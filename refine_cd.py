"""Coordinate-descent refinement of a GPTQ solution (cf. CDQuant, 2024), blocked so that the heavy work is one matmul
per block of columns.

Problem: minimise cost(K) = e^T H e per row, e = q(K) - w, where code K[r, j] in {0..QMAX} selects the tick
lo[r, j] + scale[r, j] * T[K[r, j]] (T = arange for a uniform grid, or a non-uniform template).
Move: change one code by +-1 if the exact cost change  2*d*G[r, j] + d^2*H[j, j]  is negative (G = e H is the gradient/2).
Never increases the cost; stops when no single move helps or after `sweeps` passes."""
import numpy as np

def refine(W, K, LO, SC, H, T, sweeps=3, block=64):
    """W: (rows, n) original weights. K: (rows, n) int codes. LO, SC: (rows, n) per-entry grid origin and step.
    H: (n, n) symmetric PSD (already damped/normalised as desired). T: (QMAX+1,) tick template.
    Returns (K_new, Q_new, n_flips)."""
    rows, n = W.shape; QMAX = len(T) - 1; T = T.astype(np.float32)
    K = K.astype(np.int64).copy(); H = H.astype(np.float32); SC = SC.astype(np.float32); dH = np.diag(H).copy()
    E = (LO + SC * T[K] - W).astype(np.float32); G = E @ H; flips = 0
    for _ in range(sweeps):
        changed = 0
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n); B = b1 - b0; Gb = G[:, b0:b1].copy(); Hbb = H[b0:b1, b0:b1]; D = np.zeros((rows, B), dtype=np.float32)
            for i in range(B):
                j = b0 + i; k = K[:, j]; sc = SC[:, j]
                d_up = sc * (T[np.minimum(k + 1, QMAX)] - T[k]); d_dn = sc * (T[np.maximum(k - 1, 0)] - T[k])
                del_up = np.where(k < QMAX, 2 * d_up * Gb[:, i] + d_up * d_up * dH[j], np.inf)
                del_dn = np.where(k > 0, 2 * d_dn * Gb[:, i] + d_dn * d_dn * dH[j], np.inf)
                up = (del_up < 0) & (del_up <= del_dn); dn = (del_dn < 0) & ~up
                if not (up.any() or dn.any()): continue
                dd = np.where(up, d_up, np.where(dn, d_dn, 0.0)).astype(np.float32)
                K[:, j] = k + up.astype(np.int64) - dn.astype(np.int64); D[:, i] = dd; Gb += np.outer(dd, Hbb[i]); changed += int(up.sum() + dn.sum())
            if D.any(): G += D @ H[b0:b1, :]; E[:, b0:b1] += D
        flips += changed
        if changed == 0: break
    return K, (LO + SC * T[K]).astype(np.float32), flips

if __name__ == "__main__":                       # self-test against a plain column-by-column implementation
    rng = np.random.default_rng(0); rows, n, QMAX = 40, 192, 7
    X = rng.normal(size=(600, n)) @ (np.eye(n) + 0.3 * rng.normal(size=(n, n)) / np.sqrt(n)); H = X.T @ X / 600; H += 0.01 * np.mean(np.diag(H)) * np.eye(n)
    W = rng.normal(size=(rows, n)) * 0.05; lo = W.min(1, keepdims=True) * np.ones((1, n)); sc = (W.max(1, keepdims=True) - W.min(1, keepdims=True)) / QMAX * np.ones((1, n))
    for T in (np.arange(QMAX + 1.0), np.array([0, 1.34, 2.296, 3.113, 3.887, 4.704, 5.66, 7.0])):
        K0 = np.abs(W[:, :, None] - (lo[:, :, None] + sc[:, :, None] * T[None, None, :])).argmin(2)
        cost = lambda K: np.einsum("ri,ij,rj->r", lo + sc * T[K] - W, H, lo + sc * T[K] - W).sum()
        K1, Q1, f1 = refine(W, K0, lo, sc, H, T, sweeps=20, block=64)
        K2 = K0.copy(); E = lo + sc * T[K2] - W; G = E @ H; f2 = 0                    # naive reference, float64, immediate updates
        for _ in range(20):
            ch = 0
            for j in range(n):
                k = K2[:, j]; du = sc[:, j] * (T[np.minimum(k + 1, QMAX)] - T[k]); dd_ = sc[:, j] * (T[np.maximum(k - 1, 0)] - T[k])
                a = np.where(k < QMAX, 2 * du * G[:, j] + du * du * H[j, j], np.inf); b = np.where(k > 0, 2 * dd_ * G[:, j] + dd_ * dd_ * H[j, j], np.inf)
                up = (a < 0) & (a <= b); dn = (b < 0) & ~up; d = np.where(up, du, np.where(dn, dd_, 0.0))
                K2[:, j] = k + up - dn; G += np.outer(d, H[j]); ch += int(up.sum() + dn.sum())
            f2 += ch
            if ch == 0: break
        print(f"template {'uniform' if T[1]==1 else 'non-uniform'}: cost start {cost(K0):.5f} -> blocked {cost(K1):.5f} ({f1} flips) | naive {cost(K2):.5f} ({f2} flips) | "
              f"codes identical: {np.array_equal(K1, K2)} | codes in range: {K1.min() >= 0 and K1.max() <= QMAX} | Q consistent: {np.allclose(Q1, lo + sc * T[K1], atol=1e-6)}")


def refit_grid(W, K, LO, SC, H, T, group=64, ridge=1e-6):
    """Dom's line fit in the TRUE metric. With the codes K fixed, each group g of a row has two free numbers, offset b_g
    and step a_g, and the row's values are q_j = b_g + a_g*T[K_j]. The layer cost e^T H e (e = q - w) is a quadratic in
    the 2M numbers (M = n/group), so its minimum solves one small linear system per row:
        (A^T H A) theta = A^T H w,      A = design matrix with columns [T[K] restricted to g] and [1 restricted to g].
    A tiny proximal ridge keeps degenerate groups (all codes equal) at their current values; because the current grid is
    feasible, the cost never increases. Returns per-entry (LO, SC) and per-group (BI, SCg)."""
    rows, n = W.shape; M = n // group; Tm = T.astype(np.float64)[K]                         # (rows, n) template value per weight
    Hd = H.astype(np.float64); HW = W.astype(np.float64) @ Hd                               # (rows, n)
    Naa = np.empty((rows, M, M)); Nab = np.empty((rows, M, M)); Nbb = np.empty((M, M)); Nba = np.empty((rows, M, M))
    for gi in range(M):
        sl = slice(gi * group, (gi + 1) * group)
        Y = Tm[:, sl] @ Hd[sl, :]                                                            # (rows, n): row-specific
        Z = Hd[sl, :].sum(axis=0)                                                            # (n,): same for all rows
        Yg = Y.reshape(rows, M, group); Zg = Z.reshape(M, group); Tg = Tm.reshape(rows, M, group)
        Naa[:, gi, :] = (Yg * Tg).sum(axis=2); Nab[:, gi, :] = Yg.sum(axis=2)
        Nba[:, gi, :] = (Zg[None] * Tg).sum(axis=2); Nbb[gi, :] = Zg.sum(axis=1)
    N = np.empty((rows, 2 * M, 2 * M)); N[:, :M, :M] = Naa; N[:, :M, M:] = Nab; N[:, M:, :M] = Nba; N[:, M:, M:] = Nbb[None]
    rhs = np.concatenate([(HW * Tm).reshape(rows, M, group).sum(axis=2), HW.reshape(rows, M, group).sum(axis=2)], axis=1)
    th0 = np.concatenate([SC.reshape(rows, M, group)[:, :, 0], LO.reshape(rows, M, group)[:, :, 0]], axis=1).astype(np.float64)
    D = np.einsum("rii->ri", N).copy(); D = np.maximum(D, 1e-30) * ridge + 1e-30
    idx = np.arange(2 * M); N[:, idx, idx] += D; rhs = rhs + D * th0
    th = np.linalg.solve(N, rhs[..., None])[..., 0]
    a = th[:, :M]; b = th[:, M:]; bad = ~np.isfinite(a) | ~np.isfinite(b) | (a <= 0)                # keep the old grid where the fit is unusable
    a = np.where(bad, th0[:, :M], a); b = np.where(bad, th0[:, M:], b)
    return np.repeat(b, group, axis=1), np.repeat(a, group, axis=1), b.astype(np.float32), a.astype(np.float32)


def refine_with_refit(W, K, LO, SC, H, T, rounds=2, sweeps=3, group=64):
    """Alternate: grid refit (exact least squares in the layer metric) and code refinement (coordinate descent)."""
    flips = 0
    K, Q, f = refine(W, K, LO, SC, H, T, sweeps=sweeps); flips += f
    for _ in range(rounds):
        LO, SC, BIg, SCg = refit_grid(W, K, LO, SC, H, T, group)
        K, Q, f = refine(W, K, LO, SC, H, T, sweeps=sweeps); flips += f
    BIg = LO.reshape(len(W), -1, group)[:, :, 0].astype(np.float32); SCg = SC.reshape(len(W), -1, group)[:, :, 0].astype(np.float32)
    return K, (LO + SC * T.astype(np.float64)[K]).astype(np.float32), SCg, BIg, flips


def refine_z(W, K, LO, SC, Hs, T, z=1.0, sweeps=2, block=64):
    """Dom's rule (2026-10-02): believe a flip only if its gain is large compared with its own sampling noise.
    Hs is a list of m Hessians computed on m disjoint pieces of the calibration text (same normalisation, same ridge),
    so the usual objective uses their mean. For a candidate flip the cost change is linear in the Hessian, hence the
    full-sample change is the mean of the m per-piece changes delta_p, and its standard error is sd(delta_p)/sqrt(m).
        score = -mean(delta_p) / (sd(delta_p) / sqrt(m));   accept iff mean < 0 and score > z.
    z = -inf reproduces refine() on the mean Hessian (accept every improving flip)."""
    rows, n = W.shape; m = len(Hs); QMAX = len(T) - 1; T = T.astype(np.float32); K = K.astype(np.int64).copy(); SC = SC.astype(np.float32)
    Hs = [np.asarray(h, dtype=np.float32) for h in Hs]; dHs = np.stack([np.diag(h) for h in Hs])   # (m, n); no copy if already float32
    E = (LO + SC * T[K] - W).astype(np.float32); Gs = [E @ h for h in Hs]; flips = 0; rm = np.sqrt(m)
    for _ in range(sweeps):
        changed = 0
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n); B = b1 - b0; Gb = np.stack([g[:, b0:b1] for g in Gs]); Hbb = [h[b0:b1, b0:b1] for h in Hs]  # Gb: (m, rows, B)
            D = np.zeros((rows, B), dtype=np.float32)
            for i in range(B):
                j = b0 + i; k = K[:, j]; sc = SC[:, j]; best = np.zeros(rows, dtype=np.float32); bestmean = np.zeros(rows, dtype=np.float32); step = np.zeros(rows, dtype=np.int64)
                for sgn, ok in ((1, k < QMAX), (-1, k > 0)):
                    d = sc * (T[np.clip(k + sgn, 0, QMAX)] - T[k])
                    dl = 2 * d[None, :] * Gb[:, :, i] + (d * d)[None, :] * dHs[:, j][:, None]            # (m, rows)
                    mean = dl.mean(0)
                    score = -mean / (dl.std(0, ddof=1) / rm + 1e-30) if m > 1 else np.full(rows, np.inf, dtype=np.float32)
                    acc = ok & (mean < 0) & (score > z) & (mean < bestmean)
                    best = np.where(acc, d, best); bestmean = np.where(acc, mean, bestmean); step = np.where(acc, sgn, step)
                if not step.any(): continue
                K[:, j] = k + step; D[:, i] = best; changed += int((step != 0).sum())
                for p in range(m): Gb[p] += np.outer(best, Hbb[p][i])
            if D.any():
                for p in range(m): Gs[p] += D @ Hs[p][b0:b1, :]
        flips += changed
        if changed == 0: break
    return K, (LO + SC * T[K]).astype(np.float32), flips


def refine_best_first(W, K, LO, SC, Hs, T, z=None, sweeps=3, block=64):
    """Largest-gain-first refinement (per row, inside each block of columns), vectorised over rows:
    at every step each row picks the candidate flip with the largest current gain in the block (optionally only among
    flips whose score exceeds z, Dom's rule, when several Hessian pieces are given), applies it, updates its gradients,
    and repeats until no row has an improving flip in the block. Hs: list of pieces (one piece = plain refinement)."""
    rows, n = W.shape; m = len(Hs); QMAX = len(T) - 1; T = T.astype(np.float32); K = K.astype(np.int64).copy(); SC = SC.astype(np.float32)
    Hs = [np.asarray(h, dtype=np.float32) for h in Hs]; Hm = Hs[0] if m == 1 else sum(Hs) / m; dHs = np.stack([np.diag(h) for h in Hs]); dHm = dHs.mean(0)
    E = (LO + SC * T[K] - W).astype(np.float32); Gs = [E @ h for h in Hs]; Gm = Gs[0] if m == 1 else sum(Gs) / m; flips = 0; rm = np.sqrt(m); ar = np.arange(rows)
    for _ in range(sweeps):
        changed = 0
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n); B = b1 - b0; Kb = K[:, b0:b1]; Sb = SC[:, b0:b1]
            Gb = [g[:, b0:b1].copy() for g in Gs]; Gbm = Gb[0] if m == 1 else sum(Gb) / m; Hbb = [h[b0:b1, b0:b1] for h in Hs]; Hbbm = Hbb[0] if m == 1 else sum(Hbb) / m
            D = np.zeros((rows, B), dtype=np.float32)
            for _step in range(2 * B):
                best_gain = np.zeros(rows, dtype=np.float32); best_j = np.full(rows, -1); best_d = np.zeros(rows, dtype=np.float32); best_s = np.zeros(rows, dtype=np.int64)
                for sgn in (1, -1):
                    ok = (Kb + sgn >= 0) & (Kb + sgn <= QMAX); d = Sb * (T[np.clip(Kb + sgn, 0, QMAX)] - T[Kb])          # (rows, B)
                    gain = -(2 * d * Gbm + d * d * dHm[None, b0:b1]); gain = np.where(ok, gain, -np.inf)
                    if z is not None and m > 1:
                        dl = np.stack([2 * d * gb + d * d * dh[None, b0:b1] for gb, dh in zip(Gb, dHs)])                   # (m, rows, B) cost changes
                        score = -dl.mean(0) / (dl.std(0, ddof=1) / rm + 1e-30); gain = np.where(score > z, gain, -np.inf)
                    j = gain.argmax(1); g = gain[ar, j]; better = g > best_gain
                    best_gain = np.where(better, g, best_gain); best_j = np.where(better, j, best_j); best_d = np.where(better, d[ar, j], best_d); best_s = np.where(better, sgn, best_s)
                act = best_gain > 0
                if not act.any(): break
                r = ar[act]; j = best_j[act]; dd = best_d[act]
                Kb[r, j] += best_s[act]; D[r, j] += dd; changed += len(r)
                for p in range(m): Gb[p][r] += dd[:, None] * Hbb[p][j]
                Gbm = Gb[0] if m == 1 else sum(Gb) / m
            K[:, b0:b1] = Kb
            if D.any():
                for p in range(m): Gs[p] += D @ Hs[p][b0:b1, :]
                Gm = Gs[0] if m == 1 else sum(Gs) / m
        flips += changed
        if changed == 0: break
    return K, (LO + SC * T[K]).astype(np.float32), flips
