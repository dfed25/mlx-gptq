"""Core of the PyTorch port of Dom's mlx-gptq recipe (gptq_mlx.py + refine_cd.py), as importable functions with the bit
width as a parameter: grid search (--mse), alternating grid fit (--lloyd), integer-zero-point constraint, GPTQ with
error feedback, coordinate-descent refinement, exact (or integer-constrained) refit in the layer metric."""
import torch, numpy as np, os
DEV = os.environ.get("DOMQ_DEV", "cuda" if torch.cuda.is_available() else "cpu"); F64 = dict(dtype=torch.float64, device=DEV)
def empty_cache():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
def mem_gb():
    return f"{torch.cuda.memory_allocated() / 2**30:.1f}/{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- grid + GPTQ (float64, as in gptq_mlx.py)
def quant_group(w, lo, hi, QMAX):
    scale = ((hi - lo) / QMAX).clamp_min(1e-8)
    q = ((w - lo[:, None]) / scale[:, None]).round().clamp(0, QMAX)
    return q * scale[:, None] + lo[:, None]

def grid_for_group(grp, QMAX, mse=True, lloyd=True, int_zero=False, fixed_zero=None):
    """grp (rows, g) float64 -> per-row (lo, hi): min-max, then --mse range search, then --lloyd alternating fit."""
    lo, hi = grp.min(1).values, grp.max(1).values
    if mse:
        best = None
        for p in np.linspace(1.0, 0.6, 21):
            lo2, hi2 = lo * p, hi * p; e = ((quant_group(grp, lo2, hi2, QMAX) - grp) ** 2).sum(1)
            if best is None: best, blo, bhi = e, lo2.clone(), hi2.clone()
            else:
                better = e < best; best = torch.where(better, e, best); blo = torch.where(better, lo2, blo); bhi = torch.where(better, hi2, bhi)
        lo, hi = blo, bhi
    if lloyd:
        e_cur = ((quant_group(grp, lo, hi, QMAX) - grp) ** 2).sum(1); lo0, hi0 = grp.min(1).values, grp.max(1).values
        for p in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            l = lo0 * p; st = ((hi0 - lo0) * p / QMAX).clamp_min(1e-8)
            for _ in range(12):
                k = ((grp - l[:, None]) / st[:, None]).round().clamp(0, QMAX)
                kb = k.mean(1, keepdim=True); xb = grp.mean(1, keepdim=True); sxx = ((k - kb) ** 2).sum(1); ok = sxx > 0
                a = torch.where(ok, ((k - kb) * (grp - xb)).sum(1) / torch.where(ok, sxx, torch.ones_like(sxx)), st).clamp_min(1e-8)
                b = xb[:, 0] - a * kb[:, 0]
                if torch.allclose(a, st) and torch.allclose(b, l): break
                st, l = a, b
            h = l + QMAX * st; e_new = ((quant_group(grp, l, h, QMAX) - grp) ** 2).sum(1)
            better = e_new < e_cur; e_cur = torch.where(better, e_new, e_cur); lo = torch.where(better, l, lo); hi = torch.where(better, h, hi)
    if int_zero:                                            # constrained alternating fit from the current grid (and from min-max starts)
        e_cur = torch.full_like(lo, float("inf")); blo, bhi = lo.clone(), hi.clone(); lo0, hi0 = grp.min(1).values, grp.max(1).values
        for start in [(lo, hi)] + [(lo0 * p, lo0 * p + (hi0 - lo0) * p) for p in (1.0, 0.8, 0.6)]:
            l, h = start; st = ((h - l) / QMAX).clamp_min(1e-8)
            for _ in range(12):
                k = ((grp - l[:, None]) / st[:, None]).round().clamp(0, QMAX)
                s_new, l_new, e_new = int_zero_fit(grp, k, QMAX, fixed_zero)
                if torch.allclose(s_new, st) and torch.allclose(l_new, l): break
                st, l = s_new, l_new
            better = e_new < e_cur; e_cur = torch.where(better, e_new, e_cur); blo = torch.where(better, l_new, blo); bhi = torch.where(better, l_new + QMAX * s_new, bhi)
        lo, hi = blo, bhi
    return lo, hi

def int_zero_fit(grp, k, QMAX, fixed_zero=None):
    """Codes k fixed: best (step, offset) with offset = -z*step for integer z: per z the least-squares step through the
    origin, s_z = <u, w> / <u, u> with u = k - z; pick the z with the smallest residual. Returns (step, offset, residual)."""
    best_e = None
    for z in ([fixed_zero] if fixed_zero is not None else range(int(QMAX) + 1)):
        u = k - z; su2 = (u * u).sum(1); suw = (u * grp).sum(1); ok = su2 > 0
        s = torch.where(ok, suw / torch.where(ok, su2, torch.ones_like(su2)), torch.zeros_like(su2)); s = s.clamp_min(1e-8)
        e = ((grp - s[:, None] * u) ** 2).sum(1)
        if best_e is None: best_e, bs, bz = e, s, torch.full_like(s, float(z))
        else:
            better = e < best_e; best_e = torch.where(better, e, best_e); bs = torch.where(better, s, bs); bz = torch.where(better, torch.full_like(s, float(z)), bz)
    return bs, -bz * bs, best_e

def upper_chol_of_inverse(Hbox, b=1024):
    """U upper-triangular with H^-1 = U^T U, computed WITHOUT holding H^-1 (three n^2 float64 copies would not fit for n = 27648).
    With P the index reversal, chol(P H P) = L' gives H = Ut Ut^T with Ut = P L' P upper-triangular, and then U = Ut^-1,
    obtained by a blocked in-place inverse of the upper-triangular Ut (as LAPACK trtri). Hbox = [H] is consumed."""
    H = Hbox.pop(); n = H.shape[0]
    Hf = H.flip(0, 1); del H; empty_cache()
    Lp = torch.linalg.cholesky(Hf); del Hf; empty_cache()
    Ut = Lp.flip(0, 1); del Lp; empty_cache()
    for j0 in range(0, n, b):
        j1 = min(j0 + b, n); I = torch.eye(j1 - j0, dtype=Ut.dtype, device=Ut.device)
        Dinv = torch.linalg.solve_triangular(Ut[j0:j1, j0:j1], I, upper=True)
        if j0 > 0: Ut[:j0, j0:j1] = -(Ut[:j0, :j0] @ Ut[:j0, j0:j1]) @ Dinv        # Ut[:j0, :j0] already holds A^-1
        Ut[j0:j1, j0:j1] = Dinv
    return Ut

def gptq(W, Hbox, QMAX, group=64, damp=0.01, block=128, mse=True, lloyd=True, int_zero=False, fixed_zero=None, static=None, kq=None):
    """W (rows, cols) float32 cuda; Hbox = [H] a one-element list holding the float64 Hessian (mean over calibration
    tokens), consumed here so that no second reference keeps it alive. Returns Q float32, codes uint8, SC, BI (rows, cols/g) float32."""
    rows, cols = W.shape; H = Hbox.pop(); g = group
    dead = torch.diag(H) == 0; H[dead, dead] = 1
    H.diagonal().add_(damp * torch.diag(H).mean())
    U = upper_chol_of_inverse([H]); del H; empty_cache()
    W = W.double().clone(); W[:, dead] = 0
    Q = torch.zeros((rows, cols), dtype=torch.float32, device=DEV); codes = torch.zeros((rows, cols), dtype=torch.uint8, device=DEV)
    SC = torch.zeros((rows, cols // g), dtype=torch.float32, device=DEV); BI = torch.zeros_like(SC)
    if static is not None: SLO, SSC, GID = static["lo"], static["scale"], static["gid"]           # act-order: grids fixed per ORIGINAL group, columns arrive permuted
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols); Wb = W[:, i1:i2].clone(); Err = torch.zeros_like(Wb); Ub = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            if static is not None: lo, scale = SLO[:, GID[j]], SSC[:, GID[j]]
            elif kq is not None and j % 256 == 0:                       # K-quant: fit the 256/g group grids of this super-block from the current weights, project, fix
                from kquant import kq_project
                grp = torch.cat([Wb[:, i:], W[:, i2:j + 256]], 1) if j + 256 > i2 else Wb[:, i:i + 256]
                LOs = []; SCs = []
                for k in range(256 // g):
                    lo_k, hi_k = grid_for_group(grp[:, k * g:(k + 1) * g], QMAX, mse, lloyd, int_zero, fixed_zero); LOs.append(lo_k); SCs.append(((hi_k - lo_k) / QMAX).clamp_min(1e-8))
                LOp, SCp, _ = kq_project(torch.stack(LOs, 1), torch.stack(SCs, 1), kq); KQC = (LOp, SCp)
                lo, scale = KQC[0][:, 0], KQC[1][:, 0]; SC[:, j // g] = scale.float(); BI[:, j // g] = lo.float()
            elif kq is not None and j % g == 0:
                lo, scale = KQC[0][:, (j % 256) // g], KQC[1][:, (j % 256) // g]; SC[:, j // g] = scale.float(); BI[:, j // g] = lo.float()
            elif j % g == 0:
                grp = torch.cat([Wb[:, i:], W[:, i2:j + g]], 1) if j + g > i2 else Wb[:, i:i + g]
                lo, hi = grid_for_group(grp, QMAX, mse, lloyd, int_zero, fixed_zero); scale = ((hi - lo) / QMAX).clamp_min(1e-8)
                SC[:, j // g] = scale.float(); BI[:, j // g] = lo.float()
            w = Wb[:, i]; d = Ub[i, i]
            q = ((w - lo) / scale).round().clamp(0, QMAX); qd = q * scale + lo; Q[:, j] = qd.float(); codes[:, j] = q.to(torch.uint8)
            err = (w - qd) / d
            Wb[:, i:] -= torch.outer(err, Ub[i, i:]); Err[:, i] = err
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    del U, W; empty_cache(); return Q, codes, SC, BI

# ---------------------------------------------------------------- refinement + refit (refine_cd.py), memory-slim:
# codes uint8, per-GROUP offsets/steps (rows, M) in float64, the Hessian kept once in float32 and widened to float64
# only inside the per-row refit solves. Same arithmetic as Dom's refine()/refit_grid() up to fp32 rounding of H.
def expand(Xg, g): return Xg.repeat_interleave(g, 1)

def refine(W, K, LOg, SCg, H, T, QMAX, group=64, sweeps=3, block=64):
    """Coordinate descent on the codes: flip a code by +-1 when 2*d*G + d^2*H_jj < 0 (exact cost change). Never increases e^T H e.
    W fp32 (rows, n); K uint8; LOg, SCg (rows, M); H fp32 (n, n); T (QMAX+1,)."""
    rows, n = W.shape; g = group; T = T.float(); K = K.clone(); dH = torch.diag(H).clone(); SCf = SCg.float(); LOf = LOg.float()
    E = expand(LOf, g) + expand(SCf, g) * T[K.long()] - W; G = E @ H; flips = 0; INF = torch.tensor(float("inf"), device=DEV)
    for _ in range(sweeps):
        changed = torch.zeros((), dtype=torch.int64, device=DEV)
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n); B = b1 - b0; Gb = G[:, b0:b1].clone(); Hbb = H[b0:b1, b0:b1]; D = torch.zeros((rows, B), device=DEV)
            for i in range(B):
                j = b0 + i; k = K[:, j].long(); sc = SCf[:, j // g]
                d_up = sc * (T[(k + 1).clamp(max=int(QMAX))] - T[k]); d_dn = sc * (T[(k - 1).clamp(min=0)] - T[k])
                del_up = torch.where(k < QMAX, 2 * d_up * Gb[:, i] + d_up * d_up * dH[j], INF)
                del_dn = torch.where(k > 0, 2 * d_dn * Gb[:, i] + d_dn * d_dn * dH[j], INF)
                up = (del_up < 0) & (del_up <= del_dn); dn = (del_dn < 0) & ~up
                dd = torch.where(up, d_up, torch.where(dn, d_dn, torch.zeros_like(d_up)))
                K[:, j] = (k + up.long() - dn.long()).to(torch.uint8); D[:, i] = dd; Gb += torch.outer(dd, Hbb[i]); changed += (up | dn).sum()
            G += D @ H[b0:b1, :]; E[:, b0:b1] += D
        c = int(changed.item()); flips += c
        if c == 0: break
    del E, G; return K, flips

def refit_grid(W, K, LOg, SCg, H, T, group=64, ridge=1e-6, budget=4e8, colblock=2048):
    """Dom's exact least-squares refit of every group's (offset, step) in the layer metric: one (2M x 2M) float64 solve per
    row, rows in chunks sized by `budget` bytes. H is fp32 and widened to float64 piecewise."""
    rows, n = W.shape; M = n // group; Td = T.double()
    LOn = torch.empty((rows, M), **F64); SCn = torch.empty((rows, M), **F64)
    rc = max(1, min(rows, int(budget // (4 * M * M * 8))))
    Zs = torch.stack([H[gi * group:(gi + 1) * group, :].double().sum(0) for gi in range(M)])      # (M, n)
    Nbb = Zs.view(M, M, group).sum(2)
    for r0 in range(0, rows, rc):
        r1 = min(r0 + rc, rows); c = r1 - r0; Wc = W[r0:r1].double(); Tm = Td[K[r0:r1].long()]; Tg = Tm.view(c, M, group)
        HW = torch.empty((c, n), **F64)
        for j0 in range(0, n, colblock): j1 = min(j0 + colblock, n); HW[:, j0:j1] = Wc @ H[:, j0:j1].double()
        Naa = torch.empty((c, M, M), **F64); Nab = torch.empty((c, M, M), **F64); Nba = torch.empty((c, M, M), **F64)
        for gi in range(M):
            sl = slice(gi * group, (gi + 1) * group); Y = Tm[:, sl] @ H[sl, :].double(); Yg = Y.view(c, M, group); Zg = Zs[gi].view(M, group)
            Naa[:, gi, :] = (Yg * Tg).sum(2); Nab[:, gi, :] = Yg.sum(2); Nba[:, gi, :] = (Zg[None] * Tg).sum(2)
        N = torch.empty((c, 2 * M, 2 * M), **F64); N[:, :M, :M] = Naa; N[:, :M, M:] = Nab; N[:, M:, :M] = Nba; N[:, M:, M:] = Nbb[None]
        rhs = torch.cat([(HW * Tm).view(c, M, group).sum(2), HW.view(c, M, group).sum(2)], 1)
        th0 = torch.cat([SCg[r0:r1], LOg[r0:r1]], 1).double()
        del Naa, Nab, Nba, HW, Y, Tm, Tg, Wc
        D = N.diagonal(dim1=1, dim2=2).clone(); D = D.clamp_min(1e-30) * ridge + 1e-30
        N.diagonal(dim1=1, dim2=2).add_(D); rhs = rhs + D * th0
        th = torch.linalg.solve(N, rhs.unsqueeze(-1)).squeeze(-1); del N
        a = th[:, :M]; b = th[:, M:]; bad = ~torch.isfinite(a) | ~torch.isfinite(b) | (a <= 0)
        SCn[r0:r1] = torch.where(bad, th0[:, :M], a); LOn[r0:r1] = torch.where(bad, th0[:, M:], b)
    empty_cache(); return LOn, SCn

def refit_metric(H, mode, block):
    """Structured variants of the refit metric (all PSD, exact importances kept except avg/max). See refine_nb.py (MLX)."""
    n = H.shape[0]
    if mode == "full": return H
    d = torch.diag(H)
    if mode in ("diag", "avg", "max"):
        d2 = d.clone()
        if mode != "diag" and block > 1:
            nb = n // block; db = d[:nb * block].view(nb, block); pooled = db.mean(1) if mode == "avg" else db.max(1).values
            d2[:nb * block] = pooled.repeat_interleave(block)
        return torch.diag(d2)
    nb = n // block; assert nb * block == n, f"n={n} not a multiple of block {block}"
    if mode == "bd":
        Hb = torch.zeros_like(H)
        for i in range(nb): sl = slice(i * block, (i + 1) * block); Hb[sl, sl] = H[sl, sl]
        return Hb
    if mode == "pool":
        M = H.view(nb, block, nb, block).mean((1, 3)); Hp = M.repeat_interleave(block, 0).repeat_interleave(block, 1)
        idx = torch.arange(n, device=H.device); Hp[idx, idx] = torch.maximum(d, Hp[idx, idx]); return Hp
    raise ValueError(mode)

def refit_int(W, K, LOg, SCg, H, T, QMAX, group=64, sweeps=2, fixed_zero=None):
    """Dom's refit under the integer-zero-point constraint: for each group in turn, with every other group fixed, try all
    zero points z; for each, the step minimising e^T H e is closed-form (quadratic in s), keep the best; apply only if the
    layer cost decreases. Exact cost accounting through G = E H as in refine(). Returns new (LOg, SCg)."""
    rows, n = W.shape; M = n // group; g = group; T = T.float(); SCg = SCg.float().clone(); LOg = LOg.float().clone()
    E = expand(LOg, g) + expand(SCg, g) * T[K.long()] - W; G = E @ H; zs = torch.tensor([float(fixed_zero)], device=DEV) if fixed_zero is not None else torch.arange(int(QMAX) + 1, device=DEV, dtype=torch.float32)
    for _ in range(sweeps):
        changed = 0
        for gi in range(M):
            sl = slice(gi * g, (gi + 1) * g); k = T[K[:, sl].long()]; c = E[:, sl] + W[:, sl]; Hgg = H[sl, sl]; Gg = G[:, sl]
            kH = k @ Hgg; kHk = (kH * k).sum(1); kH1 = kH.sum(1); oneH1 = Hgg.sum(); kHc = (kH * c).sum(1); oneHc = (c @ Hgg).sum(1)
            kG = (k * Gg).sum(1); oneG = Gg.sum(1); cHc = ((c @ Hgg) * c).sum(1); cG = (c * Gg).sum(1); const = cHc - 2 * cG
            uHu = kHk[:, None] - 2 * zs[None, :] * kH1[:, None] + zs[None, :] ** 2 * oneH1            # (rows, 16)
            uHc = kHc[:, None] - zs[None, :] * oneHc[:, None]; uG = kG[:, None] - zs[None, :] * oneG[:, None]
            ok = uHu > 1e-12; s = torch.where(ok, (uHc - uG) / torch.where(ok, uHu, torch.ones_like(uHu)), torch.zeros_like(uHu)); ok = ok & (s > 0)
            f = torch.where(ok, s * s * uHu + 2 * s * (uG - uHc) + const[:, None], torch.full_like(uHu, float("inf")))
            fbest, zidx = f.min(1); sbest = s[torch.arange(rows, device=DEV), zidx]; zbest = zs[zidx]; apply = fbest < -1e-9 * (cHc.abs() + 1e-30)   # zbest = the zero VALUE (zs[i] != i when fixed_zero is set)
            if not bool(apply.any()): continue
            u = k - zbest[:, None]; Q_new = sbest[:, None] * u; D = torch.where(apply[:, None], Q_new - c, torch.zeros_like(c))
            E[:, sl] += D; G += D @ H[sl, :]
            SCg[:, gi] = torch.where(apply, sbest, SCg[:, gi]); LOg[:, gi] = torch.where(apply, -zbest * sbest, LOg[:, gi]); changed += int(apply.sum())
        if changed == 0: break
    del E, G; return LOg.double(), SCg.double()

def refine_with_refit(W, K, LOg, SCg, H, T, QMAX, rounds=2, sweeps=3, group=64, int_zero=False, metric="full", block=16, fixed_zero=None, kq=None):
    flips = 0; K, f = refine(W, K, LOg, SCg, H, T, QMAX, group, sweeps=sweeps); flips += f
    Hfit = refit_metric(H, metric, block); meta = None
    if kq is not None: from kquant import kq_project
    for _ in range(rounds):
        LOg, SCg = refit_int(W, K, LOg, SCg, H, T, QMAX, group, fixed_zero=fixed_zero) if int_zero else refit_grid(W, K, LOg, SCg, Hfit, T, group)
        if kq is not None:                                              # back onto the K-quant set (weights: diag(H) summed per group), then let the refinement re-pick codes
            wg = torch.diag(H).double().reshape(-1, group).sum(1)[None, :].expand(LOg.shape[0], -1)
            LOg, SCg, meta = kq_project(LOg, SCg, kq, wg)
        K, f = refine(W, K, LOg, SCg, H, T, QMAX, group, sweeps=sweeps); flips += f
    Q = expand(LOg.float(), group) + expand(SCg.float(), group) * T.float()[K.long()]
    return (K, Q, SCg.float(), LOg.float(), flips) if kq is None else (K, Q, SCg.float(), LOg.float(), flips, meta)

