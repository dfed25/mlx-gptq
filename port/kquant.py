"""K-quant constraint for Dom's grids (llama.cpp Q2_K/Q3_K/Q4_K/Q5_K/Q6_K layouts): per super-block of 256 weights one fp16
super-scale d (and super-min dmin), per sub-block an integer sub-scale sc (and sub-min m) of a few bits:
    w = d*sc*q - dmin*m        (q4_k, q5_k: 32-weight sub-blocks, 6-bit sc and m;  q2_k: 16-weight, 4-bit sc and m)
    w = d*sc*(q - z)           (q3_k: 16-weight, 6-bit sc, z = 4;  q6_k: 16-weight, 8-bit signed sc, z = 32)
In the affine language of the recipe (w = scale*q + bias per group of `sb` columns): scale = d*sc, bias = -dmin*m (or -z*scale).
kq_project() maps free per-group (bias, scale) onto that set, searching d per (row, super-block) in the layer-weighted least squares."""
import torch, numpy as np

KQ = {"q4_k": dict(sb=32, smax=63, mmax=63, zfix=None, qmax=15, bpw=4.5),
      "q5_k": dict(sb=32, smax=63, mmax=63, zfix=None, qmax=31, bpw=5.5),
      "q2_k": dict(sb=16, smax=15, mmax=15, zfix=None, qmax=3, bpw=2.625),
      "q3_k": dict(sb=16, smax=31, mmax=None, zfix=4, qmax=7, bpw=3.4375),
      "q6_k": dict(sb=16, smax=127, mmax=None, zfix=32, qmax=63, bpw=6.5625)}
SB = 256

def _fit_super(V, vmax, wgt, nf=16, fmax=1.25, kmin=0):
    """V (rows, nsb, gps) >= 0 (or any sign, one shared sign per super-block): choose per (row, super-block) a fp16 super value D and
    integers k in [0, vmax] with V ~ D*k, minimising sum_j wgt_j (V_j - D k_j)^2 over a search of D between max|V|/vmax and fmax times it."""
    rows, nsb, gps = V.shape; A = V.abs(); amax = A.max(-1, keepdim=True).values; sgn = torch.where(V.gather(-1, A.argmax(-1, keepdim=True)) < 0, -1.0, 1.0)
    base = (amax / vmax).clamp_min(1e-12); best = None
    for f in torch.linspace(1.0, fmax, nf, dtype=V.dtype, device=V.device):
        D = (base * f).half().double(); k = (A / D).round().clamp(kmin, vmax); e = (wgt * (A - D * k) ** 2).sum(-1, keepdim=True)
        if best is None: best, bD, bk = e, D, k
        else: better = e < best; best = torch.where(better, e, best); bD = torch.where(better, D, bD); bk = torch.where(better, k, bk)
    return bD * sgn, bk            # D carries the sign (fp16-exact), k are the integers

def kq_project(LO, SC, kind, wgt=None):
    """LO, SC: (rows, ngroups) float64 per-group bias and scale (group = sub-block). Returns (LO', SC', meta) with the projected grids and
    meta = dict(d, dmin (rows, nsb) fp16-exact float64, sc, m (rows, ngroups) integer) for an exact pack."""
    k = KQ[kind]; rows, ng = SC.shape; gps = SB // k["sb"]; assert ng % gps == 0, f"{ng} groups not a multiple of {gps}"
    nsb = ng // gps; W = (torch.ones_like(SC) if wgt is None else wgt).reshape(rows, nsb, gps)
    D, sc = _fit_super(SC.reshape(rows, nsb, gps), k["smax"], W, kmin=1); SCp = (D * sc).reshape(rows, ng).clamp_min(1e-8)     # sub-scale >= 1: a zero step would divide the code assignment by zero
    if k["zfix"] is not None: LOp = -k["zfix"] * SCp; Dm = torch.zeros_like(D); m = torch.zeros_like(sc)
    else:
        Dm, m = _fit_super((-LO).reshape(rows, nsb, gps), k["mmax"], W); LOp = (-(Dm * m)).reshape(rows, ng)
    return LOp, SCp, dict(d=D.squeeze(-1), dmin=Dm.squeeze(-1), sc=sc.reshape(rows, ng).to(torch.int16), m=m.reshape(rows, ng).to(torch.int16))

# ---------------------------------------------------------------- packers (ggml block layouts, QK_K = 256), numpy uint8 (rows, nsb, block_bytes)
def _f16(x): return np.ascontiguousarray(np.asarray(x, dtype=np.float64).astype("<f2")).view(np.uint8)
def _scales_k4(sc, m):
    """12-byte scale/min packing of Q4_K/Q5_K (get_scale_min_k4): sc, m (..., 8) ints 0..63 -> (..., 12) uint8."""
    sc = sc.astype(np.uint8); m = m.astype(np.uint8); out = np.zeros(sc.shape[:-1] + (12,), dtype=np.uint8)
    out[..., 0:4] = (sc[..., 0:4] & 63) | ((sc[..., 4:8] >> 4) << 6)
    out[..., 4:8] = (m[..., 0:4] & 63) | ((m[..., 4:8] >> 4) << 6)
    out[..., 8:12] = (sc[..., 4:8] & 15) | ((m[..., 4:8] & 15) << 4)
    return out
def pack_q4_k(q, d, dmin, sc, m):
    """q (rows, cols) ints 0..15, d/dmin (rows, nsb) fp16-exact, sc/m (rows, nsb*8). Block: d, dmin, scales[12], qs[128] (low nibble = w[0:32], high = w[32:64] per 64)."""
    rows, cols = q.shape; nsb = cols // 256; q = q.reshape(rows, nsb, 4, 2, 32).astype(np.uint8)
    qs = (q[:, :, :, 0, :] | (q[:, :, :, 1, :] << 4)).reshape(rows, nsb, 128)
    return np.concatenate([_f16(d).reshape(rows, nsb, 2), _f16(dmin).reshape(rows, nsb, 2), _scales_k4(sc.reshape(rows, nsb, 8), m.reshape(rows, nsb, 8)), qs], -1)
def pack_q5_k(q, d, dmin, sc, m):
    """Block: d, dmin, scales[12], qh[32] (bit (j//32) of weight j%32... per ggml: high bit of w[j] in qh[j%32] bit j//32), qs[128] low nibbles as Q4_K."""
    rows, cols = q.shape; nsb = cols // 256; q = q.reshape(rows, nsb, 256).astype(np.uint16); lo = (q & 15).astype(np.uint8); hi = (q >> 4).astype(np.uint8)
    l = lo.reshape(rows, nsb, 4, 2, 32); qs = (l[:, :, :, 0, :] | (l[:, :, :, 1, :] << 4)).reshape(rows, nsb, 128)
    h = hi.reshape(rows, nsb, 8, 32); qh = np.zeros((rows, nsb, 32), dtype=np.uint8)
    for j in range(8): qh |= (h[:, :, j, :] << j).astype(np.uint8)
    return np.concatenate([_f16(d).reshape(rows, nsb, 2), _f16(dmin).reshape(rows, nsb, 2), _scales_k4(sc.reshape(rows, nsb, 8), m.reshape(rows, nsb, 8)), qh, qs], -1)
def _qs_2bit(q2):
    """2-bit codes (rows, nsb, 256) -> (rows, nsb, 64): per 128-weight half, byte l holds weights 32*j + l (j = 0..3) in bit pair j (Q2_K/Q3_K qs layout)."""
    rows, nsb, _ = q2.shape; h = q2.reshape(rows, nsb, 2, 4, 32).astype(np.uint8); out = np.zeros((rows, nsb, 2, 32), dtype=np.uint8)
    for j in range(4): out |= (h[:, :, :, j, :] << (2 * j)).astype(np.uint8)
    return out.reshape(rows, nsb, 64)
def pack_q3_k(q, d, sc):
    """q (rows, cols) ints 0..7 (value = q - 4), d (rows, nsb) fp16-exact, sc (rows, nsb*16) ints 0..31 (stored as sc + 32, ggml's 6-bit signed scale).
    Block: hmask[32] (bit 4n+j of byte l = high bit of weight 128n + 32j + l; set means q >= 4), qs[64], scales[12], d."""
    rows, cols = q.shape; nsb = cols // 256; q = q.reshape(rows, nsb, 256).astype(np.uint8)
    hi = (q >> 2).reshape(rows, nsb, 2, 4, 32); hm = np.zeros((rows, nsb, 32), dtype=np.uint8)
    for n in range(2):
        for j in range(4): hm |= (hi[:, :, n, j, :] << (4 * n + j)).astype(np.uint8)
    qs = _qs_2bit(q & 3)
    sv = (sc.reshape(rows, nsb, 16).astype(np.int32) + 32).astype(np.uint8); sb = np.zeros((rows, nsb, 12), dtype=np.uint8)
    for i in range(4):
        sb[..., i] = (sv[..., i] & 15) | ((sv[..., 8 + i] & 15) << 4)
        sb[..., 4 + i] = (sv[..., 4 + i] & 15) | ((sv[..., 12 + i] & 15) << 4)
        sb[..., 8 + i] = (sv[..., i] >> 4) | ((sv[..., 4 + i] >> 4) << 2) | ((sv[..., 8 + i] >> 4) << 4) | ((sv[..., 12 + i] >> 4) << 6)
    return np.concatenate([hm, qs, sb, _f16(d).reshape(rows, nsb, 2)], -1)
def pack_q2_k(q, d, dmin, sc, m):
    """q (rows, cols) ints 0..3, d/dmin (rows, nsb), sc/m (rows, nsb*16) ints 0..15. Block: scales[16] (low nibble scale, high nibble min, per 16-weight sub-block), qs[64], d, dmin."""
    rows, cols = q.shape; nsb = cols // 256; q = q.reshape(rows, nsb, 256).astype(np.uint8)
    sv = (sc.reshape(rows, nsb, 16).astype(np.uint8) & 15) | ((m.reshape(rows, nsb, 16).astype(np.uint8) & 15) << 4)
    return np.concatenate([sv, _qs_2bit(q), _f16(d).reshape(rows, nsb, 2), _f16(dmin).reshape(rows, nsb, 2)], -1)

def pack_q6_k(q, d, sc):
    """q (rows, cols) ints 0..63 (value = q - 32), d (rows, nsb) fp16-exact, sc (rows, nsb*16) ints 0..127 (int8 scales).
    Block: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] int8, d. Per 128-weight half: weight 32k + l -> ql[l (k even) / l+32 (k odd)]
    low nibble (k < 2) or high nibble (k >= 2); qh[l] bits 2k..2k+1."""
    rows, cols = q.shape; nsb = cols // 256; q = q.reshape(rows, nsb, 2, 4, 32).astype(np.uint8); lo = q & 15; hi = q >> 4
    ql = np.zeros((rows, nsb, 2, 64), dtype=np.uint8); qh = np.zeros((rows, nsb, 2, 32), dtype=np.uint8)
    ql[:, :, :, :32] = lo[:, :, :, 0, :] | (lo[:, :, :, 2, :] << 4); ql[:, :, :, 32:] = lo[:, :, :, 1, :] | (lo[:, :, :, 3, :] << 4)
    for k in range(4): qh |= (hi[:, :, :, k, :] << (2 * k)).astype(np.uint8)
    return np.concatenate([ql.reshape(rows, nsb, 128), qh.reshape(rows, nsb, 64), sc.reshape(rows, nsb, 16).astype(np.int8).view(np.uint8), _f16(d).reshape(rows, nsb, 2)], -1)

def rtn_kquant(Wt, kind, chunk=4096):
    """Round-to-nearest K-quant of a plain weight matrix (embeddings / head): Dom's per-group grid fit (MSE search + Lloyd, fixed zero for
    the symmetric kinds), projected per super-block. Rows are processed in chunks (a 152k x 1536 embedding would need >20 GB otherwise).
    Returns block bytes (rows, nsb, block)."""
    if Wt.shape[0] > chunk: return np.concatenate([rtn_kquant(Wt[i:i + chunk], kind, chunk) for i in range(0, Wt.shape[0], chunk)], 0)
    from quant_core import grid_for_group, DEV
    k = KQ[kind]; g = k["sb"]; W = Wt.to(DEV).double(); rows, cols = W.shape; assert cols % 256 == 0      # the chunk is cast on the GPU; the caller may pass bf16
    grp = W.reshape(rows * (cols // g), g); QMAX = float(k["qmax"])
    lo, hi = grid_for_group(grp, QMAX, True, True, k["zfix"] is not None, k["zfix"]); sc = ((hi - lo) / QMAX).clamp_min(1e-8)
    LOp, SCp, meta = kq_project(lo.reshape(rows, cols // g), sc.reshape(rows, cols // g), kind)
    q = ((grp - LOp.reshape(-1)[:, None]) / SCp.reshape(-1)[:, None]).round().clamp(0, QMAX).reshape(rows, cols).to(torch.int64).cpu().numpy()
    d = meta["d"].cpu().numpy(); dm = meta["dmin"].cpu().numpy(); s = meta["sc"].cpu().numpy(); m = meta["m"].cpu().numpy()
    if kind == "q4_k": return pack_q4_k(q, d, dm, s, m)
    if kind == "q5_k": return pack_q5_k(q, d, dm, s, m)
    if kind == "q6_k": return pack_q6_k(q, d, s)
    if kind == "q3_k": return pack_q3_k(q, d, s)
    if kind == "q2_k": return pack_q2_k(q, d, dm, s, m)
