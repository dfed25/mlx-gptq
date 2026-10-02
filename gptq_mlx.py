"""Error-feedback rounding (GPTQ, Frantar et al. 2022) for the linear layers of an MLX model, group-wise affine
quantization compatible with MLX's format (w ≈ scale*q + bias, q in [0, 2^bits-1], one (scale, bias) per group).

Idea: rounding a column of W to the grid makes an error; instead of accepting it, adjust the not-yet-rounded columns
so that the layer's OUTPUT on real activations changes as little as possible. With H = X^T X (X = calibration inputs)
the optimal adjustment is  W[:, j+1:] -= err_j * U[j, j+1:] / U[j, j]  with U the upper Cholesky factor of H^{-1}.
Layers are processed in order; each layer's calibration inputs come from the already-quantized previous layers.

    python gptq_mlx.py MODEL_PATH OUT_PATH BITS [--group 64] [--nsamples 128] [--seqlen 512] [--damp 0.01]
Writes OUT_PATH with the DEQUANTIZED weights (bf16) so quality can be evaluated with the unchanged model code;
packing into MLX's quantized format is a separate step."""
import sys, time, argparse, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("bits", type=int)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--nsamples", type=int, default=128)
ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--block", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-sequential", action="store_true", help="ablation: calibrate every layer on the UNQUANTIZED model's activations (as mlx_lm.quant.gptq does)")
ap.add_argument("--grid-from-original", action="store_true", help="ablation: take each group's grid from the original weights instead of the error-updated ones (as mlx_lm.quant.gptq does)")
ap.add_argument("--lloyd", action="store_true", help="per-group grid by alternating nearest-tick assignment and least-squares line fit (Dom's loop), 4 starts, kept only where it beats the --mse/min-max grid")
ap.add_argument("--template", default=None, help="npy file with 2^bits tick positions scaled to [0, 2^bits-1] (non-uniform grid: tick k at lo + step*t_k); evaluation only, not packable into MLX's affine format")
ap.add_argument("--refine", type=int, default=0, help="after GPTQ, this many sweeps of coordinate-descent refinement of the codes on the layer objective (refine_cd.py)")
ap.add_argument("--refine-z", type=float, default=None, help="Dom's rule: accept a flip only if its gain exceeds this many standard errors, estimated from --pieces disjoint pieces of the calibration text (refine_cd.refine_z)")
ap.add_argument("--pieces", type=int, default=1, help="keep the Hessian in this many pieces (calibration batches dealt round-robin); needed for --refine-z")
ap.add_argument("--order", default="column", choices=["column", "best"], help="refinement order: column (left to right) or best (largest gain first per row inside each block; refine_cd.refine_best_first)")
ap.add_argument("--refit", type=int, default=0, help="with --refine: this many rounds of alternating grid refit (least squares in the layer metric, refine_cd.refit_grid) and code refinement")
ap.add_argument("--refine-damp", type=float, default=0.01, help="ridge added to the normalised Hessian in the refinement objective (larger = trust the calibration data less)")
ap.add_argument("--max-layers", type=int, default=0, help="smoke test: process only this many layers and exit without saving")
ap.add_argument("--eval", action="store_true", help="after quantization, print WikiText-2 perplexity of the in-memory model (no reload)")
ap.add_argument("--no-save", action="store_true", help="do not write the model (experiments: avoids a 3 GB write)")
ap.add_argument("--mse", action="store_true", help="per-group grid by error search (shrink the min-max range) instead of plain min-max")
args = ap.parse_args()
QMAX = 2 ** args.bits - 1

TPL = np.load(args.template).astype(np.float64) if args.template else None
def nearest_tick(w, lo, scale):
    """Template grid: index of the nearest tick lo + scale*TPL[k] for each entry of w (rows x n)."""
    ticks = lo[:, None] + scale[:, None] * TPL[None, :]
    return np.abs(w[:, :, None] - ticks[:, None, :]).argmin(axis=2)
def quant_group(w, lo, hi):
    """Round the columns w (rows x g) to the affine grid with per-row min lo and max hi. Returns dequantized values."""
    scale = np.maximum((hi - lo) / QMAX, 1e-8)
    if TPL is not None:
        return lo[:, None] + scale[:, None] * TPL[nearest_tick(w, lo, scale)]
    q = np.clip(np.round((w - lo[:, None]) / scale[:, None]), 0, QMAX)
    return q * scale[:, None] + lo[:, None]

def gptq(W, H):
    """W: (rows, cols) float32 numpy (a Linear's weight, x @ W.T). H: (cols, cols). Returns dequantized W'."""
    W = W.astype(np.float64).copy(); W0 = W.copy(); cols = W.shape[1]; H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H[np.diag_indices(cols)] += args.damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H); U = np.linalg.cholesky(Hinv).T          # upper triangular: Hinv = U^T U
    Q = np.zeros_like(W); g = args.group; codes = np.zeros(W.shape, dtype=np.int64); SC = np.zeros((W.shape[0], cols // g), dtype=np.float32); BI = np.zeros_like(SC)
    for i1 in range(0, cols, args.block):
        i2 = min(i1 + args.block, cols); Wb = W[:, i1:i2].copy(); Err = np.zeros_like(Wb); Ub = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            if j % g == 0:                                  # grid for this group from the CURRENT (error-updated) weights
                grp = np.concatenate([Wb[:, i:min(i + g, i2 - i1)]] + ([W[:, i2:j + g]] if j + g > i2 else []), axis=1)
                if args.grid_from_original: grp = W0[:, j:j + g]
                lo, hi = grp.min(axis=1), grp.max(axis=1)
                if args.mse:                                # shrink the range: try p in [0.6, 1], keep the p with least squared error per row
                    best_err = None
                    for pfrac in np.linspace(1.0, 0.6, 21):
                        lo2, hi2 = lo * pfrac, hi * pfrac
                        e = ((quant_group(grp, lo2, hi2) - grp) ** 2).sum(axis=1)
                        if best_err is None: best_err, blo, bhi = e, lo2.copy(), hi2.copy()
                        else:
                            better = e < best_err; best_err = np.where(better, e, best_err); blo = np.where(better, lo2, blo); bhi = np.where(better, hi2, bhi)
                    lo, hi = blo, bhi
                if args.lloyd:                              # alternating assign / line-fit, multi-start; keep per row only if better
                    e_cur = ((quant_group(grp, lo, hi) - grp) ** 2).sum(axis=1)
                    lo0, hi0 = grp.min(axis=1), grp.max(axis=1)
                    for pfrac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                        l, st = lo0 * pfrac, np.maximum((hi0 - lo0) * pfrac / QMAX, 1e-8)
                        for _ in range(12):                 # 12 rounds capture >99.9% of the gain (grid_lloyd_iters.py)
                            k = np.clip(np.round((grp - l[:, None]) / st[:, None]), 0, QMAX) if TPL is None else TPL[nearest_tick(grp, l, st)]
                            kb = k.mean(axis=1, keepdims=True); xb = grp.mean(axis=1, keepdims=True); sxx = ((k - kb) ** 2).sum(axis=1)
                            ok = sxx > 0
                            a = np.where(ok, ((k - kb) * (grp - xb)).sum(axis=1) / np.where(ok, sxx, 1.0), st); a = np.maximum(a, 1e-8)
                            b = xb[:, 0] - a * kb[:, 0]
                            if np.allclose(a, st) and np.allclose(b, l): break
                            st, l = a, b
                        h = l + QMAX * st; e_new = ((quant_group(grp, l, h) - grp) ** 2).sum(axis=1)
                        better = e_new < e_cur; e_cur = np.where(better, e_new, e_cur); lo = np.where(better, l, lo); hi = np.where(better, h, hi)
                SC[:, j // g] = np.maximum((hi - lo) / QMAX, 1e-8); BI[:, j // g] = lo
            w = Wb[:, i]; d = Ub[i, i]
            q = quant_group(w[:, None], lo, hi)[:, 0]; Q[:, j] = q; codes[:, j] = np.clip(np.round((q - lo) / np.maximum((hi - lo) / QMAX, 1e-8)), 0, QMAX) if TPL is None else nearest_tick(w[:, None], lo, np.maximum((hi - lo) / QMAX, 1e-8))[:, 0]
            err = (w - q) / d
            Wb[:, i:] -= np.outer(err, Ub[i, i:]); Err[:, i] = err
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    return Q.astype(np.float32), codes, SC, BI

t0 = time.time(); mx.random.seed(args.seed); np.random.seed(args.seed)
model, tok = load(args.model)
ids = tok.encode(open("wikitext2_train.txt").read()); rng = np.random.default_rng(args.seed)
starts = rng.integers(0, len(ids) - args.seqlen, size=args.nsamples)
calib = mx.array(np.stack([ids[s:s + args.seqlen] for s in starts]))          # (nsamples, seqlen)
inner = model.model; h = inner.embed_tokens(calib); mask = create_attention_mask(h, None); mx.eval(h)

class Capture:
    """Wraps a Linear: passes calls through, accumulates H = X^T X of its inputs (float32)."""
    def __init__(self, lin): self.lin = lin; self.Hp = [None] * args.pieces; self.n = 0; self.calls = 0
    def __call__(self, x):
        X = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        HH = X.T @ X; mx.eval(HH); p = self.calls % args.pieces; self.calls += 1
        self.Hp[p] = HH if self.Hp[p] is None else self.Hp[p] + HH; self.n += X.shape[0]
        return self.lin(x)
    @property
    def H(self):
        tot = None
        for h in self.Hp:
            if h is not None: tot = h if tot is None else tot + h
        return tot

NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
def get(layer, name):
    obj = layer
    for part in name.split("."): obj = getattr(obj, part)
    return obj
def setm(layer, name, val):
    parts = name.split("."); obj = layer
    for part in parts[:-1]: obj = getattr(obj, part)
    setattr(obj, parts[-1], val)

BS = 8; CODES = {}; FLIPS = []
for li, layer in enumerate(inner.layers):
    caps = {n: Capture(get(layer, n)) for n in NAMES}
    for n, c in caps.items(): setm(layer, n, c)
    for b in range(0, args.nsamples, BS): mx.eval(layer(h[b:b + BS], mask, None))   # accumulate H on fp inputs of this layer
    if args.no_sequential:                                         # next layer's inputs from the UNQUANTIZED layer
        outs = [layer(h[b:b + BS], mask, None) for b in range(0, args.nsamples, BS)]; h_next = mx.concatenate(outs); mx.eval(h_next)
    for n, c in caps.items():
        lin = c.lin; W = np.array(lin.weight.astype(mx.float32)); Hn = np.array(c.H) / c.n
        Wq, codes, SC, BI = gptq(W, Hn)
        if args.refine > 0:
            from refine_cd import refine
            Hr = Hn.astype(np.float64); dead = np.diag(Hr) == 0; Hr = Hr / max(np.mean(np.diag(Hr)), 1e-30); Hr[np.diag_indices(len(Hr))] += args.refine_damp
            Wr = W.astype(np.float64).copy(); Wr[:, dead] = 0
            T = TPL if TPL is not None else np.arange(QMAX + 1.0)
            if args.order == "best":
                from refine_cd import refine_best_first, refit_grid
                if args.refine_z is not None:
                    md = max(np.mean(np.diag(Hn)), 1e-30); npc = c.n / args.pieces; Hs = []
                    for hp in [h for h in c.Hp if h is not None]:
                        a = np.array(hp).astype(np.float32); a /= np.float32(npc * md); a[np.diag_indices(len(a))] += np.float32(args.refine_damp); Hs.append(a)
                else: Hs = [Hr.astype(np.float32)]
                LOf = np.repeat(BI, args.group, axis=1).astype(np.float64); SCf = np.repeat(SC, args.group, axis=1).astype(np.float64); nfl = 0
                codes, Wq, f = refine_best_first(Wr, codes, LOf, SCf, Hs, T, z=args.refine_z, sweeps=args.refine); nfl += f
                for _r in range(args.refit):
                    LOf, SCf, BI, SC = refit_grid(Wr, codes, LOf, SCf, Hr, T, args.group)
                    codes, Wq, f = refine_best_first(Wr, codes, LOf, SCf, Hs, T, z=args.refine_z, sweeps=args.refine); nfl += f
                BI = LOf.reshape(len(Wr), -1, args.group)[:, :, 0].astype(np.float32); SC = SCf.reshape(len(Wr), -1, args.group)[:, :, 0].astype(np.float32); del Hs
            elif args.refine_z is not None:
                from refine_cd import refine_z
                md = max(np.mean(np.diag(Hn)), 1e-30); npc = c.n / args.pieces
                Hs = []
                for hp in [h for h in c.Hp if h is not None]:     # float32 pieces, built without float64 temporaries
                    a = np.array(hp).astype(np.float32); a /= np.float32(npc * md); a[np.diag_indices(len(a))] += np.float32(args.refine_damp); Hs.append(a)
                codes, Wq, nfl = refine_z(Wr, codes, np.repeat(BI, args.group, axis=1), np.repeat(SC, args.group, axis=1), Hs, T, z=args.refine_z, sweeps=args.refine); del Hs
            elif args.refit > 0:
                from refine_cd import refine_with_refit
                codes, Wq, SC, BI, nfl = refine_with_refit(Wr, codes, np.repeat(BI, args.group, axis=1).astype(np.float64), np.repeat(SC, args.group, axis=1).astype(np.float64), Hr, T, rounds=args.refit, sweeps=args.refine, group=args.group)
            else:
                codes, Wq, nfl = refine(Wr, codes, np.repeat(BI, args.group, axis=1), np.repeat(SC, args.group, axis=1), Hr, T, sweeps=args.refine)
            FLIPS.append(nfl / codes.size)
        lin.weight = mx.array(Wq).astype(mx.bfloat16); setm(layer, n, lin)
        CODES[f"model.layers.{li}.{n}"] = (codes.astype(np.uint8), SC, BI)
    if args.no_sequential: h = h_next
    else:
        outs = [layer(h[b:b + BS], mask, None) for b in range(0, args.nsamples, BS)]; h = mx.concatenate(outs); mx.eval(h)
    mx.clear_cache(); print(f"layer {li:2d} done  ({time.time() - t0:.0f}s)" + (f"  refine: {100 * np.mean(FLIPS[-len(NAMES):]):.1f}% of codes changed" if args.refine > 0 else ""), flush=True)
    if args.max_layers and li + 1 >= args.max_layers: print("smoke test finished, nothing saved"); sys.exit(0)

if args.eval:
    from ppl_wikitext import wikitext_ppl
    pp, ntok = wikitext_ppl(model, tok, 20); print(f"{args.out}: wikitext-2 perplexity {pp:.3f} over {ntok} tokens (20 windows of 2048)", flush=True)
if args.no_save: print(f"done, not saved  ({time.time() - t0:.0f}s)"); sys.exit(0)
from mlx_lm.utils import save_model, save_config
import pathlib, shutil
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
save_model(out, model); 
for f in pathlib.Path(args.model).glob("*"):
    if f.suffix != ".safetensors" and f.is_file() and not (out / f.name).exists(): shutil.copy(f, out / f.name)
np.savez_compressed(out / "gptq_codes.npz", **{k + "|codes": v[0] for k, v in CODES.items()}, **{k + "|scales": v[1] for k, v in CODES.items()}, **{k + "|biases": v[2] for k, v in CODES.items()})
print(f"saved dequantized {args.bits}-bit GPTQ model to {out}  ({time.time() - t0:.0f}s)")
