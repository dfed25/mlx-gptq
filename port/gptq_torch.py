"""PyTorch/CUDA port of Dom's mlx-gptq recipe (github.com/dfed25/mlx-gptq: gptq_mlx.py + refine_cd.py), streaming the
model through the GPU one decoder layer at a time so a 27B model can be quantized on a 24 GB card.
Same algorithm, same defaults, same numerics (float64 for GPTQ and the refit, float32 for the refinement):
  sequential GPTQ (each layer calibrated on the already-quantized layers before it), group-wise affine grid
  (w = scale*q + bias), --mse range search, --lloyd alternating grid fit, --refine coordinate descent on the codes in the
  layer metric, --refit exact least-squares refit of every group's grid alternated with the refinement.
    python gptq_torch.py MODEL OUT BITS [--group 64] [--nsamples 128] [--seqlen 512] [--mse] [--lloyd] [--refine 3] [--refit 2]
Writes OUT as an HF float16 checkpoint with the DEQUANTIZED weights (loads in vLLM/transformers unchanged) and
OUT/codes/*.safetensors with the exact codes, scales and biases per linear layer."""
import argparse, time, json, os, math, glob, torch, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from safetensors.torch import save_file
from stream_util import Streamer, ShardWriter
from quant_core import *

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("bits", type=int)
ap.add_argument("--group", type=int, default=64); ap.add_argument("--nsamples", type=int, default=128); ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--block", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mse", action="store_true"); ap.add_argument("--lloyd", action="store_true")
ap.add_argument("--refine", type=int, default=0); ap.add_argument("--refit", type=int, default=0); ap.add_argument("--refine-damp", type=float, default=0.01)
ap.add_argument("--dtype", default="float16"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-layers", type=int, default=0)
ap.add_argument("--calib", default="wikitext2_train.txt")
ap.add_argument("--sym8", action="store_true", help="8-bit layers use a fixed zero point of 128 (symmetric, packs to GGUF Q8_0)")
ap.add_argument("--bits-map", default=None, help="JSON {regex: bits} overriding BITS per linear (first match wins; names are the text-model parameter names)")
ap.add_argument("--refit-metric", default="full", choices=["full", "diag", "avg", "max", "bd", "pool"], help="metric for the grid refit: full H (Dom); diag(H); diag pooled over --refit-block neighbours (avg/max); block-diagonal H (bd); cross terms pooled per block tile with exact diagonal (pool)")
ap.add_argument("--refit-block", type=int, default=16)
ap.add_argument("--order", default="none", choices=["none", "act", "perm", "permall"], help="column ordering by usage (diag of the Hessian, descending): act = quantize in that order, groups stay in the original column space (static grids; any format); perm = also cluster columns into groups in that order on the dimensions a checkpoint can physically permute (down_proj input, o_proj input within kv-head groups, hidden dim with --global-order); permall = every linear in its own order (upper bound, not deployable)")
ap.add_argument("--global-order", default=None, help="torch file with one hidden-dim order shared by all layers (from hidden_order.py), used by --order perm")
ap.add_argument("--kq", default=None, choices=["q4_k", "q5_k", "q3_k", "q2_k", "q6_k"], help="fit the grids under the llama.cpp K-quant constraint of this type (sets group = sub-block size, bits, and the fixed zero for the symmetric types); codes pack exactly into that GGUF type")
ap.add_argument("--kq-map", default=None, help="JSON {regex: K-quant kind} per linear (first match wins; names are the text-model parameter names), overriding --kq; each linear then uses that kind's sub-block size, levels and zero")
ap.add_argument("--token-weights", default=None, help="yaqa_weights.py file: per-linear, per-token weights w_t for the Hessian H = sum_t w_t x_t x_t^T (YAQA sketch, B = I); must come from the same calib / nsamples / seqlen / seed")
ap.add_argument("--codes-only", action="store_true", help="do not write the dequantized checkpoint (54 GB for a 27B): OUT gets the codes plus symlinks to the source files, which is all codes2gguf.py needs")
ap.add_argument("--imatrix", default=None, help="torch file {hf linear name: mean x^2 per input column} (imatrix_tools.py list): rescale each linear's Hessian so its diagonal matches these statistics (H' = s H s, s = sqrt(theirs / ours)); the correlation structure stays ours. A proxy for 'our recipe on their calibration data'")
ap.add_argument("--int-zero", action="store_true", help="constrain every group's offset to an integer zero point (offset = -z*step, z in 0..2^bits-1): packs losslessly into AWQ/GPTQ formats; the grid fit and the refit are done under this constraint")
args = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
import re
BITS_MAP = [(re.compile(k), int(v)) for k, v in (json.load(open(args.bits_map)).items() if args.bits_map else [])]
MAPTXT = " + map" if BITS_MAP else ""
def bits_for(name):
    for rx, b in BITS_MAP:
        if rx.search(name): return b
    return args.bits

# ---------------------------------------------------------------- per-linear recipe (in whatever column space the caller hands in)
def quantize_linear(W, Hn, QMAX, FZ, static, perm=None, inv=None, group=None, int_zero=None, kq=None):
    group = args.group if group is None else group; int_zero = args.int_zero if int_zero is None else int_zero; kq = args.kq if kq is None and not KQMAP else kq
    """W float32 (rows, cols), Hn float32 Hessian (mean over tokens). static/perm/inv: act-order mode (GPTQ runs on the permuted
    columns with grids fixed per original group, then everything is mapped back before refine/refit). Returns Wq, codes, SC, BI, nfl."""
    if perm is not None: Hbox = [Hn[perm][:, perm].double()]; Win = W[:, perm].contiguous()
    else: Hbox = [Hn.double()]; Win = W
    Hn_cpu = Hn.cpu(); del Hn; empty_cache()
    Wq, codes, SC, BI = gptq(Win, Hbox, QMAX, group, args.damp, args.block, args.mse, args.lloyd, int_zero, FZ, static, kq); empty_cache()
    if perm is not None: Wq = Wq[:, inv].contiguous(); codes = codes[:, inv].contiguous(); SC = static["scale"].float(); BI = static["lo"].float(); del Win
    Hn = Hn_cpu.to(DEV); nfl = 0
    if args.refine > 0:
        dead = torch.diag(Hn) == 0; Hn /= max(torch.diag(Hn).mean().item(), 1e-30); Hn.diagonal().add_(args.refine_damp)
        Wr = W.clone(); Wr[:, dead] = 0; T = torch.arange(QMAX + 1.0, device=DEV)
        if args.refit > 0:
            out = refine_with_refit(Wr, codes, BI.double(), SC.double(), Hn, T, QMAX, rounds=args.refit, sweeps=args.refine, group=group, int_zero=int_zero, metric=args.refit_metric, block=args.refit_block, fixed_zero=FZ, kq=kq)
            codes, Wq, SC, BI, nfl = out[:5]; KQMETA[0] = out[5] if len(out) > 5 else None
        else:
            codes, nfl = refine(Wr, codes, BI.double(), SC.double(), Hn, T, QMAX, group, sweeps=args.refine); Wq = expand(BI, group) + expand(SC, group) * T.float()[codes.long()]
        del Wr
    del Hn; return Wq, codes, SC, BI, nfl

GLOBAL = torch.load(args.global_order).to(DEV) if args.global_order else None
IMAT = torch.load(args.imatrix) if args.imatrix else None
TW = None
if args.token_weights:
    _tw = torch.load(args.token_weights); assert _tw["nsamples"] == args.nsamples and _tw["seqlen"] == args.seqlen and _tw["seed"] == args.seed, "token weights were made for different windows"
    TW = {k: v / v.mean() for k, v in _tw["weights"].items()}; print(f"token weights from {args.token_weights}: {len(TW)} linears (YAQA sketch, B = I)")
def rescale_h(name, Hn):
    """Column-rescale Hn so that diag(Hn') = the imatrix diagonal for this linear (in place); returns the geometric-mean scale for the log."""
    if IMAT is None or name not in IMAT: return 1.0
    u = IMAT[name].to(Hn.device, torch.float32); d = torch.diag(Hn).clamp_min(1e-12); sc = torch.sqrt(u.clamp_min(1e-12) / d)
    Hn.mul_(sc[:, None]).mul_(sc[None, :]); return float(torch.exp(torch.log(sc).mean()))
KQMETA = [None]
from kquant import KQ
KQMAP = [(re.compile(k), v) for k, v in (json.load(open(args.kq_map)).items() if args.kq_map else [])]
if args.kq and not KQMAP:
    KQS = KQ[args.kq]; assert args.group == KQS["sb"], f"--kq {args.kq} needs --group {KQS['sb']}"; args.int_zero = KQS["zfix"] is not None
def kq_for(name):
    for rx, k in KQMAP:
        if rx.search(name): return k
    return args.kq
def order_for(n, lin, Hn):
    """Column order (descending usage = diag of the Hessian) for this linear under --order, or None."""
    if args.order == "none": return None
    d = torch.diag(Hn)
    if args.order in ("act", "permall"): return torch.argsort(d, descending=True)
    cfg = S.text_cfg; hidden = cfg.hidden_size; nq = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", nq) or nq; hd = getattr(cfg, "head_dim", None) or hidden // nq
    if n.endswith("down_proj"): return torch.argsort(d, descending=True)                               # intermediate dim: permute gate/up rows to match
    if n.endswith("o_proj") and lin.in_features == nq * hd:                                              # within each kv group the same order of the head_dim positions (v_proj rows follow)
        r = nq // nkv; D = d.view(nkv, r, hd).sum(1); o = torch.argsort(D, dim=1, descending=True)
        base = torch.arange(nq * hd, device=d.device).view(nkv, r, hd); return torch.gather(base, 2, o[:, None, :].expand(nkv, r, hd)).reshape(-1)
    if GLOBAL is not None and lin.in_features == hidden: return GLOBAL
    return None

# ---------------------------------------------------------------- driver
t0 = time.time(); torch.manual_seed(args.seed)
S = Streamer(args.model, dtype=getattr(torch, args.dtype), device=DEV); tok = S.tok
ids = tok.encode(open(args.calib).read()); rng = np.random.default_rng(args.seed)
starts = rng.integers(0, len(ids) - args.seqlen, size=args.nsamples)
calib = torch.tensor(np.stack([ids[s:s + args.seqlen] for s in starts]))
KW = S.capture_kwargs(calib[:args.bs]); hs = [S.embed(calib[b:b + args.bs]) for b in range(0, args.nsamples, args.bs)]; S.offload_head()
W_out = ShardWriter(args.out, S); os.makedirs(os.path.join(args.out, "codes"), exist_ok=True)
print(f"{args.model}: {S.n_layers} layers order={args.order} kq={args.kq}{' + map' if KQMAP else ''} ({S.layout} layout, types {sorted(set(S.layer_types))}), calib {calib.shape}, dtype {args.dtype}, bits {args.bits}{MAPTXT} group {args.group} mse={args.mse} lloyd={args.lloyd} refine={args.refine} refit={args.refit} metric={args.refit_metric}/{args.refit_block}", flush=True)
FLIPS = []; BITS_USED = {}
for li in range(S.n_layers):
    tl = time.time(); layer = S.load_layer(li); kw = S.kw_for(li, KW); LINEARS = S.lin_names(layer); lins = {n: layer.get_submodule(n) for n in LINEARS}
    acc = {n: {"H": torch.zeros((m.in_features, m.in_features), dtype=torch.float32, device=DEV), "n": 0, "pos": 0} for n, m in lins.items()}
    def mk(n):
        def hook(mod, inp, out):
            X = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if TW is not None:                                                                  # YAQA: weight each token by the end-to-end gradient energy at this linear's output
                w = TW[f"model.layers.{li}.{n}"][acc[n]["pos"]:acc[n]["pos"] + X.shape[0]].to(X.device); acc[n]["pos"] += X.shape[0]
                acc[n]["H"].addmm_((X * w[:, None]).T, X); acc[n]["n"] += w.sum().item()
            else: acc[n]["H"].addmm_(X.T, X); acc[n]["n"] += X.shape[0]
        return hook
    hds = [lins[n].register_forward_hook(mk(n)) for n in LINEARS]
    for h in hs: S.run_layer(layer, h, kw)
    for hd in hds: hd.remove()
    codes_out = {}
    for n in LINEARS:
        lin = lins[n]; W = lin.weight.data.float(); Hn = acc[n]["H"] / acc[n]["n"]; acc[n]["H"] = None; rescale_h(f"model.layers.{li}.{n}", Hn)
        b = bits_for(f"model.layers.{li}.{n}"); QMAX = float(2 ** b - 1); BITS_USED[f"model.layers.{li}.{n}"] = b
        FZ = 128 if (args.sym8 and b == 8) else None; G = args.group; IZ = args.int_zero; KQK = kq_for(f"model.layers.{li}.{n}") if (args.kq or KQMAP) else None
        if KQK: k = KQ[KQK]; QMAX = float(k["qmax"]); FZ = k["zfix"]; G = k["sb"]; IZ = FZ is not None; b = {"q2_k": 2, "q3_k": 3, "q4_k": 4, "q5_k": 5, "q6_k": 6}[KQK]; BITS_USED[f"model.layers.{li}.{n}"] = KQK
        p = order_for(n, lin, Hn)
        if p is None:
            Wq, codes, SC, BI, nfl = quantize_linear(W, Hn, QMAX, FZ, None, group=G, int_zero=IZ, kq=KQK)
        elif args.order == "act":                                                       # permuted GPTQ with grids fixed per original group, refine/refit in the original space
            g = G; W64 = W.double(); ng = W.shape[1] // g
            SLO = torch.empty((W.shape[0], ng), **F64); SSC = torch.empty_like(SLO)
            for gi in range(ng):
                lo, hi = grid_for_group(W64[:, gi * g:(gi + 1) * g], QMAX, args.mse, args.lloyd, IZ, FZ); SLO[:, gi] = lo; SSC[:, gi] = ((hi - lo) / QMAX).clamp_min(1e-8)
            del W64; inv = torch.argsort(p)
            Wq, codes, SC, BI, nfl = quantize_linear(W, Hn, QMAX, FZ, dict(lo=SLO, scale=SSC, gid=(p // g).tolist()), perm=p, inv=inv, group=G, int_zero=IZ, kq=KQK)
        else:                                                                             # fully permuted: groups contiguous in usage order, refine/refit there, un-permute the result
            inv = torch.argsort(p); Wp = W[:, p].contiguous(); Hp = Hn[p][:, p].contiguous(); del Hn
            Wq, codes, SC, BI, nfl = quantize_linear(Wp, Hp, QMAX, FZ, None, group=G, int_zero=IZ, kq=KQK); Wq = Wq[:, inv].contiguous(); del Wp, Hp
        if args.refine > 0: FLIPS.append(nfl / codes.numel())
        lin.weight.data = Wq.to(lin.weight.dtype)
        ck = S.ckpt_key(f"model.layers.{li}.{n}")
        codes_out[ck + ".codes"] = codes.to(torch.uint8).cpu(); codes_out[ck + ".scales"] = SC.half().cpu(); codes_out[ck + ".biases"] = BI.half().cpu()
        if IZ: codes_out[ck + ".zeros"] = (-BI / SC).round().clamp(0, QMAX).to(torch.uint8).cpu()
        codes_out[ck + ".bits"] = torch.tensor([b], dtype=torch.uint8)
        if KQK and KQMETA[0] is not None:
            mt = KQMETA[0]; codes_out[ck + ".kd"] = mt["d"].half().cpu(); codes_out[ck + ".kdmin"] = mt["dmin"].half().cpu(); codes_out[ck + ".ksc"] = mt["sc"].cpu(); codes_out[ck + ".km"] = mt["m"].cpu(); KQMETA[0] = None
        if p is not None and args.order != "act": codes_out[ck + ".perm"] = p.to(torch.int32).cpu()          # codes/scales are in the permuted column space
        del W, Wq, codes, SC, BI
        empty_cache(); print(f"    {n}: {mem_gb()}" + (f"  order {args.order}" if p is not None else ""), flush=True)
    hs = [S.run_layer(layer, h, kw) for h in hs]                                   # next layer's inputs from the QUANTIZED layer
    if not args.codes_only: W_out.write(f"layer{li:03d}", {n: layer.get_parameter(n[len(f'model.layers.{li}.'):]).data.half() for n in S.layer_param_names(li)})
    save_file(codes_out, os.path.join(args.out, "codes", f"layer{li:03d}.safetensors"))
    S.free_layer(li); del acc
    print(f"layer {li:2d} done  ({time.time() - t0:.0f}s, {time.time() - tl:.0f}s this layer)" + (f"  refine: {100 * np.mean(FLIPS[-len(LINEARS):]):.1f}% of codes changed" if args.refine else "") + f"  mem {mem_gb()}", flush=True)
    if args.max_layers and li + 1 >= args.max_layers: print("smoke test finished (partial checkpoint not usable)"); raise SystemExit(0)
if args.codes_only:
    for f in glob.glob(os.path.join(S.path, "*")):
        dst = os.path.join(args.out, os.path.basename(f))
        if not os.path.exists(dst): os.symlink(os.path.abspath(f), dst)
else:
    rest = {n: p.data.half().cpu() for n, p in S.model.named_parameters() if not n.startswith("model.layers.") and not (n == "lm_head.weight" and S.tied)}
    W_out.write("embed", rest); W_out.finish()
json.dump(BITS_USED, open(os.path.join(args.out, "bits_used.json"), "w"), indent=1)
print(f"saved dequantized {args.bits}-bit model to {args.out}  ({time.time() - t0:.0f}s)")
