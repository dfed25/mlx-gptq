"""Turn a model with DEQUANTIZED weights on an affine grid (output of gptq_mlx.py) into a real MLX quantized model:
recover each group's grid (bias = min, scale = (max-min)/(2^bits-1) over the group's achieved values, exact whenever the
group uses at least two distinct levels: we recover the step as the smallest positive difference between distinct
values, and the origin from the min), pack the codes with pack_affine.pack, and save with quantization config.
    python build_quantized.py DEQ_MODEL_DIR OUT_DIR BITS [--group 64]"""
import sys, argparse, pathlib, shutil, json, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import save_model
from pack_affine import pack

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("bits", type=int); ap.add_argument("--group", type=int, default=64)
ap.add_argument("--embed-bits", type=int, default=8, help="bits for the (tied) embedding / output projection; 8 costs nothing in quality, 4 costs ~0.5 perplexity")
a = ap.parse_args(); QMAX = 2 ** a.bits - 1; g = a.group

def to_codes(W):
    """W: (rows, K) float32 numpy on a per-group affine grid. Returns codes, scales, biases (rows, K/g)."""
    rows, K = W.shape; Wg = W.reshape(rows, K // g, g)
    lo = Wg.min(axis=2); hi = Wg.max(axis=2)
    # step: smallest positive gap between distinct values in the group; the grid has QMAX steps between lo and lo+QMAX*step
    srt = np.sort(Wg, axis=2); gaps = np.diff(srt, axis=2); gaps[gaps <= 1e-7] = np.inf; step = gaps.min(axis=2)
    step = np.where(np.isinf(step), 1.0, step)                       # constant group: any step, all codes 0
    # the achieved values are lo + k*step with k in [0, QMAX] only if (hi-lo)/step <= QMAX; otherwise our step guess is a multiple: refine
    k_span = np.round((hi - lo) / step)
    bad = k_span > QMAX
    step = np.where(bad, (hi - lo) / QMAX, step)                    # fall back to full-range grid (may re-round slightly)
    q = np.clip(np.round((Wg - lo[..., None]) / step[..., None]), 0, QMAX)
    err = np.abs(lo[..., None] + q * step[..., None] - Wg).max()
    return q.reshape(rows, K).astype(np.int64), step.astype(np.float32), lo.astype(np.float32), float(err), int(bad.sum())

model, tok = load(a.src)
n_lin = 0; worst = 0.0; nbad = 0
npz = pathlib.Path(a.src) / "gptq_codes.npz"; EXACT = dict(np.load(npz)) if npz.exists() else None
print("exact codes from gptq_codes.npz:", EXACT is not None)
def convert(module):
    global n_lin, worst, nbad
    for name, child in list(module.named_modules()):
        pass
for li, layer in enumerate(model.model.layers):
    for path in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        parent = layer; parts = path.split(".")
        for p in parts[:-1]: parent = getattr(parent, p)
        lin = getattr(parent, parts[-1])
        W = np.array(lin.weight.astype(mx.float32))
        key = f"model.layers.{li}.{path}"
        if EXACT is not None and key + "|codes" in EXACT:
            q, s, b, err, bad = EXACT[key + "|codes"].astype(np.int64), EXACT[key + "|scales"], EXACT[key + "|biases"], 0.0, 0
        else:
            q, s, b, err, bad = to_codes(W)
        worst = max(worst, err); nbad += bad
        ql = nn.QuantizedLinear(W.shape[1], W.shape[0], bias="bias" in lin, group_size=g, bits=a.bits)
        ql.weight = mx.array(pack(q, a.bits)); ql.scales = mx.array(s).astype(mx.float16); ql.biases = mx.array(b).astype(mx.float16)
        if "bias" in lin: ql.bias = lin.bias
        setattr(parent, parts[-1], ql); n_lin += 1
# the embedding (tied to the output projection) is read every token: quantize it too (8-bit by default: free in quality)
emb = model.model.embed_tokens; EB = a.embed_bits
if isinstance(emb, nn.Embedding):
    qe = nn.QuantizedEmbedding.from_embedding(emb, group_size=g, bits=EB); model.model.embed_tokens = qe
    print(f"embedding quantized to {EB} bits")
print(f"converted {n_lin} linear layers; max reconstruction error {worst:.2e}; groups needing full-range fallback {nbad}")
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
save_model(out, model)
cfg = json.load(open(pathlib.Path(a.src) / "config.json")); cfg["quantization"] = {"group_size": g, "bits": a.bits}
if EB != a.bits: cfg["quantization"]["model.embed_tokens"] = {"group_size": g, "bits": EB}
cfg["quantization_config"] = cfg["quantization"]
json.dump(cfg, open(out / "config.json", "w"), indent=2)
for f in pathlib.Path(a.src).glob("*"):
    if f.suffix not in (".safetensors", ".npz") and f.is_file() and f.name != "config.json" and not (out / f.name).exists(): shutil.copy(f, out / f.name)
print("saved", out)
