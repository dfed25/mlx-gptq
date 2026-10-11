"""Export a gptq_torch.py output (dequantized HF checkpoint + codes/) to AutoAWQ GEMM format so vLLM's int4 Marlin kernel
can run it. The affine grid's offset is rounded to an integer zero point (zero = round(-bias/scale)), so this is an
APPROXIMATION of the model; the mean relative weight change is printed.   python codes2awq.py SRC_DIR OUT_DIR"""
import sys, os, json, glob, shutil, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src, "config.json"))); G = None
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
def pack_awq(x):
    R, N = x.shape; o = np.zeros((R, N // 8), dtype=np.uint32); x = x.astype(np.uint32)
    for j, oo in enumerate(AWQ_ORDER): o |= x[:, oo::8] << (4 * j)
    return torch.from_numpy(o.view(np.int32))
wmap = {}; total = 0; stats = []
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(idx.values()))
for sh in shards:
    tensors = {}
    with safe_open(os.path.join(src, sh), "pt") as f:
        for k in f.keys(): tensors[k] = f.get_tensor(k)
    cf = os.path.join(src, "codes", sh)
    if os.path.exists(cf):
        with safe_open(cf, "pt") as f:
            names = sorted({k.rsplit(".", 1)[0] for k in f.keys()})
            for n in names:
                q = f.get_tensor(n + ".codes").numpy().astype(np.int64); S = f.get_tensor(n + ".scales").float().numpy(); B = f.get_tensor(n + ".biases").float().numpy()
                rows, K = q.shape; G = K // S.shape[1]; g = G
                z = f.get_tensor(n + ".zeros").numpy().astype(np.int64) if (n + ".zeros") in f.keys() else np.clip(np.round(-B / S), 0, 15).astype(np.int64)
                Wd = S.repeat(g, 1) * (q - z.repeat(g, 1)); Wx = S.repeat(g, 1) * q + B.repeat(g, 1)
                stats.append(float(np.abs(Wd - Wx).mean() / (np.abs(Wx).mean() + 1e-12)))
                tensors.pop(n + ".weight")
                tensors[n + ".qweight"] = pack_awq(q.T); tensors[n + ".qzeros"] = pack_awq(z.T); tensors[n + ".scales"] = torch.from_numpy(S.T.astype(np.float16).copy())
    save_file(tensors, os.path.join(out, sh), metadata={"format": "pt"})
    for k, v in tensors.items(): wmap[k] = sh; total += v.numel() * v.element_size()
    print("wrote", sh, flush=True)
json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
skip = ["lm_head"] + (["in_proj_a", "in_proj_b", "visual", "vision", "mtp"] if "text_config" in cfg else [])
cfg["quantization_config"] = {"quant_method": "awq", "bits": 4, "group_size": G, "zero_point": True, "version": "gemm", "modules_to_not_convert": skip}
if "text_config" in cfg: cfg["text_config"].pop("quantization_config", None)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in glob.glob(os.path.join(src, "*")):
    if os.path.isfile(f) and not f.endswith(".safetensors") and os.path.basename(f) not in ("config.json", "model.safetensors.index.json"): shutil.copy(f, out)
print(f"AWQ export of {src} -> {out}: {total/1e9:.2f} GB, group {G}, mean relative weight change from rounding zero points {np.mean(stats):.4f}")
