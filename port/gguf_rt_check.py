"""Round-trip check: every tensor in a packed GGUF must dequantize to the HF safetensors it was built from.
usage: gguf_rt_check.py packed.gguf hf_model_dir"""
import sys, os, json, glob, numpy as np, torch, gguf
from gguf import GGUFReader, dequantize
from safetensors import safe_open
G, M = sys.argv[1], sys.argv[2]
hf = {}
for f in sorted(glob.glob(os.path.join(M, "*.safetensors"))):
    with safe_open(f, "pt") as s:
        for k in s.keys(): hf[k] = f
def load(k):
    with safe_open(hf[k], "pt") as s: return s.get_tensor(k).float()
def hfname(n):
    if n == "token_embd.weight": return "model.embed_tokens.weight"
    if n == "output.weight": return "lm_head.weight" if "lm_head.weight" in hf else "model.embed_tokens.weight"
    if n == "output_norm.weight": return "model.norm.weight"
    p = n.split("."); li = p[1]; rest = ".".join(p[2:])
    m = {"attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
         "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
         "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    base, suf = rest.rsplit(".", 1)
    if rest in m: return f"model.layers.{li}.{m[rest]}"
    return f"model.layers.{li}.{m[base]}.{suf}" if base in m else None
r = GGUFReader(G); worst = 0; n = 0; bad = []
for t in r.tensors:
    k = hfname(t.name)
    if k is None or k not in hf: print("SKIP", t.name, k); continue
    a = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32)).reshape(list(reversed([int(x) for x in t.shape])))
    b = load(k)
    if a.shape != b.shape: bad.append((t.name, "shape", tuple(a.shape), tuple(b.shape))); continue
    d = (a - b).abs().max().item(); rel = d / (b.abs().max().item() + 1e-12); n += 1; worst = max(worst, rel)
    if rel > 1e-3: bad.append((t.name, t.tensor_type.name, d, rel))
print(f"checked {n} tensors, worst rel max-abs diff {worst:.2e}")
for x in bad: print("BAD", x)
print("RT OK" if not bad else "RT FAIL")
