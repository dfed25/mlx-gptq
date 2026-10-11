"""Build a --kq-map for gptq_torch.py from the per-tensor types of a reference GGUF (unsloth's "dynamic" allocation), so our recipe can be
run with THEIR allocation: Q4_K / IQ4_XS / IQ4_NL -> q4_k, Q5_K -> q5_k, Q6_K / Q8_0 -> q6_k, Q3_K / IQ3_S -> q3_k. Prints the bits per
weight of the mapped allocation (K-quant sizes) next to the reference file's own, so the size difference is known.
    python unsloth_map.py REF.gguf MODEL_DIR OUT.json"""
import sys, re, json, collections, gguf
from safetensors import safe_open
from kquant import KQ
ref, model, out = sys.argv[1:4]
G2H = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
       "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj"}
T2K = {"Q4_K": "q4_k", "IQ4_XS": "q4_k", "IQ4_NL": "q4_k", "Q5_K": "q5_k", "Q6_K": "q6_k", "Q8_0": "q6_k", "Q3_K": "q3_k", "IQ3_S": "q3_k"}
REFBPW = {"Q4_K": 4.5, "IQ4_XS": 4.25, "IQ4_NL": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5, "Q3_K": 3.4375, "IQ3_S": 3.44}
r = gguf.GGUFReader(ref); m = {}; ours = theirs = n = 0; stats = collections.Counter()
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]; opened = {}
def numel(key):
    f = idx[key]
    if f not in opened: opened[f] = safe_open(f"{model}/{f}", "pt")
    s = opened[f].get_slice(key).get_shape(); return s[0] * s[1]
for t in r.tensors:
    mm = re.match(r"blk\.(\d+)\.(\w+)\.weight$", t.name)
    if not mm or mm.group(2) not in G2H: continue
    li, g = int(mm.group(1)), mm.group(2); hf = f"model.layers.{li}.{G2H[g]}"; kind = T2K[t.tensor_type.name]
    if f"model.language_model.layers.{li}.{G2H[g]}.weight" not in idx: continue                    # blk.64 is the MTP (nextn) layer, not part of the text model
    m[f"^{re.escape(hf)}$"] = kind; k = numel(f"model.language_model.layers.{li}.{G2H[g]}.weight")
    ours += KQ[kind]["bpw"] * k; theirs += REFBPW[t.tensor_type.name] * k; n += k; stats[(t.tensor_type.name, kind)] += 1
json.dump(m, open(out, "w"), indent=0)
print(f"{len(m)} linears; reference allocation {theirs / n:.3f} bpw over the linears, K-quant version {ours / n:.3f} bpw ({(ours - theirs) / 8 / 2**20:+.0f} MiB)")
print("type mapping counts:", dict(stats))
