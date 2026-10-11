"""Offline rotation of a Qwen2/Llama-style HF checkpoint (QuaRot's weight-only part; arXiv 2404.00456):
  1. fuse RMSNorm gains into the following linear layers (norm weights become 1);
  2. rotate the residual stream by an orthogonal Q: embeddings E <- E Q, lm_head W <- W Q, q/k/v/gate/up W <- W Q,
     o/down W <- Q^T W  (y = W x with x = Q x' gives W' = W Q; contributions to the residual get Q^T);
  3. rotate attention values per head by a Hadamard R (d x d): v_proj rows and bias <- R v, o_proj columns <- R^T.
RMSNorm without gain commutes with orthogonal maps, RoPE acts on q/k outputs which are unchanged, and softmax mixes
tokens not channels, so the rotated model computes exactly the same function. Q = Hadamard * random signs / sqrt(n)
when a Hadamard of order n exists in this file (2^k, 2^k*12, 2^k*20), else a Haar-random orthogonal matrix.
    python rotate.py SRC OUT [--seed 0]          (streams shard by shard: works for 32B on 15 GB of RAM)"""
import argparse, json, os, glob, shutil, torch, numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--random-orth", action="store_true")
a = ap.parse_args(); src, out = a.src, a.out; os.makedirs(out, exist_ok=True)
if not os.path.isdir(src):
    from huggingface_hub import snapshot_download; src = snapshot_download(src)
cfg = json.load(open(os.path.join(src, "config.json"))); tc = cfg.get("text_config", cfg); PFX = "language_model." if "text_config" in cfg else ""
n = tc["hidden_size"]; L = tc["num_hidden_layers"]; nh = tc["num_attention_heads"]; nkv = tc.get("num_key_value_heads", nh); d = tc.get("head_dim", n // nh)
tied = cfg.get("tie_word_embeddings", False)

def paley(q):
    """Paley-I Hadamard of order q+1 for prime q = 3 mod 4."""
    res = {(i * i) % q for i in range(1, q)}; chi = lambda x: 0 if x % q == 0 else (1 if x % q in res else -1)
    Qm = np.array([[chi(j - i) for j in range(q)] for i in range(q)]); H = np.ones((q + 1, q + 1), dtype=np.int64)
    H[1:, 1:] = Qm + np.eye(q, dtype=np.int64); H[1:, 0] = -1; return H
def hadamard(m):
    """Hadamard of order m = 2^k * b for b in {1, 12, 20} (Sylvester doubling of a Paley base), else None."""
    for b in (20, 12, 1):
        if m % b == 0 and (m // b) & (m // b - 1) == 0:
            H = {1: np.ones((1, 1), dtype=np.int64), 12: paley(11), 20: paley(19)}[b]; k = (m // b).bit_length() - 1
            for _ in range(k): H = np.block([[H, H], [H, -H]])
            return H
    return None
def orth(size, gen):
    H = None if a.random_orth else hadamard(size)
    if H is None:
        M = torch.randn(size, size, generator=gen, dtype=torch.float64); Qm, R = torch.linalg.qr(M); Qm = Qm * torch.sign(torch.diag(R)); return Qm
    s = torch.randint(0, 2, (size,), generator=gen, dtype=torch.float64) * 2 - 1
    return torch.from_numpy(H.astype(np.float64)) * s[None, :] / np.sqrt(size)
gen = torch.Generator().manual_seed(a.seed); Q = orth(n, gen); R = orth(d, torch.Generator().manual_seed(a.seed + 1)); Rb = torch.block_diag(*([R] * nkv)); Rq = torch.block_diag(*([R] * nh))
assert torch.allclose(Q.T @ Q, torch.eye(n, dtype=torch.float64), atol=1e-9) and torch.allclose(R.T @ R, torch.eye(d, dtype=torch.float64), atol=1e-9)
print(f"rotation: n={n} ({'Hadamard' if hadamard(n) is not None and not a.random_orth else 'random orthogonal'}), head_dim={d}, layers={L}, heads={nh}/{nkv}, tied={tied}", flush=True)

idx = os.path.join(src, "model.safetensors.index.json")
wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {k: "model.safetensors" for f in glob.glob(os.path.join(src, "*.safetensors")) for k in safe_open(f, "pt").keys()}
files = {}
for k, f in wmap.items(): files.setdefault(f, []).append(k)
def get(k):
    with safe_open(os.path.join(src, wmap[k]), "pt") as sf: return sf.get_tensor(k).double()
key = lambda s: PFX + s
new_map = {}; total = 0
def write(name, tensors):
    global total
    t = {k: v.to(torch.float16).contiguous() for k, v in tensors.items()}; save_file(t, os.path.join(out, name), metadata={"format": "pt"})
    for k, v in t.items(): new_map[k] = name; total += v.numel() * 2
# embeddings + head
E = get(key("model.embed_tokens.weight")); gN = get(key("model.norm.weight"))
Wlm = (E if tied else get(key("lm_head.weight"))) * gN[None, :]
write("embed.safetensors", {key("model.embed_tokens.weight"): E @ Q, key("model.norm.weight"): torch.ones(n, dtype=torch.float64), key("lm_head.weight"): Wlm @ Q})
del E, Wlm
for li in range(L):
    p = key(f"model.layers.{li}."); T = {}
    g1 = get(p + "input_layernorm.weight"); g2 = get(p + "post_attention_layernorm.weight")
    for nm in ("q_proj", "k_proj", "v_proj"):
        W = get(p + f"self_attn.{nm}.weight") * g1[None, :]; W = W @ Q
        if nm == "v_proj": W = Rb @ W
        T[p + f"self_attn.{nm}.weight"] = W
        bk = p + f"self_attn.{nm}.bias"
        if bk in wmap: b = get(bk); T[bk] = (Rb @ b) if nm == "v_proj" else b
    T[p + "self_attn.o_proj.weight"] = Q.T @ (get(p + "self_attn.o_proj.weight") @ Rq.T)
    for nm in ("gate_proj", "up_proj"): T[p + f"mlp.{nm}.weight"] = (get(p + f"mlp.{nm}.weight") * g2[None, :]) @ Q
    T[p + "mlp.down_proj.weight"] = Q.T @ get(p + "mlp.down_proj.weight")
    T[p + "input_layernorm.weight"] = torch.ones(n, dtype=torch.float64); T[p + "post_attention_layernorm.weight"] = torch.ones(n, dtype=torch.float64)
    for k in files.get(wmap.get(p + "input_layernorm.weight"), []):
        if k.startswith(p) and k not in T and ".norm" in k: T[k] = get(k)         # q_norm/k_norm (Gemma3) pass through
    write(f"layer{li:03d}.safetensors", T); print(f"layer {li} rotated", flush=True) if li % 8 == 0 else None
json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
cfg2 = dict(tc); cfg2["tie_word_embeddings"] = False; cfg2["torch_dtype"] = "float16"; cfg2["dtype"] = "float16"
if PFX: cfg2["architectures"] = ["Gemma3ForCausalLM"]
json.dump(cfg2, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in glob.glob(os.path.join(src, "*")):
    b = os.path.basename(f)
    if os.path.isfile(f) and (b.startswith("tokenizer") or b in ("special_tokens_map.json", "generation_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "added_tokens.json")): shutil.copy(f, os.path.join(out, b))
torch.save({"Q": Q, "R": R, "seed": a.seed}, os.path.join(out, "rotation.pt"))
print(f"wrote rotated checkpoint {out}: {total/1e9:.2f} GB")
