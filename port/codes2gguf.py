"""Pack a gptq_torch.py output (dequantized HF checkpoint + codes/) into a llama.cpp GGUF, losslessly: Dom's affine grid at
group 32 with integer zero points IS the Q4_1 / Q5_1 block (w = d*q + m with m = -z*d), and 8-bit layers with the fixed
zero point 128 are Q8_0 (w = d*(q-128)). Everything else (embeddings, head, norms, conv, decay) is written in F16 after the
llama.cpp converter's transforms (value-head reorder, -exp(A_log), +1 on norms for the Qwen3.5 family). All metadata
(hyperparameters, tokenizer) is copied from a reference GGUF of the same model.
    python codes2gguf.py MODEL_DIR REFERENCE.gguf OUT.gguf [--embed-q8]"""
import sys, os, json, re, argparse, numpy as np, torch, gguf
from safetensors import safe_open
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType as Q

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ref"); ap.add_argument("out"); ap.add_argument("--embed-q8", action="store_true", help="store token embeddings and output head as Q8_0 (symmetric round-to-nearest) instead of F16")
ap.add_argument("--temp-file", action="store_true", help="stream tensor data through a temp file instead of holding the whole model in RAM (gguf-py buffers every tensor until the final write: 16 GB for a 27B; needed on a 15 GB box)")
ap.add_argument("--head-kq", default=None, choices=["q6_k", "q5_k", "q4_k"], help="K-quant type for the output head when it should differ from --embed-kq (unsloth: Q4_K embeddings, Q6_K head)")
ap.add_argument("--head-codes", default=None, help="head_gptq.py codes file: pack the output head from these GPTQ codes (its K-quant kind) instead of round-to-nearest")
ap.add_argument("--embed-kq", default=None, choices=["q6_k", "q5_k", "q4_k"], help="store token embeddings and output head as this K-quant type (Dom's grid fit per group, super-block projected; what bartowski/unsloth do)")
a = ap.parse_args(); M = a.model
cfg = json.load(open(os.path.join(M, "config.json"))); tc = cfg.get("text_config", cfg); vl = "text_config" in cfg
PFX = "model.language_model." if vl else "model."
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
def hf(key):
    with safe_open(os.path.join(M, idx[key]), "pt") as f: return f.get_tensor(key)
class LazyCodes:
    """key -> tensor, loaded from the codes shards on demand (a 27B's codes are ~26 GB; never hold them all)."""
    def __init__(self, d):
        self.where = {}
        for f in sorted(os.listdir(d)):
            with safe_open(os.path.join(d, f), "pt") as sf:
                for k in sf.keys(): self.where[k] = os.path.join(d, f)
    def __contains__(self, k): return k in self.where
    def __iter__(self): return iter(self.where)
    def items(self):
        for k in self.where: yield k, self[k]
    def __getitem__(self, k):
        with safe_open(self.where[k], "pt") as sf: return sf.get_tensor(k)
codes = LazyCodes(os.path.join(M, "codes"))
quant = {k[:-6] for k in codes if k.endswith(".codes")}
PERM = {k[:-5]: codes[k].numpy().astype(np.int64) for k in codes if k.endswith(".perm")}       # usage-clustered models: codes are in a permuted column order
GPERM = next((v for k, v in PERM.items() if k.endswith("self_attn.q_proj")), None)                  # the shared hidden-dim order (same for q/k/v/gate/up)
if PERM: print(f"permuted model: {len(PERM)} linears carry a column order; hidden-dim order {'shared' if GPERM is not None else 'none'}")
nk = tc.get("linear_num_key_heads", 0) or 0; nv = tc.get("linear_num_value_heads", 0) or 0; hk = tc.get("linear_key_head_dim", 0) or 0; hv = tc.get("linear_value_head_dim", 0) or 0
reorder = nk > 0 and nv > 0 and nk != nv; vpk = nv // nk if reorder else 1
def tile(x, dim, head_dim):                      # grouped -> tiled (the converter's _reorder_v_heads)
    if not reorder: return x
    shape = list(x.shape); dim = dim % len(shape); new = shape[:dim] + [nk, vpk, head_dim] + shape[dim + 1:]
    y = x.reshape(*new); perm = list(range(len(new))); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return y.permute(*perm).contiguous().reshape(*shape)

# ---- block packers (little-endian), x: codes (rows, cols) uint8, s/b: (rows, cols/32) fp16
def pack_q4_1(q, s, b):
    rows, cols = q.shape; nb = cols // 32; q = q.reshape(rows, nb, 32).astype(np.uint8)
    qs = (q[:, :, :16] & 0xF) | ((q[:, :, 16:] & 0xF) << 4)
    out = np.zeros((rows, nb, 20), dtype=np.uint8); out[:, :, 0:2] = s.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2); out[:, :, 2:4] = b.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2); out[:, :, 4:] = qs
    return out.reshape(rows, nb * 20)
def pack_q5_1(q, s, b):
    rows, cols = q.shape; nb = cols // 32; q = q.reshape(rows, nb, 32).astype(np.uint32)
    qs = ((q[:, :, :16] & 0xF) | ((q[:, :, 16:] & 0xF) << 4)).astype(np.uint8)
    qh = np.zeros((rows, nb), dtype=np.uint32)
    for j in range(32): qh |= ((q[:, :, j] >> 4) & 1) << j
    out = np.zeros((rows, nb, 24), dtype=np.uint8); out[:, :, 0:2] = s.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2); out[:, :, 2:4] = b.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 4:8] = qh.astype("<u4").view(np.uint8).reshape(rows, nb, 4); out[:, :, 8:] = qs
    return out.reshape(rows, nb * 24)
def pack_q8_0(q, s):
    rows, cols = q.shape; nb = cols // 32; qi = (q.reshape(rows, nb, 32).astype(np.int16) - 128).astype(np.int8)
    out = np.zeros((rows, nb, 34), dtype=np.uint8); out[:, :, 0:2] = s.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2); out[:, :, 2:] = qi.view(np.uint8)
    return out.reshape(rows, nb * 34)
def q8_rtn(w):                                   # symmetric round-to-nearest Q8_0 for embeddings/head when asked
    rows, cols = w.shape; nb = cols // 32; wb = w.reshape(rows, nb, 32).astype(np.float32); d = np.abs(wb).max(2) / 127.0; d[d == 0] = 1e-8
    q = np.clip(np.round(wb / d[:, :, None]), -127, 127).astype(np.int8)
    out = np.zeros((rows, nb, 34), dtype=np.uint8); out[:, :, 0:2] = d.astype("<f2", order="C").view(np.uint8).reshape(rows, nb, 2); out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(rows, nb * 34)

def kq_payload(q, kd, kdm, ksc, km, bits):
    """Exact K-quant pack of codes (rows, cols) with the super-block metadata (kd/kdm per 256, ksc/km per sub-block); the kind follows from bits and the sub-block size."""
    from kquant import pack_q4_k, pack_q5_k, pack_q3_k, pack_q2_k, pack_q6_k
    g = q.shape[1] // ksc.shape[1]; kind = {(4, 32): "q4_k", (5, 32): "q5_k", (3, 16): "q3_k", (2, 16): "q2_k", (6, 16): "q6_k"}[(bits, g)]
    if kind == "q4_k": return pack_q4_k(q, kd, kdm, ksc, km), q.shape, Q.Q4_K
    if kind == "q5_k": return pack_q5_k(q, kd, kdm, ksc, km), q.shape, Q.Q5_K
    if kind == "q3_k": return pack_q3_k(q, kd, ksc), q.shape, Q.Q3_K
    if kind == "q2_k": return pack_q2_k(q, kd, kdm, ksc, km), q.shape, Q.Q2_K
    if kind == "q6_k": return pack_q6_k(q, kd, ksc), q.shape, Q.Q6_K

def quantized(hfname, rowperm=None, colperm=None):
    """GGUF payload for a quantized linear: (bytes, logical shape, qtype)."""
    hfname = hfname[:-7] if hfname.endswith(".weight") else hfname                   # codes are keyed by the linear's name, without .weight
    q = codes[hfname + ".codes"].numpy(); s = codes[hfname + ".scales"].numpy(); b = codes[hfname + ".biases"].numpy(); bits = int(codes[hfname + ".bits"][0]); g = q.shape[1] // s.shape[1]
    if rowperm is not None: q, s, b = q[rowperm], s[rowperm], b[rowperm]
    if colperm is not None:
        q = q[:, colperm]; gp = colperm.reshape(-1, g)[:, 0] // g; s, b = s[:, gp], b[:, gp]
    q, s, b = np.ascontiguousarray(q), np.ascontiguousarray(s), np.ascontiguousarray(b)
    if hfname + ".kd" in codes and colperm is None:                                   # K-quant constrained grids: exact pack from the super-block metadata (a column permutation would scatter the super-blocks: those tensors fall through to Q4_1)
        kd = codes[hfname + ".kd"].float().numpy(); kdm = codes[hfname + ".kdmin"].float().numpy(); ksc = codes[hfname + ".ksc"].numpy(); km = codes[hfname + ".km"].numpy()
        if rowperm is not None: kd, kdm, ksc, km = kd[rowperm], kdm[rowperm], ksc[rowperm], km[rowperm]
        return kq_payload(q, kd, kdm, ksc, km, bits)
    if g == 16 and colperm is not None:                                               # a Q6_K / Q3_K linear on a permuted input dimension (linear-attention out_proj): no 16-group block type exists, so dequantize the exact codes and store Q8_0 (near-lossless for 6-bit codes; +2 bpw on these few tensors)
        W = q.astype(np.float32) * np.repeat(s.astype(np.float32), g, 1) + np.repeat(b.astype(np.float32), g, 1); return q8_rtn(W), q.shape, Q.Q8_0
    assert g == 32, f"{hfname}: group {g}; GGUF Q4_1/Q5_1/Q8_0 need group 32"
    if bits == 4: return pack_q4_1(q, s, b), q.shape, Q.Q4_1
    if bits == 5: return pack_q5_1(q, s, b), q.shape, Q.Q5_1
    if bits == 8:
        z = codes[hfname + ".zeros"].numpy(); assert (z == 128).all(), f"{hfname}: 8-bit layers must use the fixed zero point 128 (--sym8)"
        return pack_q8_0(q, s), q.shape, Q.Q8_0
    raise ValueError(f"{hfname}: {bits} bits has no lossless GGUF block type (use 4, 5 or 8)")

def f16(t): return t.float().numpy().astype(np.float16)
r = GGUFReader(a.ref); arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8"); q35 = arch in ("qwen35", "qwen3next", "qwen35moe")
w = GGUFWriter(a.out, arch, use_temp_file=a.temp_file)
skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "general.file_type", "general.quantization_version"}
for name, field in r.fields.items():
    if name in skip or not field.types: continue
    main = field.types[0]; sub = field.types[-1] if len(field.types) > 1 else None
    if main == GGUFValueType.ARRAY:
        val = [str(bytes(field.parts[i]), "utf-8") for i in field.data] if sub == GGUFValueType.STRING else [pv for i in field.data for pv in field.parts[i].tolist()]
        w.add_array(name, val)
    elif main == GGUFValueType.STRING: w.add_string(name, str(bytes(field.parts[-1]), "utf-8"))
    else: w.add_key_value(name, field.parts[-1].tolist()[0], main)
w.add_quantization_version(gguf.GGML_QUANT_VERSION); w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_1)
sizes = {}
def add(name, data, shape=None, qtype=None):
    if qtype is None: w.add_tensor(name, data); sizes[name] = data.nbytes
    else: w.add_tensor(name, data.reshape(int(shape[0]), -1), raw_dtype=qtype); sizes[name] = data.nbytes   # byte shape (rows, bytes per row); the writer derives the logical columns from the type size
def emb_payload(E, kind=None):
    kind = kind or a.embed_kq
    if kind: from kquant import rtn_kquant; return (rtn_kquant(E, kind), E.shape, getattr(Q, kind.upper()))        # cast per row chunk inside (a 248k x 5120 float32 copy is 5 GB)
    return (q8_rtn(E.float().numpy()), E.shape, Q.Q8_0) if a.embed_q8 else (f16(E),)
def hcols(t): return t[:, torch.as_tensor(GPERM)] if GPERM is not None else t                  # permute the residual-stream (hidden) dimension
def hvec(t): return t[torch.as_tensor(GPERM)] if GPERM is not None else t
E = hcols(hf(PFX + "embed_tokens.weight")); add("token_embd.weight", *emb_payload(E)); del E
norm = hvec(hf(PFX + "norm.weight").float()); add("output_norm.weight", (norm + 1 if q35 else norm).numpy().astype(np.float32))
if "lm_head.weight" in idx:
    if a.head_codes:                                                                          # GPTQ'd head (head_gptq.py): exact pack of its codes
        assert GPERM is None, "--head-codes on a permuted model is not supported"
        with safe_open(a.head_codes, "pt") as sf: hq = {k: sf.get_tensor(k) for k in sf.keys()}
        add("output.weight", *kq_payload(hq["codes"].numpy(), hq["kd"].float().numpy(), hq["kdmin"].float().numpy(), hq["ksc"].numpy(), hq["km"].numpy(), int(hq["bits"][0]))); del hq
    else: H = hcols(hf("lm_head.weight")); add("output.weight", *emb_payload(H, a.head_kq)); del H
L = tc["num_hidden_layers"]; types = tc.get("layer_types") or ["full_attention"] * L
ROPE_PERM = arch in ("llama", "mistral", "llama4")                                                    # llama.cpp interleaves q/k rows for these archs (convert_hf_to_gguf LlamaModel.permute)
nq_a = tc["num_attention_heads"]; nkv_a = tc.get("num_key_value_heads") or nq_a; hd_a = tc.get("head_dim") or tc["hidden_size"] // nq_a
QKPERM = {n: np.arange(nh * hd_a).reshape(nh, 2, hd_a // 2).swapaxes(1, 2).reshape(-1) for n, nh in (("q_proj", nq_a), ("k_proj", nkv_a))} if ROPE_PERM else {}
if ROPE_PERM: print(f"arch {arch}: q/k rows interleaved for the rotary layout")
rowperm_v = tile(torch.arange(nv * hv).unsqueeze(-1), 0, hv).squeeze(-1).numpy() if reorder else None
rowperm_1 = tile(torch.arange(nv).unsqueeze(-1), 0, 1).squeeze(-1).numpy() if reorder else None
colperm_v = tile(torch.arange(nv * hv).unsqueeze(0), 1, hv).squeeze(0).numpy() if reorder else None
for li in range(L):
    p = PFX + f"layers.{li}."; g = f"blk.{li}."
    n1 = hvec(hf(p + "input_layernorm.weight").float()); add(g + "attn_norm.weight", (n1 + 1 if q35 else n1).numpy().astype(np.float32))
    n2 = hvec(hf(p + "post_attention_layernorm.weight").float()); add(g + ("post_attention_norm.weight" if q35 else "ffn_norm.weight"), (n2 + 1 if q35 else n2).numpy().astype(np.float32))
    p_down = PERM.get(p + "mlp.down_proj"); p_o = PERM.get(p + "self_attn.o_proj")                   # row orders implied by the permuted inputs of down_proj / o_proj
    if p_o is not None:                                                                               # o_proj columns = concat of q heads' values; v rows follow the per-kv-group order
        nq_ = tc["num_attention_heads"]; nkv_ = tc.get("num_key_value_heads") or nq_; hd_ = tc.get("head_dim") or tc["hidden_size"] // nq_; r_ = nq_ // nkv_
        o_ = p_o.reshape(nkv_, r_, hd_)[:, 0, :] - (np.arange(nkv_) * r_ * hd_)[:, None]; v_rows = (np.arange(nkv_)[:, None] * hd_ + o_).reshape(-1)
    else: v_rows = None
    rows_h = GPERM
    for hfn, gn in (("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")):
        add(g + gn + ".weight", *quantized(p + hfn + ".weight", rowperm=(rows_h if hfn == "mlp.down_proj" else p_down)))
    if types[li] == "linear_attention":
        qd = hk * nk
        qkv = p + "linear_attn.in_proj_qkv.weight"; rp = np.concatenate([np.arange(2 * qd), 2 * qd + rowperm_v]) if reorder else None
        add(g + "attn_qkv.weight", *quantized(qkv, rowperm=rp))
        add(g + "attn_gate.weight", *quantized(p + "linear_attn.in_proj_z.weight", rowperm=rowperm_v))
        add(g + "ssm_out.weight", *quantized(p + "linear_attn.out_proj.weight", colperm=colperm_v))
        for hfn, gn in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
            t = hf(p + f"linear_attn.{hfn}.weight").float(); t = t[rowperm_1] if reorder else t; add(g + gn + ".weight", f16(t))
        A = hf(p + "linear_attn.A_log").float(); A = A[rowperm_1] if reorder else A; add(g + "ssm_a", (-torch.exp(A)).numpy().astype(np.float32))
        dt = hf(p + "linear_attn.dt_bias").float(); dt = dt[rowperm_1] if reorder else dt; add(g + "ssm_dt.bias", dt.numpy().astype(np.float32))
        cv = hf(p + "linear_attn.conv1d.weight").float().squeeze(1); qk = 2 * qd
        if reorder: cv = torch.cat([cv[:qk], cv[qk:][rowperm_v]], 0)
        add(g + "ssm_conv1d.weight", cv.numpy().astype(np.float32)); add(g + "ssm_norm.weight", hf(p + "linear_attn.norm.weight").float().numpy().astype(np.float32))
    else:
        for hfn, gn in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            rp = v_rows if hfn == "v_proj" else (rows_h if hfn == "o_proj" else None)
            if hfn in QKPERM: assert rp is None, "usage-clustered q/k rows with the rotary interleave are not composed"; rp = QKPERM[hfn]
            add(g + gn + ".weight", *quantized(p + f"self_attn.{hfn}.weight", rowperm=rp))
            if (p + f"self_attn.{hfn}.bias") in idx:
                bb = hf(p + f"self_attn.{hfn}.bias").float(); bb = bb[torch.as_tensor(rp)] if rp is not None else bb; add(g + gn + ".bias", bb.numpy().astype(np.float32))
        for hfn, gn in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
            if (p + f"self_attn.{hfn}.weight") in idx: t = hf(p + f"self_attn.{hfn}.weight").float(); add(g + gn + ".weight", (t + 1 if q35 else t).numpy().astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
tot = sum(sizes.values()); print(f"wrote {a.out}: {len(sizes)} tensors, {tot/1e9:.2f} GB; arch {arch}; reference tensors {len(r.tensors)}")
