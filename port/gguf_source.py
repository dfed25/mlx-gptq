"""Read a llama.cpp GGUF of a Qwen3.5/3.8 (hybrid linear-attention) or Qwen2/3 model and hand back HF-named fp16 tensors,
inverting the converter's transforms (llama.cpp conversion/qwen.py: V-head reorder grouped->tiled in the linear-attention
projections, A = -exp(A_log), dt_bias -> dt_proj.bias, conv1d squeeze, +1 on every norm weight except linear_attn.norm).
Quantized tensors are dequantized with gguf-py (exact for the GGUF's own arithmetic)."""
import re, numpy as np, torch, gguf
from gguf.quants import dequantize

class GGUFSource:
    def __init__(self, path, text_cfg, layout="qwen_vl"):
        self.r = gguf.GGUFReader(path); self.t = {t.name: t for t in self.r.tensors}; self.cfg = text_cfg; self.layout = layout
        self.nk = getattr(text_cfg, "linear_num_key_heads", 0) or 0; self.nv = getattr(text_cfg, "linear_num_value_heads", 0) or 0
        self.hk = getattr(text_cfg, "linear_key_head_dim", 0) or 0; self.hv = getattr(text_cfg, "linear_value_head_dim", 0) or 0
        self.reorder = self.nk > 0 and self.nv > 0 and self.nk != self.nv; self.vpk = self.nv // self.nk if self.reorder else 1
        self.types = {}
        f = self.r.fields.get("general.architecture"); self.arch = bytes(f.parts[f.data[0]]).decode() if f is not None else ""
        self.plus1 = 1.0 if self.arch.startswith(("qwen35", "qwen3next", "gemma")) else 0.0        # llama.cpp stores these archs' RMS norms as w+1
        self.ffn_norm = "post_attention_norm.weight" if self.arch.startswith(("qwen35", "qwen3next")) else "ffn_norm.weight"
        self.rope_perm = self.arch in ("llama", "mistral", "llama4")                                    # llama.cpp's converter interleaves q/k rows (GPT-J rotary layout) for these archs
        nq = getattr(text_cfg, "num_attention_heads", 0); nkv = getattr(text_cfg, "num_key_value_heads", None) or nq; hd = getattr(text_cfg, "head_dim", None) or (getattr(text_cfg, "hidden_size", 0) // max(nq, 1))
        self.qk_inv = {}
        if self.rope_perm:
            for name, nh in (("q_proj", nq), ("k_proj", nkv)):
                idx = torch.arange(nh * hd).reshape(nh, 2, hd // 2).transpose(1, 2).reshape(-1); self.qk_inv[name] = torch.argsort(idx)   # gguf = hf[idx]  ->  hf = gguf[argsort(idx)]
    def names(self): return list(self.t)
    def has_output(self): return "output.weight" in self.t
    # ---- raw
    def raw(self, gname):
        t = self.t[gname]; self.types[gname] = t.tensor_type.name
        if t.tensor_type.name in ("F32", "F16", "BF16"):
            a = np.asarray(t.data)
            if t.tensor_type.name == "BF16": a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            a = a.astype(np.float32)
        else:
            d = np.asarray(t.data); shape = [int(x) for x in reversed(list(t.shape))]
            if d.ndim == 2 and d.shape[0] >= 8192:                       # big quantized matrices (embeddings / head): dequantize in row chunks into one fp32 buffer (a 15 GB box cannot hold gguf-py's temporaries for 248k x 5120)
                out = torch.empty((d.shape[0], int(np.prod(shape)) // d.shape[0]), dtype=torch.float32); C = 4096
                for r0 in range(0, d.shape[0], C): out[r0:r0 + C] = torch.from_numpy(dequantize(d[r0:r0 + C], t.tensor_type).astype(np.float32).reshape(min(C, d.shape[0] - r0), -1))
                return out.reshape(shape)
            a = dequantize(d, t.tensor_type).astype(np.float32)
        shape = [int(x) for x in reversed(list(t.shape))]            # ggml ne order -> numpy
        return torch.from_numpy(a.reshape(shape))
    # ---- inverse of _reorder_v_heads (grouped -> tiled): tiled -> grouped along `dim`
    def untile(self, x, dim, head_dim):
        if not self.reorder: return x
        shape = list(x.shape); dim = dim % len(shape)
        new = shape[:dim] + [self.vpk, self.nk, head_dim] + shape[dim + 1:]
        y = x.reshape(*new); perm = list(range(len(new))); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return y.permute(*perm).contiguous().reshape(*shape)
    # ---- HF name -> tensor
    def strip(self, name):
        if self.layout == "qwen_vl" and name.startswith("model.language_model."): return "model." + name[len("model.language_model."):]
        if self.layout == "gemma_vl" and name.startswith("language_model."): return name[len("language_model."):]
        return name
    def get(self, hf_name):
        n = self.strip(hf_name)
        if n == "model.embed_tokens.weight": return self.raw("token_embd.weight")
        if n == "lm_head.weight": return self.raw("output.weight") if "output.weight" in self.t else self.raw("token_embd.weight")
        if n == "model.norm.weight": return self.raw("output_norm.weight") - self.plus1
        m = re.match(r"model\.layers\.(\d+)\.(.+)", n); assert m, n; b, rest = m.group(1), m.group(2); g = f"blk.{b}."
        if rest == "input_layernorm.weight": return self.raw(g + "attn_norm.weight") - self.plus1
        if rest == "post_attention_layernorm.weight": return self.raw(g + self.ffn_norm) - self.plus1
        if rest.startswith("self_attn."):
            k = rest[len("self_attn."):]
            if k == "q_norm.weight": return self.raw(g + "attn_q_norm.weight") - self.plus1
            if k == "k_norm.weight": return self.raw(g + "attn_k_norm.weight") - self.plus1
            base = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}[k.split(".")[0]]
            w = self.raw(g + base + "." + k.split(".")[1])
            return w[self.qk_inv[k.split(".")[0]]] if k.split(".")[0] in self.qk_inv else w
        if rest.startswith("mlp."):
            k = rest[len("mlp."):]; base = {"gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}[k.split(".")[0]]
            return self.raw(g + base + "." + k.split(".")[1])
        if rest.startswith("linear_attn."):
            k = rest[len("linear_attn."):]
            if k == "in_proj_qkv.weight":
                w = self.raw(g + "attn_qkv.weight"); qd = self.hk * self.nk
                return torch.cat([w[:2 * qd], self.untile(w[2 * qd:], 0, self.hv)], 0)
            if k == "in_proj_z.weight": return self.untile(self.raw(g + "attn_gate.weight"), 0, self.hv)
            if k == "in_proj_a.weight": return self.untile(self.raw(g + "ssm_alpha.weight"), 0, 1)
            if k == "in_proj_b.weight": return self.untile(self.raw(g + "ssm_beta.weight"), 0, 1)
            if k == "out_proj.weight": return self.untile(self.raw(g + "ssm_out.weight"), 1, self.hv)
            if k == "A_log": a = self.raw(g + "ssm_a"); return torch.log(-self.untile(a.unsqueeze(-1), 0, 1).squeeze(-1))
            if k == "dt_bias": return self.untile(self.raw(g + "ssm_dt.bias").unsqueeze(-1), 0, 1).squeeze(-1)
            if k == "conv1d.weight":
                w = self.raw(g + "ssm_conv1d.weight"); qk = 2 * self.hk * self.nk
                w = torch.cat([w[:qk], self.untile(w[qk:], 0, self.hv)], 0); return w.unsqueeze(1)
            if k == "norm.weight": return self.raw(g + "ssm_norm.weight")
        raise KeyError(hf_name)
