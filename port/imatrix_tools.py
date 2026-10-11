"""Read a llama.cpp imatrix (GGUF format: '<tensor>.in_sum2' = sum over tokens of x^2 per input column, '<tensor>.counts' = tokens) and
compare its per-column mean x^2 (= the diagonal of our calibration Hessian) with ours, per linear.
    python imatrix_tools.py list IMATRIX.gguf
    python imatrix_tools.py compare IMATRIX.gguf MODEL_DIR --calib calib_v4.txt [--nsamples 256] [--layers 0,15,31,47,63]
Output of compare: per linear, the Pearson correlation of log diag, the ratio of total energy, and the fraction of columns where
one side is >3x the other. Also exports IMATRIX.diag.pt = {hf linear name: mean x^2 (in_features,)} for gptq_torch.py --imatrix."""
import sys, re, argparse, numpy as np, torch
def load_imatrix(path):
    import gguf
    r = gguf.GGUFReader(path); t = {x.name: x for x in r.tensors}; out = {}
    for n in t:
        if n.endswith(".in_sum2"):
            base = n[:-len(".in_sum2")]; s2 = np.asarray(t[n].data, dtype=np.float64).reshape(-1); c = np.asarray(t[base + ".counts"].data, dtype=np.float64).reshape(-1)
            out[base] = s2 / np.maximum(c.mean() if c.size < s2.size else c, 1)
    kv = {k: (v.parts[v.data[0]].tolist() if v.data else None) for k, v in r.fields.items() if k.startswith("imatrix")}
    return out, kv
G2HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
        "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj"}
def hf_name(g):
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight", g)
    return f"model.layers.{m.group(1)}.{G2HF[m.group(2)]}" if m and m.group(2) in G2HF else None
def untile_fn(model_dir):
    """llama.cpp's converter reorders the value heads (grouped -> tiled) on the INPUT of ssm_out (linear_attn.out_proj); the imatrix diagonal for that tensor is in
    the tiled order. Returns f(name, vec) that puts it back in HF order (identity for every other tensor / for non-hybrid models)."""
    import json, os
    c = json.load(open(os.path.join(model_dir, "config.json"))); c = c.get("text_config", c)
    nk = c.get("linear_num_key_heads", 0) or 0; nv = c.get("linear_num_value_heads", 0) or 0; hv = c.get("linear_value_head_dim", 0) or 0
    if not (nk and nv and nk != nv): return lambda n, v: v
    vpk = nv // nk
    def f(n, v):
        if not n.endswith("linear_attn.out_proj") or v.size != nv * hv: return v
        return v.reshape(vpk, nk, hv).transpose(1, 0, 2).reshape(-1)          # tiled -> grouped (GGUFSource.untile on dim 0)
    return f
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":                                                                       # python imatrix_tools.py list IMATRIX.gguf MODEL_DIR
        d, kv = load_imatrix(sys.argv[2]); print(kv); print(len(d), "tensors"); ut = untile_fn(sys.argv[3]) if len(sys.argv) > 3 else (lambda n, v: v)
        for n in list(d)[:12]: print(f"  {n:40s} -> {hf_name(n)}  n={d[n].size}  mean x^2 {d[n].mean():.4g}")
        torch.save({hf_name(n): torch.tensor(ut(hf_name(n), v), dtype=torch.float32) for n, v in d.items() if hf_name(n)}, sys.argv[2] + ".diag.pt"); print("wrote", sys.argv[2] + ".diag.pt", "(out_proj untiled)" if len(sys.argv) > 3 else "(no model dir: out_proj left in GGUF order)")
    elif cmd == "compare":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("imatrix"); ap.add_argument("model"); ap.add_argument("--calib", default="calib_v4.txt"); ap.add_argument("--nsamples", type=int, default=256)
        ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--layers", default="0,15,31,47,63"); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--bs", type=int, default=4)
        a = ap.parse_args(); from stream_util import Streamer; from quant_core import DEV
        U, kv = load_imatrix(a.imatrix); ut = untile_fn(a.model); U = {hf_name(n): ut(hf_name(n), v) for n, v in U.items() if hf_name(n)}; print("imatrix:", kv)
        S = Streamer(a.model, dtype=getattr(torch, a.dtype), device=DEV); ids = S.tok.encode(open(a.calib).read()); rng = np.random.default_rng(0)
        st = rng.integers(0, len(ids) - a.seqlen, size=a.nsamples); X = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in st])); KW = S.capture_kwargs(X[:a.bs]); hs = [S.embed(X[b:b + a.bs]) for b in range(0, a.nsamples, a.bs)]; S.offload_head()
        want = set(int(x) for x in a.layers.split(","))
        for li in range(S.n_layers):
            layer = S.load_layer(li); kw = S.kw_for(li, KW)
            if li in want:
                names = S.lin_names(layer); lins = {n: layer.get_submodule(n) for n in names}; acc = {n: torch.zeros(m.in_features, dtype=torch.float64, device=DEV) for n, m in lins.items()}; cnt = [0]
                def mk(n):
                    def hook(mod, inp, out): X_ = inp[0].reshape(-1, inp[0].shape[-1]).double(); acc[n] += (X_ * X_).sum(0); 
                    return hook
                hds = [lins[n].register_forward_hook(mk(n)) for n in names]
                for h in hs: S.run_layer(layer, h, kw)
                for hd in hds: hd.remove()
                ntok = a.nsamples * a.seqlen
                for n in names:
                    key = f"model.layers.{li}.{n}"; ours = (acc[n] / ntok).cpu().numpy()
                    if key not in U: print(f"  {key}: not in imatrix"); continue
                    u = U[key]
                    if u.size != ours.size: print(f"  {key}: size mismatch {u.size} vs {ours.size}"); continue
                    lo, lu = np.log(ours + 1e-12), np.log(u + 1e-12); r = np.corrcoef(lo, lu)[0, 1]; ratio = u.sum() / ours.sum(); far = np.mean(np.abs(lu - lo) > np.log(3))
                    print(f"  {key:45s} corr(log diag) {r:.3f}  energy unsloth/ours {ratio:.2f}  columns >3x apart {100*far:.1f}%", flush=True)
            hs = [S.run_layer(layer, h, kw) for h in hs]; S.free_layer(li)
