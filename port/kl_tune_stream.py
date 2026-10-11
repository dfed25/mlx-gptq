"""Layer-streamed KL fine-tune for a model larger than the card (Qwen3.8-27B on one 24 GB A10G, 30 GB RAM): the same objective as
kl_tune.py (codes fixed; fp16 super-scales d / dmin of every K-quant linear as log-multipliers through the fp16 rounding, RMSNorm
weights and biases, trained to minimise KL(teacher || student)), with
  1. the bf16 teacher streamed ONCE over the calibration windows and its top-K log-probs per token cached (K = 256: ~200 MB);
  2. each training pass streaming the student layer by layer: forward without gradients storing every layer's input, then backward
     from the last layer with recompute (one layer's codes and rebuilt weights on the card at a time; the small trained parameters
     and the Adam state stay resident on the GPU, the codes are read from disk per pass);
  3. the student's embeddings / head taken from the packed GGUF (Q4_K / Q6_K), so the fine-tune sees the deployed model.
Export: OUT/ = symlinks to the source shards + tuned.safetensors (norms, biases) with an index pointing at it, and OUT/codes/ =
symlinks to the original code shards + zz_tuned_layerNNN.safetensors overriding .kd/.kdmin/.scales/.biases; codes2gguf.py packs it.
    python kl_tune_stream.py SRC_DIR QUANT_DIR OUT_DIR --gguf PACKED.gguf [--calib calib_v4.txt] [--nwin 256] [--bs 8] [--steps 200]"""
import argparse, os, json, math, time, glob, shutil, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from stream_util import Streamer
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("quant"); ap.add_argument("out"); ap.add_argument("--gguf", default=None, help="packed GGUF of QUANT: embeddings / head / norms are taken from it")
ap.add_argument("--calib", default="calib_v4.txt"); ap.add_argument("--nwin", type=int, default=256); ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--steps", type=int, default=200); ap.add_argument("--accum", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-norm", type=float, default=1e-5)
ap.add_argument("--topk", type=int, default=256); ap.add_argument("--cache", default=None, help="teacher top-K cache file (default OUT_DIR/teacher_topk.pt)"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--chunk", type=int, default=128, help="positions per logits chunk")
ap.add_argument("--packed", default=None, help="packed codes dir from pack_codes.py (half the bytes per pass)"); ap.add_argument("--codes-ram", action="store_true", help="hold every packed shard in RAM for the whole run (needs ~17 GB on the 27B)")
ap.add_argument("--max-layers", type=int, default=0, help="smoke test: train through this many layers only (loss on their output is meaningless; checks the machinery)")
a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); DEV = "cuda"; dt = getattr(torch, a.dtype); ZFIX = {3: 4, 6: 32}
os.makedirs(a.out, exist_ok=True); cache = a.cache or os.path.join(a.out, "teacher_topk.pt")
def log(*x): print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)

# ---------------------------------------------------------------- phase A: teacher top-K
S = Streamer(a.src, dtype=dt, device=DEV); tok = S.tok
ids = tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed); starts = rng.integers(0, len(ids) - a.seqlen, size=a.nwin)
X = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in starts]))
if not os.path.exists(cache):
    log(f"teacher pass: {a.nwin} x {a.seqlen} tokens of {a.calib}, top-{a.topk}")
    KW = S.capture_kwargs(X[:a.bs]); H = [S.embed(X[b:b + a.bs]) for b in range(0, a.nwin, a.bs)]
    for li in range(S.n_layers):
        layer = S.load_layer(li); H = [S.run_layer(layer, h, S.kw_for(li, KW)) for h in H]; S.free_layer(li)
    TI = torch.empty((a.nwin, a.seqlen, a.topk), dtype=torch.int32); TP = torch.empty((a.nwin, a.seqlen, a.topk), dtype=torch.float16); TT = torch.empty((a.nwin, a.seqlen), dtype=torch.float16)
    w = 0
    for h in H:
        for i in range(h.shape[0]):
            for c0 in range(0, a.seqlen, a.chunk):
                c1 = min(c0 + a.chunk, a.seqlen); lp = torch.log_softmax(S.logits(h[i:i + 1, c0:c1])[0].float(), -1); v, ix = lp.topk(a.topk, -1)
                TI[w, c0:c1] = ix.int().cpu(); TP[w, c0:c1] = v.half().cpu(); TT[w, c0:c1] = (1 - v.exp().sum(-1)).clamp_min(1e-9).log().half().cpu(); del lp       # tail mass; clamped: fp32 sums can exceed 1
            w += 1
    torch.save({"X": X, "idx": TI, "lp": TP, "tail": TT, "calib": a.calib, "topk": a.topk}, cache); del H; log(f"teacher cache written: {cache}")
C = torch.load(cache); assert torch.equal(C["X"], X), "cache was built from different windows (seed / calib / nwin)"; TI, TP, TT = C["idx"], C["lp"], C["tail"]
torch.cuda.empty_cache()

# ---------------------------------------------------------------- student pieces
class Build(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, dmin, Q, sc, m, g, zfix):
        rows, cols = Q.shape; ng = sc.shape[1]; gps = ng // d.shape[1]
        Sg = d.repeat_interleave(gps, 1) * sc; Bg = -(dmin.repeat_interleave(gps, 1) * m)
        ctx.save_for_backward(Q, sc, m); ctx.g = g; ctx.gps = gps; ctx.zfix = zfix
        return (Q.float() - zfix) * Sg.repeat_interleave(g, 1) + Bg.repeat_interleave(g, 1)
    @staticmethod
    def backward(ctx, gW):
        Q, sc, m = ctx.saved_tensors; g, gps = ctx.g, ctx.gps; rows, cols = Q.shape; ng = sc.shape[1]; gW = gW.float()
        A = (gW * (Q.float() - ctx.zfix)).view(rows, ng, g).sum(-1); B = gW.view(rows, ng, g).sum(-1)
        return (A * sc).view(rows, -1, gps).sum(-1), -(B * m).view(rows, -1, gps).sum(-1), None, None, None, None, None
class KQLinear(nn.Module):
    """Persistent part (theta, phi, d0, dmin0, bias) lives on the GPU for the whole run; Q / sc / m are attached per layer visit."""
    def __init__(self, d, dmin, bias, zfix):
        super().__init__(); self.register_buffer("d0", d.half().to(DEV)); self.register_buffer("dmin0", dmin.half().to(DEV)); self.zfix = zfix
        self.theta = nn.Parameter(torch.zeros(d.shape, device=DEV)); self.phi = nn.Parameter(torch.zeros(d.shape, device=DEV))
        self.bias = nn.Parameter(bias.float().to(DEV)) if bias is not None else None; self.Q = self.sc = self.m = None
    def cur(self):
        d = self.d0.float() * torch.exp(self.theta); dm = self.dmin0.float() * torch.exp(self.phi)
        return d + (d.half().float() - d).detach(), dm + (dm.half().float() - dm).detach()
    def weight(self):
        d, dm = self.cur(); return Build.apply(d, dm, self.Q, self.sc, self.m, self.Q.shape[1] // self.sc.shape[1], self.zfix)
    def forward(self, x): return F.linear(x, self.weight().to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
CODEF = {}; 
for f in sorted(glob.glob(os.path.join(a.quant, "codes", "*.safetensors"))):
    with safe_open(f, "pt") as sf:
        for k in sf.keys(): CODEF[k] = f
RAM = {}
def packed_shard(li):
    f = os.path.join(a.packed, f"layer{li:03d}.safetensors")
    if a.codes_ram:
        if li not in RAM:
            with safe_open(f, "pt") as sf: RAM[li] = {k: sf.get_tensor(k).pin_memory() for k in sf.keys()}
        return RAM[li]
    with safe_open(f, "pt") as sf: return {k: sf.get_tensor(k) for k in sf.keys()}
def layer_codes(li):
    """{module name in the layer: (Q uint8 (rows, cols), sc, m float)} for layer li, from the packed shard (nibbles unpacked on the GPU) or the raw codes."""
    pre = S.ckpt_key(f"model.layers.{li}."); out = {}
    if a.packed:
        sh = packed_shard(li)
        for k in sh:
            if k.endswith(".q"):
                n = k[:-2]; q = sh[k].to(DEV, non_blocking=True)
                if int(sh[n + ".bits"][0]) == 4: Q = torch.empty((q.shape[0], q.shape[1] * 2), dtype=torch.uint8, device=DEV); Q[:, 0::2] = q & 15; Q[:, 1::2] = q >> 4
                else: Q = q
                out[n[len(pre):]] = (Q, sh[n + ".sc"].to(DEV).float(), sh[n + ".m"].to(DEV).float())
        return out
    f = CODEF[next(k for k in CODEF if k.startswith(pre) and k.endswith(".codes"))]
    with safe_open(f, "pt", device="cpu") as sf:
        for k in sf.keys():
            if k.startswith(pre) and k.endswith(".codes"):
                n = k[:-6]; out[n[len(pre):]] = (sf.get_tensor(k).to(DEV, non_blocking=True), sf.get_tensor(n + ".ksc").to(DEV).float(), sf.get_tensor(n + ".km").to(DEV).float())
    return out
# persistent trained state: per layer the KQLinear modules and the small params (norms, biases)
KQ = {}; P = {}; NQ = set(); ORIG = {}
for li in range(S.n_layers):
    pre = S.ckpt_key(f"model.layers.{li}."); layer = S.layers[li]; KQ[li] = {}; ORIG[li] = {}
    for k in CODEF:
        if k.startswith(pre) and k.endswith(".codes"):
            n = k[:-6]; local = n[len(pre):]; lin = layer.get_submodule(local); ORIG[li][local] = lin
            with safe_open(CODEF[k], "pt") as sf: d = sf.get_tensor(n + ".kd"); dm = sf.get_tensor(n + ".kdmin"); bits = int(sf.get_tensor(n + ".bits")[0])
            b = S.load(f"model.layers.{li}.{local}.bias") if f"model.layers.{li}.{local}.bias" in [p for p in S.layer_param_names(li)] else None
            KQ[li][local] = KQLinear(d, dm, b, ZFIX.get(bits, 0)); NQ.add(f"model.layers.{li}.{local}.weight")
    for n in S.layer_param_names(li):
        local = n[len(f"model.layers.{li}."):]
        if n in NQ or any(local.startswith(q + ".") for q in KQ[li]): continue          # quantized weights and their biases live in KQLinear
        if "norm" in local or local.endswith(".bias"): P[n] = nn.Parameter(S.load(n).float().to(DEV))
params_d = [p for li in KQ for m in KQ[li].values() for p in (m.theta, m.phi)]; params_b = [m.bias for li in KQ for m in KQ[li].values() if m.bias is not None]; params_n = list(P.values())
log(f"student: {sum(len(v) for v in KQ.values())} K-quant linears, {sum(p.numel() for p in params_d)/1e6:.1f}M scale multipliers, {sum(p.numel() for p in params_n + params_b)/1e3:.0f}k norm/bias values; GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")
if a.gguf:                                                                       # deployed embeddings / head / final norm
    from gguf_source import GGUFSource; G = GGUFSource(a.gguf, S.text_cfg, S.layout)
    for n in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        if n == "lm_head.weight" and S.tied: continue
        S.model.get_parameter(n).data = G.get(S.ckpt_key(n)).to(dt).to(DEV)
    log(f"embeddings / head from {a.gguf}: {G.types}")
FN = nn.Parameter(S.model.model.norm.weight.data.float().clone().to(DEV)); params_n.append(FN)         # final norm is trained too
groups = [{"params": params_d, "lr": a.lr}, {"params": params_n + params_b, "lr": a.lr_norm}]
opt = torch.optim.Adam(groups, betas=(0.9, 0.99)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

from accelerate.utils import set_module_tensor_to_device
def visit(li):
    """Load layer li's NON-quantized parameters from the source shards (the quantized weights are never read: the KQLinear modules replace them), attach its codes; returns (layer, overrides)."""
    layer = S.layers[li]
    for n in S.layer_param_names(li):
        if n in NQ or n[len(f"model.layers.{li}."):].rsplit(".", 1)[0] in KQ[li]: continue
        set_module_tensor_to_device(S.model, n, DEV, value=S.load(n))
    for n, b in layer.named_buffers(): set_module_tensor_to_device(layer, n, DEV, value=b)
    codes = layer_codes(li)
    for local, (Q, sc, m) in codes.items():
        mod = KQ[li][local]; mod.Q, mod.sc, mod.m = Q, sc, m; parent = layer.get_submodule(local.rsplit(".", 1)[0]) if "." in local else layer; setattr(parent, local.rsplit(".", 1)[-1], mod)
    ov = {n[len(f"model.layers.{li}."):]: P[n].to(dt) for n in P if n.startswith(f"model.layers.{li}.")}
    return layer, ov
def leave(li, layer):
    for local, mod in KQ[li].items():                                                                  # the original modules back in place BEFORE free_layer (it walks named_parameters)
        mod.Q = mod.sc = mod.m = None; parent = layer.get_submodule(local.rsplit(".", 1)[0]) if "." in local else layer
        setattr(parent, local.rsplit(".", 1)[-1], ORIG[li][local])
    S.free_layer(li)
NL = a.max_layers or S.n_layers
def run_layer(layer, ov, h, kw, grad):
    with torch.set_grad_enabled(grad):
        out = torch.func.functional_call(layer, ov, (h,), kw); return out[0] if isinstance(out, (tuple, list)) else out
def head(h):
    """logits of the student for hidden states h (bs, T, H): the model's own final RMSNorm with the trained weight, then the (frozen, GGUF) head."""
    v = torch.func.functional_call(S.model.model.norm, {"weight": FN.to(dt)}, (h,)); return S.model.lm_head(v)
def kl_chunk(lg, ti, tp, tt):
    """KL(teacher || student) on the teacher's top-K + tail: sum_k p_k (lp_k - lq_k) + p_tail (lt - log q_tail)."""
    lq = torch.log_softmax(lg.float(), -1); lqk = lq.gather(-1, ti.long()); p = tp.float().exp(); pt = tt.float().exp()
    qtail = (1 - lqk.exp().sum(-1)).clamp_min(1e-9).log()
    return ((p * (tp.float() - lqk)).sum(-1) + pt * (tt.float() - qtail)).mean()
KW = S.capture_kwargs(X[:a.bs])
def one_pass(wins, train=True):
    """One forward (no grad, inputs stored) + one backward (recompute) over the windows `wins`; returns the KL."""
    xb = X[wins]; h = S.embed(xb); stored = []
    for li in range(NL):
        layer, ov = visit(li); stored.append(h.to("cpu", non_blocking=True)); h = run_layer(layer, ov, h, S.kw_for(li, KW), False); leave(li, layer)
    hf = h.detach().requires_grad_(train); kl = 0.0
    for c0 in range(0, a.seqlen, a.chunk):
        c1 = min(c0 + a.chunk, a.seqlen); lg = head(hf[:, c0:c1]); l = kl_chunk(lg, TI[wins, c0:c1].to(DEV), TP[wins, c0:c1].to(DEV), TT[wins, c0:c1].to(DEV)) * ((c1 - c0) / a.seqlen)
        if train: l.backward()
        kl += l.item(); del lg, l
    if not train: return kl
    g = hf.grad; del hf
    for li in reversed(range(NL)):
        layer, ov = visit(li); hin = stored[li].to(DEV).requires_grad_(True); out = run_layer(layer, ov, hin, S.kw_for(li, KW), True); out.backward(g); g = hin.grad; del out, hin; leave(li, layer); torch.cuda.empty_cache()
    return kl
log(f"training: {a.steps} steps x {a.accum} x {a.bs} windows of {a.seqlen}, lr {a.lr} / norms {a.lr_norm}, {NL} layers"); run = None
for step in range(a.steps):
    ts = time.time(); opt.zero_grad(set_to_none=True); kl = 0.0
    for _ in range(a.accum): kl += one_pass(torch.as_tensor(rng.integers(0, a.nwin, size=a.bs))) / a.accum
    if a.accum > 1:
        for p in params_d + params_n + params_b:
            if p.grad is not None: p.grad /= a.accum
    opt.step(); sched.step(); run = kl if run is None else 0.9 * run + 0.1 * kl
    log(f"step {step+1:4d}  KL {kl:.5f} (ema {run:.5f})  lr {sched.get_last_lr()[0]:.2e}  {time.time()-ts:.0f}s/step  GPU peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

# ---------------------------------------------------------------- export
os.makedirs(os.path.join(a.out, "codes"), exist_ok=True)
for f in glob.glob(os.path.join(a.src, "*")):
    b = os.path.basename(f); dst = os.path.join(a.out, b)
    if b in ("model.safetensors.index.json",) or os.path.exists(dst): continue
    os.symlink(os.path.abspath(f), dst)
ix = os.path.join(a.src, "model.safetensors.index.json")
idx = json.load(open(ix)) if os.path.exists(ix) else {"metadata": {}, "weight_map": {k: os.path.basename(v) for k, v in S.wmap.items()}}; tuned = {}
for n, p in P.items(): tuned[S.ckpt_key(n)] = p.detach().to(dt).cpu()
for li in KQ:
    for local, m in KQ[li].items():
        if m.bias is not None: tuned[S.ckpt_key(f"model.layers.{li}.{local}.bias")] = m.bias.detach().to(dt).cpu()
tuned[S.ckpt_key("model.norm.weight")] = FN.detach().to(dt).cpu()
save_file(tuned, os.path.join(a.out, "tuned.safetensors"), metadata={"format": "pt"})
for k in tuned: idx["weight_map"][k] = "tuned.safetensors"
json.dump({"metadata": idx.get("metadata", {}), "weight_map": idx["weight_map"]}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
for f in sorted(set(CODEF.values())):
    dst = os.path.join(a.out, "codes", os.path.basename(f))
    if not os.path.exists(dst): os.symlink(os.path.abspath(f), dst)
for li in KQ:
    pre = S.ckpt_key(f"model.layers.{li}."); new = {}
    for local, m in KQ[li].items():
        with torch.no_grad():
            d, dm = m.cur(); n = pre + local
            with safe_open(CODEF[n + ".codes"], "pt") as sf: sc = sf.get_tensor(n + ".ksc").float(); mm = sf.get_tensor(n + ".km").float()
            gps = sc.shape[1] // d.shape[1]; d = d.cpu(); dm = dm.cpu()
            new[n + ".kd"] = d.half(); new[n + ".kdmin"] = dm.half(); new[n + ".scales"] = (d.repeat_interleave(gps, 1) * sc).half(); new[n + ".biases"] = (-(dm.repeat_interleave(gps, 1) * mm) - m.zfix * (d.repeat_interleave(gps, 1) * sc)).half()
    save_file(new, os.path.join(a.out, "codes", f"zz_tuned_layer{li:03d}.safetensors"))
shutil.copy(os.path.join(a.quant, "bits_used.json"), os.path.join(a.out, "bits_used.json")) if os.path.exists(os.path.join(a.quant, "bits_used.json")) else None
json.dump({"src": a.src, "quant": a.quant, "gguf": a.gguf, "steps": a.steps, "bs": a.bs, "accum": a.accum, "lr": a.lr, "lr_norm": a.lr_norm, "nwin": a.nwin, "topk": a.topk, "calib": a.calib, "final_kl_ema": run}, open(os.path.join(a.out, "kl_tune.json"), "w"), indent=1)
log(f"exported {a.out}")
