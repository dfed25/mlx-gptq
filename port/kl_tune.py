"""KL fine-tune of the CONTINUOUS parameters of a K-quant-constrained model with the integer codes fixed: the fp16 super-scales
d / dmin of every quantized linear (trained through the fp16 rounding, straight-through), the RMSNorm weights and the attention
biases, minimising KL(teacher || student) on calibration windows against the unquantized model. Everything trained is stored
as-is in the GGUF (d/dmin per super-block, norms and biases in F32), so the exported file is exactly the trained model.
Exports the gptq_torch.py layout (dequantized shards + codes/ with new .kd/.kdmin/.scales/.biases); codes2gguf.py packs it unchanged.
    python kl_tune.py TEACHER_DIR QUANT_DIR OUT_DIR [--calib calib_mixed.txt] [--steps 300] [--bs 4] [--micro 2] [--seqlen 512] [--lr 3e-3] [--lr-norm 2e-4]"""
import argparse, os, json, math, time, shutil, glob, copy, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("teacher"); ap.add_argument("quant"); ap.add_argument("out")
ap.add_argument("--calib", default="calib_mixed.txt"); ap.add_argument("--heldout", default="wikitext2_test.txt"); ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--bs", type=int, default=4); ap.add_argument("--micro", type=int, default=2); ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--lr", type=float, default=3e-3, help="Adam lr on the log-multipliers of d and dmin (relative step per iteration)")
ap.add_argument("--lr-norm", type=float, default=2e-4, help="Adam lr on RMSNorm weights and attention biases"); ap.add_argument("--no-norm", action="store_true", help="train d/dmin only")
ap.add_argument("--eval-every", type=int, default=50); ap.add_argument("--eval-win", type=int, default=4); ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--accum", type=int, default=4, help="gradient accumulation: batches per optimizer step (10M scale parameters need a large effective batch)")
ap.add_argument("--opt", default="adam", choices=["adam", "blockmax"], help="adam, or the user's Packed BlockMax (fused Triton kernel packed16_triton.py: momentum in the low 8 bits of the fp32 parameter, one decaying max |g| per 16 parameters)")
ap.add_argument("--param", default="sb", choices=["sb", "row", "tensor"], help="which multipliers to train on d/dmin: per super-block (all of them), one per output row, or one per tensor")
a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); DEV = "cuda"

ZFIX = {3: 4, 6: 32}                                                                             # symmetric K-quant kinds: w = d*sc*(q - z)
class Build(torch.autograd.Function):
    """W = (Q - zfix) * (d*sc) - dmin*m, expanded per sub-block; gradient only to d and dmin (per super-block)."""
    @staticmethod
    def forward(ctx, d, dmin, Q, sc, m, g, zfix):
        rows, cols = Q.shape; ng = sc.shape[1]; gps = ng // d.shape[1]
        S = d.repeat_interleave(gps, 1) * sc; B = -(dmin.repeat_interleave(gps, 1) * m)
        ctx.save_for_backward(Q, sc, m); ctx.g = g; ctx.gps = gps; ctx.zfix = zfix
        return (Q.float() - zfix) * S.repeat_interleave(g, 1) + B.repeat_interleave(g, 1)
    @staticmethod
    def backward(ctx, gW):
        Q, sc, m = ctx.saved_tensors; g, gps = ctx.g, ctx.gps; rows, cols = Q.shape; ng = sc.shape[1]; gW = gW.float()
        A = (gW * (Q.float() - ctx.zfix)).view(rows, ng, g).sum(-1); Bg = gW.view(rows, ng, g).sum(-1)
        return (A * sc).view(rows, -1, gps).sum(-1), -(Bg * m).view(rows, -1, gps).sum(-1), None, None, None, None, None

class KQLinear(nn.Module):
    def __init__(self, lin, Q, d, dmin, sc, m, zfix=0):
        super().__init__(); self.g = Q.shape[1] // sc.shape[1]; self.zfix = zfix
        self.register_buffer("Q", Q.to(DEV)); self.register_buffer("sc", sc.float().to(DEV)); self.register_buffer("m", m.float().to(DEV))
        self.register_buffer("d0", d.float().to(DEV)); self.register_buffer("dmin0", dmin.float().to(DEV))
        shp = {"sb": self.d0.shape, "row": (self.d0.shape[0], 1), "tensor": (1, 1)}[a.param]
        self.theta = nn.Parameter(torch.zeros(shp, device=DEV)); self.phi = nn.Parameter(torch.zeros(shp, device=DEV)); self.bias = lin.bias
    def cur(self):
        d = self.d0 * torch.exp(self.theta); dm = self.dmin0 * torch.exp(self.phi)
        return d + (d.half().float() - d).detach(), dm + (dm.half().float() - dm).detach()           # fp16 rounding, straight-through
    def weight_fp32(self):
        d, dm = self.cur(); return Build.apply(d, dm, self.Q, self.sc, self.m, self.g, self.zfix)
    def forward(self, x): return F.linear(x, self.weight_fp32().to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

# ---- models
tok = AutoTokenizer.from_pretrained(a.teacher)
teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.float16).to(DEV).eval(); [p.requires_grad_(False) for p in teacher.parameters()]
student = AutoModelForCausalLM.from_pretrained(a.quant, dtype=torch.float32).to(DEV); student.config.use_cache = False
[p.requires_grad_(False) for p in student.parameters()]
CODES = {}
for f in sorted(glob.glob(os.path.join(a.quant, "codes", "*.safetensors"))): CODES[os.path.basename(f)] = load_file(f)
KQ = {}
for fn, sh in CODES.items():
    for k in sh:
        if not k.endswith(".codes"): continue
        n = k[:-6]; assert n + ".kd" in sh, f"{n}: not a K-quant model (no .kd)"
        lin = student.get_submodule(n); mod = KQLinear(lin, sh[k], sh[n + ".kd"], sh[n + ".kdmin"], sh[n + ".ksc"], sh[n + ".km"], ZFIX.get(int(sh[n + ".bits"][0]), 0))
        with torch.no_grad():                                                                        # sanity: rebuilt weights == checkpoint weights
            W0 = mod.weight_fp32(); err = (W0 - lin.weight.float()).abs().max().item(); ref = lin.weight.float().abs().max().item()
            assert err <= 2e-3 * ref + 1e-6, f"{n}: rebuilt weight differs from the checkpoint by {err:.3e} (max |w| {ref:.3e})"
        parent = student.get_submodule(n.rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], mod); KQ[n] = (fn, mod); del lin
torch.cuda.empty_cache()
params_d = [p for _, m in KQ.values() for p in (m.theta, m.phi)]
params_n = []
if not a.no_norm:
    for n, p in student.named_parameters():
        if n in {f"{k}.theta" for k in KQ} | {f"{k}.phi" for k in KQ}: continue
        if "norm" in n or n.endswith(".bias"): p.requires_grad_(True); params_n.append(p)
print(f"student: {len(KQ)} K-quant linears, {sum(p.numel() for p in params_d)} scale params ({a.param}), {sum(p.numel() for p in params_n)} norm/bias params; {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); student.train()
groups = [{"params": params_d, "lr": a.lr}] + ([{"params": params_n, "lr": a.lr_norm}] if params_n else [])
if a.opt == "adam": opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
else:
    import packed16_triton as PK; opt = PK.CandidateOptimizer(groups, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"optimizer: Packed BlockMax (fused kernel), state {sum((p.numel() + 15) // 16 * 4 for g in groups for p in g['params'])/2**20:.1f} MiB vs Adam {sum(p.numel() * 8 for g in groups for p in g['params'])/2**20:.1f} MiB")
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

# ---- data
ids = tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed)
hid = tok.encode(open(a.heldout).read()); H0 = len(hid) - a.eval_win * 1024; HW = torch.tensor([hid[H0 + w * 1024:H0 + (w + 1) * 1024] for w in range(a.eval_win)])   # the END of the text: the evaluator uses the first 8 x 2048 tokens
def batch():
    st = rng.integers(0, len(ids) - a.seqlen, size=a.bs); return torch.tensor(np.stack([ids[s:s + a.seqlen] for s in st]))
def kl_of(x, train):
    """mean over tokens of KL(teacher || student) on ids x (micro-batched); backward inside when train."""
    tot = 0.0
    for i in range(0, x.shape[0], a.micro):
        xb = x[i:i + a.micro].to(DEV)
        with torch.no_grad(): tl = torch.log_softmax(teacher(input_ids=xb).logits.float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train): sl = student(input_ids=xb).logits
        sl = torch.log_softmax(sl.float(), -1); kl = (tl.exp() * (tl - sl)).sum(-1).mean() * (xb.shape[0] / x.shape[0])
        if train: kl.backward()
        tot += kl.item(); del tl, sl, kl
    return tot
with torch.no_grad(): student.eval(); kl0 = kl_of(HW, False); student.train()
print(f"held-out KL before tuning: {kl0:.5f} ({a.eval_win} x 1024 tokens of {a.heldout})", flush=True)
def trainable_state(): return {k: v.detach().clone() for k, v in student.state_dict().items() if k.endswith((".theta", ".phi", ".bias")) or "norm" in k}
init_state = trainable_state(); best = (kl0, None); run = None
for step in range(a.steps):
    opt.zero_grad(set_to_none=True); kl = sum(kl_of(batch(), True) for _ in range(a.accum)) / a.accum
    for p in params_d + params_n:
        if p.grad is not None: p.grad /= a.accum
    opt.step(); sched.step(); run = kl if run is None else 0.9 * run + 0.1 * kl
    if (step + 1) % 10 == 0: print(f"step {step+1:4d}  train KL {kl:.5f} (ema {run:.5f})  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        with torch.no_grad(): student.eval(); h = kl_of(HW, False); student.train()
        print(f"  held-out KL {h:.5f} (start {kl0:.5f}, best {best[0]:.5f})", flush=True)
        if h < best[0]: best = (h, trainable_state())
if best[1] is not None: student.load_state_dict(best[1], strict=False); print(f"kept the best held-out state: KL {best[0]:.5f}")
else: student.load_state_dict(init_state, strict=False); print("no held-out improvement; exporting the initial state")

# ---- export (same layout as the input dir)
student.eval(); os.makedirs(os.path.join(a.out, "codes"), exist_ok=True)
for f in glob.glob(os.path.join(a.quant, "*")):
    b = os.path.basename(f)
    if os.path.isfile(f) and not f.endswith(".safetensors"): shutil.copy(f, os.path.join(a.out, b))
sd = {k: v.detach() for k, v in student.state_dict().items()}
for f in sorted(glob.glob(os.path.join(a.quant, "*.safetensors"))):
    sh = load_file(f); new = {}
    for k, v in sh.items():
        n = k[:-7] if k.endswith(".weight") else None
        if n in KQ:
            with torch.no_grad(): new[k] = KQ[n][1].weight_fp32().half().cpu()
        elif k in sd and ("norm" in k or k.endswith(".bias")): new[k] = sd[k].to(v.dtype).cpu()
        else: new[k] = v
    save_file(new, os.path.join(a.out, os.path.basename(f)), metadata={"format": "pt"})
for fn, sh in CODES.items():
    new = dict(sh)
    for n, (f2, mod) in KQ.items():
        if f2 != fn: continue
        with torch.no_grad():
            d, dm = mod.cur(); gps = mod.sc.shape[1] // d.shape[1]
            new[n + ".kd"] = d.half().cpu(); new[n + ".kdmin"] = dm.half().cpu()
            new[n + ".scales"] = (d.repeat_interleave(gps, 1) * mod.sc).half().cpu(); new[n + ".biases"] = (-(dm.repeat_interleave(gps, 1) * mod.m) - mod.zfix * d.repeat_interleave(gps, 1) * mod.sc).half().cpu()
    save_file(new, os.path.join(a.out, "codes", fn))
json.dump({"teacher": a.teacher, "quant": a.quant, "steps": a.steps, "lr": a.lr, "lr_norm": a.lr_norm, "accum": a.accum, "bs": a.bs, "param": a.param, "opt": a.opt, "heldout_kl_start": kl0, "heldout_kl_best": best[0], "calib": a.calib}, open(os.path.join(a.out, "kl_tune.json"), "w"), indent=1)
print(f"exported {a.out}  ({time.time()-t0:.0f}s)")
