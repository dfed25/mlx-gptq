"""YAQA-style token weights for the layer Hessians (Tseng et al. 2025, Kronecker-factored sketch of the end-to-end KL Hessian with B = I):
for every linear, A = sum_t ||delta_t||^2 x_t x_t^T, where delta_t is the gradient of the model's own log-likelihood (y_t sampled from the
model: the Fisher = the KL Hessian at the teacher) at that linear's OUTPUT for token t. GPTQ's X^T X is the special case ||delta_t||^2 = 1.
This script runs the fp16 model forward + backward on the SAME calibration windows gptq_torch.py draws (same calib / nsamples / seqlen / seed)
and saves {hf linear name: weights (nsamples*seqlen,) float32}; gptq_torch.py --token-weights FILE then builds the weighted Hessians.
    python yaqa_weights.py MODEL_DIR OUT.pt --calib calib_v4.txt --nsamples 256 [--seqlen 512] [--seed 0] [--bs 4] [--greedy]"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--calib", default="calib_v4.txt"); ap.add_argument("--nsamples", type=int, default=256)
ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--dtype", default="float16")
ap.add_argument("--greedy", action="store_true", help="use the realised next tokens instead of sampling from the model (empirical Fisher)"); ap.add_argument("--sample-seed", type=int, default=1)
a = ap.parse_args(); t0 = time.time(); DEV = "cuda"; dt = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model); ids = tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed)
starts = rng.integers(0, len(ids) - a.seqlen, size=a.nsamples); X = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in starts]))
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(DEV).eval(); [p.requires_grad_(False) for p in model.parameters()]
model.config.use_cache = False; torch.manual_seed(a.sample_seed)
lins = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("model.layers.") and m.in_features >= 256 and m.out_features >= 256}
W = {n: torch.zeros(a.nsamples * a.seqlen, dtype=torch.float32) for n in lins}; pos = [0]
def mk(n):
    def hook(mod, gin, gout):
        g = gout[0].reshape(-1, gout[0].shape[-1]).float(); W[n][pos[0]:pos[0] + g.shape[0]] = (g * g).sum(-1).cpu()
    return hook
hds = [m.register_full_backward_hook(mk(n)) for n, m in lins.items()]
model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))          # parameters are frozen; this makes gradients flow to every activation
for b in range(0, a.nsamples, a.bs):
    xb = X[b:b + a.bs].to(DEV); out = model(input_ids=xb).logits.float()
    lp = torch.log_softmax(out, -1)
    if a.greedy: y = torch.cat([xb[:, 1:], xb[:, :1]], 1)                                        # realised next tokens (last position: wrap, negligible)
    else: y = torch.multinomial(lp.exp().reshape(-1, lp.shape[-1]), 1).reshape(lp.shape[:2])     # y ~ p_model: Fisher information = KL Hessian
    loss = -lp.gather(-1, y.unsqueeze(-1)).sum(); loss.backward(); pos[0] += xb.numel(); del out, lp, loss
    if (b // a.bs) % 16 == 0: print(f"  {b + a.bs}/{a.nsamples} windows ({time.time()-t0:.0f}s)", flush=True)
for h in hds: h.remove()
stats = {n: (w.mean().item(), w.std().item() / max(w.mean().item(), 1e-30)) for n, w in W.items()}
torch.save({"weights": W, "calib": a.calib, "nsamples": a.nsamples, "seqlen": a.seqlen, "seed": a.seed, "greedy": a.greedy}, a.out)
cv = np.mean([s[1] for s in stats.values()]); print(f"wrote {a.out}: {len(W)} linears x {a.nsamples*a.seqlen} tokens; mean coefficient of variation of the token weights {cv:.2f} (0 = GPTQ's uniform weighting); {time.time()-t0:.0f}s")
