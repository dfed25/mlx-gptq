"""KL divergence and top-1 agreement of a quantized model against the unquantized reference, streamed layer by layer (same
windows as the perplexity protocol). KL(ref || cand) averaged over tokens, top-1 agreement = fraction of positions where both
models' argmax agree, plus both perplexities. The reference log-probs are kept in fp16 on the CPU (nwin x ctx x vocab).
    python kld_stream.py REF_MODEL (--cand DIR | --cand-gguf FILE) [--ref-gguf FILE] [--text wikitext2_test.txt] [--nwin 8]"""
import argparse, math, json, time, os, gc, torch
from stream_util import Streamer
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--cand", default=None); ap.add_argument("--cand-gguf", default=None); ap.add_argument("--ref-gguf", default=None)
ap.add_argument("--nwin", type=int, default=8); ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--dtype", default="float16"); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--text", default="wikitext2_test.txt"); ap.add_argument("--out", default=None); ap.add_argument("--win-offset", type=int, default=0, help="skip this many windows first (split a run into pieces on a small-RAM box; token-weighted means recombine exactly)")
a = ap.parse_args(); dt = getattr(torch, a.dtype); t0 = time.time(); DEV = os.environ.get("DOMQ_DEV", "cuda" if torch.cuda.is_available() else "cpu")

def run(path, gguf):
    """Yields per window (log-probs fp32 [ctx-1, V] on DEV, targets); returns the streamer afterwards for cleanup."""
    S = Streamer(path, dtype=dt, device=DEV, gguf=gguf)
    ids = S.tok.encode(open(a.text).read()); assert len(ids) >= (a.win_offset + a.nwin) * a.ctx
    X = torch.tensor([ids[w * a.ctx:(w + 1) * a.ctx] for w in range(a.win_offset, a.win_offset + a.nwin)])
    KW = S.capture_kwargs(X[:a.bs]); KW_last = S.capture_kwargs(X[-(a.nwin % a.bs or a.bs):]) if a.nwin % a.bs else KW
    H = [S.embed(X[b:b + a.bs]) for b in range(0, a.nwin, a.bs)]
    for li in range(S.n_layers):
        layer = S.load_layer(li); H = [S.run_layer(layer, h, S.kw_for(li, KW if h.shape[0] == a.bs else KW_last)) for h in H]; S.free_layer(li)
    def windows():                                                   # chunks of CH positions: a (CH x vocab) fp32 log-prob tensor at a time (248k vocab x 2048 would be 2 GB per temporary)
        for h, b in zip(H, range(0, a.nwin, a.bs)):
            for i in range(h.shape[0]):
                for c0 in range(0, a.ctx - 1, CH):
                    c1 = min(c0 + CH, a.ctx - 1); lg = S.logits(h[i:i + 1, c0:c1])[0].float(); yield torch.log_softmax(lg, -1), X[b + i, c0 + 1:c1 + 1].to(lg.device); del lg
    return S, windows()
CH = 256

S, ref = run(a.model, a.ref_gguf)
ref_lp = []; nll_ref = 0.0; n = 0
for lp, t in ref:
    nll_ref += -lp[torch.arange(t.numel(), device=lp.device), t].sum().item(); n += t.numel(); ref_lp.append(lp.half().cpu()); del lp
del ref, S; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
S, cand = run(a.cand or a.model, a.cand_gguf)
kl = 0.0; agree = 0; nll_cand = 0.0; top5 = 0
for (lq, t), lp in zip(cand, ref_lp):
    lp = lp.to(lq.device).float(); p = lp.exp()
    kl += (p * (lp - lq)).sum().item(); agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
    nll_cand += -lq[torch.arange(t.numel(), device=lq.device), t].sum().item()
    top5 += (lq.topk(5, -1).indices == lp.argmax(-1, keepdim=True)).any(-1).sum().item(); del lp, p, lq
res = {"ref": a.ref_gguf or a.model, "cand": a.cand_gguf or a.cand or a.model, "text": a.text, "tokens": n, "kl_mean": kl / n, "top1_agree": agree / n,
       "ref_top1_in_cand_top5": top5 / n, "ppl_ref": math.exp(nll_ref / n), "ppl_cand": math.exp(nll_cand / n), "s": time.time() - t0}
print(json.dumps(res)); print(f"{res['cand']}: KL(ref||cand) {res['kl_mean']:.5f}, top-1 agreement {100*res['top1_agree']:.2f}%, ref top-1 in cand top-5 {100*res['ref_top1_in_cand_top5']:.2f}%, ppl ref {res['ppl_ref']:.3f} cand {res['ppl_cand']:.3f} over {n} tokens, {time.time()-t0:.0f}s")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
