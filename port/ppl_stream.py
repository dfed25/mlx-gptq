"""Dom's WikiText-2 protocol (20 windows of 2048 tokens) on a checkpoint streamed through the GPU one layer at a time,
so the fp16 baseline of a 27B model can be measured on a 24 GB card.   python ppl_stream.py MODEL [--nwin 20] [--dtype bfloat16]"""
import argparse, math, json, time, torch
from stream_util import Streamer
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--nwin", type=int, default=20); ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--dtype", default="float16"); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--text", default="wikitext2_test.txt"); ap.add_argument("--out", default=None); ap.add_argument("--gguf", default=None, help="read weights from this GGUF file (MODEL gives config + tokenizer)")
a = ap.parse_args(); dt = getattr(torch, a.dtype); t0 = time.time()
import os; DEV = os.environ.get("DOMQ_DEV", "cuda" if torch.cuda.is_available() else "cpu")
S = Streamer(a.model, dtype=dt, device=DEV, gguf=a.gguf)
ids = S.tok.encode(open(a.text).read()); assert len(ids) >= a.nwin * a.ctx
X = torch.tensor([ids[w * a.ctx:(w + 1) * a.ctx] for w in range(a.nwin)])
KW = S.capture_kwargs(X[:a.bs]); KW_last = S.capture_kwargs(X[-(a.nwin % a.bs or a.bs):]) if a.nwin % a.bs else KW
H = [S.embed(X[b:b + a.bs]) for b in range(0, a.nwin, a.bs)]
for li in range(S.n_layers):
    layer = S.load_layer(li)
    H = [S.run_layer(layer, h, S.kw_for(li, KW if h.shape[0] == a.bs else KW_last)) for h in H]
    S.free_layer(li)
tot = 0.0; n = 0
for h, b in zip(H, range(0, a.nwin, a.bs)):
    for i in range(h.shape[0]):
        lg = S.logits(h[i:i + 1])[0, :-1].float(); tgt = X[b + i, 1:].to(lg.device)
        tot += torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item(); n += tgt.numel(); del lg
ppl = math.exp(tot / n)
res = {"model": a.gguf or a.model, "backend": "stream", "dtype": a.dtype, "nwin": a.nwin, "ctx": a.ctx, "tokens": n, "ppl": ppl, "s": time.time() - t0}
print(json.dumps(res)); print(f"{a.gguf or a.model}: wikitext-2 perplexity {ppl:.3f} over {n} tokens ({a.nwin} windows of {a.ctx}), {time.time()-t0:.0f}s")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
