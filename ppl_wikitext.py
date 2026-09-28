"""Standard perplexity on WikiText-2 test: non-overlapping windows of 2048 tokens (first NWIN windows, for speed).
Usage: ppl_wikitext.py MODEL_PATH [NWIN]"""
import sys, math, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
def wikitext_ppl(model, tok, nwin=20, ctx=2048):
    ids = tok.encode(open("wikitext2_test.txt").read()); tot = 0.0; n = 0
    for w in range(nwin):
        x = mx.array(ids[w*ctx:(w+1)*ctx])[None]
        logits = model(x[:, :-1]); logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        nll = -mx.take_along_axis(logp, x[:, 1:][..., None], axis=-1).squeeze(-1); mx.eval(nll)
        tot += nll.sum().item(); n += nll.shape[1]; mx.clear_cache()
    return math.exp(tot / n), n
if __name__ == "__main__":
    m, tok = load(sys.argv[1]); nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    pp, n = wikitext_ppl(m, tok, nwin); print(f"{sys.argv[1]}: wikitext-2 perplexity {pp:.3f} over {n} tokens ({nwin} windows of 2048)")
