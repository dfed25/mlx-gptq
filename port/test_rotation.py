"""Rotation equivalence test on CPU: logits of the rotated model must match the original on a few windows."""
import sys, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
src, rot = sys.argv[1], sys.argv[2]; ctx = 512; nwin = 3
tok = AutoTokenizer.from_pretrained(src); ids = tok.encode(open("wikitext2_test.txt").read())[: nwin * ctx]
X = torch.tensor(ids).view(nwin, ctx)
def nll(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval(); tot = 0.0; n = 0; last = None
    with torch.no_grad():
        for i in range(nwin):
            lg = m(X[i:i+1]).logits[0, :-1]; tot += torch.nn.functional.cross_entropy(lg, X[i, 1:], reduction="sum").item(); n += ctx - 1; last = lg
    return math.exp(tot / n), last
p0, l0 = nll(src); p1, l1 = nll(rot)
print(f"original ppl {p0:.4f}  rotated ppl {p1:.4f}  max |logit diff| {(l0 - l1).abs().max():.3e}  (fp32 CPU, {nwin} windows of {ctx})")
