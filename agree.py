"""Where is the code ability lost? Teacher-forced on HumanEval (prompt + canonical solution), count how often a model's
top choice differs from the fp16 model's at solution positions ('disagreement'), and how often it is not the canonical
token ('error'). Then two layer scans on the 3-bit model:
  HEAL i : the 3-bit model with layer i replaced by its fp16 version  -> how much disagreement goes away
  HURT i : the fp16 model with only layer i replaced by its 3-bit version -> how much disagreement that layer alone creates
Usage: agree.py QUANT_MODEL_DIR [scan]"""
import sys, numpy as np, mlx.core as mx
from mlx_lm import load
from datasets import load_dataset
A, tok = load("models/qwen1.5b-fp16"); B, _ = load(sys.argv[1]); scan = len(sys.argv) > 2
ds = load_dataset("openai/openai_humaneval", split="test")
seqs = []
for ex in ds:
    p = tok.encode(ex["prompt"]); s = tok.encode(ex["prompt"] + ex["canonical_solution"]); seqs.append((s[:1024], min(len(p), 1023)))
def top1(model):
    out = []
    for s, start in seqs:
        lg = model(mx.array(s)[None, :-1]); a = mx.argmax(lg[0, start - 1:], axis=-1); mx.eval(a); out.append(np.array(a))
    return np.concatenate(out)
gold = np.concatenate([np.array(s[start:]) for s, start in seqs])
ref = top1(A); n = len(ref)
def report(name, pred): return f"{name}: disagreement with fp16 {100*(pred != ref).mean():.2f}% | not the canonical token {100*(pred != gold).mean():.2f}%"
print(f"{n} solution positions over 164 problems | fp16: not the canonical token {100*(ref != gold).mean():.2f}%")
base = top1(B); print(report(sys.argv[1], base))
if scan:
    L = len(A.model.layers); d0 = (base != ref).mean(); heal = []; hurt = []
    for i in range(L):
        keep = B.model.layers[i]; B.model.layers[i] = A.model.layers[i]; h = (top1(B) != ref).mean(); B.model.layers[i] = keep
        keepA = A.model.layers[i]; A.model.layers[i] = B.model.layers[i]; u = (top1(A) != ref).mean(); A.model.layers[i] = keepA
        heal.append(d0 - h); hurt.append(u); print(f"layer {i:2d}: HEAL removes {100*(d0-h):.2f} points of {100*d0:.2f} | HURT alone creates {100*u:.2f}", flush=True)
    order = np.argsort(heal)[::-1]; print("most healing layers:", [int(i) for i in order[:8]]); print("most hurting layers:", [int(i) for i in np.argsort(hurt)[::-1][:8]])
