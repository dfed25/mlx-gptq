"""GSM8K test-set accuracy (1319 questions) of a model or of a GGUF file (dequantized exactly: the file is what is scored).
Zero-shot chain of thought with the chat template and Qwen's instruction "Please reason step by step, and put your final
answer within \\boxed{}", greedy decoding, no repetition penalty (same settings for every model). The prediction is the last
number inside the last \\boxed{} (else the last number in the answer); a question is correct when it matches the gold exactly.
    python gsm8k_eval.py SRC [--gguf FILE] --out results/gsm8k/NAME.json [--n 1319] [--bs 32] [--max-new 512]
    python gsm8k_eval.py --compare A.json B.json [C.json ...]      # paired comparison of A with each other file (McNemar exact test)"""
import argparse, json, os, re, time, math, torch
os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("src", nargs="?"); ap.add_argument("--gguf", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--n", type=int, default=1319); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max-new", type=int, default=512); ap.add_argument("--dtype", default="float16")
ap.add_argument("--compare", nargs="+", default=None)
a = ap.parse_args()
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def to_num(s):
    try: return float(s.replace(",", "").replace("$", "").strip().rstrip("."))
    except Exception: return None
def extract(ans):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", ans)
    for src in ([boxed[-1]] if boxed else []) + [ans]:
        nums = NUM.findall(src)
        if nums:
            v = to_num(nums[-1])
            if v is not None: return v
    return None

if a.compare:
    R = [json.load(open(f)) for f in a.compare]; names = [os.path.basename(f)[:-5] for f in a.compare]
    for n, r in zip(names, R): print(f"  {n:32s} {100 * r['acc']:6.2f}%  ({sum(r['correct'])}/{len(r['correct'])})  truncated {r.get('truncated', '?')}")
    A = R[0]["correct"]
    for n, r in zip(names[1:], R[1:]):
        B = r["correct"]; ab = sum(x and not y for x, y in zip(A, B)); ba = sum(y and not x for x, y in zip(A, B)); d = ab + ba
        p = min(1.0, 2 * sum(math.comb(d, k) for k in range(min(ab, ba) + 1)) / 2 ** d) if d else 1.0
        print(f"  {names[0]} vs {n}: both right {sum(x and y for x, y in zip(A, B))}, only {names[0]} {ab}, only {n} {ba}, McNemar exact p = {p:.3f}")
    raise SystemExit

from stream_util import Streamer
t0 = time.time(); S = Streamer(a.src, dtype=getattr(torch, a.dtype), device="cuda", gguf=a.gguf)
for li in range(S.n_layers): S.load_layer(li)                                                  # the whole model on the GPU (small models only)
model = S.model.eval(); tok = S.tok; tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
eos = sorted({tok.eos_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set()))
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test"); qs = [r["question"].strip() for r in ds][:a.n]; golds = [to_num(r["answer"].split("####")[-1]) for r in ds][:a.n]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + INSTR}], tokenize=False, add_generation_prompt=True) for q in qs]
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])); outs = [None] * len(prompts); trunc = 0
name = os.path.basename(a.gguf) if a.gguf else a.src; print(f"{name}: model loaded ({time.time() - t0:.0f}s), {len(prompts)} questions, batch {a.bs}, max {a.max_new} new tokens", flush=True)
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda"); L = enc.input_ids.shape[1]
    with torch.no_grad(): g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    for j, i in enumerate(idx):
        gen = g[j, L:]; keep = gen != tok.pad_token_id; outs[i] = tok.decode(gen[keep], skip_special_tokens=True); trunc += int(keep.sum().item() >= a.max_new and gen[-1].item() not in eos)
    if (b // a.bs) % 10 == 0: print(f"  {b + len(idx)}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
preds = [extract(o) for o in outs]; correct = [p is not None and gd is not None and abs(p - gd) < 1e-6 for p, gd in zip(preds, golds)]; acc = sum(correct) / len(correct)
print(f"{name}: GSM8K accuracy {100 * acc:.2f}% ({sum(correct)}/{len(correct)}), truncated answers {trunc}, no number {sum(p is None for p in preds)}, {time.time() - t0:.0f}s", flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.src, "gguf": a.gguf, "n": len(correct), "acc": acc, "truncated": trunc, "correct": correct, "preds": preds, "golds": golds, "answers": outs}, open(a.out, "w"))
