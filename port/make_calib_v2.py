"""Calibration mix v2: the v1 mix (WikiText-2 train + Python source, calib_mixed.txt) plus CHAT-TEMPLATE-FORMATTED instruction
data from the local HF datasets cache (GSM8K train with worked answers, MMLU dev+validation questions with the answer letter),
rendered with the model's own chat template so the special tokens and the system prompt are part of the calibration stream.
Also writes a held-out chat test file (GSM8K test + MMLU test, disjoint from the calibration items) for the KL evaluator.
    python make_calib_v2.py --tok MODEL_DIR [--base calib_mixed.txt] [--out calib_v2.txt] [--test chat_test.txt] [--chat-bytes 1600000]"""
import argparse, os, random
os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--tok", required=True); ap.add_argument("--base", default="calib_mixed.txt"); ap.add_argument("--out", default="calib_v2.txt")
ap.add_argument("--test", default="chat_test.txt"); ap.add_argument("--chat-bytes", type=int, default=1600000); ap.add_argument("--test-bytes", type=int, default=450000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed); tok = AutoTokenizer.from_pretrained(a.tok)
def render(user, assistant):
    return tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], tokenize=False) + "\n"
L = "ABCD"
def mmlu_items(split):
    out = []
    for subj in SUBJECTS:
        try: ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception as e: print(f"skip {subj}/{split}: {e}"); continue
        for r in ds:
            q = r["question"].strip() + "\n" + "\n".join(f"{L[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer with the letter."
            out.append(render(q, f"The answer is {L[r['answer']]}."))
    return out
def gsm_items(split):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [render(r["question"].strip(), r["answer"].replace("####", "The answer is").strip()) for r in ds]
SUBJECTS = sorted(d for d in os.listdir(os.path.expanduser("~/.cache/huggingface/datasets/cais___mmlu")) if not d.endswith(".lock"))
calib = gsm_items("train") + mmlu_items("validation") + mmlu_items("dev"); rng.shuffle(calib)
test = gsm_items("test") + mmlu_items("test"); rng.shuffle(test)
def take(items, nbytes):
    s = []; n = 0
    for it in items:
        if n >= nbytes: break
        s.append(it); n += len(it.encode())
    return "".join(s)
chat = take(calib, a.chat_bytes); base = open(a.base).read()
open(a.out, "w").write(base + "\n" + chat); open(a.test, "w").write(take(test, a.test_bytes))
nt = len(tok.encode(chat)); print(f"{a.out}: base {len(base.encode())/1e6:.2f} MB + chat {len(chat.encode())/1e6:.2f} MB ({len(calib)} items available, {nt} chat tokens); {a.test}: {os.path.getsize(a.test)/1e6:.2f} MB held-out ({len(test)} items available)")
print("sample:\n" + chat[:600])
