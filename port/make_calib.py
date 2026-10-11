"""One-shot calibration mix (the v4 recipe) for any chat model, from local sources only:
WikiText-2 train + Python source (weighted x2) + chat-template GSM8K train / MMLU dev+val + UltraChat (train_sft sample) + Wikipedia
in 8 languages, all rendered with THIS model's chat template. Also writes the held-out sets chat_test.txt (GSM8K/MMLU test),
ultrachat_test.txt, multi_test.txt (disjoint slices).     python make_calib.py --tok MODEL_DIR --out calib.txt [--data ~/domq/data]
Needs: wikitext2_train.txt, python_calib.txt in the cwd; the HF datasets cache for openai/gsm8k and cais/mmlu; --data with
ultrachat_{train,test}_sft.jsonl and wiki_{de,fr,es,ja,zh,ru,pt,it}.txt (fetch_calib_extra.py)."""
import argparse, os, json, random
os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--tok", required=True); ap.add_argument("--out", default="calib.txt"); ap.add_argument("--data", default=os.path.expanduser("~/domq/data"))
ap.add_argument("--wiki-bytes", type=int, default=1500000); ap.add_argument("--py-bytes", type=int, default=1500000); ap.add_argument("--qa-bytes", type=int, default=1600000)
ap.add_argument("--chat-bytes", type=int, default=1500000); ap.add_argument("--multi-bytes", type=int, default=1500000); ap.add_argument("--test-bytes", type=int, default=450000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed); tok = AutoTokenizer.from_pretrained(a.tok); L = "ABCD"; LANGS = ["de", "fr", "es", "ja", "zh", "ru", "pt", "it"]
def render(msgs): return tok.apply_chat_template(msgs, tokenize=False) + "\n"
def take(items, nbytes):
    s = []; n = 0
    for it in items:
        if n >= nbytes: break
        s.append(it); n += len(it.encode())
    return "".join(s)
SUBJECTS = sorted(d for d in os.listdir(os.path.expanduser("~/.cache/huggingface/datasets/cais___mmlu")) if not d.endswith(".lock"))
def mmlu(split):
    out = []
    for subj in SUBJECTS:
        try: ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception: continue
        for r in ds: out.append(render([{"role": "user", "content": r["question"].strip() + "\n" + "\n".join(f"{L[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer with the letter."}, {"role": "assistant", "content": f"The answer is {L[r['answer']]}."}]))
    return out
def gsm(split): return [render([{"role": "user", "content": r["question"].strip()}, {"role": "assistant", "content": r["answer"].replace("####", "The answer is").strip()}]) for r in load_dataset("openai/gsm8k", "main", split=split)]
def ultra(path, nbytes):
    s = []; n = 0
    for line in open(path):
        if n >= nbytes: break
        t = render(json.loads(line)); s.append(t); n += len(t.encode())
    return "".join(s)
def multi(nbytes, test):
    per = nbytes // len(LANGS); out = []
    for l in LANGS: t = open(f"{a.data}/wiki_{l}.txt").read(); out.append(t[-per:] if test else t[:per])
    rng.shuffle(out); return "\n\n".join(out)
qa = gsm("train") + mmlu("validation") + mmlu("dev"); rng.shuffle(qa); qa_test = gsm("test") + mmlu("test"); rng.shuffle(qa_test)
parts = [open("wikitext2_train.txt").read()[:a.wiki_bytes], open("python_calib.txt").read()[:a.py_bytes], take(qa, a.qa_bytes), ultra(f"{a.data}/ultrachat_train_sft.jsonl", a.chat_bytes), multi(a.multi_bytes, False), open("python_calib.txt").read()[:a.py_bytes]]
open(a.out, "w").write("\n".join(parts)); open("chat_test.txt", "w").write(take(qa_test, a.test_bytes)); open("ultrachat_test.txt", "w").write(ultra(f"{a.data}/ultrachat_test_sft.jsonl", a.test_bytes)); open("multi_test.txt", "w").write(multi(a.test_bytes, True))
print(f"{a.out}: {os.path.getsize(a.out)/1e6:.2f} MB = wiki {len(parts[0].encode())/1e6:.1f} + python {2*len(parts[1].encode())/1e6:.1f} (x2) + QA {len(parts[2].encode())/1e6:.1f} + UltraChat {len(parts[3].encode())/1e6:.1f} + multilingual {len(parts[4].encode())/1e6:.1f} MB; held-outs chat_test / ultrachat_test / multi_test written ({len(tok.encode(open(a.out).read()))} tokens)")
