"""Calibration mix v3 = v2 (wiki + Python + chat-template GSM8K/MMLU) + real multi-turn chat (UltraChat, rendered with the model's chat
template) + multilingual Wikipedia (de fr es ja zh ru pt it). Held-out files: ultrachat_test.txt (test_sft split) and multi_test.txt (the tail
of each language's sample, disjoint from the calibration slice).
    python make_calib_v3.py --tok MODEL_DIR [--base calib_v2.txt] [--out calib_v3.txt] [--data ~/domq/data] [--chat-bytes 1500000] [--multi-bytes 1500000]"""
import argparse, os, json, random
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--tok", required=True); ap.add_argument("--base", default="calib_v2.txt"); ap.add_argument("--out", default="calib_v3.txt")
ap.add_argument("--data", default=os.path.expanduser("~/domq/data")); ap.add_argument("--chat-bytes", type=int, default=1500000); ap.add_argument("--multi-bytes", type=int, default=1500000)
ap.add_argument("--test-bytes", type=int, default=450000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed); tok = AutoTokenizer.from_pretrained(a.tok); LANGS = ["de", "fr", "es", "ja", "zh", "ru", "pt", "it"]
def chat(path, nbytes):
    s = []; n = 0
    for line in open(path):
        if n >= nbytes: break
        t = tok.apply_chat_template(json.loads(line), tokenize=False) + "\n"; s.append(t); n += len(t.encode())
    return "".join(s)
def multi(nbytes, test):
    per = nbytes // len(LANGS); out = []
    for l in LANGS:
        t = open(f"{a.data}/wiki_{l}.txt").read()
        out.append(t[-per:] if test else t[:per])                     # calibration from the head, held-out from the tail (samples are 2.5 MB; both slices < 0.2 MB)
    rng.shuffle(out); return "\n\n".join(out)
base = open(a.base).read(); uc = chat(f"{a.data}/ultrachat_train_sft.jsonl", a.chat_bytes); ml = multi(a.multi_bytes, False)
open(a.out, "w").write(base + "\n" + uc + "\n" + ml)
open("ultrachat_test.txt", "w").write(chat(f"{a.data}/ultrachat_test_sft.jsonl", a.test_bytes)); open("multi_test.txt", "w").write(multi(a.test_bytes, True))
print(f"{a.out}: base {len(base.encode())/1e6:.2f} MB + ultrachat {len(uc.encode())/1e6:.2f} MB ({len(tok.encode(uc))} tokens) + multilingual {len(ml.encode())/1e6:.2f} MB ({len(tok.encode(ml))} tokens); held-out ultrachat_test.txt {os.path.getsize('ultrachat_test.txt')/1e6:.2f} MB, multi_test.txt {os.path.getsize('multi_test.txt')/1e6:.2f} MB ({len(tok.encode(open('multi_test.txt').read()))} tokens)")
