"""Why does 3-bit fail HumanEval? Generate completions (greedy, lm-eval stop sequences), then classify each problem:
PASS, SYNTAX (prompt+completion does not parse), or LOGIC (parses, runs, but fails the tests / raises / times out).
Usage: humaneval_diag.py MODEL_DIR TAG"""
import sys, json, ast, subprocess, tempfile, os, mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from datasets import load_dataset
model_dir, tag = sys.argv[1], sys.argv[2]; model, tok = load(model_dir); sampler = make_sampler(temp=0.0)
ds = load_dataset("openai/openai_humaneval", split="test"); STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
rows = []
for ex in ds:
    pid = tok.encode(ex["prompt"]); gen = []
    for resp in stream_generate(model, tok, prompt=pid, max_tokens=512, sampler=sampler):
        gen.append(resp.token)
        if any(s in tok.decode(gen[-12:]) for s in STOPS): break
    out = tok.decode(pid + gen)[len(tok.decode(pid)):]                     # decode jointly so the first token's leading spaces survive
    cut = len(out)
    for s in STOPS:
        i = out.find(s)
        if i != -1: cut = min(cut, i)
    comp = out[:cut]; prog = ex["prompt"] + comp
    try: ast.parse(prog); syntax_ok = True
    except SyntaxError: syntax_ok = False
    verdict = "SYNTAX"
    if syntax_ok:
        src = prog + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src); path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, timeout=10); verdict = "PASS" if r.returncode == 0 else "LOGIC"
        except subprocess.TimeoutExpired: verdict = "LOGIC"
        os.unlink(path)
    rows.append({"task": ex["task_id"], "verdict": verdict, "completion": comp})
n = len(rows); c = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "SYNTAX", "LOGIC")}
print(f"{tag}: PASS {c['PASS']}/{n} = {100*c['PASS']/n:.1f}% | SYNTAX errors {c['SYNTAX']} ({100*c['SYNTAX']/n:.1f}%) | LOGIC failures {c['LOGIC']} ({100*c['LOGIC']/n:.1f}%)")
json.dump(rows, open(f"eval_out/diag_{tag}.json", "w"), indent=1)
