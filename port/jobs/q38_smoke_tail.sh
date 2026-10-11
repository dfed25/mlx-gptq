#!/bin/bash
# Redo the packing + kernel steps of the 0.8B smoke test and write the ok marker if everything agrees.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; rm -f results/q38_smoke.ok
echo "== 0.8B awq packing + kernel (retry) $(date -u +%T)"; rm -rf models/q35_08b_int4_awq; $PY codes2awq.py models/q35_08b_int4 models/q35_08b_int4_awq 2>&1 | tail -1; ls models/q35_08b_int4_awq | tr "\n" " "; echo
$PY ../ppl_vllm.py models/q35_08b_int4_awq --quant awq_marlin --dtype bfloat16 --text wikitext2_test.txt --out results/ppl_q35_08b_int4_awq.json > results/log_awq_08b.txt 2>&1; grep -E "perplexity" results/log_awq_08b.txt | tail -1 || grep -E "Error|error" results/log_awq_08b.txt | tail -3
$PY ../humaneval_vllm.py models/q35_08b_int4_awq --quant awq_marlin --dtype bfloat16 --out results/humaneval_q35_08b_int4_awq.json 2>&1 | grep -E "pass@1|rror" | tail -1
$PY - <<'PYEOF'
import json
try:
    a = json.load(open("results/ppl_stream_q35_08b_fp16.json"))["ppl"]; b = json.load(open("results/ppl_q35_08b_fp16.json"))["ppl"]
    c = json.load(open("results/ppl_stream_q35_08b_int4.json"))["ppl"]; d = json.load(open("results/ppl_q35_08b_int4_awq.json"))["ppl"]
    ok = abs(a - b) < 0.05 and abs(c - d) < 0.05 and c < 2 * a
    print(f"streamed fp16 {a:.3f} vs vLLM fp16 {b:.3f}; int4 exact {c:.3f} vs kernel {d:.3f} -> {'OK' if ok else 'MISMATCH'}")
    if ok: open("results/q38_smoke.ok", "w").write("ok")
except Exception as e: print("smoke incomplete:", e)
PYEOF
echo "== SMOKE TAIL DONE $(date -u +%T)"
