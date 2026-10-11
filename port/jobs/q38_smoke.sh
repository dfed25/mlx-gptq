#!/bin/bash
# Smoke test of the hybrid-architecture port on Qwen3.5-0.8B, plus a regression check on Qwen2.5-1.5B. Writes results/q38_smoke.ok on success.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; mkdir -p results models; rm -f results/q38_smoke.ok
echo "== download 0.8B $(date -u +%T)"; $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='models/q35_08b_src'))" 2>&1 | tail -1
echo "== regression: Qwen2.5-1.5B streamed fp16 ppl (expect 9.371) $(date -u +%T)"; $PY ppl_stream.py Qwen/Qwen2.5-1.5B-Instruct 2>&1 | grep -E "perplexity|rror"
echo "== 0.8B fp16: streamed vs vLLM (must agree) $(date -u +%T)"
$PY ppl_stream.py models/q35_08b_src --dtype bfloat16 --out results/ppl_stream_q35_08b_fp16.json 2>&1 | grep -E "perplexity|rror"
$PY ../ppl_vllm.py models/q35_08b_src --dtype bfloat16 --text wikitext2_test.txt --out results/ppl_q35_08b_fp16.json 2>&1 | grep -E "perplexity|rror"
echo "== 0.8B quantize 4-bit int-zero $(date -u +%T)"; $PY gptq_torch.py models/q35_08b_src models/q35_08b_int4 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype bfloat16 2>&1 | grep -E "layers \(|^layer  0|^layer 23|saved|Traceback|Error"
$PY ppl_stream.py models/q35_08b_int4 --dtype bfloat16 --out results/ppl_stream_q35_08b_int4.json 2>&1 | grep -E "perplexity|rror"
echo "== 0.8B awq packing + kernel $(date -u +%T)"; $PY codes2awq.py models/q35_08b_int4 models/q35_08b_int4_awq 2>&1 | tail -1
$PY ../ppl_vllm.py models/q35_08b_int4_awq --quant awq_marlin --text wikitext2_test.txt --out results/ppl_q35_08b_int4_awq.json 2>&1 | grep -E "perplexity|rror"
$PY ../humaneval_vllm.py models/q35_08b_int4_awq --quant awq_marlin --out results/humaneval_q35_08b_int4_awq.json 2>&1 | grep -E "pass@1|rror"
$PY - <<'PYEOF'
import json, os
try:
    a = json.load(open("results/ppl_stream_q35_08b_fp16.json"))["ppl"]; b = json.load(open("results/ppl_q35_08b_fp16.json"))["ppl"]
    c = json.load(open("results/ppl_stream_q35_08b_int4.json"))["ppl"]; d = json.load(open("results/ppl_q35_08b_int4_awq.json"))["ppl"]
    ok = abs(a - b) < 0.05 and abs(c - d) < 0.05 and c < 2 * a
    print(f"streamed fp16 {a:.3f} vs vLLM fp16 {b:.3f}; int4 exact {c:.3f} vs kernel {d:.3f} -> {'OK' if ok else 'MISMATCH'}")
    if ok: open("results/q38_smoke.ok", "w").write("ok")
except Exception as e: print("smoke incomplete:", e)
PYEOF
echo "== SMOKE DONE $(date -u +%T)"
