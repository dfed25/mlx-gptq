#!/bin/bash
# Qwen2.5-32B-Instruct, Dom's recipe + integer zero points (--int-zero): smoke 2 layers, full run, exact ppl, lossless AWQ
# packing, kernel ppl + HumanEval + serving. Memory-safe engine settings for the 19 GB model on the 22 GB card.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; SRC=../models/qwen32_src; OUT=models/qwen32_int4
rm -rf models/qwen32_int_smoke $OUT ${OUT}_awq
echo "== smoke 2 layers $(date -u +%T)"; $PY gptq_torch.py $SRC models/qwen32_int_smoke 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype float16 --max-layers 2 2>&1 | grep -E "^layer|^    mlp.down|smoke|Traceback|Error"
[ -f models/qwen32_int_smoke/layer001.safetensors ] || { echo "SMOKE FAILED"; exit 1; }; rm -rf models/qwen32_int_smoke
echo "== quantize 32B int-zero $(date -u +%T)"; $PY gptq_torch.py $SRC $OUT 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype float16 2>&1 | grep -E "^layer|saved|Traceback|Error"
echo "== exact ppl, streamed $(date -u +%T)"; $PY ppl_stream.py $OUT --bs 2 --out results/ppl_stream_q32_int4.json 2>&1 | grep -E "perplexity|rror"
echo "== awq packing (lossless) $(date -u +%T)"; $PY codes2awq.py $OUT ${OUT}_awq 2>&1 | tail -1
cd ~/domq; export GPU_UTIL=0.95 MAX_LEN=2048 EXTRA_SERVE="--enforce-eager --max-num-seqs 16"; M=port/${OUT}_awq
echo "== kernel ppl $(date -u +%T)"; $PY ppl_vllm.py $M --quant awq_marlin --gpu-util 0.97 --eager --max-num-seqs 1 --max-batched 512 --out results/ppl_q32_int4_awq.json > results/rerun_ppl_q32_int4_awq.log 2>&1; grep -E "perplexity" results/rerun_ppl_q32_int4_awq.log | tail -1
echo "== kernel humaneval $(date -u +%T)"; $PY humaneval_vllm.py $M --quant awq_marlin --gpu-util 0.97 --eager --max-num-seqs 6 --max-batched 1024 --max-len 1024 --out results/humaneval_q32_int4_awq.json > results/rerun_he_q32_int4_awq.log 2>&1; grep -E "pass@1" results/rerun_he_q32_int4_awq.log | tail -1
echo "== serving $(date -u +%T)"; bash bench_serve.sh q32_int4_awq $M awq_marlin 2>&1 | grep -E "throughput|TTFT|TPOT|failed"
echo "== 32B INT ALL DONE $(date -u +%T)"
