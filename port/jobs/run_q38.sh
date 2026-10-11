#!/bin/bash
# Qwen3.8-27B: Dom's recipe + integer zero points via the port. Gated on results/q38_smoke.ok and on the download.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; SRC=models/q38_27b_src; OUT=models/q38_27b_int4
while [ ! -f results/dl_q38.done ] || ! grep -q "SMOKE DONE" results/q38_smoke.out 2>/dev/null; do sleep 60; done
[ -f results/q38_smoke.ok ] || { echo "SMOKE NOT OK; aborting"; exit 1; }
echo "== fp16 baseline ppl, streamed $(date -u +%T)"; $PY ppl_stream.py $SRC --dtype bfloat16 --bs 2 --out results/ppl_stream_q38_fp16.json 2>&1 | grep -E "perplexity|rror"
echo "== smoke 2 layers $(date -u +%T)"; rm -rf models/q38_smoke2; $PY gptq_torch.py $SRC models/q38_smoke2 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype bfloat16 --max-layers 4 2>&1 | grep -E "layers \(|^layer|^    |smoke|Traceback|Error"
[ -f models/q38_smoke2/layer003.safetensors ] || { echo "SMOKE2 FAILED"; exit 1; }; rm -rf models/q38_smoke2
echo "== quantize 27B int-zero $(date -u +%T)"; rm -rf $OUT; $PY gptq_torch.py $SRC $OUT 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype bfloat16 2>&1 | grep -E "^layer|saved|Traceback|Error"
echo "== exact ppl, streamed $(date -u +%T)"; $PY ppl_stream.py $OUT --dtype bfloat16 --bs 2 --out results/ppl_stream_q38_int4.json 2>&1 | grep -E "perplexity|rror"
echo "== awq packing $(date -u +%T)"; $PY codes2awq.py $OUT ${OUT}_awq 2>&1 | tail -1
cd ~/domq; export GPU_UTIL=0.95 MAX_LEN=2048 EXTRA_SERVE="--enforce-eager --max-num-seqs 16"; M=port/${OUT}_awq
echo "== kernel ppl $(date -u +%T)"; $PY ppl_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 1 --max-batched 512 --out results/ppl_q38_int4_awq.json > results/log_ppl_q38.log 2>&1; grep -E "perplexity|Error" results/log_ppl_q38.log | tail -1
echo "== kernel humaneval $(date -u +%T)"; $PY humaneval_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 6 --max-batched 1024 --max-len 1024 --out results/humaneval_q38_int4_awq.json > results/log_he_q38.log 2>&1; grep -E "pass@1|Error" results/log_he_q38.log | tail -1
echo "== serving $(date -u +%T)"; bash bench_serve.sh q38_int4_awq $M awq_marlin 2>&1 | grep -E "throughput|TTFT|TPOT|failed"
echo "== Q38 ALL DONE $(date -u +%T)"
