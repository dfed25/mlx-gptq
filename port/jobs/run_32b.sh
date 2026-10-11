#!/bin/bash
# Qwen2.5-32B-Instruct: Dom's recipe at 4 bits via the PyTorch port, then the benchmark rows. Gated on the 1.5B validation
# (port must reproduce Dom's 9.532 within 0.09) and on the downloads.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; mkdir -p results models
while ! grep -q "PORT TEST DONE" results/port_test.out 2>/dev/null; do sleep 60; done
PPL=$($PY -c "import json; print(json.load(open('results/ppl_stream_port4.json'))['ppl'])" 2>/dev/null || echo 99)
if $PY -c "import sys; sys.exit(0 if float('$PPL') < 9.62 else 1)"; then echo "validation ok: port 4-bit ppl $PPL (Dom's MLX code: 9.532)"; else echo "VALIDATION FAILED: ppl $PPL; aborting"; exit 1; fi
while [ ! -f results/dl32.done ]; do sleep 60; done
SRC=../models/qwen32_src
echo "== fp16 baseline ppl, streamed $(date -u +%T)"; $PY ppl_stream.py $SRC --dtype float16 --bs 2 --out results/ppl_stream_q32_fp16.json 2>&1 | grep -E "perplexity|rror"
echo "== quantize 32B $(date -u +%T)"; $PY gptq_torch.py $SRC models/qwen32_port4 4 --mse --lloyd --refine 3 --refit 2 --dtype float16 2>&1 | grep -vE "Warning|warn"
echo "== ppl of exact output, streamed $(date -u +%T)"; $PY ppl_stream.py models/qwen32_port4 --bs 2 --out results/ppl_stream_q32_port4.json 2>&1 | grep -E "perplexity|rror"
echo "== awq export $(date -u +%T)"; $PY codes2awq.py models/qwen32_port4 models/qwen32_port4_awq 2>&1 | tail -1
cd ~/domq; export GPU_UTIL=0.95 MAX_LEN=2048 EXTRA_SERVE="--enforce-eager --max-num-seqs 16"
for row in "q32_port4_awq port/models/qwen32_port4_awq awq_marlin" "q32_awq models/qwen32_awq awq_marlin" "q32_gptq4 models/qwen32_gptq4 gptq_marlin"; do set -- $row
  echo "=== $1  $2  $3  $(date -u +%T)"
  $PY ppl_vllm.py $2 --quant $3 --gpu-util 0.95 --out results/ppl_$1.json 2>&1 | grep -E "perplexity|rror" | tail -2
  $PY humaneval_vllm.py $2 --quant $3 --gpu-util 0.95 --out results/humaneval_$1.json 2>&1 | grep -E "pass@1|rror" | tail -2
  bash bench_serve.sh $1 $2 $3
done
echo "== 32B ALL DONE $(date -u +%T)"
