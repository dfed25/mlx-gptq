#!/bin/bash
# Relaunch of the 32B run from the quantization step (fp16 baseline 4.761 already in results/ppl_stream_q32_fp16.json).
# A two-layer smoke test with peak-memory logging must pass before the full run.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; SRC=../models/qwen32_src
rm -rf models/qwen32_port4 models/qwen32_smoke
echo "== smoke 2 layers $(date -u +%T)"; $PY gptq_torch.py $SRC models/qwen32_smoke 4 --mse --lloyd --refine 3 --refit 2 --dtype float16 --max-layers 2 2>&1 | grep -vE "Warning|warn|Token indices"
grep -q "smoke test finished" <($PY - <<'PYEOF'
import os; print("smoke test finished" if os.path.exists("models/qwen32_smoke/layer001.safetensors") else "smoke failed")
PYEOF
) || { echo "SMOKE FAILED"; exit 1; }
rm -rf models/qwen32_smoke
echo "== quantize 32B $(date -u +%T)"; $PY gptq_torch.py $SRC models/qwen32_port4 4 --mse --lloyd --refine 3 --refit 2 --dtype float16 2>&1 | grep -vE "Warning|warn|Token indices"
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
