#!/bin/bash
# Offline rows for the 32B models with memory-safe engine settings (eager, few sequences, bounded batched tokens).
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python
for row in "q32_port4_awq port/models/qwen32_port4_awq awq_marlin" "q32_awq models/qwen32_awq awq_marlin" "q32_gptq4 models/qwen32_gptq4 gptq_marlin"; do set -- $row
  echo "== ppl $1 $(date -u +%T)"; $PY ppl_vllm.py $2 --quant $3 --gpu-util 0.97 --eager --max-num-seqs 2 --max-batched 2048 --out results/ppl_$1.json > results/rerun_ppl_$1.log 2>&1; grep -E "perplexity|Available KV|Error" results/rerun_ppl_$1.log | tail -2 | cut -c1-150
  echo "== humaneval $1 $(date -u +%T)"; $PY humaneval_vllm.py $2 --quant $3 --gpu-util 0.97 --eager --max-num-seqs 6 --max-batched 1024 --max-len 1024 --out results/humaneval_$1.json > results/rerun_he_$1.log 2>&1; grep -E "pass@1|Available KV|Error" results/rerun_he_$1.log | tail -2 | cut -c1-150
done
echo "== RERUN32B DONE $(date -u +%T)"
