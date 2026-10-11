#!/bin/bash
# After run_32b_b.sh: redo any 32B row step whose result file is missing, with FULL logs kept in results/rerun_*.log.
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0 GPU_UTIL=0.95 MAX_LEN=2048 EXTRA_SERVE="--enforce-eager --max-num-seqs 16"; PY=~/venv-vllm/bin/python
while pgrep -f "[r]un_32b_b.sh" >/dev/null; do sleep 60; done
for row in "q32_port4_awq port/models/qwen32_port4_awq awq_marlin" "q32_awq models/qwen32_awq awq_marlin" "q32_gptq4 models/qwen32_gptq4 gptq_marlin"; do set -- $row
  [ -f results/ppl_$1.json ] || { echo "== rerun ppl $1 $(date -u +%T)"; $PY ppl_vllm.py $2 --quant $3 --gpu-util 0.95 --out results/ppl_$1.json > results/rerun_ppl_$1.log 2>&1; grep -E "perplexity|Error" results/rerun_ppl_$1.log | tail -2; }
  [ -f results/humaneval_$1.json ] || { echo "== rerun humaneval $1 $(date -u +%T)"; $PY humaneval_vllm.py $2 --quant $3 --gpu-util 0.95 --out results/humaneval_$1.json > results/rerun_he_$1.log 2>&1; grep -E "pass@1|Error" results/rerun_he_$1.log | tail -2; }
  [ -f results/bench_$1.json ] || { echo "== rerun bench $1 $(date -u +%T)"; bash bench_serve.sh $1 $2 $3 2>&1 | tail -3; }
done
echo "== RERUN32 DONE $(date -u +%T)"; ls results/ | grep -E "q32" | tr "\n" " "
