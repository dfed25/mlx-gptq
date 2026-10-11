#!/bin/bash
# Task benchmarks via lm-eval-harness (vLLM backend) for finished variants.  usage: bash tasks_box.sh PATH:name ...
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false; PY=~/venv-vllm/bin/python
while pgrep -f "[n]b_box.sh" >/dev/null; do sleep 60; done
for spec in "$@"; do IFS=: read path name <<< "$spec"
  echo "=== tasks $name $(date -u +%T)"
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,dtype=float16,gpu_memory_utilization=0.7,max_model_len=4096,seed=0" \
     --tasks arc_easy,arc_challenge,hellaswag,winogrande,mmlu,gsm8k --batch_size auto --seed 0 --output_path results/lmeval_$name 2>&1 | grep -E "^\|[a-z ]" | grep -vE "^\| *-" 
done
echo "== TASKS DONE $(date -u +%T)"
