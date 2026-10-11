#!/bin/bash
# lm-eval task benchmarks on the int4 kernel for the 32B models (memory-safe engine settings for 19 GB of weights on a 22 GB card).
# usage: bash tasks32.sh PATH:name:quant ...
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false; PY=~/venv-vllm/bin/python; mkdir -p results
for spec in "$@"; do IFS=: read path name quant <<< "$spec"
  echo "=== tasks $name $(date -u +%T)"
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,quantization=$quant,dtype=float16,gpu_memory_utilization=0.97,max_model_len=1536,enforce_eager=True,max_num_seqs=3,max_num_batched_tokens=1024,seed=0" \
     --tasks arc_easy,arc_challenge,hellaswag,winogrande,mmlu,gsm8k --batch_size 3 --seed 0 --output_path results/lmeval_$name > results/lmeval_$name.log 2>&1
  grep -E "^\|[a-z ]" results/lmeval_$name.log | grep -vE "^\| *-" | grep -E "arc_|hellaswag|winogrande|mmlu |gsm8k" | tr -s " " | cut -c1-90
  grep -E "Error|error" results/lmeval_$name.log | grep -v "indexing errors" | tail -2
done
echo "== TASKS32 DONE $(date -u +%T)"
