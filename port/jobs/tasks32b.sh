#!/bin/bash
# lm-eval on the int4 kernel for 32B models, with a per-model time budget of about 5 h on one A10G:
# ARC-e/c, WinoGrande and GSM8K in full; HellaSwag on its first 2500 examples; MMLU on the first 50 questions of each of the 57
# subjects (2850 questions). The same fixed subsets for every model.   usage: bash tasks32b.sh PATH:name:quant ...
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false; PY=~/venv-vllm/bin/python; mkdir -p results
ARGS="dtype=float16,gpu_memory_utilization=0.97,max_model_len=1536,enforce_eager=True,max_num_seqs=3,max_num_batched_tokens=1024,seed=0"
for spec in "$@"; do IFS=: read path name quant <<< "$spec"
  echo "=== tasks $name $(date -u +%T)"
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,quantization=$quant,$ARGS" --tasks arc_easy,arc_challenge,winogrande,gsm8k --batch_size 3 --seed 0 --output_path results/lmeval_${name}_a > results/lmeval_${name}_a.log 2>&1
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,quantization=$quant,$ARGS" --tasks hellaswag --limit 2500 --batch_size 3 --seed 0 --output_path results/lmeval_${name}_b > results/lmeval_${name}_b.log 2>&1
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,quantization=$quant,$ARGS" --tasks mmlu --limit 50 --batch_size 3 --seed 0 --output_path results/lmeval_${name}_c > results/lmeval_${name}_c.log 2>&1
  for s in a b c; do grep -E "^\|[a-z ]" results/lmeval_${name}_$s.log | grep -vE "^\| *-" | grep -E "arc_|hellaswag|winogrande|mmlu |gsm8k" | tr -s " " | cut -c1-90; grep -E "Error" results/lmeval_${name}_$s.log | grep -v "indexing errors" | tail -1; done
  echo "--- $name done $(date -u +%T)"
done
echo "== TASKS32 DONE $(date -u +%T)"
