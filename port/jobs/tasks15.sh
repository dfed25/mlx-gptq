#!/bin/bash
# After tasks32.sh: the 1.5B trio on the int4 kernel (Dom's integer-offset model, Qwen AWQ, Qwen GPTQ-Int4), normal settings.
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false; PY=~/venv-vllm/bin/python
while pgrep -f "^bash tasks32.sh" >/dev/null || [ ! -f port/models/cand_norotint4_awq/.copied ]; do sleep 120; done
for spec in "port/models/cand_norotint4_awq:int4_awq_15b:awq_marlin" "models/qwen_awq:awq_15b:awq_marlin" "models/qwen_gptq4:gptq4_15b:gptq_marlin"; do IFS=: read path name quant <<< "$spec"
  echo "=== tasks $name $(date -u +%T)"
  $PY -m lm_eval --model vllm --model_args "pretrained=$path,quantization=$quant,dtype=float16,gpu_memory_utilization=0.7,max_model_len=4096,seed=0" --tasks arc_easy,arc_challenge,hellaswag,winogrande,mmlu,gsm8k --batch_size auto --seed 0 --output_path results/lmeval_$name > results/lmeval_$name.log 2>&1
  grep -E "^\|[a-z ]" results/lmeval_$name.log | grep -vE "^\| *-" | grep -E "arc_|hellaswag|winogrande|mmlu |gsm8k" | tr -s " " | cut -c1-90
done
echo "== TASKS15 DONE $(date -u +%T)"
