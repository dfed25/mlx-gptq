#!/bin/bash
# After run_q38.sh: lm-eval tasks for the Qwen3.8-27B integer-offset model on the int4 kernel, same subsets as the 32B runs.
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false; PY=~/venv-vllm/bin/python
while pgrep -f "^bash run_q38.sh" >/dev/null; do sleep 120; done
M=port/models/q38_27b_int4_awq; [ -f $M/config.json ] || { echo "no 27B awq model; aborting"; exit 1; }
ARGS="pretrained=$M,quantization=awq_marlin,dtype=bfloat16,gpu_memory_utilization=0.95,max_model_len=1536,enforce_eager=True,max_num_seqs=4,max_num_batched_tokens=1024,seed=0"
echo "=== tasks q38_int4_awq $(date -u +%T)"
$PY -m lm_eval --model vllm --model_args "$ARGS" --tasks arc_easy,arc_challenge,winogrande,gsm8k --batch_size 4 --seed 0 --output_path results/lmeval_q38_a > results/lmeval_q38_a.log 2>&1
$PY -m lm_eval --model vllm --model_args "$ARGS" --tasks hellaswag --limit 2500 --batch_size 4 --seed 0 --output_path results/lmeval_q38_b > results/lmeval_q38_b.log 2>&1
$PY -m lm_eval --model vllm --model_args "$ARGS" --tasks mmlu --limit 50 --batch_size 4 --seed 0 --output_path results/lmeval_q38_c > results/lmeval_q38_c.log 2>&1
for s in a b c; do grep -E "^\|[a-z ]" results/lmeval_q38_$s.log | grep -vE "^\| *-" | grep -E "arc_|hellaswag|winogrande|mmlu |gsm8k" | tr -s " " | cut -c1-90; grep -E "Error" results/lmeval_q38_$s.log | grep -v "indexing errors" | tail -1; done
echo "== TASKS27 DONE $(date -u +%T)"
