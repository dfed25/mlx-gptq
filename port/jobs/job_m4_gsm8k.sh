#!/bin/bash
# GSM8K (full test set, zero-shot CoT, greedy) on the 1.5B files: bf16 source, our v4+fine-tune Q8 file, bartowski Q4_K_M, our untuned v4 file, and the YAQA
# fine-tuned file when it exists. Then the paired comparison (McNemar) of our file against each other one. Does lower KL show up on a task?
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1; PY=~/venv-vllm/bin/python; mkdir -p results/gsm8k
run() { n=$1; shift; echo "== gsm8k $n $(date -u +%T)"; $PY gsm8k_eval.py models/qwen15_src "$@" --out results/gsm8k/$n.json --bs 32 --max-new 768 2>&1 | grep -E "accuracy|rror|Traceback|loaded" | cut -c1-220; }
run ours_v4tune_q8 --gguf models/gguf/q15_kq4_v4_ridge_tune_q8.gguf
run bartowski_q4km --gguf models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
run bf16
run ours_v4_q8 --gguf models/gguf/q15_kq4_v4_ridge_q8.gguf
f=models/gguf/q15_kq4_v4_yaqa_tune_q8.gguf; [ -f $f ] && run yaqa_tune_q8 --gguf $f
echo "== paired comparison"; $PY gsm8k_eval.py --compare results/gsm8k/ours_v4tune_q8.json results/gsm8k/bartowski_q4km.json results/gsm8k/bf16.json results/gsm8k/ours_v4_q8.json $( [ -f results/gsm8k/yaqa_tune_q8.json ] && echo results/gsm8k/yaqa_tune_q8.json ) 2>&1 | cut -c1-220
echo "== GSM8K M4 DONE $(date -u +%T)"
