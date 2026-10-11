#!/bin/bash
# The v3-calibrated + KL-fine-tuned 1.5B as a FILE (Q8_0 embeddings, 946 MiB) on the four non-wiki held-outs, to compare file against file with bartowski's Q4_K_M.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
f=models/gguf/q15_kq4_v3_ridge_tune_q8.gguf
for text in code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld q15_kq4_v3_ridge_tune_q8 on $tag"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text $text --nwin 8 --out results/kld/q15_kq4_v3_ridge_tune_q8_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== FINALKL M4 DONE $(date -u +%T)"
