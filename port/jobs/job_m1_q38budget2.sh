#!/bin/bash
# Sensitivity pass + budget only (no disk needed): K-quant forms q4_k/q5_k/q6_k, target = the bpw that matches unsloth's file size with Q4_K embeddings / Q6_K head.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld budgets
U=$(stat -c %s models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf); O=$(stat -c %s models/gguf/q38_dom_q4_k_e6.gguf)
T=$($PY -c "
import json; c=json.load(open('models/q38_27b_src/config.json')); c=c.get('text_config',c); V=c['vocab_size']; H=c['hidden_size']; N=24.33e9
O=$O - V*H*(6.5625-4.5)/8; print(f'{4.5+8*($U-O)/N:.3f}')")
echo "== budget: unsloth $U bytes, ours(e6) $O bytes -> target $T bpw over the linears $(date -u +%T)"
$PY bit_budget.py models/q38_27b_src budgets/q38_kq456_$T.json --target-bpw $T --kq-forms q4_k,q5_k,q6_k --calib calib_mixed.txt --dtype bfloat16 --bs 4 --table budgets/q38_kq456_table.json 2>&1 | tee results/q38_kqdyn_budget.log | grep -E "budget:|wrote|loaded|Traceback|Error|scored" | cut -c1-240
[ -f budgets/q38_kq456_$T.json ] && echo "== Q38 BUDGET2 DONE $(date -u +%T)" || echo "BUDGET FAILED"
