#!/bin/bash
# Sensitivity pass on Qwen2.5-1.5B at group 32, allocations for several targets.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results budgets
for t in 4.0 4.5 5.0 5.5; do echo "== budget target $t bpw $(date -u +%T)"; $PY bit_budget.py models/qwen15_src budgets/q15_g32_bpw$t.json --target-bpw $t --group 32 --widths 3,4,5,8 2>&1 | grep -E "budget:|wrote|Traceback|Error"; done
echo "== BUDGET DONE $(date -u +%T)"
