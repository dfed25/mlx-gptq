#!/bin/bash
# Redo the 27B kernel ppl + HumanEval (the driver ran them from ~/domq where the data files were missing on m1), after tasks27.sh.
set -u; cd ~/domq; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; M=port/models/q38_27b_int4_awq
while pgrep -f "^bash tasks27.sh" >/dev/null; do sleep 120; done
cp -n port/humaneval.parquet port/wikitext2_test.txt . 2>/dev/null
echo "== kernel ppl $(date -u +%T)"; $PY ppl_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 1 --max-batched 512 --text wikitext2_test.txt --out results/ppl_q38_int4_awq.json > results/log_ppl_q38.log 2>&1; grep -E "perplexity|Error" results/log_ppl_q38.log | grep -v "indexing errors" | tail -1
echo "== kernel humaneval $(date -u +%T)"; $PY humaneval_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 6 --max-batched 1024 --max-len 1024 --out results/humaneval_q38_int4_awq.json > results/log_he_q38.log 2>&1; grep -E "pass@1|Error" results/log_he_q38.log | grep -v "indexing errors" | tail -1
echo "== Q38 FIX DONE $(date -u +%T)"
