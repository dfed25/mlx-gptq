#!/bin/bash
# Qwen3.8-27B at 2 bits, Dom's recipe with fp16 offsets, mixed calibration (same calib_mixed.txt as the m1 4-bit run).
# Quality-only (no 2-bit kernel): streamed ppl on WikiText-2 and on the code corpus. Waits for the source download.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; SRC=models/q38_27b_src
while [ ! -f results/dl_q38_m2.done ]; do sleep 60; done
echo "== 2-bit, Dom's fp16 offsets, mixed calibration $(date -u +%T)"; rm -rf models/q38_27b_int2_mix
$PY gptq_torch.py $SRC models/q38_27b_int2_mix 2 --mse --lloyd --refine 3 --refit 2 --dtype bfloat16 --calib calib_mixed.txt 2>&1 | grep -E "^layer (0|31|63) |saved|Traceback|Error"
$PY ppl_stream.py models/q38_27b_int2_mix --dtype bfloat16 --bs 2 --out results/ppl_stream_q38_int2_mix_wiki.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/q38_27b_int2_mix --dtype bfloat16 --bs 2 --text code_test.txt --out results/ppl_stream_q38_int2_mix_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
echo "== Q38 2BIT DONE $(date -u +%T)"
