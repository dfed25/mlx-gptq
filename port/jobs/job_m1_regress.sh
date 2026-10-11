#!/bin/bash
# Regression of the refactored quantizer on Qwen2.5-1.5B: 4-bit int-zero g64 must reproduce ppl 9.625 (streamed).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python
echo "== regress 4-bit int-zero g64 $(date -u +%T)"; rm -rf models/reg_int4
$PY gptq_torch.py Qwen/Qwen2.5-1.5B-Instruct models/reg_int4 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype float16 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error"
$PY ppl_stream.py models/reg_int4 --out results/ppl_stream_reg_int4.json 2>&1 | grep -E "perplexity|rror"
echo "== REGRESS DONE $(date -u +%T)"
