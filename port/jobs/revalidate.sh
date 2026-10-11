#!/bin/bash
# Re-validate the memory-slim quantizer on Qwen2.5-1.5B (expect ~9.56 as the first port, Dom's MLX code 9.532).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; mkdir -p results models
echo "== quantize $(date -u +%T)"; $PY gptq_torch.py models/qwen15_src models/qwen15_slim4 4 --mse --lloyd --refine 3 --refit 2 --dtype float16 2>&1 | grep -vE "Warning|warn|Token indices"
echo "== ppl stream $(date -u +%T)"; $PY ppl_stream.py models/qwen15_slim4 --out results/ppl_stream_slim4.json 2>&1 | grep -E "perplexity|rror"
echo "== REVALIDATION DONE $(date -u +%T)"
