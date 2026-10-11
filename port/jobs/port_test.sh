#!/bin/bash
# Validate the port on Qwen2.5-1.5B-Instruct (cached on the box): streaming fp16 ppl must give ~9.372, the full recipe
# at 4 bits must give ~9.53 (Dom's MLX code on the Mac: 9.532 dequantized), and vLLM must load the written checkpoint.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; mkdir -p results models
echo "== stream ppl fp16 base $(date -u +%T)"; $PY ppl_stream.py Qwen/Qwen2.5-1.5B-Instruct --out results/ppl_stream_base.json 2>&1 | grep -vE "Warning|warn" | tail -3
echo "== port quantize 4-bit $(date -u +%T)"; $PY gptq_torch.py Qwen/Qwen2.5-1.5B-Instruct models/qwen15_port4 4 --mse --lloyd --refine 3 --refit 2 --dtype float16 2>&1 | grep -vE "Warning|warn"
echo "== stream ppl of port output $(date -u +%T)"; $PY ppl_stream.py models/qwen15_port4 --out results/ppl_stream_port4.json 2>&1 | grep -vE "Warning|warn" | tail -2
echo "== vllm ppl of port output $(date -u +%T)"; $PY ../ppl_vllm.py models/qwen15_port4 --text ../wikitext2_test.txt --out results/ppl_vllm_port4.json 2>&1 | grep perplexity
echo "== PORT TEST DONE $(date -u +%T)"
