#!/bin/bash
# Group-32 references on Qwen2.5-1.5B: uniform 4-bit and uniform 5-bit int-zero (the Q4_1 / Q5_1 equivalents), streamed ppl on wiki + code.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results
for b in 4 5; do echo "== uniform ${b}-bit int-zero g32 $(date -u +%T)"; rm -rf models/q15_g32_u$b
  $PY gptq_torch.py models/qwen15_src models/q15_g32_u$b $b --group 32 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype float16 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error"
  $PY ppl_stream.py models/q15_g32_u$b --out results/ppl_stream_q15_g32_u$b.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py models/q15_g32_u$b --text code_test.txt --out results/ppl_stream_q15_g32_u${b}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
done
echo "== G32 DONE $(date -u +%T)"
