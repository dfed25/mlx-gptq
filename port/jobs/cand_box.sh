#!/bin/bash
# Replacement-candidate runs on one box.  usage: bash cand_box.sh SRC name:rot:int:bits[:seed] ...
# rot=1 rotates SRC (randomized Hadamard, seed) and checks equivalence on CPU first; int=1 constrains offsets to integer zero points.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; SRC=$1; shift; mkdir -p results models
while pgrep -f "[t]asks_box.sh|[n]b_box.sh" >/dev/null; do sleep 60; done
for spec in "$@"; do IFS=: read name rot intz bits seed <<< "$spec"; seed=${seed:-0}
  echo "=== $name rot=$rot int=$intz bits=$bits seed=$seed $(date -u +%T)"
  M=$SRC
  if [ "$rot" = 1 ]; then M=models/qwen15_rot_s$seed
    if [ ! -f $M/rotation.pt ]; then $PY rotate.py $SRC $M --seed $seed 2>&1 | tail -1; CUDA_VISIBLE_DEVICES= $PY test_rotation.py $SRC $M 2>&1 | grep "original ppl"; fi
  fi
  F=(); [ "$intz" = 1 ] && F=(--int-zero)
  $PY gptq_torch.py $M models/cand_$name $bits --mse --lloyd --refine 3 --refit 2 "${F[@]}" --dtype float16 2>&1 | grep -E "^layer 27|saved|Traceback|Error"
  $PY ppl_stream.py models/cand_$name --out results/ppl_stream_cand_$name.json 2>&1 | grep -E "perplexity|rror"
  $PY ../ppl_vllm.py models/cand_$name --text wikitext2_test.txt --out results/ppl_cand_$name.json 2>&1 | grep -E "perplexity|rror"
  $PY ../humaneval_vllm.py models/cand_$name --out results/humaneval_cand_$name.json 2>&1 | grep -E "pass@1|rror"
  if [ "$intz" = 1 ] && [ "$bits" = 4 ]; then
    $PY codes2awq.py models/cand_$name models/cand_${name}_awq 2>&1 | tail -1
    $PY ../ppl_vllm.py models/cand_${name}_awq --quant awq_marlin --text wikitext2_test.txt --out results/ppl_cand_${name}_awq.json 2>&1 | grep -E "perplexity|rror"
    $PY ../humaneval_vllm.py models/cand_${name}_awq --quant awq_marlin --out results/humaneval_cand_${name}_awq.json 2>&1 | grep -E "pass@1|rror"
    (cd ~/domq && bash bench_serve.sh cand_${name}_awq port/models/cand_${name}_awq awq_marlin 2>&1 | grep -E "throughput|TTFT|TPOT|failed")
  fi
done
echo "== CAND DONE $(date -u +%T)"
