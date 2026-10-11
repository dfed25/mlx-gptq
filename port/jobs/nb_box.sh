#!/bin/bash
# Refit-metric variants on one box.  usage: bash nb_box.sh SRC name:metric:block:refit ...   (refit=0 -> refinement only)
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0; PY=~/venv-vllm/bin/python; mkdir -p results models; SRC=$1; shift
for spec in "$@"; do IFS=: read name metric block refit <<< "$spec"
  echo "=== $name metric=$metric block=$block refit=$refit $(date -u +%T)"
  R=(); [ "$refit" = "0" ] || R=(--refit 2 --refit-metric $metric --refit-block $block)
  $PY gptq_torch.py $SRC models/nb_$name 3 --mse --lloyd --refine 3 "${R[@]}" --dtype float16 2>&1 | grep -E "^layer 27|saved|Traceback|Error"
  $PY ppl_stream.py models/nb_$name --out results/ppl_stream_nb_$name.json 2>&1 | grep -E "perplexity|rror"
  $PY ../ppl_vllm.py models/nb_$name --text wikitext2_test.txt --out results/ppl_nb_$name.json 2>&1 | grep -E "perplexity|rror"
  $PY ../humaneval_vllm.py models/nb_$name --out results/humaneval_nb_$name.json 2>&1 | grep -E "pass@1|rror"
done
echo "== NB BOX DONE $(date -u +%T)"
