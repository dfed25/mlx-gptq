#!/bin/bash
# After the Adam variants: the same fine-tune with the user's Packed BlockMax (fused Triton kernel) in place of Adam, per-super-block and per-row multipliers.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
while ! grep -q "KLTUNE M4 DONE" results/job_m4_kltune.out 2>/dev/null; do sleep 120; done
base=q15_kq4_mix_damp05
for job in "tune_e:--calib calib_mixed.txt --steps 300 --lr 1e-4 --lr-norm 1e-5 --accum 4 --opt blockmax" "tune_f:--calib calib_mixed.txt --steps 300 --lr 3e-4 --lr-norm 1e-5 --accum 4 --param row --opt blockmax"; do name=${base}_${job%%:*}; extra=${job#*:}
  echo "== kl_tune $name ($extra) $(date -u +%T)"; rm -rf models/$name
  $PY kl_tune.py models/qwen15_src models/$base models/$name $extra 2>&1 | grep -vE "Warning|warn|Token indices" | cut -c1-200
  [ -f models/$name/kl_tune.json ] || { echo "TUNE FAILED $name"; continue; }
  for text in wikitext2_test.txt code_test.txt chat_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
  $PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
done
echo "== KLTUNE2 M4 DONE $(date -u +%T)"
