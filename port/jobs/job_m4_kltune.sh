#!/bin/bash
# KL fine-tune of the ridged Q4_K 1.5B (codes fixed; fp16 super-scales + norms + biases trained vs the fp16 model). Baseline chat KL of the
# untuned model first; then three variants; KL on wiki / code / chat (dequantized dir, 8 windows, as the earlier rows) + wiki ppl + Q6_K-embedding pack.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
base=q15_kq4_mix_damp05
[ -f chat_test.txt ] || $PY make_calib_v2.py --tok models/qwen15_src 2>&1 | grep -vE "Warning|warn" | head -3
[ -f results/kld/${base}_chat.json ] || { echo "== baseline chat KL $base $(date -u +%T)"; $PY kld_stream.py models/qwen15_src --cand models/$base --text chat_test.txt --nwin 8 --out results/kld/${base}_chat.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; }
for job in "tune_a:--calib calib_mixed.txt --steps 300 --lr 1e-4 --lr-norm 1e-5 --accum 4" "tune_b:--calib calib_mixed.txt --steps 300 --lr 3e-4 --lr-norm 1e-5 --accum 4 --param row" "tune_c:--calib calib_mixed.txt --steps 300 --lr 3e-5 --lr-norm 1e-5 --accum 4" "tune_d:--calib calib_v2.txt --steps 300 --lr 1e-4 --lr-norm 1e-5 --accum 4"; do name=${base}_${job%%:*}; extra=${job#*:}
  echo "== kl_tune $name ($extra) $(date -u +%T)"; rm -rf models/$name
  $PY kl_tune.py models/qwen15_src models/$base models/$name $extra 2>&1 | grep -vE "Warning|warn|Token indices" | cut -c1-200
  [ -f models/$name/kl_tune.json ] || { echo "TUNE FAILED $name"; continue; }
  for text in wikitext2_test.txt code_test.txt chat_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
  $PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
done
f=models/gguf/${base}_tune_a_e6.gguf; $PY codes2gguf.py models/${base}_tune_a models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf $f --embed-kq q6_k 2>&1 | tail -1 | cut -c1-200
echo "== packed $(ls -l --block-size=M $f | awk '{print $5}')"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text wikitext2_test.txt --nwin 8 --out results/kld/${base}_tune_a_e6_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
echo "== KLTUNE M4 DONE $(date -u +%T)"
