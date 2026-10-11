#!/bin/bash
# The final 1.5B candidate: v4-calibrated ridged Q4_K (from m3) + KL fine-tune on the v4 text; KL on the 5 held-outs, packs Q8/Q6_K, the Q8 file on all 5 held-outs.
# packed wiki KL; bartowski Q4_K_M on the 4 other held-outs for the final table.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
base=q15_kq4_v4_ridge; name=${base}_tune; [ -f models/$base/model.safetensors.index.json ] || { echo "no $base on this box"; exit 1; }
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; [ -f results/kld/bartowski_q4_k_m_$tag.json ] && continue
  echo "== kld bartowski on $tag"; $PY kld_stream.py models/qwen15_src --cand-gguf models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --text $text --nwin 8 --out results/kld/bartowski_q4_k_m_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== kl_tune $name (calib_v3w) $(date -u +%T)"; rm -rf models/$name
$PY kl_tune.py models/qwen15_src models/$base models/$name --calib calib_v3w.txt --steps 300 --lr 1e-4 --lr-norm 1e-5 --accum 4 2>&1 | grep -vE "Warning|warn|Token indices" | cut -c1-200
[ -f models/$name/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
$PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
for spec in "q8:--embed-q8" "e6:--embed-kq q6_k"; do tag=${spec%%:*}; f=models/gguf/${name}_$tag.gguf; $PY codes2gguf.py models/$name models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf $f ${spec#*:} 2>&1 | tail -1 | cut -c1-160
  echo "== $f $(ls -l --block-size=M $f | awk '{print $5}')"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text wikitext2_test.txt --nwin 8 --out results/kld/${name}_${tag}_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== TUNEV4 M4 DONE $(date -u +%T)"
f=models/gguf/${name}_q8.gguf; for text in code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld ${name}_q8 on $tag"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text $text --nwin 8 --out results/kld/${name}_q8_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== TUNEV4 FILE M4 DONE $(date -u +%T)"
