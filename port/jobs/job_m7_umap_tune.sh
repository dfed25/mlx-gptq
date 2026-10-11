#!/bin/bash
# KL fine-tune of the allocation file on m7 (L40S): the existing q38_kqdyn_umap codes (unsloth's per-tensor map, v4 text, 256 windows) copied from m1;
# pack_codes, streamed fine-tune (same settings as the v4 tune: 100 steps x 16, lr 1e-4), pack, ppl + KL on the 5 held-outs.
# Compare with q38_dom_dyn_umap.gguf (wiki 0.0092 / 96.07%, code 0.0147 / 98.35%, multi 0.0086 / 95.91%) and unsloth (0.0080 / 0.0186 / 0.0078).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR
base=q38_kqdyn_umap; tn=${base}_tune; g=models/gguf/q38_dom_dyn_umap.gguf; ft=models/gguf/q38_dom_dyn_umap_tune.gguf
while [ ! -f results/umap_codes_copied ]; do sleep 60; done; echo "== codes present $(date -u +%T)"
[ "$(ls models/$base/codes | wc -l)" -ge 64 ] || { echo "codes incomplete"; exit 1; }; [ -f $g ] || { echo "no $g"; exit 1; }
[ -f models/$base/packed/layer063.safetensors ] || { echo "== pack codes $(date -u +%T)"; $PY pack_codes.py models/$base/codes models/$base/packed 2>&1 | tail -1; }
if [ ! -f models/$tn/kl_tune.json ]; then
  echo "== kl_tune_stream $tn (calib_v4, 256 windows, bs 16, 100 steps, packed codes) $(date -u +%T)"; rm -f models/$tn/codes/zz_tuned_* 2>/dev/null
  $PY -u kl_tune_stream.py models/q38_27b_src models/$base models/$tn --gguf $g --packed models/$base/packed --calib calib_v4.txt --nwin 256 --bs 16 --steps 100 --warmup 10 --lr 1e-4 --lr-norm 1e-5 --dtype bfloat16 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights|falling back" | tee results/${tn}_tune.log | grep --line-buffered -E "step +[0-9]*[05] |teacher|student|training|embedding|export|held|Traceback|Error" | cut -c1-200
  [ -f models/$tn/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
fi
echo "== pack tuned $(date -u +%T)"; $PY codes2gguf.py models/$tn $g $ft --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $ft ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $ft | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $ft --out results/ppl_gguf_${tn}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt multi_test.txt chat_test.txt ultrachat_test.txt; do tag=${text%%_*}; echo "== kld $tn on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $ft --text $text --nwin 6 --dtype bfloat16 --out results/kld/${tn}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 UMAP TUNE M7 DONE $(date -u +%T)"
