#!/bin/bash
# The m3 combo run's fine-tune stage, moved to m7 (L40S): codes + untuned GGUF + calibration mix copied from m3 (pipeline.sh q38_combo: our budget,
# 1024 windows, Q4_K embeddings, Q5_K RTN head). pack_codes, kl_tune_stream 100 x 16 over the 1024-window pool (pipeline settings), pack tuned
# with the same flags, ppl + KL on the 5 held-outs. Compare with the untuned file (m3 scores it) and unsloth (0.0080 / 0.0186 / 0.0078).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR
W=models/q38_combo; Q=$W/quant; T=$W/tuned; A=$W/q38_combo_untuned.gguf; B=$W/q38_combo_tuned.gguf; REF=models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf; CAL=$W/calib.txt
while [ ! -f results/combo_codes_copied ]; do sleep 60; done; echo "== codes present $(date -u +%T)"
[ "$(ls $Q/codes | wc -l)" -ge 64 ] || { echo "codes incomplete"; exit 1; }; [ -f $A ] || { echo "no $A"; exit 1; }
[ -f $Q/packed/layer063.safetensors ] || { echo "== pack codes $(date -u +%T)"; $PY pack_codes.py $Q/codes $Q/packed 2>&1 | tail -1; }
if [ ! -f $T/kl_tune.json ]; then
  echo "== kl_tune_stream combo (pipeline calib, 1024-window pool, bs 16, 100 steps, packed codes) $(date -u +%T)"; rm -f $T/codes/zz_tuned_* 2>/dev/null
  $PY -u kl_tune_stream.py models/q38_27b_src $Q $T --gguf $A --packed $Q/packed --calib $CAL --nwin 1024 --bs 16 --steps 100 --lr 1e-4 --lr-norm 1e-5 --dtype bfloat16 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights|falling back" | tee results/q38_combo_tune_m7.log | grep --line-buffered -E "step +[0-9]*[05] |teacher|student|training|embedding|export|held|Traceback|Error" | cut -c1-200
  [ -f $T/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
fi
echo "== pack tuned $(date -u +%T)"; $PY codes2gguf.py $T $REF $B --embed-kq q4_k --head-kq q5_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $B ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $B | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $B --out results/ppl_gguf_q38_combo_tuned.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt multi_test.txt chat_test.txt ultrachat_test.txt; do tag=${text%%_*}; echo "== kld combo_tuned on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $B --text $text --nwin 6 --dtype bfloat16 --out results/kld/q38_combo_tuned_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 COMBO TUNE M7 DONE $(date -u +%T)"
