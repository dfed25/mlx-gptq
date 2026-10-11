#!/bin/bash
# 27B KL fine-tune on m1 (layer-streamed, packed codes served from the page cache): the v4-calibrated budget-map model, fine-tuned on the v4 text; export, pack, KL on the 5 held-outs.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
while ! grep -q "Q38 V4 DONE" results/job_m1_q38v4.out 2>/dev/null; do sleep 120; done
base=q38_kqdyn_v4; name=${base}_tune; g=models/gguf/q38_dom_dyn_v4.gguf
[ "$(ls models/$base/codes | wc -l)" -ge 64 ] || { echo "codes incomplete"; exit 1; }; [ -f $g ] || { echo "no $g"; exit 1; }
[ -f models/$base/packed/layer063.safetensors ] || { echo "== pack codes $(date -u +%T)"; $PY pack_codes.py models/$base/codes models/$base/packed 2>&1 | tail -1; }
echo "== kl_tune_stream $name (calib_v4, 256 windows, bs 16, 100 steps, packed codes) $(date -u +%T)"; rm -f models/$name/codes/zz_tuned_* 2>/dev/null
$PY -u kl_tune_stream.py models/q38_27b_src models/$base models/$name --gguf $g --packed models/$base/packed --calib calib_v4.txt --nwin 256 --bs 16 --steps 100 --warmup 10 --lr 1e-4 --lr-norm 1e-5 --dtype bfloat16 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights" | tee results/${name}_tune.log | grep --line-buffered -E "step +[0-9]*[05] |teacher|student|training|embeddings|exported|Traceback|Error" | cut -c1-200
[ -f models/$name/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
f=models/gguf/q38_dom_dyn_v4_tune.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name $g $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $f --out results/ppl_gguf_${name}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 V4TUNE M1 DONE $(date -u +%T)"
