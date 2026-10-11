#!/bin/bash
# The combination on unsloth's allocation (m1): our recipe with unsloth's per-tensor map (budgets/q38_unsloth_map.json) + 1,024 calibration windows
# (calib_v4) + ridge 0.05, codes only; pack (Q4_K embeddings, Q6_K head, same as q38_dom_dyn_umap.gguf); KL wiki/code/multi on the pre-tune file;
# then the streamed KL fine-tune (100 x 16, packed codes), pack, ppl + KL on the 5 held-outs. Compare with q38_dom_dyn_umap.gguf
# (0.0092 / 0.0147 / 0.0086) and unsloth UD-Q4_K_M (0.0080 / 0.0186 / 0.0078).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR
MAP=budgets/q38_unsloth_map.json; REF=models/gguf/q38_dom_dyn_umap.gguf; [ -f $MAP ] || { echo "no map"; exit 1; }; [ -f $REF ] || { echo "no reference gguf"; exit 1; }
free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 90 ] || { echo "DISK: only ${free}G free"; exit 1; }
name=q38_kqdyn_umapn1k; f=models/gguf/q38_dom_dyn_umapn1k.gguf
if [ "$(ls models/$name/codes 2>/dev/null | wc -l)" -lt 64 ]; then
  echo "== quantize $name (unsloth's allocation, calib_v4, 1024 windows, ridge 0.05, codes only) $(date -u +%T)"; rm -rf models/$name
  $PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map $MAP --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 1024 --codes-only 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
  [ "$(ls models/$name/codes 2>/dev/null | wc -l)" -ge 64 ] || { echo "QUANT FAILED"; exit 1; }
fi
if [ ! -f $f ]; then
  echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name $REF $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
  [ -f $f ] || { echo "PACK FAILED"; exit 1; }
fi
echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}; [ -f results/kld/${name}_$tag.json ] && continue; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 UMAPN1K PRE-TUNE DONE $(date -u +%T)"
# ---- fine-tune
base=$name; tn=${base}_tune; ft=models/gguf/q38_dom_dyn_umapn1k_tune.gguf
[ -f models/$base/packed/layer063.safetensors ] || { echo "== pack codes $(date -u +%T)"; $PY pack_codes.py models/$base/codes models/$base/packed 2>&1 | tail -1; }
if [ ! -f models/$tn/kl_tune.json ]; then
  echo "== kl_tune_stream $tn (calib_v4, 256 windows, bs 16, 100 steps, packed codes) $(date -u +%T)"; rm -f models/$tn/codes/zz_tuned_* 2>/dev/null
  $PY -u kl_tune_stream.py models/q38_27b_src models/$base models/$tn --gguf $f --packed models/$base/packed --calib calib_v4.txt --nwin 256 --bs 16 --steps 100 --warmup 10 --lr 1e-4 --lr-norm 1e-5 --dtype bfloat16 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights" | tee results/${tn}_tune.log | grep --line-buffered -E "step +[0-9]*[05] |teacher|student|training|embedding|export|held|Traceback|Error" | cut -c1-200
  [ -f models/$tn/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
fi
echo "== pack tuned $(date -u +%T)"; $PY codes2gguf.py models/$tn $f $ft --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $ft ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $ft | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $ft --out results/ppl_gguf_${tn}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt multi_test.txt chat_test.txt ultrachat_test.txt; do tag=${text%%_*}; echo "== kld $tn on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $ft --text $text --nwin 6 --dtype bfloat16 --out results/kld/${tn}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 UMAPN1K M1 DONE $(date -u +%T)"
