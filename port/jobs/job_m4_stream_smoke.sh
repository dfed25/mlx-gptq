#!/bin/bash
# Validate the layer-streamed fine-tuner on the 1.5B (m4): pack the untuned v4 model with Q8_0 embeddings, 3-step smoke, then a real run
# (200 steps, 8 windows per step) to compare with kl_tune.py's result on the same codes (dequantized wiki KL 0.0227, Q8 file 0.0229).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
REF=models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf; base=q15_kq4_v4_ridge; g=models/gguf/${base}_q8.gguf
[ -f $g ] || $PY codes2gguf.py models/$base $REF $g --embed-q8 2>&1 | tail -1 | cut -c1-160
echo "== smoke $(date -u +%T)"; rm -rf models/${base}_stream_smoke
$PY kl_tune_stream.py models/qwen15_src models/$base models/${base}_stream_smoke --gguf $g --calib calib_v3w.txt --nwin 16 --bs 4 --steps 3 --warmup 1 --dtype float16 2>&1 | grep -vE "Warning|warn|Token indices|Loading weights" | cut -c1-200
[ -f models/${base}_stream_smoke/kl_tune.json ] || { echo "SMOKE FAILED"; exit 1; }
f=models/gguf/${base}_stream_smoke_q8.gguf; $PY codes2gguf.py models/${base}_stream_smoke $REF $f --embed-q8 2>&1 | tail -1 | cut -c1-160
$PY kld_stream.py models/qwen15_src --cand-gguf $f --text wikitext2_test.txt --nwin 8 --out results/kld/${base}_stream_smoke_q8_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
echo "== full run $(date -u +%T)"; name=${base}_stream; rm -rf models/$name
$PY kl_tune_stream.py models/qwen15_src models/$base models/$name --gguf $g --calib calib_v3w.txt --nwin 256 --bs 8 --steps 200 --lr 1e-4 --lr-norm 1e-5 --dtype float16 2>&1 | grep -vE "Warning|warn|Token indices|Loading weights" | grep -E "step +[0-9]*0 |teacher|student|training|exported|Traceback|Error" | cut -c1-200
[ -f models/$name/kl_tune.json ] || { echo "FULL FAILED"; exit 1; }
f=models/gguf/${name}_q8.gguf; $PY codes2gguf.py models/$name $REF $f --embed-q8 2>&1 | tail -1 | cut -c1-160
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld ${name}_q8 on $tag"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text $text --nwin 8 --out results/kld/${name}_q8_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== STREAM M4 DONE $(date -u +%T)"
