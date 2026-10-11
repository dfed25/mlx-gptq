#!/bin/bash
# YAQA test on Qwen2.5-1.5B: token weights from the end-to-end KL gradient (sampled Fisher) on the v4 calibration windows, then Dom's ridged Q4_K recipe with the
# weighted Hessians; KL ×5 (dequantized, 8 windows) vs the unweighted twin q15_kq4_v4_ridge; then the KL fine-tune on top and KL ×5 again; pack Q8 + file wiki KL.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
TWF=results/yaqa_q15_v4.pt; [ -f $TWF ] || { echo "== yaqa weights $(date -u +%T)"; $PY yaqa_weights.py models/qwen15_src $TWF --calib calib_v3w.txt --nsamples 256 --seqlen 512 --seed 0 --bs 4 2>&1 | grep -vE "Warning|warn|Token indices|Loading" | tail -2 | cut -c1-200; }
[ -f $TWF ] || { echo "WEIGHTS FAILED"; exit 1; }
name=q15_kq4_v4_yaqa; echo "== quantize $name (v4 text, 256 windows, ridge 0.05, YAQA token weights) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/qwen15_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq q4_k --dtype float16 --calib calib_v3w.txt --nsamples 256 --token-weights $TWF 2>&1 | grep -E "^layer (0|27) |token weights|saved|Traceback|Error" | cut -c1-160
[ -f models/$name/model.safetensors.index.json ] || { echo "QUANT FAILED"; exit 1; }
$PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
tn=${name}_tune; echo "== kl_tune $tn $(date -u +%T)"; rm -rf models/$tn
$PY kl_tune.py models/qwen15_src models/$name models/$tn --calib calib_v3w.txt --steps 300 --lr 1e-4 --lr-norm 1e-5 --accum 4 2>&1 | grep -vE "Warning|warn|Token indices|Loading weights" | grep -E "held-out|kept|exported|Traceback|Error" | tail -3 | cut -c1-160
[ -f models/$tn/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $tn on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$tn --text $text --nwin 8 --out results/kld/${tn}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
$PY ppl_stream.py models/$tn --out results/ppl_stream_$tn.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"; $PY ppl_stream.py models/$tn --text code_test.txt --out results/ppl_stream_${tn}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
f=models/gguf/${tn}_q8.gguf; $PY codes2gguf.py models/$tn models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf $f --embed-q8 2>&1 | tail -1 | cut -c1-160; echo "== $f $(ls -l --block-size=M $f | awk '{print $5}')"
$PY kld_stream.py models/qwen15_src --cand-gguf $f --text wikitext2_test.txt --nwin 8 --out results/kld/${tn}_q8_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
echo "== YAQA15 M4 DONE $(date -u +%T)"
