#!/bin/bash
# v4 = the v3 mix with the Python slice weighted twice AND 256 calibration windows (both helped separately), ridged Q4_K; ppl + KL on the 5 held-outs; then copy to m4 for the fine-tune.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
while ! grep -q "CALIB3B M3 DONE" results/job_m3_calib3b.out 2>/dev/null; do sleep 120; done
name=q15_kq4_v4_ridge; echo "== quantize $name (calib_v3w, 256 windows) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/qwen15_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq q4_k --dtype float16 --calib calib_v3w.txt --nsamples 256 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error" | cut -c1-160
$PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== V4 M3 DONE $(date -u +%T)"
