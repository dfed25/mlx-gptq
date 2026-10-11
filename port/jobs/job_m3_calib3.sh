#!/bin/bash
# After the v2 job: calibration mix v3 (v2 + UltraChat + multilingual Wikipedia) on the 1.5B, ridged Q4_K; KL on wiki / code / chat / ultrachat / multilingual
# for the v3 model, and the two new held-outs for the v1 and v2 models from the previous job.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
while ! grep -q "CALIB2 M3 DONE" results/job_m3_calib2.out 2>/dev/null; do sleep 120; done
$PY make_calib_v3.py --tok models/qwen15_src 2>&1 | grep -vE "Warning|warn|Token indices" | tail -2
echo "== fp16 reference ppl on the new held-outs"; for text in ultrachat_test.txt multi_test.txt; do $PY ppl_stream.py models/qwen15_src --text $text --nwin 8 --out results/ppl_stream_q15_fp16_${text%%_*}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  $text: /"; done
for name in q15_kq4_v1_ridge_m3 q15_kq4_v2_ridge; do for text in ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done; done
name=q15_kq4_v3_ridge; echo "== quantize $name (calib_v3) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/qwen15_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq q4_k --dtype float16 --calib calib_v3.txt 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error" | cut -c1-160
$PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
f=models/gguf/${name}_e6.gguf; $PY codes2gguf.py models/$name models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf $f --embed-kq q6_k 2>&1 | tail -1 | cut -c1-200; echo "== packed $(ls -l --block-size=M $f | awk '{print $5}')"
echo "== CALIB3 M3 DONE $(date -u +%T)"
