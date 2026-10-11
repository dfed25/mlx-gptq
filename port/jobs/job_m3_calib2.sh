#!/bin/bash
# Calibration mix v2 (v1 + chat-template-formatted GSM8K / MMLU) on the 1.5B under the Q4_K constraint, ridged refinement: ppl + KL on wiki / code / chat
# against the same recipe on the v1 mix (q15_kq4_mix_damp05 on m4: wiki KL 0.0261 / 92.24%, code 0.0354 / 94.34%).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
$PY make_calib_v2.py --tok models/qwen15_src 2>&1 | grep -vE "Warning|warn" | head -3
for job in "q15_kq4_v2_ridge:--calib calib_v2.txt" "q15_kq4_v1_ridge_m3:--calib calib_mixed.txt"; do name=${job%%:*}; extra=${job#*:}
  echo "== quantize $name ($extra) $(date -u +%T)"; rm -rf models/$name
  $PY gptq_torch.py models/qwen15_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq q4_k --dtype float16 $extra 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error" | cut -c1-160
  $PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
  for text in wikitext2_test.txt code_test.txt chat_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
done
f=models/gguf/q15_kq4_v2_ridge_e6.gguf; $PY codes2gguf.py models/q15_kq4_v2_ridge models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf $f --embed-kq q6_k 2>&1 | tail -1 | cut -c1-200
echo "== packed $(ls -l --block-size=M $f | awk '{print $5}')"
echo "== CALIB2 M3 DONE $(date -u +%T)"
