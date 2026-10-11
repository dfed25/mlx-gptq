#!/bin/bash
# 27B with the v4 calibration (chat-template GSM8K/MMLU + UltraChat + 8-language Wikipedia + Python x2, 256 windows), same budget map and ridge as q38_kqdyn.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
[ -f calib_v4.txt ] || { $PY make_calib_v2.py --tok models/q38_27b_src 2>&1 | grep -vE "Warning|warn|Token indices" | head -2; $PY make_calib_v3.py --tok models/q38_27b_src 2>&1 | grep -vE "Warning|warn|Token indices" | head -2
  $PY - <<'PY'
s = open("calib_mixed.txt").read(); py = s[len(s)//2:]; assert py.count("def ") > 200, "second half of calib_mixed.txt is not the Python slice"
open("calib_v4.txt", "w").write(open("calib_v3.txt").read() + "\n" + py); print(f"calib_v4.txt: {len(open('calib_v4.txt').read().encode())/1e6:.2f} MB")
PY
}
free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 110 ] || { echo "DISK: only ${free}G free, need 110G"; exit 1; }
name=q38_kqdyn_v4; MAP=budgets/q38_kq456_4.783.json; echo "== quantize $name (calib_v4, 256 windows, refine-damp 0.05, map $MAP) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map $MAP --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 256 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ -f models/$name/model.safetensors.index.json ] || { echo "QUANT FAILED"; exit 1; }
$PY ppl_stream.py models/$name --dtype bfloat16 --out results/ppl_stream_${name}_wiki.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --dtype bfloat16 --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
f=models/gguf/q38_dom_dyn_v4.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
for text in chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; for pair in q38_kqdyn:models/gguf/q38_dom_dyn.gguf unsloth:models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf; do n=${pair%%:*}; g=${pair#*:}
  [ -f results/kld/${n}_$tag.json ] && continue; echo "== kld $n on $tag $(date -u +%T)"; $PY kld_stream.py models/q38_27b_src --cand-gguf $g --text $text --nwin 6 --dtype bfloat16 --out results/kld/${n}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done; done
echo "== Q38 V4 DONE $(date -u +%T)"
