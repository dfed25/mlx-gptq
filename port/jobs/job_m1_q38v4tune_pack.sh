#!/bin/bash
# Resume of job_m1_q38v4tune.sh from the pack step (the first pack ran out of disk).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld; base=q38_kqdyn_v4; name=${base}_tune; g=models/gguf/q38_dom_dyn_v4.gguf
free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 20 ] || { echo "DISK: only ${free}G free"; exit 1; }
f=models/gguf/q38_dom_dyn_v4_tune.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name $g $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $f --out results/ppl_gguf_${name}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 V4TUNE M1 DONE $(date -u +%T)"
