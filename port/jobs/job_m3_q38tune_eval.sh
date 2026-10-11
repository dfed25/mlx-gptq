#!/bin/bash
# Pack (streamed through a temp file: gguf-py buffers the whole model otherwise, too much for 15 GB) + evaluation of the narrow file's fine-tune on m3.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR; base=q38_kqdyn; name=${base}_tune; g=models/gguf/q38_dom_dyn.gguf
f=models/gguf/q38_dom_dyn_tune.gguf; [ -f $f ] || { echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name $g $f --embed-kq q4_k --head-kq q6_k --temp-file 2>&1 | grep -vE "Token indices|Warning" | tail -2 | cut -c1-200; }
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $f --out results/ppl_gguf_${name}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 TUNE M3 DONE $(date -u +%T)"
