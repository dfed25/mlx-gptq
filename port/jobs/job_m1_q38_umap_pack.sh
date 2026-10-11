#!/bin/bash
# Resume of job_m1_q38_umap.sh from the pack, in case that job runs out of disk (36 GB free at layer 21, codes grow ~0.5 GB/layer, pack needs 16 GB).
# Waits for the main job to exit; if it did not finish, waits for >= 20 GB free (the user clears space), then packs and scores.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
name=q38_kqdyn_umap; f=models/gguf/q38_dom_dyn_umap.gguf
while pgrep -f "^bash job_m1_q38_umap.sh" >/dev/null; do sleep 300; done
grep -q "Q38 UMAP M1 DONE" results/job_m1_q38_umap.out 2>/dev/null && { echo "main job finished on its own"; exit 0; }
[ "$(ls models/$name/codes 2>/dev/null | wc -l)" -ge 64 ] || { echo "codes incomplete: the quantization itself failed"; exit 1; }
while [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -lt 20 ]; do echo "waiting for >= 20G free disk before the pack ($(date -u +%T))"; sleep 1800; done
rm -f $f; echo "== pack (resume) $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 UMAP M1 DONE $(date -u +%T)"
