#!/bin/bash
# Our recipe with UNSLOTH's per-tensor allocation (their UD-Q4_K_M types mapped to K-quants: IQ4_XS/IQ4_NL -> Q4_K, Q8_0 -> Q6_K, IQ3_S -> Q3_K), v4 calibration,
# ridge 0.05, codes only; pack with Q4_K embeddings + Q6_K head; KL wiki/code/multi. Tests whether their wiki advantage is the allocation. Waits for the head test.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
until grep -q "Q38 HEAD M1 DONE" results/job_m1_q38_head.out 2>/dev/null; do sleep 300; done
MAP=budgets/q38_unsloth_map.json; [ -f $MAP ] || { echo "no map"; exit 1; }
free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 45 ] || { echo "DISK: only ${free}G free"; exit 1; }
name=q38_kqdyn_umap; echo "== quantize $name (unsloth's allocation as K-quants, calib_v4, 256 windows, ridge 0.05, codes only) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map $MAP --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 256 --codes-only 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ "$(ls models/$name/codes 2>/dev/null | wc -l)" -ge 64 ] || { echo "QUANT FAILED"; exit 1; }
f=models/gguf/q38_dom_dyn_umap.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 UMAP M1 DONE $(date -u +%T)"
