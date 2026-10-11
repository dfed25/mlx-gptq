#!/bin/bash
# Unsloth's calibration statistics on our recipe: (1) compare their imatrix diagonal with ours (calib_v4) on 5 layers; (2) quantize with our Hessians column-rescaled
# to their diagonal (--imatrix), same map + ridge, codes only (no 54 GB shards); pack; KL on wiki / code / multi (6 windows).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
IM=models/gguf/imatrix_unsloth.gguf.diag.pt; $PY imatrix_tools.py list models/gguf/imatrix_unsloth.gguf models/q38_27b_src | tail -1
echo "== compare imatrix diagonal vs ours (calib_v4, 256 windows) $(date -u +%T)"
$PY imatrix_tools.py compare models/gguf/imatrix_unsloth.gguf models/q38_27b_src --calib calib_v4.txt --nsamples 256 --layers 0,8,47 2>&1 | grep -vE "Warning|warn|Token indices|falling back" | tee results/imatrix_compare.txt | cut -c1-200
name=q38_kqdyn_imat; echo "== quantize $name (calib_v4 correlations, unsloth imatrix diagonal, map + ridge 0.05, codes only) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map budgets/q38_kq456_4.783.json --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 256 --imatrix $IM --codes-only 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ "$(ls models/$name/codes 2>/dev/null | wc -l)" -ge 64 ] || { echo "QUANT FAILED"; exit 1; }
f=models/gguf/q38_dom_dyn_imat.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
echo "== Q38 IMAT M1 DONE $(date -u +%T)"
