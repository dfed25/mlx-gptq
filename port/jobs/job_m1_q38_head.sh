#!/bin/bash
# GPTQ'd output head on the 27B (v4 + fine-tune codes): Dom's recipe on lm_head under Q6_K / Q5_K / Q4_K with the head Hessian from the quantized model's
# final-norm outputs (calib_v4, 256 windows), vs round-to-nearest at the same kinds. Variants are packed one at a time from models/q38_kqdyn_v4_tune,
# scored by KL on wiki / code / multi (6 windows) and removed. Baseline = q38_dom_dyn_v4_tune.gguf (RTN Q6_K head): wiki 0.0122 / code 0.0173 / multi 0.0114.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
until grep -q "Q38 IMAT M1 DONE" results/job_m1_q38_imat.out 2>/dev/null; do sleep 300; done
echo "== head GPTQ (Hessian from the v4+tune GGUF, 256 windows) $(date -u +%T)"
$PY head_gptq.py models/q38_27b_src results/q38_head --gguf models/gguf/q38_dom_dyn_v4_tune.gguf --calib calib_v4.txt --nsamples 256 --seqlen 512 --bs 4 --dtype bfloat16 --kinds q6_k,q5_k,q4_k --refine 3 --refit 2 --refine-damp 0.05 2>&1 | grep -vE "Warning|warn|Token indices|Loading|rows [0-9]" | cut -c1-200
for v in gptq_q6_k gptq_q5_k gptq_q4_k rtn_q5_k rtn_q4_k; do
  kind=${v#*_}; mode=${v%%_*}; f=models/gguf/q38_dom_dyn_v4_tune_head_$v.gguf; extra=""
  if [ $mode = gptq ]; then [ -f results/q38_head_$kind.safetensors ] || { echo "no head codes for $kind"; continue; }; extra="--head-codes results/q38_head_$kind.safetensors"; fi
  echo "== pack $v $(date -u +%T)"; $PY codes2gguf.py models/q38_kqdyn_v4_tune models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq $kind $extra 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
  [ -f $f ] || { echo "PACK FAILED $v"; continue; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
  for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $v on $tag $(date -u +%T)"
    $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/q38_v4tune_head_${v}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
  rm -f $f
done
echo "== Q38 HEAD M1 DONE $(date -u +%T)"
