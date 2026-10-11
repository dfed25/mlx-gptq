#!/bin/bash
# Qwen3.8-27B with a MIXED calibration set (about half WikiText-2 train, half Python source from the vllm package, which is
# disjoint from the transformers files used as the code test corpus): 4 bits with integer zero points, then 2 bits with
# Dom's fp16 offsets (no 2-bit kernel exists, so that one is quality-only). Each model: streamed ppl on WikiText-2 and on
# the code corpus; the 4-bit also packed to AWQ for kernel ppl and HumanEval.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; SRC=models/q38_27b_src
V=~/venv-vllm/lib64/python3.11/site-packages/vllm
{ head -c 1500000 wikitext2_train.txt; echo; find $V/model_executor/layers $V/v1/core $V/engine -name "*.py" -size +8k | sort | head -60 | xargs cat | head -c 1500000; } > calib_mixed.txt
echo "calib_mixed.txt: $(wc -c < calib_mixed.txt) bytes (test corpora: wikitext2_test.txt, code_test.txt from transformers)"
ev() { # $1 model dir, $2 name, $3 extra ppl_stream args
  $PY ppl_stream.py $1 $3 --dtype bfloat16 --bs 2 --out results/ppl_stream_$2_wiki.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py $1 $3 --dtype bfloat16 --bs 2 --text code_test.txt --out results/ppl_stream_$2_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
}
echo "== 4-bit int-zero, mixed calibration $(date -u +%T)"; rm -rf models/q38_27b_int4_mix
$PY gptq_torch.py $SRC models/q38_27b_int4_mix 4 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype bfloat16 --calib calib_mixed.txt 2>&1 | grep -E "^layer (0|31|63) |saved|Traceback|Error"
ev models/q38_27b_int4_mix q38_int4_mix ""
echo "== awq packing $(date -u +%T)"; $PY codes2awq.py models/q38_27b_int4_mix models/q38_27b_int4_mix_awq 2>&1 | tail -1
cd ~/domq; M=port/models/q38_27b_int4_mix_awq
$PY ppl_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 1 --max-batched 512 --text wikitext2_test.txt --out results/ppl_q38_int4_mix_awq.json > results/log_ppl_q38mix.log 2>&1; grep -E "perplexity" results/log_ppl_q38mix.log | tail -1 | sed "s/^/  kernel: /"
$PY humaneval_vllm.py $M --quant awq_marlin --dtype bfloat16 --gpu-util 0.97 --eager --max-num-seqs 6 --max-batched 1024 --max-len 1024 --out results/humaneval_q38_int4_mix_awq.json > results/log_he_q38mix.log 2>&1; grep -E "pass@1" results/log_he_q38mix.log | tail -1 | sed "s/^/  kernel: /"
cd ~/domq/port
echo "== 2-bit, Dom's fp16 offsets, mixed calibration $(date -u +%T)"; rm -rf models/q38_27b_int2_mix
$PY gptq_torch.py $SRC models/q38_27b_int2_mix 2 --mse --lloyd --refine 3 --refit 2 --dtype bfloat16 --calib calib_mixed.txt 2>&1 | grep -E "^layer (0|31|63) |saved|Traceback|Error"
ev models/q38_27b_int2_mix q38_int2_mix ""
echo "== Q38 MIX DONE $(date -u +%T)"
