#!/bin/bash
# After the download and the regression finish: quantize g32 uniform 4-bit on m1, pack to GGUF Q4_1, round-trip check,
# then wiki+code ppl of the packed file and of bartowski's Q4_K_M / Q4_K_S / Q5_K_M through the same streamed protocol.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python
while ! grep -q "DL15 DONE" results/dl_gguf15.out 2>/dev/null || pgrep -f "^bash job_m1_regress.sh" >/dev/null; do sleep 60; done
echo "== prerequisites ready $(date -u +%T)"
REF=models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
if [ ! -d models/q15_g32_u4/codes ]; then echo "== quantize g32 u4 $(date -u +%T)"
  $PY gptq_torch.py models/qwen15_src models/q15_g32_u4 4 --group 32 --mse --lloyd --refine 3 --refit 2 --int-zero --dtype float16 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error"
fi
echo "== pack Q4_1 $(date -u +%T)"
$PY codes2gguf.py models/q15_g32_u4 $REF models/gguf/q15_dom_q4_1.gguf 2>&1 | tail -15
echo "== round trip $(date -u +%T)"
$PY gguf_rt_check.py models/gguf/q15_dom_q4_1.gguf models/q15_g32_u4 2>&1 | tail -8
for g in q15_dom_q4_1 Qwen2.5-1.5B-Instruct-Q4_K_M Qwen2.5-1.5B-Instruct-Q4_K_S Qwen2.5-1.5B-Instruct-Q5_K_M; do
  f=models/gguf/$g.gguf; [ -f $f ] || continue; echo "== ppl $g $(du -m $f | cut -f1) MB $(date -u +%T)"
  $PY ppl_stream.py models/qwen15_src --gguf $f --out results/ppl_gguf_$g.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py models/qwen15_src --gguf $f --text code_test.txt --out results/ppl_gguf_${g}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
done
echo "== GGUF15 DONE $(date -u +%T)"
