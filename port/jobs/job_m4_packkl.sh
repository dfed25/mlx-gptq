#!/bin/bash
# Attribute the packed-vs-dequantized KL gap of the fine-tuned 1.5B (dequantized dir 0.0233, Q6_K-embedding file 0.0267): the same codes packed with
# F16 and Q8_0 embeddings/head, the untuned ridged model packed with Q6_K, and bartowski's Q4_K_M, all on the same evaluator (wiki, 8 windows).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
while ! grep -q "KLTUNE2 M4 DONE" results/job_m4_kltune2.out 2>/dev/null; do sleep 120; done
REF=models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
for spec in "q15_kq4_mix_damp05_tune_a:f16:" "q15_kq4_mix_damp05_tune_a:q8:--embed-q8" "q15_kq4_mix_damp05:e6:--embed-kq q6_k"; do m=${spec%%:*}; r=${spec#*:}; tag=${r%%:*}; flags=${r#*:}
  f=models/gguf/${m}_${tag}.gguf; [ -f $f ] || $PY codes2gguf.py models/$m $REF $f $flags 2>&1 | tail -1 | cut -c1-160
  echo "== $f $(ls -l --block-size=M $f | awk '{print $5}')"; $PY kld_stream.py models/qwen15_src --cand-gguf $f --text wikitext2_test.txt --nwin 8 --out results/kld/${m}_${tag}_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
done
echo "== bartowski Q4_K_M"; $PY kld_stream.py models/qwen15_src --cand-gguf $REF --text wikitext2_test.txt --nwin 8 --out results/kld/bartowski_q4_k_m_wikitext2.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
echo "== PACKKL M4 DONE $(date -u +%T)"
