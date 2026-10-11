#!/bin/bash
# v3 follow-ups: (a) 256 calibration windows instead of 128; (b) the Python slice of the base mix weighted twice (calib_v3w). Ridged Q4_K, KL on the 5 held-outs.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
$PY - <<'PY'
s = open("calib_mixed.txt").read(); py = s[len(s)//2:]; assert py.count("def ") > 200 and py.count("import ") > 100, "second half of calib_mixed.txt is not the Python slice"
open("calib_v3w.txt", "w").write(open("calib_v3.txt").read() + "\n" + py); print(f"calib_v3w.txt: {len(open('calib_v3w.txt').read().encode())/1e6:.2f} MB (Python slice {len(py.encode())/1e6:.2f} MB repeated)")
PY
for job in "q15_kq4_v3_n256:--calib calib_v3.txt --nsamples 256" "q15_kq4_v3w_ridge:--calib calib_v3w.txt"; do name=${job%%:*}; extra=${job#*:}
  echo "== quantize $name ($extra) $(date -u +%T)"; rm -rf models/$name
  $PY gptq_torch.py models/qwen15_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq q4_k --dtype float16 $extra 2>&1 | grep -E "^layer (0|27) |saved|Traceback|Error" | cut -c1-160
  $PY ppl_stream.py models/$name --out results/ppl_stream_$name.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
  $PY ppl_stream.py models/$name --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
  for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag"; $PY kld_stream.py models/qwen15_src --cand models/$name --text $text --nwin 8 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220; done
done
echo "== CALIB3B M3 DONE $(date -u +%T)"
