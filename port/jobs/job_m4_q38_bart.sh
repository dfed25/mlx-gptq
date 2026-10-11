#!/bin/bash
# Control: our recipe on bartowski's public imatrix calibration text (calibration_datav3.txt, ~70k tokens, overlapping windows), same map + ridge; codes only; pack; KL wiki/code/multi in 2-window pieces (15 GB box).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR budgets
while ! grep -q "SRC COPY DONE" results/copy_src_from_m1.out 2>/dev/null; do sleep 60; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 300; done        # someone else's jobs are on this GPU; wait for it to free up (never kill them)
echo "== GPU free $(date -u +%T)"
name=q38_kqdyn_bart; echo "== quantize $name (calibration_datav3, 128 windows, map + ridge 0.05, codes only) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map budgets/q38_kq456_4.783.json --dtype bfloat16 --bs 4 --calib calibration_datav3.txt --nsamples 128 --codes-only 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ "$(ls models/$name/codes 2>/dev/null | wc -l)" -ge 64 ] || { echo "QUANT FAILED"; exit 1; }
f=models/gguf/q38_dom_dyn_bart.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf $f --embed-kq q4_k --head-kq q6_k --temp-file 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}
  for off in 0 2 4; do echo "== kld $name on $tag windows $off-$((off+1)) $(date -u +%T)"; $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 2 --win-offset $off --dtype bfloat16 --out results/kld/${name}_${tag}_w$off.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-200; done
  $PY - <<PY
import json, math
parts = [json.load(open(f"results/kld/${name}_${tag}_w{o}.json")) for o in (0, 2, 4)]; n = sum(p["tokens"] for p in parts)
m = {k: sum(p[k] * p["tokens"] for p in parts) / n for k in ("kl_mean", "top1_agree", "ref_top1_in_cand_top5")}
m["ppl_ref"] = math.exp(sum(math.log(p["ppl_ref"]) * p["tokens"] for p in parts) / n); m["ppl_cand"] = math.exp(sum(math.log(p["ppl_cand"]) * p["tokens"] for p in parts) / n)
m.update(ref=parts[0]["ref"], cand=parts[0]["cand"], text="$text", tokens=n, merged_from=3); json.dump(m, open("results/kld/${name}_${tag}.json", "w"), indent=1)
print(f"$name on $tag (6 windows merged): KL {m['kl_mean']:.5f}, top-1 {100*m['top1_agree']:.2f}%, ppl ref {m['ppl_ref']:.3f} cand {m['ppl_cand']:.3f}")
PY
done
echo "== Q38 BART M4 DONE $(date -u +%T)"
