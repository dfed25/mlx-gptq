#!/bin/bash
# Test: does the Hessian estimate limit us? Same map + ridge + calib_v4 as q38_kqdyn_v4, but 1024 calibration windows instead of 256 (unsloth's imatrix uses >1.5M tokens; ours used 131k).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR
while ! grep -q "Q38 TUNE M3 KL DONE" results/job_m3_q38tune_kl.out 2>/dev/null; do sleep 120; done
[ -f budgets/q38_kq456_4.783.json ] || scp -q -o StrictHostKeyChecking=accept-new <budgets-box>:~/domq/port/budgets/q38_kq456_4.783.json budgets/ 2>/dev/null
[ -f budgets/q38_kq456_4.783.json ] || { echo "no budget map on this box"; exit 1; }
name=q38_kqdyn_v4n1k; echo "== quantize $name (calib_v4, 1024 windows, map + ridge 0.05) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map budgets/q38_kq456_4.783.json --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 1024 2>&1 | tee results/${name}_quant.log | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ -f models/$name/model.safetensors.index.json ] || { echo "QUANT FAILED"; exit 1; }
f=models/gguf/q38_dom_dyn_v4n1k.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq q6_k --temp-file 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
rm -f models/$name/*.safetensors                                                                   # the dequantized shards (54 GB) are reproducible from codes + src; keep codes
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
echo "== Q38 N1K M3 DONE $(date -u +%T)"
