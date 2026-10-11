#!/bin/bash
# Resume of job_m2_q38tune300.sh from the pack: that job's --temp-file pack spools 16 GB through /tmp, which is a 7.8 GB tmpfs on m2, so it will fail.
# This waits for the tune to finish and the main job to exit, then packs with TMPDIR on the main disk and runs the ppl + KL stages.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib TMPDIR=$HOME/domq/port/tmp; PY=~/venv-vllm/bin/python; mkdir -p results/kld $TMPDIR
base=q38_kqdyn_v4; name=${base}_tune300; g=models/gguf/q38_dom_dyn_v4.gguf
until [ -f models/$name/kl_tune.json ] && ! pgrep -f "^bash job_m2_q38tune300.sh" >/dev/null; do sleep 300; done
grep -q "Q38 TUNE300 M2 DONE" results/job_m2_q38tune300.out 2>/dev/null && { echo "main job finished on its own"; exit 0; }
echo "== resume from the pack $(date -u +%T)"; rm -f models/gguf/q38_dom_dyn_v4_tune300.gguf
f=models/gguf/q38_dom_dyn_v4_tune300.gguf; echo "== pack $(date -u +%T)"; $PY codes2gguf.py models/$name $g $f --embed-kq q4_k --head-kq q6_k --temp-file 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $f --out results/ppl_gguf_${name}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt multi_test.txt ultrachat_test.txt chat_test.txt; do tag=${text%%_*}
  for off in 0 2 4; do [ -f results/kld/${name}_${tag}_w$off.json ] && continue; echo "== kld $name on $tag windows $off-$((off+1)) $(date -u +%T)"
    $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 2 --win-offset $off --dtype bfloat16 --out results/kld/${name}_${tag}_w$off.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-200; done
  $PY - <<PY
import json, math
parts = [json.load(open(f"results/kld/${name}_${tag}_w{o}.json")) for o in (0, 2, 4)]; n = sum(p["tokens"] for p in parts)
m = {k: sum(p[k] * p["tokens"] for p in parts) / n for k in ("kl_mean", "top1_agree", "ref_top1_in_cand_top5")}
m["ppl_ref"] = math.exp(sum(math.log(p["ppl_ref"]) * p["tokens"] for p in parts) / n); m["ppl_cand"] = math.exp(sum(math.log(p["ppl_cand"]) * p["tokens"] for p in parts) / n)
m.update(ref=parts[0]["ref"], cand=parts[0]["cand"], text="$text", tokens=n, merged_from=3); json.dump(m, open("results/kld/${name}_${tag}.json", "w"), indent=1)
print(f"$name on $tag (6 windows merged): KL {m['kl_mean']:.5f}, top-1 {100*m['top1_agree']:.2f}%, ppl ref {m['ppl_ref']:.3f} cand {m['ppl_cand']:.3f}")
PY
done
echo "== Q38 TUNE300 M2 DONE $(date -u +%T)"
