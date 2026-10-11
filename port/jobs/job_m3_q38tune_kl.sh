#!/bin/bash
# KL of the narrow fine-tuned 27B on m3 (15 GB RAM: the evaluator's 6-window reference cache is 6 GB, so 3 runs of 2 windows, merged token-weighted).
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld; name=q38_kqdyn_tune; f=models/gguf/q38_dom_dyn_tune.gguf
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
echo "== Q38 TUNE M3 KL DONE $(date -u +%T)"
