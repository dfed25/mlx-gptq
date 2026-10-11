#!/bin/bash
# Longer 27B KL fine-tune on m2 (15 GB RAM, disk ~150 MB/s so every step re-reads the 16 GB of packed codes): the v4-calibrated budget-map codes,
# 300 steps x 16 windows drawn from 1024 v4 windows (the 100 x 16 run on m1 saw its 256 windows ~19 times; this one sees 1024 windows ~4.7 times:
# longer AND broader, to limit the chat/UltraChat sharpening of the short run). Then export, pack, wiki ppl, KL on the 5 held-outs in 2-window pieces.
# Baseline (100 x 16 on m1): wiki 0.0122 / code 0.0173 / chat 0.178 / UltraChat 0.097 / multi 0.0114, packed wiki ppl 6.800.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
until grep -q "M2 COPY DONE" results/copy_from_m1.out 2>/dev/null; do sleep 120; done
base=q38_kqdyn_v4; name=${base}_tune300; g=models/gguf/q38_dom_dyn_v4.gguf
[ "$(ls models/$base/codes | wc -l)" -ge 64 ] || { echo "codes incomplete"; exit 1; }
[ -f models/$base/packed/layer063.safetensors ] || { echo "== pack codes (29 GB -> 16 GB) $(date -u +%T)"; $PY pack_codes.py models/$base/codes models/$base/packed 2>&1 | tail -1 | cut -c1-200; }
[ -f models/$base/packed/layer063.safetensors ] || { echo "packed incomplete"; exit 1; }; [ -f $g ] || { echo "no $g"; exit 1; }
echo "== kl_tune_stream $name (calib_v4, 1024 windows, bs 16, 300 steps, warmup 20, packed codes) $(date -u +%T)"; rm -f models/$name/codes/zz_tuned_* 2>/dev/null
$PY -u kl_tune_stream.py models/q38_27b_src models/$base models/$name --gguf $g --packed models/$base/packed --calib calib_v4.txt --nwin 1024 --bs 16 --steps 300 --warmup 20 --lr 1e-4 --lr-norm 1e-5 --dtype bfloat16 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights" | tee results/${name}_tune.log | grep --line-buffered -E "step +[0-9]*0 |teacher|student|training|embedd|export|held|Traceback|Error" | cut -c1-200
[ -f models/$name/kl_tune.json ] || { echo "TUNE FAILED"; exit 1; }
while [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -lt 20 ]; do echo "waiting for >= 20G free disk before the pack ($(date -u +%T))"; sleep 3600; done
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
