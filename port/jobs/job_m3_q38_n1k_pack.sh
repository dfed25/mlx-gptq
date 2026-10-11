#!/bin/bash
# Resume of job_m3_q38_n1k.sh from the pack: the dequantized shards (54 GB) left 8 GB free and the --temp-file pack needs 2 x 16 GB. The pack only needs
# the codes plus the NON-linear tensors, which are identical in the source, so pack from a lite dir (symlinks to the source shards + the codes), after the shards are gone.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld
export TMPDIR=$HOME/domq/port/tmp; mkdir -p $TMPDIR     # the --temp-file pack spools 16 GB through TMPDIR; /tmp on this box is a small tmpfs
name=q38_kqdyn_v4n1k; L=models/${name}_lite; mkdir -p $L
for f in $(ls models/q38_27b_src); do [ -e $L/$f ] || ln -s $(readlink -f models/q38_27b_src/$f) $L/$f; done
[ -e $L/codes ] || ln -s $(readlink -f models/$name/codes) $L/codes; cp -f models/$name/bits_used.json $L/ 2>/dev/null
[ "$(ls $L/codes | wc -l)" -ge 64 ] || { echo "codes incomplete"; exit 1; }
free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 34 ] || { echo "DISK: only ${free}G free"; exit 1; }
f=models/gguf/q38_dom_dyn_v4n1k.gguf; echo "== pack from $L $(date -u +%T)"; $PY codes2gguf.py $L models/gguf/q38_dom_dyn.gguf $f --embed-kq q4_k --head-kq q6_k --temp-file 2>&1 | grep -vE "Token indices|Warning" | tail -1 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }; echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') $(date -u +%T)"
for text in wikitext2_test.txt code_test.txt multi_test.txt; do tag=${text%%_*}
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
echo "== Q38 N1K M3 DONE $(date -u +%T)"
