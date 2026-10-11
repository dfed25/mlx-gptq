#!/bin/bash
# End-to-end: HF model -> K-quant GGUF that targets a reference file's size, with Dom's recipe + bit budget + ridge + broad calibration + KL fine-tune,
# scored (ppl + KL / top-1 vs fp16 on 5 held-outs) against the reference. Every stage is skipped when its output exists; rerun to resume.
#   bash pipeline.sh NAME MODEL(HF id or dir) REF.gguf [options]
#   --target-bpw X      average bits/weight over the linears (default: match REF.gguf's size)      --forms q4_k,q5_k,q6_k
#   --embed q8_0|q6_k|q4_k   embedding kind (default q8_0 for tied models, q4_k otherwise)  --head q6_k (untied models; default q6_k)
#   --nsamples 256 --refine 3 --refit 2 --ridge 0.05 --tune-steps 300 --tune-bs 16 --stream (layer-streamed fine-tune for big models) --dtype float16|bfloat16 --nwin 8 (KL windows)
#   --head-rtn          round-to-nearest head instead of the default GPTQ'd head (untied models; head_gptq.py, Hessian from the quantized model's final-norm outputs)
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib HF_HUB_ENABLE_HF_TRANSFER=0; PY=~/venv-vllm/bin/python
NAME=$1; MODEL=$2; REF=$3; shift 3
TBPW=""; FORMS=q4_k,q5_k,q6_k; EMBED=""; HEAD=q6_k; NS=256; REFINE=3; REFIT=2; RIDGE=0.05; TSTEPS=300; TBS=16; STREAM=0; DT=float16; NWIN=8; HEADGPTQ=1
while [ $# -gt 0 ]; do case $1 in --target-bpw) TBPW=$2; shift 2;; --forms) FORMS=$2; shift 2;; --embed) EMBED=$2; shift 2;; --head) HEAD=$2; shift 2;; --nsamples) NS=$2; shift 2;; --refine) REFINE=$2; shift 2;; --refit) REFIT=$2; shift 2;;
  --ridge) RIDGE=$2; shift 2;; --tune-steps) TSTEPS=$2; shift 2;; --tune-bs) TBS=$2; shift 2;; --stream) STREAM=1; shift;; --dtype) DT=$2; shift 2;; --nwin) NWIN=$2; shift 2;; --head-rtn) HEADGPTQ=0; shift;; *) echo "unknown option $1"; exit 1;; esac; done
W=models/$NAME; mkdir -p $W results/$NAME results/kld budgets; LOG=results/$NAME/pipeline.log; say() { echo "== $* $(date -u +%T)" | tee -a $LOG; }
LOWRAM=$([ $(free -g | awk 'NR==2{print $2}') -ge 28 ] && echo 0 || echo 1)
export TMPDIR=$W/tmp; mkdir -p $TMPDIR; PACKF=$([ $LOWRAM = 0 ] || echo --temp-file)     # 15 GB boxes: spool the pack through a temp file on the big disk (/tmp is a small tmpfs there)
# 0. source checkpoint
SRC=$W/src; if [ ! -f $SRC/config.json ]; then say "fetch $MODEL"
  if [ -d "$MODEL" ]; then ln -s $(readlink -f $MODEL) $SRC; else $PY -c "from huggingface_hub import snapshot_download; import os; p=snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','*.tiktoken']); os.symlink(p, '$SRC')"; fi; fi
TIED=$($PY -c "import json; c=json.load(open('$SRC/config.json')); c=c.get('text_config',c); print(int(bool(c.get('tie_word_embeddings', False))))")
[ -n "$EMBED" ] || EMBED=$([ "$TIED" = 1 ] && echo q8_0 || echo q4_k)
EFLAG=$([ "$EMBED" = q8_0 ] && echo "--embed-q8" || echo "--embed-kq $EMBED"); [ "$TIED" = 1 ] || EFLAG="$EFLAG --head-kq $HEAD"
# 1. calibration (the v4 recipe rendered with this model's chat template) + held-outs
[ -f python_calib.txt ] || $PY -c "s=open('calib_mixed.txt').read(); py=s[len(s)//2:]; assert py.count('def ')>200; open('python_calib.txt','w').write(py)"
CAL=$W/calib.txt; if [ ! -f $CAL ]; then say "calibration mix"; $PY make_calib.py --tok $SRC --out $CAL 2>&1 | grep -vE "Warning|warn|cached|offline|Found the" | tail -1 | tee -a $LOG; fi
# 2. sensitivity table + bit budget to the target
TAB=budgets/${NAME}_table.json; if [ ! -f $TAB ]; then say "sensitivity pass ($FORMS)"
  $PY bit_budget.py $SRC budgets/${NAME}_tmp.json --target-bpw 4.5 --kq-forms $FORMS --calib $CAL --nsamples $NS --dtype $DT --bs 4 --table $TAB 2>&1 | grep -E "budget:|wrote|Traceback|Error" | tee -a $LOG; fi
[ -n "$TBPW" ] || TBPW=$($PY size_target.py $SRC $REF $TAB --embed $EMBED --head $HEAD)
MAP=budgets/${NAME}_bpw$TBPW.json; if [ ! -f $MAP ]; then say "budget: target $TBPW bpw (embeddings $EMBED$([ "$TIED" = 1 ] || echo ", head $HEAD"))"
  $PY bit_budget.py $SRC $MAP --target-bpw $TBPW --kq-forms $FORMS --table $TAB 2>&1 | grep -E "budget:|Traceback|Error" | tee -a $LOG; fi
# 3. quantize (Dom's recipe under the K-quant constraint, per-linear kinds from the map, ridged refinement)
Q=$W/quant; if [ ! -f $Q/model.safetensors.index.json ]; then say "quantize: refine $REFINE / refit $REFIT / ridge $RIDGE, $NS windows"
  $PY gptq_torch.py $SRC $Q 4 --group 32 --mse --lloyd --refine $REFINE --refit $REFIT --refine-damp $RIDGE --kq-map $MAP --dtype $DT --bs 4 --calib $CAL --nsamples $NS $([ $STREAM = 1 ] && echo --codes-only) 2>&1 | tee results/$NAME/quant.log | grep -E "^layer [0-9]+ done|saved|Traceback|Error" | tail -3 | tee -a $LOG; fi
[ -f $Q/model.safetensors.index.json ] || { say "QUANTIZE FAILED"; exit 1; }
# 4. pack the untuned model
A=$W/${NAME}_untuned.gguf; if [ ! -f $A ]; then say "pack untuned ($EFLAG)"; $PY codes2gguf.py $Q $REF $A $EFLAG $PACKF 2>&1 | tail -1 | tee -a $LOG; fi
# 4b. GPTQ'd output head (untied models): Dom's recipe on lm_head with the head's Hessian from the quantized model's final-norm outputs, then repack
HC=""; if [ "$TIED" = 0 ] && [ $HEADGPTQ = 1 ]; then HF=$W/head_$HEAD.safetensors
  if [ ! -f $HF ]; then say "head GPTQ ($HEAD, $NS windows)"; $PY head_gptq.py $SRC $W/head --gguf $A --calib $CAL --nsamples $NS --dtype $DT --bs 4 --kinds $HEAD --refine $REFINE --refit $REFIT --refine-damp $RIDGE 2>&1 | grep -E "head error|Traceback|Error" | tee -a $LOG; fi
  if [ -f $HF ]; then HC="--head-codes $HF"; A2=$W/${NAME}_untuned_hq.gguf; [ -f $A2 ] || { say "repack untuned with the GPTQ head"; $PY codes2gguf.py $Q $REF $A2 $EFLAG $HC $PACKF 2>&1 | tail -1 | tee -a $LOG; }; [ -f $A2 ] && { rm -f $A; A=$A2; }; else say "head GPTQ failed; keeping the RTN head"; fi          # the RTN-head file is superseded (16 GB on a 27B)
fi
# 5. KL fine-tune of the continuous parameters (codes fixed), on the same calibration text
T=$W/tuned; if [ ! -f $T/kl_tune.json ]; then say "KL fine-tune: $TSTEPS steps x $TBS windows$([ $STREAM = 1 ] && echo " (layer-streamed)")"
  if [ $STREAM = 1 ]; then [ -f $Q/packed/layer000.safetensors ] || $PY pack_codes.py $Q/codes $Q/packed 2>&1 | tail -1
    $PY -u kl_tune_stream.py $SRC $Q $T --gguf $A --packed $Q/packed --calib $CAL --nwin $NS --bs $TBS --steps $TSTEPS --lr 1e-4 --lr-norm 1e-5 --dtype $DT $([ $(free -g | awk 'NR==2{print $2}') -ge 28 ] && echo --codes-ram) 2>&1 | grep --line-buffered -vE "Warning|warn|Token indices|Loading weights" | tee results/$NAME/tune.log | grep --line-buffered -E "step +[0-9]*0 |exported|Traceback|Error" | tail -3 | tee -a $LOG
  else ACC=$(( TBS / 4 )); $PY kl_tune.py $SRC $Q $T --calib $CAL --steps $TSTEPS --bs 4 --accum $ACC --lr 1e-4 --lr-norm 1e-5 2>&1 | grep -vE "Warning|warn|Token indices|Loading weights" | tee results/$NAME/tune.log | grep -E "held-out|exported|Traceback|Error" | tail -4 | tee -a $LOG; fi; fi
[ -f $T/kl_tune.json ] || { say "TUNE FAILED"; exit 1; }; rm -rf $Q/packed                       # the packed-codes cache is only needed by the streamed tune
B=$W/${NAME}_tuned.gguf; if [ ! -f $B ]; then say "pack tuned"; $PY codes2gguf.py $T $REF $B $EFLAG $HC $PACKF 2>&1 | tail -1 | tee -a $LOG; fi
# 6. evaluation: fp16 reference ppl, then ppl (wiki, code) + KL / top-1 on 5 held-outs for REF, untuned, tuned
say "evaluate"; cp -n chat_test.txt ultrachat_test.txt multi_test.txt results/$NAME/ 2>/dev/null
for text in wikitext2_test.txt code_test.txt; do t=${text%%_*}; [ -f results/$NAME/ppl_fp16_$t.json ] || $PY ppl_stream.py $SRC --dtype $DT --text $text --out results/$NAME/ppl_fp16_$t.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  fp16 $t: /" | tee -a $LOG; done
for spec in "ref:$REF" "untuned:$A" "tuned:$B"; do tag=${spec%%:*}; g=${spec#*:}
  for text in wikitext2_test.txt code_test.txt; do t=${text%%_*}; [ -f results/$NAME/ppl_${tag}_$t.json ] || $PY ppl_stream.py $SRC --dtype $DT --gguf $g --text $text --out results/$NAME/ppl_${tag}_$t.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  $tag $t: /" | tee -a $LOG; done
  for text in wikitext2_test.txt code_test.txt chat_test.txt ultrachat_test.txt multi_test.txt; do t=${text%%_*}; [ -f results/$NAME/kld_${tag}_$t.json ] && continue
    if [ $LOWRAM = 1 ] && [ $NWIN -gt 2 ]; then                                          # 15 GB boxes: the evaluator's reference cache for 6 windows is 6 GB; run 2-window pieces and merge token-weighted
      for off in $(seq 0 2 $((NWIN - 1))); do [ -f results/$NAME/kld_${tag}_${t}_w$off.json ] || $PY kld_stream.py $SRC --cand-gguf $g --text $text --nwin 2 --win-offset $off --dtype $DT --out results/$NAME/kld_${tag}_${t}_w$off.json 2>&1 | grep -E "rror|Traceback" | tee -a $LOG; done
      $PY - <<PY | tee -a $LOG
import json, math, glob
parts = [json.load(open(f)) for f in sorted(glob.glob("results/$NAME/kld_${tag}_${t}_w*.json"))]; n = sum(p["tokens"] for p in parts)
m = {k: sum(p[k] * p["tokens"] for p in parts) / n for k in ("kl_mean", "top1_agree", "ref_top1_in_cand_top5")}
m["ppl_ref"] = math.exp(sum(math.log(p["ppl_ref"]) * p["tokens"] for p in parts) / n); m["ppl_cand"] = math.exp(sum(math.log(p["ppl_cand"]) * p["tokens"] for p in parts) / n)
m.update(ref=parts[0]["ref"], cand=parts[0]["cand"], text="$text", tokens=n, merged_from=len(parts)); json.dump(m, open("results/$NAME/kld_${tag}_$t.json", "w"), indent=1)
print(f"  $tag $t ({n} tokens, {len(parts)} pieces): KL {m['kl_mean']:.5f}, top-1 {100*m['top1_agree']:.2f}%")
PY
    else $PY kld_stream.py $SRC --cand-gguf $g --text $text --nwin $NWIN --dtype $DT --out results/$NAME/kld_${tag}_$t.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-200 | sed "s/^/  $tag $t: /" | tee -a $LOG
    fi; done
done
# 7. summary
$PY - <<PY | tee results/$NAME/summary.md
import json, os
R = "results/$NAME"; rows = [("$MODEL fp16/bf16", "fp16", None), ("$(basename $REF) (reference)", "ref", "$REF"), ("ours, untuned", "untuned", "$A"), ("ours, tuned", "tuned", "$B")]
def j(p): return json.load(open(p)) if os.path.exists(p) else None
print(f"# $NAME: Dom's recipe (map to $TBPW bpw, ridge $RIDGE, $NS windows, head {'GPTQ ' if '$HC' else 'RTN '}$HEAD, fine-tune $TSTEPS x $TBS) vs reference\\n")
print("| file | MiB | wiki ppl | code ppl | wiki KL / top-1 | code KL / top-1 | chat KL / top-1 | UltraChat KL / top-1 | multilingual KL / top-1 |"); print("|---|---|---|---|---|---|---|---|---|")
for label, tag, path in rows:
    c = [label, f"{os.path.getsize(path)/2**20:,.0f}" if path else ""]
    for t in ("wikitext2", "code"): d = j(f"{R}/ppl_{tag}_{t}.json"); c.append(f"{d['perplexity']:.3f}" if d and 'perplexity' in d else (f"{d['ppl']:.3f}" if d and 'ppl' in d else ""))
    for t in ("wikitext2", "code", "chat", "ultrachat", "multi"): d = j(f"{R}/kld_{tag}_{t}.json"); c.append(f"{d['kl_mean']:.4f} / {100*d['top1_agree']:.2f}%" if d else "")
    print("| " + " | ".join(c) + " |")
PY
say "PIPELINE DONE"
