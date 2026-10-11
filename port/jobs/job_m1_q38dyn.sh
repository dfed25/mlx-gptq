#!/bin/bash
# After the ridged 27B and the KL chain: spend the size headroom. Sensitivity pass under the K-quant forms q4_k/q5_k/q6_k, greedy budget to the
# average bpw that makes the packed file (Q4_K embeddings, Q6_K head, unsloth's layout) the size of unsloth's UD-Q4_K_M, quantize with the map
# (ridge chosen from the ridged-vs-full KL result), ppl, pack, packed ppl, KL on wiki + code.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python; mkdir -p results/kld budgets
while ! grep -q "Q38 KQ RIDGE DONE" results/job_m1_q38kq_ridge.out 2>/dev/null || ! grep -q "Q38 KLD2 DONE" results/job_m1_q38kld2.out 2>/dev/null; do sleep 300; done
# free the ridged run's dequantized shards (codes + GGUF stay; a symlink dir keeps it packable)
d=models/q38_kq4_mix_ridge_pack; if [ ! -d $d ]; then mkdir -p $d; for f in models/q38_27b_src/*; do ln -s ../../$f $d/; done; ln -s ../q38_kq4_mix_ridge/codes $d/codes; cp models/q38_kq4_mix_ridge/bits_used.json $d/; fi
rm -f models/q38_kq4_mix_ridge/*.safetensors; echo "== disk after cleanup: $(df -h ~ | awk 'NR==2{print $4}') free $(date -u +%T)"
U=$(stat -c %s models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf); O=$(stat -c %s models/gguf/q38_dom_q4_k_e6.gguf 2>/dev/null || echo 0)
T=$($PY -c "
import json; c=json.load(open('models/q38_27b_src/config.json')); c=c.get('text_config',c); V=c['vocab_size']; H=c['hidden_size']; N=24.33e9
O=$O - V*H*(6.5625-4.5)/8 if $O>0 else 0        # ours with Q4_K embeddings instead of Q6_K
print(f'{4.5+8*($U-O)/N:.3f}' if O>0 else '4.75')")
echo "== budget: unsloth $U bytes, ours(e6) $O bytes -> target $T bpw over the linears $(date -u +%T)"
$PY bit_budget.py models/q38_27b_src budgets/q38_kq456_$T.json --target-bpw $T --kq-forms q4_k,q5_k,q6_k --calib calib_mixed.txt --dtype bfloat16 --bs 4 --table budgets/q38_kq456_table.json 2>&1 | grep -E "budget:|wrote|loaded|Traceback|Error" | cut -c1-240
[ -f budgets/q38_kq456_$T.json ] || { echo "BUDGET FAILED"; exit 1; }
RD=$($PY -c "
import json,os
def kl(p): return json.load(open(p))['kl_mean'] if os.path.exists(p) else 9
r=kl('results/kld/q38_kq4_mix_ridge_wikitext2.json'); f=min(kl('results/kld/q38_ours_wikitext2.json'), kl('results/kld/q38_ours_e6_wikitext2.json'))
print('0.05' if r < f else '0.01')")
name=q38_kqdyn; echo "== quantize $name (refine-damp $RD, map budgets/q38_kq456_$T.json) $(date -u +%T)"; rm -rf models/$name
$PY gptq_torch.py models/q38_27b_src models/$name 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp $RD --kq-map budgets/q38_kq456_$T.json --dtype bfloat16 --bs 4 --calib calib_mixed.txt 2>&1 | grep -E "^layer ([0-9]|[1-6][0-9]) done|saved|Traceback|Error" | cut -c1-160
[ -f models/$name/model.safetensors.index.json ] || { echo "QUANT FAILED"; exit 1; }
$PY ppl_stream.py models/$name --dtype bfloat16 --out results/ppl_stream_${name}_wiki.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  wiki: /"
$PY ppl_stream.py models/$name --dtype bfloat16 --text code_test.txt --out results/ppl_stream_${name}_code.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  code: /"
f=models/gguf/q38_dom_dyn.gguf; echo "== pack map + Q4_K embeddings / Q6_K head $(date -u +%T)"
$PY codes2gguf.py models/$name models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf $f --embed-kq q4_k --head-kq q6_k 2>&1 | grep -vE "Token indices|Warning" | tail -2 | cut -c1-200
[ -f $f ] || { echo "PACK FAILED"; exit 1; }
echo "== packed size $(ls -l --block-size=M $f | awk '{print $5}') vs unsloth $(ls -l --block-size=M models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf | awk '{print $5}') $(date -u +%T)"
$PY ppl_stream.py models/q38_27b_src --dtype bfloat16 --gguf $f --out results/ppl_gguf_${name}.json 2>&1 | grep -E "perplexity|rror" | sed "s/^/  packed wiki: /"
for text in wikitext2_test.txt code_test.txt; do tag=${text%%_*}; echo "== kld $name on $tag $(date -u +%T)"
  $PY kld_stream.py models/q38_27b_src --cand-gguf $f --text $text --nwin 6 --dtype bfloat16 --out results/kld/${name}_$tag.json 2>&1 | grep -E "KL\(|rror|Traceback" | cut -c1-220
done
echo "== Q38 DYN DONE $(date -u +%T)"
