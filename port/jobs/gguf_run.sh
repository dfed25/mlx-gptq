#!/bin/bash
# Validate the GGUF loader (0.8B BF16 GGUF must give the fp16 ppl; 27B Q4_K_M tensors must match HF after inversion), then
# measure the 27B Q4_K_M perplexity on Dom's protocol by streaming. Waits for the downloads and for the GPU to be free.
set -u; cd ~/domq/port; export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$HOME/domq/pylib; PY=~/venv-vllm/bin/python
while [ ! -f models/gguf/.done ]; do sleep 60; done
G27=models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf; G08=models/gguf/Qwen3.5-0.8B-BF16.gguf
echo "== gguf tensor names (27B, blk.0 and blk.3) $(date -u +%T)"; $PY - <<'PYEOF'
import gguf; r = gguf.GGUFReader("models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
for t in r.tensors:
    if t.name.startswith(("blk.0.", "blk.3.")) or "blk." not in t.name: print(f"  {t.name:32s} {t.tensor_type.name:7s} {[int(x) for x in reversed(list(t.shape))]}")
PYEOF
echo "== 27B tensor check vs HF (layers 0 and 3) $(date -u +%T)"; $PY gguf_check.py models/q38_27b_src $G27 0 3 2>&1 | grep -vE "Warning|warn" | tail -40
while pgrep -f "^bash tasks27.sh|^bash q38_fix.sh" >/dev/null; do sleep 120; done
echo "== 0.8B BF16 GGUF streamed ppl (expect 16.900) $(date -u +%T)"; $PY ppl_stream.py models/q35_08b_src --gguf $G08 --dtype bfloat16 2>&1 | grep -E "perplexity|rror" | tail -2
echo "== 27B UD-Q4_K_M streamed ppl $(date -u +%T)"; $PY ppl_stream.py models/q38_27b_src --gguf $G27 --dtype bfloat16 --bs 2 --out results/ppl_stream_q38_ggufQ4KM.json 2>&1 | grep -E "perplexity|rror" | tail -2
echo "== GGUF DONE $(date -u +%T)"
