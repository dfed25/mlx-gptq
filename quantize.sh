#!/bin/bash
# Full pipeline: fp16 MLX model -> GPTQ (dequantized, for evaluation) -> packed MLX quantized model in float16.
#   ./quantize.sh MODEL_DIR OUT_DIR BITS [GROUP]
set -e
SRC=$1; OUT=$2; BITS=$3; G=${4:-64}
[ -f wikitext2_train.txt ] || python get_wikitext.py
python gptq_mlx.py "$SRC" "$OUT-deq" "$BITS" --group "$G" --mse
python ppl_wikitext.py "$OUT-deq" 20
python build_quantized.py "$OUT-deq" "$OUT-bf16" "$BITS" --group "$G" --embed-bits "${EMBED_BITS:-8}"
python cast_fp16.py "$OUT-bf16" "$OUT"
rm -rf "$OUT-bf16"
python ppl_wikitext.py "$OUT" 20
echo "done: $OUT (the dequantized copy $OUT-deq can be deleted)"
