# Measurements (M4 Pro, 20 GPU cores, 24 GB, MLX 0.32, mlx-lm 0.31)

WikiText-2 test perplexity, 20 windows of 2048 tokens. Bits per weight include the fp16 scale and offset per group of 64.

## Qwen2.5-1.5B-Instruct

| method | bits/weight | perplexity |
|---|---|---|
| fp16 | 16 | 9.379 |
| mlx-community 4-bit (round to nearest) | 4.5 | 10.669 |
| **GPTQ 4-bit, alternating grid fit (`--mse --lloyd`), packed, 8-bit embedding** | 4.5 | **9.596** (dequantized 9.594) |
| GPTQ 4-bit, error-minimising grid (dequantized weights, fp16 embedding) | 4.5 | 9.653 |
| GPTQ 4-bit, min-max grid (dequantized weights, fp16 embedding) | 4.5 | 9.704 |
| GPTQ 4-bit, min-max grid, packed with a 4-bit embedding | 4.5 | 10.163 |
| GPTQ 3-bit, group 32, error-minimising grid | 4.0 | 10.468 |
| GPTQ 3-bit, error-minimising grid (+ 8-bit embedding) | 3.5 | 10.899 |
| GPTQ 3-bit, min-max grid | 3.5 | 11.299 |
| round to nearest 3-bit | 3.5 | 21.459 |
| GPTQ 2-bit | 2.5 | 689 |

Decode speed (alternating A/B, medians of 5): community 4-bit 138.5 tok/s, GPTQ 4-bit float16 with 4-bit embedding
137.8, GPTQ 3-bit + 8-bit embedding 125.8. The packed 4-bit model with the error-minimising grid and an 8-bit
embedding has not been measured yet on Qwen.

## Qwen2.5-1.5B-Instruct: mlx-lm's own quantizers, same evaluation, all packed models

| method | bits/weight | perplexity | size | decode tok/s (same session) |
|---|---|---|---|---|
| mlx-community 4-bit (round to nearest, 4-bit embedding) | 4.5 | 10.669 | | 195.4 |
| `mlx_lm.gptq` 4-bit, group 64 (6-bit embedding fallback) | 4.80 | 10.754 | 903 MB | 188.1 |
| `mlx_lm.awq` 4-bit, group 64 (4-bit embedding, group 32) | | 10.498 | 853 MB | |
| **mlx-gptq 4-bit, error-minimising grid, 8-bit embedding** | | **9.658** | 951 MB | 179.8 |

Same 128 x 512 calibration budget for all three calibrated methods (mlx-lm's use their own calibration text). The
speed differences track the embedding width (4, 6, 8 bits); with a 4-bit embedding our model matched the community
model's speed in an earlier measurement. A layer-by-layer ablation of the three recipe differences is the next step.

## SmolLM2-1.7B-Instruct

| method | bits/weight | perplexity |
|---|---|---|
| fp16 | 16 | 8.939 |
| round to nearest 4-bit | 4.5 | 10.537 |
| GPTQ 4-bit, error-minimising grid (dequantized weights) | 4.5 | 9.409 |
| **GPTQ 4-bit, packed, 8-bit embedding (the `quantize.sh` output)** | 4.5 | **9.412** |
| GPTQ 4-bit, packed, 4-bit embedding | 4.5 | 9.923 |
| GPTQ 3-bit, error-minimising grid (dequantized weights) | 3.5 | 11.493 |

Decode speed (alternating A/B, medians of 5, same session): round to nearest 4-bit 119.3 tok/s (922 MB), GPTQ 4-bit
with 8-bit embedding 113.7 tok/s (970 MB).

## The alternating grid fit (`--lloyd`)

For each group of 64 weights the 16-level grid is described by an offset and a step. Given an assignment of weights to
levels, the least-squares offset and step are an ordinary line fit of weight against level index; given the grid, the
best assignment is nearest-level. Alternating the two never increases the group's squared error and stops after finitely
many rounds, but can stop in a non-optimal valley (example: 0, 1, 6, 6, 9, 12 with 3 levels), so it is started from four
shrunk min-max grids and the result is kept per group only where it beats the range-search grid. On 60,000 real groups it
lowers the squared rounding error by about 8% over the range search; on the whole model 9.653 → 9.594.

## Settings

128 calibration sequences of 512 tokens from WikiText-2 train (seed 0), damping 1% of the mean Hessian diagonal,
block size 128, grid search over range fractions 1.0 to 0.6 in 21 steps.
