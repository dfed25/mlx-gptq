# Measurements (M4 Pro, 20 GPU cores, 24 GB, MLX 0.32, mlx-lm 0.31)

WikiText-2 test perplexity, 20 windows of 2048 tokens. Bits per weight include the fp16 scale and offset per group of 64.

## Qwen2.5-1.5B-Instruct

| method | bits/weight | perplexity |
|---|---|---|
| fp16 | 16 | 9.379 |
| mlx-community 4-bit (round to nearest) | 4.5 | 10.669 |
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

## Settings

128 calibration sequences of 512 tokens from WikiText-2 train (seed 0), damping 1% of the mean Hessian diagonal,
block size 128, grid search over range fractions 1.0 to 0.6 in 21 steps.
