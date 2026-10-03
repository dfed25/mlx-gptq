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

## Qwen2.5-1.5B-Instruct at 3 and 2 bits: the grid fit, refinement and weighted refit (2026-10-02)

All runs: group 64, same calibration (128 x 512 tokens of WikiText-2 train), evaluated in process with `--eval`;
the three-seed spread of this pipeline is 0.02. Shipped numbers in earlier rows of this file used `--mse` alone.

| 3-bit recipe | perplexity |
|---|---|
| `--mse` (range-search grid, the earlier default) | 10.899 |
| `--mse --lloyd` (alternating grid fit) | 10.654 |
| `--mse --lloyd --refine 3` (coordinate-descent refinement of the codes on the layer objective) | 10.552 |
| `--mse --lloyd --refine 3 --refit 2` (plus least-squares refit of every group's grid in the layer metric) | **10.379** |
| `--mse --lloyd --template --refine 3` (learned non-uniform ticks; not packable in MLX's affine format) | 10.374 |
| `--mse --lloyd --refine 2 --refine-z 1.0 --pieces 4` (flips accepted only above 1 standard error) | 10.534 |

| 2-bit recipe | perplexity |
|---|---|
| `--mse` alone, earlier measurement with the min-max grid | 689 |
| `--mse --lloyd` | 21.94 |
| `--mse --lloyd --refine 3` | 19.63 |

Second model, SmolLM2-1.7B-Instruct, 3-bit, same settings: `--mse` 11.493 → full recipe **10.198** (fp16 8.939): half of
the 3-bit gap closed.

At 4 bits the refinement adds nothing (`--mse --lloyd` 9.594, with `--refine 3` 9.605). Reference: fp16 9.379.

The refinement is coordinate descent on e^T H e (cf. CDQuant, 2024), implemented in `refine_cd.py`; on held-out
activations it lowers a layer's objective by 12-15% over GPTQ (21-24% on the calibration activations, so part of the
apparent gain is optimism). The refit solves, per row, the exact least-squares problem for all group offsets and steps
under the same objective and never increases it. `--refine-z` is a significance filter: a flip is accepted only if its
gain exceeds z standard errors estimated from disjoint pieces of the calibration text; on real data it reaches the
same perplexity with a third of the flips.

## Qwen2.5-1.5B-Instruct: mlx-lm's own quantizers, same evaluation, all packed models

| method | bits/weight | perplexity | size | decode tok/s (same session) |
|---|---|---|---|---|
| mlx-community 4-bit (round to nearest, 4-bit embedding) | 4.5 | 10.669 | | 195.4 |
| `mlx_lm.gptq` 4-bit, group 64 (6-bit embedding fallback) | 4.80 | 10.754 | 903 MB | 188.1 |
| `mlx_lm.awq` 4-bit, group 64 (4-bit embedding, group 32) | | 10.498 | 853 MB | |
| **mlx-gptq 4-bit, error-minimising grid, 8-bit embedding** | | **9.658** | 951 MB | 179.8 |

Same 128 x 512 calibration budget for all calibrated methods. The speed differences track the embedding width (4, 6, 8
bits); with a 4-bit embedding our model matched the community model's speed in an earlier measurement.

**Calibration text matters, and we calibrate in-domain.** Our default calibrates on WikiText-2 train and evaluates on
WikiText-2 test. Swapping calibration texts (same everything else):

| method | calibration text | perplexity |
|---|---|---|
| this repository (`--mse`, fp16 embedding) | WikiText-2 train | 9.653 |
| this repository (`--mse`, fp16 embedding) | mlx-lm's generic calibration_v5.txt | 9.945 |
| `mlx_lm.gptq`, current main (index bug fixed), 6-bit embedding | calibration_v5.txt | 10.408 |
| `mlx_lm.gptq`, current main, 6-bit embedding | WikiText-2 train | 10.283 |
| `mlx_lm.gptq`, current main, 4-bit embedding | calibration_v5.txt | 10.878 |

So about 0.3 of our lead on this benchmark is in-domain calibration; about 0.45 remains on identical text. An ablation of
the recipe differences (sequential calibration, grid timing) accounts for at most 0.07 of it.

**Noise floors.** This pipeline, three calibration seeds (Qwen, `--mse`): 9.653 / 9.644 / 9.635, spread 0.02, so the
grid-search gain (0.05) and the grid-fit gain (0.06) are real. `mlx_lm.gptq` (current main) is deterministic for a fixed
seed but far less stable: seed 123 gives 10.408 and seed 7 gives 10.152 on its default calibration text; accumulating its
Hessian in float32 instead of the activation dtype moves the result to 9.897 with WikiText calibration and to 10.866 with
its default text; computing its inverse-Hessian chain in float64 changes nothing further. Its default calibration file is
small (about 100k tokens), so 128 x 512 samples cover most of it. The cause of the instability is not identified; the
dtype of the Hessian and of its inverse are ruled out.

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
