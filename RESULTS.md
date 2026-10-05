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

## Independent evaluation on NVIDIA (M. Federico, 2026-10-04; A10G, vLLM 0.29, float16)

Our MLX checkpoints were unpacked (converter verified bit-exact against `mx.quantize`) and scored with the same
WikiText-2 protocol through vLLM: fp16 9.372 (ours 9.379), published 4-bit 9.596 (9.596), 3-bit 10.380 (10.38),
3-bit + DWQ 10.184 (10.18). Three findings beyond our own measurements:

| model | WikiText-2 | HumanEval pass@1 (164, greedy) |
|---|---|---|
| fp16 | 9.372 | 37.2% |
| Qwen official GPTQ-Int8 | 9.381 | 37.2% |
| ours, full recipe at 4 bits (`--mse --lloyd --refine 3 --refit 2`), exact | **9.537** | 34.8% |
| ours, published 4-bit (grid fit only), exact | 9.596 | 32.3% |
| Qwen official AWQ 4-bit | 10.161 | 34.1% |
| ours, 3-bit + DWQ, exact | 10.184 | 13.4% |
| ours, published 3-bit, exact | 10.380 | 14.0% |
| Qwen official GPTQ-Int4 | 10.398 | 27.4% |
| ours, 4-bit forced to an integer zero point (AWQ format) | 10.44–10.50 | 22.6–28.0% |

1. **The refit at 4 bits**: his run of the full recipe gave 9.537 vs 9.596. Our own re-measurement (three runs: 9.575, 9.583
   in memory, 9.586 packed) finds a smaller perplexity gain, at the pipeline's noise floor, but a clear code gain: HumanEval
   39.0% vs 32.3% (about 2 standard errors). The 0.05 gap between his 9.537 and our 9.58 is unexplained. The published
   4-bit model is now the full-recipe one.
2. **3-bit collapses on code**: about a third of the fp16 pass rate, while perplexity hides it. The 3-bit models are text
   models; this is now stated on their cards. HumanEval now runs in this pipeline (`mlx_lm.evaluate`, lm-eval 0.4.13) and
   reproduces his numbers: fp16 38.4%, published 4-bit 32.3%, 3-bit 14.0%, 3-bit + DWQ 14.0%. Two 3-bit experiments:

   | 3-bit variant | WikiText-2 | HumanEval |
   |---|---|---|
   | recipe, WikiText calibration (shipped) | 10.38 | 14.0% |
   | recipe, calibration on half WikiText + half Python source | 10.73 | **23.2%** |
   | recipe, layers 0-1 and 26-27 at 4 bits (+0.14 bits/weight) | 10.25 | 17.7% |

   The calibration text decides a large part of the code ability; a mixed set trades 0.35 perplexity for 9 points of code.
3. **The grid's fp16 offset does not survive integer zero-point formats** (GPTQ/AWQ/Marlin kernels): rounding the offset
   costs 0.9 perplexity and 10 points of HumanEval. The advantage currently lives in MLX's affine format. The fix, `--zero-point int`,
   builds the integer-zero-point constraint into the initial grid, the grid fit and the refit (closed-form step for each
   of the 16 candidate zero points per group, coordinate descent over groups). Full model, evaluated in MLX with the
   constraint: **4-bit 9.662** (free offset 9.586; Qwen's AWQ 10.16, GPTQ-Int4 10.40; converting the free model
   afterwards 10.44–10.50) and **3-bit 10.704** (free 10.38). Packed, the 4-bit model scores 9.663 and 33.5% on HumanEval in MLX; converted to the
   AWQ layout with M. Federico's `mlx2hf.py --awq` the weights change by 0.05% (free-offset models: about 10%). The export
   is at huggingface.co/dfed24/Qwen2.5-1.5B-Instruct-gptq-4bit-int4-awq. **Verified on an NVIDIA A10G through vLLM's int4
   Marlin kernel (2026-10-05): perplexity 9.6625 and HumanEval 33.5%, identical to the MLX numbers**, against 10.161 / 34.1%
   for Qwen's official AWQ 4-bit and 10.398 / 27.4% for its GPTQ-Int4 on the same box and protocol. An offline rotation
   (QuaRot-style) was tested independently on the PyTorch port of this pipeline: no difference at 4 bits (all cells
   within a few hundredths); at 3 bits it helps the integer-zero-point version (10.70 -> 10.47) and barely changes the
   free-offset one (10.39 -> 10.35).

### Larger models (2026-10-05; M. Federico's PyTorch port of this pipeline, A10G, vLLM 0.29, one run per row)

| model | fp16 | ours, 4-bit | Qwen official AWQ 4-bit |
|---|---|---|---|
| Qwen2.5-7B-Instruct, integer zero points, on the int4 kernel | 7.145 / 70.1% | **7.288** / 67.1% | 7.583 / 64.6% |
| Qwen2.5-32B-Instruct, free offsets (streamed, not servable on the int4 kernel) | 4.761 | **4.892** | 5.043 / 65.9% |
| Qwen2.5-32B-Instruct, free offsets rounded afterwards, on the int4 kernel | | 5.079 / 64.6% | 5.043 / 65.9% |

Cells are WikiText-2 perplexity / HumanEval pass@1. The 7B model is at
huggingface.co/dfed24/Qwen2.5-7B-Instruct-gptq-4bit-int4-awq. The lead over Qwen's AWQ shrinks with size (0.50 at 1.5B,
0.30 at 7B on the kernel; 0.15 at 32B for the free-offset model), the HumanEval differences among 4-bit models are
inside the standard error (about 3.6 points), the calibration is in-domain for the perplexity test, and the 32B model
has not been made with integer zero points.

## Where the code ability is lost (2026-10-05)

`agree.py` runs a model teacher-forced over the 164 HumanEval reference solutions (8,872 positions) and counts how often
its top choice differs from the fp16 model's: 4.4% for the 4-bit full recipe, 4.9% for the 4-bit integer-zero-point
model, 11.3% for the 3-bit model. This tracks HumanEval (39%, 33.5%, 14%) and needs no code execution. A layer scan at
3 bits shows the damage is diffuse: restoring any single layer to fp16 removes at most 0.6 of the 11.3 points, and
quantizing any single layer alone creates 1 to 2.7 points. `humaneval_diag.py` sorts the 3-bit failures: of the 59
problems fp16 solves, 39 are lost, mostly to repetitive loops and placeholder stubs rather than syntax errors.

**On noise.** The 0.02 quoted in this file is the spread over calibration seeds on one machine. The same recipe and seed
gave 9.537 and 9.554 on two other machines and 9.575 to 9.586 here, so differences of about 0.05 between machines are
not meaningful.

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

Comparators and extensions at 3 bits (Qwen): `mlx_lm.awq` 3-bit with an 8-bit embedding 12.434; `mlx_lm.dwq` 3-bit fine-tuned
from the round-to-nearest start with 256 x 512 tokens of WikiText-2 train (a reduced budget) 13.545; the same DWQ run
started from our 3-bit model **10.184**; our recipe with 512 instead of 128 calibration sequences 10.352 (no real gain).

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
