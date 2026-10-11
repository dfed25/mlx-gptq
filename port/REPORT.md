# Dom's mlx-gptq models on NVIDIA: quality, code ability and serving speed under vLLM

Date: 2026-10-03/04. Box: m2 (AWS g5.xlarge, NVIDIA A10G 24 GB, vLLM 0.29.0, torch 2.13, float16).
Modelled on jarvislabs.ai's "vLLM quantization complete guide" (Qwen2.5-32B on an H200), scaled to the model Dom actually
quantized, Qwen2.5-1.5B-Instruct. Source: `WRITEUP_dom.pdf`, github.com/dfed25/mlx-gptq, huggingface.co/dfed24.

## Method

- **Perplexity**: Dom's own protocol, reproduced through vLLM's prompt log-probabilities (`ppl_vllm.py`): WikiText-2 test
  (raw, paragraphs joined by blank lines), the first 20 non-overlapping windows of 2048 tokens, token-level perplexity.
  Every row is scored with the same kernels that serve it. Cross-check: fp16 gives 9.372 here vs 9.379 in Dom's MLX run.
- **HumanEval pass@1**: greedy, completion-style prompts with lm-eval-harness's stop sequences, 164 problems executed with a
  10 s timeout (`humaneval_vllm.py`). Dom's writeup has no code metric; this is new.
- **Serving**: `vllm serve` + `vllm bench serve`, 200 prompts, concurrency 10, as in the blog, but with the `random` dataset
  (512 input / 128 output tokens, `--ignore-eos`) instead of ShareGPT (`bench_serve.sh`). Numbers are output tokens/s, mean
  time to first token, mean time per output token.
- **Dom's MLX checkpoints** were converted with `mlx2hf.py`, which unpacks MLX's affine format (weight = scale x code + bias,
  groups of 64) and was verified bit-exact against `mx.quantize`. Two exports: **exact fp16** (his model, to the last bit,
  but running as fp16 weights), and for 4-bit an **AWQ-format copy** whose group offsets are rounded to integer zero
  points so vLLM's int4 Marlin kernel can run it. The AWQ copy is NOT his model: rounding the offsets changes the weights
  by ~10% in mean relative terms. There is no 3-bit kernel in this vLLM (GPTQ is Marlin-only, 4/8-bit symmetric; AWQ is
  4-bit only), so the 3-bit rows are quality-only.
- **dom4full** is a model I made on the M3 Max with Dom's shipped recipe (`quantize.sh ... 4`, i.e. `--mse --lloyd
  --refine 3 --refit 2`, 128 x 512 WikiText-2 train calibration, 8-bit embedding, 47 min). His published 4-bit on HF used
  the range-search grid alone.

## Results

| row | what | vLLM kernel | checkpoint MB | WikiText-2 ppl | HumanEval pass@1 | out tok/s | TTFT ms | TPOT ms | weights in GPU GiB |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | Qwen2.5-1.5B-Instruct, unquantized | none | 3087 | **9.372** | **37.2%** | 927 | 151 | 9.7 | 2.98 |
| gptq8_marlin | Qwen's official GPTQ-Int8 | gptq_marlin | 1810 | 9.381 | 37.2% | 1285 | 179 | 6.4 | 1.74 |
| dom4full_fp16 | Dom's full recipe at 4 bits (made here), exact | fp16 | 3087 (794 packed) | **9.537** | **34.8%** | 927 | 151 | 9.7 | 2.98 |
| dom4_fp16 | Dom's published 4-bit, exact | fp16 | 3087 (985 packed) | 9.596 | 32.3% | 930 | 154 | 9.6 | 2.98 |
| awq / awq_marlin | Qwen's official AWQ 4-bit (g128) | awq, awq_marlin | 1615 | 10.161 | 34.1% | 1550 | 170 | 5.2 | 1.10 |
| dom3dwq_fp16 | Dom's 3-bit + DWQ fine-tune, exact | fp16 | 3087 (822 packed) | 10.184 | 13.4% | 931 | 154 | 9.6 | 2.98 |
| dom3_fp16 | Dom's published 3-bit, exact | fp16 | 3087 (822 packed) | 10.380 | 14.0% | 927 | 148 | 9.7 | 2.98 |
| gptq / gptq_marlin | Qwen's official GPTQ-Int4 (g128, sym) | gptq_marlin | 1150 | 10.398 | 27.4% | 1557 | 169 | 5.1 | 1.10 |
| dom4full_awq | dom4full with zero points rounded | awq_marlin | 1173 | 10.444 | 22.6% | 1533 | 172 | 5.2 | 1.12 |
| dom4_awq | dom4 with zero points rounded | awq_marlin | 1173 | 10.496 | 28.0% | 1531 | 171 | 5.2 | 1.12 |

"Packed" sizes are the MLX checkpoints as Dom ships them; the exact rows run as fp16 so their checkpoint and GPU size are the
fp16 model's. In vLLM 0.29 `awq` and `awq_marlin`, and `gptq` and `gptq_marlin`, resolve to the same Marlin kernels, which
is why those pairs are identical (the blog's 10x gap between plain AWQ and Marlin-AWQ no longer exists).

## Findings

1. **Dom's perplexity numbers reproduce on NVIDIA to three decimals**: 3-bit 10.380 (his 10.38), 4-bit 9.596 (his 9.596),
   3-bit+DWQ 10.184 (his 10.18), fp16 9.372 (his 9.379). Nothing in his table is an MLX or Apple artifact.
2. **At 4 bits his rounding beats both of Qwen's official 4-bit releases on quality**, by a wide margin: 9.54-9.60 vs AWQ
   10.16 and GPTQ-Int4 10.40, at a smaller packed size (794-985 MB vs 1150-1615 MB). About 0.3 of that is in-domain
   calibration, as he states on the model cards.
3. **The writeup's "at 4 bits nothing beyond the grid loop helps" is not what his own shipped recipe does.** Running
   `quantize.sh` at 4 bits (refinement + weighted refit) gives 9.537 vs 9.596 for the published model, and +2.5 points of
   HumanEval. The gain (0.06) is three times his stated noise floor (0.02). RESULTS.md attributes the 4-bit published model
   to `--mse --lloyd`; the refit was not re-measured at 4 bits.
4. **HumanEval, which his writeup does not measure, is where 3-bit falls apart.** 4-bit keeps 87-94% of the fp16 pass@1;
   both 3-bit models keep about 37%. The DWQ fine-tune that improves 3-bit perplexity by 0.2 does nothing for code (13.4%
   vs 14.0%). "A third to a half of the 3-bit gap closed" is true for perplexity and false for code generation.
5. **His grid does not survive a standard int4 kernel.** Rounding the group offsets to integer zero points so the model can
   run on vLLM's AWQ/Marlin kernel costs 0.9 of perplexity (9.54/9.60 -> 10.44/10.50) and 10-12 HumanEval points, landing
   it below Qwen's own AWQ. The quality advantage lives in the arbitrary fp16 offset that MLX's affine format stores and
   NVIDIA's int4 kernels do not. His 3-bit models cannot run on an NVIDIA int kernel in vLLM at all. On this stack his
   models are therefore a quality result, not a deployment artifact; to get the speed column (1.65x fp16 throughput, 1.1 GiB
   of weights) one would need a custom kernel with fp16 offsets, or a format with integer offsets (then the ruler must be
   fitted with that constraint, which his refit does not do).
6. **Interesting target for his refit**: the weighted refit could be re-derived with the constraint zero = integer (a
   discrete least squares over 16 candidate offsets per group). That would make the method deployable on GPTQ/AWQ kernels.

## Caveats

- One model (1.5B), one GPU, one seed per row. Dom's three-seed noise floor is 0.02 perplexity; HumanEval at n=164 has a
  standard error of about 3.5 points, so pass@1 differences under ~7 points between single runs are not significant.
  The 3-bit collapse (37 -> 14) and the AWQ-rounding loss (35 -> 23) are far outside that; dom4full vs dom4 (34.8 vs 32.3) is not.
- Speed uses a synthetic random dataset, not ShareGPT; the relative ordering (int4 Marlin 1.65x fp16, int8 1.4x) is what
  matters. At 1.5B and concurrency 10 the A10G is far from saturated.
- The flashinfer sampler had to be disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`): no nvcc on the box. It does not affect greedy
  results or perplexity.

## Files

`domq/`: `mlx2hf.py` (converter), `ppl_vllm.py`, `humaneval_vllm.py`, `bench_serve.sh`, `run_row.sh`, `run_all.sh`,
`run_extra.sh`, `summarize.py`, `mac_run.sh`; raw results in `results/` (`ppl_*.json`, `humaneval_*.json` with per-task
verdicts, `bench_*.json` with full latency percentiles, `meta.json`, `TABLE.md`); Dom's repo clone and the full-recipe 4-bit
model in `mlx-gptq/models/qwen1.5b-gptq4`. On m2, `~/domq/models/` still holds the downloaded and converted checkpoints (about
14 GB) for re-runs.

## Addendum (2026-10-04): neighbour-pooled importance in the refit (the user's BlockMax idea)

Question: can the per-input importances in Dom's weighted refit be replaced by one statistic per block of 16 neighbouring
inputs (as Packed BlockMax uses one decaying maximum per block of 16 weights), to denoise the calibration luck Dom
identified and to make the refit a closed-form weighted line fit instead of a per-row linear solve? Dom's MLX pipeline,
Qwen2.5-1.5B-Instruct, 3 bits, `--mse --lloyd --refine 3 --refit 2`, same seed and calibration for every row, in-process
evaluation, M3 Max (`mlx-gptq/gptq_mlx_nb.py`, `refine_nb.py`, `nb_run.sh`, `nb_run2.sh`, logs in `mlx-gptq/nb_results/`).

| refit metric | perplexity |
|---|---|
| no refit (refinement only), the floor | 10.487 |
| full calibration Hessian (Dom's refit) | **10.459** |
| diagonal of the Hessian, no pooling | 10.623 |
| diagonal averaged over blocks of 16 neighbours | 12.817 |
| diagonal max-pooled over blocks of 16 (the BlockMax rule) | 12.745 |

Result: negative, and clearly so. Any diagonal metric makes the refit worse than not refitting, and pooling over
neighbours is catastrophic (+2.3 perplexity), whether by mean or by max. The reason is visible in the structure of these
layers: a few outlier input channels dominate the Hessian, and the refit's job is to protect exactly those while using
the cross terms (which inputs fire together) to place the ruler. Pooling spreads an outlier's importance over its 15 quiet
neighbours (mean) or lifts the neighbours to the outlier's level (max); either way the ruler is fitted to the wrong
objective, and the code refinement that follows can only move codes by one tick, so it cannot repair a misplaced ruler.
The block-max rule works for optimizer scaling because there the quantity is a per-coordinate step size that only has to be
roughly right; here the metric is what the fit minimises, and its off-diagonal structure is the point.

Side finding: in this environment (mlx 0.32+ as installed today, our own fp16 conversion of the source) Dom's refit is
worth 0.03 over the refinement alone (10.487 -> 10.459), not the 0.17 his RESULTS.md reports (10.552 -> 10.379), while the
4-bit full recipe came out 0.06 better than his table. The control numbers should be re-established on his machine before
either difference is interpreted.

### Cross-term pooling (2026-10-04, same setup; the user's follow-up: pool the off-diagonals, not the importances)

| refit metric | Mac (MLX) ppl | A10G port ppl | port HumanEval |
|---|---|---|---|
| no refit (floor) | 10.487 | 10.491 | 9.8% |
| full H (Dom) | **10.459** | **10.391** | **14.0%** |
| diagonal | 10.623 | 10.648 | 12.8% |
| block-diagonal, 64-groups (bd64) | 10.629 | 10.595 | 9.8% |
| block-diagonal, 16-blocks (bd16) | 10.690 | 10.650 | 9.1% |
| cross terms pooled per 16-tile, exact diagonal (pool16) | 10.600 | 10.644 | 12.8% |
| cross terms pooled per 64-tile (pool64) | 10.688 | 10.699 | 14.0% |
| diag max-pooled over 16 (BlockMax rule) | 12.745 | 12.738 | 9.1% |
| diag mean-pooled over 16 | 12.817 | - | - |

All PSD by construction, exact importances kept. Every structured metric is worse than not refitting at all, on both
implementations; the ordering is stable across them. Keeping cross terms only within the quantization group (bd64), which
would make the refit a closed-form per-group solve, loses as much as dropping them entirely: the refit's value is in the
correlations BETWEEN groups, i.e. in the dense coupling of the whole row. Pooling the cross terms to tile means (pool16,
pool64) does not recover it either. Conclusion: no neighbour-structured approximation of the metric survives; the row-wise
dense solve is the method. The live alternative is to remove the structure the refit exploits (rotation; see the
replacement-candidate runs) rather than to approximate the metric.

### Task benchmarks of the refit variants (2026-10-05; lm-eval-harness 0.4.13 on vLLM, 0-shot except GSM8K 5-shot)

3-bit Qwen2.5-1.5B-Instruct from the A10G port; accuracy in percent; standard errors about 0.9 (ARC-e), 1.4 (ARC-c),
0.5 (HellaSwag), 1.3 (WinoGrande), 0.4 (MMLU), 1.3 (GSM8K). Raw files: `results/lmeval/`, `results/lmeval_table.json`.

| model | WikiText ppl | HumanEval | ARC-e | ARC-c | HellaSwag | WinoGrande | MMLU | GSM8K | mean of 6 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 9.372 | 37.2 | 76.8 | 43.5 | 50.8 | 62.9 | 60.2 | 58.4 | 58.8 |
| full H (Dom) | 10.391 | 14.0 | 74.1 | 41.1 | 46.6 | 63.9 | 53.9 | 31.2 | **51.8** |
| no refit | 10.491 | 9.8 | 74.7 | 40.4 | 46.6 | 63.0 | 54.0 | 29.9 | 51.4 |
| bd16 | 10.650 | 9.1 | 74.3 | 40.6 | 45.6 | 63.8 | 52.0 | 28.9 | 50.9 |
| pool64 | 10.699 | 14.0 | 73.2 | 38.7 | 45.9 | 63.0 | 51.0 | 31.8 | 50.6 |
| pool16 | 10.644 | 12.8 | 71.6 | 39.0 | 45.6 | 62.7 | 53.0 | 30.3 | 50.4 |
| diagonal | 10.648 | 12.8 | 72.1 | 38.8 | 45.6 | 63.5 | 52.4 | 29.1 | 50.2 |
| bd64 | 10.595 | 9.8 | 71.4 | 37.3 | 45.5 | 63.0 | 52.6 | 27.6 | 49.6 |
| max16 (BlockMax) | 12.738 | 9.1 | 71.6 | 38.8 | 44.2 | 62.7 | 44.8 | 19.4 | 46.9 |

The task ranking agrees with perplexity: Dom's full metric is best on the mean (its lead over no refit is +0.4, inside
noise on every single task), every structured metric sits below no refit by 0.5 to 1.8 points, and the max-pooled rule
loses 9 points of MMLU and 10 of GSM8K. Note what 3 bits costs regardless of refit: GSM8K falls from 58 to about 30 and
HumanEval from 37 to 14, while the multiple-choice tasks lose only 2 to 6 points; perplexity understates the damage to
reasoning and code at this bit width.

## The replacement candidate: rotation and integer zero points (2026-10-05)

Built in `port/`: `rotate.py` (QuaRot-style offline rotation: RMSNorm gains fused into the following projections,
randomized Hadamard on the residual stream, per-head Hadamard on the attention values; exact, verified on CPU: same
perplexity to four decimals) and `gptq_torch.py --int-zero` (Dom's alternating grid fit with the offset constrained to an
integer zero point, and his weighted refit re-derived under that constraint as group-wise coordinate descent in the layer
metric with the closed-form optimal step for each of the 16 candidate zero points; never increases the cost). The
integer-offset models pack losslessly into AWQ format (`codes2awq.py` uses the saved zero points) and run on vLLM's int4
Marlin kernel. Qwen2.5-1.5B-Instruct, A10G, same calibration and seed as everything above; "kernel" rows are the AWQ
packing measured through the kernel.

| 4 bits | ppl | HumanEval | kernel ppl / HumanEval | kernel tok/s |
|---|---|---|---|---|
| Dom's recipe, fp16 offsets (reference) | 9.554 | 34.8 (Mac-made copy) | cannot run; rounded: 10.496 / 28.0 | 1531 |
| rotation + Dom's recipe | 9.589 | 29.3 | cannot run | |
| **integer offsets, no rotation** | **9.625** | **34.1** | **9.625 / 33.5** | **1589** |
| rotation + integer offsets, seed 0 | 9.629 | 34.1 | 9.630 / 34.1 | 1587 |
| rotation + integer offsets, seed 1 | 9.619 | 31.7 | 9.619 / 32.3 | 1592 |
| Qwen official AWQ (g128) | | | 10.161 / 34.1 | 1550 |
| Qwen official GPTQ-Int4 (g128, sym) | | | 10.398 / 27.4 | 1557 |

| 3 bits (no int3 kernel in vLLM; fp16-exact only) | ppl | HumanEval |
|---|---|---|
| Dom's recipe, fp16 offsets | 10.391 | 14.0 |
| rotation + Dom's recipe | 10.352 | 14.6 |
| integer offsets, no rotation | 10.695 | 10.4 |
| rotation + integer offsets, seed 0 / seed 1 | 10.479 / 10.458 | 14.6 / 17.1 |

Findings. (1) At 4 bits the integer-offset constraint costs 0.07 perplexity against Dom's fp16 offsets and nothing on
HumanEval, against 0.94 and 7 points when his offsets are rounded after the fact; the packing is exact (kernel numbers
equal fp16 numbers to three decimals) and the model runs at 1.7x fp16 throughput. It beats Qwen's AWQ by 0.54 and its
GPTQ-Int4 by 0.77 on the same kernel. (2) Rotation is unnecessary at 4 bits: the four rotated/unrotated cells differ by
hundredths, inside the 0.02 noise floor, and rotation makes Dom's own recipe slightly worse. (3) At 3 bits rotation DOES
matter for the constrained version (10.70 -> 10.47) and helps Dom's recipe a little (10.39 -> 10.35): the fewer the levels,
the more the outlier structure hurts a constrained grid. (4) HumanEval differences among the 4-bit cells (29 to 34%) are
within the 3.5-point standard error. Recommendation: ship Dom's recipe + integer-offset fit/refit at 4 bits with no
rotation; keep rotation as an option for 3 bits, where no kernel exists yet anyway.

## Qwen2.5-32B-Instruct (2026-10-05): the PyTorch port at scale, one A10G

Dom's full recipe at 4 bits (`--mse --lloyd --refine 3 --refit 2`, 128 x 512 WikiText-2 train calibration, fp16 embedding
and head), streamed through the 24 GB card one layer at a time: 64 layers at about 11 minutes each, 19.2 GiB peak. The
exact model is a 65 GB fp16 checkpoint that no single A10G can serve, so its quality is measured by the streaming evaluator;
the kernel rows are the AWQ packing with rounded zero points (the integer-offset version was not run at 32B: 12 GPU hours).
Serving: 200 prompts, concurrency 16, 512 in / 128 out, `--enforce-eager`; all three int4 models are memory-bandwidth bound
on this card and run at the same speed. Raw files: `results/q32/`.

| 32B, 4 bits | WikiText-2 ppl | HumanEval | out tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| fp16 (streamed) | **4.761** | not measurable on one A10G | | | |
| Dom's recipe, exact fp16 offsets (streamed) | **4.892** | not measurable | | | |
| Dom's recipe, AWQ format with rounded offsets (int4 kernel) | 5.079 | 64.6% | 117 | 3321 | 60.1 |
| Qwen official AWQ (int4 kernel) | 5.043 | 65.9% | 118 | 3288 | 59.8 |
| Qwen official GPTQ-Int4 (int4 kernel) | 5.126 | 61.0% | 117 | 3273 | 60.0 |

Findings. (1) The port scales: Dom's exact recipe closes the 4-bit gap at 32B to 0.13 over fp16 and beats Qwen's AWQ by
0.15 and its GPTQ-Int4 by 0.23, with the same in-domain calibration caveat as at 1.5B. (2) Rounding his offsets costs 0.19
at 32B (4.892 -> 5.079), which puts the rounded copy 0.04 behind Qwen's AWQ; the HumanEval of the three kernel models is
within noise (61 to 66%, standard error 3.7). (3) The lead of his method shrinks with size, as expected (0.6 at 1.5B, 0.15 at
32B against AWQ), but it does not vanish; the integer-offset variant, which cost 0.07 instead of 0.94 at 1.5B, is the obvious
next run if a deployable 32B result is wanted. (4) Speed is format-independent at this size on this card.

### Qwen2.5-32B-Instruct, the deployable version (2026-10-06): Dom's recipe + integer zero points

Same run as above with `--int-zero` (14.5 h on one A10G, 19.2 GiB peak), packed losslessly into AWQ (`.zeros` kept exact).

| 32B, 4 bits | WikiText-2 ppl | HumanEval | out tok/s |
|---|---|---|---|
| fp16 (streamed) | 4.761 | | |
| Dom's recipe, fp16 offsets, exact (streamed; not kernel-runnable) | 4.892 | | |
| **Dom's recipe + integer zero points, exact (streamed)** | **4.919** | | |
| **the same through the int4 Marlin kernel (AWQ packing)** | **4.920** | **63.4%** | 117 |
| Dom's recipe, offsets rounded after the fact (int4 kernel) | 5.079 | 64.6% | 117 |
| Qwen official AWQ (int4 kernel) | 5.043 | 65.9% | 118 |
| Qwen official GPTQ-Int4 (int4 kernel) | 5.126 | 61.0% | 117 |

The integer-offset constraint costs 0.027 at 32B (0.07 at 1.5B) and the packing is lossless (4.919 -> 4.920). On the kernel the
model beats Qwen's AWQ by 0.12 and its GPTQ-Int4 by 0.21 at identical speed; HumanEval of the four kernel models (61 to 66%)
is within the 3.7-point standard error. The lead over the official AWQ is 0.12 at 32B against 0.54 at 1.5B: the method's
advantage shrinks with model size but survives, and this is the first version of it that a standard inference stack can run.

## Qwen3.8-27B (2026-10-06): the port on a hybrid linear-attention model

Architecture: 64 layers, three of four with gated linear attention (DeltaNet-style), one with full attention, 248k vocabulary,
untied head. The port discovers the linears per layer type (the two 48-wide gating projections and the conv stay fp16, as
vLLM expects), records one set of layer kwargs per layer type, and writes the checkpoint in the source layout so vLLM loads
it as the original. Dom's recipe with integer zero points, 4 bits, group 64, WikiText-2 calibration; 64 layers at 8.5 min
each on one A10G, 8.5 GiB peak; AWQ packing 19.2 GB (the official FP8 is 30.9 GB; no official int4 exists for this model).

| Qwen3.8-27B | WikiText-2 ppl | HumanEval | ARC-e | ARC-c | HellaSwag* | WinoGrande | MMLU* | GSM8K | serving tok/s |
|---|---|---|---|---|---|---|---|---|---|
| bf16 (streamed) | **6.773** | | | | | | | | |
| **Dom's recipe + integer offsets, exact (streamed)** | **6.826** | | | | | | | | |
| the same on the int4 Marlin kernel (AWQ packing) | 6.817 | **79.9%** | 82.6 | 56.2 | 57.0 | 75.6 | 83.6 | 77.7 | 32.5 |
| unsloth UD-Q4_K_M GGUF (16.5 GB), dequantized and streamed | 6.773 | not runnable here | | | | | | | |

*HellaSwag on its first 2500 examples, MMLU on the first 50 questions of each subject (the same fixed subsets as the 32B runs).

Findings. (1) The gap to bf16 is 0.05 (0.8%), the smallest of any model measured in this project; the linear-attention
projections quantize very gently (the refinement moved 3 to 8% of their codes against 10 to 20% on Qwen2.5 layers). The
packing is lossless within the bf16/fp16 difference of the two paths (6.826 streamed in bf16, 6.817 on the fp16 kernel).
(2) The serving row is 3.6x slower than the Qwen2.5-32B on the same card (32.5 vs 117 tok/s): vLLM 0.29's linear-attention
path on an A10G in eager mode without the fast kernels, not the quantization; relevant for anyone deploying this family on
Ampere cards. (3) unsloth's UD-Q4_K_M, read through a validated GGUF loader (`port/gguf_source.py`: inverse of the llama.cpp
converter's value-head reorder, decay and norm transforms; every unquantized tensor matches the HF original exactly, the
0.8B BF16 GGUF reproduces the fp16 perplexity 16.900), scores 6.773 on this protocol: equal to bf16. Its recipe is not
uniform 4-bit (5-bit attention and linear-attention projections, 8-bit gates, a mix of 3/4-bit feed-forwards) and it is
importance-matrix calibrated, most likely on WikiText-style text, so on WikiText-2 it enjoys the same in-domain advantage
Dom's writeup warns about; a second corpus is needed before reading 6.773 against 6.826 (see below).

### Second corpus for the 27B (2026-10-06): Python source, 20 windows of 2048 tokens

Text: seven `modeling_*.py` files from the transformers package (306 KB), a corpus none of the three models was calibrated
on (`port/code_ppl.sh`). The absolute perplexities are low because this code is public and in the base model's training
data; the gaps are what matter.

| Qwen3.8-27B | WikiText-2 ppl (gap) | Python-source ppl (gap) |
|---|---|---|
| bf16 | 6.773 | 1.371 |
| Dom's recipe + integer offsets, WikiText-2 calibration | 6.826 (+0.8%) | 1.437 (+4.8%) |
| unsloth UD-Q4_K_M (importance-matrix calibrated, mixed bits) | 6.773 (+0.0%) | 1.390 (+1.4%) |

Reading. On WikiText the GGUF's tie with bf16 is an in-domain artifact (one window: bf16 5.433, GGUF 5.534, mean |logit
diff| 0.13; a 20-window average equal to bf16 requires beating bf16 elsewhere, which only a calibration tuned to that text
produces). On code, where neither calibration set is in-domain, the picture reverses: unsloth's model is 3.4 points
closer to bf16 than ours. Two causes, both fixable and both already identified in Dom's writeup: (1) our calibration is
WikiText-2 only, so the grids are tuned for prose, while unsloth's importance matrix comes from a mixed corpus that
includes code; (2) unsloth's recipe spends more bits where it matters (5-bit attention projections, 8-bit gates) and fewer
on the feed-forward blocks, whereas ours is uniform 4-bit. The honest statement for the 27B is therefore: at uniform
4 bits with in-domain calibration, Dom's recipe is within 0.8% of bf16 on WikiText and 4.8% on code; a mixed calibration
set (Dom measured the cost of generic text at 0.05 on the 8B) is the next run, and a per-layer bit budget is the next idea.

### Task benchmarks of the 32B int4 models on the kernel (2026-10-06)

lm-eval-harness on vLLM, identical engine settings (3 sequences, 1536 context, eager); ARC and WinoGrande in full, GSM8K
5-shot in full, HellaSwag on its first 2500 examples, MMLU on the first 50 questions per subject. GSM8K is given with both
of lm-eval's extractors: "strict" requires the "The answer is N" format, "flexible" takes the last number in the output and
is unstable when a model keeps writing after its answer. Standard errors about 1.0 to 1.5 points on every task.

| Qwen2.5-32B, 4 bits, int4 kernel | ARC-e | ARC-c | HellaSwag* | WinoGrande | MMLU* | GSM8K strict | GSM8K flexible | mean (strict) |
|---|---|---|---|---|---|---|---|---|
| Dom's recipe + integer offsets (ppl 4.920) | 82.7 | 57.2 | 57.4 | 73.3 | 82.2 | 79.8 | 72.9 | 72.1 |
| Qwen official AWQ (ppl 5.043) | 81.8 | 57.5 | 57.4 | 73.7 | 82.5 | 83.9 | 59.8 | 72.8 |
| Qwen official GPTQ-Int4 (ppl 5.126) | 81.6 | 56.9 | 57.0 | 71.9 | 82.0 | 82.1 | 81.9 | 71.9 |

At 32B the three deployable 4-bit models are indistinguishable on tasks: every multiple-choice score is within a point,
and the strict GSM8K spread (80 to 84) is within two standard errors. The 0.12 to 0.21 perplexity advantage of Dom's recipe
does not show up on these benchmarks at this size, where all three models are already close to bf16; at 1.5B and 3 bits
it did (HumanEval 34% vs 27% for GPTQ-Int4, GSM8K 31% vs 19% for the worst variant). Perplexity is the right instrument for
ranking quantizers; task scores say whether the ranking matters, and at 32B/4-bit it stops mattering.

### Task benchmarks of the 1.5B int4 models on the kernel (2026-10-06)

Full tasks (no subsets), lm-eval-harness on vLLM; GSM8K strict is the stable extractor. fp16 row from the earlier run.

| Qwen2.5-1.5B, 4 bits | ARC-e | ARC-c | HellaSwag | WinoGrande | MMLU | GSM8K strict | mean |
|---|---|---|---|---|---|---|---|
| fp16 | 76.8 | 43.5 | 50.8 | 62.9 | 60.2 | 52.8 | 57.8 |
| Dom's recipe + integer offsets, int4 kernel (ppl 9.625) | 74.7 | 42.2 | 50.1 | 63.8 | 59.0 | 54.3 | **57.3** |
| Qwen official AWQ (ppl 10.161) | 74.2 | 41.0 | 49.7 | 63.6 | 57.4 | 51.5 | 56.2 |
| Qwen official GPTQ-Int4 (ppl 10.398) | 75.9 | 42.6 | 49.4 | 63.3 | 57.4 | 50.2 | 56.5 |

At 1.5B the perplexity ranking does show on tasks, modestly: Dom's deployable model is 0.5 points below fp16 on the
mean and 0.8 to 1.1 above the two official releases, with the gap concentrated in MMLU (+1.6) and GSM8K (+2.8 to +4.1),
the knowledge and reasoning tasks; the multiple-choice tasks are within noise. Together with the 32B table: his recipe's
advantage is real and measurable where quantization bites (small models, low bits) and vanishes where it does not.

### Mixed calibration on the 27B (2026-10-06/07)

Calibration text: 1.5 MB of WikiText-2 train followed by 1.5 MB of Python source from the vllm package (disjoint from the
transformers files of the code test corpus); 128 windows of 512 tokens drawn from it as before (`port/run_q38_mix.sh`).

| Qwen3.8-27B, 4 bits, integer offsets | WikiText-2 ppl (gap) | Python source ppl (gap) | kernel ppl | HumanEval |
|---|---|---|---|---|
| bf16 | 6.773 | 1.371 | | |
| WikiText-only calibration | 6.826 (+0.8%) | 1.437 (+4.8%) | 6.817 | 79.9% |
| mixed calibration | 6.848 (+1.1%) | 1.409 (+2.8%) | 6.843 | 79.3% |
| unsloth UD-Q4_K_M | 6.773 (+0.0%) | 1.390 (+1.4%) | | |
| **2 bits**, Dom's fp16 offsets, mixed calibration | 8.590 (+26.8%) | 2.234 (+63%) | not kernel-runnable | |

Mixing the calibration text costs 0.02 on WikiText and closes 40% of the code gap, with HumanEval unchanged within noise;
the packing stays lossless. The remaining distance to unsloth's model on code is the bit allocation (5-bit attention
projections, 8-bit gates, 3/4-bit feed-forwards against our uniform 4 bits), not the calibration. A per-layer bit budget
under Dom's layer metric is the natural next step. The 2-bit row (2026-10-07, m2, 7.6 h) is the floor of the uniform recipe on this model:
+27% on WikiText and +63% on code, so a usable 2-bit 27B would need the mixed-width allocation (unsloth's UD-Q2_K_XL keeps
attention and gates at 4-6 bits) rather than a uniform grid.

### Qwen2.5-1.5B comparators at group 32 (2026-10-07): bartowski's K-quants on the same protocol

The GGUF loader (`port/gguf_source.py`) now reads the architecture from the file (Qwen2 norms are stored as-is and the second
norm is `ffn_norm`; the Qwen3.5 +1 and `post_attention_norm` conventions apply only to that family). fp16 code perplexity added.

| Qwen2.5-1.5B-Instruct | size MB | WikiText-2 ppl (gap) | Python source ppl (gap) |
|---|---|---|---|
| fp16 | 3090 | 9.372 | 2.579 |
| Dom's recipe, fp16 offsets, 4 bits g64 (not kernel-runnable) | | 9.554 (+1.9%) | |
| Dom's recipe + integer offsets, 4 bits g64, AWQ (refactor regression 2026-10-07: 9.625 reproduced) | 1173 | 9.625 (+2.7%) | |
| bartowski Q2_K | 645 | 12.567 (+34%) | 3.846 (+49%) |
| Dom's recipe + integer offsets, uniform 2 bits g32, mixed calibration (m4) | ~800 | 21.862 (+133%) | 5.024 (+95%) |
| Dom's recipe + integer offsets, uniform 3 bits g32, mixed calibration (m3) | ~1120 | 10.600 (+13.1%) | 3.047 (+18.1%) |
| bartowski Q3_K_M | 786 | 10.365 (+10.6%) | 2.910 (+12.8%) |
| bartowski Q4_K_S | 897 | 9.698 (+3.5%) | 2.685 (+4.1%) |
| bartowski Q4_K_M | 941 | 9.659 (+3.1%) | 2.672 (+3.6%) |
| bartowski Q5_K_M | 1073 | 9.438 (+0.7%) | 2.611 (+1.2%) |
| Dom's recipe + integer offsets, 4 bits g32 (m4; GGUF Q4_1 pack 1233 MB scores identically) | 1233 | 9.562 (+2.0%) | 2.836 (+10.0%) |

Reading (sizes of our rows = codes + fp16 scale/offset per 32 + fp16 embeddings, as Q4_1/Q5_1 packing would store them).
At 3 and 2 bits the uniform affine grid loses to the K-quants on both corpora: Q3_K_M is 0.24 better on wiki at fewer
bytes, and Q2_K is not a uniform 2-bit model at all (4-bit scales per 16 weights inside 256-weight super-blocks, some
tensors kept at 4 bits), which the 3/2 mixes (running) are the fair comparison for. Our uniform 4-bit with integer offsets beats Q4_K_M on WikiText at 25% more bytes (fp16 scale + zero per 64 in AWQ
against the K-quant's 4.85 bits per weight); Q5_K_M at 1073 MB is the size-matched K-quant and is 0.19 better. The group-32
uniform 4/5-bit references and the mixed budgets (m4, m3) are the matched-size comparison.

### Column order by usage (2026-10-07, Mac, Dom's MLX code): negative at 4 bits

Three ways to exploit per-column usage (the diagonal of the Hessian) inside Dom's recipe, Qwen2.5-1.5B, 4 bits, group 64,
full recipe (`mlx-gptq/gptq_mlx_ord.py --usage-order`): act = GPTQ in descending-usage order with the groups kept in the
original column space (grids therefore fixed per group before error feedback; any format); perm = the same order with the
columns physically clustered into contiguous groups on the dimensions a checkpoint can permute (feed-forward intermediate
dimension per layer, value-head positions within each kv group, hidden dimension by one order shared across layers from a
pre-pass; rank correlation of per-linear usage with the shared order 0.54); permall = every linear in its own order (upper
bound, not deployable).

| Qwen2.5-1.5B, 4 bits g64 | WikiText-2 ppl | Python source ppl | codes changed by refinement |
|---|---|---|---|
| baseline (natural order) | 9.532 | 2.887 | 20.7% |
| act | 9.582 | 2.912 | 9.9% |
| perm | 9.575 | 2.901 | 4.2% |
| permall (upper bound) | 9.566 | 2.896 | 2.7% |

Reading: the act-order gain documented for plain GPTQ does not survive Dom's refinement + refit, which revisit every code in
the full layer metric after the fact (the ordered runs leave the refinement far fewer codes to move, and end worse). Clustering
by usage does not tighten per-row ranges. Closed at 4 bits; the 3-bit repeat is running.

### Column order by usage at 3 bits (2026-10-07, Mac): positive for the deployable clustering

Same three variants, Qwen2.5-1.5B, 3 bits, group 64, Dom's full recipe (fp16 offsets).

| Qwen2.5-1.5B, 3 bits g64 | WikiText-2 ppl | Python source ppl | codes changed by refinement |
|---|---|---|---|
| baseline (natural order) | 10.459 | 4.188 | 24.7% |
| act | 10.403 | 4.238 | 12.3% |
| perm (deployable: down_proj input, v-head positions per kv group, shared hidden order) | **10.318** | **4.097** | 5.3% |
| permall (upper bound) | 10.246 | 4.124 | 3.1% |

Reading: at 3 bits the clustering wins on both corpora (-1.3% wiki, -2.2% code; the deployable version captures 2/3 of the
upper bound's wiki gain and all of its code gain) where at 4 bits it lost 0.4%: the coarser the
grid, the more a group's (offset, step) suffers from mixing a hot column with cold ones. Usage ordering alone (act) is a wash.

### Per-layer bit mixes on the 1.5B vs the K-quants at matched bytes (2026-10-07)

Budgets from the sensitivity pass (`port/bit_budget.py`: error e^T H e / w^T H w per linear at each width, greedy by gain
per byte; targets are TOTAL bits per weight counting 1 bpw of fp16 scale+offset per 32). Integer offsets, group 32, full
recipe. Packed size = codes + scales + 8-bit embeddings/head.

| Qwen2.5-1.5B | packed MB | WikiText-2 ppl | Python source ppl |
|---|---|---|---|
| mix 4.5 bpw (3b 50% / 4b 50%), wiki calibration (m2) | ~970 | 9.991 | 3.397 |
| bartowski Q4_K_M | 941 | 9.659 | 2.672 |
| bartowski Q3_K_M | 786 | 10.365 | 2.910 |
| mix 3.25 bpw (2b 75% / 3b 25%), mixed calibration (m4) | ~770 | 19.277 | 4.599 |
| bartowski Q2_K | 645 | 12.567 | 3.846 |
| mix 3.5 bpw (2b 50% / 3b 50%), mixed calibration (m3) | ~810 | 15.085 | 4.004 |
| mix 3.75 bpw (2b 25% / 3b 75%), mixed calibration (m4) | ~850 | 12.913 | 3.480 |
| mix 5.0 bpw (3b 3% / 4b 94% / 5b 3%), wiki calibration (m2) | ~1050 | 9.586 | 2.882 |
| mix 5.5 bpw (3b 3% / 4b 43% / 5b 54%), wiki calibration (m2) | ~1135 | 9.472 | 2.745 |
| bartowski Q5_K_M | 1073 | 9.438 | 2.611 |

Reading: at matched bytes the mixes lose to the K-quants, and the cause is the packing, not the recipe: Q4_1/Q5_1-style
storage spends 1.0 bpw on fp16 scale+offset per 32 weights, so a "4.5 bpw" file carries 3.5 bits of codes against Q4_K_M's
4.5 (its super-block of 256 shares one fp16 scale and uses 6-bit sub-scales: 0.35 bpw overhead). That one bit is worth more
than any allocation or ordering gain. Next: fit Dom's grids under the K-quant constraint (sub-scale/sub-min quantized to 6
bits against one fp16 super-scale per 256), which packs into standard Q4_K/Q3_K/Q2_K types at their sizes.

### Dom's recipe under the K-quant constraint (2026-10-07): matched-size win over Q4_K_M on WikiText

`port/kquant.py` + `--kq q4_k` in `port/gptq_torch.py`: the 8 group grids of every 256-weight super-block are fitted from the
error-updated weights at the super-block start (Dom's MSE search + Lloyd fit), projected onto the Q4_K set (one fp16 super-scale
and super-min per 256, 6-bit integer sub-scale and sub-min per 32; the super-scale chosen by a 16-point search in the
diag(H)-weighted least squares) and fixed for the GPTQ pass; after every exact refit the grids are projected again and the
refinement re-picks the codes. The codes pack into standard GGUF Q4_K blocks (packer verified bit-exact against gguf-py's
dequantizer; the packed file reproduces the HF checkpoint's perplexity within the 8-bit embedding rounding).

| Qwen2.5-1.5B | packed MB | WikiText-2 ppl (gap) | Python source ppl (gap) |
|---|---|---|---|
| fp16 | 3090 | 9.372 | 2.579 |
| **Dom's recipe, Q4_K-constrained, wiki calibration, packed GGUF Q4_K + Q8_0 embeddings (m3)** | **946** | **9.514 (+1.5%)** | 2.797 (+8.5%) |
| **Dom's recipe, Q4_K-constrained, MIXED calibration (m3)** | **946** | **9.583 (+2.3%)** | **2.662 (+3.2%)** |
| **the same, packed with Q6_K embeddings as the reference does (m3)** | **892** | **9.602 (+2.5%)** | **2.665 (+3.3%)** |
| bartowski Q4_K_M (Q4_K/Q6_K mix, imatrix) | 941 | 9.659 (+3.1%) | 2.672 (+3.6%) |
| Dom's recipe + integer offsets, uniform 4 bits g32, Q4_1 packing | 1233 | 9.562 (+2.0%) | 2.836 (+10.0%) |
| bartowski Q5_K_M | 1073 | 9.438 (+0.7%) | 2.611 (+1.2%) |
| Dom's recipe, Q5_K-constrained, mixed calibration, packed Q5_K + Q8_0 embeddings (m3) | ~1135 | 9.425 (+0.6%) | 2.601 (+0.9%) |
| the same, packed with Q6_K embeddings (m3) | 1048 | 9.441 (+0.7%) | 2.604 (+1.0%) |

Reading: at the same file size (946 vs 941 MB) the constrained recipe halves Q4_K_M's WikiText gap (1.5% vs 3.1%); it also beats
our own Q4_1 packing at 23% fewer bytes, because the K-quant form keeps a fine (super-min-scaled) offset where the int4 kernels
force an integer zero point. With the mixed calibration and the reference's own Q6_K embeddings, a SMALLER file (892 vs 941 MB) beats Q4_K_M on BOTH
corpora (9.602 vs 9.659 wiki, 2.665 vs 2.672 code):
the first matched-size win over the llama.cpp importance-matrix K-quants, with Dom's recipe doing the fitting. Next: the 27B under
the same constraint against unsloth's UD-Q4_K_M (m1, after the running 4.5-bpw mix), and Q5_K on the 1.5B against Q5_K_M. Q3_K (16-weight sub-blocks, 3.44 bpw) is being debugged on m4 (a later layer's Hessian
goes non-positive-definite: hidden states overflow fp16 after a badly quantized layer; the one-layer smoke shows the GPTQ-stage
grids far from the refit, 79% of codes changed).

### Q3_K-constrained (2026-10-07, m4): the symmetric 3-bit form costs more than it saves

| Qwen2.5-1.5B, mixed calibration | packed MB | WikiText-2 ppl | Python source ppl |
|---|---|---|---|
| Dom's recipe, Q3_K-constrained (16-weight sub-blocks, fixed zero 4, 5-bit effective sub-scales), packed Q3_K | ~800 | 10.812 | 3.084 |
| Dom's recipe + integer offsets, uniform 3 bits g32, Q4_1-style exact packing | ~1120 | 10.600 | 3.047 |
| bartowski Q3_K_M (Q3_K with attn_v / part of ffn_down at Q4_K-Q5_K) | 786 | 10.365 | 2.910 |
| Dom's recipe, Q3_K-constrained + usage clustering (--order perm, shared hidden order from the mixed text) | 726 (packed: 10.695 / 3.073) | 10.646 | 3.070 |
| Dom's recipe, per-linear Q3_K/Q4_K map at 3.6 bpw (sensitivity-ranked, `--kq-map`; m3) | ~780 | 10.563 | 3.021 |
| **map 3.8 bpw (Q3_K 65% / Q4_K 35%) + usage clustering (m4)** | **784** (packed: 10.254 / 2.946) | **10.209** | 2.943 |
| map 3.8 bpw plain (m3) | ~815 | 10.374 | 2.959 |
| map 3.6 bpw + usage clustering (m4) | 752 (packed: 10.474 / 3.011) | 10.432 | 3.008 |
| map 4.0 bpw (Q3_K 47% / Q4_K 53%) plain (m3) | ~846 | 10.123 | 2.846 |
| **map 4.0 bpw + usage clustering (m3)** | **814** (packed, Q6_K embeddings) | **9.951** | **2.826** |
| the same as the packed GGUF file (permutation-aware pack: gate/up rows, v rows, embeddings/norms/output rows reordered to match) | 814 | 9.994 | 2.829 |
| 3-form map q2_k 18% / q3_k 52% / q4_k 30% at 3.6 bpw + clustering (m4) | ~780 | 10.590 | 3.086 |
| 3-form map q2_k 26% / q3_k 57% / q4_k 17% at 3.4 bpw + clustering (m4) | ~750 | 10.890 | 3.179 |

Reading: Q3_K's symmetric 8-level grid (q - 4) loses ~25% in layer error against a free offset at the grid-fit stage (0.196 vs
0.159 relative on q_proj of layer 0), and Q3_K_M keeps the sensitive tensors at 4-5 bits. Beating it needs a per-layer choice of
K-quant kind (Q3_K / Q4_K by the sensitivity table), i.e. the bit-budget allocator applied to K-quant forms: `--kq-map`, next.
Reading of the maps (packed sizes in MiB, Q6_K embeddings): at the reference's own size the clustered 3.8-bpw map (784 MiB) is
1% better than Q3_K_M (786 MiB) on WikiText (10.254 vs 10.365 as packed files) and 1% worse on code (2.946 vs 2.910): a tie at 3 bits;
the 4.0-bpw clustered map (814 MiB) is under the Q3_K_M-Q4_K_M line on both corpora. Adding Q2_K as a third form is negative (3.6 bpw: 10.59 vs 10.43): the sensitivity proxy (no error
feedback) under-estimates the Q2_K damage. A bug found on the way: the fixed-zero refit (`refit_int`) used the candidate index as the zero value; fixed (also affects any
`--sym8` 8-bit layer refit; no reported number used one).

Reference file anatomy (gguf-py, 2026-10-07): bartowski 1.5B Q3_K_M / Q4_K_M / Q5_K_M all store token_embd as Q6_K (191 MB,
tied output); Q4_K_M blocks = 168 Q4_K + 28 Q6_K tensors (788 MB), Q5_K_M = 168 Q5_K + 28 Q6_K (927 MB), Q3_K_M = 112 Q3_K + 81
Q4_K + 3 Q5_K (626 MB). unsloth Qwen3.8-27B-UD-Q4_K_M: 131 Q5_K (4.8 GB) + 117 IQ4_XS (4.8 GB) + 103 Q4_K (3.5 GB) + 29 Q6_K +
7 Q3_K + 7 IQ4_NL + 4 IQ3_S + 106 Q8_0 small tensors; token_embd Q4_K (715 MB), output Q6_K; blocks 14.7 GB of 16.5. Our rows
above use Q8_0 embeddings (234 MB at 1.5B); with Q6_K embeddings the Q4_K file is ~905 MB and the Q5_K file ~1090 MB.

Note (2026-10-07): the 27B 4.5-bpw Q3/Q4 exact-packing mix on m1 quantized all 64 layers (10.5 h) but its export hit a full
disk (348 MB free); the codes are kept (`models/q38_g32_mix4.5/codes`), the dequantized checkpoint was not written and the model
is unscored. Given the 1.5B results (exact packing loses at matched bytes) it was superseded by the Q4_K-constrained 27B run.

Sizes in this section are MiB from `ls -l --block-size=M` (the reference Q4_K_M is 941 MiB = 986.0 MB decimal). Packing a
usage-clustered model needs the permutations applied to the neighbours (`codes2gguf.py` reads the stored `.perm` per linear:
gate/up rows follow down_proj's order, v rows follow o_proj's per-kv-group order, embeddings columns / norms / o_proj and
down_proj rows / output follow the shared hidden order); verified on the 4.0 map (packed 9.994 vs unpacked 9.951, the
difference being the Q6_K embeddings; a naive pack scored 7e7).

### KL divergence and top-1 agreement vs fp16 (2026-10-07, m3): unsloth's own metric family

`port/kld_stream.py`: same windows as the perplexity protocol (first 8 of 20), reference log-probs kept in fp16, KL(ref || cand)
averaged over tokens, top-1 agreement, and both perplexities (these 8-window perplexities differ from the 20-window ones above).
unsloth's Dynamic 3.0 write-up (unsloth.ai/docs/basics/dynamic-3.0-ggufs, read 2026-10-07) ranks quantizations by KL divergence
and top-1 agreement rather than perplexity, with a calibration set built for agentic coding / chat / multilingual text; its first
release is the Qwen3.8-27B family we compare against (the UD-Q4_K_M file here is its 4-bit member).

| Qwen2.5-1.5B (8 windows) | MiB | wiki KL | wiki top-1 | wiki ppl | code KL | code top-1 | code ppl |
|---|---|---|---|---|---|---|---|
| fp16 | 3090 | 0 | 100% | 8.452 | 0 | 100% | 2.804 |
| Dom's recipe, Q4_K-constrained, mixed calibration | 892 | 0.0313 | 90.95% | 8.675 | 0.0388 | 94.09% | 2.880 |
| bartowski Q4_K_M | 941 | 0.0282 | 91.70% | 8.722 | 0.0381 | 94.03% | 2.888 |
| Dom's recipe, Q5_K-constrained, mixed calibration | 1048 | 0.0110 | 94.11% | 8.511 | 0.0120 | 96.77% | 2.825 |
| bartowski Q5_K_M | 1073 | 0.0093 | 94.47% | 8.486 | 0.0105 | 96.73% | 2.838 |
| Dom's recipe, Q3_K/Q4_K map 3.8 bpw + clustering | 784 | 0.1148 | 84.06% | 9.319 | 0.1427 | 88.97% | 3.094 |
| bartowski Q3_K_M | 786 | 0.1063 | 84.34% | 9.420 | 0.1332 | 89.02% | 3.097 |
| Dom's recipe, map 3.8 bpw + clustering, ridge 0.05 (m4) | 784 | 0.1132 | 84.30% | 9.347 | 0.1408 | 89.15% | 3.089 |

Reading: the pattern holds at every size: on perplexity Dom's recipe wins or ties at equal or fewer bytes (4-bit: 8.675 vs 8.722;
5-bit: tie; 3.8 bpw: 9.319 vs 9.420), on KL / top-1 the imatrix K-quant is slightly ahead (4-bit wiki 0.028 vs 0.031; 5-bit 0.0093 vs
0.0110; 3-bit 0.106 vs 0.115), with code a tie throughout. Dom's layer-metric objective lowers the loss on the realised token; the imatrix
K-quant keeps the full distribution closer. For the 27B both metric families are reported.

### Does the refinement trade KL for perplexity? (2026-10-07, m4): yes

Q4_K-constrained, mixed calibration, Qwen2.5-1.5B, with and without Dom's refinement + refit (8-window KL / top-1, 20-window ppl).

| | wiki ppl | wiki KL | wiki top-1 | code ppl | code KL | code top-1 |
|---|---|---|---|---|---|---|
| no refinement (super-block GPTQ only) | 9.648 | 0.0294 | 91.73% | 2.668 | 0.0388 | 93.91% |
| **refine 1 / refit 1 (light)** | **9.581** | **0.0276** | **92.11%** | **2.659** | **0.0378** | **94.46%** |
| **refine 3 / refit 2 with refine-damp 0.05 (ridged)** | **9.562** | **0.0261** | **92.24%** | 2.666 | **0.0354** | 94.34% |
| refine 1 / refit 1 with refine-damp 0.05 | 9.583 | 0.0266 | 91.85% | 2.670 | 0.0359 | 94.20% |
| refine 3 / refit 2 | 9.583 | 0.0313 | 90.95% | 2.662 | 0.0388 | 94.09% |
| bartowski Q4_K_M (941 MiB vs our 892) | 9.659 | 0.0282 | 91.70% | 2.672 | 0.0381 | 94.03% |

Reading (dequantized checkpoints vs bartowski's file; packed with Q6_K embeddings our files are +0.0037 KL on this tied model, see
the KL fine-tune section): the unrefined checkpoint already matches Q4_K_M on KL / top-1 and ties its perplexity at 5% fewer bytes; the refinement buys
-0.7% wiki perplexity for +7% wiki KL and -0.8 points of top-1 agreement (code unchanged). The coordinate descent in the layer
metric over-fits the calibration objective: it sharpens the realised tokens at the expense of the rest of the distribution.
The light setting (1 sweep / 1 refit round) keeps all of the perplexity gain AND improves KL / top-1 over the unrefined model:
it beats Q4_K_M on every metric on both corpora at 5% fewer bytes. The extra sweeps are the over-fit. Recommended recipe from here:
`--refine 1 --refit 1`. The ridged variant (full sweeps, refine-damp 0.05) is better still on wiki ppl, KL on both corpora and wiki top-1: a stronger
ridge on the normalised Hessian in the refinement objective removes the over-fit while keeping Dom's full recipe. Recommended:
`--refine 3 --refit 2 --refine-damp 0.05` (or `--refine 1 --refit 1` for speed). The 27B run in flight uses Dom's default
ridge 0.01; a ridged 27B is queued after it.

### Qwen3.8-27B under the Q4_K constraint (2026-10-07, m1): level with unsloth's Dynamic 3.0 file

Dom's full recipe (refine 3 / refit 2, ridge 0.01) fitted under the Q4_K constraint, group 32, mixed calibration, 64 layers at
8.3 min each (8.8 h), exact (streamed bf16) evaluation of the quantized weights:

| Qwen3.8-27B | GB | WikiText-2 ppl (gap) | Python source ppl (gap) |
|---|---|---|---|
| bf16 | 54 | 6.773 | 1.371 |
| **Dom's recipe, Q4_K-constrained, mixed calibration** (exact, streamed) | | **6.800 (+0.4%)** | **1.394 (+1.7%)** |
| the same packed as GGUF Q4_K + Q8_0 embeddings/head (851 tensors) | 15,783 MiB | 6.804 | |
| the same packed as GGUF Q4_K + Q6_K embeddings/head | 15,195 MiB | 6.807 | 1.394 |
| unsloth UD-Q4_K_M (Dynamic 3.0; Q4_K embeddings, Q6_K head) | 15,702 MiB | 6.773 (+0.0%) | 1.390 (+1.4%) |
| Dom's recipe + integer offsets, uniform 4-bit AWQ, mixed calibration | 19.2 | 6.848 (+1.1%) | 1.409 (+2.8%) |

Reading: the K-quant form closes most of the gap to unsloth at a smaller file (6.848 -> 6.800 wiki, 1.409 -> 1.394 code); what
remains is 0.4% on each corpus, with unsloth's wiki number carrying its in-domain advantage. The ridged refinement (refine-damp
0.05, the 1.5B's best setting) is running on the 27B overnight. Packed: 15,783 MiB vs unsloth's 15,702 (ours carries Q8_0 embeddings + head, 2.5 GB; a Q6_K repack is queued and should land
near 15.1 GiB). The first pack hit a non-contiguous dtype view and then the 30 GB RAM limit (the packer held all 26 GB of codes);
both fixed (contiguous copies after the permutations; lazy code loading).

KL divergence / top-1 agreement vs bf16 (6 windows of 2048 tokens, `kld_stream.py`, same evaluator as the 1.5B rows):

| Qwen3.8-27B GGUF | MiB | wiki KL | wiki top-1 | wiki ppl | code KL | code top-1 | code ppl |
|---|---|---|---|---|---|---|---|
| ours Q4_K-constrained, Q8_0 embeddings/head | 15,783 | 0.01571 | 95.24% | 6.334 | 0.02539 | 97.80% | 1.292 |
| ours Q4_K-constrained, Q6_K embeddings/head | 15,195 | 0.01597 | 94.91% | 6.335 | | | |
| **unsloth UD-Q4_K_M (Dynamic 3.0)** | 15,702 | **0.00797** | **95.98%** | 6.278 | 0.01857 | 98.14% | 1.288 |
| **ours, budget map (q4_k 72% / q5_k 28%) + ridge 0.05, Q4_K embeddings / Q6_K head** | 15,705 | 0.01301 | 95.16% | 6.283 | **0.01687** | **98.23%** | 1.283 |

(ppl here on the 6 KL windows; bf16 6.237 / 1.271.) Reading: at matched perplexity (+0.4% on the 20-window protocol) unsloth's file has
HALF our KL to bf16 on wiki and 27% less on code: the per-linear Hessian objective buys perplexity on the realised tokens and pays in the
rest of the distribution, the same signature as the 1.5B refinement study, and the gap is larger at 27B. The Q6_K embeddings cost
0.0003 KL for 588 MiB. The levers that fixed it on the 1.5B (ridged refinement, chat-template calibration, KL fine-tune of the
super-scales, each a 7-50% KL reduction) are the ones queued on the 27B, plus the size headroom (507 MiB at Q6_K embeddings,
~830 MiB with Q4_K embeddings like unsloth's) going to Q5_K / Q6_K on the most sensitive linears.
2026-10-08 09:xx UTC: the ridged 27B run (refine-damp 0.05, flat Q4_K) died at layer 48 of 64 when m1's disk filled (its
dequantized shards + codes need 80 GB; 54 GB were free). Rather than rerun it alone, the ridge is folded into the bit-allocation run
(budget under q4_k / q5_k / q6_k to 4.783 bpw over the linears = unsloth's file size with Q4_K embeddings and Q6_K head), so one
8.5 h run carries both. The sensitivity pass (80 min, no error feedback, forms q4_k / q5_k / q6_k) and the greedy budget to 4.783 bpw
give: q4_k 71.7%, q5_k 28.2%, q6_k 0.1% of the 24.3B linear weights. What it promotes (count of linears at Q5_K or above): v_proj 14/16,
o_proj 9/16, k_proj 7/16 of the full-attention layers (two v_proj and one k_proj to Q6_K), q_proj none; in the linear-attention layers
out_proj 26/48 and in_proj_qkv 23/48, the gate in_proj_z 3/48; in the MLPs up_proj 37/64, down_proj 11/64, gate_proj 1/64. That is
unsloth's own pattern (attention and value paths up, gates and feed-forward down), found here by the layer metric.

Result (2026-10-08, 8.9 h quantize + 35 min evaluate/pack): exact 6.802 wiki / **1.387 code** (flat Q4_K 6.800 / 1.394; unsloth
6.773 / 1.390), packed **15,705 MiB** against unsloth's 15,702 (6.803 as a file). KL / top-1 vs bf16 in the table above: wiki
0.0130 / 95.16% (flat 0.0160 / 94.9%: -19%), code **0.0169 / 98.23%**, now ahead of unsloth's 0.0186 / 98.14%. On wiki unsloth is
still 1.6x closer on KL at a tie in perplexity (6.283 vs 6.278 on the KL windows); the two 1.5B levers not yet applied to the 27B
are the v4 calibration (chat / multilingual text) and the KL fine-tune of the super-scales, in that order.

v4 calibration on the 27B (2026-10-09, same map and ridge, 256 windows, 9.3 h): exact 6.835 wiki / 1.389 code (narrow mix: 6.802 /
1.387), packed 15,705 MiB. Files vs bf16 on the five held-outs (6 windows of 2048; KL / top-1):

| Qwen3.8-27B GGUF | MiB | wiki | code | chat (GSM8K/MMLU)* | UltraChat* | multilingual |
|---|---|---|---|---|---|---|
| ours, map + ridge, narrow calibration (wiki + Python) | 15,705 | 0.0130 / 95.16% | **0.0169 / 98.23%** | 0.129 / 91.74% | 0.094 / 90.19% | 0.0156 / 94.73% |
| **ours, map + ridge, v4 calibration** | 15,705 | 0.0123 / 95.29% | 0.0184 / 98.17% | **0.102 / 91.87%** | **0.041 / 92.91%** | 0.0116 / 95.47% |
| ours, map + ridge, narrow calibration + KL fine-tune on the v4 text (100 x 16, m3) | 15,705 | 0.0123 / 95.29% | 0.0167 / 98.12% | 0.169 / 88.91% | 0.156 / 84.40% | 0.0153 / 94.67% |
| **ours, map + ridge, v4 calibration + KL fine-tune (100 x 16 windows, 2 h on m1)** | 15,705 | **0.0122 / 95.34%** | **0.0173 / 98.16%** | 0.178 / 87.93% | 0.097 / 87.09% | **0.0114 / 95.51%** |
| ours, map + ridge, v4 calibration, 1024 windows instead of 256 (m3, 9.6 h) | 15,705 | 0.0118 / 95.38% | **0.0161 / 98.23%** | | | 0.0111 / 95.70% |
| ours, map + ridge, bartowski's public calibration text (calibration_datav3, 128 windows, m4) | 15,705 | 0.0151 / 94.92% | 0.0293 / 97.58% | | | 0.0174 / 93.48% |
| unsloth UD-Q4_K_M | 15,702 | **0.0080 / 95.98%** | 0.0186 / 98.14% | 0.119 / 92.09% | 0.059 / 91.74% | **0.0078 / 96.38%** |

*bf16's own perplexity on these two sets is 527 (chat) and 18.0 (UltraChat) for this model, against 5.8 / 4.3 for the 1.5B: the
GSM8K / MMLU answers and UltraChat turns rendered through Qwen3.8's chat template are text this model finds implausible (likely
the thinking-mode layout), so those two columns are relative comparisons only. Reading: the broad calibration halves UltraChat KL
(0.094 -> 0.041), trims chat, wiki (-5%) and multilingual (-26%), costs 9% on code; against unsloth the v4 file is ahead on code,
chat and UltraChat and behind on wiki (1.5x) and multilingual (1.5x). The KL fine-tune of the v4 file (layer-streamed, 100 steps x 16 windows, 71 s/step with the packed
codes in m1's page cache, ~2 h + 1 h to pack and score) moves wiki -1%, code -6%, multilingual -1% and packed wiki ppl 6.835 ->
6.800: real but a third of what it gave the 1.5B and SmolLM2, on a third of their token budget (0.8M vs 2.5M) at the same lr; a
longer schedule is the obvious next test. On the two chat held-outs it goes the other way (0.102 -> 0.178, 0.041 -> 0.097): the
tuned model becomes much MORE confident on that text than bf16 (UltraChat ppl 11.5 vs bf16's 18.0; chat 71 vs 527), so its KL to the
reference grows. That is the template problem again: bf16 is itself diffuse on chat rendered through Qwen3.8's template, the
fine-tune (whose calibration text includes such chat) sharpens the student there, and KL(bf16 || student) charges for the
sharpening. Those two columns do not measure quality for this model; a chat held-out rendered in the model's thinking format
would. Best overall against unsloth on the trustworthy columns: the v4 + fine-tune file is ahead on code and within 1.5x on wiki and

Four times the calibration windows (1024 x 512 tokens, same v4 mix) is the cheapest gain found on the 27B: wiki -4%, code -13%, multilingual -4% against the 256-window file at the same bytes, with nothing else changed. The 300-step fine-tune running on m2 already uses 1024 windows.
multilingual; unsloth keeps wiki and multilingual. The narrow file's fine-tune (m3, 6 min/step disk-bound) packs to wiki ppl
6.767 as a file (below bf16's 6.773: the fine-tune text sharpens it on wiki-like prose); its KL runs are on m3 in 2-window pieces.
Packer fixes on the way: the embedding / head are now cast per row chunk, and `--temp-file` streams the tensors through a temp
file (gguf-py buffers the whole model in RAM until the final write; 16 GB for a 27B, fatal on a 15 GB box).

### Calibration mix v2: chat-template data (2026-10-07/08, m3): chat KL halved

The model is an Instruct model, and nothing in the calibration stream (WikiText + Python) contained a single chat token. Mix v2
adds 1.6 MB of GSM8K-train and MMLU dev/validation items rendered through the model's own chat template (system prompt, `<|im_start|>`
turns, worked answers). Held-out chat set: GSM8K test + MMLU test, disjoint items. Ridged Q4_K recipe on Qwen2.5-1.5B, 8-window KL / top-1:

| calibration | wiki KL | wiki top-1 | code KL | code top-1 | chat KL | chat top-1 | UltraChat KL | UltraChat top-1 | multilingual KL | multilingual top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (wiki + Python; damp05 on m4, reproduced bit-for-bit by a rerun on m3) | 0.0261 | 92.24% | 0.0354 | 94.34% | 0.0490 | 92.79% | 0.0386 | 91.41% | 0.0497 | 89.46% |
| **v2 (v1 + chat-template GSM8K / MMLU)** | 0.0271 | 92.07% | 0.0367 | 94.14% | **0.0242** | **94.22%** | 0.0294 | 92.28% | 0.0533 | 88.56% |
| **v3 (v2 + UltraChat 1.5 MB + Wikipedia in 8 languages 1.5 MB)** | 0.0260 | 92.02% | 0.0414 | 94.14% | 0.0270 | 93.88% | **0.0228** | **93.40%** | **0.0324** | **91.37%** |
| v3 with 256 calibration windows instead of 128 | 0.0256 | 92.13% | 0.0402 | 94.05% | 0.0261 | 94.10% | 0.0221 | 93.53% | 0.0305 | 91.85% |
| v3 with the Python slice weighted twice (v3w), 128 windows | 0.0257 | 92.01% | 0.0399 | 94.25% | 0.0263 | 93.83% | 0.0241 | 92.93% | 0.0319 | 91.44% |
| **v4 = v3w with 256 windows** | **0.0249** | **92.55%** | **0.0383** | **94.46%** | 0.0261 | 94.00% | 0.0234 | 92.82% | 0.0310 | 91.77% |
| bartowski Q4_K_M, a FILE (941 MiB; imatrix on a broad chat/multilingual/code mix) | 0.0282 | 91.70% | 0.0381 | 94.03% | 0.0334 | 93.84% | 0.0255 | 92.61% | 0.0317 | 91.15% |

Reading: chat KL halves (0.049 -> 0.024) and chat top-1 gains 1.4 points for +0.001 KL on wiki and code. Before this change the
quantized model was at its worst exactly where an Instruct model is used. On real multi-turn chat the model never saw (UltraChat test)
v2 is 24% lower KL; on multilingual Wikipedia (8 languages, none in either calibration) it is 7% worse, the one place v1's plain prose
helped. v3 (7.6 MB, same 128 x 512 windows) fixes that: multilingual KL -35% vs v1, UltraChat -41%, wiki back to v1's level, chat
between v1 and v2; the one cost is code, +17% KL (0.0354 -> 0.0414; ppl 2.666 -> 2.684), because Python fell from half of the
mix to a fifth and 128 windows now cover five domains. Two follow-ups are running on m3: v3 with 256 windows, and v3 with the
Python slice weighted twice. The v2 and v3 files pack to 892 MiB like v1. bartowski's row explains unsloth's advantage: its imatrix
calibration is already broad, so it sits at our v3 level on chat / UltraChat / multilingual, 30-40% below our v1 there, and ahead
on code; as dequantized checkpoints v3 only ties it, and the Q6_K tied embedding costs our files another 0.004. The fine-tune on the
v3 text (running) is what has to carry the file past it. 256 windows instead of 128 on v3 helps on every held-out (1-6%; code
0.0414 -> 0.0402, ppl 2.684 -> 2.671): with five domains in the mix the Hessians want more samples; use 256 from here. Weighting Python twice (v3w) trims code KL by 4%
(0.0414 -> 0.0399, ppl unchanged at 2.684) at no cost elsewhere; v4 = both: the best broad calibration on wiki and code (code KL
0.0383, level with bartowski's 0.0381; code ppl 2.663 = v1's) while keeping v3's chat / UltraChat / multilingual gains. v4 is the
calibration from here; its KL fine-tune is the final 1.5B candidate. (A v3 with real
multi-turn chat (UltraChat) and multilingual Wikipedia, with two further held-out sets, is running; fp16 reference ppl on those:
UltraChat 4.276, multilingual 7.650.)

### KL fine-tune of the continuous parameters (2026-10-08, m4): every metric improves

`kl_tune.py`: codes fixed; the fp16 super-scales d / dmin of every Q4_K linear (10.2M values, trained as log-multipliers through the
fp16 rounding, straight-through), the RMSNorm weights and the attention biases (145k values) are trained to minimise KL(fp16 || quantized)
on calibration windows (512 tokens, 16 per step with accumulation, 300 steps, Adam lr 1e-4 cosine, norms 1e-5; 32 min). Everything
trained is stored as-is in the GGUF, so the file is exactly the tuned model. Start: the ridged Q4_K 1.5B (damp05).

| Qwen2.5-1.5B Q4_K-constrained | wiki ppl | wiki KL | wiki top-1 | code ppl | code KL | code top-1 | chat KL | chat top-1 |
|---|---|---|---|---|---|---|---|---|
| ridged (start) | 9.562 | 0.0261 | 92.24% | 2.666 | 0.0354 | 94.34% | 0.0490 | 92.79% |
| **+ KL fine-tune (tune_a)** | **9.502** | **0.0233** | **92.51%** | **2.660** | **0.0328** | **94.57%** | **0.0457** | **92.95%** |
| + KL fine-tune, one multiplier per output row instead of per super-block (tune_b, lr 3e-4) | 9.514 | 0.0244 | 92.40% | 2.664 | 0.0343 | 94.50% | 0.0447 | 92.79% |
| + KL fine-tune, lr 3e-5 (tune_c) | 9.504 | 0.0237 | 92.41% | 2.661 | 0.0334 | 94.53% | 0.0472 | 92.90% |
| **+ KL fine-tune on the v2 (chat-template) mix (tune_d)** | 9.503 | 0.0244 | 92.42% | 2.660 | 0.0338 | 94.55% | **0.0294** | **93.67%** |
| + KL fine-tune with the user's Packed BlockMax instead of Adam, same lr (tune_e) | 9.504 | 0.0238 | 92.42% | 2.870* | 0.0335 | 94.58% | 0.0472 | 92.93% |
| + KL fine-tune, Packed BlockMax, per-row multipliers, lr 3e-4 (tune_f) | 9.506 | 0.0240 | 92.46% | 2.871* | 0.0337 | 94.52% | 0.0462 | 92.94% |
| bartowski Q4_K_M (941 MiB vs our 892) | 9.659 | 0.0282 | 91.70% | 2.672 | 0.0381 | 94.03% | | |

Reading: -11% wiki KL, -7% code KL, -7% chat KL, better top-1 and perplexity everywhere, at the same bytes; the file is now ahead of
the imatrix Q4_K_M on every metric including KL (0.0233 vs 0.0282 wiki), which the refinement alone did not achieve.
First attempt (negative, kept for the record): Adam lr 5e-4, no accumulation, 2k tokens per step: held-out KL rose from step one
(0.0343 -> 0.039), the exported model had better perplexity (9.502) but worse KL (0.0300) - the same over-fit signature as the
refinement. With 10M scale parameters and sign-like per-parameter updates, every step moved every fp16 scale one ulp in a noisy
direction. Larger effective batch (4x) and 5-10x lower learning rates fixed it; the run keeps the best held-out state (held-out =
the last 4 x 1024 tokens of the wiki test, disjoint from the 8 evaluation windows). Variants running: per-row multipliers, lr 3e-5,
v2 calibration, and the user's Packed BlockMax in place of Adam (optimizer state is not the memory here: 80 MB vs the two models
and the 152k-vocabulary logits). Per-row multipliers (tune_b) and lr 3e-5 (tune_c) land within 0.001 of tune_a: the gain is
robust to those choices. Packed BlockMax (fused kernel, momentum in the low bits of the fp32 parameters) in place of Adam at the
same lr (tune_e) lands within 0.0005 of Adam on every metric: a drop-in here (*code ppl on the 8 KL windows; 20-window code ppl
pending). Fine-tuning on the chat-template mix instead (tune_d) keeps wiki / code within 0.001 of tune_a and takes
chat KL from 0.049 to 0.029 (-40%), without re-quantizing: the training text of the fine-tune steers the distribution where the
calibration of the codes did not.
Packed-file caveat: the tune_a codes packed as GGUF with Q6_K embeddings + head score wiki KL 0.0267 / 91.46% against 0.0233 /
92.51% for the dequantized checkpoint (ppl 8.590 vs 8.581 on the same windows). The 1.5B ties its head to the embeddings, so the
Q6_K rounding of that one tensor moves every logit. Attribution: the same codes packed with F16 embeddings score 0.02329 / 92.51%,
identical to the dequantized checkpoint to all printed digits, so the export and the pack are exact and the whole 0.0034 is the
Q6_K embedding/head. Files on the same evaluator (wiki, 8 windows):

| Qwen2.5-1.5B GGUF file | MiB | wiki KL | wiki top-1 |
|---|---|---|---|
| ridged, untuned, Q6_K embeddings (= the handover file q15_kq4_mix_e6's recipe) | 892 | 0.0298 | 91.07% |
| bartowski Q4_K_M | 941 | 0.0282 | 91.70% |
| **ridged + KL fine-tune, Q6_K embeddings** | **892** | **0.0267** | 91.46% |
| **ridged + KL fine-tune, Q8_0 embeddings** | **946** | **0.0235** | **92.61%** |
| ridged + KL fine-tune, F16 embeddings | 1,155 | 0.0233 | 92.51% |

Reading: on the 1.5B the tied embedding/head at Q6_K costs 0.0037 KL whether tuned or not, enough that the UNTUNED Q6_K file is
behind bartowski's Q4_K_M on KL (0.0298 vs 0.0282) even though it is ahead on perplexity; the earlier dequantized-vs-file rows
flattered us by that amount. The fine-tuned file is ahead of Q4_K_M at both sizes: 0.0267 at 49 MiB less, 0.0235 at 5 MiB more.
Ship Q8_0 embeddings for tied small models (54 MiB over Q6_K, 0.0002 over F16); for the untied 27B the Q6_K head cost 0.0003.

Layer-streamed version for models larger than the card (`kl_tune_stream.py`, for the 27B): the teacher is streamed once and its
top-256 log-probs per token cached (KL computed on the top-256 + the tail mass); each pass streams the student layer by layer,
forward without gradients storing every layer's input, then backward with recompute, one layer's codes on the card at a time;
embeddings / head come from the packed GGUF so the student is the deployed model. Validated on the 1.5B (v4 codes, 200 steps x 8
windows, Q8_0 file): wiki 0.0236 / 92.73%, code 0.0360, chat 0.0234, UltraChat 0.0216, multilingual 0.0299, against 0.0229 /
0.0353 / 0.0220 / 0.0205 / 0.0288 for the in-memory `kl_tune.py` with 3x the tokens (untuned 0.0249): the top-K objective and
the streaming reproduce the gain; the small gap is the shorter schedule.

### The combination on the 1.5B (2026-10-08, m3 + m4): v3 calibration + KL fine-tune on the v3 text

Ridged Q4_K codes from the v3 calibration (m3), then `kl_tune.py` on the v3 text (300 steps, Adam lr 1e-4, accum 4; held-out wiki
KL 0.0362 -> 0.0326 during training). Dequantized checkpoints unless marked FILE; 8-window KL / top-1; 20-window ppl.

| Qwen2.5-1.5B | MiB | wiki ppl | code ppl | wiki KL | wiki top-1 | code KL | code top-1 | chat KL | chat top-1 | UltraChat KL | UltraChat top-1 | multilingual KL | multilingual top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 calibration, ridged (the handover recipe) | | 9.562 | 2.666 | 0.0261 | 92.24% | 0.0354 | 94.34% | 0.0490 | 92.79% | 0.0386 | 91.41% | 0.0497 | 89.46% |
| v3 calibration, ridged | | 9.573 | 2.684 | 0.0260 | 92.02% | 0.0414 | 94.14% | 0.0270 | 93.88% | 0.0228 | 93.40% | 0.0324 | 91.37% |
| **v3 + KL fine-tune on v3** | | **9.532** | 2.672 | **0.0240** | **92.32%** | 0.0393 | **94.14%** | **0.0231** | **94.19%** | **0.0199** | **93.78%** | **0.0296** | **91.87%** |
| v4 calibration (v3w + 256 windows), ridged | | 9.591 | 2.663 | 0.0249 | 92.55% | 0.0383 | 94.46% | 0.0261 | 94.00% | 0.0234 | 92.82% | 0.0310 | 91.77% |
| **v4 + KL fine-tune on v4 (the final candidate)** | | **9.524** | **2.651** | **0.0227** | **92.93%** | **0.0352** | **94.71%** | **0.0220** | **94.31%** | 0.0205 | 93.17% | **0.0286** | **92.12%** |
| **v4 + fine-tune, FILE, Q8_0 embeddings** | 946 | | | **0.0229** | **92.82%** | **0.0353** | **94.69%** | **0.0220** | **94.38%** | **0.0205** | **93.24%** | **0.0288** | **91.98%** |
| v4 + fine-tune, FILE, Q6_K embeddings | 892 | | | 0.0261 | 91.81% | | | | | | | | |
| **v3 + fine-tune, FILE, Q8_0 embeddings** | 946 | | | **0.0242** | **92.32%** | 0.0394 | 94.07% | **0.0233** | **94.19%** | **0.0200** | **93.80%** | **0.0298** | **91.77%** |
| v3 + fine-tune, FILE, Q6_K embeddings | 892 | | | 0.0273 | 91.30% | | | | | | | | |
| bartowski Q4_K_M, FILE | 941 | 9.659 | 2.672 | 0.0282 | 91.70% | 0.0381 | 94.03% | 0.0334 | 93.84% | 0.0255 | 92.61% | 0.0317 | 91.15% |

Reading: against the handover recipe the combination is -8% KL on wiki, -53% on chat, -48% on UltraChat, -40% on multilingual, +11% on
code; against bartowski's file at the same size (946 vs 941 MiB, Q8_0 embeddings) it is ahead on wiki (0.0242 vs 0.0282), chat,
UltraChat and multilingual and on perplexity, and behind on code KL only (0.0394 vs 0.0381, file against file; the Q8_0 file
reproduces the dequantized checkpoint within 0.0002 on every held-out). v4 (Python weighted twice + 256 windows) + the fine-tune on the v4 text closes the code gap too: 0.0352 vs bartowski's 0.0381 KL,
94.71% vs 94.03% top-1, and it is the best row on every held-out but UltraChat (where v3's is 0.0006 lower). Against the handover
recipe: -13% KL on wiki, -1% on code, -55% chat, -47% UltraChat, -42% multilingual, and lower perplexity on both corpora (9.524 /
2.651 vs 9.562 / 2.666). The Q8_0 file (946 MiB) reproduces it within 0.0002 on all five held-outs: file against file, at 5 MiB more
than bartowski's Q4_K_M, it is -19% KL on wiki, -7% code, -34% chat, -20% UltraChat, -9% multilingual, with higher top-1 everywhere.
The Q6_K-embedding file (892 MiB, 49 MiB smaller than bartowski's) still beats it on wiki KL (0.0261 vs 0.0282).

## End-to-end pipeline on a new model (2026-10-09, m4): SmolLM2-1.7B-Instruct

`port/pipeline.sh NAME MODEL REF.gguf` runs the whole recipe unattended and resumably: fetch; the v4 calibration mix rendered with
the model's own chat template (`make_calib.py`) + the three held-out sets; sensitivity pass under q4_k / q5_k / q6_k; a bit budget
sized to the reference file (`size_target.py`: reference bytes minus the embedding / head at the chosen kinds, over the linear
weights); Dom's recipe with the map and the ridge; pack; KL fine-tune (in-memory, or `--stream` for models larger than the card);
pack; perplexity + KL / top-1 on wiki, code, chat, UltraChat, multilingual for the reference, the untuned and the tuned file;
a summary table. First model outside the Qwen family: SmolLM2-1.7B-Instruct (HuggingFaceTB, Apache, Llama architecture, tied
embeddings, 49k vocabulary), reference bartowski's Q4_K_M. 2 h 20 min on one A10G (sensitivity 6 min, quantize 64 min at 256
windows, fine-tune 40 min, evaluation 20 min). The budget landed at 4.68 bpw (q4_k 82%, q5_k 17%) for Q8_0 embeddings.

| SmolLM2-1.7B-Instruct | MiB | wiki ppl | code ppl | wiki KL / top-1 | code KL / top-1 | chat KL / top-1 | UltraChat KL / top-1 | multilingual KL / top-1 |
|---|---|---|---|---|---|---|---|---|
| bf16 | 3,266 | 8.941 | 3.070 | | | | | |
| bartowski Q4_K_M (imatrix) | 1,007 | 9.241 | 3.151 | 0.0412 / 91.41% | 0.0367 / 94.13% | 0.0643 / 90.62% | 0.0334 / 93.20% | 0.0652 / 88.55% |
| ours, untuned | 1,003 | 9.121 | 3.098 | 0.0298 / 92.06% | 0.0196 / 95.46% | 0.0313 / 92.90% | 0.0217 / 93.56% | 0.0436 / 89.99% |
| **ours, tuned** | **1,003** | **9.093** | **3.087** | **0.0271 / 92.44%** | **0.0176 / 95.65%** | **0.0232 / 93.92%** | **0.0189 / 94.28%** | **0.0389 / 90.71%** |

Reading: at 4 MiB less than bartowski's file, KL to bf16 is 34% lower on wiki, 52% on code, 64% on chat, 43% on UltraChat and
40% on multilingual, with higher top-1 everywhere and half the perplexity gap (wiki +1.7% vs +3.4% over bf16). The margins are
larger than on Qwen2.5-1.5B because the Llama-architecture 1.7B quantizes harder and the imatrix Q4_K_M suffers more; the
fine-tune adds its usual 9-26%. Found on the way: llama.cpp's converter interleaves the rows of q_proj / k_proj for the Llama
family (rotary layout); the GGUF reader now un-interleaves them (the reference first scored ppl 1372) and the writer interleaves
ours, so the file is correct for llama.cpp; Qwen2 / Qwen3.5 files were never affected.

## Related work (2026-10-09): two fast inference stacks for Qwen3.8-27B, and what they say about weights

Both items the user forwarded are inference engineering, not weight quantization. They sit on top of a
4-bit checkpoint and get their speed from speculation, caching and KV/activation quantization. The weight
format they need is the thing that touches our work.

**Mirai / uzu** (github.com/trymirai/uzu, Rust + Metal). Speculative decoding (DFlash-Weaver, rollback-free
tree verification): 92-117 tok/s for a 9B on an M5, 114 tok/s for the 27B on an M5 Max. Their checkpoint
`trymirai/Qwen3.8-27B-M` is 13,859 MiB plus a 984 MiB speculator: 4-bit asymmetric, group 64, bf16 scales,
4-bit zero points, Hadamard rotation, 8-bit activations. PTQ is YAQA (arXiv 2505.22988, end-to-end KL
Hessian sketch) followed by distillation of scales and zero points, calibrated on OpenHermes-2.5. They
claim "comparable to UD-Q4_K_M" with a KL chart and no numbers. This is the closest published recipe to
ours (Hessian-based PTQ, then a continuous-parameter fine-tune with codes fixed). Our YAQA test on the
1.5B (`yaqa_weights.py`, `gptq_torch.py --token-weights`) is the direct check of their PTQ step.

**syv-ai / qwen38-27b-rtx3090** ("HyperQwen", r/LocalLLaMA). One RTX 3090 at 250 W: 127 tok/s single
stream on ordinary chat, 381 tok/s when the answer is quoted from a 25k-token document (DFlash2 drafts 7
tokens, the remaining 8 of a 16-token block are filled from the prompt, 15 of 16 accepted), 1,035 tok/s
decode at 64 concurrent, prefix caching cuts a second question's TTFT from 22.4 s to 0.56 s. Stack: vLLM
0.30 with ~70 patches, int8 Marlin GEMMs on selected layers, int8-QK prefill attention, a 4/2-bit
"KVarN" KV cache (Sinkhorn-scaled), int8 head and embeddings. Weights are third-party W4A16 checkpoints
(compressed-tensors pack-quantized: AutoRound, or asymmetric AWQ g128 with zero points); they do not
quantize the body themselves. Quality table (their own test texts, so not comparable to our wiki numbers):
W4A16 ppl en 10.68 / code 3.05, GSM8K 95.5%, IFBench 78.3 vs 79.5 for bf16 ("about one point"). Each
int8-activation step costs ppl, mostly on code (10.68 -> 10.93 with int8 on all layers). Head: int8 head
10.68, int4 round-to-nearest head 10.81, int4 GPTQ head 10.77.

What carries over:
- The speed layer is format-agnostic as long as the weights are in a kernel-native format. Our K-quant
  files target llama.cpp/MLX; to sit under vLLM/Marlin we would export uniform int4 g128 with fp16 scales
  (GPTQ/compressed-tensors layout). The integer-offset refit (2026-10-05 section) already produces that
  layout and beat AWQ at equal bits on the 1.5B, so the recipe transfers; only the packer is missing.
- Their head numbers confirm the head is worth GPTQ rather than round-to-nearest: GPTQ at 4 bits recovers
  about a third of the RTN loss. Our `--head-kq` path is RTN; on the untied 27B a GPTQ'd Q5_K/Q4_K head
  would free 0.3-0.5 GB for the linears. Untested.
- They evaluate with GSM8K and IFBench, not only perplexity. We have GSM8K in the HF cache; a 200-question
  greedy run on the final 1.5B files (ours vs bartowski) is cheap and would answer "does KL translate to
  tasks". The 27B would need llama.cpp or vLLM-GGUF to generate.
- Both stacks train a small drafter on the target model's own output distribution. Drafter acceptance
  depends on the quantized target's distribution, so lower KL to bf16 should also mean higher acceptance
  with a drafter trained on bf16 output. That is a selling point for Dom's recipe, not a change to it.

### What unsloth's wiki advantage is made of (2026-10-09/10)

Unsloth's calibration text is not published (their docs: "hand-curated", >1.5M tokens for Dynamic 2.0, "refined for agentic coding,
chat and multilingual" for 3.0; a GitHub request for the raw data was closed without it). They say they deliberately exclude
Wikipedia-related calibration data because it over-fits the WikiText benchmark, so their wiki lead is not in-domain calibration.
Their imatrix for the 27B IS published (`imatrix_unsloth.gguf`, 13 MiB: 1,251 chunks, ~640k tokens, dataset name
`unsloth_calibration_dataset`), and it holds exactly the per-input-column mean x^2 of every linear, i.e. the diagonal of our
calibration Hessian, so the two calibrations can be compared directly (`imatrix_tools.py compare`, ours = calib_v4, 256 windows):

| linear (layers 0, 8, 47, 63 shown) | corr(log diag) | energy unsloth / ours | columns >3x apart |
|---|---|---|---|
| hidden-dim inputs (q/k/v, qkv, gate/up, in_proj_z) | 0.98-0.996 | 0.84-1.01 | 0% |
| o_proj / linear_attn.out_proj | 0.82-0.88 | 0.77-1.21 | 0.7-8% |
| mlp.down_proj (intermediate dim) | 0.60-0.90 | 0.72-1.02 | 1-16% |

Reading: on the residual-stream inputs the two calibrations agree almost perfectly (the statistics there are dominated by the
model, not the text); they differ on the post-nonlinearity inputs of down_proj and on the attention outputs, where the text
decides which intermediate features fire: 5-16% of those columns differ by more than 3x. That is where a calibration mix lives.
Three tests, all with the same bit map and ridge: (a) `--imatrix`: our recipe with our correlation structure but the Hessian
diagonal rescaled to unsloth's statistics (m1, codes only); (b) 1024 calibration windows instead of 256 (m3): is the Hessian
estimate the limit; (c) bartowski's public `calibration_datav3.txt` as the calibration text (m4): does a generic community mix
behave like theirs. Results below when they land.

**PrismML Ternary Bonsai 2 27B** (huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf, whitepaper September 2026).
Qwen3.8-27B re-made with ternary weights {-1, 0, +1}, fp16 scale per 128, in a fixed blockwise Hadamard basis
(block 1024, SpinQuant-style fixed signs): 1.76 bpw as packed, 5.93 GB for the language model vs 53.8 GB bf16
and 17.6 GB for unsloth's UD-Q4_K_XL. Everything is ternary (embeddings, head, all projections) except the
linear-attention state path (in_proj_a/b, conv1d, A_log, dt_bias) and the norms, 0.1% of parameters kept in
bf16: the same tensors our quantizer skips. Needs their llama.cpp / MLX forks (custom PTQ1_0 / PQ2_0 kernels);
47 tok/s on an M5 Max, 142 tok/s on an RTX 5090. The method is not disclosed ("proprietary Caltech IP"); a
ternary model at this quality is training-based (quantization-aware training with distillation from the base),
not post-training quantization, and no token count or compute is given.
Their numbers (thinking mode, xhigh effort, sampled at temperature 1, 20 benchmarks): bf16 85.4, Bonsai 83.9,
IQ2_XXS 75.2; in the model card's 14-benchmark table UD-Q4_K_XL scores 85.18 vs bf16 86.32 and Bonsai 84.78.
The average hides where the loss sits: GPQA Diamond 90.5 -> 85.8, MMLU-Redux 91.5 -> 89.1, Terminal-Bench 2.1
69.7 -> 52.8, SWE-bench Verified 80.6 -> 60.8, tau2-Bench 82.7 -> 80.2; at medium effort the gap widens
(82.6 -> 79.3 average, AIME25 86.3 -> 74.6, tau2 79.5 -> 70.9). Thinking budget compensates for weight damage.
What it means for this work:
- It confirms the regime boundary we measured on the 1.5B: post-training quantization under 3 bits collapses
  (IQ2_XXS loses 10 points at 2.2 bpw; our Q2_K/Q3_K maps lost most of the KL budget), and getting below that
  needs training the codes, not better rounding. Dom's recipe competes at 4-5 bpw, where the loss is ~1 point
  and the question is which 4-bit file is closest to bf16. Bonsai is a different product: a 6 GB model for
  16 GB laptops that gives up long-horizon agentic quality for footprint.
- The natural extension of our KL fine-tune is their direction: our `kl_tune` trains only scales and offsets
  with codes fixed; straight-through training of the codes themselves (QAT with the bf16 teacher) on top of
  Dom's initialisation is the step that would let a Q3_K-size file approach Q4 quality. Testable on the 1.5B
  with the existing Q3_K run as the starting point; a 27B version needs multi-GPU training, not our A10Gs.
- Their evaluation shows why KL to bf16 across domains is the more honest metric: an averaged benchmark score
  at high thinking effort can read 98% while agentic tasks lose a quarter. Our held-out set (wiki, code, chat,
  UltraChat, multilingual) is the cheap version of that check; the GSM8K run on the 1.5B files adds one task.
- Hadamard rotation again appears only where it is needed (ternary); our integer-offset test found it
  unnecessary at 4 bits. Keeping the linear-attention state path and norms in full precision is standard.

### YAQA token weights on the 1.5B (2026-10-09, m4): no gain over the plain Hessian

Mirai's recipe uses YAQA (arXiv 2505.22988): the layer Hessian is weighted per token by the end-to-end KL gradient energy at
the linear's output, H = sum_t ||delta_t||^2 x_t x_t^T, instead of GPTQ's uniform sum_t x_t x_t^T. `yaqa_weights.py` computes
the weights from one forward/backward of the bf16 model on the same 256 v4 windows (sampled tokens, Kronecker B = I; 62 s on
the 1.5B; mean coefficient of variation of the token weights 9.3, so the weighting is far from uniform), and
`gptq_torch.py --token-weights` applies them. Everything else identical to the v4 ridged run (map-free Q4_K, ridge 0.05,
256 windows), then the same KL fine-tune.

| 1.5B, dequantized unless noted | wiki ppl | code ppl | wiki KL / top-1 | code | chat | UltraChat | multi |
|---|---|---|---|---|---|---|---|
| v4 ridged (uniform Hessian) | 9.591 | 2.663 | 0.0249 / 92.55% | 0.0383 / 94.46% | 0.0261 / 94.00% | 0.0234 / 92.82% | 0.0310 / 91.77% |
| v4 ridged, YAQA weights | 9.592 | 2.672 | 0.0261 / 92.24% | 0.0376 / 94.50% | 0.0246 / 93.89% | 0.0236 / 93.19% | 0.0325 / 91.01% |
| v4 + KL fine-tune (the candidate) | 9.524 | 2.651 | **0.0227 / 92.93%** | **0.0352 / 94.71%** | **0.0220 / 94.31%** | **0.0205 / 93.17%** | **0.0286 / 92.12%** |
| YAQA + KL fine-tune | 9.531 | 2.668 | 0.0250 / 92.43% | 0.0366 / 94.59% | 0.0240 / 93.92% | 0.0223 / 93.34% | 0.0306 / 91.49% |
| YAQA + fine-tune, FILE Q8_0 (946 MiB) | | | 0.0251 / 92.29% | | | | |

Before the fine-tune the weighting is a wash (wiki and multilingual worse, code and chat better); after the fine-tune it is
behind the uniform twin on all five held-outs by 4-10%. The fine-tune gained less on the YAQA model (held-out 0.0359 -> 0.0339)
than on the uniform one, so the two are not complementary either. Our implementation is the cheap sketch (one sample per token,
B = I, no Kronecker factor on the output side); the paper's full version averages more samples and uses both factors. At 4.5 bpw
on this model the uniform Hessian plus the fine-tune is the better use of the compute. Not pursued further.

### GSM8K on the 1.5B files (2026-10-09, m4): the quantized files tie each other, all lose 3 points to bf16

Full GSM8K test set (1,319 questions), zero-shot chain of thought with the chat template and "put your final answer within
\boxed{}", greedy, 768 new tokens, scored on the exact file through our dequantizer (`gsm8k_eval.py`). Same prompt and
decoding for every model.

| Qwen2.5-1.5B-Instruct | GSM8K | right | truncated | vs our file (McNemar exact) |
|---|---|---|---|---|
| bf16 | **75.82%** | 1000 | 8 | only bf16 108, only ours 65, p = 0.001 |
| bartowski Q4_K_M (941 MiB) | 72.78% | 960 | 14 | only bartowski 114, only ours 111, p = 0.89 |
| ours v4 ridged, Q8 embeddings (946 MiB) | 72.86% | 961 | 10 | p = 0.81 |
| **ours v4 + KL fine-tune, Q8 (946 MiB)** | 72.55% | 957 | 11 | |
| ours YAQA + fine-tune, Q8 (946 MiB) | 73.46% | 969 | 13 | p = 0.44 |

Every 4-bit file loses the same 3 points to bf16 (significant), and the four files are statistically indistinguishable from each
other (about 110 questions flip each way between any pair). The KL differences between the files (ours 0.0229 vs bartowski
0.0282 on wiki; 0.0220 vs 0.0334 on chat) are real but do not show on this task at this sample size: a 1-point GSM8K difference
needs roughly 4x more questions to resolve, and the 1.5B's task accuracy is dominated by its own reasoning errors, not by
quantization noise. What the task eval does say is that none of the recipes damaged the model beyond the common 4-bit cost.
The honest summary for Dom: the KL suite ranks the files, GSM8K confirms none is broken, and task-level separation between
4-bit recipes of a 1.5B needs either a harder task or the 27B.

### What unsloth's wiki advantage is made of (2026-10-10, m1): not their calibration statistics

Unsloth's UD-Q4_K_M of the 27B beats our v4 file on wiki KL (0.0080 vs 0.0123) and multilingual (0.0078 vs 0.0116) and ties on code
(0.0186 vs 0.0184). Their calibration text is not published (issue #5500 closed without it), but their imatrix is
(`imatrix_unsloth.gguf`, 1,251 chunks of 512 = 640k tokens from a dataset they label "unsloth_calibration_dataset"; they state they
avoid Wikipedia-style calibration). The imatrix is the per-input-column mean square of the activations, i.e. the diagonal of our
Hessian, so two things can be compared directly.

**Their diagonal vs ours** (`imatrix_tools.py compare`, our Hessians from calib_v4 on 256 windows, layers 0 / 8 / 47 / 63, after the
value-head untiling fix for out_proj): correlation of the log diagonals 0.98-0.995 on the attention / MLP inputs (the residual stream),
0.82-0.88 on the linear-attention out_proj inputs, 0.60-0.79 on down_proj inputs (the SwiGLU product, the most data-dependent
statistic); energy ratios 0.72-1.21; columns more than 3x apart 0-16% (down_proj). Their data and ours light up the same columns on the
residual stream; the intermediate activations differ most, as expected from different text.

**Our recipe with their diagonal** (`gptq_torch.py --imatrix`: each linear's Hessian column-rescaled so its diagonal equals theirs,
correlations kept, same map, ridge, 256 windows; packed to 15,705 MiB with Q4_K embeddings and Q6_K head):

| 27B file, 6 windows | wiki KL / top-1 | code | multi |
|---|---|---|---|
| ours, v4 calibration (baseline) | 0.0123 / 95.29% | 0.0184 / 98.17% | 0.0116 / 95.47% |
| ours, v4 correlations + unsloth's imatrix diagonal | 0.0132 / 95.44% | 0.0198 / 98.18% | 0.0121 / 95.40% |
| ours, v4 calibration, 1024 windows instead of 256 (m3) | 0.0118 / 95.38% | **0.0161 / 98.23%** | 0.0111 / 95.70% |
| ours, bartowski's calibration_datav3 text, 128 windows (m4) | 0.0151 / 94.92% | 0.0293 / 97.58% | 0.0167 / 94.94% |
| ours, v4, 256 windows, UNSLOTH'S per-tensor allocation (15,763 MiB) | 0.0092 / 96.07% | **0.0147 / 98.35%** | 0.0086 / 95.91% |
| the same + KL fine-tune, 100 x 16 (m7, 75 min on the L40S) | 0.0095 / 96.12% | **0.0142 / 98.50%** | 0.0086 / 95.99% |
| unsloth's allocation + 1,024 windows, no fine-tune (m1, 15,763 MiB) | 0.0087 / 96.01% | 0.0144 / 98.49% | **0.0075 / 96.54%** |
| the same + KL fine-tune (m1) | 0.0091 / 95.97% | 0.0141 / 98.52% | 0.0076 / 96.50% |
| our budget + 1,024 windows + Q5_K head + fine-tune (pipeline.sh combo, m3 quantize / m7 tune, 15,854 MiB) | 0.0109 / 95.54% | 0.0141 / 98.21% | 0.0102 / 95.74% |
| unsloth UD-Q4_K_M | 0.0080 / 95.98% | 0.0186 / 98.14% | 0.0078 / 96.11% |

Importing their activation statistics makes our file slightly worse on all three held-outs (wiki +7%, code +8%, multilingual +4%), not better. Their wiki advantage therefore does
not come from the calibration data. What is left: (a) their per-tensor type allocation ("dynamic": which tensors get Q5_K / Q6_K / Q8_0),
which differs from our sensitivity-driven budget at the same total bytes, and (b) the quantizer itself (llama.cpp's importance-weighted
round-to-nearest with a per-block scale search vs GPTQ error feedback), whose error has a different structure. (a) is directly testable:
read the tensor types out of their GGUF, run our recipe with exactly their map (equal size by construction), and compare.

**Fourth test, their allocation** (m1, `job_m1_q38_umap.sh`: the tensor types read out of UD-Q4_K_M and mapped to our K-quant kinds,
IQ4_XS/IQ4_NL -> Q4_K, Q8_0 -> Q6_K, IQ3_S -> Q3_K; v4 text, 256 windows, ridge 0.05, packed 15,763 MiB, 0.4% above their 15,705):
wiki 0.0092 / 96.07%, code 0.0147 / 98.35%, multilingual 0.0086 / 95.91% (ppl 6.269 / 1.281 / 4.943 vs bf16 6.237 / 1.271 / 4.928).
This is the answer. Against our own sensitivity-driven budget at the same text and windows the KL drops 25% on wiki, 20% on code and
26% on multilingual; against unsloth's file we are now 21% ahead on code, ahead on top-1 everywhere (96.07 vs 95.98, 98.35 vs 98.14),
and within 0.0012 on wiki and 0.0008 on multilingual. Three quarters of the wiki gap was the allocation, i.e. which tensors get the
extra bits, not the quantizer and not the data; the remaining quarter is the quantizer's error structure plus the two levers not yet
applied to this file (1,024 windows, -4% wiki / -12% code above; the KL fine-tune, -1 to -6% on the 27B). Our greedy budget
(`bit_budget.py`, error per byte from a round-to-nearest fit without error feedback) is therefore the weakest part of the recipe on
this model: it ranks the linears by a proxy that GPTQ then changes. Next: (1) diff the two maps to see where the bytes moved (expected:
unsloth keeps attention q/k/v and the linear-attention projections higher and spends less on the MLP), (2) run the allocation with
1,024 windows and the fine-tune, which should put the file ahead of unsloth on all three held-outs at equal size, (3) a budget that
scores candidates with the actual GPTQ error (error-feedback-aware sensitivity) instead of the plain fit.

**The fine-tune on the allocation file** (m7, `job_m7_umap_tune.sh`: the umap codes copied from m1, `kl_tune_stream` 100 steps x 16 on the
v4 text, the same settings as the v4 tune; 38 s/step on the L40S vs ~3 min on the A10G, 75 minutes end to end; packed 15,763 MiB):
wiki 0.0095 / 96.12%, code 0.0142 / 98.50%, multilingual 0.0086 / 95.99%; wiki ppl 6.754 (20 x 2048) vs unsloth 6.773 and our v4 tune
6.800. Small and mixed, as on the 27B before: code -3%, wiki KL +3% with top-1 up, multilingual flat; top-1 agreement improves on all
three. With the allocation already fixed the continuous parameters have little left to absorb; the 1,024-window run on m1 is the lever
with the larger measured effect, and its fine-tuned file is the one to compare against unsloth.

**Unsloth's allocation with 1,024 windows** (m1, `job_m1_q38_umapn1k.sh`, before its fine-tune): wiki 0.0087 / 96.01%, code 0.0144 / 98.49%,
multilingual 0.0075 / 96.54% at 15,763 MiB. Against unsloth's file (15,705 MiB) this is ahead on multilingual (0.0075 vs 0.0078), far ahead
on code (0.0144 vs 0.0186), ahead on top-1 on all three held-outs, and within 9% on wiki (0.0087 vs 0.0080), with 0.4% more bytes. The two
levers add up as measured separately: the allocation took wiki from 0.0123 to 0.0092, the windows from 0.0092 to 0.0087 (the same -5% they
gave on the old map). The fine-tuned version of this file is scored below when it lands.
The pipeline's own combination (our greedy budget at the size target, 1,024 windows, GPTQ head attempted and fallen back to RTN Q5_K,
fine-tune; 15,854 MiB, 1% above unsloth) lands at 0.0109 / 0.0141 / 0.0102 (untuned, scored on m7: 0.0106 / 95.43%, 0.0144 / 98.21%,
0.0103 / 95.68%, so its fine-tune changed nothing either): with the windows and the tune equal, the allocation alone is worth 20-25% of the
KL on wiki and multilingual. The budget is what to fix in `pipeline.sh`. The fine-tune on the 1,024-window allocation file is likewise
flat on code (0.0144 -> 0.0141) and multilingual (0.0075 -> 0.0076) and 5% worse on wiki (0.0087 -> 0.0091; wiki ppl 6.757 vs 6.754
for the 256-window tune and 6.773 for unsloth): on the 27B, once the allocation and the Hessian are right, the
100 x 16 fine-tune of super-scales and norms is not worth its hours; it earned its keep on the 1.5B (-11% wiki KL) and not here.

**Third test, bartowski's public calibration text** (`calibration_datav3.txt`, the file his imatrix files are built from: ~70k tokens of
mixed English prose, chat and trivia, no code; 128 windows with overlap, same map and ridge): wiki 0.0151 (+23% vs our v4 mix), code 0.0293
(+59%), multilingual 0.0167 (+44%; ppl 4.965 vs 4.927 bf16). Less data and no code or non-English text, and the file shows it on every held-out.
(m4's copy of `multi_test.txt` turned out to differ from the one on m1/m3, same size, different order, bf16 ppl 6.11 vs 4.93; the multilingual
number above is a re-score on the m1 file, the m4-file score 0.0174 is kept as `q38_kqdyn_bart_multi_m4oldtext.json`. The m4 file is now the
m1 one, so any earlier m4-only multilingual number is only comparable with other m4 numbers.) Taken with the
imatrix test, calibration DATA is ruled out as the source of unsloth's wiki edge: neither their statistics nor the other public
calibration set beats our mix on our recipe, and our own mix with four times the windows is the lever that works (the 1024-window row).
What remains is the allocation (their K-quant type per tensor) and the quantizer's error structure.

**Is the Hessian estimate the limit?** (m3, `job_m3_q38_n1k.sh`: the same v4 text, map and ridge, but 1,024 calibration windows of 512
tokens instead of 256, 4x the statistics; same file size.) Four times the windows cuts the KL by 4% on wiki (0.0123 -> 0.0118),
12% on code (0.0184 -> 0.0161, now clearly ahead of unsloth's 0.0186) and 4% on multilingual (0.0116 -> 0.0111); ppl 6.320 vs 6.237
bf16 on wiki, 1.277 vs 1.271 on code. So the 256-window Hessian was noticeably under-sampled for the code-heavy layers (the ridge had
been covering for it) and 1,024 windows should be the default for the 27B (quantization time 9.7 h on the slow-disk m3 vs 8.5 h,
most of the extra cost is the calibration forward passes), but it moves wiki by a tenth of the 0.0043 gap to unsloth: the Hessian
estimate is not where the wiki advantage lives either. (The bartowski-text control above was run twice: a stale copy of its job script
woke from the GPU-wait loop and deleted the finished codes under the pack, 9 GPU-hours lost; lesson, never edit a job script while a
copy of it is running, bash reads it by byte offset.)

### GPTQ on the output head (2026-10-10, m1): one tier down for free

The packer's head path was round-to-nearest (Dom's per-group grid fit, super-block projected, no Hessian). `head_gptq.py` runs the
full recipe on the 248k x 5120 head instead: the head's Hessian is the final-norm output of the QUANTIZED model over the 256 v4 windows
(so the head is calibrated on the inputs it will see), then GPTQ + ridged refine/refit under the K-quant constraint, in row chunks
(rows are independent given H; 17 min per kind). `codes2gguf.py --head-codes` packs the result. Error in the head's own layer metric
(x^T H x) relative to RTN of the same kind: Q6_K 0.26, Q5_K 0.33, Q4_K 0.29 (plain MSE 1.4-1.6x: the recipe spends weight error where
the inputs do not go). Everything else is the fine-tuned v4 file; KL on 6 windows per held-out.

| 27B, v4 + fine-tune codes, head variant | file MiB | wiki KL / top-1 | code | multi |
|---|---|---|---|---|
| RTN Q6_K (the shipped file) | 15,705 | 0.0122 / 95.34% | 0.0173 / 98.16% | 0.0114 / 95.51% |
| GPTQ Q6_K | 15,705 | **0.01205 / 95.46%** | **0.01715 / 98.16%** | **0.01136 / 95.61%** |
| GPTQ Q5_K | 15,544 | 0.01227 / 95.13% | 0.01730 / 98.09% | 0.01173 / 95.33% |
| RTN Q5_K | 15,544 | 0.01283 / 94.98% | 0.01730 / 98.14% | 0.01212 / 95.32% |
| GPTQ Q4_K | 15,392 | 0.01350 / 94.83% | 0.01759 / 98.06% | 0.01330 / 94.75% |
| RTN Q4_K | 15,392 | 0.01575 / 93.96% | 0.01792 / 97.99% | 0.01514 / 94.12% |

- GPTQ on the head is worth 1-4% on wiki and multilingual at equal head size (Q6_K: -1%; Q5_K: -4% / -3%; Q4_K: -14% / -12%), nothing
  on code, where the head is not the bottleneck. The gain grows as the head gets coarser, as expected.
- A GPTQ Q5_K head equals the shipped RTN Q6_K head within noise (wiki +0.5%, code 0%, multi +3%) at 161 MB less. A Q4_K head costs
  11% wiki / 17% multi even with GPTQ: the head's share of the KL is real below 5.5 bits.
- Follow-up: rerun the 27B budget with the head fixed at GPTQ Q5_K and the 161 MB handed to the linears (+0.05 bpw), together with the
  1024-window calibration above. Both are now pipeline options: `--head q5_k` with a head-codes stage after quantize, and `--nsamples 1024`.
