# mlx-gptq

GPTQ (error-feedback) quantization for MLX language models on Apple Silicon, packed into MLX's own quantized
format so the result loads and runs with `mlx_lm` at the same speed as the models you download today.

The 4-bit models on `mlx-community` are made by round-to-nearest (`mlx_lm convert -q`). On the two models measured so
far, GPTQ with an error-minimising grid removes about 70% of their quality loss, at the same size and speed within 5%.

Note: mlx-lm itself ships calibration-based quantizers as separate commands (`mlx_lm.gptq`, `mlx_lm.awq`, `mlx_lm.dwq`,
`mlx_lm.dynamic_quant`). Its GPTQ supports 2/4/8 bits, computes all Hessians once from the unquantized model and takes
each group's grid from the original weights. This repository differs in three details from the GPTQ paper's recipe:
layers are calibrated sequentially on already-quantized predecessors, each group's grid is chosen from the
error-updated weights, and the grid range is searched for least error (`--mse`); it also supports 3 bits. On Qwen2.5-1.5B,
same evaluation, all packed: `mlx_lm.gptq` 10.75, `mlx_lm.awq` 10.50, round to nearest 10.67, this repository 9.66 (details
and sizes in RESULTS.md; `dwq` not yet measured).

| model | fp16 | 4-bit round to nearest (what `mlx_lm convert -q` and mlx-community ship) | **mlx-gptq 4-bit** |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 8.94 | 10.54 (922 MB, 119 tok/s) | **9.41** (970 MB, 114 tok/s) |
| Qwen2.5-1.5B-Instruct | 9.38 | 10.67 | **9.65** (dequantized weights; packed 4-bit-embedding version 10.16, see RESULTS.md) |

WikiText-2 test perplexity (lower is better), 20 windows of 2048 tokens, group size 64, MacBook Pro M4 Pro, MLX 0.32.
The SmolLM2 row is the packed model exactly as `quantize.sh` produces it (8-bit embedding, float16). The 8-bit
embedding is why it is 5% larger and 5% slower than the round-to-nearest model; with a 4-bit embedding
(`EMBED_BITS=4`) size and speed match the community model and the perplexity rises by about 0.5.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
# an fp16 MLX model (from mlx_lm convert, or any MLX model directory)
python -m mlx_lm convert --hf-path Qwen/Qwen2.5-1.5B-Instruct --mlx-path models/qwen1.5b-fp16 --dtype float16

# the whole pipeline: calibration data, GPTQ, quality check, packing, float16 cast, quality check
./quantize.sh models/qwen1.5b-fp16 models/qwen1.5b-gptq4 4

python -m mlx_lm generate --model models/qwen1.5b-gptq4 --prompt "Explain GPTQ in one paragraph."
```

Runtime for a 1.5B model on an M4 Pro: about 11 minutes for GPTQ, a minute for the rest.

The steps, if you want them separately:

| script | what it does |
|---|---|
| `get_wikitext.py` | downloads WikiText-2 raw (train for calibration, test for evaluation) |
| `gptq_mlx.py MODEL OUT BITS [--group 64] [--mse] [--nsamples 128] [--seqlen 512] [--damp 0.01] [--block 128]` | GPTQ on every linear layer; writes the dequantized weights plus the exact codes (`gptq_codes.npz`) |
| `ppl_wikitext.py MODEL [NWIN]` | WikiText-2 perplexity of any MLX model |
| `build_quantized.py DEQ OUT BITS [--group 64] [--embed-bits 8]` | packs the exact codes into `QuantizedLinear` layers; the tied embedding is quantized separately (8-bit by default) |
| `cast_fp16.py SRC OUT` | casts bfloat16 parameters to float16 (30% faster decoding) |
| `pack_affine.py` | the packing routine, self-tested bit-exact against `mx.quantize` for 2, 3, 4, 6 and 8 bits |

## How it works

Rounding one column of a weight matrix to the grid makes an error. Instead of accepting it, GPTQ (Frantar, Ashkboos,
Hoefler, Alistarh 2022) adjusts the columns that are not yet rounded so that the layer's output on real inputs
changes as little as possible. With H = XᵀX computed from calibration inputs, the adjustment is the corresponding
row of the upper Cholesky factor of H⁻¹. Layers are processed in order and each is calibrated on the outputs of the
already-quantized layers before it. The `--mse` grid search shrinks each group's min-max range when clipping the
extremes costs less than the coarser step costs the rest of the group.

Two details matter for speed on Apple GPUs and are handled by the pipeline: the tied embedding/output projection is
read on every token, so it is quantized as well (8-bit is free in quality, 4-bit costs about 0.5 perplexity but
matches the community models' size and speed), and the model is stored in float16, not bfloat16.

## Limits

- Tested on Qwen2 and Llama architectures (`model.model.layers[i].self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`).
  Other architectures need the projection list in `build_quantized.py` extended.
- Calibration uses WikiText-2 train; perplexity is measured on WikiText-2 test. Numbers on other text will differ.
- The 3-bit models are smaller but not faster than 4-bit on current MLX kernels.

## Credits

GPTQ is by Frantar et al. (ICLR 2023). MLX and mlx-lm are by Apple. Built by Domenic Federico with Claude, September
2026; every number above is reproducible with the scripts in this repository.

MIT licence.
