# port/: the GPTQ-K recipe for llama.cpp GGUF files (PyTorch, one NVIDIA GPU)

A PyTorch port of the mlx-gptq recipe (sequential GPTQ with error feedback, MSE range search, Lloyd grid fit, refinement passes,
ridge-damped refit) that writes **stock llama.cpp GGUF files**: the quantizer is run under the K-quant constraint (Q3_K / Q4_K / Q5_K /
Q6_K super-blocks of 256 with 6-bit sub-scales, projected exactly as llama.cpp stores them), so the file is bit-exact for llama.cpp's
kernels and needs no custom runtime. Validated on Qwen2.5-1.5B, SmolLM2-1.7B (Llama family) and Qwen3.8-27B (Qwen3.5 hybrid
attention); on the 27B the file beats unsloth's UD-Q4_K_M on code (-23% KL to bf16) and multilingual text at the same size. The full
experimental log is `REPORT.md`; the 27B handover and the Hugging Face publishing note are in `notes/`.

## One command

    bash pipeline.sh NAME MODEL_DIR REFERENCE.gguf [--nsamples 1024] [--kq-map budgets/MAP.json] [--target-bpw X] [--embed q4_k] [--head q6_k] [--stream] [--tune-steps 100 --tune-bs 16]

Runs every stage and skips the ones whose output exists: calibration mix -> sensitivity pass -> bit budget to the reference file's size
-> quantize -> pack -> KL fine-tune -> pack -> KL / perplexity on five held-outs for the untuned, tuned and reference files, then
`results/NAME/summary.md`. Needs on the box: the bf16 checkpoint (HF layout), a reference GGUF of the same model (any quant; its
metadata and tokenizer are copied), `wikitext2_train.txt`, `python_calib.txt`, the HF datasets cache for GSM8K / MMLU / UltraChat and
the 8-language Wikipedia samples (`fetch_calib_extra.py`), gguf-py on `PYTHONPATH`, torch + transformers. A 1.7B takes ~2.5 h on an
A10G; a 27B ~9 h to quantize (`--stream` for the fine-tune on a 24 GB card; `--codes-only` keeps the 54 GB dequantized checkpoint off
the disk).

## The stages as scripts

| step | script | what it does |
|---|---|---|
| calibration | `make_calib.py` (`make_calib_v2/v3.py`, `fetch_calib_extra.py`) | the v4 mix: wikitext-2 train, Python, chat-template GSM8K/MMLU, UltraChat, 8-language Wikipedia; held-outs `code_test`, `chat_test`, `ultrachat_test`, `multi_test` |
| allocation | `bit_budget.py --kq-forms q3_k,q4_k,q5_k,q6_k` -> `size_target.py` | per-linear sensitivity (error per byte of a plain fit) and a greedy budget to a target size. **On the 27B unsloth's per-tensor allocation beats this budget by 20-26% KL** (`REPORT.md`, "What unsloth's wiki advantage is made of"); `unsloth_map.py` reads a map out of their GGUF (`budgets/q38_unsloth_map.json`) and `--kq-map` uses it |
| quantize | `gptq_torch.py MODEL OUT 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 --kq-map MAP --calib calib_v4.txt --nsamples 1024 --codes-only` | the recipe, layer-streamed (`stream_util.py`, `quant_core.py`, `kquant.py`); options tested and documented in REPORT.md: `--imatrix` (unsloth's statistics, no gain), `--token-weights` (YAQA, no gain), rotation (`rotate.py`, unnecessary at 4 bits) |
| pack | `codes2gguf.py QUANT_DIR REFERENCE.gguf OUT.gguf --embed-kq q4_k --head-kq q6_k [--temp-file]` | writes the GGUF from the codes (`gguf_source.py` reads one back, inverting the converter's transforms, incl. the Llama q/k rotary interleave); `head_gptq.py` is the GPTQ head variant, `gguf_check.py` / `gguf_rt_check.py` round-trip checks |
| fine-tune | `pack_codes.py`, `kl_tune.py` (in memory) / `kl_tune_stream.py` (layer-streamed) | KL(bf16 || student) training of the fp16 super-scales (through fp16 rounding), norms and biases with the codes fixed. -11% wiki KL on the 1.5B; flat on the 27B once allocation and Hessian are right (ship untuned there) |
| evaluate | `kld_stream.py`, `ppl_stream.py`, `gsm8k_eval.py` | KL to bf16, top-1 agreement, perplexity over 6 x 2048 tokens per held-out; GSM8K on the exact file |

`jobs/` holds every job script that produced a number in REPORT.md (box names m1-m7 are our A10G / L40S machines); `budgets/` the
27B maps (ours at 4.783 bpw, unsloth's, and the per-tensor bits of the shipped 27B file). `imatrix_tools.py` compares a llama.cpp
imatrix with our Hessian diagonal; `yaqa_weights.py`, `hidden_order.py`, `codes2awq.py` are the negative-result experiments.

## Results in one table (KL(bf16 || file) / top-1 agreement, 6 x 2048 tokens per held-out, same protocol for every row)

| model, file (MiB) | wiki | code | multilingual |
|---|---|---|---|
| Qwen2.5-1.5B, bartowski Q4_K_M (941) | 0.0282 / 92.19% | 0.0381 / 94.41% | 0.0317 / 91.57% |
| Qwen2.5-1.5B, ours v4 + fine-tune, Q8 embeddings (946) | **0.0229 / 92.82%** | **0.0353 / 94.69%** | **0.0288 / 91.98%** |
| SmolLM2-1.7B, bartowski Q4_K_M (1,007) | 0.0412 | 0.0367 | 0.0652 |
| SmolLM2-1.7B, ours tuned (1,003) | **0.0271 / 92.44%** | **0.0176 / 95.65%** | **0.0389** |
| Qwen3.8-27B, unsloth UD-Q4_K_M (15,705) | **0.0080 / 95.98%** | 0.0186 / 98.14% | 0.0078 / 96.11% |
| Qwen3.8-27B, ours with unsloth's allocation, 1,024 windows (15,763) | 0.0087 / 96.01% | **0.0144 / 98.49%** | **0.0075 / 96.54%** |

Everything here was measured through gguf-py's dequantizer against the bf16 model on the same box, not through llama.cpp's kernels;
run `llama-perplexity` on a file before publishing it (`notes/HF_PUBLISH_NOTE_FOR_DOM.md`).
