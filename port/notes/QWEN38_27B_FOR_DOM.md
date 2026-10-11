# Qwen3.8-27B under the Q4_K constraint: the handover (2026-10-11)

What is in this folder (`~/from_marcello/qwen38_27b/`):

- `gguf/q38_dom_dyn_umapn1k.gguf` (15,763 MiB): the file to compare with unsloth's UD-Q4_K_M (15,705 MiB). Your recipe (sequential
  GPTQ with error feedback, MSE range search, Lloyd, refine 3 / refit 2, ridge 0.05) under the K-quant constraint, with UNSLOTH'S
  per-tensor type allocation (read out of their GGUF, `unsloth_map.py` -> `budgets/q38_unsloth_map.json`), calib_v4 text, 1,024 windows
  of 512 tokens. Q4_K embeddings, Q6_K round-to-nearest head. (`q38_dom_dyn_umap.gguf`, if present, is the 256-window twin: the same
  recipe with a quarter of the calibration, wiki 0.0092 / code 0.0147 / multilingual 0.0086.)
- `code/`: every script (`gptq_torch.py`, `kquant.py`, `quant_core.py`, `codes2gguf.py`, `gguf_source.py`, `kld_stream.py`,
  `ppl_stream.py`, `bit_budget.py`, `unsloth_map.py`, `kl_tune_stream.py`, `pack_codes.py`, `pipeline.sh`, the job_*.sh runs),
  the calibration text, the five held-out texts and `budgets/` (our sensitivity tables and unsloth's map).
- `results_kld/`: the KL JSONs of every 27B file scored on m7 (the KL protocol: 6 windows of 2,048 tokens per held-out, bf16 teacher).
- `REPORT.md`: the full log; the section "What unsloth's wiki advantage is made of" has the whole story below in detail.

## The numbers (27B, KL(bf16 || file) / top-1 agreement, 6 x 2048 tokens)

| file | MiB | wiki | code | multilingual |
|---|---|---|---|---|
| unsloth UD-Q4_K_M | 15,705 | 0.0080 / 95.98% | 0.0186 / 98.14% | 0.0078 / 96.11% |
| ours, our greedy budget, 256 windows (the 10-07 file) | 15,705 | 0.0123 / 95.29% | 0.0184 / 98.17% | 0.0116 / 95.47% |
| ours, unsloth's allocation, 256 windows | 15,763 | 0.0092 / 96.07% | 0.0147 / 98.35% | 0.0086 / 95.91% |
| **ours, unsloth's allocation, 1,024 windows** | 15,763 | **0.0087 / 96.01%** | **0.0144 / 98.49%** | **0.0075 / 96.54%** |

## What we learned about the gap to unsloth

1. Not their calibration data: our recipe with their published imatrix statistics (the diagonal of our Hessian rescaled to theirs) is
   slightly worse everywhere; bartowski's public calibration text is far worse (wiki +23%, code +59%).
2. Three quarters of their wiki lead is the per-tensor allocation (which linears get Q5_K / Q6_K). Swapping only the map cut our KL by
   20-26% on every held-out. Our greedy budget ranks linears by a round-to-nearest proxy that the error feedback then changes; it is the
   weak step of the recipe on this model.
3. A quarter is the Hessian estimate: 4x the calibration windows is worth -5% on wiki and -12% on code.
4. The KL fine-tune of the fp16 super-scales and norms (100 steps x 16 windows) is flat on the 27B once the allocation and the Hessian
   are right (code 0.0144 -> 0.0141, multilingual 0.0075 -> 0.0076); it earned -11% on the 1.5B and nothing here. Ship the untuned file.
5. What remains on wiki (0.0087 vs 0.0080, 9%) is the quantizer's error structure (llama.cpp's importance-weighted RTN with block scale
   search vs GPTQ error feedback), with our file ahead on code (-23%), multilingual and top-1 agreement on all three.

## How to reproduce (box with a 24 GB GPU, 30 GB RAM, the bf16 source in models/q38_27b_src)

    python gptq_torch.py models/q38_27b_src models/q38_kqdyn_umapn1k 4 --group 32 --mse --lloyd --refine 3 --refit 2 --refine-damp 0.05 \
        --kq-map budgets/q38_unsloth_map.json --dtype bfloat16 --bs 4 --calib calib_v4.txt --nsamples 1024 --codes-only
    python codes2gguf.py models/q38_kqdyn_umapn1k REFERENCE.gguf out.gguf --embed-kq q4_k --head-kq q6_k
    python kld_stream.py models/q38_27b_src --cand-gguf out.gguf --text wikitext2_test.txt --nwin 6 --dtype bfloat16 --out kld.json

(~9 h for the quantization on an A10G; `--codes-only` keeps the 54 GB dequantized checkpoint off the disk; the reference GGUF only
provides metadata, any GGUF of the same model works.) The map was read out of unsloth's file with `unsloth_map.py` (IQ4_XS/IQ4_NL -> Q4_K,
Q8_0 -> Q6_K, IQ3_S -> Q3_K).

Open ends worth your time: a budget that scores candidates with the real GPTQ error instead of the RTN proxy (so we stop borrowing their
map), and a diff of the two maps to see where unsloth spends the bytes (attention vs MLP) — both in REPORT.md under "Next".
