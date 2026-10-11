# Publishing the Qwen3.8-27B GPTQ-K file on Hugging Face: note for Dom (2026-10-11)

Dom, this is everything you need to put the 27B file up. The file is on m5 at `~/from_marcello/qwen38_27b/gguf/q38_dom_dyn_umapn1k.gguf`
(15,763 MiB); the scripts that made it and the KL results are next to it (`code/`, `results_kld/`, `REPORT.md`,
`QWEN38_27B_FOR_DOM.md`). Below: what the file is, a ready-to-paste model card, the upload steps, and two checks to run before you press
publish.

## What the file is

- **Base model**: Qwen3.8-27B (the Qwen3.5-family hybrid linear-attention + gated-attention 27B; HF class `Qwen3_5ForConditionalGeneration`,
  `model_type qwen3_5`, 64 layers, hidden 5120, vocab 248,320, untied head; Apache-2.0 per the source LICENSE). The GGUF is the
  **language model only** (`general.architecture = qwen35`, same layout as unsloth's text GGUF); there is no vision projector in it, so
  for image input users need an `mmproj` file (unsloth ships one in their GGUF repo; it is not part of this release).
- **Quantization**: your recipe, i.e. sequential GPTQ with error feedback, MSE range search, Lloyd grid fit, `--refine 3 --refit 2`, ridge
  0.05, under the llama.cpp K-quant constraint (Q4_K / Q5_K / Q6_K / Q3_K super-blocks of 256 with 6-bit sub-scales, exact for llama.cpp's
  arithmetic). **The per-tensor type allocation is unsloth's**: the tensor types were read out of their `Qwen3.8-27B-UD-Q4_K_M.gguf` and
  mapped to K-quants (IQ4_XS / IQ4_NL -> Q4_K, Q8_0 -> Q6_K, IQ3_S -> Q3_K; `code/unsloth_map.py`, map in `code/budgets/q38_unsloth_map.json`).
  Say so in the card; it is the honest description and it is why the file is the same size as theirs. Their calibration data and
  imatrix were NOT used (we tested them; they do not help this recipe). Token embeddings Q4_K, output head Q6_K (round-to-nearest, as in
  llama.cpp), norms / linear-attention state path (A_log, dt_bias, conv1d) in F32/F16 as llama.cpp expects. About 4.9 bits per weight over
  the whole file (16.5 GB / 27B parameters); the exact per-tensor map is `bits_used.json` in `code/` (and inside the quantized dir).
- **Calibration**: our `calib_v4.txt` mix (wikitext-2 train, Python source, chat-template GSM8K and MMLU train questions, UltraChat
  conversations, Wikipedia in 8 languages, Python weighted twice), 1,024 windows of 512 tokens = 524k tokens, sampled with seed 0.
- **Packed with** `code/codes2gguf.py` through gguf-py (metadata and tokenizer copied from a reference GGUF of the same model).

## The numbers to put in the card

KL(bf16 || file) and top-1 agreement with the bf16 model, 6 windows of 2,048 tokens per held-out (`code/kld_stream.py`, bf16 teacher on
the same box); wikitext-2 perplexity over 20 windows of 2,048 (`code/ppl_stream.py`). Same protocol for every row.

| GGUF (MiB) | wiki KL / top-1 | code KL / top-1 | multilingual KL / top-1 | wikitext-2 ppl |
|---|---|---|---|---|
| bf16 reference | 0 / 100% | 0 / 100% | 0 / 100% | 6.237 (6 x 2048) |
| unsloth UD-Q4_K_M (15,702) | 0.0080 / 95.98% | 0.0186 / 98.14% | 0.0078 / 96.11% | 6.773 |
| **this file (15,763)** | **0.0087 / 96.01%** | **0.0144 / 98.49%** | **0.0075 / 96.54%** | 6.264 (6 x 2048; 20 x 2048 run pending on m1) |

In words: at 0.4% more bytes than unsloth's file, 23% lower KL on code, 4% lower on multilingual text, higher top-1 agreement on all three
held-outs, 9% higher KL on wiki. Do not quote chat-template numbers (the bf16 reference itself is unstable under this model's thinking
template; reference perplexity 525 on that held-out). No task benchmark was run on the 27B files; on the 1.5B the four 4-bit recipes were
statistically indistinguishable on GSM8K (all lost 3 points to bf16), so the KL suite is the ranking, not a task score.

## Ready-to-paste model card (README.md of the HF repo)

```markdown
---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - gptq
  - quantized
  - qwen3.8
quantized_by: dfed25
---

# Qwen3.8-27B GPTQ-K (GGUF, 4.9 bpw)

A 4-bit GGUF of Qwen3.8-27B quantized with sequential GPTQ (error feedback, MSE range search, Lloyd grid fit, two refinement passes)
under the llama.cpp K-quant constraint, so the file runs in stock llama.cpp with Q4_K / Q5_K / Q6_K / Q3_K kernels. The per-tensor type
allocation follows unsloth's UD-Q4_K_M; the weights inside every block are chosen by GPTQ on 524k tokens of mixed calibration text
(wiki, code, chat, multilingual). Language model only; for images use an mmproj file for Qwen3.8-27B.

| file | size | wiki KL / top-1 | code KL / top-1 | multilingual KL / top-1 |
|---|---|---|---|---|
| unsloth UD-Q4_K_M | 15.3 GiB | 0.0080 / 95.98% | 0.0186 / 98.14% | 0.0078 / 96.11% |
| **Qwen3.8-27B-GPTQ-Q4_K.gguf** | 15.4 GiB | 0.0087 / 96.01% | 0.0144 / 98.49% | 0.0075 / 96.54% |

KL(bf16 || quantized) and top-1 agreement with the bf16 model over 6 x 2,048 tokens of held-out wikitext-2, Python code and
8-language Wikipedia; lower KL is closer to the original model. Same protocol for both rows.

Method and code: https://github.com/dfed25/mlx-gptq (the GGUF path is the `port/` scripts; the file was made with
`gptq_torch.py --kq-map ... --nsamples 1024 --codes-only` and `codes2gguf.py`).
```

Adjust `base_model` to the exact HF id of the base checkpoint (the source README is titled "Qwen3.8-27B"; confirm the org/name on the
hub), and the method link to wherever you put the port scripts.

## Upload steps

```bash
pip install -U huggingface_hub
hf auth login                                   # once
hf repo create dfed25/Qwen3.8-27B-GPTQ-GGUF --type model
cd ~/from_marcello/qwen38_27b
cp gguf/q38_dom_dyn_umapn1k.gguf Qwen3.8-27B-GPTQ-Q4_K.gguf        # llama.cpp's naming convention: <model>-<quant>.gguf
hf upload dfed25/Qwen3.8-27B-GPTQ-GGUF Qwen3.8-27B-GPTQ-Q4_K.gguf Qwen3.8-27B-GPTQ-Q4_K.gguf
hf upload dfed25/Qwen3.8-27B-GPTQ-GGUF README.md README.md
```

A single 16.5 GB file is fine for the hub (the per-file limit is 50 GB); no need to split. If you want the 256-window twin up as well
(`q38_dom_dyn_umap.gguf`, same size, wiki 0.0092 / code 0.0147 / multilingual 0.0086), name it `...-Q4_K-c256.gguf` and say in the
card that it is the same recipe with a quarter of the calibration.

## Two checks before publishing

1. **Run it in llama.cpp.** Everything above was measured through gguf-py's dequantizer with the bf16 model as the reference, not
   through llama.cpp's kernels. The packing is bit-exact K-quant and the 1.5B files of the same packer ran fine, but the 27B GGUF has
   not been through `llama-cli` by us. Build a recent llama.cpp (one with Qwen3.5/3.8 support), then:
   `llama-perplexity -m Qwen3.8-27B-GPTQ-Q4_K.gguf -f wikitext-2-raw/wiki.test.raw -c 2048` and compare with the same command on
   unsloth's file; the ratio should match the table (ours a touch higher on wiki). Then a short `llama-cli -m ... -p "..."` chat to see
   it talks. If anything is off, the KL JSONs in `results_kld/` tell you which held-out to look at.
2. **Credit and license.** The card above credits unsloth for the allocation and keeps Apache-2.0 from the base model; keep both.
   Qwen's Apache-2.0 text is `models/q38_27b_src/LICENSE` on any of the boxes if the hub wants a LICENSE file.

If you would rather publish under a different repo name or with the tuned variant (it is not better; the fine-tune was flat on the 27B,
see REPORT.md), say so and I will stage it.
