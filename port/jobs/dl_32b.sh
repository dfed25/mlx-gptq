#!/bin/bash
# Approved downloads for the 32B run (2026-10-04): source bf16 + Qwen's official AWQ and GPTQ-Int4.
set -u; cd ~/domq; mkdir -p models port/results
~/venv-vllm/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo, name in [("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen32_awq"), ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", "qwen32_gptq4"), ("Qwen/Qwen2.5-32B-Instruct", "qwen32_src")]:
    p = snapshot_download(repo, local_dir=f"models/{name}", max_workers=4); print("downloaded", repo, "->", p, flush=True)
PYEOF
du -sh models/qwen32_* && touch port/results/dl32.done && echo "DL DONE $(date -u +%T)"
