#!/bin/bash
# The combined 27B run: pipeline.sh with 1024 calibration windows, GPTQ Q5_K head (161 MB to the linears via the size target), streamed 100 x 16 fine-tune,
# against unsloth's UD-Q4_K_M. Waits for >= 90 GB free (the user clears scored artifacts) and for the reference GGUF copy.
set -u; cd ~/domq/port
until [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -ge 90 ] && [ -f models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf ] && [ "$(stat -c %s models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf)" -ge 16400000000 ]; do sleep 300; done
echo "== combo start $(date -u +%T) ($(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)G free)"
bash pipeline.sh q38_combo models/q38_27b_src models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf --nsamples 1024 --head q5_k --forms q4_k,q5_k,q6_k --embed q4_k --stream --tune-steps 100 --tune-bs 16 --dtype bfloat16 --nwin 6
echo "== COMBO M3 DONE $(date -u +%T)"
