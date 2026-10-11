"""Tensor-level validation of gguf_source against the HF bf16 checkpoint for chosen layers: exact match for unquantized
tensors (norms, A_log, dt_bias, conv1d), per-row relative error for quantized matrices (a wrong row/column permutation
shows up as rows with ~100% error).   python gguf_check.py HF_DIR GGUF layers..."""
import sys, json, torch, numpy as np
from stream_util import Streamer
from gguf_source import GGUFSource
hf, gg = sys.argv[1], sys.argv[2]; layers = [int(x) for x in sys.argv[3:]] or [0]
S = Streamer(hf, dtype=torch.bfloat16, device="cpu")                      # HF side, CPU, bf16 to fit RAM
G = GGUFSource(gg, S.text_cfg, S.layout)
print(f"gguf: {len(G.names())} tensors; reorder={G.reorder} (k heads {G.nk}, v heads {G.nv})")
for n in G.names()[:0]: pass
bad = 0
for li in layers:
    for name in S.layer_param_names(li):
        a = S.load(name).float(); ck = S.ckpt_key(name)
        try: b = G.get(ck).float()
        except KeyError as e: print(f"  MISSING {name}: {e}"); bad += 1; continue
        if a.shape != b.shape: print(f"  SHAPE {name}: hf {tuple(a.shape)} gguf {tuple(b.shape)}"); bad += 1; continue
        gname = [k for k in G.types][-1]; qt = G.types[gname]
        if a.ndim <= 1 or qt in ("F32", "F16", "BF16"):
            d = (a - b).abs().max().item(); ok = d < 1e-2 * (a.abs().max().item() + 1e-6)
            print(f"  {'ok ' if ok else 'BAD'} {name:50s} {qt:5s} max|diff| {d:.2e}", flush=True); bad += (not ok)
        else:
            rel = ((a - b).norm(dim=1) / (a.norm(dim=1) + 1e-9)); ok = rel.max().item() < 0.5 and rel.mean().item() < 0.2
            print(f"  {'ok ' if ok else 'BAD'} {name:50s} {qt:5s} row rel err mean {rel.mean():.3f} max {rel.max():.3f}", flush=True); bad += (not ok)
for name in ("model.norm.weight", "model.embed_tokens.weight"):
    a = S.load(name).float(); b = G.get(S.ckpt_key(name)).float(); rel = ((a - b).norm() / (a.norm() + 1e-9)).item(); del b
    print(f"  {'ok ' if rel < 0.2 else 'BAD'} {name:50s} rel err {rel:.4f}", flush=True); bad += (rel >= 0.2)
print("CHECK", "PASSED" if bad == 0 else f"FAILED ({bad} problems)")
