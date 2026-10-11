"""Dom's recipe on the OUTPUT HEAD under the K-quant constraint (codes2gguf.py --head-kq is round-to-nearest). The head's
Hessian is collected from the QUANTIZED model's final-norm outputs (the inputs the head will actually see): the quantized
model is read from a GGUF (--gguf) or, without it, from the source. Rows are independent given H, so the head is quantized
in row chunks. Writes OUTPREFIX_<kind>.safetensors (codes + super-block metadata) for codes2gguf.py --head-codes, caches the
Hessian in OUTPREFIX_H.pt, and prints the head's error in the layer metric relative to the RTN K-quant of the same kind.
    python head_gptq.py SRC OUTPREFIX --gguf QUANT.gguf --calib calib_v4.txt --nsamples 256 --seqlen 512 --kinds q6_k,q5_k,q4_k"""
import argparse, os, time, torch, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from safetensors import safe_open
from safetensors.torch import save_file
from stream_util import Streamer
from quant_core import gptq, refine_with_refit, empty_cache, DEV
from kquant import KQ, rtn_kquant
import gguf
from gguf.quants import dequantize

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--gguf", default=None, help="GGUF of the quantized model: its layers produce the head's inputs")
ap.add_argument("--calib", default="calib_v4.txt"); ap.add_argument("--nsamples", type=int, default=256); ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--kinds", default="q6_k,q5_k,q4_k")
ap.add_argument("--refine", type=int, default=3); ap.add_argument("--refit", type=int, default=2); ap.add_argument("--refine-damp", type=float, default=0.05); ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--block", type=int, default=128)
ap.add_argument("--rows", type=int, default=16384, help="rows per GPTQ chunk"); ap.add_argument("--max-rows", type=int, default=0, help="smoke test: only the first rows of the head")
a = ap.parse_args(); torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
S = Streamer(a.src, dtype=getattr(torch, a.dtype), device=DEV, gguf=a.gguf); tok = S.tok; hid = S.text_cfg.hidden_size

# ---- Hessian of the head's inputs (final-norm outputs of the quantized model) over the calibration windows
HF = f"{a.out}_H.pt"
if os.path.exists(HF): H = torch.load(HF).to(DEV); print(f"head Hessian from {HF}", flush=True)
else:
    ids = tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed); starts = rng.integers(0, len(ids) - a.seqlen, size=a.nsamples)
    calib = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in starts]))
    KW = S.capture_kwargs(calib[:a.bs]); hs = [S.embed(calib[b:b + a.bs]) for b in range(0, a.nsamples, a.bs)]; S.offload_head()
    for li in range(S.n_layers):
        layer = S.load_layer(li); kw = S.kw_for(li, KW); hs = [S.run_layer(layer, h, kw) for h in hs]; S.free_layer(li)
        if li % 8 == 0 or li == S.n_layers - 1: print(f"  layer {li} done ({time.time() - t0:.0f}s)", flush=True)
    H = torch.zeros((hid, hid), dtype=torch.float32, device=DEV); n = 0
    with torch.no_grad():
        for h in hs: X = S.model.model.norm(h).reshape(-1, hid).float(); H.addmm_(X.T, X); n += X.shape[0]
    H /= n; torch.save(H.cpu(), HF); del hs; empty_cache()
    print(f"head Hessian from {n} tokens of the {'GGUF' if a.gguf else 'source'} model's final-norm outputs -> {HF} ({time.time() - t0:.0f}s)", flush=True)

# ---- the head weight from the SOURCE checkpoint (bf16), rows in chunks
key = S.ckpt_key("model.embed_tokens.weight" if S.tied else "lm_head.weight")
with safe_open(S.wmap[key], "pt") as sf: W = sf.get_tensor(key).float()
if a.max_rows: W = W[:a.max_rows]
rows, cols = W.shape; assert cols == hid; print(f"head {rows} x {cols} ({'tied to the embeddings' if S.tied else 'untied'}), chunks of {a.rows} rows", flush=True)
for kind in a.kinds.split(","):
    k = KQ[kind]; QMAX = float(k["qmax"]); FZ = k["zfix"]; G = k["sb"]; IZ = FZ is not None; bits = {"q4_k": 4, "q5_k": 5, "q6_k": 6}[kind]; T = torch.arange(QMAX + 1.0, device=DEV)
    parts = {"codes": [], "kd": [], "kdmin": [], "ksc": [], "km": []}; err = {"g": 0.0, "r": 0.0}; mse = {"g": 0.0, "r": 0.0}; flips = 0; tk = time.time()
    for r0 in range(0, rows, a.rows):
        Wc = W[r0:r0 + a.rows].to(DEV); Hn = H.clone()
        Wq, codes, SC, BI = gptq(Wc, [Hn.double()], QMAX, G, a.damp, a.block, True, True, IZ, FZ, None, kind); empty_cache()
        dead = torch.diag(Hn) == 0; Hn /= max(torch.diag(Hn).mean().item(), 1e-30); Hn.diagonal().add_(a.refine_damp); Wr = Wc.clone(); Wr[:, dead] = 0
        out = refine_with_refit(Wr, codes, BI.double(), SC.double(), Hn, T, QMAX, rounds=a.refit, sweeps=a.refine, group=G, int_zero=IZ, metric="full", block=16, fixed_zero=FZ, kq=kind)
        codes, Wq, SC, BI, nfl, mt = out[:6]; flips += nfl; del Wr, Hn
        E = Wc - Wq.float(); err["g"] += float((E @ H * E).sum()); mse["g"] += float((E * E).sum())
        Wrtn = torch.from_numpy(dequantize(rtn_kquant(Wc.to(torch.bfloat16).cpu(), kind), getattr(gguf.GGMLQuantizationType, kind.upper())).astype(np.float32).reshape(Wc.shape)).to(DEV)
        E = Wc - Wrtn; err["r"] += float((E @ H * E).sum()); mse["r"] += float((E * E).sum()); del Wrtn, E
        parts["codes"].append(codes.to(torch.uint8).cpu()); parts["kd"].append(mt["d"].half().cpu()); parts["kdmin"].append(mt["dmin"].half().cpu()); parts["ksc"].append(mt["sc"].cpu()); parts["km"].append(mt["m"].cpu())
        del Wc, Wq, codes, SC, BI; empty_cache(); print(f"  {kind}: rows {r0}-{min(r0 + a.rows, rows)} ({time.time() - tk:.0f}s)", flush=True)
    out_f = f"{a.out}_{kind}.safetensors"
    save_file({"codes": torch.cat(parts["codes"]), "kd": torch.cat(parts["kd"]), "kdmin": torch.cat(parts["kdmin"]), "ksc": torch.cat(parts["ksc"]), "km": torch.cat(parts["km"]), "bits": torch.tensor([bits], dtype=torch.uint8)}, out_f)
    print(f"{kind}: head error GPTQ/RTN in the layer metric {err['g'] / err['r']:.3f}, plain MSE {mse['g'] / mse['r']:.3f}; refine changed {100 * flips / (rows * cols):.1f}% of codes; wrote {out_f} ({time.time() - tk:.0f}s)", flush=True)
print(f"done ({time.time() - t0:.0f}s)")
