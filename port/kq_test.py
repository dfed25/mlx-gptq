"""Self-test of the K-quant projection and packers against gguf-py's dequantizer."""
import torch, numpy as np, gguf
from gguf import dequantize, GGMLQuantizationType as Q
from kquant import kq_project, pack_q4_k, pack_q5_k, pack_q3_k, pack_q2_k, pack_q6_k, rtn_kquant, KQ
torch.manual_seed(0)
for kind, packer in (("q4_k", pack_q4_k), ("q5_k", pack_q5_k), ("q3_k", pack_q3_k), ("q2_k", pack_q2_k), ("q6_k", pack_q6_k)):
    k = KQ[kind]; rows, cols = 8, 512; g = k["sb"]; ng = cols // g
    SC = (torch.rand(rows, ng, dtype=torch.float64) * 0.02 + 0.001); LO = -(torch.rand(rows, ng, dtype=torch.float64) * 0.3) * SC * k["qmax"]
    LOp, SCp, meta = kq_project(LO, SC, kind)
    rel = ((SCp - SC).abs() / SC).max().item(); relm = ((LOp - LO).abs() / LO.abs().clamp_min(1e-9)).max().item()
    assert (meta["sc"] >= 0).all() and (meta["sc"] <= k["smax"]).all()
    if k["mmax"] is not None: assert (meta["m"] >= 0).all() and (meta["m"] <= k["mmax"]).all()
    q = torch.randint(0, k["qmax"] + 1, (rows, cols))
    exp = (LOp.repeat_interleave(g, 1) + SCp.repeat_interleave(g, 1) * q.double()).numpy()
    blk = packer(q.numpy(), meta["d"].numpy(), meta["sc"].numpy()) if k["zfix"] is not None else packer(q.numpy(), meta["d"].numpy(), meta["dmin"].numpy(), meta["sc"].numpy(), meta["m"].numpy())
    got = dequantize(blk.reshape(rows, -1), getattr(Q, kind.upper())).reshape(rows, cols).astype(np.float64)
    err = np.abs(got - exp).max() / np.abs(exp).max()
    print(f"{kind}: projection max rel scale err {rel:.4f}, min err {relm:.4f}; pack->gguf dequant max rel diff {err:.2e} {'OK' if err < 1e-3 else 'FAIL'}; block bytes {blk.shape[-1]}")

# RTN embedding path: dequantized result must be close to the matrix (relative error small) for every kind
W = torch.randn(16, 1024, dtype=torch.float32) * 0.05
for kind in ("q6_k", "q5_k", "q4_k", "q3_k", "q2_k"):
    blk = rtn_kquant(W, kind); got = dequantize(blk.reshape(16, -1), getattr(Q, kind.upper())).reshape(16, 1024)
    print(f"rtn {kind}: rel rms err {np.sqrt(((got - W.numpy()) ** 2).mean()) / W.numpy().std():.4f}, {blk.shape[-1]} bytes/block")
