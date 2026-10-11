# IDEA: PACKED BLOCKMAX, fused: one GPU pass reads the weight and the gradient and writes the weight. The momentum is unpacked from, updated, and repacked into the weight's low 8 bits inside the same pass. Only three data streams per step (AdamW: seven).
import math
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "packed16_triton"
DEFAULT_LR = 4e-3
K = 16
BB = 64


@triton.jit
def _packed_kernel(p_ptr, g_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps,
                   K: tl.constexpr, BB: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0)
    p = tl.load(p_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + blk, mask=bmask, other=0.0)
    v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))

    bits = p.to(tl.int32, bitcast=True)
    q = bits & 255
    q = tl.where(q > 127, q - 256, q)
    qf = q.to(tl.float32)
    mag = tl.exp2((tl.abs(qf) - 101.0) * 0.25)
    m = tl.where(q > 0, mag, tl.where(q < 0, -mag, 0.0))          # code 0 means exactly zero
    m = m + (1.0 - b1) * (g - m)

    idx = tl.floor(tl.log2(tl.maximum(tl.abs(m), 1e-30)) * 4.0 + 0.5) + 101.0
    idx = tl.minimum(tl.maximum(idx, 1.0), 127.0)
    sgn = tl.where(m > 0, 1.0, tl.where(m < 0, -1.0, 0.0))
    code = (idx * sgn).to(tl.int32) & 255

    w = p * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)
    out = ((w.to(tl.int32, bitcast=True) & -256) | code).to(tl.float32, bitcast=True)
    tl.store(p_ptr + off, out, mask=mask)
    tl.store(v_ptr + blk, v, mask=bmask)


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                n = p.numel()
                nb = (n + K - 1) // K
                if "v" not in st:
                    st["step"] = 0
                    st["v"] = torch.zeros(nb, device=p.device, dtype=p.dtype)
                    p.data.view(torch.int32).bitwise_and_(-256)
                st["step"] += 1
                t = st["step"]
                _packed_kernel[((nb + BB - 1) // BB,)](
                    p.data, p.grad, st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
                    (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], K=K, BB=BB)
