"""Pack integer codes q (rows x K, values in [0, 2^bits-1]) into MLX's affine quantized layout, so that a model rounded
by OUR method (GPTQ) can be loaded as a real QuantizedLinear and run with MLX's kernels.
Layout (verified below against mx.quantize): along each row, codes are packed little-endian, `bits` bits each, into a
byte stream, which is viewed as uint32 (little-endian) of shape (rows, K*bits/32)."""
import numpy as np, mlx.core as mx

def pack(q, bits):
    q = np.asarray(q, dtype=np.uint64); rows, K = q.shape
    assert (K * bits) % 32 == 0
    per_word = 32 // np.gcd(32, bits) if bits in (3, 5, 6) else 32 // bits     # values per 32-bit word for byte-aligned packs
    # general: pack via 96-bit chunks (lcm of 32 and bits*something): simplest exact method = big-int per row chunk of 32 values*bits
    chunk = 32                                     # 32 codes -> 32*bits bits -> bits uint32 words
    out = np.zeros((rows, K * bits // 32), dtype=np.uint32)
    for c in range(0, K, chunk):
        block = q[:, c:c + chunk]                  # (rows, 32)
        acc = np.zeros(rows, dtype=object)
        for j in range(chunk):
            acc = acc | (block[:, j].astype(object) << (bits * j))
        for w in range(bits):
            out[:, (c // chunk) * bits + w] = np.array([(int(a) >> (32 * w)) & 0xFFFFFFFF for a in acc], dtype=np.uint32)
    return out

def codes_from_mlx(W, bits, gs):
    wq, s, b = mx.quantize(W, group_size=gs, bits=bits); mx.eval(wq, s, b)
    Wd = mx.dequantize(wq, s, b, group_size=gs, bits=bits)
    W32 = np.array(W.astype(mx.float32)); S = np.repeat(np.array(s.astype(mx.float32)), gs, axis=1); B = np.repeat(np.array(b.astype(mx.float32)), gs, axis=1)
    q = np.clip(np.round((np.array(Wd.astype(mx.float32)) - B) / S), 0, 2 ** bits - 1).astype(np.int64)
    return q, np.array(wq), s, b

if __name__ == "__main__":
    mx.random.seed(1)
    for bits in (2, 3, 4, 6, 8):
        W = mx.random.normal((16, 256)).astype(mx.float16)
        q, wq_ref, s, b = codes_from_mlx(W, bits, 64)
        mine = pack(q, bits)
        ok = mine.shape == wq_ref.shape and np.array_equal(mine, wq_ref.astype(np.uint32) if wq_ref.dtype != np.uint32 else wq_ref)
        print(f"{bits}-bit: packed shape {mine.shape} vs mlx {wq_ref.shape} dtype {wq_ref.dtype}: {'MATCH' if ok else 'DIFFERENT'}")
