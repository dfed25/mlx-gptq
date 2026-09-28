"""Cast every floating-point parameter of a saved MLX model to float16 (our conversions came out bfloat16, which is
slower on the Apple GPU and mixes dtypes with the float16 quantization scales). Usage: cast_fp16.py SRC OUT"""
import sys, pathlib, shutil, mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import save_model
from mlx.utils import tree_map
src, out = sys.argv[1], pathlib.Path(sys.argv[2])
m, tok = load(src)
m.update(tree_map(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a, m.parameters()))
out.mkdir(exist_ok=True); save_model(out, m)
for f in pathlib.Path(src).glob("*"):
    if f.suffix != ".safetensors" and f.is_file() and not (out / f.name).exists(): shutil.copy(f, out / f.name)
print("saved", out)
