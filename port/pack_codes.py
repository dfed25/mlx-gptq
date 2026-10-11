"""Pack a codes/ directory for the streamed fine-tuner: 4-bit codes two per byte (5/6-bit codes stay one per byte), sub-scales / sub-mins
as uint8, super-scales as fp16. Halves the bytes read per pass on a Q4_K-dominated model.   python pack_codes.py QUANT_DIR/codes OUT_DIR"""
import sys, os, glob, torch
from safetensors.torch import save_file
from safetensors import safe_open
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); tot = 0
for f in sorted(glob.glob(os.path.join(src, "layer*.safetensors"))):
    new = {}
    with safe_open(f, "pt") as sf:
        for k in sf.keys():
            if not k.endswith(".codes"): continue
            n = k[:-6]; q = sf.get_tensor(k); bits = int(sf.get_tensor(n + ".bits")[0])
            new[n + ".q"] = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous() if bits == 4 else q
            new[n + ".bits"] = sf.get_tensor(n + ".bits"); new[n + ".sc"] = sf.get_tensor(n + ".ksc").to(torch.uint8); new[n + ".m"] = sf.get_tensor(n + ".km").to(torch.uint8)
            new[n + ".kd"] = sf.get_tensor(n + ".kd"); new[n + ".kdmin"] = sf.get_tensor(n + ".kdmin")
    o = os.path.join(out, os.path.basename(f)); save_file(new, o); tot += os.path.getsize(o); print(os.path.basename(f), f"{os.path.getsize(o)/2**20:.0f} MiB", flush=True)
print(f"packed {tot/2**30:.1f} GiB into {out}")
