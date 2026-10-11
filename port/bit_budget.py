"""Sensitivity pass + bit allocation (the unsloth-style per-tensor budget, chosen by Dom's layer metric).
For every linear layer: collect H = X^T X on the calibration text (sequentially, on the UNQUANTIZED model), fit Dom's grid
(range search + alternating fit, integer zero points) at each candidate width WITHOUT error feedback, and score the
rounding error in the layer metric, err_b = e^T H e / (w^T H w) summed over rows. Then a greedy budget: start every layer
at the lowest width, repeatedly buy the upgrade with the largest error reduction per byte until the target average bits
per weight (over the quantized linears) is reached. Writes the bits map (JSON {exact-name regex: bits}) and the table.
    python bit_budget.py MODEL OUT.json --target-bpw 4.5 [--widths 3,4,5,8] [--group 32] [--calib ...]"""
import argparse, json, time, re, torch, numpy as np
from stream_util import Streamer
from quant_core import grid_for_group, DEV
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--target-bpw", type=float, default=4.5)
ap.add_argument("--widths", default="3,4,5,8"); ap.add_argument("--group", type=int, default=32); ap.add_argument("--nsamples", type=int, default=128); ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--calib", default="wikitext2_train.txt"); ap.add_argument("--dtype", default="float16"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale-bytes", type=float, default=4.0, help="bytes of scale+offset per group (fp16 scale + fp16 min, as in GGUF Q4_1/Q5_1)")
ap.add_argument("--kq-forms", default=None, help="comma list of K-quant kinds (e.g. q3_k,q4_k): score each linear under each form (super-block projected grids) and allocate by form; writes {regex: kind} usable as --kq-map")
ap.add_argument("--max-layers", type=int, default=0, help="smoke test: score only this many layers")
ap.add_argument("--fit-int-zero", action="store_true", help="score with the integer-zero-point constrained fit (2^bits slower; the plain fit ranks widths the same)")
ap.add_argument("--table", default=None, help="sensitivity table to reuse (written on the first run; later targets skip the pass)")
a = ap.parse_args(); t0 = time.time()
FORMS = a.kq_forms.split(",") if a.kq_forms else None
if FORMS:
    from kquant import KQ, kq_project; widths = FORMS
else: widths = [int(x) for x in a.widths.split(",")]
import os
if a.table and os.path.exists(a.table):
    rows_out = json.load(open(a.table))["rows"]; rows_out = [dict(r, err={(k if FORMS else int(k)): v for k, v in r["err"].items()}) for r in rows_out]
    print(f"loaded sensitivity table {a.table} ({len(rows_out)} linears)")
else:
  S = Streamer(a.model, dtype=getattr(torch, a.dtype), device=DEV); tok = S.tok
  ids = tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed); starts = rng.integers(0, len(ids) - a.seqlen, size=a.nsamples)
  calib = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in starts])); KW = S.capture_kwargs(calib[:a.bs]); hs = [S.embed(calib[b:b + a.bs]) for b in range(0, a.nsamples, a.bs)]; S.offload_head()
  rows_out = []
  for li in range(S.n_layers):
      layer = S.load_layer(li); kw = S.kw_for(li, KW); names = S.lin_names(layer); lins = {n: layer.get_submodule(n) for n in names}
      acc = {n: {"H": torch.zeros((m.in_features, m.in_features), dtype=torch.float32, device=DEV), "n": 0} for n, m in lins.items()}
      def mk(n):
          def hook(mod, inp, out):
              X = inp[0].reshape(-1, inp[0].shape[-1]).float(); acc[n]["H"].addmm_(X.T, X); acc[n]["n"] += X.shape[0]
          return hook
      hds = [lins[n].register_forward_hook(mk(n)) for n in names]
      for h in hs: S.run_layer(layer, h, kw)
      for hd in hds: hd.remove()
      for n in names:
          W = lins[n].weight.data.float(); H = acc[n]["H"] / acc[n]["n"]; acc[n]["H"] = None; rows, cols = W.shape; g = a.group
          energy = ((W @ H) * W).sum().item() + 1e-30; errs = {}
          for b in widths:
              if FORMS:                                                                               # K-quant form: group = sub-block, grids projected per super-block
                  k = KQ[b]; g = k["sb"]; QMAX = float(k["qmax"]); grp = W.reshape(rows * (cols // g), g).double()
                  lo, hi = grid_for_group(grp, QMAX, True, True, k["zfix"] is not None, k["zfix"]); sc = ((hi - lo) / QMAX).clamp_min(1e-8)
                  LOp, SCp, _ = kq_project(lo.reshape(rows, cols // g), sc.reshape(rows, cols // g), b); lo = LOp.reshape(-1); sc = SCp.reshape(-1)
              else:
                  g = a.group; QMAX = float(2 ** b - 1); grp = W.reshape(rows * (cols // g), g).double()              # every group of every row at once
                  lo, hi = grid_for_group(grp, QMAX, True, True, a.fit_int_zero); sc = ((hi - lo) / QMAX).clamp_min(1e-8)
              q = ((grp - lo[:, None]) / sc[:, None]).round().clamp(0, QMAX); Q = (q * sc[:, None] + lo[:, None]).float().reshape(rows, cols)
              E = Q - W; errs[b] = ((E @ H) * E).sum().item() / energy
          rows_out.append({"name": f"model.layers.{li}.{n}", "params": rows * cols, "err": errs}); del W, H
      hs = [S.run_layer(layer, h, kw) for h in hs]; S.free_layer(li)
      if a.max_layers and li + 1 >= a.max_layers: break
      print(f"layer {li:2d} scored ({time.time() - t0:.0f}s): " + ", ".join(f"{r['name'].split('.',3)[-1]}: " + "/".join(f"{r['err'][b]:.4f}" for b in widths) for r in rows_out[-len(names):]), flush=True)
  if a.table: json.dump({"rows": rows_out, "widths": widths, "group": a.group, "model": a.model}, open(a.table, "w"), indent=1); print(f"wrote sensitivity table {a.table} ({time.time() - t0:.0f}s)")
# ---- greedy budget
total = sum(r["params"] for r in rows_out); bytes_per_w = (lambda b: KQ[b]["bpw"] / 8) if FORMS else (lambda b: b / 8 + a.scale_bytes / a.group)
cur = {r["name"]: widths[0] for r in rows_out}
def bpw(): return 8 * sum(r["params"] * bytes_per_w(cur[r["name"]]) for r in rows_out) / total
while bpw() < a.target_bpw:
    best = None
    for r in rows_out:
        b = cur[r["name"]]; i = widths.index(b)
        if i + 1 >= len(widths): continue
        nb = widths[i + 1]; gain = (r["err"][b] - r["err"][nb]) * r["params"]; cost = r["params"] * (bytes_per_w(nb) - bytes_per_w(b))
        if best is None or gain / cost > best[0]: best = (gain / cost, r["name"], nb)
    if best is None: break
    cur[best[1]] = best[2]
summary = {b: sum(r["params"] for r in rows_out if cur[r["name"]] == b) / total for b in widths}
print(f"budget: target {a.target_bpw} bpw -> {bpw():.3f} bpw over {total/1e9:.2f}B quantized weights; share per width: " + ", ".join(f"{b}{'' if FORMS else 'b'} {100*v:.1f}%" for b, v in summary.items()))
json.dump({"^" + re.escape(k) + "$": v for k, v in cur.items()}, open(a.out, "w"), indent=1)
json.dump({"rows": rows_out, "alloc": cur, "bpw": bpw(), "target": a.target_bpw, "group": a.group}, open(a.out.replace(".json", "_table.json"), "w"), indent=1)
print(f"wrote {a.out} ({time.time() - t0:.0f}s)")
