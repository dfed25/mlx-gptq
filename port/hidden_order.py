"""One hidden-dimension order shared by every layer: stream the model once over the calibration windows, accumulate the
diagonal of X^T X at every linear that reads the residual stream (in_features == hidden_size, i.e. q/k/v/gate/up; o_proj and
down_proj read other dimensions), normalise each linear's diagonal by its mean, average, sort descending.
    python hidden_order.py MODEL OUT.pt [--nsamples 128 --seqlen 512 --calib wikitext2_train.txt]"""
import argparse, os, time, torch, numpy as np
from stream_util import Streamer
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--nsamples", type=int, default=128); ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--calib", default="wikitext2_train.txt"); ap.add_argument("--dtype", default="float16"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); DEV = os.environ.get("DOMQ_DEV", "cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
S = Streamer(a.model, dtype=getattr(torch, a.dtype), device=DEV); hidden = S.text_cfg.hidden_size
ids = S.tok.encode(open(a.calib).read()); rng = np.random.default_rng(a.seed); starts = rng.integers(0, len(ids) - a.seqlen, size=a.nsamples)
calib = torch.tensor(np.stack([ids[s:s + a.seqlen] for s in starts]))
KW = S.capture_kwargs(calib[:a.bs]); hs = [S.embed(calib[b:b + a.bs]) for b in range(0, a.nsamples, a.bs)]; S.offload_head()
total = torch.zeros(hidden, dtype=torch.float64); count = 0; per_layer = []
for li in range(S.n_layers):
    layer = S.load_layer(li); kw = S.kw_for(li, KW); lins = {n: layer.get_submodule(n) for n in S.lin_names(layer)}
    acc = {n: torch.zeros(hidden, dtype=torch.float64) for n, m in lins.items() if m.in_features == hidden and not n.endswith(("o_proj", "down_proj"))}
    def mk(n):
        def hook(mod, inp, out): X = inp[0].reshape(-1, inp[0].shape[-1]).double(); acc[n] += (X * X).sum(0).cpu()
        return hook
    hds = [lins[n].register_forward_hook(mk(n)) for n in acc]
    hs = [S.run_layer(layer, h, kw) for h in hs]
    for hd in hds: hd.remove()
    for n, d in acc.items(): d = d / d.mean().clamp_min(1e-30); total += d; count += 1; per_layer.append(d)
    S.free_layer(li)
mean = total / count; order = torch.argsort(mean, descending=True)
# how consistent is the order across layers: Spearman of each layer's diag against the mean
rk = torch.argsort(torch.argsort(mean)).double(); cors = []
for d in per_layer:
    r = torch.argsort(torch.argsort(d)).double(); cors.append(torch.corrcoef(torch.stack([rk, r]))[0, 1].item())
torch.save(order.cpu(), a.out)
print(f"{a.model}: hidden {hidden}, {count} linears over {S.n_layers} layers; top channel usage ratio max/median {mean.max().item() / mean.median().item():.1f}; "
      f"rank correlation of per-linear usage with the shared order: mean {np.mean(cors):.2f} min {np.min(cors):.2f}; saved {a.out} ({time.time() - t0:.0f}s)")
