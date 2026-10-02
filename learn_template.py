"""Learn a tick template for B bits from a model's weights: Lloyd-Max (1-D k-means) on weights normalised per group of
64 by (x - mean)/std, pooled over a sample of layers; saved scaled to [0, 2^B - 1].  learn_template.py MODEL BITS OUT.npy"""
import sys, numpy as np, mlx.core as mx
from mlx_lm import load
model, _ = load(sys.argv[1]); B = int(sys.argv[2]); L = 2 ** B; rng = np.random.default_rng(0); Zs = []
for li in range(0, len(model.model.layers), 3):
    for lin in (model.model.layers[li].self_attn.q_proj, model.model.layers[li].mlp.down_proj, model.model.layers[li].mlp.gate_proj):
        G = np.array(lin.weight.astype(mx.float32)).reshape(-1, 64); G = G[rng.choice(len(G), 3000, replace=False)]
        Zs.append(((G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True)).ravel())
Z = np.concatenate(Zs); c = np.quantile(Z, np.linspace(0.5 / L, 1 - 0.5 / L, L))
for _ in range(40):
    idx = np.abs(Z[:, None] - c[None, :]).argmin(1); c = np.array([Z[idx == j].mean() if (idx == j).any() else c[j] for j in range(L)])
t = np.sort(c); t = (t - t.min()) / (t.max() - t.min()) * (L - 1); np.save(sys.argv[3], t); print("template", np.round(t, 3))
