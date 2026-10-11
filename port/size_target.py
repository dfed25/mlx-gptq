"""Average bits per weight over the quantized linears that makes a packed file the size of a reference GGUF, given the embedding / head
kinds: target = 8 * (ref_bytes - embed_bytes - head_bytes - small) / N_linear.    python size_target.py MODEL_DIR REF.gguf TABLE.json --embed q8_0|q6_k|q4_k [--head q6_k] [--extra-mib N]"""
import argparse, json, os
BPW = {"f16": 16, "q8_0": 8.5, "q6_k": 6.5625, "q5_k": 5.5, "q4_k": 4.5}
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ref"); ap.add_argument("table"); ap.add_argument("--embed", default="q8_0"); ap.add_argument("--head", default=None); ap.add_argument("--extra-mib", type=float, default=8.0, help="metadata + norms allowance")
a = ap.parse_args(); c = json.load(open(os.path.join(a.model, "config.json"))); c = c.get("text_config", c); V = c["vocab_size"]; H = c["hidden_size"]
tied = c.get("tie_word_embeddings", False); N = sum(r["params"] for r in json.load(open(a.table))["rows"])
emb = V * H * BPW[a.embed] / 8; head = 0 if tied else V * H * BPW[a.head or a.embed] / 8
lin = os.path.getsize(a.ref) - emb - head - a.extra_mib * 2**20; print(f"{8 * lin / N:.3f}")
