"""Stream a Hugging Face causal LM through the GPU one decoder layer at a time, so a model larger than the card (or the
box's RAM) can be calibrated, quantized and evaluated. Handles text-only checkpoints (Qwen2, Qwen3, Llama), the language
model inside Gemma3ForConditionalGeneration (`language_model.model.` prefix) and inside Qwen3.5/3.8
ForConditionalGeneration (`model.language_model.` prefix, hybrid linear/full attention layers, untied lm_head)."""
import json, os, glob, shutil, torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MIN_DIM = 256          # linears narrower than this (Qwen3.5's in_proj_a / in_proj_b gates) stay in fp16, as vLLM expects

class StopForward(Exception): pass

class Streamer:
    def __init__(self, path, dtype=torch.float16, device="cuda", gguf=None):
        self.gguf = gguf
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download; path = snapshot_download(path)
        self.path = path; self.dtype = dtype; self.device = device
        cfg = AutoConfig.from_pretrained(path); self.full_cfg = cfg
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx): self.wmap = {k: os.path.join(path, v) for k, v in json.load(open(idx))["weight_map"].items()}
        else:
            self.wmap = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with safe_open(f, "pt") as sf:
                    for k in sf.keys(): self.wmap[k] = f
        keys = list(self.wmap)
        # checkpoint layout -> text-model parameter names
        if any(k.startswith("model.language_model.") for k in keys): self.layout = "qwen_vl"      # Qwen3.5/3.8: model.language_model.*, lm_head.*
        elif any(k.startswith("language_model.model.") for k in keys): self.layout = "gemma_vl"   # Gemma3: language_model.model.*, language_model.lm_head.*
        else: self.layout = "text"
        tc = getattr(cfg, "text_config", None)
        if tc is not None and self.layout != "text":
            self.text_cfg = tc; self.text_cfg.torch_dtype = dtype
            from transformers import AutoModelForCausalLM
            with init_empty_weights(): self.model = AutoModelForCausalLM.from_config(tc, dtype=dtype)
        else:
            self.text_cfg = cfg
            from transformers import AutoModelForCausalLM
            with init_empty_weights(): self.model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        if self.gguf:
            from gguf_source import GGUFSource; self.gsrc = GGUFSource(self.gguf, self.text_cfg, self.layout)
        self.model.eval(); self.layers = self.model.model.layers; self.n_layers = len(self.layers)
        self.layer_types = list(getattr(self.text_cfg, "layer_types", None) or ["default"] * self.n_layers)
        self._open = {}
        self.tied = (not self.gsrc.has_output()) if self.gguf else (self.ckpt_key("lm_head.weight") not in self.wmap)
        for name, _ in list(self.model.named_parameters()):
            if not name.startswith("model.layers."):
                if name == "lm_head.weight" and self.tied: continue
                set_module_tensor_to_device(self.model, name, device, value=self.load(name))
        if self.tied: self.model.lm_head.weight = self.model.model.embed_tokens.weight
        for name, buf in list(self.model.named_buffers()):
            if not name.startswith("model.layers."): set_module_tensor_to_device(self.model, name, device, value=buf)
        self.tok = AutoTokenizer.from_pretrained(path)

    # ---- names
    def ckpt_key(self, name):
        if self.layout == "qwen_vl": return "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        if self.layout == "gemma_vl": return "language_model." + name
        return name
    def extra_keys(self):
        """Checkpoint tensors outside the text model (vision tower, mtp head ...), to copy through or drop."""
        text = {self.ckpt_key(n) for n, _ in self.model.named_parameters()} | {self.ckpt_key(n) for n, _ in self.model.named_buffers()}
        return [k for k in self.wmap if k not in text]
    def load(self, name):
        if self.gguf: return self.gsrc.get(self.ckpt_key(name)).to(self.dtype)
        key = self.ckpt_key(name); f = self.wmap[key]
        if f not in self._open: self._open[f] = safe_open(f, "pt", device="cpu")
        return self._open[f].get_tensor(key).to(self.dtype)
    def layer_param_names(self, li): return [n for n, _ in self.model.named_parameters() if n.startswith(f"model.layers.{li}.")]
    @staticmethod
    def lin_names(layer):
        """The linear layers to quantize in a decoder layer: every nn.Linear wide enough on both sides (skips Qwen3.5's gates)."""
        return [n for n, m in layer.named_modules() if isinstance(m, nn.Linear) and m.in_features >= MIN_DIM and m.out_features >= MIN_DIM]
    # ---- layer streaming
    def load_layer(self, li):
        for n in self.layer_param_names(li): set_module_tensor_to_device(self.model, n, self.device, value=self.load(n))
        for n, b in self.layers[li].named_buffers(): set_module_tensor_to_device(self.layers[li], n, self.device, value=b)
        return self.layers[li]
    def free_layer(self, li):
        for n in self.layer_param_names(li): set_module_tensor_to_device(self.model, n, "meta")
        (torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    def offload_head(self):
        for n in ("model.embed_tokens.weight", "lm_head.weight"):
            try: set_module_tensor_to_device(self.model, n, "cpu", value=self.model.get_parameter(n).data)
            except Exception: pass
        if self.tied: self.model.lm_head.weight = self.model.model.embed_tokens.weight
        (torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    # ---- running
    def capture_kwargs(self, ids):
        """Run the model's own forward (masks, rotary embeddings) and keep the kwargs it passes to the FIRST layer of each
        layer type (hybrid models get a different attention mask per type). Every layer up to the last one needed is
        replaced by a pass-through stub (their weights are on the meta device), the last one raises. Returns {type: kwargs}."""
        firsts = {}
        for i, t in enumerate(self.layer_types): firsts.setdefault(t, i)
        last = max(firsts.values()); box = {}; saved = {}
        class Stub(nn.Module):
            def __init__(s, t, record, final): super().__init__(); s.t = t; s.record = record; s.final = final
            def forward(s, hidden_states, *args, **kw):
                if s.record: box[s.t] = kw
                if s.final: raise StopForward
                return hidden_states
        for i in range(last + 1): saved[i] = self.layers[i]; self.layers[i] = Stub(self.layer_types[i], firsts[self.layer_types[i]] == i, i == last)
        try:
            with torch.no_grad(): self.model(input_ids=ids.to(self.device), use_cache=False)
        except StopForward: pass
        finally:
            for i, m in saved.items(): self.layers[i] = m
        out = {}
        for t, kw in box.items():
            kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "use_cache", "output_attentions")}
            kw["use_cache"] = False; out[t] = kw
        self._kw = out; return out
    def kw_for(self, li, kws=None): return (kws or self._kw)[self.layer_types[li]]
    def embed(self, ids):
        with torch.no_grad(): return self.model.model.embed_tokens(ids.to(self.device))
    @staticmethod
    def run_layer(layer, h, kw):
        with torch.no_grad(): out = layer(h, **kw)
        return out[0] if isinstance(out, (tuple, list)) else out
    def logits(self, h):
        with torch.no_grad(): h = self.model.model.norm(h); return self.model.lm_head(h)

class ShardWriter:
    """Write a checkpoint shard by shard in the SOURCE layout (same key names and config as the input, so vLLM loads it
    exactly as it loads the original), with an index. Non-text tensors (vision tower) are copied through; `mtp.*` dropped."""
    def __init__(self, out, streamer):
        self.out = out; os.makedirs(out, exist_ok=True); self.wmap = {}; self.s = streamer; self.total = 0
    def write(self, name, tensors):
        f = f"{name}.safetensors"; tensors = {self.s.ckpt_key(k): v.detach().to("cpu").contiguous() for k, v in tensors.items()}
        save_file(tensors, os.path.join(self.out, f), metadata={"format": "pt"})
        for k, v in tensors.items(): self.wmap[k] = f; self.total += v.numel() * v.element_size()
    def finish(self, extra_cfg=None):
        extra = [k for k in self.s.extra_keys() if not k.startswith("mtp.")]
        if extra:
            byfile = {}
            for k in extra: byfile.setdefault(self.s.wmap[k], []).append(k)
            for i, (f, ks) in enumerate(sorted(byfile.items())):
                with safe_open(f, "pt") as sf: t = {k: sf.get_tensor(k) for k in ks}
                fn = f"extra{i:02d}.safetensors"; save_file(t, os.path.join(self.out, fn), metadata={"format": "pt"})
                for k, v in t.items(): self.wmap[k] = fn; self.total += v.numel() * v.element_size()
        json.dump({"metadata": {"total_size": self.total}, "weight_map": self.wmap}, open(os.path.join(self.out, "model.safetensors.index.json"), "w"), indent=1)
        cfg = json.load(open(os.path.join(self.s.path, "config.json"))); cfg.pop("quantization_config", None)
        cfg["torch_dtype"] = "float16"; cfg["dtype"] = "float16"
        if "text_config" in cfg: cfg["text_config"]["dtype"] = "float16"; cfg["text_config"].pop("torch_dtype", None)
        if extra_cfg: cfg.update(extra_cfg)
        json.dump(cfg, open(os.path.join(self.out, "config.json"), "w"), indent=2)
        for f in glob.glob(os.path.join(self.s.path, "*")):
            b = os.path.basename(f)
            if os.path.isfile(f) and not f.endswith(".safetensors") and b not in ("config.json", "model.safetensors.index.json"): shutil.copy(f, os.path.join(self.out, b))
