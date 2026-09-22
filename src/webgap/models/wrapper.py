"""Attach WebGAPPlugin into a frozen Qwen3-VL via decoder-layer wrapping."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from webgap.config import ExperimentConfig
from webgap.models.plugin import GraphBatch, WebGAPPlugin


class _Ctx:
    gb: GraphBatch | None = None
    enabled: bool = True


def _unwrap_peft(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _language_layers(root: nn.Module):
    base = _unwrap_peft(root)
    # Qwen3VLForConditionalGeneration.model.language_model.layers
    if hasattr(base, "model") and hasattr(base.model, "language_model"):
        return base.model.language_model.layers
    if hasattr(base, "language_model"):
        return base.language_model.layers
    raise AttributeError("Cannot locate language_model.layers")


def _vision_module(root: nn.Module):
    base = _unwrap_peft(root)
    if hasattr(base, "model") and hasattr(base.model, "visual"):
        return base.model.visual
    return None


class WebGAPModel(nn.Module):
    def __init__(self, backbone: nn.Module, plugin: WebGAPPlugin, insert_every: int, cfg: ExperimentConfig):
        super().__init__()
        self.backbone = backbone
        self.plugin = plugin
        self.cfg = cfg
        self.ctx = _Ctx()
        self._hooks = []
        layers = _language_layers(backbone)
        n = len(layers)
        self.insert_layers = [i for i in range(n) if (i % insert_every == insert_every - 1)]
        self._patch_layers(layers)

    def _patch_layers(self, layers):
        plugin = self.plugin
        ctx = self.ctx

        for idx in self.insert_layers:
            layer = layers[idx]
            orig = layer.forward

            def make_wrapped(orig_fwd, layer_idx):
                def wrapped(*args, **kwargs):
                    hidden = orig_fwd(*args, **kwargs)
                    if not ctx.enabled or ctx.gb is None:
                        return hidden
                    if torch.is_tensor(hidden):
                        # Prefill only: decode uses KV cache already mixed with structure.
                        if hidden.shape[1] <= 1:
                            return hidden
                        return plugin(hidden, ctx.gb)
                    return hidden

                return wrapped

            layer.forward = make_wrapped(orig, idx)
            self._hooks.append(layer)

    def set_graph(self, gb: GraphBatch | None):
        self.ctx.gb = gb

    def enable(self, flag: bool = True):
        self.ctx.enabled = flag

    def freeze_backbone(self):
        vis = _vision_module(self.backbone)
        if vis is not None:
            for p in vis.parameters():
                p.requires_grad = False
        # peft already freezes base; keep vision frozen even if not peft
        try:
            base = _unwrap_peft(self.backbone)
            if hasattr(base, "model") and hasattr(base.model, "visual"):
                for p in base.model.visual.parameters():
                    p.requires_grad = False
        except Exception:
            pass

    def forward(self, **kwargs):
        gb = kwargs.pop("graph_batch", None)
        if gb is not None:
            self.set_graph(gb)
        return self.backbone(**kwargs)

    def generate(self, **kwargs):
        gb = kwargs.pop("graph_batch", None)
        if gb is not None:
            self.set_graph(gb)
        return self.backbone.generate(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if fn is not None:
            fn(**kwargs)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @property
    def config(self):
        return self.backbone.config

    def parameters_for_optim(self):
        plugin_params = [p for p in self.plugin.parameters() if p.requires_grad]
        lora_params = []
        other = []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                lora_params.append(p)
            else:
                other.append(p)
        return plugin_params, lora_params, other


def load_qwen3vl(cfg: ExperimentConfig, device: str = "cuda"):
    import os

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from webgap.constants import MODELS_ROOT

    os.environ.setdefault("HF_HOME", str(MODELS_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_ROOT / "hub"))
    dtype = torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" else torch.float16
    path = cfg.model.name_or_path
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    # pixel budget
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = cfg.model.max_pixels
            ip.min_pixels = cfg.model.min_pixels
        elif hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["longest_edge"] = cfg.model.max_pixels
            ip.size["shortest_edge"] = cfg.model.min_pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=True,
    )
    model.to(device)
    return model, processor


def attach_lora(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    lc = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lc)


def build_webgap(cfg: ExperimentConfig, device: str = "cuda", ckpt_dir: str | None = None) -> tuple[WebGAPModel, Any]:
    backbone, processor = load_qwen3vl(cfg, device=device)
    hidden = backbone.config.text_config.hidden_size
    need_lora = cfg.train.baseline in ("webgap", "lora", "graphtoken", "random_anchor")
    if ckpt_dir:
        from pathlib import Path
        from peft import PeftModel

        adapter = Path(ckpt_dir) / "backbone"
        if need_lora and adapter.exists():
            backbone = PeftModel.from_pretrained(backbone, str(adapter), is_trainable=True)
    elif need_lora:
        backbone = attach_lora(backbone, cfg)

    plugin = WebGAPPlugin(hidden, cfg.plugin).to(device=device, dtype=next(backbone.parameters()).dtype)
    if ckpt_dir:
        from pathlib import Path

        ppt = Path(ckpt_dir) / "plugin.pt"
        if ppt.exists() and cfg.train.baseline in ("webgap", "random_anchor"):
            blob = torch.load(ppt, map_location=device, weights_only=False)
            plugin.load_state_dict(blob["plugin"], strict=False)

    wrap = WebGAPModel(backbone, plugin, cfg.plugin.insert_every, cfg)
    wrap.freeze_backbone()
    if cfg.train.baseline in ("lora", "graphtoken", "zeroshot"):
        wrap.enable(False)
        for p in wrap.plugin.parameters():
            p.requires_grad = False
    if cfg.plugin.random_anchor_perm or cfg.train.baseline == "random_anchor":
        cfg.plugin.random_anchor_perm = True
    if cfg.train.gradient_checkpointing and ckpt_dir is None:
        wrap.gradient_checkpointing_enable()
        try:
            wrap.backbone.config.use_cache = False
        except Exception:
            pass
    wrap.to(device)
    return wrap, processor
