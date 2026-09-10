"""Unified greedy generation for frozen / LoRA / WebGAP Qwen3-VL (and later other families)."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from webgap.config import ExperimentConfig
from webgap.models.wrapper import build_webgap, load_qwen3vl


@torch.no_grad()
def generate_qwen3vl(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}
    n = proc["input_ids"].shape[1]
    tok = processor.tokenizer
    out = model.generate(
        **proc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


def load_eval_model(cfg: ExperimentConfig, ckpt: str | None, baseline: str):
    """P3 zeroshot: raw backbone. P4 WebGAP: wrapped plugin model."""
    cfg.train.baseline = baseline
    if baseline in ("webgap", "lora", "graphtoken", "random_anchor"):
        wrap, processor = build_webgap(cfg, device=cfg.device, ckpt_dir=ckpt)
        wrap.eval()
        return wrap, processor, True
    backbone, processor = load_qwen3vl(cfg, device=cfg.device)
    backbone.eval()
    return backbone, processor, False
