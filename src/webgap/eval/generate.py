"""Unified greedy generation for frozen / LoRA / WebGAP and other MLLM families."""

from __future__ import annotations

import os
from typing import Callable

import torch
from PIL import Image

from webgap.config import ExperimentConfig
from webgap.constants import MODELS_ROOT
from webgap.models.wrapper import build_webgap, load_qwen3vl

# P3 zeroshot zoo. WebGAP plugin still only wraps Qwen3-VL.
FAMILY_QWEN3VL = "qwen3vl"
FAMILY_INTERNVL = "internvl"
FAMILY_LLAVA_OV = "llava_ov"
FAMILY_MINICPM = "minicpm"
KNOWN_FAMILIES = (FAMILY_QWEN3VL, FAMILY_INTERNVL, FAMILY_LLAVA_OV, FAMILY_MINICPM)


def _ensure_model_cache_env() -> None:
    os.environ.setdefault("HF_HOME", str(MODELS_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_ROOT / "hub"))


def infer_family(path: str, explicit: str | None = None) -> str:
    if explicit and explicit != "auto":
        if explicit not in KNOWN_FAMILIES:
            raise ValueError(f"unknown family {explicit!r}; expected one of {KNOWN_FAMILIES}")
        return explicit
    blob = path.lower().replace("_", "-")
    if "internvl" in blob:
        return FAMILY_INTERNVL
    if "llava" in blob or "onevision" in blob:
        return FAMILY_LLAVA_OV
    if "minicpm" in blob:
        return FAMILY_MINICPM
    return FAMILY_QWEN3VL


@torch.no_grad()
def generate_hf_chat(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    """Chat-template + processor path used by Qwen3-VL and InternVL3.5-HF."""
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
    tok = getattr(processor, "tokenizer", processor)
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None) or eos
    out = model.generate(
        **proc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


generate_qwen3vl = generate_hf_chat


_INTERNVL_NO_THINK = (
    "Answer the user directly in the requested format. "
    "Do not write <think> tags or a reasoning trace."
)


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    if text.lstrip().startswith("<think>"):
        return ""
    return text.strip()


@torch.no_grad()
def generate_internvl(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    """InternVL3.5-HF official path, thinking OFF (same greedy table as Instruct)."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": _INTERNVL_NO_THINK}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
    n = inputs["input_ids"].shape[1]
    think_id = getattr(processor, "tokenizer", processor).convert_tokens_to_ids("<think>")
    ban = [[think_id]] if isinstance(think_id, int) and think_id >= 0 else None
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        bad_words_ids=ban,
    )
    return _strip_think(processor.decode(out[0, n:], skip_special_tokens=True))


@torch.no_grad()
def generate_minicpm(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    msgs = [{"role": "user", "content": [image, prompt]}]
    tok = processor
    out = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tok,
        sampling=False,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return str(out).strip()


@torch.no_grad()
def generate_llava_ov(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    """Official HuggingFace LLaVA-OneVision chat template (greedy)."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
    n = inputs["input_ids"].shape[1]
    tok = getattr(processor, "tokenizer", processor)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


def generate_fn_for(family: str) -> Callable[..., str]:
    if family == FAMILY_MINICPM:
        return generate_minicpm
    if family == FAMILY_INTERNVL:
        return generate_internvl
    if family == FAMILY_LLAVA_OV:
        return generate_llava_ov
    return generate_hf_chat


def load_hf_vlm(cfg: ExperimentConfig, device: str = "cuda"):
    """Generic AutoModelForImageTextToText loader (InternVL-HF, etc.)."""
    _ensure_model_cache_env()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" else torch.float16
    path = cfg.model.name_or_path
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def load_llava_ov(cfg: ExperimentConfig, device: str = "cuda"):
    """Load the HuggingFace conversion of LLaVA-OneVision (not the LLaVA-NeXT original)."""
    _ensure_model_cache_env()
    import json
    from pathlib import Path

    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    path = cfg.model.name_or_path
    cfg_path = Path(path) / "config.json"
    if cfg_path.exists():
        arch = (json.loads(cfg_path.read_text()).get("architectures") or [""])[0]
        if arch == "LlavaQwenForCausalLM":
            raise RuntimeError(
                f"{path} is the lmms-lab LLaVA-NeXT checkpoint (LlavaQwenForCausalLM), "
                "which transformers cannot load. Use llava-hf/llava-onevision-qwen2-7b-ov-hf "
                "(same weights, official HF conversion)."
            )
    dtype = torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(path)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.to(device)
    model.eval()
    return model, processor


def load_minicpm(cfg: ExperimentConfig, device: str = "cuda"):
    """MiniCPM-V-4.5 remote code never calls post_init(); transformers 5.x requires all_tied_weights_keys."""
    _ensure_model_cache_env()
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    dtype = torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" else torch.float16
    path = cfg.model.name_or_path
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    orig = PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            tied = getattr(self, "_tied_weights_keys", None) or {}
            self.all_tied_weights_keys = tied if isinstance(tied, dict) else {k: k for k in tied}
        return orig(self, *args, **kwargs)

    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched
    try:
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=cfg.model.attn_implementation,
        )
    finally:
        PreTrainedModel._move_missing_keys_from_meta_to_device = orig
    if not hasattr(model, "all_tied_weights_keys"):
        model.post_init()
    model.to(device)
    model.eval()
    return model, tok


def load_eval_model(
    cfg: ExperimentConfig,
    ckpt: str | None,
    baseline: str,
    family: str = FAMILY_QWEN3VL,
):
    """P3 zeroshot: raw backbone. P4 WebGAP: wrapped plugin model (Qwen3-VL only)."""
    cfg.train.baseline = baseline
    if baseline in ("webgap", "lora", "graphtoken", "random_anchor"):
        if family != FAMILY_QWEN3VL:
            raise ValueError("WebGAP / LoRA / GraphToken wrap is Qwen3-VL-only in P3/P4")
        wrap, processor = build_webgap(cfg, device=cfg.device, ckpt_dir=ckpt)
        wrap.eval()
        return wrap, processor, True
    if family == FAMILY_QWEN3VL:
        backbone, processor = load_qwen3vl(cfg, device=cfg.device)
        backbone.eval()
        return backbone, processor, False
    if family == FAMILY_MINICPM:
        return (*load_minicpm(cfg, device=cfg.device), False)
    if family == FAMILY_LLAVA_OV:
        return (*load_llava_ov(cfg, device=cfg.device), False)
    return (*load_hf_vlm(cfg, device=cfg.device), False)
