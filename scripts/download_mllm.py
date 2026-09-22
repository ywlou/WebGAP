#!/usr/bin/env python3
"""Download P3 MLLM zoo weights into /data/models (never into the WebGAP repo)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models"))

# local_dir is MODELS_ROOT / local_rel
ZOO = {
    "qwen3vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
    "internvl35-8b": ("OpenGVLab/InternVL3_5-8B-HF", "OpenGVLab/InternVL3_5-8B-HF"),
    "internvl35-14b": ("OpenGVLab/InternVL3_5-14B-HF", "OpenGVLab/InternVL3_5-14B-HF"),
    "qwen3vl-30b-a3b": ("Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    # HF conversion of lmms-lab/llava-onevision-qwen2-7b-ov (same weights; original is LLaVA-NeXT, not transformers-native).
    "llava-ov-7b": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "minicpm-v45": ("openbmb/MiniCPM-V-4_5", "openbmb/MiniCPM-V-4_5"),
}


def _looks_complete(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    weights = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
    return (dest / "config.json").exists() and len(weights) > 0


def download_one(key: str) -> Path:
    if key not in ZOO:
        raise SystemExit(f"unknown model {key!r}; choose from {sorted(ZOO)}")
    repo_id, rel = ZOO[key]
    dest = MODELS_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _looks_complete(dest):
        print(f"[skip] {key} already at {dest}")
        return dest
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(MODELS_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_ROOT / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    print(f"[download] {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        resume_download=True,
    )
    if not _looks_complete(dest):
        raise RuntimeError(f"download finished but {dest} has no weights")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", choices=["all", *ZOO.keys()])
    args = ap.parse_args()
    keys = list(ZOO) if "all" in args.model else args.model
    for key in keys:
        dest = download_one(key)
        print(f"[ok] {key}: {dest}")


if __name__ == "__main__":
    main()
