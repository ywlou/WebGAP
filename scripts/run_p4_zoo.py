#!/usr/bin/env python3
"""P4: Frozen is qwen3vl8b_zeroshot. Re-run LoRA / GraphToken / WebGAP on the P3 suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models")) / "Qwen" / "Qwen3-VL-8B-Instruct"

JOBS = [
    ("visualwebbench", None),
    ("screenspot_v2", "web"),
    ("websrc", None),
    ("guiact", None),
    ("mind2web", "test_website"),
    ("mind2web", "test_task"),
    ("mind2web", "test_domain"),
]

METHODS = [
    ("lora_sft_public", "lora", "outputs/runs/lora_sft/checkpoint", None),
    ("graphtoken_sft_public", "graphtoken", "outputs/runs/graphtoken_sft/checkpoint", None),
    ("webgap_sft_public", "webgap", "outputs/runs/webgap_sft/checkpoint", "visual"),
    ("webgap_dom_public", "webgap", "outputs/runs/webgap_dom/checkpoint", "dom"),
    ("webgap_dual_public", "webgap", "outputs/runs/webgap_dual/checkpoint", "dual"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Substring of run-name, e.g. webgap_sft")
    ap.add_argument("--skip-mind2web", action="store_true")
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    env.setdefault("HF_HOME", "/data/models")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    jobs = JOBS
    if args.skip_mind2web:
        jobs = [j for j in jobs if j[0] != "mind2web"]

    for run_name, baseline, ckpt, mode in METHODS:
        if args.only and args.only not in run_name:
            continue
        ck = ROOT / ckpt
        if not ck.exists():
            print(f"[skip missing ckpt] {run_name} {ck}", flush=True)
            continue
        out_root = ROOT / "outputs" / "mllm_baselines" / run_name
        for bench, subset in jobs:
            tag = bench if bench != "mind2web" else f"mind2web_{subset}"
            metrics_path = out_root / f"{tag}_metrics.json"
            if metrics_path.exists():
                print(f"[skip] {run_name}/{tag}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--benchmark",
                bench,
                "--run-name",
                run_name,
                "--baseline",
                baseline,
                "--ckpt",
                str(ck),
                "--model-path",
                str(MODEL),
                "--family",
                "qwen3vl",
            ]
            if mode:
                cmd.extend(["--anchor-mode", mode])
            if subset:
                cmd.extend(["--subset", subset])
            if args.max_samples and args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
