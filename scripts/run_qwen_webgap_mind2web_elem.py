#!/usr/bin/env python3
"""Qwen main-table gap: Mind2Web element choice for WebGAP-V and WebGAP-Dual only.

Does not train, does not run Frozen / LoRA / GraphToken / InternVL.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models")) / "Qwen" / "Qwen3-VL-8B-Instruct"
SPLITS = ("test_task", "test_website", "test_domain")
METHODS = (
    ("webgap_sft_public", "outputs/runs/webgap_sft/checkpoint", "visual"),
    ("webgap_dual_public", "outputs/runs/webgap_dual/checkpoint", "dual"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--only", default=None, help="visual or dual")
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    env.setdefault("HF_HOME", "/data/models")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for run_name, ckpt, mode in METHODS:
        if args.only and args.only not in mode and args.only not in run_name:
            continue
        ck = ROOT / ckpt
        if not ck.exists():
            raise SystemExit(f"missing checkpoint {ck}")
        for split in SPLITS:
            metrics = ROOT / "outputs" / "mllm_baselines" / run_name / f"mind2web_{split}_elem_metrics.json"
            if metrics.exists() and args.max_samples <= 0:
                print(f"[skip] {metrics}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--benchmark",
                "mind2web",
                "--protocol",
                "element",
                "--subset",
                split,
                "--run-name",
                run_name,
                "--baseline",
                "webgap",
                "--ckpt",
                str(ck),
                "--anchor-mode",
                mode,
                "--model-path",
                str(MODEL),
                "--family",
                "qwen3vl",
            ]
            if args.max_samples and args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
            print("+", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "aggregate_qwen_main_table.py")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
