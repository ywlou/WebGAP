#!/usr/bin/env python3
"""Run the P3 zeroshot suite for one local model under /data/models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models"))

JOBS = [
    ("visualwebbench", None),
    ("screenspot_v2", "web"),
    ("websrc", None),
    ("guiact", None),
    ("mind2web", "test_website"),
    ("mind2web", "test_task"),
    ("mind2web", "test_domain"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-path", required=True, help="Must be under /data/models")
    ap.add_argument("--family", default="auto")
    ap.add_argument("--skip-mind2web", action="store_true")
    args = ap.parse_args()

    model = Path(args.model_path).resolve()
    root = MODELS_ROOT.resolve()
    if root not in model.parents and model != root:
        raise SystemExit(f"model-path {model} is not under {root}")

    jobs = JOBS
    if args.skip_mind2web:
        jobs = [j for j in jobs if j[0] != "mind2web"]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WEBGAP_MODELS_DIR", str(MODELS_ROOT))
    env.setdefault("HF_HOME", str(MODELS_ROOT))

    out_root = ROOT / "outputs" / "mllm_baselines" / args.run_name
    for bench, subset in jobs:
        tag = bench if bench != "mind2web" else f"mind2web_{subset}"
        metrics_path = out_root / f"{tag}_metrics.json"
        if metrics_path.exists():
            print(f"[skip] {tag} ({metrics_path.name} exists)", flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            bench,
            "--run-name",
            args.run_name,
            "--model-path",
            str(model),
            "--family",
            args.family,
        ]
        if subset:
            cmd.extend(["--subset", subset])
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
