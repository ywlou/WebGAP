#!/usr/bin/env python3
"""Resumable experiment driver. Skip stages whose stamp files exist."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PY = sys.executable
STAMP = ROOT / "outputs" / "pipeline_stamps"
LOG = ROOT / "logs"


def run(cmd: list[str], stamp: str | None = None, env: dict | None = None):
    STAMP.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    if stamp and (STAMP / stamp).exists():
        print(f"[skip] {stamp}", flush=True)
        return 0
    print("[run]", " ".join(cmd), flush=True)
    e = os.environ.copy()
    e["PYTHONPATH"] = str(SRC) + os.pathsep + e.get("PYTHONPATH", "")
    e["HF_ENDPOINT"] = e.get("HF_ENDPOINT", "https://hf-mirror.com")
    if env:
        e.update(env)
    logf = LOG / f"{stamp or 'cmd'}.log"
    with logf.open("a", encoding="utf-8") as f:
        f.write("\n\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), env=e, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"[fail] {cmd} -> {p.returncode}  see {logf}", flush=True)
        raise SystemExit(p.returncode)
    if stamp:
        (STAMP / stamp).write_text("ok\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-pages", type=int, default=None)
    args = ap.parse_args()

    cfg = "configs/default.yaml"
    python = [PY]
    if args.smoke:
        run(python + ["scripts/generate_webforge.py", "--smoke"], stamp="gen_smoke")
        run(
            python
            + [
                "scripts/train.py",
                "--run-name",
                "smoke_webgap",
                "--baseline",
                "webgap",
                "--max-steps",
                "4",
                "--max-samples",
                "8",
                "--epochs",
                "1",
            ],
            stamp="train_smoke",
        )
        run(
            python
            + [
                "scripts/eval.py",
                "--run-name",
                "smoke_webgap",
                "--ckpt",
                "outputs/runs/smoke_webgap/checkpoint",
                "--split",
                "heldout",
                "--max-samples",
                "8",
                "--tag",
                "smoke",
            ],
            stamp="eval_smoke",
        )
        print("SMOKE OK")
        return

    gen = [PY, "scripts/generate_webforge.py"]
    if args.train_pages:
        gen += ["--train-pages", str(args.train_pages)]
    run(gen, stamp="gen_full")

    run(python + ["scripts/download_benchmarks.py"], stamp="dl_bench")

    if not args.skip_train:
        # Main method
        run(
            python
            + [
                "scripts/train.py",
                "--run-name",
                "webgap_sft",
                "--baseline",
                "webgap",
                "--stage",
                "sft",
                "--epochs",
                "2",
            ],
            stamp="train_webgap",
        )
        run(
            python
            + [
                "scripts/train.py",
                "--run-name",
                "lora_sft",
                "--baseline",
                "lora",
                "--stage",
                "sft",
                "--epochs",
                "2",
            ],
            stamp="train_lora",
        )
        run(
            python
            + [
                "scripts/train.py",
                "--run-name",
                "graphtoken_sft",
                "--baseline",
                "graphtoken",
                "--stage",
                "sft",
                "--epochs",
                "2",
            ],
            stamp="train_graphtoken",
        )
        # ERPR-off retrain (ablation)
        run(
            python
            + [
                "scripts/train.py",
                "--run-name",
                "webgap_no_erpr",
                "--baseline",
                "webgap",
                "--stage",
                "sft",
                "--epochs",
                "1.5",
                "--no-erpr",
            ],
            stamp="train_no_erpr",
        )

    evals = [
        ("zeroshot", None, "zeroshot", []),
        ("webgap_sft", "outputs/runs/webgap_sft/checkpoint", "webgap", []),
        ("lora_sft", "outputs/runs/lora_sft/checkpoint", "lora", []),
        ("graphtoken_sft", "outputs/runs/graphtoken_sft/checkpoint", "graphtoken", []),
        ("webgap_no_erpr", "outputs/runs/webgap_no_erpr/checkpoint", "webgap", []),
        ("webgap_no_gaca", "outputs/runs/webgap_sft/checkpoint", "webgap", ["--no-gaca"]),
        ("webgap_no_trb", "outputs/runs/webgap_sft/checkpoint", "webgap", ["--no-trb"]),
        ("webgap_rand_anchor", "outputs/runs/webgap_sft/checkpoint", "random_anchor", ["--random-anchors"]),
    ]
    for name, ckpt, baseline, extra in evals:
        for split, n in [("heldout", 500), ("leaked_templates", 200)]:
            if split == "leaked_templates" and name not in ("webgap_sft", "lora_sft", "zeroshot"):
                continue
            cmd = python + [
                "scripts/eval.py",
                "--run-name",
                name,
                "--baseline",
                baseline,
                "--split",
                split,
                "--max-samples",
                str(n),
                "--tag",
                f"{name}_{split}",
            ]
            if ckpt:
                cmd += ["--ckpt", ckpt]
            cmd += extra
            run(cmd, stamp=f"eval_{name}_{split}")

    run(python + ["scripts/eval_public.py"], stamp="eval_public")
    run(python + ["scripts/efficiency.py"], stamp="efficiency")
    run(
        python + ["scripts/generate_webforge.py", "--hard-only", "--hard-pages", "320"],
        stamp="gen_hard",
    )
    run(python + ["scripts/eval_hard.py", "--max-samples", "400"], stamp="eval_hard")
    run(
        python
        + [
            "scripts/generate_webforge.py",
            "--hard-only",
            "--hard-pages",
            "240",
            "--split-name",
            "hard_train",
            "--seed0",
            "77777",
        ],
        stamp="gen_hard_train",
    )
    run(
        python
        + [
            "scripts/train.py",
            "--run-name",
            "webgap_hardmix",
            "--baseline",
            "webgap",
            "--epochs",
            "1",
            "--max-samples",
            "800",
            "--ckpt",
            "outputs/runs/webgap_sft/checkpoint",
            "--split",
            "hard_train",
        ],
        stamp="train_webgap_hardmix",
    )
    run(
        python
        + [
            "scripts/eval.py",
            "--run-name",
            "webgap_hardmix",
            "--ckpt",
            "outputs/runs/webgap_hardmix/checkpoint",
            "--baseline",
            "webgap",
            "--split",
            "hard_struct",
            "--max-samples",
            "400",
            "--tag",
            "webgap_hardmix_hard",
        ],
        stamp="eval_webgap_hardmix",
    )
    run(python + ["scripts/plot_results.py"], stamp="plots")
    run(python + ["scripts/write_report.py"], stamp="report")
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
