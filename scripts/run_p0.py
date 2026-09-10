#!/usr/bin/env python3
"""P0 driver: WebClash data + dual-order anchors + SSL/replay + HTML attach.

Stamps live in outputs/pipeline_stamps/. Delete a stamp to rerun that stage.
"""

from __future__ import annotations

import argparse
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


def run(cmd: list[str], stamp: str | None = None):
    STAMP.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    if stamp and (STAMP / stamp).exists():
        print(f"[skip] {stamp}", flush=True)
        return 0
    print("[run]", " ".join(cmd), flush=True)
    e = os.environ.copy()
    e["PYTHONPATH"] = str(SRC) + os.pathsep + e.get("PYTHONPATH", "")
    e["HF_ENDPOINT"] = e.get("HF_ENDPOINT", "https://hf-mirror.com")
    logf = LOG / f"{stamp or 'cmd'}.log"
    with logf.open("a", encoding="utf-8") as f:
        f.write("\n\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), env=e, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"[fail] {stamp or cmd} -> {p.returncode}  see {logf}", flush=True)
        raise SystemExit(p.returncode)
    if stamp:
        (STAMP / stamp).write_text("ok\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    py = [PY]

    run(py + ["scripts/test_dual_anchors.py"], stamp="test_dual")

    if not args.skip_data:
        hp, mp = (24, 16) if args.smoke else (800, 400)
        run(py + ["scripts/generate_webforge.py", "--hard-only", "--hard-pages", str(hp)], stamp="gen_hard_v2")
        run(py + ["scripts/generate_webforge.py", "--medium-only", "--medium-pages", str(mp)], stamp="gen_medium")
        run(py + ["scripts/generate_webforge.py", "--long-only"], stamp="gen_long")
        run(
            py
            + [
                "scripts/generate_webforge.py",
                "--hard-only",
                "--hard-pages",
                "240" if not args.smoke else "12",
                "--split-name",
                "hard_train",
                "--seed0",
                "77777",
            ],
            stamp="gen_hard_train_v2",
        )
        run(py + ["scripts/download_benchmarks.py", "--attach-html"], stamp="dl_html")

    if args.skip_train:
        return

    steps = "6" if args.smoke else None
    samples = "16" if args.smoke else None

    def train(run_name, extra):
        cmd = py + ["scripts/train.py", "--run-name", run_name] + extra
        if steps:
            cmd += ["--max-steps", steps, "--max-samples", samples, "--epochs", "1"]
        run(cmd, stamp=f"train_{run_name}")

    # Dual-order from the existing visual SFT (new gate / order embeddings).
    train(
        "webgap_dom",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dom",
            "--ckpt",
            "outputs/runs/webgap_sft/checkpoint",
            "--epochs",
            "1",
            "--max-samples",
            "1500" if not args.smoke else "16",
        ],
    )
    train(
        "webgap_dual",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dual",
            "--ckpt",
            "outputs/runs/webgap_sft/checkpoint",
            "--epochs",
            "1",
            "--max-samples",
            "1500" if not args.smoke else "16",
        ],
    )
    # Mix a little conflict + open replay, lower LR — aims to lift dom probe without VWB collapse.
    train(
        "webgap_dual_mix",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dual",
            "--ckpt",
            "outputs/runs/webgap_sft/checkpoint",
            "--mix-splits",
            "hard_train",
            "--replay-ratio",
            "0.15",
            "--lr-lora",
            "5e-5",
            "--lr-plugin",
            "5e-5",
            "--epochs",
            "1",
            "--max-samples",
            "1200" if not args.smoke else "16",
        ],
    )
    # Stage-1 SSL then a short low-LR SFT (negative-transfer control).
    train(
        "webgap_ssl",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dual",
            "--stage",
            "ssl",
            "--epochs",
            "1",
            "--max-samples",
            "800" if not args.smoke else "16",
        ],
    )
    train(
        "webgap_ssl_sft",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dual",
            "--ckpt",
            "outputs/runs/webgap_ssl/checkpoint",
            "--replay-ratio",
            "0.2",
            "--lr-lora",
            "5e-5",
            "--lr-plugin",
            "5e-5",
            "--epochs",
            "1",
            "--max-samples",
            "2000" if not args.smoke else "16",
        ],
    )
    # Learning curve on conflict pages (saves checkpoint-25/50/...).
    train(
        "webgap_dual_curve",
        [
            "--baseline",
            "webgap",
            "--anchor-mode",
            "dual",
            "--ckpt",
            "outputs/runs/webgap_sft/checkpoint",
            "--split",
            "hard_train",
            "--max-steps",
            "400" if not args.smoke else "6",
            "--max-samples",
            "800" if not args.smoke else "16",
            "--epochs",
            "1",
        ],
    )
    train(
        "lora_hard_curve",
        [
            "--baseline",
            "lora",
            "--ckpt",
            "outputs/runs/lora_sft/checkpoint",
            "--split",
            "hard_train",
            "--max-steps",
            "400" if not args.smoke else "6",
            "--max-samples",
            "800" if not args.smoke else "16",
            "--epochs",
            "1",
        ],
    )
    train(
        "graphtoken_hard_curve",
        [
            "--baseline",
            "graphtoken",
            "--ckpt",
            "outputs/runs/graphtoken_sft/checkpoint",
            "--split",
            "hard_train",
            "--max-steps",
            "400" if not args.smoke else "6",
            "--max-samples",
            "800" if not args.smoke else "16",
            "--epochs",
            "1",
        ],
    )

    n_eval = "80" if args.smoke else "1500"
    n_med = "80" if args.smoke else "800"
    run(py + ["scripts/eval_hard.py", "--split", "hard_struct", "--max-samples", n_eval], stamp="eval_webclash_hard")
    run(py + ["scripts/eval_hard.py", "--split", "medium_struct", "--max-samples", n_med], stamp="eval_webclash_med")
    run(py + ["scripts/eval_public.py"], stamp="eval_public_v2")
    run(py + ["scripts/plot_results.py"], stamp="plots_v2")
    run(py + ["scripts/write_report.py"], stamp="report_v2")


if __name__ == "__main__":
    main()
