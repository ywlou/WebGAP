#!/usr/bin/env python3
"""Evaluate on hard / medium WebClash splits (visual vs document order)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.eval.infer import evaluate_webforge


# name, ckpt, baseline, no_gaca, no_trb, rand, anchor_mode
JOBS = [
    ("zeroshot", None, "zeroshot", False, False, False, "visual"),
    ("lora_sft", "outputs/runs/lora_sft/checkpoint", "lora", False, False, False, "visual"),
    ("graphtoken_sft", "outputs/runs/graphtoken_sft/checkpoint", "graphtoken", False, False, False, "visual"),
    ("webgap_sft", "outputs/runs/webgap_sft/checkpoint", "webgap", False, False, False, "visual"),
    ("webgap_no_gaca", "outputs/runs/webgap_sft/checkpoint", "webgap", True, False, False, "visual"),
    ("webgap_rand_anchor", "outputs/runs/webgap_sft/checkpoint", "random_anchor", False, False, True, "visual"),
    ("webgap_dom", "outputs/runs/webgap_dom/checkpoint", "webgap", False, False, False, "dom"),
    ("webgap_dual", "outputs/runs/webgap_dual/checkpoint", "webgap", False, False, False, "dual"),
    ("webgap_dual_mix", "outputs/runs/webgap_dual_mix/checkpoint", "webgap", False, False, False, "dual"),
    ("webgap_ssl_sft", "outputs/runs/webgap_ssl_sft/checkpoint", "webgap", False, False, False, "dual"),
    ("webgap_dual_rand_vis", "outputs/runs/webgap_dual/checkpoint", "random_anchor", False, False, True, "dual"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="hard_struct")
    p.add_argument("--max-samples", type=int, default=1500)
    p.add_argument("--only", default=None, help="run name substring filter")
    args = p.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    for name, ckpt, baseline, no_gaca, no_trb, rand_anc, mode in JOBS:
        if args.only and args.only not in name:
            continue
        ck = ckpt if ckpt and Path(ckpt).exists() else None
        if name.startswith("webgap_dom") or name.startswith("webgap_dual"):
            if ck is None:
                print(f"[skip] {name}: no checkpoint yet", flush=True)
                continue
        cfg.run_name = name
        cfg.train.baseline = baseline
        cfg.plugin.use_gaca = not no_gaca
        cfg.plugin.use_trb = not no_trb
        cfg.plugin.random_anchor_perm = rand_anc
        cfg.plugin.anchor_mode = mode
        print(f"==== {args.split} {name} mode={mode} ====", flush=True)
        evaluate_webforge(cfg, ck, split=args.split, max_samples=args.max_samples, tag=f"{name}_{args.split}")


if __name__ == "__main__":
    main()
