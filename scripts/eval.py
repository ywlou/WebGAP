#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.eval.infer import evaluate_webforge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="heldout")
    p.add_argument("--baseline", default=None)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--tag", default="eval")
    p.add_argument("--random-anchors", action="store_true")
    p.add_argument("--no-gaca", action="store_true")
    p.add_argument("--no-trb", action="store_true")
    p.add_argument("--anchor-mode", default=None, choices=["visual", "dom", "dual"])
    args = p.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.baseline:
        cfg.train.baseline = args.baseline
    if args.random_anchors:
        cfg.plugin.random_anchor_perm = True
        cfg.train.baseline = "random_anchor"
    if args.no_gaca:
        cfg.plugin.use_gaca = False
    if args.no_trb:
        cfg.plugin.use_trb = False
    if args.anchor_mode:
        cfg.plugin.anchor_mode = args.anchor_mode
    ckpt = args.ckpt
    if ckpt is None and cfg.train.baseline != "zeroshot":
        ckpt = str(Path(cfg.train.output_dir) / cfg.run_name / "checkpoint")
    evaluate_webforge(cfg, ckpt, split=args.split, max_samples=args.max_samples, tag=args.tag)


if __name__ == "__main__":
    main()
