#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.train.loop import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--baseline", default=None, choices=["webgap", "lora", "graphtoken", "zeroshot", "random_anchor"])
    p.add_argument("--stage", default=None, choices=["ssl", "sft"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-gaca", action="store_true")
    p.add_argument("--no-trb", action="store_true")
    p.add_argument("--no-erpr", action="store_true")
    p.add_argument("--ckpt", default=None, help="continue from adapter/plugin checkpoint")
    p.add_argument("--split", default=None, help="WebForge split folder to train on (default train)")
    p.add_argument("--anchor-mode", default=None, choices=["visual", "dom", "dual"])
    p.add_argument("--replay-ratio", type=float, default=None)
    p.add_argument("--mix-splits", default=None, help="comma-separated extra splits (e.g. hard_train)")
    p.add_argument("--lr-lora", type=float, default=None)
    p.add_argument("--lr-plugin", type=float, default=None)
    args = p.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.baseline:
        cfg.train.baseline = args.baseline
    if args.stage:
        cfg.train.stage = args.stage
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.max_samples is not None:
        cfg.train.max_train_samples = args.max_samples
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.no_gaca:
        cfg.plugin.use_gaca = False
    if args.no_trb:
        cfg.plugin.use_trb = False
    if args.no_erpr:
        cfg.plugin.use_erpr = False
    if args.ckpt:
        cfg.train.resume = args.ckpt
    if args.split:
        cfg.data.sft_split = args.split
    if args.anchor_mode:
        cfg.plugin.anchor_mode = args.anchor_mode
    if args.replay_ratio is not None:
        cfg.train.replay_ratio = args.replay_ratio
    if args.mix_splits is not None:
        cfg.data.mix_splits = args.mix_splits
    if args.lr_lora is not None:
        cfg.train.lr_lora = args.lr_lora
    if args.lr_plugin is not None:
        cfg.train.lr_plugin = args.lr_plugin
    train(cfg)


if __name__ == "__main__":
    main()
