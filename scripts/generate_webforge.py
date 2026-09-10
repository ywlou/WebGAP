#!/usr/bin/env python3
"""Generate WebForge synthetic pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.constants import HELDOUT_TEMPLATES, TRAIN_TEMPLATES
from webgap.data.webforge import generate_all, generate_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-pages", type=int, default=None)
    p.add_argument("--diag-pages", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny split for debugging")
    p.add_argument("--hard-only", action="store_true", help="only (re)generate a hard split")
    p.add_argument("--medium-only", action="store_true")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--hard-pages", type=int, default=800)
    p.add_argument("--medium-pages", type=int, default=400)
    p.add_argument("--split-name", default=None)
    p.add_argument("--seed0", type=int, default=None)
    args = p.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.train_pages:
        cfg.data.train_pages = args.train_pages
    if args.diag_pages:
        cfg.data.diag_pages = args.diag_pages
    if args.smoke:
        cfg.data.train_pages = 24
        cfg.data.diag_pages = 12
        cfg.data.hard_pages = 8
    if args.hard_only or args.medium_only or args.long_only:
        from webgap.constants import HARD_TEMPLATES, LONG_TEMPLATES, MEDIUM_TEMPLATES
        from webgap.data.webforge import generate_split

        if args.hard_only:
            tpls, n, split, seed = HARD_TEMPLATES, args.hard_pages if not args.smoke else 8, args.split_name or "hard_struct", args.seed0 or 20260910
        elif args.medium_only:
            tpls, n, split, seed = MEDIUM_TEMPLATES, args.medium_pages if not args.smoke else 8, args.split_name or "medium_struct", args.seed0 or 30360910
        else:
            tpls, n, split, seed = LONG_TEMPLATES, 24 if not args.smoke else 4, args.split_name or "long_vis", args.seed0 or 40460910
        generate_split(
            cfg.data.webforge_dir,
            n,
            tpls,
            cfg.data.image_width,
            cfg.data.image_height,
            cfg.data.jpeg_quality,
            seed0=seed,
            split=split,
        )
        return
    generate_all(cfg.data)


if __name__ == "__main__":
    main()
