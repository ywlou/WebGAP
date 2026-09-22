#!/usr/bin/env python3
"""Generate P5 paired WebForge pages into outputs/intervention/data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.data.intervention import generate_intervention_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/intervention/data")
    ap.add_argument("--n-visual", type=int, default=48)
    ap.add_argument("--n-dom", type=int, default=48)
    ap.add_argument("--n-conflict", type=int, default=32)
    ap.add_argument("--n-curve", type=int, default=32)
    args = ap.parse_args()
    path = generate_intervention_split(
        args.out,
        n_visual=args.n_visual,
        n_dom=args.n_dom,
        n_conflict=args.n_conflict,
        n_curve=args.n_curve,
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
