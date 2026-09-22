#!/usr/bin/env python3
"""P6: train dual+conflict-gate from webgap_dual, then hard_struct + P5.

Does not re-run P3/P4/P5 for other methods. Public P4 for the new ckpt is later.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "runs" / "webgap_dual_cg" / "checkpoint"


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    env.setdefault("HF_HOME", "/data/models")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    py = sys.executable

    if not CKPT.exists():
        _run(
            [
                py,
                str(ROOT / "scripts" / "train.py"),
                "--run-name",
                "webgap_dual_cg",
                "--baseline",
                "webgap",
                "--anchor-mode",
                "dual",
                "--conflict-gate",
                "--ckpt",
                "outputs/runs/webgap_dual/checkpoint",
                "--epochs",
                "1",
                "--max-samples",
                "1500",
            ],
            env,
        )
    else:
        print(f"[skip train] {CKPT}", flush=True)

    _run([py, str(ROOT / "scripts" / "eval_hard.py"), "--only", "webgap_dual_cg"], env)
    _run([py, str(ROOT / "scripts" / "eval_p5.py"), "--only", "webgap_dual_cg"], env)


if __name__ == "__main__":
    main()
