#!/usr/bin/env python3
"""P5 full eval, then P4 public zoo. Skip existing metrics."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/WebGAP")
LOG = ROOT / "outputs/intervention/p45.log"


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    env.setdefault("HF_HOME", "/data/models")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    run([sys.executable, str(ROOT / "scripts" / "eval_p5.py")])
    run([sys.executable, str(ROOT / "scripts" / "run_p4_zoo.py")])
    log("P5 then P4 finished")


if __name__ == "__main__":
    main()
