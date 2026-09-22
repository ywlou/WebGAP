#!/usr/bin/env python3
"""After LLaVA zoo finishes: refresh README, smoke MiniCPM, run MiniCPM zoo.

Does not start P4. Public-set WebGAP/LoRA/GraphToken still need graph construction
inside evaluate.py (current path is zeroshot-only).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/WebGAP")
LLAVA_DONE = ROOT / "outputs/mllm_baselines/llava_ov_7b_zeroshot/mind2web_test_domain_metrics.json"
MINICPM = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models")) / "openbmb" / "MiniCPM-V-4_5"
LOG = ROOT / "outputs/mllm_baselines/p3_continue.log"


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_file(path: Path, label: str) -> None:
    log(f"wait {label}: {path}")
    while not path.exists():
        time.sleep(30)
    log(f"found {label}")


def pids_matching(pattern: str) -> list[str]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    self = str(os.getpid())
    return [p for p in r.stdout.split() if p and p != self]


def gpu_free_of(pattern: str) -> None:
    log(f"wait process gone: {pattern}")
    while pids_matching(pattern):
        time.sleep(20)
    log(f"gone: {pattern}")


def minicpm_complete() -> bool:
    weights = list(MINICPM.glob("*.safetensors")) + list(MINICPM.glob("*.bin"))
    return (MINICPM / "config.json").exists() and len(weights) > 0


def ensure_minicpm() -> None:
    if minicpm_complete():
        log(f"minicpm already at {MINICPM}")
        return
    dl = "download_mllm.py minicpm-v45"
    if pids_matching(dl):
        log("wait existing MiniCPM download")
        while pids_matching(dl):
            time.sleep(20)
        if not minicpm_complete():
            raise SystemExit("MiniCPM download process exited but weights are incomplete")
        return
    log("start MiniCPM download")
    run([sys.executable, str(ROOT / "scripts" / "download_mllm.py"), "minicpm-v45"])
    if not minicpm_complete():
        raise SystemExit(f"download finished but {MINICPM} has no weights")


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    env.setdefault("HF_HOME", "/data/models")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    wait_file(LLAVA_DONE, "llava domain metrics")
    gpu_free_of("evaluate.py --run-name llava_ov_7b_zeroshot")
    gpu_free_of("run_p3_zoo.py --run-name llava_ov_7b_zeroshot")
    time.sleep(5)
    run([sys.executable, str(ROOT / "scripts" / "write_p3_readme.py")])

    ensure_minicpm()
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            "screenspot_v2",
            "--run-name",
            "minicpm_v45_zeroshot_smoke",
            "--model-path",
            str(MINICPM),
            "--family",
            "minicpm",
            "--max-samples",
            "2",
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_p3_zoo.py"),
            "--run-name",
            "minicpm_v45_zeroshot",
            "--model-path",
            str(MINICPM),
            "--family",
            "minicpm",
        ]
    )
    run([sys.executable, str(ROOT / "scripts" / "write_p3_readme.py")])
    log("P3 MiniCPM zoo finished. Next is P4 (needs graph path in evaluate.py); do not auto-start.")


if __name__ == "__main__":
    main()
