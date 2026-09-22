#!/usr/bin/env python3
"""P8 data-efficiency on hard_train / hard_struct.

1. Re-eval existing hard_curve snapshots (steps 0/25/50/100/200/400).
2. Train+eval 1% and 10% of the 800-item hard_train mix (missing small-N).
Writes slim rows to outputs/data_efficiency/. Does not rerun P3/P4/P5.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "data_efficiency"
RUNS = ROOT / "outputs" / "runs"

# Same 800 hard_train items as P0 curve. 1% / 10% fill the left of the plot.
FRACTIONS = (
    (0.01, 8, 25),
    (0.10, 80, 50),
)

CURVES = (
    # run_dir, baseline, mode, step0_ckpt
    ("lora_hard_curve", "lora", "visual", "outputs/runs/lora_sft/checkpoint"),
    ("graphtoken_hard_curve", "graphtoken", "visual", "outputs/runs/graphtoken_sft/checkpoint"),
    ("webgap_dual_curve", "webgap", "dual", "outputs/runs/webgap_sft/checkpoint"),
)
STEPS = (0, 25, 50, 100, 200, 400)
EVAL_N = 400  # fixed subset via dataset seed; enough to rank methods


def _env() -> dict[str, str]:
    e = os.environ.copy()
    e.setdefault("PYTHONPATH", str(ROOT / "src"))
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    e.setdefault("WEBGAP_MODELS_DIR", "/data/models")
    e.setdefault("HF_HOME", "/data/models")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=_env())


def _metrics_path(run_name: str) -> Path:
    return RUNS / run_name / "eval" / f"p8_hard_struct_metrics.json"


def _eval(run_name: str, ckpt: str, baseline: str, mode: str) -> None:
    mp = _metrics_path(run_name)
    if mp.exists():
        print(f"[skip eval] {run_name}", flush=True)
        return
    if not (ROOT / ckpt).exists():
        print(f"[skip missing ckpt] {run_name} {ckpt}", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval.py"),
        "--run-name",
        run_name,
        "--ckpt",
        ckpt,
        "--baseline",
        baseline,
        "--anchor-mode",
        mode,
        "--split",
        "hard_struct",
        "--max-samples",
        str(EVAL_N),
        "--tag",
        "p8",
    ]
    _run(cmd)


def _train_frac(run_name: str, baseline: str, mode: str, ckpt0: str, n: int, steps: int) -> None:
    ck = RUNS / run_name / "checkpoint"
    if ck.exists():
        print(f"[skip train] {run_name}", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--run-name",
        run_name,
        "--baseline",
        baseline,
        "--anchor-mode",
        mode,
        "--ckpt",
        ckpt0,
        "--split",
        "hard_train",
        "--max-samples",
        str(n),
        "--max-steps",
        str(steps),
        "--epochs",
        "1",
    ]
    _run(cmd)


def _probe(metrics: dict) -> dict[str, float]:
    m = metrics.get("metrics") or metrics
    by = m.get("by_probe") or {}
    return {
        "em": float(m.get("em") or 0.0),
        "visual": float((by.get("visual") or {}).get("em") or 0.0),
        "dom": float((by.get("dom") or {}).get("em") or 0.0),
        "bind": float((by.get("bind") or {}).get("em") or 0.0),
        "n": int(m.get("n") or 0),
    }


def _collect() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RUNS.glob("*/eval/p8_hard_struct_metrics.json")):
        blob = json.loads(p.read_text())
        rec = {
            "run": p.parents[1].name,
            "path": str(p.relative_to(ROOT)),
            **_probe(blob),
        }
        rows.append(rec)
    (OUT / "points.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = ["run", "n", "em", "visual", "dom", "bind", "path"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (OUT / "points.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[p8] wrote {OUT / 'points.json'} n={len(rows)}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run_dir, baseline, mode, ckpt0 in CURVES:
        _eval(f"{run_dir}_p8_s0", ckpt0, baseline, mode)
        for st in STEPS:
            if st == 0:
                continue
            ck = f"outputs/runs/{run_dir}/checkpoint-{st}"
            _eval(f"{run_dir}_p8_s{st}", ck, baseline, mode)

    for run_dir, baseline, mode, ckpt0 in CURVES:
        short = run_dir.replace("_hard_curve", "").replace("_curve", "")
        for frac, n, steps in FRACTIONS:
            name = f"{short}_p8_n{n}"
            _train_frac(name, baseline, mode, ckpt0, n, steps)
            _eval(name, f"outputs/runs/{name}/checkpoint", baseline, mode)

    _collect()


if __name__ == "__main__":
    main()
