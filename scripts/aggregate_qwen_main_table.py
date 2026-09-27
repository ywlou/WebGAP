#!/usr/bin/env python3
"""Write the Qwen3-VL-8B-Instruct main table from existing metric json.

Mind2Web element accuracy is read only from ``*_elem_metrics.json``.
Action-string F1 is not copied into that column.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data/WebGAP")
BASE = ROOT / "outputs" / "mllm_baselines"
OUT = ROOT / "outputs" / "main_table" / "qwen3vl8b_instruct.json"

# Main-table rows that already have checkpoints. Empty rows stay null.
ROWS = [
    ("Frozen", "qwen3vl8b_zeroshot"),
    ("LoRA", "lora_sft_public"),
    ("GraphToken", "graphtoken_sft_public"),
    ("HTML prompt", None),
    ("Layout Prompt", None),
    ("Set-of-Mark", None),
    ("UI token selection", None),
    ("Action-head attention", None),
    ("WebGAP-V", "webgap_sft_public"),
    ("WebGAP-Dual", "webgap_dual_public"),
]
SPLITS = ("test_task", "test_website", "test_domain")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def row_metrics(run: str | None) -> dict:
    empty = {
        "websrc_em": None,
        "vwb_element_grounding": None,
        "vwb_action_grounding": None,
        "screenspot_web": None,
        "mind2web_element_acc": {s: None for s in SPLITS},
    }
    if run is None:
        return empty
    folder = BASE / run
    websrc = _load(folder / "websrc_metrics.json")
    vwb = _load(folder / "visualwebbench_metrics.json")
    ss = _load(folder / "screenspot_v2_metrics.json")
    if websrc:
        empty["websrc_em"] = _pct(websrc["metrics"]["em"] * 100)
    if vwb:
        by = vwb["metrics"]["by_task"]
        empty["vwb_element_grounding"] = _pct(by["element_ground"]["accuracy"])
        empty["vwb_action_grounding"] = _pct(by["action_ground"]["accuracy"])
    if ss:
        empty["screenspot_web"] = _pct(ss["metrics"]["accuracy"])
    for split in SPLITS:
        elem = _load(folder / f"mind2web_{split}_elem_metrics.json")
        if elem:
            empty["mind2web_element_acc"][split] = _pct(elem["metrics"]["element_acc"] * 100)
    return empty


def main() -> None:
    table = {
        "backbone": "Qwen3-VL-8B-Instruct",
        "mind2web_element_protocol": (
            "positive candidates plus 50 negatives, seed 42, no DeBERTa ranker. "
            "test_task / test_website / test_domain are the three official test splits."
        ),
        "rows": [{"method": name, "run": run, **row_metrics(run)} for name, run in ROWS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
