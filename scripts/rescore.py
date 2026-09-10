#!/usr/bin/env python3
"""Re-score saved prediction jsonl files after metric/extractor changes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.eval.metrics import score_predictions
from webgap.utils.io import save_json


def main():
    root = Path("/data/WebGAP/outputs/runs")
    n = 0
    for pred in root.glob("**/eval/*_preds.jsonl"):
        rows = []
        with pred.open() as f:
            for line in f:
                rows.append(json.loads(line))
        if not rows or "gold" not in rows[0]:
            continue
        metrics = score_predictions(rows)
        out = pred.with_name(pred.name.replace("_preds.jsonl", "_metrics.json"))
        # keep split/tag if present
        blob = {"metrics": metrics, "n": len(rows), "rescored": True, "file": str(pred)}
        save_json(blob, out)
        print(pred.parent.parent.name, pred.name, "EM", round(metrics["em"], 4), "F1", round(metrics["f1"], 4), "StHR", round(metrics["sthr"], 4))
        n += 1
    print("rescored", n, "files")


if __name__ == "__main__":
    main()
