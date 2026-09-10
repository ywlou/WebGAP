"""Load VisualWebBench / WebSRC-style public eval (screenshot + QA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from webgap.data.graph import spatial_grid_nodes
from webgap.data.html_dom import html_to_nodes, maybe_load_sidecar_boxes
from webgap.utils.io import load_jsonl


class ScreenshotQADataset(Dataset):
    """Generic {image, question, answer} jsonl with spatial-grid fallback graph."""

    def __init__(self, jsonl_path: str | Path, max_samples: int = -1):
        self.rows = list(load_jsonl(jsonl_path))
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        img = Image.open(r["image"]).convert("RGB")
        w, h = img.size
        nodes = r.get("nodes")
        if not nodes:
            html = r.get("html") or ""
            box_map = None
            if r.get("box_path"):
                box_map = maybe_load_sidecar_boxes(r["box_path"])
            if html:
                nodes = html_to_nodes(html, w, h, box_map)
            else:
                html_path = r.get("html_path")
                if html_path:
                    try:
                        html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
                        nodes = html_to_nodes(html, w, h, box_map)
                    except Exception:
                        nodes = None
        if not nodes:
            nodes = spatial_grid_nodes(w, h)
        return {
            "image": img,
            "question": r["question"],
            "answer": str(r.get("answer", "")),
            "type": r.get("type", "pub"),
            "id": r.get("id", str(idx)),
            "nodes": nodes,
            "orig_w": w,
            "orig_h": h,
            "page_id": r.get("id", str(idx)),
            "template": r.get("task", "public"),
            "heldout": True,
        }
