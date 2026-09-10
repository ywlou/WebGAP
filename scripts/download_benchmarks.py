#!/usr/bin/env python3
"""Download compact public web-understanding eval splits (no 100GB dumps)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from webgap.utils.io import ensure_dir

OUT = Path("/data/WebGAP/data/benchmarks")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _as_image(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str):
        raw = img.strip()
        if raw.startswith("http"):
            return None
        try:
            if "," in raw and raw.lower().startswith("data:"):
                raw = raw.split(",", 1)[1]
            buf = base64.b64decode(raw)
            return Image.open(io.BytesIO(buf)).convert("RGB")
        except Exception:
            return None
    return None


def _save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=85)


def dump_visualwebbench(limit_per_task: int = 80) -> int:
    from datasets import load_dataset

    out = ensure_dir(OUT / "visualwebbench")
    img_dir = ensure_dir(out / "images")
    tasks = ["webqa", "heading_ocr", "element_ocr", "web_caption", "element_ground", "action_prediction", "action_ground"]
    rows = []
    n = 0
    for task in tasks:
        try:
            ds = load_dataset("visualwebbench/VisualWebBench", task, split="test")
        except Exception as e:
            print("skip", task, e)
            continue
        print("VWB", task, len(ds), ds.column_names)
        for i, ex in enumerate(ds):
            if i >= limit_per_task:
                break
            img = _as_image(ex.get("image"))
            if img is None:
                continue
            q = ex.get("question") or ex.get("prompt") or "Describe this webpage."
            a = ex.get("answer") or ""
            if isinstance(a, list):
                a = a[0] if a else ""
            ip = img_dir / f"{task}_{i:05d}.jpg"
            _save_image(img, ip)
            rows.append(
                {
                    "id": f"vwb_{task}_{i}",
                    "image": str(ip),
                    "question": str(q),
                    "answer": str(a),
                    "task": task,
                    "type": "pub",
                }
            )
            n += 1
    with (out / "eval.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("VisualWebBench wrote", n)
    return n


def dump_websrc(limit: int = 800) -> int:
    from datasets import load_dataset

    out = ensure_dir(OUT / "websrc")
    img_dir = ensure_dir(out / "images")
    ds = None
    for split in ("dev", "train"):
        try:
            ds = load_dataset("rootsautomation/websrc", split=split)
            print("loaded websrc", split, len(ds))
            break
        except Exception as e:
            print("fail websrc", split, e)
    if ds is None:
        return 0
    rows = []
    n = 0
    for i, ex in enumerate(ds):
        if n >= limit:
            break
        img = _as_image(ex.get("image"))
        if img is None:
            continue
        q = ex.get("question") or ""
        a = ex.get("answer") or ""
        if isinstance(a, list):
            a = a[0] if a else ""
        ip = img_dir / f"{i:05d}.jpg"
        _save_image(img, ip)
        rows.append(
            {
                "id": f"websrc_{i}",
                "image": str(ip),
                "question": str(q),
                "answer": str(a),
                "task": "websrc",
                "type": "pub",
                "page_id": str(ex.get("page_id") or ""),
                "domain": str(ex.get("domain") or ""),
            }
        )
        n += 1
        if n % 100 == 0:
            print("websrc", n)
    with (out / "eval.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("WebSRC wrote", n)
    return n


def attach_websrc_html() -> int:
    """Join real HTML onto websrc/eval.jsonl when files or HF dumps are available."""
    eval_path = OUT / "websrc" / "eval.jsonl"
    if not eval_path.exists():
        print("no websrc eval.jsonl")
        return 0
    html_dir = ensure_dir(OUT / "websrc" / "html")
    rows = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    # 1) local html/{page_id}.html
    n = 0
    for r in rows:
        pid = str(r.get("page_id") or "")
        cand = html_dir / f"{pid}.html"
        if pid and cand.exists():
            r["html_path"] = str(cand)
            n += 1
    # 2) try HF dataset X-LANCE/WebSRC_v1.0 (may include html)
    if n == 0:
        try:
            from datasets import load_dataset

            ds = load_dataset("X-LANCE/WebSRC_v1.0", split="dev")
            print("X-LANCE/WebSRC_v1.0 columns", ds.column_names)
            by_id = {}
            for ex in ds:
                html = ex.get("html") or ex.get("HTML") or ""
                pid = str(ex.get("page_id") or "")
                if html and pid:
                    by_id[pid] = html
            for r in rows:
                pid = str(r.get("page_id") or "")
                if pid in by_id:
                    p = html_dir / f"{pid}.html"
                    if not p.exists():
                        p.write_text(by_id[pid], encoding="utf-8")
                    r["html_path"] = str(p)
                    n += 1
        except Exception as e:
            print("X-LANCE html attach skipped:", e)
    # 3) last resort: write a minimal HTML *from stored question/answer only* — not used.
    with eval_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("attached html for", n, "/", len(rows), "websrc rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vwb", action="store_true")
    ap.add_argument("--skip-websrc", action="store_true")
    ap.add_argument("--attach-html", action="store_true", help="try to join real WebSRC HTML onto eval.jsonl")
    args = ap.parse_args()
    ensure_dir(OUT)
    n1 = n2 = 0
    vwb_ok = (OUT / "visualwebbench" / "eval.jsonl").exists()
    src_ok = (OUT / "websrc" / "eval.jsonl").exists()
    if not args.skip_vwb and not vwb_ok:
        n1 = dump_visualwebbench()
    elif vwb_ok:
        print("keep existing visualwebbench")
    if not args.skip_websrc and not src_ok:
        n2 = dump_websrc()
    elif src_ok:
        print("keep existing websrc")
    if args.attach_html:
        attach_websrc_html()
    print({"visualwebbench": n1, "websrc": n2})
    sys.exit(0)


if __name__ == "__main__":
    main()
