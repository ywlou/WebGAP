#!/usr/bin/env python3
"""Materialize WebSRC official DEV subset after unzipping train+dev.zip."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from webgap.utils.io import ensure_dir, save_json

ROOT = Path("/data/WebGAP/data/benchmarks/websrc_official")
RELEASE = ROOT / "raw" / "release"
SURVEY = Path("/data/WebGAP/outputs/benchmark_survey")


def main(qa_per_page: int = 2, seed: int = 42):
    split_csv = RELEASE / "dataset_split.csv"
    dev_sites = set()
    with split_csv.open() as f:
        for row in csv.DictReader(f):
            if (row.get("split") or "").strip().lower() == "dev":
                domain = (row["domain"] or "").strip()
                site = str(int(row["website"]))
                dev_sites.add((domain, site.zfill(2), site))
    print("dev sites", sorted(dev_sites))
    pages: dict[str, list] = {}
    html_index: dict[str, Path] = {}
    img_index: dict[str, Path] = {}
    n_csv = 0
    for domain, zsite, site in dev_sites:
        folder = RELEASE / domain / zsite
        if not folder.exists():
            folder = RELEASE / domain / site
        csv_path = folder / "dataset.csv"
        proc = folder / "processed_data"
        if not csv_path.exists():
            print("missing", csv_path)
            continue
        if proc.exists():
            for p in proc.iterdir():
                if p.suffix.lower() in {".html", ".htm"}:
                    html_index[p.stem] = p
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    img_index[p.stem] = p
        with csv_path.open(encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                n_csv += 1
                qid = str(row.get("id") or "")
                if len(qid) < 9:
                    continue
                page_id = qid[2:9]
                pages.setdefault(page_id, []).append(
                    {
                        "id": qid,
                        "question": row.get("question") or "",
                        "answer": row.get("answer") or "",
                        "element_id": row.get("element_id"),
                        "answer_start": row.get("answer_start"),
                        "page_id": page_id,
                        "domain": domain,
                        "website": zsite,
                    }
                )
    rng = random.Random(seed)
    img_dir = ensure_dir(ROOT / "eval_images")
    html_dir = ensure_dir(ROOT / "eval_html")
    eval_rows = []
    for pid in sorted(pages):
        qs = pages[pid]
        rng.shuffle(qs)
        html_src = html_index.get(pid)
        img_src = img_index.get(pid)
        html_path = ""
        img_path = ""
        if html_src:
            dest = html_dir / f"{pid}.html"
            if not dest.exists():
                dest.write_bytes(html_src.read_bytes())
            html_path = str(dest)
        if img_src:
            dest = img_dir / f"{pid}.jpg"
            if not dest.exists():
                Image.open(img_src).convert("RGB").save(dest, quality=85)
            img_path = str(dest)
        for r in qs[:qa_per_page]:
            r["image"] = img_path
            r["html_path"] = html_path
            r["type"] = "websrc_pos"
            r["task"] = "websrc"
            eval_rows.append(r)
    path = ROOT / "eval_dev_subset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "id": "websrc_official",
        "source": "X-LANCE/WebSRC_v1.0",
        "license": "cc-by-4.0",
        "n_csv_rows_dev": n_csv,
        "n_pages": len(pages),
        "n": len(eval_rows),
        "qa_per_page": qa_per_page,
        "seed": seed,
        "dev_sites": sorted(list(dev_sites)),
        "with_html": sum(1 for r in eval_rows if r.get("html_path")),
        "with_image": sum(1 for r in eval_rows if r.get("image")),
        "subset": True,
        "subset_note": "Official DEV websites from dataset_split.csv; 2 QA per page, seed 42. Not the hidden test leaderboard.",
        "path": str(path),
        "bytes_gb": round(sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) / 1024**3, 3),
    }
    save_json(meta, SURVEY / "websrc_official_metadata.json")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
