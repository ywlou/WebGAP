#!/usr/bin/env python3
"""P1: download the five locked public benchmarks. Does not overwrite old subsets."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from webgap.utils.io import ensure_dir, save_json

ROOT = Path("/data/WebGAP")
BENCH = ROOT / "data" / "benchmarks"
SURVEY = ROOT / "outputs" / "benchmark_survey"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", os.environ.get("WEBGAP_MODELS_DIR", "/data/models"))


def _as_image(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        raw = bytes(img)
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            try:
                return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            except Exception:
                return None
    if isinstance(img, dict) and "bytes" in img:
        return _as_image(img.get("bytes"))
    if isinstance(img, str):
        p = Path(img)
        if len(img) < 4096 and p.exists():
            return Image.open(p).convert("RGB")
        s = img.strip()
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        try:
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        except Exception:
            return None
    return None


def _parse_box_tag(text) -> list[float] | None:
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(text or ""))]
    return nums[:4] if len(nums) >= 4 else None


def _save_jpg(img: Image.Image, path: Path, quality: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=quality)


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def _write_meta(name: str, payload: dict) -> None:
    save_json(payload, SURVEY / f"{name}_metadata.json")
    print("[meta]", name, json.dumps({k: payload[k] for k in ("n", "bytes_gb", "path") if k in payload}, ensure_ascii=False))


def _abort_if_huge(name: str, actual_gb: float, planned_gb: float, mult: float = 2.0) -> None:
    if planned_gb > 0 and actual_gb > planned_gb * mult:
        raise RuntimeError(f"{name}: actual {actual_gb:.2f}GB > {mult}x planned {planned_gb}GB — stop")


def dump_visualwebbench() -> dict:
    from datasets import load_dataset

    out = ensure_dir(BENCH / "visualwebbench")
    img_dir = ensure_dir(out / "images_full")
    tasks = [
        "web_caption",
        "webqa",
        "heading_ocr",
        "element_ocr",
        "element_ground",
        "action_prediction",
        "action_ground",
    ]
    rows = []
    counts = {}
    for task in tasks:
        ds = load_dataset("visualwebbench/VisualWebBench", task, split="test")
        print("VWB", task, len(ds), ds.column_names, flush=True)
        n_ok = 0
        for i, ex in enumerate(ds):
            img = _as_image(ex.get("image"))
            if img is None:
                continue
            ip = img_dir / f"{task}_{i:05d}.jpg"
            _save_jpg(img, ip)
            gold = ex.get("answer")
            if hasattr(gold, "tolist"):
                gold = gold.tolist()
            options = ex.get("options")
            if options is not None and hasattr(options, "tolist"):
                options = options.tolist()
            bbox = ex.get("bbox")
            if bbox is not None and hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            isize = ex.get("image_size")
            if isize is not None and hasattr(isize, "tolist"):
                isize = isize.tolist()
            q = ex.get("question") or ex.get("elem_desc") or ex.get("instruction") or ""
            rows.append(
                {
                    "id": str(ex.get("id") or f"vwb_{task}_{i}"),
                    "task": task,
                    "task_type": str(ex.get("task_type") or task),
                    "website": str(ex.get("website") or ""),
                    "image": str(ip),
                    "image_size": isize,
                    "question": str(q),
                    "answer": gold,
                    "options": options,
                    "bbox": bbox,
                    "elem_desc": str(ex.get("elem_desc") or ""),
                    "instruction": str(ex.get("instruction") or ""),
                    "type": "vwb_official",
                }
            )
            n_ok += 1
        counts[task] = n_ok
    path = out / "full.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gb = _dir_gb(img_dir) + path.stat().st_size / (1024**3)
    _abort_if_huge("visualwebbench", gb, 1.18)
    meta = {
        "id": "visualwebbench",
        "source": "visualwebbench/VisualWebBench",
        "license": "apache-2.0",
        "n": len(rows),
        "by_task": counts,
        "path": str(path),
        "bytes_gb": round(gb, 3),
        "subset": False,
        "kept_old_eval_jsonl": (out / "eval.jsonl").exists(),
    }
    _write_meta("visualwebbench", meta)
    return meta


def dump_websrc_official(qa_per_page: int = 2, seed: int = 42) -> dict:
    from huggingface_hub import snapshot_download

    dest = ensure_dir(BENCH / "websrc_official" / "hf")
    print("snapshot X-LANCE/WebSRC_v1.0 ->", dest, flush=True)
    snapshot_download(
        repo_id="X-LANCE/WebSRC_v1.0",
        repo_type="dataset",
        local_dir=str(dest),
    )
    gb = _dir_gb(dest)
    _abort_if_huge("websrc_official", gb, 0.649)
    # Official release lives in zip; unpack then pack DEV subset.
    raw = ensure_dir(BENCH / "websrc_official" / "raw")
    import zipfile

    for zp in dest.glob("*.zip"):
        marker = raw / (zp.stem + ".unpacked")
        if marker.exists():
            continue
        print("unzip", zp.name, flush=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(raw)
        marker.write_text("ok\n")
    pack = Path(__file__).resolve().parent / "pack_websrc_official.py"
    if pack.exists() and (BENCH / "websrc_official" / "raw" / "release" / "dataset_split.csv").exists():
        import runpy

        runpy.run_path(str(pack), run_name="__main__")
        meta_path = SURVEY / "websrc_official_metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    csvs = list(dest.rglob("dataset.csv")) + list((BENCH / "websrc_official" / "raw").rglob("dataset.csv"))
    print("found dataset.csv", len(csvs), flush=True)
    pages: dict[str, list[dict]] = {}
    html_index: dict[str, Path] = {}
    img_index: dict[str, Path] = {}
    for p in dest.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".html", ".htm"}:
            html_index[p.stem] = p
            html_index[p.name] = p
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img_index[p.stem] = p
            img_index[p.name] = p
    n_csv = 0
    for csv_path in csvs:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                        "csv": str(csv_path),
                    }
                )
    rng = random.Random(seed)
    eval_rows = []
    img_dir = ensure_dir(BENCH / "websrc_official" / "eval_images")
    html_dir = ensure_dir(BENCH / "websrc_official" / "eval_html")
    page_ids = sorted(pages)
    for pid in page_ids:
        qs = pages[pid]
        rng.shuffle(qs)
        chosen = qs[:qa_per_page]
        html_src = html_index.get(pid) or html_index.get(f"{pid}.html")
        img_src = img_index.get(pid) or img_index.get(f"{pid}.png") or img_index.get(f"{pid}.jpg")
        html_path = None
        img_path = None
        if html_src:
            html_path = html_dir / f"{pid}.html"
            if not html_path.exists():
                html_path.write_bytes(html_src.read_bytes())
        if img_src:
            img_path = img_dir / f"{pid}.jpg"
            if not img_path.exists():
                try:
                    im = Image.open(img_src).convert("RGB")
                    _save_jpg(im, img_path)
                except Exception:
                    img_path = None
        for r in chosen:
            r["image"] = str(img_path) if img_path else ""
            r["html_path"] = str(html_path) if html_path else ""
            r["type"] = "websrc_pos"
            r["task"] = "websrc"
            eval_rows.append(r)
    eval_path = BENCH / "websrc_official" / "eval_dev_subset.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_html = sum(1 for r in eval_rows if r.get("html_path"))
    with_img = sum(1 for r in eval_rows if r.get("image"))
    meta = {
        "id": "websrc_official",
        "source": "X-LANCE/WebSRC_v1.0",
        "license": "cc-by-4.0",
        "dump_gb": round(gb, 3),
        "n_csv_rows": n_csv,
        "n_pages_indexed": len(pages),
        "n_eval": len(eval_rows),
        "qa_per_page": qa_per_page,
        "seed": seed,
        "with_html": with_html,
        "with_image": with_img,
        "subset": True,
        "subset_note": "DEV-style page-stratified 2 QA/page seed=42; not the hidden test leaderboard",
        "path": str(eval_path),
        "bytes_gb": round(_dir_gb(BENCH / "websrc_official"), 3),
        "n": len(eval_rows),
    }
    _write_meta("websrc_official", meta)
    return meta


def _free_gb(path: str = "/data") -> float:
    import shutil

    return shutil.disk_usage(path).free / (1024**3)


def _assert_disk(min_free_gb: float = 50.0) -> float:
    free = _free_gb()
    print(f"disk free {free:.1f}GB", flush=True)
    if free < min_free_gb:
        raise RuntimeError(f"disk free {free:.1f}GB < {min_free_gb}GB — stop")
    return free


def _hf_download_with_retry(repo_id: str, filename: str, attempts: int = 6) -> str:
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception as e:
            last = e
            wait = min(90, 8 * attempt)
            print(
                f"  download fail {filename} attempt {attempt}/{attempts}: {type(e).__name__}: {e}; sleep {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to download {filename}") from last


def dump_mind2web() -> dict:
    """Download TEST splits only. load_dataset(repo, split=...) would also fetch train-*-of-00027."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    Image.MAX_IMAGE_PIXELS = None
    out_root = ensure_dir(BENCH / "mind2web")
    splits = ("test_website", "test_task", "test_domain")
    api = HfApi()
    repo_files = api.list_repo_files("osunlp/Multimodal-Mind2Web", repo_type="dataset")
    wanted: list[str] = []
    for f in repo_files:
        if not f.endswith(".parquet"):
            continue
        if f.startswith("data/train-"):
            print("skip train file", f, flush=True)
            continue
        if any(f.startswith(f"data/{s}-") for s in splits):
            wanted.append(f)
    wanted = sorted(wanted)
    print("mind2web test parquet files", len(wanted), wanted, flush=True)
    local_files: dict[str, list[str]] = {s: [] for s in splits}
    for fn in wanted:
        _assert_disk(50.0)
        print("download", fn, flush=True)
        path = _hf_download_with_retry("osunlp/Multimodal-Mind2Web", fn)
        size_mb = Path(path).stat().st_size / (1024**2)
        print(f"  got {path} ({size_mb:.1f}MB)", flush=True)
        for s in splits:
            if fn.startswith(f"data/{s}-"):
                local_files[s].append(path)
                break

    counts = {}
    n_with_image = 0
    for split in splits:
        files = local_files[split]
        if not files:
            raise RuntimeError(f"no local parquet for {split}")
        print("packing", split, "files", len(files), flush=True)
        split_dir = ensure_dir(out_root / split)
        img_dir = ensure_dir(split_dir / "images")
        html_dir = ensure_dir(split_dir / "html")
        rows = []
        i = 0
        for fp in files:
            pf = pq.ParquetFile(fp)
            print("  parquet", fp, "rows", pf.metadata.num_rows, "schema", pf.schema_arrow.names, flush=True)
            for batch in pf.iter_batches(batch_size=8):
                names = list(batch.schema.names)
                cols = {n: batch.column(n) for n in names}
                for j in range(batch.num_rows):
                    ex = {n: cols[n][j].as_py() for n in names}
                    img = _as_image(ex.get("screenshot"))
                    ip = img_dir / f"{i:05d}.jpg"
                    if img is not None:
                        if not ip.exists():
                            _save_jpg(img, ip, quality=80)
                        n_with_image += 1
                    else:
                        ip = None
                    html = ex.get("cleaned_html") or ex.get("raw_html") or ""
                    hp = html_dir / f"{i:05d}.html"
                    if html:
                        hp.write_text(html if isinstance(html, str) else str(html), encoding="utf-8", errors="ignore")
                    op = ex.get("operation")
                    if not isinstance(op, (str, dict, list)):
                        op = str(op)
                    pos = ex.get("pos_candidates")
                    if pos is not None and hasattr(pos, "tolist"):
                        pos = pos.tolist()
                    neg = ex.get("neg_candidates")
                    if neg is not None and hasattr(neg, "tolist"):
                        neg = neg.tolist()
                    rows.append(
                        {
                            "id": str(ex.get("action_uid") or f"{split}_{i}"),
                            "annotation_id": str(ex.get("annotation_id") or ""),
                            "split": split,
                            "website": str(ex.get("website") or ""),
                            "domain": str(ex.get("domain") or ""),
                            "subdomain": str(ex.get("subdomain") or ""),
                            "question": str(ex.get("confirmed_task") or ""),
                            "target_action_index": str(ex.get("target_action_index") or ""),
                            "target_action_reprs": str(ex.get("target_action_reprs") or ""),
                            "operation": op,
                            "pos_candidates": pos,
                            "n_neg": len(neg) if isinstance(neg, list) else 0,
                            "image": str(ip) if ip else "",
                            "html_path": str(hp) if html else "",
                            "task": "mind2web",
                            "type": "dom_element",
                        }
                    )
                    i += 1
                    if i % 200 == 0:
                        _assert_disk(50.0)
                        print(f"  {split} {i}", flush=True)
        jp = split_dir / "eval.jsonl"
        with jp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
        print("wrote", split, len(rows), flush=True)
        gb_so_far = _dir_gb(out_root)
        _abort_if_huge("multimodal_mind2web", gb_so_far, 10.0, mult=2.5)
    gb = _dir_gb(out_root)
    _abort_if_huge("multimodal_mind2web", gb, 10.0, mult=2.5)
    meta = {
        "id": "multimodal_mind2web",
        "source": "osunlp/Multimodal-Mind2Web",
        "license": "OpenRAIL",
        "splits": counts,
        "n": sum(counts.values()),
        "n_with_image": n_with_image,
        "path": str(out_root),
        "bytes_gb": round(gb, 3),
        "subset": False,
        "train_downloaded": False,
    }
    _write_meta("multimodal_mind2web", meta)
    return meta


def _screenspot_split_from_name(name: str) -> str:
    n = name.lower()
    if "web" in n:
        return "web"
    if "mobile" in n:
        return "mobile"
    if "desktop" in n:
        return "desktop"
    return "other"


def _unpack_screenspot_images(raw: Path, dest: Path) -> Path:
    """Unzip screenspotv2_image.zip if present; return directory that contains PNG files."""
    import zipfile

    zips = list(raw.glob("*.zip")) + list(raw.rglob("screenspotv2_image.zip"))
    img_root = dest
    for zp in zips:
        print("unzip", zp, flush=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
    # images may land in dest/ or dest/images/ etc.
    pngs = list(dest.rglob("*.png")) + list(raw.rglob("*.png"))
    print("screenspot pngs", len(pngs), flush=True)
    return dest


def dump_screenspot_v2() -> dict:
    from huggingface_hub import snapshot_download

    out = ensure_dir(BENCH / "screenspot_v2")
    img_dir = ensure_dir(out / "images")
    raw = ensure_dir(out / "hf")
    print("snapshot OS-Copilot/ScreenSpot-v2", flush=True)
    snapshot_download(
        "OS-Copilot/ScreenSpot-v2",
        repo_type="dataset",
        local_dir=str(raw),
    )
    extracted = ensure_dir(raw / "extracted")
    _unpack_screenspot_images(raw, extracted)
    png_index: dict[str, Path] = {}
    for p in list(extracted.rglob("*")) + list(raw.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        png_index[p.name] = p
        png_index[p.stem] = p
    print("screenspot image index", len(png_index), flush=True)
    json_files = [p for p in raw.glob("*.json") if ".cache" not in p.parts]
    rows = []
    splits = {}
    i_global = 0
    for jf in sorted(json_files):
        split = _screenspot_split_from_name(jf.name)
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for ex in data:
            if not isinstance(ex, dict):
                continue
            i = i_global
            i_global += 1
            img_rel = str(ex.get("img_filename") or ex.get("image") or "")
            src = png_index.get(Path(img_rel).name) or png_index.get(Path(img_rel).stem)
            if src is None and img_rel:
                cand = extracted / img_rel
                if cand.exists():
                    src = cand
            ip = ""
            if src and Path(src).exists():
                ipath = img_dir / f"{split}_{i:05d}.jpg"
                if not ipath.exists():
                    try:
                        _save_jpg(Image.open(src).convert("RGB"), ipath)
                    except Exception as e:
                        print("img fail", src, e, flush=True)
                if ipath.exists():
                    ip = str(ipath)
            bbox = ex.get("bbox") or ex.get("gt_bbox")
            platform = str(ex.get("data_source") or "")
            rows.append(
                {
                    "id": f"ssv2_{split}_{i}",
                    "image": ip,
                    "question": ex.get("instruction") or ex.get("question") or "",
                    "bbox": bbox,
                    "platform": platform,
                    "split": split,
                    "elem_type": str(ex.get("data_type") or ex.get("ui_type") or ""),
                    "img_filename": img_rel,
                    "task": "screenspot_v2",
                    "type": "point_ground",
                }
            )
            splits[split] = splits.get(split, 0) + 1
    if not rows:
        raise RuntimeError("ScreenSpot-v2 could not be parsed")
    plats = {}
    n_with = 0
    for r in rows:
        plats[r.get("platform") or ""] = plats.get(r.get("platform") or "", 0) + 1
        n_with += int(bool(r.get("image")))
    print("screenspot splits", splits, "platforms", plats, "with_image", n_with, flush=True)
    path = out / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gb = _dir_gb(out)
    _abort_if_huge("screenspot_v2", gb, 2.5)
    meta = {
        "id": "screenspot_v2",
        "source": "OS-Copilot/ScreenSpot-v2",
        "license": "apache-2.0",
        "n": len(rows),
        "n_web": splits.get("web", 0),
        "n_with_image": n_with,
        "splits": splits,
        "platforms": plats,
        "path": str(path),
        "bytes_gb": round(gb, 3),
        "subset": False,
        "primary_report": "web split (screenspot_web_v2.json)",
    }
    _write_meta("screenspot_v2", meta)
    return meta


def dump_guiact() -> dict:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out = ensure_dir(BENCH / "guiact_web_single")
    json_path = Path(
        hf_hub_download(
            repo_id="yiye2023/GUIAct",
            filename="web-single_test_data.json",
            repo_type="dataset",
            local_dir=str(out / "hf"),
        )
    )
    img_parq = Path(
        hf_hub_download(
            repo_id="yiye2023/GUIAct",
            filename="web-single_test_images.parquet",
            repo_type="dataset",
            local_dir=str(out / "hf"),
        )
    )
    print("guiact json", json_path, "parquet", img_parq, flush=True)
    items = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("data") if isinstance(items.get("data"), list) else list(items.values())
    img_dir = ensure_dir(out / "images")
    pf = pq.ParquetFile(img_parq)
    print("guiact parquet schema", pf.schema_arrow, "rows", pf.metadata.num_rows, flush=True)
    img_map: dict[str, str] = {}
    row_i = 0
    for batch in pf.iter_batches(batch_size=16):
        names = list(batch.schema.names)
        id_name = next((c for c in ("__index_level_0__", "index_level_0", "image_id", "id") if c in names), None)
        b64_name = "base64" if "base64" in names else next((c for c in names if c != id_name), None)
        for j in range(batch.num_rows):
            key = str(batch.column(id_name)[j].as_py()) if id_name else str(row_i)
            blob = batch.column(b64_name)[j].as_py() if b64_name else None
            img = _as_image(blob)
            if img is not None:
                ip = img_dir / f"{key}.jpg"
                if not ip.exists():
                    _save_jpg(img, ip)
                img_map[str(key)] = str(ip)
                img_map[str(row_i)] = str(ip)
            row_i += 1
            if row_i % 100 == 0:
                print(f"  guiact decoded {row_i}/{pf.metadata.num_rows}", flush=True)
    print("guiact image keys", len(img_map), flush=True)
    rows = []
    for i, ex in enumerate(items):
        if not isinstance(ex, dict):
            continue
        image_id = str(ex.get("image_id") or ex.get("id") or i)
        uid = str(ex.get("uid") or image_id)
        labels = ex.get("actions_label") or []
        first = labels[0] if labels and isinstance(labels[0], dict) else {}
        el = first.get("element") if isinstance(first.get("element"), dict) else {}
        bbox = _parse_box_tag(el.get("absolute") or el.get("related"))
        action_name = str(first.get("name") or "")
        img_path = img_map.get(image_id) or img_map.get(uid) or img_map.get(str(i)) or ""
        isize = ex.get("image_size") or {}
        rows.append(
            {
                "id": uid,
                "image_id": image_id,
                "image": img_path,
                "question": str(ex.get("instruction") or ex.get("question") or ""),
                "bbox": bbox,
                "action": action_name,
                "width": (isize or {}).get("width") if isinstance(isize, dict) else None,
                "height": (isize or {}).get("height") if isinstance(isize, dict) else None,
                "task": "guiact_web_single",
                "type": "point_ground",
            }
        )
    path = out / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gb = _dir_gb(out)
    _abort_if_huge("guiact_web_single_test", gb, 1.5)
    meta = {
        "id": "guiact_web_single_test",
        "source": "yiye2023/GUIAct",
        "license": "CC-BY-4.0",
        "n": len(rows),
        "n_with_image": sum(1 for r in rows if r.get("image")),
        "n_with_bbox": sum(1 for r in rows if r.get("bbox")),
        "path": str(path),
        "bytes_gb": round(gb, 3),
        "subset": True,
        "subset_note": "official web-single TEST split only, not train",
    }
    _write_meta("guiact_web_single_test", meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default="",
        help="comma list: visualwebbench,websrc,mind2web,screenspot,guiact",
    )
    args = ap.parse_args()
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    ensure_dir(BENCH)
    ensure_dir(SURVEY)
    t0 = time.time()
    results = {}

    def run(key, fn):
        if wanted and key not in wanted:
            print("skip", key)
            return
        print("=" * 60, key, flush=True)
        results[key] = fn()

    run("visualwebbench", dump_visualwebbench)
    run("websrc", dump_websrc_official)
    run("screenspot", dump_screenspot_v2)
    run("guiact", dump_guiact)
    run("mind2web", dump_mind2web)  # largest last
    summary = {
        "elapsed_sec": round(time.time() - t0, 1),
        "results": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "bytes_gb", "path", "splits")} for k, v in results.items()},
        "total_gb": round(sum(_dir_gb(p) for p in BENCH.iterdir() if p.is_dir()), 3),
    }
    save_json(summary, SURVEY / "p1_download_summary.json")
    print("P1 done", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
