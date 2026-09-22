#!/usr/bin/env python3
"""Unified evaluation entry (P2). Writes to outputs/mllm_baselines or outputs/structural_eval.

Does not overwrite outputs/runs/. Official VisualWebBench metrics are used for that
benchmark; short-answer EM is not applied to caption/grounding tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.eval.generate import generate_fn_for, infer_family, load_eval_model
from webgap.eval.grounding import SCREENSPOT_PROMPT, score_point_ground, score_websrc
from webgap.eval.public_graph import StructuredGenerate
from webgap.eval.metrics import exact_match, score_predictions, token_f1
from webgap.eval.vwb_official import build_prompt, max_new_tokens_for_task, score_vwb
from webgap.utils.io import ensure_dir, load_jsonl, save_json

BENCH = Path("/data/WebGAP/data/benchmarks")


class PredSink:
    """Append-only jsonl with resume-by-id. Metrics are still written only when the split finishes."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("id")
                if rid is not None:
                    self.done[str(rid)] = rec
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")
        if self.done:
            print(f"[resume] {path.name}: {len(self.done)} preds already on disk", flush=True)

    def cached(self, row_id) -> dict | None:
        if row_id is None:
            return None
        return self.done.get(str(row_id))

    def write(self, rec: dict) -> None:
        rid = rec.get("id")
        if rid is not None:
            self.done[str(rid)] = rec
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def _emit(sink: PredSink | None, rec: dict) -> dict:
    if sink is not None:
        sink.write(rec)
    return rec


def _load_rows(jsonl: Path, max_samples: int) -> list[dict]:
    rows = list(load_jsonl(jsonl))
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _open_image(path: str) -> Image.Image | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def _predict(generate, gen_model, processor, img, prompt, device: str, max_new_tokens: int, row: dict | None = None) -> str:
    if hasattr(generate, "bind_row"):
        generate.bind_row(row)
    return generate(gen_model, processor, img, prompt, device=device, max_new_tokens=max_new_tokens)


def _filter_platform(rows: list[dict], subset: str) -> list[dict]:
    if not subset or subset in {"all", "*"}:
        return rows
    key = subset.lower()
    out = []
    for r in rows:
        split = str(r.get("split") or "").lower()
        if split == key:
            out.append(r)
            continue
        blob = " ".join(str(r.get(k) or "") for k in ("split", "platform", "data_source", "elem_type", "id")).lower()
        if key in blob:
            out.append(r)
    if not out:
        raise RuntimeError(f"screenspot subset={subset!r} matched 0 rows (refusing to fall back to all)")
    return out


@torch.no_grad()
def run_visualwebbench(
    model, processor, device: str, max_samples: int, wrapped: bool, generate, sink: PredSink | None = None
) -> tuple[list[dict], dict]:
    src = BENCH / "visualwebbench" / "full.jsonl"
    if not src.exists():
        src = BENCH / "visualwebbench" / "eval.jsonl"
    rows = _load_rows(src, max_samples)
    preds = []
    gen_model = model.backbone if wrapped and hasattr(model, "backbone") else model
    for r in tqdm(rows, desc="vwb_official"):
        cached = sink.cached(r.get("id")) if sink else None
        if cached is not None:
            preds.append(cached)
            continue
        img = _open_image(r.get("image") or "")
        if img is None:
            continue
        prompt = build_prompt(r)
        task = r.get("task") or "webqa"
        pred = _predict(generate, gen_model, processor, img, prompt, device, max_new_tokens_for_task(task), r)
        preds.append(
            _emit(
                sink,
                {
                    "id": r.get("id"),
                    "task": task,
                    "pred": pred,
                    "gold": r.get("answer"),
                    "website": r.get("website"),
                },
            )
        )
    return preds, score_vwb(preds)


@torch.no_grad()
def run_screenspot(
    model, processor, device: str, max_samples: int, subset: str, wrapped: bool, generate, sink: PredSink | None = None
) -> tuple[list[dict], dict]:
    src = BENCH / "screenspot_v2" / "eval.jsonl"
    rows = _filter_platform(_load_rows(src, -1), subset)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    preds = []
    gen_model = model.backbone if wrapped and hasattr(model, "backbone") else model
    for r in tqdm(rows, desc=f"screenspot:{subset}"):
        cached = sink.cached(r.get("id")) if sink else None
        if cached is not None:
            preds.append(cached)
            continue
        img = _open_image(r.get("image") or "")
        if img is None:
            continue
        w, h = img.size
        prompt = SCREENSPOT_PROMPT.format(instruction=r.get("question") or "")
        pred = _predict(generate, gen_model, processor, img, prompt, device, 32, r)
        preds.append(
            _emit(
                sink,
                {
                    "id": r.get("id"),
                    "pred": pred,
                    "bbox": r.get("bbox"),
                    "platform": r.get("platform") or "",
                    "width": w,
                    "height": h,
                    "question": r.get("question"),
                },
            )
        )
    return preds, score_point_ground(preds)


@torch.no_grad()
def run_guiact(
    model, processor, device: str, max_samples: int, wrapped: bool, generate, sink: PredSink | None = None
) -> tuple[list[dict], dict]:
    src = BENCH / "guiact_web_single" / "eval.jsonl"
    rows = _load_rows(src, max_samples)
    preds = []
    n_skip_bbox = 0
    gen_model = model.backbone if wrapped and hasattr(model, "backbone") else model
    for r in tqdm(rows, desc="guiact"):
        bbox = r.get("bbox")
        if not bbox and isinstance(r.get("action"), dict):
            bbox = (r["action"] or {}).get("bbox")
        if not bbox:
            n_skip_bbox += 1
            continue
        cached = sink.cached(r.get("id")) if sink else None
        if cached is not None:
            preds.append(cached)
            continue
        img = _open_image(r.get("image") or "")
        if img is None:
            continue
        w, h = img.size
        prompt = SCREENSPOT_PROMPT.format(instruction=r.get("question") or "")
        pred = _predict(generate, gen_model, processor, img, prompt, device, 32, r)
        preds.append(
            _emit(
                sink,
                {
                    "id": r.get("id"),
                    "pred": pred,
                    "bbox": bbox,
                    "platform": "web",
                    "width": w,
                    "height": h,
                },
            )
        )
    metrics = score_point_ground(preds)
    metrics["n_skip_no_bbox"] = n_skip_bbox
    metrics["note"] = "Point-in-bbox on items with an absolute click box; multi-action/type-only items skipped."
    return preds, metrics


@torch.no_grad()
def run_websrc(
    model, processor, device: str, max_samples: int, wrapped: bool, generate, sink: PredSink | None = None
) -> tuple[list[dict], dict]:
    src = BENCH / "websrc_official" / "eval_dev_subset.jsonl"
    rows = _load_rows(src, max_samples)
    preds = []
    gen_model = model.backbone if wrapped and hasattr(model, "backbone") else model
    prompt_t = (
        "Look at the webpage screenshot. Answer the question with the shortest span from the page. "
        "Do not explain.\n\nQuestion: {q}"
    )
    for r in tqdm(rows, desc="websrc_pos"):
        cached = sink.cached(r.get("id")) if sink else None
        if cached is not None:
            preds.append(cached)
            continue
        img = _open_image(r.get("image") or "")
        if img is None:
            continue
        pred = _predict(
            generate,
            gen_model,
            processor,
            img,
            prompt_t.format(q=r.get("question") or ""),
            device,
            32,
            r,
        )
        preds.append(
            _emit(
                sink,
                {
                    "id": r.get("id"),
                    "pred": pred,
                    "gold": r.get("answer"),
                    "answer": r.get("answer"),
                    "element_id": r.get("element_id"),
                    "html_path": r.get("html_path"),
                },
            )
        )
    return preds, score_websrc(preds)


@torch.no_grad()
def run_mind2web(
    model, processor, device: str, max_samples: int, split: str, wrapped: bool, generate, sink: PredSink | None = None
) -> tuple[list[dict], dict]:
    src = BENCH / "mind2web" / split / "eval.jsonl"
    rows = _load_rows(src, max_samples)
    preds = []
    gen_model = model.backbone if wrapped and hasattr(model, "backbone") else model
    prompt_t = (
        "You are a web agent. Given the screenshot and the user task, output the NEXT action "
        "as a short string (CLICK/TYPE/SELECT and the target element). Do not explain.\n\n"
        "Task: {task}\nNext action:"
    )
    em = f1 = 0.0
    for r in tqdm(rows, desc=f"mind2web:{split}"):
        gold = str(r.get("target_action_reprs") or "")
        cached = sink.cached(r.get("id")) if sink else None
        if cached is not None:
            preds.append(cached)
            em += float(cached.get("em") or exact_match(str(cached.get("pred") or ""), gold))
            f1 += token_f1(str(cached.get("pred") or ""), gold)
            continue
        img = _open_image(r.get("image") or "")
        if img is None:
            continue
        pred = _predict(
            generate,
            gen_model,
            processor,
            img,
            prompt_t.format(task=r.get("question") or ""),
            device,
            48,
            r,
        )
        hit = exact_match(pred, gold)
        em += float(hit)
        f1 += token_f1(pred, gold)
        preds.append(
            _emit(
                sink,
                {
                    "id": r.get("id"),
                    "split": split,
                    "website": r.get("website"),
                    "domain": r.get("domain"),
                    "pred": pred,
                    "gold": gold,
                    "em": hit,
                },
            )
        )
    n = max(len(preds), 1)
    metrics = {
        "n": len(preds),
        "element_em": em / n,
        "action_f1": f1 / n,
        "split": split,
        "note": "Subset-safe step metric: match predicted next-action string to official target_action_reprs (not full task SR).",
    }
    return preds, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["visualwebbench", "screenspot_v2", "guiact", "websrc", "mind2web"])
    ap.add_argument("--run-name", default="qwen3vl8b_zeroshot")
    ap.add_argument("--baseline", default="zeroshot", choices=["zeroshot", "lora", "graphtoken", "webgap", "random_anchor"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model-path", default=None, help="Override configs/default.yaml model.name_or_path; must live under /data/models")
    ap.add_argument("--family", default="auto", choices=["auto", "qwen3vl", "internvl", "llava_ov", "minicpm"])
    ap.add_argument("--anchor-mode", default=None, help="visual|dom|dual; P4 WebGAP only")
    ap.add_argument("--conflict-gate", action="store_true", help="P6 dual conflict-aware mix")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--subset", default="web", help="screenspot platform filter; mind2web split name")
    ap.add_argument("--out-root", default="outputs/mllm_baselines")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    cfg.run_name = args.run_name
    cfg.train.baseline = args.baseline
    if args.model_path:
        cfg.model.name_or_path = args.model_path
    if args.anchor_mode:
        cfg.plugin.anchor_mode = args.anchor_mode
    if args.conflict_gate:
        cfg.plugin.conflict_gate = True
    family = infer_family(cfg.model.name_or_path, args.family)
    out = ensure_dir(Path(args.out_root) / args.run_name)
    tag = args.benchmark if args.benchmark != "mind2web" else f"mind2web_{args.subset}"
    metrics_path = out / f"{tag}_metrics.json"
    pred_path = out / f"{tag}_preds.jsonl"
    if metrics_path.exists() and args.max_samples <= 0:
        print(f"[skip] {metrics_path} already exists", flush=True)
        return

    generate = generate_fn_for(family)
    model, processor, wrapped = load_eval_model(cfg, args.ckpt, args.baseline, family=family)
    if wrapped:
        generate = StructuredGenerate(model, cfg, args.baseline)

    sink = PredSink(pred_path)
    try:
        if args.benchmark == "visualwebbench":
            preds, metrics = run_visualwebbench(
                model, processor, cfg.device, args.max_samples, wrapped, generate, sink
            )
        elif args.benchmark == "screenspot_v2":
            preds, metrics = run_screenspot(
                model, processor, cfg.device, args.max_samples, args.subset, wrapped, generate, sink
            )
        elif args.benchmark == "guiact":
            preds, metrics = run_guiact(
                model, processor, cfg.device, args.max_samples, wrapped, generate, sink
            )
        elif args.benchmark == "websrc":
            preds, metrics = run_websrc(
                model, processor, cfg.device, args.max_samples, wrapped, generate, sink
            )
        else:
            split = args.subset if args.subset in {"test_website", "test_task", "test_domain"} else "test_website"
            preds, metrics = run_mind2web(
                model, processor, cfg.device, args.max_samples, split, wrapped, generate, sink
            )
    finally:
        sink.close()

    save_json(
        {
            "metrics": metrics,
            "benchmark": args.benchmark,
            "run_name": args.run_name,
            "baseline": args.baseline,
            "model": cfg.model.name_or_path,
            "family": family,
            "anchor_mode": cfg.plugin.anchor_mode,
            "conflict_gate": bool(cfg.plugin.conflict_gate),
            "ckpt": args.ckpt,
            "n": len(preds),
            "subset": args.subset,
            "max_samples": args.max_samples,
        },
        metrics_path,
    )
    with pred_path.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
