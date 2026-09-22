#!/usr/bin/env python3
"""Evaluate Frozen / LoRA / GraphToken / WebGAP on P5 paired interventions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.eval.generate import load_eval_model
from webgap.eval.metrics import exact_match, normalize_answer
from webgap.eval.public_graph import generate_structured, make_anchor_perm
from webgap.utils.io import ensure_dir, load_jsonl, save_json

PERM_KINDS = ("identity", "local", "near", "global", "random")

METHODS = [
    ("zeroshot", "zeroshot", None, None),
    ("lora_sft", "lora", "outputs/runs/lora_sft/checkpoint", None),
    ("graphtoken_sft", "graphtoken", "outputs/runs/graphtoken_sft/checkpoint", None),
    ("webgap_sft", "webgap", "outputs/runs/webgap_sft/checkpoint", "visual"),
    ("webgap_dom", "webgap", "outputs/runs/webgap_dom/checkpoint", "dom"),
    ("webgap_dual", "webgap", "outputs/runs/webgap_dual/checkpoint", "dual"),
    ("webgap_dual_cg", "webgap", "outputs/runs/webgap_dual_cg/checkpoint", "dual"),
]


def _score_pairs(rows: list[dict]) -> dict:
    """Flip rate: same pair_id+probe, arm a vs b, gold differs and pred follows gold."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("family") not in {"visual_reorder", "dom_reorder"}:
            continue
        by[(r["pair_id"], r["probe"])][r["arm"]] = r
    n = n_gold_flip = n_pred_flip = n_follow = 0
    by_probe = defaultdict(lambda: {"n": 0, "gold_flip": 0, "pred_flip": 0, "follow": 0})
    for (pid, probe), arms in by.items():
        if "a" not in arms or "b" not in arms:
            continue
        a, b = arms["a"], arms["b"]
        n += 1
        gold_flip = normalize_answer(a["gold"]) != normalize_answer(b["gold"])
        pred_changed = not exact_match(a["pred"], b["pred"])
        follow = gold_flip and exact_match(a["pred"], a["gold"]) and exact_match(b["pred"], b["gold"])
        if gold_flip:
            n_gold_flip += 1
        if pred_changed:
            n_pred_flip += 1
        if follow:
            n_follow += 1
        rec = by_probe[probe]
        rec["n"] += 1
        rec["gold_flip"] += int(gold_flip)
        rec["pred_flip"] += int(pred_changed)
        rec["follow"] += int(follow)
    def _rate(x, d):
        return x / d if d else 0.0
    out = {
        "n_pairs": n,
        "gold_flip": n_gold_flip,
        "pred_changed": n_pred_flip,
        "follow_gold_when_flips": n_follow,
        "pred_changed_rate": _rate(n_pred_flip, n),
        "follow_rate_given_gold_flip": _rate(n_follow, n_gold_flip),
        "by_probe": {},
    }
    for probe, rec in by_probe.items():
        out["by_probe"][probe] = {
            **rec,
            "pred_changed_rate": _rate(rec["pred_flip"], rec["n"]),
            "follow_rate_given_gold_flip": _rate(rec["follow"], rec["gold_flip"]),
        }
    return out


def _conflict_metrics(rows: list[dict]) -> dict:
    by_probe = defaultdict(lambda: {"n": 0, "em": 0.0})
    for r in rows:
        if r.get("family") != "conflict":
            continue
        p = r.get("probe") or "other"
        by_probe[p]["n"] += 1
        by_probe[p]["em"] += float(exact_match(r["pred"], r["gold"]))
    out = {}
    for p, rec in by_probe.items():
        out[p] = {"n": rec["n"], "em": rec["em"] / rec["n"] if rec["n"] else 0.0}
    return out


def _curve_metrics(rows: list[dict]) -> dict:
    by_kind = defaultdict(lambda: {"n": 0, "em": 0.0})
    for r in rows:
        if r.get("family") != "anchor_curve":
            continue
        k = r.get("perm_kind") or "identity"
        by_kind[k]["n"] += 1
        by_kind[k]["em"] += float(exact_match(r["pred"], r["gold"]))
    return {k: {"n": v["n"], "em": v["em"] / v["n"] if v["n"] else 0.0} for k, v in by_kind.items()}


def eval_method(cfg, index: Path, run_name: str, baseline: str, ckpt: str | None, mode: str | None, max_samples: int) -> dict:
    if mode:
        cfg.plugin.anchor_mode = mode
    cfg.plugin.conflict_gate = "cg" in run_name
    cfg.run_name = run_name
    cfg.train.baseline = baseline
    model, processor, wrapped = load_eval_model(cfg, ckpt, baseline, family="qwen3vl")
    rows = list(load_jsonl(index))
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    preds = []
    rng = np.random.default_rng(0)
    for r in tqdm(rows, desc=run_name):
        img = Image.open(r["image"]).convert("RGB")
        family = r.get("family")
        if family == "anchor_curve" and baseline in ("webgap", "random_anchor"):
            kinds = PERM_KINDS
        elif family == "anchor_curve":
            kinds = ("identity",)
        else:
            kinds = (None,)
        for kind in kinds:
            perm = None
            if kind and kind != "identity":
                perm = make_anchor_perm(kind, cfg.plugin.max_anchors, cfg.plugin.max_anchors, rng)
            pred = generate_structured(
                model,
                processor,
                img,
                r["question"],
                device=cfg.device,
                max_new_tokens=32,
                max_k=cfg.plugin.max_anchors,
                baseline=baseline,
                nodes=r.get("nodes"),
                perm=perm,
            )
            rec = {
                "id": r["id"] if kind in (None, "identity") else f"{r['id']}_{kind}",
                "pair_id": r["pair_id"],
                "arm": r["arm"],
                "family": r["family"],
                "probe": r.get("probe"),
                "pred": pred,
                "gold": r["answer"],
                "perm_kind": kind or "na",
            }
            preds.append(rec)
    metrics = {
        "pairs": _score_pairs(preds),
        "conflict": _conflict_metrics(preds),
        "anchor_curve": _curve_metrics(preds),
        "n": len(preds),
        "baseline": baseline,
        "anchor_mode": cfg.plugin.anchor_mode,
    }
    out = ensure_dir(Path("outputs/intervention") / run_name)
    save_json({"metrics": metrics, "run_name": run_name}, out / "metrics.json")
    with (out / "preds.jsonl").open("w", encoding="utf-8") as f:
        for rec in preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="outputs/intervention/data/index.jsonl")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    cfg.model.name_or_path = str(Path("/data/models/Qwen/Qwen3-VL-8B-Instruct"))
    for run_name, baseline, ckpt, mode in METHODS:
        if args.only and args.only not in run_name:
            continue
        ck = str(Path(ckpt)) if ckpt else None
        if ck and not Path(ck).exists():
            print("[skip missing]", run_name)
            continue
        outp = Path("outputs/intervention") / run_name / "metrics.json"
        if outp.exists() and args.max_samples < 0:
            print("[skip]", run_name)
            continue
        eval_method(cfg, Path(args.index), run_name, baseline, ck, mode, args.max_samples)


if __name__ == "__main__":
    main()
