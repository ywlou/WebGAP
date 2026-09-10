#!/usr/bin/env python3
"""Evaluate checkpoints on public screenshot-QA jsonl (spatial-grid graph fallback)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.data.collator import VLMCollator
from webgap.data.public import ScreenshotQADataset
from webgap.eval.metrics import score_predictions
from webgap.models.wrapper import build_webgap
from webgap.train.loop import move_graph_batch, filter_batch
from webgap.utils.io import ensure_dir, save_json


@torch.no_grad()
def eval_jsonl(cfg: ExperimentConfig, jsonl: Path, ckpt: str | None, tag: str, max_samples: int = 400):
    if not jsonl.exists():
        print("missing", jsonl)
        return None
    model, processor = build_webgap(cfg, device=cfg.device, ckpt_dir=ckpt)
    model.eval()
    use_plugin = cfg.train.baseline in ("webgap", "random_anchor")
    model.enable(use_plugin)
    ds = ScreenshotQADataset(jsonl, max_samples=max_samples)
    if len(ds) == 0:
        return None
    collator = VLMCollator(processor, max_k=cfg.plugin.max_anchors, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    dtype = next(model.plugin.parameters()).dtype
    rows = []
    for batch in tqdm(loader, desc=tag):
        gb = batch.pop("graph_batch")
        meta = batch.pop("meta")
        batch = filter_batch(batch)
        batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        gb = move_graph_batch(gb, cfg.device, dtype)
        model.set_graph(gb if use_plugin else None)
        n = batch["input_ids"].shape[1]
        gen = model.generate(**batch, max_new_tokens=64, do_sample=False, use_cache=True)
        pred = processor.tokenizer.decode(gen[0, n:], skip_special_tokens=True).strip()
        rows.append({"id": meta[0]["id"], "type": meta[0]["type"], "pred": pred, "gold": meta[0]["answer"], "hops": 1})
    metrics = score_predictions(rows)
    out = ensure_dir(Path(cfg.train.output_dir) / cfg.run_name / "eval")
    save_json({"metrics": metrics, "tag": tag, "n": len(rows)}, out / f"{tag}_metrics.json")
    with (out / f"{tag}_preds.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(tag, json.dumps(metrics, indent=2))
    return metrics


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-samples", type=int, default=300)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    jobs = [
        ("zeroshot", None, "zeroshot", "visual"),
        ("lora_sft", "outputs/runs/lora_sft/checkpoint", "lora", "visual"),
        ("webgap_sft", "outputs/runs/webgap_sft/checkpoint", "webgap", "visual"),
        ("webgap_dual", "outputs/runs/webgap_dual/checkpoint", "webgap", "dual"),
        ("webgap_dual_mix", "outputs/runs/webgap_dual_mix/checkpoint", "webgap", "dual"),
        ("webgap_ssl_sft", "outputs/runs/webgap_ssl_sft/checkpoint", "webgap", "dual"),
    ]
    benches = [
        Path("/data/WebGAP/data/benchmarks/visualwebbench/eval.jsonl"),
        Path("/data/WebGAP/data/benchmarks/websrc/eval.jsonl"),
    ]
    for name, ckpt, baseline, mode in jobs:
        if args.only and args.only not in name:
            continue
        cfg.run_name = name
        cfg.train.baseline = baseline
        cfg.plugin.anchor_mode = mode
        ck = ckpt if ckpt and Path(ckpt).exists() else None
        if ckpt and ck is None:
            print("[skip missing]", name)
            continue
        for b in benches:
            tag = f"pub_{b.parent.name}_{name}"
            eval_jsonl(cfg, b, ck, tag, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
