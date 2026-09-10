#!/usr/bin/env python3
"""Prefill latency / peak memory microbenchmark (paper Table: efficiency)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.config import ExperimentConfig
from webgap.data.collator import VLMCollator, WebForgeQADataset
from webgap.models.wrapper import build_webgap
from webgap.train.loop import move_graph_batch, filter_batch
from webgap.utils.io import ensure_dir, save_json


def bench(tag: str, baseline: str, ckpt: str | None, n: int = 20, anchor_mode: str = "visual"):
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    cfg.train.baseline = baseline
    cfg.plugin.anchor_mode = anchor_mode
    cfg.run_name = "efficiency"
    model, processor = build_webgap(cfg, device="cuda", ckpt_dir=ckpt)
    model.eval()
    use_plugin = baseline in ("webgap", "random_anchor")
    model.enable(use_plugin)
    split = Path(cfg.data.webforge_dir) / "heldout"
    ds = WebForgeQADataset(split / "index.jsonl", split, max_samples=n, seed=0)
    collator = VLMCollator(processor, max_k=cfg.plugin.max_anchors, train=False)
    dtype = next(model.plugin.parameters()).dtype
    times = []
    mems = []
    seqs = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i in range(min(n, len(ds))):
        batch = collator([ds[i]])
        gb = batch.pop("graph_batch")
        batch.pop("meta", None)
        batch = filter_batch(batch)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        gb = move_graph_batch(gb, "cuda", dtype)
        model.set_graph(gb if use_plugin else None)
        seqs.append(int(batch["input_ids"].shape[1]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.generate(**batch, max_new_tokens=8, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mems.append(torch.cuda.max_memory_allocated() / 1024**3)
    rec = {
        "tag": tag,
        "baseline": baseline,
        "n": len(times),
        "latency_s_mean": sum(times) / len(times),
        "latency_s_p50": sorted(times)[len(times) // 2],
        "peak_mem_gb": max(mems),
        "seq_mean": sum(seqs) / len(seqs),
        "times": times,
        "seqs": seqs,
    }
    print(json.dumps({k: rec[k] for k in rec if k not in ("times", "seqs")}, indent=2))
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    out_dir = ensure_dir("/data/WebGAP/figures/data")
    rows = []
    n = 50
    jobs = [
        ("B0_zeroshot", "zeroshot", None, "visual"),
        ("B1_lora", "lora", "/data/WebGAP/outputs/runs/lora_sft/checkpoint", "visual"),
        ("M_webgap_visual", "webgap", "/data/WebGAP/outputs/runs/webgap_sft/checkpoint", "visual"),
        ("M_webgap_dual", "webgap", "/data/WebGAP/outputs/runs/webgap_dual/checkpoint", "dual"),
    ]
    for tag, baseline, ckpt, mode in jobs:
        if ckpt and not Path(ckpt).exists():
            print("skip missing", tag)
            continue
        rec = bench(tag, baseline, ckpt, n=n, anchor_mode=mode)
        rec["anchor_mode"] = mode
        rows.append(rec)
    save_json(rows, out_dir / "efficiency_raw.json")
    import csv

    keys = ["tag", "baseline", "latency_s_mean", "peak_mem_gb", "seq_mean", "n"]
    with open(out_dir / "efficiency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("wrote figures/data/efficiency.csv")


if __name__ == "__main__":
    main()
