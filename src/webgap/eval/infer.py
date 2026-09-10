"""Greedy generation over a dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from webgap.config import ExperimentConfig
from webgap.data.collator import VLMCollator, WebForgeQADataset
from webgap.eval.metrics import score_predictions
from webgap.models.plugin import GraphBatch
from webgap.models.wrapper import build_webgap
from webgap.train.loop import move_graph_batch, filter_batch
from webgap.utils.io import ensure_dir, save_json


@torch.no_grad()
def evaluate_webforge(
    cfg: ExperimentConfig,
    ckpt_dir: str | Path | None,
    split: str = "heldout",
    max_samples: int = -1,
    tag: str = "eval",
) -> dict:
    device = cfg.device
    model, processor = build_webgap(cfg, device=device, ckpt_dir=str(ckpt_dir) if ckpt_dir else None)
    model.eval()
    use_plugin = cfg.train.baseline in ("webgap", "random_anchor")
    model.enable(use_plugin)

    index = Path(cfg.data.webforge_dir) / split / "index.jsonl"
    image_root = Path(cfg.data.webforge_dir) / split
    ds = WebForgeQADataset(
        index,
        image_root,
        max_samples=max_samples if max_samples > 0 else cfg.eval.max_samples,
        seed=cfg.train.seed,
        graphtoken=cfg.train.baseline == "graphtoken",
        ssl=False,
    )
    collator = VLMCollator(processor, max_k=cfg.plugin.max_anchors, max_seq_len=cfg.train.max_seq_len, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    rows = []
    dtype = next(model.plugin.parameters()).dtype
    for batch in tqdm(loader, desc=f"eval:{split}:{tag}"):
        gb: GraphBatch = batch.pop("graph_batch")
        meta = batch.pop("meta")
        batch = filter_batch(batch)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        gb = move_graph_batch(gb, device, dtype)
        if cfg.plugin.random_anchor_perm:
            gb.perm = torch.randperm(gb.assignment.size(-1), device=device)
        model.set_graph(gb if use_plugin else None)
        input_len = batch["input_ids"].shape[1]
        gen = model.generate(
            **batch,
            max_new_tokens=cfg.train.max_new_tokens_eval,
            do_sample=False,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        new_ids = gen[0, input_len:]
        pred = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        gold = meta[0]["answer"]
        rows.append(
            {
                "id": meta[0]["id"],
                "type": meta[0]["type"],
                "template": meta[0].get("template"),
                "probe": meta[0].get("probe", ""),
                "pred": pred,
                "gold": gold,
                "hops": 2 if meta[0]["type"] == "H5" else 1,
            }
        )
    metrics = score_predictions(rows)
    out_dir = ensure_dir(Path(cfg.train.output_dir) / cfg.run_name / "eval")
    save_json({"metrics": metrics, "n": len(rows), "split": split, "tag": tag}, out_dir / f"{tag}_{split}_metrics.json")
    with (out_dir / f"{tag}_{split}_preds.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def _strip_peft(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model
