"""Clean micro-batch training loop."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from webgap.config import ExperimentConfig
from webgap.data.collator import VLMCollator, WebForgeQADataset
from webgap.models.plugin import GraphBatch
from webgap.models.wrapper import build_webgap
from webgap.utils.io import ensure_dir, save_json
from webgap.utils.seed import set_seed


KEEP_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
    "labels",
    "position_ids",
}


def filter_batch(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k in KEEP_KEYS}


def move_graph_batch(gb: GraphBatch, device, dtype) -> GraphBatch:
    def _t(x, floating=False):
        if x is None:
            return None
        if floating:
            return x.to(device=device, dtype=dtype)
        return x.to(device)

    return GraphBatch(
        assignment=_t(gb.assignment, True),
        rel_idx=_t(gb.rel_idx),
        k_valid=_t(gb.k_valid),
        anchor_types=_t(gb.anchor_types),
        visual_mask=_t(gb.visual_mask),
        perm=None if gb.perm is None else gb.perm.to(device),
        assignment_dom=_t(gb.assignment_dom, True),
        rel_idx_dom=_t(gb.rel_idx_dom),
        k_valid_dom=_t(gb.k_valid_dom),
        anchor_types_dom=_t(gb.anchor_types_dom),
        order_ids=_t(gb.order_ids),
        order_ids_dom=_t(gb.order_ids_dom),
    )


def train(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.train.seed)
    device = cfg.device
    run_dir = ensure_dir(Path(cfg.train.output_dir) / cfg.run_name)
    cfg.save(run_dir / "config.yaml")

    model, processor = build_webgap(cfg, device=device, ckpt_dir=cfg.train.resume)
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        try:
            model.backbone.config.use_cache = False
        except Exception:
            pass
    plugin_params, lora_params, other = model.parameters_for_optim()
    groups = []
    if plugin_params:
        groups.append({"params": plugin_params, "lr": cfg.train.lr_plugin})
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.train.lr_lora})
    if other:
        groups.append({"params": other, "lr": cfg.train.lr_lora})
    if not groups:
        raise RuntimeError("No trainable parameters — check baseline/freezing.")

    ssl = cfg.train.stage == "ssl"
    cfg.plugin.compute_ssl = ssl
    split_dir = Path(cfg.data.webforge_dir) / (cfg.data.sft_split or "train")
    max_samples = cfg.data.ssl_pages if ssl else cfg.data.sft_samples
    if cfg.train.max_train_samples and cfg.train.max_train_samples > 0:
        max_samples = cfg.train.max_train_samples
    replay = float(getattr(cfg.train, "replay_ratio", 0.0) or 0.0)
    ds = WebForgeQADataset(
        split_dir / "index.jsonl",
        split_dir,
        max_samples=-1 if getattr(cfg.data, "mix_splits", "") else max_samples,
        seed=cfg.train.seed,
        graphtoken=cfg.train.baseline == "graphtoken",
        ssl=ssl,
        replay_ratio=replay,
    )
    extra = str(getattr(cfg.data, "mix_splits", "") or "").strip()
    if extra and not ssl:
        parts = [ds]
        for name in extra.split(","):
            name = name.strip()
            if not name:
                continue
            d2 = Path(cfg.data.webforge_dir) / name
            if (d2 / "index.jsonl").exists():
                parts.append(
                    WebForgeQADataset(
                        d2 / "index.jsonl",
                        d2,
                        max_samples=-1,
                        seed=cfg.train.seed + 1,
                        graphtoken=cfg.train.baseline == "graphtoken",
                        ssl=False,
                        replay_ratio=replay,
                    )
                )
        # flatten rows then cap
        rows = []
        for p in parts:
            rows.extend(p.rows)
        rng = __import__("random").Random(cfg.train.seed)
        rng.shuffle(rows)
        if max_samples and max_samples > 0:
            rows = rows[:max_samples]
        ds.rows = rows
    collator = VLMCollator(processor, max_k=cfg.plugin.max_anchors, max_seq_len=cfg.train.max_seq_len, train=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collator)

    accum = max(1, cfg.train.grad_accum)
    opt_steps_target = cfg.train.max_steps if cfg.train.max_steps > 0 else int(cfg.train.num_epochs * max(1, len(ds) / accum))
    warmup = max(1, int(opt_steps_target * cfg.train.warmup_ratio))
    opt = AdamW(groups, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, warmup, max(opt_steps_target, 1))

    use_plugin = cfg.train.baseline in ("webgap", "random_anchor")
    model.train()
    model.enable(use_plugin)
    dtype = next(model.plugin.parameters()).dtype
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] run={cfg.run_name} trainable={n_train/1e6:.2f}M samples={len(ds)} opt_steps={opt_steps_target} accum={accum}", flush=True)

    def _save_ckpt(tag: str):
        ck = ensure_dir(run_dir / tag)
        torch.save({"plugin": model.plugin.state_dict(), "cfg": cfg.to_dict()}, ck / "plugin.pt")
        try:
            model.backbone.save_pretrained(ck / "backbone")
            processor.save_pretrained(ck / "processor")
        except Exception as e:
            print("save_pretrained failed:", e, flush=True)
        return ck

    logs = []
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    opt_step = 0
    micro = 0
    loss_acc = 0.0
    epoch = 0
    extra_saves = {25, 50, 100, 200, 400}

    while opt_step < opt_steps_target:
        epoch += 1
        for batch in loader:
            gb = batch.pop("graph_batch")
            batch.pop("meta", None)
            batch = filter_batch(batch)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            gb = move_graph_batch(gb, device, dtype)
            model.set_graph(gb if use_plugin else None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                if ssl:
                    aux = model.plugin.last_aux
                    rec_l = aux.get("rec", out.logits.new_zeros(()))
                    con_l = aux.get("con", out.logits.new_zeros(()))
                    loss = cfg.plugin.rec_lambda * rec_l + cfg.plugin.con_lambda * con_l
                    if torch.is_tensor(loss) and loss.ndim > 0:
                        loss = loss.mean()
                    if (not torch.is_tensor(loss)) or float(loss.detach()) == 0.0:
                        if out.loss is not None:
                            loss = 0.05 * out.loss
                else:
                    loss = out.loss
                    if loss is None:
                        raise RuntimeError("Model did not return loss; labels missing.")
                    if use_plugin and cfg.plugin.use_erpr:
                        erpr = model.plugin.last_aux.get("erpr", loss.new_zeros(()))
                        loss = loss + cfg.plugin.erpr_lambda * erpr
                loss = loss / accum
            loss.backward()
            loss_acc += float(loss.item()) * accum
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.train.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % cfg.train.logging_steps == 0 or opt_step == 1:
                rec = {
                    "step": opt_step,
                    "loss": loss_acc / cfg.train.logging_steps,
                    "lr": sched.get_last_lr()[0],
                    "epoch": epoch,
                    "secs": round(time.time() - t0, 1),
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
                    "anchor_mode": getattr(cfg.plugin, "anchor_mode", "visual"),
                }
                if use_plugin:
                    rec["erpr"] = float(model.plugin.last_aux.get("erpr", torch.tensor(0.0)).detach().float().cpu())
                    rec["gate"] = float(model.plugin.last_aux.get("gate", torch.tensor(0.0)).detach().float().cpu())
                    rec["conflict"] = float(model.plugin.last_aux.get("conflict", torch.tensor(0.0)).detach().float().cpu())
                    rec["ssl_rec"] = float(model.plugin.last_aux.get("rec", torch.tensor(0.0)).detach().float().cpu())
                print(rec, flush=True)
                logs.append(rec)
                loss_acc = 0.0
            if opt_step in extra_saves or (cfg.train.save_steps > 0 and opt_step % cfg.train.save_steps == 0):
                _save_ckpt(f"checkpoint-{opt_step}")
                print(f"[train] snapshot checkpoint-{opt_step}", flush=True)
            if opt_step >= opt_steps_target:
                break
        if opt_step >= opt_steps_target:
            break

    ckpt = _save_ckpt("checkpoint")
    save_json(
        {"log": logs, "trainable_m": n_train / 1e6, "opt_steps": opt_step, "samples": len(ds), "secs": time.time() - t0},
        run_dir / "train_log.json",
    )
    print(f"[train] saved {ckpt} in {time.time()-t0:.0f}s", flush=True)
    return run_dir
