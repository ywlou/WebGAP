"""Build a public-screenshot graph and run greedy decode with STAR / GraphToken."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from webgap.constants import IMAGE_TOKEN_ID_QWEN3VL
from webgap.data.collator import build_assignment_for_ids
from webgap.data.graph import build_graph, serialize_graph_text, spatial_grid_nodes
from webgap.data.html_dom import html_to_nodes, maybe_load_sidecar_boxes
from webgap.models.plugin import GraphBatch
from webgap.train.loop import move_graph_batch


def nodes_for_screenshot(
    image: Image.Image,
    html_path: str | None = None,
    html: str | None = None,
    box_path: str | None = None,
    nodes: list[dict] | None = None,
) -> list[dict]:
    if nodes:
        return nodes
    w, h = image.size
    box_map = maybe_load_sidecar_boxes(box_path) if box_path else None
    blob = html or ""
    if not blob and html_path:
        p = Path(html_path)
        if p.exists():
            blob = p.read_text(encoding="utf-8", errors="ignore")
    if blob:
        try:
            out = html_to_nodes(blob, w, h, box_map)
            if out:
                return out
        except Exception:
            pass
    return spatial_grid_nodes(w, h)


def graph_batch_from_proc(
    proc: dict,
    nodes: list[dict],
    orig_w: int,
    orig_h: int,
    max_k: int,
    image_token_id: int,
) -> GraphBatch:
    ids = proc["input_ids"][0]
    grid = proc.get("image_grid_thw")
    gthw = grid[0] if grid is not None else None
    pack = build_assignment_for_ids(
        ids.detach().cpu(),
        gthw.detach().cpu() if torch.is_tensor(gthw) else gthw,
        nodes,
        orig_w,
        orig_h,
        max_k,
        image_token_id=image_token_id,
    )
    return GraphBatch(
        assignment=torch.tensor(np.asarray(pack["assign"])[None], dtype=torch.float32),
        rel_idx=torch.tensor(np.asarray(pack["rel"])[None], dtype=torch.long),
        k_valid=torch.tensor([int(pack["k"])], dtype=torch.long),
        anchor_types=torch.tensor(np.asarray(pack["types"])[None], dtype=torch.long),
        visual_mask=torch.tensor(np.asarray(pack["vis_mask"])[None], dtype=torch.bool),
        assignment_dom=torch.tensor(np.asarray(pack["assign_dom"])[None], dtype=torch.float32),
        rel_idx_dom=torch.tensor(np.asarray(pack["rel_dom"])[None], dtype=torch.long),
        k_valid_dom=torch.tensor([int(pack["k_dom"])], dtype=torch.long),
        anchor_types_dom=torch.tensor(np.asarray(pack["types_dom"])[None], dtype=torch.long),
        order_ids=torch.tensor(np.asarray(pack["order"])[None], dtype=torch.long),
        order_ids_dom=torch.tensor(np.asarray(pack["order_dom"])[None], dtype=torch.long),
    )


def make_anchor_perm(kind: str, k_valid: int, max_k: int, rng: np.random.Generator) -> list[int]:
    """Identity / local swap / cycle / reverse / uniform random over the first k_valid columns."""
    k = max(int(k_valid), 1)
    k = min(k, max_k)
    inner = list(range(k))
    if kind == "identity":
        pass
    elif kind == "local":
        for i in range(0, k - 1, 2):
            inner[i], inner[i + 1] = inner[i + 1], inner[i]
    elif kind == "near":
        if k > 1:
            inner = inner[1:] + inner[:1]
    elif kind == "global":
        inner = inner[::-1]
    elif kind == "random":
        rng.shuffle(inner)
    else:
        raise ValueError(f"unknown perm kind {kind!r}")
    return inner + list(range(k, max_k))


@torch.no_grad()
def generate_structured(
    wrap,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 64,
    max_k: int = 48,
    baseline: str = "webgap",
    html_path: str | None = None,
    html: str | None = None,
    box_path: str | None = None,
    nodes: list[dict] | None = None,
    perm: torch.Tensor | list[int] | None = None,
) -> str:
    """Greedy decode. GraphToken prepends linearized graph; WebGAP calls set_graph."""
    nd = nodes
    if baseline in ("webgap", "random_anchor", "graphtoken"):
        nd = nodes_for_screenshot(image, html_path=html_path, html=html, box_path=box_path, nodes=nodes)
    text_prompt = prompt
    if baseline == "graphtoken":
        text_prompt = serialize_graph_text(build_graph(nd)) + "\n\n" + prompt
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}
    use_plugin = baseline in ("webgap", "random_anchor")
    backbone = wrap.backbone if hasattr(wrap, "backbone") else wrap
    if use_plugin and hasattr(wrap, "set_graph"):
        tok = getattr(processor, "tokenizer", processor)
        image_token_id = getattr(tok, "image_token_id", None) or IMAGE_TOKEN_ID_QWEN3VL
        gb = graph_batch_from_proc(proc, nd, image.size[0], image.size[1], max_k, int(image_token_id))
        dtype = next(wrap.plugin.parameters()).dtype
        gb = move_graph_batch(gb, device, dtype)
        if perm is not None:
            gb.perm = torch.tensor(perm, device=device, dtype=torch.long) if not torch.is_tensor(perm) else perm.to(device)
        elif baseline == "random_anchor":
            gb.perm = torch.randperm(gb.assignment.size(-1), device=device)
        wrap.set_graph(gb)
        wrap.enable(True)
    elif hasattr(wrap, "set_graph"):
        wrap.set_graph(None)
        wrap.enable(False)
    n = proc["input_ids"].shape[1]
    tok = getattr(processor, "tokenizer", processor)
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None) or eos
    out = backbone.generate(
        **proc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


class StructuredGenerate:
    """Drop-in for evaluate.py generate(model, processor, image, prompt, ...)."""

    def __init__(self, wrap, cfg, baseline: str):
        self.wrap = wrap
        self.cfg = cfg
        self.baseline = baseline
        self.html_path = None
        self.nodes = None
        self.perm = None
        self._logged = False

    def bind_row(self, row: dict | None):
        self.html_path = None if row is None else (row.get("html_path") or None)
        self.nodes = None if row is None else row.get("nodes")

    def __call__(self, gen_model, processor, image, prompt, device: str = "cuda", max_new_tokens: int = 64) -> str:
        text = generate_structured(
            self.wrap,
            processor,
            image,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            max_k=self.cfg.plugin.max_anchors,
            baseline=self.baseline,
            html_path=self.html_path,
            nodes=self.nodes,
            perm=self.perm,
        )
        if not self._logged and self.baseline in ("webgap", "random_anchor") and getattr(self.wrap, "ctx", None) is not None:
            gb = self.wrap.ctx.gb
            if gb is not None:
                print(
                    f"[graph] baseline={self.baseline} k_vis={int(gb.k_valid[0])} k_dom={int(gb.k_valid_dom[0]) if gb.k_valid_dom is not None else 0}",
                    flush=True,
                )
            self._logged = True
        return text
