"""Dataset + collator: screenshot + QA + STAR assignment tensors."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from webgap.data.graph import (
    build_graph,
    dom_anchor_pack,
    serialize_graph_text,
    visual_anchor_pack,
    visual_token_bboxes,
)
from webgap.models.plugin import GraphBatch
from webgap.constants import IMAGE_TOKEN_ID_QWEN3VL, VISION_START_ID_QWEN3VL, VISION_END_ID_QWEN3VL


SYSTEM_PROMPT = (
    "Look at the webpage screenshot. "
    "Answer with ONLY the final short answer (a name, number, yes/no, or short phrase). "
    "Do not explain."
)


class WebForgeQADataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        image_root: str | Path,
        max_samples: int = -1,
        seed: int = 42,
        graphtoken: bool = False,
        ssl: bool = False,
        types: tuple[str, ...] | None = None,
        replay_ratio: float = 0.0,
    ):
        self.image_root = Path(image_root)
        self.graphtoken = graphtoken
        self.ssl = ssl
        rows = []
        split_dir = str(self.image_root)
        with Path(index_path).open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["_split_dir"] = split_dir
                qas = rec.get("qa") or []
                if ssl:
                    rows.append({"page": rec, "qa": {"question": "Summarize the layout of this webpage in one sentence.", "answer": rec["meta"]["template"], "type": "SSL"}})
                    continue
                for qa in qas:
                    if types and qa.get("type") not in types:
                        continue
                    rows.append({"page": rec, "qa": qa})
        rng = random.Random(seed)
        rng.shuffle(rows)
        if replay_ratio and replay_ratio > 0 and not ssl:
            by_page = {}
            for r in rows:
                by_page[r["page"]["id"]] = r["page"]
            open_rows = []
            for rec in by_page.values():
                tmpl = str((rec.get("meta") or {}).get("template") or "webpage")
                open_rows.append(
                    {
                        "page": rec,
                        "qa": {
                            "question": "In one short sentence, describe the main layout of this webpage (sections, not pixel colors).",
                            "answer": tmpl.replace("_", " "),
                            "type": "OPEN",
                            "probe": "replay",
                        },
                    }
                )
            rng.shuffle(open_rows)
            n_open = max(1, int(len(rows) * replay_ratio))
            rows = rows + open_rows[:n_open]
            rng.shuffle(rows)
        else:
            rows = [r for r in rows if r["qa"].get("type") != "OPEN" and r["qa"].get("probe") != "replay"]
        if max_samples and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        rec, qa = row["page"], row["qa"]
        root = Path(rec.get("_split_dir") or self.image_root)
        img_path = Path(rec["image"]) if Path(rec["image"]).is_absolute() else root / rec["image"]
        if not img_path.exists():
            img_path = self.image_root / rec["image"]
        image = Image.open(img_path).convert("RGB")
        question = qa["question"]
        if self.graphtoken:
            g = build_graph(rec["nodes"])
            question = serialize_graph_text(g) + "\n\nQuestion: " + question
        question = SYSTEM_PROMPT + "\n\n" + question
        return {
            "image": image,
            "question": question,
            "answer": str(qa["answer"]),
            "type": qa.get("type", ""),
            "id": rec["id"] + "::" + qa["question"][:48],
            "nodes": rec["nodes"],
            "orig_w": rec["meta"]["width"],
            "orig_h": rec["meta"]["height"],
            "page_id": rec["id"],
            "template": rec["meta"]["template"],
            "heldout": rec["meta"].get("heldout", False),
            "probe": qa.get("probe", ""),
        }


def _find_visual_span(input_ids: torch.Tensor, image_token_id: int) -> tuple[int, int]:
    ids = input_ids.tolist()
    pos = [i for i, t in enumerate(ids) if t == image_token_id]
    if not pos:
        return 0, 0
    return pos[0], pos[-1] + 1


def _token_boxes_for_ids(input_ids, image_grid_thw, orig_w, orig_h, merge_size, image_token_id):
    start, end = _find_visual_span(input_ids, image_token_id)
    n_vis = max(0, end - start)
    boxes = None
    if n_vis > 0 and image_grid_thw is not None:
        thw = image_grid_thw
        if torch.is_tensor(thw):
            thw = thw.detach().cpu().tolist()
        if isinstance(thw[0], (list, tuple)):
            t, h, w = thw[0]
        else:
            t, h, w = thw
        gh, gw = max(int(h) // merge_size, 1), max(int(w) // merge_size, 1)
        boxes = visual_token_bboxes(orig_w, orig_h, gh, gw)
        if boxes.shape[0] != n_vis:
            if boxes.shape[0] > n_vis:
                boxes = boxes[:n_vis]
            else:
                reps = int(np.ceil(n_vis / max(boxes.shape[0], 1)))
                boxes = np.tile(boxes, (reps, 1))[:n_vis]
    return start, end, n_vis, boxes


def _scatter_assign(assign_tok: np.ndarray, s: int, start: int, end: int, k: int, max_k: int) -> np.ndarray:
    assign = np.zeros((s, max_k), dtype=np.float32)
    if end > start and assign_tok is not None and assign_tok.shape[0] > 0:
        n = min(end - start, assign_tok.shape[0])
        assign[start : start + n, : assign_tok.shape[1]] = assign_tok[:n]
    if k > 0:
        text = np.ones((s,), dtype=np.bool_)
        text[start:end] = False
        assign[text, :k] = 1.0 / k
    return assign


def build_assignment_for_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    nodes: list[dict],
    orig_w: int,
    orig_h: int,
    max_k: int,
    merge_size: int = 2,
    image_token_id: int = IMAGE_TOKEN_ID_QWEN3VL,
) -> dict[str, np.ndarray | int]:
    """Visual + DOM STAR packs aligned to the token sequence."""
    s = input_ids.shape[0]
    graph = build_graph(nodes)
    start, end, n_vis, boxes = _token_boxes_for_ids(input_ids, image_grid_thw, orig_w, orig_h, merge_size, image_token_id)
    vis_mask = np.zeros((s,), dtype=np.bool_)
    vis_mask[start:end] = True
    if boxes is None:
        boxes = np.zeros((max(n_vis, 1), 4), dtype=np.float32)
        boxes[:, 2:] = 1.0
    vis = visual_anchor_pack(graph, boxes, max_k)
    dom = dom_anchor_pack(graph, boxes, max_k)
    vis_a = _scatter_assign(vis["assign"], s, start, end, int(vis["k"]), max_k)
    dom_a = _scatter_assign(dom["assign"], s, start, end, int(dom["k"]), max_k)
    return {
        "assign": vis_a,
        "rel": vis["rel"],
        "types": vis["types"],
        "k": int(vis["k"]),
        "order": vis["order"],
        "assign_dom": dom_a,
        "rel_dom": dom["rel"],
        "types_dom": dom["types"],
        "k_dom": int(dom["k"]),
        "order_dom": dom["order"],
        "vis_mask": vis_mask,
    }


class VLMCollator:
    def __init__(self, processor, max_k: int = 48, max_seq_len: int = 3072, train: bool = True, merge_size: int = 2):
        self.processor = processor
        self.max_k = max_k
        self.max_seq_len = max_seq_len
        self.train = train
        self.merge_size = merge_size
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.image_token_id = getattr(tok, "image_token_id", IMAGE_TOKEN_ID_QWEN3VL) or IMAGE_TOKEN_ID_QWEN3VL
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _messages(self, question: str):
        text = question if question.startswith("Look at the webpage") else (SYSTEM_PROMPT + "\n\n" + question)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        images = [f["image"] for f in features]
        questions = [f["question"] for f in features]
        answers = [f["answer"] for f in features]
        texts = []
        for q, a in zip(questions, answers):
            msgs = self._messages(q)
            prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if self.train:
                texts.append(prompt + a + self.processor.tokenizer.eos_token)
            else:
                texts.append(prompt)
        proc = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        input_ids = proc["input_ids"]
        labels = None
        if self.train:
            labels = torch.full_like(input_ids, -100)
            attn = proc["attention_mask"]
            for i in range(input_ids.size(0)):
                ans_ids = self.processor.tokenizer(
                    answers[i] + self.processor.tokenizer.eos_token, add_special_tokens=False
                )["input_ids"]
                n_ans = len(ans_ids)
                seq = input_ids[i]
                if n_ans <= 0:
                    continue
                valid = attn[i].eq(1).nonzero(as_tuple=False).view(-1)
                if len(valid) >= n_ans:
                    labels[i, valid[-n_ans:]] = seq[valid[-n_ans:]]
                else:
                    labels[i, -n_ans:] = seq[-n_ans:]
        # graphs
        grid = proc.get("image_grid_thw")
        vis_as, vis_rel, vis_ty, vis_ord = [], [], [], []
        dom_as, dom_rel, dom_ty, dom_ord = [], [], [], []
        vis_masks, k_vis, k_dom = [], [], []
        bsz, slen = input_ids.shape
        for i, f in enumerate(features):
            gthw = grid[i] if grid is not None else None
            pack = build_assignment_for_ids(
                input_ids[i],
                gthw,
                f["nodes"],
                f["orig_w"],
                f["orig_h"],
                self.max_k,
                merge_size=self.merge_size,
                image_token_id=self.image_token_id,
            )

            def _pad_len(a, vm):
                if a.shape[0] < slen:
                    pad = np.zeros((slen - a.shape[0], a.shape[1]), dtype=np.float32)
                    vm_pad = np.zeros((slen - vm.shape[0],), dtype=np.bool_)
                    return np.concatenate([a, pad], 0), np.concatenate([vm, vm_pad], 0)
                if a.shape[0] > slen:
                    return a[:slen], vm[:slen]
                return a, vm

            a, vm = _pad_len(pack["assign"], pack["vis_mask"])
            ad, _ = _pad_len(pack["assign_dom"], pack["vis_mask"])
            vis_as.append(a)
            vis_rel.append(pack["rel"])
            vis_ty.append(pack["types"])
            vis_ord.append(pack["order"])
            vis_masks.append(vm)
            k_vis.append(pack["k"])
            dom_as.append(ad)
            dom_rel.append(pack["rel_dom"])
            dom_ty.append(pack["types_dom"])
            dom_ord.append(pack["order_dom"])
            k_dom.append(pack["k_dom"])
        gb = GraphBatch(
            assignment=torch.tensor(np.stack(vis_as), dtype=torch.float32),
            rel_idx=torch.tensor(np.stack(vis_rel), dtype=torch.long),
            k_valid=torch.tensor(k_vis, dtype=torch.long),
            anchor_types=torch.tensor(np.stack(vis_ty), dtype=torch.long),
            visual_mask=torch.tensor(np.stack(vis_masks), dtype=torch.bool),
            assignment_dom=torch.tensor(np.stack(dom_as), dtype=torch.float32),
            rel_idx_dom=torch.tensor(np.stack(dom_rel), dtype=torch.long),
            k_valid_dom=torch.tensor(k_dom, dtype=torch.long),
            anchor_types_dom=torch.tensor(np.stack(dom_ty), dtype=torch.long),
            order_ids=torch.tensor(np.stack(vis_ord), dtype=torch.long),
            order_ids_dom=torch.tensor(np.stack(dom_ord), dtype=torch.long),
        )
        batch = dict(proc)
        if labels is not None:
            batch["labels"] = labels
        batch["graph_batch"] = gb
        batch["meta"] = [
            {
                "id": f["id"],
                "type": f["type"],
                "answer": f["answer"],
                "template": f.get("template"),
                "probe": f.get("probe", ""),
            }
            for f in features
        ]
        return batch
