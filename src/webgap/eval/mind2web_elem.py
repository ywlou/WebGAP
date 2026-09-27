"""Mind2Web element-choice protocol for the Qwen main table.

Positive candidates are kept. Negatives are a fixed sample of 50 (seed 42),
not the DeBERTa top-50 used by MindAct. Scores from this protocol stay in
``*_elem_metrics.json`` and are not MindAct / SeeAct Element Accuracy.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

BENCH = Path("/data/WebGAP/data/benchmarks/mind2web")
PARQUET_ROOT = Path(
    "/data/WebGAP/.hf_cache/hub/datasets--osunlp--Multimodal-Mind2Web/snapshots/"
    "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889/data"
)
NEG_K = 50
SEED = 42

_OP_RE = re.compile(r"\b(CLICK|TYPE|SELECT|HOVER)\b", re.I)
_ID_RE = re.compile(r"\bID\s*[:=]?\s*(\d+)\b", re.I)


def _loads(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _attrs(obj: dict) -> dict:
    raw = obj.get("attributes")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def summarize(raw: str) -> dict | None:
    obj = _loads(raw) if isinstance(raw, str) else {}
    if not obj:
        return None
    attrs = _attrs(obj)
    node_id = str(obj.get("backend_node_id") or attrs.get("backend_node_id") or "").strip()
    if not node_id:
        return None
    text = ""
    for key in ("aria-label", "placeholder", "value", "title", "alt", "name"):
        val = attrs.get(key)
        if val:
            text = str(val).replace("\n", " ").strip()
            break
    return {"id": node_id, "tag": str(obj.get("tag") or "node"), "text": text[:40]}


def _line(index: int, cand: dict) -> str:
    text = cand["text"]
    extra = f' "{text}"' if text else ""
    return f"[{index}] ID {cand['id']} <{cand['tag']}>{extra}"


def sample_negatives(neg_raw: list, uid: str, k: int = NEG_K) -> list[dict]:
    parsed = []
    for raw in neg_raw or []:
        item = summarize(raw)
        if item is not None:
            parsed.append(item)
    if len(parsed) <= k:
        return parsed
    rng = random.Random(f"{SEED}:{uid}")
    return rng.sample(parsed, k)


def build_choice(pos_raw: list, neg_raw: list, uid: str) -> dict:
    pos = [c for c in (summarize(x) for x in (pos_raw or [])) if c]
    neg = sample_negatives(neg_raw, uid)
    # Drop negatives that reuse a positive id.
    pos_ids = {c["id"] for c in pos}
    neg = [c for c in neg if c["id"] not in pos_ids]
    cands = pos + neg
    return {
        "pos_ids": sorted(pos_ids),
        "candidates": cands,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def prompt_for(task: str, choice: dict) -> str:
    lines = [_line(i, c) for i, c in enumerate(choice["candidates"])]
    body = "\n".join(lines) if lines else "(no candidates)"
    return (
        "You are a web agent. Choose the next element from the candidate list "
        "and the operation. Reply with exactly one line:\n"
        "ID <backend_node_id> OP <CLICK|TYPE|SELECT|HOVER> VALUE <text or empty>\n\n"
        f"Task: {task}\n"
        f"Candidates:\n{body}\n"
        "Answer:"
    )


def parse_prediction(text: str) -> tuple[str, str, str]:
    node_id = ""
    match = _ID_RE.search(text or "")
    if match:
        node_id = match.group(1)
    elif (text or "").strip().isdigit():
        node_id = text.strip()
    op = ""
    op_match = _OP_RE.search(text or "")
    if op_match:
        op = op_match.group(1).upper()
    value = ""
    val_match = re.search(r"\bVALUE\s*[:=]?\s*(.*)$", text or "", re.I)
    if val_match:
        value = val_match.group(1).strip()
    return node_id, op, value


def gold_operation(raw) -> tuple[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"op": raw}
    if not isinstance(raw, dict):
        return str(raw or "").upper(), ""
    op = str(raw.get("op") or raw.get("original_op") or "").upper()
    value = str(raw.get("value") or "")
    return op, value


def score_one(pred: str, choice: dict, operation) -> dict:
    node_id, op, value = parse_prediction(pred)
    gold_op, gold_value = gold_operation(operation)
    element_hit = bool(node_id) and node_id in set(choice["pos_ids"])
    pred_op = f"{op} {value}".strip()
    gold_op_s = f"{gold_op} {gold_value}".strip()
    return {
        "pred_id": node_id,
        "pred_op": op,
        "pred_value": value,
        "element_hit": element_hit,
        "op_f1": _token_f1(pred_op, gold_op_s),
        "gold_op": gold_op,
        "gold_value": gold_value,
    }


def _token_f1(pred: str, gold: str) -> float:
    ps = pred.lower().split()
    gs = gold.lower().split()
    if not ps or not gs:
        return 0.0
    from collections import Counter

    pc, gc = Counter(ps), Counter(gs)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ps)
    recall = overlap / len(gs)
    return 2 * precision * recall / (precision + recall)


def candidate_path(split: str) -> Path:
    return BENCH / split / "elem_candidates.jsonl"


def parquet_files(split: str) -> list[Path]:
    return sorted(PARQUET_ROOT.glob(f"{split}-*.parquet"))


def build_candidate_file(split: str) -> Path:
    """Write one compact choice record per action. Skips the screenshot column."""
    out = candidate_path(split)
    if out.exists() and out.stat().st_size > 0:
        return out
    files = parquet_files(split)
    if not files:
        raise FileNotFoundError(f"no parquet for {split} under {PARQUET_ROOT}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for path in files:
            pf = pq.ParquetFile(path)
            cols = ["action_uid", "pos_candidates", "neg_candidates", "operation"]
            for batch in pf.iter_batches(batch_size=32, columns=cols):
                uids = batch.column("action_uid").to_pylist()
                poss = batch.column("pos_candidates").to_pylist()
                negs = batch.column("neg_candidates").to_pylist()
                ops = batch.column("operation").to_pylist()
                for uid, pos, neg, op in zip(uids, poss, negs, ops):
                    choice = build_choice(pos or [], neg or [], str(uid))
                    rec = {"id": str(uid), "operation": op, **choice}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 500 == 0:
                        print(f"  {split} candidates {n}", flush=True)
    tmp.replace(out)
    print(f"wrote {out} n={n}", flush=True)
    return out


def load_candidate_map(split: str) -> dict[str, dict]:
    path = build_candidate_file(split)
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["id"])] = rec
    return out
