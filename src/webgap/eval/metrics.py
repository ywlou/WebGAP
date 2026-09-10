"""Answer normalization, EM/F1, StHR."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(pred: str) -> str:
    """Pull a short answer out of a verbose VLM reply."""
    pred = (pred or "").strip()
    if not pred:
        return ""
    bolds = re.findall(r"\*\*([^*]+)\*\*", pred)
    if bolds:
        cand = bolds[-1].strip()
        if len(cand) < 80:
            pred = cand
    # "Overview — Peak Bottle" -> last clause
    if "—" in pred:
        pred = pred.split("—")[-1].strip()
    if " – " in pred:
        pred = pred.split(" – ")[-1].strip()
    lines = [ln.strip().strip("-• ") for ln in pred.splitlines() if ln.strip()]
    skip_pfx = ("based on", "looking at", "the image", "from the", "according to")
    short = [
        ln
        for ln in lines
        if len(ln) <= 64 and not ln.lower().startswith(skip_pfx) and not ln.endswith(":")
    ]
    if short:
        return short[-1]
    return lines[-1] if lines else pred


def exact_match(pred: str, gold: str) -> bool:
    p = normalize_answer(extract_answer(pred))
    g = normalize_answer(gold)
    if p == g:
        return True
    # gold is a whole-word substring of the extracted answer (tab prefixes etc.)
    if g and (p.endswith(g) or p.startswith(g) or f" {g} " in f" {p} "):
        return True
    return False


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(extract_answer(pred)).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_entities(pred: str, gold: str) -> bool:
    """Entity hit: gold tokens of length>=2 appear in prediction (or EM)."""
    if exact_match(pred, gold):
        return True
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    if not g:
        return False
    if g in p:
        return True
    toks = [t for t in g.split() if len(t) >= 2]
    if not toks:
        return g in p
    return all(t in p.split() or t in p for t in toks)


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: pred, gold, type, ..."""
    n = len(rows)
    em = sum(exact_match(r["pred"], r["gold"]) for r in rows) / max(n, 1)
    f1 = sum(token_f1(r["pred"], r["gold"]) for r in rows) / max(n, 1)
    substr = sum(normalize_answer(r["gold"]) in normalize_answer(r["pred"]) for r in rows if normalize_answer(r["gold"])) / max(n, 1)
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r.get("type") or "UNK", []).append(r)
    type_scores = {}
    for t, rs in by_type.items():
        type_scores[t] = {
            "n": len(rs),
            "em": sum(exact_match(r["pred"], r["gold"]) for r in rs) / len(rs),
            "f1": sum(token_f1(r["pred"], r["gold"]) for r in rs) / len(rs),
        }
    by_probe: dict[str, list] = {}
    for r in rows:
        p = r.get("probe") or ""
        if p:
            by_probe.setdefault(p, []).append(r)
    probe_scores = {
        t: {
            "n": len(rs),
            "em": sum(exact_match(r["pred"], r["gold"]) for r in rs) / len(rs),
            "f1": sum(token_f1(r["pred"], r["gold"]) for r in rs) / len(rs),
        }
        for t, rs in by_probe.items()
    }
    by_tmpl: dict[str, list] = {}
    for r in rows:
        tm = r.get("template") or ""
        if tm:
            by_tmpl.setdefault(tm, []).append(r)
    tmpl_scores = {
        t: {
            "n": len(rs),
            "em": sum(exact_match(r["pred"], r["gold"]) for r in rs) / len(rs),
            "f1": sum(token_f1(r["pred"], r["gold"]) for r in rs) / len(rs),
        }
        for t, rs in by_tmpl.items()
    }
    # StHR: among entity-correct samples, fraction with wrong relation (not EM)
    ent_ok = [r for r in rows if contains_entities(r["pred"], r["gold"])]
    struct_wrong = [r for r in ent_ok if not exact_match(r["pred"], r["gold"])]
    sth = len(struct_wrong) / max(len(ent_ok), 1)
    hops = {
        "single": [r for r in rows if int(r.get("hops") or 1) <= 1],
        "multi": [r for r in rows if int(r.get("hops") or 1) >= 2],
    }
    hop_scores = {
        k: {
            "n": len(v),
            "em": sum(exact_match(r["pred"], r["gold"]) for r in v) / max(len(v), 1),
            "f1": sum(token_f1(r["pred"], r["gold"]) for r in v) / max(len(v), 1),
        }
        for k, v in hops.items()
    }
    return {
        "n": n,
        "em": em,
        "f1": f1,
        "em_substr": substr,
        "sthr": sth,
        "entity_hit": len(ent_ok) / max(n, 1),
        "by_type": type_scores,
        "by_hop": hop_scores,
        "by_probe": probe_scores,
        "by_template": tmpl_scores,
    }


def bootstrap_em_ci(rows: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Item-level bootstrap 95% CI for exact match."""
    import random

    n = len(rows)
    if n == 0:
        return {"em": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0}
    hits = [1.0 if exact_match(r["pred"], r["gold"]) else 0.0 for r in rows]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        acc = sum(hits[rng.randrange(n)] for _ in range(n)) / n
        stats.append(acc)
    stats.sort()
    lo = stats[int(0.025 * (n_boot - 1))]
    hi = stats[int(0.975 * (n_boot - 1))]
    return {"em": sum(hits) / n, "ci95_lo": float(lo), "ci95_hi": float(hi), "n": n}
