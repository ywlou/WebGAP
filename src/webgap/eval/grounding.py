"""Point-in-bbox grounding and WebSRC Path Overlap Score."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from webgap.eval.metrics import exact_match, token_f1, normalize_answer


def parse_point(pred: str, width: int, height: int) -> tuple[float, float] | None:
    """Parse a click point. Accepts [x,y] in pixels, 0-1, 0-100, or 0-1000."""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", pred or "")]
    if len(nums) < 2:
        return None
    x, y = nums[0], nums[1]
    w, h = max(float(width), 1.0), max(float(height), 1.0)
    mx = max(abs(x), abs(y), abs(nums[0]), abs(nums[1]) if len(nums) > 1 else 0)
    if 0 <= x <= 1.01 and 0 <= y <= 1.01 and mx <= 1.01:
        return x * w, y * h
    if 0 <= x <= 100.5 and 0 <= y <= 100.5 and mx <= 100.5:
        return (x / 100.0) * w, (y / 100.0) * h
    if 0 <= x <= 1000.5 and 0 <= y <= 1000.5 and mx <= 1000.5 and (w > 1000 or h > 1000 or mx > 100):
        return (x / 1000.0) * w, (y / 1000.0) * h
    return x, y


def bbox_xyxy(bbox, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    nums = [float(x) for x in list(bbox)[:4]]
    x1, y1, a, b = nums
    if 0 <= x1 <= 1.01 and 0 <= y1 <= 1.01 and 0 <= a <= 1.5 and 0 <= b <= 1.5:
        x1, y1, a, b = x1 * width, y1 * height, a * width, b * height
    x2, y2 = (a, b) if a > x1 and b > y1 and a > 1.5 else (x1 + a, y1 + b)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def point_in_bbox(pred: str, bbox, width: int, height: int) -> bool:
    pt = parse_point(pred, width, height)
    box = bbox_xyxy(bbox, width, height)
    if pt is None or box is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


SCREENSPOT_PROMPT = """Locate the UI element that matches this instruction. Output only the click point as [x, y] where x and y are percentages from 0 to 100.

Instruction: {instruction}
"""


def score_point_ground(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    n = 0
    by_plat: dict[str, list[int]] = {}
    for r in rows:
        w = int(r.get("width") or r.get("orig_w") or 0)
        h = int(r.get("height") or r.get("orig_h") or 0)
        ok = point_in_bbox(r.get("pred") or "", r.get("bbox") or r.get("gold"), w, h)
        hits += int(ok)
        n += 1
        plat = str(r.get("platform") or r.get("data_source") or "all")
        by_plat.setdefault(plat, []).append(int(ok))
    acc = 100.0 * hits / max(n, 1)
    return {
        "n": n,
        "accuracy": acc,
        "by_platform": {k: {"n": len(v), "accuracy": 100.0 * sum(v) / max(len(v), 1)} for k, v in by_plat.items()},
    }


class _TidParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.paths: dict[str, list[str]] = {}
        self.texts: dict[str, str] = {}
        self._cur_tid: str | None = None

    def handle_starttag(self, tag, attrs):
        ad = {k.lower(): v for k, v in attrs}
        tid = ad.get("tid") or ad.get("data-tid")
        self.stack.append(tid if tid is not None else f"{tag}:{len(self.stack)}")
        if tid is not None:
            self.paths[str(tid)] = list(self.stack)
            self.texts.setdefault(str(tid), "")
            self._cur_tid = str(tid)

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()
        self._cur_tid = None
        for t in reversed(self.stack):
            if t.isdigit() or (t and t[0].isdigit()):
                self._cur_tid = t
                break

    def handle_data(self, data):
        if self._cur_tid is not None:
            self.texts[self._cur_tid] = (self.texts.get(self._cur_tid) or "") + " " + (data or "")


def html_tid_path(html: str, tid: str) -> set[str]:
    p = _TidParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:
        return set()
    return set(p.paths.get(str(tid), []))


def html_tid_for_answer(html: str, answer: str) -> str | None:
    p = _TidParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:
        return None
    g = normalize_answer(answer)
    if not g:
        return None
    hits = []
    for tid, text in p.texts.items():
        nt = normalize_answer(text)
        if g and g in nt:
            hits.append((len(nt), tid))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def path_overlap(p1: set[str], p2: set[str]) -> float:
    if not p1 and not p2:
        return 1.0
    if not p1 or not p2:
        return 0.0
    return len(p1 & p2) / len(p1 | p2)


def score_websrc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    em = f1 = pos = 0.0
    n_pos = 0
    n = len(rows)
    for r in rows:
        pred = r.get("pred") or ""
        gold = str(r.get("gold") or r.get("answer") or "")
        em += float(exact_match(pred, gold))
        f1 += token_f1(pred, gold)
        html = r.get("html") or ""
        if not html and r.get("html_path"):
            try:
                html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                html = ""
        gold_tid = str(r.get("element_id") or "")
        if html and gold_tid and gold_tid not in {"-1", ""}:
            gold_path = html_tid_path(html, gold_tid)
            pred_tid = html_tid_for_answer(html, pred)
            pred_path = html_tid_path(html, pred_tid) if pred_tid else set()
            pos += path_overlap(pred_path, gold_path)
            n_pos += 1
        elif gold_tid in {"-1", ""}:
            # yes/no: POS undefined; skip
            pass
    return {
        "n": n,
        "em": em / max(n, 1),
        "f1": f1 / max(n, 1),
        "pos": pos / max(n_pos, 1),
        "n_pos": n_pos,
        "note": "POS is span-aligned (map predicted text to a tid, then Jaccard of root-to-tid paths). Yes/no items excluded from POS.",
    }
