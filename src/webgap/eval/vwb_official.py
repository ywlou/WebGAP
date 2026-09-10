"""Official VisualWebBench prompts and metrics (Liu et al., 2024).

Scoring follows VisualWebBench/utils/eval_utils.py: Rouge-F for caption/OCR/WebQA,
letter accuracy for 8-way grounding / action tasks. Prompts copied from
VisualWebBench/utils/prompts.py.
"""

from __future__ import annotations

import re
from typing import Any

from webgap.eval.metrics import normalize_answer

WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

HEADING_OCR_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""

WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        ndp = [0]
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            ndp.append(prev + 1 if x == y else max(ndp[-1], dp[j]))
            prev = cur
        dp = ndp
    return dp[-1]


def rouge_n_f(pred: str, gold: str, n: int = 1) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if n > 1:
        pt = [" ".join(pt[i : i + n]) for i in range(len(pt) - n + 1)]
        gt = [" ".join(gt[i : i + n]) for i in range(len(gt) - n + 1)]
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    from collections import Counter

    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    prec = overlap / max(len(pt), 1)
    rec = overlap / max(len(gt), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l_f(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    lcs = _lcs_len(pt, gt)
    prec = lcs / len(pt)
    rec = lcs / len(gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def parse_multi_choice_response(response: str, all_choices: list[str] | None = None) -> str:
    """Official VisualWebBench letter parser."""
    import numpy as np

    all_choices = all_choices or [chr(ord("A") + i) for i in range(8)]
    response = response or ""
    if len(response.strip()) == 1:
        return response.strip().upper()
    if not response.strip():
        return "a"
    if re.match(r"[A-Z]\.", response.strip()):
        return response.strip()[0]
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " "
    candidates = []
    ans_with_brack = False
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates:
        return "z"
    if len(candidates) > 1:
        start_indexes = []
        for can in candidates:
            needle = f"({can})" if ans_with_brack else f" {can} "
            start_indexes.append(response.rfind(needle))
        return candidates[int(np.argmax(start_indexes))]
    return candidates[0]


def extract_caption(pred: str) -> str:
    m = re.search(r'content\s*=\s*"([^"]*)"', pred or "", re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"content\s*=\s*'([^']*)'", pred or "", re.I)
    if m:
        return m.group(1).strip()
    return (pred or "").strip()


def extract_element_ocr(pred: str) -> str:
    m = re.search(r"bounding box is:\s*(.*)$", pred or "", re.I | re.S)
    if m:
        return m.group(1).strip().strip('"')
    return (pred or "").strip()


def bbox_ratio_str(bbox, image_size) -> str:
    if not bbox:
        return "[0.00, 0.00, 1.00, 1.00]"
    nums = [float(x) for x in list(bbox)[:4]]
    w = h = 1.0
    if image_size and len(image_size) >= 2:
        w, h = float(image_size[0] or 1), float(image_size[1] or 1)
        if w < h and w <= 4:  # sometimes stored as H,W
            w, h = float(image_size[1]), float(image_size[0])
    w, h = max(w, 1.0), max(h, 1.0)
    x1, y1, a, b = nums
    # [x,y,w,h] vs [x1,y1,x2,y2]
    if a <= 1.5 and b <= 1.5 and x1 <= 1.5 and y1 <= 1.5:
        x2, y2 = a, b
        if x2 < x1 or y2 < y1:
            x2, y2 = x1 + max(a, 0.01), y1 + max(b, 0.01)
        return f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
    if a > x1 and b > y1 and a > 1.5 and b > 1.5:
        # x2,y2
        x2, y2 = a, b
    else:
        x2, y2 = x1 + a, y1 + b
    return f"[{x1 / w:.2f}, {y1 / h:.2f}, {x2 / w:.2f}, {y2 / h:.2f}]"


def build_prompt(row: dict[str, Any]) -> str:
    task = row.get("task") or row.get("task_type") or ""
    if task == "web_caption":
        return WEB_CAPTION_PROMPT
    if task == "heading_ocr":
        return HEADING_OCR_PROMPT
    if task == "webqa":
        return WEBQA_PROMPT.format(question=row.get("question") or "")
    if task == "element_ocr":
        return ELEMENT_OCR_PROMPT.format(bbox_ratio=bbox_ratio_str(row.get("bbox"), row.get("image_size")))
    if task == "element_ground":
        desc = row.get("elem_desc") or row.get("question") or ""
        return ELEMENT_GROUND_PROMPT.format(element_desc=desc)
    if task == "action_prediction":
        opts = row.get("options") or []
        if opts and isinstance(opts[0], (list, tuple)):
            opts = [str(x) for x in opts]
        choices = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(opts))
        return ACTION_PREDICTION_PROMPT.format(
            bbox_ratio=bbox_ratio_str(row.get("bbox"), row.get("image_size")),
            choices_text=choices,
        )
    if task == "action_ground":
        return ACTION_GROUND_PROMPT.format(instruction=row.get("instruction") or row.get("question") or "")
    return str(row.get("question") or "Describe this webpage.")


def score_vwb(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: task, pred, gold (gold may be str, list[str], or int)."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r.get("task") or "unk", []).append(r)
    task_scores: dict[str, Any] = {}
    headline = []
    for task, rs in by.items():
        preds = [r["pred"] for r in rs]
        golds = [r["gold"] for r in rs]
        if task in ("element_ground", "action_ground", "action_prediction"):
            hits = []
            for pred, gold in zip(preds, golds):
                letter = parse_multi_choice_response(pred)
                try:
                    gi = int(gold)
                    hits.append(ord(letter) - ord("A") == gi)
                except (TypeError, ValueError):
                    hits.append(False)
            acc = 100.0 * sum(hits) / max(len(hits), 1)
            task_scores[task] = {"n": len(rs), "accuracy": acc, "primary": acc}
            headline.append(acc)
        elif task == "webqa":
            f1s = []
            for pred, gold in zip(preds, golds):
                glist = gold if isinstance(gold, list) else [str(gold)]
                f1s.append(max(rouge_n_f(pred, g, 1) for g in glist if str(g).strip()) if any(str(g).strip() for g in glist) else 0.0)
            f1 = 100.0 * (sum(f1s) / max(len(f1s), 1))
            task_scores[task] = {"n": len(rs), "f1": f1, "primary": f1}
            headline.append(f1)
        else:
            xs, ys = [], []
            for pred, gold in zip(preds, golds):
                p = extract_caption(pred) if task == "web_caption" else extract_element_ocr(pred) if task == "element_ocr" else pred
                g = gold[0] if isinstance(gold, list) else str(gold)
                xs.append(p)
                ys.append(g)
            r1 = 100.0 * sum(rouge_n_f(p, g, 1) for p, g in zip(xs, ys)) / max(len(xs), 1)
            r2 = 100.0 * sum(rouge_n_f(p, g, 2) for p, g in zip(xs, ys)) / max(len(xs), 1)
            rl = 100.0 * sum(rouge_l_f(p, g) for p, g in zip(xs, ys)) / max(len(xs), 1)
            task_scores[task] = {"n": len(rs), "rouge_1": r1, "rouge_2": r2, "rouge_l": rl, "primary": rl}
            headline.append(rl)
    avg = sum(headline) / max(len(headline), 1) if headline else 0.0
    return {
        "n": len(rows),
        "official_avg": avg,
        "by_task": task_scores,
        "note": "Official VisualWebBench average of per-task primary scores (Rouge-L or accuracy or WebQA Rouge-1 F).",
    }


def max_new_tokens_for_task(task: str) -> int:
    if task in ("element_ground", "action_ground", "action_prediction"):
        return 8
    if task == "webqa":
        return 32
    if task == "heading_ocr":
        return 64
    if task == "element_ocr":
        return 96
    return 128
