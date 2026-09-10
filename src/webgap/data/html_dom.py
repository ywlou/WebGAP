"""Parse real HTML into WebForge-style nodes so GACA/TRB can run on WebSRC."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

SKIP_TAGS = {"script", "style", "noscript", "svg", "meta", "link", "br", "hr"}
ROLE_MAP = {
    "html": "page",
    "body": "page",
    "header": "header",
    "nav": "nav",
    "a": "nav_item",
    "button": "button",
    "table": "table",
    "tr": "row",
    "td": "cell",
    "th": "cell",
    "img": "image",
    "figcaption": "caption",
    "figure": "card",
    "li": "list_item",
    "ul": "list",
    "ol": "list",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "label": "label",
    "input": "field",
    "form": "form",
    "footer": "footer",
    "aside": "sidebar",
    "section": "section",
    "article": "card",
}


class _TreeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "html", "attrs": {}, "children": [], "text": "", "parent": None}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.stack.append({"tag": tag, "attrs": dict(attrs), "children": [], "text": "", "parent": self.stack[-1], "skip": True})
            return
        node = {"tag": tag, "attrs": dict(attrs), "children": [], "text": "", "parent": self.stack[-1]}
        self.stack[-1]["children"].append(node)
        if tag not in {"img", "input", "meta"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        tag = tag.lower()
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i]["tag"] == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        if self.stack:
            self.stack[-1]["text"] += " " + (data or "")


def _bbox_from_attrs(attrs: dict, fallback: list[float]) -> list[float]:
    raw = attrs.get("data-bbox") or attrs.get("bbox") or ""
    if raw:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(raw))]
        if len(nums) >= 4:
            return [nums[0], nums[1], max(1.0, nums[2]), max(1.0, nums[3])]
    for key in ("left", "top", "width", "height"):
        if key not in attrs:
            return fallback
    try:
        return [float(attrs["left"]), float(attrs["top"]), max(1.0, float(attrs["width"])), max(1.0, float(attrs["height"]))]
    except (TypeError, ValueError):
        return fallback


def html_to_nodes(html: str, width: int = 1280, height: int = 800, box_map: dict | None = None) -> list[dict[str, Any]]:
    """Best-effort DOM. Uses element bboxes when present; otherwise a simple block layout."""
    parser = _TreeParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return _spatial_fallback(width, height)
    nodes: list[dict] = []
    cursor_y = [8.0]

    def walk(el, parent_id, depth, sib, region_id):
        if el.get("skip"):
            return
        tag = el.get("tag") or "div"
        text = re.sub(r"\s+", " ", (el.get("text") or "")).strip()[:80]
        attrs = el.get("attrs") or {}
        nid = len(nodes)
        role = ROLE_MAP.get(tag, "section")
        if tag == "a" and depth <= 2:
            role = "nav_item"
        fb = [16.0, cursor_y[0], max(80.0, width - 32.0), 22.0]
        bbox = _bbox_from_attrs(attrs, fb)
        if box_map and attrs.get("id") in box_map:
            bbox = [float(x) for x in box_map[attrs["id"]][:4]]
        # scale into screenshot if boxes look like a different coordinate system
        if bbox[0] + bbox[2] > width * 1.2 or bbox[1] + bbox[3] > height * 1.2:
            scale_x = width / max(bbox[0] + bbox[2], 1.0)
            scale_y = height / max(bbox[1] + bbox[3], 1.0)
            s = min(scale_x, scale_y, 1.0)
            bbox = [bbox[0] * s, bbox[1] * s, max(1.0, bbox[2] * s), max(1.0, bbox[3] * s)]
        nodes.append(
            {
                "node_id": nid,
                "tag": tag,
                "role": role,
                "text": text or attrs.get("alt") or attrs.get("aria-label") or tag,
                "parent_id": parent_id,
                "sibling_index": sib,
                "depth": depth,
                "region_id": region_id if depth > 1 else nid,
                "bbox": bbox,
                "extra": {"id": attrs.get("id", "")},
            }
        )
        if bbox is fb:
            cursor_y[0] += 22.0
        child_region = nid
        for j, ch in enumerate(el.get("children") or []):
            walk(ch, nid, depth + 1, j, child_region)

    walk(parser.root, None, 0, 0, 0)
    if len(nodes) < 3:
        return _spatial_fallback(width, height)
    # cap very bushy trees so max_anchors=48 still covers the interesting set
    if len(nodes) > 120:
        keep_roles = {"page", "header", "nav", "nav_item", "cell", "row", "table", "image", "caption", "button", "heading", "card"}
        kept = [n for n in nodes if n["role"] in keep_roles or n["depth"] <= 2]
        remap = {n["node_id"]: i for i, n in enumerate(kept[:120])}
        out = []
        for n in kept[:120]:
            nn = dict(n)
            nn["node_id"] = remap[n["node_id"]]
            pid = n.get("parent_id")
            nn["parent_id"] = remap.get(pid) if pid in remap else None
            out.append(nn)
        return out
    return nodes


def _spatial_fallback(width: int, height: int) -> list[dict]:
    from webgap.data.graph import spatial_grid_nodes

    return spatial_grid_nodes(width, height, 4, 4)


def maybe_load_sidecar_boxes(path) -> dict | None:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None
