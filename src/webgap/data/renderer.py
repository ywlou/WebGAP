"""Minimal box-layout + PIL renderer with exact DOM-like nodes and bboxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


@dataclass
class Node:
    node_id: int
    tag: str
    role: str
    text: str = ""
    parent_id: int | None = None
    sibling_index: int = 0
    depth: int = 0
    region_id: int = 0
    bbox: list[float] = field(default_factory=lambda: [0, 0, 0, 0])  # x,y,w,h
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tag": self.tag,
            "role": self.role,
            "text": self.text,
            "parent_id": self.parent_id,
            "sibling_index": self.sibling_index,
            "depth": self.depth,
            "region_id": self.region_id,
            "bbox": [round(x, 1) for x in self.bbox],
            "extra": self.extra,
        }


class PageBuilder:
    """Immediate-mode layout: callers push boxes; we record nodes + draw."""

    def __init__(self, width: int, height: int, bg: str = "#f6f7fb", seed: int = 0):
        self.width = width
        self.height = height
        self.bg = bg
        self.img = Image.new("RGB", (width, height), bg)
        self.draw = ImageDraw.Draw(self.img)
        self.nodes: list[Node] = []
        self._next_id = 0
        self._region_counter = 0
        page = self._add("html", "page", "", None, 0, 0)
        page.bbox = [0, 0, width, height]
        self.page_id = page.node_id

    def new_region(self) -> int:
        rid = self._region_counter
        self._region_counter += 1
        return rid

    def _add(self, tag, role, text, parent_id, depth, sibling_index, region_id=0, extra=None) -> Node:
        n = Node(
            node_id=self._next_id,
            tag=tag,
            role=role,
            text=text,
            parent_id=parent_id,
            sibling_index=sibling_index,
            depth=depth,
            region_id=region_id,
            extra=extra or {},
        )
        self._next_id += 1
        self.nodes.append(n)
        return n

    def rect(self, xywh, fill, outline=None, width=1, radius=0):
        x, y, w, h = [int(v) for v in xywh]
        box = [x, y, x + w, y + h]
        if radius > 0:
            self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        else:
            self.draw.rectangle(box, fill=fill, outline=outline, width=width)

    def text(self, xy, s, size=16, fill="#111827", bold=False, mono=False, max_w=None):
        fnt = font(size, bold=bold, mono=mono)
        x, y = xy
        if max_w is not None:
            s = self._fit(s, fnt, max_w)
        self.draw.text((x, y), s, font=fnt, fill=fill)
        tw, th = _text_size(self.draw, s, fnt)
        return tw, th, s

    def _fit(self, s: str, fnt, max_w: int) -> str:
        if _text_size(self.draw, s, fnt)[0] <= max_w:
            return s
        ell = "…"
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            cand = s[:mid] + ell
            if _text_size(self.draw, cand, fnt)[0] <= max_w:
                lo = mid + 1
            else:
                hi = mid
        return s[: max(0, lo - 1)] + ell

    def add_box(
        self,
        tag: str,
        role: str,
        text: str,
        bbox,
        parent_id: int,
        depth: int,
        sibling_index: int,
        region_id: int,
        extra=None,
    ) -> Node:
        n = self._add(tag, role, text, parent_id, depth, sibling_index, region_id, extra)
        n.bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        return n

    def to_image(self) -> Image.Image:
        return self.img
