"""Heterogeneous web graph + STAR visual-token assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from webgap.constants import (
    DOM_ANCHOR_ROLES,
    NUM_REL_TYPES,
    REL_ABOVE,
    REL_ANCESTOR,
    REL_BELOW,
    REL_CHILD,
    REL_CMA,
    REL_CONTAINED,
    REL_CONTAINS,
    REL_DESCENDANT,
    REL_LEFT,
    REL_NONE,
    REL_OVERLAP,
    REL_PARENT,
    REL_RIGHT,
    REL_SAME_REGION,
    REL_SIB_NEXT,
    REL_SIB_PREV,
)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def _contains(a, b, slack=4.0) -> bool:
    return a[0] - slack <= b[0] and a[1] - slack <= b[1] and a[0] + a[2] + slack >= b[0] + b[2] and a[1] + a[3] + slack >= b[1] + b[3]


@dataclass
class WebGraph:
    nodes: list[dict]
    edges: list[tuple[int, int, int]]  # src, dst, rel
    n_regions: int
    node_to_region: list[int]
    region_types: list[int]
    extra: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def build_graph(nodes: list[dict], max_degree: int = 8) -> WebGraph:
    n = len(nodes)
    by_id = {nd["node_id"]: i for i, nd in enumerate(nodes)}
    edges: list[tuple[int, int, int]] = []

    # DOM parent/child / sibling / ancestor
    children: dict[int, list[int]] = {}
    for i, nd in enumerate(nodes):
        pid = nd.get("parent_id")
        if pid is None or pid not in by_id:
            continue
        p = by_id[pid]
        edges.append((p, i, REL_CHILD))
        edges.append((i, p, REL_PARENT))
        children.setdefault(p, []).append(i)
        # climb ancestors
        climb = nodes[p]
        hops = 0
        while climb.get("parent_id") in by_id and hops < 4:
            a = by_id[climb["parent_id"]]
            edges.append((a, i, REL_DESCENDANT))
            edges.append((i, a, REL_ANCESTOR))
            climb = nodes[a]
            hops += 1

    for p, ch in children.items():
        # Source/DOM order, not visual order. Spatial LEFT/RIGHT edges already
        # encode geometry; mixing the two made TRB a duplicate of the layout.
        ch_sorted = sorted(ch, key=lambda j: (int(nodes[j].get("sibling_index", 0)), j))
        for a, b in zip(ch_sorted, ch_sorted[1:]):
            edges.append((a, b, REL_SIB_NEXT))
            edges.append((b, a, REL_SIB_PREV))

    # Spatial among "interesting" nodes (not page/header giant boxes)
    interesting = [
        i
        for i, nd in enumerate(nodes)
        if nd["role"] in {"card", "cell", "nav_item", "button", "image", "caption", "list_item", "stat", "tab", "field", "label"}
    ]
    for a in interesting:
        ba = nodes[a]["bbox"]
        ca = _center(ba)
        scored = []
        for b in interesting:
            if a == b:
                continue
            bb = nodes[b]["bbox"]
            cb = _center(bb)
            dist = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
            scored.append((dist, b, bb, cb))
        scored.sort()
        for dist, b, bb, cb in scored[:max_degree]:
            iou = _iou(ba, bb)
            rel = REL_NONE
            if _contains(ba, bb):
                rel = REL_CONTAINS
            elif _contains(bb, ba):
                rel = REL_CONTAINED
            elif iou > 0.15:
                rel = REL_OVERLAP
            else:
                dx, dy = cb[0] - ca[0], cb[1] - ca[1]
                if abs(dx) >= abs(dy):
                    rel = REL_RIGHT if dx > 0 else REL_LEFT
                else:
                    rel = REL_BELOW if dy > 0 else REL_ABOVE
            edges.append((a, b, rel))

    # Cross-modal: image ↔ caption. Prefer the figure/DOM parent (so a caption
    # drawn under the *other* image still binds to its own figure). Spatial
    # fallback only for unpaired captions.
    imgs = [i for i, nd in enumerate(nodes) if nd["role"] == "image"]
    caps = [i for i, nd in enumerate(nodes) if nd["role"] == "caption"]
    paired_cap = set()
    for i in imgs:
        for j in caps:
            if nodes[i].get("parent_id") is None:
                continue
            if nodes[i].get("parent_id") == nodes[j].get("parent_id"):
                edges.append((i, j, REL_CMA))
                edges.append((j, i, REL_CMA))
                paired_cap.add(j)
    for i in imgs:
        for j in caps:
            if j in paired_cap:
                continue
            if _iou(nodes[i]["bbox"], nodes[j]["bbox"]) > 0:
                edges.append((i, j, REL_CMA))
                edges.append((j, i, REL_CMA))
            elif nodes[j]["bbox"][1] >= nodes[i]["bbox"][1] and abs(_center(nodes[i]["bbox"])[0] - _center(nodes[j]["bbox"])[0]) < 80:
                if abs(nodes[j]["bbox"][1] - (nodes[i]["bbox"][1] + nodes[i]["bbox"][3])) < 40:
                    edges.append((i, j, REL_CMA))
                    edges.append((j, i, REL_CMA))

    # same region
    for i, nd in enumerate(nodes):
        for j, nd2 in enumerate(nodes):
            if i >= j:
                continue
            if nd.get("region_id", 0) == nd2.get("region_id", 0) and nd["role"] not in {"page"}:
                edges.append((i, j, REL_SAME_REGION))
                edges.append((j, i, REL_SAME_REGION))

    regions = sorted({int(nd.get("region_id", 0)) for nd in nodes})
    region_index = {r: k for k, r in enumerate(regions)}
    node_to_region = [region_index[int(nd.get("region_id", 0))] for nd in nodes]
    # region type = majority role hashed into a small set
    region_types = []
    for r in regions:
        roles = [nd["role"] for nd in nodes if int(nd.get("region_id", 0)) == r]
        key = roles[0] if roles else "section"
        region_types.append(hash(key) % 16)

    return WebGraph(
        nodes=nodes,
        edges=edges,
        n_regions=len(regions),
        node_to_region=node_to_region,
        region_types=region_types,
    )


def region_relation_matrix(graph: WebGraph, max_k: int) -> np.ndarray:
    """[K,K] discrete relation between region anchors (strongest typed edge)."""
    k = min(graph.n_regions, max_k)
    rel = np.zeros((k, k), dtype=np.int64)
    # priority: CMA > parent > sibling > spatial > same_region
    prio = {
        REL_CMA: 10,
        REL_PARENT: 9,
        REL_CHILD: 9,
        REL_SIB_NEXT: 8,
        REL_SIB_PREV: 8,
        REL_CONTAINS: 7,
        REL_CONTAINED: 7,
        REL_LEFT: 6,
        REL_RIGHT: 6,
        REL_ABOVE: 6,
        REL_BELOW: 6,
        REL_SAME_REGION: 3,
        REL_OVERLAP: 5,
        REL_ANCESTOR: 4,
        REL_DESCENDANT: 4,
    }
    best = np.zeros((k, k), dtype=np.int64)
    for s, d, r in graph.edges:
        a = graph.node_to_region[s]
        b = graph.node_to_region[d]
        if a >= k or b >= k or a == b:
            continue
        sc = prio.get(r, 1)
        if sc >= best[a, b]:
            best[a, b] = sc
            rel[a, b] = r
    # spatial fallback using region bboxes
    region_bb = []
    for rid in range(k):
        xs = [graph.nodes[i]["bbox"] for i, rr in enumerate(graph.node_to_region) if rr == rid]
        if not xs:
            region_bb.append([0, 0, 1, 1])
            continue
        x0 = min(b[0] for b in xs)
        y0 = min(b[1] for b in xs)
        x1 = max(b[0] + b[2] for b in xs)
        y1 = max(b[1] + b[3] for b in xs)
        region_bb.append([x0, y0, x1 - x0, y1 - y0])
    for a in range(k):
        for b in range(k):
            if a == b or rel[a, b] != REL_NONE:
                continue
            ca, cb = _center(region_bb[a]), _center(region_bb[b])
            dx, dy = cb[0] - ca[0], cb[1] - ca[1]
            if abs(dx) >= abs(dy):
                rel[a, b] = REL_RIGHT if dx > 0 else REL_LEFT
            else:
                rel[a, b] = REL_BELOW if dy > 0 else REL_ABOVE
    return rel


def visual_token_bboxes(orig_w: int, orig_h: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Return [H*W, 4] bboxes in original screenshot pixels (x,y,w,h)."""
    boxes = np.zeros((grid_h * grid_w, 4), dtype=np.float32)
    tw, th = orig_w / grid_w, orig_h / grid_h
    idx = 0
    for i in range(grid_h):
        for j in range(grid_w):
            boxes[idx] = [j * tw, i * th, tw, th]
            idx += 1
    return boxes


def star_assignment(
    token_bboxes: np.ndarray,
    graph: WebGraph,
    max_k: int,
) -> np.ndarray:
    """Soft assignment [n_tokens, K] from token boxes to region anchors via IoU."""
    k = min(graph.n_regions, max_k)
    n_tok = token_bboxes.shape[0]
    assign = np.zeros((n_tok, k), dtype=np.float32)
    region_bb = []
    for rid in range(k):
        xs = [graph.nodes[i]["bbox"] for i, rr in enumerate(graph.node_to_region) if rr == rid]
        if not xs:
            region_bb.append([0, 0, 1, 1])
            continue
        x0 = min(b[0] for b in xs)
        y0 = min(b[1] for b in xs)
        x1 = max(b[0] + b[2] for b in xs)
        y1 = max(b[1] + b[3] for b in xs)
        region_bb.append([x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)])
    for t in range(n_tok):
        tb = token_bboxes[t]
        for r in range(k):
            assign[t, r] = _iou(tb, region_bb[r])
        s = assign[t].sum()
        if s < 1e-6:
            # nearest region by center
            tc = _center(tb)
            dists = [(_center(region_bb[r])[0] - tc[0]) ** 2 + (_center(region_bb[r])[1] - tc[1]) ** 2 for r in range(k)]
            assign[t, int(np.argmin(dists))] = 1.0
        else:
            assign[t] /= s
    return assign


def spatial_grid_nodes(width: int, height: int, gh: int = 4, gw: int = 4) -> list[dict]:
    """Fallback graph when HTML/DOM is unavailable (real screenshots)."""
    nodes = [
        {
            "node_id": 0,
            "tag": "html",
            "role": "page",
            "text": "",
            "parent_id": None,
            "sibling_index": 0,
            "depth": 0,
            "region_id": 0,
            "bbox": [0, 0, width, height],
            "extra": {},
        }
    ]
    nid = 1
    cw, ch = width / gw, height / gh
    for i in range(gh):
        for j in range(gw):
            nodes.append(
                {
                    "node_id": nid,
                    "tag": "div",
                    "role": "section",
                    "text": f"cell-{i}-{j}",
                    "parent_id": 0,
                    "sibling_index": i * gw + j,
                    "depth": 1,
                    "region_id": nid,
                    "bbox": [j * cw, i * ch, cw, ch],
                    "extra": {},
                }
            )
            nid += 1
    return nodes


def _region_bboxes(graph: WebGraph, k: int) -> list[list[float]]:
    region_bb = []
    for rid in range(k):
        xs = [graph.nodes[i]["bbox"] for i, rr in enumerate(graph.node_to_region) if rr == rid]
        if not xs:
            region_bb.append([0.0, 0.0, 1.0, 1.0])
            continue
        x0 = min(b[0] for b in xs)
        y0 = min(b[1] for b in xs)
        x1 = max(b[0] + b[2] for b in xs)
        y1 = max(b[1] + b[3] for b in xs)
        region_bb.append([x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)])
    return region_bb


def _rank_xy(bboxes: list[list[float]], max_k: int) -> np.ndarray:
    """Reading-order ranks (top-to-bottom, then left-to-right)."""
    k = min(len(bboxes), max_k)
    order = np.zeros((max_k,), dtype=np.int64)
    if k == 0:
        return order
    keys = [(_center(bboxes[i])[1], _center(bboxes[i])[0], i) for i in range(k)]
    keys.sort()
    for rank, (*_, i) in enumerate(keys):
        order[i] = rank
    return order


def select_dom_anchor_indices(nodes: list[dict], max_k: int) -> list[int]:
    """One DOM anchor per interesting node, in *document* (insertion) order."""
    preferred = [i for i, nd in enumerate(nodes) if nd.get("role") in DOM_ANCHOR_ROLES]
    if len(preferred) < 3:
        preferred = [i for i, nd in enumerate(nodes) if nd.get("role") not in {"page", "header", "footer"}]
    preferred.sort(key=lambda i: (int(nodes[i].get("node_id", i)), i))
    if len(preferred) <= max_k:
        return preferred
    # Keep high-value roles, then fill remaining slots in document order.
    prio_roles = {"nav_item", "caption", "image", "card", "button", "tab", "cell"}
    head = [i for i in preferred if nodes[i].get("role") in prio_roles]
    rest = [i for i in preferred if i not in set(head)]
    chosen = (head + rest)[:max_k]
    chosen.sort(key=lambda i: (int(nodes[i].get("node_id", i)), i))
    return chosen


def node_relation_matrix(graph: WebGraph, anchor_idx: list[int], max_k: int) -> np.ndarray:
    """[K,K] typed relations between DOM-node anchors (not visual regions)."""
    k = min(len(anchor_idx), max_k)
    rel = np.zeros((max_k, max_k), dtype=np.int64)
    mapping = {i: a for a, i in enumerate(anchor_idx[:k])}
    # edges stored as node list indices
    prio = {
        REL_CMA: 10,
        REL_PARENT: 9,
        REL_CHILD: 9,
        REL_SIB_NEXT: 8,
        REL_SIB_PREV: 8,
        REL_CONTAINS: 7,
        REL_CONTAINED: 7,
        REL_LEFT: 6,
        REL_RIGHT: 6,
        REL_ABOVE: 6,
        REL_BELOW: 6,
        REL_SAME_REGION: 3,
        REL_OVERLAP: 5,
        REL_ANCESTOR: 4,
        REL_DESCENDANT: 4,
    }
    best = np.zeros((k, k), dtype=np.int64)
    for s, d, r in graph.edges:
        a = mapping.get(s)
        b = mapping.get(d)
        if a is None or b is None or a == b:
            continue
        sc = prio.get(r, 1)
        if sc >= best[a, b]:
            best[a, b] = sc
            rel[a, b] = r
    return rel


def star_assignment_boxes(token_bboxes: np.ndarray, target_bboxes: list[list[float]], max_k: int) -> np.ndarray:
    """Soft IoU assignment [n_tokens, max_k] onto an arbitrary list of boxes."""
    k = min(len(target_bboxes), max_k)
    n_tok = token_bboxes.shape[0]
    assign = np.zeros((n_tok, max_k), dtype=np.float32)
    if k <= 0:
        return assign
    for t in range(n_tok):
        tb = token_bboxes[t]
        for r in range(k):
            assign[t, r] = _iou(tb, target_bboxes[r])
        s = float(assign[t, :k].sum())
        if s < 1e-6:
            tc = _center(tb)
            dists = [
                (_center(target_bboxes[r])[0] - tc[0]) ** 2 + (_center(target_bboxes[r])[1] - tc[1]) ** 2 for r in range(k)
            ]
            assign[t, int(np.argmin(dists))] = 1.0
        else:
            assign[t, :k] /= s
    return assign


def visual_anchor_pack(graph: WebGraph, token_bboxes: np.ndarray, max_k: int) -> dict[str, np.ndarray | int]:
    k = min(max(graph.n_regions, 1), max_k)
    rel = region_relation_matrix(graph, max_k)
    if rel.shape[0] < max_k:
        rr = np.zeros((max_k, max_k), dtype=np.int64)
        rr[: rel.shape[0], : rel.shape[1]] = rel
        rel = rr
    types = np.zeros((max_k,), dtype=np.int64)
    nrt = np.array(graph.region_types[:k], dtype=np.int64)
    types[: len(nrt)] = nrt
    bb = _region_bboxes(graph, k)
    assign = star_assignment_boxes(token_bboxes, bb, max_k)
    order = _rank_xy(bb, max_k)
    return {"assign": assign, "rel": rel, "types": types, "k": k, "order": order}


def dom_anchor_pack(graph: WebGraph, token_bboxes: np.ndarray, max_k: int) -> dict[str, np.ndarray | int]:
    idx = select_dom_anchor_indices(graph.nodes, max_k)
    k = len(idx)
    rel = node_relation_matrix(graph, idx, max_k)
    types = np.zeros((max_k,), dtype=np.int64)
    order = np.zeros((max_k,), dtype=np.int64)
    bbs = []
    for a, ni in enumerate(idx):
        nd = graph.nodes[ni]
        bbs.append(nd["bbox"])
        types[a] = hash(nd.get("role", "section")) % 16
        # document-order rank among selected anchors (not visual x)
        order[a] = a
    if k == 0:
        bbs = [[0.0, 0.0, 1.0, 1.0]]
        k = 1
        types[0] = 1
    assign = star_assignment_boxes(token_bboxes, bbs, max_k)
    return {"assign": assign, "rel": rel, "types": types, "k": k, "order": order}


def serialize_graph_text(graph: WebGraph, max_nodes: int = 40) -> str:
    """GraphToken-style linearized structure (baseline B3)."""
    lines = ["[WEB GRAPH]"]
    for nd in graph.nodes[:max_nodes]:
        if nd["role"] in {"page"}:
            continue
        txt = (nd["text"] or "")[:40].replace("\n", " ")
        lines.append(f"N{nd['node_id']}:{nd['role']} \"{txt}\"")
    # a few typed edges
    shown = 0
    for s, d, r in graph.edges:
        if shown >= 60:
            break
        if r in (REL_PARENT, REL_CHILD, REL_SIB_NEXT, REL_CMA, REL_CONTAINS):
            lines.append(f"E {s}-{d}:{r}")
            shown += 1
    return "\n".join(lines)
