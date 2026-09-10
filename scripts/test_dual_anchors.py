#!/usr/bin/env python3
"""Sanity checks: visual regions collapse siblings; DOM anchors keep source order."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from webgap.data.graph import (
    build_graph,
    dom_anchor_pack,
    select_dom_anchor_indices,
    visual_anchor_pack,
    visual_token_bboxes,
)
from webgap.data.webforge import render_page
from webgap.constants import REL_SIB_NEXT


def main():
    img, nodes, graph, qas, meta = render_page("rtl_toolbar", 7, 1280, 800)
    nav = [n for n in nodes if n["role"] == "nav_item" and n["bbox"][1] < 70]
    assert len(nav) == 5, nav
    xs = [n["bbox"][0] for n in nav]
    # DOM index 0 is rightmost (row-reverse)
    assert xs[0] > xs[-1], xs
    vis = visual_anchor_pack(graph, visual_token_bboxes(1280, 800, 25, 40), 48)
    idx = select_dom_anchor_indices(nodes, 48)
    n_nav_dom = sum(1 for i in idx if nodes[i]["role"] == "nav_item" and nodes[i]["bbox"][1] < 70)
    print("rtl n_regions", graph.n_regions, "k_vis", vis["k"], "n_nav_dom", n_nav_dom, "n_nodes", len(nodes))
    assert n_nav_dom == 5, n_nav_dom
    assert vis["k"] < len(nodes)
    boxes = visual_token_bboxes(1280, 800, 25, 40)
    dom = dom_anchor_pack(graph, boxes, 48)
    # sibling edges exist between DOM nav anchors
    has_sib = int((dom["rel"] == REL_SIB_NEXT).sum())
    print("dom k", dom["k"], "sib_next edges", has_sib)
    assert has_sib >= 4, has_sib
    # document order ranks are 0..k-1 not visual x
    nav_ranks = []
    for a, ni in enumerate(idx):
        if nodes[ni]["role"] == "nav_item" and nodes[ni]["bbox"][1] < 70:
            nav_ranks.append((int(dom["order"][a]), nodes[ni]["bbox"][0], nodes[ni]["sibling_index"]))
    nav_ranks.sort()
    print("dom nav (rank, x, sib)", nav_ranks)
    assert nav_ranks[0][2] == 0
    assert nav_ranks[0][1] > nav_ranks[-1][1]

    img2, nodes2, g2, qas2, _ = render_page("crossed_figures", 11, 1280, 800)
    caps = [n for n in nodes2 if n["role"] == "caption"]
    imgs = [n for n in nodes2 if n["role"] == "image"]
    print("crossed captions", [(c["text"], c["parent_id"], c["bbox"][0]) for c in caps])
    assert len(caps) == 2 and len(imgs) == 2
    # caption of left figure is drawn under the right image (larger x)
    left_fig = [n for n in nodes2 if n["role"] == "card"][0]
    cap_of_left = [c for c in caps if c["parent_id"] == left_fig["node_id"]][0]
    assert cap_of_left["bbox"][0] > 500, cap_of_left["bbox"]

    img3, nodes3, g3, _, _ = render_page("swap_nav_pair", 3, 1280, 800)
    nav3 = [n for n in nodes3 if n["role"] == "nav_item" and n["bbox"][1] < 70]
    assert len(nav3) == 5, len(nav3)
    print("ok dual-anchor diagnostics")


if __name__ == "__main__":
    main()
