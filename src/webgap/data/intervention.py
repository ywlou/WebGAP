"""Paired WebForge pages for P5: visual reorder, DOM reorder, conflict, STAR perm curve."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from webgap.data.content import COLOR_HEX, COLORS, COMPANIES, NAV_ITEMS, PRODUCTS, pick, picks
from webgap.data.renderer import PageBuilder
from webgap.data.templates import _footer
from webgap.data.webforge import render_page
from webgap.utils.io import ensure_dir

CONFLICT_TEMPLATES = ("rtl_toolbar", "crossed_figures", "swap_nav_pair")


def _nav_page(seed: int, labels: list[str], visual_reverse: bool, width: int = 1280, height: int = 800):
    rng = random.Random(seed)
    pb = PageBuilder(width, height, bg="#f3f4f6", seed=seed)
    accent = COLOR_HEX[pick(rng, COLORS)]
    company = pick(rng, COMPANIES)
    rid = pb.new_region()
    header = pb.add_box("header", "header", company, [0, 0, pb.width, 64], pb.page_id, 1, 0, rid)
    pb.rect([0, 0, pb.width, 64], fill=accent)
    pb.text((24, 18), company, size=20, fill="#ffffff", bold=True)
    n = len(labels)
    slot = (pb.width - 320) / max(n, 1)
    nav_node = pb.add_box("nav", "nav", " | ".join(labels), [280, 0, pb.width - 300, 64], header.node_id, 2, 0, rid)
    items = []
    for i, lab in enumerate(labels):
        x = pb.width - 40 - (i + 1) * slot if visual_reverse else 340 + i * slot
        nd = pb.add_box("a", "nav_item", lab, [x, 16, min(slot - 8, 120), 32], nav_node.node_id, 3, i, rid)
        pb.text((x, 20), lab, size=15, fill="#f8fafc")
        items.append({"text": lab, "node_id": nd.node_id, "index": i, "x": x})
    product = pick(rng, PRODUCTS)
    rid2 = pb.new_region()
    pb.add_box("section", "hero", product, [0, 64, pb.width, 220], pb.page_id, 1, 1, rid2)
    pb.rect([0, 64, pb.width, 220], fill="#e2e8f0")
    pb.text((40, 110), product, size=32, fill="#0f172a", bold=True)
    order = "right-to-left" if visual_reverse else "left-to-right"
    pb.text((40, 160), f"Navigation is drawn {order} (source order preserved).", size=14, fill="#334155")
    _footer(pb, rng, pb.page_id, pb.height - 52)
    visual = sorted(items, key=lambda d: d["x"])
    facts = {
        "source_first": items[0]["text"],
        "visual_leftmost": visual[0]["text"],
        "visual_rightmost": visual[-1]["text"],
        "hero_product": product,
        "nav": items,
    }
    return pb.to_image(), [n.as_dict() for n in pb.nodes], facts


def _dom_reorder_nodes(nodes: list[dict]) -> list[dict]:
    """Reverse sibling_index among nav_item children of the header nav; pixels unchanged."""
    out = copy.deepcopy(nodes)
    nav_items = [n for n in out if n.get("role") == "nav_item" and n.get("depth", 0) >= 3]
    # header nav is typically depth 3; footer items also nav_item — keep only those under the first nav
    if len(nav_items) < 2:
        nav_items = [n for n in out if n.get("role") == "nav_item"]
    header_nav = [n for n in nav_items if n.get("bbox", [0, 0, 0, 0])[1] < 80]
    if len(header_nav) < 2:
        header_nav = nav_items[:5]
    order = sorted(header_nav, key=lambda n: int(n.get("sibling_index", 0)))
    for i, nd in enumerate(reversed(order)):
        nd["sibling_index"] = i
    return out


def _qs_nav(facts: dict) -> list[dict]:
    return [
        {"question": "What is the leftmost navigation item on the screen?", "answer": facts["visual_leftmost"], "probe": "visual"},
        {"question": "What is the first item in the navigation source order (item 1 in the markup list)?", "answer": facts["source_first"], "probe": "dom"},
        {"question": "What product name is shown in the hero section?", "answer": facts["hero_product"], "probe": "bind"},
    ]


def generate_intervention_split(
    out_dir: str | Path,
    n_visual: int = 48,
    n_dom: int = 48,
    n_conflict: int = 32,
    n_curve: int = 32,
    seed0: int = 20260915,
) -> Path:
    root = ensure_dir(out_dir)
    img_dir = ensure_dir(root / "images")
    rows = []

    def _save(img, name: str) -> str:
        p = img_dir / name
        img.save(p, "JPEG", quality=88)
        return str(p)

    rng = random.Random(seed0)
    for i in range(n_visual):
        seed = seed0 + i
        labels = picks(random.Random(seed), NAV_ITEMS, 5)
        img_a, nodes_a, fa = _nav_page(seed, labels, visual_reverse=False)
        img_b, nodes_b, fb = _nav_page(seed, labels, visual_reverse=True)
        pair_id = f"vis_{i:04d}"
        pa = _save(img_a, f"{pair_id}_a.jpg")
        pb = _save(img_b, f"{pair_id}_b.jpg")
        for arm, path, nodes, facts in (("a", pa, nodes_a, fa), ("b", pb, nodes_b, fb)):
            for q in _qs_nav(facts):
                rows.append(
                    {
                        "id": f"{pair_id}_{arm}_{q['probe']}",
                        "pair_id": pair_id,
                        "arm": arm,
                        "family": "visual_reorder",
                        "image": path,
                        "nodes": nodes,
                        "question": q["question"],
                        "answer": q["answer"],
                        "probe": q["probe"],
                    }
                )

    for i in range(n_dom):
        seed = seed0 + 1000 + i
        labels = picks(random.Random(seed), NAV_ITEMS, 5)
        img, nodes, facts = _nav_page(seed, labels, visual_reverse=False)
        pair_id = f"dom_{i:04d}"
        path = _save(img, f"{pair_id}.jpg")
        nodes_b = _dom_reorder_nodes(nodes)
        # After reversing sibling_index, source_first is the old last header nav item.
        header = sorted(
            [n for n in nodes if n.get("role") == "nav_item" and n.get("bbox", [0, 0, 0, 0])[1] < 80],
            key=lambda n: int(n.get("sibling_index", 0)),
        )
        facts_b = dict(facts)
        if len(header) >= 2:
            facts_b["source_first"] = header[-1]["text"]
        for arm, nds, fac in (("a", nodes, facts), ("b", nodes_b, facts_b)):
            for q in _qs_nav(fac):
                rows.append(
                    {
                        "id": f"{pair_id}_{arm}_{q['probe']}",
                        "pair_id": pair_id,
                        "arm": arm,
                        "family": "dom_reorder",
                        "image": path,
                        "nodes": nds,
                        "question": q["question"],
                        "answer": q["answer"],
                        "probe": q["probe"],
                    }
                )

    for i in range(n_conflict):
        tmpl = CONFLICT_TEMPLATES[i % len(CONFLICT_TEMPLATES)]
        seed = seed0 + 2000 + i
        img, nodes, graph, qas, meta = render_page(tmpl, seed, 1280, 800)
        path = _save(img, f"conf_{i:04d}.jpg")
        for j, qa in enumerate(qas):
            if qa.get("probe") not in {"visual", "dom", "bind"}:
                continue
            rows.append(
                {
                    "id": f"conf_{i:04d}_{j}",
                    "pair_id": f"conf_{i:04d}",
                    "arm": "solo",
                    "family": "conflict",
                    "template": tmpl,
                    "image": path,
                    "nodes": nodes,
                    "question": qa["question"],
                    "answer": str(qa["answer"]),
                    "probe": qa.get("probe", ""),
                }
            )

    for i in range(n_curve):
        seed = seed0 + 3000 + i
        img, nodes, graph, qas, meta = render_page("navbar_hero", seed, 1280, 800)
        path = _save(img, f"curve_{i:04d}.jpg")
        picked = [qa for qa in qas if qa.get("probe") in {"visual", "bind", "dom"}][:2] or qas[:2]
        for j, qa in enumerate(picked):
            rows.append(
                {
                    "id": f"curve_{i:04d}_{j}",
                    "pair_id": f"curve_{i:04d}",
                    "arm": "solo",
                    "family": "anchor_curve",
                    "image": path,
                    "nodes": nodes,
                    "question": qa["question"],
                    "answer": str(qa["answer"]),
                    "probe": qa.get("probe", ""),
                }
            )

    index = root / "index.jsonl"
    with index.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "n_rows": len(rows),
        "n_visual_pairs": n_visual,
        "n_dom_pairs": n_dom,
        "n_conflict": n_conflict,
        "n_curve": n_curve,
        "seed0": seed0,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return index
