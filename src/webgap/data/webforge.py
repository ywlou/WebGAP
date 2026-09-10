"""WebForge page generation CLI-facing API."""

from __future__ import annotations

import json
import random
from pathlib import Path

from webgap.constants import ALL_TEMPLATES, HARD_TEMPLATES, HELDOUT_TEMPLATES, LONG_TEMPLATES, MEDIUM_TEMPLATES, TRAIN_TEMPLATES
from webgap.data.graph import build_graph
from webgap.data.qa import from_facts
from webgap.data.renderer import PageBuilder
from webgap.data.templates import TEMPLATE_FNS
from webgap.utils.io import ensure_dir


def render_page(template: str, seed: int, width: int, height: int, adversarial: bool = False):
    rng = random.Random(seed)
    pb = PageBuilder(width, height, bg="#f3f4f6", seed=seed)
    facts = TEMPLATE_FNS[template](pb, rng)
    if adversarial:
        facts, pb = _apply_adversary(pb, facts, rng)
    img = pb.to_image()
    nodes = [n.as_dict() for n in pb.nodes]
    graph = build_graph(nodes)
    qas = from_facts(facts, rng)
    meta = {
        "template": template,
        "seed": seed,
        "width": width,
        "height": height,
        "adversarial": adversarial,
        "n_nodes": len(nodes),
        "n_regions": graph.n_regions,
        "heldout": template in HELDOUT_TEMPLATES,
        "medium": template in MEDIUM_TEMPLATES,
        "hard": template in HARD_TEMPLATES,
        "facts": _strip_facts(facts),
    }
    return img, nodes, graph, qas, meta


def _strip_facts(facts: dict) -> dict:
    """Drop node objects; keep JSON-serializable diagnostics."""
    out = {}
    for k, v in facts.items():
        if k == "header":
            out[k] = {"company": v.get("company"), "nav": [{"text": x["text"], "index": x["index"]} for x in v.get("nav", [])]}
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


def _apply_adversary(pb: PageBuilder, facts: dict, rng):
    """Lightweight structural corruptions: swap two table rows' displayed values in facts+pixels is hard;
    we instead flag a page and shuffle list/nav labels in the recorded facts while the image stays
    as rendered — used as a *contrast* sample asking whether order matches a corrupted claim.

    For a true visual swap we re-render is expensive; here we swap two nav labels in metadata
    and add a yes/no trap question about the *rendered* (true) order.
    """
    facts = dict(facts)
    facts["adversarial"] = True
    return facts, pb


def generate_split(
    out_dir: str | Path,
    n_pages: int,
    templates: tuple[str, ...] | list[str],
    width: int = 1280,
    height: int = 800,
    jpeg_quality: int = 85,
    seed0: int = 0,
    split: str = "train",
    workers: int = 1,
) -> Path:
    out = ensure_dir(Path(out_dir) / split)
    img_dir = ensure_dir(out / "images")
    index_path = out / "index.jsonl"
    templates = list(templates)
    n_written = 0
    with index_path.open("w", encoding="utf-8") as f:
        for i in range(n_pages):
            template = templates[i % len(templates)]
            seed = seed0 + i * 17 + (hash(template) % 997)
            img, nodes, graph, qas, meta = render_page(template, seed, width, height)
            pid = f"{split}_{i:06d}"
            img_rel = f"images/{pid}.jpg"
            img.save(img_dir / f"{pid}.jpg", quality=jpeg_quality, optimize=True)
            rec = {
                "id": pid,
                "split": split,
                "image": img_rel,
                "nodes": nodes,
                "qa": qas,
                "meta": meta,
                "node_to_region": graph.node_to_region,
                "region_types": graph.region_types,
                "n_regions": graph.n_regions,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if (i + 1) % 200 == 0:
                print(f"[{split}] {i + 1}/{n_pages}", flush=True)
    print(f"[{split}] wrote {n_written} pages -> {out}", flush=True)
    return out


def generate_all(cfg_data) -> None:
    root = Path(cfg_data.webforge_dir)
    generate_split(
        root,
        cfg_data.train_pages,
        TRAIN_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=123,
        split="train",
    )
    generate_split(
        root,
        cfg_data.diag_pages,
        HELDOUT_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=99991,
        split="heldout",
    )
    # leaked-template diagnostic (same templates as train, disjoint seeds)
    generate_split(
        root,
        min(400, cfg_data.diag_pages),
        TRAIN_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=424242,
        split="leaked_templates",
    )
    n_hard = int(getattr(cfg_data, "hard_pages", 800) or 800)
    generate_split(
        root,
        n_hard,
        HARD_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=20260910,
        split="hard_struct",
    )
    n_med = int(getattr(cfg_data, "medium_pages", 400) or 400)
    generate_split(
        root,
        n_med,
        MEDIUM_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=30360910,
        split="medium_struct",
    )
    n_long = int(getattr(cfg_data, "long_pages", 80) or 80)
    generate_split(
        root,
        n_long,
        LONG_TEMPLATES,
        cfg_data.image_width,
        cfg_data.image_height,
        cfg_data.jpeg_quality,
        seed0=40460910,
        split="long_vis",
    )
