"""Programmatic webpage templates with exact structure facts for QA."""

from __future__ import annotations

from webgap.data.content import (
    ADJECTIVES,
    BREADCRUMBS,
    CAPTIONS,
    CITIES,
    COLOR_HEX,
    COLORS,
    COMPANIES,
    CONFUSABLE_FAMILIES,
    FIRST_NAMES,
    FOOTER_LINKS,
    FORM_LABELS,
    LAST_NAMES,
    NAV_ITEMS,
    PLAN_NAMES,
    PRODUCTS,
    SENTENCES,
    STATUS,
    TAB_NAMES,
    TABLE_HEADERS,
    WIZARD_STEPS,
    email_of,
    money,
    pick,
    picks,
    sku,
)
from webgap.data.renderer import PageBuilder


def _header_nav(pb: PageBuilder, rng, accent: str, n_nav: int | None = None):
    company = pick(rng, COMPANIES)
    n_nav = n_nav or rng.randint(4, 6)
    nav = picks(rng, NAV_ITEMS, n_nav)
    rid = pb.new_region()
    header = pb.add_box("header", "header", company, [0, 0, pb.width, 64], pb.page_id, 1, 0, rid)
    pb.rect([0, 0, pb.width, 64], fill=accent)
    pb.text((24, 18), company, size=22, fill="#ffffff", bold=True)
    nav_node = pb.add_box("nav", "nav", " | ".join(nav), [320, 0, pb.width - 340, 64], header.node_id, 2, 0, rid)
    gap = (pb.width - 360) / max(n_nav, 1)
    items = []
    for i, lab in enumerate(nav):
        x = 340 + i * gap
        nd = pb.add_box("a", "nav_item", lab, [x, 16, min(gap - 8, 120), 32], nav_node.node_id, 3, i, rid)
        pb.text((x, 20), lab, size=16, fill="#f8fafc")
        items.append({"text": lab, "node_id": nd.node_id, "index": i})
    return {"company": company, "nav": items, "header_id": header.node_id, "nav_id": nav_node.node_id, "region": rid}


def _footer(pb: PageBuilder, rng, parent_id, y):
    rid = pb.new_region()
    links = picks(rng, FOOTER_LINKS, 4)
    h = pb.height - y
    node = pb.add_box("footer", "footer", " · ".join(links), [0, y, pb.width, h], parent_id, 1, 99, rid)
    pb.rect([0, y, pb.width, h], fill="#111827")
    pb.text((24, y + 16), " © 2026  " + "   ·   ".join(links), size=13, fill="#e5e7eb")
    items = []
    for i, lab in enumerate(links):
        nd = pb.add_box("a", "nav_item", lab, [180 + i * 140, y + 12, 120, 24], node.node_id, 2, i, rid)
        items.append({"text": lab, "node_id": nd.node_id, "index": i})
    return {"footer_id": node.node_id, "links": items}


def _image_block(pb, xywh, parent, depth, sib, rid, label, color, caption=None):
    x, y, w, h = xywh
    pb.rect([x, y, w, h], fill=color, outline="#0f172a", width=2, radius=8)
    pb.text((x + 10, y + h / 2 - 10), label, size=16, fill="#ffffff", bold=True, max_w=w - 20)
    img_n = pb.add_box("img", "image", label, xywh, parent, depth, sib, rid, extra={"caption": caption or ""})
    cap_n = None
    if caption:
        cap_n = pb.add_box(
            "figcaption",
            "caption",
            caption,
            [x, y + h + 4, w, 22],
            parent,
            depth,
            sib + 1,
            rid,
            extra={"image_text": label},
        )
        pb.text((x + 4, y + h + 4), caption, size=13, fill="#334155", max_w=w - 8)
    return img_n, cap_n


def navbar_hero(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    product = pick(rng, PRODUCTS)
    slogan = pick(rng, SENTENCES)
    rid = pb.new_region()
    hero = pb.add_box("section", "hero", product, [0, 64, pb.width, 210], pb.page_id, 1, 1, rid)
    pb.rect([0, 64, pb.width, 210], fill="#e2e8f0")
    pb.text((40, 100), product, size=36, fill="#0f172a", bold=True)
    pb.text((40, 150), slogan, size=16, fill="#334155", max_w=pb.width - 420)
    img_n, cap_n = _image_block(
        pb, [pb.width - 360, 84, 300, 160], hero.node_id, 2, 0, rid, pick(rng, CAPTIONS), accent, pick(rng, CAPTIONS)
    )
    btn = "Shop " + product.split()[0]
    btn_n = pb.add_box("button", "button", btn, [40, 200, 180, 40], hero.node_id, 2, 2, rid)
    pb.rect([40, 200, 180, 40], fill=accent, radius=6)
    pb.text((58, 208), btn, size=16, fill="#ffffff", bold=True)
    facts = {
        "template": "navbar_hero",
        "header": hdr,
        "hero_product": product,
        "slogan": slogan,
        "button": btn,
        "image": img_n.text,
        "caption": cap_n.text if cap_n else "",
        "section_name": product,
        "contained": [(btn, product), (img_n.text, product)],
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def card_grid(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    cols, rows = 3, 2
    products = picks(rng, PRODUCTS, cols * rows)
    prices = [money(rng) for _ in products]
    captions = picks(rng, CAPTIONS, cols * rows)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Featured", [0, 72, pb.width, 620], pb.page_id, 1, 1, rid)
    pb.text((32, 80), "Featured catalog", size=22, fill="#0f172a", bold=True)
    cards = []
    gap, left, top = 24, 32, 120
    cw = (pb.width - left * 2 - gap * (cols - 1)) // cols
    ch = 230
    for i, prod in enumerate(products):
        r, c = divmod(i, cols)
        x = left + c * (cw + gap)
        y = top + r * (ch + gap)
        card = pb.add_box("article", "card", prod, [x, y, cw, ch], section.node_id, 2, i, rid)
        pb.rect([x, y, cw, ch], fill="#ffffff", outline="#cbd5e1", width=2, radius=10)
        color = COLOR_HEX[COLORS[i % len(COLORS)]]
        img_n, cap_n = _image_block(pb, [x + 12, y + 12, cw - 24, 110], card.node_id, 3, 0, rid, captions[i], color, captions[i])
        pb.text((x + 14, y + 150), prod, size=16, fill="#0f172a", bold=True, max_w=cw - 28)
        pb.text((x + 14, y + 176), prices[i], size=16, fill=accent, bold=True)
        adj = pick(rng, ADJECTIVES)
        pb.text((x + 14, y + 200), adj, size=13, fill="#64748b")
        cards.append(
            {
                "title": prod,
                "price": prices[i],
                "caption": captions[i],
                "image": captions[i],
                "row": r,
                "col": c,
                "adj": adj,
                "node_id": card.node_id,
            }
        )
    facts = {"template": "card_grid", "header": hdr, "cards": cards, "section_name": "Featured catalog"}
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def data_table(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    n_rows = rng.randint(5, 7)
    headers = ["Item", "SKU", "Price", "Stock", "Region"]
    items = picks(rng, PRODUCTS, n_rows)
    rows = []
    for it in items:
        rows.append(
            {
                "Item": it,
                "SKU": sku(rng),
                "Price": money(rng),
                "Stock": str(rng.randint(2, 80)),
                "Region": pick(rng, CITIES),
            }
        )
    rid = pb.new_region()
    title = "Inventory"
    section = pb.add_box("section", "section", title, [24, 80, pb.width - 48, 620], pb.page_id, 1, 1, rid)
    pb.text((32, 88), title + " table", size=22, fill="#0f172a", bold=True)
    table = pb.add_box("table", "table", title, [32, 130, pb.width - 64, 40 + 36 * (n_rows + 1)], section.node_id, 2, 0, rid)
    col_w = (pb.width - 64) / len(headers)
    # header row
    pb.rect([32, 130, pb.width - 64, 36], fill=accent)
    head_cells = []
    for j, h in enumerate(headers):
        x = 32 + j * col_w
        pb.text((x + 8, 136), h, size=15, fill="#ffffff", bold=True)
        cell = pb.add_box("th", "cell", h, [x, 130, col_w, 36], table.node_id, 3, j, rid, extra={"row": 0, "col": j, "header": True})
        head_cells.append(cell)
    body = []
    for i, row in enumerate(rows):
        y = 166 + i * 36
        bg = "#ffffff" if i % 2 == 0 else "#f1f5f9"
        pb.rect([32, y, pb.width - 64, 36], fill=bg, outline="#cbd5e1")
        row_n = pb.add_box("tr", "row", row["Item"], [32, y, pb.width - 64, 36], table.node_id, 3, i + 1, rid, extra={"row": i + 1})
        rec = {"row_index": i + 1, "values": row, "node_id": row_n.node_id, "cells": []}
        for j, h in enumerate(headers):
            x = 32 + j * col_w
            val = row[h]
            pb.text((x + 8, y + 8), val, size=14, fill="#0f172a", max_w=col_w - 16)
            cell = pb.add_box(
                "td",
                "cell",
                val,
                [x, y, col_w, 36],
                row_n.node_id,
                4,
                j,
                rid,
                extra={"row": i + 1, "col": j, "header": h},
            )
            rec["cells"].append({"header": h, "value": val, "col": j, "node_id": cell.node_id})
        body.append(rec)
    facts = {
        "template": "data_table",
        "header": hdr,
        "table_name": title,
        "headers": headers,
        "rows": body,
        "n_rows": n_rows,
        "n_cols": len(headers),
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def contact_form(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    labels = picks(rng, FORM_LABELS, 5)
    rid = pb.new_region()
    title = "Contact " + hdr["company"]
    section = pb.add_box("section", "form", title, [80, 90, 640, 560], pb.page_id, 1, 1, rid)
    pb.rect([80, 90, 640, 560], fill="#ffffff", outline="#cbd5e1", width=2, radius=12)
    pb.text((104, 108), title, size=22, fill="#0f172a", bold=True)
    fields = []
    for i, lab in enumerate(labels):
        y = 160 + i * 72
        lab_n = pb.add_box("label", "label", lab, [110, y, 560, 22], section.node_id, 2, i * 2, rid)
        pb.text((110, y), lab, size=14, fill="#334155", bold=True)
        placeholder = "Enter " + lab.lower()
        field = pb.add_box("input", "field", placeholder, [110, y + 24, 560, 36], section.node_id, 2, i * 2 + 1, rid, extra={"label": lab})
        pb.rect([110, y + 24, 560, 36], fill="#f8fafc", outline="#94a3b8", radius=4)
        pb.text((118, y + 32), placeholder, size=14, fill="#94a3b8")
        fields.append({"label": lab, "placeholder": placeholder, "node_id": field.node_id, "label_id": lab_n.node_id})
    btn = "Send message"
    btn_n = pb.add_box("button", "button", btn, [110, 160 + 5 * 72, 200, 42], section.node_id, 2, 99, rid)
    pb.rect([110, 160 + 5 * 72, 200, 42], fill=accent, radius=6)
    pb.text((130, 170 + 5 * 72), btn, size=16, fill="#ffffff", bold=True)
    side_rid = pb.new_region()
    img_n, cap_n = _image_block(
        pb, [780, 140, 420, 260], pb.page_id, 1, 2, side_rid, pick(rng, CAPTIONS), accent, pick(rng, CAPTIONS)
    )
    facts = {
        "template": "contact_form",
        "header": hdr,
        "form_title": title,
        "fields": fields,
        "button": btn,
        "image": img_n.text,
        "caption": cap_n.text if cap_n else "",
        "contained": [(f["label"], title) for f in fields] + [(btn, title)],
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def article_sidebar(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    title = pick(rng, PRODUCTS) + " notes"
    paras = picks(rng, SENTENCES, 4, unique=False)
    toc = picks(rng, ["Specs", "Care", "Warranty", "FAQ", "Downloads", "Related"], 4)
    rid = pb.new_region()
    article = pb.add_box("article", "section", title, [32, 80, 820, 620], pb.page_id, 1, 1, rid)
    pb.text((40, 90), title, size=26, fill="#0f172a", bold=True)
    y = 140
    para_nodes = []
    for i, p in enumerate(paras):
        pb.text((40, y), p, size=16, fill="#1e293b", max_w=780)
        n = pb.add_box("p", "text", p, [40, y, 780, 48], article.node_id, 2, i, rid)
        para_nodes.append(p)
        y += 56
    img_n, cap_n = _image_block(pb, [40, y, 360, 160], article.node_id, 2, 10, rid, pick(rng, CAPTIONS), accent, pick(rng, CAPTIONS))
    sid_rid = pb.new_region()
    side = pb.add_box("aside", "sidebar", "On this page", [880, 80, 360, 520], pb.page_id, 1, 2, sid_rid)
    pb.rect([880, 80, 360, 520], fill="#ffffff", outline="#cbd5e1", width=2, radius=10)
    pb.text((900, 100), "On this page", size=18, fill="#0f172a", bold=True)
    toc_items = []
    for i, t in enumerate(toc):
        yy = 150 + i * 44
        pb.rect([900, yy, 320, 36], fill="#f1f5f9", radius=4)
        pb.text((912, yy + 8), f"{i + 1}. {t}", size=15, fill=accent, bold=True)
        n = pb.add_box("a", "list_item", t, [900, yy, 320, 36], side.node_id, 2, i, sid_rid)
        toc_items.append({"text": t, "index": i, "node_id": n.node_id})
    facts = {
        "template": "article_sidebar",
        "header": hdr,
        "title": title,
        "paragraphs": para_nodes,
        "toc": toc_items,
        "image": img_n.text,
        "caption": cap_n.text if cap_n else "",
        "contained": [(t["text"], "On this page") for t in toc_items],
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def product_list(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    ordered = bool(rng.randint(0, 1))
    n = rng.randint(5, 7)
    items = picks(rng, PRODUCTS, n)
    rid = pb.new_region()
    title = "Packing list" if ordered else "Related items"
    section = pb.add_box("section", "section", title, [48, 88, 900, 600], pb.page_id, 1, 1, rid)
    pb.text((56, 96), title, size=22, fill="#0f172a", bold=True)
    list_n = pb.add_box("ol" if ordered else "ul", "list", title, [56, 140, 860, 40 * n], section.node_id, 2, 0, rid)
    recs = []
    for i, it in enumerate(items):
        y = 140 + i * 48
        mark = f"{i + 1}." if ordered else "•"
        pb.rect([56, y, 860, 42], fill="#ffffff", outline="#e2e8f0")
        pb.text((70, y + 10), f"{mark}  {it}", size=16, fill="#0f172a", bold=True)
        extra = pick(rng, ADJECTIVES)
        pb.text((520, y + 10), extra, size=14, fill="#64748b")
        nd = pb.add_box("li", "list_item", it, [56, y, 860, 42], list_n.node_id, 3, i, rid, extra={"index": i + 1, "adj": extra})
        recs.append({"text": it, "index": i + 1, "adj": extra, "node_id": nd.node_id})
    facts = {
        "template": "product_list",
        "header": hdr,
        "title": title,
        "ordered": ordered,
        "items": recs,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def breadcrumb_gallery(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    crumbs = BREADCRUMBS[: rng.randint(3, 5)]
    rid = pb.new_region()
    bc = pb.add_box("nav", "breadcrumb", " / ".join(crumbs), [32, 80, pb.width - 64, 36], pb.page_id, 1, 1, rid)
    crumb_items = []
    x = 40
    for i, c in enumerate(crumbs):
        t = c if i == 0 else " / " + c
        tw, _, _ = pb.text((x, 86), t, size=14, fill=accent if i < len(crumbs) - 1 else "#0f172a", bold=i == len(crumbs) - 1)
        nd = pb.add_box("a", "nav_item", c, [x, 80, tw + 8, 28], bc.node_id, 2, i, rid)
        crumb_items.append({"text": c, "index": i, "node_id": nd.node_id})
        x += tw + 4
    n_img = 6
    caps = picks(rng, CAPTIONS, n_img)
    images = []
    g_rid = pb.new_region()
    gallery = pb.add_box("section", "section", "Gallery", [24, 130, pb.width - 48, 560], pb.page_id, 1, 2, g_rid)
    pb.text((32, 136), "Gallery", size=22, fill="#0f172a", bold=True)
    cols = 3
    for i, cap in enumerate(caps):
        r, c = divmod(i, cols)
        x = 40 + c * 400
        y = 180 + r * 240
        color = COLOR_HEX[COLORS[i % len(COLORS)]]
        img_n, cap_n = _image_block(pb, [x, y, 360, 160], gallery.node_id, 2, i, g_rid, cap, color, cap)
        images.append({"image": cap, "caption": cap, "row": r, "col": c, "node_id": img_n.node_id})
    facts = {
        "template": "breadcrumb_gallery",
        "header": hdr,
        "breadcrumbs": crumb_items,
        "images": images,
        "section_name": "Gallery",
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def pricing_cards(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    plans = PLAN_NAMES[:3]
    prices = [money(rng, 9, 29), money(rng, 39, 79), money(rng, 99, 199)]
    feats = [
        ["1 seat", "Email support", "Community"],
        ["5 seats", "Priority inbox", "SSO"],
        ["Unlimited seats", "Phone support", "Audit log"],
    ]
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Pricing", [20, 80, pb.width - 40, 620], pb.page_id, 1, 1, rid)
    pb.text((40, 90), "Pricing", size=24, fill="#0f172a", bold=True)
    cards = []
    for i, plan in enumerate(plans):
        x = 40 + i * 410
        card = pb.add_box("article", "card", plan, [x, 140, 380, 480], section.node_id, 2, i, rid)
        pb.rect([x, 140, 380, 480], fill="#ffffff", outline=accent if i == 1 else "#cbd5e1", width=3, radius=12)
        pb.text((x + 24, 160), plan, size=22, fill="#0f172a", bold=True)
        pb.text((x + 24, 200), prices[i] + " / mo", size=28, fill=accent, bold=True)
        feat_recs = []
        for j, f in enumerate(feats[i]):
            yy = 270 + j * 48
            pb.text((x + 36, yy), "•  " + f, size=16, fill="#1e293b")
            nd = pb.add_box("li", "list_item", f, [x + 24, yy, 330, 36], card.node_id, 3, j, rid)
            feat_recs.append({"text": f, "index": j, "node_id": nd.node_id})
        btn = "Choose " + plan
        btn_n = pb.add_box("button", "button", btn, [x + 24, 540, 320, 44], card.node_id, 3, 10, rid)
        pb.rect([x + 24, 540, 320, 44], fill=accent, radius=6)
        pb.text((x + 90, 550), btn, size=16, fill="#ffffff", bold=True)
        cards.append({"plan": plan, "price": prices[i], "features": feat_recs, "button": btn, "index": i, "node_id": card.node_id})
    facts = {"template": "pricing_cards", "header": hdr, "cards": cards, "section_name": "Pricing"}
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def dashboard_stats(pb: PageBuilder, rng) -> dict:
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    stats = [
        ("Orders", str(rng.randint(120, 980))),
        ("Refunds", str(rng.randint(2, 40))),
        ("Revenue", money(rng, 800, 9000)),
        ("Alerts", str(rng.randint(0, 12))),
    ]
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Overview", [24, 80, pb.width - 48, 200], pb.page_id, 1, 1, rid)
    pb.text((32, 88), "Overview", size=22, fill="#0f172a", bold=True)
    recs = []
    for i, (k, v) in enumerate(stats):
        x = 32 + i * 310
        card = pb.add_box("div", "stat", k, [x, 130, 290, 130], section.node_id, 2, i, rid, extra={"value": v})
        pb.rect([x, 130, 290, 130], fill="#ffffff", outline="#cbd5e1", width=2, radius=10)
        pb.text((x + 20, 148), k, size=14, fill="#64748b")
        pb.text((x + 20, 178), v, size=28, fill=accent, bold=True)
        recs.append({"name": k, "value": v, "index": i, "node_id": card.node_id})
    # mini table
    n_rows = 4
    headers = ["Owner", "City", "Status"]
    owners = [pick(rng, FIRST_NAMES) + " " + pick(rng, LAST_NAMES) for _ in range(n_rows)]
    t_rid = pb.new_region()
    table_n = pb.add_box("table", "table", "Team", [32, 330, pb.width - 64, 280], pb.page_id, 1, 2, t_rid)
    pb.text((32, 310), "Team", size=18, fill="#0f172a", bold=True)
    col_w = (pb.width - 64) / 3
    pb.rect([32, 330, pb.width - 64, 36], fill=accent)
    for j, h in enumerate(headers):
        pb.text((40 + j * col_w, 338), h, size=14, fill="#ffffff", bold=True)
        pb.add_box("th", "cell", h, [32 + j * col_w, 330, col_w, 36], table_n.node_id, 2, j, t_rid, extra={"row": 0, "col": j})
    body = []
    for i, owner in enumerate(owners):
        y = 366 + i * 40
        city = pick(rng, CITIES)
        st = pick(rng, STATUS)
        pb.rect([32, y, pb.width - 64, 40], fill="#ffffff" if i % 2 == 0 else "#f8fafc", outline="#e2e8f0")
        vals = [owner, city, st]
        rec = {"row_index": i + 1, "values": dict(zip(headers, vals)), "cells": []}
        row_n = pb.add_box("tr", "row", owner, [32, y, pb.width - 64, 40], table_n.node_id, 2, i + 1, t_rid)
        for j, val in enumerate(vals):
            pb.text((40 + j * col_w, y + 10), val, size=14, fill="#0f172a")
            cell = pb.add_box("td", "cell", val, [32 + j * col_w, y, col_w, 40], row_n.node_id, 3, j, t_rid, extra={"row": i + 1, "col": j, "header": headers[j]})
            rec["cells"].append({"header": headers[j], "value": val, "col": j})
        body.append(rec)
    facts = {
        "template": "dashboard_stats",
        "header": hdr,
        "stats": recs,
        "table_name": "Team",
        "headers": headers,
        "rows": body,
        "n_rows": n_rows,
        "n_cols": 3,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def tabs_panel(pb: PageBuilder, rng) -> dict:
    """Held-out family: tabbed content."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    tabs = TAB_NAMES[: rng.randint(3, 5)]
    active = rng.randint(0, len(tabs) - 1)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Product tabs", [32, 80, pb.width - 64, 620], pb.page_id, 1, 1, rid)
    tab_recs = []
    tw = (pb.width - 80) / len(tabs)
    for i, t in enumerate(tabs):
        x = 40 + i * tw
        bg = accent if i == active else "#e2e8f0"
        fg = "#ffffff" if i == active else "#0f172a"
        pb.rect([x, 90, tw - 8, 44], fill=bg, radius=6)
        pb.text((x + 16, 100), t, size=16, fill=fg, bold=True)
        nd = pb.add_box("button", "tab", t, [x, 90, tw - 8, 44], section.node_id, 2, i, rid, extra={"active": i == active})
        tab_recs.append({"text": t, "index": i, "active": i == active, "node_id": nd.node_id})
    panel_text = pick(rng, SENTENCES)
    product = pick(rng, PRODUCTS)
    panel = pb.add_box("div", "panel", tabs[active], [40, 150, pb.width - 80, 480], section.node_id, 2, 99, rid)
    pb.rect([40, 150, pb.width - 80, 480], fill="#ffffff", outline="#cbd5e1", width=2)
    pb.text((64, 170), f"{tabs[active]} — {product}", size=22, fill="#0f172a", bold=True)
    pb.text((64, 220), panel_text, size=16, fill="#334155", max_w=pb.width - 160)
    img_n, cap_n = _image_block(pb, [64, 280, 420, 220], panel.node_id, 3, 0, rid, pick(rng, CAPTIONS), accent, pick(rng, CAPTIONS))
    facts = {
        "template": "tabs_panel",
        "header": hdr,
        "tabs": tab_recs,
        "active_tab": tabs[active],
        "product": product,
        "panel_text": panel_text,
        "image": img_n.text,
        "caption": cap_n.text if cap_n else "",
        "contained": [(product, tabs[active]), (img_n.text, tabs[active])],
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def comparison_matrix(pb: PageBuilder, rng) -> dict:
    """Held-out family: product comparison matrix (table-like but unseen chrome)."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    plans = PLAN_NAMES[:4]
    attrs = ["Storage", "Seats", "SLA", "SSO", "API"]
    grid = {
        "Storage": [pick(rng, ["10 GB", "100 GB", "1 TB", "Unlimited"]) for _ in plans],
        "Seats": [str(n) for n in [1, 5, 20, 99]],
        "SLA": ["none", "99.9%", "99.95%", "99.99%"],
        "SSO": ["No", "No", "Yes", "Yes"],
        "API": ["No", "Yes", "Yes", "Yes"],
    }
    rid = pb.new_region()
    title = "Plan comparison"
    section = pb.add_box("section", "section", title, [24, 80, pb.width - 48, 640], pb.page_id, 1, 1, rid)
    pb.text((32, 90), title, size=22, fill="#0f172a", bold=True)
    table = pb.add_box("table", "table", title, [32, 140, pb.width - 64, 500], section.node_id, 2, 0, rid)
    n_cols = 1 + len(plans)
    col_w = (pb.width - 64) / n_cols
    pb.rect([32, 140, pb.width - 64, 48], fill=accent)
    pb.add_box("th", "cell", "Feature", [32, 140, col_w, 48], table.node_id, 3, 0, rid, extra={"row": 0, "col": 0})
    pb.text((44, 152), "Feature", size=15, fill="#ffffff", bold=True)
    headers = ["Feature"] + plans
    for j, p in enumerate(plans):
        pb.text((44 + (j + 1) * col_w, 152), p, size=15, fill="#ffffff", bold=True)
        pb.add_box("th", "cell", p, [32 + (j + 1) * col_w, 140, col_w, 48], table.node_id, 3, j + 1, rid, extra={"row": 0, "col": j + 1})
    body = []
    for i, attr in enumerate(attrs):
        y = 188 + i * 56
        pb.rect([32, y, pb.width - 64, 56], fill="#ffffff" if i % 2 == 0 else "#f1f5f9", outline="#cbd5e1")
        row_n = pb.add_box("tr", "row", attr, [32, y, pb.width - 64, 56], table.node_id, 3, i + 1, rid)
        pb.text((44, y + 16), attr, size=15, fill="#0f172a", bold=True)
        rec = {"row_index": i + 1, "values": {"Feature": attr}, "cells": [{"header": "Feature", "value": attr, "col": 0}]}
        rec["values"].update({plans[j]: grid[attr][j] for j in range(len(plans))})
        pb.add_box("td", "cell", attr, [32, y, col_w, 56], row_n.node_id, 4, 0, rid, extra={"row": i + 1, "col": 0, "header": "Feature"})
        for j, p in enumerate(plans):
            val = grid[attr][j]
            pb.text((44 + (j + 1) * col_w, y + 16), val, size=15, fill="#0f172a")
            pb.add_box("td", "cell", val, [32 + (j + 1) * col_w, y, col_w, 56], row_n.node_id, 4, j + 1, rid, extra={"row": i + 1, "col": j + 1, "header": p})
            rec["cells"].append({"header": p, "value": val, "col": j + 1})
        body.append(rec)
    facts = {
        "template": "comparison_matrix",
        "header": hdr,
        "table_name": title,
        "headers": headers,
        "rows": body,
        "n_rows": len(attrs),
        "n_cols": n_cols,
        "plans": plans,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def checkout_wizard(pb: PageBuilder, rng) -> dict:
    """Held-out family: multi-step checkout."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent)
    steps = WIZARD_STEPS
    active = rng.randint(0, 3)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Checkout", [40, 80, pb.width - 80, 640], pb.page_id, 1, 1, rid)
    pb.text((48, 90), "Checkout", size=22, fill="#0f172a", bold=True)
    step_recs = []
    for i, s in enumerate(steps):
        x = 60 + i * 240
        bg = accent if i == active else "#e2e8f0"
        fg = "#ffffff" if i == active else "#334155"
        pb.rect([x, 140, 200, 40], fill=bg, radius=20)
        pb.text((x + 28, 150), f"{i + 1}. {s}", size=14, fill=fg, bold=True)
        nd = pb.add_box("li", "list_item", s, [x, 140, 200, 40], section.node_id, 2, i, rid, extra={"active": i == active, "index": i + 1})
        step_recs.append({"text": s, "index": i + 1, "active": i == active, "node_id": nd.node_id})
        if i < len(steps) - 1:
            pb.text((x + 204, 148), "→", size=16, fill="#94a3b8")
    product = pick(rng, PRODUCTS)
    price = money(rng)
    city = pick(rng, CITIES)
    name = pick(rng, FIRST_NAMES) + " " + pick(rng, LAST_NAMES)
    panel = pb.add_box("div", "panel", steps[active], [60, 210, pb.width - 120, 430], section.node_id, 2, 99, rid)
    pb.rect([60, 210, pb.width - 120, 430], fill="#ffffff", outline="#cbd5e1", width=2, radius=10)
    pb.text((88, 230), f"Step {active + 1}: {steps[active]}", size=20, fill="#0f172a", bold=True)
    facts_lines = [
        f"Item: {product}",
        f"Price: {price}",
        f"Ship to: {city}",
        f"Contact: {name}",
    ]
    for i, line in enumerate(facts_lines):
        pb.text((88, 290 + i * 40), line, size=16, fill="#1e293b")
        pb.add_box("p", "text", line, [88, 290 + i * 40, 600, 32], panel.node_id, 3, i, rid)
    img_n, cap_n = _image_block(pb, [720, 280, 420, 220], panel.node_id, 3, 10, rid, pick(rng, CAPTIONS), accent, product)
    btn = "Continue to " + (steps[active + 1] if active + 1 < len(steps) else "Finish")
    pb.rect([88, 560, 260, 44], fill=accent, radius=6)
    pb.text((110, 570), btn, size=16, fill="#ffffff", bold=True)
    pb.add_box("button", "button", btn, [88, 560, 260, 44], panel.node_id, 3, 20, rid)
    facts = {
        "template": "checkout_wizard",
        "header": hdr,
        "steps": step_recs,
        "active_step": steps[active],
        "active_index": active + 1,
        "product": product,
        "price": price,
        "city": city,
        "name": name,
        "button": btn,
        "image": img_n.text,
        "caption": cap_n.text if cap_n else product,
        "contained": [(product, steps[active]), (btn, steps[active])],
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def confusable_catalog(pb: PageBuilder, rng) -> dict:
    """Hard: near-duplicate product titles so OCR of a heading is not enough."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=5)
    family = list(pick(rng, CONFUSABLE_FAMILIES))
    rng.shuffle(family)
    titles = family[:6]
    # Tight price band so a nearby number is a plausible distractor.
    base = rng.randint(18, 26)
    prices = [f"${base + i * 0.5:.2f}" for i in range(6)]
    rng.shuffle(prices)
    skus = [f"SKU-{rng.randint(400, 499)}{i}" for i in range(6)]
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Catalog", [16, 72, pb.width - 32, 670], pb.page_id, 1, 1, rid)
    pb.text((24, 80), "Catalog — similar SKUs", size=18, fill="#0f172a", bold=True)
    cards = []
    for i, title in enumerate(titles):
        row, col = divmod(i, 3)
        x, y = 24 + col * 410, 120 + row * 300
        card = pb.add_box("article", "card", title, [x, y, 392, 280], section.node_id, 2, i, rid)
        pb.rect([x, y, 392, 280], fill="#ffffff", outline="#cbd5e1", width=2, radius=8)
        pb.text((x + 14, y + 12), title, size=15, fill="#0f172a", bold=True, max_w=360)
        pb.text((x + 14, y + 42), skus[i], size=12, fill="#64748b", mono=True)
        pb.text((x + 14, y + 68), prices[i], size=22, fill=accent, bold=True)
        img_n, cap_n = _image_block(
            pb,
            [x + 14, y + 110, 364, 120],
            card.node_id,
            3,
            0,
            rid,
            title,
            accent,
            f"Photo of {title}",
        )
        cards.append(
            {
                "title": title,
                "price": prices[i],
                "sku": skus[i],
                "row": row,
                "col": col,
                "index": i,
                "caption": cap_n.text if cap_n else "",
                "node_id": card.node_id,
            }
        )
    facts = {"template": "confusable_catalog", "header": hdr, "cards": cards, "family": titles}
    _footer(pb, rng, pb.page_id, pb.height - 48)
    return facts


def dense_ledger(pb: PageBuilder, rng) -> dict:
    """Hard: compact numeric table; answers are specific cells, not the title."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=4)
    n_rows = 8
    headers = ["SKU", "Qty", "Unit", "Total", "Bin"]
    rid = pb.new_region()
    title = "Warehouse ledger"
    section = pb.add_box("section", "section", title, [20, 72, pb.width - 40, 680], pb.page_id, 1, 1, rid)
    pb.text((28, 80), title, size=18, fill="#0f172a", bold=True)
    table = pb.add_box("table", "table", title, [28, 118, pb.width - 56, 580], section.node_id, 2, 0, rid)
    col_w = (pb.width - 56) / 5
    pb.rect([28, 118, pb.width - 56, 32], fill=accent)
    for j, h in enumerate(headers):
        pb.text((36 + j * col_w, 124), h, size=13, fill="#ffffff", bold=True)
        pb.add_box("th", "cell", h, [28 + j * col_w, 118, col_w, 32], table.node_id, 3, j, rid, extra={"row": 0, "col": j})
    body = []
    seed_sku = rng.randint(1040, 1080)
    for i in range(n_rows):
        y = 150 + i * 36
        sku_s = f"A-{seed_sku + i}"
        qty = str(10 + i)
        unit = f"${(9.10 + i * 0.07):.2f}"
        total = f"${(9.10 + i * 0.07) * (10 + i):.2f}"
        bin_s = f"B-{12 + (i % 6):02d}"
        vals = [sku_s, qty, unit, total, bin_s]
        bg = "#ffffff" if i % 2 == 0 else "#f1f5f9"
        pb.rect([28, y, pb.width - 56, 36], fill=bg, outline="#e2e8f0")
        row_n = pb.add_box("tr", "row", sku_s, [28, y, pb.width - 56, 36], table.node_id, 3, i + 1, rid)
        rec = {"row_index": i + 1, "values": dict(zip(headers, vals)), "cells": []}
        for j, val in enumerate(vals):
            pb.text((36 + j * col_w, y + 8), val, size=13, fill="#0f172a", mono=True)
            pb.add_box(
                "td",
                "cell",
                val,
                [28 + j * col_w, y, col_w, 36],
                row_n.node_id,
                4,
                j,
                rid,
                extra={"row": i + 1, "col": j, "header": headers[j]},
            )
            rec["cells"].append({"header": headers[j], "value": val, "col": j})
        body.append(rec)
    facts = {
        "template": "dense_ledger",
        "header": hdr,
        "table_name": title,
        "headers": headers,
        "rows": body,
        "n_rows": n_rows,
        "n_cols": 5,
    }
    _footer(pb, rng, pb.page_id, pb.height - 44)
    return facts


def rtl_toolbar(pb: PageBuilder, rng) -> dict:
    """Hard: visual left-to-right ≠ DOM sibling_index (flex-direction: row-reverse)."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    company = pick(rng, COMPANIES)
    nav = picks(rng, NAV_ITEMS, 5)
    rid = pb.new_region()
    header = pb.add_box("header", "header", company, [0, 0, pb.width, 64], pb.page_id, 1, 0, rid)
    pb.rect([0, 0, pb.width, 64], fill=accent)
    pb.text((24, 18), company, size=20, fill="#ffffff", bold=True)
    nav_node = pb.add_box("nav", "nav", " | ".join(nav), [280, 0, pb.width - 300, 64], header.node_id, 2, 0, rid)
    slot = (pb.width - 320) / 5
    items = []
    for i, lab in enumerate(nav):
        # sibling_index i is document order; x decreases so i=0 is rightmost.
        x = pb.width - 40 - (i + 1) * slot
        nd = pb.add_box("a", "nav_item", lab, [x, 16, min(slot - 8, 120), 32], nav_node.node_id, 3, i, rid)
        pb.text((x, 20), lab, size=15, fill="#f8fafc")
        items.append({"text": lab, "node_id": nd.node_id, "index": i, "x": x})
    visual = sorted(items, key=lambda d: d["x"])
    product = pick(rng, PRODUCTS)
    rid2 = pb.new_region()
    hero = pb.add_box("section", "hero", product, [0, 64, pb.width, 220], pb.page_id, 1, 1, rid2)
    pb.rect([0, 64, pb.width, 220], fill="#e2e8f0")
    pb.text((40, 110), product, size=32, fill="#0f172a", bold=True)
    pb.text((40, 160), "Navigation is drawn right-to-left (source order preserved).", size=14, fill="#334155")
    facts = {
        "template": "rtl_toolbar",
        "header": {"company": company, "nav": items},
        "nav_source": items,
        "nav_visual": visual,
        "source_first": items[0]["text"],
        "visual_leftmost": visual[0]["text"],
        "visual_rightmost": visual[-1]["text"],
        "hero_product": product,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def crossed_figures(pb: PageBuilder, rng) -> dict:
    """Hard: each figure's caption is drawn under the *other* image."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=4)
    left_p, right_p = picks(rng, PRODUCTS, 2)
    left_cap, right_cap = picks(rng, CAPTIONS, 2)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Studio pair", [24, 80, pb.width - 48, 620], pb.page_id, 1, 1, rid)
    pb.text((32, 90), "Studio pair", size=20, fill="#0f172a", bold=True)
    # Left figure owns left image + left caption node, but caption pixels sit under the right image.
    fig_l = pb.add_box("figure", "card", left_p, [40, 140, 560, 480], section.node_id, 2, 0, rid)
    fig_r = pb.add_box("figure", "card", right_p, [680, 140, 560, 480], section.node_id, 2, 1, rid)
    img_l, _ = _image_block(pb, [56, 160, 520, 240], fig_l.node_id, 3, 0, rid, left_p, accent, None)
    img_r, _ = _image_block(pb, [696, 160, 520, 240], fig_r.node_id, 3, 0, rid, right_p, COLOR_HEX[pick(rng, COLORS)], None)
    # Crossed caption *boxes* (DOM parent = own figure; pixels under the other image).
    cap_l = pb.add_box(
        "figcaption",
        "caption",
        left_cap,
        [696, 416, 520, 28],
        fig_l.node_id,
        3,
        1,
        rid,
        extra={"image_text": left_p, "visual_under": right_p},
    )
    pb.text((704, 416), left_cap, size=14, fill="#334155", max_w=500)
    cap_r = pb.add_box(
        "figcaption",
        "caption",
        right_cap,
        [56, 416, 520, 28],
        fig_r.node_id,
        3,
        1,
        rid,
        extra={"image_text": right_p, "visual_under": left_p},
    )
    pb.text((64, 416), right_cap, size=14, fill="#334155", max_w=500)
    pb.text((56, 460), f"Left product: {left_p}", size=14, fill="#0f172a")
    pb.text((696, 460), f"Right product: {right_p}", size=14, fill="#0f172a")
    facts = {
        "template": "crossed_figures",
        "header": hdr,
        "left_product": left_p,
        "right_product": right_p,
        "left_caption_dom": left_cap,
        "right_caption_dom": right_cap,
        "text_under_left_image": right_cap,
        "text_under_right_image": left_cap,
        "image_left": img_l.text,
        "image_right": img_r.text,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def swap_nav_pair(pb: PageBuilder, rng) -> dict:
    """Medium: only one adjacent pair is visually swapped; rest of the bar matches DOM."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    company = pick(rng, COMPANIES)
    nav = picks(rng, NAV_ITEMS, 5)
    rid = pb.new_region()
    header = pb.add_box("header", "header", company, [0, 0, pb.width, 64], pb.page_id, 1, 0, rid)
    pb.rect([0, 0, pb.width, 64], fill=accent)
    pb.text((24, 18), company, size=20, fill="#ffffff", bold=True)
    nav_node = pb.add_box("nav", "nav", " | ".join(nav), [280, 0, pb.width - 300, 64], header.node_id, 2, 0, rid)
    slot = (pb.width - 320) / 5
    swap_i = rng.randint(1, 3)
    visual_pos = list(range(5))
    visual_pos[swap_i], visual_pos[swap_i + 1] = visual_pos[swap_i + 1], visual_pos[swap_i]
    items = []
    for i, lab in enumerate(nav):
        col = visual_pos[i]
        x = 300 + col * slot
        nd = pb.add_box("a", "nav_item", lab, [x, 16, min(slot - 8, 120), 32], nav_node.node_id, 3, i, rid)
        pb.text((x, 20), lab, size=15, fill="#f8fafc")
        items.append({"text": lab, "node_id": nd.node_id, "index": i, "x": x, "visual_col": col})
    visual = sorted(items, key=lambda d: d["x"])
    product = pick(rng, PRODUCTS)
    rid2 = pb.new_region()
    hero = pb.add_box("section", "hero", product, [0, 64, pb.width, 220], pb.page_id, 1, 1, rid2)
    pb.rect([0, 64, pb.width, 220], fill="#e2e8f0")
    pb.text((40, 110), product, size=32, fill="#0f172a", bold=True)
    pb.text((40, 160), f"One adjacent pair is drawn swapped (source indices {swap_i + 1}/{swap_i + 2}).", size=14, fill="#334155")
    facts = {
        "template": "swap_nav_pair",
        "header": {"company": company, "nav": items},
        "nav_source": items,
        "nav_visual": visual,
        "source_first": items[0]["text"],
        "visual_leftmost": visual[0]["text"],
        "swap_a": items[swap_i]["text"],
        "swap_b": items[swap_i + 1]["text"],
        "hero_product": product,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def offset_captions(pb: PageBuilder, rng) -> dict:
    """Medium: each caption is shifted toward the neighbor but still closer to its own image."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=4)
    left_p, right_p = picks(rng, PRODUCTS, 2)
    left_cap, right_cap = picks(rng, CAPTIONS, 2)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Studio pair", [24, 80, pb.width - 48, 620], pb.page_id, 1, 1, rid)
    pb.text((32, 90), "Studio pair", size=20, fill="#0f172a", bold=True)
    fig_l = pb.add_box("figure", "card", left_p, [40, 140, 560, 480], section.node_id, 2, 0, rid)
    fig_r = pb.add_box("figure", "card", right_p, [680, 140, 560, 480], section.node_id, 2, 1, rid)
    img_l, _ = _image_block(pb, [56, 160, 520, 240], fig_l.node_id, 3, 0, rid, left_p, accent, None)
    img_r, _ = _image_block(pb, [696, 160, 520, 240], fig_r.node_id, 3, 0, rid, right_p, COLOR_HEX[pick(rng, COLORS)], None)
    # Shift 160px toward the other image; still overlapping own figure more than the other.
    cap_l = pb.add_box("figcaption", "caption", left_cap, [56 + 160, 416, 400, 28], fig_l.node_id, 3, 1, rid, extra={"image_text": left_p})
    pb.text((64 + 160, 416), left_cap, size=14, fill="#334155", max_w=380)
    cap_r = pb.add_box("figcaption", "caption", right_cap, [696 - 80, 416, 400, 28], fig_r.node_id, 3, 1, rid, extra={"image_text": right_p})
    pb.text((704 - 80, 416), right_cap, size=14, fill="#334155", max_w=380)
    facts = {
        "template": "offset_captions",
        "header": hdr,
        "left_product": left_p,
        "right_product": right_p,
        "left_caption_dom": left_cap,
        "right_caption_dom": right_cap,
        "text_under_left_image": left_cap,
        "text_under_right_image": right_cap,
        "image_left": img_l.text,
        "image_right": img_r.text,
    }
    _footer(pb, rng, pb.page_id, pb.height - 52)
    return facts


def near_confusable(pb: PageBuilder, rng) -> dict:
    """Medium: similar titles but color + size cues make the target easier than confusable_catalog."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=5)
    family = list(pick(rng, CONFUSABLE_FAMILIES))
    rng.shuffle(family)
    titles = family[:4]
    prices = [money(rng) for _ in titles]
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Catalog", [16, 72, pb.width - 32, 670], pb.page_id, 1, 1, rid)
    pb.text((24, 80), "Catalog", size=22, fill="#0f172a", bold=True)
    cards = []
    for i, title in enumerate(titles):
        col = i % 2
        row = i // 2
        x, y = 40 + col * 620, 130 + row * 300
        color = COLOR_HEX[COLORS[i % len(COLORS)]]
        card = pb.add_box("article", "card", title, [x, y, 580, 270], section.node_id, 2, i, rid)
        pb.rect([x, y, 580, 270], fill="#ffffff", outline=color, width=4, radius=8)
        pb.rect([x, y, 16, 270], fill=color)
        pb.text((x + 28, y + 16), title, size=22, fill="#0f172a", bold=True, max_w=520)
        sku_s = f"SKU-{rng.randint(200, 299)}{i}"
        pb.text((x + 28, y + 56), sku_s, size=14, fill="#64748b", mono=True)
        pb.text((x + 28, y + 88), prices[i], size=26, fill=color, bold=True)
        img_n, cap_n = _image_block(pb, [x + 28, y + 130, 524, 110], card.node_id, 3, 0, rid, title, color, f"Photo of {title}")
        cards.append({"title": title, "price": prices[i], "sku": sku_s, "row": row, "col": col, "index": i, "caption": cap_n.text if cap_n else "", "node_id": card.node_id})
    facts = {"template": "near_confusable", "header": hdr, "cards": cards, "family": titles}
    _footer(pb, rng, pb.page_id, pb.height - 48)
    return facts


def compact_ledger(pb: PageBuilder, rng) -> dict:
    """Medium: 4×3 table with larger type; still cell-level answers."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=4)
    n_rows = 4
    headers = ["SKU", "Qty", "Bin"]
    rid = pb.new_region()
    title = "Stock sheet"
    section = pb.add_box("section", "section", title, [40, 80, pb.width - 80, 640], pb.page_id, 1, 1, rid)
    pb.text((48, 90), title, size=22, fill="#0f172a", bold=True)
    table = pb.add_box("table", "table", title, [48, 140, pb.width - 96, 480], section.node_id, 2, 0, rid)
    col_w = (pb.width - 96) / 3
    pb.rect([48, 140, pb.width - 96, 44], fill=accent)
    for j, h in enumerate(headers):
        pb.text((64 + j * col_w, 150), h, size=16, fill="#ffffff", bold=True)
        pb.add_box("th", "cell", h, [48 + j * col_w, 140, col_w, 44], table.node_id, 3, j, rid, extra={"row": 0, "col": j})
    body = []
    seed_sku = rng.randint(2040, 2080)
    for i in range(n_rows):
        y = 196 + i * 80
        sku_s = f"C-{seed_sku + i}"
        qty = str(4 + i * 2)
        bin_s = f"D-{3 + i:02d}"
        vals = [sku_s, qty, bin_s]
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        pb.rect([48, y, pb.width - 96, 76], fill=bg, outline="#e2e8f0")
        row_n = pb.add_box("tr", "row", sku_s, [48, y, pb.width - 96, 76], table.node_id, 3, i + 1, rid)
        rec = {"row_index": i + 1, "values": dict(zip(headers, vals)), "cells": []}
        for j, val in enumerate(vals):
            pb.text((64 + j * col_w, y + 24), val, size=18, fill="#0f172a", mono=True)
            pb.add_box("td", "cell", val, [48 + j * col_w, y, col_w, 76], row_n.node_id, 4, j, rid, extra={"row": i + 1, "col": j, "header": headers[j]})
            rec["cells"].append({"header": headers[j], "value": val, "col": j})
        body.append(rec)
    facts = {"template": "compact_ledger", "header": hdr, "table_name": title, "headers": headers, "rows": body, "n_rows": n_rows, "n_cols": 3}
    _footer(pb, rng, pb.page_id, pb.height - 44)
    return facts


def tall_mosaic(pb: PageBuilder, rng) -> dict:
    """Long visual page: 4×4 tiles so visual-token count is high enough for ERPR."""
    accent = COLOR_HEX[pick(rng, COLORS)]
    hdr = _header_nav(pb, rng, accent, n_nav=6)
    titles = picks(rng, PRODUCTS, 8) + picks(rng, PRODUCTS, 8)
    rid = pb.new_region()
    section = pb.add_box("section", "section", "Mosaic", [8, 70, pb.width - 16, pb.height - 130], pb.page_id, 1, 1, rid)
    pb.text((16, 76), "Mosaic wall", size=16, fill="#0f172a", bold=True)
    tiles = []
    cols, rows = 4, 4
    gap, left, top = 10, 16, 104
    cw = (pb.width - left * 2 - gap * (cols - 1)) // cols
    ch = 150
    for i in range(16):
        r, c = divmod(i, cols)
        x = left + c * (cw + gap)
        y = top + r * (ch + gap)
        title = titles[i % len(titles)] + f" {i+1}"
        card = pb.add_box("article", "card", title, [x, y, cw, ch], section.node_id, 2, i, rid)
        color = COLOR_HEX[COLORS[i % len(COLORS)]]
        pb.rect([x, y, cw, ch], fill=color, outline="#0f172a", width=1, radius=4)
        pb.text((x + 6, y + 8), title, size=12, fill="#ffffff", bold=True, max_w=cw - 12)
        price = money(rng)
        pb.text((x + 6, y + 40), price, size=14, fill="#f8fafc")
        tiles.append({"title": title, "price": price, "row": r, "col": c, "index": i})
    facts = {"template": "tall_mosaic", "header": hdr, "cards": tiles, "n_tiles": 16}
    _footer(pb, rng, pb.page_id, pb.height - 48)
    return facts


TEMPLATE_FNS = {
    "navbar_hero": navbar_hero,
    "card_grid": card_grid,
    "data_table": data_table,
    "contact_form": contact_form,
    "article_sidebar": article_sidebar,
    "product_list": product_list,
    "breadcrumb_gallery": breadcrumb_gallery,
    "pricing_cards": pricing_cards,
    "dashboard_stats": dashboard_stats,
    "tabs_panel": tabs_panel,
    "comparison_matrix": comparison_matrix,
    "checkout_wizard": checkout_wizard,
    "confusable_catalog": confusable_catalog,
    "dense_ledger": dense_ledger,
    "rtl_toolbar": rtl_toolbar,
    "crossed_figures": crossed_figures,
    "swap_nav_pair": swap_nav_pair,
    "offset_captions": offset_captions,
    "near_confusable": near_confusable,
    "compact_ledger": compact_ledger,
    "tall_mosaic": tall_mosaic,
}
