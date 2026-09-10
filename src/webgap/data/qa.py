"""Rule-based H1–H5 questions from template facts. Answers are exact strings."""

from __future__ import annotations

from typing import Any

from webgap.constants import HARD_TEMPLATES, MEDIUM_TEMPLATES


def _qa(q: str, a: str, htype: str, extra: dict | None = None) -> dict[str, Any]:
    rec = {"question": q, "answer": str(a), "type": htype, "hops": 2 if htype == "H5" else 1}
    if extra:
        rec.update(extra)
    return rec


def from_facts(facts: dict, rng) -> list[dict]:
    t = facts.get("template")
    qs: list[dict] = []
    hdr = facts.get("header") or {}
    nav = hdr.get("nav") or []

    conflict = t in HARD_TEMPLATES or t in MEDIUM_TEMPLATES
    if t not in HARD_TEMPLATES and t not in MEDIUM_TEMPLATES and nav:
        i = rng.randrange(len(nav))
        qs.append(_qa(f"What is the navigation item at position {i + 1} from the left?", nav[i]["text"], "H3"))
        if i + 1 < len(nav):
            qs.append(
                _qa(
                    f"Which navigation label is immediately to the right of '{nav[i]['text']}'?",
                    nav[i + 1]["text"],
                    "H2",
                )
            )
        qs.append(_qa("How many items are in the top navigation bar?", str(len(nav)), "H3"))

    if t == "navbar_hero":
        qs.append(_qa(f"Which page section contains the button '{facts['button']}'?", facts["hero_product"], "H1"))
        qs.append(_qa("What is the caption under the hero image?", facts["caption"], "H4"))
        qs.append(_qa("What product name is shown in the hero heading?", facts["hero_product"], "H3"))
        if nav:
            qs.append(
                _qa(
                    f"The first navigation item is '{nav[0]['text']}'. What product is advertised in the hero?",
                    facts["hero_product"],
                    "H5",
                )
            )

    if t == "card_grid":
        cards = facts["cards"]
        c = cards[rng.randrange(len(cards))]
        qs.append(_qa(f"What is the price shown on the card titled '{c['title']}'?", c["price"], "H1"))
        qs.append(_qa(f"What caption is under the image on the '{c['title']}' card?", c["caption"], "H4"))
        # spatial neighbors in grid
        right = [x for x in cards if x["row"] == c["row"] and x["col"] == c["col"] + 1]
        if right:
            qs.append(_qa(f"Which product card is immediately to the right of '{c['title']}'?", right[0]["title"], "H2"))
        below = [x for x in cards if x["row"] == c["row"] + 1 and x["col"] == c["col"]]
        if below:
            qs.append(_qa(f"Which product card is directly below '{c['title']}'?", below[0]["title"], "H2"))
        qs.append(
            _qa(
                f"The card in row {c['row'] + 1}, column {c['col'] + 1} shows which product?",
                c["title"],
                "H3",
            )
        )

    if t in ("data_table", "dashboard_stats", "comparison_matrix") and "rows" in facts:
        rows = facts["rows"]
        headers = facts["headers"]
        r = rows[rng.randrange(len(rows))]
        # pick a non-first column if possible
        col = 1 if len(headers) > 1 else 0
        h = headers[col]
        val = r["values"][h] if h in r["values"] else list(r["values"].values())[col]
        qs.append(
            _qa(
                f"In the {facts.get('table_name', 'table')}, what is the value in row {r['row_index']}, column '{h}'?",
                val,
                "H3",
            )
        )
        first_h = headers[0]
        first_v = r["values"][first_h]
        qs.append(_qa(f"Which row of the table contains '{first_v}' in column '{first_h}'?", str(r["row_index"]), "H1"))
        if r["row_index"] < facts["n_rows"]:
            nxt = rows[r["row_index"]]  # next row (1-index)
            qs.append(
                _qa(
                    f"Which value appears in column '{first_h}' in the row immediately below '{first_v}'?",
                    nxt["values"][first_h],
                    "H2",
                )
            )
        if len(headers) >= 3:
            h2 = headers[2]
            qs.append(
                _qa(
                    f"For the row whose '{first_h}' is '{first_v}', what is the '{h2}'?",
                    r["values"][h2],
                    "H5",
                )
            )
        qs.append(_qa(f"How many data rows are in the {facts.get('table_name', 'table')} (excluding the header)?", str(facts["n_rows"]), "H3"))

    if t == "contact_form":
        fields = facts["fields"]
        f = fields[rng.randrange(len(fields))]
        qs.append(_qa(f"Does the form titled '{facts['form_title']}' contain a field labeled '{f['label']}'?", "yes", "H1"))
        if fields:
            i = rng.randrange(len(fields) - 1) if len(fields) > 1 else 0
            if i + 1 < len(fields):
                qs.append(
                    _qa(
                        f"Which form field is immediately below '{fields[i]['label']}'?",
                        fields[i + 1]["label"],
                        "H2",
                    )
                )
        qs.append(_qa("What is the submit button label on the form?", facts["button"], "H3"))
        qs.append(_qa("What is the caption of the image to the right of the form?", facts["caption"], "H4"))
        qs.append(
            _qa(
                f"The second form field is '{fields[1]['label']}'. What is the form title that contains it?",
                facts["form_title"],
                "H5",
            )
        )

    if t == "article_sidebar":
        toc = facts["toc"]
        qs.append(_qa("Which sidebar heading contains the on-page links?", "On this page", "H1"))
        i = rng.randrange(len(toc))
        qs.append(_qa(f"What is the {i + 1}th item in the 'On this page' sidebar list?", toc[i]["text"], "H3"))
        if i + 1 < len(toc):
            qs.append(_qa(f"Which sidebar item is immediately below '{toc[i]['text']}'?", toc[i + 1]["text"], "H2"))
        qs.append(_qa("What is the caption of the figure in the article?", facts["caption"], "H4"))
        qs.append(
            _qa(
                f"The first sidebar link is '{toc[0]['text']}'. What is the article title?",
                facts["title"],
                "H5",
            )
        )

    if t == "product_list":
        items = facts["items"]
        i = rng.randrange(len(items))
        qs.append(_qa(f"What is item number {items[i]['index']} in '{facts['title']}'?", items[i]["text"], "H3"))
        if i + 1 < len(items):
            qs.append(_qa(f"Which list item comes immediately after '{items[i]['text']}'?", items[i + 1]["text"], "H2"))
        qs.append(_qa(f"How many items are in the list titled '{facts['title']}'?", str(len(items)), "H3"))
        qs.append(_qa(f"Does the list '{facts['title']}' contain '{items[i]['text']}'?", "yes", "H1"))
        qs.append(
            _qa(
                f"The first list item is '{items[0]['text']}'. What adjective is shown next to item {items[-1]['index']}?",
                items[-1]["adj"],
                "H5",
            )
        )

    if t == "breadcrumb_gallery":
        bc = facts["breadcrumbs"]
        qs.append(_qa(f"What is the last breadcrumb (current page)?", bc[-1]["text"], "H3"))
        if len(bc) >= 2:
            qs.append(_qa(f"Which breadcrumb is immediately left of '{bc[-1]['text']}'?", bc[-2]["text"], "H2"))
        imgs = facts["images"]
        im = imgs[rng.randrange(len(imgs))]
        qs.append(_qa(f"What is the caption of the gallery image in row {im['row'] + 1}, column {im['col'] + 1}?", im["caption"], "H4"))
        right = [x for x in imgs if x["row"] == im["row"] and x["col"] == im["col"] + 1]
        if right:
            qs.append(_qa(f"Which image caption is immediately to the right of '{im['caption']}'?", right[0]["caption"], "H2"))
        qs.append(
            _qa(
                f"The first breadcrumb is '{bc[0]['text']}'. How many images are in the Gallery section?",
                str(len(imgs)),
                "H5",
            )
        )

    if t == "pricing_cards":
        cards = facts["cards"]
        c = cards[rng.randrange(len(cards))]
        qs.append(_qa(f"What monthly price is listed for the '{c['plan']}' plan?", c["price"], "H1"))
        qs.append(_qa(f"What is the first feature under '{c['plan']}'?", c["features"][0]["text"], "H3"))
        if c["index"] + 1 < len(cards):
            qs.append(_qa(f"Which plan card is immediately to the right of '{c['plan']}'?", cards[c["index"] + 1]["plan"], "H2"))
        qs.append(_qa(f"The button on the '{c['plan']}' card says what?", c["button"], "H4"))
        qs.append(
            _qa(
                f"The leftmost plan is '{cards[0]['plan']}'. What is the last feature of '{cards[-1]['plan']}'?",
                cards[-1]["features"][-1]["text"],
                "H5",
            )
        )

    if t == "dashboard_stats":
        st = facts["stats"]
        s = st[rng.randrange(len(st))]
        qs.append(_qa(f"What value is shown in the '{s['name']}' stat card?", s["value"], "H1"))
        if s["index"] + 1 < len(st):
            qs.append(_qa(f"Which stat card is immediately to the right of '{s['name']}'?", st[s["index"] + 1]["name"], "H2"))

    if t == "tabs_panel":
        tabs = facts["tabs"]
        qs.append(_qa("Which tab is currently selected (active)?", facts["active_tab"], "H3"))
        qs.append(_qa(f"Does the '{facts['active_tab']}' panel contain the product '{facts['product']}'?", "yes", "H1"))
        i = 0
        if len(tabs) > 1:
            qs.append(_qa(f"Which tab label is immediately to the right of '{tabs[0]['text']}'?", tabs[1]["text"], "H2"))
        qs.append(_qa("What is the caption of the image inside the active panel?", facts["caption"], "H4"))
        qs.append(
            _qa(
                f"The first tab is '{tabs[0]['text']}'. What product heading appears in the active panel?",
                facts["product"],
                "H5",
            )
        )

    if t == "checkout_wizard":
        steps = facts["steps"]
        qs.append(_qa("Which checkout step is currently active?", facts["active_step"], "H3"))
        qs.append(_qa(f"Does the '{facts['active_step']}' panel contain the item '{facts['product']}'?", "yes", "H1"))
        if facts["active_index"] < len(steps):
            nxt = steps[facts["active_index"]]["text"]  # next because active_index is 1-based
            # steps[active_index] is the next step if active_index < len
            pass
        if facts["active_index"] < len(steps):
            qs.append(
                _qa(
                    f"Which wizard step is immediately after '{facts['active_step']}'?",
                    steps[facts["active_index"]]["text"],
                    "H2",
                )
            )
        qs.append(_qa("What city is the order shipping to?", facts["city"], "H3"))
        qs.append(_qa("The image in the checkout panel is captioned with which product name?", facts["caption"], "H4"))
        qs.append(
            _qa(
                f"Step {facts['active_index']} is '{facts['active_step']}'. What is the item price shown in that step?",
                facts["price"],
                "H5",
            )
        )

    if t == "confusable_catalog":
        cards = facts["cards"]
        c = cards[rng.randrange(len(cards))]
        qs.append(_qa(f"What price is shown on the card titled '{c['title']}'?", c["price"], "H1", {"probe": "bind"}))
        qs.append(_qa(f"What SKU is printed on the '{c['title']}' card?", c["sku"], "H1", {"probe": "bind"}))
        right = [x for x in cards if x["row"] == c["row"] and x["col"] == c["col"] + 1]
        if right:
            qs.append(
                _qa(
                    f"Which product card is immediately to the right of '{c['title']}'?",
                    right[0]["title"],
                    "H2",
                    {"probe": "bind"},
                )
            )
        below = [x for x in cards if x["row"] == c["row"] + 1 and x["col"] == c["col"]]
        if below:
            qs.append(
                _qa(
                    f"Which product card is directly below '{c['title']}'?",
                    below[0]["title"],
                    "H2",
                    {"probe": "bind"},
                )
            )
        qs.append(
            _qa(
                f"The card in row {c['row'] + 1}, column {c['col'] + 1} shows which product?",
                c["title"],
                "H3",
                {"probe": "bind"},
            )
        )
        qs.append(_qa(f"What is the caption under the image on the '{c['title']}' card?", c["caption"], "H4", {"probe": "bind"}))
        other = cards[(c["index"] + 1) % len(cards)]
        qs.append(
            _qa(
                f"The SKU on '{c['title']}' is '{c['sku']}'. What is the price of '{other['title']}'?",
                other["price"],
                "H5",
                {"probe": "bind"},
            )
        )

    if t == "dense_ledger":
        rows = facts["rows"]
        r = rows[rng.randrange(len(rows))]
        h = facts["headers"][rng.randrange(1, len(facts["headers"]))]
        qs.append(
            _qa(
                f"In the {facts['table_name']}, what is the value in row {r['row_index']}, column '{h}'?",
                r["values"][h],
                "H3",
                {"probe": "cell"},
            )
        )
        sku0 = r["values"]["SKU"]
        qs.append(_qa(f"Which ledger row number has SKU '{sku0}'?", str(r["row_index"]), "H1", {"probe": "cell"}))
        qs.append(_qa(f"For SKU '{sku0}', what is the Bin?", r["values"]["Bin"], "H5", {"probe": "cell"}))
        qs.append(_qa(f"How many data rows are in the {facts['table_name']} (excluding the header)?", str(facts["n_rows"]), "H3"))
        if r["row_index"] < facts["n_rows"]:
            nxt = rows[r["row_index"]]
            qs.append(
                _qa(
                    f"Which SKU appears in the row immediately below '{sku0}'?",
                    nxt["values"]["SKU"],
                    "H2",
                    {"probe": "cell"},
                )
            )

    if t == "rtl_toolbar":
        vis = facts["nav_visual"]
        src = facts["nav_source"]
        qs.append(
            _qa(
                "What is the leftmost navigation item on the screen?",
                facts["visual_leftmost"],
                "H3",
                {"probe": "visual"},
            )
        )
        qs.append(
            _qa(
                "What is the rightmost navigation item on the screen?",
                facts["visual_rightmost"],
                "H3",
                {"probe": "visual"},
            )
        )
        qs.append(
            _qa(
                "What is the first item in the navigation source order (item 1 in the markup list)?",
                facts["source_first"],
                "H3",
                {"probe": "dom"},
            )
        )
        qs.append(
            _qa(
                f"Which navigation label is immediately to the right of '{vis[0]['text']}' on the screen?",
                vis[1]["text"],
                "H2",
                {"probe": "visual"},
            )
        )
        qs.append(
            _qa(
                f"In source order, which label comes immediately after '{src[0]['text']}'?",
                src[1]["text"],
                "H2",
                {"probe": "dom"},
            )
        )
        qs.append(_qa("How many items are in the top navigation bar?", str(len(src)), "H3"))
        qs.append(
            _qa(
                f"The leftmost on-screen nav item is '{facts['visual_leftmost']}'. What product is in the hero?",
                facts["hero_product"],
                "H5",
                {"probe": "visual"},
            )
        )

    if t == "crossed_figures":
        qs.append(
            _qa(
                "What text sits immediately below the left image on the screen?",
                facts["text_under_left_image"],
                "H4",
                {"probe": "visual"},
            )
        )
        qs.append(
            _qa(
                "What text sits immediately below the right image on the screen?",
                facts["text_under_right_image"],
                "H4",
                {"probe": "visual"},
            )
        )
        qs.append(
            _qa(
                "According to the left figure grouping, what is the caption of the left product image?",
                facts["left_caption_dom"],
                "H4",
                {"probe": "dom"},
            )
        )
        qs.append(
            _qa(
                "According to the right figure grouping, what is the caption of the right product image?",
                facts["right_caption_dom"],
                "H4",
                {"probe": "dom"},
            )
        )
        qs.append(_qa("What product name is shown as the left image label?", facts["left_product"], "H3", {"probe": "bind"}))
        qs.append(
            _qa(
                f"The left product is '{facts['left_product']}'. What caption belongs to that figure?",
                facts["left_caption_dom"],
                "H5",
                {"probe": "dom"},
            )
        )

    if t == "near_confusable":
        cards = facts["cards"]
        c = cards[rng.randrange(len(cards))]
        qs.append(_qa(f"What price is shown on the card titled '{c['title']}'?", c["price"], "H1", {"probe": "bind"}))
        qs.append(_qa(f"What SKU is printed on the '{c['title']}' card?", c["sku"], "H1", {"probe": "bind"}))
        right = [x for x in cards if x["row"] == c["row"] and x["col"] == c["col"] + 1]
        if right:
            qs.append(_qa(f"Which product card is immediately to the right of '{c['title']}'?", right[0]["title"], "H2", {"probe": "bind"}))

    if t == "compact_ledger":
        rows = facts["rows"]
        r = rows[rng.randrange(len(rows))]
        h = facts["headers"][rng.randrange(1, len(facts["headers"]))]
        qs.append(_qa(f"In the {facts['table_name']}, what is the value in row {r['row_index']}, column '{h}'?", r["values"][h], "H3", {"probe": "cell"}))
        sku0 = r["values"]["SKU"]
        qs.append(_qa(f"Which ledger row number has SKU '{sku0}'?", str(r["row_index"]), "H1", {"probe": "cell"}))
        qs.append(_qa(f"For SKU '{sku0}', what is the Bin?", r["values"]["Bin"], "H5", {"probe": "cell"}))

    if t == "swap_nav_pair":
        vis = facts["nav_visual"]
        src = facts["nav_source"]
        qs.append(_qa("What is the leftmost navigation item on the screen?", facts["visual_leftmost"], "H3", {"probe": "visual"}))
        qs.append(_qa("What is the first item in the navigation source order (item 1 in the markup list)?", facts["source_first"], "H3", {"probe": "dom"}))
        qs.append(_qa(f"In source order, which label comes immediately after '{src[0]['text']}'?", src[1]["text"], "H2", {"probe": "dom"}))
        qs.append(_qa(f"Which navigation label is immediately to the right of '{vis[0]['text']}' on the screen?", vis[1]["text"], "H2", {"probe": "visual"}))
        qs.append(_qa(f"The two swapped labels are '{facts['swap_a']}' and '{facts['swap_b']}'. What is the first source-order item?", facts["source_first"], "H5", {"probe": "dom"}))

    if t == "offset_captions":
        qs.append(_qa("According to the left figure grouping, what is the caption of the left product image?", facts["left_caption_dom"], "H4", {"probe": "dom"}))
        qs.append(_qa("According to the right figure grouping, what is the caption of the right product image?", facts["right_caption_dom"], "H4", {"probe": "dom"}))
        qs.append(_qa("What text sits immediately below the left image on the screen?", facts["text_under_left_image"], "H4", {"probe": "visual"}))
        qs.append(_qa(f"The left product is '{facts['left_product']}'. What caption belongs to that figure?", facts["left_caption_dom"], "H5", {"probe": "dom"}))

    if t == "tall_mosaic":
        cards = facts["cards"]
        c = cards[rng.randrange(len(cards))]
        qs.append(_qa(f"What price is on the tile titled '{c['title']}'?", c["price"], "H1", {"probe": "cell"}))
        qs.append(_qa(f"The tile in row {c['row'] + 1}, column {c['col'] + 1} shows which title?", c["title"], "H3", {"probe": "bind"}))

    # Generic yes/no negatives for H1 (skip on conflict pages — it inflates EM).
    if not conflict:
        qs.append(_qa("Does the top header contain a navigation bar?", "yes", "H1"))

    # Dedup by question string
    seen = set()
    out = []
    for q in qs:
        if q["question"] in seen:
            continue
        seen.add(q["question"])
        out.append(q)
    rng.shuffle(out)
    if t in HARD_TEMPLATES or t in MEDIUM_TEMPLATES:
        return out[:16]
    if t == "tall_mosaic":
        return out[:8]
    return out[:10]


def adversarial_qa(facts: dict, kind: str) -> list[dict]:
    """Questions that remain valid after a known structural corruption."""
    # The visual structure after corruption is the new ground truth; we store both.
    return []
