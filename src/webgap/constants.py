"""Shared constants for graph relations, hallucination types, and templates."""

from __future__ import annotations

import os
from pathlib import Path

# Base-model weights live outside the repo. Override with WEBGAP_MODELS_DIR.
MODELS_ROOT = Path(os.environ.get("WEBGAP_MODELS_DIR", "/data/models"))
DEFAULT_QWEN3VL_8B = MODELS_ROOT / "Qwen" / "Qwen3-VL-8B-Instruct"

# Structural hallucination taxonomy (paper Table 1).
HALLU_TYPES = ("H1", "H2", "H3", "H4", "H5")
HALLU_NAMES = {
    "H1": "containment",
    "H2": "adjacency",
    "H3": "order",
    "H4": "cross_modal_alignment",
    "H5": "multi_hop",
}

# Discrete TRB relation IDs. Index 0 is "no typed edge" (bias forced to 0).
REL_NONE = 0
REL_PARENT = 1
REL_CHILD = 2
REL_SIB_PREV = 3
REL_SIB_NEXT = 4
REL_SAME_REGION = 5
REL_LEFT = 6
REL_RIGHT = 7
REL_ABOVE = 8
REL_BELOW = 9
REL_CONTAINS = 10
REL_CONTAINED = 11
REL_OVERLAP = 12
REL_CMA = 13
REL_ANCESTOR = 14
REL_DESCENDANT = 15
NUM_REL_TYPES = 16

REL_NAMES = [
    "none",
    "parent",
    "child",
    "sib_prev",
    "sib_next",
    "same_region",
    "left",
    "right",
    "above",
    "below",
    "contains",
    "contained",
    "overlap",
    "cma",
    "ancestor",
    "descendant",
]

# Template families. Held-out families never appear in training pages.
TRAIN_TEMPLATES = (
    "navbar_hero",
    "card_grid",
    "data_table",
    "contact_form",
    "article_sidebar",
    "product_list",
    "breadcrumb_gallery",
    "pricing_cards",
    "dashboard_stats",
)
HELDOUT_TEMPLATES = (
    "tabs_panel",
    "comparison_matrix",
    "checkout_wizard",
)
# Never used in SFT. Stress tests where OCR of large headings is not enough.
HARD_TEMPLATES = (
    "confusable_catalog",
    "dense_ledger",
    "rtl_toolbar",
    "crossed_figures",
)
# Milder visual≠DOM conflict: the missing middle of the easy/hard poles.
MEDIUM_TEMPLATES = (
    "swap_nav_pair",
    "offset_captions",
    "near_confusable",
    "compact_ledger",
)
# Long visual sequences so ERPR entropy hinge can actually fire (CH3).
LONG_TEMPLATES = ("tall_mosaic",)
ALL_TEMPLATES = TRAIN_TEMPLATES + HELDOUT_TEMPLATES + HARD_TEMPLATES + MEDIUM_TEMPLATES + LONG_TEMPLATES

# Node roles that become *individual* DOM anchors (not collapsed into a visual region).
DOM_ANCHOR_ROLES = (
    "nav_item",
    "card",
    "cell",
    "image",
    "caption",
    "list_item",
    "button",
    "tab",
    "field",
    "label",
    "heading",
    "stat",
    "breadcrumb",
    "row",
)

NODE_ROLES = (
    "page",
    "header",
    "nav",
    "nav_item",
    "hero",
    "section",
    "card",
    "table",
    "row",
    "cell",
    "form",
    "field",
    "button",
    "list",
    "list_item",
    "image",
    "caption",
    "heading",
    "text",
    "footer",
    "tab",
    "panel",
    "stat",
    "breadcrumb",
    "sidebar",
    "label",
)

IMAGE_TOKEN_ID_QWEN3VL = 151655
VISION_START_ID_QWEN3VL = 151652
VISION_END_ID_QWEN3VL = 151653
