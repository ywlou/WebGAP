"""Closed vocabularies used by WebForge so answers are unambiguous."""

from __future__ import annotations

COMPANIES = [
    "Northwind", "BlueHarbor", "CedarLabs", "SummitPeak", "NovaTrail",
    "IronLeaf", "SilverOak", "BrightForge", "QuietHarbor", "LumenWorks",
    "MapleCourt", "Riverstone", "AmberField", "Skyline Hub", "CopperNest",
]
PRODUCTS = [
    "Atlas Mug", "Nimbus Lamp", "Cedar Desk", "Orbit Chair", "Harbor Bag",
    "Lumen Keyboard", "Peak Bottle", "Willow Notebook", "Forge Headphones",
    "Drift Monitor", "Canyon Jacket", "Meadow Sneakers", "Quartz Watch",
    "Pine Backpack", "Solstice Speaker",
]
# Near-duplicate titles used only by hard_struct pages.
CONFUSABLE_FAMILIES = [
    [
        "Peak Bottle",
        "Peak Bottle Pro",
        "Peak Bottle Mini",
        "Peak Bottle Lite",
        "Peak Bottle Pro Max",
        "Peak Bottle Set",
    ],
    [
        "Atlas Mug",
        "Atlas Mug XL",
        "Atlas Mug Pro",
        "Atlas Mug Mini",
        "Atlas Mug Duo",
        "Atlas Mug Set",
    ],
    [
        "Orbit Chair",
        "Orbit Chair Plus",
        "Orbit Chair Mini",
        "Orbit Chair Air",
        "Orbit Chair Pro",
        "Orbit Chair Set",
    ],
    [
        "Harbor Bag",
        "Harbor Bag Mini",
        "Harbor Bag Pro",
        "Harbor Bag Tote",
        "Harbor Bag Pack",
        "Harbor Bag Set",
    ],
]
NAV_ITEMS = [
    "Home", "Products", "Pricing", "Docs", "Blog", "About", "Support",
    "Careers", "Login", "Cart", "Gallery", "News", "Contact",
]
CITIES = [
    "Seattle", "Austin", "Boston", "Denver", "Portland", "Chicago",
    "Miami", "Oslo", "Lisbon", "Kyoto", "Dublin", "Toronto",
]
FIRST_NAMES = [
    "Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Luna", "Owen",
    "Ivy", "Caleb", "Nora", "Miles", "Chloe", "Jonas", "Elena",
]
LAST_NAMES = [
    "Reed", "Park", "Nguyen", "Patel", "Garcia", "Keller", "Shaw",
    "Brooks", "Ibrahim", "Costa", "Andersen", "Vogel", "Sato",
]
COLORS = [
    "navy", "teal", "coral", "olive", "maroon", "indigo", "amber", "slate",
]
COLOR_HEX = {
    "navy": "#1f3a5f",
    "teal": "#0f766e",
    "coral": "#e06c75",
    "olive": "#6b8f3c",
    "maroon": "#7f1d1d",
    "indigo": "#3730a3",
    "amber": "#d97706",
    "slate": "#334155",
}
TABLE_HEADERS = [
    "Item", "SKU", "Price", "Stock", "Region", "Rating", "Owner", "Status",
]
STATUS = ["In stock", "Low", "Backorder", "Shipped", "Active", "Paused"]
PLAN_NAMES = ["Starter", "Growth", "Business", "Enterprise"]
FORM_LABELS = ["Full name", "Email", "Company", "Role", "Message", "Phone", "City"]
CAPTIONS = [
    "Harbor sunrise", "Cedar workshop", "Peak trail map", "Studio desk",
    "Autumn meadow", "Night market", "River ferry", "Library stacks",
]
SENTENCES = [
    "We ship worldwide within three business days.",
    "Every plan includes email support and a public status page.",
    "The catalog is updated every Monday at 09:00 UTC.",
    "Members can filter by region, stock level, and rating.",
    "Use the sidebar to jump between documentation chapters.",
    "Prices exclude tax and are shown in USD.",
    "The dashboard summarizes orders, refunds, and inventory alerts.",
    "Click a card to open the product specification sheet.",
]
FOOTER_LINKS = ["Privacy", "Terms", "Status", "Press", "Cookies"]
BREADCRUMBS = ["Home", "Catalog", "Outdoor", "Lighting", "Details"]
TAB_NAMES = ["Overview", "Specs", "Reviews", "Shipping", "FAQ"]
WIZARD_STEPS = ["Cart", "Address", "Payment", "Review", "Done"]
UNITS = ["kg", "cm", "pcs", "gb", "mhz"]
ADJECTIVES = ["compact", "durable", "silent", "modular", "portable", "matte"]


def pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def picks(rng, seq, k, unique=True):
    seq = list(seq)
    if unique:
        rng.shuffle(seq)
        return seq[:k]
    return [pick(rng, seq) for _ in range(k)]


def money(rng, lo=8, hi=420):
    return f"${rng.randint(lo, hi)}.{rng.choice(['00', '50', '99'])}"


def sku(rng):
    return f"{pick(rng, 'ABCDEFGH')}{rng.randint(100, 999)}-{rng.randint(10, 99)}"


def email_of(name, company, rng):
    handle = name.lower().replace(" ", ".")
    domain = company.lower().replace(" ", "") + rng.choice([".com", ".io", ".co"])
    return f"{handle}@{domain}"
