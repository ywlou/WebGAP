#!/usr/bin/env python3
"""Paper figures from raw CSV/JSON. Edit style() — do not edit PNGs by hand.

Mirrors: figures/scripts/plot_results.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from webgap.eval.metrics import bootstrap_em_ci
from webgap.utils.io import ensure_dir, load_json, load_jsonl

ROOT = Path("/data/WebGAP")
DATA = ROOT / "figures" / "data"
PDF = ROOT / "figures" / "pdf"
PNG = ROOT / "figures" / "png"
SKIP_RUNS = {"smoke_webgap", "zeroshot_smoke"}


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, name: str):
    ensure_dir(PDF)
    ensure_dir(PNG)
    fig.savefig(PDF / f"{name}.pdf")
    fig.savefig(PNG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def _is_heldout_file(name: str) -> bool:
    n = name.lower()
    if "hard" in n or "leaked" in n or "pub_" in n or "smoke" in n:
        return False
    return "heldout" in n


def collect_metrics():
    rows = []
    eval_root = ROOT / "outputs" / "runs"
    if not eval_root.exists():
        return pd.DataFrame()
    for run in eval_root.iterdir():
        if run.name in SKIP_RUNS:
            continue
        ev = run / "eval"
        if not ev.exists():
            continue
        for p in ev.glob("*_metrics.json"):
            blob = load_json(p)
            m = blob.get("metrics") or blob
            rows.append(
                {
                    "run": run.name,
                    "file": p.name,
                    "tag": blob.get("tag", p.stem),
                    "split": blob.get("split", ""),
                    "n": m.get("n"),
                    "em": m.get("em"),
                    "f1": m.get("f1"),
                    "sthr": m.get("sthr"),
                    "entity_hit": m.get("entity_hit"),
                    "by_type": json.dumps(m.get("by_type", {})),
                    "by_hop": json.dumps(m.get("by_hop", {})),
                    "by_probe": json.dumps(m.get("by_probe", {})),
                    "by_template": json.dumps(m.get("by_template", {})),
                }
            )
    df = pd.DataFrame(rows)
    ensure_dir(DATA)
    if len(df):
        df.to_csv(DATA / "all_metrics.csv", index=False)
    return df


def _pick(df: pd.DataFrame, run: str, pred) -> pd.Series | None:
    hit = df[df["run"] == run]
    hit = hit[hit["file"].map(pred)]
    if len(hit) == 0:
        return None
    return hit.iloc[0]


def fig_main_bars(df: pd.DataFrame):
    want = ["zeroshot", "lora_sft", "graphtoken_sft", "webgap_sft"]
    labels = ["B0 Frozen", "B1 LoRA", "B3 GraphToken", "M WebGAP"]
    sub = [_pick(df, w, _is_heldout_file) for w in want]
    if not any(x is not None for x in sub):
        return
    x = np.arange(len(want))
    width = 0.36
    em = [0 if r is None else r["em"] for r in sub]
    f1 = [0 if r is None else r["f1"] for r in sub]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x - width / 2, em, width, label="EM", color="#2563eb")
    ax.bar(x + width / 2, f1, width, label="F1", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("WebHallu-Struct held-out templates")
    fig.tight_layout()
    pd.DataFrame({"method": labels, "em": em, "f1": f1}).to_csv(DATA / "fig_main_bars.csv", index=False)
    savefig(fig, "fig_main_bars")


def fig_sthr(df: pd.DataFrame):
    want = ["zeroshot", "lora_sft", "graphtoken_sft", "webgap_sft"]
    labels = ["B0", "B1", "B3", "M"]
    vals = []
    for w in want:
        r = _pick(df, w, _is_heldout_file)
        vals.append(float(r["sthr"]) if r is not None else np.nan)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(labels, vals, color=["#94a3b8", "#64748b", "#0ea5e9", "#16a34a"])
    ax.set_ylabel("StHR (lower is better)")
    finite = [v for v in vals if np.isfinite(v)]
    ax.set_ylim(0, max(0.2, (max(finite) * 1.25 if finite else 1)))
    pd.DataFrame({"method": labels, "sthr": vals}).to_csv(DATA / "fig_sthr.csv", index=False)
    savefig(fig, "fig_sthr")


def fig_h_types(df: pd.DataFrame):
    hit = _pick(df, "webgap_sft", _is_heldout_file)
    base = _pick(df, "zeroshot", _is_heldout_file)
    lora = _pick(df, "lora_sft", _is_heldout_file)
    if hit is None or base is None:
        return
    t1 = json.loads(hit["by_type"])
    t0 = json.loads(base["by_type"])
    tL = json.loads(lora["by_type"]) if lora is not None else {}
    keys = ["H1", "H2", "H3", "H4", "H5"]
    y1 = [t1.get(k, {}).get("em", 0) for k in keys]
    y0 = [t0.get(k, {}).get("em", 0) for k in keys]
    yL = [tL.get(k, {}).get("em", 0) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.bar(x - 0.25, y0, 0.24, label="B0 Frozen", color="#94a3b8")
    ax.bar(x, yL, 0.24, label="B1 LoRA", color="#64748b")
    ax.bar(x + 0.25, y1, 0.24, label="M WebGAP", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Exact Match")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    pd.DataFrame({"type": keys, "zeroshot_em": y0, "lora_em": yL, "webgap_em": y1}).to_csv(
        DATA / "fig_htype.csv", index=False
    )
    savefig(fig, "fig_htype")


def fig_ablation(df: pd.DataFrame):
    mapping = [
        ("webgap_sft", "Full"),
        ("webgap_no_gaca", "−GACA"),
        ("webgap_no_trb", "−TRB"),
        ("webgap_no_erpr", "−ERPR"),
        ("webgap_rand_anchor", "Rand. anchors"),
    ]
    ems, names = [], []
    for run, lab in mapping:
        r = _pick(df, run, _is_heldout_file)
        if r is None:
            r = df[df["run"] == run]
            r = r.iloc[0] if len(r) else None
        if r is not None:
            names.append(lab)
            ems.append(float(r["em"]))
    if not names:
        return
    colors = ["#16a34a" if n == "Full" else ("#dc2626" if "Rand" in n else "#2563eb") for n in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.bar(names, ems, color=colors)
    ax.set_ylabel("Held-out EM")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    pd.DataFrame({"variant": names, "em": ems}).to_csv(DATA / "fig_ablation.csv", index=False)
    savefig(fig, "fig_ablation")


def fig_efficiency():
    p = DATA / "efficiency.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = ["#94a3b8", "#16a34a", "#64748b"][: len(df)]
    ax.bar(df["tag"], df["latency_s_mean"], color=colors)
    ax.set_ylabel("Mean generate latency (s)")
    plt.xticks(rotation=15)
    savefig(fig, "fig_efficiency")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(df["tag"], df["peak_mem_gb"], color="#0f766e")
    ax.set_ylabel("Peak allocated memory (GB)")
    plt.xticks(rotation=15)
    savefig(fig, "fig_efficiency_mem")


def fig_leakage(df: pd.DataFrame):
    recs = []
    for run, lab in [("webgap_sft", "WebGAP"), ("lora_sft", "LoRA"), ("zeroshot", "Frozen")]:
        h = _pick(df, run, _is_heldout_file)
        l = _pick(df, run, lambda n: "leaked" in n.lower())
        if h is not None and l is not None:
            recs.append({"method": lab, "heldout": h["em"], "leaked": l["em"]})
    if not recs:
        return
    d = pd.DataFrame(recs)
    d.to_csv(DATA / "fig_leakage.csv", index=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(x - 0.18, d["heldout"], 0.36, label="Held-out templates", color="#2563eb")
    ax.bar(x + 0.18, d["leaked"], 0.36, label="Seen templates", color="#cbd5e1")
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"])
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    savefig(fig, "fig_leakage")


def fig_public(df: pd.DataFrame):
    recs = []
    for run, lab in [("zeroshot", "B0 Frozen"), ("lora_sft", "B1 LoRA"), ("webgap_sft", "M WebGAP")]:
        v = _pick(df, run, lambda n: "visualwebbench" in n.lower())
        w = _pick(df, run, lambda n: "websrc" in n.lower())
        recs.append(
            {
                "method": lab,
                "visualwebbench_em": None if v is None else v["em"],
                "websrc_em": None if w is None else w["em"],
            }
        )
    d = pd.DataFrame(recs)
    if d["visualwebbench_em"].isna().all() and d["websrc_em"].isna().all():
        return
    d.to_csv(DATA / "fig_public.csv", index=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.bar(x - 0.18, d["visualwebbench_em"].fillna(0), 0.36, label="VisualWebBench", color="#7c3aed")
    ax.bar(x + 0.18, d["websrc_em"].fillna(0), 0.36, label="WebSRC (sample)", color="#0ea5e9")
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"], rotation=10)
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Public screenshot transfer (spatial-grid graph)")
    savefig(fig, "fig_public")


def fig_hard(df: pd.DataFrame):
    mapping = [
        ("zeroshot", "B0 Frozen"),
        ("lora_sft", "B1 LoRA"),
        ("graphtoken_sft", "B3 GraphToken"),
        ("webgap_sft", "M WebGAP"),
        ("webgap_hardmix", "M + hard SFT"),
        ("lora_hardmix", "B1 + hard SFT"),
        ("graphtoken_hardmix", "B3 + hard SFT"),
        ("webgap_no_gaca", "−GACA"),
        ("webgap_rand_anchor", "Rand. anchors"),
    ]
    recs = []
    for run, lab in mapping:
        r = _pick(df, run, lambda n: "hard" in n.lower())
        if r is None:
            continue
        recs.append({"method": lab, "run": run, "em": r["em"], "f1": r["f1"], "n": r["n"]})
    if not recs:
        return
    d = pd.DataFrame(recs)
    d.to_csv(DATA / "fig_hard.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    palette = ["#94a3b8", "#64748b", "#0ea5e9", "#16a34a", "#a78bfa", "#38bdf8", "#f59e0b", "#2563eb", "#dc2626"]
    ax.bar(d["method"], d["em"], color=palette[: len(d)])
    ax.set_ylabel("Hard-struct EM")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=18)
    ax.set_title("Diagnostic pages (confusable / dense / RTL / crossed captions)")
    savefig(fig, "fig_hard")


def fig_hard_probe(df: pd.DataFrame):
    recs = []
    for run, lab in [
        ("zeroshot", "B0"),
        ("lora_sft", "B1"),
        ("graphtoken_sft", "B3"),
        ("webgap_sft", "M"),
        ("webgap_rand_anchor", "Rand"),
    ]:
        r = _pick(df, run, lambda n: "hard" in n.lower())
        if r is None:
            continue
        probes = json.loads(r["by_probe"] or "{}")
        row = {"method": lab}
        for k in ("visual", "dom", "bind", "cell"):
            row[k] = probes.get(k, {}).get("em", np.nan)
        recs.append(row)
    if not recs:
        return
    d = pd.DataFrame(recs)
    d.to_csv(DATA / "fig_hard_probe.csv", index=False)
    keys = [c for c in ("visual", "dom", "bind", "cell") if d[c].notna().any()]
    if not keys:
        return
    x = np.arange(len(d))
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    palette = {"visual": "#0ea5e9", "dom": "#16a34a", "bind": "#f59e0b", "cell": "#7c3aed"}
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, d[k].fillna(0), width, label=k, color=palette[k])
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"])
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=len(keys))
    ax.set_title("Hard-struct probes")
    savefig(fig, "fig_hard_probe")


def fig_hard_template(df: pd.DataFrame):
    recs = []
    tmpls = ["confusable_catalog", "dense_ledger", "rtl_toolbar", "crossed_figures"]
    short = ["Confusable", "Dense table", "RTL nav", "Crossed cap."]
    for run, lab in [("zeroshot", "B0"), ("lora_sft", "B1"), ("graphtoken_sft", "B3"), ("webgap_sft", "M")]:
        r = _pick(df, run, lambda n: "hard" in n.lower())
        if r is None:
            continue
        by = json.loads(r["by_template"] or "{}")
        row = {"method": lab}
        for t in tmpls:
            row[t] = by.get(t, {}).get("em", np.nan)
        recs.append(row)
    if not recs:
        return
    d = pd.DataFrame(recs)
    d.to_csv(DATA / "fig_hard_template.csv", index=False)
    x = np.arange(len(tmpls))
    width = 0.2
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = {"B0": "#94a3b8", "B1": "#64748b", "B3": "#0ea5e9", "M": "#16a34a"}
    shown = [lab for lab in ("B0", "B1", "B3", "M") if (d["method"] == lab).any()]
    for i, lab in enumerate(shown):
        hit = d[d["method"] == lab]
        if len(hit) == 0:
            continue
        vals = [float(hit.iloc[0][t]) if pd.notna(hit.iloc[0][t]) else 0 for t in tmpls]
        ax.bar(x + (i - (len(shown) - 1) / 2) * width, vals, width, label=lab, color=colors[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=10)
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    savefig(fig, "fig_hard_template")


def fig_train_loss():
    recs = []
    for run, lab in [
        ("webgap_sft", "WebGAP"),
        ("lora_sft", "LoRA"),
        ("graphtoken_sft", "GraphToken"),
        ("webgap_no_erpr", "WebGAP −ERPR"),
    ]:
        p = ROOT / "outputs" / "runs" / run / "train_log.json"
        if not p.exists():
            continue
        blob = load_json(p)
        for row in blob.get("log") or []:
            recs.append(
                {
                    "run": lab,
                    "step": row.get("step"),
                    "loss": row.get("loss"),
                    "lr": row.get("lr"),
                    "mem_gb": row.get("mem_gb"),
                    "erpr": row.get("erpr"),
                }
            )
    if not recs:
        return
    d = pd.DataFrame(recs)
    d.to_csv(DATA / "fig_train_loss.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for lab, g in d.groupby("run"):
        ax.plot(g["step"], g["loss"], label=lab, linewidth=1.4)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("SFT loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    savefig(fig, "fig_train_loss")


def write_bootstrap_table():
    """95% CI from prediction jsonl (item bootstrap)."""
    jobs = [
        ("zeroshot", "heldout", "zeroshot/eval/zeroshot_heldout_heldout_preds.jsonl"),
        ("lora", "heldout", "lora_sft/eval/lora_sft_heldout_heldout_preds.jsonl"),
        ("graphtoken", "heldout", "graphtoken_sft/eval/graphtoken_sft_heldout_heldout_preds.jsonl"),
        ("webgap", "heldout", "webgap_sft/eval/webgap_sft_heldout_heldout_preds.jsonl"),
        ("rand_anchor", "heldout", "webgap_rand_anchor/eval/webgap_rand_anchor_heldout_heldout_preds.jsonl"),
        ("zeroshot", "hard", "zeroshot/eval/zeroshot_hard_hard_struct_preds.jsonl"),
        ("lora", "hard", "lora_sft/eval/lora_sft_hard_hard_struct_preds.jsonl"),
        ("graphtoken", "hard", "graphtoken_sft/eval/graphtoken_sft_hard_hard_struct_preds.jsonl"),
        ("webgap", "hard", "webgap_sft/eval/webgap_sft_hard_hard_struct_preds.jsonl"),
        ("rand_anchor", "hard", "webgap_rand_anchor/eval/webgap_rand_anchor_hard_hard_struct_preds.jsonl"),
        ("no_gaca", "hard", "webgap_no_gaca/eval/webgap_no_gaca_hard_hard_struct_preds.jsonl"),
        ("webgap", "hardmix", "webgap_hardmix/eval/webgap_hardmix_hard_hard_struct_preds.jsonl"),
        ("lora", "hardmix", "lora_hardmix/eval/lora_hardmix_hard_hard_struct_preds.jsonl"),
        ("graphtoken", "hardmix", "graphtoken_hardmix/eval/graphtoken_hardmix_hard_hard_struct_preds.jsonl"),
    ]
    rows = []
    for method, split, rel in jobs:
        p = ROOT / "outputs" / "runs" / rel
        if not p.exists():
            continue
        data = list(load_jsonl(p))
        ci = bootstrap_em_ci(data, n_boot=1000, seed=0)
        rows.append({"method": method, "split": split, **ci})
    if not rows:
        return
    pd.DataFrame(rows).to_csv(DATA / "bootstrap_em.csv", index=False)


def mirror_script():
    dest = ROOT / "figures" / "scripts"
    ensure_dir(dest)
    src = Path(__file__).resolve()
    shutil.copy(src, dest / "plot_results.py")


def main():
    style()
    df = collect_metrics()
    if len(df):
        fig_main_bars(df)
        fig_sthr(df)
        fig_h_types(df)
        fig_ablation(df)
        fig_leakage(df)
        fig_public(df)
        fig_hard(df)
        fig_hard_probe(df)
        fig_hard_template(df)
    fig_efficiency()
    fig_train_loss()
    write_bootstrap_table()
    mirror_script()
    print("done plots", len(df), "metric files")


if __name__ == "__main__":
    main()
