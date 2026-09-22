#!/usr/bin/env python3
"""Rewrite outputs/mllm_baselines/README.md from on-disk *_metrics.json files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data/WebGAP/outputs/mllm_baselines")

ZOO = [
    ("qwen3vl8b_zeroshot", "Qwen3-VL-8B-Instruct"),
    ("internvl35_8b_zeroshot", "InternVL3.5-8B-HF"),
    ("internvl35_14b_zeroshot", "InternVL3.5-14B-HF"),
    ("llava_ov_7b_zeroshot", "LLaVA-OneVision-7B (HF conversion)"),
    ("minicpm_v45_zeroshot", "MiniCPM-V-4.5"),
]


def _load(run: str, tag: str) -> dict | None:
    p = ROOT / run / f"{tag}_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt_vwb(blob: dict) -> str:
    m = blob["metrics"]
    return f"{m.get('n', blob.get('n'))} | official_avg **{m['official_avg']:.2f}**"


def _fmt_point(blob: dict) -> str:
    m = blob["metrics"]
    extra = ""
    if m.get("n_skip_no_bbox"):
        extra = f"（跳过无框 {m['n_skip_no_bbox']}）"
    return f"{m.get('n', blob.get('n'))} | point-in-bbox **{m['accuracy']:.2f}%**{extra}"


def _fmt_websrc(blob: dict) -> str:
    m = blob["metrics"]
    em = m.get("em", m.get("exact_match", 0))
    f1 = m.get("f1", 0)
    pos = m.get("pos", m.get("POS", 0))
    if em <= 1:
        em, f1, pos = 100 * em, 100 * f1, 100 * pos
    return f"{m.get('n', blob.get('n'))} | EM {em:.1f} / F1 {f1:.1f} / **POS {pos:.1f}**"


def _fmt_m2w(blob: dict) -> str:
    m = blob["metrics"]
    f1 = m["action_f1"]
    em = m.get("element_em", 0)
    if f1 <= 1:
        f1 *= 100
    if 0 <= em <= 1:
        em *= 100
    return f"{m.get('n', blob.get('n'))} | element_em {em:.1f}% / action_f1 {f1:.1f}%"


def _rows(run: str) -> list[str]:
    specs = [
        ("visualwebbench", "VisualWebBench 官方七类", _fmt_vwb, "visualwebbench_metrics.json"),
        ("screenspot_v2", "ScreenSpot-v2 web", _fmt_point, "screenspot_v2_metrics.json"),
        ("websrc", "WebSRC official DEV subset", _fmt_websrc, "websrc_metrics.json"),
        ("guiact", "GUIAct web-single", _fmt_point, "guiact_metrics.json"),
        ("mind2web_test_website", "Mind2Web test_website", _fmt_m2w, "mind2web_test_website_metrics.json"),
        ("mind2web_test_task", "Mind2Web test_task", _fmt_m2w, "mind2web_test_task_metrics.json"),
        ("mind2web_test_domain", "Mind2Web test_domain", _fmt_m2w, "mind2web_test_domain_metrics.json"),
    ]
    out = []
    for tag, name, fmt, fname in specs:
        blob = _load(run, tag)
        if blob is None:
            out.append(f"| {name} | — | **未完成** | `{run}/{fname}` |")
            continue
        n_and_metric = fmt(blob)
        n, metric = n_and_metric.split(" | ", 1)
        out.append(f"| {name} | {n} | {metric} | `{run}/{fname}` |")
    return out


def main() -> None:
    done, pending = [], []
    sections = []
    for run, title in ZOO:
        d = ROOT / run
        if not d.exists():
            pending.append(run)
            continue
        rows = _rows(run)
        complete = all("未完成" not in r for r in rows)
        (done if complete else pending).append(run)
        sections.append(f"### {title}\n\n| 基准 | n | 主指标 | 文件 |\n| --- | --- | --- | --- |\n" + "\n".join(rows))
    head = [
        "# mllm_baselines",
        "",
        "P3 零样本结果写在这里，不覆盖 `outputs/runs/`。",
        "",
        f"已完成：{', '.join(f'`{x}`' for x in done) or '无'}。",
        f"未完成：{', '.join(f'`{x}`' for x in pending) or '无'}。",
        "",
        "权重在 `/data/models/`。续跑命令见仓库根目录 `交接文档.md` 第 4 节。",
        "",
        "Mind2Web 指标是下一步动作字符串匹配，不是 Task SR。旧 VWB 短答 EM 协议作废。",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(head + sections) + "\n", encoding="utf-8")
    print("wrote", ROOT / "README.md")
    print("done", done)
    print("pending", pending)


if __name__ == "__main__":
    main()
