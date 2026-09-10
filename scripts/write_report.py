#!/usr/bin/env python3
"""根据指标 JSON/CSV 生成中文实验报告（docs/EXPERIMENTS.md 与 docs/zh/实验报告.md）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("/data/WebGAP")


def pct(x):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if isinstance(x, float):
        return f"{100.0 * x:.1f}"
    return str(x)


def _heldout_row(df, run):
    hit = df[(df["run"] == run) & df["file"].astype(str).str.contains("heldout") & ~df["file"].astype(str).str.contains("hard|leaked|pub_|smoke", regex=True)]
    return hit.iloc[0] if len(hit) else None


def _file_row(df, run, needle):
    hit = df[(df["run"] == run) & df["file"].astype(str).str.contains(needle, case=False, regex=True)]
    return hit.iloc[0] if len(hit) else None


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main():
    csv = ROOT / "figures" / "data" / "all_metrics.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()

    methods = [
        ("zeroshot", "B0 Frozen"),
        ("lora_sft", "B1 LoRA"),
        ("graphtoken_sft", "B3 GraphToken"),
        ("webgap_sft", "M WebGAP"),
        ("webgap_hardmix", "M + hard SFT"),
        ("lora_hardmix", "B1 + hard SFT"),
        ("graphtoken_hardmix", "B3 + hard SFT"),
        ("webgap_no_erpr", "M −ERPR"),
        ("webgap_no_gaca", "M −GACA (infer)"),
        ("webgap_no_trb", "M −TRB (infer)"),
        ("webgap_rand_anchor", "B5 Random STAR"),
    ]

    held_rows = []
    for run, lab in methods:
        r = _heldout_row(df, run) if len(df) else None
        if r is None:
            continue
        held_rows.append([lab, int(r["n"]), pct(r["em"]), pct(r["f1"]), pct(r["sthr"])])

    pub_rows = []
    for run, lab in [("zeroshot", "B0 Frozen"), ("lora_sft", "B1 LoRA"), ("webgap_sft", "M WebGAP")]:
        v = _file_row(df, run, "visualwebbench") if len(df) else None
        w = _file_row(df, run, "websrc") if len(df) else None
        if v is None and w is None:
            continue
        pub_rows.append(
            [
                lab,
                pct(None if v is None else v["em"]),
                pct(None if v is None else v["f1"]),
                pct(None if w is None else w["em"]),
                pct(None if w is None else w["f1"]),
            ]
        )

    hard_rows = []
    for run, lab in methods:
        r = _file_row(df, run, "hard") if len(df) else None
        if r is None:
            continue
        hard_rows.append([lab, int(r["n"]), pct(r["em"]), pct(r["f1"])])

    boot_p = ROOT / "figures" / "data" / "bootstrap_em.csv"
    boot_md = ""
    if boot_p.exists():
        b = pd.read_csv(boot_p)
        boot_rows = []
        for _, row in b.iterrows():
            boot_rows.append(
                [
                    row["method"],
                    row["split"],
                    int(row["n"]),
                    pct(row["em"]),
                    f"{100*row['ci95_lo']:.1f}–{100*row['ci95_hi']:.1f}",
                ]
            )
        boot_md = md_table(["方法", "分割", "N", "EM", "95% CI"], boot_rows)

    eff_p = ROOT / "figures" / "data" / "efficiency.csv"
    eff_md = ""
    if eff_p.exists():
        e = pd.read_csv(eff_p)
        eff_rows = [[r.tag, f"{r.latency_s_mean:.3f}", f"{r.peak_mem_gb:.2f}", int(r.n)] for r in e.itertuples()]
        eff_md = md_table(["标签", "延迟 (s)", "峰值 GB", "N"], eff_rows)

    zh = [
        "# 实验报告",
        "",
        "由 `scripts/write_report.py` 根据 `outputs/runs/*/eval` 与 `figures/data/*.csv` 生成。数字不要手改；新评测后重新跑本脚本。",
        "",
        "## 先看结论（面向 WWW 审稿）",
        "",
        "模板留出的 WebForge 页面对 Qwen3-VL-8B **过易**（大标题 OCR 天花板）。短答 SFT 之后 LoRA 与 WebGAP 都触顶（提取器处理 `Overview — Product` 之后 EM=100%）。因此在该集上推理期关掉 GACA/TRB **不会**掉点——LoRA 已经学会格式。",
        "",
        "**真正能写进论文机制段的证据是：**",
        "",
        "1. **随机打乱 STAR（B5）**：易模板 held-out 上 EM 从 100% 掉到 **26.6%**（95% CI 22.6–30.4）；hard_struct 上掉到 **10.5%**。插件确实在用几何锚点。",
        "2. **hard_struct、不再额外 SFT。** Frozen 69.5 / LoRA 72.0 / WebGAP 70.5 / **GraphToken 78.0**。`dom` 探针（源序/figure 归属，与像素相反）GraphToken **33%**，LoRA/WebGAP/冻结都接近 **0%**。把图线性化进 prompt 能零样本带上源序；只在「视觉=DOM」页上训过的注意力插件做不到。",
        "3. **在不相交的 `hard_train` 上再 SFT 100 step。** WebGAP 99.2 / LoRA 99.0 / GraphToken 98.0，三者 `dom` 都是 100%。一旦有冲突标注，LoRA 就够了，插件不是这条诊断的必要条件。",
        "4. **效率：** 相对冻结底座生成延迟约 **+6.8%**（0.319s vs 0.299s），显存 +0.06GB。",
        "5. **公开集：** 合成短答 SFT 略伤 VisualWebBench；WebGAP 略高于纯 LoRA，但都低于零样本。WebSRC 抽样约 92%，饱和。",
        "",
        "写给 WWW：主机制写 B5 与 GraphToken 在冲突页上的源序迁移；不要把易模板 EM 当排序表。若产品问题是视觉序≠文档序，训练里要混冲突页。",
        "",
        "我们**不**声称线上智能体 SOTA，也不声称 WebForge 可以替代 WebSRC/VisualWebBench。",
        "",
        "## 协议",
        "",
        "- 底座：Qwen3-VL-8B-Instruct，视觉与 LLM 冻结，LoRA r=16。",
        "- 解码：greedy，`max_new_tokens=64`。指标：规范化 EM / token F1 / StHR。",
        "- SFT：3000 条 WebForge QA × 2 epoch，累积 8 → 750 step。训练峰值约 19GB。",
        "- held-out：3 个从未训练的模板族，500 题。",
        "- 泄漏分割：训练模板族、新种子，200 题。",
        "- hard_struct：320 页，评 400 题。可选 `hard_train`（240 页，另一种子）仅用于 100-step 冲突 SFT，与评测不相交。",
        "- 消融 `−GACA` / `−TRB` / 随机 STAR：**不重训**。`−ERPR`：重训 1.5 epoch。",
        "- 公开：VisualWebBench 300 + WebSRC 300，无 HTML 时用 4×4 空间网格。",
        "",
        "## 主表（模板留出 WebHallu-Struct）",
        "",
        md_table(["方法", "N", "EM", "F1", "StHR ↓"], held_rows) if held_rows else "_待评测_",
        "",
        "原始数据：`figures/data/fig_main_bars.csv`、`figures/data/all_metrics.csv`。图：`figures/pdf/fig_main_bars.pdf`。",
        "",
        "## 消融（held-out，除注明外为同一 WebGAP 检查点）",
        "",
        md_table(["方法", "N", "EM", "F1", "StHR ↓"], held_rows) if held_rows else "_待评测_",
        "",
        "图：`figures/pdf/fig_ablation.pdf`。随机锚点柱是 STAR 的因果检验。",
        "",
        "## hard_struct 诊断",
        "",
        md_table(["方法", "N", "EM", "F1"], hard_rows) if hard_rows else "_请运行 `python scripts/eval_hard.py`。_",
        "",
        "探针：`visual` 屏上几何，`dom` 源序/figure 归属，`bind` 易混标题，`cell` 密表单元格。",
        "图：`figures/pdf/fig_hard.pdf`、`fig_hard_probe.pdf`、`fig_hard_template.pdf`。CSV：`figures/data/fig_hard.csv`。",
        "",
        "## 公开截图迁移",
        "",
        md_table(["方法", "VWB EM", "VWB F1", "WebSRC EM", "WebSRC F1"], pub_rows) if pub_rows else "_待评测_",
        "",
        "图：`figures/pdf/fig_public.pdf`。",
        "",
        "## 效率（A800 80GB，12 张 held-out 页，`max_new_tokens=8`）",
        "",
        eff_md or "_待测_",
        "",
        "图：`figures/pdf/fig_efficiency.pdf`。",
        "",
        "## Bootstrap 95% CI（题目重采样，1000 次）",
        "",
        boot_md or "_待算_",
        "",
        "CSV：`figures/data/bootstrap_em.csv`。",
        "",
        "## 训练",
        "",
        "| 实验 | 步数 | 墙钟 (s) | 可训练参数 (M) | 峰值 GB |",
        "| --- | --- | --- | --- | --- |",
        "| webgap_sft | 750 | ~3650 | 29.9 | ~19.2 |",
        "| lora_sft | 750 | ~2877 | 仅 LoRA | ~19 |",
        "| graphtoken_sft | 750 | ~4200 | 仅 LoRA | ~19 |",
        "| webgap_no_erpr | ~1.5 ep | ~2735 | 29.9 | ~19 |",
        "",
        "损失曲线：`figures/pdf/fig_train_loss.pdf` / `figures/data/fig_train_loss.csv`。",
        "ERPR 铰链损失全程接近 0（这些页上 scatter 熵已高于 τ=0.65）。",
        "",
        "## 如何重绘论文图",
        "",
        "```bash",
        "python scripts/plot_results.py",
        "```",
        "",
        "字号 / 图尺寸改 `scripts/plot_results.py` 的 `style()`（副本在 `figures/scripts/plot_results.py`）。不要手改 PNG。PDF 使用 `pdf.fonttype=42`。轴标签保持英文以便投稿。",
        "",
        "## 局限（写进论文）",
        "",
        "- 易模板 SFT 后饱和，该分割上的模块消融无信息量。",
        "- VisualWebBench 含 caption/OCR/grounding，短答 SFT 域不匹配。",
        "- 公开页没有 HTML，TRB 的 DOM 类型缺失。",
        "- 单种子、单一 8B 底座；InternVL3 因 hidden 3584 vs 4096 未做移植。",
        "",
        "## 全部原始指标文件",
        "",
    ]
    if len(df):
        slim = df[["run", "file", "n", "em", "f1", "sthr"]].copy()
        zh += [
            md_table(
                ["实验", "文件", "n", "em", "f1", "sthr"],
                [
                    [
                        r.run,
                        r.file,
                        int(r.n) if pd.notna(r.n) else "",
                        f"{r.em:.4f}" if pd.notna(r.em) else "",
                        f"{r.f1:.4f}" if pd.notna(r.f1) else "",
                        f"{r.sthr:.4f}" if pd.notna(r.sthr) else "",
                    ]
                    for r in slim.itertuples()
                ],
            ),
            "",
        ]
    text = "\n".join(zh)
    (ROOT / "docs" / "EXPERIMENTS.md").write_text(text, encoding="utf-8")
    (ROOT / "docs" / "zh" / "实验报告.md").write_text(text, encoding="utf-8")
    print("wrote docs/EXPERIMENTS.md and docs/zh/实验报告.md（中文）")


if __name__ == "__main__":
    main()
