# WebClash：视觉序 vs 文档序冲突基准

正式评测贡献的名字：**WebClash**（Visual-order vs Document-order Conflict）。
代码里的分割名仍是英文：`hard_struct`（难）/ `medium_struct`（中）/ 易模板 `heldout`。

## 要测什么

冻结 MLLM 把截图切成视觉 token 后，默认按**屏幕几何**组织信息。网页还有一条正交的序：**DOM / 标记源序**（`sibling_index`、figure 父子）。当两条序一致时，OCR 和大标题就能做对（易模板 held-out 触顶）。当两条序故意相反时，才能分开：

- 只看像素的模型
- 把图写进 prompt 的模型（GraphToken）
- 在注意力里注入图的模型（WebGAP，视觉锚点 / DOM 锚点 / 双序门控）

## 三档难度

| 档 | 分割 | 冲突强度 | 用途 |
| --- | --- | --- | --- |
| 易 | `heldout` | 视觉≈DOM，大字 OCR | 格式对齐；**不能排序方法** |
| 中 | `medium_struct` | 一对导航对调、图注偏移、近重复但有色块、4×3 表 | **主排序表候选** |
| 难 | `hard_struct` | 整条 RTL、图注完全交叉、易混 SKU、密表 | 机制诊断；`dom` 探针 |

中档模板：`swap_nav_pair`、`offset_captions`、`near_confusable`、`compact_ledger`。
难档模板：`rtl_toolbar`、`crossed_figures`、`confusable_catalog`、`dense_ledger`。

生成：

```bash
python scripts/generate_webforge.py --hard-only --hard-pages 800
python scripts/generate_webforge.py --medium-only --medium-pages 400
```

评测种子与 `hard_train`（续训）不相交（`hard_train` seed 77777）。不要在 `hard_struct` / `medium_struct` 上调参。

## 探针

每题可带 `probe`：

| probe | 问的是 |
| --- | --- |
| `visual` | 屏上几何（最左、图正下方的字） |
| `dom` | 源序 / figure 归属（与像素相反时才难） |
| `bind` | 近重复标题绑对卡片 |
| `cell` | 表格单元格 |

报告总体 EM **和** `dom` 探针。只报总体会把「看屏幕看对了」和「懂文档序」混在一起。

## 在整篇论文里的位置

WebClash **不是**公开 benchmark，也不再承担整篇论文的主 empirical 表。它是唯一能做「只改一条结构轴」的成对干预仪器。公开主表见 [BENCHMARK_SURVEY.md](BENCHMARK_SURVEY.md) 与 [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)。medium_struct 已接近饱和（89–98%），排序请看 hard_struct 的 `dom` 探针和公开 grounding/DOM 选择。

## 建议的受控诊断表

在 **hard_struct 的 `dom` 探针** 上比 B0 / B1 LoRA / B3 GraphToken / M_visual / M_dom / M_dual。易 held-out 与 medium_struct 只进附录。

学习曲线：在 `hard_train` 上存 `checkpoint-{25,50,100,200,400}`，再评 `hard_struct`，避免 100 步又顶到 99%。

## 基线含义

- LoRA 赢、插件不赢：冲突是**监督**问题。
- GraphToken 赢、视觉锚点输：注意力里的几何聚类在喂错误序。
- DOM / 双序锚点追上 GraphToken：结构应走**文档通道**，不能只靠 IoU。
- 双序在 B5（只打乱视觉 STAR）时掉点小于纯视觉：门控会倒向 DOM。
