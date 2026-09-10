# 数据集

## WebForge（训练 + 诊断）

程序化页面，不是爬取 HTML。公开网页转储缺少无噪声的拓扑标注，许可也乱。

| 分割 | 模板 | 默认规模 | 用途 |
| --- | --- | --- | --- |
| `train` | 9 族（`navbar_hero` … `dashboard_stats`） | 10 000 页 | SFT / SSL |
| `heldout` | 3 个未见族（`tabs_panel`、`comparison_matrix`、`checkout_wizard`） | 2 000 页 | WebHallu-Struct（易） |
| `leaked_templates` | 训练族、新种子 | 400 页 | 记忆差距 |
| `hard_struct` | 4 个难冲突族 | **800 页 / ~4800 题** | WebClash 难档（评测） |
| `medium_struct` | 4 个中等冲突族 | 400 页 / ~1450 题 | WebClash 中档（排序主表候选） |
| `hard_train` | 难档模板，种子 77777 | 240 页 | 冲突续训；与评测不相交 |
| `long_vis` | `tall_mosaic` | 24 页 | ERPR 长视觉序列诊断 |

图像 1280×800 JPEG q=85。约 50–80 KB/页，合计数 GB。

**H1–H5** 问题由驱动渲染器的同一批事实生成。答案是短字符串。

不要在 `heldout` 或 `hard_struct` 上调参。泄漏分割只用来报告「留出模板 vs 见过模板」的差距。

### hard_struct 族（默认不进 SFT）

| 模板 | 打破什么 |
| --- | --- |
| `confusable_catalog` | 近重复标题（`Peak Bottle` vs `Peak Bottle Pro`）+ 很近的价格带 |
| `dense_ledger` | 紧凑 8×5 数字表；黄金答案是某个单元格，不是页标题 |
| `rtl_toolbar` | 视觉从左到右 ≠ DOM `sibling_index`（row-reverse 导航） |
| `crossed_figures` | 每个 figure 的图注**画在另一张图下面**；CMA 仍跟 figure 父节点 |

QA 可带 `probe` ∈ `{visual, dom, bind, cell}`，便于按机制拆 EM。

### medium_struct 族

| 模板 | 相对难档减轻了什么 |
| --- | --- |
| `swap_nav_pair` | 只对调一对相邻导航，不是整条 RTL |
| `offset_captions` | 图注向邻居偏移，但仍更靠近自己的图 |
| `near_confusable` | 近重复标题，但色块+更大字号 |
| `compact_ledger` | 4×3 大字表，不是 8×5 密表 |

完整协议见 [WEBCLASH.md](WEBCLASH.md)。

## 公开评测

第一轮只用了截图子集，且短答 EM 会饱和或指标错配。完整调研、体积、许可与 **P1 下载清单**见 [BENCHMARK_SURVEY.md](BENCHMARK_SURVEY.md)。计划把公开主表换成官方协议（VWB 七类、WebSRC POS、Mind2Web 元素准确率、ScreenSpot-v2 Web），而不是再扩 WebForge 短答。

| 集合 | 来源 | 本仓库现状 | 图 |
| --- | --- | --- | --- |
| VisualWebBench | `visualwebbench/VisualWebBench` | **P1 全量 1536**（`full.jsonl`）；旧 526 短答 `eval.jsonl` 仅作附录 | 无 HTML → 4×4 网格 |
| WebSRC 截图 | `rootsautomation/websrc` **dev** | 已存 800 / 评 300，EM ~92%（附录饱和） | 空间网格 |
| WebSRC 官方 | `X-LANCE/WebSRC_v1.0` | **已下**：DEV 站 849 页 / 1697 QA（2/页 seed 42） | HTML + 截图 |
| Multimodal-Mind2Web | `osunlp/Multimodal-Mind2Web` | **已下三个 test**（6418 步，未下 train） | 截图 + cleaned HTML |
| ScreenSpot-v2 | `OS-Copilot/ScreenSpot-v2` | **已下** 1272（web 437） | 元素框，无 DOM |
| GUIAct web-single | `yiye2023/GUIAct` | **已下 test** 1410（1089 有点击框） | 截图 + 框 |

没有 HTML 时，网格单元之间仍有空间 TRB；DOM 类型缺失。若 `eval.jsonl` 带 `html_path`（`scripts/download_benchmarks.py --attach-html`），则用 `html_to_nodes` 建真实 DOM。

**不下载** WebQA 原图（51×1GB 分片）、Mind2Web 300GB raw、WebLINX-full 318GB、WebArena 原版 Docker、完整 Common Crawl。WebQA (CVPR 2022) 是维基多跳问答，不是网页结构。

## 训练里没有什么

留出模板、`hard_struct`、公开评测图，以及（默认）泄漏分割。第二阶段 SFT 只从 `train` 抽 3 000 条 QA（`data.sft_samples`）。

## 磁盘口径（本机）

权重约 18 GB，WebForge + 基准数 GB，检查点（LoRA + 插件）远小于 1 GB。完整跑完后 `/data` 仍约有 240 GB 空闲。
