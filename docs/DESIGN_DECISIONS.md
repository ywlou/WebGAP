# 方案修订说明（面向 WWW）

本文记录实现相对 `研究方案.md` 的偏离及原因。标准是：WWW Graph-for-the-Web 审稿人会不会接受这条声称？

原始问题设定（三维 Web 异构图、结构性幻觉 H1–H5、冻结 MLLM 插件）保留。下列改动是为了**可运行、可审稿**，不是缩小贡献。

## 1. WebQA (CVPR 2022) 是错误基准

Chang 等，*WebQA: Multihop and Multimodal QA*（CVPR 2022）是开放域**维基百科**图文问答，不是 HTML/DOM/布局理解。把它当「网页结构」主表是事实错误，容易被拒。

**替换。** 核心证据：

- **WebHallu-Struct**（自建）：模板留出的 H1–H5，带精确盒子。
- **WebSRC**（EMNLP 2021）：真实网页结构阅读（抽样，不是全量 40 万）。
- **VisualWebBench**（2024）：约 1.5k 真实站点任务；评测可下载的截图问答子集。

Mind2Web / GQA / TextVQA / POPE 保持可选，不是主声称所必需。

## 2. 稠密 token–token TRB 按原文不可实现

每头 4096×4096 偏置、32 头、18 个插入层、bf16，量级是**数 GB**，且与 8B 视觉语言模型在 80GB 上所需的 FlashAttention/SDPA 内核不兼容。

方案里的关系类型（父、兄弟序、在左、caption-of）定义在 **DOM/布局节点**上，不在子词或 32×32 patch token 上。

**改动（更强，不是更弱）。** TRB 是小表 `b[rel_type, head]`，加在**锚点–锚点** logits 上（`k≈区域数≤48`）。STAR 把视觉 token 绑到这些锚点。复杂度 `O(nk + k²)` 且 `k ≪ n`。这更接近 Graphormer（图节点上的偏置），而且真正能跑。

GTLM（2026）把图注意力偏置注入 LLM，面向文本属性图。我们不同：**多模态网页图**、几何接地的 token 赋值、冻结 MLLM 插件而不是从零训图语言模型。

## 3. STAR：让「结构定义锚点」落地

Perceiver/Q-Former 的潜变量与内容无关。WikiWeb2M Prefix Global 用的是*内容* token。原稿要结构定义锚点，但没写 token↔节点映射。

**STAR（Structure–Token Assignment Routing）：** 每个视觉 token 从 `image_grid_thw` 得到盒子（Qwen3-VL merge 2 × patch 16 ⇒ 约 32px 格子）。赋值是相对渲染区域盒（或无 HTML 真实截图上的 4×4 空间网格）的 IoU。问题 token 读全部锚点（全局读出）。

这才是「锚点是 DOM 子树 / 空间区域」的实际机制。

## 4. GACA 只在 prefill（KV cache）

在 `S=1` 的 decode 步上跑 GACA 会用单个新 token 重建锚点并毁掉它们。我们在 **prefill** 隐状态上跑 GACA；decode 使用已经混合过的前缀 KV。这符合「结构是页面的属性，不是每个生成词的属性」。

残差**零初始化**（`out_proj` 与 `tanh(gate)`）保证第 0 步预训练模型完好（ControlNet 式）。实现上只把 `out_proj` 置零，避免门与投影同时为零导致无梯度。

## 5. 放弃 vLLM

自定义解码层插件不是 vLLM 即插即用。评测用 `model.generate`（greedy）。吞吐更低；数字是 A800 上诚实的墙钟。

## 6. InternVL3 零样本移植不是免费午餐

InternVL3-8B 用 Qwen2.5-7B（`hidden=3584`）；Qwen3-VL-8B 为 `hidden=4096`，视觉栈与 DeepStack 层都不同。线性适配器不等于「同一套插件零样本」。跨家族当作 **stretch**，不是主声称，除非训了维数匹配的适配器。同底座 LoRA vs 插件才是公平对比。

## 7. 训练配方（算力）

代码里保留两阶段 SSL+SFT（`--stage ssl|sft`）。默认驱动跑 **3k WebForge QA 的 SFT**（2 epoch），因为这些问题本身就是拓扑监督；SSL 重构可选。默认开 ERPR；关 ERPR 的那次是唯一需要重训的消融。

种子：驱动里一个种子；报告里对题目做 bootstrap。若还有 GPU 时间再加种子。

## 8. 我们声称什么（以及不声称什么）

我们声称：(i) STAR 赋值是**因果的**——推理期置换它会把 held-out EM 从天花板打到 26.6%，权重不变；(ii) 插件便宜（A800 上生成延迟 +6.8%）；(iii) 在 **hard_struct**（大标题 OCR 不够用）上可以在不触顶的情况下测量几何接地的图注入。

我们**不**声称 Mind2Web 智能体 SOTA，也不声称插件无需适配器即可跨架构，也不声称易 WebForge 留出模板够当 WWW 主表（短答 SFT 后会饱和）。VisualWebBench 是迁移压力测试，不是「结构 SOTA」。

## 9. 必须诚实引用的相关工作

Graphormer、LayoutLM、WikiWeb2M Prefix Global、Perceiver/Q-Former、GraphToken、FastV/VAR、Dong 等 2021 秩坍缩、GTLM 2026、「When Graph Tokens Sink」2026。最后这篇有用：额外图 token 可能变成 sink；ERPR 以及**不**往序列里插额外 token，是直接回应。

## 10. 易分割饱和（本机实验教训）

Qwen3-VL-8B 在大字号合成页上已经能解大部分 H1–H5（冻结 EM 90.6%，短答提取器之后）。LoRA 随后到 100%。该分割上推理期 GACA/TRB 消融没有信息量。必须搭配 **B5 随机 STAR** 与 **hard_struct**。

## 11. 这张 GPU 上真正移动指针的是什么

易留出模板在短答 SFT 后饱和（LoRA = WebGAP = EM 100%）。插件仍有**因果**足迹：置换 STAR 列后 EM 掉到 26.6%（held-out）和 10.5%（hard_struct）。

在 **视觉≠DOM** 页上、*不做*冲突 SFT 时，**GraphToken（prompt 序列化）在源序探针上优于注意力插件**（33% vs ~0%）。在不相交的 `hard_train` 上再跑 100 个优化步后，LoRA 与 WebGAP 都到约 99%，且 `dom`=100%。因此：

- STAR 赋值被用到了（B5）。
- 源序/视觉冲突主要是**监督**问题，而不是再堆一个模块就能零样本解决的问题；除非把图写进 prompt（B3）。
- 不要声称 GACA 在易页 SFT 之后就能零样本解开交叉图注。

这比一张饱和的排序表更像能投 WWW 的故事。

## 12. 失败变成方法：视觉-文档双序锚点

区域级 STAR 把同一 `<nav>` 下的全部 `nav_item` 收成 **一个** 视觉锚点，TRB 兄弟边变成自环被丢弃。冲突页上 GACA 喂的是屏幕几何，所以 −GACA 反而升、`dom` 探针为 0、GraphToken（按源序列出节点）能到 33%。

**改法。** 三条消融共用同一套权重接口：

- `visual`：旧区域锚点（几何）
- `dom`：每个有信息的 DOM 节点一个锚点，赋值仍是对该节点 BBox 的 IoU，但 TRB/顺序嵌入走 `sibling_index`
- `dual`：两路并行 + token 级门控（初始化偏视觉，避免冲掉已有 SFT）

受控诊断仍用 WebClash 难档的 `dom` 探针（见 [WEBCLASH.md](WEBCLASH.md)）；**论文公开主表**改为别人的官方协议，见 [BENCHMARK_SURVEY.md](BENCHMARK_SURVEY.md)。易 held-out 不再排序方法。ERPR 降为辅助组件：短页熵已高于 τ，不能写「正则压制了幻觉」。

论文故事从「三模块全面 SOTA」收成：**视觉几何与 DOM 结构冲突时，冻结 MLLM 该信哪一边、图该从哪条路径注入**——几何锚点会在序冲突上提供错误结构（B5），GraphToken 的显式源序有时更强，因此需要双序与冲突门控。

## 13. 评测空间（相对第一轮）

第一轮把 WebForge 短答 EM 和抽样 WebSRC EM 当主证据，天花板与指标错配已经写进 [EXPERIMENTS.md](EXPERIMENTS.md)。下一阶段不另起项目：公开集做真实性与 DOM grounding，WebForge 只做成对干预。计划见 [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)。不删除 GraphToken 领先的结果。
