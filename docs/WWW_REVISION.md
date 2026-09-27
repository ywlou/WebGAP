# 面向 WWW 的修订：同底座上把 WebGAP 做成主表第一

数字以 `outputs/` 里的 json 和 `src/webgap/` 的实现为准。`研究方案.md`、`docs/EXPERIMENTS.md`、`docs/BENCHMARK_SURVEY.md` 里有多处已过时或说满的句子，下文逐条标出。

投稿目标是 The Web Conference。论文问题改成：**如何把网页结构有效注入一个预训练 MLLM，以提升网页理解。** 比较范围只限同一底座、同一训练数据、同一预算、同一输出协议下的结构增强插件。底座只用 **Qwen3-VL-8B-Instruct** 和 **InternVL3.5-8B**。更大的 14B、30B 不再做。

主表要追求的结果是：在这个公平比较池里，主模型 **WebGAP-V** 在主基准上最好。现在的权重还做不到，尤其是 ScreenSpot-Web。

**本轮只补 Qwen3-VL-8B-Instruct 主表里 WebGAP 还空着的 Mind2Web 元素选择。** WebSRC、VWB Element Grounding、VWB Action Grounding、ScreenSpot-Web 已写进 `outputs/main_table/qwen3vl8b_instruct.json`。不训、不测 Frozen、LoRA、GraphToken，以及 HTML prompt、Layout Prompt、Set-of-Mark、UI token selection、Action-head attention。不跑 InternVL。待确认后只跑 WebGAP-V 和 WebGAP-Dual：正候选加上随机 50 个负候选（seed 42），没有 MindAct 的排序器，因此不能和附录里的 55.1 直接比。三个分割都报。动作字符串 F1 不写入这一列。

---

## 1. 现有实验已经说清楚的事

这些结论方向稳定，后面改故事时应当保留，不要再刷掉。

**强 MLLM 在「看截图、抄可见字、点一个框」上已经很高。** Qwen3-VL-8B 零样本：VisualWebBench 官方均分 79.8，ScreenSpot-v2 Web 点在框内 91.1%，GUIAct 有框项 76.3%，WebSRC official DEV subset EM 86.7。InternVL3.5-14B 的 ScreenSpot Web 也到 89.0。这些列上再比 1 个点，没有论文信息。

**几何锚点被用到了，而且会伤真实点选。** 同一套 WebGAP 权重，推理期打乱视觉 STAR：易模板 EM 100% → 26.6%，hard_struct 70.5% → 10.5%。公开点选上，双序插件把 ScreenSpot Web 从 91.1% 拉到 33.4%，GUIAct 从 76.3% 拉到 35.1%。依赖和增益是两件事。B5 证明依赖；P4 证明这种依赖在真实 grounding 上是有害的。

**源序问题靠「把结构写成文本」，不靠注意力插件。** hard_struct 1500 题的 `dom` 探针：GraphToken 35.4%，LoRA 0.6%，视觉 WebGAP 0%，双序 9.6%。P5 冲突页的 DOM 题（交接文档的总表没列这一列，json 里有）：文本结构 58.8%，双序 45.9%，冲突门控 44.7%，DOM 锚点 35.3%，LoRA 与视觉锚点都是 32.9%，冻结 0%。P8 更干净：冲突监督只有 8 条时，`dom` 探针文本结构 40.2%，双序 16.1%，LoRA 13.8%；25 步之后文本结构已到 96.6%，100 步后三种学法都到 100%。低数据区赢的是显式文本，插件不比 LoRA 省数据。

**DOM 序的成对干预，谁都没跟上。** P5 里 gold 翻转且两臂都答对的比例：视觉重排上 WebGAP 三个变体都是 100%（n=48 对），DOM 重排上冻结、LoRA、文本结构、三种 WebGAP 全是 0%（n=48 对）。DOM 锚点模式对视觉置换完全不敏感（曲线五档都是 100%），说明它没在用视觉赋值；它同样不跟踪 DOM 序被改写。冲突页上 35% 的 DOM 题做对，不能写成「模型在用文档序」。

**冲突门控是负结果。** `webgap_dual_cg` 的 hard_struct `dom` 从双序的 9.6% 掉到 1.6%，P5 全局置换从 12.5% 掉到 0%。不要再在 `mix` 上加一项序差。

---



## 2. 文档说满、代码并不是那样



### 2.1 名叫 GraphToken 的基线不是 GraphToken

`研究方案.md` 第 6.3 节把 B3 写成「小型 GNN 在模型外编码结构、特征拼入序列」，并引用 Perozzi 等 2024 的 GraphToken。实现是 `serialize_graph_text`：最多 40 个节点的角色和截断文本，加上最多 60 条边的整数编号，拼进 prompt。没有 GNN，没有 soft token。

它在 DOM 题上领先，说明的是：**同一张截图旁边放一段显式结构文本，比在隐状态里加几何锚点更有效。** 论文里应改名为 text-structure 或 linearized DOM。继续叫 GraphToken，审稿人会按原文要求一个 GNN 编码器。

### 2.2 TRB 没有加进主干自注意力

方案里的公式是主干 logits `QK/sqrt(d) + γ b_{t(i,j)}`，作用在 token×token 上。`plugin.py` 的 `_trb_bias` 只加在锚点–锚点的小注意力上，`k ≤ 48`。`DESIGN_DECISIONS.md` 把这写成工程上更合理的改法，这句可以保留。不能再写「类型化偏置直接调制每个视觉 token 对每个视觉 token 的注意力」。CH2（局部关系无类型偏置）目前没有被这个模块回答。

### 2.3 GACA 的汇聚不是按图邻接裁剪的

方案写「汇聚与回写均受图邻接约束（非全连接）」。`_run_stream` 里 gather 的 `key_mask` 是全 1，每个锚点看整段序列。STAR 的 IoU 只用于一开始的 `einsum` 池化，随后的交叉注意力是全连接。和 Perceiver 的差别只剩：池化初值来自盒子 IoU，以及锚点之间有一张小的关系偏置。这仍然可以描述，但「图邻接约束的双向拓扑通道」这句话要删。

### 2.4 ERPR 不是有效秩，而且没起作用

实现是 scatter 注意力的熵铰链，相对 `τ=0.65`。训练日志里该项接近 0。核范数幂迭代没有实现。去掉它重训，held-out 99.8% 对 100%。贡献列表里的第四条应删除，附录用一句说明即可。

### 2.5 三通道正交没有被消融支持

GACA 和 TRB 在同一个插件子层里，TRB 不接触主干 logits，ERPR 只看插件自己的 scatter。推理期关掉 GACA，hard_struct 400 题从 70.5% 升到 72.2%。不存在「三模块可独立开关、组合超加」的表。方法节改成一条残差旁路：几何池化、锚点间小图、散回序列。双序是后来加的门控，不是第三个正交通道。

### 2.6 StHR 在这套数据上没有分辨率

`score_predictions` 把 StHR 定义成：gold 子串已经出现在预测里、但 EM 仍失败的比例。冻结 held-out EM 90.6%，StHR 仍是 0。错例要么整句对不上、实体也没出现，要么被宽松 EM（gold 作为整词子串即算对）吸走了。H1–H5 在易页上 SFT 后全是 100%。这个指标不能当贡献。

宽松 EM 还会抬高合成集分数：`exact_match` 允许 gold 出现在抽取句的头、尾或中间。写主结果前应同时报严格归一化相等的 EM。

### 2.7 WebSRC 的 POS 被答案文本反查绑死

官方 POS 衡量的是预测证据节点和标注节点在 DOM 路径上的重合，用来抓住「字抄对了、节点错了」。`score_websrc` 用预测字符串在 HTML 里找包含该字符串的最短节点，再和 gold tid 比路径 Jaccard。字对了，POS 会跟着高。当前 POS 77–81、EM 86–88，方法和冻结差 3 个点以内，不能当成结构定位成绩。要让 POS 有意义，模型必须另输出节点 id，禁止用答案字符串回查。

### 2.8 Mind2Web 主表列的不是元素准确率

`BENCHMARK_SURVEY.md` 和 `NEXT_EXPERIMENTS.md` 的 Table 1/2 仍写「Mind2Web 元素准确率」。`evaluate.py` 的 prompt 只给截图和任务句，要求自由生成 `CLICK/TYPE/SELECT` 短句，再和 `target_action_reprs`（形如 `[div] foo -> CLICK`）做 EM/F1。`element_em` 全是 0。动作 F1 大约 21–30%，文本结构比冻结高约 7 个点，但这个协议不是社区的 Element Accuracy / Operation F1 / Step SR。

下载脚本其实留下了 `pos_candidates`，负例只存了 `n_neg`，负例内容被丢掉。官方逐步评测需要正例加负例候选上的选择。在负例补回之前，Mind2Web 这一列不能进主表。交接文档后半已经写明协议不对；调研文档的表头还没改。以代码为准。

### 2.9 真实 HTML 上的盒子很多是假的

`html_to_nodes` 在元素没有 `data-bbox` 时，用纵向光标每行加 22 像素造一个盒子。STAR 的 IoU 就打在这些假盒子上。无 HTML 的截图则退回 4×4 网格。因此 ScreenSpot、GUIAct、VisualWebBench 上的插件不包含 DOM 关系；WebSRC 和 Mind2Web 上的「DOM 锚点」几何也不可靠。文档里「有 HTML 则建 DOM 图」容易被读成几何接地的真实布局。论文里应分开写：拓扑来自标签树，坐标在缺少渲染盒时是占位。

这也解释得了公开点选为什么崩：插件把已经很强的视觉隐状态残差拉向错误区域。

### 2.10 若干文档互相打架


| 位置                                     | 问题                                                                 |
| -------------------------------------- | ------------------------------------------------------------------ |
| `研究方案.md` 贡献 1–4、H1–H5、跨家族零样本、WebQA 主表 | 与后来的实现和结果相反。只可作历史，不能再当论文大纲                                         |
| `docs/EXPERIMENTS.md` 标题「主表（模板留出）」     | 易页 100% 被放在主表。`OVERVIEW.md` 已说这张表不能排序                              |
| `docs/WEBCLASH.md`                     | 中档曾被写成主排序候选；同文后半又写 89–98% 饱和                                       |
| `docs/BENCHMARK_SURVEY.md` 第 6 节       | 仍写 P1 尚未执行。数据已经在 `data/benchmarks/`                                |
| `docs/NEXT_EXPERIMENTS.md` 开头          | 写 `evaluate.py` 尚未实现。脚本已经在跑完 P3–P5                                 |
| 交接文档 P5 总表                             | 冲突页只报了 visual EM。同文件 json 里 DOM 题文本结构 58.8%、双序 45.9%，这是合成集上最有差距的一列 |


LLaVA-OneVision 的 ScreenSpot 4.4%、GUIAct 2.4% 是坐标格式没接上。交接文档已承认。这组数字不能用来证明「结构弱点跨家族」。LLaVA 在 WebSRC POS 74.9、VWB 55.8 上仍可用，grounding 列应标成格式失败。

---



## 3. 思考方式上该停的地方

**把「插件被使用」写成「插件更好」。** B5 和 P5 视觉跟上 100% 只说明残差通路接在几何上。审稿人接受这个机制句，不会因此接受方法排名。

**用同一个生成器训练、诊断、干预。** WebForge 的易页、hard_struct、P5 成对页都是同一套 PIL 模板。模板族留出挡得住「背某一张版式」，挡不住「学会这种字体、这种问法、这种短答」。公开点选上的大幅退步就是域外结果。合成页留作仪器，公开页承担排名。

**指标饱和之后继续加模块。** 双序把 `dom` 从 0 抬到 9.6，冲突门控又压回 1.6。P8 说明再给几十步冲突监督，LoRA 就和插件一起到顶。下一轮的增量应放在评测接口和真实页冲突切片，而不是第四个门控。

**用非官方协议制造可分的分数。** Mind2Web 字符串 F1、POS 字符串反查、短答 EM 打 VisualWebBench，都会得到「有高低」的表，社区对不上。改进要求里禁止为了降分去改标签；同样不能为了能跑完而换一套更松的指标，再把列名写成官方名字。

**主表塞进太多家族，方法行却只有三条。** 五个 MLLM 的零样本是能力普查，占满表格之后，同预算的结构注入只剩下 LoRA、一段自制文本、插件三变体。WWW 审稿人会拿 SeeAct（截图加 HTML）、MindAct（候选上选择）、Set-of-Mark 来比。这些都还没做。

**把「有时注意力好、有时提示词好」写成主结论。** 那是上一版测量故事。这一版要的是传统主表：同一底座上的结构增强方法里，WebGAP 最好。现有权重还输在 ScreenSpot-Web 和 VWB Action Grounding，所以先改模型，再写排名。排名只在第 6 节的统一约束下成立。缩短别人的训练、换更差的提示词、禁止别人使用同等 LoRA，都不做。

---



## 4. 建议改成的论文

题目方向：**WebGAP: Injecting Webpage Structure into Multimodal LLMs for Web Understanding。**

主张写成：在预训练 MLLM 上，用注意力级结构插件注入网页几何，并在同一底座、同一数据、同一预算下优于其他结构增强插件。主模型是 **WebGAP-V**（视觉几何进注意力）。WebGAP-DOM 和 WebGAP-Dual 留作变体；Dual 的 ScreenSpot-Web 只有 33.4，在点选修好之前不当最终模型。

比较池是结构增强插件，完整 GUI Agent 不进主表。


| 档 | 方法 | 结构从哪进 | 主表角色 |
| --- | --- | --- | --- |
| 无适应 | Frozen | 不进 | 底座 |
| 只加参数 | LoRA | 权重 | 排除「多训一下就行」 |
| 结构写成语言 | GraphToken（短线性化 DOM） | 提示词 | 核心对照 |
| 结构写成语言 | HTML prompt（HtmlRAG，WWW 2025 的清洗思路） | 提示词 | 必做。只借输入格式，不复现其 RAG |
| 结构写成语言 | Layout Prompt（文本、类型、盒子） | 提示词 | 必做 |
| 图像标注 | Set-of-Mark（Yang 等 2023；SeeAct / UGround 用过） | 截图 | 必做 |
| 视觉 token | UI token selection（ShowUI，CVPR 2025 的选择模块） | 自注意力里丢冗余视觉 token | 必做。不复现其 2B 全模型 |
| 注意力定位 | Action-head attention（GUI-Actor，NeurIPS 2025） | 一个定位 token 对视觉 patch 做注意力 | 必做。与 WebGAP 最接近的已发表注意力机制 |
| 注意力结构 | **WebGAP-V** | 多锚点残差 | 主模型 |
| 注意力变体 | WebGAP-DOM、WebGAP-Dual | 锚点残差 | 消融。Dual 接地修好再考虑升格 |


表里的 GraphToken 仍是 `serialize_graph_text`。引用 Talk like a Graph（Fatemi 等，ICLR 2024）。Perozzi 等 2024 的 GraphToken 是 GNN soft token，现有代码没有实现，这行不写成那篇论文的复现。

2025 年之后和本题最接近的两篇，主表只移植机制，不搬它们的原文分数：

- **GUI-Actor（NeurIPS 2025）**：加一个定位 token，用注意力在视觉 patch 上选出动作区域，骨干可以冻结、只训这个头。它不用 DOM。WebGAP-V 是多个结构锚点进残差。两者都是注意力，问的问题不同：一个定位一个目标，一个把整页结构注进去。
- **ShowUI（CVPR 2025）**：把截图当成连通图，在自注意力里丢掉冗余视觉 token。主表只做这一步选择，不训它的 2B 模型，也不用它的 256K 数据。
- **HtmlRAG（WWW 2025）**：论证 HTML 比纯文本更适合作为知识格式，并做了清洗和剪枝。主表用它的清洗后 HTML 作为提示词行。它的检索、剪枝模型和 QA 数据集不进实验。

已发表的完整模型放附录：UGround（ICLR 2025）、Aria-UI（Findings of ACL 2025）、ShowUI 全文、UI-TARS、Magma、MultiUI、SeeClick、CogAgent。它们的 ScreenSpot 或 Mind2Web 数字和本表协议不同，格子留在附录。TAG（训练无关的注意力读出，arXiv 2024）和 Doc-AGround（Findings of ACL 2026，文档图）写进相关工作；主表不单列，避免和 GUI-Actor 行重复。

### 4.1 论文要回答的三件事

**RQ1. 同一 Qwen3-VL-8B-Instruct 上，哪种结构增强最好？**  
主表 1。目标是 WebGAP-V 在 WebSRC、VWB Element Grounding、VWB Action Grounding、ScreenSpot-v2 Web、Mind2Web Element Accuracy 上整体最好。现在还不是：VWB Action Grounding 上 GraphToken 91.3 高于 WebGAP-V 的 90.3；ScreenSpot-Web 上 LoRA 91.3、冻结 91.1，WebGAP-V 只有 73.2。

**RQ2. 这个名次是否换一个 8B 底座仍然成立？**  
主表 2，只做 InternVL3.5-8B。至少复现 Frozen、GraphToken、WebGAP-V。14B 和 30B 不做。

**RQ3. 涨点是否来自结构插件，而不是多训了参数？**  
模块消融（STAR、GACA、TRB、ERPR）加锚点置换。现有证据里 ERPR 损失接近 0，GACA 关掉后 hard EM 从 70.5 到 72.2。消融若显示某模块没有增益，就从贡献里删掉，不保留空模块来凑四段式。

### 4.2 故事顺序

1. 问题是如何把网页结构注入预训练 MLLM。
2. 在 Qwen3-VL-8B-Instruct 上，用统一约束比较 Frozen、LoRA、GraphToken、HTML prompt、Layout Prompt、Set-of-Mark、ShowUI 式 token 选择、GUI-Actor 式注意力头、WebGAP-V。
3. 主模型 WebGAP-V 在主基准上最好。Dual 若仍明显伤害点选，就写成变体。
4. 在 InternVL3.5-8B 上复现关键行，说明插件不绑死在 Qwen 上。
5. 消融和锚点置换说明增益来自结构通路。
6. 附录给出原文设定下的已发表 Web / GUI 方法，说明文献位置。

### 4.3 冲突子集怎么定义才站得住

不要把模型做错的题挑出来叫 hard set。规则要在看预测之前定好，并公开 id 列表。

在有渲染盒的页面上：

- 对可见元素按盒子中心做阅读序（先行后列），再按 DOM `sibling_index` 做文档序，算 Kendall τ。τ 低于预定阈值的页面进入高冲突桶。
- 另两个不依赖「挑错题」的桶：视口外仍可点击的节点；页面上文本高度重复、必须靠邻域才能区分的节点（多个相同按钮文案）。

Mind2Web 的 cleaned HTML 经常没有可靠渲染盒。`html_to_nodes` 的假盒子不能用来算 τ。需要二选一：用 Playwright 对测试页重渲染（先做 300–500 页的子集，写明 subset），或只用 WebSRC 里带 `data-bbox` 的元素。没有真实盒子的样本不进冲突桶。

合成 WebClash 继续负责「只改一条轴」的配对实验。它不叫公开榜，不占 Table 1。

---



## 5. 主表留哪些基准

主基准选和「网页结构 + 视觉定位」直接相关、而且方法之间还能分开的列。


| 列 | 数据 | 指标 | 角色 |
| --- | --- | --- | --- |
| 结构阅读 | WebSRC 官方 DEV subset（1,697 QA / 849 页，seed 42） | 文本 EM | 主表 |
| 元素定位 | VisualWebBench Element Grounding | 官方 accuracy | 主表 |
| 动作定位 | VisualWebBench Action Grounding | 官方 accuracy | 主表 |
| 网页点选 | ScreenSpot-v2 Web（本仓库 n=437） | 点在框内 | 主表。这是当前最大短板 |
| 元素选择 | Multimodal-Mind2Web 的 test_task、test_website、test_domain | 官方 Element Accuracy、Operation F1。输入是截图加正负候选，输出候选 id | 主表。现有动作字符串 F1 不填进此列 |


VisualWebBench 七任务均分、caption、webqa、heading OCR、element OCR 进附录。均分里有大量与结构注入无关的题。GUIAct web-single 有框项作副表，用来核对点选是否一起修好；无框的 321 条继续跳过并写明。

补负例：`eval.jsonl` 里现在只有 `pos_candidates` 和 `n_neg`。要从原来的 parquet 把 `neg_candidates` 写回来，所有方法共用这一套候选和输出格式。

冲突桶（τ、视口外、重复文案）和合成干预留在机制表，不占主排名。规则仍要在看预测之前定好。假盒子不能用来算 τ。

### 5.1 移出主表

- WebForge 易 held-out、泄漏模板、H1–H5、StHR、medium_struct。
- VisualWebBench 七任务均分，以及 caption、webqa、heading OCR、element OCR。
- 旧 WebSRC 的文本反查 POS。
- Mind2Web 动作字符串 F1 和全 0 的 element_em。约 21–29 的字符串分数不抄进 Element Accuracy。
- LLaVA 的点选分数。ScreenSpot 4.4、GUIAct 2.4 是格式失败。
- InternVL3.5-14B、更大的 Qwen、MiniCPM。零样本可以留在附录普查，不再训练。

不必新增：WebArena、VisualWebArena、WebVoyager、OSWorld、Android、DocVQA。Design2Code 只作可选探针。

---



## 6. 公平竞争池

所有主表行共用下面的约束。这是排名成立的条件。

- 底座：Qwen3-VL-8B-Instruct，或 InternVL3.5-8B。两张表分开，不混。
- 训练数据：同一份 WebForge 训练集。禁止额外网页数据、OCR 引擎、浏览器交互。
- 可训练参数量和训练步数同一量级。WebGAP 开 LoRA 时，LoRA、GraphToken、HTML prompt、Layout Prompt、Set-of-Mark、UI token selection、Action-head attention 也开同等 LoRA。GUI-Actor 原文可以只训注意力头；本表若只训头，WebGAP 的可训练量也要收在同一量级，不能一边只训头、一边全量 LoRA。
- 推理：同一分辨率、同一贪心解码、同一最大生成长度、同一提示词骨架。结构信息只替换各自的注入槽位。
- Mind2Web：同一套正负候选，输出元素 id。
- 插件在所有评测任务上保持开启。不为 ScreenSpot 单独关掉 WebGAP。

明确不做：把 GraphToken 或 Layout Prompt 的提示词改差、少训几步、换更小的 rank，用来制造 WebGAP 第一。现有结果里 GraphToken 的动作字符串 F1 更高，那一列本身就不进主表；下一轮用 Element Accuracy 重测，谁高谁写进表。

### 6.1 必须出现在 Qwen 主表的行

| 行 | 来源 | 状态 | 作用 |
| --- | --- | --- | --- |
| Frozen | — | 已有 | 底座 |
| LoRA | — | 已有 | 参数适应 |
| GraphToken | Talk like a Graph，ICLR 2024 | 已有 | 短文本。核心对照 |
| HTML prompt | HtmlRAG，WWW 2025 | 未做 | 清洗后的 HTML 进提示词 |
| Layout Prompt | 元素列表常见做法 | 未做 | 文本、类型、盒子进提示词 |
| Set-of-Mark | Yang 等 2023 | 未做 | 截图编号框 |
| UI token selection | ShowUI，CVPR 2025 | 未做 | 只移植视觉 token 选择 |
| Action-head attention | GUI-Actor，NeurIPS 2025 | 未做 | 只移植定位 token 的视觉注意力头 |
| WebGAP-V | 本文 | 已有，点选未达标 | 主模型 |
| WebGAP-DOM / Dual | 本文 | 已有 | 变体。Dual 在 ScreenSpot 修好前不升格 |

候选元素选择是评测协议，所有行都走，不单独变成一个「方法」。

### 6.2 InternVL3.5-8B 最少复现三行

Frozen 已有零样本。还要训 GraphToken、GUI-Actor 式注意力头和 WebGAP-V。HTML prompt、ShowUI 式 token 选择有余力再补。不再给 14B 接插件。

### 6.3 不移植

UI-TARS、ShowUI、Magma、MultiUI、SeeClick、CogAgent、UGround 不迁到这两个 8B 上。它们的训练数据、动作空间和目标函数都是另一套体系。附录只引用原文设定下的数字。

---



## 7. 主表

投稿时主表 1 和主表 2 上，WebGAP-V 应是同协议结构增强里的第一名。下面填的是**现在已经跑出来的数**。空格是还没做的行。Mind2Web 整列是 —†：当前 json 里约 21–29 的是动作字符串 F1，没有元素 id，不能填进 Element Accuracy。GraphToken 这一行是线性化 DOM 文本，不是 GNN soft token。

现在还不能写「WebGAP 已经最好」。Qwen 上 WebGAP-V 的 WebSRC 88.0 高于 Frozen 86.7 和 GraphToken 85.9，VWB Element Grounding 94.4 与 LoRA 并列、高于 Frozen 93.0；VWB Action Grounding 90.3 低于 GraphToken 的 91.3；ScreenSpot-Web 73.2 低于 LoRA 91.3、Frozen 91.1 和 GraphToken 84.9。Dual 的 Element Grounding 94.9 最高，ScreenSpot-Web 只有 33.4，所以 Dual 留在变体行。

### 表 1. Qwen3-VL-8B-Instruct，同协议结构增强

目标：改完插件并补齐空行之后，WebGAP-V 在这五列上整体第一。


| 方法 | 结构从哪进 | WebSRC | VWB-EG | VWB-AG | ScreenSpot-Web | Mind2Web Elem. Acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 不进 | 86.7 | 93.0 | 87.4 | 91.1 | —† |
| LoRA | 权重 | 86.7 | 94.4 | 88.3 | 91.3 | —† |
| GraphToken | 短 DOM 文本 | 85.9 | 93.7 | 91.3 | 84.9 | —† |
| HTML prompt | 清洗后 HTML | — | — | — | — | —† |
| Layout Prompt | 文本 + 盒子 | — | — | — | — | —† |
| Set-of-Mark | 截图编号框 | — | — | — | — | —† |
| UI token selection | 丢掉冗余视觉 token | — | — | — | — | —† |
| Action-head attention | 定位 token → 视觉 patch | — | — | — | — | —† |
| **WebGAP-V** | 结构锚点 → 注意力 | **88.0** | **94.4** | 90.3 | 73.2 | —† |
| WebGAP-Dual | 视觉锚点 + DOM 锚点 | 85.6 | 94.9 | 89.3 | 33.4 | —† |


† 官方 Element Accuracy 要求从候选里选出元素。当前只生成动作短句，留空。WebGAP-DOM 的现有数（WebSRC 88.1，VWB-EG 93.9，VWB-AG 86.4，ScreenSpot-Web 60.9）放进消融，不占主模型行。GUIAct 有框项现有数：Frozen 76.3，LoRA 76.3，GraphToken 74.3，WebGAP-V 69.5，WebGAP-DOM 62.0，Dual 35.1。副表，用来确认 ScreenSpot 修好时 GUIAct 一起回来。

### 表 2. InternVL3.5-8B，同一套方法

只填已有的冻结零样本。GraphToken、Action-head attention、WebGAP-V 训练后再填。14B 不进这张表。InternVL3.5-8B 冻结：WebSRC 86.3，VWB 七任务均分 72.6（附录），VWB-EG 72.6，VWB-AG 75.7，ScreenSpot-Web 73.9，GUIAct 49.4。


| 方法 | WebSRC | VWB-EG | VWB-AG | ScreenSpot-Web | Mind2Web Elem. Acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 86.3 | 72.6 | 75.7 | 73.9 | —† |
| GraphToken | — | — | — | — | —† |
| Action-head attention | — | — | — | — | —† |
| **WebGAP-V** | — | — | — | — | —† |


### 表 3. 消融

在 ScreenSpot 达标之后的 WebGAP-V 上做。行是去掉 STAR、去掉 GACA、去掉 TRB、去掉 ERPR。现有合成页上 ERPR 几乎没损失，GACA 关掉后 hard EM 70.5 → 72.2。没有增益的模块不写进贡献。

### 表 4. 机制，不代替排名

合成页，标题写 synthetic。用来说明涨点依赖锚点赋值，而不是只多训了 LoRA。


| 方法 | Visual Flip Follow | Identity EM | Global Permutation EM |
| --- | ---: | ---: | ---: |
| Frozen | 0 | — | — |
| LoRA | 50 | — | — |
| GraphToken | 8 | — | — |
| WebGAP-V | 100 | 100 | 12.5 |
| WebGAP-Dual | 100 | 93.8 | 12.5 |


Global Permutation 是锚点–区域赋值整体反转，不是均匀随机（均匀随机时 visual 为 25）。每条轴现在 48 对，正文报百分比前加到至少 200 对，或加配对 bootstrap。DOM 翻转目前各方法都是 0，这一列先放附录，不拿来证明「谁更会用 DOM」。

### 附录. 已发表参照

原文设定，不与表 1–2 排名。只列 Element Accuracy。


| 方法 | Model / Setting | Cross-Task | Cross-Website | Cross-Domain |
| --- | --- | ---: | ---: | ---: |
| MindAct | Flan-T5-XL, trained on Mind2Web | 55.1 | 42.0 | 42.1 |
| SeeAct | GPT-4V, choice | 46.4 | 38.0 | 42.4 |
| CogAgent | image coordinates, SeeAct Table 2 | 22.4 | 18.4 | 20.6 |
| SeeClick | GPT-4o planner + SeeAct-V | 32.1 | 33.1 | 33.5 |
| UGround-V1 | 7B locator + GPT-4o planner | 50.7 | 48.1 | 48.5 |


MindAct 的 Operation F1 与 Step SR（Cross-Task 75.7 / 52.0）不放进本表。UI-TARS、ShowUI、Magma、MultiUI 写进相关工作，不填数。

### 贡献（数字达标之后才这样写）

1. 一个结构插件：把网页视觉几何转成注意力级锚点，主模型是 WebGAP-V。
2. 在 Qwen3-VL-8B-Instruct 和 InternVL3.5-8B 上，与同一数据、同一预算的结构增强插件比较，WebGAP-V 在主基准上最好。
3. 锚点置换说明模型用到了注入的结构。

在表 1 的 ScreenSpot-Web 和 VWB-AG 仍低于对照时，摘要不写第 2 条。

---



## 8. 先改插件，再补基线

当前要改的是 WebGAP-V 的公开点选，目标是主表第一，而不是再加一个门控。

成功标准，全部在第 6 节的统一约束下：

- ScreenSpot-Web 回到至少不低于冻结底座（现在 91.1；WebGAP-V 是 73.2）。GUIAct 有框项同时回到不低于冻结的 76.3。
- VWB Action Grounding 高于 GraphToken 的 91.3（现在 90.3）。
- VWB Element Grounding 保持不低于 LoRA 的 94.4。
- WebSRC EM 保持不低于现在的 88.0。
- Mind2Web Element Accuracy 在同一候选协议下高于表 1 里其余各行。

插件在点选任务上保持开启。用残差缩放、插入层、锚点数量、训练时混入公开点选页来修，不用「grounding 任务推理时关掉插件」。

Dual 单独修。若修完后 ScreenSpot 仍远低于 WebGAP-V，Dual 留在消融。conflict-gate 不再训。ERPR 不再当贡献。14B 和 30B 不训。

---



## 9. 写作时直接改掉的句子

- 「三通道正交注入冻结 MLLM」。改成：一条共享残差插件，用盒子 IoU 做初始池化，在最多 48 个锚点之间加关系偏置。主模型是视觉锚点 WebGAP-V。
- 「GraphToken 基线」写成 GNN soft token 复现。改成：线性化 DOM 文本，引用 Talk like a Graph。
- 「Mind2Web 元素准确率」在现有 json 上。改成：下一步动作字符串的 token F1，不能与 Element Accuracy 比较。
- 「WebSRC POS 证明点对了节点」。改成：由预测文本反查 tid 的路径 Jaccard，与文本 EM 耦合。
- 「有时提示词更好，有时注意力更好」当作主结论。主结论改成同协议下 WebGAP-V 最好；在 ScreenSpot 和 VWB-AG 达标之前不写这句话。
- 「ERPR 压制注意力坍缩」。删除。
- 易模板 100% 任何「主表」标题。改成附录，标题写 OCR 天花板。
- 用更差的提示词、更少的步数或更小的 LoRA 让 GraphToken、HTML prompt、ShowUI 式选择或 GUI-Actor 式注意力头掉分。删除。排名只来自第 6 节的统一约束。

---



## 10. 实验顺序

本轮的第 0 步是 Mind2Web 元素选择，只跑 WebGAP-V（`outputs/runs/webgap_sft/checkpoint`，`anchor_mode=visual`）和 WebGAP-Dual（`outputs/runs/webgap_dual/checkpoint`）。脚本是 `scripts/run_qwen_webgap_mind2web_elem.py`。跑完再执行 `scripts/aggregate_qwen_main_table.py`。其余行以后再说。

不移植 UI-TARS、ShowUI、Magma、MultiUI、SeeClick、CogAgent。不训 14B。不在本轮训 InternVL。

1. 改 WebGAP-V，把 ScreenSpot-Web 和 GUIAct 拉回不低于冻结底座，同时保住 WebSRC 88.0 和 VWB-EG 94.4，并把 VWB-AG 做到高于 91.3。这是主表能写「第一」的前提。排在元素选择之后，本轮不跑。
2. 同一预算补四行，都只移植机制：HTML prompt（HtmlRAG 的清洗，不要它的检索器）、Layout Prompt、ShowUI 的 UI token selection、GUI-Actor 的定位注意力头。Set-of-Mark 放在这四行之后。全部与改完的 WebGAP-V 用同一数据、同一解码重跑表 1。这些论文在 WebSRC 或本协议 Mind2Web 上没有可直接抄的数字，空着等自己跑。
3. Mind2Web 元素选择的代码已接上，本轮只填 WebGAP-V 和 WebGAP-Dual。其它方法的这一列仍空着。协议是正候选加 50 个负候选，不是附录里的官方排序器结果。
4. InternVL3.5-8B 上至少训 GraphToken、GUI-Actor 式注意力头和 WebGAP-V，填表 2。有余力再加 HTML prompt 和 ShowUI 式 token 选择。
5. 在达标的 WebGAP-V 上做 STAR / GACA / TRB / ERPR 消融，以及锚点置换。没有增益的模块从贡献里删掉。

多种子：Mind2Web Element Accuracy 用 3 个数据折或配对 bootstrap。合成干预每条轴加到 200 对。不要为了方差重训五个插件。