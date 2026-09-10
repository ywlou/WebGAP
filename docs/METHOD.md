# 方法（实现对照）

论文模块 → 代码：

| 论文 | 文件 | 说明 |
| --- | --- | --- |
| WebForge | `src/webgap/data/templates.py`、`renderer.py`、`qa.py`、`webforge.py` | PIL 布局，精确盒子 |
| 异构图 | `src/webgap/data/graph.py` | DOM + 空间 + CMA 边 |
| STAR | `visual_anchor_pack` / `dom_anchor_pack` | 视觉：token→**区域** IoU；DOM：token→**节点** IoU |
| GACA | `WebGAPPlugin` | 视觉流 / DOM 流 / 双序门控（`anchor_mode`） |
| TRB | `trb_table`、`_trb_bias` | 视觉流：区域图；DOM 流：节点图（含源序兄弟边） |
| 顺序嵌入 | `order_emb` | 视觉流=阅读序；DOM 流=文档序 `0..k-1` |
| 门控 | `mix` | token 级 sigmoid；bias=+2 起步偏视觉 |
| ERPR | scatter 熵铰链 | 辅助正则；短页上通常不触发，不作主创新 |
| B5 | `GraphBatch.perm` | **只打乱视觉 STAR**；双序模型应能倒向 DOM |
| ERPR | scatter 映射熵，相对 `erpr_tau` 的铰链 | 仅训练 |
| 插入 | `src/webgap/models/wrapper.py` | 每隔 `insert_every` 层，权重共享 |
| B3 GraphToken | `serialize_graph_text` 拼进问题 | 同一 LoRA 预算 |
| B5 | `GraphBatch.perm` 列置换 | 推理 |
| hard_struct | `templates.py` + `qa.py` | 易混 / 密表 / RTL / 交叉图注 |

GACA 残差为 `x + tanh(gate) * W_out(Δ)`，`W_out` 零初始化。

仅 prefill：包装层在 `hidden.shape[1] <= 1` 时跳过插件，保证 decode KV 合法。

兄弟 TRB 类型跟 `sibling_index`（标记源序）。空间 LEFT/RIGHT 跟盒子中心。CMA 优先 figure/DOM 父节点，因此图注可以画在*另一张*图下面。

**双序锚点（相对第一版视觉区域锚点的翻盘）：** 区域级 STAR 把同一导航条的全部 `nav_item` 收成一个锚点，源序兄弟边在 TRB 上变成自环被丢掉——冲突页上等于把「屏幕几何」当成答案。DOM 锚点一人一节点，TRB 兄弟边还在；双序门控让问题 token 在「看屏幕」和「看标记序」之间选。消融：`--anchor-mode visual|dom|dual`。

受控冲突协议见 [WEBCLASH.md](WEBCLASH.md)。公开主表与下载清单见 [BENCHMARK_SURVEY.md](BENCHMARK_SURVEY.md)；不要用易模板 100% 排序方法。
