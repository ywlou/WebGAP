# 实验报告

由 `scripts/write_report.py` 根据 `outputs/runs/*/eval` 与 `figures/data/*.csv` 生成。数字不要手改；新评测后重新跑本脚本。

## 先看结论（面向 WWW 审稿）

模板留出的 WebForge 页面对 Qwen3-VL-8B **过易**（大标题 OCR 天花板）。短答 SFT 之后 LoRA 与 WebGAP 都触顶（提取器处理 `Overview — Product` 之后 EM=100%）。因此在该集上推理期关掉 GACA/TRB **不会**掉点——LoRA 已经学会格式。

**真正能写进论文机制段的证据是：**

1. **随机打乱 STAR（B5）**：易模板 held-out 上 EM 从 100% 掉到 **26.6%**（95% CI 22.6–30.4）；hard_struct 上掉到 **10.5%**。插件确实在用几何锚点。
2. **hard_struct、不再额外 SFT。** Frozen 69.5 / LoRA 72.0 / WebGAP 70.5 / **GraphToken 78.0**。`dom` 探针（源序/figure 归属，与像素相反）GraphToken **33%**，LoRA/WebGAP/冻结都接近 **0%**。把图线性化进 prompt 能零样本带上源序；只在「视觉=DOM」页上训过的注意力插件做不到。
3. **在不相交的** `hard_train` **上再 SFT 100 step。** WebGAP 99.2 / LoRA 99.0 / GraphToken 98.0，三者 `dom` 都是 100%。一旦有冲突标注，LoRA 就够了，插件不是这条诊断的必要条件。
4. **效率：** 相对冻结底座生成延迟约 **+6.8%**（0.319s vs 0.299s），显存 +0.06GB。
5. **公开集：** 合成短答 SFT 略伤 VisualWebBench；WebGAP 略高于纯 LoRA，但都低于零样本。WebSRC 抽样约 92%，饱和。

写给 WWW：主机制写 B5 与 GraphToken 在冲突页上的源序迁移；不要把易模板 EM 当排序表。若产品问题是视觉序≠文档序，训练里要混冲突页。

我们**不**声称线上智能体 SOTA，也不声称 WebForge 可以替代 WebSRC/VisualWebBench。

## 协议

- 底座：Qwen3-VL-8B-Instruct，视觉与 LLM 冻结，LoRA r=16。
- 解码：greedy，`max_new_tokens=64`。指标：规范化 EM / token F1 / StHR。
- SFT：3000 条 WebForge QA × 2 epoch，累积 8 → 750 step。训练峰值约 19GB。
- held-out：3 个从未训练的模板族，500 题。
- 泄漏分割：训练模板族、新种子，200 题。
- hard_struct：320 页，评 400 题。可选 `hard_train`（240 页，另一种子）仅用于 100-step 冲突 SFT，与评测不相交。
- 消融 `−GACA` / `−TRB` / 随机 STAR：**不重训**。`−ERPR`：重训 1.5 epoch。
- 公开：VisualWebBench 300 + WebSRC 300，无 HTML 时用 4×4 空间网格。



## 主表（模板留出 WebHallu-Struct）


| 方法              | N   | EM    | F1    | StHR ↓ |
| --------------- | --- | ----- | ----- | ------ |
| B0 Frozen       | 500 | 90.6  | 89.5  | 0.0    |
| B1 LoRA         | 500 | 100.0 | 100.0 | 0.0    |
| B3 GraphToken   | 500 | 99.2  | 98.7  | 0.0    |
| M WebGAP        | 500 | 100.0 | 100.0 | 0.0    |
| M −ERPR         | 500 | 99.8  | 98.7  | 0.0    |
| M −GACA (infer) | 500 | 100.0 | 100.0 | 0.0    |
| M −TRB (infer)  | 500 | 100.0 | 100.0 | 0.0    |
| B5 Random STAR  | 500 | 26.6  | 23.4  | 0.0    |


原始数据：`figures/data/fig_main_bars.csv`、`figures/data/all_metrics.csv`。图：`figures/pdf/fig_main_bars.pdf`。

## 消融（held-out，除注明外为同一 WebGAP 检查点）


| 方法              | N   | EM    | F1    | StHR ↓ |
| --------------- | --- | ----- | ----- | ------ |
| B0 Frozen       | 500 | 90.6  | 89.5  | 0.0    |
| B1 LoRA         | 500 | 100.0 | 100.0 | 0.0    |
| B3 GraphToken   | 500 | 99.2  | 98.7  | 0.0    |
| M WebGAP        | 500 | 100.0 | 100.0 | 0.0    |
| M −ERPR         | 500 | 99.8  | 98.7  | 0.0    |
| M −GACA (infer) | 500 | 100.0 | 100.0 | 0.0    |
| M −TRB (infer)  | 500 | 100.0 | 100.0 | 0.0    |
| B5 Random STAR  | 500 | 26.6  | 23.4  | 0.0    |


图：`figures/pdf/fig_ablation.pdf`。随机锚点柱是 STAR 的因果检验。

## hard_struct 诊断


| 方法              | N    | EM   | F1   |
| --------------- | ---- | ---- | ---- |
| B0 Frozen       | 1500 | 66.7 | 66.8 |
| B1 LoRA         | 1500 | 72.0 | 72.0 |
| B3 GraphToken   | 400  | 78.0 | 78.0 |
| M WebGAP        | 1500 | 70.6 | 70.7 |
| M + hard SFT    | 400  | 99.2 | 99.2 |
| B1 + hard SFT   | 400  | 99.0 | 99.0 |
| B3 + hard SFT   | 400  | 98.0 | 98.0 |
| M −GACA (infer) | 400  | 72.2 | 72.2 |
| B5 Random STAR  | 400  | 10.5 | 9.8  |


探针：`visual` 屏上几何，`dom` 源序/figure 归属，`bind` 易混标题，`cell` 密表单元格。
图：`figures/pdf/fig_hard.pdf`、`fig_hard_probe.pdf`、`fig_hard_template.pdf`。CSV：`figures/data/fig_hard.csv`。

## 公开截图迁移


| 方法        | VWB EM | VWB F1 | WebSRC EM | WebSRC F1 |
| --------- | ------ | ------ | --------- | --------- |
| B0 Frozen | 18.3   | 32.5   | 92.3      | 93.2      |
| B1 LoRA   | 15.3   | 23.9   | 92.0      | 93.3      |
| M WebGAP  | 16.7   | 28.0   | 92.0      | 93.3      |


图：`figures/pdf/fig_public.pdf`。

## 效率（A800 80GB，12 张 held-out 页，`max_new_tokens=8`）


| 标签          | 延迟 (s) | 峰值 GB | N   |
| ----------- | ------ | ----- | --- |
| B0_zeroshot | 0.299  | 16.71 | 12  |
| M_webgap    | 0.319  | 16.76 | 12  |
| B1_lora     | 0.305  | 16.76 | 12  |


图：`figures/pdf/fig_efficiency.pdf`。

## Bootstrap 95% CI（题目重采样，1000 次）


| 方法          | 分割      | N   | EM    | 95% CI      |
| ----------- | ------- | --- | ----- | ----------- |
| zeroshot    | heldout | 500 | 90.6  | 87.8–93.2   |
| lora        | heldout | 500 | 100.0 | 100.0–100.0 |
| graphtoken  | heldout | 500 | 99.2  | 98.4–99.8   |
| webgap      | heldout | 500 | 100.0 | 100.0–100.0 |
| rand_anchor | heldout | 500 | 26.6  | 22.6–30.4   |
| zeroshot    | hard    | 400 | 69.5  | 65.0–74.2   |
| lora        | hard    | 400 | 72.0  | 67.5–76.5   |
| graphtoken  | hard    | 400 | 78.0  | 73.8–82.2   |
| webgap      | hard    | 400 | 70.5  | 66.0–74.8   |
| rand_anchor | hard    | 400 | 10.5  | 7.5–13.2    |
| no_gaca     | hard    | 400 | 72.2  | 67.8–76.8   |
| webgap      | hardmix | 400 | 99.2  | 98.2–100.0  |
| lora        | hardmix | 400 | 99.0  | 98.0–100.0  |
| graphtoken  | hardmix | 400 | 98.0  | 96.5–99.2   |


CSV：`figures/data/bootstrap_em.csv`。

## 训练


| 实验             | 步数      | 墙钟 (s) | 可训练参数 (M) | 峰值 GB |
| -------------- | ------- | ------ | --------- | ----- |
| webgap_sft     | 750     | ~3650  | 29.9      | ~19.2 |
| lora_sft       | 750     | ~2877  | 仅 LoRA    | ~19   |
| graphtoken_sft | 750     | ~4200  | 仅 LoRA    | ~19   |
| webgap_no_erpr | ~1.5 ep | ~2735  | 29.9      | ~19   |


损失曲线：`figures/pdf/fig_train_loss.pdf` / `figures/data/fig_train_loss.csv`。
ERPR 铰链损失全程接近 0（这些页上 scatter 熵已高于 τ=0.65）。

## 如何重绘论文图

```bash
python scripts/plot_results.py
```

字号 / 图尺寸改 `scripts/plot_results.py` 的 `style()`（副本在 `figures/scripts/plot_results.py`）。不要手改 PNG。PDF 使用 `pdf.fonttype=42`。轴标签保持英文以便投稿。

## 局限（写进论文）

- 易模板 SFT 后饱和，该分割上的模块消融无信息量。
- VisualWebBench 含 caption/OCR/grounding，短答 SFT 域不匹配。
- 公开页没有 HTML，TRB 的 DOM 类型缺失。
- 单种子、单一 8B 底座；InternVL3 因 hidden 3584 vs 4096 未做移植。



## 全部原始指标文件


| 实验                   | 文件                                                            | n    | em     | f1     | sthr   |
| -------------------- | ------------------------------------------------------------- | ---- | ------ | ------ | ------ |
| graphtoken_hardmix   | graphtoken_hardmix_hard_hard_struct_metrics.json              | 400  | 0.9800 | 0.9800 | 0.0000 |
| webgap_sft           | pub_visualwebbench_webgap_sft_metrics.json                    | 300  | 0.1667 | 0.2802 | 0.0741 |
| webgap_sft           | webgap_sft_hard_struct_hard_struct_metrics.json               | 1500 | 0.7060 | 0.7068 | 0.0000 |
| webgap_sft           | pub_websrc_webgap_sft_metrics.json                            | 300  | 0.9200 | 0.9333 | 0.0000 |
| webgap_sft           | webgap_sft_hard_hard_struct_metrics.json                      | 400  | 0.7050 | 0.7050 | 0.0000 |
| webgap_sft           | webgap_sft_leaked_templates_leaked_templates_metrics.json     | 200  | 1.0000 | 1.0000 | 0.0000 |
| webgap_sft           | webgap_sft_heldout_heldout_metrics.json                       | 500  | 1.0000 | 1.0000 | 0.0000 |
| webgap_sft           | webgap_sft_medium_struct_medium_struct_metrics.json           | 800  | 0.9812 | 0.9812 | 0.0000 |
| lora_sft             | lora_sft_leaked_templates_leaked_templates_metrics.json       | 200  | 1.0000 | 1.0000 | 0.0000 |
| lora_sft             | pub_websrc_lora_sft_metrics.json                              | 300  | 0.9200 | 0.9333 | 0.0000 |
| lora_sft             | lora_sft_hard_struct_hard_struct_metrics.json                 | 1500 | 0.7200 | 0.7204 | 0.0000 |
| lora_sft             | lora_sft_medium_struct_medium_struct_metrics.json             | 800  | 0.9750 | 0.9750 | 0.0000 |
| lora_sft             | lora_sft_hard_hard_struct_metrics.json                        | 400  | 0.7200 | 0.7200 | 0.0000 |
| lora_sft             | pub_visualwebbench_lora_sft_metrics.json                      | 300  | 0.1533 | 0.2392 | 0.0417 |
| lora_sft             | lora_sft_heldout_heldout_metrics.json                         | 500  | 1.0000 | 1.0000 | 0.0000 |
| webgap_no_erpr       | webgap_no_erpr_heldout_heldout_metrics.json                   | 500  | 0.9980 | 0.9867 | 0.0000 |
| zeroshot             | zeroshot_leaked_templates_leaked_templates_metrics.json       | 200  | 0.9400 | 0.9400 | 0.0000 |
| zeroshot             | zeroshot_medium_struct_medium_struct_metrics.json             | 800  | 0.8287 | 0.8287 | 0.0000 |
| zeroshot             | zeroshot_heldout_heldout_metrics.json                         | 500  | 0.9060 | 0.8947 | 0.0000 |
| zeroshot             | zeroshot_hard_struct_hard_struct_metrics.json                 | 1500 | 0.6673 | 0.6679 | 0.0000 |
| zeroshot             | pub_websrc_zeroshot_metrics.json                              | 300  | 0.9233 | 0.9322 | 0.0000 |
| zeroshot             | pub_visualwebbench_zeroshot_metrics.json                      | 300  | 0.1833 | 0.3250 | 0.1129 |
| zeroshot             | zeroshot_hard_hard_struct_metrics.json                        | 400  | 0.6950 | 0.6950 | 0.0000 |
| webgap_no_gaca       | webgap_no_gaca_hard_hard_struct_metrics.json                  | 400  | 0.7225 | 0.7225 | 0.0000 |
| webgap_no_gaca       | webgap_no_gaca_heldout_heldout_metrics.json                   | 500  | 1.0000 | 1.0000 | 0.0000 |
| webgap_no_gaca       | webgap_no_gaca_medium_struct_medium_struct_metrics.json       | 800  | 0.9650 | 0.9650 | 0.0000 |
| webgap_no_gaca       | webgap_no_gaca_hard_struct_hard_struct_metrics.json           | 1500 | 0.7167 | 0.7175 | 0.0000 |
| webgap_ssl_sft       | webgap_ssl_sft_hard_struct_hard_struct_metrics.json           | 1500 | 0.7260 | 0.7264 | 0.0000 |
| webgap_ssl_sft       | pub_visualwebbench_webgap_ssl_sft_metrics.json                | 300  | 0.1467 | 0.2182 | 0.0222 |
| webgap_ssl_sft       | webgap_ssl_sft_medium_struct_medium_struct_metrics.json       | 800  | 0.8912 | 0.8912 | 0.0000 |
| webgap_ssl_sft       | pub_websrc_webgap_ssl_sft_metrics.json                        | 300  | 0.9133 | 0.9233 | 0.0000 |
| webgap_dual_mix      | webgap_dual_mix_medium_struct_medium_struct_metrics.json      | 800  | 0.9187 | 0.9187 | 0.0000 |
| webgap_dual_mix      | pub_websrc_webgap_dual_mix_metrics.json                       | 300  | 0.8267 | 0.8517 | 0.0198 |
| webgap_dual_mix      | pub_visualwebbench_webgap_dual_mix_metrics.json               | 300  | 0.1333 | 0.2098 | 0.0698 |
| webgap_dual_mix      | webgap_dual_mix_hard_struct_hard_struct_metrics.json          | 1500 | 0.7093 | 0.7102 | 0.0000 |
| webgap_dual          | pub_websrc_webgap_dual_metrics.json                           | 300  | 0.9133 | 0.9267 | 0.0000 |
| webgap_dual          | webgap_dual_hard_struct_hard_struct_metrics.json              | 1500 | 0.7093 | 0.7106 | 0.0000 |
| webgap_dual          | webgap_dual_medium_struct_medium_struct_metrics.json          | 800  | 0.9788 | 0.9788 | 0.0000 |
| webgap_dual          | pub_visualwebbench_webgap_dual_metrics.json                   | 300  | 0.1667 | 0.2755 | 0.0909 |
| webgap_dual_rand_vis | webgap_dual_rand_vis_medium_struct_medium_struct_metrics.json | 800  | 0.2263 | 0.1394 | 0.0109 |
| webgap_dual_rand_vis | webgap_dual_rand_vis_hard_struct_hard_struct_metrics.json     | 1500 | 0.1780 | 0.0914 | 0.0000 |
| webgap_dom           | webgap_dom_hard_struct_hard_struct_metrics.json               | 1500 | 0.7107 | 0.7115 | 0.0000 |
| webgap_dom           | webgap_dom_medium_struct_medium_struct_metrics.json           | 800  | 0.9775 | 0.9775 | 0.0000 |
| webgap_hardmix       | webgap_hardmix_hard_hard_struct_metrics.json                  | 400  | 0.9925 | 0.9925 | 0.0000 |
| webgap_no_trb        | webgap_no_trb_heldout_heldout_metrics.json                    | 500  | 1.0000 | 1.0000 | 0.0000 |
| graphtoken_sft       | graphtoken_sft_hard_hard_struct_metrics.json                  | 400  | 0.7800 | 0.7800 | 0.0000 |
| graphtoken_sft       | graphtoken_sft_medium_struct_medium_struct_metrics.json       | 800  | 0.9500 | 0.9500 | 0.0000 |
| graphtoken_sft       | graphtoken_sft_hard_struct_hard_struct_metrics.json           | 1500 | 0.7867 | 0.7870 | 0.0000 |
| graphtoken_sft       | graphtoken_sft_heldout_heldout_metrics.json                   | 500  | 0.9920 | 0.9873 | 0.0000 |
| webgap_rand_anchor   | webgap_rand_anchor_medium_struct_medium_struct_metrics.json   | 800  | 0.1350 | 0.0946 | 0.2987 |
| webgap_rand_anchor   | webgap_rand_anchor_hard_hard_struct_metrics.json              | 400  | 0.1050 | 0.0984 | 0.3438 |
| webgap_rand_anchor   | webgap_rand_anchor_hard_struct_hard_struct_metrics.json       | 1500 | 0.1173 | 0.0890 | 0.3714 |
| webgap_rand_anchor   | webgap_rand_anchor_heldout_heldout_metrics.json               | 500  | 0.2660 | 0.2343 | 0.0000 |
| lora_hardmix         | lora_hardmix_hard_hard_struct_metrics.json                    | 400  | 0.9900 | 0.9900 | 0.0000 |


