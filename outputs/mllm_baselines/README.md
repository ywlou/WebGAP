# mllm_baselines

P3 零样本结果写在这里，不覆盖 `outputs/runs/`。

当前跑：`qwen3vl8b_zeroshot`（本地 `checkpoints/Qwen3-VL-8B-Instruct`，greedy）。

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **79.75** | `qwen3vl8b_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **91.08%** | `qwen3vl8b_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 86.7 / F1 90.4 / **POS 77.5** | `qwen3vl8b_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **76.31%**（跳过无框 321） | `qwen3vl8b_zeroshot/guiact_metrics.json` |
| Mind2Web test splits | — | **尚未评** | |

VisualWebBench 分任务（primary）：caption Rouge-L 29.0，webqa 88.0，heading 74.9，element OCR 95.6，element ground 93.0，action pred 90.4，action ground 87.4。旧短答 EM 协议作废。
