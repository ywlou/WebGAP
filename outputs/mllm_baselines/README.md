# mllm_baselines

P3 零样本结果写在这里，不覆盖 `outputs/runs/`。

已完成：`qwen3vl8b_zeroshot`, `internvl35_8b_zeroshot`, `internvl35_14b_zeroshot`, `llava_ov_7b_zeroshot`, `minicpm_v45_zeroshot`。
未完成：无。

权重在 `/data/models/`。续跑命令见仓库根目录 `交接文档.md` 第 4 节。

Mind2Web 指标是下一步动作字符串匹配，不是 Task SR。旧 VWB 短答 EM 协议作废。

### Qwen3-VL-8B-Instruct

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **79.75** | `qwen3vl8b_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **91.08%** | `qwen3vl8b_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 86.7 / F1 90.4 / **POS 77.5** | `qwen3vl8b_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **76.31%**（跳过无框 321） | `qwen3vl8b_zeroshot/guiact_metrics.json` |
| Mind2Web test_website | 1019 | element_em 0.0% / action_f1 21.3% | `qwen3vl8b_zeroshot/mind2web_test_website_metrics.json` |
| Mind2Web test_task | 1338 | element_em 0.0% / action_f1 21.6% | `qwen3vl8b_zeroshot/mind2web_test_task_metrics.json` |
| Mind2Web test_domain | 4050 | element_em 0.0% / action_f1 24.1% | `qwen3vl8b_zeroshot/mind2web_test_domain_metrics.json` |
### InternVL3.5-8B-HF

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **72.62** | `internvl35_8b_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **73.91%** | `internvl35_8b_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 86.3 / F1 90.3 / **POS 77.6** | `internvl35_8b_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **49.40%**（跳过无框 321） | `internvl35_8b_zeroshot/guiact_metrics.json` |
| Mind2Web test_website | 1019 | element_em 0.0% / action_f1 9.2% | `internvl35_8b_zeroshot/mind2web_test_website_metrics.json` |
| Mind2Web test_task | 1338 | element_em 0.0% / action_f1 7.5% | `internvl35_8b_zeroshot/mind2web_test_task_metrics.json` |
| Mind2Web test_domain | 4050 | element_em 0.0% / action_f1 10.5% | `internvl35_8b_zeroshot/mind2web_test_domain_metrics.json` |
### InternVL3.5-14B-HF

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **75.18** | `internvl35_14b_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **89.02%** | `internvl35_14b_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 86.3 / F1 82.7 / **POS 71.7** | `internvl35_14b_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **75.67%**（跳过无框 321） | `internvl35_14b_zeroshot/guiact_metrics.json` |
| Mind2Web test_website | 1019 | element_em 0.0% / action_f1 18.4% | `internvl35_14b_zeroshot/mind2web_test_website_metrics.json` |
| Mind2Web test_task | 1338 | element_em 0.0% / action_f1 19.4% | `internvl35_14b_zeroshot/mind2web_test_task_metrics.json` |
| Mind2Web test_domain | 4050 | element_em 0.0% / action_f1 22.3% | `internvl35_14b_zeroshot/mind2web_test_domain_metrics.json` |
### LLaVA-OneVision-7B (HF conversion)

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **55.81** | `llava_ov_7b_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **4.35%** | `llava_ov_7b_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 79.8 / F1 85.1 / **POS 74.9** | `llava_ov_7b_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **2.39%**（跳过无框 321） | `llava_ov_7b_zeroshot/guiact_metrics.json` |
| Mind2Web test_website | 1019 | element_em 0.0% / action_f1 12.4% | `llava_ov_7b_zeroshot/mind2web_test_website_metrics.json` |
| Mind2Web test_task | 1338 | element_em 0.0% / action_f1 11.1% | `llava_ov_7b_zeroshot/mind2web_test_task_metrics.json` |
| Mind2Web test_domain | 4050 | element_em 0.0% / action_f1 13.8% | `llava_ov_7b_zeroshot/mind2web_test_domain_metrics.json` |
### MiniCPM-V-4.5

| 基准 | n | 主指标 | 文件 |
| --- | --- | --- | --- |
| VisualWebBench 官方七类 | 1536 | official_avg **76.40** | `minicpm_v45_zeroshot/visualwebbench_metrics.json` |
| ScreenSpot-v2 web | 437 | point-in-bbox **63.39%** | `minicpm_v45_zeroshot/screenspot_v2_metrics.json` |
| WebSRC official DEV subset | 1697 | EM 80.1 / F1 81.5 / **POS 78.8** | `minicpm_v45_zeroshot/websrc_metrics.json` |
| GUIAct web-single | 1089 | point-in-bbox **51.70%**（跳过无框 321） | `minicpm_v45_zeroshot/guiact_metrics.json` |
| Mind2Web test_website | 1019 | element_em 0.0% / action_f1 20.3% | `minicpm_v45_zeroshot/mind2web_test_website_metrics.json` |
| Mind2Web test_task | 1338 | element_em 0.0% / action_f1 21.1% | `minicpm_v45_zeroshot/mind2web_test_task_metrics.json` |
| Mind2Web test_domain | 4050 | element_em 0.0% / action_f1 24.4% | `minicpm_v45_zeroshot/mind2web_test_domain_metrics.json` |
