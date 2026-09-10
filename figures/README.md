# 论文图

| 路径 | 内容 |
| --- | --- |
| `data/*.csv` | 绘图源表（建议进 git） |
| `pdf/*.pdf` | 矢量图（`pdf.fonttype=42`） |
| `png/*.png` | 栅格预览 + `sample_*.jpg` 样例页 |
| `scripts/plot_results.py` | `../../scripts/plot_results.py` 的副本 |

坐标轴、图例、方法名保持**英文**（投英文会议）。改字号、刻度、`figsize` 请改 `style()` 后重跑：

```bash
python scripts/plot_results.py
```

不要为了改 legend 去 GUI 里改 PNG。
