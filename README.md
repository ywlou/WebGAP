# WebGAP

**迁移或新开对话请先读 [交接文档.md](交接文档.md)**：当前研究计划、已完成实验、下一步（现停在 P3 零样本 zoo）、以及哪些数字不能当论文主表。

**Web Graph-Anchored attention Plugin**：面向冻结多模态大模型的网页结构注意力插件。

代码、脚本、配置、Git 路径一律用英文。本仓库的**说明文档均为中文**。论文图的坐标轴/图例仍为英文（方便投 The Web Conference），改图请用 `figures/data/*.csv` 与 `scripts/plot_results.py`。

WebGAP 把网页结构（DOM 拓扑、二维布局、图文对齐）通过三条正交通道注入冻结视觉语言模型：

| 模块 | 通道 | 作用 |
| --- | --- | --- |
| **STAR** | token ↔ 节点 | 用视觉 token 盒与区域 BBox 的 IoU 做几何赋值 |
| **GACA** | 全局 | 序列与少量**结构定义**图锚点之间的双向交叉注意力 |
| **TRB** | 局部 | 锚点–锚点注意力上的类型加性偏置（DOM / 空间 / 跨模态） |
| **ERPR** | 健康度 | 仅训练：GACA 映射熵下限，避免结构被平均掉 |

默认底座：**Qwen3-VL-8B-Instruct**（冻结 + LoRA）。训练数据由 **WebForge** 程序化生成（精确盒子 + H1–H5 结构题）。评测为模板留出的 WebHallu-Struct，外加可选公开截图基准。

文档入口：[docs/OVERVIEW.md](docs/OVERVIEW.md)（已完成任务与各实验动机/效果）、[docs/WEBCLASH.md](docs/WEBCLASH.md)（视觉序/文档序冲突基准）、[docs/BENCHMARK_SURVEY.md](docs/BENCHMARK_SURVEY.md)（公开基准调研 / 评测空间 P0）、[docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md)（与已有工作合成一篇论文的下一阶段计划）、[docs/USAGE.md](docs/USAGE.md)（用法）、[docs/DATA.md](docs/DATA.md)（数据）、[docs/METHOD.md](docs/METHOD.md)（方法对照）、[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)（实验表）、[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)（方案修订）。`docs/zh/` 为同内容备份（总览见 [docs/zh/工作与实验说明.md](docs/zh/工作与实验说明.md)）。

## 相对原始方案改了什么

见 [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)。短版：

1. **WebQA (CVPR 2022) 不是网页结构数据集**（维基片段+图），已删除。
2. 4k 长度下 token×token TRB 在 FlashAttention / 80GB 上不可跑。TRB 放在**锚点图**上。
3. vLLM 无法挂自定义层插件；评测用 Hugging Face `generate`。
4. 无 HTML 的真实截图使用 **4×4 空间网格**回退图。

## 仓库结构

```
src/webgap/          # 库（英文代码）
  data/              # WebForge、图、STAR、collator
  models/            # GACA/TRB/ERPR 插件 + Qwen3-VL 包装
  train/             # SFT / SSL
  eval/              # EM/F1/StHR + 生成
scripts/             # 生成 / 训练 / 评测 / 绘图 / 流水线
configs/default.yaml
data/webforge/       # 生成页（不进 git）
figures/{data,pdf,png,scripts}
docs/                # 中文文档（文件名为英文）
docs/zh/             # 中文文档备份
```

## 环境

```bash
conda activate torch2.4_cuda12.1   # 本机为 torch 2.6 + CUDA 12.4（transformers 5.x 需要）
export PYTHONPATH=/data/WebGAP/src
export HF_ENDPOINT=https://hf-mirror.com
# 权重：/data/WebGAP/checkpoints/Qwen3-VL-8B-Instruct
```

Python 3.10，单卡 NVIDIA A800 80GB。磁盘远低于 300GB（8B 权重约 18GB，WebForge 约 3–8GB，公开评测数 GB 量级）。

## 复现

```bash
# 1) 合成页（CPU）
python scripts/generate_webforge.py

# 2) 可选公开基准
python scripts/download_benchmarks.py

# 3) 训练主方法与基线（GPU）
python scripts/train.py --run-name webgap_sft --baseline webgap --epochs 2
python scripts/train.py --run-name lora_sft --baseline lora --epochs 2
python scripts/train.py --run-name graphtoken_sft --baseline graphtoken --epochs 2

# 4) 评测（模板留出）
python scripts/eval.py --run-name webgap_sft --ckpt outputs/runs/webgap_sft/checkpoint \
    --split heldout --max-samples 800 --tag webgap_heldout

python scripts/generate_webforge.py --hard-only --hard-pages 320
python scripts/eval_hard.py --max-samples 400

# 可选：在不相交的 hard_train 上继续 SFT
python scripts/train.py --run-name webgap_hardmix --baseline webgap --epochs 1 \
  --max-samples 800 --ckpt outputs/runs/webgap_sft/checkpoint --split hard_train

# 5) 出图与中文实验报告
python scripts/plot_results.py
python scripts/write_report.py

# 全流程（outputs/pipeline_stamps/ 可断点）
python scripts/run_all.py
python scripts/run_all.py --smoke    # 4 step 冒烟
```

基线：`zeroshot`（B0）、`lora`（B1）、`graphtoken`（B3，图线性化进问题）、`random_anchor`（B5，推理期置换 STAR 列）。`--no-gaca` / `--no-trb` 为推理期消融（不重训）。`--no-erpr` 需要重训。

**往论文里抄表之前先读 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。** 易模板 held-out 在 SFT 后触顶；真正有信息量的是随机 STAR 坍塌、`hard_struct`（源序探针上 GraphToken 领先）、以及效率微基准。

## 指标

- **EM / token F1**：短答，greedy 解码。
- **StHR**：在仍提到黄金实体的预测中，关系答错的比例。越低越好。分 H1–H5 报告。
- **Held-out vs 泄漏模板**：量化模板记忆。

## 引用

```
@inproceedings{webgap,
  title={WebGAP: Graph-Anchored Attention for Web Structure Understanding in Frozen MLLMs},
  author={WebGAP authors},
  booktitle={The Web Conference},
  year={2026}
}
```

## 许可

Apache-2.0。Qwen3-VL 权重遵循其自身许可。
