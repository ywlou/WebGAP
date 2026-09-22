# 使用说明

代码与命令为英文；本文档为中文。

## 环境

- GPU：1× NVIDIA A800 80GB
- Conda 环境 `torch2.4_cuda12.1`（本机已原地升级为 **PyTorch 2.6.0+cu124**，因为 `transformers` 5.x 需要 torch ≥ 2.5）
- `export PYTHONPATH=/data/WebGAP/src`
- `export HF_ENDPOINT=https://hf-mirror.com`（可选；权重经 ModelScope 拉取）
- 权重：`/data/models/Qwen/Qwen3-VL-8B-Instruct`（`WEBGAP_MODELS_DIR`，禁止放进仓库）

若 CUDA OOM：在 `configs/default.yaml` 降低 `model.max_pixels`（默认 `802816`）和 `train.max_seq_len`。

可选安装：

```bash
pip install -e .
```

## 生成数据

```bash
python scripts/generate_webforge.py                 # 1 万训练 + 2 千留出 + 泄漏模板 + hard_struct
python scripts/generate_webforge.py --smoke         # 调试用小分割
python scripts/generate_webforge.py --hard-only --hard-pages 320
python scripts/download_benchmarks.py               # VisualWebBench + WebSRC 抽样
```

输出：`data/webforge/{train,heldout,leaked_templates,hard_struct}/index.jsonl` 与 `images/*.jpg`。

每条 jsonl 含 `nodes`（盒子 + DOM 式父节点）、`qa`（H1–H5，可选 `probe`）、`meta.template`。

## 训练

```bash
python scripts/train.py --config configs/default.yaml \
  --run-name webgap_sft --baseline webgap --stage sft --epochs 2
```

--baseline` ∈ `{webgap, lora, graphtoken, zeroshot, random_anchor}`。

`--anchor-mode visual|dom|dual`：视觉区域锚点 / DOM 节点锚点 / 双序门控。`--no-erpr` 关闭熵铰链并重训。`--max-steps` / `--max-samples` 用于冒烟。`--mix-splits hard_train` 把冲突页混进 SFT。`--replay-ratio 0.15` 混入开放布局描述，减轻短答过拟合。

`--no-erpr` 关闭熵铰链并重训。`--max-steps` / `--max-samples` 用于冒烟。

检查点：`outputs/runs/<run>/checkpoint/{plugin.pt,backbone/,processor/}`，另有 `train_log.json`。

冲突页继续训练（与评测 `hard_struct` 种子不相交）：

```bash
python scripts/train.py --run-name webgap_hardmix --baseline webgap --epochs 1 \
  --max-samples 800 --ckpt outputs/runs/webgap_sft/checkpoint --split hard_train
```

## 评测

```bash
python scripts/eval.py --run-name webgap_sft \
  --ckpt outputs/runs/webgap_sft/checkpoint \
  --split heldout --max-samples 500 --tag webgap_heldout

python scripts/eval.py --run-name webgap_rand_anchor \
  --ckpt outputs/runs/webgap_sft/checkpoint \
  --split heldout --max-samples 500 --random-anchors --tag webgap_rand_anchor

python scripts/eval_hard.py --max-samples 400
python scripts/eval_public.py
python scripts/efficiency.py
python scripts/rescore.py          # 改过 extract_answer 之后重打分
```

`--no-gaca` / `--no-trb` 推理期关闭模块（同一权重）。`--random-anchors` 置换 STAR 列（B5）。

分割：`train` | `heldout` | `leaked_templates` | `hard_struct`。

## 图与报告

```bash
python scripts/plot_results.py
python scripts/write_report.py
```

- 原始数列：`figures/data/*.csv`
- 矢量 PDF：`figures/pdf/*.pdf`（`pdf.fonttype=42`）
- 字号 / 图尺寸：改 `scripts/plot_results.py` 的 `style()`（镜像在 `figures/scripts/plot_results.py`）
- 不要为了改图例去手改 PNG；论文轴标签保持英文以便投稿

`write_report.py` 只生成**中文**实验报告：`docs/EXPERIMENTS.md` 与 `docs/zh/实验报告.md`。

## 全流程

`python scripts/run_all.py` 可通过 `outputs/pipeline_stamps/<name>` 断点续跑。删掉某个戳记即重跑该阶段。日志：`logs/<stamp>.log`。

```bash
python scripts/run_all.py --smoke    # 4 step 冒烟
```
