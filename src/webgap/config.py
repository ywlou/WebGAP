"""YAML/CLI experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from webgap.constants import DEFAULT_QWEN3VL_8B


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    name_or_path: str = str(DEFAULT_QWEN3VL_8B)
    processor_path: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 256 * 28 * 28
    freeze_vision: bool = True
    freeze_llm: bool = True


@dataclass
class PluginConfig:
    bottleneck: int = 768
    num_heads: int = 8
    max_anchors: int = 48
    insert_every: int = 2  # insert at layers 1,3,5,...
    dropout: float = 0.0
    use_gaca: bool = True
    use_trb: bool = True
    use_erpr: bool = True
    erpr_tau: float = 0.65
    erpr_lambda: float = 0.05
    rec_lambda: float = 0.2
    con_lambda: float = 0.1
    gate_init: float = 0.0
    share_across_layers: bool = True
    random_anchor_perm: bool = False  # B5: permute visual STAR columns
    # visual = region IoU (geometry); dom = per-node IoU + source-order TRB;
    # dual = both streams + token-wise gate (initially visual-heavy).
    anchor_mode: str = "visual"  # visual | dom | dual
    gate_bias_init: float = 2.0
    # P6: add an explicit visual-vs-DOM order-conflict feature into mix(x).
    # False keeps dual identical to the P0/P4 checkpoint.
    conflict_gate: bool = False
    compute_ssl: bool = False


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class TrainConfig:
    output_dir: str = str(ROOT / "outputs" / "runs")
    stage: str = "sft"  # ssl | sft
    num_epochs: float = 1.0
    max_steps: int = -1
    micro_batch_size: int = 1
    grad_accum: int = 8
    lr_plugin: float = 1e-4
    lr_lora: float = 2e-4
    replay_ratio: float = 0.0  # mix OPEN layout-description SFT to reduce short-answer overfit
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_seq_len: int = 3072
    max_new_tokens_eval: int = 64
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 200
    eval_every_steps: int = 0
    gradient_checkpointing: bool = True
    num_workers: int = 2
    resume: str | None = None
    max_train_samples: int = -1
    baseline: str = "webgap"  # webgap | lora | graphtoken | zeroshot


@dataclass
class DataConfig:
    webforge_dir: str = str(ROOT / "data" / "webforge")
    train_pages: int = 10000
    diag_pages: int = 2000
    image_width: int = 1280
    image_height: int = 800
    jpeg_quality: int = 85
    sft_samples: int = 3000
    ssl_pages: int = 8000
    public_eval_dir: str = str(ROOT / "data" / "benchmarks")
    websrc_samples: int = 1500
    include_adversarial: bool = True
    hard_pages: int = 800
    medium_pages: int = 400
    long_pages: int = 80
    sft_split: str = "train"
    mix_splits: str = ""  # comma-separated extra splits mixed into SFT (e.g. hard_train)


@dataclass
class EvalConfig:
    split: str = "heldout"
    max_samples: int = -1
    batch_size: int = 1
    greedy: bool = True
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    plugin: PluginConfig = field(default_factory=PluginConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "webgap"
    device: str = "cuda"

    @staticmethod
    def from_yaml(path: str | os.PathLike) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ExperimentConfig":
        def _merge(dc_cls, payload):
            if payload is None:
                return dc_cls()
            valid = {k: v for k, v in payload.items() if k in dc_cls.__dataclass_fields__}
            return dc_cls(**valid)

        cfg = ExperimentConfig(
            model=_merge(ModelConfig, raw.get("model")),
            plugin=_merge(PluginConfig, raw.get("plugin")),
            lora=_merge(LoRAConfig, raw.get("lora")),
            train=_merge(TrainConfig, raw.get("train")),
            data=_merge(DataConfig, raw.get("data")),
            eval=_merge(EvalConfig, raw.get("eval")),
        )
        if "run_name" in raw:
            cfg.run_name = raw["run_name"]
        if "device" in raw:
            cfg.device = raw["device"]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
