"""Nested dataclass configuration. OmegaConf merges YAML files and `key.subkey=value` overrides onto the defaults,
converts types and rejects unknown keys."""
from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class ModelConfig:  # mirrors the keyword arguments of NanoTabICLv2
    embed_dim: int = 128
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    feature_group_size: int = 3
    n_cls_cols: int = 4
    n_cls_rows: int = 128
    ln_bias: bool = True


@dataclass
class DataConfig:
    task: str = "classification"  # "classification" or "regression"
    max_classes: int = 10  # classification: output dim, upper bound for the sampled number of classes
    n_quantiles: int = 999  # regression: output dim, quantile levels are linspace(0, 1, n_quantiles + 2)[1:-1]
    micro_batch_size: int = 4  # datasets per micro-batch; they share n_rows, n_train and n_features
    min_seq_len: int = 1024
    max_seq_len: int = 1024
    log_seq_len: bool = True  # sample n_rows log-uniformly instead of uniformly
    min_train_frac: float = 0.3  # n_train / n_rows is sampled uniformly from [min_train_frac, max_train_frac]
    max_train_frac: float = 0.9
    min_features: int = 1
    max_features: int = 100
    log_n_features: bool = False
    max_cat_size: int = 100  # maximum cardinality of categorical features
    filter_unpredictable: bool = True  # reject datasets on which ExtraTrees does not beat the mean predictor
    num_workers: int = -1  # CPU processes generating data, -1 = all cores, 0 = generate in the main process


@dataclass
class OptimConfig:
    max_steps: int = 500_000
    accum_steps: int = 16  # micro-batches per optimizer step (batch size = accum_steps * data.micro_batch_size)
    lr: float = 8e-4  # Muon step size, scaled to match the update RMS of AdamW (so on the AdamW scale)
    weight_decay: float = 0.01
    momentum: float = 0.9
    matched_adamw_rms: float = 0.2
    warmup_frac: float = 0.01  # linear warmup for this fraction of max_steps, then cosine decay to zero
    grad_clip: float = 10.0
    amp_dtype: str = "float32"  # autocast dtype: "float32" (off), "bfloat16" or "float16"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 0
    device: str = "auto"  # "auto" picks cuda if available, else cpu
    out_dir: str = "runs/default"  # checkpoints and metrics log; training resumes from out_dir/latest.pt if present
    init_from: str | None = None  # checkpoint whose model weights initialize a fresh run (e.g. the previous stage)
    save_every: int = 500
    log_every: int = 10
    wandb_project: str | None = None  # log to Weights & Biases if set


def to_config(*sources) -> Config:  # sources: dicts or OmegaConf configs, merged left to right onto the defaults
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), *sources))


def load_config(yaml_paths: list[str] = (), overrides: list[str] = ()) -> Config:
    # example: load_config(["configs/regression.yaml", "configs/stage2.yaml"], ["optim.lr=2e-4"])
    return to_config(*map(OmegaConf.load, yaml_paths), OmegaConf.from_dotlist(list(overrides)))
