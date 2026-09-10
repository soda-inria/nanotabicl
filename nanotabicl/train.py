"""Training loop: gradient accumulation over micro-batches, AMP, warmup + cosine schedule, checkpointing, logging."""
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .config import Config, OptimConfig, load_config
from .data import iter_batches
from .model import NanoTabICLv2
from .muon import Muon
from .runtime import build_model, resolve_device


def cross_entropy_loss(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # pred: (n_batch, n_test, max_classes), y: (n_batch, n_test) class labels stored as floats
    loss = F.cross_entropy(pred.flatten(0, 1), y.flatten().long())
    return loss, {"accuracy": (pred.argmax(dim=-1) == y).float().mean()}

def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # pred: (n_batch, n_test, n_quantiles) at levels linspace(0, 1, n_quantiles + 2)[1:-1], y: (n_batch, n_test)
    alphas = torch.linspace(0, 1, pred.shape[-1] + 2, device=pred.device)[1:-1]
    errors = y[..., None] - pred
    loss = torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
    return loss, {"median_mse": (pred.detach()[..., pred.shape[-1] // 2] - y).square().mean()}


def lr_schedule(step: int, cfg: OptimConfig) -> float:  # linear warmup, then cosine decay to zero
    warmup = cfg.warmup_frac * cfg.max_steps
    if step < warmup:
        return cfg.lr * step / max(1.0, warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1.0, cfg.max_steps - warmup)))


def train(cfg: Config) -> NanoTabICLv2:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    torch.set_float32_matmul_precision("high")  # allow TF32 matmuls on GPUs
    os.makedirs(cfg.out_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), os.path.join(cfg.out_dir, "config.yaml"))

    model = build_model(cfg).to(device)
    optimizer = Muon(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
                     momentum=cfg.optim.momentum, matched_adamw_rms=cfg.optim.matched_adamw_rms)
    step = 0

    ckpt_path = os.path.join(cfg.out_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resuming from {ckpt_path} at step {step}")
    elif cfg.init_from:  # start a new run (e.g. the next curriculum stage) from the weights of another checkpoint
        model.load_state_dict(torch.load(cfg.init_from, map_location=device, weights_only=False)["model"])
        print(f"Initialized model weights from {cfg.init_from}")

    amp_dtype = getattr(torch, cfg.optim.amp_dtype)
    autocast = torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)  # bf16 needs no loss scaling
    loss_fn = pinball_loss if cfg.data.task == "regression" else cross_entropy_loss
    loader = iter_batches(cfg.data, seed=cfg.seed + step)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device={device}, "
          f"batch size={cfg.optim.accum_steps * cfg.data.micro_batch_size}", flush=True)

    if cfg.wandb_project:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), dir=cfg.out_dir, resume="allow")

    model.train()
    while step < cfg.optim.max_steps:
        t_start, t_data, metrics = time.time(), 0.0, {}
        lr = optimizer.param_groups[0]["lr"] = lr_schedule(step, cfg.optim)
        for _ in range(cfg.optim.accum_steps):
            t0 = time.time()
            x, y, n_train = next(loader)
            t_data += time.time() - t0
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                pred = model(x, y[:, :n_train])
            loss, extra = loss_fn(pred.float(), y[:, n_train:])
            scaler.scale(loss / cfg.optim.accum_steps).backward()
            for name, value in {"loss": loss.detach(), **extra}.items():  # stays on device, no sync per micro-batch
                metrics[name] = metrics.get(name, 0.0) + value / cfg.optim.accum_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip or float("inf"))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == cfg.optim.max_steps:
            metrics = {name: value.item() for name, value in metrics.items()}
            metrics.update(step=step, lr=lr, grad_norm=grad_norm.item(),
                           step_time=time.time() - t_start, data_wait=t_data)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)
            with open(os.path.join(cfg.out_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if cfg.wandb_project:
                wandb.log(metrics, step=step)
        if step > 0 and step % cfg.save_every == 0 or step == cfg.optim.max_steps:
            if step > cfg.save_every:
                os.rename(ckpt_path, os.path.join(cfg.out_dir, "previous.pt"))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "config": asdict(cfg)}, ckpt_path)
    return model


if __name__ == "__main__":  # usage: python -m nanotabicl.train [config.yaml ...] [key.subkey=value ...]
    yaml_paths = [arg for arg in sys.argv[1:] if "=" not in arg]
    train(load_config(yaml_paths, [arg for arg in sys.argv[1:] if "=" in arg]))
