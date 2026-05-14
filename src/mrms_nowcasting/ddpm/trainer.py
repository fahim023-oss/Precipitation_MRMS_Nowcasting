"""
Training utilities for conditional DDPM nowcasting.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrms_nowcasting.ddpm.diffusion import DiffusionSchedule, q_sample
from mrms_nowcasting.ddpm.losses import ddpm_training_loss


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    """
    Convert dataset layout from (B, T, C, H, W) to (B, C, T, H, W).
    """

    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")

    if x.shape[2] == 1 and x.shape[1] != 1:
        x = x.permute(0, 2, 1, 3, 4).contiguous()

    return x


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device | str,
    epoch: int,
    grad_clip: float = 1.0,
    lambda_noise: float = 1.0,
) -> float:
    """
    Train DDPM for one epoch.
    """

    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [DDPM train]")

    for condition, target in progress:
        condition = _to_channel_first(condition.to(device, non_blocking=True))
        target = _to_channel_first(target.to(device, non_blocking=True))

        batch_size = target.shape[0]

        timesteps = torch.randint(
            low=0,
            high=schedule.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_target, true_noise = q_sample(
            x_start=target,
            timesteps=timesteps,
            schedule=schedule,
        )

        predicted_noise = model(
            x_t=noisy_target,
            condition=condition,
            timesteps=timesteps,
        )

        loss = ddpm_training_loss(
            predicted_noise=predicted_noise,
            true_noise=true_noise,
            lambda_noise=lambda_noise,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device | str,
    lambda_noise: float = 1.0,
) -> float:
    """
    Compute validation noise-prediction loss.
    """

    model.eval()
    total_loss = 0.0

    for condition, target in loader:
        condition = _to_channel_first(condition.to(device, non_blocking=True))
        target = _to_channel_first(target.to(device, non_blocking=True))

        batch_size = target.shape[0]

        timesteps = torch.randint(
            low=0,
            high=schedule.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_target, true_noise = q_sample(
            x_start=target,
            timesteps=timesteps,
            schedule=schedule,
        )

        predicted_noise = model(
            x_t=noisy_target,
            condition=condition,
            timesteps=timesteps,
        )

        loss = ddpm_training_loss(
            predicted_noise=predicted_noise,
            true_noise=true_noise,
            lambda_noise=lambda_noise,
        )

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    config: Dict | None = None,
) -> None:
    """
    Save DDPM checkpoint.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Dict:
    """
    Load DDPM checkpoint.

    Supports both checkpoint dictionaries and raw model.state_dict().
    """

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    model.load_state_dict(checkpoint)

    return {
        "model_state_dict": checkpoint,
        "epoch": None,
        "best_val_loss": None,
        "config": None,
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    schedule: DiffusionSchedule,
    device: torch.device | str,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    grad_clip: float = 1.0,
    lambda_noise: float = 1.0,
    config: Dict | None = None,
) -> Tuple[List[float], List[float], float]:
    """
    Train DDPM with validation and early stopping.
    """

    best_val_loss = float("inf")
    wait = 0

    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            epoch=epoch,
            grad_clip=grad_clip,
            lambda_noise=lambda_noise,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            schedule=schedule,
            device=device,
            lambda_noise=lambda_noise,
        )

        if scheduler is not None:
            scheduler.step()

        train_history.append(train_loss)
        val_history.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )

            print(f"  ✓ Best DDPM checkpoint saved: {checkpoint_path}")
        else:
            wait += 1
            print(f"  No improvement: {wait}/{patience}")

            if wait >= patience:
                print("\nEarly stopping triggered.")
                break

    return train_history, val_history, best_val_loss
