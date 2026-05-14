"""
Training utilities for the ConvLSTM baseline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrms_nowcasting.convlstm.losses import (
    nowcast_loss,
    teacher_forcing_ratio,
)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int,
    output_steps: int,
    heavy_threshold: float,
    heavy_weight: float,
    grad_clip: float = 1.0,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.0,
    teacher_forcing_epochs: int = 20,
) -> float:
    """
    Train the ConvLSTM model for one epoch.
    """

    model.train()

    tf_ratio = teacher_forcing_ratio(
        epoch=epoch,
        start_ratio=teacher_forcing_start,
        end_ratio=teacher_forcing_end,
        decay_epochs=teacher_forcing_epochs,
    )

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [train] tf={tf_ratio:.2f}",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(
            x,
            y=y,
            teacher_forcing_ratio=tf_ratio,
            future_steps=output_steps,
        )

        loss = nowcast_loss(
            prediction=prediction,
            target=y,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
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
    device: torch.device | str,
    output_steps: int,
    heavy_threshold: float,
    heavy_weight: float,
) -> float:
    """
    Compute validation loss.
    """

    model.eval()

    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(
            x,
            future_steps=output_steps,
        )

        loss = nowcast_loss(
            prediction=prediction,
            target=y,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
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
    Save a training checkpoint.
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
    Load a checkpoint.

    Supports both:
        1. New dictionary checkpoints.
        2. Old raw model.state_dict() checkpoints.
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
    device: torch.device | str,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    output_steps: int,
    heavy_threshold: float,
    heavy_weight: float,
    grad_clip: float = 1.0,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.0,
    teacher_forcing_epochs: int = 20,
    config: Dict | None = None,
) -> Tuple[List[float], List[float], float]:
    """
    Train with validation and early stopping.
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
            device=device,
            epoch=epoch,
            output_steps=output_steps,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
            grad_clip=grad_clip,
            teacher_forcing_start=teacher_forcing_start,
            teacher_forcing_end=teacher_forcing_end,
            teacher_forcing_epochs=teacher_forcing_epochs,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            device=device,
            output_steps=output_steps,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
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

            print(f"  ✓ Best checkpoint saved: {checkpoint_path}")
        else:
            wait += 1
            print(f"  No improvement: {wait}/{patience}")

            if wait >= patience:
                print("\nEarly stopping triggered.")
                break

    return train_history, val_history, best_val_loss
