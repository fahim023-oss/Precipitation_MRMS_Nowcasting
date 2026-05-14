#!/usr/bin/env python3
"""
Train conditional DDPM for MRMS precipitation nowcasting.

Run from repository root:

    python scripts/train_ddpm.py --config configs/ddpm.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mrms_nowcasting.data import build_dataloaders
from mrms_nowcasting.ddpm.diffusion import DiffusionSchedule
from mrms_nowcasting.ddpm.model import ConditionalUNet3D, count_parameters
from mrms_nowcasting.ddpm.trainer import fit, load_checkpoint
from mrms_nowcasting.metrics import compute_metrics, format_metrics_table
from mrms_nowcasting.visualization import plot_training_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conditional DDPM for MRMS nowcasting."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpm.yaml",
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> str:
    """
    Resolve relative paths from repository root.
    Absolute paths are left unchanged.
    """

    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    return str(REPO_ROOT / path_obj)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_folder = resolve_path(config["data"]["data_folder"])

    input_steps = int(config["data"]["input_steps"])
    output_steps = int(config["data"]["output_steps"])
    precip_key = str(config["data"]["precip_key"])

    rain_norm = float(config["normalization"]["rain_norm"])
    log_transform = bool(config["normalization"]["log_transform"])

    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_folder=data_folder,
        input_steps=input_steps,
        output_steps=output_steps,
        precip_key=precip_key,
        rain_norm=rain_norm,
        log_transform=log_transform,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        batch_size=int(config["loader"]["batch_size"]),
        num_workers=int(config["loader"]["num_workers"]),
        pin_memory=bool(config["loader"]["pin_memory"]),
    )

    model = ConditionalUNet3D(
        in_channels=int(config["model"]["in_channels"]),
        condition_channels=int(config["model"]["condition_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
        time_embedding_dim=int(config["model"]["time_embedding_dim"]),
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    diffusion_schedule = DiffusionSchedule(
        timesteps=int(config["diffusion"]["timesteps"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"]["eta_min"]),
    )

    checkpoint_path = resolve_path(config["outputs"]["checkpoint_path"])
    metrics_path = resolve_path(config["outputs"]["metrics_path"])
    training_curve_path = resolve_path(config["outputs"]["training_curve_path"])

    train_history, val_history, best_val_loss = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        schedule=diffusion_schedule,
        device=device,
        epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        checkpoint_path=checkpoint_path,
        grad_clip=float(config["training"]["grad_clip"]),
        lambda_noise=float(config["loss"]["lambda_noise"]),
        config=config,
    )

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    plot_training_curve(
        train_history=train_history,
        val_history=val_history,
        output_path=training_curve_path,
    )

    # Load best checkpoint before reporting validation/test noise-prediction metrics.
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    print("\nDDPM training finished.")
    print("Note: DDPM direct RMSE/CSI evaluation requires sampling.")
    print("Sampling/physics-guided evaluation will be handled by a separate script.")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Best validation DDPM noise loss: {best_val_loss:.6f}\n")

    print(f"Summary saved -> {metrics_path}")


if __name__ == "__main__":
    main()
