#!/usr/bin/env python3
"""
Train ConvLSTM baseline for MRMS precipitation nowcasting.

Run from repository root:

    python scripts/train_convlstm.py --config configs/convlstm.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mrms_nowcasting.data import build_dataloaders
from mrms_nowcasting.metrics import compute_metrics, format_metrics_table
from mrms_nowcasting.convlstm.model import ConvLSTMNowcast, count_parameters
from mrms_nowcasting.convlstm.trainer import fit, load_checkpoint
from mrms_nowcasting.visualization import (
    plot_prediction_example,
    plot_training_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM baseline for MRMS nowcasting."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/convlstm.yaml",
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
    Resolve relative paths from the repository root.
    Absolute paths are left unchanged.
    """

    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    return str(REPO_ROOT / path_obj)


def threshold_to_model_space(
    threshold_mmhr: float,
    rain_norm: float,
    log_transform: bool,
) -> float:
    """
    Convert a physical precipitation threshold to model space.
    """

    value = float(threshold_mmhr)

    if log_transform:
        value = math.log1p(value)

    value = value / rain_norm

    return value


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

    train_loader, val_loader, test_loader, test_dataset = build_dataloaders(
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

    model = ConvLSTMNowcast(
        hidden1=int(config["model"]["hidden1"]),
        hidden2=int(config["model"]["hidden2"]),
        kernel_size=int(config["model"]["kernel_size"]),
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

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
    prediction_figure_path = resolve_path(
        config["outputs"]["prediction_figure_path"]
    )

    heavy_threshold = threshold_to_model_space(
        threshold_mmhr=float(config["loss"]["heavy_threshold_mmhr"]),
        rain_norm=rain_norm,
        log_transform=log_transform,
    )

    train_history, val_history, best_val_loss = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        checkpoint_path=checkpoint_path,
        output_steps=output_steps,
        heavy_threshold=heavy_threshold,
        heavy_weight=float(config["loss"]["heavy_weight"]),
        grad_clip=float(config["training"]["grad_clip"]),
        teacher_forcing_start=float(
            config["training"]["teacher_forcing_start"]
        ),
        teacher_forcing_end=float(
            config["training"]["teacher_forcing_end"]
        ),
        teacher_forcing_epochs=int(
            config["training"]["teacher_forcing_epochs"]
        ),
        config=config,
    )

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    plot_training_curve(
        train_history=train_history,
        val_history=val_history,
        output_path=training_curve_path,
    )

    # Load best checkpoint before test evaluation.
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    print("\n=== Test Evaluation ===")

    thresholds = list(config["evaluation"]["thresholds_mmhr"])

    results = compute_metrics(
        model=model,
        loader=test_loader,
        device=device,
        output_steps=output_steps,
        thresholds=thresholds,
        rain_norm=rain_norm,
        log_transform=log_transform,
    )

    metric_text = format_metrics_table(
        results=results,
        thresholds=thresholds,
    )

    print(metric_text)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(metric_text)

    print(f"\nMetrics saved -> {metrics_path}")

    plot_prediction_example(
        model=model,
        dataset=test_dataset,
        sample_index=0,
        device=device,
        output_path=prediction_figure_path,
        output_steps=output_steps,
        lead_times=config["plotting"]["lead_times"],
        rain_norm=rain_norm,
        log_transform=log_transform,
        title="ConvLSTM MRMS Precipitation Nowcasting",
    )


if __name__ == "__main__":
    main()
