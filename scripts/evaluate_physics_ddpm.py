#!/usr/bin/env python3
"""
Evaluate physics-guided DDPM nowcasting.

This script:
    1. Loads a trained DDPM checkpoint.
    2. Generates multiple DDPM realizations for each input sequence.
    3. Selects the best realization using physics-guided scoring.
    4. Computes RMSE, MAE, CSI, POD, and FAR.

Run from repository root:

    python scripts/evaluate_physics_ddpm.py --config configs/physics_ddpm.yaml
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
from tqdm import tqdm

# Allow running from repository root without installing package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mrms_nowcasting.data import build_dataloaders
from mrms_nowcasting.ddpm.diffusion import DiffusionSchedule
from mrms_nowcasting.ddpm.model import ConditionalUNet3D, count_parameters
from mrms_nowcasting.ddpm.trainer import load_checkpoint
from mrms_nowcasting.metrics import (
    default_lead_time_buckets,
    format_metrics_table,
    inverse_transform,
)
from mrms_nowcasting.physics_ddpm.inference import physics_guided_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate physics-guided DDPM for MRMS nowcasting."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/physics_ddpm.yaml",
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
    path_obj = Path(path)

    if path_obj.is_absolute():
        return str(path_obj)

    return str(REPO_ROOT / path_obj)


def update_metric_counts(
    results_accumulator: dict,
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    thresholds: list[float],
    output_steps: int,
) -> None:
    """
    Update RMSE/MAE and categorical metric accumulators.

    prediction_mm, target_mm shape:
        (B, C, T, H, W)
    """

    buckets = default_lead_time_buckets(output_steps)

    for bucket_name, time_indices in buckets.items():
        pred_bucket = prediction_mm[:, :, time_indices]
        target_bucket = target_mm[:, :, time_indices]

        results_accumulator["squared_errors"][bucket_name].append(
            ((pred_bucket - target_bucket) ** 2).mean()
        )
        results_accumulator["absolute_errors"][bucket_name].append(
            np.abs(pred_bucket - target_bucket).mean()
        )

        for threshold in thresholds:
            observed = target_bucket >= threshold
            forecast = pred_bucket >= threshold

            results_accumulator["hits"][bucket_name][threshold] += np.logical_and(
                observed,
                forecast,
            ).sum()

            results_accumulator["misses"][bucket_name][threshold] += np.logical_and(
                observed,
                np.logical_not(forecast),
            ).sum()

            results_accumulator["false_alarms"][bucket_name][threshold] += (
                np.logical_and(
                    np.logical_not(observed),
                    forecast,
                ).sum()
            )


def finalize_metrics(
    accumulator: dict,
    thresholds: list[float],
    output_steps: int,
) -> dict:
    """
    Convert accumulated counts/errors into final metric dictionary.
    """

    import math

    buckets = default_lead_time_buckets(output_steps)
    results = {}

    for bucket_name in buckets:
        rmse = math.sqrt(
            float(np.mean(accumulator["squared_errors"][bucket_name]))
        )
        mae = float(np.mean(accumulator["absolute_errors"][bucket_name]))

        results[bucket_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "threshold_metrics": {},
        }

        for threshold in thresholds:
            h = accumulator["hits"][bucket_name][threshold]
            m = accumulator["misses"][bucket_name][threshold]
            f = accumulator["false_alarms"][bucket_name][threshold]

            csi = h / (h + m + f + 1e-9)
            pod = h / (h + m + 1e-9)
            far = f / (h + f + 1e-9)

            results[bucket_name]["threshold_metrics"][threshold] = {
                "CSI": float(csi),
                "POD": float(pod),
                "FAR": float(far),
            }

    return results


def make_accumulator(thresholds: list[float], output_steps: int) -> dict:
    buckets = default_lead_time_buckets(output_steps)

    return {
        "squared_errors": {name: [] for name in buckets},
        "absolute_errors": {name: [] for name in buckets},
        "hits": {
            name: {threshold: 0.0 for threshold in thresholds}
            for name in buckets
        },
        "misses": {
            name: {threshold: 0.0 for threshold in thresholds}
            for name in buckets
        },
        "false_alarms": {
            name: {threshold: 0.0 for threshold in thresholds}
            for name in buckets
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    input_steps = int(config["data"]["input_steps"])
    output_steps = int(config["data"]["output_steps"])
    rain_norm = float(config["normalization"]["rain_norm"])
    log_transform = bool(config["normalization"]["log_transform"])

    data_folder = resolve_path(config["data"]["data_folder"])

    _, _, test_loader, _ = build_dataloaders(
        data_folder=data_folder,
        input_steps=input_steps,
        output_steps=output_steps,
        precip_key=str(config["data"]["precip_key"]),
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

    checkpoint_path = resolve_path(config["outputs"]["checkpoint_path"])

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")

    diffusion_schedule = DiffusionSchedule(
        timesteps=int(config["diffusion"]["timesteps"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
        device=device,
    )

    thresholds = list(config["evaluation"]["thresholds_mmhr"])
    accumulator = make_accumulator(
        thresholds=thresholds,
        output_steps=output_steps,
    )

    max_batches = config["evaluation"].get("max_batches", None)

    for batch_index, (condition, target) in enumerate(
        tqdm(test_loader, desc="Physics-guided DDPM evaluation")
    ):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        best_prediction, best_indices, scores = physics_guided_sample(
            model=model,
            condition=condition,
            schedule=diffusion_schedule,
            output_steps=output_steps,
            num_realizations=int(config["physics_guidance"]["num_realizations"]),
            mass_weight=float(config["physics_guidance"]["mass_weight"]),
            temporal_weight=float(config["physics_guidance"]["temporal_weight"]),
            spatial_weight=float(config["physics_guidance"]["spatial_weight"]),
            persistence_weight=float(
                config["physics_guidance"]["persistence_weight"]
            ),
        )

        # target is dataset layout (B, T, C, H, W), convert to channel-first.
        if target.shape[2] == 1 and target.shape[1] != 1:
            target = target.permute(0, 2, 1, 3, 4).contiguous()

        prediction_mm = inverse_transform(
            best_prediction,
            rain_norm=rain_norm,
            log_transform=log_transform,
        ).cpu().numpy()

        target_mm = inverse_transform(
            target,
            rain_norm=rain_norm,
            log_transform=log_transform,
        ).cpu().numpy()

        update_metric_counts(
            results_accumulator=accumulator,
            prediction_mm=prediction_mm,
            target_mm=target_mm,
            thresholds=thresholds,
            output_steps=output_steps,
        )

    results = finalize_metrics(
        accumulator=accumulator,
        thresholds=thresholds,
        output_steps=output_steps,
    )

    metric_text = format_metrics_table(
        results=results,
        thresholds=thresholds,
    )

    print(metric_text)

    metrics_path = resolve_path(config["outputs"]["metrics_path"])
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(metric_text)

    print(f"\nMetrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
