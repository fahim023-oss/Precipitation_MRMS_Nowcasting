"""
Evaluation metrics for MRMS precipitation nowcasting.

Metrics are computed in physical precipitation space after inverse transform.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm


MetricDict = Dict[str, Dict]


def inverse_transform(
    x: torch.Tensor,
    rain_norm: float = 20.0,
    log_transform: bool = True,
) -> torch.Tensor:
    """
    Convert model-space precipitation back to mm/hr.
    """

    x = x * rain_norm

    if log_transform:
        x = torch.expm1(x)

    return x.clamp(min=0.0)


def default_lead_time_buckets(output_steps: int = 30) -> Dict[str, List[int]]:
    """
    Create lead-time buckets.
    """

    if output_steps < 30:
        return {
            "overall": list(range(output_steps)),
        }

    return {
        "t01-10": list(range(0, 10)),
        "t11-20": list(range(10, 20)),
        "t21-30": list(range(20, 30)),
        "overall": list(range(0, output_steps)),
    }


@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    output_steps: int = 30,
    thresholds: Iterable[float] = (0.5, 2.0, 5.0, 10.0),
    rain_norm: float = 20.0,
    log_transform: bool = True,
) -> MetricDict:
    """
    Compute RMSE, MAE, CSI, POD, and FAR.

    Parameters
    ----------
    model : torch.nn.Module
        Trained nowcasting model.
    loader : DataLoader
        Evaluation dataloader.
    device : str or torch.device
        Inference device.
    output_steps : int
        Number of forecast frames.
    thresholds : iterable of float
        Precipitation thresholds in mm/hr.
    rain_norm : float
        Normalization value used by dataset.
    log_transform : bool
        Whether dataset used log1p transform.
    """

    model.eval()

    thresholds = list(thresholds)
    buckets = default_lead_time_buckets(output_steps)

    squared_errors = {name: [] for name in buckets}
    absolute_errors = {name: [] for name in buckets}

    hits = {
        name: {threshold: 0.0 for threshold in thresholds}
        for name in buckets
    }
    misses = {
        name: {threshold: 0.0 for threshold in thresholds}
        for name in buckets
    }
    false_alarms = {
        name: {threshold: 0.0 for threshold in thresholds}
        for name in buckets
    }

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(
            x,
            future_steps=output_steps,
        )

        y_mm = inverse_transform(
            y,
            rain_norm=rain_norm,
            log_transform=log_transform,
        ).cpu().numpy()

        pred_mm = inverse_transform(
            prediction,
            rain_norm=rain_norm,
            log_transform=log_transform,
        ).cpu().numpy()

        for bucket_name, time_indices in buckets.items():
            y_bucket = y_mm[:, time_indices]
            pred_bucket = pred_mm[:, time_indices]

            squared_errors[bucket_name].append(
                ((y_bucket - pred_bucket) ** 2).mean()
            )
            absolute_errors[bucket_name].append(
                np.abs(y_bucket - pred_bucket).mean()
            )

            for threshold in thresholds:
                observed = y_bucket >= threshold
                forecast = pred_bucket >= threshold

                hits[bucket_name][threshold] += np.logical_and(
                    observed,
                    forecast,
                ).sum()

                misses[bucket_name][threshold] += np.logical_and(
                    observed,
                    np.logical_not(forecast),
                ).sum()

                false_alarms[bucket_name][threshold] += np.logical_and(
                    np.logical_not(observed),
                    forecast,
                ).sum()

    results: MetricDict = {}

    for bucket_name in buckets:
        rmse = math.sqrt(float(np.mean(squared_errors[bucket_name])))
        mae = float(np.mean(absolute_errors[bucket_name]))

        results[bucket_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "threshold_metrics": {},
        }

        for threshold in thresholds:
            h = hits[bucket_name][threshold]
            m = misses[bucket_name][threshold]
            f = false_alarms[bucket_name][threshold]

            csi = h / (h + m + f + 1e-9)
            pod = h / (h + m + 1e-9)
            far = f / (h + f + 1e-9)

            results[bucket_name]["threshold_metrics"][threshold] = {
                "CSI": float(csi),
                "POD": float(pod),
                "FAR": float(far),
            }

    return results


def format_metrics_table(
    results: MetricDict,
    thresholds: Iterable[float] = (0.5, 2.0, 5.0, 10.0),
) -> str:
    """
    Format metric dictionary as text table.
    """

    thresholds = list(thresholds)
    lines = []

    for bucket_name, values in results.items():
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Lead-time bucket: {bucket_name}")
        lines.append(f"RMSE : {values['RMSE']:.4f} mm/hr")
        lines.append(f"MAE  : {values['MAE']:.4f} mm/hr")
        lines.append("")
        lines.append(f"{'Threshold':>10}  {'CSI':>8}  {'POD':>8}  {'FAR':>8}")

        for threshold in thresholds:
            metric_values = values["threshold_metrics"][threshold]
            lines.append(
                f"{threshold:>8.1f}    "
                f"{metric_values['CSI']:>8.4f}  "
                f"{metric_values['POD']:>8.4f}  "
                f"{metric_values['FAR']:>8.4f}"
            )

    return "\n".join(lines)


def print_metrics(
    results: MetricDict,
    thresholds: Iterable[float] = (0.5, 2.0, 5.0, 10.0),
) -> None:
    """
    Print metric table.
    """

    print(format_metrics_table(results, thresholds=thresholds))
