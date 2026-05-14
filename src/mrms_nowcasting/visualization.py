"""
Visualization utilities for MRMS precipitation nowcasting.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import matplotlib

# Safe for HPC / Codespaces / non-interactive runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from mrms_nowcasting.metrics import inverse_transform


def make_precip_colormap() -> Tuple[ListedColormap, BoundaryNorm, List[float]]:
    """
    Create a discrete precipitation colormap.
    """

    colors = [
        "white",
        "#c0e8c0",
        "#00a600",
        "#f0f000",
        "#e07000",
        "#e00000",
        "#c000c0",
        "#7030c0",
    ]

    bounds = [0, 0.1, 1, 5, 10, 20, 40, 70, 150]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


def plot_training_curve(
    train_history: Iterable[float],
    val_history: Iterable[float],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Plot training and validation loss curves.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(train_history), label="Train")
    plt.plot(list(val_history), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Training curve saved -> {output_path}")


@torch.no_grad()
def plot_prediction_example(
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    device: torch.device | str,
    output_path: str,
    output_steps: int = 30,
    lead_times: Iterable[int] = (0, 9, 19, 29),
    rain_norm: float = 20.0,
    log_transform: bool = True,
    title: str = "MRMS Precipitation Nowcasting",
    dpi: int = 150,
) -> None:
    """
    Plot qualitative nowcasting example.

    Layout:
        column 1 : last observed frame
        column 2 : ground truth
        column 3 : prediction
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.eval()

    cmap, norm, bounds = make_precip_colormap()

    x, y = dataset[sample_index]
    x_batch = x.unsqueeze(0).to(device)

    prediction = model(
        x_batch,
        future_steps=output_steps,
    )

    x_mm = inverse_transform(
        x,
        rain_norm=rain_norm,
        log_transform=log_transform,
    ).squeeze().cpu().numpy()

    y_mm = inverse_transform(
        y,
        rain_norm=rain_norm,
        log_transform=log_transform,
    ).squeeze().cpu().numpy()

    pred_mm = inverse_transform(
        prediction.squeeze(0),
        rain_norm=rain_norm,
        log_transform=log_transform,
    ).squeeze().cpu().numpy()

    lead_times = list(lead_times)

    fig, axes = plt.subplots(
        len(lead_times),
        3,
        figsize=(12, 3.7 * len(lead_times)),
    )

    if len(lead_times) == 1:
        axes = axes.reshape(1, 3)

    image_handle = None

    for row, lead_idx in enumerate(lead_times):
        if lead_idx >= y_mm.shape[0]:
            raise IndexError(
                f"Requested lead index {lead_idx}, "
                f"but target sequence has only {y_mm.shape[0]} frames."
            )

        lead_label = f"t+{lead_idx + 1} ({(lead_idx + 1) * 2} min)"

        if row == 0:
            image_handle = axes[row, 0].imshow(
                x_mm[-1],
                origin="lower",
                cmap=cmap,
                norm=norm,
            )
            axes[row, 0].set_title(
                "Last Observed Frame",
                fontsize=11,
                fontweight="bold",
            )
        else:
            axes[row, 0].axis("off")

        axes[row, 1].imshow(
            y_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 1].set_title(
            f"Ground Truth {lead_label}",
            fontsize=10,
        )

        image_handle = axes[row, 2].imshow(
            pred_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 2].set_title(
            f"Prediction {lead_label}",
            fontsize=10,
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    colorbar = fig.colorbar(
        image_handle,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        pad=0.03,
        fraction=0.03,
        aspect=60,
        ticks=bounds,
        extend="max",
    )

    colorbar.set_label("Precipitation (mm/hr)", fontsize=12)

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Prediction figure saved -> {output_path}")
