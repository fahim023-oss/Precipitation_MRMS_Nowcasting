"""
Loss functions for ConvLSTM precipitation nowcasting.
"""

from __future__ import annotations

import torch


def nowcast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    heavy_threshold: float,
    heavy_weight: float,
) -> torch.Tensor:
    """
    Weighted nowcasting loss in transformed precipitation space.

    The loss combines:
        0.5 * L1 + 0.5 * MSE + heavy-rain weighted L1

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted sequence, shape (B, T, 1, H, W).
    target : torch.Tensor
        Target sequence, shape (B, T, 1, H, W).
    heavy_threshold : float
        Heavy-rain threshold in the same transformed space as target.
    heavy_weight : float
        Extra weight for heavy-rain pixels.
    """

    absolute_error = torch.abs(prediction - target)
    squared_error = (prediction - target) ** 2

    base_loss = 0.5 * absolute_error.mean() + 0.5 * squared_error.mean()

    heavy_mask = (target > heavy_threshold).float()
    heavy_count = heavy_mask.sum().clamp(min=1.0)

    heavy_loss = (
        heavy_weight * absolute_error * heavy_mask
    ).sum() / heavy_count

    return base_loss + heavy_loss


def teacher_forcing_ratio(
    epoch: int,
    start_ratio: float = 1.0,
    end_ratio: float = 0.0,
    decay_epochs: int = 20,
) -> float:
    """
    Linearly decay teacher-forcing ratio.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    start_ratio : float
        Initial teacher-forcing ratio.
    end_ratio : float
        Final teacher-forcing ratio.
    decay_epochs : int
        Number of epochs over which the ratio decays.
    """

    if decay_epochs <= 0:
        return end_ratio

    if epoch >= decay_epochs:
        return end_ratio

    progress = epoch / decay_epochs
    ratio = start_ratio + progress * (end_ratio - start_ratio)

    return float(ratio)
