"""
Physics-guided scoring utilities for DDPM nowcasting.

These utilities score generated DDPM realizations using simple physical
consistency terms:

    1. Mass consistency
    2. Temporal smoothness
    3. Spatial smoothness
    4. Persistence/advection-style consistency

The lowest total score is selected as the best realization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def temporal_smoothness_score(sequence: torch.Tensor) -> torch.Tensor:
    """
    Score abrupt temporal changes.

    Parameters
    ----------
    sequence : torch.Tensor
        Sequence with shape (B, C, T, H, W).

    Returns
    -------
    torch.Tensor
        Score per batch item, shape (B,).
    """

    if sequence.shape[2] < 2:
        return torch.zeros(sequence.shape[0], device=sequence.device)

    diff = sequence[:, :, 1:] - sequence[:, :, :-1]
    return diff.abs().mean(dim=(1, 2, 3, 4))


def spatial_smoothness_score(sequence: torch.Tensor) -> torch.Tensor:
    """
    Score noisy spatial gradients.

    Parameters
    ----------
    sequence : torch.Tensor
        Sequence with shape (B, C, T, H, W).

    Returns
    -------
    torch.Tensor
        Score per batch item, shape (B,).
    """

    grad_y = sequence[..., 1:, :] - sequence[..., :-1, :]
    grad_x = sequence[..., :, 1:] - sequence[..., :, :-1]

    score_y = grad_y.abs().mean(dim=(1, 2, 3, 4))
    score_x = grad_x.abs().mean(dim=(1, 2, 3, 4))

    return score_y + score_x


def mass_consistency_score(
    prediction: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """
    Score inconsistency between the final observed mass and predicted mean mass.

    This is a lightweight physical prior:
        the future precipitation field should not have an unrealistically large
        domain-total jump relative to the last observed frame.

    Parameters
    ----------
    prediction : torch.Tensor
        Generated future sequence, shape (B, C, T_out, H, W).
    condition : torch.Tensor
        Input sequence, shape (B, C, T_in, H, W).

    Returns
    -------
    torch.Tensor
        Score per batch item, shape (B,).
    """

    last_observed = condition[:, :, -1]
    future_mean = prediction.mean(dim=2)

    last_mass = last_observed.mean(dim=(1, 2, 3))
    future_mass = future_mean.mean(dim=(1, 2, 3))

    return torch.abs(future_mass - last_mass)


def persistence_score(
    prediction: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """
    Score departure from a persistence baseline.

    Persistence baseline:
        future fields are compared to the last observed frame.

    Parameters
    ----------
    prediction : torch.Tensor
        Generated future sequence, shape (B, C, T_out, H, W).
    condition : torch.Tensor
        Input sequence, shape (B, C, T_in, H, W).

    Returns
    -------
    torch.Tensor
        Score per batch item, shape (B,).
    """

    last_observed = condition[:, :, -1:]
    persistence = last_observed.expand_as(prediction)

    return torch.abs(prediction - persistence).mean(dim=(1, 2, 3, 4))


def physics_score(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    mass_weight: float = 1.0,
    temporal_weight: float = 0.2,
    spatial_weight: float = 0.2,
    persistence_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute weighted physics-guided score.

    Lower is better.

    Parameters
    ----------
    prediction : torch.Tensor
        Generated future sequence, shape (B, C, T_out, H, W).
    condition : torch.Tensor
        Input sequence, shape (B, C, T_in, H, W).
    mass_weight : float
        Weight for mass consistency.
    temporal_weight : float
        Weight for temporal smoothness.
    spatial_weight : float
        Weight for spatial smoothness.
    persistence_weight : float
        Weight for persistence consistency.

    Returns
    -------
    total_score : torch.Tensor
        Total score per batch item, shape (B,).
    components : dict
        Individual component scores.
    """

    mass = mass_consistency_score(
        prediction=prediction,
        condition=condition,
    )

    temporal = temporal_smoothness_score(prediction)
    spatial = spatial_smoothness_score(prediction)

    persistence = persistence_score(
        prediction=prediction,
        condition=condition,
    )

    total = (
        mass_weight * mass
        + temporal_weight * temporal
        + spatial_weight * spatial
        + persistence_weight * persistence
    )

    components = {
        "mass": mass,
        "temporal": temporal,
        "spatial": spatial,
        "persistence": persistence,
        "total": total,
    }

    return total, components
