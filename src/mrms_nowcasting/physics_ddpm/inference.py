"""
Physics-guided DDPM inference utilities.

This module generates multiple DDPM realizations and selects the one with the
lowest physics-guided score.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from mrms_nowcasting.ddpm.diffusion import DiffusionSchedule, sample_ddpm
from mrms_nowcasting.physics_ddpm.scoring import physics_score


def to_channel_first(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor layout from (B, T, C, H, W) to (B, C, T, H, W).

    If the tensor is already channel-first, it is returned unchanged.
    """

    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")

    if x.shape[2] == 1 and x.shape[1] != 1:
        x = x.permute(0, 2, 1, 3, 4).contiguous()

    return x


@torch.no_grad()
def generate_realizations(
    model: torch.nn.Module,
    condition: torch.Tensor,
    schedule: DiffusionSchedule,
    output_steps: int = 30,
    num_realizations: int = 20,
) -> torch.Tensor:
    """
    Generate multiple DDPM realizations.

    Parameters
    ----------
    model : torch.nn.Module
        Trained DDPM noise-prediction model.
    condition : torch.Tensor
        Input sequence, shape (B, C, T_in, H, W) or (B, T_in, C, H, W).
    schedule : DiffusionSchedule
        DDPM diffusion schedule.
    output_steps : int
        Number of future frames to generate.
    num_realizations : int
        Number of stochastic DDPM samples.

    Returns
    -------
    torch.Tensor
        Realizations with shape (R, B, C, T_out, H, W).
    """

    model.eval()

    condition = to_channel_first(condition)
    batch_size, channels, _, height, width = condition.shape

    output_shape = (
        batch_size,
        channels,
        output_steps,
        height,
        width,
    )

    realizations = []

    for _ in range(num_realizations):
        sample = sample_ddpm(
            model=model,
            condition=condition,
            output_shape=output_shape,
            schedule=schedule,
        )
        realizations.append(sample)

    return torch.stack(realizations, dim=0)


@torch.no_grad()
def select_best_realization(
    realizations: torch.Tensor,
    condition: torch.Tensor,
    mass_weight: float = 1.0,
    temporal_weight: float = 0.2,
    spatial_weight: float = 0.2,
    persistence_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Select the best realization using physics-guided scores.

    Parameters
    ----------
    realizations : torch.Tensor
        Generated samples, shape (R, B, C, T_out, H, W).
    condition : torch.Tensor
        Input sequence, shape (B, C, T_in, H, W) or (B, T_in, C, H, W).
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
    best : torch.Tensor
        Best generated sequence, shape (B, C, T_out, H, W).
    best_indices : torch.Tensor
        Best realization index for each batch item, shape (B,).
    component_scores : dict
        Score components for all realizations.
    """

    condition = to_channel_first(condition)

    if realizations.ndim != 6:
        raise ValueError(
            "Expected realizations with shape (R, B, C, T, H, W), "
            f"got {tuple(realizations.shape)}"
        )

    num_realizations, batch_size = realizations.shape[:2]

    total_scores = []
    all_components: Dict[str, list[torch.Tensor]] = {
        "mass": [],
        "temporal": [],
        "spatial": [],
        "persistence": [],
        "total": [],
    }

    for realization_index in range(num_realizations):
        prediction = realizations[realization_index]

        total, components = physics_score(
            prediction=prediction,
            condition=condition,
            mass_weight=mass_weight,
            temporal_weight=temporal_weight,
            spatial_weight=spatial_weight,
            persistence_weight=persistence_weight,
        )

        total_scores.append(total)

        for key, value in components.items():
            all_components[key].append(value)

    score_matrix = torch.stack(total_scores, dim=0)  # (R, B)
    best_indices = torch.argmin(score_matrix, dim=0)  # (B,)

    best = []

    for batch_index in range(batch_size):
        best.append(realizations[best_indices[batch_index], batch_index])

    best = torch.stack(best, dim=0)

    component_scores = {
        key: torch.stack(value, dim=0)
        for key, value in all_components.items()
    }

    return best, best_indices, component_scores


@torch.no_grad()
def physics_guided_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    schedule: DiffusionSchedule,
    output_steps: int = 30,
    num_realizations: int = 20,
    mass_weight: float = 1.0,
    temporal_weight: float = 0.2,
    spatial_weight: float = 0.2,
    persistence_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generate multiple DDPM realizations and return the physics-selected one.

    Returns
    -------
    best : torch.Tensor
        Physics-selected nowcast, shape (B, C, T_out, H, W).
    best_indices : torch.Tensor
        Selected realization index for each batch item.
    scores : dict
        Physics-guided scores for all realizations.
    """

    condition = to_channel_first(condition)

    realizations = generate_realizations(
        model=model,
        condition=condition,
        schedule=schedule,
        output_steps=output_steps,
        num_realizations=num_realizations,
    )

    best, best_indices, scores = select_best_realization(
        realizations=realizations,
        condition=condition,
        mass_weight=mass_weight,
        temporal_weight=temporal_weight,
        spatial_weight=spatial_weight,
        persistence_weight=persistence_weight,
    )

    return best, best_indices, scores
