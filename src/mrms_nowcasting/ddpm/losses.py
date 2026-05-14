"""
Loss functions for conditional DDPM precipitation nowcasting.

The core DDPM objective is noise prediction:

    L_noise = MSE(predicted_noise, true_noise)

Additional optional losses can be used to improve precipitation structure:
    - reconstruction loss
    - heavy-rain weighted loss
    - dry-region penalty
    - smoothness penalty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def noise_prediction_loss(
    predicted_noise: torch.Tensor,
    true_noise: torch.Tensor,
) -> torch.Tensor:
    """
    Standard DDPM noise-prediction loss.
    """

    return F.mse_loss(predicted_noise, true_noise)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L1 reconstruction loss.
    """

    return F.l1_loss(prediction, target)


def heavy_rain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Extra L1 loss over heavy-rain pixels.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted precipitation sequence.
    target : torch.Tensor
        Target precipitation sequence.
    threshold : float
        Heavy-rain threshold in model space.
    weight : float
        Multiplicative weight.
    """

    mask = (target > threshold).float()
    count = mask.sum().clamp(min=1.0)

    loss = torch.abs(prediction - target) * mask
    return weight * loss.sum() / count


def dry_region_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Penalize false precipitation in dry target regions.
    """

    dry_mask = (target <= threshold).float()
    count = dry_mask.sum().clamp(min=1.0)

    loss = torch.abs(prediction) * dry_mask
    return loss.sum() / count


def temporal_smoothness_loss(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize abrupt temporal changes.

    Expected shape:
        (B, C, T, H, W)
    """

    if prediction.shape[2] < 2:
        return prediction.new_tensor(0.0)

    diff = prediction[:, :, 1:] - prediction[:, :, :-1]
    return torch.mean(torch.abs(diff))


def spatial_smoothness_loss(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize noisy spatial gradients.

    Expected shape:
        (B, C, T, H, W)
    """

    grad_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    grad_x = prediction[..., :, 1:] - prediction[..., :, :-1]

    return torch.mean(torch.abs(grad_y)) + torch.mean(torch.abs(grad_x))


def ddpm_training_loss(
    predicted_noise: torch.Tensor,
    true_noise: torch.Tensor,
    lambda_noise: float = 1.0,
) -> torch.Tensor:
    """
    Main DDPM training loss.

    This intentionally keeps the training objective simple and stable:
    the model predicts the Gaussian noise added during q_sample().
    """

    return lambda_noise * noise_prediction_loss(
        predicted_noise=predicted_noise,
        true_noise=true_noise,
    )
