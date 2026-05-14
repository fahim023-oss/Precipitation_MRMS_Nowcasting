"""
Diffusion utilities for conditional DDPM nowcasting.

This module contains the noise schedule and forward diffusion process.

Notation
--------
x_0 : clean target sequence
x_t : noisy target sequence at diffusion step t
eps : Gaussian noise
"""

from __future__ import annotations

import torch


class DiffusionSchedule:
    """
    DDPM beta/alpha schedule.

    Parameters
    ----------
    timesteps : int
        Number of diffusion steps.
    beta_start : float
        Starting beta value.
    beta_end : float
        Ending beta value.
    device : str or torch.device
        Device where schedule tensors are stored.
    """

    def __init__(
        self,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.device = device

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            self.timesteps,
            device=device,
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance used during reverse sampling.
        self.posterior_variance = self.betas.clone()

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        """
        Move schedule tensors to a new device.
        """

        return DiffusionSchedule(
            timesteps=self.timesteps,
            beta_start=float(self.betas[0].detach().cpu()),
            beta_end=float(self.betas[-1].detach().cpu()),
            device=device,
        )


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """
    Extract timestep-specific values and reshape for broadcasting.

    Parameters
    ----------
    values : torch.Tensor
        Tensor with shape (T,).
    timesteps : torch.Tensor
        Tensor with shape (B,).
    target_shape : torch.Size
        Shape of target tensor, usually (B, C, T, H, W).

    Returns
    -------
    torch.Tensor
        Extracted values reshaped to (B, 1, 1, 1, 1).
    """

    batch_size = timesteps.shape[0]

    out = values.gather(0, timesteps)

    return out.reshape(
        batch_size,
        *((1,) * (len(target_shape) - 1)),
    )


def q_sample(
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion process.

    Adds Gaussian noise to clean target x_start at timestep t.

    Parameters
    ----------
    x_start : torch.Tensor
        Clean target sequence, shape (B, C, T, H, W).
    timesteps : torch.Tensor
        Diffusion timestep indices, shape (B,).
    schedule : DiffusionSchedule
        Diffusion schedule.
    noise : torch.Tensor or None
        Optional noise tensor. If None, random Gaussian noise is sampled.

    Returns
    -------
    x_noisy : torch.Tensor
        Noisy target sequence x_t.
    noise : torch.Tensor
        The Gaussian noise that was added.
    """

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar_t = extract(
        schedule.sqrt_alpha_bars,
        timesteps,
        x_start.shape,
    )

    sqrt_one_minus_alpha_bar_t = extract(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        x_start.shape,
    )

    x_noisy = (
        sqrt_alpha_bar_t * x_start
        + sqrt_one_minus_alpha_bar_t * noise
    )

    return x_noisy, noise


@torch.no_grad()
def p_sample(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    condition: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """
    One reverse diffusion step.

    Parameters
    ----------
    model : torch.nn.Module
        Noise-prediction model.
    x_t : torch.Tensor
        Current noisy target, shape (B, C, T, H, W).
    condition : torch.Tensor
        Conditioning input sequence, usually previous MRMS frames.
    timesteps : torch.Tensor
        Current diffusion step, shape (B,).
    schedule : DiffusionSchedule
        Diffusion schedule.

    Returns
    -------
    torch.Tensor
        Denoised sample x_{t-1}.
    """

    beta_t = extract(schedule.betas, timesteps, x_t.shape)
    sqrt_one_minus_alpha_bar_t = extract(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        x_t.shape,
    )
    sqrt_recip_alpha_t = extract(
        schedule.sqrt_recip_alphas,
        timesteps,
        x_t.shape,
    )

    predicted_noise = model(
        x_t=x_t,
        condition=condition,
        timesteps=timesteps,
    )

    model_mean = sqrt_recip_alpha_t * (
        x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t
    )

    # No noise is added at final step.
    if timesteps.min().item() == 0:
        return model_mean

    posterior_variance_t = extract(
        schedule.posterior_variance,
        timesteps,
        x_t.shape,
    )

    noise = torch.randn_like(x_t)

    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    condition: torch.Tensor,
    output_shape: tuple[int, int, int, int, int],
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """
    Generate one DDPM nowcast sample.

    Parameters
    ----------
    model : torch.nn.Module
        Noise-prediction model.
    condition : torch.Tensor
        Conditioning input sequence.
    output_shape : tuple
        Shape of generated target sequence: (B, C, T, H, W).
    schedule : DiffusionSchedule
        Diffusion schedule.

    Returns
    -------
    torch.Tensor
        Generated target sequence.
    """

    device = condition.device
    x_t = torch.randn(output_shape, device=device)

    for step in reversed(range(schedule.timesteps)):
        timesteps = torch.full(
            (output_shape[0],),
            step,
            device=device,
            dtype=torch.long,
        )

        x_t = p_sample(
            model=model,
            x_t=x_t,
            condition=condition,
            timesteps=timesteps,
            schedule=schedule,
        )

    return x_t
