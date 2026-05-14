"""
Conditional 3D U-Net for DDPM precipitation nowcasting.

The DDPM model predicts Gaussian noise added to the target future sequence.

Expected tensor layout:
    x_t       : (B, 1, T_out, H, W)
    condition : (B, 1, T_in,  H, W) or (B, T_in, 1, H, W)
    timesteps : (B,)

The model internally resizes the conditioning sequence along the time dimension
so that it can be concatenated with x_t.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.

    Parameters
    ----------
    timesteps : torch.Tensor
        Diffusion timesteps, shape (B,).
    embedding_dim : int
        Embedding dimension.

    Returns
    -------
    torch.Tensor
        Time embedding, shape (B, embedding_dim).
    """

    half_dim = embedding_dim // 2
    device = timesteps.device

    scale = math.log(10000) / max(half_dim - 1, 1)

    frequencies = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32) * -scale
    )

    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

    embedding = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1,
    )

    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class TimeMLP(nn.Module):
    """
    MLP for timestep embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_time_embedding(
            timesteps=timesteps,
            embedding_dim=self.embedding_dim,
        )
        return self.net(emb)


class ConvBlock3D(nn.Module):
    """
    3D convolution block with optional time conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

        if time_dim is not None:
            self.time_proj = nn.Linear(time_dim, out_channels)
        else:
            self.time_proj = None

        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        if self.time_proj is not None and time_embedding is not None:
            time_bias = self.time_proj(time_embedding)
            time_bias = time_bias[:, :, None, None, None]
            h = h + time_bias

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h


class DownBlock3D(nn.Module):
    """
    Downsampling block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.block = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=time_dim,
        )

        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x, time_embedding)
        down = self.downsample(h)
        return h, down


class UpBlock3D(nn.Module):
    """
    Upsampling block with skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        self.block = ConvBlock3D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            time_dim=time_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)

        # In case odd spatial sizes create one-pixel mismatch.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.block(x, time_embedding)


class ConditionalUNet3D(nn.Module):
    """
    Conditional 3D U-Net for DDPM noise prediction.

    Parameters
    ----------
    in_channels : int
        Channels of noisy target sequence. Usually 1.
    condition_channels : int
        Channels of conditioning sequence. Usually 1.
    out_channels : int
        Output channels. Usually 1 because model predicts noise.
    base_channels : int
        Base channel width.
    time_embedding_dim : int
        Sinusoidal timestep embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        condition_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        time_hidden_dim = 4 * base_channels

        self.time_mlp = TimeMLP(
            embedding_dim=time_embedding_dim,
            hidden_dim=time_hidden_dim,
        )

        total_input_channels = in_channels + condition_channels

        self.input_block = ConvBlock3D(
            in_channels=total_input_channels,
            out_channels=base_channels,
            time_dim=time_hidden_dim,
        )

        self.down1 = DownBlock3D(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            time_dim=time_hidden_dim,
        )

        self.down2 = DownBlock3D(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            time_dim=time_hidden_dim,
        )

        self.middle = ConvBlock3D(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            time_dim=time_hidden_dim,
        )

        self.up2 = UpBlock3D(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_dim=time_hidden_dim,
        )

        self.up1 = UpBlock3D(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
            time_dim=time_hidden_dim,
        )

        self.output_head = nn.Sequential(
            nn.Conv3d(base_channels + base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, T, C, H, W) to (B, C, T, H, W) if needed.
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D tensor, got shape {tuple(x.shape)}"
            )

        # Dataset returns (B, T, 1, H, W). Convert to channel-first.
        if x.shape[2] == 1 and x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x

    @staticmethod
    def _match_condition_time(
        condition: torch.Tensor,
        target_time: int,
    ) -> torch.Tensor:
        """
        Resize condition sequence to match target time dimension.
        """

        if condition.shape[2] == target_time:
            return condition

        return F.interpolate(
            condition,
            size=(target_time, condition.shape[-2], condition.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy target sequence, shape (B, 1, T_out, H, W).
        condition : torch.Tensor
            Conditioning input sequence.
        timesteps : torch.Tensor
            Diffusion timesteps, shape (B,).

        Returns
        -------
        torch.Tensor
            Predicted noise, same shape as x_t.
        """

        x_t = self._ensure_channel_first(x_t)
        condition = self._ensure_channel_first(condition)
        condition = self._match_condition_time(
            condition=condition,
            target_time=x_t.shape[2],
        )

        time_embedding = self.time_mlp(timesteps)

        x = torch.cat([x_t, condition], dim=1)

        x0 = self.input_block(x, time_embedding)

        skip1, x1 = self.down1(x0, time_embedding)
        skip2, x2 = self.down2(x1, time_embedding)

        x_mid = self.middle(x2, time_embedding)

        x = self.up2(x_mid, skip2, time_embedding)
        x = self.up1(x, skip1, time_embedding)

        x = torch.cat([x, x0], dim=1)

        return self.output_head(x)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
