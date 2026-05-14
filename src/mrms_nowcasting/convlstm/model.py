"""
ConvLSTM model components for MRMS precipitation nowcasting.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


TensorState = Tuple[torch.Tensor, torch.Tensor]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> TensorState:
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str,
    ) -> TensorState:
        h = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )
        c = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )
        return h, c


class StackedConvLSTMEncoder(nn.Module):
    """Two-layer stacked ConvLSTM encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden1: int = 64,
        hidden2: int = 96,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.cell1 = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden1,
            kernel_size=kernel_size,
        )

        self.bn1 = nn.BatchNorm2d(hidden1)

        self.cell2 = ConvLSTMCell(
            in_channels=hidden1,
            hidden_channels=hidden2,
            kernel_size=kernel_size,
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[TensorState, TensorState]:
        batch_size, seq_len, _, height, width = x_seq.shape
        device = x_seq.device

        h1, c1 = self.cell1.init_hidden(batch_size, height, width, device)
        h2, c2 = self.cell2.init_hidden(batch_size, height, width, device)

        for t in range(seq_len):
            h1, c1 = self.cell1(x_seq[:, t], h1, c1)
            h1_bn = self.bn1(h1)
            h2, c2 = self.cell2(h1_bn, h2, c2)

        return (h1, c1), (h2, c2)


class StackedConvLSTMDecoder(nn.Module):
    """Two-layer stacked ConvLSTM decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden1: int = 64,
        hidden2: int = 96,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.cell1 = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden1,
            kernel_size=kernel_size,
        )

        self.cell2 = ConvLSTMCell(
            in_channels=hidden1,
            hidden_channels=hidden2,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        state1: TensorState,
        state2: TensorState,
    ) -> Tuple[torch.Tensor, TensorState, TensorState]:
        h1, c1 = self.cell1(decoder_input, *state1)
        h2, c2 = self.cell2(h1, *state2)

        return h2, (h1, c1), (h2, c2)


class ConvLSTMNowcast(nn.Module):
    """
    Encoder-decoder ConvLSTM for precipitation nowcasting.

    Input shape:
        (batch, input_steps, 1, height, width)

    Output shape:
        (batch, output_steps, 1, height, width)
    """

    def __init__(
        self,
        hidden1: int = 64,
        hidden2: int = 96,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.encoder = StackedConvLSTMEncoder(
            in_channels=1,
            hidden1=hidden1,
            hidden2=hidden2,
            kernel_size=kernel_size,
        )

        self.decoder = StackedConvLSTMDecoder(
            in_channels=1,
            hidden1=hidden1,
            hidden2=hidden2,
            kernel_size=kernel_size,
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden2, hidden2 // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden2 // 2, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        future_steps: int = 30,
    ) -> torch.Tensor:
        state1, state2 = self.encoder(x)

        outputs = []
        decoder_input = x[:, -1]

        for t in range(future_steps):
            h2, state1, state2 = self.decoder(
                decoder_input=decoder_input,
                state1=state1,
                state2=state2,
            )

            prediction = self.head(h2)
            outputs.append(prediction)

            if self.training and y is not None and teacher_forcing_ratio > 0.0:
                use_ground_truth = (
                    torch.rand(1, device=x.device).item() < teacher_forcing_ratio
                )
                decoder_input = y[:, t] if use_ground_truth else prediction.detach()
            else:
                decoder_input = prediction.detach()

        return torch.stack(outputs, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
