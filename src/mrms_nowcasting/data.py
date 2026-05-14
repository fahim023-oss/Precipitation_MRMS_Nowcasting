"""
Dataset and dataloader utilities for MRMS precipitation nowcasting.

Each MRMS sample is expected to be a `.npz` file containing:

    precip: array with shape (90, 128, 128)

Default task:
    60 input frames -> 30 forecast frames
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MRMSDataset(Dataset):
    """
    Dataset for MRMS precipitation cubes.

    Parameters
    ----------
    folder : str
        Folder containing `.npz` files.
    input_steps : int
        Number of input frames.
    output_steps : int
        Number of forecast frames.
    precip_key : str
        Name of precipitation variable inside each `.npz` file.
    rain_norm : float
        Normalization factor. Precipitation is divided by this value.
    log_transform : bool
        If True, apply log1p transform before normalization.
    """

    def __init__(
        self,
        folder: str,
        input_steps: int = 60,
        output_steps: int = 30,
        precip_key: str = "precip",
        rain_norm: float = 20.0,
        log_transform: bool = True,
    ) -> None:
        self.folder = folder
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.precip_key = precip_key
        self.rain_norm = rain_norm
        self.log_transform = log_transform

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Data folder not found: {folder}")

        self.files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".npz")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found in {folder}. "
                "Place MRMS files in data/mrms_3hr_cubes_128/."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(self.folder, self.files[index])

        with np.load(path) as npz:
            if self.precip_key not in npz:
                raise KeyError(
                    f"Key '{self.precip_key}' not found in {path}. "
                    f"Available keys: {list(npz.keys())}"
                )

            cube = npz[self.precip_key].astype(np.float32)

        required_steps = self.input_steps + self.output_steps

        if cube.shape[0] < required_steps:
            raise ValueError(
                f"{path} has {cube.shape[0]} frames, "
                f"but {required_steps} are required."
            )

        cube = np.nan_to_num(cube, nan=0.0)
        cube = np.maximum(cube, 0.0)

        if self.log_transform:
            cube = np.log1p(cube)

        cube = cube / self.rain_norm

        x = torch.from_numpy(cube[: self.input_steps]).unsqueeze(1)
        y = torch.from_numpy(
            cube[self.input_steps : self.input_steps + self.output_steps]
        ).unsqueeze(1)

        return x, y

    def get_filename(self, index: int) -> str:
        return self.files[index]


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Subset, Subset, Subset]:
    """
    Chronological train/validation/test split.
    """

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_total))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    data_folder: str,
    input_steps: int = 60,
    output_steps: int = 30,
    precip_key: str = "precip",
    rain_norm: float = 20.0,
    log_transform: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
    """
    Build train, validation, and test dataloaders.
    """

    dataset = MRMSDataset(
        folder=data_folder,
        input_steps=input_steps,
        output_steps=output_steps,
        precip_key=precip_key,
        rain_norm=rain_norm,
        log_transform=log_transform,
    )

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        "Dataset split -> "
        f"train: {len(train_dataset)} | "
        f"val: {len(val_dataset)} | "
        f"test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, test_dataset

