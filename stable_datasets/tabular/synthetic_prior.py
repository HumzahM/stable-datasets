"""Synthetic prior data loader for nanoTabPFN training.

Loads pre-generated synthetic tabular datasets from an HDF5 dump file.
Each batch contains ``(X, y, train_test_split_index)`` tensors sampled
from random MLP priors — the same data format used by nanoTabPFN's
training loop.

The HDF5 file can be downloaded from Figshare::

    curl -L -H "User-Agent: Mozilla/5.0" -H "Referer: https://figshare.com/" \
        -o 300k_150x5_2.h5 \
        "https://figshare.com/ndownloader/files/58932628?private_link=63fc1ada93e42e388e63"

Usage::

    from stable_datasets.tabular import SyntheticPrior

    prior = SyntheticPrior("300k_150x5_2.h5", num_steps=2500, batch_size=32)
    for batch in prior:
        x = batch["x"]           # (batch_size, max_rows, max_features)
        y = batch["y"]           # (batch_size, max_rows)
        split = batch["train_test_split_index"]  # int
"""

from __future__ import annotations

import h5py
import torch
from torch.utils.data import DataLoader


class _SyntheticPriorDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that streams batches from an HDF5 prior dump.

    This is the underlying dataset used by :class:`SyntheticPrior`.  It
    yields pre-batched dicts so the wrapping ``DataLoader`` should use
    ``batch_size=None``.
    """

    def __init__(self, filename: str, num_steps: int, batch_size: int):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.pointer = 0

        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer:end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())

                x = torch.from_numpy(
                    f["X"][self.pointer:end, :max_seq_in_batch, :num_features]
                )
                y = torch.from_numpy(
                    f["y"][self.pointer:end, :max_seq_in_batch]
                )
                train_test_split_index = f["single_eval_pos"][self.pointer:end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    self.pointer = 0

                yield dict(
                    x=x,
                    y=y,
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps


def SyntheticPrior(
    filename: str,
    num_steps: int,
    batch_size: int,
) -> DataLoader:
    """Create a DataLoader that streams synthetic prior data from an HDF5 dump.

    Each iteration yields ``num_steps`` pre-batched dicts.  When the file
    is exhausted the pointer wraps around to the beginning.

    Device placement is left to the training framework (e.g. PyTorch
    Lightning moves batches to the accelerator automatically).

    Args:
        filename: Path to the HDF5 file.
        num_steps: Number of batches per epoch.
        batch_size: Number of datasets per batch.

    Returns:
        A standard ``DataLoader`` wrapping a :class:`_SyntheticPriorDataset`.
    """
    dataset = _SyntheticPriorDataset(filename, num_steps, batch_size)
    return DataLoader(dataset, batch_size=None)
