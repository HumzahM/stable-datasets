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

    def __init__(self, filename: str, num_steps: int, batch_size: int,
                 seed: int | None = None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.seed = seed
        self.pointer = 0

        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            total = int(f["X"].shape[0])

        if seed is None:
            self._batch_starts = None
        else:
            # Shuffle only within the samples the sequential loader would consume
            # before wrapping: the first num_steps * batch_size rows. Every seed
            # sees the same sample set, differing only in order.
            n_perm = min(num_steps, total // batch_size)
            g = torch.Generator().manual_seed(int(seed))
            perm = torch.randperm(n_perm, generator=g).tolist()
            self._batch_starts = [i * batch_size for i in perm]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for step in range(self.num_steps):
                if self._batch_starts is None:
                    start = self.pointer
                    self.pointer += self.batch_size
                    if self.pointer >= f["X"].shape[0]:
                        self.pointer = 0
                else:
                    start = self._batch_starts[step % len(self._batch_starts)]
                end = start + self.batch_size
                num_features = f["num_features"][start:end].max()
                num_datapoints_batch = f["num_datapoints"][start:end]
                max_seq_in_batch = int(num_datapoints_batch.max())

                x = torch.from_numpy(
                    f["X"][start:end, :max_seq_in_batch, :num_features]
                )
                y = torch.from_numpy(
                    f["y"][start:end, :max_seq_in_batch]
                )
                train_test_split_index = f["single_eval_pos"][start:end]

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
    seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader that streams synthetic prior data from an HDF5 dump.

    Each iteration yields ``num_steps`` pre-batched dicts.  When ``seed`` is
    ``None`` batches are read sequentially and the pointer wraps at EOF.
    When ``seed`` is given, a per-seed permutation of batch starts is used
    so different seeds see different data orders.

    Device placement is left to the training framework (e.g. PyTorch
    Lightning moves batches to the accelerator automatically).

    Args:
        filename: Path to the HDF5 file.
        num_steps: Number of batches per epoch.
        batch_size: Number of datasets per batch.
        seed: If given, shuffle batch order with this seed.

    Returns:
        A standard ``DataLoader`` wrapping a :class:`_SyntheticPriorDataset`.
    """
    dataset = _SyntheticPriorDataset(filename, num_steps, batch_size, seed=seed)
    return DataLoader(dataset, batch_size=None)
