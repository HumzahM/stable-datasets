"""Base class for a single tabular ML table.

``TabularDataset`` wraps a PyArrow table (all columns, including the target)
alongside optional pre-defined train/test split indices for n-fold cross-
validation.  It is the shared return type for all tabular suite loaders
(TabArena, and future suites).

Unlike ``StableDataset``, tabular datasets are stored as a single Arrow IPC
file rather than shards, because tabular datasets are typically small enough
to fit in memory and require whole-table access for fold operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd
import pyarrow as pa


@dataclass(frozen=True)
class TabularTaskInfo:
    """Static metadata for a single tabular ML task.

    ``n_rows`` and ``n_features`` always reflect the *full* dataset, even when
    this object is attached to a fold subset.
    """

    task_id: int
    task_name: str
    problem_type: str  # "binary" | "multiclass" | "regression"
    target_col: str
    n_rows: int
    n_features: int
    n_folds: int
    n_repeats: int


# Splits are stored as {repeat_id: {fold_id: (train_indices, test_indices)}}.
# Indices are plain Python lists of ints so they serialise to JSON cleanly.
_Splits = dict[int, dict[int, tuple[list[int], list[int]]]]


class TabularDataset:
    """Arrow-backed tabular ML dataset with optional pre-defined CV splits.

    Construction is handled by suite loaders (e.g. ``TabArena``); you should
    not need to instantiate this class directly.

    Row access::

        ds[0]              # dict of column values for one row
        ds[10:50]          # new TabularDataset with rows 10-49
        for row in ds: ... # iterate all rows as dicts

    Data as pandas / Arrow::

        ds.to_pandas()     # pd.DataFrame (full table, features + target)
        ds.X               # pa.Table  (feature columns only)
        ds.y               # pa.ChunkedArray  (target column)

    Cross-validation::

        train, test = ds.get_fold(fold=0, repeat=0)
        for fold, repeat, train, test in ds.iter_folds():
            ...

    Metadata::

        ds.task_id, ds.task_name, ds.problem_type
        ds.target_col, ds.n_folds, ds.n_repeats
        ds.info            # TabularTaskInfo dataclass
    """

    def __init__(
        self,
        table: pa.Table,
        info: TabularTaskInfo,
        *,
        splits: _Splits | None = None,
    ):
        self._table = table
        self._info = info
        # splits is None for fold subsets (they have no further CV splits).
        self._splits: _Splits = splits or {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def info(self) -> TabularTaskInfo:
        return self._info

    @property
    def task_id(self) -> int:
        return self._info.task_id

    @property
    def task_name(self) -> str:
        return self._info.task_name

    @property
    def problem_type(self) -> str:
        return self._info.problem_type

    @property
    def target_col(self) -> str:
        return self._info.target_col

    @property
    def n_folds(self) -> int:
        return self._info.n_folds

    @property
    def n_repeats(self) -> int:
        return self._info.n_repeats

    # ------------------------------------------------------------------
    # Row-level access
    # ------------------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """The underlying Arrow table (all columns, including the target)."""
        return self._table

    @property
    def X(self) -> pa.Table:
        """Feature columns only (all columns except the target)."""
        cols = [c for c in self._table.column_names if c != self._info.target_col]
        return self._table.select(cols)

    @property
    def y(self) -> pa.ChunkedArray:
        """Target column as a PyArrow ChunkedArray."""
        return self._table.column(self._info.target_col)

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int | slice) -> dict | TabularDataset:
        """Return a row dict (int) or a new in-memory TabularDataset (slice)."""
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            return {col: self._table.column(col)[idx].as_py() for col in self._table.column_names}
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return self._take(indices)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __iter__(self) -> Iterator[dict]:
        """Iterate all rows as plain Python dicts."""
        for i in range(len(self)):
            yield {col: self._table.column(col)[i].as_py() for col in self._table.column_names}

    def to_pandas(self) -> pd.DataFrame:
        """Return the full table as a pandas DataFrame (features + target)."""
        return self._table.to_pandas()

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def get_fold(self, fold: int = 0, repeat: int = 0) -> tuple[TabularDataset, TabularDataset]:
        """Return ``(train, test)`` for the given fold/repeat pair.

        Both datasets are in-memory Arrow subsets with no splits of their own.
        Indices are the pre-defined OpenML splits stored at download time.

        Args:
            fold: Zero-based fold index (0 to ``n_folds - 1``).
            repeat: Zero-based repeat index (0 to ``n_repeats - 1``).

        Returns:
            A ``(train, test)`` tuple of ``TabularDataset`` instances.

        Raises:
            ValueError: If this dataset has no pre-defined splits, or if the
                requested fold/repeat combination does not exist.
        """
        if not self._splits:
            raise ValueError(
                "This TabularDataset has no pre-defined splits. "
                "Fold subsets (returned by get_fold / iter_folds) cannot be split further."
            )
        try:
            train_idx, test_idx = self._splits[repeat][fold]
        except KeyError:
            raise ValueError(
                f"No splits for repeat={repeat}, fold={fold}. "
                f"Available: {self.n_repeats} repeat(s), {self.n_folds} fold(s)."
            )
        return self._take(train_idx), self._take(test_idx)

    def iter_folds(self) -> Iterator[tuple[int, int, TabularDataset, TabularDataset]]:
        """Yield ``(fold, repeat, train, test)`` for every fold/repeat pair.

        Iterates repeats in ascending order, and folds within each repeat in
        ascending order.

        Raises:
            ValueError: If this dataset has no pre-defined splits.
        """
        if not self._splits:
            raise ValueError(
                "This TabularDataset has no pre-defined splits. "
                "Fold subsets (returned by get_fold / iter_folds) cannot be split further."
            )
        for repeat in sorted(self._splits):
            for fold in sorted(self._splits[repeat]):
                train, test = self.get_fold(fold=fold, repeat=repeat)
                yield fold, repeat, train, test

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _take(self, indices: list[int]) -> TabularDataset:
        """Return a new in-memory TabularDataset with only the given rows."""
        return TabularDataset(self._table.take(indices), self._info)

    def __repr__(self) -> str:
        return (
            f"TabularDataset("
            f"task={self._info.task_name!r}, "
            f"n_rows={len(self)}, "
            f"problem_type={self._info.problem_type!r}, "
            f"n_folds={self._info.n_folds}, "
            f"n_repeats={self._info.n_repeats}"
            f")"
        )
