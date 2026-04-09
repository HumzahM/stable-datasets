"""TabArena benchmark suite loader.

Exposes the full TabArena benchmark (NeurIPS 2025 Datasets & Benchmarks) as a
collection of ``TabularDataset`` objects.  Task IDs are fetched from the
OpenML study "tabarena-v0.1" and cached in memory after the first call.
Each task's data and pre-defined fold/repeat splits are downloaded from
OpenML and cached locally as Arrow IPC + JSON on first use.

Homepage: https://openreview.net/forum?id=jZqCqpCLdU
Code:     https://github.com/autogluon/tabarena/

Cache layout (under ``~/.stable_datasets/processed/tabarena/``)::

    task_<task_id>/
    ├── data.arrow       Arrow IPC file — full table, all columns incl. target
    ├── splits.json      {repeat: {fold: [train_indices, test_indices]}}
    └── metadata.json    TabularTaskInfo fields

Usage::

    from stable_datasets.tabular import TabArena

    # List all task IDs in the suite
    ids = TabArena.task_ids()

    # Load a single task (downloads + caches on first use)
    ds = TabArena(task_id=363621)
    ds = TabArena(task_name="credit-g")   # slower: scans cache then OpenML

    # Cross-validation
    train, test = ds.get_fold(fold=0, repeat=0)
    for fold, repeat, train, test in ds.iter_folds():
        ...

    # Iterate the whole suite
    for ds in TabArena.iter_tasks():
        train, test = ds.get_fold(fold=0, repeat=0)

    # Get preprocessed numpy arrays for nanoTabPFN evaluation
    datasets = TabArena.get_numpy_datasets(max_features=10, max_instances=200)
    for name, (X, y) in datasets.items():
        clf.fit(X_train, y_train)
        ...
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from loguru import logger as logging
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer

from stable_datasets.arrow_dataset import _mmap_ipc
from stable_datasets.schema import Version
from .base import TabularBaseDatasetBuilder, TabularDataset, TabularTaskInfo, _Splits

_SUITE_NAME = "tabarena-v0.1"


class TabArena(TabularBaseDatasetBuilder):
    """TabArena benchmark suite: ~51 curated OpenML tabular datasets.

    Construct with a task ID or name to load a single :class:`TabularDataset`::

        ds = TabArena(task_id=363621)
        ds = TabArena(task_name="credit-g")

    Suite-level classmethods::

        TabArena.task_ids()       # list of all OpenML task IDs
        TabArena.iter_tasks()     # iterator of TabularDataset objects
        TabArena.load(task_id=…)  # alias for TabArena(task_id=…)

    Requires the ``openml`` package (``pip install openml``).
    """

    VERSION = Version("0.1.0")

    SOURCE = {
        "homepage": "https://openreview.net/forum?id=jZqCqpCLdU",
        "code": "https://github.com/autogluon/tabarena/",
        "citation": """@inproceedings{erickson2025tabarena,
                        title={TabArena: A Living Benchmark for Machine Learning on Tabular Data},
                        author={Nick Erickson and Lennart Purucker and Andrej Tschalzev and David Holzm{\\"u}ller and Prateek Mutalik Desai and David Salinas and Frank Hutter},
                        booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
                        year={2025},
                        url={https://openreview.net/forum?id=jZqCqpCLdU}
                    }""",
    }

    # In-process cache: populated on first call to task_ids().
    _suite_task_ids: ClassVar[list[int] | None] = None

    # ------------------------------------------------------------------
    # TabularBaseDatasetBuilder implementation
    # ------------------------------------------------------------------

    def _build_tabular_dataset(
        self, task_id: int | None = None, task_name: str | None = None
    ) -> TabularDataset:
        if task_id is None and task_name is None:
            raise ValueError("Provide either task_id or task_name.")
        if task_id is not None and task_name is not None:
            raise ValueError("Provide only one of task_id or task_name, not both.")

        if task_id is None:
            task_id = self._resolve_task_name(task_name)

        cache_dir = self._task_cache_dir(task_id)

        if _is_cached(cache_dir):
            return _load_from_cache(cache_dir)

        return _download_and_cache(task_id, cache_dir)

    def _task_cache_dir(self, task_id: int) -> Path:
        return self._processed_cache_dir / "tabarena" / f"task_{task_id}"

    def _resolve_task_name(self, task_name: str) -> int:
        """Resolve a dataset name to a task ID.

        Fast path: scan ``metadata.json`` files in the local cache.
        Slow path: fetch each task's metadata from OpenML until a match is found.
        """
        base = self._processed_cache_dir / "tabarena"
        if base.exists():
            for task_dir in base.iterdir():
                meta_path = task_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("task_name") == task_name:
                        return meta["task_id"]

        import openml

        logging.warning(
            f"Task name {task_name!r} not found in local cache. "
            "Querying OpenML for each task — this may be slow. "
            "Use task_id for faster lookups."
        )
        for tid in self.__class__.task_ids():
            task = openml.tasks.get_task(tid, download_data=False, download_splits=False)
            ds = task.get_dataset(download_data=False)
            if ds.name == task_name:
                return tid

        raise ValueError(
            f"No TabArena task with name {task_name!r}. "
            "Use TabArena.task_ids() to list available task IDs."
        )

    # ------------------------------------------------------------------
    # Task discovery
    # ------------------------------------------------------------------

    @classmethod
    def task_ids(cls) -> list[int]:
        """Return the list of OpenML task IDs in the TabArena suite.

        The result is fetched from OpenML on the first call and cached in
        memory for subsequent calls within the same process.
        """
        if cls._suite_task_ids is None:
            import openml

            logging.info(f"Fetching task list from OpenML suite {_SUITE_NAME!r}...")
            suite = openml.study.get_suite(_SUITE_NAME)
            cls._suite_task_ids = list(suite.tasks)
            logging.info(f"TabArena suite contains {len(cls._suite_task_ids)} tasks.")
        return cls._suite_task_ids

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        task_id: int | None = None,
        task_name: str | None = None,
        processed_cache_dir: Path | str | None = None,
    ) -> TabularDataset:
        """Load a single TabArena task, downloading and caching if needed.

        Alias for ``TabArena(task_id=…)`` / ``TabArena(task_name=…)``.

        Args:
            task_id: OpenML task ID.  Provide either this or ``task_name``.
            task_name: Dataset name (e.g. ``"credit-g"``).  Resolved by
                scanning the local cache first, then querying OpenML — prefer
                ``task_id`` when performance matters.
            processed_cache_dir: Override the base directory for processed
                caches.  Defaults to ``~/.stable_datasets/processed/``.
                Respects the ``STABLE_DATASETS_CACHE_DIR`` environment variable.

        Returns:
            A :class:`~stable_datasets.tabular.TabularDataset` with all rows
            and pre-defined fold/repeat splits ready for use.
        """
        return cls(
            task_id=task_id,
            task_name=task_name,
            processed_cache_dir=processed_cache_dir,
        )

    @classmethod
    def iter_tasks(
        cls,
        task_ids: list[int] | None = None,
        processed_cache_dir: Path | str | None = None,
    ) -> Iterator[TabularDataset]:
        """Iterate over TabArena tasks, yielding one ``TabularDataset`` at a time.

        Args:
            task_ids: Subset of task IDs to iterate.  Defaults to all tasks
                in the suite (from :meth:`task_ids`).
            processed_cache_dir: Passed through to :meth:`load`.
        """
        ids = task_ids if task_ids is not None else cls.task_ids()
        for tid in ids:
            yield cls.load(task_id=tid, processed_cache_dir=processed_cache_dir)

    # ------------------------------------------------------------------
    # Preprocessed numpy datasets (for nanoTabPFN evaluation)
    # ------------------------------------------------------------------

    @classmethod
    def get_numpy_datasets(
        cls,
        max_features: int = 10,
        max_instances: int = 200,
        max_classes: int = 2,
        max_missing_pct: float = 0.0,
        min_minority_pct: float = 2.5,
        processed_cache_dir: Path | str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Load TabArena classification tasks as preprocessed numpy arrays.

        Filters, preprocesses, and subsamples tasks for use with sklearn-style
        classifiers (e.g. nanoTabPFN).  Each returned dataset has:

        - Numerical columns cast to float, categorical columns ordinal-encoded
        - Constant columns dropped
        - Target label-encoded to ``0 .. n_classes-1``
        - Rows subsampled (stratified) to ``max_instances``

        Args:
            max_features:      Skip tasks with more than this many features.
            max_instances:     Subsample to this many rows (stratified).
            max_classes:       Skip tasks with more than this many classes.
            max_missing_pct:   Skip tasks where missing-value percentage exceeds this.
            min_minority_pct:  Skip tasks where minority class percentage is below this.
            processed_cache_dir: Passed through to :meth:`load`.

        Returns:
            Dict mapping dataset name to ``(X, y)`` numpy arrays.
        """
        import openml
        from openml.tasks import TaskType

        datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for task_id in cls.task_ids():
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue

            dataset = task.get_dataset(download_data=False)
            qualities = dataset.qualities

            if qualities["NumberOfFeatures"] > max_features:
                continue
            if qualities["NumberOfClasses"] > max_classes:
                continue
            if qualities["PercentageOfInstancesWithMissingValues"] > max_missing_pct:
                continue
            if qualities["MinorityClassPercentage"] < min_minority_pct:
                continue

            X, y, _, _ = dataset.get_data(
                target=task.target_name, dataset_format="dataframe",
            )

            if max_instances < len(y):
                _, X, _, y = train_test_split(
                    X, y, test_size=max_instances, stratify=y, random_state=0,
                )

            X = X.to_numpy(copy=True)
            y = y.to_numpy(copy=True)

            y = LabelEncoder().fit_transform(y)

            preprocessor = _get_feature_preprocessor(X)
            X = preprocessor.fit_transform(X)

            datasets[dataset.name] = (X, y)

        return datasets


# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------


def _get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessor that drops constant columns, casts numerics, and ordinal-encodes categoricals."""
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan = X[col].dropna().unique()
        if len(unique_non_nan) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_count = X[col].notna().sum()
        numeric_count = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_count == numeric_count)
        cat_mask.append(non_nan_count != numeric_count)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(
            lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x,
        )),
        ("to_numeric", FunctionTransformer(
            lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy(),
        )),
    ])
    cat_transformer = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan,
        )),
    ])

    return ColumnTransformer(transformers=[
        ("num", num_transformer, num_mask),
        ("cat", cat_transformer, cat_mask),
    ])


# ------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ------------------------------------------------------------------


def _is_cached(cache_dir: Path) -> bool:
    return (
        (cache_dir / "data.arrow").exists()
        and (cache_dir / "splits.json").exists()
        and (cache_dir / "metadata.json").exists()
    )


def _download_and_cache(task_id: int, cache_dir: Path) -> TabularDataset:
    """Download a task from OpenML and write it to the local Arrow cache."""
    import openml

    logging.info(f"Downloading TabArena task {task_id} from OpenML...")
    task = openml.tasks.get_task(task_id, download_splits=True, download_data=True)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

    problem_type = _infer_problem_type(task)
    n_repeats, n_folds, _ = task.get_split_dimensions()

    # Collect all fold/repeat split indices.
    splits: _Splits = {}
    for repeat in range(n_repeats):
        splits[repeat] = {}
        for fold in range(n_folds):
            train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
            splits[repeat][fold] = (train_idx.tolist(), test_idx.tolist())

    # Combine X and y into one DataFrame; reset index so Arrow row positions
    # are 0-based and match the OpenML split indices.
    import pandas as pd

    df = pd.concat([X, y.rename(task.target_name)], axis=1).reset_index(drop=True)

    info = TabularTaskInfo(
        task_id=task_id,
        task_name=dataset.name,
        problem_type=problem_type,
        target_col=task.target_name,
        n_rows=len(df),
        n_features=len(X.columns),
        n_folds=n_folds,
        n_repeats=n_repeats,
    )

    # Write atomically: temp dir → rename.
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir.parent, prefix=f".task_{task_id}_tmp_"))
    try:
        _write_arrow(df, tmp_dir / "data.arrow")
        (tmp_dir / "splits.json").write_text(json.dumps(_encode_splits(splits)))
        (tmp_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "task_id": info.task_id,
                    "task_name": info.task_name,
                    "problem_type": info.problem_type,
                    "target_col": info.target_col,
                    "n_rows": info.n_rows,
                    "n_features": info.n_features,
                    "n_folds": info.n_folds,
                    "n_repeats": info.n_repeats,
                },
                indent=2,
            )
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        shutil.move(str(tmp_dir), str(cache_dir))
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logging.info(f"Cached task {task_id} ({info.task_name!r}) to {cache_dir}")

    table = _mmap_ipc(cache_dir / "data.arrow")
    return TabularDataset(table, info, splits=splits)


def _load_from_cache(cache_dir: Path) -> TabularDataset:
    """Load a TabularDataset from an existing local Arrow + JSON cache."""
    meta = json.loads((cache_dir / "metadata.json").read_text())
    info = TabularTaskInfo(
        task_id=meta["task_id"],
        task_name=meta["task_name"],
        problem_type=meta["problem_type"],
        target_col=meta["target_col"],
        n_rows=meta["n_rows"],
        n_features=meta["n_features"],
        n_folds=meta["n_folds"],
        n_repeats=meta["n_repeats"],
    )
    splits = _decode_splits(json.loads((cache_dir / "splits.json").read_text()))
    table = _mmap_ipc(cache_dir / "data.arrow")
    return TabularDataset(table, info, splits=splits)


def _infer_problem_type(task) -> str:
    if task.task_type == "Supervised Regression":
        return "regression"
    labels = getattr(task, "class_labels", None)
    if labels is not None and len(labels) == 2:
        return "binary"
    return "multiclass"


def _write_arrow(df, path: Path) -> None:
    """Write a pandas DataFrame to an Arrow IPC file."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _encode_splits(splits: _Splits) -> dict:
    """Convert splits to a JSON-serialisable form (int keys → str keys)."""
    return {
        str(repeat): {
            str(fold): [train, test]
            for fold, (train, test) in folds.items()
        }
        for repeat, folds in splits.items()
    }


def _decode_splits(raw: dict) -> _Splits:
    """Restore splits from JSON (str keys → int keys)."""
    return {
        int(repeat): {
            int(fold): (pair[0], pair[1])
            for fold, pair in folds.items()
        }
        for repeat, folds in raw.items()
    }
