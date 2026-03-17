"""GitTables corpus loader.

Exposes GitTables — a large-scale corpus of ~1 M relational tables extracted
from GitHub CSV files — as a collection of ``TabularDataset`` objects.

Tables are hosted on Zenodo (record 6517052) as zip archives.  The Zenodo
file manifest is fetched from the REST API on first use and cached in memory.
Zip archives are downloaded to ``~/.stable_datasets/downloads/gittables/`` on
demand.  Individual tables can optionally be cached as Arrow IPC files under
``~/.stable_datasets/processed/gittables/``.

Homepage:  https://gittables.github.io
Paper:     https://dl.acm.org/doi/10.1145/3588930
Zenodo:    https://zenodo.org/records/6517052

Cache layout::

    ~/.stable_datasets/
    ├── downloads/gittables/
    │   └── <zip_name>.zip                  raw Zenodo zip archives
    └── processed/gittables/
        └── <zip_stem>/<csv_stem>/
            ├── data.arrow                  Arrow IPC file
            └── metadata.json              TabularTaskInfo fields

Usage::

    from stable_datasets.tabular import GitTables

    # List zip archives available in the Zenodo record
    zips = GitTables.zip_files()

    # Load a single table (downloads zip + caches Arrow IPC on first use)
    ds = GitTables.load(zip_name="GitTables_1.zip", table_name="some_table.csv")

    # Access data (no pre-defined ML splits for GitTables tables)
    df = ds.to_pandas()
    print(ds.info)

    # Iterate every table in the corpus (streams one zip at a time)
    for ds in GitTables.iter_tables():
        df = ds.to_pandas()

    # Iterate tables within a single archive
    for ds in GitTables.iter_tables(zip_name="GitTables_1.zip"):
        df = ds.to_pandas()
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import ClassVar, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import requests
from loguru import logger as logging

from stable_datasets.arrow_dataset import _mmap_ipc
from stable_datasets.schema import Version
from stable_datasets.utils import _default_dest_folder
from .base import TabularBaseDatasetBuilder, TabularDataset, TabularTaskInfo

_ZENODO_RECORD_ID = "6517052"
_ZENODO_API_URL = f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}"


class GitTables(TabularBaseDatasetBuilder):
    """GitTables corpus: ~1 M relational tables extracted from GitHub CSV files.

    GitTables is a large-scale corpus of relational tables collected from CSV
    files hosted on GitHub.  It is widely used for table representation
    learning, column type inference, and data discovery research.

    Construct with a zip archive name and table name to load a single
    :class:`TabularDataset`::

        ds = GitTables.load(zip_name="GitTables_1.zip", table_name="some_table.csv")
        # or equivalently:
        ds = GitTables(task_name="GitTables_1.zip/some_table.csv")

    Suite-level classmethods::

        GitTables.zip_files()                              # Zenodo archive listing
        GitTables.iter_tables()                            # all tables (streams)
        GitTables.iter_tables(zip_name="GitTables_1.zip") # one archive only

    Note:
        GitTables tables have no pre-defined ML train/test splits.  The full
        table is available via ``ds.to_pandas()`` or ``ds.table``.
        Calling ``ds.get_fold()`` or ``ds.iter_folds()`` will raise
        ``ValueError`` because no splits are defined.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://gittables.github.io",
        "citation": """@article{hulsebos2023gittables,
                        title={Gittables: A large-scale corpus of relational tables},
                        author={Hulsebos, Madelon and Demiralp, {\\c{C}}agatay and Groth, Paul},
                        journal={Proceedings of the ACM on Management of Data},
                        volume={1},
                        number={1},
                        pages={1--17},
                        year={2023},
                        publisher={ACM New York, NY, USA}
                    }""",
    }

    # In-process cache for the Zenodo file manifest.
    _zenodo_manifest: ClassVar[list[dict] | None] = None

    # ------------------------------------------------------------------
    # TabularBaseDatasetBuilder implementation
    # ------------------------------------------------------------------

    def _build_tabular_dataset(
        self, task_id: int | None = None, task_name: str | None = None
    ) -> TabularDataset:
        """Load a single table identified by its encoded ``task_name``.

        ``task_name`` must be of the form ``"<zip_name>/<csv_name>"``
        (e.g. ``"GitTables_1.zip/some_table.csv"``).
        Prefer :meth:`load` for a friendlier interface.
        """
        if task_name is None:
            raise ValueError(
                "Provide a table via GitTables.load(zip_name=..., table_name=...) "
                "or GitTables(task_name='<zip_name>/<csv_name>')."
            )

        zip_name, csv_name = _parse_table_name(task_name)
        cache_dir = self._table_cache_dir(zip_name, csv_name)

        if _is_cached(cache_dir):
            return _load_from_cache(cache_dir)

        return _download_and_cache(
            zip_name=zip_name,
            csv_name=csv_name,
            cache_dir=cache_dir,
            download_dir=self._raw_download_dir / "gittables",
            manifest=self.__class__.zip_files(),
        )

    def _table_cache_dir(self, zip_name: str, csv_name: str) -> Path:
        zip_stem = Path(zip_name).stem
        csv_stem = Path(csv_name).stem
        return self._processed_cache_dir / "gittables" / zip_stem / csv_stem

    # ------------------------------------------------------------------
    # Corpus discovery
    # ------------------------------------------------------------------

    @classmethod
    def zip_files(cls) -> list[dict]:
        """Return the list of zip archives in the GitTables Zenodo record.

        Each entry is a dict with keys:

        * ``name`` — archive filename (e.g. ``"GitTables_1.zip"``)
        * ``url``  — direct download URL
        * ``size`` — file size in bytes

        The result is fetched from the Zenodo REST API on the first call and
        cached in memory for the lifetime of the process.
        """
        if cls._zenodo_manifest is None:
            logging.info(
                f"Fetching GitTables file manifest from Zenodo record {_ZENODO_RECORD_ID!r}..."
            )
            resp = requests.get(_ZENODO_API_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cls._zenodo_manifest = [
                {
                    "name": f["key"],
                    "url": f["links"]["self"],
                    "size": f["size"],
                }
                for f in data["files"]
                if f["key"].endswith(".zip")
            ]
            logging.info(
                f"GitTables corpus: {len(cls._zenodo_manifest)} zip archive(s) on Zenodo."
            )
        return cls._zenodo_manifest

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        zip_name: str,
        table_name: str,
        processed_cache_dir: Path | str | None = None,
    ) -> TabularDataset:
        """Load a single GitTables table, downloading and caching if needed.

        Downloads the containing zip archive (once, to the raw-download cache),
        extracts the requested CSV, converts it to Arrow IPC format, and
        caches it for future use.

        Args:
            zip_name: Zenodo archive filename (e.g. ``"GitTables_1.zip"``).
                Use :meth:`zip_files` to list available archives.
            table_name: CSV filename within the archive (e.g. ``"some_table.csv"``).
            processed_cache_dir: Override the processed-cache root directory.
                Defaults to ``~/.stable_datasets/processed/``.
                Respects the ``STABLE_DATASETS_CACHE_DIR`` environment variable.

        Returns:
            A :class:`~stable_datasets.tabular.TabularDataset` with all rows
            of the table.  No pre-defined CV splits are available.
        """
        return cls(
            task_name=f"{zip_name}/{table_name}",
            processed_cache_dir=processed_cache_dir,
        )

    @classmethod
    def iter_tables(
        cls,
        zip_name: str | None = None,
        processed_cache_dir: Path | str | None = None,
        cache_tables: bool = True,
    ) -> Iterator[TabularDataset]:
        """Iterate over GitTables tables, yielding one ``TabularDataset`` per CSV.

        Zip archives are downloaded one at a time and their CSV files are
        streamed without loading the full archive into memory at once.
        Tables that fail to parse are skipped with a warning.

        Args:
            zip_name: If provided, restrict iteration to this archive only.
                Otherwise all archives in the Zenodo record are iterated in
                corpus order.
            processed_cache_dir: Passed through to :meth:`load`.  Applies only
                when ``cache_tables=True``.
            cache_tables: Whether to write Arrow IPC cache files for each
                loaded table.  Set to ``False`` when iterating the full corpus
                to avoid creating ~1 M small files on disk.  Defaults to
                ``True``.
        """
        manifests = cls.zip_files()
        if zip_name is not None:
            manifests = [m for m in manifests if m["name"] == zip_name]
            if not manifests:
                raise ValueError(
                    f"Archive {zip_name!r} not found in Zenodo manifest. "
                    "Use GitTables.zip_files() to list available archives."
                )

        download_dir = _default_dest_folder() / "gittables"

        for entry in manifests:
            zname = entry["name"]
            zip_path = _ensure_zip_downloaded(entry["url"], zname, download_dir)
            logging.info(f"Streaming tables from {zname!r}...")
            for csv_path_in_zip in _list_csv_names_in_zip(zip_path):
                csv_name = Path(csv_path_in_zip).name
                try:
                    if cache_tables:
                        yield cls.load(
                            zip_name=zname,
                            table_name=csv_name,
                            processed_cache_dir=processed_cache_dir,
                        )
                    else:
                        yield _read_table_from_zip(zip_path, csv_path_in_zip, csv_name)
                except Exception as exc:
                    logging.warning(f"Skipping {zname}/{csv_name}: {exc}")


# ------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ------------------------------------------------------------------


def _parse_table_name(task_name: str) -> tuple[str, str]:
    """Parse ``'<zip_name>/<csv_name>'`` → ``(zip_name, csv_name)``."""
    parts = task_name.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid table name {task_name!r}. "
            "Expected '<zip_name>/<csv_name>' "
            "(e.g. 'GitTables_1.zip/some_table.csv')."
        )
    return parts[0], parts[1]


def _is_cached(cache_dir: Path) -> bool:
    return (
        (cache_dir / "data.arrow").exists()
        and (cache_dir / "metadata.json").exists()
    )


def _load_from_cache(cache_dir: Path) -> TabularDataset:
    """Load a GitTables table from an existing Arrow + JSON cache."""
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
    table = _mmap_ipc(cache_dir / "data.arrow")
    return TabularDataset(table, info)


def _download_and_cache(
    zip_name: str,
    csv_name: str,
    cache_dir: Path,
    download_dir: Path,
    manifest: list[dict],
) -> TabularDataset:
    """Download the zip containing ``csv_name``, extract, cache, and return a TabularDataset."""
    url = next((m["url"] for m in manifest if m["name"] == zip_name), None)
    if url is None:
        raise ValueError(
            f"Archive {zip_name!r} not found in Zenodo manifest. "
            "Use GitTables.zip_files() to list available archives."
        )

    zip_path = _ensure_zip_downloaded(url, zip_name, download_dir)
    df = _extract_csv_from_zip(zip_path, csv_name)

    info = TabularTaskInfo(
        task_id=0,
        task_name=csv_name,
        problem_type="unknown",
        target_col="",
        n_rows=len(df),
        n_features=len(df.columns),
        n_folds=0,
        n_repeats=0,
    )

    table = pa.Table.from_pandas(df, preserve_index=False)

    # Atomic write: temp dir → rename.
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir.parent, prefix=".tmp_"))
    try:
        _write_arrow(table, tmp_dir / "data.arrow")
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

    logging.info(f"Cached table {csv_name!r} to {cache_dir}")
    return TabularDataset(table, info)


def _ensure_zip_downloaded(url: str, zip_name: str, download_dir: Path) -> Path:
    """Return the local path of the zip, downloading it from Zenodo if needed."""
    zip_path = download_dir / zip_name
    if zip_path.exists():
        return zip_path

    download_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(".part")

    logging.info(f"Downloading {zip_name!r} from Zenodo...")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            from tqdm import tqdm

            with open(tmp_path, "wb") as f, tqdm(
                total=total or None, unit="B", unit_scale=True, desc=zip_name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=65_536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp_path.rename(zip_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logging.info(f"Downloaded {zip_name!r} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def _list_csv_names_in_zip(zip_path: Path) -> list[str]:
    """Return all CSV entry paths within a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [name for name in zf.namelist() if name.lower().endswith(".csv")]


def _extract_csv_from_zip(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """Extract and parse a CSV from a zip archive, matching by basename."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [n for n in zf.namelist() if Path(n).name == csv_name]
        if not matches:
            raise ValueError(f"CSV {csv_name!r} not found in {zip_path.name}.")
        with zf.open(matches[0]) as f:
            return pd.read_csv(f)


def _read_table_from_zip(
    zip_path: Path, csv_path_in_zip: str, csv_name: str
) -> TabularDataset:
    """Parse a CSV entry from a zip and return an in-memory TabularDataset (no disk cache)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_path_in_zip) as f:
            df = pd.read_csv(f)

    info = TabularTaskInfo(
        task_id=0,
        task_name=csv_name,
        problem_type="unknown",
        target_col="",
        n_rows=len(df),
        n_features=len(df.columns),
        n_folds=0,
        n_repeats=0,
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    return TabularDataset(table, info)


def _write_arrow(table: pa.Table, path: Path) -> None:
    """Write a PyArrow Table to an Arrow IPC file."""
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
