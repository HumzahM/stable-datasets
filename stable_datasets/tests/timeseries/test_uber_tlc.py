import numpy as np
import pytest

from stable_datasets.timeseries import UberTLC


@pytest.mark.parametrize("config, length", [("hourly", 4344), ("daily", 181)])
def test_uber_tlc_dataset(tmp_path, config, length):
    cache = {"download_dir": str(tmp_path / "downloads"), "processed_cache_dir": str(tmp_path / "processed")}

    # Test 1: Check the splits and their sizes.
    ds = UberTLC(config, split=None, **cache)
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert len(train) == len(test) == 262

    # Test 2: Check the sample fields and types.
    first = test[0]
    assert set(first) == {"series", "item_id", "start"}
    assert isinstance(first["item_id"], str)
    assert isinstance(first["start"], str)
    assert (first["item_id"], first["start"]) == ("location_001", "2015-01-01T00:00:00")
    assert test[-1]["item_id"] == "location_265"

    # Test 3: Check the series lengths: test is the full series, train drops the forecast horizon.
    _, horizon, _ = UberTLC.SETTINGS[config]
    test_series = [np.asarray(s, np.float64) for s in test.table.column("series").to_pylist()]
    train_series = [np.asarray(s, np.float64) for s in train.table.column("series").to_pylist()]
    assert {len(s) for s in test_series} == {length}
    assert all(np.array_equal(a, b[:-horizon]) for a, b in zip(train_series, test_series))

    # Test 4: Check every one of the 14,270,479 trips in the raw CSV is counted once.
    assert sum(s.sum() for s in test_series) == 14_270_479
    assert all(s.min() >= 0 for s in test_series)


def test_uber_tlc_settings():
    assert UberTLC.SETTINGS == {"hourly": ("h", 24, 24), "daily": ("D", 7, 1)}
