import zipfile

import numpy as np

from stable_datasets.schema import Sequence
from stable_datasets.timeseries import UberTLC


def _builder(builder_cls, *args):
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder, *args)
    return builder


def _fake_download(monkeypatch, path):
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda specs, dest_folder: [path] * len(specs))


def test_uber_tlc_metadata():
    for config in ("hourly", "daily"):
        builder = _builder(UberTLC, config)
        assert set(builder.info.features) == {"series", "item_id", "start"}
        assert isinstance(builder.info.features["series"], Sequence)
    assert UberTLC.SETTINGS == {"hourly": ("h", 24, 24), "daily": ("D", 7, 1)}


def test_uber_tlc_smoke(tmp_path, monkeypatch):
    rows = [
        ("B02617", "2015-01-01 00:10:00", "B02617", 7),
        ("B02617", "2015-01-01 00:50:00", "B02617", 7),
        ("B02617", "2015-01-02 05:00:00", "B02617", 7),
        ("B02617", "2015-06-30 23:59:00", "B02617", 42),
    ]
    csv = "Dispatching_base_num,Pickup_date,Affiliated_base_num,locationID\n" + "".join(
        ",".join(map(str, r)) + "\n" for r in rows
    )
    archive = tmp_path / "uber-raw-data-janjune-15.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("uber-raw-data-janjune-15.csv", csv)
    _fake_download(monkeypatch, archive)

    hourly = UberTLC("hourly", split="test", processed_cache_dir=str(tmp_path / "processed")).with_format("numpy")
    assert [hourly[i]["item_id"] for i in range(len(hourly))] == ["location_007", "location_042"]
    zone7 = np.asarray(hourly[0]["series"])
    assert len(zone7) == 181 * 24 and zone7.sum() == 3 and zone7[0] == 2 and zone7[24 + 5] == 1
    assert hourly[1]["series"][-1] == 1

    daily = UberTLC("daily", split="train", processed_cache_dir=str(tmp_path / "processed")).with_format("numpy")
    np.testing.assert_array_equal(daily[0]["series"][:3], [2, 1, 0])
    assert len(daily[0]["series"]) == 181 - 7
