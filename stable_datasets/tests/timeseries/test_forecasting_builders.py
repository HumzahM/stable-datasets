import zipfile

import numpy as np

from stable_datasets.schema import Sequence
from stable_datasets.timeseries import ElectricityHourly, UberTLC


def _builder(builder_cls, *args):
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder, *args)
    return builder


def _fake_download(monkeypatch, path):
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda specs, dest_folder: [path] * len(specs))


def test_forecasting_builders_metadata():
    for builder in [_builder(ElectricityHourly), _builder(UberTLC, "hourly"), _builder(UberTLC, "daily")]:
        assert set(builder.info.features) == {"series", "item_id", "start"}
        assert isinstance(builder.info.features["series"], Sequence)
    assert (ElectricityHourly.FREQ, ElectricityHourly.PREDICTION_LENGTH, ElectricityHourly.SEASONALITY) == (
        "h",
        24,
        24,
    )
    assert UberTLC.SETTINGS == {"hourly": ("h", 24, 24), "daily": ("D", 7, 1)}


def test_electricity_hourly_smoke(tmp_path, monkeypatch):
    values = [np.arange(30), np.arange(30) * 2]
    tsf = "# comment\n@relation Electricity\n@frequency hourly\n@data\n" + "".join(
        f"T{i + 1}:2012-01-01 00-00-01:{','.join(str(v) for v in s)}\n" for i, s in enumerate(values)
    )
    archive = tmp_path / "electricity_hourly_dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("electricity_hourly_dataset.tsf", tsf)
    _fake_download(monkeypatch, archive)

    test = ElectricityHourly(split="test", processed_cache_dir=str(tmp_path / "processed")).with_format("numpy")
    train = ElectricityHourly(split="train", processed_cache_dir=str(tmp_path / "processed")).with_format("numpy")
    assert len(test) == len(train) == 2
    assert test[1]["item_id"] == "T2"
    assert test[1]["start"] == "2012-01-01T00:00:01"
    np.testing.assert_array_equal(test[1]["series"], values[1])
    np.testing.assert_array_equal(train[1]["series"], values[1][:-24])


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
