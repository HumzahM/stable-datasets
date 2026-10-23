import zipfile

import numpy as np

from stable_datasets.schema import Sequence
from stable_datasets.timeseries import ElectricityHourly


def _builder(builder_cls, *args):
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder, *args)
    return builder


def _fake_download(monkeypatch, path):
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda specs, dest_folder: [path] * len(specs))


def test_electricity_hourly_metadata():
    builder = _builder(ElectricityHourly)
    assert set(builder.info.features) == {"series", "item_id", "start"}
    assert isinstance(builder.info.features["series"], Sequence)
    assert (ElectricityHourly.FREQ, ElectricityHourly.PREDICTION_LENGTH, ElectricityHourly.SEASONALITY) == (
        "h",
        24,
        24,
    )


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
