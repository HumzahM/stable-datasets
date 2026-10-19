"""Forecasting example: reproduce Chronos (Ansari et al., 2024) Table 8 MASE scores.

    python examples/forecasting.py                      # Seasonal Naive only (exact, seconds)
    python examples/forecasting.py --chronos mini base  # also Chronos-T5 (pip install chronos-forecasting)

Protocol (Chronos / GluonTS): forecast the last ``PREDICTION_LENGTH`` points of every ``test`` series from the
rest of it; MASE = mean absolute error of the point forecast divided by the in-sample seasonal-naive error, then
averaged over series. Seasonal Naive is deterministic, so it matches the paper exactly; Chronos forecasts the
median of 20 sample paths from the last 512 points (as in the paper), so it matches to sampling noise (the paper averages 3 seeds).
"""

import argparse

import numpy as np

from stable_datasets.timeseries import ElectricityHourly, UberTLC


#: Table 8 of the Chronos paper (arXiv:2403.07815): MASE per model.
PAPER = {
    "Electricity (Hourly)": {"seasonal_naive": 1.840, "mini": 1.348, "small": 1.477, "base": 1.590, "large": 1.439},
    "Uber TLC (Hourly)": {"seasonal_naive": 0.931, "mini": 0.689, "small": 0.677, "base": 0.673, "large": 0.670},
    "Uber TLC (Daily)": {"seasonal_naive": 1.378, "mini": 0.906, "small": 0.870, "base": 0.839, "large": 0.821},
}
#: name -> (load the test split, prediction length, MASE seasonality)
DATASETS = {
    "Electricity (Hourly)": (
        lambda: ElectricityHourly(split="test"),
        ElectricityHourly.PREDICTION_LENGTH,
        ElectricityHourly.SEASONALITY,
    ),
    "Uber TLC (Hourly)": (lambda: UberTLC("hourly", split="test"), *UberTLC.SETTINGS["hourly"][1:]),
    "Uber TLC (Daily)": (lambda: UberTLC("daily", split="test"), *UberTLC.SETTINGS["daily"][1:]),
}


def mase(series, forecast, horizon, season):
    train, target = series[:-horizon], series[-horizon:]
    scale = np.mean(np.abs(train[season:] - train[:-season]))
    return np.mean(np.abs(target - forecast)) / scale


def seasonal_naive(series, horizon, season):
    return np.resize(series[:-horizon][-season:], horizon)


def chronos_medians(pipeline, series, horizon, seed, context=512, batch=64):
    import torch

    torch.manual_seed(seed)
    medians = []
    for i in range(0, len(series), batch):
        contexts = [torch.tensor(s[:-horizon][-context:]) for s in series[i : i + batch]]
        samples = pipeline.predict(contexts, prediction_length=horizon, num_samples=20)
        medians.append(samples.float().median(dim=1).values.numpy())
    return np.concatenate(medians)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chronos", nargs="*", default=[], choices=["mini", "small", "base", "large"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16, help="series per Chronos call")
    args = parser.parse_args()

    data = {}
    for name, (load, horizon, season) in DATASETS.items():
        ds = load()
        data[name] = (horizon, season, [np.asarray(ds[i]["series"], np.float64) for i in range(len(ds))])

    def report(model, scores):
        for name, value in scores.items():
            paper = PAPER[name][model]
            print(f"{model:15s} {name:22s} MASE {value:.3f}   paper {paper:.3f}   diff {value - paper:+.3f}")

    report(
        "seasonal_naive",
        {
            name: np.mean([mase(s, seasonal_naive(s, h, m), h, m) for s in series])
            for name, (h, m, series) in data.items()
        },
    )
    for size in args.chronos:
        import torch
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        scores = {}
        for name, (h, m, series) in data.items():
            per_seed = []
            for seed in range(args.seeds):
                medians = chronos_medians(pipeline, series, h, seed, batch=args.batch_size)
                per_seed.append(np.mean([mase(s, f, h, m) for s, f in zip(series, medians)]))
            scores[name] = float(np.mean(per_seed))
        report(size, scores)


if __name__ == "__main__":
    main()
