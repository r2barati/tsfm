from __future__ import annotations

import numpy as np
import pandas as pd

from .data import shift_label


def mase(actual: np.ndarray, point: np.ndarray, history: np.ndarray, season_length: int = 1) -> float:
    actual = np.asarray(actual, dtype=float)
    point = np.asarray(point, dtype=float)
    history = np.asarray(history, dtype=float)
    if len(actual) != len(point) or len(history) <= season_length:
        return float("nan")
    scale = np.mean(np.abs(history[season_length:] - history[:-season_length]))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(actual - point)) / scale)


def pinball(actual: np.ndarray, quantile_values: np.ndarray, levels: tuple[float, ...] = (0.1, 0.5, 0.9)) -> float:
    actual = np.asarray(actual, dtype=float).reshape(-1, 1)
    forecast = np.asarray(quantile_values, dtype=float)
    if forecast.shape != (len(actual), len(levels)):
        return float("nan")
    q = np.asarray(levels, dtype=float).reshape(1, -1)
    err = actual - forecast
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual, lower, upper = map(lambda x: np.asarray(x, dtype=float), (actual, lower, upper))
    return float(np.mean((actual >= lower) & (actual <= upper)))


def score_forecast_rows(forecasts: pd.DataFrame, actual_by_series: dict[str, pd.Series],
                        shift_rules: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    scored = []
    for (series_id, model, horizon, origin_id), group in forecasts.groupby(
            ["series_id", "model", "horizon_weeks", "origin_id"], sort=False):
        if series_id not in actual_by_series:
            continue
        target_weeks = [pd.Timestamp(x).date() for x in group["target_week_start"]]
        actual_series = actual_by_series[series_id]
        if not all(day in actual_series.index and np.isfinite(actual_series.loc[day]) for day in target_weeks):
            continue
        actual = np.array([actual_series.loc[day] for day in target_weeks], dtype=float)
        point = group["point"].to_numpy(dtype=float)
        history = np.asarray(json_array(group.iloc[0]["input_values_json"]), dtype=float)
        quantiles = group[["q10", "q50", "q90"]].to_numpy(dtype=float)
        score = {"series_id": series_id, "model": model, "horizon_weeks": int(horizon),
                 "origin_id": origin_id, "n_targets": len(actual),
                 "mase": mase(actual, point, history, season_length=1),
                 "actual_mean": float(actual.mean()), "forecast_mean": float(point.mean())}
        if shift_rules is not None:
            labels = [shift_label(value, shift_rules.get(series_id)) for value in actual]
            score["shift_label"] = "shift" if "shift" in labels else ("in_range" if "unclassified" not in labels else "unclassified")
            score["shift_weeks"] = int(labels.count("shift"))
        if np.isfinite(quantiles).all():
            score["pinball_mean"] = pinball(actual, quantiles)
            score["coverage_80"] = interval_coverage(actual, quantiles[:, 0], quantiles[:, 2])
        else:
            score["pinball_mean"] = float("nan")
            score["coverage_80"] = float("nan")
        scored.append(score)
    return pd.DataFrame(scored)


def json_array(value: str) -> list[float]:
    import json
    return json.loads(value)


def rank_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["domain", "horizon_weeks", "model", "mean_mase", "mean_rank"])
    data = scores.copy()
    data["domain"] = data["series_id"].map(lambda value: "github" if value.startswith("github:") else "ieso")
    means = data.groupby(["domain", "horizon_weeks", "model"], as_index=False)["mase"].mean()
    means["rank"] = means.groupby(["domain", "horizon_weeks"])["mase"].rank(method="average", ascending=True)
    return means.rename(columns={"mase": "mean_mase", "rank": "mean_rank"})


def bootstrap_mean_difference(scores: pd.DataFrame, model_a: str, model_b: str,
                              *, repetitions: int = 2000, seed: int = 20260927) -> tuple[float, float, float]:
    """Paired two-way cluster bootstrap over repository/series and origin/cutoff."""
    subset = scores[scores["model"].isin([model_a, model_b])].copy()
    pivot = subset.pivot_table(index=["series_id", "origin_id"], columns="model", values="mase", aggfunc="mean")
    if model_a not in pivot or model_b not in pivot:
        return float("nan"), float("nan"), float("nan")
    pivot = pivot.dropna(subset=[model_a, model_b])
    if pivot.empty:
        return float("nan"), float("nan"), float("nan")
    diff = (pivot[model_a] - pivot[model_b]).rename("diff").reset_index()
    series = diff["series_id"].unique()
    origins = diff["origin_id"].unique()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen_series = rng.choice(series, size=len(series), replace=True)
        chosen_origins = rng.choice(origins, size=len(origins), replace=True)
        blocks = diff[diff["series_id"].isin(chosen_series) & diff["origin_id"].isin(chosen_origins)]
        if len(blocks):
            samples.append(float(blocks["diff"].mean()))
    observed = float(diff["diff"].mean())
    if not samples:
        return observed, float("nan"), float("nan")
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return observed, float(lo), float(hi)


def bootstrap_rank_stability(development: pd.DataFrame, prospective: pd.DataFrame, *,
                             domain: str, horizon: int, repetitions: int = 2000,
                             seed: int = 20260927) -> tuple[float, float, float]:
    """Mean cutoff-level rank correlation; resample series and origins/cutoffs as blocks."""
    from scipy.stats import spearmanr

    dev = development.copy()
    fut = prospective.copy()
    dev["domain"] = dev["series_id"].map(lambda value: "github" if value.startswith("github:") else "ieso")
    fut["domain"] = fut["series_id"].map(lambda value: "github" if value.startswith("github:") else "ieso")
    dev = dev[(dev["domain"] == domain) & (dev["horizon_weeks"] == horizon) &
              (~dev["model"].str.startswith("ttm_"))]
    fut = fut[(fut["domain"] == domain) & (fut["horizon_weeks"] == horizon) &
              (~fut["model"].str.startswith("ttm_"))]
    models = sorted(set(dev["model"]) & set(fut["model"]))
    if len(models) < 2 or dev.empty or fut.empty:
        return float("nan"), float("nan"), float("nan")

    def ranks(data: pd.DataFrame, unit_col: str, time_col: str,
              unit_weights: dict[str, int] | None = None,
              time_weights: dict[str, int] | None = None) -> pd.Series:
        subset = data[data["model"].isin(models)].copy()
        weights = np.ones(len(subset), dtype=float)
        if unit_weights is not None:
            weights *= subset[unit_col].map(unit_weights).fillna(0).to_numpy()
        if time_weights is not None:
            weights *= subset[time_col].map(time_weights).fillna(0).to_numpy()
        subset["_w"] = weights
        subset = subset[(subset["_w"] > 0) & np.isfinite(subset["mase"].to_numpy(dtype=float))]
        if subset.empty:
            return pd.Series(dtype=float)
        subset["_weighted_mase"] = subset["mase"] * subset["_w"]
        totals = subset.groupby("model")["_weighted_mase"].sum()
        weight_totals = subset.groupby("model")["_w"].sum()
        means = totals / weight_totals
        return means.rank(method="average", ascending=True)

    dev["origin_id"] = dev["origin_id"].astype(str)
    fut["cutoff"] = fut["cutoff"].astype(str)
    dev_units, dev_times = dev["series_id"].unique(), dev["origin_id"].unique()
    fut_units, fut_times = fut["series_id"].unique(), fut["cutoff"].unique()
    hist_rank = ranks(dev, "series_id", "origin_id")
    observed_cutoff_correlations = []
    for cutoff in fut_times:
        current = fut[fut["cutoff"] == cutoff]
        current_rank = ranks(current, "series_id", "cutoff")
        left, right = hist_rank.align(current_rank, join="inner")
        if len(left) >= 2:
            rho = float(spearmanr(left, right).statistic)
            if np.isfinite(rho):
                observed_cutoff_correlations.append(rho)
    if not observed_cutoff_correlations:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(observed_cutoff_correlations))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        du = rng.choice(dev_units, size=len(dev_units), replace=True)
        dt = rng.choice(dev_times, size=len(dev_times), replace=True)
        fu = rng.choice(fut_units, size=len(fut_units), replace=True)
        ft = rng.choice(fut_times, size=len(fut_times), replace=True)
        du_count = pd.Series(du).value_counts().to_dict()
        dt_count = pd.Series(dt).value_counts().to_dict()
        fu_count = pd.Series(fu).value_counts().to_dict()
        a = ranks(dev, "series_id", "origin_id", du_count, dt_count)
        cutoff_rhos = []
        for cutoff in ft:
            current = fut[fut["cutoff"] == cutoff]
            b = ranks(current, "series_id", "cutoff", fu_count, {cutoff: 1})
            left, right = a.align(b, join="inner")
            if len(left) >= 2:
                value = float(spearmanr(left, right).statistic)
                if np.isfinite(value):
                    cutoff_rhos.append(value)
        if cutoff_rhos:
            samples.append(float(np.mean(cutoff_rhos)))
    if not samples:
        return observed, float("nan"), float("nan")
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return observed, float(lo), float(hi)
