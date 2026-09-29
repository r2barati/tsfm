from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from prospective_cpu.data import parse_github_activity, parse_ieso_csv, shift_label
from prospective_cpu.metrics import bootstrap_rank_stability, mase, pinball
from prospective_cpu.runner import _continuous_histories, _forecast_series


def test_ieso_market_hour_parser_keeps_dst_labeled_hours() -> None:
    lines = [r"\Hourly Demand Report,,,", r"\Created at 2026-09-27 07:30:15,,,", r"\For 2024,,,",
             "Date,Hour,Market Demand,Ontario Demand"]
    for day in ("2024-03-10", "2024-11-03"):
        lines.extend(f"{day},{hour},20000,17000" for hour in range(1, 25))
    frame = parse_ieso_csv(("\n".join(lines) + "\n").encode())
    assert len(frame) == 48
    assert frame.groupby("date").size().tolist() == [24, 24]
    assert frame["hour_ending"].min() == 1
    assert frame["hour_ending"].max() == 24


def test_ieso_rejects_unexpected_hour_label() -> None:
    payload = b"Date,Hour,Market Demand,Ontario Demand\n2024-01-01,25,1,2\n"
    with pytest.raises(ValueError, match="hour-ending"):
        parse_ieso_csv(payload)


def test_github_weekly_activity_checks_daily_sums_and_omits_partial_week() -> None:
    sunday = int(datetime(2026, 9, 13, tzinfo=timezone.utc).timestamp())
    data = [{"week": sunday, "total": 7, "days": [1] * 7},
            {"week": sunday + 7 * 86400, "total": 7, "days": [1] * 7}]
    payload = json.dumps(data).encode()
    series, qa = parse_github_activity(payload, as_of=datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert len(series) == 2
    assert qa["weeks_complete"] == 2
    broken = [{**data[0], "total": 8}]
    with pytest.raises(ValueError, match="does not match"):
        parse_github_activity(json.dumps(broken).encode(), as_of=datetime(2026, 9, 27, tzinfo=timezone.utc))


def test_github_saturday_23_utc_anchor_maps_to_sunday_start() -> None:
    anchor = int(datetime(2025, 10, 4, 23, tzinfo=timezone.utc).timestamp())
    payload = json.dumps([{"week": anchor, "total": 7, "days": [1] * 7}]).encode()
    series, _ = parse_github_activity(payload, as_of=datetime(2025, 10, 12, tzinfo=timezone.utc))
    assert list(series.index) == [date(2025, 10, 5)]


def test_mase_and_pinball_known_values() -> None:
    assert mase(np.array([3.0]), np.array([1.0]), np.array([0.0, 2.0, 4.0])) == 1.0
    assert pinball(np.array([2.0]), np.array([[1.0, 2.0, 3.0]])) == pytest.approx(0.0666666667)


def test_rank_bootstrap_ignores_missing_targets_and_is_reproducible() -> None:
    models = {"a": 1.0, "b": 2.0, "c": 3.0}
    dev_rows = []
    future_rows = []
    for unit_idx in range(4):
        unit = f"github:repo-{unit_idx}"
        for origin_idx in range(3):
            for model, offset in models.items():
                value = offset + unit_idx / 10 + origin_idx / 100
                dev_rows.append({"series_id": unit, "model": model, "horizon_weeks": 1,
                                 "origin_id": f"dev-{origin_idx}", "mase": value})
                future_rows.append({"series_id": unit, "model": model, "horizon_weeks": 1,
                                    "cutoff": f"future-{origin_idx}", "mase": value})
    dev_rows[0]["mase"] = np.nan
    dev = pd.DataFrame(dev_rows)
    future = pd.DataFrame(future_rows)
    first = bootstrap_rank_stability(dev, future, domain="github", horizon=1, repetitions=50)
    second = bootstrap_rank_stability(dev, future, domain="github", horizon=1, repetitions=50)
    assert first == second
    assert first[0] == pytest.approx(1.0)
    assert np.isfinite(first[1]) and np.isfinite(first[2])


def test_shift_rule_is_frozen_and_scale_aware() -> None:
    rule = {"scale": "log1p", "lower_q01": 0.0, "upper_q99": 2.0}
    assert shift_label(100.0, rule) == "shift"
    assert shift_label(1.0, rule) == "in_range"


def test_continuity_guard_rejects_gaps_and_nan() -> None:
    weekly = pd.Series([1.0, 2.0, 3.0], index=[date(2026, 1, 4), date(2026, 1, 11), date(2026, 1, 25)])
    assert _continuous_histories({"gap": weekly}, 2) == {}
    weekly.iloc[1] = np.nan
    assert _continuous_histories({"nan": weekly}, 2) == {}


def test_forecast_wrapper_passes_only_the_prefix_and_uses_requested_horizons() -> None:
    class Spy:
        name = "seasonal_naive"
        def predict(self, history, horizon, *, season_length):
            assert np.array_equal(history, np.array([1.0, 2.0, 3.0, 4.0]))
            return type("Forecast", (), {"point": np.array([2.0, 3.0, 4.0, 1.0]),
                                         "q10": np.full(4, np.nan), "q50": np.full(4, np.nan),
                                         "q90": np.full(4, np.nan), "probabilistic": False})()
    rows = _forecast_series(Spy(), "test", np.array([1.0, 2.0, 3.0, 4.0]), date(2026, 1, 4),
                            "origin", "github", season_length=4)
    assert [row["horizon_weeks"] for row in rows] == [1, 4]
    assert [row["point"] for row in rows] == [2.0, 1.0]
