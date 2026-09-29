from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sunday_start(day: date) -> date:
    return day - timedelta(days=(day.weekday() + 1) % 7)


def parse_ieso_csv(payload: bytes) -> pd.DataFrame:
    from io import BytesIO

    frame = pd.read_csv(BytesIO(payload), comment="\\", dtype={"Date": str, "Hour": str})
    required = {"Date", "Hour", "Ontario Demand"}
    if not required.issubset(frame.columns):
        raise ValueError(f"IESO CSV missing required columns: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise").dt.date
    frame["hour_ending"] = pd.to_numeric(frame["Hour"], errors="raise")
    frame["value"] = pd.to_numeric(frame["Ontario Demand"], errors="raise")
    if frame[["date", "hour_ending", "value"]].isna().any().any():
        raise ValueError("IESO demand contains missing date/hour/value")
    if not frame["hour_ending"].between(1, 24).all():
        raise ValueError("IESO hour-ending labels must be in 1..24")
    if (frame["value"] < 0).any():
        raise ValueError("IESO demand must be nonnegative")
    return frame[["date", "hour_ending", "value"]]


def load_ieso_files(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = [parse_ieso_csv(path.read_bytes()) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    duplicate_dates = int(frame.duplicated(["date", "hour_ending"]).sum())
    if duplicate_dates:
        raise ValueError(f"Duplicate IESO date/hour records across source files: {duplicate_dates}")
    daily = frame.groupby("date", as_index=False).agg(n_records=("value", "size"), mean_value=("value", "mean"))
    bad_daily = daily.loc[daily["n_records"] != 24, "date"].astype(str).tolist()
    # IESO publishes 24 market-hour records on each local market date, including DST transition dates.
    frame["week_start"] = frame["date"].map(_sunday_start)
    daily["week_start"] = daily["date"].map(_sunday_start)
    daily_coverage = daily.groupby("week_start").agg(n_days=("date", "nunique"),
                                                       min_daily_records=("n_records", "min"),
                                                       max_daily_records=("n_records", "max"))
    weekly = frame.groupby("week_start").agg(value=("value", "mean"), n_records=("value", "size"),
                                               n_days=("date", "nunique"))
    weekly = weekly.join(daily_coverage[["min_daily_records", "max_daily_records"]])
    # Retain a week with one missing hourly point without imputing it.
    weekly["complete"] = ((weekly["n_days"] == 7) & (weekly["n_records"] >= 167) &
                          (weekly["min_daily_records"] >= 23) & (weekly["max_daily_records"] <= 24))
    full = weekly.loc[weekly["complete"], "value"].astype(float)
    expected_weeks = pd.date_range(min(weekly.index), max(weekly.index), freq="7D").date
    missing_weeks = sorted(set(expected_weeks) - set(full.index))
    qa = {"hourly_records": int(len(frame)), "daily_dates": int(len(daily)),
          "daily_record_count_min": int(daily["n_records"].min()),
          "daily_record_count_max": int(daily["n_records"].max()),
          "daily_dates_not_24_records": bad_daily,
          "weeks_total": int(len(weekly)), "weeks_eligible": int(weekly["complete"].sum()),
          "weeks_incomplete": int((~weekly["complete"]).sum()),
          "weeks_missing_or_incomplete": [str(day) for day in missing_weeks],
          "weeks_with_one_missing_hour_included": [str(day) for day in weekly.index[(weekly["n_records"] == 167) & weekly["complete"]]],
          "dst_probe": {str(day): int(daily.loc[daily["date"].astype(str) == day, "n_records"].sum())
                        for day in ("2024-03-10", "2024-11-03", "2025-03-09", "2025-11-02")}}
    return full, qa


def load_ieso_snapshot(directory: Path) -> tuple[pd.Series, dict[str, Any]]:
    paths = sorted((directory / "ieso").glob("PUB_Demand_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No IESO annual CSV files under {directory}")
    values, qa = load_ieso_files(paths)
    return values.sort_index(), qa


def parse_github_activity(payload: bytes, *, as_of: datetime | None = None) -> tuple[pd.Series, dict[str, Any]]:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("GitHub commit activity response must be a JSON list")
    rows: dict[date, int] = {}
    now_utc = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    for item in data:
        if not isinstance(item, dict) or not {"week", "total", "days"}.issubset(item):
            raise ValueError("Each GitHub activity week needs week, total, and days")
        week_timestamp = datetime.fromtimestamp(int(item["week"]), timezone.utc)
        # GitHub emits Sunday midnight; some older buckets encode the same Sunday as
        # Saturday 23:00Z. The API defines the seven `days` entries as Sunday-first.
        if week_timestamp.weekday() == 6:
            week = week_timestamp.date()
        elif week_timestamp.weekday() == 5 and week_timestamp.hour == 23 and week_timestamp.minute == 0:
            week = week_timestamp.date() + timedelta(days=1)
        else:
            raise ValueError(f"Unrecognized GitHub weekly anchor timestamp: {week_timestamp.isoformat()}")
        days = item["days"]
        if len(days) != 7 or any(int(x) < 0 for x in days):
            raise ValueError(f"GitHub daily activity must contain seven nonnegative counts ({week})")
        total = int(item["total"])
        if total != sum(map(int, days)):
            raise ValueError(f"GitHub weekly total does not match daily counts ({week})")
        if week + timedelta(days=7) <= now_utc:
            rows[week] = total
    if not rows:
        empty = pd.Series(dtype=float, name="value")
        return empty, {"weeks_returned": len(data), "weeks_complete": 0, "missing_weeks": []}
    idx = pd.date_range(min(rows), max(rows), freq="7D")
    values = pd.Series([float(rows.get(ts.date(), np.nan)) for ts in idx], index=idx.date, name="value")
    missing = [str(day) for day, value in values.items() if pd.isna(value)]
    qa = {"weeks_returned": len(data), "weeks_complete": len(rows), "missing_weeks": missing,
          "latest_complete_week": str(max(rows))}
    return values, qa


def load_github_snapshot(directory: Path, as_of: datetime | None = None) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    series: dict[str, pd.Series] = {}
    qa = {}
    for path in sorted((directory / "github").glob("*.json")):
        repo = path.stem.replace("__", "/")
        raw = path.read_bytes()
        if not raw:
            series[repo] = pd.Series(dtype=float, name="value")
            qa[repo] = {"empty_response": True, "weeks_complete": 0}
            continue
        try:
            values, record = parse_github_activity(raw, as_of=as_of)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            values, record = pd.Series(dtype=float, name="value"), {"invalid_response": True}
        series[repo] = values
        qa[repo] = record
    return series, qa


def filter_complete_through(series: pd.Series, cutoff_date: date) -> pd.Series:
    keep = [week for week in series.index if week + timedelta(days=6) < cutoff_date]
    return series.loc[keep]


def compute_shift_rules(series_by_id: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
    rules = {}
    for series_id, values in series_by_id.items():
        finite = np.asarray(values.dropna(), dtype=float)
        if not len(finite):
            continue
        transformed = np.log1p(np.maximum(finite, 0)) if series_id.startswith("github:") else finite
        rules[series_id] = {"scale": "log1p" if series_id.startswith("github:") else "identity",
                            "lower_q01": float(np.quantile(transformed, 0.01)),
                            "upper_q99": float(np.quantile(transformed, 0.99)),
                            "n_historical_weeks": int(len(transformed))}
    return rules


def shift_label(value: float, rule: dict[str, float] | None) -> str:
    if rule is None or not np.isfinite(value):
        return "unclassified"
    v = float(np.log1p(max(value, 0))) if rule["scale"] == "log1p" else float(value)
    return "shift" if v < rule["lower_q01"] or v > rule["upper_q99"] else "in_range"
