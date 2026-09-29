from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY_FILE = ROOT / "study.yaml"
DATA = ROOT / "data"
SNAPSHOTS = DATA / "snapshots"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
OUTCOMES = RESULTS / "outcomes"
SCORES = RESULTS / "scores"
REPORTS = ROOT / "reports"
SCHEDULE = ROOT / "schedule"
MANIFEST = ROOT / "manifests" / "study_manifest.json"


def read_study() -> dict:
    return yaml.safe_load(STUDY_FILE.read_text(encoding="utf-8"))


def second_tuesdays(start: date, count: int) -> list[date]:
    result: list[date] = []
    year, month = start.year, start.month
    while len(result) < count:
        first = date(year, month, 1)
        days_to_tuesday = (1 - first.weekday()) % 7
        second_tuesday = first + timedelta(days=days_to_tuesday + 7)
        if second_tuesday >= start:
            result.append(second_tuesday)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
