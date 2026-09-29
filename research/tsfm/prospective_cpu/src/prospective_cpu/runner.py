from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import (MANIFEST, OUTCOMES, REPORTS, RESULTS, RUNS, SCHEDULE, SNAPSHOTS,
                     read_study, second_tuesdays, write_json)
from .data import (compute_shift_rules, filter_complete_through, load_github_snapshot,
                   load_ieso_snapshot, shift_label)
from .forecasters import MODEL_NAMES, ModelRunner
from .metrics import bootstrap_rank_stability, rank_summary, score_forecast_rows
from .sources import collect_source_snapshot, repositories_from_manifest, utc_now

TORONTO = ZoneInfo("America/Toronto")


def _hash_values(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError("Frozen study manifest missing; run `tsfm-prospective freeze` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _latest_freeze() -> Path:
    return SNAPSHOTS / _load_manifest()["freeze_snapshot_id"]


def _model_list(domain: str, include_ttm: bool = True) -> list[str]:
    names = list(MODEL_NAMES)
    if include_ttm and domain == "ieso":
        names.append("ttm_512_week_context")
    return names


def _series_at(directory: Path, as_of: datetime) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    ieso, ieso_qa = load_ieso_snapshot(directory)
    gh, gh_qa = load_github_snapshot(directory, as_of=as_of)
    series = {"ieso:ontario-demand": filter_complete_through(ieso, as_of.astimezone(TORONTO).date())}
    series.update({f"github:{repo}": filter_complete_through(values, as_of.astimezone(TORONTO).date())
                   for repo, values in gh.items()})
    return series, {"ieso": ieso_qa, "github": gh_qa}


def _write_input_rows(path: Path, series: dict[str, pd.Series], cutoff: datetime) -> dict[str, np.ndarray]:
    records = []
    histories = {}
    for series_id, values in series.items():
        values = values.sort_index()
        if not len(values) or values.isna().any():
            continue
        weeks = list(values.index)
        if any((weeks[index] - weeks[index - 1]).days != 7 for index in range(1, len(weeks))):
            continue
        data = values.to_numpy(dtype=float)
        histories[series_id] = data
        for week, value in zip(weeks, data):
            records.append({"series_id": series_id, "week_start": week.isoformat(), "value": float(value),
                            "week_end": (week + timedelta(days=6)).isoformat(),
                            "cutoff_timestamp": cutoff.isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return histories


def _continuous_histories(series: dict[str, pd.Series], minimum: int) -> dict[str, np.ndarray]:
    histories = {}
    for series_id, values in series.items():
        values = values.sort_index()
        if len(values) < minimum or values.isna().any():
            continue
        weeks = list(values.index)
        if any((weeks[index] - weeks[index - 1]).days != 7 for index in range(1, len(weeks))):
            continue
        histories[series_id] = values.to_numpy(dtype=float)
    return histories


def _forecast_series(model_runner: ModelRunner, series_id: str, history: np.ndarray,
                     last_week: date, origin_id: str, domain: str, *, season_length: int,
                     horizons: tuple[int, ...] = (1, 4)) -> list[dict[str, Any]]:
    longest = max(horizons)
    result = model_runner.predict(history, longest, season_length=season_length)
    rows = []
    for horizon in horizons:
        idx = horizon - 1
        rows.append({"series_id": series_id, "domain": domain, "model": model_runner.name,
                     "origin_id": origin_id, "horizon_weeks": horizon,
                     "input_last_week_start": last_week.isoformat(),
                     "input_last_week_end": (last_week + timedelta(days=6)).isoformat(),
                     "target_week_start": (last_week + timedelta(days=7 * horizon)).isoformat(),
                     "n_input_weeks": int(len(history)), "input_hash": _hash_values(history),
                     "input_values_json": json.dumps(history.tolist(), separators=(",", ":")),
                     "point": float(result.point[idx]), "q10": float(result.q10[idx]),
                     "q50": float(result.q50[idx]), "q90": float(result.q90[idx]),
                     "probabilistic": bool(result.probabilistic)})
    return rows


def _checkpoint_records(model_runs: list[ModelRunner]) -> dict[str, Any]:
    return {model.name: {"model_repo": model.manifest.get("model_revisions", {}).get(model.name, {}).get("repo_id"),
                         "revision": model.revision,
                         "state_dict_sha256": model.checkpoint_hash}
            for model in model_runs}


def run_cutoff(*, as_of: datetime | None = None, confirmatory: bool = False, include_ttm: bool = True) -> Path:
    manifest = _load_manifest()
    study = read_study()
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("Cutoff timestamp must include a timezone")
    actual_now = datetime.now(TORONTO)
    if confirmatory and as_of is not None and abs((actual_now - as_of.astimezone(TORONTO)).total_seconds()) > 120:
        raise ValueError("Confirmatory runs must use the live cutoff timestamp; historical --as-of values are development only")
    as_of = actual_now if confirmatory else (as_of or actual_now)
    local_as_of = as_of.astimezone(TORONTO)
    if confirmatory and actual_now.date().isoformat() not in manifest["cutoffs"]:
        raise ValueError(f"{local_as_of.date()} is not one of the 12 preregistered cutoff dates")
    role = "confirmatory" if confirmatory else "development"
    origin_id = local_as_of.strftime("%Y-%m-%d")
    run_dir = RUNS / role / origin_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable run already exists: {run_dir}")
    source_id = "origin_" + local_as_of.strftime("%Y%m%dT%H%M%S%z")
    source_dir, source_manifest = collect_source_snapshot(source_id)
    series, qa = _series_at(source_dir, local_as_of)
    shift_path = RESULTS / "shift_rules.json"
    if not shift_path.exists():
        setup_series, _ = _series_at(_latest_freeze(), datetime.fromisoformat(manifest["frozen_at_local"]))
        rules = compute_shift_rules(setup_series)
        write_json(shift_path, {"fitted_at": manifest["frozen_at_local"], "rules": rules})
    shift_rules = json.loads(shift_path.read_text(encoding="utf-8"))["rules"]
    min_context = int(study["minimum_context_weeks"])
    eligible = _continuous_histories(series, min_context)
    if not eligible:
        raise RuntimeError(f"No continuous series has the prespecified minimum context of {min_context} complete weeks")
    run_dir.mkdir(parents=True, exist_ok=False)
    input_file = run_dir / "series_inputs.csv"
    _write_input_rows(input_file, {key: series[key].loc[:list(series[key].dropna().index)[-1]]
                                   for key in eligible}, local_as_of)
    # Keep a byte-for-byte source snapshot in the run directory, so the run can be replayed without network access.
    run_source_dir = run_dir / "source_snapshot"
    import shutil
    shutil.copytree(source_dir, run_source_dir)
    forecast_rows: list[dict[str, Any]] = []
    checkpoint_info = {}
    for domain in ("ieso", "github"):
        domain_series = {key: values for key, values in eligible.items() if key.startswith(domain + ":")}
        if not domain_series:
            continue
        for model_name in _model_list(domain, include_ttm=include_ttm):
            if model_name == "ttm_512_week_context" and len(domain_series.get("ieso:ontario-demand", [])) < 512:
                logging.warning("Skipping TTM; IESO has fewer than 512 complete input weeks")
                continue
            runner = ModelRunner(model_name, manifest)
            try:
                if model_name not in ("seasonal_naive", "autoets"):
                    runner.load()
                checkpoint_info[model_name] = {"repo_id": manifest.get("model_revisions", {}).get(model_name, {}).get("repo_id"),
                                               "revision": runner.revision,
                                               "state_dict_sha256": runner.checkpoint_hash}
                for series_id, history in domain_series.items():
                    last_week = list(series[series_id].dropna().index)[-1]
                    season_length = int(study["domains"][domain]["season_length"])
                    try:
                        forecast_rows.extend(_forecast_series(runner, series_id, history, last_week, origin_id,
                                                              domain, season_length=season_length))
                    except Exception as exc:
                        logging.exception("Forecast failed: model=%s series=%s", model_name, series_id)
                        forecast_rows.append({"series_id": series_id, "domain": domain, "model": model_name,
                                              "origin_id": origin_id, "horizon_weeks": 0, "error": str(exc)})
            finally:
                runner.close()
    forecast_frame = pd.DataFrame(forecast_rows)
    valid_rows = forecast_frame[forecast_frame["horizon_weeks"].isin([1, 4])].copy() if not forecast_frame.empty else forecast_frame
    if valid_rows.empty:
        raise RuntimeError("Every model forecast failed; inspect run log")
    cutoff_day = local_as_of.date()
    if not ((pd.to_datetime(valid_rows["input_last_week_end"]).dt.date < cutoff_day).all()):
        raise AssertionError("Leakage guard failed: an input week ends on/after the cutoff date")
    # Shift labels are attached after inference only; no outcomes or targets entered the model call.
    valid_rows["shift_reference"] = "frozen_2026-09-27_q01_q99"
    valid_rows["training_input_as_of"] = local_as_of.isoformat()
    valid_rows.to_csv(run_dir / "forecasts.csv", index=False)
    forecast_errors = forecast_frame[forecast_frame.get("horizon_weeks", pd.Series(dtype=int)) == 0]
    write_json(run_dir / "run_manifest.json", {
        "origin_id": origin_id, "role": role, "as_of_timestamp": local_as_of.isoformat(),
        "as_of_timezone": "America/Toronto", "source_snapshot_id": source_id,
        "source_snapshot_acquired_at_utc": source_manifest.get("created_at_utc"),
        "source_snapshot_manifest": source_manifest, "data_qa": qa,
        "forecast_input_file": "series_inputs.csv", "forecast_file": "forecasts.csv",
        "models": checkpoint_info, "minimum_context_weeks": min_context,
        "forecast_error_count": int(len(forecast_errors)),
        "forecast_errors": forecast_errors[["series_id", "model", "error"]].to_dict("records")
        if not forecast_errors.empty else [],
        "horizons_weeks": [1, 4], "target_values_loaded_for_inference": False,
        "all_inference_parameters_cpu": True, "shift_rule_ids": list(shift_rules),
        "created_at_utc": utc_now(),
    })
    return run_dir


def _outcomes_at(now_local: datetime) -> tuple[Path, dict[str, pd.Series], dict[str, Any]]:
    out_id = "score_" + now_local.strftime("%Y%m%dT%H%M%S%z")
    out_dir, snapshot_manifest = collect_source_snapshot(out_id, destination=OUTCOMES / out_id)
    series, qa = _series_at(out_dir, now_local)
    write_json(out_dir / "outcome_qa.json", qa)
    return out_dir, series, snapshot_manifest


def score_completed(*, now: datetime | None = None) -> Path:
    now = now or datetime.now(TORONTO)
    outcome_dir, actuals, outcome_manifest = _outcomes_at(now)
    shift_rules_path = RESULTS / "shift_rules.json"
    shift_rules = json.loads(shift_rules_path.read_text(encoding="utf-8"))["rules"] if shift_rules_path.exists() else None
    rows = []
    cutoff_status = []
    for run_dir in sorted(RUNS.glob("confirmatory/*")):
        if not run_dir.is_dir() or not (run_dir / "forecasts.csv").exists():
            continue
        forecasts = pd.read_csv(run_dir / "forecasts.csv")
        scoreable = score_forecast_rows(forecasts, actuals, shift_rules=shift_rules)
        if not scoreable.empty:
            scoreable["cutoff"] = run_dir.name
            scoreable["outcome_snapshot"] = outcome_dir.name
            rows.extend(scoreable.to_dict("records"))
        cutoff_status.append({"cutoff": run_dir.name, "forecast_exists": True,
                              "scored_groups": 0 if scoreable.empty else int(len(scoreable))})
    score_frame = pd.DataFrame(rows)
    score_id = now.strftime("%Y%m%dT%H%M%S%z")
    score_dir = RESULTS / "scores" / score_id
    score_dir.mkdir(parents=True, exist_ok=False)
    score_frame.to_csv(score_dir / "scores.csv", index=False)
    ranks = rank_summary(score_frame.rename(columns={"cutoff": "origin_id"})) if not score_frame.empty else pd.DataFrame()
    ranks.to_csv(score_dir / "rank_summary.csv", index=False)
    write_json(score_dir / "score_manifest.json", {"scored_at": now.isoformat(),
                "outcome_snapshot": str(outcome_dir),
                "outcome_snapshot_acquired_at_utc": outcome_manifest.get("created_at_utc"),
                "outcome_snapshot_manifest": outcome_manifest,
                "confirmatory_cutoffs_found": cutoff_status, "score_rows": len(score_frame)})
    return score_dir


def replay_score(score_id: str) -> dict[str, Any]:
    score_dir = RESULTS / "scores" / score_id
    score_manifest = json.loads((score_dir / "score_manifest.json").read_text(encoding="utf-8"))
    outcome_dir = Path(score_manifest["outcome_snapshot"])
    scored_at = datetime.fromisoformat(score_manifest["scored_at"])
    actuals, qa = _series_at(outcome_dir, scored_at)
    shift_rules_path = RESULTS / "shift_rules.json"
    shift_rules = json.loads(shift_rules_path.read_text(encoding="utf-8"))["rules"] if shift_rules_path.exists() else None
    rows = []
    for item in score_manifest["confirmatory_cutoffs_found"]:
        run_dir = RUNS / "confirmatory" / item["cutoff"]
        forecast_path = run_dir / "forecasts.csv"
        if not forecast_path.exists():
            continue
        frame = score_forecast_rows(pd.read_csv(forecast_path), actuals, shift_rules=shift_rules)
        if not frame.empty:
            frame["cutoff"] = run_dir.name
            frame["outcome_snapshot"] = outcome_dir.name
            rows.extend(frame.to_dict("records"))
    replay = pd.DataFrame(rows)
    expected_path = score_dir / "scores.csv"
    expected = pd.read_csv(expected_path) if expected_path.stat().st_size else pd.DataFrame()
    keys = ["series_id", "model", "horizon_weeks", "origin_id", "cutoff"]
    expected = expected.sort_values(keys).reset_index(drop=True) if not expected.empty else expected
    replay = replay.sort_values(keys).reset_index(drop=True) if not replay.empty else replay
    if list(expected.columns) != list(replay.columns) or len(expected) != len(replay):
        raise AssertionError("Replay does not reproduce the saved score table shape/columns")
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(expected[column].to_numpy(dtype=float), replay[column].to_numpy(dtype=float),
                               equal_nan=True, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"Replay mismatch in metric column {column}")
        elif not expected[column].astype(str).equals(replay[column].astype(str)):
            raise AssertionError(f"Replay mismatch in column {column}")
    result = {"score_id": score_id, "reproduced": True, "score_rows": len(expected), "outcome_qa": qa}
    write_json(score_dir / "replay_verification.json", result)
    return result


def run_development_backtest(*, origins: int = 12, include_ttm: bool = True) -> Path:
    """Retrospective development only. Each origin uses a prefix; later values are scoring-only."""
    manifest = _load_manifest()
    setup_dir = _latest_freeze()
    setup_as_of = datetime.fromisoformat(manifest["frozen_at_local"])
    series, qa = _series_at(setup_dir, setup_as_of)
    study = read_study()
    dev_dir = RESULTS / "development" / f"backtest_{datetime.now(TORONTO).strftime('%Y%m%dT%H%M%S%z')}"
    dev_dir.mkdir(parents=True, exist_ok=False)
    dev_rows, score_actuals = [], {}
    checkpoint_info: dict[str, Any] = {}
    origin_specs: list[tuple[str, dict[str, np.ndarray], dict[str, date]]] = []
    for domain in ("ieso", "github"):
        chosen = {k: v.dropna() for k, v in series.items() if k.startswith(domain + ":") and len(v.dropna()) >= 32}
        if not chosen:
            continue
        common_weeks = sorted(set.intersection(*(set(s.index) for s in chosen.values())))
        min_ctx = 52 if domain == "ieso" else 26
        last_index = len(common_weeks) - 5
        first_index = min_ctx - 1
        if last_index < first_index:
            continue
        positions = np.unique(np.linspace(first_index, last_index, min(origins, last_index - first_index + 1), dtype=int))
        for position in positions:
            origin_week = common_weeks[position]
            target_weeks = common_weeks[position + 1:position + 5]
            if len(target_weeks) != 4:
                continue
            input_values = {}
            last_weeks = {}
            for sid, s in chosen.items():
                prefix = s.loc[:origin_week]
                if len(prefix) >= min_ctx:
                    input_values[sid] = prefix.to_numpy(dtype=float)
                    last_weeks[sid] = origin_week
            origin_id = f"{domain}_{origin_week.isoformat()}"
            for model_name in _model_list(domain, include_ttm=include_ttm):
                if model_name == "ttm_512_week_context" and len(input_values.get("ieso:ontario-demand", [])) < 512:
                    continue
                runner = ModelRunner(model_name, manifest)
                try:
                    if model_name not in ("seasonal_naive", "autoets"):
                        runner.load()
                    checkpoint_info[model_name] = {
                        "repo_id": manifest.get("model_revisions", {}).get(model_name, {}).get("repo_id"),
                        "revision": runner.revision,
                        "state_dict_sha256": runner.checkpoint_hash,
                    }
                    for sid, history in input_values.items():
                        if sid.startswith("github:") and model_name == "ttm_512_week_context":
                            continue
                        try:
                            dev_rows.extend(_forecast_series(runner, sid, history, last_weeks[sid], origin_id,
                                                             domain, season_length=study["domains"][domain]["season_length"]))
                        except Exception as exc:
                            logging.exception("Development forecast failed %s %s %s", model_name, sid, origin_id)
                    score_actuals.update({sid: series[sid] for sid in input_values})
                finally:
                    runner.close()
    all_rows = pd.DataFrame(dev_rows)
    all_rows.to_csv(dev_dir / "forecasts.csv", index=False)
    # Development scores are retrospective from the frozen setup vintage and are not confirmatory.
    scores = score_forecast_rows(all_rows, score_actuals) if not all_rows.empty else pd.DataFrame()
    scores.to_csv(dev_dir / "scores.csv", index=False)
    (dev_dir / "qa.json").write_text(json.dumps(qa, indent=2, default=str), encoding="utf-8")
    write_json(dev_dir / "backtest_manifest.json", {"role": "development_only", "origins_requested": origins,
                "origin_rows": int(all_rows["origin_id"].nunique()) if not all_rows.empty else 0,
                "origin_ids": sorted(all_rows["origin_id"].unique().tolist()) if not all_rows.empty else [],
                "frozen_setup_vintage": manifest["frozen_at_local"], "gpu_used": False,
                "models": checkpoint_info,
                "known_limitation": "GitHub statistics are a current one-year API vintage; revisions may affect retrospective values."})
    return dev_dir


def make_report() -> Path:
    manifest = _load_manifest()
    planned = manifest["cutoffs"]
    scored_files = sorted((RESULTS / "scores").glob("*/scores.csv")) if (RESULTS / "scores").exists() else []
    current_scores = pd.read_csv(scored_files[-1]) if scored_files and Path(scored_files[-1]).stat().st_size else pd.DataFrame()
    dev_score_files = sorted((RESULTS / "development").glob("backtest_*/scores.csv")) if (RESULTS / "development").exists() else []
    dev = pd.read_csv(dev_score_files[-1]) if dev_score_files and Path(dev_score_files[-1]).stat().st_size else pd.DataFrame()
    dev_rank = rank_summary(dev) if not dev.empty else pd.DataFrame()
    prospective_dirs = {p.name: p for p in (RUNS / "confirmatory").glob("*") if p.is_dir()} if (RUNS / "confirmatory").exists() else {}
    prospective_manifests: dict[str, dict[str, Any]] = {}
    for cutoff, run_dir in prospective_dirs.items():
        manifest_path = run_dir / "run_manifest.json"
        forecast_path = run_dir / "forecasts.csv"
        if manifest_path.exists() and forecast_path.exists():
            prospective_manifests[cutoff] = json.loads(manifest_path.read_text(encoding="utf-8"))
    prospective_exists = set(prospective_manifests)
    rows = []
    for cutoff in planned:
        if cutoff not in prospective_dirs:
            state = "pending"
        elif cutoff not in prospective_manifests:
            state = "missing (incomplete run)"
        else:
            errors = int(prospective_manifests[cutoff].get("forecast_error_count", 0))
            state = f"partial ({errors} model failures)" if errors else "forecast recorded"
        rows.append(f"| {cutoff} | {state} |")
    report = [
        "# CPU-only prospective TSFM evaluation",
        "",
        f"Generated: {datetime.now(TORONTO).isoformat(timespec='minutes')} (America/Toronto)",
        "",
        "## Confirmatory status",
        "",
        f"Forward-only cutoffs recorded: **{len(prospective_exists)}/{len(planned)}**. No future result is inferred or backfilled.",
        "",
        "| Scheduled cutoff | Status |",
        "|---|---|",
        *rows,
        "",
        "## Historical development evidence",
        "",
        "Historical rankings are development evidence only. GitHub activity is retrieved from a rolling 52-week endpoint and its past values may be revised; it is not treated as a point-in-time archive.",
        "",
    ]
    if dev_rank.empty:
        report.append("No development backtest has been run yet.")
    else:
        report.extend(["| Domain | Horizon (weeks) | Model | Mean MASE | Mean rank |", "|---|---:|---|---:|---:|"])
        for row in dev_rank.sort_values(["domain", "horizon_weeks", "mean_rank"]).itertuples(index=False):
            report.append(f"| {row.domain} | {row.horizon_weeks} | {row.model} | {row.mean_mase:.4f} | {row.mean_rank:.2f} |")
    report.extend(["", "## Prospective scores", ""])
    if current_scores.empty:
        report.append("No scheduled forecast has a completed 1- or 4-week target yet.")
    else:
        current_scores["domain"] = current_scores["series_id"].map(lambda x: "github" if x.startswith("github:") else "ieso")
        grouped = current_scores.groupby(["domain", "horizon_weeks", "model"], as_index=False)["mase"].mean()
        grouped["rank"] = grouped.groupby(["domain", "horizon_weeks"])["mase"].rank(method="average")
        report.extend(["| Domain | Horizon (weeks) | Model | Mean MASE | Rank |", "|---|---:|---|---:|---:|"])
        for row in grouped.sort_values(["domain", "horizon_weeks", "rank"]).itertuples(index=False):
            report.append(f"| {row.domain} | {row.horizon_weeks} | {row.model} | {row.mase:.4f} | {row.rank:.2f} |")
        if not dev.empty and "cutoff" in current_scores:
            report.extend(["", "### Historical-to-prospective rank stability", "",
                           "Spearman correlation of the frozen historical model ordering with the prospective ordering; 95% intervals resample series and origins/cutoffs as blocks.",
                           "", "| Domain | Horizon | Rank correlation | 95% block-bootstrap interval |", "|---|---:|---:|---:|"])
            for domain in ("ieso", "github"):
                for horizon in (1, 4):
                    estimate, low, high = bootstrap_rank_stability(dev, current_scores, domain=domain, horizon=horizon)
                    if np.isfinite(estimate):
                        report.append(f"| {domain} | {horizon} | {estimate:.3f} | [{low:.3f}, {high:.3f}] |")
                    else:
                        report.append(f"| {domain} | {horizon} | pending | pending |")
    report.extend(["", "Scores are computed only after actual target weeks become complete in a later source snapshot. MASE uses a one-week naïve scale. The weekly seasonal-naïve/AutoETS periods are fixed at 52 for IESO and 4 for GitHub to respect the GitHub history limit. Probabilistic columns are scored only when a model provides them.", ""])
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "results.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def register_task() -> dict[str, Any]:
    import subprocess
    script = Path(__file__).resolve().parents[2] / "scripts" / "register_task.ps1"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Task Scheduler registration failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
