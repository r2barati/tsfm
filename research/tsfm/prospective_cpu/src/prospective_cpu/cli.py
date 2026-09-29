from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import MANIFEST, ROOT, SCHEDULE, second_tuesdays, write_json
from .runner import make_report, register_task, replay_score, run_cutoff, run_development_backtest, score_completed
from .sources import freeze_study

TORONTO = ZoneInfo("America/Toronto")


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None or value.lower() == "now":
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = datetime.combine(date.fromisoformat(value), time(23, 59), tzinfo=TORONTO)
        return parsed.astimezone(TORONTO)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 'now', YYYY-MM-DD, or an ISO timestamp with timezone") from exc


def _setup_schedule() -> Path:
    from .config import read_study
    study = read_study()
    first = date.fromisoformat(str(study["prospective"]["start_date"]))
    dates = [day.isoformat() for day in second_tuesdays(first, int(study["prospective"]["count"]))]
    SCHEDULE.mkdir(parents=True, exist_ok=True)
    path = SCHEDULE / "cutoffs.json"
    write_json(path, {"timezone": study["timezone"], "local_time": study["prospective"]["local_time"],
                      "catch_up": study["prospective"]["catch_up"], "cutoffs": dates})
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsfm-prospective", description="CPU-only prospective TSFM study runner")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze repositories, checkpoint revisions, and setup data vintage")
    freeze.add_argument("--force", action="store_true", help="replace a previously frozen manifest and setup snapshot")
    run = commands.add_parser("run", help="forecast one live or explicitly dated development cutoff")
    run.add_argument("--as-of", default=None, help="now, YYYY-MM-DD, or ISO timestamp (dated runs are development)")
    run.add_argument("--confirmatory", action="store_true", help="require the live timestamp on a preregistered cutoff date")
    run.add_argument("--without-ttm", action="store_true", help="skip the supplemental IESO-only TTM model")
    backtest = commands.add_parser("backtest", help="run the retrospective development backtest")
    backtest.add_argument("--origins", type=int, default=12)
    backtest.add_argument("--without-ttm", action="store_true")
    commands.add_parser("score", help="score any completed forecast horizons with a new outcome snapshot")
    replay = commands.add_parser("replay-score", help="recompute one score snapshot from its archived inputs")
    replay.add_argument("score_id", help="timestamped directory name under results/scores")
    commands.add_parser("report", help="regenerate the concise study report")
    commands.add_parser("register-task", help="register the current-user 12-run Windows Task Scheduler task")
    commands.add_parser("prepare-schedule", help="write the frozen 12-date cutoff list")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.command}_{datetime.now(TORONTO).strftime('%Y%m%dT%H%M%S%z')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    try:
        if args.command == "freeze":
            path = freeze_study(force=args.force)
            _setup_schedule()
            print(f"Frozen setup snapshot: {path}")
            print(f"Study manifest: {MANIFEST}")
        elif args.command == "run":
            run_dir = run_cutoff(as_of=_parse_as_of(args.as_of), confirmatory=args.confirmatory,
                                 include_ttm=not args.without_ttm)
            run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            run_manifest["log_file"] = str(log_path.resolve())
            write_json(run_dir / "run_manifest.json", run_manifest)
            print(f"Forecast run: {run_dir}")
        elif args.command == "backtest":
            result = run_development_backtest(origins=args.origins, include_ttm=not args.without_ttm)
            manifest_path = result / "backtest_manifest.json"
            backtest_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            backtest_manifest["log_file"] = str(log_path.resolve())
            write_json(manifest_path, backtest_manifest)
            print(f"Development backtest: {result}")
        elif args.command == "score":
            result = score_completed()
            manifest_path = result / "score_manifest.json"
            score_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            score_manifest["log_file"] = str(log_path.resolve())
            write_json(manifest_path, score_manifest)
            print(f"Score snapshot: {result}")
        elif args.command == "replay-score":
            print(json.dumps(replay_score(args.score_id), indent=2))
        elif args.command == "report":
            print(f"Report: {make_report()}")
        elif args.command == "register-task":
            if not MANIFEST.exists():
                raise FileNotFoundError("Freeze the study before registering its scheduler")
            _setup_schedule()
            print(json.dumps(register_task(), indent=2))
        elif args.command == "prepare-schedule":
            print(f"Cutoff dates: {_setup_schedule()}")
        return 0
    except Exception as exc:
        logging.exception("Command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
