# CPU-only prospective TSFM study

This is a self-contained Python 3.13 project for H2 in `../research_opportunities.md`. It uses CPU inference only, never trains models, and freezes a narrow claim to three compact Chronos checkpoints, two statistical baselines, the selected public data, and the 12 prespecified cutoffs.

## Set up

Run these commands from this directory in PowerShell:

```powershell
uv python pin 3.13.2
uv lock
uv sync --locked --extra dev
uv run --locked tsfm-prospective freeze
uv run --locked tsfm-prospective backtest --origins 12
uv run --locked tsfm-prospective report
uv run --locked tsfm-prospective register-task
```

The lock pins CPU-only `torch==2.10.0+cpu`, compatible with the pinned TTM package. The environment lives in `.venv`; downloaded Hugging Face checkpoints are cached under `.cache/huggingface` by the scheduled wrapper. Do not activate a GPU or change the frozen model/data manifests after the first prospective cutoff.

## Commands

```powershell
uv run --locked tsfm-prospective freeze [--force]
uv run --locked tsfm-prospective run --as-of 2026-09-27
uv run --locked tsfm-prospective run --confirmatory
uv run --locked tsfm-prospective score
uv run --locked tsfm-prospective replay-score <score-snapshot-id>
uv run --locked tsfm-prospective report
```

`freeze` selects the 20 repositories by GitHub star rank, captures the setup data vintage, and resolves model revisions. A date supplied to `run` is development-only. Confirmatory mode rejects any date other than one of the 12 scheduled local dates and rejects backdated timestamps. It forecasts only complete weeks, saves model inputs before any outcomes are read, and stores the exact source snapshot in the run directory. `score` takes a later data snapshot and scores only target weeks now complete. Missing repositories, weeks, origins, and cutoffs remain missing.

The scheduler is current-user, least-privilege, and run-only-when-logged-on. `StartWhenAvailable` is enabled. Its wrapper requires an invocation on a frozen cutoff date within ten minutes of 09:00 local time; a delayed catch-up logs the missed cutoff and exits without making a forecast. Failures are written to `logs/` and Task Scheduler history.

## Results

- `manifests/study_manifest.json`: locked cohort, checkpoint revisions, data source hashes, and cutoff dates.
- `data/snapshots/`: byte-exact origin and setup inputs.
- `results/development/`: historical development forecasts and scores.
- `results/runs/confirmatory/`: prospective forecasts and per-run input/checkpoint manifests.
- `results/outcomes/` and `results/scores/`: later actual-data vintages, metric tables, and replay checks.
- `logs/`: timestamped command logs and scheduled-task transcripts linked from the run/score manifests.
- `reports/results.md`: concise status and results; before the first forecast it explicitly reports 0/12.

Details, hypotheses, exclusions, and analysis rules are in [protocol.md](protocol.md). The first prospective cutoff is 13 October 2026, so there are no prospective results at setup.
