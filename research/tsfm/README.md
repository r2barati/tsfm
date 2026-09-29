# TSFM research and experiments

This directory contains the TSFM literature review, research map, and two prospective forecasting studies.

## Studies

- [`prospective_cpu/`](prospective_cpu/README.md) is the Ontario demand and GitHub activity pilot. It includes the frozen manifests, timestamped source snapshots, historical forecasts, scores, quality checks, runner, and Windows scheduler artifacts.
- [`prospective_v2/`](prospective_v2/) is a Git submodule linked to the separately preregistered cross-domain EIA-930 and GHCN-Daily study: [r2barati/tsfm-prospective-v2](https://github.com/r2barati/tsfm-prospective-v2). Its protocol, model/data manifests, code, reports, and public releases remain versioned in that repository.

## Research notes

The surrounding Markdown, BibTeX, and CSV files track the model taxonomy, benchmark audit, literature review, research opportunities, open problems, and 2026 state of the art.

## Data and generated files

The pilot's source snapshots, non-empty recorded execution logs, and completed development outputs are included for replay. Large raw v2 upstream archives, model caches, and the active v2 backtest's unfinished runtime files remain excluded according to the v2 repository's ignore rules; their source locations and hashes are recorded in its manifests. Re-download upstream data from the cited official sources when replaying that study. Third-party dataset and checkpoint terms remain separate from the code license.

The exported pilot Task Scheduler XML has its machine-specific workspace path replaced with `%PROSPECTIVE_CPU_ROOT%`; use `prospective_cpu/scripts/register_task.ps1` to create a task for a local checkout.
