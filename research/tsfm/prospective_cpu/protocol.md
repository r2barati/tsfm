# Frozen study protocol

## Question and scope

Test whether the ranking of five zero-shot forecasting methods in a historical development backtest predicts their ranking on future observations. The confirmatory claim applies only to these frozen model checkpoints, the Ontario demand series and selected public repositories, weekly aggregation, and 1- and 4-week horizons. It does not claim that the result generalizes to all TSFMs or domains.

The five shared methods are Chronos-Bolt Tiny, Chronos-T5 Tiny, Chronos-2 Small, seasonal naive, and AutoETS. The supplemental TTM lane uses one fixed 512-week-context weekly checkpoint on IESO only. TTM ranks are never pooled with GitHub.

## Data and point-in-time rules

### Ontario demand

Use the IESO annual `PUB_Demand_<YEAR>.csv` source, 2002 through the current year, and the `Ontario Demand` column. Each input record is an IESO market hour ending label. The archived format carries 24 records per local market date, including spring and fall DST transition dates; preserve these market labels and do not fabricate missing/duplicate civil-clock hours. Assign each record to the Ontario calendar date and group dates into Sunday–Saturday weeks. A weekly mean is eligible when all seven dates are present, each date has at least 23 and at most 24 hourly records, and the week has at least 167 of 168 records. For a 167-record week, average the 167 observed values without imputation and flag the missing hourly record. Otherwise keep the week missing. Exclude the partial current week at each cutoff.

### GitHub activity

At setup, query public GitHub Search by language with `stars:>10000` and an explicit descending-star sort, then verify primary language, archive/fork status, and exact star count through the repository REST endpoint. Freeze the first five eligible results in Python, JavaScript, Go, and Rust, including raw search pages, rank, and star count. At each origin, fetch `stats/commit_activity` for only those repositories. Validate seven nonnegative daily counts and `total == sum(days)`. The API's maximum one-year lookback constrains history. Its seven daily counts start on Sunday; canonicalize the epoch to its Sunday date, including the observed Saturday 23:00 UTC anchors that denote the following Sunday. Check normalized buckets for seven-day spacing. Include a week only after its full Sunday–Saturday interval has elapsed; do not extrapolate partial-week counts. Preserve 0-commit weeks as zero and missing API weeks as missing.

The IESO series has annual season length 52. The GitHub baselines use fixed 4-week season length because their available history does not support an annual period with useful independent development origins. The model classes are the same across domains, but these domain-specific seasonal periods are declared up front.

### Time and leakage controls

Record the actual local cutoff timestamp, source acquisition time, weekly input end date, and snapshot hashes. Use only complete weeks ending before the scheduled cutoff date. The inference API receives only its input prefix; target actuals are opened only by `score`. Model-side guards check finite, continuous weekly histories, input hashes, exact final input week, CPU parameter devices, and CUDA unavailability. Forecasts and source snapshots are immutable.

Prospective dates are the second Tuesday at 09:00 America/Toronto beginning 13 October 2026, 12 dates total. `StartWhenAvailable` is enabled, but the wrapper records a forecast only within ten minutes of 09:00 on a frozen cutoff date. A delayed catch-up is logged and left missing; there is no backfilling. A failed or missed origin stays missing, with its reason in the task log/report.

## Model and compute controls

Run on the inspected Windows PC (Python 3.13.2, four CPU cores, 16 GiB RAM). Use the isolated `uv` lock and CPU-only PyTorch 2.10.0. Use float32; limit torch to four threads; load one neural checkpoint at a time; set deterministic seeds for sample-based Chronos quantiles. No training, GPU, or hosted inference. Resolve Hugging Face commit SHAs before prospective collection; record the SHA-256 of every loaded state dictionary. TTM's fixed branch is `512-48-ft-r2.1`; its first four outputs provide the requested horizons.

Chronos checkpoints provide forecast distributions/quantiles. AutoETS contributes its model-based 80% prediction interval where available. Seasonal naive is point-only. Report pinball loss and 80% interval coverage only when the forecast method supplies intervals/quantiles; do not synthesize uncertainty for point-only methods.

## Historical development phase

Use the frozen setup vintage only to implement and develop the analysis. Select up to 12 evenly spaced rolling origins over the available history, with at least 52 complete IESO input weeks or 26 GitHub weeks and four fully observed target weeks. Never pass targets to the forecaster. These retrospective results determine no prospective model changes. GitHub historical counts are a current rolling API vintage, potentially revised; this limitation is reported and those results are not treated as point-in-time evidence.

At setup, freeze a per-series shift rule from historical values: transform GitHub commit totals with `log1p`, leave IESO values on the original scale, and label a future weekly observation as shifted if it lies outside the setup-vintage 1st–99th percentile interval. This label is descriptive and does not affect forecasts, model choice, or the primary ranking.

## Confirmatory analysis

- Primary metric: MASE using a one-week naïve scale, computed from the origin's input history. Lower is better.
- Report MASE and model ranks separately for each domain and horizon; summarize repository means and each cutoff separately. Do not pool IESO and GitHub.
- Historical model ranks are frozen before the prospective phase. Compare them with the prospective model order at each cutoff using Spearman rank correlation. Report a 95% two-way block-bootstrap interval by resampling repositories (series for IESO) and development origins/cutoffs.
- Report pinball loss and 80% interval coverage only for methods that provide predictive distributions or intervals.
- Report shift labels using only the frozen setup thresholds. Report missing outcomes, API gaps, failed model runs, and missed cutoffs explicitly.
- Forecast targets are scored only after a later source snapshot contains the full target week. Every score snapshot includes the immutable outcome source vintage and can be recomputed with `replay-score`.

## Interpretation limits

This is a prospective measurement protocol, not a guarantee of model availability or uninterrupted internet connectivity. The confirmatory study is incomplete until its 12 scheduled origins and their 1-/4-week outcomes have been collected. No results should be described as confirmatory before then. Public repository selection, IESO revisions, API outages, model licensing/availability, and CPU run failures are recorded as data limitations rather than silently repaired.
