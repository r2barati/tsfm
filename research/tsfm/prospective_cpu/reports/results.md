# CPU-only prospective TSFM evaluation

Generated: 2026-09-27T14:42-04:00 (America/Toronto)

## Confirmatory status

Forward-only cutoffs recorded: **0/12**. No future result is inferred or backfilled.

| Scheduled cutoff | Status |
|---|---|
| 2026-10-13 | pending |
| 2026-11-10 | pending |
| 2026-12-08 | pending |
| 2027-01-12 | pending |
| 2027-02-09 | pending |
| 2027-03-09 | pending |
| 2027-04-13 | pending |
| 2027-05-11 | pending |
| 2027-06-08 | pending |
| 2027-07-13 | pending |
| 2027-08-10 | pending |
| 2027-09-14 | pending |

## Historical development evidence

Historical rankings are development evidence only. GitHub activity is retrieved from a rolling 52-week endpoint and its past values may be revised; it is not treated as a point-in-time archive.

| Domain | Horizon (weeks) | Model | Mean MASE | Mean rank |
|---|---:|---|---:|---:|
| github | 1 | chronos2_small | 1.5442 | 1.00 |
| github | 1 | autoets | 1.5537 | 2.00 |
| github | 1 | chronos_bolt_tiny | 1.5716 | 3.00 |
| github | 1 | chronos_t5_tiny | 1.5888 | 4.00 |
| github | 1 | seasonal_naive | 1.8650 | 5.00 |
| github | 4 | chronos2_small | 2.2622 | 1.00 |
| github | 4 | chronos_bolt_tiny | 2.2959 | 2.00 |
| github | 4 | autoets | 2.3569 | 3.00 |
| github | 4 | chronos_t5_tiny | 2.3833 | 4.00 |
| github | 4 | seasonal_naive | 2.5353 | 5.00 |
| ieso | 1 | chronos2_small | 0.9800 | 1.00 |
| ieso | 1 | autoets | 1.0143 | 2.00 |
| ieso | 1 | chronos_bolt_tiny | 1.0151 | 3.00 |
| ieso | 1 | ttm_512_week_context | 1.1239 | 4.00 |
| ieso | 1 | seasonal_naive | 1.1292 | 5.00 |
| ieso | 1 | chronos_t5_tiny | 1.3613 | 6.00 |
| ieso | 4 | chronos_bolt_tiny | 0.7632 | 1.00 |
| ieso | 4 | chronos2_small | 0.8235 | 2.00 |
| ieso | 4 | seasonal_naive | 1.1501 | 3.00 |
| ieso | 4 | autoets | 1.1550 | 4.00 |
| ieso | 4 | chronos_t5_tiny | 1.1839 | 5.00 |
| ieso | 4 | ttm_512_week_context | 1.1961 | 6.00 |

## Prospective scores

No scheduled forecast has a completed 1- or 4-week target yet.

Scores are computed only after actual target weeks become complete in a later source snapshot. MASE uses a one-week naïve scale. The weekly seasonal-naïve/AutoETS periods are fixed at 52 for IESO and 4 for GitHub to respect the GitHub history limit. Probabilistic columns are scored only when a model provides them.
