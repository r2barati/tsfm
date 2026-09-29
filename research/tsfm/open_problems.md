# Open scientific problems in time-series foundation models

**Cutoff: 27 September 2026.**

## 1. Generalization needs operational definitions

“Unseen dataset” is weaker than unseen domain, new temporal regime, unseen frequency, unseen horizon, unseen variable relations, or future observations that did not exist at training time. TIME improves freshness; TS-Arena and Impermanent use prospective protocols. A clean study should stratify all these axes and record corpus cutoff. [TIME](https://arxiv.org/abs/2602.12147), [TS-Arena](https://ts-arena.live/about), [benchmark leakage study](https://arxiv.org/abs/2510.13654)

**Open question:** Which transfer signal is learned—local shape, frequency, domain semantics, global event history, or entity-specific memorization?

## 2. Pretraining data quality and transfer

Cross-domain pretraining can help unrelated domains and hurt apparently related domains, depending on frequency and temporal dynamics. Synthetic data can help distribution coverage, but the generators often encode assumptions and may carry implicit priors from public benchmarks. [Cross-domain study](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/), [Chronos](https://arxiv.org/abs/2403.07815)

**Open question:** What measurable data properties predict positive transfer before expensive training?

## 3. Multivariate signal versus spurious coupling

Chronos-2 group attention, Toto multivariate forecasting, Timer-XL, and TimesFM-3 expand cross-series learning. But covariate associations can be unstable, misaligned, or contaminated by leakage; dimensions and missing channels can change model behavior.

**Open question:** Does a model learn stable conditional relationships, or merely exploit co-movement seen in pretraining? Use variable permutation, intervention, missing-channel, and relation-shift tests. [Chronos-2](https://arxiv.org/abs/2510.15821), [Timer-XL](https://arxiv.org/abs/2410.04803)

## 4. Marginal uncertainty is not path uncertainty

Quantile models may achieve excellent marginal WQL while failing to generate coherent trajectories. Inventory replenishment over time, storage dispatch, and control depend on autocorrelation and joint tail events. Flow and token-sampling models can generate paths, but sample coherence, calibration, and cost are not routinely measured. [Sundial](https://arxiv.org/abs/2502.00816), [Chronos](https://arxiv.org/abs/2403.07815), [Moirai 2.0](https://arxiv.org/abs/2511.11698)

**Open question:** Which downstream decisions need joint paths rather than calibrated marginal quantiles?

## 5. Calibration under shift

Average CRPS/WQL can look acceptable while forecast intervals miscover during structural breaks or rare events. Need conditional calibration by horizon, regime, series scale, and event severity; online recalibration must not use future data.

**Open question:** Can reusable TSFMs support reliable selective prediction or abstention when shift is detected?

## 6. Causal intervention and known-future covariates

Known-future promotions or calendars are inputs; intervention effects are a different estimand. A forecaster using observed promotional history may confuse selection effects with causal response. None of broad TSFM interfaces grants causal identification by architecture alone.

**Open question:** How should TSFMs represent the difference between observed covariates, planned actions, and hypothetical interventions? Test with randomized promotion data, natural experiments, or known causal simulators.

## 7. Forecast metric to decision utility

A single-period asymmetric inventory example yields MAE/decision ranking reversal. One energy paper shows decision-focused Moirai fine-tuning can reduce average daily costs versus prediction-focused tuning, but no standard multi-model benchmark evaluates this. [Beichter et al.](https://arxiv.org/abs/2503.01936)

**Open question:** When do better MASE/CRPS/WQL rankings predict lower sequential cost, and when do they reverse? Quantify policy regret and realized utility in inventory, storage, resource allocation and control.

## 8. Scaling laws remain partial

Toto 2.0 and Time-MoE report within-family scaling; Moirai 2.0 reports parameter plateaus. Size, data volume, data quality, training tokens, tokenization, active FLOPs and benchmark contamination are entangled. [Toto 2.0](https://arxiv.org/abs/2605.20119), [Time-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [Moirai 2.0](https://arxiv.org/abs/2511.11698)

**Open question:** Which resource buys real transfer: parameters, independent series, unique temporal regimes, calibrated synthetic trajectories, or target-domain context?

## 9. Long context and long horizon are conflated

A model may accept 16K context yet ignore distant history; a long horizon may extrapolate beyond any repeated seasonal coverage. Context ablations, history permutation, and train/test horizon extrapolation should be reported separately.

## 10. Rare events and regime shifts

Benchmark aggregates reward common patterns; smooth forecasts can miss spikes and breaks. Test event recall, tail calibration, false alarm burden, and policy cost when rare events carry asymmetric losses.

## 11. Streaming models and online learning

TiRex-2's recurrent state is relevant to telemetry and industrial streams. Need evaluate state updates with missing/out-of-order data, concept drift, cold starts, numerical stability, and end-to-end latency against cached Transformer KV and batch forecasts. [TiRex-2](https://arxiv.org/abs/2607.01204)

## 12. General-purpose task representations

MOMENT and Timer aim beyond forecasting, but classification, imputation, anomaly detection, and forecasting do not share one obvious optimal objective. A representation that reconstructs local samples may not expose decision-relevant event or causal features. [MOMENT](https://arxiv.org/abs/2402.03885), [Timer](https://arxiv.org/abs/2402.02368)

## 13. Benchmark integrity and leaderboard adaptation

Static public benchmarks become training or selection data. Fresh datasets can still have opaque source history; public histories can share shocks across entities. Live pre-registration prevents direct target exposure but narrows domain coverage. [GIFT-Eval](https://arxiv.org/abs/2410.10393), [TS-Arena](https://arxiv.org/abs/2512.20761)

## 14. License and reproducibility constraints

Current candidate models differ in checkpoint licenses, data provenance, and compute transparency. TimesFM-3 has non-commercial terms; Toto 2.0 releases Apache weights; model/data code permissions are not interchangeable. The paper matrix records release details where verifiable. [TimesFM-3 card](https://huggingface.co/google/timesfm-3.0-pytorch), [Toto repo](https://github.com/DataDog/toto)

## Five most consequential open problems

1. **Do forecasting rankings predict sequential decision value?** Establish a benchmark and theory-led error decomposition.
2. **What temporal knowledge transfers under true future shift?** Prospective evaluation plus concept-shift strata.
3. **When is joint-path uncertainty necessary?** Compare quantiles, marginal samples, and copula/joint trajectory outputs under path-dependent policies.
4. **Which pretraining scale dimension matters?** Orthogonalize data diversity, tokens, parameters, context and compute.
5. **Can multivariate TSFMs identify robust relations?** Test relation shift, missing channels, covariate availability, and interventions.

The five strongest candidates and exact experiments are specified in [research opportunities](research_opportunities.md).
