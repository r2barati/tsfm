# Research opportunities: 15 falsifiable TSFM hypotheses

**Review cutoff:** 27 September 2026. **GPU figures are planning estimates from the proposed experiment, not claims reported in the cited papers.** Each experiment assumes released checkpoints and focuses compute on inference/fine-tuning, not pretraining a billion-parameter model from scratch.

## H1 — Forecast metrics can mis-rank sequential policies

- **Question / hypothesis:** Does forecast quality predict cumulative operational utility? **H:** Across repeated inventory and energy tasks, model ranking by MASE/CRPS will reverse under at least one economically plausible cost matrix, and direct decision-focused fine-tuning will reduce policy regret without necessarily improving MASE.
- **Why unanswered:** One Moirai building-energy study reports 9.45% cost reduction from decision-focused fine-tuning, but there is no shared multi-model benchmark across operations tasks. [Beichter et al.](https://arxiv.org/abs/2503.01936)
- **Closest work:** Smart Predict, then Optimize; Beichter et al.; probabilistic inventory/newsvendor forecasting.
- **Experiment/data:** Freeze forecasts from Chronos-2, TimesFM-2.5, Toto 2.0 small, TTM, TiRex, statistical baselines. Use M5/Walmart hierarchy for replenishment and BuildingsBench or open day-ahead energy data for dispatch/storage. Rolling origins only; prespecify costs and constraints. Evaluate MASE/CRPS/WQL, service level, stockout/holding cost, energy cost, cumulative regret, and confidence intervals. Fine-tune Moirai/Chronos/TTM with prediction versus decision losses on identical train windows.
- **Compute:** 1–2 GPUs with 24–48GB for batched inference; 1–4 GPUs for PEFT runs. Simulator/bootstraps are CPU-heavy. No foundation pretraining.
- **Falsification:** No ranking reversals across defensible costs, or decision-focused adaptation fails to improve held-out utility over prediction-focused training.
- **If supported:** A decision-oriented TSFM benchmark and an empirical characterization of when proper forecast scores translate into value.
- **If rejected:** Evidence that proper probabilistic quality is a robust proxy for these decisions under the tested conditions; this would justify forecast metrics in those regimes.

## H2 — Prospective temporal shift changes the model ranking

- **Question / hypothesis:** Do static zero-shot ranks predict future performance? **H:** Model rank variance across prospective cutoffs is large, and a static benchmark’s top model is not reliably top under later drift.
- **Why unanswered:** TIME adds fresh retrospective tasks; TS-Arena and Impermanent provide prospective evaluation but are domain-limited. [TIME](https://arxiv.org/abs/2602.12147), [TS-Arena](https://ts-arena.live/about), [Impermanent](https://arxiv.org/abs/2603.08707)
- **Closest work:** GIFT-Eval, TIME, TS-Arena, Impermanent, TSFM leakage audit.
- **Experiment/data:** Keep the Ontario/GitHub 12-cutoff study as a pilot. For v2, freeze 20 EIA-930 balancing-authority demand series and 20 GHCN-Daily station-temperature series using the selection algorithms in `research/tsfm/prospective_v2/protocol_v2.md`. Compare 12 frozen methods (five compact TSFMs and seven statistical/simple baselines), with a retrospective development phase and 24 prospective monthly cutoffs at 1- and 4-week horizons, subject to the preregistered power decision. The primary statistic is cross-domain historical-to-prospective rank transfer; within-domain stability and input-only shift/reversal analysis are separate prespecified outcomes. Preserve failures and missing outcomes.
- **Compute:** CPU-only inference on the inspected four-core, 16-GiB Windows machine; Python 3.13.2 in an isolated `uv` environment with CPU-only PyTorch. Batch eligible series, run one neural checkpoint at a time, and do no training, GPU, or cloud inference. All 12 pinned methods passed a 256-week CPU smoke at internal forecast steps 2 and 5; the largest observed process RSS was 1.45 GB (1.35 GiB). The smoke took 461 seconds for two series across both steps. Linear scaling to 40 series gives about 2.6 hours of inference per cutoff. EIA setup parsing streamed 59,401 weekly rows from the 70-authority archive in 955 seconds with about 136 MB RSS. The archive is 692 MB compressed / 4.36 GB expanded; each live cutoff currently downloads the same bulk archive and streams only the frozen authorities, so budget about 16.6 GB of EIA downloads across 24 cutoffs and provisionally add 16 minutes of parsing per cutoff. The revised planning estimate is 2.9 hours per cutoff and about 70 hours for 24 prospective cutoffs, plus roughly 62 hours for the 24-origin historical development backtest. These are extrapolations under the current shared-machine load, not measured full-cohort runtimes; per-cutoff acquisition/parse and inference timings will replace them. The workload is 40 series × 2 horizons × 12 methods = 960 series-horizon-method forecast groups per cutoff (23,040 groups across 24 cutoffs). Horizon 1 scores the first full week starting after the cutoff week; horizon 4 scores four such weeks, requiring model steps 2–5 from the last complete weekly input. Schedule the v2 task at 09:30 Toronto time after the existing pilot's 09:00 task.
- **Falsification:** Static ranks strongly predict prospective ranks with narrow uncertainty across domains and cutoffs, and no meaningful drift in calibration.
- **If supported:** A defensible time-aware generalization claim and benchmark methodology.
- **If rejected:** Static evaluations have external validity in those tasks; prospective evaluation can focus on event rarity and utility instead.

## H3 — Joint-path uncertainty helps only for path-dependent decisions

- **Question / hypothesis:** Are marginal quantiles sufficient? **H:** For one-step asymmetric decisions, quantile forecasts are competitive; for multi-period storage/inventory/control with ramping or capacity constraints, calibrated joint path samples lower realized cost.
- **Why unanswered:** Chronos-Bolt/Chronos-2/Moirai 2.0 emphasize quantile outputs; Chronos/Sundial produce samples, but standard evaluations rarely score joint temporal dependence. [Chronos](https://arxiv.org/abs/2403.07815), [Sundial](https://arxiv.org/abs/2502.00816), [Moirai 2.0](https://arxiv.org/abs/2511.11698)
- **Closest work:** Probabilistic forecasting and stochastic MPC; Sundial; probabilistic energy benchmarks.
- **Experiment/data:** Compare quantile outputs, independent marginal sampling, Chronos sampled trajectories, Sundial flow samples, and a copula-coupled quantile baseline on ERCOT/ENTSO-E load/price and M5 demand. Evaluate marginal CRPS, energy score, variogram score, coverage, ramp/event likelihood, storage cost and risk.
- **Compute:** 1–2 GPUs (24–48GB) for inference/sampling; CPU optimization solver for policies.
- **Falsification:** Joint samples do not improve path scores or decision cost after calibration/compute matching.
- **If supported:** Establish when joint output is necessary and give TSFM authors a target beyond marginal WQL.
- **If rejected:** Fast marginal quantiles may be sufficient for these policies; reduces the need for expensive path generators.

## H4 — Cross-series attention is brittle when relations shift

- **Question / hypothesis:** Does multivariate attention learn causal/stable relationships? **H:** Native multivariate models help when relationships persist, but are worse than univariate or retrievable-history baselines when correlations change, channels go missing, or spurious variables are added.
- **Why unanswered:** Chronos-2 and new multivariate releases show accuracy gains on benchmark tasks but no comprehensive relation-shift test. [Chronos-2](https://arxiv.org/abs/2510.15821), [TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- **Closest work:** Timer-XL, Chronos-2, Toto, multivariate forecasting robustness literature.
- **Experiment/data:** Use synthetic structural time-series equations with interventions plus electricity/traffic panels. Train/evaluate with stable links, reversed/removed links, permuted channels, missing variates, and unseen group sizes. Compare Chronos-2, Toto, TimesFM-3 when accessible, TiRex-2, univariate TSFMs, VAR and regularized linear baselines.
- **Compute:** 1–2 GPUs (24–48GB); synthetic generation and statistical baselines mostly CPU.
- **Falsification:** Cross-series models retain consistent gains under link shift and missingness, without calibration or cost penalties.
- **If supported:** Failure taxonomy and robust relation-selection architecture.
- **If rejected:** Current group/attention mechanisms already provide sufficient invariance; identify what training distribution explains success.

## H5 — Data diversity can dominate parameter count

- **Question / hypothesis:** Which resource drives zero-shot transfer: unique series regimes or parameters? **H:** At fixed compute, expanding regime diversity gives more OOD gain than increasing parameters after a moderate size.
- **Why unanswered:** Toto 2.0 and Time-MoE show within-family size trends; Moirai 2.0 shows a plateau; cross-domain study reports negative transfer. No shared factorial study isolates dimensions. [Toto 2.0](https://arxiv.org/abs/2605.20119), [Time-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [cross-domain study](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/)
- **Closest work:** Time-MoE scaling; Toto scaling; Chronos synthetic-data ablations; domain transfer study.
- **Experiment/data:** Train 20M–100M model family on fixed token budgets with factorial variation of number of unique source series, domain/frequency diversity, synthetic/real proportion, and repetition. Evaluate on hidden domains plus prospective series; measure score per training FLOP and per unique regime.
- **Compute:** 8–32 A100/H100 GPU-days for 20M–100M controlled pretraining; much smaller if existing model checkpoints permit continued pretraining.
- **Falsification:** Parameter/data volume predicts OOD performance better than regime diversity, or factorial effects are negligible.
- **If supported:** A time-series scaling law for curation and data mixture.
- **If rejected:** Stronger case for parameter scaling or domain-specialized training; narrows investment priorities.

## H6 — Global event exposure explains “zero-shot” gains

- **Question / hypothesis:** Do models borrow global shocks learned elsewhere? **H:** Models exposed to an event in a different series predict its timing/signature in unseen series better than models trained before that event, even when test series identity is novel.
- **Why unanswered:** Leakage work theorizes and measures global-pattern memorization, but there is no controlled event-exposure benchmark with matched model weights. [Meyer et al.](https://arxiv.org/abs/2510.13654)
- **Closest work:** GIFT leakage study; TS-Arena; benchmark contamination work.
- **Experiment/data:** Pre/post-cutoff checkpoints around COVID, energy crisis, and rate regimes; paired series with and without event exposure; synthetic event injection. Score the event window separately and use identical forecast origins.
- **Compute:** 1 GPU for inference; training cost depends on available historical checkpoints; approximately 1–2 GPU-days for controlled small model.
- **Falsification:** Event-exposed models show no measurable gain over matched pre-event models on novel series.
- **If supported:** Clarifies zero-shot claims and motivates event-embargo protocols.
- **If rejected:** Global shock transfer is less important than series-specific shape or context.

## H7 — Selecting in-context examples beats nearest-neighbor heuristics

- **Question / hypothesis:** Can support-set selection improve in-context adaptation? **H:** A learned selector based on temporal features and task-specific utility improves forecast and decision metrics over random, domain-label, or Euclidean nearest-neighbor support selection.
- **Why unanswered:** ICF and Chronos-2 use related series but optimal selection is open. [ICF](https://proceedings.mlr.press/v267/faw25b.html), [Chronos-2](https://arxiv.org/abs/2510.15821)
- **Closest work:** In-Context Fine-Tuning; Chronos-2; retrieval-augmented forecasting.
- **Experiment/data:** Monash/GIFT/fev tasks with multiple series per dataset; compare retrieval rules and a small selector; avoid test labels in support search. Measure gain vs context cost.
- **Compute:** 1 GPU (24GB) for inference and selector training; CPU index building.
- **Falsification:** Learned selection does not beat random or simple pattern matching at equal context tokens.
- **If supported:** Practical sample-efficient adaptation without target gradient updates.
- **If rejected:** Cross-series gains arise from aggregate context rather than fine selection.

## H8 — Horizon extrapolation is distinct from long context

- **Question / hypothesis:** Does context scaling improve unseen forecast horizon? **H:** Increasing context helps only when seasonal coverage spans the new horizon; it does not substitute for horizon extrapolation training.
- **Why unanswered:** 16K context support and long-context architectures are often conflated with long-horizon performance. [TimesFM repo](https://github.com/google-research/timesfm), [Timer-XL](https://arxiv.org/abs/2410.04803), [Moirai 2.0](https://arxiv.org/abs/2511.11698)
- **Closest work:** Timer-XL; TimesFM 2.5; Moirai 2.0 long-horizon analysis.
- **Experiment/data:** Factorial test of context (short/long) and horizon (seen/unseen) on hourly/daily electricity, weather, M4, TIME. Ablate old context and train horizon curriculum.
- **Compute:** 1–2 GPUs (24–48GB) for inference and small PEFT study.
- **Falsification:** Longer context consistently improves out-of-range horizons independent of seasonal coverage.
- **If supported:** Separate context- and horizon-scaling axes and improve model cards.
- **If rejected:** Long input is a generalization aid; quantify where useful.

## H9 — Decision-focused tuning can improve value without degrading general forecasts

- **Question / hypothesis:** Can utility specialization preserve general-purpose competence? **H:** PEFT with multi-objective forecast and decision losses improves target utility while retaining zero-shot performance outside the target domain.
- **Why unanswered:** One energy case demonstrates cost gains but not cross-domain retention. [Beichter et al.](https://arxiv.org/abs/2503.01936)
- **Closest work:** Decision-focused fine-tuning; Moirai; smart predict-then-optimize.
- **Experiment/data:** Tune adapters on one of buildings, inventory, or energy storage; evaluate target and held-out general GIFT/TIME tasks. Compare forecast-only, decision-only, multi-objective and LoRA.
- **Compute:** 1–2 GPUs (24–48GB); PEFT on 100M–300M backbones.
- **Falsification:** Utility gains vanish out of sample or catastrophic forecast transfer loss is unavoidable.
- **If supported:** Modular specialist adapters over reusable TSFM priors.
- **If rejected:** Decision objectives require full retraining or target-specific models.

## H10 — Calibration degrades earlier than point accuracy under shift

- **Question / hypothesis:** Do uncertainty estimates become unreliable before MASE changes? **H:** Conditional coverage and tail calibration degrade under regime shifts while point scores move modestly, and online calibration improves risk-sensitive action quality.
- **Why unanswered:** Benchmarks report CRPS/WQL averages; conditional rare-event calibration is less measured.
- **Closest work:** Chronos probabilistic outputs; Sundial; TS-Arena live evaluation.
- **Experiment/data:** Energy price/load and retail demand across COVID/weather/event windows; evaluate interval coverage and tail loss by regime; compare conformal/online recalibration.
- **Compute:** 1 GPU (24GB), mostly evaluation.
- **Falsification:** Calibration tracks point metrics and online recalibration gives no decision benefit.
- **If supported:** Shift-aware uncertainty layer and operational trust criteria.
- **If rejected:** Avoids extra calibrator complexity.

## H11 — Synthetic multivariate training can teach transferable dependence

- **Question / hypothesis:** Are synthetic “multivariatizers” sufficient for real covariate dependence? **H:** Synthetic dependence templates improve zero-shot cross-series skill only when template families span the target relation structures; otherwise they underperform real multivariate pretraining.
- **Why unanswered:** Chronos-2 uses synthetic multivariate structures; TimesFM-3 and Toto train multivariately at scale, but data recipes differ. [Chronos-2](https://arxiv.org/abs/2510.15821)
- **Closest work:** Chronos-2; multivariate forecasting; synthetic data in Chronos.
- **Experiment/data:** Generate known contemporaneous, lagged, nonlinear, and regime-switching links; compare synthetic-only, real-only and mixed pretraining on hidden relation families.
- **Compute:** 4–16 A100 GPU-days for medium-scale controlled models; downstream test 1 GPU.
- **Falsification:** Real and synthetic pretraining perform similarly across relation shifts without template coverage effects.
- **If supported:** Design rules for synthetic relation generators.
- **If rejected:** Synthetic dependence may not be the bottleneck; data alignment/covariates matter more.

## H12 — Streaming xLSTM is cheaper only at matched forecast service

- **Question / hypothesis:** Does recurrent streaming lower total cost without sacrificing accuracy? **H:** TiRex-2 reduces cumulative compute for frequent refreshes at long histories, but its advantage shrinks for batched infrequent forecasts.
- **Why unanswered:** Model-team claims describe bounded state, while Transformer batching/caching has different workload tradeoffs. [TiRex-2](https://arxiv.org/abs/2607.01204)
- **Closest work:** TiRex/TiRex-2; Timer-XL; streaming forecasting systems.
- **Experiment/data:** Identical industrial/telemetry streams; measure energy, latency p50/p99, throughput, memory and forecast quality for 1, 100, 10K streams and different update rates; include cold start and missing points.
- **Compute:** One GPU plus CPU edge device; no training needed.
- **Falsification:** Transformer caching/batching matches or beats recurrent throughput/cost at equal forecast quality.
- **If supported:** Workload-aware model selection and streaming benchmark.
- **If rejected:** The architecture's bounded-state advantage does not translate to service-level benefit.

## H13 — Covariates help through task availability and alignment, not count

- **Question / hypothesis:** Are known-future covariates useful when aligned and point-in-time valid? **H:** A few correctly aligned covariates reduce decision cost more than many weak or misaligned covariates; naive covariate inclusion can worsen transfer.
- **Why unanswered:** fev-bench contains 46 covariate tasks; Chronos-2 supports them, but the causal/value of each feature varies. [fev-bench](https://arxiv.org/abs/2509.26468), [Chronos-2](https://arxiv.org/abs/2510.15821)
- **Closest work:** Chronos-2; TabPFN-TS; event covariate study.
- **Experiment/data:** Retail promotions/calendar and energy weather/day-ahead forecasts; lag and misalign inputs deliberately; compare absent, aligned, noisy and future-unavailable covariates.
- **Compute:** 1 GPU (24GB) for inference and small adapter.
- **Falsification:** Covariate count or domain label alone predicts gains, with no alignment effect.
- **If supported:** Point-in-time covariate audit and value-of-information metric.
- **If rejected:** Benefits may be dataset-specific and not generalize across covariate families.

## H14 — A general representation is better for downstream tasks only after adaptation

- **Question / hypothesis:** Does masked pretraining yield useful general time-series features? **H:** MOMENT frozen features underperform task-specific training, but light linear/PEFT adaptation improves low-label classification, imputation and anomaly detection across new domains.
- **Why unanswered:** MOMENT's multi-task capability is broad but adaptation protocols and metrics vary. [MOMENT](https://arxiv.org/abs/2402.03885)
- **Closest work:** MOMENT; Timer; TSLib tasks.
- **Experiment/data:** UCR/UEA classification, anomaly detection corpora, imputation benchmarks plus new domain split; compare frozen probe, LoRA, full fine-tune and supervised-from-scratch.
- **Compute:** 1–2 GPUs (24–48GB); small downstream datasets.
- **Falsification:** Frozen/PEFT models do not improve low-label performance or do not transfer beyond pretraining-adjacent domains.
- **If supported:** Characterizes when one temporal representation supports multiple tasks.
- **If rejected:** Forecasting and analysis may require separate pretraining objectives.

## H15 — Event/text side information helps only when point-in-time grounded

- **Question / hypothesis:** Do text descriptions and event records improve TSFM transfer? **H:** Text/event features help rare events only when timestamp-aligned and available at forecast origin; retrieval after the fact produces apparent gains that disappear under point-in-time evaluation.
- **Why unanswered:** Multimodal models (ChatTime, CALF) and event-covariate examples exist, but contamination and publication-time controls are inconsistent. [ChatTime](https://ojs.aaai.org/index.php/AAAI/article/download/33384/35539), [CALF](https://ojs.aaai.org/index.php/AAAI/article/view/34082)
- **Closest work:** ChatTime; CALF; event covariate papers; LLMTime textual context.
- **Experiment/data:** Public energy/news and retail event streams with timestamp archive; compare no text, contemporaneous text, delayed/future text, retrieved similar event and shuffled text. Score forecast plus policy utility.
- **Compute:** 1 GPU (24GB) for TSFM; text encoder may need a second GPU or API budget.
- **Falsification:** Point-in-time text yields no improvement over structured event flags or shuffled controls.
- **If supported:** A temporal multimodal forecasting protocol with leakage-safe retrieval.
- **If rejected:** Structured covariates are sufficient, simplifying multimodal system design.

## Strongest five and tradeoffs

1. **H1 decision metric versus sequential utility.** Highest practical and conceptual value; feasible with open models and simulators; direct connection to operations/RL. Main risk is simulator realism and cost assumptions. Start with several prespecified costs and report the full Pareto frontier.
2. **H2 prospective generalization.** Strong methodology and timely leakage concern; inexpensive inference; use TS-Arena plus extra domains. Main risk is prospective duration and narrow initial domains.
3. **H3 joint path uncertainty.** A sharply testable gap between marginal quantile leaders and sample generators; moderate compute. Main risk is policy results depending on optimizer choices; freeze policies and include oracle/naive controls.
4. **H4 cross-series relation shift.** Strong scientific mechanism and direct relevance to multivariate releases; moderate compute with synthetic ground truth and real validation. Main risk is domain confounding; make synthetic causal structure primary and real datasets confirmatory.
5. **H5 controlled scaling/data diversity.** Highest theoretical value; costly. A good paper needs controlled models and compute; a cheaper study can use continued pretraining and 20M–100M sizes. Main risk is scope and compute; preregister the factorial design before training.

A strong combined paper could use H1 as the contribution, H2 as its contamination-safe evaluation protocol, and H3/H4 as mechanism analyses, but avoid bundling so many tasks that no result is decisive.

