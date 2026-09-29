# TSFM frontier report: what is the next unanswered question?

**State-of-the-art review cutoff: 27 September 2026.** This synthesis distinguishes peer-reviewed papers, preprints, releases, and announcements. SOTA is task-, metric-, protocol-, checkpoint-, and date-specific.

## 1. What the field believed in 2024

The working thesis was that a single broadly pretrained time-series model could replace one-model-per-dataset forecasting for many zero-shot tasks. TimesFM demonstrated patch-based decoder-only forecasting from large corpora; Chronos showed that quantizing scaled values and training language-model objectives yields useful probabilistic transfer; Moirai solved heterogeneous frequency/variate handling with any-variate attention and multi-patch projections. MOMENT and Timer broadened the ambition beyond forecasting. [TimesFM](https://arxiv.org/abs/2310.10688), [Chronos](https://arxiv.org/abs/2403.07815), [Moirai](https://arxiv.org/abs/2402.02592), [MOMENT](https://arxiv.org/abs/2402.03885), [Timer](https://arxiv.org/abs/2402.02368)

## 2. What changed in 2025

The field changed the output mechanism and adaptation protocol: Chronos-Bolt uses direct quantiles for speed; Sundial uses flow matching for continuous trajectory sampling; MoE families test capacity/compute tradeoffs; Timer-XL broadens multivariate and long context; ICF and Chronos-2 use related series in context. There was no convergence on one universally superior output representation. [Sundial](https://arxiv.org/abs/2502.00816), [Time-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [Timer-XL](https://arxiv.org/abs/2410.04803), [ICF](https://proceedings.mlr.press/v267/faw25b.html), [Chronos-2](https://arxiv.org/abs/2510.15821)

## 3. What changed in 2026

Several public releases now advertise multivariate capability and scaling: Toto 2.0 (4M–2.5B, report), TimesFM-3 (330M, Google announcement), and TiRex-2 (recurrent xLSTM preprint). TabPFN-TS continued changing with the new TabPFN backbone. In evaluation, TIME introduced fresh task-centered data; TS-Arena (KDD 2026) and Impermanent added prospective/live protocols. Many latest model claims are too recent for independent replications as of this cutoff. [Toto 2.0](https://arxiv.org/abs/2605.20119), [TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [TiRex-2](https://arxiv.org/abs/2607.01204), [TIME](https://arxiv.org/abs/2602.12147), [TS-Arena](https://ts-arena.live/about)

## 4. Current SOTA architecture families

- Continuous patch autoregression: TimesFM, Toto, Timer; strong scalable forecast interface.
- Discrete token generation: Chronos; probabilistic trajectories and synthetic-data leverage.
- Direct multi-horizon quantile decoders: Chronos-Bolt, Chronos-2, Moirai 2.0; efficient uncertainty marginals.
- Masked representation learners: Moirai 1, MOMENT; broad adaptation potential.
- Flow matching: Sundial; flexible distribution sampling.
- MoE: Time-MoE, Moirai-MoE, Toto 2.0; increased total capacity with sparse compute.
- Mixer: TTM; compact transfer under constrained resources.
- Recurrent xLSTM: TiRex/TiRex-2; streaming state and bounded update computation.
- Tabular foundation repurposing: TabPFN-TS; a meaningful baseline for whether native TS pretraining is needed.

Detailed attributes, released checkpoints, data, task support, compute and licenses are recorded in [paper_matrix.csv](paper_matrix.csv). For model-specific rankings see [sota_2026.md](sota_2026.md).

## 5. Current benchmark frontier

GIFT-Eval offers broad domain/frequency/multivariate coverage and a non-leaking pretraining set; it also directly reports contamination effects. fev-bench adds covariates and bootstrap intervals. TIME adds fresh tasks and pattern-level diagnostics. BOOM tests observability but overlaps Toto's training domain. Chronos Benchmark II is useful historical OOD comparison but is public and may have entered later corpora. Monash and LTSF remain useful for classic methods, not contamination-proof OOD claims. TS-Arena and Impermanent test forecasts before outcomes exist, but their initial domain coverage is narrow. [GIFT-Eval](https://arxiv.org/abs/2410.10393), [fev-bench](https://arxiv.org/abs/2509.26468), [TIME](https://arxiv.org/abs/2602.12147), [BOOM](https://arxiv.org/abs/2505.14766), [TS-Arena](https://ts-arena.live/about), [Impermanent](https://arxiv.org/abs/2603.08707)

No listed suite fully evaluates decision consequences or policy feedback.

## 6. What scaling laws we actually know

Within-family evidence: Time-MoE reports gains up to 2.4B; Toto 2.0 reports gains from 4M to 2.5B. Against that, Moirai 2.0 reports model-size plateau and declines at long horizons; TimesFM 2.5 is smaller than 2.0 yet longer-context. Cross-domain experiments show negative transfer can happen. **Established:** some recipes scale within measured ranges. **Not established:** a universal loss law in parameter/data/compute/context or a compute-optimal allocation rule. [Time-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [Toto 2.0](https://arxiv.org/abs/2605.20119), [Moirai 2.0](https://arxiv.org/abs/2511.11698), [TimesFM](https://github.com/google-research/timesfm), [cross-domain study](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/)

## 7. What remains scientifically unresolved

1. Transfer to truly future observations and unseen regimes.
2. Forecast uncertainty calibration under drift and rare events.
3. Marginal quantiles versus joint trajectory quality for sequential use.
4. Stable cross-series relations versus spurious co-movement.
5. Which training resource—unique regimes, points, parameters, context, or compute—buys OOD performance.
6. Whether better predictive scores translate to better operational decisions.
7. Causal response versus observational covariate conditioning.
8. General-purpose representation versus task-specific objectives.

See [open problems](open_problems.md).

## 8. Popular assumptions with weak evidence

- **“More parameters always help.”** Supported within selected recipes, contradicted by reported plateaus and smaller newer releases.
- **“Long context means long-horizon skill.”** Input capacity is not an ablation showing effective use.
- **“Multivariate is automatically better.”** Related variables help only if available, aligned, and stable.
- **“Zero-shot means uncontaminated.”** Benchmark overlap and global-shock transfer are documented concerns.
- **“Probabilistic means calibrated.”** CRPS/WQL averages do not imply correct tails or joint paths.
- **“Forecast SOTA is operational SOTA.”** Forecast losses encode a chosen scoring rule, not arbitrary cost functions.
- **“Foundation model means large model.”** TTM and TabPFN-TS show reusable priors can be compact or borrowed from other modalities.

## 9. Five strongest open research problems

1. **Forecast quality versus sequential utility:** quantify ranking reversals, cost sensitivity, and decision-focused adaptation.
2. **Prospective generalization under temporal drift:** connect static benchmark ranks to live future performance.
3. **Joint-path uncertainty:** identify tasks where trajectory dependence changes decisions.
4. **Robust multivariate structure:** intervene on/shift cross-series relations and measure failure.
5. **Controlled scaling:** disentangle model size, unique data diversity, synthetic data, context and training compute.

## 10. Connection to decision foundation models

The conventional pipeline is observations → forecast → score. A decision foundation model needs observations and metadata → belief/state → action-conditioned predictive distribution → policy/optimizer → environment → utility feedback. Model-based RL and world models learn action consequences; Decision Transformer learns action/reward-conditioned trajectories; offline RL learns from logged decisions; stochastic optimization maps distributions to actions; decision-focused learning trains prediction for downstream value. [Smart Predict, then Optimize](https://doi.org/10.1287/mnsc.2020.3922), [Decision Transformer](https://arxiv.org/abs/2106.01345), [decision-focused TSFM study](https://arxiv.org/abs/2503.01936)

A forecast is not a causal simulator. If promotions, prices, or controls are actions, observed correlations are confounded. A future research model should distinguish observed covariates from interventions, represent uncertainty over paths, respect constraints, and be judged by regret and realized utility.

## 11. Most promising paper direction

**Decision-aware and shift-aware evaluation of TSFMs.** The contribution is strongest if it is a shared benchmark plus a falsifiable empirical finding, rather than another leaderboard or another adapter. Start with two environments—multi-period inventory and energy storage—where actions are transparent and utility is auditable. Compare the same pinned forecasts across point forecast, probabilistic score, decision loss, and rolling realized cost.

## 12. Exact experiments that would resolve the core questions

- Freeze Chronos-2, TimesFM-2.5, Toto 2.0 small, TTM and statistical baselines by checkpoint hash; include TimesFM-3/TiRex-2 only as explicitly provisional new-release rows.
- Create rolling-origin folds that respect release timestamps; maintain hidden prospective windows where possible.
- Score forecast MAE/MASE, CRPS/WQL, calibration and joint energy/variogram scores.
- Run prespecified newsvendor, multi-period replenishment and storage policies with fixed costs/capacity; compute service level, expected cost, tail cost, action regret and cumulative utility.
- Compare point-loss-trained, quantile-trained, and decision-focused PEFT versions on identical target training windows. Hold compute/search budget equal.
- Bootstrap across datasets/entities and time origins; publish confidence intervals, full per-task distributions, and cost-weight sensitivity rather than only aggregate ranks.
- Test covariate and relation shift by withholding future features, permuting channels, removing groups, and simulating structural breaks.
- Publish data lineage, point-in-time feature audit, model/version/license, hardware, throughput and exact prompts/inference configuration.

The candidate hypotheses and falsification rules are detailed in [research_opportunities.md](research_opportunities.md).

### What would constitute a genuinely new result?

- **Engineering improvement:** Faster or smaller inference, with matched accuracy and hardware. Useful, but not a new scientific claim by itself.
- **Benchmark improvement:** A higher score on a known suite. Meaningful only with pinned versions, leakage controls, uncertainty intervals, and strong baselines.
- **Empirical scientific finding:** A reproducible mechanism-level result—for example, identifying when multivariate transfer fails under relation shift or showing why a larger corpus causes negative transfer.
- **New evaluation methodology:** Prospective, point-in-time-safe evaluation that predicts real deployment reliability or decision utility and can be reproduced independently.
- **Theoretical contribution:** A condition under which a predictive representation or score guarantees decision value, or a bound linking shift/calibration to policy regret.
- **New foundation-model architecture:** A distinct representation/objective/information pathway that demonstrates controlled improvements in clean transfer and downstream utility, not only a leaderboard win.

The most convincing next result would establish a reusable connection between temporal prediction, calibrated beliefs under shift, and realized sequential utility—and show where the connection fails.
