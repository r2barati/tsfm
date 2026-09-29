# Scientific taxonomy of time-series foundation models

**Cutoff: 27 September 2026.** A system may fit several categories. Benchmark labels should identify the exact checkpoint and inference protocol.

## 1. Representation

| Representation | Examples | Benefit | Limitation to test |
|---|---|---|---|
| Numeric strings / digit tokens | LLMTime | Reuses pretrained LM; text metadata and prompting are easy | Number tokenization is arbitrary; context is wasted; calibration can fail. GPT-4 underperformed GPT-3 in one setup. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf) |
| Scale-normalized quantization | Chronos | Categorical LM loss, sampleable futures, mature tooling | Quantization error and serial decoding; scalar values become tokens. [Chronos](https://arxiv.org/abs/2403.07815) |
| Continuous patches | TimesFM, Timer, Toto | Compresses long streams; larger output chunks reduce serial decoding | Patch size is an inductive choice; spikes and phase shifts can be smoothed. [TimesFM](https://arxiv.org/abs/2310.10688) |
| Multi-size or adaptive patches | Moirai, TTM, Moirai-MoE | Addresses frequency and local-scale variation | Added projections and specialization choices need ablations. [Moirai](https://arxiv.org/abs/2402.02592), [TTM](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1) |
| Lag feature tokens | Lag-Llama | Compact, scale-aware autoregression and probabilistic output | Univariate by default; depends on lag construction. [Lag-Llama](https://arxiv.org/abs/2310.08278) |
| Masked continuous patches | MOMENT, Moirai | Efficient representation learning and task adaptation | Reconstruction objective may not align with forecasting or decisions. [MOMENT](https://arxiv.org/abs/2402.03885) |
| Tabular temporal features | TabPFN-TS | Strong control for small data; easy feature/covariate integration | Depends on feature schema and horizon limits; not learned universal dynamics. [TabPFN-TS](https://arxiv.org/abs/2501.02945) |

## 2. Backbone and information mixing

| Backbone | Examples | Evidence/diagnostic |
|---|---|---|
| Decoder-only Transformer, primarily series independent | TimesFM 1/2/2.5, Timer, Chronos-Bolt | Strong reusable prior and patch API; early versions do not natively mix series. |
| Decoder-only with channel or multivariate mixing | Toto, Timer-XL, TimesFM-3 announcement | Test by removing cross-series inputs, varying dimension, permuting channels, and masking covariates. |
| Encoder / masked modeling | Moirai-1, MOMENT, Chronos-2 | Bidirectional input context; the prediction head/objective determines probabilistic and horizon behavior. |
| Sparse MoE | Time-MoE, Moirai-MoE | Separates total capacity from active compute; study routing by pattern and shift. |
| Mixer / MLP | TTM | Tiny CPU-friendly models can transfer; size is not a proxy for utility. |
| Recurrent xLSTM | TiRex, TiRex-2 | Streaming state can bound update cost; test state drift and equal-hardware speed. |
| General tabular Transformer | TabPFN-TS | Prior comes from synthetic tables; temporal meaning comes from feature engineering. |
| General LLM adapted to time | LLMTime, GPT4TS, Time-LLM, CALF, ChatTime | Language or text-side context may help; an LLM backbone alone does not establish temporal transfer. [CALF, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34082), [ChatTime, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33384/35539) |

## 3. Prediction mechanisms

- **Autoregressive categorical generation:** Chronos. Provides trajectories, with quantization and sequential inference costs.
- **Autoregressive continuous patches:** TimesFM, Toto, Timer-XL. Token efficient but may be point forecasts unless equipped with uncertainty heads.
- **Direct multi-horizon quantiles:** Chronos-Bolt, Chronos-2, Moirai 2.0. Fast marginal probabilities; marginal quantiles do not imply coherent joint paths.
- **Mixture likelihood:** Moirai-1. Flexible multimodal marginals subject to parametric adequacy.
- **Masked reconstruction plus task head:** MOMENT. Transferable features may still require task adaptation.
- **Flow matching:** Sundial. Flexible trajectory samples; evaluate calibration, path quality, and sampling cost.
- **Multi-token/contiguous patch masking:** Moirai 2.0 and Toto 2.0 recipes. Improve throughput but change the learning target as well.

## 4. Adaptation protocol

| Protocol | Definition | Representative work |
|---|---|---|
| Zero-shot | No target-domain gradient updates; may still have benchmark or global-event contamination | Public TSFM leaderboard results |
| Few-shot / in-context | Related target-domain examples enter inference context | TimesFM In-Context Fine-Tuning; Chronos-2 group attention |
| Fine-tuning | Target training split updates weights | MOMENT, TTM, TimesFM |
| PEFT | Small parameter subset updated | TimesFM LoRA examples; energy Moirai fine-tuning |
| Decision-focused PEFT | Optimize application value through an optimization layer/surrogate | Moirai dispatchable feeder study |

Do not combine these into one leaderboard without protocol columns. “Zero-shot” is not equivalent to “unseen data.”

## 5. Information structure

- **Univariate:** one series forecast from its own history.
- **Multivariate:** multiple channels are jointly predicted, with a defined cross-channel information path. Running an independent univariate model per channel is not native multivariate reasoning.
- **Cross-series learning:** information transfers across entities, either via support examples (ICF) or joint/group attention (Chronos-2).
- **Covariates:** distinguish past-only, known-future (calendar/promotions), and future-but-uncertain (weather predictions). Enforce point-in-time availability.
- **Static metadata:** can condition entity/domain but may reveal identity or benchmark labels.
- **Text/multimodal conditioning:** distinct from numeric-to-text serialization. Text descriptions and reports must be evaluated with controlled retrieval and timestamp policy.

## 6. Scaling axes and evidence

| Axis | Evidence by cutoff | Defensible conclusion |
|---|---|---|
| Parameter count | Time-MoE and Toto 2.0 show within-family gains; Moirai 2.0 reports plateau | Scaling may help within a recipe; no universal law established |
| Data volume | TimesFM/Chronos show gains from broad data and synthetic augmentation; cross-domain transfer can hurt | Data quality/domain mix/objective mediate benefit |
| Context length | TimesFM 2.5 supports 16K; Timer-XL targets long context | Supported length is not evidence that older context helps; ablate age-specific slices |
| Number of series/variates | Chronos-2, Toto, and TimesFM-3 aim at joint forecasts | Measure accuracy, missing-channel robustness, and compute as dimension increases |
| Forecast horizon | Direct heads improve throughput | Publish horizon-wise accuracy, calibration, and latency |
| Inference compute | Chronos-Bolt and multi-token prediction speed up; flow sampling adds cost | Compare throughput/memory at fixed hardware, batch, context, horizon |
| Training compute | Often omitted or not comparable | Parameter size alone cannot support scaling law claims |

## 7. Forecast quality to decision quality

Let H be history, b a predictive belief over future paths, a an action, and Y the realized path. A forecast loss L(b,Y) and utility U(a,Y) answer different questions. The optimal action is a*(b)=argmax_a E_b[U(a,Y)]. A proper scoring rule incentivizes an honest predictive distribution for its defined random variable, but better score under one scoring rule does not guarantee better action value for every utility. There is no universal monotone theorem from a scalar MASE, CRPS, or WQL ranking to arbitrary sequential utility; Blackwell ordering compares information structures, which is a stronger condition.

[Blackwell's comparison of experiments](https://doi.org/10.1214/aoms/1177729032) gives a stronger sufficient notion: an information structure that is more informative than another (the weaker signal can be simulated from the stronger) is weakly better for all decision problems when the decision maker may ignore information. A ranking by MASE, CRPS, or WQL is much weaker and does not prove Blackwell dominance.

**Inventory ranking reversal.** Demand is 0 with probability 0.8 and 10 with probability 0.2. Forecaster A predicts 0 and has MAE 2. Forecaster B predicts 10 and has MAE 8, so MAE ranks A better. With underage cost 9 per unit and overage cost 1, expected asymmetric inventory loss is 18 for A and 8 for B, so the decision objective ranks B better. The Bayes-optimal order quantity is the 0.9 demand quantile, namely 10.

For multi-period inventory, storage, finance, and control, a marginal distribution at each horizon is insufficient when utility depends on joint paths. Evaluate temporal dependence, calibration by regime, constraints, policy adaptation, and cumulative realized cost. Calibration matters when decisions are sensitive to tails, but calibration is not a substitute for utility evaluation.

## 8. Sequential-decision connection

A decision-capable pipeline is:

**observations + metadata → latent state/belief → action-conditioned future distribution → policy or optimizer → environment → realized utility → online update**

The ordinary TSFM pipeline ends after forecast. Decision Transformer and offline RL model action/reward-conditioned behavior; world models predict action consequences; model-predictive control repeatedly replans; stochastic and distributionally robust optimization map distributions into feasible actions. These literatures provide tools, not evidence that current TSFMs already perform action-conditioned causal forecasting.

Important distinction: a covariate association is not an intervention effect. “What demand follows a promotion?” on historical data can be confounded by targeting. Interventions require randomized data, causal structure, or explicit assumptions and tests.

## References for theoretical bridges

- Blackwell, D. “Equivalent Comparisons of Experiments.” *Annals of Mathematical Statistics*, 1953. A more informative signal weakly improves every Bayes decision problem.
- Elmachtoub and Grigas, [“Smart Predict, then Optimize”](https://doi.org/10.1287/mnsc.2020.3922), *Management Science*, 2022. It formalizes loss functions tied to downstream optimization decisions.
- Beichter et al. “Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization.” 2025; reports a Moirai application. [Paper](https://arxiv.org/abs/2503.01936)
- Faw et al. “In-Context Fine-Tuning for Time-Series Foundation Models.” ICML 2025. [Paper](https://proceedings.mlr.press/v267/faw25b.html)
- Chen et al. “Decision Transformer: Reinforcement Learning via Sequence Modeling.” NeurIPS 2021. [Paper](https://arxiv.org/abs/2106.01345)


