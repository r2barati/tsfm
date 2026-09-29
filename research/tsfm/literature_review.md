# Time-series foundation models: literature review

**Review cutoff: 27 September 2026.** This is a source-verified research map, not a claim that one universal SOTA exists. It distinguishes peer-reviewed papers, preprints, code/model releases, and company announcements. Unreported fields are not inferred.

## Executive synthesis

The central scientific change since 2024 is not simply “bigger Transformers.” It is the move from independent-series forecasting toward four choices: reusable temporal priors with zero-shot inference; related-series in-context adaptation; covariate-aware or native multivariate forecasting; and domain-specific scaling/deployment. No architecture dominates all four.

Three findings survive comparison across primary sources:

1. Transfer is real but conditional. Chronos, TimesFM, Moirai, and TTM demonstrate useful zero/few-shot transfer, while domain studies find performance depends on pretraining-domain match and may not beat small task-specific models. [Chronos](https://arxiv.org/abs/2403.07815), [Moirai](https://arxiv.org/abs/2402.02592), [TTM](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1), [How Foundational are Foundation Models for Time Series Forecasting?](https://arxiv.org/abs/2510.00742)

2. Data and task interface matter at least as much as parameter count. Toto 2.0 reports systematic gains from 4M to 2.5B parameters under one family recipe, while Moirai 2.0 reports parameter plateauing and longer-horizon weakness; TimesFM 2.5 reduces its backbone from 500M to 200M while increasing context. These results are not direct contradictions because objectives, data, horizons, and evaluations differ. [Toto 2.0](https://arxiv.org/abs/2605.20119), [Moirai 2.0](https://arxiv.org/abs/2511.11698), [TimesFM releases](https://github.com/google-research/timesfm)

3. “Zero-shot” is an evaluation protocol, not a property proved by a model card. A model may be zero-shot for a user task while having seen the benchmark series or globally shared shocks during pretraining. GIFT-Eval demonstrates score inflation from known overlap; later work describes global-pattern contamination; TIME and prospective live benchmarks improve evaluation. [GIFT-Eval](https://arxiv.org/abs/2410.10393), [Benchmarking Challenges and Requirements](https://arxiv.org/abs/2510.13654), [TIME](https://arxiv.org/abs/2602.12147), [TS-Arena](https://arxiv.org/abs/2512.20761)

**Most consequential gap:** Public TSFM comparisons mostly score forecasts, not repeated operational decisions. They rarely measure stockouts, overstock, energy dispatch cost, or capacity provisioning. One 2025 study provides a decision-focused Moirai fine-tuning example for building-energy optimization, but no shared cross-model sequential decision benchmark exists. [Beichter et al.](https://arxiv.org/abs/2503.01936)

## 1. Scope and definition

Use a capability-based definition: a model is foundation-like when broad pretraining supports transfer across series, tasks, or domains through a reusable interface. A parameter threshold is unsuitable. TTM begins around one million parameters; Chronos-2 is 120M; Toto 2.0 ranges 4M–2.5B; TabPFN regression can be repurposed through temporal features. [TTM](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1), [Chronos-2](https://arxiv.org/abs/2510.15821), [Toto 2.0](https://arxiv.org/abs/2605.20119), [TabPFN-TS](https://arxiv.org/abs/2501.02945)

Distinguish native time-series pretraining; general pretrained models adapted through numeric serialization or learned temporal adapters; and tabular priors applied to time-indexed features. The last two can be useful without learning a universal temporal representation.

## 2. Scientific generations

### 2023: test whether language-model priors transfer

LLMTime serialized numbers as text and showed general LLMs can extrapolate simple temporal patterns. It also documented a failure: GPT-4 could perform worse than GPT-3 because numeric tokenization and calibration matter. Lag-Llama used a decoder-only model with lag features and probabilistic Student-t output as a compact, open univariate baseline. [LLMTime, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf), [Lag-Llama](https://arxiv.org/abs/2310.08278)

This was proof of feasibility, not evidence that an LLM understands physical dynamics or that textual digits are efficient.

### 2024: purpose-built pretraining and reusable interfaces

TimesFM, Chronos, and Moirai established complementary recipes. TimesFM used continuous input/output patches with a decoder-only Transformer, at 200M parameters and 100B real-world points. Chronos scaled and quantized values and trained T5/GPT-family backbones by categorical likelihood, then sampled trajectories. Moirai targeted heterogeneity with multi-size patch projections, any-variate attention, probabilistic mixture outputs, and LOTSA. [TimesFM](https://arxiv.org/abs/2310.10688), [Chronos](https://arxiv.org/abs/2403.07815), [Moirai](https://arxiv.org/abs/2402.02592)

MOMENT broadened the objective from forecasting to masked reconstruction and multiple downstream tasks. Timer made next-token generation a shared adaptation interface for forecasting, imputation, and anomaly detection. TTM showed that a compact mixer with resolution-aware pretraining can transfer in multivariate forecasting at a fraction of the size. [MOMENT](https://arxiv.org/abs/2402.03885), [Timer](https://arxiv.org/abs/2402.02368), [TTM](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1)

Targeted ablations provide the best mechanism evidence: Chronos tested synthetic-data, size, and tokenization choices; Moirai tested patch-size and any-variate design; TTM tested adaptive patching and resolution handling. There is no cross-paper ablation holding data and compute fixed.

### 2025: efficient outputs, MoE, and in-context adaptation

Chronos-Bolt replaced autoregressive categorical sampling with patch-level direct quantile prediction, reducing sequential inference at the cost of not directly sampling coherent paths. Sundial used flow matching to model continuous next-patch distributions. Time-MoE and Moirai-MoE tested sparse experts. Timer-XL extended generative prediction to long-context and multivariate inputs. [Chronos repository](https://github.com/amazon-science/chronos-forecasting), [Sundial](https://arxiv.org/abs/2502.00816), [Time-MoE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [Moirai-MoE, ICML 2025](https://openreview.net/pdf?id=SrEOUSyJcR), [Timer-XL](https://arxiv.org/abs/2410.04803)

A second shift used context as adaptation. In-Context Fine-Tuning trained TimesFM to consume related target-domain series and reported performance comparable to target-domain fine-tuning. Chronos-2 used group attention to share information across targets and covariates. [ICML 2025 paper](https://proceedings.mlr.press/v267/faw25b.html), [Chronos-2](https://arxiv.org/abs/2510.15821)

### 2026: multivariate releases, scaling claims, and prospective evaluation

- **TimesFM-3** (Google announcement, 31 Aug 2026): 330M parameters, claimed more than 1T real and synthetic points, native multivariate prediction, single-pass inference. The announcement claims leading benchmark results; it is company-authored and no peer-reviewed paper was identified at the cutoff. Released weights use a non-commercial license. [Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [model card](https://huggingface.co/google/timesfm-3.0-pytorch)
- **Toto 2.0** (May 2026 report): Apache 2.0 family, 4M–2.5B; authors report within-family scaling gains and results on BOOM, GIFT-Eval, and TIME. This is the clearest current scaling study, but it is one family and has privileged observability data. [Paper](https://arxiv.org/abs/2605.20119), [official repo](https://github.com/DataDog/toto)
- **TiRex-2** (July 2026 preprint): recurrent xLSTM multivariate forecasting with covariates and streaming updates. Bounded state/update costs are distinctive; vendor comparisons need independent replication. [Paper](https://arxiv.org/abs/2607.01204), [project](https://nx-ai.github.io/tirex-2/)
- **TabPFN-TS** changes with its underlying tabular model. At 15 Sep 2026 the official adapter defaults to TabPFN-3.5, so results without package and checkpoint versions are not reproducible. It remains feature-engineered tabular regression, not native temporal pretraining. [Integration](https://github.com/PriorLabs/tabpfn-time-series), [TabPFN-3.5 report](https://arxiv.org/abs/2609.17895)
- **Benchmarking** shifted toward fresh tasks and prospective evaluation: TIME has 50 new datasets and 98 tasks; TS-Arena is a KDD 2026 prospective platform; Impermanent uses live prequential evaluation. [TIME](https://arxiv.org/abs/2602.12147), [TS-Arena](https://ts-arena.live/about), [Impermanent](https://arxiv.org/abs/2603.08707)

## 3. Architecture families

1. **Continuous patch autoregression:** TimesFM, Timer/Timer-XL, Toto. Patches reduce token count and allow larger forecast blocks; a long context only helps when training and evaluation show it is used.
2. **Discrete probabilistic generation:** Chronos. Quantization reuses LM tooling and supports path sampling, with quantization error and serial decoding costs.
3. **Direct multi-horizon quantiles:** Chronos-Bolt, Chronos-2, Moirai 2.0. Efficient for marginal quantiles, but marginal quantiles do not define joint temporal trajectories.
4. **Masked representation learning:** Moirai-1, MOMENT. Useful for heterogeneous input and adaptation; representation quality does not guarantee zero-shot forecast quality.
5. **Continuous generative distributions:** Sundial flow matching. Flexible sampling needs calibration, path-quality, and compute evaluation beyond marginal CRPS.
6. **Sparse experts and scaling:** Time-MoE, Moirai-MoE, Toto 2.0. Compare active FLOPs, total parameters, data, and training tokens separately.
7. **Recurrent/streaming:** TiRex/TiRex-2 use xLSTM. Persistent state may suit streams; compare with optimized attention under identical histories and hardware.
8. **Tabular repurposing:** TabPFN-TS maps lag, calendar, and covariate features to a tabular prior. It tests whether TS-specific pretraining is necessary.
9. **LLM adaptation:** LLMTime, GPT4TS, Time-LLM, CALF and ChatTime bring language priors or text-side information; performance remains task-specific. [CALF, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34082), [ChatTime, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33384/35539)

## 4. Scaling evidence

- **Parameters:** Time-MoE reports improvement through 2.4B; Toto 2.0 reports monotonic improvement from 4M to 2.5B. These support within-family scaling for tested tasks, not a universal scaling law. [Time-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html), [Toto 2.0](https://arxiv.org/abs/2605.20119)
- **Data:** TimesFM and Chronos show benefit from broad data and synthetic augmentation; cross-domain pretraining can also cause negative transfer when frequencies or dynamics mismatch. [TimesFM](https://arxiv.org/abs/2310.10688), [Chronos](https://arxiv.org/abs/2403.07815), [cross-domain study](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/)
- **Context:** TimesFM 2.5 supports 16K context, but supported length is not evidence that older history helps. Moirai 2.0 reports worse results at longer horizons; extrapolation and objective matter.
- **Horizon:** Direct heads improve throughput, but long-horizon error and uncertainty remain. Publish horizon-wise scores.
- **Inference compute:** Chronos-Bolt and multi-token output increase throughput. Flow models buy flexible sampling at additional inference cost. Compare latency/memory with fixed batch, series count, context, and horizon.

No primary study gives a community-accepted cross-family scaling law controlling data quality, training compute, inference compute, context, and a clean prospective test.

## 5. Contradictions and evidence limits

- Chronos found gains from model size and synthetic data; Moirai 2.0 reports size plateau and favors efficiency; Time-MoE and Toto 2.0 report monotonic within-family gains. Conditions differ, so this is an open scientific disagreement.
- Rankings change across GIFT-Eval, fev-bench, BOOM, Chronos Benchmark II, TIME, and LTSF because the tasks differ in covariates, horizons, dimensionality, metrics, and contamination.
- Company claims are not independent confirmation. TimesFM-3 is new; Toto 2.0 uses proprietary observability data; TiRex-2 has a fresh streaming claim.
- Forecast score need not reduce decision cost. One Moirai energy study reports 9.45% lower average daily cost with decision-focused versus prediction-focused fine-tuning, but this is one application/model. [Beichter et al.](https://arxiv.org/abs/2503.01936)
- Cross-series learning helps only when relationships are useful at inference. It can introduce spurious links, missingness sensitivity, and scaling with variables.

## 6. Closest justified research thesis

The next paper should test whether forecasts preserve decision-relevant information under shift and path dependence. Build paired evaluation with point-in-time clean and prospective data; exact model/checkpoint lineage; direct quantiles and sampled joint trajectories; operational environments with fixed costs/constraints; forecast accuracy, calibration, action regret, service levels, and cumulative utility; and rolling or prospective evaluation with intervals. One decision-focused case study exists; a shared cross-model benchmark does not.

See [model taxonomy](model_taxonomy.md), [benchmark audit](benchmark_audit.md), [SOTA assessment](sota_2026.md), [open problems](open_problems.md), and [research opportunities](research_opportunities.md).
