from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)
MODEL_NAMES = ("chronos_bolt_tiny", "chronos_t5_tiny", "chronos2_small", "seasonal_naive", "autoets")


@dataclass
class Forecast:
    point: np.ndarray
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    probabilistic: bool


def _weight_digest(module: Any) -> str:
    """Hash the loaded checkpoint tensors, independent of serialization container metadata."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _extract_chronos_quantiles(values: np.ndarray, quantiles: Any, mean: Any) -> Forecast:
    q = quantiles.detach().to("cpu").numpy()
    point_mean = mean.detach().to("cpu").numpy()
    if q.ndim == 3:
        q = q[0]
    if q.shape[0] == len(QUANTILES):
        q = q.T
    if q.shape != (4, len(QUANTILES)):
        raise ValueError(f"Unexpected Chronos quantile output shape {q.shape}")
    if point_mean.ndim == 2:
        point_mean = point_mean[0]
    return Forecast(point=q[:4, 1].astype(float), q10=q[:4, 0], q50=q[:4, 1], q90=q[:4, 2], probabilistic=True)


class ModelRunner:
    """One model instance at a time; every loaded parameter must be on CPU."""

    def __init__(self, name: str, manifest: dict[str, Any], *, torch_module: Any | None = None):
        self.name = name
        self.manifest = manifest
        self.torch = torch_module
        self.model: Any | None = None
        self.checkpoint_hash: str | None = None
        self.revision: str | None = None

    def load(self) -> None:
        import torch

        if torch.cuda.is_available():
            raise RuntimeError("CUDA is visible; this experiment requires CPU-only inference")
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.torch = torch
        record = self.manifest.get("model_revisions", {}).get(self.name, {})
        self.revision = record.get("revision")
        if self.name == "chronos_bolt_tiny":
            from chronos import ChronosBoltPipeline
            self.model = ChronosBoltPipeline.from_pretrained(
                record["repo_id"], revision=self.revision, device_map="cpu", torch_dtype=torch.float32,
            )
            inner = self.model.model
        elif self.name == "chronos_t5_tiny":
            from chronos import ChronosPipeline
            self.model = ChronosPipeline.from_pretrained(
                record["repo_id"], revision=self.revision, device_map="cpu", torch_dtype=torch.float32,
            )
            inner = self.model.model
        elif self.name == "chronos2_small":
            from chronos import Chronos2Pipeline
            self.model = Chronos2Pipeline.from_pretrained(
                record["repo_id"], revision=self.revision, device_map="cpu", torch_dtype=torch.float32,
            )
            inner = self.model.model
        elif self.name == "ttm_512_week_context":
            from tsfm_public import TinyTimeMixerForPrediction
            repo_id = record["repo_id"]
            # Fixed weekly r2.1 checkpoint; it predicts 48 weeks and the requested first 1/4 are scored.
            self.model = TinyTimeMixerForPrediction.from_pretrained(repo_id, revision=self.revision)
            self.model.to("cpu").eval()
            inner = self.model
            self.revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        else:
            raise ValueError(f"Model {self.name!r} is loaded through a statistical forecast path")
        inner.eval()
        if any(parameter.device.type != "cpu" for parameter in inner.parameters()):
            raise RuntimeError(f"{self.name} has a non-CPU parameter")
        self.checkpoint_hash = _weight_digest(inner)

    def predict(self, history: np.ndarray, horizon: int, *, season_length: int = 1) -> Forecast:
        if len(history) < 2 or not np.isfinite(history).all():
            raise ValueError("Forecast history must contain at least two finite values")
        if self.name == "seasonal_naive":
            if len(history) < season_length:
                raise ValueError(f"Need {season_length} observations for seasonal-naive")
            point = np.array([history[-season_length + (i % season_length)] for i in range(horizon)], dtype=float)
            missing = np.full(horizon, np.nan)
            return Forecast(point=point, q10=missing.copy(), q50=missing.copy(), q90=missing.copy(), probabilistic=False)
        if self.name == "autoets":
            from statsforecast import StatsForecast
            from statsforecast.models import AutoETS
            import pandas as pd

            period = max(1, int(season_length))
            model = AutoETS(season_length=period, model="ZZZ")
            frame = pd.DataFrame({"unique_id": ["series"] * len(history),
                                  "ds": pd.date_range("2000-01-02", periods=len(history), freq="W-SUN"),
                                  "y": history.astype(float)})
            forecast = StatsForecast(models=[model], freq="W-SUN", n_jobs=1).forecast(df=frame, h=horizon, level=[80])
            point = forecast["AutoETS"].to_numpy(dtype=float)
            low = forecast.get("AutoETS-lo-80", pd.Series(np.nan, index=forecast.index)).to_numpy(dtype=float)
            high = forecast.get("AutoETS-hi-80", pd.Series(np.nan, index=forecast.index)).to_numpy(dtype=float)
            # StatsForecast supplies an 80% predictive interval, not a full quantile function.
            q10, q90 = low, high
            return Forecast(point=point, q10=q10, q50=point.copy(), q90=q90,
                            probabilistic=bool(np.isfinite(low).all() and np.isfinite(high).all()))
        if self.model is None:
            raise RuntimeError(f"Model {self.name} is not loaded")
        torch = self.torch
        if self.name == "chronos2_small":
            outputs = self.model.predict([torch.tensor(history, dtype=torch.float32)], prediction_length=horizon)
            out = outputs[0].detach().to("cpu").numpy()
            levels = list(self.model.quantiles)
            qrow = out[0]  # (quantile, horizon) for one univariate input
            q10 = qrow[levels.index(0.1), :horizon]
            q50 = qrow[levels.index(0.5), :horizon]
            q90 = qrow[levels.index(0.9), :horizon]
            return Forecast(point=q50.astype(float), q10=q10.astype(float), q50=q50.astype(float),
                            q90=q90.astype(float), probabilistic=True)
        if self.name == "ttm_512_week_context":
            context = history[-512:]
            mean = float(np.mean(context))
            scale = float(np.std(context)) or 1.0
            normalized = (context - mean) / scale
            inputs = torch.tensor(normalized, dtype=torch.float32).reshape(1, -1, 1)
            frequency = torch.tensor([9], dtype=torch.long)  # tsfm_public maps weekly ('W') to token 9.
            with torch.inference_mode():
                output = self.model(past_values=inputs, freq_token=frequency, return_loss=False)
            point = output.prediction_outputs.detach().to("cpu").numpy()[0, :horizon, 0] * scale + mean
            missing = np.full(horizon, np.nan)
            return Forecast(point=point.astype(float), q10=missing.copy(), q50=point.astype(float),
                            q90=missing.copy(), probabilistic=False)
        if self.name in ("chronos_bolt_tiny", "chronos_t5_tiny"):
            torch.manual_seed(20260927)
            tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
            sampling = {"num_samples": 100} if self.name == "chronos_t5_tiny" else {}
            quantiles, mean = self.model.predict_quantiles(
                inputs=tensor, prediction_length=horizon, quantile_levels=list(QUANTILES),
                **sampling,
            )
            return _extract_chronos_quantiles(history, quantiles, mean)
        raise ValueError(f"No forecast implementation for {self.name}")

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.torch is not None:
            # CPU tensors are released by dropping the pipeline and collecting Python objects.
            pass

