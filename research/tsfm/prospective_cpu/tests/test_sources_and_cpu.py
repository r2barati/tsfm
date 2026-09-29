from __future__ import annotations

import requests
import pytest

from prospective_cpu.forecasters import ModelRunner
from prospective_cpu.sources import request_with_retry


def test_http_retry_handles_github_stats_202(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, status: int):
            self.status_code = status
            self.headers = {}
            self.content = b"[]"
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)
    class Session:
        calls = 0
        def __init__(self):
            self.headers = {}
        def get(self, *args, **kwargs):
            Session.calls += 1
            return Response(202 if Session.calls == 1 else 200)
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr("prospective_cpu.sources.time.sleep", lambda _: None)
    response = request_with_retry("https://api.github.com/repos/a/b/stats/commit_activity",
                                  attempts=3, stats_202_is_retryable=True)
    assert response.status_code == 200
    assert Session.calls == 2


def test_model_loader_refuses_cuda_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CPU-only"):
        ModelRunner("chronos_t5_tiny", {}).load()


def test_seasonal_naive_is_cpu_point_forecast() -> None:
    runner = ModelRunner("seasonal_naive", {})
    forecast = runner.predict(__import__("numpy").array([1., 2., 3., 4.]), 4, season_length=4)
    assert forecast.point.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not forecast.probabilistic
