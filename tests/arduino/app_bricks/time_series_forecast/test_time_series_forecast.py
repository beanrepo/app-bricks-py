# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import math
from email.message import Message
from unittest.mock import patch

import numpy as np
import pytest

import arduino.app_bricks.time_series_forecast.time_series_forecast as module
from arduino.app_bricks.time_series_forecast import Forecast, ForecastError, ModelInfo, TimeSeriesForecast
from arduino.app_utils import http

BASE_URL = "http://timestral-runner:1345"
LEVELS = [0.1, 0.5, 0.9]
INFO = {
    "model": "mistralai/Timestral-5M",
    "parameters": 5_210_000,
    "patch_size": 32,
    "context_length": 32768,
    "max_prediction_length": 1024,
    "quantile_levels": LEVELS,
    "supports_covariates": True,
    "supports_known_covariates": True,
    "threads": 4,
    "load_seconds": 2.0,
}


def response(status_code: int, payload: object) -> http.Response:
    return http.Response(status_code, Message(), json.dumps(payload).encode())


@pytest.fixture
def runner():
    """Replace the HTTP calls with a scripted runner.

    `runner.posts` lists the (URL, JSON payload) of the forecasts sent, `runner.forecast` is the payload
    the runner answers, `runner.ready` the status codes `/ready` answers in turn.
    """
    state = {
        "ready": [200],
        "forecast": {
            "median": [3.0, 3.0],
            "quantiles": [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
            "quantile_levels": LEVELS,
            "prediction_length": 2,
            "inference_ms": 12.5,
        },
    }
    posts: list[tuple[str, dict]] = []

    def get(url: str, *, timeout: float | None = None, **kwargs: object) -> http.Response:
        if url == f"{BASE_URL}/ready":
            status = state["ready"].pop(0) if len(state["ready"]) > 1 else state["ready"][0]
            return response(status, {"status": "ready"} if status == 200 else {"detail": "model loading"})
        if url == f"{BASE_URL}/api/info":
            return response(200, INFO)
        raise AssertionError(f"unexpected GET {url}")

    def post(url: str, *, json: dict, timeout: float | None = None, **kwargs: object) -> http.Response:
        posts.append((url, json))
        answer = state["forecast"]
        return response(answer.get("status", 200), answer)

    with patch.object(http, "get", side_effect=get), patch.object(http, "post", side_effect=post), patch.object(module.time, "sleep"):
        yield type("Runner", (), {"state": state, "posts": posts})


def test_start_waits_for_the_runner_and_logs_the_model(runner):
    runner.state["ready"] = [503, 503, 200]
    brick = TimeSeriesForecast()
    brick.start()
    assert brick.model_info == ModelInfo(
        model="mistralai/Timestral-5M",
        parameters=5_210_000,
        patch_size=32,
        context_length=32768,
        max_prediction_length=1024,
        quantile_levels=(0.1, 0.5, 0.9),
        supports_covariates=True,
        supports_known_covariates=True,
    )
    brick.stop()


def test_start_fails_when_the_runner_never_becomes_ready(runner, monkeypatch):
    runner.state["ready"] = [503]
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(ForecastError, match="not ready after 10 s: model loading"):
        TimeSeriesForecast(ready_timeout=10).start()


def test_start_fails_when_the_runner_is_unreachable(runner, monkeypatch):
    def refused(url: str, **kwargs: object) -> http.Response:
        raise http.ConnectionFailed("connection refused")

    monkeypatch.setattr(http, "get", refused)
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(ForecastError, match="connection refused"):
        TimeSeriesForecast(ready_timeout=10).start()


def test_univariate_forecast(runner):
    forecast = TimeSeriesForecast(prediction_length=2).forecast([1.0, 2.0, 3.0])

    assert runner.posts == [(f"{BASE_URL}/api/forecast", {"context": [1.0, 2.0, 3.0], "prediction_length": 2})]
    assert isinstance(forecast, Forecast)
    assert forecast.horizon == 2
    assert forecast.quantile_levels == (0.1, 0.5, 0.9)
    assert forecast.inference_ms == 12.5
    np.testing.assert_array_equal(forecast.median, [3.0, 3.0])
    np.testing.assert_array_equal(forecast.lower(), [2.0, 2.0])
    np.testing.assert_array_equal(forecast.upper(), [4.0, 4.0])
    assert forecast.quantiles.shape == (2, 3)


def test_multivariate_forecast(runner):
    runner.state["forecast"] = {
        "median": [[2.0], [20.0]],
        "quantiles": [[[1.0, 2.0, 3.0]], [[19.0, 20.0, 21.0]]],
        "quantile_levels": LEVELS,
        "prediction_length": 1,
        "inference_ms": 1.0,
    }
    forecast = TimeSeriesForecast().forecast(np.array([[1.0, 2.0], [10.0, 20.0]]), prediction_length=1)

    assert runner.posts[0][1] == {"context": [[1.0, 2.0], [10.0, 20.0]], "prediction_length": 1}
    assert forecast.quantiles.shape == (2, 1, 3)
    np.testing.assert_array_equal(forecast.median, [[2.0], [20.0]])
    np.testing.assert_array_equal(forecast.quantile(0.9), [[3.0], [21.0]])


def test_missing_values_are_sent_as_null(runner):
    TimeSeriesForecast().forecast([1.0, None, math.nan, float("inf"), 5.0], prediction_length=2)

    assert runner.posts[0][1]["context"] == [1.0, None, None, None, 5.0]


def test_covariates_and_quantile_levels_are_sent(runner):
    TimeSeriesForecast().forecast(
        [1.0, 2.0],
        prediction_length=2,
        past_covariates=[[0.0, 1.0]],
        known_covariates=np.array([[1.0, 0.0, 1.0, 0.0], [5.0, 5.0, 5.0, 5.0]]),
        quantile_levels=(0.1, 0.9),
    )

    assert runner.posts[0][1] == {
        "context": [1.0, 2.0],
        "prediction_length": 2,
        "past_covariates": [[0.0, 1.0]],
        "known_covariates": [[1.0, 0.0, 1.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
        "quantile_levels": [0.1, 0.9],
    }


def test_unknown_quantile_level_raises(runner):
    forecast = TimeSeriesForecast().forecast([1.0, 2.0], prediction_length=2)
    with pytest.raises(ValueError, match="0.25"):
        forecast.quantile(0.25)


@pytest.mark.parametrize("context", [[], [[]], [[[1.0]]]])
def test_invalid_context_raises(runner, context):
    with pytest.raises(ValueError, match="context"):
        TimeSeriesForecast().forecast(context)
    assert runner.posts == []


def test_invalid_horizon_raises(runner):
    with pytest.raises(ValueError, match="prediction_length"):
        TimeSeriesForecast(prediction_length=0)
    with pytest.raises(ValueError, match="prediction_length"):
        TimeSeriesForecast().forecast([1.0], prediction_length=0)
    brick = TimeSeriesForecast(prediction_length=5)
    brick.prediction_length = 7
    assert brick.prediction_length == 7
    with pytest.raises(ValueError):
        brick.prediction_length = 0


def test_runner_refusal_raises_with_its_detail(runner):
    runner.state["forecast"] = {"status": 422, "detail": "prediction_length 2000 exceeds the limit of 1024"}
    with pytest.raises(ForecastError, match=r"refused the request \(422\): prediction_length 2000 exceeds"):
        TimeSeriesForecast().forecast([1.0, 2.0], prediction_length=2000)


def test_runner_validation_errors_are_joined(runner):
    runner.state["forecast"] = {
        "status": 422,
        "detail": [{"loc": ["body", "context"], "msg": "context must hold at least one value", "type": "value_error"}],
    }
    with pytest.raises(ForecastError, match="context must hold at least one value"):
        TimeSeriesForecast().forecast([1.0])


def test_unreachable_runner_raises(runner, monkeypatch):
    def refused(url: str, **kwargs: object) -> http.Response:
        raise http.ConnectionFailed("connection refused")

    monkeypatch.setattr(http, "post", refused)
    with pytest.raises(ForecastError, match="cannot be reached: connection refused"):
        TimeSeriesForecast().forecast([1.0, 2.0])
