# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for forecast_server.py, with a scripted forecaster in place of the Timestral model."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import forecast_server

LEVELS = (0.1, 0.5, 0.9)
OFFSETS = (-0.5, 0.0, 0.5)  # Added to the last context value, exact in float32


class FakeForecaster:
    """Forecasts the last context value, with the quantiles spread by OFFSETS.

    Records the keyword arguments of every call in ``calls``.
    """

    quantile_levels = LEVELS
    max_context = 32768
    supports_covariates = True
    supports_known_covariates = True

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.config = SimpleNamespace(patch_size=32)
        self.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 5)])

    def forecast(self, **kwargs):
        self.calls.append(kwargs)
        context = kwargs["context"]
        horizon = kwargs["prediction_length"]
        last = np.nan_to_num(context[..., -1:])  # (B, V, 1)
        offsets = np.array(OFFSETS, dtype=np.float32)  # (Q,)
        quantiles = np.repeat(last[..., None], horizon, axis=-2) + offsets  # (B, V, H, Q)
        return SimpleNamespace(quantiles=quantiles, quantile_levels=LEVELS, median=quantiles[..., 1])


@pytest.fixture
def forecaster(monkeypatch):
    fake = FakeForecaster()
    state = forecast_server.ModelState("fake/model", threads=2)
    state.forecaster = fake
    state.load_seconds = 1.5
    monkeypatch.setattr(forecast_server, "state", state)
    return fake


@pytest.fixture
def client(forecaster):
    # No lifespan: the fixture already loaded the fake, nothing must load in the background
    return TestClient(forecast_server.app)


def test_ready_and_info_report_the_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok", "model": "fake/model", "loaded": True, "error": None}
    assert client.get("/ready").status_code == 200
    info = client.get("/api/info").json()
    assert info["model"] == "fake/model"
    assert info["parameters"] == 5
    assert info["quantile_levels"] == list(LEVELS)
    assert info["context_length"] == 32768
    assert info["threads"] == 2


def test_not_ready_while_loading(monkeypatch):
    monkeypatch.setattr(forecast_server, "state", forecast_server.ModelState("fake/model", threads=1))
    client = TestClient(forecast_server.app)
    assert client.get("/health").json()["loaded"] is False
    assert client.get("/ready").status_code == 503
    assert client.get("/api/info").status_code == 503
    assert client.post("/api/forecast", json={"context": [1.0, 2.0], "prediction_length": 1}).status_code == 503


def test_not_ready_when_loading_failed(monkeypatch):
    state = forecast_server.ModelState("fake/model", threads=1)
    state.error = "FileNotFoundError: no such repo"
    monkeypatch.setattr(forecast_server, "state", state)
    client = TestClient(forecast_server.app)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "no such repo" in response.json()["detail"]


def test_univariate_forecast_is_flat(client, forecaster):
    body = client.post("/api/forecast", json={"context": [1.0, 2.0, 3.0], "prediction_length": 2}).json()

    assert body["quantile_levels"] == list(LEVELS)
    assert body["prediction_length"] == 2
    assert body["median"] == [3.0, 3.0]
    assert body["quantiles"] == [[2.5, 3.0, 3.5], [2.5, 3.0, 3.5]]
    assert body["inference_ms"] >= 0
    call = forecaster.calls[0]
    assert call["context"].shape == (1, 1, 3)
    assert call["context"].dtype == np.float32
    assert call["symmetric_averaging"] is True and call["clip_nonnegative"] is True
    assert "past_covariates" not in call and "known_covariates" not in call


def test_multivariate_forecast_keeps_the_variates(client, forecaster):
    body = client.post("/api/forecast", json={"context": [[1.0, 2.0], [10.0, 20.0]], "prediction_length": 1}).json()

    assert body["median"] == [[2.0], [20.0]]
    assert body["quantiles"] == [[[1.5, 2.0, 2.5]], [[19.5, 20.0, 20.5]]]
    assert forecaster.calls[0]["context"].shape == (1, 2, 2)


def test_missing_values_become_nan(client, forecaster):
    client.post("/api/forecast", json={"context": [1.0, None, 3.0], "prediction_length": 1})

    context = forecaster.calls[0]["context"][0, 0]
    assert np.isnan(context[1]) and context[0] == 1.0 and context[2] == 3.0


def test_covariates_are_passed_with_their_shapes(client, forecaster):
    request = {
        "context": [1.0, 2.0, 3.0],
        "prediction_length": 2,
        "past_covariates": [[0.0, 1.0, 0.0]],
        "known_covariates": [[1.0, 0.0, 1.0, 0.0, 1.0], [5.0, 5.0, 5.0, 5.0, 5.0]],
        "symmetric_averaging": False,
    }
    assert client.post("/api/forecast", json=request).status_code == 200

    call = forecaster.calls[0]
    assert call["past_covariates"].shape == (1, 1, 3)
    assert call["known_covariates"].shape == (1, 2, 5)
    assert call["symmetric_averaging"] is False


def test_requested_quantile_levels_select_the_columns(client):
    body = client.post("/api/forecast", json={"context": [1.0], "prediction_length": 1, "quantile_levels": [0.1, 0.9]}).json()

    assert body["quantile_levels"] == [0.1, 0.9]
    assert body["quantiles"] == [[0.5, 1.5]]
    assert body["median"] == [1.0]  # From the model's median when 0.5 is not requested


def test_unknown_quantile_level_is_rejected(client):
    response = client.post("/api/forecast", json={"context": [1.0], "prediction_length": 1, "quantile_levels": [0.25]})
    assert response.status_code == 422
    assert "0.25" in response.json()["detail"]


@pytest.mark.parametrize(
    "request_body",
    [
        {"context": [], "prediction_length": 1},
        {"context": [[1.0, 2.0], [1.0]], "prediction_length": 1},
        {"context": [1.0, 2.0], "prediction_length": 0},
        {"context": [1.0, 2.0], "prediction_length": forecast_server.MAX_PREDICTION_LENGTH + 1},
        {"context": [1.0, 2.0], "prediction_length": 1, "past_covariates": [[1.0]]},
        {"context": [1.0, 2.0], "prediction_length": 1, "known_covariates": [[1.0, 2.0]]},
    ],
)
def test_malformed_requests_are_rejected(client, forecaster, request_body):
    assert client.post("/api/forecast", json=request_body).status_code == 422
    assert forecaster.calls == []


def test_model_errors_are_reported_as_unprocessable(client, forecaster, monkeypatch):
    def failing(**kwargs):
        raise ValueError("the input is degenerate")

    monkeypatch.setattr(forecaster, "forecast", failing)
    response = client.post("/api/forecast", json={"context": [1.0, 1.0], "prediction_length": 1})
    assert response.status_code == 422
    assert response.json()["detail"] == "the input is degenerate"
