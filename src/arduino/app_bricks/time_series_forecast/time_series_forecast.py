# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Zero-shot probabilistic forecasting of time series with the Timestral foundation models.

The models run on the CPU in the ``timestral-runner`` container, this brick sends it the observed
values and returns the forecast quantiles.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arduino.app_internal.core import resolve_address
from arduino.app_utils import AppError, Logger, brick, http

logger = Logger("TimeSeriesForecast")

RUNNER_PORT = 1345
"""Port the timestral-runner container listens on, the compose file publishes it on the same number."""

READY_POLL_INTERVAL_S = 2.0
"""Seconds between two readiness checks of the runner while it loads, or downloads, the model."""

type SeriesLike = Sequence[float | None] | Sequence[Sequence[float | None]] | np.ndarray
"""One series as values along time, or one row of values per variate. None and NaN mark a missing value."""


class ForecastError(AppError):
    """The runner refused the request or could not be reached."""


@dataclass(frozen=True)
class ModelInfo:
    """The checkpoint the runner serves and its limits."""

    model: str
    """Hugging Face repository id of the checkpoint, e.g. ``mistralai/Timestral-5M``."""
    parameters: int
    patch_size: int
    """Time steps per patch, the context is left-padded to a multiple of it."""
    context_length: int
    """Most recent context steps the model reads, older values are dropped."""
    max_prediction_length: int
    """Longest horizon the runner accepts in one request."""
    quantile_levels: tuple[float, ...]
    supports_covariates: bool
    supports_known_covariates: bool


@dataclass(frozen=True)
class Forecast:
    """Quantile forecast of one series.

    ``quantiles`` is shaped ``(horizon, levels)`` for a univariate series and ``(variates, horizon, levels)``
    for a multivariate one, sorted along the last axis at the ``quantile_levels``.
    """

    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]
    inference_ms: float
    """Milliseconds the model took, without the transfer."""

    @property
    def horizon(self) -> int:
        """Forecast steps."""
        return self.quantiles.shape[-2]

    @property
    def median(self) -> np.ndarray:
        """The 0.5 quantile, the point forecast: ``(horizon,)`` or ``(variates, horizon)``."""
        return self.quantile(0.5)

    def quantile(self, level: float) -> np.ndarray:
        """The forecast at one of the ``quantile_levels``: ``(horizon,)`` or ``(variates, horizon)``."""
        if level not in self.quantile_levels:
            raise ValueError(f"Quantile level {level} is not one of {self.quantile_levels}.")
        return self.quantiles[..., self.quantile_levels.index(level)]

    def lower(self, level: float = 0.1) -> np.ndarray:
        """Lower bound of the prediction interval, the forecast at ``level``."""
        return self.quantile(level)

    def upper(self, level: float = 0.9) -> np.ndarray:
        """Upper bound of the prediction interval, the forecast at ``level``."""
        return self.quantile(level)


def _rows(values: SeriesLike, name: str) -> tuple[list[list[float | None]], bool]:
    """``values`` as rows of JSON numbers, None where missing, and whether it was a single row."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty sequence of values, or one sequence per variate.")
    return [[None if math.isnan(v) or math.isinf(v) else float(v) for v in row] for row in array.tolist()], array.shape[0] == 1


@brick
class TimeSeriesForecast:
    """Forecast the next values of a time series from its past, without training.

    Timestral is a family of time-series foundation models by Mistral AI: given the observed values of a
    series it predicts the next ``prediction_length`` steps as quantiles, so each step comes with its
    uncertainty. Any regularly sampled signal works, sensor readings, counts, energy, temperature, as long
    as the steps are evenly spaced; the model reads values, not timestamps.

    The model runs in the ``timestral-runner`` container on the CPU. The checkpoint is chosen with the
    ``TIMESTRAL_MODEL`` variable of the brick, ``mistralai/Timestral-5M`` by default, and is downloaded
    from Hugging Face the first time the app starts.

    Example:
        >>> forecaster = TimeSeriesForecast(prediction_length=12)
        >>> forecast = forecaster.forecast(temperatures)  # the last hours, one value per minute
        >>> forecast.median  # the next 12 minutes
        >>> forecast.lower(0.1), forecast.upper(0.9)  # 80% prediction interval
    """

    _APP_SERVICE_NAME = "timestral-runner"

    def __init__(self, prediction_length: int = 24, timeout: float = 120.0, ready_timeout: float = 900.0) -> None:
        """Create the brick.

        Args:
            prediction_length: Default forecast horizon in steps, ``forecast()`` can override it per call.
            timeout: Seconds to wait for a forecast. A long context on a small board can take a few seconds.
            ready_timeout: Seconds ``start()`` waits for the runner to load the model, which the first time
                includes downloading it.

        Raises:
            ValueError: On a non-positive horizon or timeout.
        """
        if prediction_length < 1:
            raise ValueError("prediction_length must be at least 1.")
        if timeout <= 0 or ready_timeout <= 0:
            raise ValueError("timeout and ready_timeout must be positive.")
        self._prediction_length = prediction_length
        self._timeout = timeout
        self._ready_timeout = ready_timeout

        host = resolve_address(self._APP_SERVICE_NAME)
        if not host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")
        self._base_url = f"http://{host}:{RUNNER_PORT}"
        self._model_info: ModelInfo | None = None

    @property
    def prediction_length(self) -> int:
        """Default forecast horizon in steps."""
        return self._prediction_length

    @prediction_length.setter
    def prediction_length(self, value: int) -> None:
        if value < 1:
            raise ValueError("prediction_length must be at least 1.")
        self._prediction_length = value

    @property
    def model_info(self) -> ModelInfo:
        """The checkpoint the runner serves, fetched once the runner is ready."""
        if self._model_info is None:
            self._model_info = self._fetch_model_info()
        return self._model_info

    def start(self) -> None:
        """Wait for the runner to have its model loaded. Called by App.run() before the loop starts."""
        self._wait_ready()
        info = self.model_info
        logger.info(f"Using {info.model} ({info.parameters / 1e6:.0f}M parameters) with a {info.context_length}-step context")

    def stop(self) -> None:
        """Nothing to release, the runner is stopped with the app."""

    def forecast(
        self,
        context: SeriesLike,
        prediction_length: int | None = None,
        *,
        past_covariates: SeriesLike | None = None,
        known_covariates: SeriesLike | None = None,
        quantile_levels: Sequence[float] | None = None,
    ) -> Forecast:
        """Forecast the steps after ``context``.

        Args:
            context: The observed values, oldest first, evenly spaced in time: a sequence of numbers for a
                univariate series, or one sequence per variate for a multivariate one (the variates are
                forecast jointly). None and NaN mark a missing value. Only the most recent
                ``model_info.context_length`` steps are read.
            prediction_length: Horizon in steps, the brick's ``prediction_length`` when omitted.
            past_covariates: Optional rows of explanatory values known over the context only, one row per
                covariate, each as long as the context.
            known_covariates: Optional rows of values known over the context and the horizon, such as a
                schedule or a calendar indicator, each as long as the context plus the horizon.
            quantile_levels: The quantiles to return, all the model's ``quantile_levels`` when omitted.

        Returns:
            The forecast quantiles, see ``Forecast``.

        Raises:
            ValueError: On an empty context or a non-positive horizon.
            ForecastError: When the runner refuses the request, e.g. a horizon over
                ``model_info.max_prediction_length``, or cannot be reached.
        """
        horizon = self._prediction_length if prediction_length is None else prediction_length
        if horizon < 1:
            raise ValueError("prediction_length must be at least 1.")
        rows, univariate = _rows(context, "context")

        payload: dict[str, Any] = {"context": rows[0] if univariate else rows, "prediction_length": horizon}
        if past_covariates is not None:
            payload["past_covariates"] = _rows(past_covariates, "past_covariates")[0]
        if known_covariates is not None:
            payload["known_covariates"] = _rows(known_covariates, "known_covariates")[0]
        if quantile_levels is not None:
            payload["quantile_levels"] = [float(q) for q in quantile_levels]

        body = self._post("/api/forecast", payload, timeout=self._timeout)
        quantiles = np.array(body["quantiles"], dtype=np.float64)  # None, never produced in practice, becomes NaN
        return Forecast(quantiles=quantiles, quantile_levels=tuple(body["quantile_levels"]), inference_ms=float(body["inference_ms"]))

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        announced = False
        while True:
            try:
                response = http.get(f"{self._base_url}/ready", timeout=5.0)
                if response.status_code == 200:
                    return
                detail = self._detail(response)
            except http.ConnectionFailed as e:
                detail = str(e)
            if time.monotonic() >= deadline:
                raise ForecastError(f"The forecast runner is not ready after {self._ready_timeout:.0f} s: {detail}")
            if not announced:
                logger.info("Waiting for the forecast runner to load the model, the first start downloads it...")
                announced = True
            time.sleep(READY_POLL_INTERVAL_S)

    def _fetch_model_info(self) -> ModelInfo:
        try:
            response = http.get(f"{self._base_url}/api/info", timeout=10.0)
        except http.ConnectionFailed as e:
            raise ForecastError(f"The forecast runner cannot be reached: {e}") from e
        if response.status_code != 200:
            raise ForecastError(f"The forecast runner has no model loaded: {self._detail(response)}")
        body = response.json()
        return ModelInfo(
            model=body["model"],
            parameters=int(body["parameters"]),
            patch_size=int(body["patch_size"]),
            context_length=int(body["context_length"]),
            max_prediction_length=int(body["max_prediction_length"]),
            quantile_levels=tuple(body["quantile_levels"]),
            supports_covariates=bool(body["supports_covariates"]),
            supports_known_covariates=bool(body["supports_known_covariates"]),
        )

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            response = http.post(f"{self._base_url}{path}", json=payload, timeout=timeout)
        except http.ConnectionFailed as e:
            raise ForecastError(f"The forecast runner cannot be reached: {e}") from e
        if response.status_code != 200:
            raise ForecastError(f"The forecast runner refused the request ({response.status_code}): {self._detail(response)}")
        return response.json()

    @staticmethod
    def _detail(response: http.Response) -> str:
        """The ``detail`` of an error response, or its text."""
        try:
            body: Any = response.json()
        except ValueError:
            return response.text
        # hasattr instead of isinstance: narrowing Any to dict or list leaves pyright with unknown element types
        detail: Any = body.get("detail") if hasattr(body, "get") else None
        if hasattr(detail, "__iter__") and not isinstance(detail, str):  # Validation errors come as a list of {loc, msg, type}
            return "; ".join(str(item.get("msg", item)) if hasattr(item, "get") else str(item) for item in detail)
        return str(detail) if detail else response.text
