# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""HTTP API serving Timestral forecasts to the app container.

The checkpoint named by ``TIMESTRAL_MODEL`` is loaded in the background when the server starts, and
downloaded first into the Hugging Face Hub cache (``HF_HOME``) when it is not there yet. ``/ready``
answers 200 once the model is loaded, so the compose healthcheck covers the download too, and
``/api/forecast`` answers 503 until then.

Timestral inference is CPU bound and runs on ``TIMESTRAL_THREADS`` torch threads: the forecasts are
serialized, two concurrent requests would only compete for the same cores.
"""

import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("forecast_server")

MODEL = os.getenv("TIMESTRAL_MODEL", "mistralai/Timestral-5M")
THREADS = int(os.getenv("TIMESTRAL_THREADS", "4"))
# A horizon decodes in blocks of 1024 steps fed back as context, bound the request to one block by default
MAX_PREDICTION_LENGTH = int(os.getenv("TIMESTRAL_MAX_PREDICTION_LENGTH", "1024"))
# The checkpoints are trained on 32768 steps and cap the context themselves, this bounds the request size
MAX_CONTEXT_LENGTH = int(os.getenv("TIMESTRAL_MAX_CONTEXT_LENGTH", "32768"))

Series = list[float | None]
"""One row of values along time, None marks a missing value."""


class ModelState:
    """The forecaster, loaded once in the background, and the outcome of the loading."""

    def __init__(self, model: str, threads: int) -> None:
        self.model = model
        self.threads = threads
        self.forecaster: Any = None
        self.error: str | None = None
        self.load_seconds: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def loaded(self) -> bool:
        return self.forecaster is not None

    def load_in_background(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self.load, name="timestral-load", daemon=True)
            self._thread.start()

    def load(self) -> None:
        """Load the checkpoint, downloading it when it is not in the Hub cache."""
        started = time.monotonic()
        try:
            import torch
            from timestral import TimestralForecaster

            torch.set_num_threads(self.threads)
            logger.info("loading %s on %d threads", self.model, self.threads)
            forecaster = TimestralForecaster.from_pretrained(self.model)
        except Exception as e:  # noqa: BLE001 - whatever failed, the readiness endpoint reports it
            logger.exception("loading %s failed", self.model)
            self.error = f"{type(e).__name__}: {e}"
            return
        self.load_seconds = time.monotonic() - started
        self.forecaster = forecaster
        logger.info("loaded %s in %.1f s", self.model, self.load_seconds)

    def forecast(self, **kwargs: Any) -> Any:
        """Run one forecast, serialized with the others."""
        with self._lock:
            return self.forecaster.forecast(**kwargs)

    def info(self) -> dict[str, Any]:
        forecaster = self.forecaster
        config = forecaster.config
        return {
            "model": self.model,
            "parameters": sum(p.numel() for p in forecaster.model.parameters()),
            "patch_size": config.patch_size,
            "context_length": forecaster.max_context,
            "max_prediction_length": MAX_PREDICTION_LENGTH,
            "quantile_levels": list(forecaster.quantile_levels),
            "supports_covariates": forecaster.supports_covariates,
            "supports_known_covariates": forecaster.supports_known_covariates,
            "threads": self.threads,
            "load_seconds": self.load_seconds,
        }


class ForecastRequest(BaseModel):
    """A forecast of one series, univariate or multivariate.

    ``context`` is the observed past, one list of values for a univariate series or one list per
    variate for a multivariate one (the variates are forecast jointly). ``None`` marks a missing
    value. Covariates are rows aligned with the context: the past ones cover the context, the known
    ones cover the context plus the horizon.
    """

    context: Series | list[Series]
    prediction_length: int = Field(ge=1)
    past_covariates: list[Series] | None = None
    known_covariates: list[Series] | None = None
    quantile_levels: list[float] | None = None
    symmetric_averaging: bool = True
    clip_nonnegative: bool = True

    @model_validator(mode="after")
    def check_shapes(self) -> "ForecastRequest":
        context = self.context_rows
        if not context or not context[0]:
            raise ValueError("context must hold at least one value")
        length = len(context[0])
        if any(len(row) != length for row in context):
            raise ValueError("every variate of the context must have the same length")
        if length > MAX_CONTEXT_LENGTH:
            raise ValueError(f"context holds {length} steps, at most {MAX_CONTEXT_LENGTH} are accepted")
        if self.prediction_length > MAX_PREDICTION_LENGTH:
            raise ValueError(f"prediction_length {self.prediction_length} exceeds the limit of {MAX_PREDICTION_LENGTH}")
        for name, rows, expected in (
            ("past_covariates", self.past_covariates, length),
            ("known_covariates", self.known_covariates, length + self.prediction_length),
        ):
            if rows is not None and any(len(row) != expected for row in rows):
                raise ValueError(f"every row of {name} must hold {expected} values")
        return self

    @property
    def univariate(self) -> bool:
        return not (self.context and isinstance(self.context[0], list))

    @property
    def context_rows(self) -> list[Series]:
        return [self.context] if self.univariate else self.context  # type: ignore[list-item, return-value]


def _as_array(rows: list[Series]) -> np.ndarray:
    """``(1, rows, length)`` float32 array of one batch, missing values as NaN."""
    return np.array([[math.nan if v is None else v for v in row] for row in rows], dtype=np.float32)[None]


def _finite(values: np.ndarray) -> list[Any]:
    """Nested lists with None where the model produced no finite value."""
    return np.where(np.isfinite(values), values.astype(object), None).tolist()


app = FastAPI(title="Timestral forecast runner", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
state = ModelState(MODEL, THREADS)


@asynccontextmanager
async def lifespan(_: FastAPI):  # noqa: ANN201
    state.load_in_background()
    yield


app.router.lifespan_context = lifespan


@app.get("/health")
def health() -> dict[str, Any]:
    """Report the process is up and where the model loading stands."""
    return {"status": "ok", "model": state.model, "loaded": state.loaded, "error": state.error}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """200 once the model is loaded, 503 while it loads or when the loading failed."""
    if state.error is not None:
        raise HTTPException(status_code=503, detail=f"model loading failed: {state.error}")
    if not state.loaded:
        raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ready", "model": state.model}


@app.get("/api/info")
def info() -> dict[str, Any]:
    """The loaded checkpoint and its limits."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "model loading")
    return state.info()


@app.post("/api/forecast")
def forecast(request: ForecastRequest) -> dict[str, Any]:
    """Forecast ``prediction_length`` steps after the context.

    The response holds the ``median`` and the ``quantiles`` as lists shaped like the context:
    ``[horizon]`` and ``[horizon][quantile]`` for a univariate series, ``[variate][horizon]`` and
    ``[variate][horizon][quantile]`` for a multivariate one, with the ``quantile_levels`` they are at.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "model loading")

    kwargs: dict[str, Any] = {
        "context": _as_array(request.context_rows),
        "prediction_length": request.prediction_length,
        "symmetric_averaging": request.symmetric_averaging,
        "clip_nonnegative": request.clip_nonnegative,
    }
    if request.past_covariates:
        kwargs["past_covariates"] = _as_array(request.past_covariates)
    if request.known_covariates:
        kwargs["known_covariates"] = _as_array(request.known_covariates)

    started = time.monotonic()
    try:
        result = state.forecast(**kwargs)
    except (ValueError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    inference_ms = (time.monotonic() - started) * 1000

    levels = list(result.quantile_levels)
    quantiles = np.asarray(result.quantiles, dtype=np.float32)[0]  # (V, H, Q), the one series of the batch
    if request.quantile_levels is not None:
        missing = [q for q in request.quantile_levels if q not in levels]
        if missing:
            raise HTTPException(status_code=422, detail=f"quantile levels {missing} are not among {levels}")
        indexes = [levels.index(q) for q in request.quantile_levels]
        quantiles = quantiles[..., indexes]
        levels = list(request.quantile_levels)
    median = quantiles[..., levels.index(0.5)] if 0.5 in levels else np.asarray(result.median, dtype=np.float32)[0]
    if request.univariate:
        quantiles, median = quantiles[0], median[0]

    return {
        "median": _finite(median),
        "quantiles": _finite(quantiles),
        "quantile_levels": levels,
        "prediction_length": request.prediction_length,
        "inference_ms": round(inference_ms, 1),
    }
