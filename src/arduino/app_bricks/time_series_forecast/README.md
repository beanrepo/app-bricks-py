# Time Series Forecast brick

Zero-shot probabilistic forecasting of sensor time series with the [Timestral](https://mistral.ai/news/timestral/) foundation models by Mistral AI, running on the CPU.

## Overview

Give the brick the past values of a regularly sampled signal and it predicts the next steps, without any training. The forecast comes as quantiles, so every step has a point estimate (the median) and a prediction interval. Any evenly spaced series works: temperature, power consumption, vibration RMS, counts, prices. The model reads values, not timestamps, so the sampling period of the context is the step of the forecast.

The model runs in the `timestral-runner` container on the CPU of the board, no NPU is involved, so the brick works on both the UNO Q and the VENTUNO Q.

## Features

- Zero-shot forecasting: no training, no model to pick for the signal at hand.
- Probabilistic output: 21 quantiles from 0.01 to 0.99, the 0.5 one being the median.
- Univariate and multivariate series: several variates of one series are forecast jointly.
- Missing values: `None` or `NaN` in the context are handled by the model.
- Covariates: explanatory rows known over the past, or over the past and the future (schedules, calendar indicators).
- A context up to 32,768 steps, a horizon up to 1,024 steps per request.

## Models

| Model | Parameters | Weights | Note |
|---|---|---|---|
| `mistralai/Timestral-5M` | 5M | 21 MB | Default, the fastest |
| `mistralai/Timestral-28M` | 28M | 111 MB | More accurate, still fast on the CPU |

The checkpoint is selected with the `TIMESTRAL_MODEL` variable of the brick and downloaded from Hugging Face the first time an app using it starts, into `/home/arduino/.arduino-bricks/timestral` on the board, shared by every app. The first start therefore needs an internet connection and takes as long as the download.

The Timestral inference code is Apache 2.0, the weights are released under a Modified MIT license: check its terms, in particular the revenue condition, at the model page on Hugging Face before using them in a product.

## Usage

```python
from arduino.app_bricks.time_series_forecast import TimeSeriesForecast
from arduino.app_utils import App

forecaster = TimeSeriesForecast(prediction_length=12)

temperatures = [21.3, 21.4, 21.6, 21.9, 22.1, 22.0, 22.3, 22.6, 22.8, 23.0, 23.1, 23.3]  # one per minute, oldest first


def loop():
    forecast = forecaster.forecast(temperatures)
    print("next 12 minutes:", forecast.median.round(1))
    print("80% interval:", forecast.lower(0.1).round(1), forecast.upper(0.9).round(1))
    App.sleep(60)


App.run(user_loop=loop)
```

A multivariate series is one row per variate, forecast jointly:

```python
forecast = forecaster.forecast([temperatures, humidities], prediction_length=6)
forecast.median.shape  # (2, 6): variates, horizon
```

Covariates known over the horizon, here an "office hours" indicator, condition the forecast:

```python
forecast = forecaster.forecast(power, prediction_length=24, known_covariates=[office_hours_past + office_hours_next_24])
```

`forecaster.model_info` tells which checkpoint is loaded, its context length and the quantile levels.

## Variables

| Variable | Default | Description |
|---|---|---|
| `TIMESTRAL_MODEL` | `mistralai/Timestral-5M` | Hugging Face repository id of the checkpoint |
| `TIMESTRAL_THREADS` | `4` | CPU threads of the inference |
