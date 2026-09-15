#!/usr/bin/env python3
"""
PRT661 – Data Science Practice
Australian Electricity Demand Forecasting
WP3 contribution module – Sudip Lamichhane (S388085)

Primary WP3 scope represented here
----------------------------------
1. Statistical forecasting family: SARIMA and SARIMAX.
2. Live weather-vintage policy for the weather-enhanced Model B pathway.
3. AEMO secondary benchmark handling and target-semantics protection.
4. Source/vintage checks so a later forecast or realised weather cannot be used
   as if it were known at an earlier forecast origin.

This module mirrors the logic later integrated into the final group Stage 3
pipeline.  It is intentionally explicit about what is a forecast, what is an
actual, and what is only a secondary benchmark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

FREQ = "30min"
INTERVALS_PER_DAY = 48


@dataclass
class StatisticalRun:
    name: str
    status: str
    forecast: pd.Series | None
    fitted: object | None = None
    detail: str = ""
    runtime_seconds: float = 0.0


def _prepare_series(history: pd.Series, train_intervals: int) -> pd.Series:
    y = history.sort_index().dropna().astype(float).iloc[-train_intervals:]
    y = y.asfreq(FREQ)
    if y.isna().any():
        y = y.interpolate(limit_direction="both")
    return y


def run_sarima(
    history: pd.Series,
    forecast_index: pd.DatetimeIndex,
    train_intervals: int = 1440,
    quick: bool = False,
) -> StatisticalRun:
    """Classical seasonal time-series benchmark with daily seasonality."""
    t0 = time.time()
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        return StatisticalRun("SARIMA", "MODEL_FAILED", None, detail=f"statsmodels unavailable: {exc}")

    try:
        train = _prepare_series(history, train_intervals)
        order = (1, 0, 0) if quick else (1, 0, 1)
        seasonal = (1, 0, 0, INTERVALS_PER_DAY)
        fitted = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=60 if quick else 120)
        pred = fitted.get_forecast(steps=len(forecast_index)).predicted_mean
        pred.index = forecast_index
        converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
        status = "MODEL_READY" if converged else "MODEL_READY_WITH_WARNING"
        return StatisticalRun(
            "SARIMA", status, pred, fitted,
            detail=f"order={order}; seasonal_order={seasonal}; converged={converged}",
            runtime_seconds=time.time() - t0,
        )
    except Exception as exc:
        return StatisticalRun("SARIMA", "MODEL_FAILED", None, detail=f"{type(exc).__name__}: {exc}", runtime_seconds=time.time()-t0)


def _safe_exogenous_columns(train_exog: pd.DataFrame, future_exog: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[str]]:
    """Drop regressors that cannot be extrapolated safely across the horizon."""
    keep, dropped = [], []
    for c in candidates:
        if c not in train_exog or c not in future_exog:
            dropped.append(f"{c}(missing)")
            continue
        tr = pd.to_numeric(train_exog[c], errors="coerce")
        fc = pd.to_numeric(future_exog[c], errors="coerce")
        if tr.notna().sum() == 0 or tr.std() <= 1e-8:
            dropped.append(f"{c}(no training variation)")
            continue
        tr_range = float(tr.max() - tr.min())
        fc_range = float(fc.max() - fc.min())
        if fc_range > 1.5 * max(tr_range, 1e-9):
            dropped.append(f"{c}(future range outside training support)")
            continue
        keep.append(c)
    return keep, dropped


def run_sarimax(
    history: pd.Series,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
    candidate_exog: list[str],
    train_intervals: int = 1440,
    quick: bool = False,
) -> StatisticalRun:
    """SARIMAX with only future-known or forecast-safe exogenous variables."""
    t0 = time.time()
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        return StatisticalRun("SARIMAX", "MODEL_FAILED", None, detail=f"statsmodels unavailable: {exc}")

    try:
        train = _prepare_series(history, train_intervals)
        tx = train_exog.reindex(train.index).ffill().bfill()
        fx = future_exog.reindex(forecast_index).ffill().bfill()
        keep, dropped = _safe_exogenous_columns(tx, fx, candidate_exog)
        if not keep:
            return StatisticalRun("SARIMAX", "MODEL_FAILED", None, detail=f"no safe exogenous columns; dropped={dropped}")

        tx, fx = tx[keep], fx[keep]
        scale = tx.std().replace(0, 1.0)
        tx, fx = tx / scale, fx / scale
        order = (1, 0, 0) if quick else (1, 0, 1)
        seasonal = (1, 0, 0, INTERVALS_PER_DAY)
        fitted = SARIMAX(
            train,
            exog=tx,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=60 if quick else 120)
        pred = fitted.get_forecast(steps=len(forecast_index), exog=fx).predicted_mean
        pred.index = forecast_index
        converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
        status = "MODEL_READY" if converged else "MODEL_READY_WITH_WARNING"
        return StatisticalRun(
            "SARIMAX", status, pred, fitted,
            detail=f"exog={keep}; dropped={dropped}; converged={converged}",
            runtime_seconds=time.time() - t0,
        )
    except Exception as exc:
        return StatisticalRun("SARIMAX", "MODEL_FAILED", None, detail=f"{type(exc).__name__}: {exc}", runtime_seconds=time.time()-t0)


def validate_weather_vintage(frame: pd.DataFrame, forecast_origin: pd.Timestamp) -> pd.DataFrame:
    """Confirm that forecast-weather rows were issued no later than the model origin."""
    out = frame.copy()
    origin = pd.Timestamp(forecast_origin)
    if "weather_forecast_origin" not in out:
        out["weather_vintage_valid"] = False
        return out
    issue = pd.to_datetime(out["weather_forecast_origin"], errors="coerce")
    source = out.get("weather_feature_source", pd.Series("", index=out.index)).astype(str)
    is_forecast = source.eq("FORECAST_7DAY")
    out["weather_vintage_valid"] = (~is_forecast) | (issue.notna() & (issue <= origin))
    return out


def apply_live_weather_policy(
    future_frame: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    fallback_columns: dict[str, str],
) -> pd.DataFrame:
    """Use genuine 7-day forecast weather only when its vintage is valid.

    Beyond the reliable forecast window, replace direct weather inputs with the
    corresponding origin-safe climatology columns supplied by Stage 2.
    """
    out = validate_weather_vintage(future_frame, forecast_origin)
    src = out.get("weather_feature_source", pd.Series("MISSING", index=out.index)).astype(str)
    valid_fc = src.eq("FORECAST_7DAY") & out["weather_vintage_valid"]

    for weather_col, fallback_col in fallback_columns.items():
        if weather_col not in out:
            continue
        if fallback_col not in out:
            continue
        use_fallback = ~valid_fc
        out.loc[use_fallback, weather_col] = out.loc[use_fallback, fallback_col]

    out.loc[valid_fc, "weather_policy"] = "FORECAST_7DAY"
    out.loc[~valid_fc, "weather_policy"] = "CLIMATOLOGY_FALLBACK"
    return out


def aemo_source_consistency(aemo_actual: pd.DataFrame, operational_actual: pd.Series, time_col: str = "SETTLEMENTDATE") -> tuple[pd.DataFrame, dict]:
    """Compare sources without calling the difference forecasting error."""
    a = aemo_actual.copy()
    a[time_col] = pd.to_datetime(a[time_col])
    op = operational_actual.rename("operational_actual_MW").copy()
    op.index = pd.to_datetime(op.index)
    j = a.set_index(time_col).join(op, how="inner").dropna()
    if j.empty:
        return j, {"comparison_status": "NO_OVERLAP"}

    a_col = "aemo_actual_MW"
    j["source_difference_MW"] = j[a_col] - j["operational_actual_MW"]
    d = j["source_difference_MW"]
    summary = {
        "matched_intervals": len(j),
        "MAE_between_sources_MW": float(d.abs().mean()),
        "RMSE_between_sources_MW": float(np.sqrt((d ** 2).mean())),
        "Bias_between_sources_MW": float(d.mean()),
        "correlation": float(j[a_col].corr(j["operational_actual_MW"])) if len(j) > 2 else np.nan,
        "comparison_status": "TARGET_MISMATCH_SECONDARY_BENCHMARK",
        "interpretation": "source consistency / target relationship check - not forecast accuracy",
    }
    return j.reset_index(), summary


def aemo_forecast_margin(
    aemo_forecast: pd.DataFrame,
    project_forecast: pd.DataFrame,
    time_col: str = "SETTLEMENTDATE",
) -> tuple[pd.DataFrame, dict]:
    """Compare forecast divergence on exact shared timestamps only."""
    a = aemo_forecast.copy()
    a[time_col] = pd.to_datetime(a[time_col])
    p = project_forecast.copy()
    p["target_timestamp"] = pd.to_datetime(p["target_timestamp"])
    j = a.merge(p[["target_timestamp", "forecast_MW"]], left_on=time_col, right_on="target_timestamp", how="inner")
    j = j.dropna(subset=["aemo_forecast_MW", "forecast_MW"]).copy()
    if j.empty:
        return j, {"comparison_status": "FORECAST_ONLY", "matched_project_intervals": 0}

    j["forecast_margin_MW"] = j["forecast_MW"] - j["aemo_forecast_MW"]
    j["absolute_forecast_margin_MW"] = j["forecast_margin_MW"].abs()
    summary = {
        "matched_project_intervals": len(j),
        "mean_margin_MW": float(j["forecast_margin_MW"].mean()),
        "mean_absolute_margin_MW": float(j["absolute_forecast_margin_MW"].mean()),
        "comparison_status": "FORECAST_ONLY",
        "interpretation": "forecast divergence / margin - neither source is scored as better",
    }
    return j, summary


if __name__ == "__main__":
    print("Sudip Lamichhane WP3 module loaded: SARIMA/SARIMAX, weather-vintage policy and AEMO benchmark governance.")
