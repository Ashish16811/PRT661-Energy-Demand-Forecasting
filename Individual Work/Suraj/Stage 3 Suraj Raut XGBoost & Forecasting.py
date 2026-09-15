#!/usr/bin/env python3
"""
PRT661 – Data Science Practice
Australian Electricity Demand Forecasting
WP3 contribution module – Suraj Raut (S391201)

Primary WP3 scope represented here
----------------------------------
1. XGBoost A/B candidate family.
2. Recursive time-series engine used by long-horizon ML forecasts.
3. Lag provenance: ACTUAL_HISTORY versus RECURSIVE_FORECAST.
4. Frozen month-start 2026 forecast construction and monthly summary.
5. Reusable current-month + next-three-month horizon logic.

The final production file integrates this time-series engine with the other WP3
workstreams.  Keeping recursion separate makes the long-horizon information
boundary explicit and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_STATE = 20260914
FREQ = "30min"
INTERVALS_PER_DAY = 48
BLOCK = INTERVALS_PER_DAY
HOURS_PER_INTERVAL = 0.5


def build_xgboost(quick: bool = False):
    """Exact XGBoost configuration used by the integrated Stage 3 design."""
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=300 if quick else 700,
        max_depth=6 if quick else 8,
        learning_rate=0.08 if quick else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def rolling_four_month_horizon(origin: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    """Current calendar month plus the next three months, including year rollover."""
    start = pd.Timestamp(origin).to_period("M").to_timestamp()
    end_period = start.to_period("M") + 3
    end = end_period.end_time.floor(FREQ)
    months = [str(start.to_period("M") + i) for i in range(4)]
    return start, end, months


def lag_provenance(
    full_index: pd.DatetimeIndex,
    origin_pos: int,
    forecast_positions: np.ndarray,
    lags: tuple[int, ...] = (48, 96, 336),
) -> pd.DataFrame:
    """Label each lag reference as actual history, recursive forecast or missing."""
    out: dict[str, np.ndarray] = {}
    n = len(full_index)
    for lag in lags:
        ref = forecast_positions - lag
        src = np.full(len(forecast_positions), "RECURSIVE_FORECAST", dtype=object)
        src[ref < 0] = "MISSING"
        src[(ref >= 0) & (ref < origin_pos)] = "ACTUAL_HISTORY"
        src[ref >= n] = "MISSING"
        out[f"lag{lag}_source"] = src
    df = pd.DataFrame(out, index=full_index[forecast_positions])
    df["recursive_input_used"] = (df != "ACTUAL_HISTORY").any(axis=1).astype(int)
    return df


def demand_features_at(y: np.ndarray, positions: np.ndarray) -> pd.DataFrame:
    """Rebuild long-horizon demand features from history or earlier predictions."""
    def gather(lag: int) -> np.ndarray:
        ref = positions - lag
        values = np.full(len(positions), np.nan, dtype=float)
        valid = (ref >= 0) & (ref < len(y))
        values[valid] = y[ref[valid]]
        return values

    data: dict[str, np.ndarray] = {
        "lag_48": gather(48),
        "lag_96": gather(96),
        "lag_336": gather(336),
    }

    # Previous day summaries anchored at t-48.
    means, peaks, mins, same7, same14, stds = [], [], [], [], [], []
    for p in positions:
        prev = y[max(0, p - 95): p - 47] if p - 47 > 0 else np.array([])
        means.append(float(np.nanmean(prev)) if len(prev) else np.nan)
        peaks.append(float(np.nanmax(prev)) if len(prev) and np.isfinite(prev).any() else np.nan)
        mins.append(float(np.nanmin(prev)) if len(prev) and np.isfinite(prev).any() else np.nan)
        vals7 = [y[p - 48*k] if p - 48*k >= 0 else np.nan for k in range(1, 8)]
        vals14 = [y[p - 48*k] if p - 48*k >= 0 else np.nan for k in range(1, 15)]
        same7.append(float(np.nanmean(vals7)))
        same14.append(float(np.nanmean(vals14)))
        stds.append(float(np.nanstd(prev, ddof=1)) if len(prev) > 1 else np.nan)

    data.update({
        "previous_day_mean": np.array(means),
        "previous_day_peak": np.array(peaks),
        "previous_day_min": np.array(mins),
        "same_half_hour_7day_mean": np.array(same7),
        "same_half_hour_14day_mean": np.array(same14),
        "rolling_mean_24h_at_t_minus_48": np.array(means),
        "rolling_std_24h_at_t_minus_48": np.array(stds),
    })
    return pd.DataFrame(data, index=positions)


@dataclass
class RecursiveForecastResult:
    forecast: pd.Series
    provenance: pd.DataFrame
    filled_feature_cells: int


def recursive_ml_forecast(
    model,
    full_index: pd.DatetimeIndex,
    y_history: np.ndarray,
    origin_pos: int,
    known_future: pd.DataFrame,
    feature_names: list[str],
    training_feature_medians: pd.Series,
) -> RecursiveForecastResult:
    """Forecast one day at a time so lagged demand can consume earlier predictions.

    `known_future` must already contain only forecast-safe calendar/weather/profile
    features. Demand-derived features are rebuilt here from the evolving y array.
    """
    y = np.asarray(y_history, dtype=float).copy()
    if len(y) < len(full_index):
        y = np.pad(y, (0, len(full_index) - len(y)), constant_values=np.nan)

    grid_pos = np.arange(origin_pos, len(full_index))
    preds = np.full(len(grid_pos), np.nan)
    filled = 0

    demand_cols = {
        "lag_48", "lag_96", "lag_336",
        "previous_day_mean", "previous_day_peak", "previous_day_min",
        "same_half_hour_7day_mean", "same_half_hour_14day_mean",
        "rolling_mean_24h_at_t_minus_48", "rolling_std_24h_at_t_minus_48",
    }

    for start in range(0, len(grid_pos), BLOCK):
        block_pos = grid_pos[start:start + BLOCK]
        block_idx = full_index[block_pos]
        X = known_future.reindex(block_idx).copy()
        dyn = demand_features_at(y, block_pos)
        dyn.index = block_idx
        for col in demand_cols:
            if col in feature_names:
                X[col] = dyn[col]
        X = X.reindex(columns=feature_names)
        filled += int(X.isna().to_numpy().sum())
        X = X.fillna(training_feature_medians)
        yhat = model.predict(X.to_numpy(dtype=np.float32))
        y[block_pos] = yhat
        preds[start:start + len(block_pos)] = yhat

    fc_index = full_index[grid_pos]
    prov = lag_provenance(full_index, origin_pos, grid_pos)
    return RecursiveForecastResult(
        forecast=pd.Series(preds, index=fc_index, name="forecast_MW"),
        provenance=prov,
        filled_feature_cells=filled,
    )


def frozen_forecast_frame(
    region: str,
    origin: pd.Timestamp,
    result: RecursiveForecastResult,
    model_name: str,
    feature_set: str,
    weather_source: pd.Series | str = "NONE_MODEL_A",
) -> pd.DataFrame:
    """Build the auditable month-start forecast product used for later verification."""
    idx = result.forecast.index
    if isinstance(weather_source, pd.Series):
        ws = weather_source.reindex(idx).fillna("MISSING").astype(str).to_numpy()
    else:
        ws = np.repeat(str(weather_source), len(idx))
    step = np.arange(1, len(idx) + 1)
    p = result.provenance.reindex(idx)
    return pd.DataFrame({
        "REGION": region,
        "forecast_origin": pd.Timestamp(origin),
        "target_timestamp": idx,
        "forecast_MW": result.forecast.to_numpy(),
        "selected_model": model_name,
        "feature_set": feature_set,
        "forecast_horizon_step": step,
        "forecast_horizon_days": np.round(step / INTERVALS_PER_DAY, 3),
        "lag48_source": p["lag48_source"].to_numpy(),
        "lag96_source": p["lag96_source"].to_numpy(),
        "lag336_source": p["lag336_source"].to_numpy(),
        "recursive_input_used": p["recursive_input_used"].to_numpy(),
        "weather_feature_source": ws,
    })


def monthly_forecast_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    """Readable monthly view of a four-month half-hour forecast."""
    f = forecast.dropna(subset=["forecast_MW"]).copy()
    f["month"] = pd.to_datetime(f["target_timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, g in f.groupby("month"):
        peak_idx = g["forecast_MW"].idxmax()
        rows.append({
            "REGION": g["REGION"].iloc[0],
            "month": month,
            "mean_forecast_MW": float(g["forecast_MW"].mean()),
            "minimum_forecast_MW": float(g["forecast_MW"].min()),
            "maximum_forecast_MW": float(g["forecast_MW"].max()),
            "peak_timestamp": g.loc[peak_idx, "target_timestamp"],
            "number_of_intervals": len(g),
            "forecast_energy_MWh": float(g["forecast_MW"].sum() * HOURS_PER_INTERVAL),
            "recursive_input_share_pct": float(100.0 * g["recursive_input_used"].mean()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Suraj Raut WP3 module loaded: XGBoost, recursive engine, lag provenance and frozen four-month forecast.")
