#!/usr/bin/env python3
"""
PRT661 – Data Science Practice
Australian Electricity Demand Forecasting
WP3 contribution module – Bishal Dahal (S388095)

Primary WP3 scope represented here
----------------------------------
1. Feature-safety gate before any model is allowed to run.
2. Seasonal-naive baselines (Naive48 and Naive336).
3. Forecast evaluation: MAE, RMSE, sMAPE, Bias and Median Absolute Error.
4. Skill against the best seasonal-naive benchmark.
5. Current-month actual verification and forecast-integrity checks.

This file is a readable role-aligned module extracted/refactored from the final
integrated Stage 3 design.  The production entry point remains the jointly
integrated 03_Modelling.py pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TIME = "SETTLEMENTDATE"
TARGET = "TOTALDEMAND"
FREQ = "30min"
INTERVALS_PER_DAY = 48
SMAPE_MIN_DENOM = 1.0

# Benchmark/context fields that must not become predictors.
FORBIDDEN_PATTERNS = [
    r"aemo", r"predispatch", r"pre_dispatch", r"marketrequirement",
    r"^temp_mean_c$", r"^temp_max_c$", r"^temp_min_c$", r"^rainfall_mm$",
    r"^humidity_pct$", r"^temp_halfhourly_c$", r"^hdd$", r"^cdd$",
]


def metrics(actual: Iterable[float], forecast: Iterable[float]) -> dict:
    """Return the Stage 3 evaluation metrics.

    Error is defined consistently as forecast - actual.
    Positive Bias therefore means over-forecasting and negative Bias means
    under-forecasting.  sMAPE excludes only intervals with an effectively zero
    denominator; the excluded count is reported instead of silently clipping it.
    """
    a = np.asarray(list(actual), dtype=float)
    f = np.asarray(list(forecast), dtype=float)
    ok = np.isfinite(a) & np.isfinite(f)
    a, f = a[ok], f[ok]

    if a.size == 0:
        return {
            "matched_intervals": 0, "MAE": np.nan, "RMSE": np.nan,
            "sMAPE": np.nan, "Bias": np.nan, "MedAE": np.nan,
            "smape_excluded_intervals": 0,
        }

    err = f - a
    denom = np.abs(f) + np.abs(a)
    usable = denom >= SMAPE_MIN_DENOM
    smape = (
        float(np.mean(200.0 * np.abs(err[usable]) / denom[usable]))
        if usable.any() else np.nan
    )
    return {
        "matched_intervals": int(a.size),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "sMAPE": smape,
        "Bias": float(np.mean(err)),
        "MedAE": float(np.median(np.abs(err))),
        "smape_excluded_intervals": int((~usable).sum()),
    }


def skill_pct(model_mae: float, baseline_mae: float) -> float:
    """Percentage MAE improvement over a baseline; positive is better."""
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return np.nan
    return 100.0 * (baseline_mae - model_mae) / baseline_mae


def audit_feature_safety(features_A: list[str], features_B: list[str]) -> pd.DataFrame:
    """Independent feature gate used before model fitting.

    Model B may contain the approved weather_* fields because Stage 2 supplies
    their provenance.  What is blocked here are benchmark/context variables or
    legacy realised-weather columns that would create leakage or circularity.
    """
    rows = []
    for label, features in (("A", features_A), ("B", features_B)):
        offenders = [
            f for f in features
            if any(re.search(p, f, flags=re.I) for p in FORBIDDEN_PATTERNS)
        ]
        rows.append({
            "feature_set": label,
            "feature_count": len(features),
            "result": "PASS" if not offenders else "FAIL",
            "offenders": ", ".join(offenders),
        })

    rows.append({
        "feature_set": "B extends A",
        "feature_count": len(set(features_B) - set(features_A)),
        "result": "PASS" if set(features_A).issubset(set(features_B)) else "FAIL",
        "offenders": "",
    })
    return pd.DataFrame(rows)


@dataclass
class NaiveForecast:
    name: str
    forecast: pd.Series
    status: str
    detail: str


def recursive_seasonal_naive(
    history: pd.Series,
    forecast_index: pd.DatetimeIndex,
    lag_intervals: int,
    name: str,
) -> NaiveForecast:
    """Deployment-honest seasonal naive.

    Once the lag points into the forecast horizon, the method consumes its own
    earlier prediction.  That is the same recursive behaviour the full
    four-month system experiences instead of pretending future actual demand is
    already known.
    """
    history = history.sort_index().astype(float)
    full_index = history.index.append(forecast_index)
    y = pd.Series(index=full_index, dtype=float)
    y.loc[history.index] = history.values

    for ts in forecast_index:
        pos = full_index.get_loc(ts)
        ref = pos - lag_intervals
        y.loc[ts] = y.iloc[ref] if ref >= 0 else np.nan

    fc = y.reindex(forecast_index)
    missing = int(fc.isna().sum())
    return NaiveForecast(
        name=name,
        forecast=fc,
        status="MODEL_READY" if missing == 0 else "MODEL_READY_WITH_WARNING",
        detail=f"recursive seasonal naive lag={lag_intervals}; missing={missing}",
    )


def run_naive48(history: pd.Series, forecast_index: pd.DatetimeIndex) -> NaiveForecast:
    return recursive_seasonal_naive(history, forecast_index, 48, "Naive48")


def run_naive336(history: pd.Series, forecast_index: pd.DatetimeIndex) -> NaiveForecast:
    return recursive_seasonal_naive(history, forecast_index, 336, "Naive336")


def score_against_naive(
    actual: pd.Series,
    candidate: pd.Series,
    naive48: pd.Series,
    naive336: pd.Series,
) -> dict:
    """Score one candidate and calculate skill against the stronger naive."""
    idx = actual.index.intersection(candidate.index).intersection(naive48.index).intersection(naive336.index)
    cand = metrics(actual.reindex(idx), candidate.reindex(idx))
    n48 = metrics(actual.reindex(idx), naive48.reindex(idx))
    n336 = metrics(actual.reindex(idx), naive336.reindex(idx))
    baseline = min(n48["MAE"], n336["MAE"])
    return {
        **cand,
        "naive48_MAE": n48["MAE"],
        "naive336_MAE": n336["MAE"],
        "best_naive_MAE": baseline,
        "skill_vs_naive_pct": skill_pct(cand["MAE"], baseline),
    }


def verify_current_month(
    forecast: pd.DataFrame,
    actual: pd.Series,
    timestamp_col: str = "target_timestamp",
    forecast_col: str = "forecast_MW",
) -> tuple[pd.DataFrame, dict]:
    """Exact-timestamp verification using actuals kept outside model training."""
    if forecast.empty:
        return pd.DataFrame(), metrics([], [])

    f = forecast[[timestamp_col, forecast_col]].copy()
    f[timestamp_col] = pd.to_datetime(f[timestamp_col])
    a = actual.rename("actual_MW").copy()
    a.index = pd.to_datetime(a.index)

    joined = f.merge(a, left_on=timestamp_col, right_index=True, how="inner")
    joined = joined.dropna(subset=[forecast_col, "actual_MW"]).copy()
    if joined.empty:
        return joined, metrics([], [])

    joined["error_MW"] = joined[forecast_col] - joined["actual_MW"]
    joined["absolute_error_MW"] = joined["error_MW"].abs()
    joined["squared_error"] = joined["error_MW"] ** 2
    denom = joined[forecast_col].abs() + joined["actual_MW"].abs()
    joined["smape_component"] = np.where(
        denom >= SMAPE_MIN_DENOM,
        200.0 * joined["absolute_error_MW"] / denom,
        np.nan,
    )
    return joined, metrics(joined["actual_MW"], joined[forecast_col])


def forecast_integrity_checks(
    forecast: pd.DataFrame,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    timestamp_col: str = "target_timestamp",
    forecast_col: str = "forecast_MW",
) -> pd.DataFrame:
    """Structural forecast checks used before accepting an output vintage."""
    f = forecast.copy()
    f[timestamp_col] = pd.to_datetime(f[timestamp_col])
    ts = f[timestamp_col]
    values = pd.to_numeric(f[forecast_col], errors="coerce")

    expected_n = int((pd.Timestamp(expected_end) - pd.Timestamp(expected_start)) / pd.Timedelta(FREQ)) + 1
    spacing = ts.sort_values().diff().dropna()
    rows = [
        ("no_NaN", "PASS" if values.notna().all() else "FAIL", f"{int(values.isna().sum())} NaN"),
        ("no_Inf", "PASS" if np.isfinite(values.dropna()).all() else "FAIL", "finite forecast values"),
        ("no_duplicate_timestamp", "PASS" if not ts.duplicated().any() else "FAIL", f"{int(ts.duplicated().sum())} duplicates"),
        ("uniform_30min_spacing", "PASS" if (spacing == pd.Timedelta(FREQ)).all() else "FAIL", "30-minute spacing"),
        ("horizon_start", "PASS" if ts.min() == pd.Timestamp(expected_start) else "WARNING", f"actual={ts.min()} expected={expected_start}"),
        ("horizon_end", "PASS" if ts.max() == pd.Timestamp(expected_end) else "WARNING", f"actual={ts.max()} expected={expected_end}"),
        ("interval_count", "PASS" if len(f) == expected_n else "WARNING", f"{len(f)} rows vs {expected_n} expected"),
        ("no_negative_forecast", "PASS" if (values.dropna() >= 0).all() else "WARNING", f"{int((values.dropna() < 0).sum())} negative values"),
    ]
    return pd.DataFrame(rows, columns=["check", "result", "detail"])


if __name__ == "__main__":
    print("Bishal Dahal WP3 module loaded: feature safety, baselines, metrics and verification.")

# Work progress: 10 September 2026 - baselines, metrics and forecast verification
