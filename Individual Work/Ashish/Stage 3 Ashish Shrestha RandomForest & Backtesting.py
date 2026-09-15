#!/usr/bin/env python3
"""
PRT661 – Data Science Practice
Australian Electricity Demand Forecasting
WP3 contribution module – Ashish Shrestha (S388084)

Primary WP3 scope represented here
----------------------------------
1. Random Forest A/B candidate family.
2. Short chronological screening logic.
3. Historical four-month analogue backtesting.
4. Deployment-safe aggregation and regional model selection.
5. A/B family comparison for understanding the incremental value of weather.

The production pipeline uses the same principles in the integrated
03_Modelling.py file.  This module keeps the allocated modelling/selection work
readable as a separate WP3 contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

RANDOM_STATE = 20260914
INTERVALS_PER_DAY = 48
TIE_TOLERANCE = 0.01

MODEL_ORDER = [
    "Naive48", "Naive336",
    "RandomForest_A", "RandomForest_B",
    "XGBoost_A", "XGBoost_B",
    "SARIMA", "SARIMAX",
]

COMPLEXITY = {
    "Naive48": 0, "Naive336": 0, "SARIMA": 2, "SARIMAX": 3,
    "RandomForest_A": 4, "XGBoost_A": 4,
    "RandomForest_B": 5, "XGBoost_B": 5,
}

FEATURE_SET_OF = {
    "Naive48": "none", "Naive336": "none",
    "RandomForest_A": "A", "XGBoost_A": "A",
    "RandomForest_B": "B", "XGBoost_B": "B",
    "SARIMA": "none", "SARIMAX": "exog_forecast_safe",
}


def build_random_forest(quick: bool = False):
    """Exact Random Forest configuration used by the integrated Stage 3 design."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=40 if quick else 150,
        max_depth=12 if quick else 18,
        min_samples_leaf=5 if quick else 2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str,
    quick: bool = False,
):
    """Fit RF-A or RF-B after the Stage 3 feature-safety gate has passed."""
    if feature_set not in {"A", "B"}:
        raise ValueError("feature_set must be 'A' or 'B'")
    model = build_random_forest(quick=quick)
    ok = X_train.notna().all(axis=1) & pd.to_numeric(y_train, errors="coerce").notna()
    X = X_train.loc[ok].astype(np.float32)
    y = pd.to_numeric(y_train.loc[ok], errors="coerce").astype(float)
    if len(X) < 2000:
        raise ValueError(f"Only {len(X)} complete rows available for RandomForest_{feature_set}")
    model.fit(X.to_numpy(dtype=np.float32), y.to_numpy())
    return model


def analogue_origins(
    anchor_month: pd.Period,
    data_start: pd.Timestamp,
    latest_actual: pd.Timestamp,
    max_origins: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return completed historical four-calendar-month windows matching the target season."""
    out = []
    for year in range(anchor_month.year - 1, data_start.year - 1, -1):
        origin = pd.Timestamp(year=year, month=anchor_month.month, day=1)
        end_period = origin.to_period("M") + 3
        end = end_period.end_time.floor("30min")
        # Need enough history before origin and a complete realised test horizon.
        if origin > data_start + pd.Timedelta(days=240) and end <= latest_actual:
            out.append((origin, end))
        if len(out) >= max_origins:
            break
    return out


def screen_decision(screen: pd.DataFrame) -> dict[str, bool]:
    """Drop only obviously weak ML configurations; screening never chooses the winner."""
    keep = {m: True for m in MODEL_ORDER}
    if screen.empty or "MAE" not in screen:
        return keep
    mean_mae = screen.groupby("model")["MAE"].mean().dropna()
    if mean_mae.empty:
        return keep
    best = float(mean_mae.min())
    for model, mae in mean_mae.items():
        if model in {"Naive48", "Naive336"}:
            continue
        if float(mae) > 2.0 * best:
            keep[model] = False
    if not any(keep[m] for m in MODEL_ORDER if m not in {"Naive48", "Naive336"}):
        keep = {m: True for m in MODEL_ORDER}
    return keep


def run_short_chronological_screen(
    history: pd.Series,
    candidate_runners: dict[str, Callable[[pd.Series, pd.DatetimeIndex], pd.Series]],
    folds: int = 3,
    test_days: int = 14,
) -> pd.DataFrame:
    """Generic three-fold chronological screen used only as an early failure filter.

    The full integrated project screens six non-statistical candidates.  Each
    runner receives training history ending immediately before its test window
    and must return predictions for the supplied test index.
    """
    from sklearn.metrics import mean_absolute_error

    y = history.sort_index().dropna()
    test_len = test_days * INTERVALS_PER_DAY
    rows = []
    for fold in range(folds, 0, -1):
        if fold * test_len >= len(y):
            continue
        cut_pos = len(y) - fold * test_len
        end_pos = len(y) - (fold - 1) * test_len
        train = y.iloc[:cut_pos]
        test = y.iloc[cut_pos:end_pos]
        for name, runner in candidate_runners.items():
            try:
                pred = runner(train, test.index).reindex(test.index)
                mae = mean_absolute_error(test, pred)
                rows.append({
                    "fold": folds - fold + 1,
                    "model": name,
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "MAE": float(mae),
                    "status": "MODEL_READY",
                })
            except Exception as exc:
                rows.append({"fold": folds - fold + 1, "model": name, "MAE": np.nan,
                             "status": "MODEL_FAILED", "detail": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def add_skill_against_naive(scores: pd.DataFrame) -> pd.DataFrame:
    """Add per-origin skill against the best of Naive48 and Naive336."""
    out = scores.copy()
    out["skill_vs_naive_pct"] = np.nan
    for (_, origin), grp in out.groupby(["region", "backtest_origin"]):
        baseline = grp[grp["model"].isin(["Naive48", "Naive336"])]["MAE"].min()
        if np.isfinite(baseline):
            idx = grp.index
            out.loc[idx, "baseline_MAE"] = baseline
            out.loc[idx, "skill_vs_naive_pct"] = 100.0 * (baseline - grp["MAE"]) / baseline
    return out


def aggregate_deployment_safe_scores(
    scores: pd.DataFrame,
    region: str,
    allowed_models: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate only deployment-safe statuses before selection."""
    grp = scores[(scores["region"] == region) & scores["MAE"].notna()].copy()
    if allowed_models is not None:
        grp = grp[grp["model"].isin(allowed_models)]
    if grp.empty:
        return pd.DataFrame()

    ready = grp[grp["status"] == "MODEL_READY"]
    if ready.empty:
        ready = grp[grp["status"] == "MODEL_READY_WITH_WARNING"]
    if ready.empty:
        return pd.DataFrame()

    agg = ready.groupby("model").agg(
        mean_backtest_MAE=("MAE", "mean"),
        median_backtest_MAE=("MAE", "median"),
        mean_RMSE=("RMSE", "mean"),
        mean_sMAPE=("sMAPE", "mean"),
        mean_Bias=("Bias", "mean"),
        mae_spread=("MAE", "std"),
        origins=("backtest_origin", "nunique"),
        skill_vs_naive_pct=("skill_vs_naive_pct", "mean"),
    ).reset_index()

    winners = []
    for _, fold in ready.groupby("backtest_origin"):
        if fold["MAE"].notna().any():
            winners.append(fold.loc[fold["MAE"].idxmin(), "model"])
    counts = pd.Series(winners).value_counts() if winners else pd.Series(dtype=float)
    agg["win_rate_pct"] = agg["model"].map(counts).fillna(0) / max(len(winners), 1) * 100.0
    return agg


def select_regional_model(scores: pd.DataFrame, region: str, allowed_models: set[str] | None = None) -> dict:
    """Primary rule: lowest mean four-month MAE; within 1%, prefer simpler/stabler."""
    agg = aggregate_deployment_safe_scores(scores, region, allowed_models)
    if agg.empty:
        return {"region": region, "selected_model": "NONE", "status": "MODEL_FAILED"}

    best_mae = float(agg["mean_backtest_MAE"].min())
    tied = agg[agg["mean_backtest_MAE"] <= best_mae * (1 + TIE_TOLERANCE)].copy()
    tied["complexity"] = tied["model"].map(COMPLEXITY).fillna(9)
    tied = tied.sort_values(["complexity", "mean_backtest_MAE", "mae_spread"], na_position="last")
    win = tied.iloc[0]
    return {
        "region": region,
        "selected_model": win["model"],
        "feature_set": FEATURE_SET_OF.get(win["model"], ""),
        "mean_backtest_MAE": float(win["mean_backtest_MAE"]),
        "mean_RMSE": float(win["mean_RMSE"]),
        "mean_sMAPE": float(win["mean_sMAPE"]),
        "mean_Bias": float(win["mean_Bias"]),
        "skill_vs_naive_pct": float(win["skill_vs_naive_pct"]),
        "win_rate_pct": float(win["win_rate_pct"]),
        "historical_origins_used": int(win["origins"]),
        "status": "MODEL_READY",
    }


def select_A_B_champions(scores: pd.DataFrame, region: str) -> tuple[dict, dict]:
    a = select_regional_model(scores, region, {"RandomForest_A", "XGBoost_A"})
    b = select_regional_model(scores, region, {"RandomForest_B", "XGBoost_B"})
    return a, b


if __name__ == "__main__":
    print("Ashish Shrestha WP3 module loaded: Random Forest, chronological screening, four-month backtesting and selection.")
