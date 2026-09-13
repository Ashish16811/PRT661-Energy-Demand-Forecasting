"""
wp3/evaluation.py - metrics, walk-forward harness and verification plots
PRT661 WP3 Modelling | Dan6: Theme 2
OWNER: Bishal Dahal (S388095) - Verification & Dashboard Lead
Action M4 from the 3 September register: "Prepare verification/evaluation
framework: MAE, RMSE, percentage error and bias."

Assessment 1 Section 7 names four metrics, and the fourth is the one most
often left out:

    MAE    average absolute error, in MW - the headline operational number
    RMSE   penalises large misses more heavily - catches bad peak hours
    MAPE   scale-free, so regions of very different size can be compared
    BIAS   mean signed error. Positive means the model over-forecasts.

Bias is reported because MAE and RMSE cannot distinguish a model that is
usually right from one that is equally wrong in both directions. Under-
forecasting risks a shortfall; over-forecasting wastes committed capacity. A
model with low MAE and large negative bias is systematically under-calling
demand, and that has to be visible.

MAPE is computed only where the actual is above MAPE_FLOOR. SA1 contains
readings at or below zero, and a percentage error against a near-zero
denominator is not a meaningful number - reporting it would put an
astronomical MAPE next to a perfectly reasonable MAE.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import INTERVALS_PER_DAY

MAPE_FLOOR = 100.0     # MW - below this a percentage error is meaningless


def metrics(y_true, y_pred):
    """The four Assessment 1 metrics, plus the counts behind them."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]

    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                "bias": np.nan, "n_mape": 0}

    err = y_pred - y_true                      # signed: positive = over-forecast
    big = np.abs(y_true) >= MAPE_FLOOR
    mape = (np.mean(np.abs(err[big] / y_true[big])) * 100
            if big.any() else np.nan)

    return {"n": int(len(y_true)),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": float(mape),
            "bias": float(np.mean(err)),
            "n_mape": int(big.sum())}


def skill_vs_baseline(scores, baseline="seasonal_naive_48", metric="MAE"):
    """
    Percentage improvement over the tier-1 benchmark, per region.

    This is the number that decides whether a model earns its complexity. A
    negative skill score means the model is worse than repeating yesterday.
    """
    rows = []
    for region, g in scores.groupby("region"):
        base = g.loc[g["model"] == baseline, metric]
        if base.empty or not np.isfinite(base.iloc[0]):
            continue
        b = base.iloc[0]
        for _, r in g.iterrows():
            rows.append({"region": region, "model": r["model"], "tier": r.get("tier", ""),
                         metric: r[metric],
                         f"skill_vs_{baseline}_pct": round(100 * (b - r[metric]) / b, 2)})
    return pd.DataFrame(rows)


def fold_table(records):
    """Per-fold results, kept so variation across folds is visible rather than
    hidden inside an average."""
    return pd.DataFrame(records)


def aggregate_folds(fold_df):
    """
    Mean and spread across walk-forward folds.

    The standard deviation is reported alongside the mean: a model that is
    excellent in three folds and poor in the fourth is not the same as one that
    is consistently mediocre, and an average alone cannot tell them apart.
    """
    g = fold_df.groupby(["region", "model", "tier"], as_index=False).agg(
        folds=("fold", "count"),
        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"),
        bias=("bias", "mean"), n=("n", "sum"))
    return g.sort_values(["region", "MAE"])


def acceptance_table(agg, baseline="seasonal_naive_48", min_skill_pct=5.0):
    """
    Action M5: the rule for moving beyond the seasonal-naive baseline, applied
    rather than described.

    A model is ACCEPTED for a region only if it beats the benchmark on MAE by
    at least min_skill_pct. Anything less is not worth the extra fitting cost,
    the extra failure modes and the loss of interpretability.
    """
    rows = []
    for region, g in agg.groupby("region"):
        base = g.loc[g["model"] == baseline, "MAE"]
        if base.empty:
            continue
        b = base.iloc[0]
        best = g.loc[g["MAE"].idxmin()]
        skill = 100 * (b - best["MAE"]) / b
        rows.append({
            "region": region,
            "baseline_MAE": round(b, 2),
            "best_model": best["model"], "best_tier": best["tier"],
            "best_MAE": round(best["MAE"], 2),
            "skill_pct": round(skill, 2),
            "decision": ("ACCEPT - beats benchmark" if skill >= min_skill_pct
                         else "REJECT - benchmark retained"),
            "rule": f"accept only if MAE improves on {baseline} by >= {min_skill_pct}%",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(agg, out_dir):
    """MAE by model and region - the headline comparison figure."""
    piv = agg.pivot_table(index="region", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(11, 4))
    piv.plot(kind="bar", ax=ax)
    ax.set(title="Day-ahead MAE by model and region (walk-forward mean)",
           ylabel="MAE (MW)", xlabel="")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, out_dir / "model_comparison_mae.png")
    return "model_comparison_mae.png"


def plot_skill(acc, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colours = ["tab:green" if s >= 5 else "tab:red" for s in acc["skill_pct"]]
    ax.bar(acc["region"], acc["skill_pct"], color=colours)
    ax.axhline(5, ls="--", c="0.4", lw=1, label="5% acceptance threshold")
    ax.set(title="Best model skill against the seasonal-naive benchmark",
           ylabel="MAE improvement (%)")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "skill_vs_baseline.png")
    return "skill_vs_baseline.png"


def plot_forecast_window(actual_df, preds, region, out_dir, days=7):
    """Actual against each model over the final week of the last fold."""
    n = days * INTERVALS_PER_DAY
    tail = actual_df.tail(n)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(tail["SETTLEMENTDATE"], tail["actual"], color="black", lw=1.4, label="actual")
    for name in preds:
        if name in tail.columns:
            ax.plot(tail["SETTLEMENTDATE"], tail[name], lw=0.9, alpha=0.85, label=name)
    ax.set(title=f"{region} - day-ahead forecasts vs actual, final {days} days",
           ylabel="MW", xlabel="")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, out_dir / f"{region}_forecast_vs_actual.png")
    return f"{region}_forecast_vs_actual.png"


def plot_error_profile(resid_df, region, out_dir):
    """
    Mean signed error by half-hour of day.

    A flat line near zero is what a well-calibrated model looks like. A shape
    here means the model is systematically wrong at particular times of day -
    usually the morning ramp or the evening peak, which are exactly the hours
    a demand forecast exists to get right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for name, g in resid_df.groupby("model"):
        g.groupby("half_hour_index")["error"].mean().plot(ax=axes[0], label=name, lw=1)
    axes[0].axhline(0, c="0.3", lw=1)
    axes[0].set(title=f"{region} - mean signed error by half-hour",
                xlabel="half-hour index", ylabel="MW (positive = over-forecast)")
    axes[0].legend(fontsize=7)

    for name, g in resid_df.groupby("model"):
        axes[1].hist(g["error"], bins=70, histtype="step", label=name)
    axes[1].axvline(0, c="0.3", lw=1)
    axes[1].set(title=f"{region} - error distribution", xlabel="MW")
    axes[1].legend(fontsize=7)
    _save(fig, out_dir / f"{region}_error_profile.png")
    return f"{region}_error_profile.png"
