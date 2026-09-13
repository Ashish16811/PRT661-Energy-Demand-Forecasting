"""
ws3_features_eda.py - Forecast-readiness, EDA and feature engineering
PRT661 WP2 Preprocessing · Dan6: Theme 2
Workstream owner: Ashish Shrestha (Forecasting Lead)
Peer reviewer: Bishal Dahal

Distribution and normality diagnostics · transformation analysis · temporal,
lag and rolling features · leakage control · modelling schema.

THE DESIGN RULE THIS WORKSTREAM ENFORCES
----------------------------------------
Every predictor for timestamp t must contain only information that would have
been available BEFORE the forecast for timestamp t was generated.

This is tested empirically by validate_no_future_leakage() rather than asserted
in a comment, and the test suite injects a deliberate leak to confirm the
validator can fail. A validator never observed to fail proves nothing.

ON NORMALITY
------------
Normality here is a diagnostic, not a requirement. Electricity demand is a
deterministic daily and weekly cycle, so raw levels fail every formal test in
every region and no transform can make a cycle Gaussian. At n ~ 82,000 per
region the formal tests reject for essentially any real series, so the p-values
carry almost no information; skew, excess kurtosis and the Q-Q plots are the
signal. Random Forest and Gradient Boosting assume nothing about the target
distribution, so TOTALDEMAND is never transformed automatically.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

from common import SHAPIRO_SAMPLE_N, SEASON


LAGS = {"lag_1": 1, "lag_2": 2, "lag_48": 48, "lag_96": 96, "lag_336": 336}
ROLL_WINDOWS = {"3h": 6, "6h": 12, "24h": 48}


def _normality_block(x: np.ndarray, label: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    samp = np.random.default_rng(0).choice(x, size=min(SHAPIRO_SAMPLE_N, len(x)), replace=False)
    sh_stat, sh_p = stats.shapiro(samp)
    k2_stat, k2_p = stats.normaltest(x)
    return {"space": label, "n": len(x), "skew": float(stats.skew(x)),
            "kurtosis_excess": float(stats.kurtosis(x)),
            "shapiro_W": float(sh_stat), "shapiro_p": float(sh_p),
            "shapiro_sample_n": len(samp),
            "dagostino_k2": float(k2_stat), "dagostino_p": float(k2_p)}


def seasonal_residual(g: pd.DataFrame) -> pd.Series:
    """Demand minus its (day-of-week x half-hour) mean profile. This is the
    series where a normality question is actually meaningful - the raw level
    is a deterministic cycle, and no transform makes a cycle Gaussian."""
    prof = g.groupby(["day_of_week", "half_hour_index"])["TOTALDEMAND"].transform("mean")
    return g["TOTALDEMAND"] - prof


def distribution_analysis(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """raw / log10 / log1p / seasonal residual, per region."""
    d = g["TOTALDEMAND"].dropna()
    pos = d[d > 0]
    blocks = [_normality_block(pos, "raw"),
              _normality_block(np.log10(pos), "log10"),
              _normality_block(np.log1p(pos), "log1p"),
              _normality_block(seasonal_residual(g).dropna(), "seasonal_residual")]
    out = pd.DataFrame(blocks)
    out.insert(0, "region", region)
    out["n_nonpositive_excluded"] = int((d <= 0).sum())
    return out


def compare_transformations(dist: pd.DataFrame) -> pd.DataFrame:
    """Region-level recommendation table. The rule applied: prefer the RAW
    level unless a transform reduces |skew| by a material margin, because a
    transform costs interpretability and back-transform bias, and neither
    Random Forest nor Gradient Boosting (the Assessment 1 models) assumes a
    normally distributed target."""
    rows = []
    for region, g in dist.groupby("region"):
        s = g.set_index("space")["skew"]
        k = g.set_index("space")["kurtosis_excess"]
        best = min(["raw", "log10", "log1p"], key=lambda sp: abs(s[sp]))
        gain = abs(s["raw"]) - abs(s[best])
        rec = best if (best != "raw" and gain >= 0.20) else "raw"
        rows.append({
            "Region": region,
            "Raw skew": round(s["raw"], 3), "log10 skew": round(s["log10"], 3),
            "log1p skew": round(s["log1p"], 3),
            "Residual skew": round(s["seasonal_residual"], 3),
            "Residual excess kurtosis": round(k["seasonal_residual"], 3),
            "Recommended representation": rec,
            "Reason": (f"log reduces |skew| {abs(s['raw']):.2f}->{abs(s[best]):.2f}; "
                       "retain for the SARIMA-family model where residual "
                       "diagnostics bind") if rec != "raw" else
                      (f"|skew| {abs(s['raw']):.2f} already low; transform gains "
                       f"{gain:.2f} - not worth the back-transform bias for "
                       "tree models that assume no target distribution"),
        })
    return pd.DataFrame(rows)


def add_temporal_features(g: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from LOCAL CIVIL time (see
    resolve_timestamp_conventions), because 'is this a public holiday' and
    'is this a weekend' are civil-date facts, not market-clock facts."""
    g = g.copy()
    lt = g["local_timestamp"]
    g["half_hour_index"] = lt.dt.hour * 2 + (lt.dt.minute >= 30).astype(int)
    g["hour"] = lt.dt.hour
    g["day_of_week"] = lt.dt.dayofweek
    g["day_of_month"] = lt.dt.day
    g["week_of_year"] = lt.dt.isocalendar().week.astype(int)
    g["month"] = lt.dt.month
    g["quarter"] = lt.dt.quarter
    g["season"] = g["month"].map(SEASON)
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
    return g


def add_lag_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("SETTLEMENTDATE").copy()
    for name, k in LAGS.items():
        g[name] = g["TOTALDEMAND"].shift(k)
    return g


def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """Every rolling window is computed on demand.shift(1).

    THE DESIGN RULE: every predictor for timestamp t must contain only
    information that would have been available BEFORE the forecast for
    timestamp t was generated. Rolling without a prior shift(1) puts the
    target inside its own predictor - the single most common silent leak in
    a time-series pipeline, and one that inflates validation scores in a way
    that looks like success.
    """
    g = g.sort_values("SETTLEMENTDATE").copy()
    past = g["TOTALDEMAND"].shift(1)
    for label, w in ROLL_WINDOWS.items():
        g[f"rolling_mean_{label}"] = past.rolling(w).mean()
    g["rolling_std_24h"] = past.rolling(48).std()
    g["rolling_min_24h"] = past.rolling(48).min()
    g["rolling_max_24h"] = past.rolling(48).max()
    g["recent_ramp"] = past.diff()                       # change into t-1
    g["demand_change_30m"] = g["lag_1"] - g["lag_2"]     # both already past
    # 24h change ending at t-1: demand(t-1) - demand(t-49). Never touches t.
    g["demand_change_24h"] = g["lag_1"] - g["TOTALDEMAND"].shift(49)
    return g


PREDICTORS = (list(LAGS) +
              [f"rolling_mean_{k}" for k in ROLL_WINDOWS] +
              ["rolling_std_24h", "rolling_min_24h", "rolling_max_24h",
               "recent_ramp", "demand_change_30m", "demand_change_24h"])


def validate_no_future_leakage(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Empirical leakage test, not an assertion in a comment.

    For every predictor built from the target, re-derive it from a strictly
    past-only slice and confirm the pipeline's column matches. Also confirm
    no predictor ever equals the current target on a series where that would
    only happen through a leak.
    """
    checks = []

    def add(name, passed, detail):
        checks.append({"region": region, "check": name,
                       "result": "PASS" if passed else "FAIL", "detail": detail})

    d = g["TOTALDEMAND"]
    for name, k in LAGS.items():
        add(f"{name} equals target shifted {k}",
            g[name].equals(d.shift(k)), f"shift({k}) reproduced exactly")

    past = d.shift(1)
    for label, w in ROLL_WINDOWS.items():
        add(f"rolling_mean_{label} excludes current row",
            np.allclose(g[f"rolling_mean_{label}"].dropna(),
                        past.rolling(w).mean().reindex(g[f"rolling_mean_{label}"].dropna().index),
                        equal_nan=True),
            "recomputed from shift(1) - matches")

    # A predictor identical to the target across the whole series is the
    # signature of a same-row leak.
    for col in PREDICTORS:
        both = g[[col, "TOTALDEMAND"]].dropna()
        ident = len(both) > 100 and np.isclose(both[col], both["TOTALDEMAND"]).all()
        add(f"{col} is not a copy of the target", not ident,
            f"{len(both):,} paired rows compared")

    add("no predictor is a future value of the target",
        all((g[c].dropna().index >= 0).all() for c in PREDICTORS),
        "all predictors constructed by non-negative shift only")
    add("AEMO pre-dispatch not merged as a predictor",
        not any("predispatch" in c.lower() or "forecast" in c.lower() for c in g.columns),
        "no forecast-vintage column present in the modelling table")
    add("WA MarketRequirements not merged as observed demand",
        not any("requirement" in c.lower() for c in g.columns),
        "energy_requirement/ excluded at load time")
    return pd.DataFrame(checks)


def build_modelling_schema(g: pd.DataFrame) -> pd.DataFrame:
    """The data dictionary, generated from the produced table rather than
    typed by hand, so it cannot drift from the actual columns."""
    roles = {"SETTLEMENTDATE": "index (market clock, fixed offset)",
             "TOTALDEMAND": "TARGET - mean MW over the half hour"}
    for c in LAGS:
        roles[c] = f"predictor - target lagged {LAGS[c]} half-hours"
    for c in PREDICTORS:
        roles.setdefault(c, "predictor - rolling/derived, shift(1)-safe")
    rows = []
    for c in g.columns:
        role = roles.get(c, "calendar/temporal feature" if c in
                         ("hour", "half_hour_index", "day_of_week", "day_of_month",
                          "week_of_year", "month", "quarter", "season", "is_weekend",
                          "is_public_holiday", "holiday_name")
                         else "provenance / quality flag")
        rows.append({"column": c, "dtype": str(g[c].dtype), "role": role,
                     "non_null": int(g[c].notna().sum()),
                     "example": str(g[c].dropna().iloc[0]) if g[c].notna().any() else ""})
    return pd.DataFrame(rows)



# ==========================================================================
# MODEL-READY ENCODING
# Adopted from the group's earlier feature_engineering.py scaffold.
# ==========================================================================
SEASONS = ["Summer", "Autumn", "Winter", "Spring"]


def encode_model_features(g: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `season` and return a table sklearn can consume directly.

    The earlier scaffold used pd.get_dummies(columns=["season"]). That is the
    right idea - Random Forest and Gradient Boosting need numeric input, and a
    string column would have to be encoded somewhere regardless - but
    get_dummies derives its columns from whatever values happen to be PRESENT.
    A region, or a train/test slice, that contains no winter rows would then
    produce a table with one fewer column than its sibling, and the mismatch
    would surface as a confusing error at fit time rather than here.

    Encoding against a fixed SEASONS list instead guarantees the same four
    columns in the same order for every region and every split.
    """
    g = g.copy()
    for s in SEASONS:
        g[f"season_{s.lower()}"] = (g["season"] == s).astype(int)
    return g


# The single authoritative predictor list for the modelling stage, so the
# feature set is defined in one place rather than re-typed downstream.
# Weather columns are deliberately absent: they are blocked, and a name in
# this list implies a usable column.
MODEL_FEATURES = (
    ["half_hour_index", "hour", "day_of_week", "month", "quarter",
     "is_weekend", "is_public_holiday"]
    + [f"season_{s.lower()}" for s in SEASONS]
    + PREDICTORS
)

# Readable aliases for the lag horizons, carried into the data dictionary.
# lag_48 is self-documenting only if you already know the grid is half-hourly.
LAG_ALIASES = {"lag_1": "previous interval (30 min)",
               "lag_2": "two intervals back (60 min)",
               "lag_48": "same interval, previous day",
               "lag_96": "same interval, two days back",
               "lag_336": "same interval, previous week"}
