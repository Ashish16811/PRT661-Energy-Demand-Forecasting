"""
ws2_source_quality.py - Source quality, WA cutover and missing data
PRT661 WP2 Preprocessing · Dan6: Theme 2
Workstream owner: Sudip Lamichhane (Data Acquisition & Correction Handling Lead)
Peer reviewer: Suraj Raut

Source integrity · WA resolution regimes · gap register · imputation policy ·
invalid values and outlier classification · provenance.

CLEANING PHILOSOPHY FOR THIS WORKSTREAM
---------------------------------------
Missing values: a genuine gap (a timestamp absent from the expected grid) is
distinguished from a resolution-regime change (a source moving 30-min -> 5-min
is not "missing data"). Only gaps of at most SHORT_GAP_MAX intervals are
interpolated, and every interpolated row carries is_imputed=1. Longer gaps stay
NaN - interpolating a multi-hour outage manufactures a smooth trajectory
through a period the data does not describe.

Outliers: flagged, never silently dropped. Only physically impossible values
(demand <= 0) are marked invalid. Statistical outliers are further classified
ISOLATED (lone extreme interval - higher prior on telemetry error) vs
SUSTAINED (multi-interval run - higher prior on a real event, e.g. a heatwave).
Electricity demand contains real record peaks and real record minima; deleting
them by IQR rule would delete exactly the events a demand forecaster most needs
to learn.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

from common import SHORT_GAP_MAX, IQR_K
from ws1_data_engineering import detect_resolution, load_wa_demand


def validate_source_integrity(raw_dir: Path, regions) -> pd.DataFrame:
    """Confirm the preprocessing inputs are exactly the acquisition outputs,
    and that datasets which are NOT observed demand stay out of the demand
    path."""
    rows = []
    for r in regions:
        p = (raw_dir / "price_and_demand" /
             ("WA_demand_2022_2026.csv" if r == "WA" else f"price_and_demand_{r}.csv"))
        rows.append({"artefact": p.name, "role": "observed demand (in scope)",
                     "present": p.exists(),
                     "bytes": p.stat().st_size if p.exists() else 0})
    for sub, role in [("energy_requirement", "WA market REQUIREMENT forecast - NOT observed demand"),
                      ("predispatch", "AEMO forecast vintages - NOT a predictor here"),
                      ("dispatch", "current-moment snapshot - not a historical series")]:
        d = raw_dir / sub
        n = len(list(d.glob("*.csv"))) if d.exists() else 0
        rows.append({"artefact": f"{sub}/ ({n} files)", "role": f"EXCLUDED: {role}",
                     "present": d.exists(), "bytes": 0})
    return pd.DataFrame(rows)


def detect_wa_resolution_regimes(raw_dir: Path) -> pd.DataFrame:
    """WA's published SOURCE_INTERVAL_MIN, cross-checked against the observed
    timestamp spacing rather than believed on its own."""
    wa = load_wa_demand(raw_dir)
    rows = []
    for res, g in wa.groupby("SOURCE_INTERVAL_MIN"):
        observed = g["SETTLEMENTDATE"].diff().dt.total_seconds().div(60).mode()
        rows.append({"declared_interval_min": int(res), "rows": len(g),
                     "start": g["SETTLEMENTDATE"].min(), "end": g["SETTLEMENTDATE"].max(),
                     "observed_modal_spacing_min": float(observed.iat[0]) if len(observed) else np.nan,
                     "sources": ", ".join(sorted(g["SOURCE"].unique())),
                     "agrees": bool(len(observed) and observed.iat[0] == res)})
    return pd.DataFrame(rows)


def reindex_and_flag_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Rebuild each resolution regime on its own complete grid so absent rows
    become explicit NaN, then interpolate ONLY runs of at most SHORT_GAP_MAX.
    A 30-minute source is never treated as five missing 5-minute readings."""
    g = g.sort_values("SETTLEMENTDATE").copy()
    expected = detect_resolution(g).bfill().ffill()
    month = g["SETTLEMENTDATE"].dt.to_period("M")
    pieces = []
    for m, res in expected.groupby(month).first().items():
        seg = g[month == m]
        if seg.empty or pd.isna(res):
            continue
        full = pd.date_range(seg["SETTLEMENTDATE"].min(),
                             seg["SETTLEMENTDATE"].max(), freq=f"{int(res)}min")
        seg = seg.set_index("SETTLEMENTDATE").reindex(full)
        seg.index.name = "SETTLEMENTDATE"
        seg["SOURCE_INTERVAL_MIN"] = seg["SOURCE_INTERVAL_MIN"].fillna(res)
        pieces.append(seg.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    out["REGION"] = out["REGION"].ffill().bfill()
    out["SOURCE"] = out["SOURCE"].ffill()

    out["is_imputed"] = 0
    na = out["TOTALDEMAND"].isna()
    run_len = na.groupby((na != na.shift()).cumsum()).transform("size")
    short = na & (run_len <= SHORT_GAP_MAX)
    out.loc[short, "TOTALDEMAND"] = out["TOTALDEMAND"].interpolate(limit=SHORT_GAP_MAX)[short]
    out.loc[short, "is_imputed"] = 1
    return out


def build_gap_register(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gap register: region, start, end, length, resolution, treatment."""
    rows = []
    for region, g in df.groupby("REGION"):
        g = g.sort_values("SETTLEMENTDATE").reset_index(drop=True)
        na = g["TOTALDEMAND"].isna() | (g["is_imputed"] == 1)
        if not na.any():
            continue
        run = (na != na.shift()).cumsum()
        for _, blk in g[na].groupby(run[na]):
            n = len(blk)
            imputed = bool(blk["is_imputed"].max())
            rows.append({
                "region": region, "start": blk["SETTLEMENTDATE"].min(),
                "end": blk["SETTLEMENTDATE"].max(), "length_intervals": n,
                "resolution_min": float(blk["SOURCE_INTERVAL_MIN"].iloc[0]),
                "treatment": "linear interpolation" if imputed else "left NaN, flagged",
                "is_imputed": int(imputed),
                "reason": ("short isolated gap <= %d intervals" % SHORT_GAP_MAX) if imputed
                          else "gap exceeds short-gap threshold; not fabricated",
            })
    return pd.DataFrame(rows, columns=["region", "start", "end", "length_intervals",
                                       "resolution_min", "treatment", "is_imputed", "reason"])


def flag_invalid_and_outliers(g: pd.DataFrame) -> pd.DataFrame:
    """Nothing is deleted. Physically impossible values are separated from
    statistical extremes, and statistical extremes are separated into
    isolated (likely telemetry) and sustained (likely genuine event)."""
    g = g.copy()
    g["is_invalid"] = (g["TOTALDEMAND"] <= 0).fillna(False)

    valid = g.loc[~g["is_invalid"], "TOTALDEMAND"]
    q1, q3 = valid.quantile([0.25, 0.75])
    lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
    g["is_stat_outlier"] = (~g["is_invalid"]) & ((g["TOTALDEMAND"] < lo) | (g["TOTALDEMAND"] > hi))
    g["is_stat_outlier"] = g["is_stat_outlier"].fillna(False)

    # Robust z-score on the median/MAD - resistant to the very outliers being
    # measured, unlike a mean/SD z-score.
    med = valid.median()
    mad = (valid - med).abs().median()
    g["robust_z"] = (g["TOTALDEMAND"] - med) / (1.4826 * mad) if mad else np.nan

    flagged = g["is_stat_outlier"] | g["is_invalid"]
    run_len = flagged.groupby((flagged != flagged.shift()).cumsum()).transform("size")
    g["outlier_type"] = np.where(~flagged, "",
                        np.where(g["is_invalid"], "invalid_impossible",
                        np.where(run_len <= 1, "isolated_extreme", "sustained_extreme_event")))
    g.attrs["iqr_bounds"] = (lo, hi)
    return g

