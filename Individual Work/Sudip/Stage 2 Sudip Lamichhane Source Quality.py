"""
PRT661 – Data Science Practice
WP2 Individual Contribution Module – Sudip Lamichhane (S388085)
Role: Data Acquisition & Correction Handling Lead
Project: Australian Electricity Demand Forecasting | Dan6

This workstream protects source integrity during cleaning: duplicates, missing
intervals, short-gap treatment, anomaly flags, public-holiday context and AEMO
benchmark separation. It was merged with the other three workstreams in the
final integrated Stage 2 pipeline.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("wp2_sudip")
TIME = "SETTLEMENTDATE"
TARGET = "TOTALDEMAND"
TARGET_FREQ_MIN = 30
SHORT_GAP_MAX = 2
IQR_K = 3.0
REGION_STATE = {"NSW1":"NSW","QLD1":"QLD","VIC1":"VIC","SA1":"SA","TAS1":"TAS","WA":"WA"}
NEM_REGIONS = ["NSW1","QLD1","VIC1","SA1","TAS1"]
REGION_INTERVAL_LABEL = {r:"ending" for r in NEM_REGIONS} | {"WA":"beginning"}


def detect_resolution(g: pd.DataFrame) -> pd.Series:
    """Shared engineering contract: infer each month's native spacing."""
    diffs = g[TIME].diff().dt.total_seconds().div(60)
    month = g[TIME].dt.to_period("M")
    modal = diffs.groupby(month).agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
    return month.map(modal)


def report_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"region": r, "rows": len(g),
                          "duplicate_timestamps": int(g[TIME].duplicated(keep=False).sum())}
                         for r, g in df.groupby("REGION")])


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["REGION", TIME], keep="last")


def find_missing_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Measure completeness against the regime that actually existed that month."""
    rows = []
    for region, g in df.groupby("REGION"):
        g = g.sort_values(TIME).reset_index(drop=True)
        expected = detect_resolution(g)
        diff = g[TIME].diff().dt.total_seconds().div(60)
        cand_curr = ((diff / expected).round() - 1).clip(lower=0)
        cand_prev = ((diff / expected.shift(1)).round() - 1).clip(lower=0)
        missing = np.minimum(cand_curr, cand_prev).fillna(0)
        rows.append({"region": region, "rows": len(g),
                     "resolution_regimes_min": sorted(expected.dropna().unique().tolist()),
                     "missing_intervals": int(missing.sum()),
                     "completeness_pct": round(100 * len(g) / (len(g)+missing.sum()), 4)})
    return pd.DataFrame(rows)


def reindex_and_flag_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Fill only very short gaps; long gaps remain visible NaN evidence."""
    g = g.sort_values(TIME).copy()
    if "SOURCE_INTERVAL_MIN" not in g:
        g["SOURCE_INTERVAL_MIN"] = np.nan
    expected = detect_resolution(g).bfill().ffill()
    month = g[TIME].dt.to_period("M")
    pieces = []
    for m, res in expected.groupby(month).first().items():
        seg = g[month == m]
        if seg.empty or pd.isna(res):
            continue
        full = pd.date_range(seg[TIME].min(), seg[TIME].max(), freq=f"{int(res)}min")
        seg = seg.set_index(TIME).reindex(full)
        seg.index.name = TIME
        seg["SOURCE_INTERVAL_MIN"] = seg["SOURCE_INTERVAL_MIN"].fillna(res)
        pieces.append(seg.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    out["REGION"] = out["REGION"].ffill().bfill()
    out["is_imputed"] = 0
    na = out[TARGET].isna()
    run_len = na.groupby((na != na.shift()).cumsum()).transform("size")
    short = na & (run_len <= SHORT_GAP_MAX)
    out.loc[short, TARGET] = out[TARGET].interpolate(limit=SHORT_GAP_MAX)[short]
    out.loc[short, "is_imputed"] = 1
    return out


def flag_invalid_and_outliers(g: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious values; do not silently delete genuine demand extremes."""
    g = g.copy()
    g["is_invalid"] = (g[TARGET] <= 0).fillna(False).astype(int)
    valid = g.loc[g["is_invalid"] == 0, TARGET]
    q1, q3 = valid.quantile([0.25, 0.75])
    lo, hi = q1 - IQR_K*(q3-q1), q3 + IQR_K*(q3-q1)
    g["is_stat_outlier"] = ((g["is_invalid"] == 0) &
                            ((g[TARGET] < lo) | (g[TARGET] > hi))).fillna(False).astype(int)
    med = valid.median(); mad = (valid-med).abs().median()
    g["robust_z"] = (g[TARGET]-med)/(1.4826*mad) if mad else np.nan
    flagged = (g["is_stat_outlier"] == 1) | (g["is_invalid"] == 1)
    run_len = flagged.groupby((flagged != flagged.shift()).cumsum()).transform("size")
    g["outlier_type"] = np.where(~flagged, "",
                         np.where(g["is_invalid"] == 1, "invalid_impossible",
                         np.where(run_len <= 1, "isolated_extreme", "sustained_extreme_event")))
    return g


def load_holidays(raw_dir: Path, region: str) -> pd.DataFrame:
    p = raw_dir / "public_holidays" / f"public_holidays_{region}_{REGION_STATE[region]}.csv"
    h = pd.read_csv(p, parse_dates=["date"])
    return h[["date", "is_public_holiday", "holiday_name"]]


def load_holidays_covering(raw_dir: Path, region: str, needed_through):
    """Extend the acquired holiday calendar when the forecast horizon goes further."""
    h = load_holidays(raw_dir, region)
    h["provenance"] = "acquired"
    acquired_to = h["date"].max(); needed_through = pd.Timestamp(needed_through)
    if acquired_to >= needed_through:
        return h, 0
    try:
        import holidays as _hol
    except ImportError:
        log.error("[%s] holiday calendar ends %s and holidays package is missing", region, acquired_to.date())
        return h, 0
    years = sorted({acquired_to.year, needed_through.year})
    cal = _hol.Australia(subdiv=REGION_STATE[region], years=years)
    extra_dates = pd.date_range(acquired_to + pd.Timedelta(days=1), needed_through, freq="D")
    extra = pd.DataFrame({"date": extra_dates})
    extra["is_public_holiday"] = extra["date"].dt.date.isin(cal).astype(int)
    extra["holiday_name"] = extra["date"].dt.date.map(lambda d: cal.get(d, ""))
    extra["provenance"] = "generated (holidays library)"
    return pd.concat([h, extra], ignore_index=True), int(extra["is_public_holiday"].sum())


def add_holiday_features(g, raw_dir, region, holidays_table=None):
    """Attach public-holiday and adjacent-day effects using local civil date."""
    h = holidays_table if holidays_table is not None else load_holidays(raw_dir, region)
    h = h[["date","is_public_holiday","holiday_name"]].rename(columns={"date":"local_date"})
    g = g.merge(h, on="local_date", how="left")
    g["is_public_holiday"] = g["is_public_holiday"].fillna(0).astype(int)
    g["holiday_name"] = g["holiday_name"].fillna("")
    dates = set(pd.DatetimeIndex(h.loc[h["is_public_holiday"].astype(bool), "local_date"]))
    day = pd.Timedelta(days=1)
    g["is_day_before_holiday"] = g["local_date"].add(day).isin(dates).astype(int)
    g["is_day_after_holiday"] = g["local_date"].sub(day).isin(dates).astype(int)
    return g


def process_predispatch(raw_dir, region, hh):
    """Keep AEMO dashboard ACTUAL/FORECAST as a secondary benchmark, never a predictor."""
    path = raw_dir / "predispatch" / f"predispatch_{region}.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path, parse_dates=[TIME])
    if "PERIODTYPE" not in df.columns:
        return None, None, None
    kind = df["PERIODTYPE"].str.upper()
    ending = REGION_INTERVAL_LABEL[region] == "ending"
    closed = label = "right" if ending else "left"
    def to_half_hourly(sub, column):
        if sub.empty:
            return pd.DataFrame(columns=[TIME,"region",column])
        s = sub.set_index(TIME)[TARGET].resample(
            f"{TARGET_FREQ_MIN}min", closed=closed, label=label).mean().dropna()
        return pd.DataFrame({TIME:s.index,"region":region,column:s.values})
    actual = to_half_hourly(df[kind == "ACTUAL"], "aemo_actual_MW")
    forecast = to_half_hourly(df[kind == "FORECAST"], "aemo_forecast_MW")
    if len(forecast):
        origin = forecast[TIME].min()
        forecast["forecast_origin"] = origin
        forecast["forecast_horizon_step"] = range(1, len(forecast)+1)
        forecast["forecast_horizon_minutes"] = ((forecast[TIME]-origin).dt.total_seconds()/60).astype(int)
        forecast["source_vintage"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M")
    consistency = None
    if len(actual):
        joined = actual.merge(hh[[TIME,TARGET]], on=TIME, how="inner").dropna()
        if len(joined):
            diff = joined[TARGET] - joined["aemo_actual_MW"]
            consistency = pd.DataFrame([{"region":region,"matched_intervals":len(joined),
                "window_start":joined[TIME].min(),"window_end":joined[TIME].max(),
                "MAE_between_sources_MW":diff.abs().mean(),"bias_MW":diff.mean(),
                "correlation":joined[TARGET].corr(joined["aemo_actual_MW"]),
                "interpretation":"source consistency check - NOT forecast accuracy"}])
    return actual, forecast, consistency


if __name__ == "__main__":
    print("WP2 Sudip Lamichhane module loaded: source quality, cleaning and benchmark separation.")
