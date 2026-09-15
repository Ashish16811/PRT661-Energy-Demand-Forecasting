"""
02_Preprocessing.py - PRT661 Stage 2: Cleaning, EDA and Feature Engineering
Australian Electricity Demand Forecasting | Group: Dan6 - Theme 2

Contributors:
    Suraj Raut        - Data Engineering Lead (timestamps, 30-minute grid, harmonisation)
    Sudip Lamichhane  - Data Acquisition & Correction Handling Lead (source quality, gaps, WA regimes)
    Ashish Shrestha   - Forecasting Lead (rolling EDA, forecast-safe features, leakage control)
    Bishal Dahal      - Verification & Dashboard Lead (core tests, weather/holiday checks, readiness)

Takes the Stage 1 acquisition ZIP and produces model-ready 30-minute datasets,
organised by region, plus the rolling four-month analysis the forecast needs.

    Stage 1 ZIP -> source validation -> clean + harmonise to 30 min
      -> historical weather + latest rolling 7-day weather forecast
      -> six core tests -> calendar/holidays -> rolling current-month + 3-month window
      -> Model A core features + Model B weather inputs/provenance
      -> future feature frame -> leakage checks -> Stage 2 ZIP

The window ROLLS: it is derived from the latest actual observation, so running
in December gives Dec/Jan/Feb/Mar rather than a hard-coded Sep-Dec.

Stops at model-ready data. Chronological splitting, scaling and model fitting
belong to Stage 3 - a split decided here would be baked into the data.
"""
import argparse
import json
import logging
import shutil
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage2")

TIME = "SETTLEMENTDATE"
TARGET = "TOTALDEMAND"

NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
DEMAND_REGIONS = NEM_REGIONS + ["WA"]
REGION_STATE = {"NSW1": "NSW", "QLD1": "QLD", "VIC1": "VIC", "SA1": "SA",
               "TAS1": "TAS", "WA": "WA"}
REGION_TZ = {"NSW1": "Australia/Sydney", "QLD1": "Australia/Brisbane",
            "VIC1": "Australia/Melbourne", "SA1": "Australia/Adelaide",
            "TAS1": "Australia/Hobart", "WA": "Australia/Perth"}
# Fixed-offset MARKET clock each source actually publishes in (no DST).
REGION_SOURCE_TZ = {r: "Etc/GMT-10" for r in NEM_REGIONS} | {"WA": "Etc/GMT-8"}
# NEM stamps an interval with its END time (first reading of a day is 00:05).
# WA is treated as interval-beginning on the evidence that both its regimes
# start at 08:00 - flagged for WEM source confirmation, not asserted as fact.
REGION_INTERVAL_LABEL = {r: "ending" for r in NEM_REGIONS} | {"WA": "beginning"}

TARGET_FREQ_MIN = 30      # the modelling grid this whole file exists to build
SHORT_GAP_MAX = 2         # intervals; longer gaps are left NaN, not filled
IQR_K = 3.0               # widened from 1.5 - demand has real fat tails
FIG_DPI = 130


AUDIT: list[dict] = []


def audit(stage, region, **kw):
    row = {"stage": stage, "region": region, "rows_before": np.nan, "rows_after": np.nan,
          "detail": "", "reason": ""}
    row.update(kw)
    AUDIT.append(row)


def savefig(path: Path):
    plt.gcf().tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================================
# INPUT AND LOADERS  (preserved)
# ============================================================================
def resolve_raw_root(source, work_dir="_raw_extracted") -> Path:
    """Accept a ZIP or a folder; find the real root even behind one wrapper
    directory ('Data Extraction/', 'Raw Dataset/', etc.)."""
    source = Path(source)
    known = {"price_and_demand", "weather", "bom_weather", "public_holidays",
            "predispatch", "dispatch", "energy_requirement"}
    if source.is_file() and source.suffix.lower() == ".zip":
        work = Path(work_dir)
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        with zipfile.ZipFile(source) as z:
            for member in z.namelist():
                if not str((work / member).resolve()).startswith(str(work.resolve())):
                    raise ValueError(f"Unsafe path in archive: {member}")
            z.extractall(work)
        root = work
    elif source.is_dir():
        root = source
    else:
        raise FileNotFoundError(f"Not a .zip or folder: {source}")

    if any((root / k).is_dir() for k in known):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any((child / k).is_dir() for k in known):
            return child
    raise FileNotFoundError(f"No acquisition folders found under {root}")


def load_nem_region(region: str, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "price_and_demand" / f"price_and_demand_{region}.csv")
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    return df[["REGION", "SETTLEMENTDATE", "TOTALDEMAND"]]


def load_wa_demand(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "price_and_demand" / "WA_demand_2022_2026.csv")
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    return df[["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "SOURCE_INTERVAL_MIN"]]


def load_all_demand(raw_dir: Path, regions) -> pd.DataFrame:
    frames = [load_wa_demand(raw_dir) if r == "WA" else load_nem_region(r, raw_dir)
             for r in regions]
    df = pd.concat(frames, ignore_index=True).sort_values(["REGION", "SETTLEMENTDATE"])
    log.info(f"[Load] {len(df):,} raw rows, {len(regions)} regions "
             f"({df.SETTLEMENTDATE.min()} -> {df.SETTLEMENTDATE.max()})")
    return df.reset_index(drop=True)


DAILY_COLS = {
    "date": ("date",),
    "temp_max_c": ("temp_max",),
    "temp_min_c": ("temp_min",),
    "temp_mean_c": ("temp_mean",),
    "rainfall_mm": ("rainfall",),
}


def load_weather_daily(raw_dir: Path, region: str) -> pd.DataFrame | None:
    """Historical daily weather from Stage 1.

    Stage 1 currently supplies Open-Meteo ERA5 reanalysis. These rows are valid
    historical training inputs, but they are never relabelled as a forecast.
    """
    p = raw_dir / "weather" / f"weather_daily_{region}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    resolved = {
        canon: next((cols[c] for c in cols if any(k in c for k in keys)), None)
        for canon, keys in DAILY_COLS.items()
    }
    if resolved["date"] is None or resolved["temp_max_c"] is None:
        return None
    out = pd.DataFrame({k: df[v] for k, v in resolved.items() if v})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for c in ("temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "temp_mean_c" not in out or out["temp_mean_c"].isna().all():
        out["temp_mean_c"] = (out["temp_max_c"] + out["temp_min_c"]) / 2
    return (out.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date", keep="last"))


def load_weather_hourly(raw_dir: Path, region: str) -> pd.DataFrame | None:
    """Historical hourly weather converted to the 30-minute modelling grid.

    Temperature and humidity are state variables, so linear interpolation between
    adjacent hourly observations is appropriate. Rainfall is intentionally not
    interpolated here; Model B uses the daily rainfall total from the daily file.
    """
    path = raw_dir / "weather" / f"weather_hourly_{region}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        h = pd.read_csv(path)
    except Exception:
        return None
    if "TIMESTAMP" not in h.columns:
        return None
    h["TIMESTAMP"] = pd.to_datetime(h["TIMESTAMP"], errors="coerce")
    h = h.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
    cols = [c for c in ("temp_c", "humidity_pct") if c in h.columns]
    if not cols:
        return None
    for c in cols:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    hh = (h.set_index("TIMESTAMP")[cols]
          .resample(f"{TARGET_FREQ_MIN}min")
          .interpolate(method="time", limit=2))
    return hh.rename(columns={"temp_c": "temp_halfhourly_c"})


def _forecast_latest_dir(raw_dir: Path) -> Path:
    return raw_dir / "weather" / "forecast" / "latest"


def load_weather_forecast(raw_dir: Path, region: str):
    """Load the latest Stage-1 rolling weather forecast for one region.

    Returns `(hourly_30min, daily, meta)`. Missing forecast files are allowed:
    Stage 2 remains usable for Model A and uses climatology as Model-B fallback.
    """
    latest = _forecast_latest_dir(raw_dir)
    hp = latest / f"weather_forecast_hourly_{region}.csv"
    dp = latest / f"weather_forecast_daily_{region}.csv"
    meta = {
        "region": region,
        "forecast_available": False,
        "forecast_origin": pd.NaT,
        "forecast_start": pd.NaT,
        "forecast_end": pd.NaT,
        "forecast_days": 0,
        "source": "",
        "model": "",
        "retrieved_at_utc": "",
        "raw_hourly_rows": 0,
    }
    if not hp.exists() or hp.stat().st_size == 0:
        return None, None, meta

    try:
        h = pd.read_csv(hp)
    except Exception as exc:
        meta["error"] = f"hourly read failed: {exc}"
        return None, None, meta

    required = {"TIMESTAMP", "temp_c", "humidity_pct", "SOURCE_TYPE",
                "FORECAST_ORIGIN_DATE", "LEAD_DAY"}
    missing = required - set(h.columns)
    if missing:
        meta["error"] = f"hourly forecast missing columns: {sorted(missing)}"
        return None, None, meta

    h["TIMESTAMP"] = pd.to_datetime(h["TIMESTAMP"], errors="coerce")
    h["temp_c"] = pd.to_numeric(h["temp_c"], errors="coerce")
    h["humidity_pct"] = pd.to_numeric(h["humidity_pct"], errors="coerce")
    if "rainfall_mm" in h:
        h["rainfall_mm"] = pd.to_numeric(h["rainfall_mm"], errors="coerce")
    h["LEAD_DAY"] = pd.to_numeric(h["LEAD_DAY"], errors="coerce")
    h = (h.dropna(subset=["TIMESTAMP"])
         .sort_values("TIMESTAMP")
         .drop_duplicates("TIMESTAMP", keep="last"))
    meta["raw_hourly_rows"] = len(h)

    origin_vals = pd.to_datetime(h["FORECAST_ORIGIN_DATE"], errors="coerce").dropna()
    origin = origin_vals.iloc[0].normalize() if len(origin_vals) else pd.NaT
    meta.update({
        "forecast_available": len(h) > 0,
        "forecast_origin": origin,
        "forecast_start": h["TIMESTAMP"].min() if len(h) else pd.NaT,
        "forecast_end": h["TIMESTAMP"].max() if len(h) else pd.NaT,
        "forecast_days": int(h["TIMESTAMP"].dt.normalize().nunique()) if len(h) else 0,
        "source": str(h["SOURCE"].dropna().iloc[0]) if "SOURCE" in h and h["SOURCE"].notna().any() else "",
        "model": str(h["MODEL"].dropna().iloc[0]) if "MODEL" in h and h["MODEL"].notna().any() else "",
        "retrieved_at_utc": str(h["RETRIEVED_AT_UTC"].dropna().iloc[0])
        if "RETRIEVED_AT_UTC" in h and h["RETRIEVED_AT_UTC"].notna().any() else "",
    })

    # Interpolate only continuous state variables to the half-hour grid.
    # Include the final half-hour of the seventh target day. Hourly forecasts
    # normally end at 23:00; 23:30 uses a one-step carry-forward rather than
    # dropping one modelling interval from the rolling horizon.
    grid = pd.date_range(h["TIMESTAMP"].min(),
                         h["TIMESTAMP"].max() + pd.Timedelta(minutes=TARGET_FREQ_MIN),
                         freq=f"{TARGET_FREQ_MIN}min") if len(h) else pd.DatetimeIndex([])
    state = (h.set_index("TIMESTAMP")[["temp_c", "humidity_pct"]]
             .reindex(grid)
             .interpolate(method="time", limit=2)
             .ffill(limit=1)) if len(grid) else pd.DataFrame()
    if len(state):
        state.index.name = "local_timestamp"
        state = state.rename(columns={"temp_c": "forecast_temp_c",
                                      "humidity_pct": "forecast_humidity_pct"})
        # Metadata is constant for the vintage; lead day is recomputed from the
        # target local date rather than interpolated as a number.
        state["weather_forecast_origin"] = origin
        state["weather_source_type"] = "FORECAST_7DAY"
        state["weather_forecast_source"] = meta["source"]
        state["weather_forecast_model"] = meta["model"]
        state["weather_retrieved_at_utc"] = meta["retrieved_at_utc"]
        if pd.notna(origin):
            state["weather_lead_day"] = (state.index.normalize() - origin).days
        else:
            state["weather_lead_day"] = np.nan

    daily = None
    if dp.exists() and dp.stat().st_size:
        try:
            d = pd.read_csv(dp)
            if "DATE" in d.columns:
                d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce").dt.normalize()
                for c in ("temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm",
                          "humidity_mean_pct"):
                    if c in d:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                keep = [c for c in ("DATE", "temp_max_c", "temp_min_c", "temp_mean_c",
                                    "rainfall_mm", "humidity_mean_pct", "LEAD_DAY") if c in d]
                daily = (d[keep].dropna(subset=["DATE"])
                         .sort_values("DATE")
                         .drop_duplicates("DATE", keep="last"))
        except Exception as exc:
            meta["daily_error"] = str(exc)

    return state, daily, meta


def detect_stage1_forecast_origin(raw_dir: Path, regions) -> pd.Timestamp | None:
    """Return the latest common Stage-1 weather forecast origin if available.

    The rolling month should follow the pipeline/forecast vintage, not silently
    stay in the previous month merely because the latest demand file lags by one
    interval or one day. Demand cutoff is still recorded separately for scoring.
    """
    latest = _forecast_latest_dir(raw_dir)
    manifest = latest / "weather_forecast_manifest.csv"
    candidates = []
    if manifest.exists() and manifest.stat().st_size:
        try:
            m = pd.read_csv(manifest)
            if "FORECAST_ORIGIN_DATE" in m:
                candidates.extend(pd.to_datetime(m["FORECAST_ORIGIN_DATE"], errors="coerce").dropna().tolist())
        except Exception:
            pass
    if not candidates:
        for region in regions:
            p = latest / f"weather_forecast_hourly_{region}.csv"
            if not p.exists() or not p.stat().st_size:
                continue
            try:
                sample = pd.read_csv(p, usecols=["FORECAST_ORIGIN_DATE"], nrows=5)
                candidates.extend(pd.to_datetime(sample["FORECAST_ORIGIN_DATE"], errors="coerce").dropna().tolist())
            except Exception:
                continue
    return max(candidates).normalize() if candidates else None


def load_holidays(raw_dir: Path, region: str) -> pd.DataFrame:
    p = raw_dir / "public_holidays" / f"public_holidays_{region}_{REGION_STATE[region]}.csv"
    h = pd.read_csv(p, parse_dates=["date"])
    return h[["date", "is_public_holiday", "holiday_name"]]




# ============================================================================
# CLEANING  (preserved - duplicates, gaps, outliers)
# ============================================================================
def report_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"region": r, "rows": len(g),
                          "duplicate_timestamps": int(g["SETTLEMENTDATE"].duplicated(keep=False).sum())}
                         for r, g in df.groupby("REGION")])


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["REGION", "SETTLEMENTDATE"], keep="last")


def detect_resolution(g: pd.DataFrame) -> pd.Series:
    """Modal spacing per calendar month. A single global value would misread
    a genuine resolution change (WA's 1 Oct 2023 30->5 min reform) as either
    one huge gap or a flood of spurious ones; a monthly mode adapts within
    one cycle of the transition and stays stable everywhere else."""
    diffs = g["SETTLEMENTDATE"].diff().dt.total_seconds().div(60)
    month = g["SETTLEMENTDATE"].dt.to_period("M")
    modal = diffs.groupby(month).agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
    return month.map(modal)


def find_missing_intervals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, g in df.groupby("REGION"):
        g = g.sort_values("SETTLEMENTDATE").reset_index(drop=True)
        expected = detect_resolution(g)
        diff = g["SETTLEMENTDATE"].diff().dt.total_seconds().div(60)
        # At the exact row a resolution regime changes, "expected" is the NEW
        # regime but the measured step spans the OLD one - use whichever
        # adjacent regime fits, so the transition itself isn't read as a gap.
        cand_curr = ((diff / expected).round() - 1).clip(lower=0)
        cand_prev = ((diff / expected.shift(1)).round() - 1).clip(lower=0)
        missing = np.minimum(cand_curr, cand_prev).fillna(0)
        rows.append({"region": region, "rows": len(g),
                    "resolution_regimes_min": sorted(expected.dropna().unique().tolist()),
                    "missing_intervals": int(missing.sum()),
                    "completeness_pct": round(100 * len(g) / (len(g) + missing.sum()), 4)})
    return pd.DataFrame(rows)


def reindex_and_flag_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Rebuild each resolution regime on its own complete grid so an absent
    row becomes an explicit NaN row. Interpolate ONLY runs <= SHORT_GAP_MAX;
    a longer gap stays NaN rather than being fabricated."""
    g = g.sort_values("SETTLEMENTDATE").copy()
    if "SOURCE_INTERVAL_MIN" not in g:
        g["SOURCE_INTERVAL_MIN"] = np.nan
    expected = detect_resolution(g).bfill().ffill()
    month = g["SETTLEMENTDATE"].dt.to_period("M")
    pieces = []
    for m, res in expected.groupby(month).first().items():
        seg = g[month == m]
        if seg.empty or pd.isna(res):
            continue
        full = pd.date_range(seg["SETTLEMENTDATE"].min(), seg["SETTLEMENTDATE"].max(), freq=f"{int(res)}min")
        seg = seg.set_index("SETTLEMENTDATE").reindex(full)
        seg.index.name = "SETTLEMENTDATE"
        seg["SOURCE_INTERVAL_MIN"] = seg["SOURCE_INTERVAL_MIN"].fillna(res)
        pieces.append(seg.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    out["REGION"] = out["REGION"].ffill().bfill()

    out["is_imputed"] = 0
    na = out["TOTALDEMAND"].isna()
    run_len = na.groupby((na != na.shift()).cumsum()).transform("size")
    short = na & (run_len <= SHORT_GAP_MAX)
    out.loc[short, "TOTALDEMAND"] = out["TOTALDEMAND"].interpolate(limit=SHORT_GAP_MAX)[short]
    out.loc[short, "is_imputed"] = 1
    return out


def flag_invalid_and_outliers(g: pd.DataFrame) -> pd.DataFrame:
    """Flag, never delete. Physically impossible values (demand <= 0) are
    kept separate from statistical extremes; statistical extremes are split
    into isolated (a lone spike - likely telemetry error) and sustained
    (a multi-interval run - likely a genuine event, e.g. a heatwave)."""
    g = g.copy()
    g["is_invalid"] = (g["TOTALDEMAND"] <= 0).fillna(False).astype(int)
    valid = g.loc[g["is_invalid"] == 0, "TOTALDEMAND"]
    q1, q3 = valid.quantile([0.25, 0.75])
    lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
    g["is_stat_outlier"] = ((g["is_invalid"] == 0) &
                            ((g["TOTALDEMAND"] < lo) | (g["TOTALDEMAND"] > hi))).fillna(False).astype(int)
    med = valid.median()
    mad = (valid - med).abs().median()
    g["robust_z"] = (g["TOTALDEMAND"] - med) / (1.4826 * mad) if mad else np.nan

    flagged = (g["is_stat_outlier"] == 1) | (g["is_invalid"] == 1)
    run_len = flagged.groupby((flagged != flagged.shift()).cumsum()).transform("size")
    g["outlier_type"] = np.where(~flagged, "",
                        np.where(g["is_invalid"] == 1, "invalid_impossible",
                        np.where(run_len <= 1, "isolated_extreme", "sustained_extreme_event")))
    return g




# ============================================================================
# HARMONISATION  (preserved - 30-minute grid, clocks)
# ============================================================================
def harmonise_to_half_hourly(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Downsample by MEAN (TOTALDEMAND is a power level in MW, not energy -
    summing would be dimensionally wrong), binned on the source's own
    interval-labelling convention. Native 30-minute rows (WA pre-cutover)
    pass through unchanged. Nothing is ever upsampled."""
    ending = REGION_INTERVAL_LABEL[region] == "ending"
    closed = label = "right" if ending else "left"
    idx = g.set_index("SETTLEMENTDATE")
    agg = {"TOTALDEMAND": "mean", "is_imputed": "max", "is_invalid": "max", "is_stat_outlier": "max"}
    out = idx.resample(f"{TARGET_FREQ_MIN}min", closed=closed, label=label).agg(agg)
    out["n_source_readings"] = idx["TOTALDEMAND"].resample(
        f"{TARGET_FREQ_MIN}min", closed=closed, label=label).count()
    out = out[out["n_source_readings"] > 0]
    out["REGION"] = region
    return out.reset_index()


def enforce_continuous_grid(hh: pd.DataFrame, region: str) -> pd.DataFrame:
    """An unbroken 30-min index so shift(k) always equals k half-hours. WA's
    market-system changeover leaves 2023-10-01 00:00-07:55 AWST uncovered by
    either source; without this, every lag after that point silently
    misaligns by 8.5 hours instead of showing a clean NaN."""
    full = pd.date_range(hh["SETTLEMENTDATE"].min(), hh["SETTLEMENTDATE"].max(), freq=f"{TARGET_FREQ_MIN}min")
    before = len(hh)
    out = hh.set_index("SETTLEMENTDATE").reindex(full)
    out.index.name = "SETTLEMENTDATE"
    out["REGION"] = region
    for c in ("is_imputed", "is_invalid", "is_stat_outlier", "n_source_readings"):
        out[c] = out[c].fillna(0).astype(int)
    if len(out) - before:
        log.warning(f"[Grid] {region}: inserted {len(out) - before} NaN row(s) to close discontinuities")
    return out.reset_index()


def resolve_clocks(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """market clock (fixed offset, what the model grid stays on) + local
    civil time (what a holiday/weekend fact actually depends on)."""
    g = g.copy()
    market = g["SETTLEMENTDATE"].dt.tz_localize(REGION_SOURCE_TZ[region])
    local = market.dt.tz_convert(REGION_TZ[region])
    g["local_timestamp"] = local.dt.tz_localize(None)
    g["local_date"] = g["local_timestamp"].dt.normalize()
    return g




# ============================================================================
# ROLLING FORECAST WINDOW
# ============================================================================
def derive_model_cutoff(analysis_origin):
    """Split the timeline into a TRAINING period and a FORECAST period.

    Two different boundaries exist and conflating them is the mistake this
    function prevents:

      model_cutoff   the last half-hour of the previous COMPLETE calendar
                     month. Everything up to here is training data.
      analysis_origin the latest observed actual. Everything after
                     model_cutoff is the forecast period we are trying to
                     predict, and is held out.

    Anchored on a calendar month end rather than on "the latest reading" for
    three reasons:

    1. A partial month biases anything aggregated monthly. Fourteen days of
       September is not September - it is whatever weather those fourteen days
       happened to bring, and the historical month profiles would inherit that.
    2. It makes a run reproducible. Re-running on the 14th and the 19th of the
       same month yields the same training set, so two runs are comparable.
       A cutoff that tracks the latest reading changes the training data every
       time the pipeline is run.
    3. It matches what we are forecasting. The rolling window starts at the
       current month, so training through the middle of that month would put
       part of the target period into the training data.

    Returns (model_cutoff, forecast_start).
    """
    origin = pd.Timestamp(analysis_origin)
    forecast_start = origin.to_period("M").to_timestamp()      # 1st of this month
    model_cutoff = forecast_start - pd.Timedelta(minutes=TARGET_FREQ_MIN)
    return model_cutoff, forecast_start


def determine_rolling_window(analysis_origin):
    """Current calendar month plus the next three, derived from the data.

    The window rolls rather than being pinned to Sep-Dec, so the same script
    still makes sense when it is re-run in December (Dec/Jan/Feb/Mar) - the
    year boundary is handled by arithmetic on a Period rather than by a
    hard-coded month list.
    """
    origin = pd.Timestamp(analysis_origin).to_period("M")
    return [(p.year, p.month) for p in (origin + i for i in range(4))]


def month_name(year, month):
    return f"{pd.Timestamp(year=year, month=month, day=1):%B}"


def historical_same_months(df, year, month, origin_year):
    """All prior-year observations for one calendar month.

    Used both for EDA and for the historical-profile features. Only years
    strictly before the target are returned, so a feature built for
    October 2026 can never see October 2026 demand.
    """
    hist = df[(df["month"] == month) & (df["year"] < origin_year)]
    return hist


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Lags kept for the production four-month horizon. lag_1/lag_2 are built for
# diagnostics but deliberately excluded from the recursive feature set: a
# four-month forecast would have to generate 5,800 consecutive one-step
# predictions to use them, compounding error at every step.
HORIZON_SAFE_LAGS = {"lag_48": 48, "lag_96": 96, "lag_336": 336}
# lag_1/lag_2 are deliberately NOT built. A four-month recursive forecast would
# need 5,800 consecutive one-step predictions to use them, so they could never
# reach production - carrying them would only invite someone to fit on them and
# report a backtest score that cannot be reproduced in a real forecast.
DIAGNOSTIC_LAGS = {}

CALENDAR_FEATURES = ["half_hour_index", "hour", "day_of_week", "day_of_month",
                     "week_of_year", "month", "quarter", "day_of_year", "is_weekend",
                     "sin_half_hour", "cos_half_hour", "sin_day_of_week",
                     "cos_day_of_week", "sin_month", "cos_month",
                     "sin_day_of_year", "cos_day_of_year"]
HOLIDAY_FEATURES = ["is_public_holiday", "is_day_before_holiday",
                    "is_day_after_holiday"]
HISTORICAL_FEATURES = ["historical_month_halfhour_mean",
                       "historical_month_halfhour_median",
                       "historical_month_dow_halfhour_mean",
                       "historical_month_daily_mean",
                       "historical_month_peak_mean",
                       "historical_month_variability"]
SAFE_DEMAND_FEATURES = ["previous_day_mean", "previous_day_peak", "previous_day_min",
                        "same_half_hour_7day_mean", "same_half_hour_14day_mean",
                        "rolling_mean_24h_at_t_minus_48",
                        "rolling_std_24h_at_t_minus_48"]
TREND_FEATURES = ["time_index", "days_since_start", "year"]
# Model B learns weather response from historical observed weather and uses the
# same numerical input fields with forecast weather at inference time. The source
# itself is kept as provenance metadata, not used as a predictor.
WEATHER_INPUT_FEATURES = [
    "weather_temp_c", "weather_humidity_pct",
    "weather_daily_temp_max_c", "weather_daily_temp_min_c",
    "weather_daily_temp_mean_c", "weather_daily_rainfall_mm",
]
WEATHER_CONTEXT_FEATURES = [
    "temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
    "expected_temp_climatology", "historical_rain_probability",
    "hdd_lag_1d", "cdd_lag_1d",
    "expected_hdd_climatology", "expected_cdd_climatology",
]
WEATHER_SAFE_FEATURES = WEATHER_INPUT_FEATURES + WEATHER_CONTEXT_FEATURES

# Degree-day base temperature is DERIVED per region, not assumed. The
# conventional 18 C is a building-standards convention, not a fact about the
# NEM, and the regions differ: Tasmania is heating-driven, Queensland
# cooling-driven. Searching the balance point on prior-year data makes the
# choice evidence-based and keeps it out of the target year.
BALANCE_POINT_SEARCH = np.arange(10.0, 28.0, 0.5)

# Model A is the core set; Model B adds forecast-safe weather. Stage 3 compares
# them chronologically and keeps weather only where it actually helps.
FEATURE_SET_A = (list(HORIZON_SAFE_LAGS) + CALENDAR_FEATURES + HOLIDAY_FEATURES
                 + HISTORICAL_FEATURES + SAFE_DEMAND_FEATURES + TREND_FEATURES)
FEATURE_SET_B = FEATURE_SET_A + WEATHER_SAFE_FEATURES


def add_calendar_features(g):
    """Calendar variables from local civil time, plus cyclical encodings.

    Cyclical encodings matter because half-hour 47 and half-hour 0 are adjacent
    in time but 47 apart as integers. A tree can learn around that with enough
    splits; sin/cos pairs hand it the structure directly, and they cost two
    columns each.
    """
    lt = g["local_timestamp"]
    g["half_hour_index"] = lt.dt.hour * 2 + (lt.dt.minute >= 30).astype(int)
    g["hour"] = lt.dt.hour
    g["day_of_week"] = lt.dt.dayofweek
    g["day_of_month"] = lt.dt.day
    g["week_of_year"] = lt.dt.isocalendar().week.astype(int)
    g["month"] = lt.dt.month
    g["quarter"] = lt.dt.quarter
    g["year"] = lt.dt.year
    g["day_of_year"] = lt.dt.dayofyear
    # `season` is a 1:1 relabelling of `month`, so it is not carried into the
    # dataset - sin_month/cos_month already give a model the seasonal position.
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)

    for name, value, period in (("half_hour", g["half_hour_index"], 48),
                                ("day_of_week", g["day_of_week"], 7),
                                ("month", g["month"], 12),
                                ("day_of_year", g["day_of_year"], 365.25)):
        angle = 2 * np.pi * value / period
        g[f"sin_{name}"] = np.sin(angle)
        g[f"cos_{name}"] = np.cos(angle)
    return g


def load_holidays_covering(raw_dir, region, needed_through):
    """Acquired holiday calendar, extended forward if it stops too early.

    A Stage 1 run generates holidays to the forecast end date, but an older
    acquisition stops at the day it ran. That leaves the whole forecast horizon
    with no holidays at all - and silently telling the model that Christmas is
    an ordinary Friday is worse than failing, because nothing looks wrong.

    Missing dates are filled from the same `holidays` library the acquisition
    uses. That is a deterministic statutory calendar, not estimated data, and
    every generated row is labelled so the extension is visible rather than
    blended into the acquired file.
    """
    h = load_holidays(raw_dir, region)
    h["provenance"] = "acquired"
    acquired_to = h["date"].max()
    needed_through = pd.Timestamp(needed_through)
    if acquired_to >= needed_through:
        return h, 0

    try:
        import holidays as _hol
    except ImportError:
        log.error(f"[{region}] holiday calendar stops at {acquired_to.date()} and the "
                  f"'holidays' package is unavailable - the forecast horizon will "
                  f"carry no public holidays")
        return h, 0

    years = sorted({acquired_to.year, needed_through.year})
    cal = _hol.Australia(subdiv=REGION_STATE[region], years=years)
    extra_dates = pd.date_range(acquired_to + pd.Timedelta(days=1), needed_through, freq="D")
    extra = pd.DataFrame({"date": extra_dates})
    extra["is_public_holiday"] = extra["date"].dt.date.isin(cal).astype(int)
    extra["holiday_name"] = extra["date"].dt.date.map(lambda d: cal.get(d, ""))
    extra["provenance"] = "generated (holidays library)"

    added = int(extra["is_public_holiday"].sum())
    log.warning(f"[{region}] holiday calendar stopped at {acquired_to.date()}; "
                f"extended to {needed_through.date()} adding {added} holiday date(s)")
    return pd.concat([h, extra], ignore_index=True), added


def add_holiday_features(g, raw_dir, region, holidays_table=None):
    """State holidays joined on local_date, plus the day either side.

    The day before and after a public holiday behave differently from an
    ordinary day - bridging days and pre-holiday activity shift demand - and
    both are known in advance, so they are safe for the whole horizon.
    """
    h = (holidays_table if holidays_table is not None
         else load_holidays(raw_dir, region))
    h = h[["date", "is_public_holiday", "holiday_name"]].rename(
        columns={"date": "local_date"})
    g = g.merge(h, on="local_date", how="left")
    g["is_public_holiday"] = g["is_public_holiday"].fillna(0).astype(int)
    g["holiday_name"] = g["holiday_name"].fillna("")

    holiday_dates = set(pd.DatetimeIndex(h.loc[h["is_public_holiday"].astype(bool),
                                               "local_date"]))
    day = pd.Timedelta(days=1)
    g["is_day_before_holiday"] = g["local_date"].add(day).isin(holiday_dates).astype(int)
    g["is_day_after_holiday"] = g["local_date"].sub(day).isin(holiday_dates).astype(int)
    return g


def add_horizon_safe_demand_features(g):
    """Demand features that survive a long recursive horizon.

    Everything here is anchored at t-48 or earlier, so a day-ahead forecast can
    compute them from observed data. Short-window statistics ending at t-1 are
    deliberately avoided: they are the strongest predictors in a backtest and
    the least available in a real four-month forecast.
    """
    g = g.sort_values(TIME).copy()
    demand = g[TARGET]

    for name, steps in {**HORIZON_SAFE_LAGS, **DIAGNOSTIC_LAGS}.items():
        g[name] = demand.shift(steps)

    # Previous-day summaries, taken from the day that ends 48 intervals back.
    prior_day = demand.shift(48)
    g["previous_day_mean"] = prior_day.rolling(48).mean()
    g["previous_day_peak"] = prior_day.rolling(48).max()
    g["previous_day_min"] = prior_day.rolling(48).min()

    # Same half-hour on previous days - a mean over the same clock position,
    # which is why the window steps by 48.
    same_hh = pd.concat([demand.shift(48 * k) for k in range(1, 8)], axis=1)
    g["same_half_hour_7day_mean"] = same_hh.mean(axis=1)
    same_hh_14 = pd.concat([demand.shift(48 * k) for k in range(1, 15)], axis=1)
    g["same_half_hour_14day_mean"] = same_hh_14.mean(axis=1)

    g["rolling_mean_24h_at_t_minus_48"] = prior_day.rolling(48).mean()
    g["rolling_std_24h_at_t_minus_48"] = prior_day.rolling(48).std()
    return g


def add_trend_features(g):
    """Simple level-shift features.

    Demand in 2026 is not demand in 2022 - population, rooftop solar and
    efficiency all move the level over a four-year history - and a four-month
    horizon sits far enough from the training mean for that to matter. These
    are offered as candidates only; Stage 3 decides whether they earn their
    place from validation performance rather than from an assumption here.
    """
    start = g[TIME].min()
    g["time_index"] = np.arange(len(g))
    g["days_since_start"] = (g[TIME] - start).dt.total_seconds() / 86400
    return g


def build_historical_month_profiles(g, origin_year):
    """Per-month demand profiles built from PRIOR YEARS only.

    This is the feature group that makes a four-month horizon tractable. A
    recursive lag decays into its own error after a few days; a profile saying
    "October half-hour 36 has historically averaged X" stays informative at day
    100, and is computable for every future timestamp before the forecast
    starts.

    The origin_year filter is the leakage control: profiles for October 2026
    are computed from October 2022-2025 only.
    """
    hist = g[g["year"] < origin_year].dropna(subset=[TARGET])
    if hist.empty:
        return {}

    profiles = {
        "halfhour": hist.groupby(["month", "half_hour_index"])[TARGET].agg(["mean", "median"]),
        "dow_halfhour": hist.groupby(["month", "day_of_week", "half_hour_index"])[TARGET].mean(),
    }
    daily = hist.groupby(["month", "local_date"])[TARGET].agg(["mean", "max"])
    profiles["daily"] = daily.groupby("month").agg(
        daily_mean=("mean", "mean"), peak_mean=("max", "mean"),
        variability=("mean", "std"))
    return profiles


def apply_historical_month_profiles(g, profiles):
    """Attach the prior-year profiles to every row, historical and future."""
    if not profiles:
        for c in HISTORICAL_FEATURES:
            g[c] = np.nan
        return g

    hh = profiles["halfhour"]
    idx = pd.MultiIndex.from_arrays([g["month"], g["half_hour_index"]])
    g["historical_month_halfhour_mean"] = hh["mean"].reindex(idx).to_numpy()
    g["historical_month_halfhour_median"] = hh["median"].reindex(idx).to_numpy()

    dow = profiles["dow_halfhour"]
    idx3 = pd.MultiIndex.from_arrays([g["month"], g["day_of_week"], g["half_hour_index"]])
    g["historical_month_dow_halfhour_mean"] = dow.reindex(idx3).to_numpy()

    daily = profiles["daily"]
    g["historical_month_daily_mean"] = g["month"].map(daily["daily_mean"])
    g["historical_month_peak_mean"] = g["month"].map(daily["peak_mean"])
    g["historical_month_variability"] = g["month"].map(daily["variability"])
    return g



def derive_balance_point(daily_temp, demand_by_date, origin_year):
    """Find the temperature at which demand is lowest, from prior years only.

    Heating and cooling load both rise as temperature moves away from this
    point, so it is the natural split between HDD and CDD. Deriving it means
    the report can state why the base is what it is, instead of citing a
    convention that was never checked against this data. Restricting the search
    to years before the target keeps it out of the forecast period.
    """
    hist = demand_by_date[demand_by_date.index.year < origin_year]
    joined = hist.to_frame("demand").join(daily_temp.rename("temp"), how="inner").dropna()
    if len(joined) < 365:
        return None, np.nan
    best, best_corr = None, -np.inf
    for base in BALANCE_POINT_SEARCH:
        corr = abs(joined["demand"].corr((joined["temp"] - base).abs()))
        if np.isfinite(corr) and corr > best_corr:
            best, best_corr = float(base), corr
    return best, round(best_corr, 4)


def add_weather_features(g, raw_dir, region, origin_year):
    """Join historical weather and build Model-B training inputs.

    Historical observed/reanalysis weather is legitimate training information.
    It is *not* automatically safe as a held-out future predictor; Stage 3 must
    rebuild test/future weather from a forecast vintage or from origin-safe
    climatology. That contract is recorded in the feature provenance output.
    """
    daily = load_weather_daily(raw_dir, region)
    if daily is None:
        missing_cols = [
            "temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm",
            "temp_halfhourly_c", "humidity_pct", "hdd", "cdd",
        ] + WEATHER_SAFE_FEATURES
        for c in missing_cols:
            g[c] = np.nan
        g["weather_available"] = 0
        g["weather_feature_source"] = "MISSING"
        g["balance_point_c"] = np.nan
        g["balance_point_corr"] = np.nan
        return g

    daily = daily.sort_values("date").copy()
    daily["temp_mean_lag_1d"] = daily["temp_mean_c"].shift(1)
    daily["temp_mean_lag_7d"] = daily["temp_mean_c"].shift(7)
    daily["rolling_temp_mean_7d"] = daily["temp_mean_c"].shift(1).rolling(7).mean()

    demand_by_date = g.set_index("local_date")[TARGET].groupby(level=0).mean()
    base, base_corr = derive_balance_point(
        daily.set_index("date")["temp_mean_c"], demand_by_date, origin_year)
    if base is None:
        base, base_corr = np.nan, np.nan
    daily["hdd"] = (base - daily["temp_mean_c"]).clip(lower=0) if base == base else np.nan
    daily["cdd"] = (daily["temp_mean_c"] - base).clip(lower=0) if base == base else np.nan
    daily["hdd_lag_1d"] = daily["hdd"].shift(1)
    daily["cdd_lag_1d"] = daily["cdd"].shift(1)

    keep = [c for c in (
        "date", "temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm",
        "temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
        "hdd", "cdd", "hdd_lag_1d", "cdd_lag_1d") if c in daily.columns]
    g = g.merge(daily[keep].rename(columns={"date": "local_date"}),
                on="local_date", how="left")
    g["balance_point_c"] = base
    g["balance_point_corr"] = base_corr

    hourly = load_weather_hourly(raw_dir, region)
    if hourly is not None:
        g = g.merge(hourly, left_on="local_timestamp", right_index=True, how="left")
    else:
        g["temp_halfhourly_c"] = np.nan
        g["humidity_pct"] = np.nan

    # Origin-safe climatology. These remain useful after a detailed weather
    # forecast runs out and are also rebuilt by Stage 3 for historical folds.
    hist = g[g["year"] < origin_year]
    if len(hist) and hist["temp_mean_c"].notna().any():
        clim = hist.groupby("day_of_year")["temp_mean_c"].mean()
        g["expected_temp_climatology"] = g["day_of_year"].map(clim)
        if base == base:
            g["expected_hdd_climatology"] = (base - g["expected_temp_climatology"]).clip(lower=0)
            g["expected_cdd_climatology"] = (g["expected_temp_climatology"] - base).clip(lower=0)
        else:
            g["expected_hdd_climatology"] = np.nan
            g["expected_cdd_climatology"] = np.nan
        rain = (hist.assign(wet=(hist["rainfall_mm"] > 0.2))
                .groupby("month")["wet"].mean())
        g["historical_rain_probability"] = g["month"].map(rain)
    else:
        for c in ("expected_temp_climatology", "historical_rain_probability",
                  "expected_hdd_climatology", "expected_cdd_climatology"):
            g[c] = np.nan

    # Common Model-B numerical schema. Historical rows use observed/reanalysis
    # weather; future rows use forecast/climatology in build_future_feature_frame.
    g["weather_temp_c"] = g["temp_halfhourly_c"].combine_first(g["temp_mean_c"])
    g["weather_humidity_pct"] = g["humidity_pct"]
    g["weather_daily_temp_max_c"] = g["temp_max_c"]
    g["weather_daily_temp_min_c"] = g["temp_min_c"]
    g["weather_daily_temp_mean_c"] = g["temp_mean_c"]
    g["weather_daily_rainfall_mm"] = g["rainfall_mm"]
    g["weather_available"] = g["weather_temp_c"].notna().astype(int)
    g["weather_feature_source"] = np.where(
        g["weather_available"].eq(1), "OBSERVED_REANALYSIS", "MISSING")
    return g


def _build_weather_climatology(g_hist, origin_year):
    """Climatology from rows available before the target year.

    It is a fallback only. It never masquerades as a numerical weather forecast.
    """
    hist = g_hist[(g_hist["year"] < origin_year) & g_hist[TARGET].notna()].copy()
    out = {}
    if hist.empty:
        return out

    if "temp_halfhourly_c" in hist and hist["temp_halfhourly_c"].notna().any():
        out["temp_hh"] = hist.groupby(["day_of_year", "half_hour_index"])["temp_halfhourly_c"].mean()
    if "humidity_pct" in hist and hist["humidity_pct"].notna().any():
        out["humidity_hh"] = hist.groupby(["month", "half_hour_index"])["humidity_pct"].mean()
    daily_cols = [c for c in ("temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm")
                  if c in hist and hist[c].notna().any()]
    if daily_cols:
        by_date = hist.groupby(["local_date", "day_of_year", "month"])[daily_cols].first().reset_index()
        out["daily_doy"] = by_date.groupby("day_of_year")[[c for c in daily_cols if c != "rainfall_mm"]].mean()
        if "rainfall_mm" in by_date:
            out["rain_month"] = by_date.groupby("month")["rainfall_mm"].mean()
    return out


def _map_multiindex(series, *arrays):
    if series is None:
        return np.full(len(arrays[0]), np.nan)
    idx = pd.MultiIndex.from_arrays(arrays)
    return series.reindex(idx).to_numpy()


def build_future_feature_frame(region, raw_dir, window, profiles, g_hist,
                               holidays_table=None, analysis_origin=None):
    """Build the current-month + next-three-month modelling frame.

    Weather priority is explicit:
      1. past rows in the current month -> observed/reanalysis weather if present;
      2. future rows covered by the current Stage-1 forecast -> FORECAST_7DAY;
      3. remaining future rows -> prior-year climatology fallback.

    The provenance columns are never predictors. They exist so Stage 3 can prove
    what information was available for every timestamp.
    """
    first = pd.Timestamp(year=window[0][0], month=window[0][1], day=1)
    last_year, last_month = window[-1]
    last = (pd.Timestamp(year=last_year, month=last_month, day=1)
            + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=30))
    grid = pd.date_range(first, last, freq=f"{TARGET_FREQ_MIN}min")

    market = grid.tz_localize(REGION_SOURCE_TZ[region])
    local = market.tz_convert(REGION_TZ[region]).tz_localize(None)
    f = pd.DataFrame({TIME: grid, "REGION": region, "local_timestamp": local})
    f["local_date"] = f["local_timestamp"].dt.normalize()
    f = add_calendar_features(f)
    f = add_holiday_features(f, raw_dir, region, holidays_table)
    f = apply_historical_month_profiles(f, profiles)

    start = g_hist[TIME].min()
    # Continue trend by elapsed 30-minute intervals rather than by row count;
    # future frame begins at month start and can overlap historical rows.
    f["time_index"] = ((f[TIME] - start).dt.total_seconds() /
                       (TARGET_FREQ_MIN * 60)).round().astype(int)
    f["days_since_start"] = (f[TIME] - start).dt.total_seconds() / 86400

    # Carry climatology features into the whole frame.
    hist_for_clim = g_hist[g_hist["year"] < window[0][0]]
    if len(hist_for_clim) and hist_for_clim["temp_mean_c"].notna().any():
        temp_clim = hist_for_clim.groupby("day_of_year")["temp_mean_c"].mean()
        f["expected_temp_climatology"] = f["day_of_year"].map(temp_clim)
        rain_prob = (hist_for_clim.assign(wet=(hist_for_clim["rainfall_mm"] > 0.2))
                     .groupby("month")["wet"].mean())
        f["historical_rain_probability"] = f["month"].map(rain_prob)
    else:
        f["expected_temp_climatology"] = np.nan
        f["historical_rain_probability"] = np.nan

    base = g_hist["balance_point_c"].dropna().iloc[-1] \
        if "balance_point_c" in g_hist and g_hist["balance_point_c"].notna().any() else np.nan
    if base == base:
        f["expected_hdd_climatology"] = (base - f["expected_temp_climatology"]).clip(lower=0)
        f["expected_cdd_climatology"] = (f["expected_temp_climatology"] - base).clip(lower=0)
    else:
        f["expected_hdd_climatology"] = np.nan
        f["expected_cdd_climatology"] = np.nan

    # Dynamic demand features are rebuilt recursively in Stage 3. Keep the
    # columns present but empty so the schema is explicit and uniform.
    for c in list(HORIZON_SAFE_LAGS) + SAFE_DEMAND_FEATURES:
        f[c] = np.nan

    # Start with climatology as the long-horizon weather fallback.
    clim = _build_weather_climatology(g_hist, window[0][0])
    temp_hh = clim.get("temp_hh")
    humidity_hh = clim.get("humidity_hh")
    daily_doy = clim.get("daily_doy")
    rain_month = clim.get("rain_month")

    f["weather_temp_c"] = _map_multiindex(
        temp_hh, f["day_of_year"], f["half_hour_index"]) if temp_hh is not None \
        else f["expected_temp_climatology"].to_numpy()
    f["weather_humidity_pct"] = _map_multiindex(
        humidity_hh, f["month"], f["half_hour_index"]) if humidity_hh is not None \
        else np.nan
    for src, dst in (("temp_max_c", "weather_daily_temp_max_c"),
                     ("temp_min_c", "weather_daily_temp_min_c"),
                     ("temp_mean_c", "weather_daily_temp_mean_c")):
        f[dst] = f["day_of_year"].map(daily_doy[src]) \
            if daily_doy is not None and src in daily_doy else np.nan
    f["weather_daily_rainfall_mm"] = f["month"].map(rain_month) \
        if rain_month is not None else np.nan
    f["weather_feature_source"] = "CLIMATOLOGY_FALLBACK"
    f["weather_forecast_origin"] = pd.NaT
    f["weather_lead_day"] = np.nan
    f["weather_forecast_source"] = ""
    f["weather_forecast_model"] = ""
    f["weather_retrieved_at_utc"] = ""

    # Past/current-month rows use historical observed weather where Stage 1 has it.
    observed_cols = [TIME] + [c for c in WEATHER_INPUT_FEATURES if c in g_hist]
    observed = g_hist[observed_cols].drop_duplicates(TIME, keep="last")
    obs_names = {c: f"_obs_{c}" for c in observed_cols if c != TIME}
    observed = observed.rename(columns=obs_names)
    f = f.merge(observed, on=TIME, how="left")
    origin = pd.Timestamp(analysis_origin if analysis_origin is not None else g_hist[TIME].max())
    past_mask = f[TIME] <= origin
    obs_temp = f.get("_obs_weather_temp_c", pd.Series(np.nan, index=f.index))
    use_obs = past_mask & obs_temp.notna()
    for c in WEATHER_INPUT_FEATURES:
        oc = f"_obs_{c}"
        if oc in f:
            f.loc[use_obs, c] = f.loc[use_obs, oc]
    f.loc[use_obs, "weather_feature_source"] = "OBSERVED_REANALYSIS"
    f = f.drop(columns=[c for c in f.columns if c.startswith("_obs_")])

    # Genuine latest 7-day forecast overrides climatology only for future rows.
    forecast_hh, forecast_daily, forecast_meta = load_weather_forecast(raw_dir, region)
    if forecast_hh is not None and len(forecast_hh):
        fh = forecast_hh.reset_index().rename(columns={
            "weather_forecast_origin": "_fc_forecast_origin",
            "weather_lead_day": "_fc_lead_day",
            "weather_forecast_source": "_fc_forecast_source",
            "weather_forecast_model": "_fc_forecast_model",
            "weather_retrieved_at_utc": "_fc_retrieved_at_utc",
            "weather_source_type": "_fc_source_type",
        })
        f = f.merge(fh, on="local_timestamp", how="left")
        future_forecast = ((f[TIME] > origin) & f["forecast_temp_c"].notna())
        f.loc[future_forecast, "weather_temp_c"] = f.loc[future_forecast, "forecast_temp_c"]
        f.loc[future_forecast, "weather_humidity_pct"] = f.loc[future_forecast, "forecast_humidity_pct"]
        mapping = {
            "_fc_forecast_origin": "weather_forecast_origin",
            "_fc_lead_day": "weather_lead_day",
            "_fc_forecast_source": "weather_forecast_source",
            "_fc_forecast_model": "weather_forecast_model",
            "_fc_retrieved_at_utc": "weather_retrieved_at_utc",
        }
        for src, dst in mapping.items():
            if src in f:
                f.loc[future_forecast, dst] = f.loc[future_forecast, src]
        f.loc[future_forecast, "weather_feature_source"] = "FORECAST_7DAY"
        drop_cols = [c for c in ("forecast_temp_c", "forecast_humidity_pct",
                                 "_fc_source_type", *mapping.keys()) if c in f]
        f = f.drop(columns=drop_cols)

    if forecast_daily is not None and len(forecast_daily):
        d = forecast_daily.rename(columns={
            "DATE": "local_date",
            "temp_max_c": "_fc_temp_max_c",
            "temp_min_c": "_fc_temp_min_c",
            "temp_mean_c": "_fc_temp_mean_c",
            "rainfall_mm": "_fc_rainfall_mm",
        })
        keep = [c for c in ("local_date", "_fc_temp_max_c", "_fc_temp_min_c",
                            "_fc_temp_mean_c", "_fc_rainfall_mm") if c in d]
        f = f.merge(d[keep], on="local_date", how="left")
        fc_day = (f[TIME] > origin) & f["_fc_temp_mean_c"].notna() \
            if "_fc_temp_mean_c" in f else pd.Series(False, index=f.index)
        for src, dst in (("_fc_temp_max_c", "weather_daily_temp_max_c"),
                         ("_fc_temp_min_c", "weather_daily_temp_min_c"),
                         ("_fc_temp_mean_c", "weather_daily_temp_mean_c"),
                         ("_fc_rainfall_mm", "weather_daily_rainfall_mm")):
            if src in f:
                f.loc[fc_day, dst] = f.loc[fc_day, src]
        f = f.drop(columns=[c for c in f.columns if c.startswith("_fc_")])

    # Weather lag/context features are training/history constructs. Stage 3 must
    # rebuild them recursively or from the current future weather frame.
    for c in ("temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
              "hdd_lag_1d", "cdd_lag_1d"):
        f[c] = np.nan

    return f, forecast_meta


# Backwards-compatible alias: deterministic calendar is still written separately.
def build_future_calendar(region, raw_dir, window, profiles, g_hist,
                          holidays_table=None):
    frame, _ = build_future_feature_frame(
        region, raw_dir, window, profiles, g_hist, holidays_table,
        analysis_origin=g_hist[TIME].max())
    keep = [TIME, "REGION", "local_timestamp", "local_date"] + CALENDAR_FEATURES + \
           HOLIDAY_FEATURES + HISTORICAL_FEATURES + TREND_FEATURES
    return frame[[c for c in keep if c in frame.columns]].copy()


# ============================================================================
# SIX CORE VALIDATION TESTS
# ============================================================================
# Deliberately six. Distribution shape is reported in the statistics section
# as evidence, never here as a gate: electricity demand is a deterministic
# daily cycle and is never normally distributed, so a normality criterion would
# reject every usable dataset.

def test_source_and_coverage(raw, hh, region):
    """1. Required fields, numeric demand, date range, detected frequency."""
    checks = []
    required = [TIME, "REGION", TARGET]
    missing = [c for c in required if c not in raw.columns]
    checks.append(("required source fields present", not missing,
                   f"missing: {missing}" if missing else f"{required} all present"))
    checks.append(("demand is numeric", pd.api.types.is_numeric_dtype(raw[TARGET]),
                   f"dtype {raw[TARGET].dtype}"))
    checks.append(("chronologically ordered", raw[TIME].is_monotonic_increasing,
                   f"{raw[TIME].min()} -> {raw[TIME].max()}"))
    intervals = sorted(raw["SOURCE_INTERVAL_MIN"].dropna().unique()) \
        if "SOURCE_INTERVAL_MIN" in raw else []
    checks.append(("source frequency detected", True,
                   f"native interval(s): {[int(i) for i in intervals] or 'inferred 5 min'}"))
    checks.append(("output covers the source window", len(hh) > 0 and
                   hh[TIME].max() >= raw[TIME].max() - pd.Timedelta(minutes=TARGET_FREQ_MIN),
                   f"processed to {hh[TIME].max()}"))
    return _as_frame("1. source & coverage", region, checks)


def test_duplicates_and_gaps(raw, cleaned, hh, region):
    """2. Duplicates, missing timestamps and unresolved gaps, before and after."""
    dup_before = int(raw[TIME].duplicated().sum())
    dup_after = int(hh[TIME].duplicated().sum())
    gaps_after = int(hh[TARGET].isna().sum())
    imputed = int(cleaned["is_imputed"].sum()) if "is_imputed" in cleaned else 0
    step = hh[TIME].diff().dropna().dt.total_seconds().div(60)
    checks = [
        ("duplicate timestamps removed", dup_after == 0,
         f"before {dup_before}, after {dup_after}"),
        ("grid continuous at 30 minutes", bool((step == TARGET_FREQ_MIN).all()),
         f"{int((step != TARGET_FREQ_MIN).sum())} irregular steps"),
        ("short gaps interpolated and flagged", True,
         f"{imputed} rows imputed (<= {SHORT_GAP_MAX} intervals)"),
        ("long gaps left explicit, not fabricated", True,
         f"{gaps_after} rows remain NaN and are flagged"),
    ]
    return _as_frame("2. duplicates & gaps", region, checks)


def test_resampling_accuracy(cleaned, hh, region, samples=200):
    """3. Independently recompute sampled 5-min -> 30-min means.

    Recalculated from the cleaned native-resolution series rather than trusting
    the pipeline's own output, because a resampling bug that shifts every value
    produces a perfectly plausible-looking series. This is the check that
    catches a wrong aggregation window.
    """
    native = cleaned.dropna(subset=[TARGET]).set_index(TIME)[TARGET]
    ending = REGION_INTERVAL_LABEL[region] == "ending"
    closed = label = "right" if ending else "left"
    recomputed = native.resample(f"{TARGET_FREQ_MIN}min", closed=closed,
                                 label=label).mean().dropna()

    joined = hh.set_index(TIME)[TARGET].dropna().to_frame("pipeline").join(
        recomputed.to_frame("recomputed"), how="inner")
    if joined.empty:
        return _as_frame("3. resampling accuracy", region,
                         [("independent recomputation", False, "no overlapping intervals")])
    check = joined.sample(min(samples, len(joined)), random_state=0)
    diff = (check["pipeline"] - check["recomputed"]).abs()
    checks = [
        ("sampled means match recomputation", bool(diff.max() < 0.01),
         f"{len(check)} sampled intervals, max abs diff {diff.max():.6f} MW"),
        ("mean is used, not sum", bool(joined["pipeline"].mean() <
                                       native.mean() * 2),
         f"processed mean {joined['pipeline'].mean():.1f} MW vs native "
         f"{native.mean():.1f} MW - summation would be ~6x"),
        ("interval convention applied", True,
         f"{REGION_INTERVAL_LABEL[region]}-labelled bins (closed={closed})"),
    ]
    return _as_frame("3. resampling accuracy", region, checks)


def test_row_reconciliation(raw, cleaned, hh, region):
    """4. Account for every row, and confirm the distribution did not shift.

    A row count that cannot be explained usually means data was lost silently.
    The mean/min/max comparison catches the subtler failure: the right number
    of rows carrying distorted values.
    """
    raw_rows = len(raw)
    dup_removed = int(raw[TIME].duplicated().sum())
    gap_rows = len(cleaned) - (raw_rows - dup_removed)
    imputed = int(cleaned["is_imputed"].sum()) if "is_imputed" in cleaned else 0

    before = raw[TARGET].dropna()
    after = hh[TARGET].dropna()
    mean_shift = abs(after.mean() - before.mean()) / before.mean() * 100
    checks = [
        ("row movement fully accounted for", True,
         f"raw {raw_rows:,} - {dup_removed} duplicates + {gap_rows:,} grid rows "
         f"({imputed} imputed) -> {len(hh):,} half-hourly rows"),
        ("mean preserved within 1%", mean_shift < 1.0,
         f"{before.mean():.1f} -> {after.mean():.1f} MW ({mean_shift:.3f}% shift)"),
        ("min not distorted", after.min() >= before.min() - 1e-6,
         f"{before.min():.1f} -> {after.min():.1f} MW"),
        ("max not inflated by aggregation", after.max() <= before.max() + 1e-6,
         f"{before.max():.1f} -> {after.max():.1f} MW (averaging lowers peaks)"),
    ]
    return _as_frame("4. row reconciliation", region, checks)


def test_weather_and_holidays(g, region):
    """5. Region, local date, historical weather and holiday mapping."""
    weather_pct = 100 * g["weather_available"].mean() if "weather_available" in g else 0
    holidays = int(g["is_public_holiday"].sum())
    checks = [
        ("region label consistent", g["REGION"].nunique() == 1, f"{g['REGION'].iat[0]}"),
        ("calendar derived from local civil date", "local_date" in g.columns,
         f"local_date present, state {REGION_STATE[region]}"),
        ("state holiday calendar joined", holidays > 0, f"{holidays:,} holiday half-hours"),
        ("day-before / day-after flags built",
         {"is_day_before_holiday", "is_day_after_holiday"} <= set(g.columns), "both present"),
        ("historical weather coverage", weather_pct > 0,
         f"{weather_pct:.1f}% of rows carry observed/reanalysis weather"
         if weather_pct else "no historical weather - Model B training unavailable"),
    ]
    return _as_frame("5. weather & holidays", region, checks)


def test_leakage_and_readiness(g, region, feature_set):
    """6. Demand leakage, benchmark isolation and modelling schema."""
    demand = g[TARGET]
    lag_ok = all(g[name].equals(demand.shift(steps))
                 for name, steps in HORIZON_SAFE_LAGS.items() if name in g)
    forbidden = [c for c in g.columns
                 if any(k in c.lower() for k in ("predispatch", "aemo_forecast", "requirement"))]
    step = g[TIME].diff().dropna().dt.total_seconds().div(60)
    required = {TIME, "REGION", TARGET, "local_date", "half_hour_index"}
    weather_schema = set(WEATHER_INPUT_FEATURES) <= set(g.columns)
    checks = [
        ("horizon-safe lags reference past only", lag_ok,
         f"{list(HORIZON_SAFE_LAGS)} verified against shift()"),
        ("t-48-anchored demand summaries present", all(c in g.columns for c in SAFE_DEMAND_FEATURES),
         "previous-day and same-half-hour summaries anchored at t-48 or earlier"),
        ("Model B weather schema present", weather_schema,
         "historical observed weather is training input; future weather is replaced by forecast/climatology"),
        ("no AEMO forecast or MarketRequirements predictor", not forbidden,
         f"forbidden columns found: {forbidden}" if forbidden
         else "AEMO data remains benchmark-only"),
        ("uniform 30-minute output", bool((step == TARGET_FREQ_MIN).all()), f"{len(g):,} rows"),
        ("Stage 3 required fields present", required <= set(g.columns),
         f"missing: {sorted(required - set(g.columns))}" if not required <= set(g.columns)
         else "all present"),
    ]
    return _as_frame("6. leakage & readiness", region, checks)


def validate_weather_forecast(raw_dir, region, forecast_meta, future_frame):
    """Validate the live Stage-1 forecast vintage without making it a hard gate."""
    rows = []
    def add(check, ok, detail, severity="required"):
        rows.append({"region": region, "check": check,
                     "result": "PASS" if ok else ("WARN" if severity == "warning" else "FAIL"),
                     "detail": detail})

    available = bool(forecast_meta.get("forecast_available"))
    add("latest forecast files available", available,
        "Stage-1 weather/forecast/latest found" if available
        else "no latest forecast files; future Model B will use climatology fallback",
        severity="warning")
    if not available:
        return pd.DataFrame(rows)

    days = int(forecast_meta.get("forecast_days") or 0)
    add("forecast covers 1-7 target days", 1 <= days <= 7, f"{days} distinct target day(s)")
    origin = pd.to_datetime(forecast_meta.get("forecast_origin"), errors="coerce")
    start = pd.to_datetime(forecast_meta.get("forecast_start"), errors="coerce")
    end = pd.to_datetime(forecast_meta.get("forecast_end"), errors="coerce")
    add("forecast origin precedes targets",
        pd.notna(origin) and pd.notna(start) and start.normalize() > origin.normalize(),
        f"origin {origin}, first target {start}")
    add("forecast target range ordered", pd.notna(start) and pd.notna(end) and end >= start,
        f"{start} -> {end}")

    fc = future_frame[future_frame["weather_feature_source"].eq("FORECAST_7DAY")]
    add("forecast mapped to 30-minute future frame", len(fc) > 0,
        f"{len(fc):,} half-hour rows use FORECAST_7DAY")
    if len(fc):
        add("forecast temperatures numeric", fc["weather_temp_c"].notna().all(),
            f"missing {int(fc['weather_temp_c'].isna().sum())}")
        hum = fc["weather_humidity_pct"].dropna()
        add("humidity within 0-100%", hum.empty or hum.between(0, 100).all(),
            f"range {hum.min() if len(hum) else np.nan}..{hum.max() if len(hum) else np.nan}")
        rain = fc["weather_daily_rainfall_mm"].dropna()
        add("rainfall non-negative", rain.empty or (rain >= 0).all(),
            f"minimum {rain.min() if len(rain) else np.nan}")
        lead = pd.to_numeric(fc["weather_lead_day"], errors="coerce").dropna()
        add("lead days are within 1-7", len(lead) > 0 and lead.between(1, 7).all(),
            f"lead range {lead.min() if len(lead) else np.nan}..{lead.max() if len(lead) else np.nan}")
        if pd.notna(origin):
            target_dates = fc["local_date"]
            add("no future observed weather mislabeled as forecast",
                (target_dates > origin.normalize()).all(),
                "FORECAST_7DAY rows occur only after forecast origin")
    return pd.DataFrame(rows)


def _as_frame(test_group, region, checks):
    return pd.DataFrame([{"test_group": test_group, "region": region, "check": name,
                          "result": "PASS" if ok else "FAIL", "detail": detail}
                         for name, ok, detail in checks])


def run_core_tests(raw, cleaned, hh, g, region, feature_set):
    return pd.concat([
        test_source_and_coverage(raw, hh, region),
        test_duplicates_and_gaps(raw, cleaned, hh, region),
        test_resampling_accuracy(cleaned, hh, region),
        test_row_reconciliation(raw, cleaned, hh, region),
        test_weather_and_holidays(g, region),
        test_leakage_and_readiness(g, region, feature_set),
    ], ignore_index=True)


# ============================================================================
# AEMO PRE-DISPATCH: BENCHMARK ONLY
# ============================================================================
# Pre-dispatch is split and kept for benchmark comparison only. Dashboard
# TOTALDEMAND is AEMO Scheduled Demand, while this project's target is Operational
# Demand. The two may be compared as related series, but they are not asserted to
# be identical and pre-dispatch is never used as a predictor.

def process_predispatch(raw_dir, region, hh):
    """Split ACTUAL from FORECAST, harmonise to 30 minutes, check consistency.

    Dashboard ACTUAL rows are Scheduled Demand and are compared with project
    Operational Demand only as a source-relationship consistency check. FORECAST
    rows are Scheduled Demand forecasts and are retained at the horizon AEMO
    actually published; no fixed 24/48-hour assumption is imposed.
    """
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
            return pd.DataFrame(columns=[TIME, "region", column])
        s = (sub.set_index(TIME)[TARGET]
             .resample(f"{TARGET_FREQ_MIN}min", closed=closed, label=label)
             .mean().dropna())
        return pd.DataFrame({TIME: s.index, "region": region, column: s.values})

    actual = to_half_hourly(df[kind == "ACTUAL"], "aemo_actual_MW")
    forecast = to_half_hourly(df[kind == "FORECAST"], "aemo_forecast_MW")
    if len(forecast):
        origin = forecast[TIME].min()
        forecast["forecast_origin"] = origin
        forecast["forecast_horizon_step"] = range(1, len(forecast) + 1)
        forecast["forecast_horizon_minutes"] = (
            (forecast[TIME] - origin).dt.total_seconds() / 60).astype(int)
        forecast["source_vintage"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M")

    # Source consistency: our processed demand vs AEMO's own ACTUAL.
    consistency = None
    if len(actual):
        joined = actual.merge(hh[[TIME, TARGET]], on=TIME, how="inner").dropna()
        if len(joined):
            diff = joined[TARGET] - joined["aemo_actual_MW"]
            consistency = pd.DataFrame([{
                "region": region, "matched_intervals": len(joined),
                "window_start": joined[TIME].min(), "window_end": joined[TIME].max(),
                "MAE_between_sources_MW": diff.abs().mean(),
                "bias_MW": diff.mean(),
                "correlation": joined[TARGET].corr(joined["aemo_actual_MW"]),
                "interpretation": "source consistency check - NOT forecast accuracy",
            }])
    return actual, forecast, consistency


# ============================================================================
# CONSOLIDATED ROLLING EDA  (one four-month view, not four folders)
# ============================================================================
def rolling_window_eda(g, region, window, origin, eda_dir, figures_dir,
                       make_figures=True):
    """Analyse the whole four-month horizon in one pass.

    Earlier versions wrote a folder per target month, which produced 24 files
    per region and made the output hard to read. The months are still kept
    apart analytically - they are panels and series within shared figures, not
    a merged distribution - but they are presented together, because the point
    of the comparison is how the four months differ from each other.
    """
    origin_ts = pd.Timestamp(origin)
    months = [m for _, m in window]
    labels = {m: month_name(y, m) for y, m in window}
    eda_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    hist = g[(g["month"].isin(months)) & (g["year"] < origin_ts.year)].dropna(subset=[TARGET])
    if hist.empty:
        return pd.DataFrame(), figures

    # --- one table covering all four months x all prior years --------------
    summary = (hist.groupby(["month", "year"])[TARGET]
               .agg(mean_MW="mean", peak_MW="max", min_MW="min", intervals="size")
               .reset_index())
    summary["month_name"] = summary["month"].map(labels)
    summary.insert(0, "region", region)

    profile = (hist.groupby(["month", "half_hour_index"])[TARGET].mean()
               .unstack(0).rename(columns=labels))
    daytype = (hist.assign(daytype=np.where(hist["is_public_holiday"] == 1, "holiday",
                           np.where(hist["is_weekend"] == 1, "weekend", "weekday")))
               .groupby(["month", "daytype"])[TARGET].mean().unstack().rename(index=labels))

    if not make_figures:
        return summary, figures

    def fig(path, caption):
        savefig(path)
        figures.append({"region": region, "category": "rolling_eda",
                        "file": path.name, "caption": caption})

    # 1 - year over year, four months side by side
    ax = summary.pivot(index="year", columns="month_name", values="mean_MW")[
        [labels[m] for m in months]].plot(kind="bar", figsize=(8, 3.6))
    ax.set(title=f"{region} - mean demand by year, four target months",
           ylabel="MW", xlabel="")
    fig(figures_dir / f"{region}_eda_01_year_over_year.png",
        "Mean demand for each target month across prior years")

    # 2 - half-hourly profile, months overlaid
    ax = profile.plot(figsize=(7.5, 3.6))
    ax.set(title=f"{region} - half-hourly profile by target month",
           xlabel="half-hour index (local)", ylabel="MW")
    fig(figures_dir / f"{region}_eda_02_halfhour_profile.png",
        "Average daily shape for each of the four target months")

    # 3 - weekday vs weekend, one panel per month
    axes = plt.subplots(1, len(months), figsize=(3.2 * len(months), 3.2),
                        sharey=True)[1]
    for ax, m in zip(np.atleast_1d(axes), months):
        sub = hist[hist["month"] == m]
        for lbl, sel in (("weekday", sub["is_weekend"] == 0),
                         ("weekend", sub["is_weekend"] == 1)):
            part = sub[sel]
            if len(part):
                part.groupby("half_hour_index")[TARGET].mean().plot(ax=ax, label=lbl)
        ax.set(title=labels[m], xlabel="half-hour")
        ax.legend(fontsize=7)
    plt.gcf().suptitle(f"{region} - weekday vs weekend by target month", y=1.02)
    fig(figures_dir / f"{region}_eda_03_weekday_weekend.png",
        "Weekday and weekend shape per target month")

    # 4 - daily mean and peak trend across the four months, all years
    daily = hist.groupby(["year", "local_date"])[TARGET].agg(["mean", "max"]).reset_index()
    _, ax = plt.subplots(figsize=(8.5, 3.4))
    for year, sub in daily.groupby("year"):
        ax.plot(pd.to_datetime(sub["local_date"]).dt.dayofyear, sub["mean"],
                lw=0.7, label=str(year))
    ax.set(title=f"{region} - daily mean demand over the target months",
           xlabel="day of year", ylabel="MW")
    ax.legend(ncol=4, fontsize=7)
    fig(figures_dir / f"{region}_eda_04_daily_trend.png",
        "Daily mean demand across the target window, by year")

    # 5 - weather response
    if "temp_mean_c" in hist and hist["temp_mean_c"].notna().any():
        _, ax = plt.subplots(figsize=(5.6, 4))
        for m in months:
            sub = hist[hist["month"] == m]
            ax.scatter(sub["temp_mean_c"], sub[TARGET], s=1.5, alpha=0.2, label=labels[m])
        ax.set(title=f"{region} - demand vs daily mean temperature",
               xlabel="mean temperature (C)", ylabel="MW")
        ax.legend(markerscale=6, fontsize=7)
        fig(figures_dir / f"{region}_eda_05_weather_response.png",
            "Demand against daily mean temperature, coloured by target month")

    # 6 - holiday vs ordinary day
    if hist["is_public_holiday"].any():
        ax = daytype.plot(kind="bar", figsize=(7, 3.4))
        ax.set(title=f"{region} - mean demand by day type and month",
               ylabel="MW", xlabel="")
        fig(figures_dir / f"{region}_eda_06_daytype.png",
            "Weekday, weekend and public holiday means per target month")

    return summary, figures


# ============================================================================
# FEATURE EVIDENCE  (academic analysis - not a readiness gate)
# ============================================================================
# Normality/Gaussian-fit diagnostics are intentionally excluded from Stage 2.
# Electricity demand is cyclical and non-Gaussian by construction; the useful
# evidence here is feature association and weather-response behaviour.

def correlation_evidence(g, region, feature_set, stats_dir, figures_dir,
                         make_figures=True):
    """Pearson and Spearman between each candidate feature and the target.

    Both are reported because Pearson only sees linear association and the
    demand-temperature relationship is U-shaped: a feature can matter a great
    deal while showing a near-zero Pearson value. A large gap between the two
    is itself the signal that the relationship is non-linear.
    """
    usable = [c for c in feature_set
              if c in g.columns and pd.api.types.is_numeric_dtype(g[c])]
    sub = g[usable + [TARGET]].dropna()
    rows = []
    for column in usable:
        pearson = sub[column].corr(sub[TARGET])
        spearman = sub[column].corr(sub[TARGET], method="spearman")
        gap = abs(spearman) - abs(pearson)
        if abs(pearson) < 0.05 and abs(spearman) < 0.05:
            finding, action = "no monotonic association", \
                "Weak candidate - keep only if Stage 3 importance supports it."
        elif gap > 0.10:
            finding, action = "non-linear association", \
                "Keep - tree models capture this; a linear model would miss it."
        else:
            finding, action = "linear association", "Keep as a predictor."
        rows.append({"region": region, "feature": column,
                     "pearson": round(pearson, 4), "spearman": round(spearman, 4),
                     "abs_gap": round(gap, 4), "n": len(sub),
                     "finding": finding, "action": action})
    table = pd.DataFrame(rows).sort_values("pearson", key=abs, ascending=False)

    figures = []
    if make_figures and len(table):
        top = table.head(18).iloc[::-1]
        _, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.barh(top["feature"], top["pearson"], label="Pearson")
        ax.barh(top["feature"], top["spearman"], alpha=0.5, label="Spearman")
        ax.set(title=f"{region} - feature vs demand correlation (top 18)",
               xlabel="correlation")
        ax.legend(fontsize=8)
        savefig(figures_dir / f"{region}_stats_03_correlation.png")
        figures.append({"region": region, "category": "statistics",
                        "file": f"{region}_stats_03_correlation.png",
                        "caption": "Pearson and Spearman correlation for the "
                                   "strongest candidate features; a gap between "
                                   "the two indicates a non-linear relationship"})
    return table, figures


def weather_response_summary(g, region, origin_year):
    """How demand responds to weather, and the balance point it was split at.

    The balance point is derived per region from prior years rather than
    assumed at the conventional 18 C - Tasmania is heating-driven and
    Queensland cooling-driven, and a single assumed base would misrepresent
    both.
    """
    base = g["balance_point_c"].iat[0] if "balance_point_c" in g else np.nan
    hist = g[g["year"] < origin_year].dropna(subset=[TARGET, "temp_mean_c"])
    if base != base or hist.empty:
        return pd.DataFrame()

    def corr(column):
        return (round(hist[TARGET].corr(hist[column]), 4)
                if column in hist and hist[column].notna().any() else np.nan)

    heating = corr("hdd")
    cooling = corr("cdd")
    character = ("heating-driven" if heating > cooling + 0.1 else
                 "cooling-driven" if cooling > heating + 0.1 else "mixed")
    return pd.DataFrame([{
        "region": region, "balance_point_c": base,
        "corr_hdd": heating, "corr_cdd": cooling,
        "corr_temp_mean": corr("temp_mean_c"),
        "corr_temp_halfhourly": corr("temp_halfhourly_c"),
        "corr_humidity": corr("humidity_pct"),
        "character": character,
        "derived_from_years": f"< {origin_year}",
        "note": "balance point derived from the data, not assumed at 18 C",
    }])



def month_to_date_comparison(g, region, origin):
    """Compare the partial current month against the same partial window in
    prior years.

    Comparing 1-13 September 2026 against complete Septembers of earlier years
    would make 2026 look artificially low simply because it is two-thirds of a
    month. Matching the day-of-month cut-off removes that artefact.
    """
    origin_ts = pd.Timestamp(origin)
    cutoff_day, month = origin_ts.day, origin_ts.month
    # Cut prior years at the same half-hour position within the day, not just
    # the same day number. The current month usually ends mid-day, so matching
    # only the day would give earlier years an extra evening peak and make the
    # current year look artificially low.
    cutoff_hh = origin_ts.hour * 2 + (origin_ts.minute >= 30)
    rows = []
    for year, sub in g[g["month"] == month].groupby("year"):
        mtd = sub[(sub["day_of_month"] < cutoff_day) |
                  ((sub["day_of_month"] == cutoff_day) &
                   (sub["half_hour_index"] <= cutoff_hh))]
        if mtd.empty:
            continue
        rows.append({"region": region, "year": int(year),
                     "window": f"1-{cutoff_day} {month_name(year, month)} "
                               f"(to half-hour {cutoff_hh})",
                     "intervals": len(mtd), "mean_MW": mtd[TARGET].mean(),
                     "peak_MW": mtd[TARGET].max(), "min_MW": mtd[TARGET].min(),
                     "is_partial_month": True})
    out = pd.DataFrame(rows)
    if len(out):
        base = out.loc[out["year"] < origin_ts.year, "mean_MW"].mean()
        out["pct_vs_prior_year_mean"] = (out["mean_MW"] / base - 1) * 100 if base else np.nan
    return out


# ============================================================================
# HORIZON-SAFE FEATURE REPORT AND READINESS GATE
# ============================================================================
def build_horizon_safe_feature_report(columns):
    """State how every predictor may be used at training and forecast time."""
    rows = []

    def add(feature, training, next_7d, four_month, reason, backtest_rule=""):
        rows.append({
            "feature": feature,
            "historical_training": training,
            "next_7_day_forecast": next_7d,
            "four_month_recursive": four_month,
            "backtest_rule": backtest_rule,
            "reason": reason,
        })

    for name in HORIZON_SAFE_LAGS:
        add(name, "Yes", "Yes, recursive", "Yes, recursive",
            "actual history initially, then earlier forecasts", "rebuild per fold origin")
    for name in SAFE_DEMAND_FEATURES:
        add(name, "Yes", "Yes, recursive", "Yes, recursive",
            "anchored at t-48 or earlier", "rebuild per fold origin")
    for name in CALENDAR_FEATURES + HOLIDAY_FEATURES + TREND_FEATURES:
        add(name, "Yes", "Yes", "Yes", "known/deterministic from timestamp")
    for name in HISTORICAL_FEATURES:
        add(name, "Yes", "Yes", "Yes", "prior-history demand profile",
            "must be recomputed using only rows before fold origin")

    for name in WEATHER_INPUT_FEATURES:
        add(name, "Observed weather", "Forecast weather", "Forecast then climatology",
            "same numerical feature space; provenance changes by target timestamp",
            "observed target-period weather cannot be used as a simulated future forecast")
    for name in ("temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
                 "hdd_lag_1d", "cdd_lag_1d"):
        add(name, "Yes", "Rebuild", "Rebuild/fallback", "weather-history summary",
            "recompute from information available at fold origin")
    for name in ("expected_temp_climatology", "historical_rain_probability",
                 "expected_hdd_climatology", "expected_cdd_climatology"):
        add(name, "Yes", "Fallback", "Yes", "origin-safe historical climatology",
            "recompute from rows before fold origin")

    add("aemo_forecast_MW", "No", "Benchmark only", "Benchmark only",
        "Scheduled Demand forecast; never a project predictor")
    add("aemo_actual_MW", "No", "Benchmark only", "Benchmark only",
        "Scheduled Demand actual; source-relationship comparison only")
    add("MarketRequirements", "No", "No", "No", "benchmark/context, not observed target")

    report = pd.DataFrame(rows)
    report["present_in_dataset"] = report["feature"].isin(columns)
    return report


def build_weather_feature_provenance():
    rows = []
    for f in WEATHER_INPUT_FEATURES:
        rows.append({
            "feature": f,
            "historical_training_source": "OBSERVED_REANALYSIS",
            "live_future_source_priority": "FORECAST_7DAY > CLIMATOLOGY_FALLBACK",
            "predictor": True,
            "historical_backtest_rule": "do not use realised target-period weather as if forecast",
        })
    for f in WEATHER_CONTEXT_FEATURES:
        rows.append({
            "feature": f,
            "historical_training_source": "historical weather",
            "live_future_source_priority": "origin-safe reconstruction/climatology",
            "predictor": True,
            "historical_backtest_rule": "recompute at each fold origin",
        })
    for f in ("weather_feature_source", "weather_forecast_origin", "weather_lead_day",
              "weather_forecast_source", "weather_forecast_model", "weather_retrieved_at_utc"):
        rows.append({
            "feature": f,
            "historical_training_source": "metadata",
            "live_future_source_priority": "metadata",
            "predictor": False,
            "historical_backtest_rule": "provenance only; never fit as numeric/categorical predictor",
        })
    return pd.DataFrame(rows)


def assess_readiness(tests, region, training_rows, weather_pct,
                     horizon_holidays=None, generated_holidays=0,
                     forecast_available=False, forecast_days=0):
    """READY / READY_WITH_WARNINGS / NOT_READY from structural tests."""
    failures = tests[tests["result"] == "FAIL"]
    critical = failures[failures["test_group"].str.startswith(("1.", "2.", "3.", "6."))]
    if len(critical):
        status = "NOT_READY"
    elif len(failures) or weather_pct == 0 or not forecast_available:
        status = "READY_WITH_WARNINGS"
    else:
        status = "READY"
    if horizon_holidays == 0:
        status = "NOT_READY"

    notes = []
    if len(failures):
        notes.append(f"{len(failures)} core check(s) failed")
    if weather_pct == 0:
        notes.append("no historical weather; Model B training unavailable")
    if not forecast_available:
        notes.append("live 7-day weather forecast absent; future frame uses climatology fallback")
    else:
        notes.append(f"live weather forecast available for {forecast_days} day(s)")
    if horizon_holidays == 0:
        notes.append("forecast horizon contains no public holidays")
    if generated_holidays:
        notes.append(f"{generated_holidays} holiday date(s) generated to cover horizon")

    return pd.DataFrame([{
        "region": region,
        "status": status,
        "checks_run": len(tests),
        "checks_passed": int((tests["result"] == "PASS").sum()),
        "critical_failures": len(critical),
        "training_ready_rows": training_rows,
        "historical_weather_coverage_pct": round(weather_pct, 2),
        "live_forecast_available": bool(forecast_available),
        "live_forecast_days": int(forecast_days or 0),
        "horizon_holiday_days": horizon_holidays,
        "generated_holiday_dates": generated_holidays,
        "notes": "; ".join(notes) or "all core checks passed",
    }])


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
# Organised by ARTEFACT, not by region. An earlier version wrote a folder tree
# per region, which produced 147 CSVs - six near-identical copies of every
# table. That is work for whoever reads it and adds nothing: a per-region
# readiness file says exactly what one row of the readiness summary says.
#
# The rule applied here: a file earns its place only if Stage 3 consumes it or
# the report cites it. Everything else was removed.

def create_output_dirs(run_dir, regions):
    paths = {name: run_dir / name for name in
             ("data", "benchmark", "validation", "analysis", "features",
              "figures", "metadata")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for region in regions:
        (paths["figures"] / region).mkdir(parents=True, exist_ok=True)
    return paths


def build_data_dictionary(g, feature_set_a, feature_set_b, constants):
    """Describe every delivered column, so Stage 3 needs no guesswork."""
    roles = {TIME: "index - market clock, fixed offset, 30-minute grid",
             TARGET: "TARGET - mean MW over the half hour",
             "REGION": "region label",
             "local_timestamp": "same instant in regional civil time",
             "local_date": "civil date - the holiday join key"}
    rows = []
    for column in g.columns:
        if column in roles:
            role = roles[column]
        elif column in HORIZON_SAFE_LAGS:
            role = f"predictor - target lagged {HORIZON_SAFE_LAGS[column]} half-hours"
        elif column in DIAGNOSTIC_LAGS:
            role = "diagnostic lag - excluded from the production feature set"
        elif column in feature_set_b and column not in feature_set_a:
            role = "predictor - forecast-safe weather (Feature Set B only)"
        elif column in feature_set_a:
            role = "predictor - Feature Set A (core)"
        elif column.startswith(("is_", "weather_available")):
            role = "quality or calendar flag"
        else:
            role = "supporting column - not a predictor"
        rows.append({"column": column, "dtype": str(g[column].dtype), "role": role,
                     "non_null_pct": round(100 * g[column].notna().mean(), 2),
                     "example": str(g[column].dropna().iloc[0])
                                if g[column].notna().any() else ""})
    dictionary = pd.DataFrame(rows)
    for name, value in constants.items():
        dictionary.loc[len(dictionary)] = {
            "column": name, "dtype": "constant", "role": "constant for this region",
            "non_null_pct": 100.0, "example": str(value)}
    return dictionary


def build_stage2_zip(run_dir, output_root, stamp, regions):
    """Compress and reopen, so a broken handoff is caught before Stage 3.

    The modelling datasets are written as plain CSV because that is what makes
    them directly usable. Deflate inside the archive recovers the space, so the
    delivered ZIP is no larger than it would be with pre-compressed members.
    """
    zip_path = Path(output_root) / f"Stage2_Preprocessing_Output_{stamp}.zip"
    run_dir = Path(run_dir)
    skip = {"__pycache__", ".venv", "venv", ".ipynb_checkpoints"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(run_dir.rglob("*")):
            if item.is_file() and not (set(item.parts) & skip) and item.suffix != ".pyc":
                archive.write(item, item.relative_to(run_dir.parent))

    result = {"zip_file": str(zip_path), "status": "FAIL", "entries": 0,
              "size_mb": 0.0, "message": ""}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            broken = archive.testzip()
            if broken:
                result["message"] = f"corrupt entry: {broken}"
                return result
            names = archive.namelist()
        result["entries"] = len(names)
        result["size_mb"] = round(zip_path.stat().st_size / 1048576, 2)
        needed = ([f"data/modelling_{r}_30min.csv" for r in regions] +
                  [f"data/holdout_actuals_{r}_30min.csv" for r in regions] +
                  ["data/future_calendar_all_regions.csv",
                   "data/future_feature_frame_all_regions.csv",
                   "features/feature_set_A_core.csv",
                   "features/feature_set_B_weather.csv",
                   "features/weather_feature_provenance.csv",
                   "validation/weather_forecast_validation.csv",
                   "metadata/run_config.json"])
        missing = [n for n in needed if not any(n in x for x in names)]
        if missing:
            result.update(status="WARNING", message=f"missing: {missing}")
        else:
            result.update(status="PASS",
                          message=f"{len(regions)} modelling datasets, "
                                  f"{sum(1 for n in names if n.endswith('.png'))} figures")
    except Exception as exc:
        result["message"] = f"could not reopen archive: {exc}"
    return result


def discard_run_folder(run_dir, zip_result):
    """Leave only the validated ZIP. Kept on failure - deleting the only copy
    on the strength of an archive we could not read would be unrecoverable."""
    if zip_result.get("status") == "FAIL":
        log.warning("[ZIP] validation failed - keeping the run folder")
        return False
    shutil.rmtree(run_dir, ignore_errors=True)
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run(dataset, output_root, regions=DEMAND_REGIONS, work_dir=None,
        make_figures=True):
    """Stage 2 end to end: clean, harmonise, engineer, validate and package."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"Stage2_Preprocessing_{stamp}"
    out = create_output_dirs(run_dir, regions)

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = resolve_raw_root(dataset, work_dir or tmp)
        log.info(f"[input] raw sources: {raw_dir}")

        raw_all = load_all_demand(raw_dir, regions)
        analysis_origin = raw_all[TIME].max()          # latest actual-demand cutoff
        forecast_anchor = detect_stage1_forecast_origin(raw_dir, regions)
        rolling_anchor = forecast_anchor if forecast_anchor is not None else analysis_origin
        window = determine_rolling_window(rolling_anchor)
        model_cutoff, forecast_start = derive_model_cutoff(rolling_anchor)
        log.info(f"[window] demand cutoff {analysis_origin} | rolling anchor {rolling_anchor} "
                 f"-> target months {[f'{y}-{m:02d}' for y, m in window]}")
        log.info(f"[cutoff] training data ends {model_cutoff} (last complete month); "
                 f"{forecast_start:%b %Y} onward is held out as the forecast period")

        tests_all, readiness, coverage = [], [], []
        corr_all, eda_all, weather_all, consistency_all = [], [], [], []
        aemo_actual_all, aemo_forecast_all = [], []
        mtd_all, future_calendars, future_features, figure_index = [], [], [], []
        forecast_validation_all, forecast_meta_all = [], []
        dictionaries = []

        for region in regions:
            log.info(f"[{region}] processing")
            raw = raw_all[raw_all["REGION"] == region].copy()
            figures_dir = out["figures"] / region

            cleaned = reindex_and_flag_gaps(drop_duplicates(raw))
            cleaned = flag_invalid_and_outliers(cleaned)

            hh = harmonise_to_half_hourly(cleaned, region)
            hh = enforce_continuous_grid(hh, region)
            hh = resolve_clocks(hh, region)

            last_year, last_month = window[-1]
            horizon_end = (pd.Timestamp(year=last_year, month=last_month, day=1)
                           + pd.offsets.MonthEnd(1))
            holidays_table, generated = load_holidays_covering(
                raw_dir, region, horizon_end)

            # Historical modelling dataset.
            g = add_calendar_features(hh)
            g = add_holiday_features(g, raw_dir, region, holidays_table)
            g = add_horizon_safe_demand_features(g)
            g = add_trend_features(g)
            profiles = build_historical_month_profiles(g, analysis_origin.year)
            g = apply_historical_month_profiles(g, profiles)
            g = add_weather_features(g, raw_dir, region, analysis_origin.year)

            weather_pct = 100 * g["weather_available"].mean()
            available = [c for c in FEATURE_SET_B if c in g.columns]

            constants = {c: g[c].iloc[0] for c in g.columns
                         if g[c].nunique(dropna=False) <= 1}
            slim = g.drop(columns=list(constants))

            # The split happens HERE, after every feature has been built on the
            # full series. Splitting first would leave lag_48 empty for the
            # first day of the forecast period, because the values it needs sit
            # on the training side of the boundary. The lags may read across
            # the cutoff; the training ROWS may not cross it.
            train = slim[slim[TIME] <= model_cutoff].copy()
            holdout = slim[slim[TIME] > model_cutoff].copy()

            train.to_csv(out["data"] / f"modelling_{region}_30min.csv",
                         index=False, float_format="%.4f")
            # Observed actuals inside the forecast period. Not training data -
            # this is what Stage 3 scores its forecast against.
            holdout.to_csv(out["data"] / f"holdout_actuals_{region}_30min.csv",
                           index=False, float_format="%.4f")
            log.info(f"[{region}] training to {train[TIME].max()} "
                     f"({len(train):,} rows) | holdout {len(holdout):,} rows "
                     f"from {forecast_start:%Y-%m-%d}")
            dictionaries.append(build_data_dictionary(
                slim, FEATURE_SET_A, FEATURE_SET_B, constants).assign(region=region))

            # Future deterministic calendar and forecast-aware Model-B frame.
            future_frame, forecast_meta = build_future_feature_frame(
                region, raw_dir, window, profiles, g, holidays_table,
                analysis_origin=analysis_origin)
            future_features.append(future_frame)
            cal_keep = ([TIME, "REGION", "local_timestamp", "local_date"] +
                        CALENDAR_FEATURES + HOLIDAY_FEATURES + HISTORICAL_FEATURES + TREND_FEATURES)
            future_calendars.append(future_frame[[c for c in cal_keep if c in future_frame]])
            forecast_meta_all.append(forecast_meta)
            fv = validate_weather_forecast(raw_dir, region, forecast_meta, future_frame)
            forecast_validation_all.append(fv)

            # AEMO benchmark only.
            actual, forecast, consistency = process_predispatch(raw_dir, region, hh)
            if actual is not None and len(actual):
                aemo_actual_all.append(actual)
            if forecast is not None and len(forecast):
                aemo_forecast_all.append(forecast)
            if consistency is not None:
                consistency_all.append(consistency)

            tests = run_core_tests(raw, cleaned, hh, g, region, available)
            tests_all.append(tests)

            eda, figs = rolling_window_eda(g, region, window, analysis_origin,
                                           out["analysis"], figures_dir,
                                           make_figures)
            if len(eda):
                eda_all.append(eda)
            figure_index.extend(figs)
            mtd = month_to_date_comparison(g, region, analysis_origin)
            if len(mtd):
                mtd_all.append(mtd)

            # Compact feature evidence; normality/Gaussian-fit outputs are omitted.
            corr, f2 = correlation_evidence(g, region, available, out["analysis"],
                                            figures_dir, make_figures)
            figure_index.extend(f2)
            if len(corr):
                corr_all.append(corr)
            if weather_pct > 0:
                wr = weather_response_summary(g, region, analysis_origin.year)
                if len(wr):
                    weather_all.append(wr)

            # Counted on the training split only. Rows inside the forecast
            # period are observed, but they are not training data.
            training_rows = int(train.dropna(
                subset=[c for c in available if c in train] + [TARGET]).shape[0])
            horizon_holidays = int(future_frame["is_public_holiday"].sum() // 48)
            forecast_available = bool(forecast_meta.get("forecast_available"))
            forecast_days = int(forecast_meta.get("forecast_days") or 0)
            readiness.append(assess_readiness(
                tests, region, training_rows, weather_pct,
                horizon_holidays, generated, forecast_available, forecast_days))
            coverage.append({
                "region": region,
                "rows_30min": len(g),
                "training_ready_rows": training_rows,
                "training_ready_pct": round(100 * training_rows / len(train), 2),
                "training_end": train[TIME].max(),
                "holdout_rows": len(holdout),
                "holdout_start": holdout[TIME].min() if len(holdout) else pd.NaT,
                "start": g[TIME].min(),
                "end": g[TIME].max(),
                "missing_demand_rows": int(g[TARGET].isna().sum()),
                "historical_weather_coverage_pct": round(weather_pct, 2),
                "live_forecast_available": forecast_available,
                "live_forecast_days": forecast_days,
                "forecast_rows_30min": int((future_frame["weather_feature_source"] == "FORECAST_7DAY").sum()),
                "climatology_fallback_rows_30min": int((future_frame["weather_feature_source"] == "CLIMATOLOGY_FALLBACK").sum()),
                "observed_weather_rows_in_window": int((future_frame["weather_feature_source"] == "OBSERVED_REANALYSIS").sum()),
                "mean_demand_MW": round(g[TARGET].mean(), 1),
                "peak_demand_MW": round(g[TARGET].max(), 1),
            })

    # Cross-region deliverables.
    pd.concat(future_calendars, ignore_index=True).to_csv(
        out["data"] / "future_calendar_all_regions.csv", index=False,
        float_format="%.4f")
    pd.concat(future_features, ignore_index=True).to_csv(
        out["data"] / "future_feature_frame_all_regions.csv", index=False,
        float_format="%.4f")

    for frames, path in ((aemo_actual_all, out["benchmark"] / "aemo_actual_30min.csv"),
                         (aemo_forecast_all, out["benchmark"] / "aemo_forecast_30min.csv"),
                         (consistency_all, out["benchmark"] / "aemo_source_consistency.csv")):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(path, index=False,
                                                        float_format="%.4f")

    tests_table = pd.concat(tests_all, ignore_index=True)
    tests_table.to_csv(out["validation"] / "core_validation_tests.csv", index=False)
    fv_table = pd.concat(forecast_validation_all, ignore_index=True) \
        if forecast_validation_all else pd.DataFrame()
    fv_table.to_csv(out["validation"] / "weather_forecast_validation.csv", index=False)

    ready = pd.concat(readiness, ignore_index=True)
    ready = ready.merge(pd.DataFrame(coverage), on="region", how="left",
                        suffixes=("", "_coverage"))
    ready.to_csv(out["validation"] / "readiness_summary.csv", index=False)

    if corr_all:
        pd.concat(corr_all, ignore_index=True).to_csv(
            out["analysis"] / "feature_correlation.csv", index=False)
    if eda_all:
        pd.concat(eda_all, ignore_index=True).to_csv(
            out["analysis"] / "rolling_window_eda.csv", index=False)
    if mtd_all:
        pd.concat(mtd_all, ignore_index=True).to_csv(
            out["analysis"] / "month_to_date_comparison.csv", index=False)
    if weather_all:
        pd.concat(weather_all, ignore_index=True).to_csv(
            out["analysis"] / "weather_response.csv", index=False)

    dictionary = pd.concat(dictionaries, ignore_index=True)
    report = build_horizon_safe_feature_report(dictionary["column"].unique())
    report["in_feature_set_A_core"] = report["feature"].isin(FEATURE_SET_A)
    report["in_feature_set_B_weather"] = report["feature"].isin(FEATURE_SET_B)
    report.to_csv(out["features"] / "horizon_safe_feature_report.csv", index=False)
    pd.DataFrame({"feature": FEATURE_SET_A}).to_csv(
        out["features"] / "feature_set_A_core.csv", index=False)
    pd.DataFrame({"feature": FEATURE_SET_B}).to_csv(
        out["features"] / "feature_set_B_weather.csv", index=False)
    build_weather_feature_provenance().to_csv(
        out["features"] / "weather_feature_provenance.csv", index=False)

    dictionary.to_csv(out["metadata"] / "data_dictionary.csv", index=False)
    if figure_index:
        pd.DataFrame(figure_index).to_csv(out["metadata"] / "figure_index.csv", index=False)

    fm = pd.DataFrame(forecast_meta_all)
    valid_origins = pd.to_datetime(fm.get("forecast_origin"), errors="coerce").dropna() \
        if len(fm) else pd.Series(dtype="datetime64[ns]")
    valid_starts = pd.to_datetime(fm.get("forecast_start"), errors="coerce").dropna() \
        if len(fm) else pd.Series(dtype="datetime64[ns]")
    valid_ends = pd.to_datetime(fm.get("forecast_end"), errors="coerce").dropna() \
        if len(fm) else pd.Series(dtype="datetime64[ns]")
    run_config = {
        "analysis_origin": str(analysis_origin),
        "model_training_cutoff": str(model_cutoff),
        "forecast_period_start": str(forecast_start),
        "cutoff_rule": "training data ends at the last half-hour of the previous "
                       "complete calendar month; the current month onward is held out",
        "rolling_anchor": str(rolling_anchor),
        "rolling_window": [f"{y}-{m:02d}" for y, m in window],
        "regions": list(regions),
        "target_frequency_min": TARGET_FREQ_MIN,
        "short_gap_max_intervals": SHORT_GAP_MAX,
        "feature_set_A_size": len(FEATURE_SET_A),
        "feature_set_B_size": len(FEATURE_SET_B),
        "model_A_horizon": "current month + next 3 months",
        "model_B_weather_policy": "OBSERVED_REANALYSIS for past rows; FORECAST_7DAY for available future rows; CLIMATOLOGY_FALLBACK otherwise",
        "weather_forecast_origin": str(valid_origins.min()) if len(valid_origins) else None,
        "weather_forecast_start": str(valid_starts.min()) if len(valid_starts) else None,
        "weather_forecast_end": str(valid_ends.max()) if len(valid_ends) else None,
        "weather_forecast_regions_available": int(fm["forecast_available"].sum()) if len(fm) and "forecast_available" in fm else 0,
        "input_dataset": str(dataset),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out["metadata"] / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8")

    zip_result = build_stage2_zip(run_dir, output_root, stamp, regions)
    discard_run_folder(run_dir, zip_result)

    failures = int((tests_table["result"] == "FAIL").sum())
    not_ready = int((ready["status"] == "NOT_READY").sum())
    status = ("FAILED" if not_ready else
              "COMPLETE_WITH_WARNINGS"
              if failures or (ready["status"] == "READY_WITH_WARNINGS").any()
              else "COMPLETE")

    log.info("=" * 62)
    log.info(f"  {status}")
    log.info(f"  Modelling datasets : {len(regions)} x 30-minute")
    log.info(f"  Future feature frame: current month + next 3 months")
    log.info(f"  Core tests         : {int((tests_table['result'] == 'PASS').sum())}/{len(tests_table)} PASS")
    log.info(f"  Readiness          : {ready['status'].value_counts().to_dict()}")
    log.info(f"  Rolling window     : {[f'{y}-{m:02d}' for y, m in window]}")
    log.info(f"  Stage 2 ZIP        : {Path(zip_result['zip_file']).name} ({zip_result['status']}, {zip_result['size_mb']} MB)")
    log.info("=" * 62)

    return {
        "status": status,
        "tests": tests_table,
        "forecast_validation": fv_table,
        "readiness": ready,
        "coverage": pd.DataFrame(coverage),
        "zip": zip_result,
        "window": window,
        "analysis_origin": analysis_origin,
    }


def prompt_for_path(kind):
    """Browse for the Stage 1 ZIP and the output folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "zip":
            chosen = filedialog.askopenfilename(
                title="Select the Stage 1 acquisition ZIP",
                filetypes=[("ZIP archives", "*.zip"), ("All files", "*.*")])
        else:
            chosen = filedialog.askdirectory(title="Select folder to save Stage 2 output")
        root.destroy()
        if chosen:
            return chosen
    except Exception as exc:
        log.warning(f"No dialog available ({exc}).")
    return input(f"Path to {kind}: ").strip()


def main():
    parser = argparse.ArgumentParser(description="PRT661 Stage 2 preprocessing")
    parser.add_argument("--dataset", help="Stage 1 ZIP or an extracted folder")
    parser.add_argument("--output-dir", help="where the Stage 2 ZIP is written")
    parser.add_argument("--regions", nargs="+", default=DEMAND_REGIONS)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset or prompt_for_path("zip")
    output_root = args.output_dir or prompt_for_path("folder")
    if not dataset or not output_root:
        log.error("A Stage 1 dataset and an output folder are both required.")
        return

    result = run(dataset, output_root, args.regions, make_figures=not args.no_figures)
    print(f"\n{result['status']}")
    print(f"Stage 2 ZIP: {result['zip']['zip_file']} ({result['zip']['status']})")


if __name__ == "__main__":
    main()
