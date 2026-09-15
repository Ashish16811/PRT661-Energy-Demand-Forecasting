"""
PRT661 – Data Science Practice
WP2 Individual Contribution Module – Suraj Raut (S391201)
Role: Data Engineering Lead
Project: Australian Electricity Demand Forecasting | Dan6

Purpose
-------
This module represents Suraj's Stage 2 engineering workstream before group
integration. It focuses on source loading, native-frequency recognition,
30-minute harmonisation, continuous time-grid construction, time-zone handling,
and calendar/cyclical features.

The final production implementation is the group-integrated Preprocessing Final.py.
This file is a contribution artefact aligned to that final implementation; Git
history remains the authoritative evidence for authorship/timing.
"""
from pathlib import Path
import shutil
import zipfile
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("wp2_suraj")
TIME = "SETTLEMENTDATE"
TARGET = "TOTALDEMAND"
TARGET_FREQ_MIN = 30
NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
DEMAND_REGIONS = NEM_REGIONS + ["WA"]
REGION_TZ = {"NSW1":"Australia/Sydney","QLD1":"Australia/Brisbane",
             "VIC1":"Australia/Melbourne","SA1":"Australia/Adelaide",
             "TAS1":"Australia/Hobart","WA":"Australia/Perth"}
REGION_SOURCE_TZ = {r:"Etc/GMT-10" for r in NEM_REGIONS} | {"WA":"Etc/GMT-8"}
REGION_INTERVAL_LABEL = {r:"ending" for r in NEM_REGIONS} | {"WA":"beginning"}


def resolve_raw_root(source, work_dir="_raw_extracted") -> Path:
    """Accept a Stage 1 ZIP or folder and locate its real data root safely."""
    source = Path(source)
    known = {"price_and_demand", "weather", "bom_weather", "public_holidays",
             "predispatch", "dispatch", "energy_requirement"}
    if source.is_file() and source.suffix.lower() == ".zip":
        work = Path(work_dir)
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        with zipfile.ZipFile(source) as z:
            root_resolved = work.resolve()
            for member in z.namelist():
                if not str((work / member).resolve()).startswith(str(root_resolved)):
                    raise ValueError(f"Unsafe path in archive: {member}")
            z.extractall(work)
        root = work
    elif source.is_dir():
        root = source
    else:
        raise FileNotFoundError(f"Not a ZIP or folder: {source}")

    if any((root / k).is_dir() for k in known):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any((child / k).is_dir() for k in known):
            return child
    raise FileNotFoundError(f"No acquisition folders found under {root}")


def load_nem_region(region: str, raw_dir: Path) -> pd.DataFrame:
    """Load one NEM regional demand file without changing the source meaning."""
    df = pd.read_csv(raw_dir / "price_and_demand" / f"price_and_demand_{region}.csv")
    df[TIME] = pd.to_datetime(df[TIME], format="%Y/%m/%d %H:%M:%S")
    return df[["REGION", TIME, TARGET]]


def load_wa_demand(raw_dir: Path) -> pd.DataFrame:
    """Load WA demand while preserving the source interval declared by Stage 1."""
    df = pd.read_csv(raw_dir / "price_and_demand" / "WA_demand_2022_2026.csv")
    df[TIME] = pd.to_datetime(df[TIME])
    return df[["REGION", TIME, TARGET, "SOURCE_INTERVAL_MIN"]]


def load_all_demand(raw_dir: Path, regions=DEMAND_REGIONS) -> pd.DataFrame:
    frames = [load_wa_demand(raw_dir) if r == "WA" else load_nem_region(r, raw_dir)
              for r in regions]
    df = pd.concat(frames, ignore_index=True).sort_values(["REGION", TIME])
    log.info("Loaded %s raw demand rows across %s regions", f"{len(df):,}", len(regions))
    return df.reset_index(drop=True)


def detect_resolution(g: pd.DataFrame) -> pd.Series:
    """Infer native spacing per month so WA's historical regime change is respected."""
    diffs = g[TIME].diff().dt.total_seconds().div(60)
    month = g[TIME].dt.to_period("M")
    modal = diffs.groupby(month).agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
    return month.map(modal)


def harmonise_to_half_hourly(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Aggregate higher-frequency MW readings to 30 minutes by MEAN, never by sum."""
    ending = REGION_INTERVAL_LABEL[region] == "ending"
    closed = label = "right" if ending else "left"
    idx = g.set_index(TIME)
    agg = {TARGET:"mean", "is_imputed":"max", "is_invalid":"max", "is_stat_outlier":"max"}
    out = idx.resample(f"{TARGET_FREQ_MIN}min", closed=closed, label=label).agg(agg)
    out["n_source_readings"] = idx[TARGET].resample(
        f"{TARGET_FREQ_MIN}min", closed=closed, label=label).count()
    out = out[out["n_source_readings"] > 0]
    out["REGION"] = region
    return out.reset_index()


def enforce_continuous_grid(hh: pd.DataFrame, region: str) -> pd.DataFrame:
    """Materialise missing half-hours as NaN so lag(48) always means exactly one day."""
    full = pd.date_range(hh[TIME].min(), hh[TIME].max(), freq=f"{TARGET_FREQ_MIN}min")
    before = len(hh)
    out = hh.set_index(TIME).reindex(full)
    out.index.name = TIME
    out["REGION"] = region
    for c in ("is_imputed", "is_invalid", "is_stat_outlier", "n_source_readings"):
        out[c] = out[c].fillna(0).astype(int)
    if len(out) > before:
        log.warning("[%s] inserted %d explicit NaN grid rows", region, len(out)-before)
    return out.reset_index()


def resolve_clocks(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Keep the fixed-offset market clock and derive regional civil time for behaviour."""
    g = g.copy()
    market = g[TIME].dt.tz_localize(REGION_SOURCE_TZ[region])
    local = market.dt.tz_convert(REGION_TZ[region])
    g["local_timestamp"] = local.dt.tz_localize(None)
    g["local_date"] = g["local_timestamp"].dt.normalize()
    return g


def add_calendar_features(g: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic calendar and cyclical fields shared by Models A and B."""
    g = g.copy()
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
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
    for name, value, period in (("half_hour", g["half_hour_index"], 48),
                                ("day_of_week", g["day_of_week"], 7),
                                ("month", g["month"], 12),
                                ("day_of_year", g["day_of_year"], 365.25)):
        angle = 2 * np.pi * value / period
        g[f"sin_{name}"] = np.sin(angle)
        g[f"cos_{name}"] = np.cos(angle)
    return g


if __name__ == "__main__":
    print("WP2 Suraj Raut module loaded: engineering, harmonisation and calendar features.")
