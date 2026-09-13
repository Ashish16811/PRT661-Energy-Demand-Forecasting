"""
ws1_data_engineering.py - Core data engineering and harmonisation
PRT661 WP2 Preprocessing · Dan6: Theme 2
Workstream owner: Suraj Raut (Data Engineering Lead)
Peer reviewer: Sudip Lamichhane

Schema validation · timestamp conventions · duplicates · gap detection ·
frequency harmonisation · grid continuity · region audit.

Two defects corrected in this workstream are worth reading before the code:

1. HALF-HOURLY AGGREGATION WAS MISALIGNED BY ONE INTERVAL.
   AEMO stamps a dispatch interval with its END time - the first reading of
   each day is 00:05, not 00:00 - so the half hour labelled 00:30 covers
   00:05..00:30. The previous implementation used pandas' default left-closed
   bins, which placed 00:30..00:55 there instead. Measured on NSW1: mean
   172.78 MW error per half hour, maximum 1,484.78 MW, 82,106 of 82,369
   buckets differing by more than 1 MW. Every value of the target variable in
   all six processed files was wrong.

2. A GRID DISCONTINUITY WOULD HAVE SILENTLY MISALIGNED EVERY LAG.
   See enforce_continuous_grid().
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

from common import (NEM_REGIONS, DEMAND_REGIONS, REGION_TZ, REGION_SOURCE_TZ,
                    REGION_INTERVAL_LABEL, REGION_STATE, TARGET_FREQ_MIN,
                    DEMAND_SCHEMA, SHORT_GAP_MAX)


def load_nem_region(region: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "price_and_demand" / f"price_and_demand_{region}.csv"
    df = pd.read_csv(path)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"],
                                          format="%Y/%m/%d %H:%M:%S")
    df = df[["REGION", "SETTLEMENTDATE", "TOTALDEMAND"]].copy()
    df["SOURCE"] = "AEMO_NEM_ARCHIVE"
    df["SOURCE_INTERVAL_MIN"] = np.nan       # not published; inferred below
    return df


def load_wa_demand(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "price_and_demand" / "WA_demand_2022_2026.csv"
    df = pd.read_csv(path)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    return df[["REGION", "SETTLEMENTDATE", "TOTALDEMAND",
               "SOURCE", "SOURCE_INTERVAL_MIN"]]


def load_all_demand(raw_dir: Path, regions=DEMAND_REGIONS) -> pd.DataFrame:
    frames = [load_wa_demand(raw_dir) if r == "WA" else load_nem_region(r, raw_dir)
              for r in regions]
    df = pd.concat(frames, ignore_index=True).sort_values(["REGION", "SETTLEMENTDATE"])
    log.info(f"[WS1] loaded {len(df):,} raw rows, {len(regions)} regions "
             f"({df.SETTLEMENTDATE.min()} -> {df.SETTLEMENTDATE.max()})")
    return df.reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on a schema drift rather than producing a wrong number
    quietly. Returns a per-column result table for the audit."""
    # Compare dtype FAMILY, not the exact spelling: pandas 2.x reports
    # object/datetime64[ns] where pandas 3.x reports str/datetime64[us] for
    # identical data, and a validator that fails on that is testing the
    # pandas version rather than the data.
    def family(dt) -> str:
        if pd.api.types.is_datetime64_any_dtype(dt):
            return "datetime"
        if pd.api.types.is_float_dtype(dt) or pd.api.types.is_integer_dtype(dt):
            return "numeric"
        if pd.api.types.is_string_dtype(dt) or dt == object:
            return "text"
        return str(dt)

    want_family = {"object": "text", "datetime64[ns]": "datetime", "float64": "numeric"}
    rows = []
    for col, want in DEMAND_SCHEMA.items():
        present = col in df.columns
        got = str(df[col].dtype) if present else "MISSING"
        ok = present and family(df[col].dtype) == want_family[want]
        rows.append({"column": col, "expected_family": want_family[want],
                     "actual_dtype": got, "status": "ok" if ok else "FAIL"})
    out = pd.DataFrame(rows)
    if (out["status"] == "FAIL").any():
        raise ValueError(f"Schema validation failed:\n{out.to_string(index=False)}")
    if not df.groupby("REGION")["SETTLEMENTDATE"].is_monotonic_increasing.all():
        raise ValueError("Timestamps are not chronologically ordered within region")
    log.info("[WS1] schema + chronological ordering validated")
    return out


def report_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Independent duplicate check. Acquisition already de-duplicates; this
    verifies that rather than trusting it."""
    rows = []
    for region, g in df.groupby("REGION"):
        rows.append({"region": region, "rows": len(g),
                     "duplicate_timestamps": int(g["SETTLEMENTDATE"].duplicated(keep=False).sum())})
    return pd.DataFrame(rows)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["REGION", "SETTLEMENTDATE"], keep="last")


def detect_resolution(g: pd.DataFrame) -> pd.Series:
    """Modal timestamp spacing per calendar month, in minutes.

    A single global median would misread WA's real 1 Oct 2023 30->5 minute
    change as either one enormous gap or a flood of spurious ones. A monthly
    mode adapts within one cycle of the transition and is stable elsewhere.
    """
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
        # At the row where the regime changes, "expected" is the NEW regime
        # but the measured step spans the OLD one. Take whichever adjacent
        # regime fits, so the transition step itself is not read as a gap.
        cand_curr = ((diff / expected).round() - 1).clip(lower=0)
        cand_prev = ((diff / expected.shift(1)).round() - 1).clip(lower=0)
        missing = np.minimum(cand_curr, cand_prev).fillna(0)
        events = missing[missing > 0]
        rows.append({
            "region": region, "present_rows": len(g),
            "resolution_regimes": sorted(expected.dropna().unique().tolist()),
            "missing_intervals_total": int(missing.sum()),
            "gap_events": int((events > 0).sum()),
            f"short_gaps_le_{SHORT_GAP_MAX}": int((events <= SHORT_GAP_MAX).sum()),
            f"long_gaps_gt_{SHORT_GAP_MAX}": int((events > SHORT_GAP_MAX).sum()),
            "completeness_pct": round(100 * len(g) / (len(g) + missing.sum()), 4),
        })
    return pd.DataFrame(rows)


def resolve_timestamp_conventions(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Make the clock explicit instead of assuming everything is local.

    source_timestamp    exactly as published
    market_timestamp    source_timestamp localised to the source's FIXED
                        offset (AEST for NEM, AWST for WA) - no DST
    canonical_timestamp the same instant in UTC, the only clock on which two
                        regions can be compared
    local_timestamp     that instant in the region's CIVIL time zone, which
                        DOES observe DST in NSW/VIC/SA/TAS
    local_date          civil calendar date - the correct key for a public
                        holiday join

    Why this matters: the modelling grid stays on the fixed-offset market
    clock, so every civil day has exactly 48 uniform half-hours. If the grid
    were built on local civil time instead, the October DST transition would
    produce a 46-interval day and the April one a 50-interval day with two
    identical 02:00-02:59 blocks - which would silently break lag_48 and any
    "48 rows per day" assumption. Calendar features are still derived from
    local civil time, because a public holiday is a civil-date fact.
    """
    g = g.copy()
    g = g.rename(columns={"SETTLEMENTDATE": "source_timestamp"}) \
        if "source_timestamp" not in g.columns else g
    src = g["source_timestamp"]
    market = src.dt.tz_localize(REGION_SOURCE_TZ[region])
    g["canonical_timestamp"] = market.dt.tz_convert("UTC").dt.tz_localize(None)
    local = market.dt.tz_convert(REGION_TZ[region])
    g["local_timestamp"] = local.dt.tz_localize(None)
    g["local_date"] = g["local_timestamp"].dt.normalize()
    return g


def harmonise_to_half_hourly(g: pd.DataFrame, region: str) -> pd.DataFrame:
    """Downsample only, and bin according to the source's own interval
    labelling convention.

    TOTALDEMAND is MW - an instantaneous power level, not an energy total.
    Summing six 5-minute MW readings would give a number six times too large
    with no physical meaning. The mean of the six readings is the average
    power over the half hour, which is the quantity a half-hourly demand
    forecast is defined on.

    Binning: AEMO stamps an interval with its END time, so the half hour
    labelled 00:30 covers readings 00:05..00:30 inclusive - closed="right",
    label="right". Using pandas' default left-closed bins instead shifts
    every value one 5-minute interval and mis-fills the first bucket. On
    NSW1 that default costs a mean 173 MW and up to 1485 MW per half hour.

    Native 30-minute rows (WA pre-cutover) pass through unchanged. Nothing is
    ever upsampled: turning one 30-minute reading into six 5-minute readings
    would fabricate observations that were never taken.
    """
    ending = REGION_INTERVAL_LABEL[region] == "ending"
    closed = label = "right" if ending else "left"
    idx = g.set_index("SETTLEMENTDATE")
    agg = {"TOTALDEMAND": "mean", "is_imputed": "max",
           "is_invalid": "max", "is_stat_outlier": "max"}
    out = idx.resample(f"{TARGET_FREQ_MIN}min", closed=closed, label=label).agg(agg)
    out["n_source_readings"] = idx["TOTALDEMAND"].resample(
        f"{TARGET_FREQ_MIN}min", closed=closed, label=label).count()
    out = out[out["n_source_readings"] > 0]          # drop empty leading bin
    out["REGION"] = region
    return out.reset_index()


def enforce_continuous_grid(hh: pd.DataFrame, region: str) -> pd.DataFrame:
    """Place the half-hourly series on an unbroken 30-minute index, so any
    period with no source coverage becomes an explicit NaN ROW rather than an
    absent one.

    This is not cosmetic. WA's market-system changeover leaves 2023-10-01
    00:00-07:55 AWST uncovered by either the legacy or the current-market
    file - 16 half-hours with no data. If those rows are merely absent,
    shift(48) reaches back 48 ROWS instead of 48 HALF-HOURS, so every lag and
    rolling window silently misaligns by 8.5 hours for the rest of the
    series. Materialising the gap as NaN keeps shift(k) equal to k half-hours
    everywhere, and the NaN then propagates honestly into the affected
    features instead of hiding as a wrong number.
    """
    full = pd.date_range(hh["SETTLEMENTDATE"].min(), hh["SETTLEMENTDATE"].max(),
                         freq=f"{TARGET_FREQ_MIN}min")
    before = len(hh)
    out = hh.set_index("SETTLEMENTDATE").reindex(full)
    out.index.name = "SETTLEMENTDATE"
    out["REGION"] = region
    for c in ("is_imputed", "is_invalid", "is_stat_outlier", "n_source_readings"):
        if c in out.columns:
            out[c] = out[c].fillna(0)
    added = len(out) - before
    if added:
        log.warning(f"[WS1] {region}: inserted {added} NaN rows to close grid "
                    f"discontinuities - lag alignment preserved")
    return out.reset_index()


def build_region_audit_table(df, dup_report, gap_report) -> pd.DataFrame:
    rows = []
    for region, g in df.groupby("REGION"):
        gaps = gap_report.set_index("region").loc[region]
        dups = dup_report.set_index("region").loc[region]
        rows.append({
            "Region": region, "Rows": len(g),
            "Start": g["SETTLEMENTDATE"].min(), "End": g["SETTLEMENTDATE"].max(),
            "Source interval (min)": "/".join(str(x) for x in gaps["resolution_regimes"]),
            "Source clock": REGION_SOURCE_TZ[region].replace("Etc/GMT-", "UTC+"),
            "Interval label": REGION_INTERVAL_LABEL[region],
            "Civil time zone": REGION_TZ[region],
            "Missing timestamps": int(gaps["missing_intervals_total"]),
            "Duplicate timestamps": int(dups["duplicate_timestamps"]),
            "Completeness %": gaps["completeness_pct"],
        })
    return pd.DataFrame(rows)

