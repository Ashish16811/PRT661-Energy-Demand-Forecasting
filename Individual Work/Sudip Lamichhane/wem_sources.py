"""
wem_sources.py - WA / WEM operational demand across the market cutover
PRT661 WP1 Data Acquisition · Dan6: Theme 2
Module owner: Sudip Lamichhane (Data Acquisition & Correction Handling Lead)

Assembles WA operational demand from three sources into one file, spanning the
WEM market-system changeover of 1 October 2023.

  legacy directory        30-minute demand, 1 Jan 2022 -> 30 Sep 2023
  current-market directory 5-minute demand, 1 Oct 2023 onwards
  daily WEM files         tail top-up, because the annual file lags by days

Design points established during WP1 (23 August minutes, section 2.3):
  · Where sources overlap the freshest wins: daily, then annual, then legacy.
  · Column names differ across the three sources, so timestamp and demand
    columns are detected STRUCTURALLY rather than by name. Detection rejects
    any column containing "forecast" or "withdrawal" - the current-market file
    carries demand and withdrawal side by side, and silently taking the wrong
    one would corrupt the target variable.
  · No resampling happens here. The 30-minute and 5-minute halves stay exactly
    as published; SOURCE and SOURCE_INTERVAL_MIN record where every row came
    from and at what frequency. Converting in acquisition would have destroyed
    the evidence WP2 needed to handle the cutover correctly.
  · tail_days bounds the daily top-up loop. Without it, a failed annual-file
    load rolled the start date back to 1 January and the loop issued roughly
    250 requests at 30-second timeouts each - the run did not crash, it
    stalled, and every later step sat behind it and never executed
    (23 August minutes, section 3.3, Fix 1).

MarketRequirements-2026.csv is downloaded separately and deliberately kept out
of the demand folder. It is a market REQUIREMENT forecast, not observed demand.
"""
import logging
import os
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

from common import (WA_OLD_BASE, WA_NEW_BASE, WA_DAILY_BASE, WA_MARKET_REQUIREMENTS_2026,
                    AEMO_CSV_HEADERS, DT_FORMATS, parse_dt)


def _parse_wa_datetime(series):
    s = series.astype(str).str.strip().str.strip('"')
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    out.loc[iso] = pd.to_datetime(s.loc[iso], errors="coerce", yearfirst=True)
    out.loc[~iso] = pd.to_datetime(s.loc[~iso], errors="coerce", dayfirst=True)
    return out

def _find_wa_timestamp(df, native_minutes):
    """Build the most likely WEM interval timestamp."""
    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    # Prefer a field that already contains a full date-time value.
    ranked = sorted(
        cols,
        key=lambda c: (
            0 if any(k in lowered[c] for k in ("timestamp", "dispatch interval", "trading interval")) else 1,
            0 if "date" in lowered[c] or "time" in lowered[c] or "interval" in lowered[c] else 1,
        ),
    )
    for col in ranked:
        if not any(k in lowered[col] for k in ("date", "time", "interval", "timestamp")):
            continue
        text = df[col].astype(str).str.strip()
        if text.str.contains(r"[-/:T]", regex=True, na=False).mean() < 0.80:
            continue
        parsed = _parse_wa_datetime(df[col])
        valid = parsed.notna() & parsed.dt.year.between(2000, 2100)
        if valid.mean() >= 0.80 and parsed[valid].nunique() / valid.sum() >= 0.80:
            return parsed

    date_cols = [c for c in cols if "date" in lowered[c]]
    interval_cols = [c for c in cols if "interval" in lowered[c] or "period" in lowered[c]]

    # Date + numeric interval number (48 legacy intervals or 288 dispatch intervals).
    for dcol in date_cols:
        base = _parse_wa_datetime(df[dcol]).dt.normalize()
        for icol in interval_cols:
            n = pd.to_numeric(df[icol], errors="coerce")
            max_interval = int(24 * 60 / native_minutes)
            if n.notna().mean() >= 0.80 and n.dropna().between(1, max_interval).all():
                return (base + pd.Timedelta(hours=8) +
                        pd.to_timedelta((n - 1) * native_minutes, unit="m"))

    # Date + clock time stored in separate columns.
    time_cols = [c for c in cols if "time" in lowered[c] or "interval" in lowered[c]]
    for dcol in date_cols:
        for tcol in time_cols:
            clock = df[tcol].astype(str).str.strip()
            if clock.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False).mean() < 0.80:
                continue
            combined = _parse_wa_datetime(
                df[dcol].astype(str).str.strip() + " " + clock)
            if combined.notna().mean() >= 0.80:
                return combined

    raise ValueError(f"Could not identify WA timestamp column. Columns: {cols}")

def _find_wa_demand_column(df):
    """Find operational demand while avoiding withdrawal/forecast fields."""
    candidates = []
    for col in df.columns:
        name = str(col).strip().lower().replace("_", " ")
        if "demand" not in name or "forecast" in name or "withdrawal" in name:
            continue
        score = 0
        if "operational" in name:
            score += 4
        if "mw" in name:
            score += 2
        if name in {"operational demand", "operational demand (mw)"}:
            score += 4
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.80:
            candidates.append((score, col))

    if not candidates:
        raise ValueError(f"Could not identify WA operational demand column. Columns: {list(df.columns)}")
    return max(candidates, key=lambda x: x[0])[1]


def _normalise_wa_frame(df, source, native_minutes):
    timestamp = _find_wa_timestamp(df, native_minutes)
    demand_col = _find_wa_demand_column(df)
    demand = pd.to_numeric(df[demand_col], errors="coerce")

    out = pd.DataFrame({"SETTLEMENTDATE": timestamp, "TOTALDEMAND": demand})
    out = (out.dropna()
              .drop_duplicates("SETTLEMENTDATE", keep="last")
              .sort_values("SETTLEMENTDATE"))
    out["REGION"] = "WA"
    out["PERIODTYPE"] = "ACTUAL"
    out["SOURCE"] = source
    out["SOURCE_INTERVAL_MIN"] = native_minutes
    return out[["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "PERIODTYPE",
                "SOURCE", "SOURCE_INTERVAL_MIN"]]


def _largest_record_list(obj):
    """Return the largest list of dictionaries inside an AEMO JSON payload."""
    found = []

    def walk(value):
        if isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                found.append(value)
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(obj)
    return max(found, key=len) if found else []


def _download_wa_csv(session, url, source, native_minutes):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    log.info(f"[WA] {source}: {len(df):,} raw rows | {url}")
    return _normalise_wa_frame(df, source, native_minutes)


def _fetch_wa_daily_tail(session, start_date, end_date):
    frames = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        stamp = day.strftime("%Y-%m-%d")
        url = f"{WA_DAILY_BASE}/OperationalDemandAndWithdrawal_{stamp}.json"
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            records = _largest_record_list(r.json())
            if not records:
                continue
            frames.append(_normalise_wa_frame(
                pd.json_normalize(records), "AEMO_WEM_DAILY_5MIN", 5))
        except Exception as e:
            log.warning(f"[WA daily] {stamp}: {e}")
        time.sleep(0.05)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_wa_demand(start_date, end_date, demand_dir, tail_days=14):
    """Build the WA/SWIS operational-demand history used as observed demand."""
    os.makedirs(demand_dir, exist_ok=True)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    end_exclusive = end + pd.Timedelta(days=1)
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "*/*"})

    legacy = []
    for year in (2022, 2023):
        url = f"{WA_OLD_BASE}/operational-demand-{year}.csv"
        try:
            legacy.append(_download_wa_csv(session, url, f"AEMO_WEM_LEGACY_{year}", 30))
        except Exception as e:
            log.error(f"[WA] Legacy {year} failed: {e}")

    new_market = []
    for year in (2023, 2024, 2025, 2026):
        url = f"{WA_NEW_BASE}/OperationalDemandWithdrawal-{year}.csv"
        try:
            new_market.append(_download_wa_csv(session, url, f"AEMO_WEM_{year}", 5))
        except Exception as e:
            log.error(f"[WA] {year} failed: {e}")

    frames = []
    if legacy:
        old = pd.concat(legacy, ignore_index=True)
        old = old[old["SETTLEMENTDATE"] < pd.Timestamp("2023-10-01")]
        frames.append(old)

    if new_market:
        modern = pd.concat(new_market, ignore_index=True)
        modern = modern[modern["SETTLEMENTDATE"] >= pd.Timestamp("2023-10-01")]
        frames.append(modern)

        # Current-year annual CSV lags by a few days, so top up the tail from
        # the WEM daily files.
        #
        # The tail is capped at tail_days. Without the cap, a missing or empty
        # annual file makes tail_start fall back to 1 January and the loop then
        # fires one request per day for the whole year - several hundred calls
        # that stall the run long before the later steps get to execute.
        current_year = pd.Timestamp.today().year
        y_now = modern[modern["SETTLEMENTDATE"].dt.year == current_year]
        tail_end = min(end.normalize(), pd.Timestamp.today().normalize())
        floor = tail_end - pd.Timedelta(days=tail_days)

        if y_now.empty:
            log.warning(f"[WA] No {current_year} annual rows - limiting the "
                        f"daily top-up to the last {tail_days} days.")
            tail_start = floor
        else:
            tail_start = max(y_now["SETTLEMENTDATE"].max().normalize(), floor)

        if tail_start <= tail_end:
            days = (tail_end - tail_start).days + 1
            log.info(f"[WA] Daily top-up: {days} day(s) "
                     f"{tail_start.date()} -> {tail_end.date()}")
            tail = _fetch_wa_daily_tail(session, tail_start, tail_end)
            if not tail.empty:
                frames.append(tail)

    if not frames:
        log.error("[WA] No operational demand data fetched.")
        return {}

    final = pd.concat(frames, ignore_index=True)
    final = final[(final["SETTLEMENTDATE"] >= start) &
                  (final["SETTLEMENTDATE"] < end_exclusive)].copy()
    final["_priority"] = final["SOURCE"].map(
        lambda x: 3 if "DAILY" in x else (2 if "WEM_20" in x else 1))
    final = (final.sort_values(["SETTLEMENTDATE", "_priority"], kind="stable")
             .drop_duplicates("SETTLEMENTDATE", keep="last")
             .drop(columns="_priority")
             .sort_values("SETTLEMENTDATE"))

    main_path = os.path.join(demand_dir, "WA_demand_2022_2026.csv")
    final.to_csv(main_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    if not final.empty:
        log.info(f"[WA] Combined: {len(final):,} rows "
                 f"({final.SETTLEMENTDATE.min()} -> {final.SETTLEMENTDATE.max()}) -> {main_path}")
    return {"combined": main_path}


def fetch_wa_market_requirements_2026(output_dir):
    """Download AEMO's raw 2026 WEM Market Requirements CSV unchanged."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "MarketRequirements-2026.csv")

    try:
        r = requests.get(
            WA_MARKET_REQUIREMENTS_2026,
            headers={"User-Agent": UA, "Accept": "text/csv,text/plain,*/*"},
            timeout=60,
        )
        r.raise_for_status()
        if r.content.lstrip().lower().startswith((b"<!doctype", b"<html")):
            raise ValueError("AEMO returned HTML instead of the CSV file")

        with open(path, "wb") as f:
            f.write(r.content)
        log.info(f"[WA requirements] 2026 raw CSV -> {path}")
        return path
    except Exception as e:
        log.error(f"[WA requirements] 2026 download failed: {e}")
        return None

