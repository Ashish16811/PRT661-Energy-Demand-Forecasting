"""
data_acquisition.py - PRT661 data extraction """

import argparse
import importlib.util
import json
import logging
import os
import re
import time
import zipfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_acquisition")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


AEMO_CSV_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": ("https://www.aemo.com.au/energy-systems/electricity/"
                "national-electricity-market-nem/data-nem/aggregated-data"),
}

DASHBOARD_URL = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
DASHBOARD_HEADERS = {
    "User-Agent": UA,
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://www.aemo.com.au",
    "Referer": ("https://www.aemo.com.au/energy-systems/electricity/"
                "national-electricity-market-nem/data-nem/data-dashboard-nem"),
}

REGION_CONFIG = {
    "NSW1": {"state": "NSW", "city": "Sydney (Observatory Hill)"},
    "QLD1": {"state": "QLD", "city": "Brisbane"},
    "VIC1": {"state": "VIC", "city": "Melbourne (Olympic Park)"},
    "SA1":  {"state": "SA",  "city": "Adelaide (Airport)"},
    "TAS1": {"state": "TAS", "city": "Hobart (Ellerslie Road)"},
    "WA":   {"state": "WA",  "city": "Perth Airport"},
}
NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]

DT_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")

WA_OLD_BASE = "https://data.wa.aemo.com.au/datafiles/operational-demand"
WA_NEW_BASE = ("https://data.wa.aemo.com.au/datafiles/"
               "operational-demand-withdrawal-csv")
WA_DAILY_BASE = ("https://data.wa.aemo.com.au/public/market-data/wemde/"
                 "operationalDemandWithdrawal/dailyFiles")
WA_MARKET_REQUIREMENTS_2026 = (
    "https://data.wa.aemo.com.au/datafiles/market-requirements-csv/"
    "MarketRequirements-2026.csv"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def prompt_for_output_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select folder to save PRT661 data")
        root.destroy()
        if folder:
            return folder
    except Exception as e:
        log.warning(f"No folder dialog available ({e}).")
    return input("Path to save data into: ").strip() or "./data/raw"


def parse_dt(series):
    s = series.astype(str).str.strip().str.strip('"')
    for fmt in DT_FORMATS:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().any():
            return out
    return pd.to_datetime(s, errors="coerce")


# ------------------------------------------------------------------
# 1. Monthly archive - deep history
# ------------------------------------------------------------------
def fetch_price_and_demand(start_date, end_date, regions, output_dir):
    """
    PRICE_AND_DEMAND_<YYYYMM>_<REGION>.csv
    Columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE

    The current in-progress month IS published and grows through the month, so
    this covers history right up to today on its own.
    """
    os.makedirs(output_dir, exist_ok=True)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1

    session = requests.Session()
    session.headers.update(AEMO_CSV_HEADERS)
    written = {}

    for region in regions:
        frames, ok, bad, diagnosed = [], 0, 0, False
        for y, m in months:
            url = ("https://www.aemo.com.au/aemo/data/nem/priceanddemand/"
                   f"PRICE_AND_DEMAND_{y}{m:02d}_{region}.csv")
            try:
                r = session.get(url, timeout=30)
                if r.status_code != 200 or r.text.lstrip().lower().startswith(("<!doctype", "<html")):
                    bad += 1
                    if not diagnosed:
                        log.warning(f"[Archive] {region} {y}-{m:02d}: HTTP {r.status_code}, "
                                    f"body starts {r.text[:120]!r}")
                        diagnosed = True
                    continue
                frames.append(pd.read_csv(StringIO(r.text)))
                ok += 1
            except Exception as e:
                log.warning(f"[Archive] {region} {y}-{m:02d}: {e}")
                bad += 1
            time.sleep(0.2)

        if not frames:
            log.error(f"[Archive] {region}: nothing fetched.")
            continue

        df = pd.concat(frames, ignore_index=True)

        # Sort and dedupe on a temporary parsed column, then drop it, so the
        # original AEMO columns and the original "YYYY/MM/DD HH:MM:SS"
        # timestamp text survive untouched. Concatenating the monthly files is
        # the only change made here.
        df["_dt"] = parse_dt(df["SETTLEMENTDATE"])
        df = (df.dropna(subset=["_dt"])
                .drop_duplicates("_dt", keep="last")
                .sort_values("_dt"))
        first, last = df["_dt"].min(), df["_dt"].max()
        df = df.drop(columns="_dt")

        out_path = os.path.join(output_dir, f"price_and_demand_{region}.csv")
        df.to_csv(out_path, index=False)
        log.info(f"[Archive] {region}: {ok} months, {len(df):,} rows "
                 f"({first} -> {last}) | columns: {list(df.columns)}")
        written[region] = out_path

    return written


# ------------------------------------------------------------------
# 2. WA (SWIS) operational demand
# ------------------------------------------------------------------
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


def fetch_wa_demand(start_date, end_date, demand_dir):
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

        # Current-year annual CSV can lag. Fill its tail from WEM daily files.
        y2026 = modern[modern["SETTLEMENTDATE"].dt.year == 2026]
        tail_start = (y2026["SETTLEMENTDATE"].max().normalize()
                      if not y2026.empty else pd.Timestamp("2026-01-01"))
        tail_end = min(end.normalize(), pd.Timestamp.today().normalize())
        if tail_start <= tail_end:
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


# ------------------------------------------------------------------
# 3+4. Dashboard API -> dispatch/ and predispatch/  (RAW, per region)
# ------------------------------------------------------------------
# The dashboard's Price and Demand tab has two views, each backed by one
# timeScale on the same endpoint:
#     Dispatch      -> {"timeScale": ["5MIN"]}
#     Pre-Dispatch  -> {"timeScale": ["30MIN"]}
# Each response covers all five regions. Splitting it by REGIONID reproduces
# exactly what the dashboard exports when you pick a region and hit download.
# Nothing else is touched: no renamed columns, no derived columns, no merging,
# no dedupe. 5 regions x 2 views = 10 files.
DASHBOARD_VIEWS = {
    "dispatch": "5MIN",
    "predispatch": "30MIN",
}


def _dashboard_request(time_scale):
    """One POST returns every region's rows for this view."""
    r = requests.post(DASHBOARD_URL, headers=DASHBOARD_HEADERS,
                      data=json.dumps({"timeScale": [time_scale]}), timeout=60)
    r.raise_for_status()
    payload = r.json()

    # Top-level key mirrors the requested scale ("5MIN"/"30MIN"). Take the first
    # non-empty list rather than hard-coding, so a renamed key doesn't break it.
    for key, value in payload.items():
        if isinstance(value, list) and value:
            log.info(f"[Dashboard] {time_scale}: {len(value)} rows under '{key}' "
                     f"| columns: {list(value[0])}")
            return pd.DataFrame(value)
    raise ValueError(f"No data list in response. Keys: {list(payload)}")


def fetch_dispatch_and_predispatch(regions, dispatch_dir, predispatch_dir,
                                   snapshot_dir=None):
    """
    Writes one raw CSV per region per view, columns untouched and in the order
    AEMO returns them. Files are overwritten each run so each file is always a
    clean snapshot rather than an accumulation - if you want the earlier pulls
    kept, pass --archive-snapshots and timestamped copies go to snapshots/.
    """
    out_dirs = {"dispatch": dispatch_dir, "predispatch": predispatch_dir}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    pulled_at = datetime.now()

    for view, scale in DASHBOARD_VIEWS.items():
        try:
            df = _dashboard_request(scale)
        except Exception as e:
            log.error(f"[{view}] {scale} request failed: {e}")
            continue

        if "REGIONID" not in df.columns:
            path = os.path.join(out_dirs[view], f"{view}_ALL_REGIONS.csv")
            df.to_csv(path, index=False)
            log.error(f"[{view}] No REGIONID column - wrote {len(df)} rows "
                      f"unsplit to {path}. Columns: {list(df.columns)}")
            continue

        for region in regions:
            sub = df[df["REGIONID"] == region]
            if sub.empty:
                log.warning(f"[{view}] {region}: no rows returned")
                continue
            path = os.path.join(out_dirs[view], f"{view}_{region}.csv")
            sub.to_csv(path, index=False)

            if "PERIODTYPE" in sub.columns:
                breakdown = sub["PERIODTYPE"].value_counts().to_dict()
            else:
                breakdown = {}
            log.info(f"[{view}] {region}: {len(sub)} rows {breakdown} -> {path}")

            if snapshot_dir:
                os.makedirs(snapshot_dir, exist_ok=True)
                stamp = pulled_at.strftime("%Y%m%d_%H%M")
                sub.to_csv(os.path.join(snapshot_dir,
                                        f"{view}_{region}_{stamp}.csv"), index=False)

        time.sleep(0.5)

    return None


# ------------------------------------------------------------------
# 5. Weather  (Open-Meteo ERA5 archive)
# ------------------------------------------------------------------
# BOM Climate Data Online was tried twice and abandoned. It has no public API:
# the only route is scraping the station page for a "dailyZippedDataFile" link,
# and BOM blocks automated clients - the reply is an HTML page, or a ZIP that
# arrives empty, depending on the day. A pipeline that has to be re-run cannot
# depend on it.
#
# Open-Meteo's historical archive is used instead. It serves ERA5 reanalysis
# (ECMWF) as plain JSON, needs no API key or sign-up, covers 1940 to about five
# days ago, and is spatially complete - no missing stations, no outages. Data
# is CC BY 4.0, so it can be redistributed with attribution. ERA5 assimilates
# BOM's own observations, so these are not a different measurement of Australia
# so much as a gridded, gap-free version of one.
#
# Two files per region:
#   weather_daily_<REGION>.csv   max / min / mean temperature, rainfall
#   weather_hourly_<REGION>.csv  hourly temperature and humidity
# NT coordinates are configured below but NT is not in REGION_CONFIG in this
# file; add it there first if NT features are wanted.
# The hourly file matters here: demand is half-hourly, so a single daily
# maximum cannot explain an evening peak. Preprocessing can interpolate the
# hourly series onto the demand grid and build HDD/CDD from it.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Capital-city coordinates and the civil time zone each region runs on. Weather
# is requested on local civil time so it joins to a local date, matching how
# the holiday calendar is built.
WEATHER_SITES = {
    "NSW1": {"lat": -33.8688, "lon": 151.2093, "tz": "Australia/Sydney",    "city": "Sydney"},
    "QLD1": {"lat": -27.4698, "lon": 153.0251, "tz": "Australia/Brisbane",  "city": "Brisbane"},
    "VIC1": {"lat": -37.8136, "lon": 144.9631, "tz": "Australia/Melbourne", "city": "Melbourne"},
    "SA1":  {"lat": -34.9285, "lon": 138.6007, "tz": "Australia/Adelaide",  "city": "Adelaide"},
    "TAS1": {"lat": -42.8821, "lon": 147.3272, "tz": "Australia/Hobart",    "city": "Hobart"},
    "WA":   {"lat": -31.9523, "lon": 115.8613, "tz": "Australia/Perth",     "city": "Perth"},
}

DAILY_VARS = ["temperature_2m_max", "temperature_2m_min",
              "temperature_2m_mean", "precipitation_sum"]
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m"]

# ERA5 is a reanalysis, so the archive trails real time by roughly five days.
# Asking beyond that returns nulls rather than an error, which would look like
# missing data instead of a known boundary.
ERA5_LAG_DAYS = 5


def _open_meteo_block(payload, key, expected):
    """Turn one Open-Meteo JSON block into a DataFrame.

    The API returns parallel arrays - {"time": [...], "temperature_2m": [...]}
    - rather than records, so the block is validated for equal lengths before
    being zipped into rows.
    """
    block = payload.get(key)
    if not block or "time" not in block:
        raise ValueError(f"no '{key}' block in response (keys: {list(payload)})")
    n = len(block["time"])
    missing = [v for v in expected if v not in block]
    if missing:
        raise ValueError(f"{key} block missing {missing}")
    bad = [v for v in expected if len(block[v]) != n]
    if bad:
        raise ValueError(f"{key} arrays out of step with time: {bad}")
    return pd.DataFrame({"time": block["time"], **{v: block[v] for v in expected}})


def fetch_weather(regions, output_dir, start, end, hourly=True):
    """Daily and hourly weather per region from the Open-Meteo ERA5 archive.

    Returns a dict of written paths. A region that fails is logged and skipped
    rather than written as a partial file.
    """
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})

    # Clip the request to what ERA5 can actually have, and say so once.
    cutoff = (pd.Timestamp.today().normalize()
              - pd.Timedelta(days=ERA5_LAG_DAYS)).date()
    end_req = min(pd.Timestamp(end).date(), cutoff)
    if end_req < pd.Timestamp(end).date():
        log.warning(f"[Weather] ERA5 trails real time by ~{ERA5_LAG_DAYS} days - "
                    f"requesting to {end_req} instead of {pd.Timestamp(end).date()}")

    written = {}
    for region in regions:
        site = WEATHER_SITES.get(region)
        if not site:
            log.warning(f"[Weather] {region}: no coordinates configured - skipped")
            continue

        params = {"latitude": site["lat"], "longitude": site["lon"],
                  "start_date": str(pd.Timestamp(start).date()),
                  "end_date": str(end_req),
                  "timezone": site["tz"], "daily": ",".join(DAILY_VARS)}
        if hourly:
            params["hourly"] = ",".join(HOURLY_VARS)

        try:
            reply = session.get(OPEN_METEO_ARCHIVE, params=params, timeout=120)
            reply.raise_for_status()
            payload = reply.json()
            if "error" in payload:
                raise ValueError(payload.get("reason", "API reported an error"))

            # --- daily ---
            day = _open_meteo_block(payload, "daily", DAILY_VARS)
            day = day.rename(columns={"time": "DATE",
                                      "temperature_2m_max": "temp_max_c",
                                      "temperature_2m_min": "temp_min_c",
                                      "temperature_2m_mean": "temp_mean_c",
                                      "precipitation_sum": "rainfall_mm"})
            day["DATE"] = pd.to_datetime(day["DATE"], errors="coerce")
            day.insert(1, "REGION", region)
            day["LOCATION"] = site["city"]
            day["SOURCE"] = "open-meteo ERA5"

            if day["temp_max_c"].notna().sum() == 0:
                raise ValueError("daily block returned no temperature values")

            path = os.path.join(output_dir, f"weather_daily_{region}.csv")
            day.to_csv(path, index=False, lineterminator="\n")
            written[f"{region}_daily"] = path
            log.info(f"[Weather] {region} ({site['city']}): {len(day):,} days "
                     f"{day['DATE'].min():%Y-%m-%d} -> {day['DATE'].max():%Y-%m-%d}, "
                     f"{int(day['temp_max_c'].isna().sum())} gap(s)")

            # --- hourly ---
            if hourly:
                hr = _open_meteo_block(payload, "hourly", HOURLY_VARS)
                hr = hr.rename(columns={"time": "TIMESTAMP",
                                        "temperature_2m": "temp_c",
                                        "relative_humidity_2m": "humidity_pct"})
                hr["TIMESTAMP"] = pd.to_datetime(hr["TIMESTAMP"], errors="coerce")
                hr.insert(1, "REGION", region)
                hr["SOURCE"] = "open-meteo ERA5"
                path = os.path.join(output_dir, f"weather_hourly_{region}.csv")
                hr.to_csv(path, index=False, lineterminator="\n")
                written[f"{region}_hourly"] = path
                log.info(f"[Weather] {region}: {len(hr):,} hourly readings")

        except Exception as exc:
            log.error(f"[Weather] {region} failed: {exc}")
        time.sleep(1)          # courtesy pause; the free tier allows plenty

    expected = len(regions) * (2 if hourly else 1)
    log.info(f"[Weather] {len(written)}/{expected} file(s) saved")
    return written


# ------------------------------------------------------------------
# 6. Public holidays
# ------------------------------------------------------------------
def holiday_calendar_end(end_date):
    """Last date the holiday calendar should reach: 31 December of the year the
    run falls in.

    Demand and weather stop at whatever the providers have published, but
    holidays are statutory and known years in advance. Stopping the calendar on
    the run date would leave the rest of the year with no holidays at all - and
    the remaining months hold Labour Day, Melbourne Cup, Christmas and Boxing
    Day, which are among the largest single-day effects on electricity demand.
    Downstream, a missing holiday does not look like missing data: it looks
    like an ordinary weekday, so nothing flags it.
    """
    return f"{pd.Timestamp(end_date).year}-12-31"


def fetch_public_holidays(start_date, end_date, regions, output_dir,
                          holiday_end=None):
    """State holiday calendars from start_date to the end of the current year.

    `end_date` is the demand cutoff and is used only to work out which year the
    run belongs to; the calendar itself always extends to 31 December so the
    forecast horizon is fully covered.
    """
    try:
        import holidays
    except ImportError:
        log.error("holidays not installed. Run: pip install holidays")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    holiday_end = holiday_end or holiday_calendar_end(end_date)
    years = list(range(int(str(start_date)[:4]), int(str(holiday_end)[:4]) + 1))
    dates = pd.date_range(start_date, holiday_end, freq="D")
    written = {}

    log.info(f"[Holidays] calendar covers {dates.min().date()} to "
             f"{dates.max().date()} (demand cutoff was {pd.Timestamp(end_date).date()})")

    for region in regions:
        state = REGION_CONFIG[region]["state"]
        cal = holidays.Australia(subdiv=state, years=years)
        df = pd.DataFrame({"date": dates})
        df["state"] = state
        df["is_public_holiday"] = df["date"].dt.date.isin(cal).astype(int)
        df["holiday_name"] = df["date"].dt.date.map(lambda d: cal.get(d, ""))
        path = os.path.join(output_dir, f"public_holidays_{region}_{state}.csv")
        df.to_csv(path, index=False)

        future = df[(df["date"] > pd.Timestamp(end_date)) &
                    (df["is_public_holiday"] == 1)]
        log.info(f"[Holidays] {region} ({state}): {int(df.is_public_holiday.sum())} days "
                 f"total, {len(future)} after the demand cutoff"
                 + (f" ({', '.join(future['holiday_name'].tolist())})" if len(future) else ""))
        written[region] = path

    return written


# ------------------------------------------------------------------
# Notes file
# ------------------------------------------------------------------
DATA_NOTES = """# Data notes

## Folders

| Folder | What it contains |
|---|---|
| `price_and_demand/` | NEM demand files plus WA/SWIS operational demand. |
| `energy_requirement/` | Raw AEMO `MarketRequirements-2026.csv`. |
| `dispatch/` | Current NEM 5-minute dispatch snapshots. |
| `predispatch/` | Current NEM 30-minute pre-dispatch snapshots. |
| `weather/` | Daily and hourly weather per region from the Open-Meteo ERA5 archive. |
| `public_holidays/` | State public-holiday calendars including WA. |

## WA demand frequency

`WA_demand_2022_2026.csv` preserves AEMO's native source frequency:

- 1 Jan 2022 to 30 Sep 2023: 30-minute WEM operational-demand observations.
- From 1 Oct 2023: 5-minute WEM operational-demand observations.

No 30-minute WA observations are interpolated or converted to 5-minute data in
this acquisition script. `SOURCE_INTERVAL_MIN` records whether each row came
from a 30-minute or 5-minute source.

`energy_requirement/MarketRequirements-2026.csv` is downloaded separately and
kept as AEMO publishes it for later forecasting work.

WA weather uses BOM Perth Airport station 009021. Weather and public-holiday
files follow local state dates. Time-zone alignment belongs in preprocessing,
not acquisition.
"""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    today = datetime.today().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="PRT661 data acquisition (v7)")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=today)
    p.add_argument("--regions", nargs="+", default=NEM_REGIONS)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--archive-snapshots", action="store_true",
                   help="Also keep a timestamped copy of each pull in snapshots/")
    p.add_argument("--skip-archive", action="store_true")
    p.add_argument("--skip-wa", action="store_true")
    p.add_argument("--skip-dashboard", action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--skip-holidays", action="store_true")
    args = p.parse_args()

    missing = [m for m in ("requests", "pandas", "holidays")
               if importlib.util.find_spec(m) is None]
    if missing:
        log.error(f"Missing: {', '.join(missing)}. Run: pip install {' '.join(missing)}")
        return

    out = Path(args.output_dir or prompt_for_output_dir())
    dirs = {
        "archive": out / "price_and_demand",
        "requirements": out / "energy_requirement",
        "dispatch": out / "dispatch",
        "predispatch": out / "predispatch",
        "snapshots": out / "snapshots",
        "weather": out / "weather",
        "holidays": out / "public_holidays",
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "DATA_NOTES.md").write_text(DATA_NOTES, encoding="utf-8")

    log.info("=" * 64)
    log.info(f"Regions: {args.regions} | Archive period: {args.start} -> {args.end}")
    log.info(f"Output: {out.resolve()}")
    log.info("=" * 64)

    feature_regions = list(dict.fromkeys(args.regions + ["WA"]))

    if not args.skip_archive:
        log.info("1/6 Monthly NEM archive (deep history)")
        fetch_price_and_demand(args.start, args.end, args.regions, str(dirs["archive"]))

    if not args.skip_wa:
        log.info("2/6 WA operational demand (native source intervals)")
        fetch_wa_demand(args.start, args.end, str(dirs["archive"]))

        log.info("3/6 WA 2026 market requirements")
        fetch_wa_market_requirements_2026(str(dirs["requirements"]))

    if not args.skip_dashboard:
        log.info("4/6 Dashboard API -> dispatch/ and predispatch/")
        fetch_dispatch_and_predispatch(
            args.regions, str(dirs["dispatch"]), str(dirs["predispatch"]),
            str(dirs["snapshots"]) if args.archive_snapshots else None)

    if not args.skip_weather:
        log.info("5/6 Weather - Open-Meteo ERA5 (including WA)")
        fetch_weather(feature_regions, str(dirs["weather"]), args.start, args.end)

    if not args.skip_holidays:
        log.info("6/6 Public holidays (through the end of the current year)")
        fetch_public_holidays(args.start, args.end, feature_regions,
                              str(dirs["holidays"]))

    log.info("=" * 64)
    log.info(f"Done -> {out.resolve()}")
    log.info("Read DATA_NOTES.md before joining the demand series.")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
