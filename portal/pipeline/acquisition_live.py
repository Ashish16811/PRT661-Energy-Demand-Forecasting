from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (NEM_REGIONS, OPEN_METEO_FORECAST, RAW_DIR, REGION_COORDS,
                     REGION_TZ, REGIONS, WEATHER_FORECAST_DAYS)
from .database import upsert_frame, utcnow_naive
from .loader import acquisition_module

log = logging.getLogger("live_acquisition")
UA = "PRT661-Energy-Forecasting-Portal/1.0"

# A failed AEMO dashboard request must never make an old local CSV look freshly
# retrieved. A recent snapshot can bridge a short provider outage; older cache
# is rejected and the database keeps its previously stored history unchanged.
PREDISPATCH_CACHE_MAX_AGE_MINUTES = int(
    os.getenv("PREDISPATCH_CACHE_MAX_AGE_MINUTES", "90")
)


def _merge_csv(path: Path, new: pd.DataFrame, time_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path)
        both = pd.concat([old, new], ignore_index=True)
    else:
        both = new.copy()
    both[time_col] = pd.to_datetime(both[time_col], errors="coerce")
    both = (both.dropna(subset=[time_col])
            .drop_duplicates(time_col, keep="last")
            .sort_values(time_col))
    both.to_csv(path, index=False, date_format="%Y/%m/%d %H:%M:%S")
    return both


def refresh_nem_current_month(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Refresh the current AEMO monthly archive when available.

    The monthly PRICE_AND_DEMAND file is a catch-up/history source. At the
    beginning of a new month AEMO may not have published the monthly CSV yet.
    A missing/unavailable monthly file must therefore not stop the live
    pipeline: recent NEM actuals are refreshed later from the current
    pre-dispatch snapshot.
    """
    now = pd.Timestamp(now or pd.Timestamp.now())
    ym = now.strftime("%Y%m")
    out = []
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "text/csv,text/plain,*/*"})

    for region in NEM_REGIONS:
        url = (
            "https://www.aemo.com.au/aemo/data/nem/priceanddemand/"
            f"PRICE_AND_DEMAND_{ym}_{region}.csv"
        )

        try:
            r = session.get(url, timeout=30)

            # AEMO's new-month aggregate can legitimately be absent for a while.
            # Treat this source as optional because the current pre-dispatch feed
            # later refreshes recent ACTUAL rows.
            if r.status_code == 404:
                log.warning(
                    "%s: AEMO monthly PRICE_AND_DEMAND file for %s is not yet "
                    "available; continuing with current/pre-dispatch data.",
                    region,
                    ym,
                )
                continue

            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except requests.RequestException as exc:
            log.warning(
                "%s: AEMO monthly PRICE_AND_DEMAND refresh failed for %s (%s); "
                "continuing with current/pre-dispatch data.",
                region,
                ym,
                exc,
            )
            continue
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            log.warning(
                "%s: could not parse AEMO monthly PRICE_AND_DEMAND file for %s "
                "(%s); continuing with current/pre-dispatch data.",
                region,
                ym,
                exc,
            )
            continue

        if not {"SETTLEMENTDATE", "TOTALDEMAND"} <= set(df.columns):
            log.warning(
                "%s: unexpected current-month columns %s; continuing with "
                "current/pre-dispatch data.",
                region,
                list(df.columns),
            )
            continue

        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
        df["TOTALDEMAND"] = pd.to_numeric(df["TOTALDEMAND"], errors="coerce")
        df = df.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])

        if df.empty:
            log.warning(
                "%s: AEMO monthly PRICE_AND_DEMAND file for %s contained no "
                "usable demand rows.",
                region,
                ym,
            )
            continue

        path = RAW_DIR / "price_and_demand" / f"price_and_demand_{region}.csv"
        _merge_csv(path, df, "SETTLEMENTDATE")

        out.append(pd.DataFrame({
            "region": region,
            "timestamp": df["SETTLEMENTDATE"],
            "demand_mw": df["TOTALDEMAND"],
            "source": "AEMO_PRICE_AND_DEMAND",
            "retrieved_at": utcnow_naive(),
        }))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def refresh_wa_tail(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fetch only recent WEM daily operational-demand rows; annual CSV is fallback."""
    now = pd.Timestamp(now or pd.Timestamp.now())
    acq = acquisition_module()
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "*/*"})
    start = (now.normalize() - pd.Timedelta(days=2))
    tail = acq._fetch_wa_daily_tail(session, start, now.normalize())
    if tail is None or tail.empty:
        year = now.year
        url = f"{acq.WA_NEW_BASE}/OperationalDemandWithdrawal-{year}.csv"
        tail = acq._download_wa_csv(session, url, f"AEMO_WEM_{year}", 5)
        tail = tail[tail["SETTLEMENTDATE"] >= start]
    if tail.empty:
        return pd.DataFrame()
    raw_path = RAW_DIR / "price_and_demand" / "WA_demand_2022_2026.csv"
    if raw_path.exists() and raw_path.stat().st_size:
        old = pd.read_csv(raw_path)
        old["SETTLEMENTDATE"] = pd.to_datetime(old["SETTLEMENTDATE"], errors="coerce")
        merged = pd.concat([old, tail], ignore_index=True)
    else:
        merged = tail.copy()
    merged["SETTLEMENTDATE"] = pd.to_datetime(merged["SETTLEMENTDATE"], errors="coerce")
    merged = (merged.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
              .drop_duplicates("SETTLEMENTDATE", keep="last")
              .sort_values("SETTLEMENTDATE"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(raw_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    # Production model operates on a 30-minute grid. Modern WA arrives at 5 minutes,
    # so the database stores the same mean-downsampled 30-minute power level used
    # by Stage 2 (never a sum/energy conversion).
    live = pd.DataFrame({
        "timestamp": pd.to_datetime(tail["SETTLEMENTDATE"]),
        "demand_mw": pd.to_numeric(tail["TOTALDEMAND"], errors="coerce"),
    }).dropna().set_index("timestamp").sort_index()
    live = live["demand_mw"].resample("30min", closed="left", label="left").mean().dropna().reset_index()
    live.insert(0, "region", "WA")
    live["source"] = "AEMO_WEM_30MIN_MEAN"
    live["retrieved_at"] = utcnow_naive()
    return live


def refresh_weather_forecast(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Current 7-day weather for Model B; archived by forecast origin."""
    now = pd.Timestamp(now or pd.Timestamp.now())
    frames = []
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})
    for region in REGIONS:
        lat, lon = REGION_COORDS[region]
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "forecast_days": WEATHER_FORECAST_DAYS,
            "timezone": REGION_TZ[region],
        }
        r = session.get(OPEN_METEO_FORECAST, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json().get("hourly", {})
        if "time" not in payload:
            continue
        f = pd.DataFrame({
            "target_timestamp": pd.to_datetime(payload["time"], errors="coerce"),
            "temp_c": pd.to_numeric(payload.get("temperature_2m"), errors="coerce"),
            "humidity_pct": pd.to_numeric(payload.get("relative_humidity_2m"), errors="coerce"),
            "rainfall_mm": pd.to_numeric(payload.get("precipitation"), errors="coerce"),
        }).dropna(subset=["target_timestamp"])
        # Model grid is half-hourly. Interpolate temperature/humidity; rainfall is carried hourly.
        f = f.set_index("target_timestamp").sort_index()
        grid = pd.date_range(f.index.min(), f.index.max(), freq="30min")
        state = f[["temp_c", "humidity_pct"]].reindex(grid).interpolate("time")
        rain = f["rainfall_mm"].reindex(grid).ffill(limit=1)
        out = state.assign(rainfall_mm=rain).reset_index(names="target_timestamp")
        out.insert(0, "region", region)
        out["forecast_origin"] = now.floor("h")
        out["source"] = "OPEN_METEO_FORECAST"
        out["retrieved_at"] = utcnow_naive()
        frames.append(out)

        # Also write the latest files in the shape Stage 2 expects on monthly retraining.
        latest = RAW_DIR / "weather" / "forecast" / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        export = out.rename(columns={
            "target_timestamp": "TIMESTAMP", "temp_c": "temp_c",
            "humidity_pct": "humidity_pct", "rainfall_mm": "rainfall_mm"})
        export["SOURCE_TYPE"] = "FORECAST_7DAY"
        export["SOURCE"] = "Open-Meteo"
        export["MODEL"] = "forecast-api"
        export["FORECAST_ORIGIN_DATE"] = now.normalize()
        export["LEAD_DAY"] = (pd.to_datetime(export["TIMESTAMP"]).dt.normalize() - now.normalize()).dt.days
        export["RETRIEVED_AT_UTC"] = utcnow_naive().isoformat()
        export["REGION"] = region
        export[["TIMESTAMP", "REGION", "temp_c", "humidity_pct", "rainfall_mm", "SOURCE_TYPE",
                "SOURCE", "MODEL", "FORECAST_ORIGIN_DATE", "LEAD_DAY", "RETRIEVED_AT_UTC"]].to_csv(
            latest / f"weather_forecast_hourly_{region}.csv", index=False)
        daily = (export.assign(DATE=pd.to_datetime(export["TIMESTAMP"]).dt.normalize())
                 .groupby("DATE", as_index=False)
                 .agg(temp_max_c=("temp_c", "max"), temp_min_c=("temp_c", "min"),
                      temp_mean_c=("temp_c", "mean"), rainfall_mm=("rainfall_mm", "sum")))
        daily["LEAD_DAY"] = (daily["DATE"] - now.normalize()).dt.days
        daily.to_csv(latest / f"weather_forecast_daily_{region}.csv", index=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refresh_aemo_predispatch_snapshot(now: pd.Timestamp | None = None) -> int:
    """Refresh AEMO pre-dispatch and safely handle short provider outages.

    Fresh files are used normally. If the AEMO dashboard request fails, an
    existing per-region CSV may be used only while its file age is within
    ``PREDISPATCH_CACHE_MAX_AGE_MINUTES``. Cached rows are explicitly labelled
    as CACHE and retain the cache retrieval time; stale files are skipped.

    This prevents a 502/503/504 or other temporary provider failure from
    silently making an old snapshot appear current in the public portal.
    """
    now = pd.Timestamp(now or pd.Timestamp.now())
    acq = acquisition_module()
    out_dir = RAW_DIR / "predispatch"
    out_dir.mkdir(parents=True, exist_ok=True)

    before_mtime = {}
    for region in NEM_REGIONS:
        path = out_dir / f"predispatch_{region}.csv"
        before_mtime[region] = path.stat().st_mtime if path.exists() else None

    fetch_status = acq.fetch_predispatch(
        NEM_REGIONS,
        str(out_dir),
        str(RAW_DIR / "snapshots"),
    )

    # New source_pipelines/data_acquisition.py returns this metadata. The mtime
    # fallback keeps this wrapper compatible with an older acquisition module.
    if isinstance(fetch_status, dict):
        fresh_regions = set(fetch_status.get("regions_written") or [])
        fetch_error = fetch_status.get("error")
    else:
        fresh_regions = set()
        fetch_error = None
        for region in NEM_REGIONS:
            path = out_dir / f"predispatch_{region}.csv"
            if not path.exists():
                continue
            old = before_mtime.get(region)
            if old is None or path.stat().st_mtime > old:
                fresh_regions.add(region)

    forecast_frames = []
    actual_frames = []
    fresh_used = []
    cache_used = []
    stale_skipped = []

    for region in NEM_REGIONS:
        path = out_dir / f"predispatch_{region}.csv"
        if not path.exists() or not path.stat().st_size:
            log.warning("%s: no predispatch snapshot is available", region)
            continue

        is_fresh = region in fresh_regions
        file_mtime_utc = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_localize(None)
        cache_age_minutes = max(
            0.0,
            (pd.Timestamp(utcnow_naive()) - file_mtime_utc).total_seconds() / 60.0,
        )

        if not is_fresh and cache_age_minutes > PREDISPATCH_CACHE_MAX_AGE_MINUTES:
            stale_skipped.append(region)
            log.warning(
                "%s: cached predispatch snapshot is %.0f minutes old "
                "(limit=%s); stale cache rejected.",
                region,
                cache_age_minutes,
                PREDISPATCH_CACHE_MAX_AGE_MINUTES,
            )
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log.warning("%s: could not read predispatch snapshot: %s", region, exc)
            continue

        required = {"SETTLEMENTDATE", "TOTALDEMAND"}
        if not required <= set(df.columns):
            log.warning("%s: unexpected predispatch columns %s", region, list(df.columns))
            continue

        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
        df["TOTALDEMAND"] = pd.to_numeric(df["TOTALDEMAND"], errors="coerce")
        df = df.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
        if df.empty:
            continue

        if "PERIODTYPE" in df.columns:
            actual = df[df["PERIODTYPE"].astype(str).str.upper().eq("ACTUAL")].copy()
            forecast = df[df["PERIODTYPE"].astype(str).str.upper().eq("FORECAST")].copy()
        else:
            actual = pd.DataFrame(columns=df.columns)
            forecast = df.copy()

        if is_fresh:
            source_actual = "AEMO_PREDISPATCH_ACTUAL_30MIN"
            source_forecast = "AEMO_PREDISPATCH_30MIN"
            retrieved_at = utcnow_naive()
            origin = now.floor("30min")
            fresh_used.append(region)
        else:
            source_actual = "AEMO_PREDISPATCH_ACTUAL_30MIN_CACHE"
            source_forecast = "AEMO_PREDISPATCH_30MIN_CACHE"
            retrieved_at = file_mtime_utc.to_pydatetime()
            # The exact provider forecast origin is not present in the CSV.
            # File modification time is the best conservative cache provenance.
            origin = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime)).floor("30min")
            cache_used.append(region)
            log.warning(
                "%s: AEMO refresh unavailable; using %.0f-minute-old cached "
                "predispatch snapshot.",
                region,
                cache_age_minutes,
            )

        if not actual.empty:
            actual_frames.append(pd.DataFrame({
                "region": region,
                "timestamp": actual["SETTLEMENTDATE"],
                "demand_mw": actual["TOTALDEMAND"],
                "source": source_actual,
                "retrieved_at": retrieved_at,
            }))

        if not forecast.empty:
            forecast_frames.append(pd.DataFrame({
                "region": region,
                "forecast_origin": origin,
                "target_timestamp": forecast["SETTLEMENTDATE"],
                "poe10_mw": np.nan,
                "poe50_mw": forecast["TOTALDEMAND"],
                "poe90_mw": np.nan,
                "source": source_forecast,
                "retrieved_at": retrieved_at,
            }))

    if fetch_error:
        log.warning("AEMO predispatch provider status: %s", fetch_error)

    log.info(
        "AEMO predispatch source state: fresh=%s cache=%s stale_skipped=%s",
        fresh_used,
        cache_used,
        stale_skipped,
    )

    if actual_frames:
        actual_frame = pd.concat(actual_frames, ignore_index=True)
        n_actual = upsert_frame("actual_demand", actual_frame, ["region", "timestamp"])
        log.info("AEMO predispatch ACTUAL refresh: %s 30-minute rows", f"{n_actual:,}")

    if not forecast_frames:
        return 0

    frame = pd.concat(forecast_frames, ignore_index=True)
    return upsert_frame(
        "aemo_forecast",
        frame,
        ["region", "forecast_origin", "target_timestamp"],
    )


def run_hourly_acquisition(now: pd.Timestamp | None = None) -> dict:
    actual = pd.concat([refresh_nem_current_month(now), refresh_wa_tail(now)], ignore_index=True)
    n_actual = upsert_frame("actual_demand", actual, ["region", "timestamp"])
    weather = refresh_weather_forecast(now)
    n_weather = upsert_frame("weather_forecast", weather,
                             ["region", "forecast_origin", "target_timestamp"])
    benchmark_notes = []
    try:
        n_predispatch = refresh_aemo_predispatch_snapshot(now)
        benchmark_notes.append(f"predispatch_forecast_rows={n_predispatch}")
    except Exception as exc:
        benchmark_notes.append(f"predispatch_warning={exc}")
        log.warning("AEMO predispatch snapshot failed: %s", exc)
    return {"actual_rows": n_actual, "weather_rows": n_weather,
            "benchmark": "; ".join(benchmark_notes)}
