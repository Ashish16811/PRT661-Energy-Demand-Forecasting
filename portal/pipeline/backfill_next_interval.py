from __future__ import annotations

import logging
import os
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from .database import init_db, read_df, upsert_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_next_interval")
PORTAL_TIMEZONE = os.getenv("PORTAL_TIMEZONE", "Australia/Adelaide")


def _utc_to_portal_wall(ts) -> pd.Timestamp:
    x = pd.Timestamp(ts)
    if x.tzinfo is None:
        x = x.tz_localize("UTC")
    else:
        x = x.tz_convert("UTC")
    return x.tz_convert(ZoneInfo(PORTAL_TIMEZONE)).tz_localize(None)


def backfill(month: str | None = None) -> pd.DataFrame:
    """Recover genuine next-interval points from forecast vintages already stored.

    This does NOT invent predictions. For each historical forecast run, it uses the
    forecast row for the first half-hour interval after the run actually occurred.
    It is useful when the live portal already stored multi-horizon vintages before
    the dedicated next_interval_forecast table was introduced.
    """
    params = {}
    where = "WHERE created_at IS NOT NULL"
    if month:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)
        # created_at is UTC in the existing table; query a generous UTC window and
        # perform the exact portal-month check after conversion.
        where += " AND created_at >= :utc_start AND created_at < :utc_end"
        params["utc_start"] = (start - pd.Timedelta(days=1)).to_pydatetime()
        params["utc_end"] = (end + pd.Timedelta(days=1)).to_pydatetime()

    d = read_df(
        f"""
        SELECT region, model_name, model_version, forecast_origin,
               target_timestamp, forecast_mw, created_at
        FROM project_forecast
        {where}
        ORDER BY created_at, region, model_name, target_timestamp
        """,
        params,
    )
    if d.empty:
        return d

    for c in ("target_timestamp", "created_at"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["forecast_mw"] = pd.to_numeric(d["forecast_mw"], errors="coerce")
    d = d.dropna(subset=["target_timestamp", "created_at", "forecast_mw"])
    if d.empty:
        return d

    rows = []
    group_cols = ["region", "model_name", "model_version", "created_at"]
    for keys, g in d.groupby(group_cols, sort=True):
        region, model_name, model_version, created_at = keys
        portal_origin = _utc_to_portal_wall(created_at)
        if month and portal_origin.strftime("%Y-%m") != month:
            continue
        requested = portal_origin.floor("30min") + pd.Timedelta(minutes=30)
        candidates = g[g["target_timestamp"] >= requested].sort_values("target_timestamp")
        if candidates.empty:
            continue
        r = candidates.iloc[0]
        target = pd.Timestamp(r["target_timestamp"])
        # Only treat this as a genuine one-step point when it is no more than one
        # 30-minute grid interval beyond the requested next timestamp.
        if target - requested > pd.Timedelta(minutes=30):
            continue
        rows.append({
            "region": region,
            "model_name": model_name,
            "model_version": model_version,
            "forecast_origin": portal_origin.floor("min"),
            "target_timestamp": target,
            "forecast_mw": float(r["forecast_mw"]),
            "lead_minutes": max(0.0, (target - portal_origin).total_seconds() / 60.0),
            "created_at": pd.Timestamp(created_at),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(
        ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
        keep="last",
    )
    upsert_frame(
        "next_interval_forecast",
        out,
        ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
    )
    log.info("Recovered %s genuine next-interval forecasts%s", f"{len(out):,}", f" for {month}" if month else "")
    return out


def main():
    init_db()
    month = sys.argv[1] if len(sys.argv) > 1 else None
    out = backfill(month)
    print(f"Next-interval backfill complete; stored {len(out):,} rows.")


if __name__ == "__main__":
    main()
