"""
PRT661 – Data Science Practice
WP1 Stage 1 contribution module – Suraj Raut (S391201)
Primary scope: AEMO NEM historical monthly archive acquisition.

This role-aligned module reproduces the NEM-history part of the final integrated
Stage 1 pipeline. It intentionally preserves provider columns and timestamp text;
analytical harmonisation belongs to Stage 2.
"""
from __future__ import annotations
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

log = logging.getLogger("suraj_nem_archive")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
AEMO_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data",
}
NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
DT_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def parse_dt(series: pd.Series) -> pd.Series:
    """Parse timestamps only for validation/sorting; raw timestamp text is preserved on output."""
    s = series.astype(str).str.strip().str.strip('"')
    for fmt in DT_FORMATS:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().any():
            return out
    return pd.to_datetime(s, errors="coerce")


def month_sequence(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_price_and_demand(start_date: str, end_date: str, regions, output_dir: str):
    """Download and consolidate AEMO PRICE_AND_DEMAND monthly CSVs per NEM region."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(AEMO_HEADERS)
    written = {}

    for region in regions:
        frames, successful_months = [], 0
        for year, month in month_sequence(start_date, end_date):
            name = f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"
            url = f"https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/{name}"
            try:
                reply = session.get(url, timeout=60)
                reply.raise_for_status()
                head = reply.content.lstrip().lower()[:80]
                if head.startswith((b"<!doctype", b"<html")):
                    raise ValueError("AEMO returned HTML instead of a CSV")
                frame = pd.read_csv(pd.io.common.BytesIO(reply.content))
                if "SETTLEMENTDATE" not in frame.columns:
                    raise ValueError(f"SETTLEMENTDATE absent; columns={list(frame.columns)}")
                frames.append(frame)
                successful_months += 1
            except Exception as exc:
                log.warning("%s %04d-%02d skipped: %s", region, year, month, exc)
            time.sleep(0.05)

        if not frames:
            log.error("%s: no archive months were retrieved", region)
            continue

        df = pd.concat(frames, ignore_index=True)
        df["_dt"] = parse_dt(df["SETTLEMENTDATE"])
        df = (df.dropna(subset=["_dt"])
                .drop_duplicates("_dt", keep="last")
                .sort_values("_dt"))
        first, last = df["_dt"].min(), df["_dt"].max()
        df = df.drop(columns="_dt")
        path = out / f"price_and_demand_{region}.csv"
        df.to_csv(path, index=False)
        log.info("%s: %s months, %s rows, %s -> %s", region, successful_months,
                 f"{len(df):,}", first, last)
        written[region] = str(path)
    return written


def main():
    p = argparse.ArgumentParser(description="Suraj WP1 – NEM historical archive acquisition")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--regions", nargs="+", default=NEM_REGIONS)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    fetch_price_and_demand(a.start, a.end, a.regions, a.output_dir)


if __name__ == "__main__":
    main()
