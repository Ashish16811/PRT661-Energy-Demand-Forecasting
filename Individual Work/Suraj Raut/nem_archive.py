"""
nem_archive.py - AEMO NEM monthly archive
PRT661 WP1 Data Acquisition · Dan6: Theme 2
Module owner: Suraj Raut (Data Engineering Lead)

Retrieves the deep historical demand series for the five NEM regions from
AEMO's monthly aggregated price-and-demand archive, one output file per region.

Design points established during WP1:
  · A single requests.Session() is reused across all monthly requests rather
    than opening a connection per file - this is the bulk-download path and
    there are roughly 57 months x 5 regions of them.
  · Responses are checked for an HTML error page before parsing. AEMO returns
    HTTP 200 with an HTML body for a missing month, so parsing without the
    check yields a confusing pandas error far from the cause.
  · Provider column names and raw timestamp text are preserved exactly as
    published. Normalisation belongs in WP2, not in acquisition.
  · Verified on 23 August: the archive reaches the current date on its own, so
    dispatch does not need to be stitched onto the end of the history
    (23 August minutes, section 2.1).
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

from common import AEMO_CSV_HEADERS, NEM_REGIONS, parse_dt


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

