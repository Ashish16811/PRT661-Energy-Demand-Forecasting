"""
nem_dispatch.py - AEMO 5-minute dashboard: dispatch and pre-dispatch
PRT661 WP1 Data Acquisition · Dan6: Theme 2
Module owner: Ashish Shrestha (Forecasting Lead)

Captures the AEMO visualisation dashboard's dispatch (actual) and pre-dispatch
(forecast) views, writing ten raw files: five regions x two views.

Design points established during WP1:
  · Two dashboard requests replace the hundreds of individual NEMWEB file
    downloads the first draft was making (23 August minutes, section 2.2).
  · Dispatch and pre-dispatch are written to SEPARATE folders and never merged.
    Pre-dispatch is a forecast produced before the event; merging it into the
    observed series would be a leakage source in every downstream model.
  · PERIODTYPE is preserved and the ACTUAL/FORECAST composition is verified on
    each pull, because both views return both row types.
  · This step runs FIRST in the integrated pipeline. The dashboard serves a
    short rolling window, so a missed snapshot is gone, whereas the archives
    are static and can be retried at any time.
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

from common import DASHBOARD_URL, DASHBOARD_HEADERS, NEM_REGIONS, parse_dt


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

