"""
features_validation.py - BOM weather, public holidays, output validation
PRT661 WP1 Data Acquisition · Dan6: Theme 2
Module owner: Bishal Dahal (Verification & Dashboard Lead)

Collects the driver variables and verifies what the whole pipeline wrote.

  fetch_bom_weather()      daily max/min temperature and rainfall per region
  fetch_public_holidays()  state-specific calendar per region
  build_validation_report() row counts, coverage, gaps and duplicates per file

Design points established during WP1:
  · Weather and holidays are collected for FEATURE_REGIONS, which includes NT
    even though NT demand is unavailable. The files are small, they cost one
    entry in the region configuration, and if NT demand is ever obtained the
    drivers are already in place rather than requiring a full re-run
    (23 August minutes, section 3.4).
  · State-specific holiday files, not one national calendar. Labour Day falls
    in March in VIC/TAS, May in QLD, June in WA and October in NSW/SA.
  · BOM retrieval validates content BEFORE writing. See the module docstring
    on fetch_bom_weather for the root-cause analysis of the earlier failure.
  · build_validation_report() is the repeatable form of the manual
    region-by-region cross-check performed against the provider websites on
    23 August (section 5), so that acceptance check is reproducible rather
    than a one-off screen-shared inspection.
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

from common import BOM_HEADERS, REGION_CONFIG, FEATURE_REGIONS, parse_dt


def _bom_extract_csv(content: bytes):
    """BOM's dailyZippedDataFile endpoint returns a ZIP archive. Return the
    bytes of the observation CSV inside it, or None if this is not a ZIP."""
    if content[:2] != b"PK":
        return None
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return None
        # BOM ships the data file plus a Note.txt; the data file is the
        # largest CSV member.
        best = max(names, key=lambda n: z.getinfo(n).file_size)
        return z.read(best)


def fetch_bom_weather(regions, output_dir, retries=2):
    """Daily max temp / min temp / rainfall per region from BOM Climate Data
    Online.

    ROOT CAUSE OF THE EARLIER FAILURE (diagnosed from the saved files, not
    guessed). Every one of the 21 files written by the previous version was a
    BOM HTML landing page saved under a .csv name. Two things caused that,
    and the session/cookie theory recorded in the earlier preprocessing notes
    was NOT one of them:

      1. WRONG p_display_type. The old code requested
         `p_display_type=dailyDataFile`, which is not a value BOM's cdio
         servlet recognises. Given an unrecognised display type it renders
         the ordinary station page and returns HTTP 200, so nothing raised.
         The correct value - visible in the saved page's own download link -
         is `dailyZippedDataFile`.

      2. RESPONSE IS A ZIP, NOT A CSV. Even with the right display type, the
         endpoint returns a ZIP archive. Writing r.content straight to a
         .csv path stores a ZIP under a CSV name, which fails later in a
         confusing place rather than here.

    Evidence the cookie theory was wrong: the p_c token is a URL parameter,
    not session state, and each saved page contained a correct, distinct,
    product-specific p_c (e.g. -872948931 for NSW max temp). p_c resolution
    was working across independent requests.get() calls. A Session is used
    below anyway - it reuses the connection and keeps headers consistent -
    but it is a robustness improvement, not the fix.

    Everything is validated before anything is written: no HTML page is ever
    saved as CSV again, and a failure logs the response snippet that caused
    it so the next run says why.
    """
    products = {"max_temperature": "122", "min_temperature": "123", "rainfall": "136"}
    os.makedirs(output_dir, exist_ok=True)
    written, failures = {}, []

    with requests.Session() as session:
        session.headers.update(BOM_HEADERS)

        for region in regions:
            station = REGION_CONFIG[region]["bom_station"]
            log.info(f"[BOM] {region} -> {REGION_CONFIG[region]['city']} (station {station})")

            for label, code in products.items():
                key = f"{region}_{label}"
                path = os.path.join(output_dir, f"bom_{region}_{label}_station{station}.csv")
                last_err = None

                for attempt in range(1, retries + 2):
                    try:
                        # Step 1 - resolve the per-product p_c token from the
                        # station page.
                        r1 = session.get(
                            "http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av"
                            f"?p_nccObsCode={code}&p_display_type=dataFile"
                            f"&p_stn_num={station}", timeout=30)
                        r1.raise_for_status()
                        m = re.search(r"p_c=(-?\d+)", r1.text)
                        if not m:
                            raise ValueError(
                                "p_c token not found in station page. First 300 chars: "
                                + r1.text[:300].replace("\n", " "))
                        p_c = m.group(1)

                        # Step 2 - download the zipped data file. Note the
                        # display type: dailyZippedDataFile, not dailyDataFile.
                        r2 = session.get(
                            "http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av"
                            f"?p_display_type=dailyZippedDataFile&p_stn_num={station}"
                            f"&p_c={p_c}&p_nccObsCode={code}&p_startYear=",
                            timeout=90)
                        r2.raise_for_status()

                        ctype = r2.headers.get("Content-Type", "").lower()
                        body = r2.content

                        # Step 3 - validate BEFORE writing.
                        if body[:2] == b"PK":
                            csv_bytes = _bom_extract_csv(body)
                            if not csv_bytes:
                                raise ValueError("ZIP contained no CSV member")
                        elif b"<html" in body[:2000].lower() or "html" in ctype:
                            raise ValueError(
                                f"BOM returned an HTML page (Content-Type: {ctype!r}), "
                                "not a data file - p_display_type or p_c is being "
                                "rejected. First 200 chars: "
                                + body[:200].decode("latin-1", "replace").replace("\n", " "))
                        elif b"," in body[:500] and b"ate" in body[:500]:
                            csv_bytes = body          # already a bare CSV
                        else:
                            raise ValueError(
                                f"Unrecognised response ({len(body)} bytes, "
                                f"Content-Type: {ctype!r}) - refusing to write it as CSV")

                        with open(path, "wb") as f:
                            f.write(csv_bytes)
                        log.info(f"[BOM] {region}/{label} -> {path} "
                                 f"({len(csv_bytes):,} bytes)")
                        written[key] = path
                        last_err = None
                        break

                    except Exception as e:
                        last_err = e
                        log.warning(f"[BOM] {region}/{label} attempt {attempt} failed: {e}")
                        time.sleep(2 * attempt)

                if last_err is not None:
                    log.error(f"[BOM] {region}/{label} FAILED after {retries + 1} "
                              f"attempts: {last_err}")
                    failures.append({"region": region, "product": label,
                                     "station": station, "error": str(last_err)})
                    # Deliberately leave no file rather than a misleading one.
                    if os.path.exists(path):
                        os.remove(path)

                time.sleep(1)

    if failures:
        import csv as _csv
        rep = os.path.join(output_dir, "bom_download_failures.csv")
        with open(rep, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["region", "product", "station", "error"])
            w.writeheader(); w.writerows(failures)
        log.error(f"[BOM] {len(failures)} downloads failed - see {rep}. "
                  "Weather features must stay unavailable until this is clean.")
    else:
        log.info(f"[BOM] all {len(written)} weather files downloaded and validated")

    return written



def fetch_public_holidays(start_date, end_date, regions, output_dir):
    try:
        import holidays
    except ImportError:
        log.error("holidays not installed. Run: pip install holidays")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))
    dates = pd.date_range(start_date, end_date, freq="D")
    written = {}

    for region in regions:
        state = REGION_CONFIG[region]["state"]
        cal = holidays.Australia(subdiv=state, years=years)
        df = pd.DataFrame({"date": dates})
        df["is_public_holiday"] = df["date"].dt.date.isin(cal).astype(int)
        df["holiday_name"] = df["date"].dt.date.map(lambda d: cal.get(d, ""))
        path = os.path.join(output_dir, f"public_holidays_{region}_{state}.csv")
        df.to_csv(path, index=False)
        log.info(f"[Holidays] {region} ({state}): {df.is_public_holiday.sum()} days")
        written[region] = path

    return written



# ------------------------------------------------------------------
# Output validation
# ------------------------------------------------------------------
def build_validation_report(output_dir, report_path=None):
    """Verify what the pipeline actually wrote, per file.

    The 23 August acceptance check was a manual region-by-region comparison
    against the provider websites with the screen shared (minutes, section 5).
    That check accepted the pipeline once; this function makes it repeatable,
    so any later run can be compared against the same criteria instead of
    being trusted because an earlier run passed.

    Reports rows, column count, timestamp coverage, duplicate timestamps and
    missing-cell counts for every CSV written. It does not judge - it records,
    so a change between runs is visible.
    """
    output_dir = Path(output_dir)
    report_path = Path(report_path or output_dir / "validation_report.csv")
    rows = []

    for csv_path in sorted(output_dir.rglob("*.csv")):
        if csv_path.name == report_path.name:
            continue
        entry = {"folder": csv_path.parent.name, "file": csv_path.name,
                 "bytes": csv_path.stat().st_size, "rows": 0, "columns": 0,
                 "timestamp_column": "", "start": "", "end": "",
                 "duplicate_timestamps": "", "missing_cells": "", "status": ""}
        # Name the cause rather than surfacing a parse error from deep inside
        # pandas. A provider that serves an HTML page or a zip under a .csv
        # name returns HTTP 200, so the only place to catch it is here.
        head = csv_path.read_bytes()[:512]
        if head[:2] == b"PK":
            entry["status"] = "ZIP_NOT_CSV - provider returned an archive"
            rows.append(entry)
            log.error(f"[validate] {csv_path.name}: zip archive saved as .csv")
            continue
        if head.decode("latin-1", "replace").lstrip().lower().startswith(("<!doctype", "<html")):
            entry["status"] = "HTML_NOT_DATA - provider served a web page"
            rows.append(entry)
            log.error(f"[validate] {csv_path.name}: HTML page saved as .csv")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            entry["status"] = f"UNREADABLE: {e}"
            rows.append(entry)
            log.error(f"[validate] {csv_path.name}: {e}")
            continue

        entry["rows"] = len(df)
        entry["columns"] = df.shape[1]
        entry["missing_cells"] = int(df.isna().sum().sum())

        ts_col = next((c for c in ("SETTLEMENTDATE", "date", "DATE", "Date")
                       if c in df.columns), None)
        if ts_col:
            ts = parse_dt(df[ts_col])
            entry["timestamp_column"] = ts_col
            if ts.notna().any():
                entry["start"] = str(ts.min())
                entry["end"] = str(ts.max())
                entry["duplicate_timestamps"] = int(ts.duplicated().sum())

        if entry["rows"] == 0:
            entry["status"] = "EMPTY"
        elif entry["duplicate_timestamps"] not in ("", 0):
            entry["status"] = "DUPLICATES PRESENT"
        else:
            entry["status"] = "ok"
        rows.append(entry)

    report = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    n_bad = int((report["status"] != "ok").sum()) if len(report) else 0
    if n_bad:
        log.warning(f"[validate] {n_bad} of {len(report)} files need attention "
                    f"- see {report_path}")
    else:
        log.info(f"[validate] {len(report)} files validated -> {report_path}")
    return report
