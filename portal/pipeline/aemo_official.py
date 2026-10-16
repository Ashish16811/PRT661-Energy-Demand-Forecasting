from __future__ import annotations

import csv
import io
import re
import zipfile
from urllib.parse import urljoin

import pandas as pd
import requests

from .database import upsert_frame, utcnow_naive

FORECAST_DIR = "https://nemweb.com.au/Reports/Current/Operational_Demand/FORECAST_HH/"
UA = "PRT661-Energy-Forecasting-Academic-Portal/1.0"


def _latest_zip(directory_url: str) -> tuple[str, bytes]:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    html = session.get(directory_url, timeout=30).text
    hrefs = re.findall(r'href=["\']([^"\']+\.zip)["\']', html, flags=re.I)
    if not hrefs:
        raise RuntimeError(f"No ZIP files listed at {directory_url}")
    href = sorted(hrefs)[-1]
    url = urljoin(directory_url, href)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return href.split("/")[-1], r.content


def _mms_csv_from_zip(blob: bytes) -> pd.DataFrame:
    """Parse AEMO MMS-style I/D rows without hard-coding field positions."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError("AEMO ZIP contains no CSV")
        text = zf.read(members[0]).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    header_idx = None
    header = None
    prefix = None
    for i, row in enumerate(rows):
        upper = [str(x).strip().upper() for x in row]
        if row and upper[0] == "I" and "REGIONID" in upper:
            header_idx = i
            # MMS files conventionally carry I,<dataset>,<table>,<version>,<fields...>.
            # Find REGIONID and infer the metadata prefix as all cells before the first field.
            region_pos = upper.index("REGIONID")
            # Date fields are also data fields; four metadata cells is the normal MMS shape.
            start = 4 if len(row) > 4 else 1
            header = [x.strip() for x in row[start:]]
            prefix = row[1:start]
            break
    if header_idx is None or header is None:
        raise ValueError("Could not find AEMO MMS header row containing REGIONID")
    data = []
    for row in rows[header_idx + 1:]:
        if not row or row[0].strip().upper() != "D":
            continue
        start = 4 if len(row) >= 4 + len(header) else max(1, len(row) - len(header))
        vals = row[start:start + len(header)]
        if len(vals) == len(header):
            data.append(vals)
    return pd.DataFrame(data, columns=header)


def refresh_aemo_operational_forecast() -> pd.DataFrame:
    filename, blob = _latest_zip(FORECAST_DIR)
    df = _mms_csv_from_zip(blob)
    cols = {c.upper(): c for c in df.columns}
    time_col = next((cols[x] for x in ("INTERVAL_DATETIME", "SETTLEMENTDATE", "DATETIME") if x in cols), None)
    region_col = cols.get("REGIONID")
    poe10 = cols.get("OPERATIONAL_DEMAND_POE10")
    poe50 = cols.get("OPERATIONAL_DEMAND_POE50")
    poe90 = cols.get("OPERATIONAL_DEMAND_POE90")
    if not time_col or not region_col or not poe50:
        raise ValueError(f"Unexpected AEMO forecast columns: {list(df.columns)}")
    origin_match = re.search(r"_(\d{12})_", filename)
    origin = pd.to_datetime(origin_match.group(1), format="%Y%m%d%H%M") if origin_match else pd.Timestamp.now().floor("30min")
    out = pd.DataFrame({
        "region": df[region_col].astype(str),
        "forecast_origin": origin,
        "target_timestamp": pd.to_datetime(df[time_col], errors="coerce"),
        "poe10_mw": pd.to_numeric(df[poe10], errors="coerce") if poe10 else None,
        "poe50_mw": pd.to_numeric(df[poe50], errors="coerce"),
        "poe90_mw": pd.to_numeric(df[poe90], errors="coerce") if poe90 else None,
        "source": "AEMO_OPERATIONAL_DEMAND_FORECAST_HH",
        "retrieved_at": utcnow_naive(),
    }).dropna(subset=["target_timestamp", "poe50_mw"])
    upsert_frame("aemo_forecast", out, ["region", "forecast_origin", "target_timestamp"])
    return out
