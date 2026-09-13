"""
common.py - shared configuration and helpers
PRT661 WP1 Data Acquisition · Dan6: Theme 2

Owned jointly. Every source module imports from here so that region lists,
headers and endpoints are defined exactly once.

DEMAND_REGIONS and FEATURE_REGIONS are deliberately two separate lists. They
were one list until the 23 August integration, when weather and holidays were
found to be collecting for the five NEM regions only - WA had never been added
when the WA module landed, and nothing failed because the step did exactly what
it was told for the wrong list (23 August minutes, section 3.2).
"""
import argparse
import json
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

import zipfile
import io

log = logging.getLogger("data_acquisition")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

BOM_HEADERS = {"User-Agent": UA, "Referer": "http://www.bom.gov.au/climate/data/"}

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
    "NSW1": {"bom_station": "066062", "state": "NSW", "city": "Sydney (Observatory Hill)"},
    "QLD1": {"bom_station": "040913", "state": "QLD", "city": "Brisbane"},
    "VIC1": {"bom_station": "086338", "state": "VIC", "city": "Melbourne (Olympic Park)"},
    "SA1":  {"bom_station": "023034", "state": "SA",  "city": "Adelaide (Airport)"},
    "TAS1": {"bom_station": "094029", "state": "TAS", "city": "Hobart (Ellerslie Road)"},
    "WA":   {"bom_station": "009021", "state": "WA",  "city": "Perth Airport"},
    "NT":   {"bom_station": "014015", "state": "NT",  "city": "Darwin Airport"},
}

# Two different region lists - keep them distinct.
#   DEMAND_REGIONS  : regions we can actually get demand data for.
#   FEATURE_REGIONS : regions we collect weather and holiday drivers for.
# NT appears only in FEATURE_REGIONS. NTESMO publishes no bulk demand download,
# but Darwin weather and NT holidays are still collected so the region can be
# added later without re-running everything.
NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
DEMAND_REGIONS = NEM_REGIONS + ["WA"]
FEATURE_REGIONS = NEM_REGIONS + ["WA", "NT"]

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
| `bom_weather/` | Daily max/min temperature and rainfall: 5 NEM regions + WA + NT. |
| `public_holidays/` | State public-holiday calendars: 5 NEM regions + WA + NT. |

## Regions: demand vs features

Two different lists, deliberately kept apart:

- **Demand** (6): NSW1, QLD1, VIC1, SA1, TAS1, WA.
- **Weather and holidays** (7): the six above, plus NT.

NT has weather and holiday features but no operational demand. NTESMO publishes
no bulk historical download, so NT demand cannot be acquired reproducibly. The
drivers are collected anyway: they are cheap, and if NT demand is ever obtained
the features are already in place. Do not expect an NT demand file - there
isn't one, by design.

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


