from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .config import REGION_MARKET, REGION_TZ

# AEMO NEM settlement timestamps are represented on the market clock (AEST,
# UTC+10 year-round). WA/WEM timestamps use AWST (UTC+8). Region-facing
# calendar features and portal labels use each region's civil timezone.
NEM_MARKET_TZ = "Australia/Brisbane"
WA_MARKET_TZ = "Australia/Perth"


def display_tz_name(region: str) -> str:
    return REGION_TZ[region]


def market_tz_name(region: str) -> str:
    return WA_MARKET_TZ if REGION_MARKET.get(region, "").startswith("WEM") else NEM_MARKET_TZ


def _aware_utc(now=None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        # Explicit naive timestamps are treated as UTC only for generic conversion
        # helpers. Operational forecasting uses market_now_naive(), which preserves
        # backward compatibility for historical/test inputs.
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def local_now(region: str, now=None) -> pd.Timestamp:
    """Current region civil time, timezone-aware (DST applied where relevant)."""
    return _aware_utc(now).tz_convert(ZoneInfo(display_tz_name(region)))


def market_now(region: str, now=None) -> pd.Timestamp:
    """Current market-clock time, timezone-aware."""
    if now is not None:
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            # Historical/test callers that supplied naive values previously meant
            # portal/market wall-clock time. Preserve that behavior.
            return ts.tz_localize(ZoneInfo(market_tz_name(region)))
    return _aware_utc(now).tz_convert(ZoneInfo(market_tz_name(region)))


def market_now_naive(region: str, now=None) -> pd.Timestamp:
    return market_now(region, now).tz_localize(None)


def next_market_half_hour(region: str, now=None) -> pd.Timestamp:
    ts = market_now_naive(region, now)
    return ts.floor("30min") + pd.Timedelta(minutes=30)


def market_naive_to_local(region: str, value) -> pd.Timestamp:
    """Convert a DB market-clock timestamp to region civil time.

    DB timestamps are intentionally timezone-naive. This helper attaches the
    market timezone first, then converts to the region's DST-aware civil zone.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo(market_tz_name(region)))
    else:
        ts = ts.tz_convert(ZoneInfo(market_tz_name(region)))
    return ts.tz_convert(ZoneInfo(display_tz_name(region)))


def region_clock_label(region: str, now=None) -> str:
    ts = local_now(region, now)
    return f"{ts:%H:%M} {ts.tzname()}"


def market_clock_label(region: str, now=None) -> str:
    ts = market_now(region, now)
    return f"{ts:%H:%M} {ts.tzname()}"
