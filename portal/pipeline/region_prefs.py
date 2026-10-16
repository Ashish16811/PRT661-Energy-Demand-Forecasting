"""Browser-timezone -> electricity-region mapping and region-preference resolution.

UI selection only: nothing here touches database timestamps or the market-clock
contract. Mapping uses exact IANA names; a region is NEVER inferred from a UTC
offset (Australia/Darwin shares Adelaide's standard offset but is not SA1).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .config import PORTAL_DEFAULT_REGION, REGION_LABELS, REGIONS

TZ_TO_REGION = {
    # NSW1 region includes the ACT; Broken Hill (Australia/Broken_Hill, alias
    # Yancowinna) is in NSW (NSW1) although it keeps South Australian time.
    "Australia/Sydney": "NSW1", "Australia/ACT": "NSW1", "Australia/Canberra": "NSW1",
    "Australia/NSW": "NSW1", "Australia/Broken_Hill": "NSW1", "Australia/Yancowinna": "NSW1",
    "Australia/Melbourne": "VIC1", "Australia/Victoria": "VIC1",
    "Australia/Brisbane": "QLD1", "Australia/Queensland": "QLD1", "Australia/Lindeman": "QLD1",
    "Australia/Adelaide": "SA1", "Australia/South": "SA1",
    "Australia/Hobart": "TAS1", "Australia/Tasmania": "TAS1", "Australia/Currie": "TAS1",
    "Australia/Perth": "WA", "Australia/West": "WA",
}
EXPLICITLY_UNSUPPORTED = {
    "Australia/Darwin": "Northern Territory is not one of the six supported regions",
    "Australia/North": "Northern Territory is not one of the six supported regions",
    "Australia/Eucla": "Eucla is not on the WA (SWIS) grid",
    "Australia/Lord_Howe": "Lord Howe Island has its own supply",
    "Australia/LHI": "Lord Howe Island has its own supply",
    "Antarctica/Macquarie": "Macquarie Island is not on a supported grid",
}
UNSUPPORTED_MESSAGE = ("Your timezone does not map directly to one of the six supported "
                       "electricity regions. Select a region.")
SOURCES = ("url", "manual", "timezone", "fallback")


def map_timezone(tz: str | None) -> tuple[str | None, str]:
    """(region or None, reason). Exact IANA match only."""
    if not tz:
        return None, "timezone not reported by the browser"
    if tz in TZ_TO_REGION:
        return TZ_TO_REGION[tz], "mapped"
    if tz in EXPLICITLY_UNSUPPORTED:
        return None, EXPLICITLY_UNSUPPORTED[tz]
    return None, "timezone outside the supported Australian regions"


def default_region() -> str:
    return PORTAL_DEFAULT_REGION if PORTAL_DEFAULT_REGION in REGIONS else "NSW1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _message(region: str, source: str, tz: str | None, unsupported: bool) -> str:
    label = REGION_LABELS.get(region, region)
    if unsupported and source == "fallback":
        return UNSUPPORTED_MESSAGE
    return {
        "timezone": f"Showing {label} — based on your timezone",
        "manual": f"Showing {label} — selected by you",
        "url": f"Showing {label} — from the shared link",
        "fallback": f"Showing {label} — default region",
    }[source]


def resolve_region(trigger: str, *, url_region: str | None = None, pref: dict | None = None,
                   browser_tz: str | None = None, clicked_region: str | None = None) -> tuple[str, dict, str]:
    """Decide the region to show.

    trigger: "manual" (map/dropdown/strip/export click), "reset" (Use my timezone),
             otherwise an automatic event (page load, timezone detected, URL change).
    Priority for automatic events: URL > stored manual > mapped timezone >
    stored timezone-derived > PORTAL_DEFAULT_REGION > NSW1.
    Returns (region, preference_to_store, message). A URL selection is shown but
    never overwrites a stored manual preference.
    """
    pref = dict(pref or {})
    mapped, _ = map_timezone(browser_tz)
    unsupported = bool(browser_tz) and mapped is None

    if trigger == "manual" and clicked_region in REGIONS:
        new = {"region": clicked_region, "selection_source": "manual", "browser_timezone": browser_tz,
               "updated_at": _now()}
        return clicked_region, new, _message(clicked_region, "manual", browser_tz, unsupported)

    if trigger == "reset":
        if mapped:
            new = {"region": mapped, "selection_source": "timezone", "browser_timezone": browser_tz, "updated_at": _now()}
            return mapped, new, _message(mapped, "timezone", browser_tz, False)
        region = pref.get("region") if pref.get("region") in REGIONS else default_region()
        new = {"region": region, "selection_source": "fallback", "browser_timezone": browser_tz, "updated_at": _now()}
        return region, new, _message(region, "fallback", browser_tz, unsupported)

    if url_region in REGIONS:
        return url_region, pref, _message(url_region, "url", browser_tz, unsupported)
    if pref.get("selection_source") == "manual" and pref.get("region") in REGIONS:
        return pref["region"], pref, _message(pref["region"], "manual", browser_tz, unsupported)
    if mapped:
        new = {"region": mapped, "selection_source": "timezone", "browser_timezone": browser_tz, "updated_at": _now()}
        return mapped, new, _message(mapped, "timezone", browser_tz, False)
    if pref.get("region") in REGIONS:
        src = pref.get("selection_source") if pref.get("selection_source") in SOURCES else "fallback"
        return pref["region"], pref, _message(pref["region"], src, browser_tz, unsupported and src == "fallback")
    region = default_region()
    new = {"region": region, "selection_source": "fallback", "browser_timezone": browser_tz, "updated_at": _now()}
    return region, new, _message(region, "fallback", browser_tz, unsupported)
