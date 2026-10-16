"""Tests 13-18: timezone mapping, unsupported zones, manual persistence, URL priority."""
import pytest

from pipeline.region_prefs import UNSUPPORTED_MESSAGE, map_timezone, resolve_region

MAP = {
    "Australia/Sydney": "NSW1", "Australia/ACT": "NSW1", "Australia/Canberra": "NSW1", "Australia/NSW": "NSW1",
    "Australia/Broken_Hill": "NSW1", "Australia/Yancowinna": "NSW1",
    "Australia/Melbourne": "VIC1", "Australia/Victoria": "VIC1",
    "Australia/Brisbane": "QLD1", "Australia/Queensland": "QLD1", "Australia/Lindeman": "QLD1",
    "Australia/Adelaide": "SA1", "Australia/South": "SA1",
    "Australia/Hobart": "TAS1", "Australia/Tasmania": "TAS1", "Australia/Currie": "TAS1",
    "Australia/Perth": "WA", "Australia/West": "WA",
}


@pytest.mark.parametrize("tz,region", MAP.items())
def test_13_timezone_mapping(tz, region):
    assert map_timezone(tz)[0] == region


@pytest.mark.parametrize("tz", ["Australia/Darwin", "Australia/North", "Australia/Eucla", "Australia/Lord_Howe",
                                "Australia/LHI", "Antarctica/Macquarie", "Asia/Kathmandu", "Europe/London", "UTC", ""])
def test_14_16_unsupported_zones_never_mapped(tz):
    assert map_timezone(tz)[0] is None


def test_14_darwin_falls_back_with_message():
    region, pref, msg = resolve_region("auto", browser_tz="Australia/Darwin")
    assert region == "NSW1" and pref["selection_source"] == "fallback" and msg == UNSUPPORTED_MESSAGE


def test_17_manual_choice_persists_over_timezone():
    _, pref, _ = resolve_region("manual", browser_tz="Australia/Adelaide", clicked_region="WA")
    assert pref["selection_source"] == "manual"
    region, pref2, msg = resolve_region("auto", pref=pref, browser_tz="Australia/Adelaide")
    assert region == "WA" and pref2["selection_source"] == "manual" and "selected by you" in msg


def test_17b_use_my_timezone_resets_manual():
    _, pref, _ = resolve_region("manual", browser_tz="Australia/Hobart", clicked_region="QLD1")
    region, pref2, _ = resolve_region("reset", pref=pref, browser_tz="Australia/Hobart")
    assert region == "TAS1" and pref2["selection_source"] == "timezone"


def test_18_url_region_beats_stored_manual_without_overwriting_it():
    _, pref, _ = resolve_region("manual", clicked_region="WA")
    region, pref2, msg = resolve_region("auto", url_region="SA1", pref=pref, browser_tz="Australia/Sydney")
    assert region == "SA1" and pref2 == pref and "shared link" in msg


def test_timezone_first_visit_and_previous_timezone_choice():
    region, pref, msg = resolve_region("auto", browser_tz="Australia/Perth")
    assert region == "WA" and pref["selection_source"] == "timezone" and "based on your timezone" in msg
    region2, _, _ = resolve_region("auto", pref=pref, browser_tz=None)
    assert region2 == "WA"
