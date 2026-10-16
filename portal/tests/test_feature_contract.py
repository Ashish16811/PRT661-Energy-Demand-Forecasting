"""Tests 6-7: exact A/B feature contracts; A contains no weather."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.feature_contract import FEATURE_SET_A, FEATURE_SET_B, WEATHER_FEATURES

WEATHER_TOKENS = ("weather", "temp", "rain", "humid", "hdd", "cdd", "climatology")


def test_06_feature_set_a_core_no_weather():
    assert len(FEATURE_SET_A) == 39 and len(set(FEATURE_SET_A)) == 39
    assert not [f for f in FEATURE_SET_A if any(t in f for t in WEATHER_TOKENS)]
    for f in ("lag_48", "lag_96", "lag_336", "is_public_holiday", "historical_month_halfhour_mean", "time_index"):
        assert f in FEATURE_SET_A


def test_07_feature_set_b_is_a_plus_15_weather():
    assert len(FEATURE_SET_B) == 54 and FEATURE_SET_B[:39] == FEATURE_SET_A
    assert len(WEATHER_FEATURES) == 15 and all(any(t in f for t in WEATHER_TOKENS) for f in WEATHER_FEATURES)


def test_07b_matches_stage2_feature_files_if_present():
    root = Path(__file__).resolve().parents[1] / "data" / "stage2" / "current"
    a = sorted(root.glob("*/features/feature_set_A_core.csv"))
    b = sorted(root.glob("*/features/feature_set_B_weather.csv"))
    if not a or not b:
        pytest.skip("Stage 2 feature lists not present in this checkout")
    assert pd.read_csv(a[-1]).iloc[:, 0].tolist() == FEATURE_SET_A
    assert pd.read_csv(b[-1]).iloc[:, 0].tolist() == FEATURE_SET_B


def test_methodology_text_no_longer_claims_weather_in_a():
    src = (Path(__file__).resolve().parents[1] / "app" / "app.py").read_text(encoding="utf-8")
    assert '"Calendar", "Holidays", "Weather")' not in src
    assert "A · core" in src and "B · weather-enhanced" in src
