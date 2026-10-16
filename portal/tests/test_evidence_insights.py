"""Evidence verification + deterministic insights (tests 12 and evidence-specific checks)."""
import json
import shutil
from pathlib import Path

import pytest

from pipeline import insights as ins
from pipeline import model_evidence as me

EXP = json.loads((me.EVIDENCE_DIR.parent / "dashboard_test_expectations.json").read_text())
REGIONS = list(EXP["selected_model"])


def test_manifest_verifies_ok():
    _, st = me.load_manifest()
    assert st.state == "OK", st.problems
    assert "Pipeline v4.0.0" in st.footer() and "2023–2025" in st.footer()


def test_missing_and_tampered_evidence(tmp_path):
    _, st = me.load_manifest(tmp_path / "nothing")
    assert st.state == "EVIDENCE_MISSING"
    copy = tmp_path / "current"
    shutil.copytree(me.EVIDENCE_DIR, copy)
    (copy / "regional_model_winners.csv").write_text("tampered", encoding="utf-8")
    _, st2 = me.load_manifest(copy)
    assert st2.state == "EVIDENCE_UNVERIFIED" and any("regional_model_winners" in p for p in st2.problems)
    assert me.evidence("does_not_exist.csv").empty


@pytest.mark.parametrize("region", REGIONS)
def test_ranking_has_exactly_ten_core_models(region):
    rank = ins.ranking_frame(region, me.evidence("model_comparison_by_region.csv"), me.evidence("model_failures.csv"),
                             EXP["selected_model"][region])
    assert sorted(rank["model"]) == sorted(me.CORE_MODELS)
    assert "GRU_L672" not in set(rank["model"]) and "XGBoost_A+CoreWeather" not in set(rank["model"])
    assert rank.loc[rank["is_selected"], "model"].tolist() == [EXP["selected_model"][region]]
    assert set(rank.loc[rank["is_new"], "model"]) == {"MLP", "GRU"}
    assert rank.iloc[0]["model"] == EXP["selected_model"][region]


def test_12_sarimax_incomplete_is_flagged():
    total_failed = 0
    for region in REGIONS:
        rank = ins.ranking_frame(region, me.evidence("model_comparison_by_region.csv"), me.evidence("model_failures.csv"),
                                 EXP["selected_model"][region])
        s = rank[rank["model"] == "SARIMAX"].iloc[0]
        total_failed += int(s["failed_runs"])
        if not s["complete"]:
            assert "SARIMAX" in ins.ranking_sentence(rank, region)
    assert total_failed == EXP["sarimax_failed_runs"]


@pytest.mark.parametrize("region", REGIONS)
def test_neural_insight_never_claims_improvement_without_evidence(region):
    nv = ins.neural_comparison_insight(region, me.evidence("neural_vs_previous_comparison.csv"))
    assert "higher error" in nv["sentence"] or "lower error" in nv["sentence"]
    if not EXP["mlp_material_improvement"][region] and not EXP["gru_material_improvement"][region]:
        assert "Neither neural model" in nv["sentence"] and "improved materially" not in nv["sentence"]
    assert "+" not in nv["sentence"]    # never a bare signed number


def test_stability_names_weak_years_honestly():
    s = ins.stability_insight("NSW1", me.evidence("model_comparison_by_year.csv"), "RandomForest_A")["sentence"]
    assert "2 of 3" in s and "2024" in s
    q = ins.stability_insight("QLD1", me.evidence("model_comparison_by_year.csv"), "RandomForest_A")["sentence"]
    assert "3 of 3" in q and "ranked" not in q


def test_peak_insight_handles_both_directions():
    tas = ins.peak_insight("TAS1", me.evidence("peak_normal_vs_high_summary.csv"), "RandomForest_A")["sentence"]
    assert "not harder" in tas
    nsw = ins.peak_insight("NSW1", me.evidence("peak_normal_vs_high_summary.csv"), "RandomForest_A")["sentence"]
    assert "harder" in nsw and "not harder" not in nsw


def test_horizon_insight_handles_non_monotonic():
    v = ins.horizon_insight("VIC1", me.evidence("model_comparison_by_horizon.csv"), "RandomForest_A")["sentence"]
    assert "does not simply rise" in v
    n = ins.horizon_insight("NSW1", me.evidence("model_comparison_by_horizon.csv"), "RandomForest_A")["sentence"]
    assert "grows steadily" in n


def test_decision_confidence_rule():
    assert ins.decision_confidence(3.0, 0.03)["clear"]
    assert not ins.decision_confidence(5.1, 0.2)["clear"]
    assert not ins.decision_confidence(1.6, 0.001)["clear"]
    w = me.evidence("regional_model_winners.csv")
    labels = {r: ins.selected_model_insight(r, w)["confidence"]["label"] for r in REGIONS}
    assert labels["QLD1"] == "Clear margin" and labels["SA1"] == "Narrow margin"


def test_weather_and_feature_set_sentences():
    br = me.evidence("model_comparison_by_region.csv")
    wx = ins.weather_insight("WA", me.evidence("weather_ablation_metrics.csv"), br, "B")
    assert "climatological" in wx["caveat"] and "weather-enhanced (B)" in wx["feature_set_sentence"]
    assert "reduced error" in wx["sentence"]
    assert "core (A)" in ins.feature_set_insight("QLD1", br, "A")


def test_drivers_caveat_and_limit():
    d = ins.drivers_insight("SA1", me.evidence("feature_importance_grouped.csv"))
    assert len(d["frame"]) <= 5 and "does not prove" in d["caveat"]


def test_direction_words_and_freshness():
    assert ins.forecast_direction_insight(1000, 1100)["direction"] == "up"
    assert ins.forecast_direction_insight(1000, 1001)["direction"] == "flat"
    assert ins.forecast_direction_insight(None, 1000)["direction"] == "unknown"
    import pandas as pd
    now = pd.Timestamp("2026-09-30 12:00")
    fr = ins.freshness(now - pd.Timedelta(minutes=90), now - pd.Timedelta(hours=3), now)
    assert fr["actual_stale"] and fr["forecast_stale"] and len(fr["messages"]) == 2
    ok = ins.freshness(now - pd.Timedelta(minutes=10), now - pd.Timedelta(minutes=40), now)
    assert not ok["actual_stale"] and not ok["forecast_stale"]
