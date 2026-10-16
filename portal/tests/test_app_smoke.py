"""App-level smoke tests on a seeded SQLite database (tests 21-27, 30).

The seed is SYNTHETIC test data written only to the temporary test database; it is
never shipped or displayed by the portal."""
from contextvars import copy_context

import numpy as np
import pandas as pd
import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict

from conftest import TEST_MODEL_DIR, make_fake_bundle
from pipeline.config import REGIONS

for r in REGIONS:
    make_fake_bundle(TEST_MODEL_DIR, r, "RandomForest_A")
make_fake_bundle(TEST_MODEL_DIR, "SA1", "RandomForest_B")
make_fake_bundle(TEST_MODEL_DIR, "WA", "RandomForest_B")      # both selected B bundles present

import app.app as A  # noqa: E402
from pipeline.database import init_db, upsert_frame  # noqa: E402
from pipeline.timezones import market_now_naive  # noqa: E402


def run(fn, trigger, *args):
    def inner():
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": trigger, "value": 1}]}))
        return fn(*args)
    return copy_context().run(inner)


def _seed():
    rows_a, rows_pf, rows_nf, rows_na, rows_ae, rows_reg = [], [], [], [], [], []
    for r in REGIONS:
        now = market_now_naive(r).floor("30min")
        ts = pd.date_range(now - pd.Timedelta(days=3), now, freq="30min")
        base = {"NSW1": 7000, "QLD1": 6000, "VIC1": 4500, "SA1": 1300, "TAS1": 1100, "WA": 2000}[r]
        y = base + 300 * np.sin(np.arange(len(ts)) / 48 * 2 * np.pi)
        rows_a.append(pd.DataFrame({"region": r, "timestamp": ts, "demand_mw": y, "source": "TEST"}))
        models = ["RandomForest_A", "RandomForest_B"] if r in ("SA1", "WA") else ["RandomForest_A"]
        origin = now - pd.Timedelta(minutes=30)
        fut = pd.date_range(now + pd.Timedelta(minutes=30), now + pd.Timedelta(days=40), freq="30min")
        for m in models:
            ver = "vB_TEST" if m.endswith("_B") else "vA_TEST"
            rows_pf.append(pd.DataFrame({"region": r, "model_name": m, "model_version": ver, "forecast_origin": origin,
                                         "target_timestamp": fut, "forecast_mw": base + 10 + 250 * np.sin(np.arange(len(fut)) / 48 * 2 * np.pi),
                                         "created_at": origin}))
            rows_nf.append(pd.DataFrame({"region": r, "model_name": m, "model_version": ver, "forecast_origin": [now],
                                         "target_timestamp": [now + pd.Timedelta(minutes=30)], "forecast_mw": [base * 1.03],
                                         "lead_minutes": [30.0], "created_at": [now]}))
            past = ts[-40:]
            for k, v in enumerate(["v_old", ver]):   # two versions in the month -> mixed-version guard
                part = past[k * 20:(k + 1) * 20]
                act = base + 300 * np.sin(np.arange(len(part)))
                fc = act + 50
                rows_na.append(pd.DataFrame({"region": r, "model_name": m, "model_version": v, "forecast_origin": part - pd.Timedelta(minutes=30),
                                             "target_timestamp": part, "forecast_mw": fc, "actual_mw": act, "error_mw": 50.0,
                                             "abs_error_mw": 50.0, "squared_error": 2500.0, "smape_component": 1.0, "evaluated_at": now}))
            rows_reg.append(pd.DataFrame({"region": [r], "model_name": [m], "model_version": [ver], "feature_set": [m[-1]],
                                          "training_start": [pd.Timestamp("2022-01-01")], "training_end": [pd.Timestamp("2026-08-31")],
                                          "model_path": [f"{r}/{m}/{ver}/bundle.joblib"], "stage2_snapshot": ["test"], "status": ["PRODUCTION"],
                                          "created_at": [now]}))
        if r != "WA":
            rows_ae.append(pd.DataFrame({"region": r, "forecast_origin": origin, "target_timestamp": fut[:96], "poe10_mw": base,
                                         "poe50_mw": base + 20, "poe90_mw": base + 40, "source": "TEST", "retrieved_at": origin}))
    upsert_frame("actual_demand", pd.concat(rows_a), ["region", "timestamp"])
    upsert_frame("project_forecast", pd.concat(rows_pf), ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"])
    upsert_frame("next_interval_forecast", pd.concat(rows_nf), ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"])
    upsert_frame("next_interval_accuracy", pd.concat(rows_na), ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"])
    upsert_frame("aemo_forecast", pd.concat(rows_ae), ["region", "forecast_origin", "target_timestamp"])
    upsert_frame("model_registry", pd.concat(rows_reg), ["region", "model_name", "model_version"])


# Create the isolated test database schema before inserting synthetic seed data.
init_db()
_seed()


def _text(component) -> str:
    return str(component)


def test_30_app_imports_and_server_exists():
    assert A.server is not None and A.DEFAULT_TAB == "dashboard" and A.NAV_KEYS == ["dashboard", "performance", "aemo", "history"]


def test_21_every_callback_id_exists_in_layout():
    from dash._callback import GLOBAL_CALLBACK_LIST
    ids = set()

    def walk(c):
        if getattr(c, "id", None) is not None and not isinstance(c.id, dict):
            ids.add(c.id)
        ch = getattr(c, "children", None)
        if isinstance(ch, (list, tuple)):
            for x in ch:
                walk(x)
        elif ch is not None and hasattr(ch, "_type"):
            walk(ch)
    walk(A.app.layout)
    dynamic = {"mc-metric"}   # rendered inside the model-comparison content
    missing = set()
    for cb in GLOBAL_CALLBACK_LIST:
        for d in cb["inputs"] + cb.get("state", []):
            if not d["id"].startswith("{") and d["id"] not in ids | dynamic:
                missing.add(d["id"])
        for o in cb["output"].strip(".").split("..."):
            cid = o.split("@")[0].rsplit(".", 1)[0]
            if not cid.startswith("{") and cid not in ids:
                missing.add(cid)
    assert not missing, missing
    assert {"national-overview-metrics", "region-share"} <= ids
    assert "region-overview-cards" not in ids


def test_served_models_resolved_from_evidence_and_bundles():
    A._prod_cfg.__globals__["production_config"].__globals__["_PROD_CACHE"].clear()
    assert A._default_model("WA") == "RandomForest_B"
    assert A._default_model("SA1") == "RandomForest_B"
    assert not A._serving("SA1").mismatch and A._serving("SA1").selected == "RandomForest_B"


def test_25_live_provenance_label_matches_plotted_model():
    out = run(A.update_live, "tabs.value", "WA", None, "48H", 10, 0, "dashboard", ["band"])
    fig, model_line = out[0], _text(out[12])
    assert "RandomForest_B" in model_line and "vB_TEST" in model_line          # chart forecasts label == served B data
    assert any(tr.name == "Typical error range" for tr in fig.data)             # band shown for selected model
    # Passing a historical A model cannot switch the live portal away from the
    # single approved production model; the callback resolves back to B.
    cmp_out = run(A.update_live, "model-switch.value", "WA", "RandomForest_A", "48H", 10, 0, "dashboard", ["band"])
    line = _text(cmp_out[12])
    assert "RandomForest_B" in line and "vB_TEST" in line
    assert any(tr.name == "Typical error range" for tr in cmp_out[0].data)


def test_sa1_selected_b_visible_on_dashboard():
    out = run(A.update_live, "tabs.value", "SA1", None, "7D", 10, 0, "dashboard", ["band"])
    line = _text(out[12])
    assert "RandomForest_B" in line and "vB_TEST" in line
    assert "RandomForest_A" not in line


def test_live_numeric_budget_direction_words():
    out = run(A.update_live, "tabs.value", "NSW1", None, "30D", 10, 0, "dashboard", ["band"])
    change = _text(out[10])
    assert "Rising" in change or "Falling" in change or "Holding" in change
    assert "%" not in change.split("title=")[0]                                   # % only in hover title
    assert "30 minutes" in _text(out[11])                                        # generated insight


def test_27_stale_guard():
    fr = A.ins.freshness(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
    assert fr["actual_stale"] and fr["forecast_stale"]
    out = run(A.update_live, "tabs.value", "QLD1", None, "48H", 10, 0, "dashboard", [])
    assert out[13] == [] or "stale" in _text(out[13])


def test_26_mixed_versions_flagged_in_performance():
    month = market_now_naive("NSW1").strftime("%Y-%m")
    out = run(A.performance, "tabs.value", "NSW1", "RandomForest_A", "month", None, month, 0, "performance")
    assert "Multiple production model versions" in _text(out[7])
    assert "On average the forecast missed" in _text(out[7])


def test_22_aemo_view_runs():
    out = run(A.aemo, "tabs.value", "NSW1", "RandomForest_A", 0, "aemo")
    assert out[0] is not None and len(out) == 6


def test_23_regional_analysis_runs():
    fig, *_ = run(A.history, "tabs.value", "VIC1", [2026], "history")
    assert fig is not None
    prof = run(A.region_profile, "tabs.value", "VIC1", None, "dashboard", 0)
    assert prof
    dfig, note = run(A.region_difficulty, "tabs.value", "history", "SA1")
    assert "does not explain its causes" in note


def test_24_downloads_work():
    card = run(A.download_model_card, "btn-download-model-card.n_clicks", 1, "SA1")
    assert "Selected model (evidence): **RandomForest_B**" in card["content"] and "Served model (live jobs): **RandomForest_B**" in card["content"]
    ev = run(A.download_model_evidence, "btn-download-model-evidence.n_clicks", 1, "WA")
    assert ev["filename"].startswith("WA_") and "MLP" in ev["content"]
    up = run(A.download_upcoming, "btn-download-upcoming.n_clicks", 1, "NSW1", "RandomForest_A", "7D")
    assert up is not None


def test_model_comparison_renders_all_regions():
    for r in REGIONS:
        cards = run(A.model_comparison, "performance-mode.value", r, "comparison", "performance")
        txt = _text(cards)
        assert "Selected for production" in txt and "Did deep learning help?" in txt
        assert "research only" in txt
    sa1 = _text(run(A.model_comparison, "performance-mode.value", "SA1", "comparison", "performance"))
    assert "Selected model" in sa1
    assert "RandomForest_B" in sa1


def test_model_switch_is_static_for_single_model_regions():
    opts, value, style, static, label, *_ = run(A.model_options, "selected-region.data", "NSW1", None, None)
    assert style == {"display": "none"} and "RandomForest_A" in _text(static) and value == "RandomForest_A"
    opts, value, style, static, label, *_ = run(A.model_options, "selected-region.data", "WA", None, None)
    assert style == {"display": "none"} and value == "RandomForest_B" and label == "Production model"
    assert [o["value"] for o in opts] == ["RandomForest_B"]


def test_17_18_choose_region_callback_manual_and_url():
    out = run(A.choose_region, "browser-tz.data", None, "NSW1", "NSW1", [], [], "Australia/Adelaide", 0, "", {}, "NSW1")
    assert out[0] == "SA1" and out[3]["selection_source"] == "timezone"
    manual = run(A.choose_region, "region-dropdown.value", None, "WA", "SA1", [], [], "Australia/Adelaide", 0, "", out[3], "SA1")
    assert manual[0] == "WA" and manual[3]["selection_source"] == "manual"
    again = run(A.choose_region, "browser-tz.data", None, "WA", "WA", [], [], "Australia/Adelaide", 0, "", manual[3], "WA")
    assert again[0] == "WA"
    url = run(A.choose_region, "url.search", None, "WA", "WA", [], [], "Australia/Adelaide", 0, "?region=tas1", manual[3], "WA")
    assert url[0] == "TAS1" and url[3] == manual[3]


def test_other_regions_strip_one_number_each():
    tiles = run(A.other_regions_strip, "refresh-ui.n_intervals", 0, "NSW1", "history")
    assert len(tiles) == 5 and "NSW1" not in [t.id["region"] for t in tiles]


def test_deep_link_navigation():
    assert run(A.navigate, "url.search", "?view=performance&mode=comparison", 0, 0, "verification") == ("performance", "comparison")
    assert run(A.navigate, "btn-why-model.n_clicks", "", 0, 1, "verification") == ("performance", "comparison")
