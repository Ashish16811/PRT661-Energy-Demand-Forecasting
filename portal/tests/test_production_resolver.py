"""Production-model correctness gate (tests 1-5, 8-11)."""
import json
import os
from pathlib import Path

import pytest

from conftest import make_fake_bundle
from pipeline import model_evidence as me

EXPECTED = {"NSW1": "RandomForest_A", "QLD1": "RandomForest_A", "VIC1": "RandomForest_A",
            "SA1": "RandomForest_B", "TAS1": "RandomForest_A", "WA": "RandomForest_B"}


def full_dir(model_dir, sa1_b=True, wa_b=True):
    for r in ("NSW1", "QLD1", "VIC1", "SA1", "TAS1", "WA"):
        make_fake_bundle(model_dir, r, "RandomForest_A")
    if sa1_b:
        make_fake_bundle(model_dir, "SA1", "RandomForest_B")
    if wa_b:
        make_fake_bundle(model_dir, "WA", "RandomForest_B")
    return model_dir


def test_01_selected_model_mapping_from_evidence():
    sel = me.selected_models()
    assert {r: v["model"] for r, v in sel.items()} == EXPECTED


def test_02_sa1_missing_b_does_not_silently_switch_to_a(model_dir):
    full_dir(model_dir, sa1_b=False)
    cfg = me.resolve_production_config(model_dir=model_dir, validation={})
    s = cfg.status["SA1"]
    assert s.selected == "RandomForest_B" and s.served == "RandomForest_B"
    assert s.state == "NO_BUNDLE" and not s.mismatch
    assert "No production bundle available" in s.message
    assert cfg.production_models["SA1"] == []
    assert cfg.training_models["SA1"] == ["RandomForest_B"]


def test_03_sa1_b_present_is_served(model_dir):
    full_dir(model_dir)
    cfg = me.resolve_production_config(model_dir=model_dir, validation={})
    assert cfg.default_model["SA1"] == "RandomForest_B"
    assert cfg.production_models["SA1"] == ["RandomForest_B"]


def test_04_full_expected_production_config(model_dir):
    full_dir(model_dir)
    cfg = me.resolve_production_config(model_dir=model_dir, validation={})
    assert cfg.default_model == EXPECTED
    assert cfg.production_models == {"NSW1": ["RandomForest_A"], "QLD1": ["RandomForest_A"], "VIC1": ["RandomForest_A"],
                                     "SA1": ["RandomForest_B"], "TAS1": ["RandomForest_A"],
                                     "WA": ["RandomForest_B"]}
    assert all(not s.mismatch for s in cfg.status.values())


def test_04b_list_order_never_decides_default(model_dir):
    full_dir(model_dir)
    cfg = me.resolve_production_config(model_dir=model_dir, validation={},
                                       fallback_models={r: ["RandomForest_A", "RandomForest_B"] for r in EXPECTED})
    assert cfg.default_model["WA"] == "RandomForest_B" and cfg.default_model["SA1"] == "RandomForest_B"


def test_05_bundle_structure_validation(model_dir):
    import joblib
    good = make_fake_bundle(model_dir, "WA", "RandomForest_B")
    assert me.validate_bundle_object(joblib.load(good), "RandomForest_B") == []
    bad = make_fake_bundle(model_dir, "SA1", "RandomForest_B", broken=True)
    assert me.validate_bundle_object(joblib.load(bad), "RandomForest_B")          # bare estimator refused
    wrong = joblib.load(good)
    assert any("feature_set" in p for p in me.validate_bundle_object(wrong, "RandomForest_A"))


def test_05b_recorded_invalid_bundle_does_not_switch_to_a(model_dir):
    full_dir(model_dir)
    path = me.resolve_pointer(me.pointer_path("WA", "RandomForest_B", model_dir), model_dir)
    validation = {"WA/RandomForest_B": {"path": str(path), "mtime": path.stat().st_mtime, "valid": False, "problems": ["test"]}}
    cfg = me.resolve_production_config(model_dir=model_dir, validation=validation)
    assert cfg.default_model["WA"] == "RandomForest_B"
    assert cfg.production_models["WA"] == [] and cfg.status["WA"].state == "NO_BUNDLE"


def test_08_legacy_docker_pointer_resolves(model_dir):
    make_fake_bundle(model_dir, "NSW1", "RandomForest_A", legacy_pointer=True)
    p = me.resolve_pointer(me.pointer_path("NSW1", "RandomForest_A", model_dir), model_dir)
    assert p is not None and p.is_file() and str(p).startswith(str(model_dir))


def test_09_relative_pointer_resolves(model_dir):
    path = make_fake_bundle(model_dir, "QLD1", "RandomForest_A", legacy_pointer=False)
    assert me.resolve_pointer(me.pointer_path("QLD1", "RandomForest_A", model_dir), model_dir) == path
    assert me.relative_pointer_text(path, model_dir) == "QLD1/RandomForest_A/vTEST/bundle.joblib"


@pytest.mark.parametrize("model", ["MLP", "GRU", "XGBoost_B", "SARIMAX"])
def test_10_11_research_models_never_production(model_dir, model):
    full_dir(model_dir)
    sel = {r: {"model": model} for r in EXPECTED}
    cfg = me.resolve_production_config(selection=sel, model_dir=model_dir, validation={})
    for r in EXPECTED:
        assert model not in cfg.production_models[r]
        assert cfg.default_model[r] in me.PRODUCTION_ALLOWLIST
        assert cfg.status[r].state == "SELECTED_NOT_ALLOWED"


def test_evidence_missing_uses_fallback(model_dir):
    full_dir(model_dir)
    cfg = me.resolve_production_config(selection={}, model_dir=model_dir, validation={})
    assert all(s.state == "EVIDENCE_MISSING" for s in cfg.status.values())
    assert cfg.default_model["SA1"] == "RandomForest_B"


REAL = os.environ.get("REAL_MODEL_DIR")


@pytest.mark.skipif(not REAL, reason="set REAL_MODEL_DIR to a folder containing the real WA/RandomForest_B bundle")
def test_wa_real_bundle_loads_and_is_feature_set_b():
    import joblib
    import numpy as np
    import pandas as pd
    from pipeline.feature_contract import FEATURE_SET_B
    p = me.resolve_pointer(me.pointer_path("WA", "RandomForest_B", Path(REAL)), Path(REAL))
    assert p is not None
    b = joblib.load(p)
    assert me.validate_bundle_object(b, "RandomForest_B") == []
    assert b["feature_set"] == "B" and list(b["feature_names"]) == FEATURE_SET_B
    X = pd.DataFrame([b["feature_medians"]])[b["feature_names"]].to_numpy(dtype=np.float32)
    assert np.isfinite(b["estimator"].predict(X)).all()
