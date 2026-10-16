"""Targeted production training CLI (SA1 RandomForest_B) — argument handling only, no training run."""
import pytest

from pipeline.model_training import training_targets
from pipeline.monthly_retrain import parse_args


def test_cli_parses_region_and_model():
    a = parse_args(["--region", "SA1", "--model", "RandomForest_B"])
    assert a.region == ["SA1"] and a.model == ["RandomForest_B"]


def test_targets_sa1_b_only():
    assert training_targets(["SA1"], ["RandomForest_B"]) == {"SA1": ["RandomForest_B"]}


def test_default_targets_use_one_approved_production_model_per_region():
    t = training_targets()
    assert t["SA1"] == ["RandomForest_B"] and t["WA"] == ["RandomForest_B"]
    assert t["NSW1"] == ["RandomForest_A"]


@pytest.mark.parametrize("bad", ["MLP", "GRU", "XGBoost_B"])
def test_research_models_cannot_be_trained_for_production(bad):
    with pytest.raises(ValueError):
        training_targets(["SA1"], [bad])


def test_unknown_region_rejected():
    with pytest.raises(ValueError):
        training_targets(["NT1"], None)
