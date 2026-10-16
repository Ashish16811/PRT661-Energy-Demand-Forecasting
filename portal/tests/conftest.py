"""Test configuration: isolated runtime dir, SQLite DB and model dir (no real bundles,
no network). Must run before any pipeline import."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="prt661_v3_tests_"))
os.environ.setdefault("RUNTIME_DIR", str(_TMP / "runtime"))
os.environ.setdefault("MODEL_DIR", str(_TMP / "models"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'runtime' / 'test_portal.db'}")
(_TMP / "runtime").mkdir(parents=True, exist_ok=True)
(_TMP / "models").mkdir(parents=True, exist_ok=True)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TEST_MODEL_DIR = Path(os.environ["MODEL_DIR"])


def make_fake_bundle(model_dir: Path, region: str, model_name: str, legacy_pointer: bool = True,
                     broken: bool = False) -> Path:
    """Tiny but structurally valid portal bundle (dict with a fitted estimator)."""
    from sklearn.ensemble import RandomForestRegressor
    from pipeline.feature_contract import FEATURE_SET_A, FEATURE_SET_B
    feats = FEATURE_SET_B if model_name.endswith("_B") else FEATURE_SET_A
    est = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=0)
    est.fit(np.random.default_rng(0).normal(size=(20, len(feats))), np.random.default_rng(1).normal(1000, 50, 20))
    bundle = {"region": region, "model_name": model_name, "model_version": "vTEST", "feature_set": model_name[-1],
              "feature_names": list(feats), "feature_medians": {f: 0.0 for f in feats},
              "training_start": "2022-01-01", "training_end": "2026-08-31 23:30:00", "estimator": est,
              "stage2_root": "/app/data/stage2/current/X"}
    if broken:
        bundle = est  # a bare estimator, like the WP4 .joblib files
    path = model_dir / region / model_name / "vTEST" / "bundle.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    pointer = f"/app/models/{region}/{model_name}/vTEST/bundle.joblib" if legacy_pointer else f"{region}/{model_name}/vTEST/bundle.joblib"
    (model_dir / region / model_name / "CURRENT.txt").write_text(pointer, encoding="utf-8")
    return path


@pytest.fixture
def model_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture(autouse=True)
def _clear_caches():
    from pipeline.model_evidence import clear_caches
    clear_caches()
    yield
    clear_caches()
