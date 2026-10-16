from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from .config import MODEL_DIR, RAW_DIR, REGIONS, STAGE2_DIR
from .model_evidence import PRODUCTION_ALLOWLIST, clear_caches, relative_pointer_text, resolve_pointer, resolve_production_config
from .database import upsert_frame, utcnow_naive
from .loader import preprocessing_module, acquisition_module

log = logging.getLogger("monthly_retraining")
RANDOM_STATE = 20260914


def _make_model(model_name: str):
    if model_name.startswith("RandomForest"):
        return RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
    if model_name.startswith("XGBoost"):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=700, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(f"Production retraining supports tree models only, got {model_name}")


def _extract_stage2(zip_path: Path, target: Path):
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    root = target
    for _ in range(3):
        if (root / "metadata" / "run_config.json").exists():
            return root
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) != 1:
            break
        root = subs[0]
    raise FileNotFoundError("metadata/run_config.json not found after Stage 2 extraction")


def _feature_list(root: Path, family: str) -> list[str]:
    name = "feature_set_A_core.csv" if family == "A" else "feature_set_B_weather.csv"
    df = pd.read_csv(root / "features" / name)
    return df["feature"].astype(str).tolist()


def prepare_monthly_stage2() -> tuple[Path, Path]:
    """Rebuild the model-ready monthly snapshot. No Stage 3 backtesting is invoked."""
    stage2 = preprocessing_module()
    acq = acquisition_module()
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    # Slow-changing inputs are refreshed once per month, not in the hourly job.
    # A full historical weather rewrite is acceptable monthly and preserves the
    # exact Stage 1/2 schema the validated models were built on.
    acq.fetch_weather(REGIONS, str(RAW_DIR / "weather"), "2022-01-01", today, hourly=True)
    acq.fetch_public_holidays("2022-01-01", today, REGIONS, str(RAW_DIR / "public_holidays"))
    # Preprocessing consumes RAW_DIR directly; hourly jobs keep current demand/weather there.
    result = stage2.run(str(RAW_DIR), str(STAGE2_DIR), regions=REGIONS, make_figures=False)
    zip_path = Path(result["zip"]["zip_file"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extracted = STAGE2_DIR / "current"
    root = _extract_stage2(zip_path, extracted)
    (STAGE2_DIR / "CURRENT_STAGE2.txt").write_text(str(root), encoding="utf-8")
    return zip_path, root


def training_targets(regions=None, models=None) -> dict:
    """Region -> production bundles to (re)train.

    Default: exactly the intended production bundle from the resolver for each
    region. Historical A/B comparisons remain evidence only and are not maintained
    as alternate live-serving bundles. ``regions``/``models`` narrow the run, e.g.
    only SA1 RandomForest_B. Only allow-listed Random Forest
    configurations can ever be trained here; no research model is retrained.
    """
    intended = resolve_production_config().training_models
    regions = list(regions) if regions else list(REGIONS)
    unknown = [r for r in regions if r not in REGIONS]
    if unknown:
        raise ValueError(f"Unknown region(s): {unknown}")
    if models:
        bad = [m for m in models if m not in PRODUCTION_ALLOWLIST]
        if bad:
            raise ValueError(f"Not an allowed production model: {bad}. Allowed: {PRODUCTION_ALLOWLIST}")
    out = {}
    for r in regions:
        chosen = [m for m in intended[r] if not models or m in models]
        if models:  # explicit request for a region/model outside the intended list is allowed only for RF A/B
            chosen = list(dict.fromkeys(chosen + [m for m in models if m in PRODUCTION_ALLOWLIST]))
        if chosen:
            out[r] = chosen
    return out


def _resolve_stage2_root_text(text: str) -> Path:
    from .live_inference import resolve_stage2_root
    return resolve_stage2_root(text.strip())


def train_promoted_models(stage2_root: Path | None = None, regions=None, models=None) -> pd.DataFrame:
    root = stage2_root or _resolve_stage2_root_text((STAGE2_DIR / "CURRENT_STAGE2.txt").read_text())
    targets = training_targets(regions, models)
    log.info("Production training targets: %s", targets)
    run_cfg = json.loads((root / "metadata" / "run_config.json").read_text(encoding="utf-8"))
    version = datetime.now().strftime("v%Y_%m_%d_%H%M")
    rows = []
    for region in targets:
        data = pd.read_csv(root / "data" / f"modelling_{region}_30min.csv",
                           parse_dates=["SETTLEMENTDATE"], low_memory=False)
        data = data.sort_values("SETTLEMENTDATE")
        for model_name in targets[region]:
            family = "B" if model_name.endswith("_B") else "A"
            features = _feature_list(root, family)
            X = data.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(data["TOTALDEMAND"], errors="coerce")
            ok = X.notna().all(axis=1) & y.notna()
            X, y = X.loc[ok], y.loc[ok]
            if len(X) < 2000:
                raise RuntimeError(f"{region}/{model_name}: only {len(X)} complete training rows")
            medians = X.median(numeric_only=True).to_dict()

            # Deployment gate only (not model selection/backtesting): keep the latest
            # 14 days aside, compare the freshly retrained candidate with the current
            # production bundle on exactly those rows, and refuse a material regression.
            test_rows = min(14 * 48, max(96, len(X) // 10))
            split = len(X) - test_rows
            candidate_check = _make_model(model_name)
            candidate_check.fit(X.iloc[:split].to_numpy(dtype=np.float32),
                                y.iloc[:split].to_numpy(dtype=float))
            candidate_mae = float(mean_absolute_error(
                y.iloc[split:], candidate_check.predict(X.iloc[split:].to_numpy(dtype=np.float32))))
            current_pointer = MODEL_DIR / region / model_name / "CURRENT.txt"
            current_mae = np.nan
            if current_pointer.exists():
                try:
                    current_path = resolve_pointer(current_pointer)
                    if current_path is None:
                        raise FileNotFoundError(current_pointer.read_text().strip())
                    current_bundle = joblib.load(current_path)
                    current_model = current_bundle["estimator"]
                    current_features = current_bundle.get("feature_names", [])
                    if current_features == features:
                        current_mae = float(mean_absolute_error(
                            y.iloc[split:], current_model.predict(X.iloc[split:].to_numpy(dtype=np.float32))))
                except Exception as exc:
                    log.warning("Could not score current %s/%s bundle: %s", region, model_name, exc)
            promote = (not np.isfinite(current_mae)) or candidate_mae <= current_mae * 1.05
            if not promote:
                rows.append({
                    "region": region, "model_name": model_name, "model_version": version,
                    "feature_set": family,
                    "training_start": pd.to_datetime(data.loc[ok, "SETTLEMENTDATE"].min()),
                    "training_end": pd.to_datetime(data.loc[ok, "SETTLEMENTDATE"].max()),
                    "model_path": str(current_pointer.read_text().strip()),
                    "stage2_snapshot": str(root), "status": "REJECTED_RECENT_MAE",
                    "created_at": utcnow_naive(),
                })
                log.warning("Rejected %s/%s candidate: recent MAE %.2f vs production %.2f",
                            region, model_name, candidate_mae, current_mae)
                continue

            model = _make_model(model_name)
            model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=float))
            bundle = {
                "region": region,
                "model_name": model_name,
                "model_version": version,
                "feature_set": family,
                "feature_names": features,
                "feature_medians": medians,
                "training_start": str(data.loc[ok, "SETTLEMENTDATE"].min()),
                "training_end": str(data.loc[ok, "SETTLEMENTDATE"].max()),
                "stage2_root": str(root),
                "stage2_run_config": run_cfg,
                "deployment_gate_candidate_mae": candidate_mae,
                "deployment_gate_current_mae": current_mae,
                "estimator": model,
            }
            path = MODEL_DIR / region / model_name / version / "bundle.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path)
            current = MODEL_DIR / region / model_name / "CURRENT.txt"
            current.parent.mkdir(parents=True, exist_ok=True)
            # v3: pointers are written relative to MODEL_DIR (portable across Docker/local).
            current.write_text(relative_pointer_text(path), encoding="utf-8")
            rows.append({
                "region": region, "model_name": model_name, "model_version": version,
                "feature_set": family,
                "training_start": pd.to_datetime(bundle["training_start"]),
                "training_end": pd.to_datetime(bundle["training_end"]),
                "model_path": str(path), "stage2_snapshot": str(root),
                "status": "PRODUCTION", "created_at": utcnow_naive(),
            })
    registry = pd.DataFrame(rows)
    clear_caches()
    # Retire previous production rows for same region/model before inserting current ones.
    upsert_frame("model_registry", registry, ["region", "model_name", "model_version"])
    return registry


def run_monthly_retraining(regions=None, models=None, reuse_stage2: bool = False) -> pd.DataFrame:
    """Monthly production retraining. ``reuse_stage2`` trains from the current
    Stage 2 snapshot instead of rebuilding it (used for targeted artefact generation)."""
    if reuse_stage2:
        root = _resolve_stage2_root_text((STAGE2_DIR / "CURRENT_STAGE2.txt").read_text())
        log.info("Reusing current Stage 2 snapshot: %s", root)
    else:
        zip_path, root = prepare_monthly_stage2()
        log.info("Stage 2 monthly snapshot: %s", zip_path)
    return train_promoted_models(root, regions=regions, models=models)
