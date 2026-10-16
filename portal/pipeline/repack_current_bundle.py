from __future__ import annotations

import argparse
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover - compatibility with older sklearn
    InconsistentVersionWarning = Warning

from .config import MODEL_DIR, STAGE2_DIR
from .database import init_db, upsert_frame, utcnow_naive
from .live_inference import resolve_stage2_root
from .model_evidence import (
    PRODUCTION_ALLOWLIST,
    clear_caches,
    pointer_path,
    relative_pointer_text,
    resolve_pointer,
    validate_bundle_object,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("repack_current_bundle")


def _probe_matrix(bundle: dict, region: str, max_rows: int = 672) -> np.ndarray:
    """Build a deterministic prediction probe from Stage 2 when available.

    The repack is a persistence-format migration, not retraining.  The probe
    verifies that predictions before and after re-serialization are identical.
    """
    features = list(bundle["feature_names"])

    roots: list[Path] = []
    if bundle.get("stage2_root"):
        roots.append(resolve_stage2_root(bundle["stage2_root"]))

    current = STAGE2_DIR / "CURRENT_STAGE2.txt"
    if current.exists():
        try:
            roots.append(resolve_stage2_root(current.read_text(encoding="utf-8").strip()))
        except Exception:
            pass

    seen = set()
    for root in roots:
        root = Path(root)
        if str(root) in seen:
            continue
        seen.add(str(root))
        path = root / "data" / f"modelling_{region}_30min.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
            X = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
            X = X.dropna()
            if not X.empty:
                return X.tail(max_rows).to_numpy(dtype=np.float32)
        except Exception as exc:
            log.warning("Could not build Stage 2 prediction probe from %s: %s", path, exc)

    # Fallback: the saved training medians are part of the bundle contract and
    # provide a valid feature vector even if the historical Stage 2 snapshot
    # has moved.
    medians = bundle.get("feature_medians", {})
    row = [float(medians.get(name, 0.0)) for name in features]
    return np.asarray([row], dtype=np.float32)


def repack_current_bundle(region: str, model_name: str, probe_rows: int = 672) -> Path:
    """Re-serialize the CURRENT production bundle under this sklearn runtime.

    No estimator is fitted and no model parameters are changed.  CURRENT.txt
    is updated only after structural validation and a before/after prediction
    equivalence check pass.
    """
    if model_name not in PRODUCTION_ALLOWLIST:
        raise ValueError(
            f"{model_name!r} is not an allowed production model. "
            f"Allowed: {PRODUCTION_ALLOWLIST}"
        )

    pointer = pointer_path(region, model_name)
    source_path = resolve_pointer(pointer)
    if source_path is None:
        raise FileNotFoundError(
            f"CURRENT bundle for {region}/{model_name} does not resolve: {pointer}"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(source_path)

    version_warnings = [
        str(w.message) for w in caught
        if isinstance(w.message, InconsistentVersionWarning)
        or "Trying to unpickle estimator" in str(w.message)
    ]
    for message in version_warnings:
        log.warning("Expected source compatibility warning during migration: %s", message)

    problems = validate_bundle_object(bundle, model_name)
    if problems:
        raise ValueError(
            f"Source bundle failed structural validation: {'; '.join(problems)}"
        )

    estimator = bundle["estimator"]
    X_probe = _probe_matrix(bundle, region, max_rows=probe_rows)
    before = np.asarray(estimator.predict(X_probe), dtype=float)

    old_version = str(bundle.get("model_version") or source_path.parent.name)
    runtime_tag = sklearn.__version__.replace(".", "_")
    new_version = (
        f"{old_version}_compat_skl{runtime_tag}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )

    migrated = dict(bundle)
    migrated["model_version"] = new_version
    migrated["artifact_migration"] = {
        "type": "sklearn_compatibility_repack",
        "source_path": str(source_path),
        "source_model_version": old_version,
        "runtime_sklearn_version": sklearn.__version__,
        "created_at": utcnow_naive().isoformat(),
        "estimator_retrained": False,
        "probe_rows": int(len(X_probe)),
    }

    target_dir = MODEL_DIR / region / model_name / new_version
    target_dir.mkdir(parents=True, exist_ok=False)
    pending = target_dir / "bundle.joblib.pending"
    final = target_dir / "bundle.joblib"

    try:
        joblib.dump(migrated, pending)

        # Reload the newly serialized artifact under the current runtime.
        with warnings.catch_warnings(record=True) as reloaded_warnings:
            warnings.simplefilter("always")
            reloaded = joblib.load(pending)

        new_version_warnings = [
            str(w.message) for w in reloaded_warnings
            if isinstance(w.message, InconsistentVersionWarning)
            or "Trying to unpickle estimator" in str(w.message)
        ]
        if new_version_warnings:
            raise RuntimeError(
                "Newly repacked bundle still emits sklearn version warning(s): "
                + " | ".join(new_version_warnings)
            )

        problems = validate_bundle_object(reloaded, model_name)
        if problems:
            raise ValueError(
                "Repacked bundle failed structural validation: " + "; ".join(problems)
            )

        after = np.asarray(reloaded["estimator"].predict(X_probe), dtype=float)
        if before.shape != after.shape or not np.allclose(
            before, after, rtol=0.0, atol=1e-12, equal_nan=True
        ):
            max_diff = float(np.nanmax(np.abs(before - after)))
            raise RuntimeError(
                f"Prediction equivalence check failed; max_abs_diff={max_diff}"
            )

        max_diff = float(np.nanmax(np.abs(before - after))) if before.size else 0.0
        reloaded["artifact_migration"]["prediction_max_abs_diff"] = max_diff
        # Persist the completed audit metadata once more.
        joblib.dump(reloaded, pending)
        os.replace(pending, final)

        # Atomic-ish pointer handoff: CURRENT remains on the old model until all
        # validation above has succeeded.
        tmp_pointer = pointer.with_suffix(".txt.tmp")
        tmp_pointer.write_text(relative_pointer_text(final), encoding="utf-8")
        os.replace(tmp_pointer, pointer)

        init_db()
        row = pd.DataFrame([{
            "region": region,
            "model_name": model_name,
            "model_version": new_version,
            "feature_set": str(reloaded.get("feature_set", "")),
            "training_start": pd.to_datetime(reloaded.get("training_start"), errors="coerce"),
            "training_end": pd.to_datetime(reloaded.get("training_end"), errors="coerce"),
            "model_path": str(final),
            "stage2_snapshot": str(reloaded.get("stage2_root", "")),
            "status": "PRODUCTION",
            "created_at": utcnow_naive(),
        }])
        upsert_frame(
            "model_registry",
            row,
            ["region", "model_name", "model_version"],
        )
        clear_caches()

        log.info(
            "%s/%s repacked without retraining: %s -> %s; "
            "sklearn=%s; probe_rows=%s; max_abs_diff=%.12g",
            region,
            model_name,
            old_version,
            new_version,
            sklearn.__version__,
            len(X_probe),
            max_diff,
        )
        return final

    except Exception:
        # Keep the original CURRENT pointer untouched on any failure.
        try:
            if pending.exists():
                pending.unlink()
            if target_dir.exists() and not any(target_dir.iterdir()):
                target_dir.rmdir()
        except Exception:
            pass
        raise


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Re-serialize an existing production bundle under the current "
            "scikit-learn runtime without retraining it."
        )
    )
    ap.add_argument("--region", required=True)
    ap.add_argument("--model", required=True, choices=PRODUCTION_ALLOWLIST)
    ap.add_argument("--probe-rows", type=int, default=672)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = repack_current_bundle(args.region, args.model, args.probe_rows)
    print(f"Repacked bundle: {path}")
    print(f"scikit-learn runtime: {sklearn.__version__}")
    print("Estimator retrained: no")
    print("CURRENT.txt updated only after prediction-equivalence validation: yes")


if __name__ == "__main__":
    main()
