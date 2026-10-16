"""Model-evidence layer and production-model resolver (portal v3.0.0).

Two responsibilities, kept deliberately separate:

1. EVIDENCE — read the curated WP4 model-evaluation files shipped in
   ``data/model_evidence/current/`` (verified against ``evidence_manifest.json``)
   and cache them. Evidence is read once per process / manifest version, never on
   the 60-second UI refresh.

2. PRODUCTION RESOLUTION — decide which Random Forest bundle serves each region.
   A model is served only when ALL of the following hold:
     a) it is selected by ``selected_model.json`` (the selection authority),
     b) it is in the production allow-list (RandomForest_A / RandomForest_B),
     c) a valid portal-format bundle exists under MODEL_DIR.
   Otherwise the explicit fallback is served and the gap is reported, never hidden.

WP4 ``.joblib`` files are bare estimators, not portal bundles; they are never
loaded here. Nothing in this module trains, promotes or deploys a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    FALLBACK_DEFAULT_MODEL,
    FALLBACK_PRODUCTION_MODELS,
    MODEL_DIR,
    PORTAL_ROLE,
    REGIONS,
    RUNTIME_DIR,
)

log = logging.getLogger(__name__)

EVIDENCE_DIR = Path(DATA_DIR) / "model_evidence" / "current"
BUNDLE_VALIDATION_FILE = Path(RUNTIME_DIR) / "bundle_validation.json"

PRODUCTION_ALLOWLIST = ("RandomForest_A", "RandomForest_B")
CORE_MODELS = ("Naive48", "Naive336", "RandomForest_A", "RandomForest_B", "XGBoost_A",
               "XGBoost_B", "SARIMA", "SARIMAX", "MLP", "GRU")
NON_SELECTION_EXPERIMENTS = {"GRU_L672": "Sensitivity experiment (14-day GRU lookback)",
                             "XGBoost_A+CoreWeather": "Weather ablation experiment"}
MODEL_INFO = {
    "Naive48": ("Baseline", "previous"), "Naive336": ("Baseline", "previous"),
    "RandomForest_A": ("Tree ensemble", "previous"), "RandomForest_B": ("Tree ensemble", "previous"),
    "XGBoost_A": ("Gradient boosting", "previous"), "XGBoost_B": ("Gradient boosting", "previous"),
    "SARIMA": ("Statistical", "previous"), "SARIMAX": ("Statistical", "previous"),
    "MLP": ("Neural network", "new"), "GRU": ("Deep sequence", "new"),
}
FEATURE_SET_LABEL = {"A": "A · core", "B": "B · weather-enhanced"}
REQUIRED_BUNDLE_KEYS = ("estimator", "feature_names", "feature_medians", "feature_set",
                        "model_version", "training_start", "training_end")


def model_feature_set(model_name: str) -> str | None:
    if model_name.endswith("_A"):
        return "A"
    if model_name.endswith("_B"):
        return "B"
    return None


def model_label(model_name: str) -> str:
    fs = model_feature_set(model_name)
    return f"{model_name} · {FEATURE_SET_LABEL[fs].split(' · ')[1]}" if fs else model_name


# =============================================================================
# 1. Evidence
# =============================================================================

@dataclass
class EvidenceStatus:
    state: str                      # OK | EVIDENCE_MISSING | EVIDENCE_UNVERIFIED
    version: str = ""
    pipeline_version: str = ""
    generated: str = ""
    selection_origins: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state == "OK"

    def footer(self) -> str:
        if self.state == "EVIDENCE_MISSING":
            return "Model evaluation evidence not available"
        gen = pd.to_datetime(self.generated, errors="coerce")
        when = f"{gen:%d %b %Y}" if pd.notna(gen) else self.version
        years = "–".join([self.selection_origins[0][:4], self.selection_origins[-1][:4]]) if self.selection_origins else ""
        text = f"Model evaluation {when} · Pipeline v{self.pipeline_version} · Selection periods {years} matched backtests"
        return text + (" · evidence unverified" if self.state == "EVIDENCE_UNVERIFIED" else "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_key(evidence_dir: Path) -> tuple:
    m = evidence_dir / "evidence_manifest.json"
    if not m.exists():
        return (str(evidence_dir), 0.0, 0)
    st = m.stat()
    return (str(evidence_dir), st.st_mtime, st.st_size)


@lru_cache(maxsize=8)
def _load_manifest_cached(key: tuple) -> tuple[dict | None, EvidenceStatus]:
    evidence_dir = Path(key[0])
    m = evidence_dir / "evidence_manifest.json"
    if not m.exists():
        return None, EvidenceStatus("EVIDENCE_MISSING", problems=[f"{m} not found"])
    try:
        manifest = json.loads(m.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt manifest
        return None, EvidenceStatus("EVIDENCE_UNVERIFIED", problems=[f"manifest unreadable: {exc}"])
    problems = []
    for name, meta in manifest.get("files", {}).items():
        p = evidence_dir / name
        if not p.exists():
            problems.append(f"missing {name}")
        elif meta.get("sha256") and _sha256(p) != meta["sha256"]:
            problems.append(f"hash mismatch {name}")
    status = EvidenceStatus(
        "EVIDENCE_UNVERIFIED" if problems else "OK",
        version=str(manifest.get("evidence_version", "")),
        pipeline_version=str(manifest.get("pipeline_version", "")),
        generated=str(manifest.get("selected_model_generated", "")),
        selection_origins=list(manifest.get("selection_origins", [])),
        problems=problems,
    )
    return manifest, status


def load_manifest(evidence_dir: Path | None = None) -> tuple[dict | None, EvidenceStatus]:
    return _load_manifest_cached(_manifest_key(Path(evidence_dir or EVIDENCE_DIR)))


@lru_cache(maxsize=64)
def _read_csv_cached(key: tuple, name: str) -> pd.DataFrame:
    p = Path(key[0]) / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as exc:
        log.warning("Evidence file %s unreadable: %s", name, exc)
        return pd.DataFrame()


def evidence(name: str, evidence_dir: Path | None = None) -> pd.DataFrame:
    """Cached evidence CSV (a copy, so callers cannot mutate the cache)."""
    return _read_csv_cached(_manifest_key(Path(evidence_dir or EVIDENCE_DIR)), name).copy()


@lru_cache(maxsize=8)
def _selected_cached(key: tuple) -> dict:
    p = Path(key[0]) / "selected_model.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("selected_model.json unreadable: %s", exc)
        return {}
    out = {}
    for r in data.get("regions", []):
        ev = r.get("backtest_evidence", {}) or {}
        out[r["region"]] = {
            "model": r.get("selected_model"),
            "feature_set": r.get("feature_set"),
            "family": r.get("model_family"),
            "development_period": r.get("development_period"),
            "runner_up": ev.get("runner_up"),
            "margin_pct": ev.get("MAE_difference_pct"),
            "training_cutoff": r.get("training_cutoff"),
        }
    return out


def selected_models(evidence_dir: Path | None = None) -> dict:
    return dict(_selected_cached(_manifest_key(Path(evidence_dir or EVIDENCE_DIR))))


def clear_caches() -> None:
    for f in (_load_manifest_cached, _read_csv_cached, _selected_cached):
        f.cache_clear()
    _PROD_CACHE.clear()


# =============================================================================
# 2. Bundles
# =============================================================================

def resolve_pointer(pointer: Path, model_dir: Path | None = None) -> Path | None:
    """Resolve a CURRENT.txt pointer portably.

    1. the path as written (absolute Docker path or absolute local path);
    2. relative to MODEL_DIR (new pointers are written relative);
    3. legacy absolute pointers rebased from their ``models/`` segment onto MODEL_DIR
       (e.g. ``/app/models/NSW1/...`` -> ``<MODEL_DIR>/NSW1/...``).
    """
    model_dir = Path(model_dir or MODEL_DIR)
    if not pointer.exists():
        return None
    raw = pointer.read_text(encoding="utf-8").strip().replace("\\", "/")
    if not raw:
        return None
    candidates = [Path(raw), model_dir / raw]
    parts = raw.split("/")
    if "models" in parts:
        idx = len(parts) - 1 - parts[::-1].index("models")
        candidates.append(model_dir.joinpath(*parts[idx + 1:]))
    for c in candidates:
        if c.is_file():
            return c
    return None


def pointer_path(region: str, model_name: str, model_dir: Path | None = None) -> Path:
    return Path(model_dir or MODEL_DIR) / region / model_name / "CURRENT.txt"


def relative_pointer_text(bundle_path: Path, model_dir: Path | None = None) -> str:
    """Pointer text for newly created bundles: relative to MODEL_DIR when possible."""
    model_dir = Path(model_dir or MODEL_DIR).resolve()
    try:
        return Path(bundle_path).resolve().relative_to(model_dir).as_posix()
    except ValueError:
        return str(bundle_path)


def validate_bundle_object(bundle, model_name: str) -> list[str]:
    """Structural checks on a loaded portal bundle. Empty list = valid."""
    problems = []
    if not isinstance(bundle, dict):
        return [f"bundle is {type(bundle).__name__}, not a dict (WP4 bare estimators are not portal bundles)"]
    for k in REQUIRED_BUNDLE_KEYS:
        if k not in bundle:
            problems.append(f"missing key {k}")
    if problems:
        return problems
    if not hasattr(bundle["estimator"], "predict"):
        problems.append("estimator has no predict()")
    if not isinstance(bundle["feature_names"], (list, tuple)) or not bundle["feature_names"]:
        problems.append("feature_names empty")
    if not isinstance(bundle["feature_medians"], dict):
        problems.append("feature_medians is not a dict")
    expected = model_feature_set(model_name)
    if expected and str(bundle.get("feature_set")) != expected:
        problems.append(f"feature_set {bundle.get('feature_set')!r} != expected {expected!r}")
    if bundle.get("model_name") not in (None, model_name):
        problems.append(f"model_name {bundle.get('model_name')!r} != {model_name!r}")
    n_features = getattr(bundle["estimator"], "n_features_in_", None)
    if n_features is not None and len(bundle["feature_names"]) != n_features:
        problems.append(f"estimator expects {n_features} features, bundle lists {len(bundle['feature_names'])}")
    return problems


def _recorded_validation() -> dict:
    if BUNDLE_VALIDATION_FILE.exists():
        try:
            return json.loads(BUNDLE_VALIDATION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def bundle_available(region: str, model_name: str, model_dir: Path | None = None,
                     validation: dict | None = None) -> tuple[bool, str, Path | None]:
    """Cheap availability check used by the resolver (no 260 MB load).

    A bundle is available when its pointer resolves to an existing non-empty file and
    no recorded deep validation (``validate-bundles``) marked that same file invalid.
    """
    path = resolve_pointer(pointer_path(region, model_name, model_dir), model_dir)
    if path is None:
        return False, "no portal bundle (CURRENT.txt missing or unresolvable)", None
    if path.stat().st_size == 0:
        return False, "bundle file is empty", path
    rec = (validation if validation is not None else _recorded_validation()).get(f"{region}/{model_name}")
    if rec and rec.get("path") == str(path) and rec.get("mtime") == path.stat().st_mtime and not rec.get("valid"):
        return False, "bundle failed validation: " + "; ".join(rec.get("problems", [])), path
    return True, "available", path


def validate_bundles(model_dir: Path | None = None, regions=None, models=PRODUCTION_ALLOWLIST,
                     write: bool = True) -> dict:
    """Deep validation (loads each bundle). Run from the worker, not the web process."""
    import joblib
    results = {}
    for region in regions or REGIONS:
        for model_name in models:
            path = resolve_pointer(pointer_path(region, model_name, model_dir), model_dir)
            if path is None:
                continue
            try:
                problems = validate_bundle_object(joblib.load(path), model_name)
            except Exception as exc:
                problems = [f"load failed: {type(exc).__name__}: {exc}"]
            results[f"{region}/{model_name}"] = {"path": str(path), "mtime": path.stat().st_mtime,
                                                 "valid": not problems, "problems": problems,
                                                 "checked_at": pd.Timestamp.utcnow().isoformat()}
    if write:
        BUNDLE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = _recorded_validation()
        existing.update(results)
        BUNDLE_VALIDATION_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    _PROD_CACHE.clear()
    return results


# =============================================================================
# 3. Production resolver
# =============================================================================

@dataclass
class ServingStatus:
    region: str
    selected: str | None           # evidence selection
    served: str                    # model the scheduled jobs forecast with by default
    available: list                # production bundles that exist (served first)
    state: str                     # SERVING_SELECTED | SELECTED_ARTEFACT_MISSING | SELECTED_NOT_ALLOWED | EVIDENCE_MISSING | NO_BUNDLE
    message: str = ""

    @property
    def mismatch(self) -> bool:
        return self.selected is not None and self.selected != self.served


@dataclass
class ProductionConfig:
    production_models: dict        # region -> [served, other available RF bundles]
    default_model: dict            # region -> served model
    training_models: dict          # region -> models the retraining job maintains
    status: dict                   # region -> ServingStatus

    def to_dict(self) -> dict:
        return {"production_models": self.production_models, "default_model": self.default_model,
                "training_models": self.training_models,
                "status": {r: asdict(s) for r, s in self.status.items()}}


def resolve_production_config(selection: dict | None = None, model_dir: Path | None = None,
                              fallback_models: dict | None = None, fallback_default: dict | None = None,
                              validation: dict | None = None) -> ProductionConfig:
    selection = selected_models() if selection is None else selection
    fallback_models = fallback_models or FALLBACK_PRODUCTION_MODELS
    fallback_default = fallback_default or FALLBACK_DEFAULT_MODEL
    prod, default, training, status = {}, {}, {}, {}
    for region in REGIONS:
        sel = (selection.get(region) or {}).get("model")
        intended = [m for m in fallback_models.get(region, []) if m in PRODUCTION_ALLOWLIST]
        if sel in PRODUCTION_ALLOWLIST and sel not in intended:
            intended = [sel] + intended
        training[region] = list(dict.fromkeys(intended))

        # The Railway web service is intentionally model-file-free. Forecasts are
        # generated by the worker service and persisted in PostgreSQL, so the web
        # process resolves the evidence-selected production model without requiring
        # a local 250+ MB bundle copy. Local/worker roles still enforce on-disk
        # bundle availability exactly as before.
        if PORTAL_ROLE == "web":
            fb = fallback_default.get(region)
            if sel in PRODUCTION_ALLOWLIST:
                served = sel
                state = "SERVING_SELECTED"
                msg = "Forecast execution is provided by the production worker service."
            elif sel:
                served = fb or (intended[0] if intended else "RandomForest_A")
                state = "SELECTED_NOT_ALLOWED"
                msg = f"{sel} is not an allowed production model; worker configuration uses {served}."
            else:
                served = fb or (intended[0] if intended else "RandomForest_A")
                state = "EVIDENCE_MISSING"
                msg = "Selection evidence unavailable; using the configured production mapping."
            prod[region] = [served]
            default[region] = served
            status[region] = ServingStatus(region, sel, served, [served], state, msg)
            continue

        avail = {m: bundle_available(region, m, model_dir, validation) for m in training[region]}
        ok = [m for m in training[region] if avail[m][0]]
        fb = fallback_default.get(region)
        if not sel:
            served = fb if fb in ok else (ok[0] if ok else fb)
            state, msg = "EVIDENCE_MISSING", "Selection evidence unavailable — serving the configured fallback."
        elif sel not in PRODUCTION_ALLOWLIST:
            served = fb if fb in ok else (ok[0] if ok else fb)
            state, msg = "SELECTED_NOT_ALLOWED", f"{sel} is not an allowed production model — serving {served}."
        elif avail[sel][0]:
            served, state, msg = sel, "SERVING_SELECTED", ""
        else:
            others = [m for m in ok if m != sel]
            served = others[0] if others else (fb or sel)
            state = "SELECTED_ARTEFACT_MISSING" if others else "NO_BUNDLE"
            msg = (f"Selected model artefact not yet available — serving {served}." if others
                   else f"No production bundle available for {region}.")
        prod[region] = [served] + [m for m in ok if m != served] if served in ok else ok
        default[region] = served
        status[region] = ServingStatus(region, sel, served, prod[region], state, msg)
    return ProductionConfig(prod, default, training, status)


_PROD_CACHE: dict = {}


def production_config(max_age_seconds: int = 300) -> ProductionConfig:
    """Cached resolution for long-running processes (web). Jobs call it once per run."""
    now = time.time()
    hit = _PROD_CACHE.get("cfg")
    if hit and now - hit[0] < max_age_seconds:
        return hit[1]
    cfg = resolve_production_config()
    _PROD_CACHE["cfg"] = (now, cfg)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence / production-resolver utilities")
    ap.add_argument("command", choices=["status", "validate-bundles"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    _, st = load_manifest()
    print(f"Evidence: {st.state} {st.version} {st.problems or ''}")
    if a.command == "validate-bundles":
        for k, v in validate_bundles().items():
            print(f"{k}: {'VALID' if v['valid'] else 'INVALID'} {v['problems'] or ''}")
    cfg = resolve_production_config()
    for r, s in cfg.status.items():
        print(f"{r}: selected={s.selected} served={s.served} available={s.available} state={s.state} {s.message}")


if __name__ == "__main__":
    main()
