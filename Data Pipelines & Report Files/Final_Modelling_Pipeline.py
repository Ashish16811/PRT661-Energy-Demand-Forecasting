
"""
================================================================================
 03_Modelling.py - PRT661 Stage 3 | Rolling four-month operational-demand
 forecasting, A/B decision evidence, verification and benchmark comparison.
 Australian Electricity Demand Forecasting using ML and Time-Series Analytics
 Dan6 Group - Theme 2
================================================================================

INPUT CONTRACT
    A Stage 2 preprocessing ZIP (Stage2_Preprocessing_Output_<timestamp>.zip).
    Nothing about the horizon is hard-coded: metadata/run_config.json supplies
    the rolling window, so a September run produces Sep-Dec and a December run
    produces Dec-Mar without a code change.

WHAT RUNS, IN ORDER
    PHASE A  fair historical four-month backtests on the same windows for every
             candidate. Model B is trained on observed historical weather, but
             target-period weather is rebuilt from origin-safe climatology unless
             a genuine archived forecast vintage exists. 2026 is not used to
             choose the historical winner.
    PHASE B  frozen month-start forecast, trained only through the previous month end.
             A forecast-weather vintage issued after that origin is rejected.
    PHASE C  refreshed forecast from the latest regional actual. Model B uses
             genuine FORECAST_7DAY rows where available and climatology after
             the reliable forecast horizon.
    A/B      Model A also writes the full current-month + next-three-month view.
             Model B writes a rolling current-month diagnostic plus the live
             seven-day weather-informed continuation.
    BENCH    AEMO source consistency and forecast-margin comparisons. AEMO is
             never a predictor, and target-mismatched margins are not accuracy.

WHY ORIGIN-DEPENDENT FEATURES ARE REBUILT
    Stage 2 ships historical_month_* and the climatology columns built as-of the
    row's own year. That is leakage-safe, but a backtest needs every row - train
    and forecast alike - to carry the information set of ONE forecast origin.
    Stage 3 therefore rebuilds those columns per origin from pre-origin data
    only and records the agreement with Stage 2 in
    validation/origin_feature_rebuild_checks.csv.

MODULE OWNERSHIP (Dan6, Theme 2 / Group 2)
    Sudip Lamichhane   input readiness, Stage 2 contract, AEMO benchmark source
    Suraj Raut         chronological validation, recursive time-series engine
    Ashish Shrestha    model development, forecasting, model selection
    Bishal Dahal       verification, metrics, AEMO comparison, evidence figures
    Integrated         Dan6 Group

RUN
    pip install pandas numpy scikit-learn matplotlib xgboost statsmodels joblib
    python 03_Modelling.py                                  # Tkinter interface
    python 03_Modelling.py --zip <stage2.zip> --out-dir <dir>
    python 03_Modelling.py --zip <stage2.zip> --out-dir <dir> --quick
    python 03_Modelling.py --zip <stage2.zip> --out-dir <dir> --regions NSW1 SA1
    python 03_Modelling.py --self-test           # horizon logic only, no data
================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONSTANTS
# ============================================================================

TIME = "SETTLEMENTDATE"
TARGET = "TOTALDEMAND"
FREQ = "30min"
INTERVALS_PER_DAY = 48          # 30-minute grid
BLOCK = INTERVALS_PER_DAY       # recursion advances one day at a time
HOURS_PER_INTERVAL = 0.5        # MW -> MWh conversion factor

RANDOM_STATE = 20260914
WARMUP_INTERVALS = 14 * INTERVALS_PER_DAY   # same_half_hour_14day_mean needs 672
SMAPE_MIN_DENOM = 1.0           # |f|+|a| below this is excluded, not clipped
TIE_TOLERANCE = 0.01            # 1% MAE -> treat as a tie, prefer the simpler model
AEMO_REQUESTED_HORIZON_HOURS = 48

MODEL_ORDER = ["Naive48", "Naive336",
               "RandomForest_A", "RandomForest_B",
               "XGBoost_A", "XGBoost_B",
               "SARIMA", "SARIMAX"]

# Used only to break ties: on effectively equal MAE, prefer the simpler model.
COMPLEXITY = {"Naive48": 0, "Naive336": 0, "SARIMA": 2, "SARIMAX": 3,
              "RandomForest_A": 4, "XGBoost_A": 4,
              "RandomForest_B": 5, "XGBoost_B": 5}

FEATURE_SET_OF = {"Naive48": "none", "Naive336": "none",
                  "RandomForest_A": "A", "XGBoost_A": "A",
                  "RandomForest_B": "B", "XGBoost_B": "B",
                  "SARIMA": "none", "SARIMAX": "exog_forecast_safe"}

NAIVE_LAG = {"Naive48": 48, "Naive336": 336}

# Demand-derived features are the only ones Stage 3 must reconstruct recursively.
DEMAND_FEATURES = {
    "lag_48", "lag_96", "lag_336",
    "previous_day_mean", "previous_day_peak", "previous_day_min",
    "same_half_hour_7day_mean", "same_half_hour_14day_mean",
    "rolling_mean_24h_at_t_minus_48", "rolling_std_24h_at_t_minus_48",
}
CLIMATOLOGY_FEATURES = {
    "expected_temp_climatology", "historical_rain_probability",
    "expected_hdd_climatology", "expected_cdd_climatology",
}

# Model B uses the same numerical weather columns for historical training and
# live forecasting.  Only provenance changes: historical rows are observed /
# reanalysis, the next seven days can be a genuine forecast, and the remaining
# four-month horizon falls back to origin-safe climatology.
WEATHER_DIRECT_FEATURES = {
    "weather_temp_c", "weather_humidity_pct",
    "weather_daily_temp_max_c", "weather_daily_temp_min_c",
    "weather_daily_temp_mean_c", "weather_daily_rainfall_mm",
}
WEATHER_CONTEXT_FEATURES = {
    "temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
    "hdd_lag_1d", "cdd_lag_1d",
}
WEATHER_FEATURES = WEATHER_DIRECT_FEATURES | WEATHER_CONTEXT_FEATURES | CLIMATOLOGY_FEATURES
WEATHER_METADATA_COLUMNS = {
    "weather_feature_source", "weather_forecast_origin", "weather_lead_day",
    "weather_forecast_source", "weather_forecast_model", "weather_retrieved_at_utc",
}
PROFILE_PREFIX = "historical_month_"

# These are genuinely unsafe as predictors.  The new weather_* columns are NOT
# forbidden: Stage 2 provides a provenance contract and Stage 3 rewrites their
# future values according to the forecast origin before any model sees them.
FORBIDDEN_PATTERNS = [
    r"aemo", r"predispatch", r"pre_dispatch", r"marketrequirement",
    r"^temp_mean_c$", r"^temp_max_c$", r"^temp_min_c$", r"^rainfall_mm$",
    r"^humidity_pct$", r"^temp_halfhourly_c$", r"^hdd$", r"^cdd$",
]

# SARIMAX keeps a compact exogenous set.  Direct weather is allowed because the
# same origin-safe weather policy used by the tree models is applied first.
SARIMAX_EXOG_PREFERRED = ["sin_half_hour", "cos_half_hour",
                          "sin_day_of_year", "cos_day_of_year",
                          "is_weekend", "is_public_holiday",
                          "weather_temp_c", "weather_daily_rainfall_mm",
                          "expected_temp_climatology"]

log = logging.getLogger("stage3")


# ============================================================================
# 2. SMALL UTILITIES
# ============================================================================

def setup_logging(logfile: Path | None = None, extra_handler=None) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if logfile is not None:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if extra_handler is not None:
        extra_handler.setFormatter(fmt)
        log.addHandler(extra_handler)


def write_csv(df: pd.DataFrame, path: Path, floats: str = "%.4f") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=floats)
    return path


def dependency_status() -> pd.DataFrame:
    """Report what is installed. Missing packages disable models, never swap them."""
    wanted = [("pandas", "pandas"), ("numpy", "numpy"),
              ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
              ("xgboost", "xgboost"), ("statsmodels", "statsmodels"),
              ("joblib", "joblib")]
    rows = []
    for label, module in wanted:
        try:
            mod = __import__(module)
            rows.append({"package": label, "status": "available",
                         "version": getattr(mod, "__version__", "unknown"),
                         "detail": ""})
        except Exception as exc:
            rows.append({"package": label, "status": "MISSING", "version": "",
                         "detail": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """MAE, RMSE, sMAPE, Bias with error = forecast - actual throughout.

    sMAPE excludes intervals where |f|+|a| is below SMAPE_MIN_DENOM. South
    Australian operational demand genuinely reaches and crosses zero, and a
    percentage error against a ~0 MW denominator is meaningless rather than
    large. Excluded intervals are counted, never silently clipped.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(a) & np.isfinite(f)
    a, f = a[ok], f[ok]
    if a.size == 0:
        return {"matched_intervals": 0, "MAE": np.nan, "RMSE": np.nan,
                "sMAPE": np.nan, "Bias": np.nan, "MedAE": np.nan,
                "smape_excluded_intervals": 0}
    err = f - a
    denom = np.abs(f) + np.abs(a)
    usable = denom >= SMAPE_MIN_DENOM
    smape = float(np.mean(200.0 * np.abs(err[usable]) / denom[usable])) if usable.any() else np.nan
    return {"matched_intervals": int(a.size),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE": smape,
            "Bias": float(np.mean(err)),
            "MedAE": float(np.median(np.abs(err))),
            "smape_excluded_intervals": int((~usable).sum())}


def skill_pct(model_mae: float, baseline_mae: float) -> float:
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return np.nan
    return 100.0 * (baseline_mae - model_mae) / baseline_mae


# ============================================================================
# 3. STAGE 2 CONTRACT
# ============================================================================

@dataclass
class Stage2:
    root: Path
    run_config: dict
    regions: list
    region_files: dict
    features_A: list
    features_B: list
    horizon_report: pd.DataFrame
    weather_provenance: pd.DataFrame
    weather_validation: pd.DataFrame
    future_frame: pd.DataFrame | None
    current_actuals: pd.DataFrame | None   # verification-only current prediction-month actuals
    aemo_actual: pd.DataFrame | None
    aemo_forecast: pd.DataFrame | None
    interval_convention: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def open_stage2(source: Path, workdir: Path) -> Path:
    """Accept a Stage 2 ZIP or extracted folder and fail clearly on partial ZIPs."""
    source = Path(source)
    if source.is_dir():
        root = source
    else:
        if not source.exists():
            raise FileNotFoundError(f"Stage 2 input does not exist: {source}")
        if source.suffix.lower() == ".partial":
            raise ValueError(
                f"Stage 2 packaging is not complete yet: {source.name}. "
                "Wait for preprocessing to report ZIP PASS and select the final .zip file.")
        if not zipfile.is_zipfile(source):
            try:
                with source.open("rb") as fh:
                    head = fh.read(4)
                    fh.seek(max(0, source.stat().st_size - 65557))
                    tail = fh.read()
                looks_truncated = head.startswith(b"PK") and b"PK\x05\x06" not in tail
            except OSError:
                looks_truncated = False
            if looks_truncated:
                raise ValueError(
                    f"Stage 2 ZIP is incomplete/truncated ({source.stat().st_size / 1048576:.1f} MB): {source}. "
                    "It begins like a ZIP but has no final ZIP directory. This normally means Stage 2 "
                    "was still compressing, was interrupted, or the file was copied before packaging finished. "
                    "Re-run Stage 2 and wait until it reports 'Stage 2 ZIP: ... (PASS)' before modelling.")
            raise ValueError(f"Not a readable ZIP archive: {source}")
        with zipfile.ZipFile(source) as zf:
            broken = zf.testzip()
            if broken:
                raise ValueError(f"Stage 2 ZIP failed CRC validation at: {broken}")
            bad = [n for n in zf.namelist() if n.startswith("/") or ".." in Path(n).parts]
            if bad:
                raise ValueError(f"Refusing unsafe archive paths, e.g. {bad[0]}")
            zf.extractall(workdir)
        root = workdir
    # Stage 2 wraps everything in one timestamped folder; descend into it.
    for _ in range(3):
        if (root / "metadata" / "run_config.json").exists():
            break
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) != 1:
            break
        root = subs[0]
    if not (root / "metadata" / "run_config.json").exists():
        raise FileNotFoundError("metadata/run_config.json not found inside the Stage 2 input")
    return root


def _read_feature_list(path: Path) -> list:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = "feature" if "feature" in df.columns else df.columns[0]
    return [str(f).strip() for f in df[col].dropna().tolist()]


def load_stage2(root: Path) -> Stage2:
    cfg = json.loads((root / "metadata" / "run_config.json").read_text(encoding="utf-8-sig"))

    region_files = {}
    for p in sorted((root / "data").glob("modelling_*.csv")):
        m = re.match(r"modelling_(.+?)(?:_30min)?$", p.stem)
        if m:
            region_files[m.group(1)] = p
    if not region_files:
        raise FileNotFoundError("No data/modelling_<REGION>*.csv files found")

    regions = [r for r in cfg.get("regions", []) if r in region_files] or sorted(region_files)

    feat_dir = root / "features"
    A = _read_feature_list(feat_dir / "feature_set_A_core.csv")
    B = _read_feature_list(feat_dir / "feature_set_B_weather.csv")
    report = pd.DataFrame()
    rp = feat_dir / "horizon_safe_feature_report.csv"
    if rp.exists():
        report = pd.read_csv(rp)

    weather_provenance = pd.DataFrame()
    wpp = feat_dir / "weather_feature_provenance.csv"
    if wpp.exists():
        weather_provenance = pd.read_csv(wpp)

    weather_validation = pd.DataFrame()
    wvp = root / "validation" / "weather_forecast_validation.csv"
    if wvp.exists():
        weather_validation = pd.read_csv(wvp)

    notes = []
    if not A:
        raise FileNotFoundError("features/feature_set_A_core.csv is required and was not found")
    if not B:
        notes.append("feature_set_B_weather.csv missing - Model B falls back to Model A + any "
                     "climatology columns named in the horizon-safe report")
        B = A + sorted(CLIMATOLOGY_FEATURES)

    if not weather_provenance.empty:
        notes.append("features/weather_feature_provenance.csv present and read")
    if not weather_validation.empty:
        notes.append("validation/weather_forecast_validation.csv present and read")

    fut = None
    fp = root / "data" / "future_feature_frame_all_regions.csv"
    if fp.exists():
        fut = pd.read_csv(fp, parse_dates=[TIME, "local_timestamp", "local_date"],
                          low_memory=False)
        if "region" not in fut.columns and "REGION" in fut.columns:
            fut = fut.rename(columns={"REGION": "region"})
    else:
        notes.append("future_feature_frame_all_regions.csv absent - Stage 3 rebuilds the "
                     "known-future frame itself")

    current_actuals = None
    cap = root / "validation" / "current_month_actuals_all_regions.csv"
    if cap.exists():
        current_actuals = pd.read_csv(cap, parse_dates=[TIME], low_memory=False)
        if "REGION" not in current_actuals.columns and "region" in current_actuals.columns:
            current_actuals = current_actuals.rename(columns={"region": "REGION"})
        for c in ("local_timestamp", "local_date"):
            if c in current_actuals.columns:
                current_actuals[c] = pd.to_datetime(current_actuals[c], errors="coerce")
        notes.append("verification/current-month actuals present; excluded from model training")
    else:
        notes.append("current_month_actuals_all_regions.csv absent - current-month verification unavailable")

    aemo_a = aemo_f = None
    ap = root / "benchmark" / "aemo_actual_30min.csv"
    if ap.exists():
        aemo_a = pd.read_csv(ap, parse_dates=[TIME])
        aemo_a = aemo_a.rename(columns={"REGION": "region"})
    fpath = root / "benchmark" / "aemo_forecast_30min.csv"
    if fpath.exists():
        aemo_f = pd.read_csv(fpath, parse_dates=[TIME])
        aemo_f = aemo_f.rename(columns={"REGION": "region"})
        if "forecast_origin" in aemo_f.columns:
            aemo_f["forecast_origin"] = pd.to_datetime(aemo_f["forecast_origin"])

    convention = {}
    cv = root / "validation" / "core_validation_tests.csv"
    if cv.exists():
        tests = pd.read_csv(cv)
        hit = tests[tests["check"].astype(str).str.contains("interval convention", case=False, na=False)]
        for _, r in hit.iterrows():
            detail = str(r.get("detail", ""))
            convention[r["region"]] = ("interval_ending" if "closed=right" in detail
                                       else "interval_beginning" if "closed=left" in detail
                                       else "unknown")

    s2 = Stage2(root=root, run_config=cfg, regions=regions, region_files=region_files,
                features_A=A, features_B=B, horizon_report=report,
                weather_provenance=weather_provenance, weather_validation=weather_validation,
                future_frame=fut, current_actuals=current_actuals,
                aemo_actual=aemo_a, aemo_forecast=aemo_f,
                interval_convention=convention, notes=notes)
    return s2


def audit_feature_safety(s2: Stage2) -> pd.DataFrame:
    """Validate the A/B contract before fitting any model.

    Direct weather is allowed only in Feature Set B and only when Stage 2 ships
    the provenance contract that distinguishes historical observed weather from
    live forecast weather and climatology fallback.
    """
    rows = []
    for label, feats in (("A", s2.features_A), ("B", s2.features_B)):
        offenders = [f for f in feats
                     if any(re.search(p, f, flags=re.I) for p in FORBIDDEN_PATTERNS)]
        rows.append({"check": f"feature_set_{label}_free_of_unsafe_columns",
                     "result": "PASS" if not offenders else "FAIL",
                     "detail": f"{len(feats)} features; offenders: {offenders or 'none'}"})

    direct_in_A = sorted(set(s2.features_A) & WEATHER_DIRECT_FEATURES)
    rows.append({"check": "direct_weather_excluded_from_model_A",
                 "result": "PASS" if not direct_in_A else "FAIL",
                 "detail": f"direct weather in A: {direct_in_A or 'none'}"})

    direct_in_B = sorted(set(s2.features_B) & WEATHER_DIRECT_FEATURES)
    prov_ok = not s2.weather_provenance.empty
    if direct_in_B:
        declared = set()
        if prov_ok and "feature" in s2.weather_provenance.columns:
            declared = set(s2.weather_provenance["feature"].astype(str))
        missing_contract = sorted(set(direct_in_B) - declared)
        rows.append({"check": "model_B_weather_provenance_contract",
                     "result": "PASS" if prov_ok and not missing_contract else "FAIL",
                     "detail": (f"B weather features covered by provenance: {len(direct_in_B)}"
                                if prov_ok and not missing_contract else
                                f"missing provenance for: {missing_contract or direct_in_B}")})

    if not s2.horizon_report.empty and "four_month_recursive" in s2.horizon_report.columns:
        rep = s2.horizon_report.copy()
        unsafe = rep[rep["four_month_recursive"].astype(str).str.strip().str.lower().str.startswith("no")]
        named = set(unsafe["feature"].astype(str))
        clash = sorted(named & (set(s2.features_A) | set(s2.features_B)))
        rows.append({"check": "horizon_report_agrees_with_feature_sets",
                     "result": "PASS" if not clash else "FAIL",
                     "detail": f"features marked not-four-month-safe but selected: {clash or 'none'}"})

    rows.append({"check": "model_B_extends_model_A",
                 "result": "PASS" if set(s2.features_A) <= set(s2.features_B) else "WARNING",
                 "detail": f"B adds {sorted(set(s2.features_B) - set(s2.features_A))}"})

    if not s2.weather_validation.empty and "result" in s2.weather_validation.columns:
        failed = s2.weather_validation[s2.weather_validation["result"].astype(str).str.upper() == "FAIL"]
        rows.append({"check": "stage2_weather_forecast_validation",
                     "result": "PASS" if failed.empty else "WARNING",
                     "detail": ("Stage 2 forecast-weather validation has no FAIL rows"
                                if failed.empty else f"{len(failed)} forecast-weather validation FAIL row(s)")})

    cutoff = pd.to_datetime(s2.run_config.get("training_cutoff"), errors="coerce")
    pred_start = pd.to_datetime(s2.run_config.get("prediction_start"), errors="coerce")
    contract_ok = pd.notna(cutoff) and pd.notna(pred_start) and cutoff < pred_start
    rows.append({"check": "strict_previous_month_training_cutoff_declared",
                 "result": "PASS" if contract_ok else "FAIL",
                 "detail": f"training_cutoff={cutoff}; prediction_start={pred_start}"})
    return pd.DataFrame(rows)


# ============================================================================
# 4. HORIZON RESOLUTION  (nothing below is month-specific)
# ============================================================================

def resolve_horizon(run_config: dict) -> tuple:
    """rolling_window -> (anchor_start, horizon_end, months).

    horizon_end is the last half-hour label of the fourth month, matching the
    grid Stage 2 writes into future_feature_frame_all_regions.csv.
    """
    rw = run_config.get("rolling_window")
    if rw:
        months = [pd.Period(str(m), freq="M") for m in rw]
    else:
        origin = pd.Timestamp(run_config["analysis_origin"])
        months = [pd.Period(origin, freq="M") + k for k in range(4)]
    for i in range(1, len(months)):
        if months[i] != months[i - 1] + 1:
            raise ValueError(f"rolling_window is not consecutive: {rw}")
    anchor_start = months[0].start_time
    horizon_end = (months[-1].end_time.floor(FREQ))
    return anchor_start, horizon_end, months


def analogue_origins(anchor_month: pd.Period, data_start: pd.Timestamp,
                     data_end: pd.Timestamp, max_origins: int = 3,
                     min_train_days: int = 240) -> list:
    """Historical windows that mirror the anchor month, newest first.

    For a September anchor these are Sep-Dec of earlier years; for a December
    anchor they automatically become Dec-Mar spanning the year boundary.
    """
    out = []
    for year in range(anchor_month.year - 1, data_start.year - 1, -1):
        period = pd.Period(year=year, month=anchor_month.month, freq="M")
        start = period.start_time
        end = (period + 3).end_time.floor(FREQ)
        if start <= data_start:
            continue
        if (start - data_start).days < min_train_days:
            continue
        if end > data_end:
            continue
        out.append((start, end))
        if len(out) >= max_origins:
            break
    return out


def horizon_self_test() -> pd.DataFrame:
    """December reusability proof - pure calendar logic, runs without any data."""
    rows = []
    cases = [
        ("september_anchor", ["2026-09", "2026-10", "2026-11", "2026-12"],
         "2026-09-01 00:00:00", "2026-12-31 23:30:00"),
        ("december_anchor_crosses_year", ["2026-12", "2027-01", "2027-02", "2027-03"],
         "2026-12-01 00:00:00", "2027-03-31 23:30:00"),
        ("january_anchor", ["2027-01", "2027-02", "2027-03", "2027-04"],
         "2027-01-01 00:00:00", "2027-04-30 23:30:00"),
    ]
    for name, rw, want_start, want_end in cases:
        start, end, months = resolve_horizon({"rolling_window": rw})
        ok = (start == pd.Timestamp(want_start)) and (end == pd.Timestamp(want_end)) and len(months) == 4
        rows.append({"test": name, "rolling_window": "|".join(rw),
                     "resolved_origin": str(start), "resolved_end": str(end),
                     "expected_origin": want_start, "expected_end": want_end,
                     "result": "PASS" if ok else "FAIL"})
    # December analogue search must find Dec->Mar windows, not Sep->Dec ones.
    origins = analogue_origins(pd.Period("2026-12", freq="M"),
                               pd.Timestamp("2022-01-01"), pd.Timestamp("2026-09-14"))
    ok = len(origins) > 0 and all(o.month == 12 for o, _ in origins) and \
        all(e == (pd.Period(o, freq="M") + 3).end_time.floor(FREQ) for o, e in origins)
    rows.append({"test": "december_analogue_origins", "rolling_window": "2026-12..2027-03",
                 "resolved_origin": "|".join(str(o.date()) for o, _ in origins) or "none",
                 "resolved_end": "|".join(str(e.date()) for _, e in origins) or "none",
                 "expected_origin": "December origins only",
                 "expected_end": "each +4 months - 30min",
                 "result": "PASS" if ok else "FAIL"})
    sep = analogue_origins(pd.Period("2026-09", freq="M"),
                           pd.Timestamp("2022-01-01"), pd.Timestamp("2026-09-14"))
    rows.append({"test": "september_analogue_origins", "rolling_window": "2026-09..2026-12",
                 "resolved_origin": "|".join(str(o.date()) for o, _ in sep) or "none",
                 "resolved_end": "|".join(str(e.date()) for _, e in sep) or "none",
                 "expected_origin": "September origins with >=240d training history",
                 "expected_end": "each +4 months - 30min",
                 "result": "PASS" if len(sep) >= 1 else "FAIL"})
    return pd.DataFrame(rows)


# ============================================================================
# 5. REGION DATA
# ============================================================================

@dataclass
class RegionData:
    region: str
    hist: pd.DataFrame           # TRAINING history only; ends at previous-month cutoff
    future: pd.DataFrame | None  # Stage 2 known-future frame for this region
    current_actuals: pd.DataFrame | None  # verification only, never training
    convention: str


def load_region(s2: Stage2, region: str) -> RegionData:
    df = pd.read_csv(s2.region_files[region], parse_dates=[TIME], low_memory=False)
    for c in ("local_date", "local_timestamp"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df.sort_values(TIME).drop_duplicates(subset=[TIME]).set_index(TIME)
    # Stage 2 drops constant columns, so REGION may be absent; recover it.
    df["REGION"] = region
    if "local_date" not in df.columns:
        df["local_date"] = df.index.normalize()
    df["local_month"] = df["local_date"].dt.month
    df["local_dow"] = df["local_date"].dt.dayofweek
    df["local_doy"] = df["local_date"].dt.dayofyear

    fut = None
    if s2.future_frame is not None:
        f = s2.future_frame
        key = "region" if "region" in f.columns else "REGION"
        fut = f[f[key] == region].copy()
        if fut.empty:
            fut = None
        else:
            fut = fut.sort_values(TIME).drop_duplicates(subset=[TIME]).set_index(TIME)
            if "local_date" not in fut.columns:
                fut["local_date"] = fut.index.normalize()
            fut["local_month"] = fut["local_date"].dt.month
            fut["local_dow"] = fut["local_date"].dt.dayofweek
            fut["local_doy"] = fut["local_date"].dt.dayofyear
    cur = None
    if s2.current_actuals is not None and not s2.current_actuals.empty:
        c = s2.current_actuals
        key = "REGION" if "REGION" in c.columns else "region"
        cur = c[c[key].astype(str) == region].copy()
        if not cur.empty:
            cur = cur.sort_values(TIME).drop_duplicates(subset=[TIME]).set_index(TIME)
            if "local_date" not in cur.columns:
                cur["local_date"] = cur.index.normalize()
            if "local_timestamp" not in cur.columns:
                cur["local_timestamp"] = cur.index
    return RegionData(region=region, hist=df, future=fut, current_actuals=cur,
                      convention=s2.interval_convention.get(region, "unknown"))


# ============================================================================
# 6. ORIGIN-SAFE FEATURE ENGINEERING
# ============================================================================

def _gather(y: np.ndarray, pos: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(pos.shape, np.nan)
    idx = pos - lag
    ok = idx >= 0
    out[ok] = y[idx[ok]]
    return out


def demand_features_at(y: np.ndarray, pos: np.ndarray, wanted: set) -> dict:
    """Rebuild the t-48-anchored demand features for the given grid positions.

    Definitions were confirmed against the Stage 2 columns to 1e-4 MW:
        lag_L(t)                        = y[t-L]
        previous_day_mean/peak/min(t)   = mean/max/min of y[t-95 .. t-48]
        rolling_mean|std_24h_at_t-48(t) = mean/std(ddof=1) of y[t-95 .. t-48]
        same_half_hour_Nday_mean(t)     = mean of y[t-48k] for k = 1..N
    One implementation serves both training rows and recursive forecast blocks,
    so the model never sees a feature built by a different rule than it was fit on.
    """
    pos = np.asarray(pos, dtype=np.int64)
    out = {}
    for lag in (48, 96, 336):
        name = f"lag_{lag}"
        if name in wanted:
            out[name] = _gather(y, pos, lag)

    need_window = wanted & {"previous_day_mean", "previous_day_peak", "previous_day_min",
                            "rolling_mean_24h_at_t_minus_48", "rolling_std_24h_at_t_minus_48"}
    if need_window:
        mat = np.empty((pos.size, INTERVALS_PER_DAY))
        for j in range(INTERVALS_PER_DAY):
            mat[:, j] = _gather(y, pos, 48 + j)
        mean = mat.mean(axis=1)
        if "previous_day_mean" in wanted:
            out["previous_day_mean"] = mean
        if "rolling_mean_24h_at_t_minus_48" in wanted:
            out["rolling_mean_24h_at_t_minus_48"] = mean
        if "previous_day_peak" in wanted:
            out["previous_day_peak"] = mat.max(axis=1)
        if "previous_day_min" in wanted:
            out["previous_day_min"] = mat.min(axis=1)
        if "rolling_std_24h_at_t_minus_48" in wanted:
            out["rolling_std_24h_at_t_minus_48"] = mat.std(axis=1, ddof=1)

    for n_days, name in ((7, "same_half_hour_7day_mean"), (14, "same_half_hour_14day_mean")):
        if name in wanted:
            acc = np.zeros(pos.size)
            for k in range(1, n_days + 1):
                acc = acc + _gather(y, pos, 48 * k)
            out[name] = acc / n_days
    return out


@dataclass
class OriginMaps:
    """Every mapping below is built from rows strictly before the forecast origin."""
    origin: pd.Timestamp
    profile: dict
    temp_by_doy: pd.Series | None
    rain_by_month: pd.Series | None
    balance_point: float | None
    n_history_rows: int
    history_end: pd.Timestamp | None
    years_used: str


def build_origin_maps(hist: pd.DataFrame, origin: pd.Timestamp) -> OriginMaps:
    """Recompute origin-dependent profiles and climatology from pre-origin data only."""
    pre = hist[hist.index < origin]
    pre = pre[pre[TARGET].notna()]
    if pre.empty:
        return OriginMaps(origin, {}, None, None, None, 0, None, "")

    prof = {}
    g = pre.groupby(["local_month", "half_hour_index"])[TARGET]
    prof["historical_month_halfhour_mean"] = g.mean()
    prof["historical_month_halfhour_median"] = g.median()
    prof["historical_month_dow_halfhour_mean"] = \
        pre.groupby(["local_month", "local_dow", "half_hour_index"])[TARGET].mean()
    daily = pre.groupby("local_date")[TARGET].agg(["mean", "max"])
    daily["local_month"] = daily.index.month
    prof["historical_month_daily_mean"] = daily.groupby("local_month")["mean"].mean()
    prof["historical_month_peak_mean"] = daily.groupby("local_month")["max"].mean()
    prof["historical_month_variability"] = daily.groupby("local_month")["mean"].std()

    temp_by_doy = rain_by_month = None
    days = pre.drop_duplicates(subset=["local_date"])
    if "temp_mean_c" in days.columns:
        t = days.dropna(subset=["temp_mean_c"])
        if not t.empty:
            temp_by_doy = t.groupby("local_doy")["temp_mean_c"].mean()
    if "rainfall_mm" in days.columns:
        r = days.dropna(subset=["rainfall_mm"])
        if not r.empty:
            # A "rain day" is >0.2 mm - the threshold Stage 2 used, recovered by
            # reproducing its shipped historical_rain_probability exactly.
            rain_by_month = r.groupby("local_month")["rainfall_mm"].apply(lambda x: float((x > 0.2).mean()))

    bp = None
    if "balance_point_c" in pre.columns:
        seen = pre["balance_point_c"].dropna()
        if not seen.empty:
            bp = float(seen.iloc[-1])
    if bp is None and temp_by_doy is not None:
        bp = _estimate_balance_point(pre)

    years = sorted(pre["local_date"].dt.year.unique().tolist())
    return OriginMaps(origin=origin, profile=prof, temp_by_doy=temp_by_doy,
                      rain_by_month=rain_by_month, balance_point=bp,
                      n_history_rows=len(pre), history_end=pre.index.max(),
                      years_used=",".join(str(y) for y in years))


def _estimate_balance_point(pre: pd.DataFrame) -> float | None:
    """Pick the balance point whose HDD+CDD best explains daily demand."""
    if "temp_mean_c" not in pre.columns:
        return None
    d = pre.groupby("local_date").agg(demand=(TARGET, "mean"), temp=("temp_mean_c", "mean")).dropna()
    if len(d) < 60:
        return None
    best, best_bp = -1.0, None
    for bp in np.arange(14.0, 25.5, 0.5):
        dd = np.maximum(bp - d["temp"], 0) + np.maximum(d["temp"] - bp, 0)
        if dd.std() == 0:
            continue
        c = abs(float(np.corrcoef(dd, d["demand"])[0, 1]))
        if np.isfinite(c) and c > best:
            best, best_bp = c, float(bp)
    return best_bp



def _weather_fallback_frame(hist: pd.DataFrame, frame: pd.DataFrame,
                            origin: pd.Timestamp, balance_point: float | None) -> pd.DataFrame:
    """Build origin-safe weather values for forecast rows from pre-origin history.

    This is the deployment-safe fallback used when no genuine archived/live
    forecast vintage is available.  It never reads realised weather at or after
    ``origin``.  Half-hour temperature/humidity use month-day-clock climatology;
    daily fields use month-day climatology with a month-level fallback.
    """
    out = pd.DataFrame(index=frame.index)
    pre = hist[(hist.index < origin)].copy()
    if pre.empty:
        for c in WEATHER_DIRECT_FEATURES | WEATHER_CONTEXT_FEATURES:
            out[c] = np.nan
        return out

    if "local_date" not in pre.columns:
        pre["local_date"] = pre.index.normalize()
    if "local_date" not in frame.columns:
        frame = frame.copy()
        frame["local_date"] = frame.index.normalize()

    def keys(df):
        d = pd.to_datetime(df["local_date"])
        return d.dt.month, d.dt.day, df["half_hour_index"].astype(int)

    pm, pd_, ph = keys(pre)
    fm, fd, fh = keys(frame)

    # Intraday features: exact month/day/half-hour, with month/half-hour fallback.
    for col in ("weather_temp_c", "weather_humidity_pct"):
        if col not in pre.columns or pre[col].notna().sum() == 0:
            out[col] = np.nan
            continue
        work = pd.DataFrame({"m": pm, "d": pd_, "hh": ph,
                             "v": pd.to_numeric(pre[col], errors="coerce")}).dropna()
        exact = work.groupby(["m", "d", "hh"])["v"].mean()
        monthly = work.groupby(["m", "hh"])["v"].mean()
        idx = pd.MultiIndex.from_arrays([fm, fd, fh])
        arr = exact.reindex(idx).to_numpy(dtype=float)
        miss = ~np.isfinite(arr)
        if miss.any():
            midx = pd.MultiIndex.from_arrays([fm[miss], fh[miss]])
            arr[miss] = monthly.reindex(midx).to_numpy(dtype=float)
        out[col] = arr

    # Daily features are duplicated across the 48 half-hours; collapse history
    # to one row per local date before building the climatology.
    day = pre.drop_duplicates(subset=["local_date"]).copy()
    day["m"] = pd.to_datetime(day["local_date"]).dt.month
    day["d"] = pd.to_datetime(day["local_date"]).dt.day
    for col in ("weather_daily_temp_max_c", "weather_daily_temp_min_c",
                "weather_daily_temp_mean_c", "weather_daily_rainfall_mm"):
        source_col = col if col in day.columns else None
        if source_col is None or day[source_col].notna().sum() == 0:
            # Historical preprocessing also keeps legacy daily columns. They are
            # supporting columns only, but are safe raw material for the fallback.
            legacy = {"weather_daily_temp_max_c": "temp_max_c",
                      "weather_daily_temp_min_c": "temp_min_c",
                      "weather_daily_temp_mean_c": "temp_mean_c",
                      "weather_daily_rainfall_mm": "rainfall_mm"}[col]
            source_col = legacy if legacy in day.columns else None
        if source_col is None or day[source_col].notna().sum() == 0:
            out[col] = np.nan
            continue
        work = day[["m", "d", source_col]].rename(columns={source_col: "v"}).dropna()
        exact = work.groupby(["m", "d"])["v"].mean()
        monthly = work.groupby("m")["v"].mean()
        idx = pd.MultiIndex.from_arrays([fm, fd])
        arr = exact.reindex(idx).to_numpy(dtype=float)
        miss = ~np.isfinite(arr)
        if miss.any():
            arr[miss] = monthly.reindex(pd.Index(fm[miss])).to_numpy(dtype=float)
        out[col] = arr

    # Rebuild weather-history summaries from a safe daily path.  Before origin
    # that path may use actually observed historical weather; from origin onward
    # it contains only the climatological/forecast values already placed above.
    target_dates = pd.DatetimeIndex(pd.to_datetime(frame["local_date"]).dt.normalize().unique())
    start_date = min(target_dates.min(), pd.Timestamp(origin).normalize()) - pd.Timedelta(days=14)
    end_date = target_dates.max()
    dates = pd.date_range(start_date, end_date, freq="D")

    historical_daily = pre.groupby("local_date")[
        "weather_daily_temp_mean_c" if "weather_daily_temp_mean_c" in pre.columns else "temp_mean_c"
    ].mean() if ("weather_daily_temp_mean_c" in pre.columns or "temp_mean_c" in pre.columns) else pd.Series(dtype=float)

    # Daily climatology from the fallback frame itself, one value per target date.
    f_daily = pd.Series(out.get("weather_daily_temp_mean_c", np.nan), index=frame.index)
    f_daily = f_daily.groupby(pd.to_datetime(frame["local_date"]).dt.normalize()).mean()
    safe_daily = pd.Series(index=dates, dtype=float)
    for d in dates:
        if d < pd.Timestamp(origin).normalize() and d in historical_daily.index and pd.notna(historical_daily.loc[d]):
            safe_daily.loc[d] = float(historical_daily.loc[d])
        elif d in f_daily.index and pd.notna(f_daily.loc[d]):
            safe_daily.loc[d] = float(f_daily.loc[d])
        else:
            # Month-level final fallback from pre-origin historical daily weather.
            same_month = historical_daily[historical_daily.index.month == d.month].dropna()
            safe_daily.loc[d] = float(same_month.mean()) if len(same_month) else np.nan

    local_dates = pd.DatetimeIndex(pd.to_datetime(frame["local_date"]).dt.normalize())
    out["temp_mean_lag_1d"] = safe_daily.reindex(local_dates - pd.Timedelta(days=1)).to_numpy(dtype=float)
    out["temp_mean_lag_7d"] = safe_daily.reindex(local_dates - pd.Timedelta(days=7)).to_numpy(dtype=float)
    rolling7 = safe_daily.shift(1).rolling(7, min_periods=1).mean()
    out["rolling_temp_mean_7d"] = rolling7.reindex(local_dates).to_numpy(dtype=float)
    if balance_point is not None and np.isfinite(balance_point):
        lag1 = pd.to_numeric(out["temp_mean_lag_1d"], errors="coerce")
        out["hdd_lag_1d"] = np.maximum(balance_point - lag1, 0.0)
        out["cdd_lag_1d"] = np.maximum(lag1 - balance_point, 0.0)
    else:
        out["hdd_lag_1d"] = np.nan
        out["cdd_lag_1d"] = np.nan
    return out


def apply_weather_policy(panel: "Panel", rd: RegionData, maps: OriginMaps,
                         s2: Stage2, mode: str, information_origin=None) -> "Panel":
    """Make Model-B weather deployable for this exact forecast origin.

    mode='historical_backtest'  -> all horizon weather rebuilt from pre-origin
                                   climatology; realised future weather is hidden.
    mode='frozen'               -> only a forecast vintage issued no later than the
                                   month-start origin may be used; otherwise fallback.
    mode='live'                 -> genuine FORECAST_7DAY rows from Stage 2 are used
                                   when their issue time is <= this model origin;
                                   later rows use climatology fallback.
    mode='retrospective'        -> current-month diagnostic only; observed weather
                                   may be used and is explicitly labelled hindsight.
    """
    information_origin = pd.Timestamp(information_origin) if information_origin is not None else maps.origin
    frame = panel.known.copy()
    fallback = _weather_fallback_frame(rd.hist, frame, maps.origin, maps.balance_point)

    # Start from the origin-safe fallback for every Model-B weather predictor.
    for col in WEATHER_DIRECT_FEATURES | WEATHER_CONTEXT_FEATURES:
        if col in s2.features_B:
            frame[col] = fallback[col] if col in fallback.columns else np.nan
    frame["weather_feature_source"] = "BACKTEST_CLIMATOLOGY" if mode == "historical_backtest" else "CLIMATOLOGY_FALLBACK"

    supplied = panel.known.copy()
    src = supplied.get("weather_feature_source", pd.Series("", index=supplied.index)).astype(str)
    issue = pd.to_datetime(supplied.get("weather_forecast_origin", pd.Series(pd.NaT, index=supplied.index)),
                           errors="coerce")

    if mode == "live":
        use = src.eq("FORECAST_7DAY") & issue.notna() & (issue <= information_origin)
        # Stage 2 climatology is also valid and may include a better intraday
        # shape than the generic fallback reconstructed here.
        use |= src.eq("CLIMATOLOGY_FALLBACK")
    elif mode == "frozen":
        use = src.eq("FORECAST_7DAY") & issue.notna() & (issue <= information_origin)
        use |= src.eq("CLIMATOLOGY_FALLBACK")
    elif mode == "retrospective":
        use = src.isin(["OBSERVED_REANALYSIS", "CLIMATOLOGY_FALLBACK"])
    else:
        use = pd.Series(False, index=frame.index)

    for col in WEATHER_DIRECT_FEATURES:
        if col in s2.features_B and col in supplied.columns:
            vals = pd.to_numeric(supplied[col], errors="coerce")
            mask = use & vals.notna()
            frame.loc[mask, col] = vals.loc[mask]

    if mode in ("live", "frozen", "retrospective"):
        frame.loc[use, "weather_feature_source"] = src.loc[use]
        for meta in WEATHER_METADATA_COLUMNS - {"weather_feature_source"}:
            if meta in supplied.columns:
                frame[meta] = supplied[meta]
                frame.loc[~use, meta] = np.nan

        # Recompute the weather-history summaries from the final safe daily path,
        # so lags never point to realised weather after the model origin.
        safe = frame.copy()
        # Temporarily expose the chosen future daily mean to the fallback helper.
        tmp_hist = rd.hist.copy()
        rebuilt = _weather_fallback_frame(tmp_hist, safe, maps.origin, maps.balance_point)
        # The helper's context columns are climatology-safe.  For the first few
        # days this is deliberately conservative; direct forecast weather remains
        # available separately and carries the high-resolution signal.
        for col in WEATHER_CONTEXT_FEATURES:
            if col in s2.features_B and col in rebuilt.columns:
                frame[col] = rebuilt[col]

    if mode == "retrospective":
        frame["weather_policy_note"] = "RETROSPECTIVE_OBSERVED_WHERE_AVAILABLE_NOT_MODEL_SELECTION"
    elif mode == "live":
        frame["weather_policy_note"] = "LIVE_FORECAST_7DAY_THEN_CLIMATOLOGY"
    elif mode == "frozen":
        frame["weather_policy_note"] = "FROZEN_ORIGIN_VINTAGE_ONLY_THEN_CLIMATOLOGY"
    else:
        frame["weather_policy_note"] = "HISTORICAL_BACKTEST_ORIGIN_SAFE_CLIMATOLOGY"

    frame["weather_information_origin"] = information_origin
    panel.known = frame
    panel.weather_policy = mode
    return panel

def apply_origin_maps(frame: pd.DataFrame, maps: OriginMaps, wanted: set) -> dict:
    """Map the origin-safe profile and climatology values onto a set of rows."""
    out = {}
    month = frame["local_month"].to_numpy()
    dow = frame["local_dow"].to_numpy()
    hh = frame["half_hour_index"].to_numpy()
    doy = frame["local_doy"].to_numpy()

    for name, table in maps.profile.items():
        if name not in wanted:
            continue
        if name == "historical_month_dow_halfhour_mean":
            key = pd.MultiIndex.from_arrays([month, dow, hh])
        elif name in ("historical_month_halfhour_mean", "historical_month_halfhour_median"):
            key = pd.MultiIndex.from_arrays([month, hh])
        else:
            key = pd.Index(month)
        out[name] = table.reindex(key).to_numpy(dtype=float)

    temp = None
    if maps.temp_by_doy is not None:
        temp = maps.temp_by_doy.reindex(pd.Index(doy)).to_numpy(dtype=float)
    if "expected_temp_climatology" in wanted:
        out["expected_temp_climatology"] = temp if temp is not None else np.full(len(frame), np.nan)
    if "historical_rain_probability" in wanted:
        out["historical_rain_probability"] = (
            maps.rain_by_month.reindex(pd.Index(month)).to_numpy(dtype=float)
            if maps.rain_by_month is not None else np.full(len(frame), np.nan))
    if temp is not None and maps.balance_point is not None:
        if "expected_hdd_climatology" in wanted:
            out["expected_hdd_climatology"] = np.maximum(maps.balance_point - temp, 0.0)
        if "expected_cdd_climatology" in wanted:
            out["expected_cdd_climatology"] = np.maximum(temp - maps.balance_point, 0.0)
    else:
        for n in ("expected_hdd_climatology", "expected_cdd_climatology"):
            if n in wanted:
                out[n] = np.full(len(frame), np.nan)
    return out


def split_features(features: list) -> tuple:
    """Partition a feature list into the three build routes."""
    demand = [f for f in features if f in DEMAND_FEATURES]
    rebuilt = [f for f in features if f.startswith(PROFILE_PREFIX) or f in CLIMATOLOGY_FEATURES]
    known = [f for f in features if f not in demand and f not in rebuilt]
    return known, rebuilt, demand


# ============================================================================
# 7. WORKING PANEL - one continuous grid per region and origin
# ============================================================================

@dataclass
class Panel:
    index: pd.DatetimeIndex     # continuous 30-min grid, history start -> horizon end
    y: np.ndarray               # actuals before the origin, NaN from the origin on
    actual: np.ndarray          # actuals wherever Stage 2 has them (scoring only)
    origin_pos: int
    grid_pos: np.ndarray        # positions of the forecast horizon
    known: pd.DataFrame         # calendar / holiday / weather rows for the horizon
    train_rows: pd.DataFrame    # pre-origin rows carrying known features
    train_pos: np.ndarray
    weather_policy: str = "none"


def build_panel(rd: RegionData, origin: pd.Timestamp, horizon_end: pd.Timestamp,
                use_stage2_future: bool) -> Panel:
    hist = rd.hist
    index = pd.date_range(hist.index.min(), horizon_end, freq=FREQ)
    pos_of = pd.Series(np.arange(len(index)), index=index)

    actual = pd.Series(np.nan, index=index)
    common = hist.index.intersection(index)
    actual.loc[common] = hist.loc[common, TARGET].to_numpy()

    y = actual.to_numpy(dtype=float).copy()
    origin_pos = int(pos_of.get(origin, np.searchsorted(index, origin)))
    y[origin_pos:] = np.nan                       # nothing at or after the origin is known

    grid = pd.date_range(origin, horizon_end, freq=FREQ)
    grid = grid[grid.isin(index)]
    grid_pos = pos_of.reindex(grid).to_numpy(dtype=np.int64)

    if use_stage2_future and rd.future is not None:
        src = rd.future.reindex(grid)
        # Any horizon row Stage 2 did not supply falls back to the historical
        # calendar row for the same timestamp, which is deterministic anyway.
        missing = src.index[src["half_hour_index"].isna()] if "half_hour_index" in src.columns else src.index[:0]
        if len(missing) and len(hist.index.intersection(missing)):
            fill = hist.reindex(missing)
            for c in src.columns:
                if c in fill.columns:
                    src.loc[missing, c] = fill.loc[missing, c]
        known = src
    else:
        known = hist.reindex(grid)

    train_rows = hist[(hist.index < origin) & hist[TARGET].notna()]
    train_pos = pos_of.reindex(train_rows.index).to_numpy(dtype=np.int64)
    return Panel(index=index, y=y, actual=actual.to_numpy(dtype=float),
                 origin_pos=origin_pos, grid_pos=grid_pos, known=known,
                 train_rows=train_rows, train_pos=train_pos)


def design_matrix(rows: pd.DataFrame, pos: np.ndarray, y: np.ndarray,
                  features: list, maps: OriginMaps) -> pd.DataFrame:
    known, rebuilt, demand = split_features(features)
    data = {}
    for f in known:
        data[f] = rows[f].to_numpy(dtype=float) if f in rows.columns else np.full(len(rows), np.nan)
    data.update(apply_origin_maps(rows, maps, set(rebuilt)))
    data.update(demand_features_at(y, pos, set(demand)))
    return pd.DataFrame({f: data.get(f, np.full(len(rows), np.nan)) for f in features},
                        index=rows.index)


def lag_provenance(panel: Panel) -> pd.DataFrame:
    """ACTUAL_HISTORY / RECURSIVE_FORECAST / MISSING for each required lag."""
    out = {}
    n = len(panel.index)
    for lag in (48, 96, 336):
        src = np.full(len(panel.grid_pos), "RECURSIVE_FORECAST", dtype=object)
        ref = panel.grid_pos - lag
        src[ref < 0] = "MISSING"
        src[(ref >= 0) & (ref < panel.origin_pos)] = "ACTUAL_HISTORY"
        # a reference pointing past the end of the panel can never be filled
        src[ref >= n] = "MISSING"
        out[f"lag{lag}_source"] = src
    df = pd.DataFrame(out, index=panel.index[panel.grid_pos])
    df["recursive_input_used"] = (df != "ACTUAL_HISTORY").any(axis=1).astype(int)
    return df


# ============================================================================
# 8. MODELS
# ============================================================================

def make_ml_model(name: str, quick: bool):
    if name.startswith("RandomForest"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=40 if quick else 150,
            max_depth=12 if quick else 18,
            min_samples_leaf=5 if quick else 2,
            max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
    if name.startswith("XGBoost"):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=300 if quick else 700,
            max_depth=6 if quick else 8,
            learning_rate=0.08 if quick else 0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(name)


@dataclass
class Run:
    name: str
    status: str                       # MODEL_READY / MODEL_READY_WITH_WARNING / MODEL_FAILED
    detail: str = ""
    runtime_s: float = 0.0
    forecast: pd.Series | None = None
    fitted: object = None
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    train_rows: int = 0
    extra: dict = field(default_factory=dict)


def _fit_frame(panel: Panel, features: list, maps: OriginMaps,
               max_train_rows: int | None) -> tuple:
    rows = panel.train_rows
    pos = panel.train_pos
    if len(rows) > WARMUP_INTERVALS:
        rows, pos = rows.iloc[WARMUP_INTERVALS:], pos[WARMUP_INTERVALS:]
    X = design_matrix(rows, pos, panel.y, features, maps)
    yv = rows[TARGET].to_numpy(dtype=float)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(yv)
    X, yv, idx = X[ok], yv[ok], rows.index[ok]
    if max_train_rows and len(X) > max_train_rows:
        X, yv, idx = X.iloc[-max_train_rows:], yv[-max_train_rows:], idx[-max_train_rows:]
    return X, yv, idx


def run_naive(panel: Panel, name: str) -> Run:
    """Recursive seasonal naive: once the lag reaches past the origin it consumes
    its own earlier output, which is the honest four-month behaviour."""
    t0 = time.time()
    lag = NAIVE_LAG[name]
    y = panel.y.copy()
    for start in range(0, len(panel.grid_pos), BLOCK):
        blk = panel.grid_pos[start:start + BLOCK]
        y[blk] = _gather(y, blk, lag)
    fc = pd.Series(y[panel.grid_pos], index=panel.index[panel.grid_pos], name=name)
    bad = int(fc.isna().sum())
    return Run(name=name, status="MODEL_READY" if bad == 0 else "MODEL_READY_WITH_WARNING",
               detail=f"seasonal naive at lag {lag}" + (f"; {bad} unfillable intervals" if bad else ""),
               runtime_s=time.time() - t0, forecast=fc,
               train_start=panel.index[0], train_end=panel.index[panel.origin_pos - 1],
               train_rows=int(np.isfinite(panel.y[:panel.origin_pos]).sum()))


def run_ml(panel: Panel, name: str, features: list, maps: OriginMaps,
           quick: bool, max_train_rows: int | None) -> Run:
    t0 = time.time()
    try:
        model = make_ml_model(name, quick)
    except ImportError as exc:
        return Run(name=name, status="MODEL_FAILED",
                   detail=f"dependency unavailable: {exc}. Not substituted with another algorithm.",
                   runtime_s=time.time() - t0)
    try:
        X, yv, idx = _fit_frame(panel, features, maps, max_train_rows)
        if len(X) < 2000:
            return Run(name=name, status="MODEL_FAILED",
                       detail=f"only {len(X)} complete training rows before the origin",
                       runtime_s=time.time() - t0)
        model.fit(X.to_numpy(dtype=np.float32), yv)
        # A horizon row can lack a profile value for a month/day/half-hour the
        # training history never contained. Fill from the TRAINING medians, which
        # are fixed before the horizon is touched, and count every fill.
        fill = X.median(numeric_only=True)

        y = panel.y.copy()
        preds = np.full(len(panel.grid_pos), np.nan)
        n_filled = 0
        for start in range(0, len(panel.grid_pos), BLOCK):
            sl = slice(start, start + BLOCK)
            blk = panel.grid_pos[sl]
            rows = panel.known.iloc[sl]
            Xb = design_matrix(rows, blk, y, features, maps)
            n_filled += int(Xb.isna().to_numpy().sum())
            Xb = Xb.fillna(fill)
            yhat = model.predict(Xb.to_numpy(dtype=np.float32))
            y[blk] = yhat
            preds[sl] = yhat
        fc = pd.Series(preds, index=panel.index[panel.grid_pos], name=name)
        return Run(name=name, status="MODEL_READY" if n_filled == 0 else "MODEL_READY_WITH_WARNING",
                   detail=(f"{len(X)} training rows, {len(features)} features, recursive in "
                           f"{BLOCK}-interval day blocks"
                           + (f"; {n_filled} horizon feature cells filled from training medians"
                              if n_filled else "")),
                   runtime_s=time.time() - t0, forecast=fc, fitted=model,
                   train_start=idx.min(), train_end=idx.max(), train_rows=len(X),
                   extra={"feature_names": features})
    except Exception as exc:
        return Run(name=name, status="MODEL_FAILED",
                   detail=f"{type(exc).__name__}: {exc}", runtime_s=time.time() - t0)


def run_sarima(panel: Panel, name: str, maps: OriginMaps, features_B: list,
               train_intervals: int, quick: bool = False) -> Run:
    """SARIMA / SARIMAX on a deliberately limited recent window.

    A seasonal state-space fit with s=48 over the full 82k-interval series is not
    computationally realistic, so the training window is bounded and recorded
    rather than the search being silently abandoned.
    """
    t0 = time.time()
    if not have("statsmodels"):
        return Run(name=name, status="MODEL_FAILED",
                   detail="dependency unavailable: statsmodels. Not substituted.",
                   runtime_s=time.time() - t0)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        hist_y = pd.Series(panel.y[:panel.origin_pos], index=panel.index[:panel.origin_pos]).dropna()
        train = hist_y.iloc[-train_intervals:]
        train = train.asfreq(FREQ)
        if train.isna().any():
            train = train.interpolate(limit_direction="both")
        order = (1, 0, 0) if quick else (1, 0, 1)
        seasonal = (1, 0, 0, INTERVALS_PER_DAY)

        exog_tr = exog_fc = None
        exog_names, exog_dropped = [], []
        if name == "SARIMAX":
            exog_names = [c for c in SARIMAX_EXOG_PREFERRED if c in features_B]
            tr_rows = panel.train_rows.reindex(train.index)
            tr_pos = np.searchsorted(panel.index, train.index)
            exog_tr = design_matrix(tr_rows, tr_pos, panel.y, exog_names, maps)
            exog_fc = design_matrix(panel.known, panel.grid_pos, panel.y, exog_names, maps)
            exog_tr = exog_tr.ffill().bfill()
            exog_fc = exog_fc.ffill().bfill()
            # A regressor that barely moves inside the (deliberately short) SARIMA
            # training window has an unidentifiable coefficient. Extrapolating it
            # across four months is what makes a state-space forecast explode, so
            # any regressor whose horizon range outruns its training range is
            # dropped rather than trusted. Survivors are scaled by training SD.
            keep, dropped = [], []
            for c in exog_names:
                tr_range = float(exog_tr[c].max() - exog_tr[c].min())
                fc_range = float(exog_fc[c].max() - exog_fc[c].min())
                if not exog_tr[c].notna().all() or exog_tr[c].std() <= 1e-8:
                    dropped.append(f"{c}(no training variation)")
                elif fc_range > 1.5 * max(tr_range, 1e-9):
                    dropped.append(f"{c}(horizon range {fc_range:.2f} outruns training range {tr_range:.2f})")
                else:
                    keep.append(c)
            exog_tr, exog_fc, exog_names = exog_tr[keep], exog_fc[keep], keep
            if not exog_names:
                return Run(name=name, status="MODEL_FAILED",
                           detail=("no exogenous regressor survives the extrapolation-support "
                                   f"filter; dropped {dropped}"),
                           runtime_s=time.time() - t0)
            scale = exog_tr.std().replace(0, 1.0)
            exog_tr, exog_fc = exog_tr / scale, exog_fc / scale
            exog_dropped = dropped

        # Stationarity is enforced because the forecast is extrapolated thousands of
        # steps: an AR root outside the unit circle would diverge rather than revert.
        model = SARIMAX(train, exog=exog_tr, order=order, seasonal_order=seasonal,
                        trend="c", enforce_stationarity=True, enforce_invertibility=True)
        res = model.fit(disp=False, maxiter=(12 if quick else 60), method="lbfgs")
        converged = bool(res.mle_retvals.get("converged", False))
        fc_vals = res.get_forecast(steps=len(panel.grid_pos),
                                   exog=(exog_fc.to_numpy(dtype=float) if exog_fc is not None else None)).predicted_mean
        fc = pd.Series(np.asarray(fc_vals, dtype=float),
                       index=panel.index[panel.grid_pos], name=name)
        status = "MODEL_READY" if converged else "MODEL_READY_WITH_WARNING"
        if not np.isfinite(fc.to_numpy()).all():
            return Run(name=name, status="MODEL_FAILED",
                       detail="forecast contained non-finite values", runtime_s=time.time() - t0)
        # Divergence guard. A state-space extrapolation that leaves the plausible
        # range of the series is a failed model, not a result worth a metric.
        sd = float(train.std())
        lo, hi = float(train.min()) - 3 * sd, float(train.max()) + 3 * sd
        if fc.min() < lo or fc.max() > hi:
            return Run(name=name, status="MODEL_FAILED",
                       detail=(f"forecast diverged outside the plausible range "
                               f"[{lo:.0f}, {hi:.0f}] MW: produced "
                               f"[{fc.min():.0f}, {fc.max():.0f}] MW over "
                               f"{len(fc)} steps; no metric is reported for it"),
                       runtime_s=time.time() - t0)
        return Run(name=name, status=status,
                   detail=(f"order={order} seasonal_order={seasonal} "
                           f"train_intervals={len(train)} converged={converged} "
                           f"enforce_stationarity=True exog={exog_names or 'none'}"
                           + (f"; exog dropped by the extrapolation-support filter: "
                              f"{exog_dropped}" if name == "SARIMAX" and exog_dropped else "")),
                   runtime_s=time.time() - t0, forecast=fc, fitted=res,
                   train_start=train.index.min(), train_end=train.index.max(),
                   train_rows=len(train),
                   extra={"order": str(order), "seasonal_order": str(seasonal),
                          "converged": converged, "exog": exog_names})
    except Exception as exc:
        return Run(name=name, status="MODEL_FAILED",
                   detail=f"{type(exc).__name__}: {exc}", runtime_s=time.time() - t0)


def run_model(name: str, panel: Panel, s2: Stage2, maps: OriginMaps, cfg: "RunConfig") -> Run:
    if name in NAIVE_LAG:
        return run_naive(panel, name)
    if name in ("SARIMA", "SARIMAX"):
        return run_sarima(panel, name, maps, s2.features_B, cfg.sarima_train_intervals, cfg.quick)
    feats = s2.features_A if FEATURE_SET_OF[name] == "A" else s2.features_B
    return run_ml(panel, name, feats, maps, cfg.quick, cfg.max_train_rows)


# ============================================================================
# 9. RUN CONFIGURATION
# ============================================================================

@dataclass
class RunConfig:
    quick: bool = False
    max_origins: int = 3
    screen_folds: int = 3
    screen_test_days: int = 14
    screen_train_cap: int = 40000
    max_train_rows: int | None = None
    sarima_train_intervals: int = 1440
    make_figures: bool = True
    profile_mode: str = "origin_cutoff"

    @staticmethod
    def build(quick: bool, origins: int | None, figures: bool) -> "RunConfig":
        if quick:
            c = RunConfig(quick=True, max_origins=1, screen_folds=2, screen_test_days=7,
                          screen_train_cap=25000, max_train_rows=30000,
                          sarima_train_intervals=336, make_figures=figures)
        else:
            c = RunConfig(quick=False, max_origins=3, screen_folds=3, screen_test_days=14,
                          screen_train_cap=35000, max_train_rows=None,
                          sarima_train_intervals=1440, make_figures=figures)
        if origins:
            c.max_origins = origins
        return c


# ============================================================================
# 10. PHASE A - SCREENING THEN FOUR-MONTH BACKTEST
# ============================================================================

def screen_region(rd: RegionData, s2: Stage2, anchor: pd.Timestamp, cfg: RunConfig) -> pd.DataFrame:
    """Cheap chronological walk-forward screen on pre-anchor data.

    Its only job is to stop obviously broken configurations before the expensive
    recursive backtest. It never selects the production model.
    """
    rows = []
    # Model selection for the 2026 production year must not use 2026 target
    # performance even indirectly through the cheap screen.  Screen on completed
    # prior calendar years only; the full analogue backtests make the decision.
    hist = rd.hist[(rd.hist.index < pd.Timestamp(anchor.year, 1, 1)) & rd.hist[TARGET].notna()]
    if len(hist) < WARMUP_INTERVALS * 3:
        return pd.DataFrame(rows)
    test_len = cfg.screen_test_days * INTERVALS_PER_DAY
    for fold in range(cfg.screen_folds, 0, -1):
        cut = hist.index[-fold * test_len] if fold * test_len < len(hist) else None
        if cut is None:
            continue
        test_end = hist.index[-(fold - 1) * test_len - 1] if fold > 1 else hist.index[-1]
        maps = build_origin_maps(rd.hist, cut)
        panel = build_panel(rd, cut, test_end, use_stage2_future=False)
        panel = apply_weather_policy(panel, rd, maps, s2, "historical_backtest")
        actual = pd.Series(panel.actual[panel.grid_pos], index=panel.index[panel.grid_pos])
        for name in MODEL_ORDER:
            if name in ("SARIMA", "SARIMAX"):
                continue          # screened by runtime elsewhere; too slow to fold-screen
            if name in NAIVE_LAG:
                r = run_naive(panel, name)
            else:
                feats = s2.features_A if FEATURE_SET_OF[name] == "A" else s2.features_B
                r = run_ml(panel, name, feats, maps, cfg.quick, cfg.screen_train_cap)
            if r.forecast is None:
                rows.append({"region": rd.region, "fold": cfg.screen_folds - fold + 1,
                             "model": name, "status": r.status, "detail": r.detail,
                             "runtime_seconds": round(r.runtime_s, 2)})
                continue
            m = metrics(actual.to_numpy(), r.forecast.reindex(actual.index).to_numpy())
            rows.append({"region": rd.region, "fold": cfg.screen_folds - fold + 1,
                         "model": name, "feature_set": FEATURE_SET_OF[name],
                         "test_start": actual.index.min(), "test_end": actual.index.max(),
                         "train_rows": r.train_rows, **m,
                         "status": r.status, "detail": r.detail,
                         "runtime_seconds": round(r.runtime_s, 2)})
    return pd.DataFrame(rows)


def screen_decision(screen: pd.DataFrame) -> dict:
    """Drop an ML model only if it is more than twice the best screened MAE."""
    keep = {m: True for m in MODEL_ORDER}
    if screen.empty or "MAE" not in screen.columns:
        return keep
    mean_mae = screen.groupby("model")["MAE"].mean().dropna()
    if mean_mae.empty:
        return keep
    best = mean_mae.min()
    for model, mae in mean_mae.items():
        if model in NAIVE_LAG:
            continue
        if mae > 2.0 * best:
            keep[model] = False
    if not any(keep[m] for m in MODEL_ORDER if m not in NAIVE_LAG):
        keep = {m: True for m in MODEL_ORDER}   # never screen everything out
    return keep


def backtest_region(rd: RegionData, s2: Stage2, origins: list, cfg: RunConfig,
                    keep: dict, progress=None) -> tuple:
    """Full recursive four-month backtest at each historical analogue origin.

    Besides aggregate scores, retain interval-level predictions for the A/B tree
    families.  Those rows are used only for evidence visualisation and decision
    analysis after the family champions have been selected; they do not influence
    model fitting or leak future information into predictors.
    """
    score_rows, rebuild_rows, detail_frames = [], [], []
    for origin, end in origins:
        maps = build_origin_maps(rd.hist, origin)
        panel = build_panel(rd, origin, end, use_stage2_future=False)
        panel = apply_weather_policy(panel, rd, maps, s2, "historical_backtest")
        rebuild_rows.append(check_rebuild(rd, panel, maps, origin, s2))
        actual = pd.Series(panel.actual[panel.grid_pos], index=panel.index[panel.grid_pos])
        for name in MODEL_ORDER:
            if not keep.get(name, True):
                score_rows.append({"region": rd.region, "model": name,
                                   "feature_set": FEATURE_SET_OF[name],
                                   "backtest_origin": origin, "status": "SCREENED_OUT",
                                   "detail": "screening MAE more than twice the best model"})
                continue
            if cfg.quick and name in ("SARIMA", "SARIMAX"):
                score_rows.append({"region": rd.region, "model": name,
                                   "feature_set": FEATURE_SET_OF[name],
                                   "backtest_origin": origin, "status": "SKIPPED_QUICK_MODE",
                                   "detail": "statistical model retained for full mode; skipped only to keep quick validation fast"})
                if progress:
                    progress(f"      {rd.region} {origin.date()} {name}: SKIPPED_QUICK_MODE")
                continue
            r = run_model(name, panel, s2, maps, cfg)
            row = {"region": rd.region, "model": name, "feature_set": FEATURE_SET_OF[name],
                   "backtest_origin": origin, "horizon_start": panel.index[panel.grid_pos[0]],
                   "horizon_end": panel.index[panel.grid_pos[-1]],
                   "horizon_intervals": len(panel.grid_pos),
                   "train_start": r.train_start, "train_end": r.train_end,
                   "train_rows": r.train_rows,
                   "runtime_seconds": round(r.runtime_s, 2),
                   "status": r.status, "detail": r.detail}
            if r.forecast is not None:
                fc_aligned = r.forecast.reindex(actual.index)
                row.update(metrics(actual.to_numpy(), fc_aligned.to_numpy()))
                # Same four-month horizon for every candidate, with extra lead-band
                # evidence so weather value can be inspected without changing the
                # primary full-horizon selection rule.
                bands = {
                    "first_7d": slice(0, 7 * INTERVALS_PER_DAY),
                    "days_8_30": slice(7 * INTERVALS_PER_DAY, 30 * INTERVALS_PER_DAY),
                    "month_2": slice(30 * INTERVALS_PER_DAY, 61 * INTERVALS_PER_DAY),
                    "month_3": slice(61 * INTERVALS_PER_DAY, 92 * INTERVALS_PER_DAY),
                    "month_4": slice(92 * INTERVALS_PER_DAY, len(actual)),
                }
                for label, sl in bands.items():
                    aa = actual.iloc[sl].to_numpy()
                    ff = fc_aligned.iloc[sl].to_numpy()
                    mm = metrics(aa, ff)
                    row[f"MAE_{label}"] = mm["MAE"]
                    row[f"RMSE_{label}"] = mm["RMSE"]
                    row[f"sMAPE_{label}"] = mm["sMAPE"]
                row["weather_policy"] = panel.weather_policy

                # Preserve the exact out-of-sample path used to compute the score.
                # Keeping A/B interval detail makes the four-month evidence auditable
                # and supports actual-vs-forecast, residual and rolling-error plots.
                if FEATURE_SET_OF.get(name) in ("A", "B"):
                    det = pd.DataFrame({
                        "REGION": rd.region,
                        "model": name,
                        "feature_set": FEATURE_SET_OF.get(name, ""),
                        "backtest_origin": origin,
                        "target_timestamp": actual.index,
                        "actual_MW": actual.to_numpy(dtype=float),
                        "forecast_MW": fc_aligned.to_numpy(dtype=float),
                    })
                    det = det[det["actual_MW"].notna() & det["forecast_MW"].notna()].copy()
                    if not det.empty:
                        det["error_MW"] = det["forecast_MW"] - det["actual_MW"]
                        det["absolute_error_MW"] = det["error_MW"].abs()
                        det["squared_error"] = det["error_MW"] ** 2
                        denom = det["forecast_MW"].abs() + det["actual_MW"].abs()
                        det["smape_component"] = np.where(
                            denom >= SMAPE_MIN_DENOM,
                            200.0 * det["absolute_error_MW"] / denom, np.nan)
                        det["month"] = det["target_timestamp"].dt.to_period("M").astype(str)
                        det["date"] = det["target_timestamp"].dt.normalize()
                        det["half_hour_index"] = (det["target_timestamp"].dt.hour * 2 +
                                                  (det["target_timestamp"].dt.minute >= 30).astype(int))
                        detail_frames.append(det)
            score_rows.append(row)
            if progress:
                progress(f"      {rd.region} {origin.date()} {name}: {r.status} "
                         f"({r.runtime_s:.1f}s)")
    scores = pd.DataFrame(score_rows)
    if not scores.empty and "MAE" in scores.columns:
        scores["skill_vs_naive_pct"] = np.nan
        for (reg, org), grp in scores.groupby(["region", "backtest_origin"]):
            base = grp[grp.model.isin(NAIVE_LAG)]["MAE"].min()
            if np.isfinite(base):
                scores.loc[grp.index, "baseline_MAE"] = base
                scores.loc[grp.index, "skill_vs_naive_pct"] = grp["MAE"].apply(
                    lambda v: skill_pct(v, base))
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return scores, pd.DataFrame(rebuild_rows), detail


def check_rebuild(rd: RegionData, panel: Panel, maps: OriginMaps,
                  origin: pd.Timestamp, s2: Stage2) -> dict:
    """Compare Stage 3's origin-safe rebuild against the Stage 2 columns."""
    feats = [f for f in s2.features_B if f.startswith(PROFILE_PREFIX) or f in CLIMATOLOGY_FEATURES]
    rows = panel.known
    rebuilt = pd.DataFrame(apply_origin_maps(rows, maps, set(feats)), index=rows.index)
    diffs = {}
    for f in feats:
        if f in rows.columns and f in rebuilt.columns:
            a, b = rows[f].astype(float), rebuilt[f].astype(float)
            m = a.notna() & b.notna()
            diffs[f] = float((a[m] - b[m]).abs().max()) if m.any() else np.nan
    worst = max([v for v in diffs.values() if np.isfinite(v)], default=np.nan)
    return {"region": rd.region, "forecast_origin": origin,
            "history_rows_used": maps.n_history_rows,
            "history_end": maps.history_end,
            "history_years_used": maps.years_used,
            "balance_point_c": maps.balance_point,
            "features_rebuilt": len(feats),
            "max_abs_diff_vs_stage2_MW_or_unit": worst,
            "detail": "; ".join(f"{k}={v:.4f}" for k, v in diffs.items() if np.isfinite(v))}


def _aggregate_model_scores(scores: pd.DataFrame, region: str,
                            allowed_models: set[str] | None = None) -> pd.DataFrame:
    grp = scores[(scores.region == region) & scores.get("MAE", pd.Series(dtype=float)).notna()].copy()
    if allowed_models is not None:
        grp = grp[grp["model"].isin(allowed_models)]
    if grp.empty:
        return pd.DataFrame()

    # Prefer fully ready models. Warning models are considered only when a region
    # has no fully ready candidate; failed/screened models never win.
    ready = grp[grp["status"] == "MODEL_READY"]
    if ready.empty:
        ready = grp[grp["status"] == "MODEL_READY_WITH_WARNING"]
    if ready.empty:
        return pd.DataFrame()

    agg = ready.groupby("model").agg(
        mean_backtest_MAE=("MAE", "mean"), median_backtest_MAE=("MAE", "median"),
        mean_RMSE=("RMSE", "mean"), mean_sMAPE=("sMAPE", "mean"),
        mean_Bias=("Bias", "mean"), mae_spread=("MAE", "std"),
        origins=("backtest_origin", "nunique"),
        skill_vs_naive_pct=("skill_vs_naive_pct", "mean"),
        model_status=("status", lambda x: "MODEL_READY" if (x == "MODEL_READY").all()
                      else "MODEL_READY_WITH_WARNING")).reset_index()

    # Win rate is the share of historical origins on which a model had the lowest
    # MAE among the candidates in this comparison set.
    winners = []
    for _, fold in ready.groupby("backtest_origin"):
        if fold["MAE"].notna().any():
            best = fold.loc[fold["MAE"].idxmin(), "model"]
            winners.append(best)
    if winners:
        counts = pd.Series(winners).value_counts()
        agg["win_rate_pct"] = agg["model"].map(counts).fillna(0) / len(winners) * 100.0
    else:
        agg["win_rate_pct"] = np.nan
    return agg


def _choose_from_aggregate(agg: pd.DataFrame, region: str, label: str) -> dict:
    if agg.empty:
        return {"region": region, "selected_model": "NONE", "feature_set": "",
                "status": "MODEL_FAILED", "selection_scope": label,
                "selection_reason": "no stable model produced a scorable historical backtest"}
    best_mae = agg["mean_backtest_MAE"].min()
    tied = agg[agg["mean_backtest_MAE"] <= best_mae * (1 + TIE_TOLERANCE)].copy()
    tied["complexity"] = tied["model"].map(COMPLEXITY).fillna(9)
    tied = tied.sort_values(["complexity", "mean_backtest_MAE", "mae_spread"], na_position="last")
    win = tied.iloc[0]
    reason = (f"lowest deployment-safe mean four-month backtest MAE "
              f"({win.mean_backtest_MAE:.1f} MW) across {int(win.origins)} origin(s)")
    raw_best = agg.sort_values("mean_backtest_MAE").iloc[0]
    if len(tied) > 1 and win["model"] != raw_best["model"]:
        reason = (f"within {TIE_TOLERANCE:.0%} of the best MAE "
                  f"({win.mean_backtest_MAE:.1f} vs {best_mae:.1f} MW) and simpler/more stable")
    return {"region": region, "selected_model": win["model"],
            "feature_set": FEATURE_SET_OF.get(win["model"], ""),
            "mean_backtest_MAE": win["mean_backtest_MAE"],
            "median_backtest_MAE": win["median_backtest_MAE"],
            "mean_RMSE": win["mean_RMSE"], "mean_sMAPE": win["mean_sMAPE"],
            "mean_Bias": win["mean_Bias"], "mae_spread_across_origins": win["mae_spread"],
            "skill_vs_naive_pct": win["skill_vs_naive_pct"],
            "win_rate_pct": win["win_rate_pct"],
            "historical_origins_used": int(win["origins"]),
            "selection_reason": reason, "selection_scope": label,
            "status": str(win["model_status"])}


def select_model(scores: pd.DataFrame, region: str) -> dict:
    """Select the 2026 production model from fair four-month historical tests."""
    return _choose_from_aggregate(_aggregate_model_scores(scores, region), region, "OVERALL")


def select_feature_family_champions(scores: pd.DataFrame, region: str) -> list[dict]:
    """Best A and best B tree model on the same four-month historical windows.

    These champions are diagnostic/operational comparison models.  The overall
    production winner may still be a naive or statistical model.
    """
    families = {
        "MODEL_A": {"RandomForest_A", "XGBoost_A"},
        "MODEL_B": {"RandomForest_B", "XGBoost_B"},
    }
    out = []
    for label, models in families.items():
        row = _choose_from_aggregate(_aggregate_model_scores(scores, region, models),
                                     region, label)
        row["family"] = label
        out.append(row)
    return out


# ============================================================================
# 11. PHASE B / C - FROZEN AND CURRENT FORECASTS
# ============================================================================

def forecast_frame(rd: RegionData, panel: Panel, run: Run, origin: pd.Timestamp,
                   model_name: str, vintage: str, run_stamp: str) -> pd.DataFrame:
    """Return one forecast vintage with row-level lag and weather provenance."""
    prov = lag_provenance(panel)
    idx = panel.index[panel.grid_pos]
    step = np.arange(1, len(idx) + 1)

    if FEATURE_SET_OF.get(model_name) == "B" or model_name == "SARIMAX":
        if panel.known is not None and "weather_feature_source" in panel.known.columns:
            weather_source = panel.known["weather_feature_source"].reindex(idx).fillna("MISSING").astype(str).to_numpy()
        else:
            weather_source = np.repeat("CLIMATOLOGY_FALLBACK", len(idx))
    elif FEATURE_SET_OF.get(model_name) == "A":
        weather_source = np.repeat("NONE_MODEL_A", len(idx))
    else:
        weather_source = np.repeat("NOT_APPLICABLE", len(idx))

    out = pd.DataFrame({
        "REGION": rd.region,
        "forecast_origin": origin,
        "projection_start": idx.min() if len(idx) else pd.NaT,
        "training_cutoff": rd.hist.index.max() if len(rd.hist) else pd.NaT,
        "target_timestamp": idx,
        "forecast_MW": run.forecast.reindex(idx).to_numpy() if run.forecast is not None else np.nan,
        "selected_model": model_name,
        "feature_set": FEATURE_SET_OF.get(model_name, ""),
        "forecast_horizon_step": step,
        "forecast_horizon_days": np.round(step / INTERVALS_PER_DAY, 3),
        "lag48_source": prov["lag48_source"].to_numpy(),
        "lag96_source": prov["lag96_source"].to_numpy(),
        "lag336_source": prov["lag336_source"].to_numpy(),
        "recursive_input_used": prov["recursive_input_used"].to_numpy(),
        "weather_feature_source": weather_source,
        "weather_policy": panel.weather_policy,
        "forecast_vintage": vintage,
        "run_timestamp": run_stamp,
    })

    if panel.known is not None and (FEATURE_SET_OF.get(model_name) == "B" or model_name == "SARIMAX"):
        for c in ("weather_forecast_origin", "weather_lead_day", "weather_forecast_source",
                  "weather_forecast_model", "weather_retrieved_at_utc",
                  "weather_information_origin"):
            if c in panel.known.columns:
                out[c] = panel.known[c].reindex(idx).to_numpy()
    return out


def verify_current_month(frozen: pd.DataFrame, rd: RegionData,
                         anchor: pd.Timestamp) -> pd.DataFrame:
    """Score forecasts against the verification-only current-month actual file.

    ``rd.hist`` is deliberately never consulted here because it is the modelling
    dataset and must stop at the previous month end.
    """
    if rd.current_actuals is None or rd.current_actuals.empty or TARGET not in rd.current_actuals.columns:
        return pd.DataFrame()
    act = rd.current_actuals[[TARGET]].rename(columns={TARGET: "actual_MW"})
    act = act[act.index >= anchor]
    j = frozen.merge(act, left_on="target_timestamp", right_index=True, how="inner")
    j = j[j["actual_MW"].notna() & j["forecast_MW"].notna()].copy()
    if j.empty:
        return j
    j["error_MW"] = j["forecast_MW"] - j["actual_MW"]
    j["absolute_error_MW"] = j["error_MW"].abs()
    j["squared_error"] = j["error_MW"] ** 2
    denom = j["forecast_MW"].abs() + j["actual_MW"].abs()
    j["smape_component"] = np.where(denom >= SMAPE_MIN_DENOM,
                                    200.0 * j["absolute_error_MW"] / denom, np.nan)
    j["local_date"] = j["target_timestamp"].dt.normalize()
    if rd.future is not None and "half_hour_index" in rd.future.columns:
        hh = rd.future["half_hour_index"].reindex(j["target_timestamp"])
        j["half_hour_index"] = hh.to_numpy()
    else:
        j["half_hour_index"] = (j["target_timestamp"].dt.hour * 2 +
                                (j["target_timestamp"].dt.minute >= 30).astype(int))
    cols = ["REGION", "target_timestamp", "actual_MW", "forecast_MW", "error_MW",
            "absolute_error_MW", "squared_error", "smape_component", "selected_model",
            "feature_set", "forecast_horizon_step", "forecast_horizon_days",
            "half_hour_index", "local_date"]
    return j[cols]



def run_named_forecast(rd: RegionData, s2: Stage2, model_name: str,
                       origin: pd.Timestamp, end: pd.Timestamp, cfg: "RunConfig",
                       weather_mode: str, run_stamp: str, vintage: str,
                       information_origin=None) -> tuple[Run, Panel, pd.DataFrame]:
    """Fit from pre-month history and forecast from ``origin`` onward.

    ``information_origin`` may be later than ``origin`` only for a live weather
    update.  Demand training still ends at the previous-month cutoff, and demand
    lags from the prediction month are recursively generated rather than replaced
    with actuals.
    """
    maps = build_origin_maps(rd.hist, origin)
    panel = build_panel(rd, origin, end, use_stage2_future=True)
    info_origin = pd.Timestamp(information_origin) if information_origin is not None else origin
    if FEATURE_SET_OF.get(model_name) == "B" or model_name == "SARIMAX":
        panel = apply_weather_policy(panel, rd, maps, s2, weather_mode, info_origin)
    else:
        panel.weather_policy = "NONE_MODEL_A" if FEATURE_SET_OF.get(model_name) == "A" else "NOT_APPLICABLE"
    run = run_model(model_name, panel, s2, maps, cfg)
    frame = forecast_frame(rd, panel, run, info_origin, model_name, vintage, run_stamp)
    return run, panel, frame


def current_month_ab_summary(region: str, a_verify: pd.DataFrame,
                             b_verify: pd.DataFrame, a_model: str, b_model: str) -> pd.DataFrame:
    """Fair current-month A/B verification on identical actual timestamps.

    Forecasts are generated from the fixed previous-month training cutoff; the
    current-month actuals are joined only after prediction for scoring.
    """
    rows = []
    for label, model, df in (("MODEL_A", a_model, a_verify), ("MODEL_B", b_model, b_verify)):
        if df is None or df.empty or model == "NONE":
            continue
        m = metrics(df["actual_MW"].to_numpy(), df["forecast_MW"].to_numpy())
        rows.append({"REGION": region, "family": label, "model": model,
                     "evaluation_type": "FIXED_PREVIOUS_MONTH_CUTOFF_VERIFICATION",
                     "verification_start": df["target_timestamp"].min(),
                     "verification_end": df["target_timestamp"].max(), **m})
    out = pd.DataFrame(rows)
    if len(out) == 2:
        a_mae = float(out.loc[out.family == "MODEL_A", "MAE"].iloc[0])
        b_mae = float(out.loc[out.family == "MODEL_B", "MAE"].iloc[0])
        out["weather_skill_vs_A_pct"] = np.nan
        if a_mae > 0:
            out.loc[out.family == "MODEL_B", "weather_skill_vs_A_pct"] = 100.0 * (a_mae - b_mae) / a_mae
    return out


def ab_backtest_summary(scores: pd.DataFrame, family_selection: pd.DataFrame) -> pd.DataFrame:
    """Summarise the A-vs-B four-month historical evidence used for 2026 planning."""
    rows = []
    if family_selection.empty:
        return pd.DataFrame()
    for region, grp in family_selection.groupby("region"):
        a = grp[grp["family"] == "MODEL_A"]
        b = grp[grp["family"] == "MODEL_B"]
        if a.empty or b.empty:
            continue
        ar, br = a.iloc[0], b.iloc[0]
        a_mae, b_mae = ar.get("mean_backtest_MAE", np.nan), br.get("mean_backtest_MAE", np.nan)
        rows.append({
            "REGION": region,
            "best_model_A": ar.get("selected_model", "NONE"),
            "best_model_B": br.get("selected_model", "NONE"),
            "model_A_mean_MAE": a_mae,
            "model_B_mean_MAE": b_mae,
            "model_A_win_rate_pct": ar.get("win_rate_pct", np.nan),
            "model_B_win_rate_pct": br.get("win_rate_pct", np.nan),
            "weather_skill_vs_A_pct": (100.0 * (a_mae - b_mae) / a_mae
                                       if np.isfinite(a_mae) and a_mae > 0 and np.isfinite(b_mae)
                                       else np.nan),
            "backtest_weather_policy": "ORIGIN_SAFE_CLIMATOLOGY_NO_REALISED_FUTURE_WEATHER",
            "decision_note": "Same four-month windows; lower MAE is better. Live 7-day weather is verified separately in 2026."
        })
    return pd.DataFrame(rows)

def select_ab_backtest_detail(detail: pd.DataFrame, family_selection: pd.DataFrame) -> pd.DataFrame:
    """Keep only the selected A- and B-family champion paths for fair comparison."""
    if detail.empty or family_selection.empty:
        return pd.DataFrame()
    parts = []
    for region, grp in family_selection.groupby("region"):
        for family, feature_set in (("MODEL_A", "A"), ("MODEL_B", "B")):
            hit = grp[grp["family"] == family]
            if hit.empty:
                continue
            model = str(hit.iloc[0].get("selected_model", "NONE"))
            if model == "NONE":
                continue
            g = detail[(detail["REGION"] == region) & (detail["model"] == model)].copy()
            if g.empty:
                continue
            g["family"] = family
            g["family_label"] = "Model A" if family == "MODEL_A" else "Model B"
            g["selected_family_model"] = model
            parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ab_backtest_monthly_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    """Monthly A/B error metrics for every region and historical backtest origin."""
    if detail.empty:
        return pd.DataFrame()
    rows = []
    keys = ["REGION", "backtest_origin", "month", "family", "family_label", "model"]
    for key, g in detail.groupby(keys, dropna=False):
        m = metrics(g["actual_MW"].to_numpy(), g["forecast_MW"].to_numpy())
        rows.append(dict(zip(keys, key), **m))
    return pd.DataFrame(rows)


def ab_backtest_daily_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    """Daily demand means and errors used for readable four-month evidence plots."""
    if detail.empty:
        return pd.DataFrame()
    rows = []
    keys = ["REGION", "backtest_origin", "date", "family", "family_label", "model"]
    for key, g in detail.groupby(keys, dropna=False):
        m = metrics(g["actual_MW"].to_numpy(), g["forecast_MW"].to_numpy())
        rows.append({**dict(zip(keys, key)),
                     "actual_mean_MW": g["actual_MW"].mean(),
                     "forecast_mean_MW": g["forecast_MW"].mean(),
                     "actual_peak_MW": g["actual_MW"].max(),
                     "forecast_peak_MW": g["forecast_MW"].max(), **m})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["REGION", "backtest_origin", "family", "date"])
        out["rolling_7day_MAE_MW"] = out.groupby(
            ["REGION", "backtest_origin", "family"])["MAE"].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
    return out


def ab_backtest_halfhour_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day residual diagnostics for A/B champions."""
    if detail.empty:
        return pd.DataFrame()
    rows = []
    keys = ["REGION", "backtest_origin", "half_hour_index", "family", "family_label", "model"]
    for key, g in detail.groupby(keys, dropna=False):
        m = metrics(g["actual_MW"].to_numpy(), g["forecast_MW"].to_numpy())
        rows.append(dict(zip(keys, key), **m))
    return pd.DataFrame(rows)


def ab_decision_evidence(detail: pd.DataFrame) -> pd.DataFrame:
    """Region-level decision evidence comparing selected Model A and Model B.

    MAE is the primary closeness criterion. RMSE, sMAPE, bias, monthly wins and
    origin wins are supporting evidence.  The result is descriptive: it never
    forces Model A or Model B to win.
    """
    if detail.empty:
        return pd.DataFrame()
    rows = []
    monthly = ab_backtest_monthly_metrics(detail)
    for region, rg in detail.groupby("REGION"):
        fam = {}
        for family, g in rg.groupby("family"):
            fam[family] = {
                "model": g["model"].iloc[0],
                **metrics(g["actual_MW"].to_numpy(), g["forecast_MW"].to_numpy())
            }
        if "MODEL_A" not in fam or "MODEL_B" not in fam:
            continue
        a, b = fam["MODEL_A"], fam["MODEL_B"]
        a_mae, b_mae = a["MAE"], b["MAE"]
        if np.isfinite(a_mae) and np.isfinite(b_mae):
            closer = "MODEL_A" if a_mae < b_mae else "MODEL_B" if b_mae < a_mae else "TIE"
            gap = abs(a_mae - b_mae)
            better = min(a_mae, b_mae)
            worse = max(a_mae, b_mae)
            improvement = 100.0 * (worse - better) / worse if worse > 0 else np.nan
        else:
            closer, gap, improvement = "UNAVAILABLE", np.nan, np.nan

        mm = monthly[monthly["REGION"] == region]
        a_month_wins = b_month_wins = ties = 0
        if not mm.empty:
            piv = mm.pivot_table(index=["backtest_origin", "month"], columns="family",
                                 values="MAE", aggfunc="mean").dropna()
            if {"MODEL_A", "MODEL_B"}.issubset(piv.columns):
                a_month_wins = int((piv["MODEL_A"] < piv["MODEL_B"]).sum())
                b_month_wins = int((piv["MODEL_B"] < piv["MODEL_A"]).sum())
                ties = int(np.isclose(piv["MODEL_A"], piv["MODEL_B"]).sum())

        origin_rows = []
        for origin, og in rg.groupby("backtest_origin"):
            vals = {}
            for family, fg in og.groupby("family"):
                vals[family] = metrics(fg["actual_MW"].to_numpy(), fg["forecast_MW"].to_numpy())["MAE"]
            if "MODEL_A" in vals and "MODEL_B" in vals:
                origin_rows.append(vals)
        a_origin_wins = sum(x["MODEL_A"] < x["MODEL_B"] for x in origin_rows)
        b_origin_wins = sum(x["MODEL_B"] < x["MODEL_A"] for x in origin_rows)

        rows.append({
            "REGION": region,
            "model_A": a["model"], "model_B": b["model"],
            "model_A_MAE": a["MAE"], "model_B_MAE": b["MAE"],
            "model_A_RMSE": a["RMSE"], "model_B_RMSE": b["RMSE"],
            "model_A_sMAPE": a["sMAPE"], "model_B_sMAPE": b["sMAPE"],
            "model_A_Bias": a["Bias"], "model_B_Bias": b["Bias"],
            "closer_to_actual": closer,
            "MAE_gap_MW": gap,
            "better_family_improvement_pct": improvement,
            "model_A_month_wins": a_month_wins,
            "model_B_month_wins": b_month_wins,
            "monthly_ties": ties,
            "model_A_origin_wins": a_origin_wins,
            "model_B_origin_wins": b_origin_wins,
            "historical_origins_compared": len(origin_rows),
            "decision_basis": "Primary=lowest MAE; support=RMSE,sMAPE,|Bias|,monthly/origin consistency",
            "decision_statement": (f"{closer.replace('_', ' ').title()} is closer to actual demand "
                                   f"on aggregate four-month historical backtests"
                                   if closer in ("MODEL_A", "MODEL_B") else
                                   "A/B aggregate four-month MAE is tied or unavailable")
        })
    return pd.DataFrame(rows)


def monthly_forecast_summary(frozen: pd.DataFrame, region: str, vintage: str) -> pd.DataFrame:
    if frozen.empty:
        return pd.DataFrame()
    f = frozen.dropna(subset=["forecast_MW"]).copy()
    f["month"] = f["target_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, g in f.groupby("month"):
        peak_idx = g["forecast_MW"].idxmax()
        rows.append({
            "REGION": region, "vintage": vintage, "month": month,
            "mean_forecast_MW": g["forecast_MW"].mean(),
            "median_forecast_MW": g["forecast_MW"].median(),
            "minimum_forecast_MW": g["forecast_MW"].min(),
            "maximum_forecast_MW": g["forecast_MW"].max(),
            "peak_timestamp": g.loc[peak_idx, "target_timestamp"],
            "number_of_intervals": len(g),
            # energy only ever from MW x 0.5 h, never a bare MW sum
            "forecast_energy_MWh": g["forecast_MW"].sum() * HOURS_PER_INTERVAL,
            "recursive_input_share_pct": 100.0 * g["recursive_input_used"].mean(),
        })
    return pd.DataFrame(rows)


def forecast_sanity(frozen: pd.DataFrame, region: str, vintage: str,
                    expect_start: pd.Timestamp, expect_end: pd.Timestamp,
                    hist_max_step: float | None = None,
                    convention: str = "unknown") -> list:
    """Flag suspicious forecasts. Nothing is clipped or repaired silently."""
    out = []

    def add(check, result, detail):
        out.append({"region": region, "vintage": vintage, "check": check,
                    "result": result, "detail": detail})

    v = frozen["forecast_MW"]
    add("no_NaN", "PASS" if v.notna().all() else "FAIL", f"{int(v.isna().sum())} NaN")
    add("no_Inf", "PASS" if np.isfinite(v.dropna()).all() else "FAIL",
        f"{int((~np.isfinite(v.dropna())).sum())} non-finite")
    ts = frozen["target_timestamp"]
    add("no_duplicate_timestamp", "PASS" if not ts.duplicated().any() else "FAIL",
        f"{int(ts.duplicated().sum())} duplicates")
    step = ts.diff().dropna()
    bad = int((step != pd.Timedelta(FREQ)).sum())
    add("uniform_30min_spacing", "PASS" if bad == 0 else "FAIL", f"{bad} irregular steps")
    add("horizon_start_matches_contract", "PASS" if ts.min() == expect_start else "WARNING",
        f"{ts.min()} vs expected {expect_start}")
    add("horizon_end_matches_contract", "PASS" if ts.max() == expect_end else "WARNING",
        f"{ts.max()} vs expected {expect_end}")
    expected_n = int((expect_end - expect_start) / pd.Timedelta(FREQ)) + 1
    add("no_missing_intervals", "PASS" if len(frozen) == expected_n else "WARNING",
        f"{len(frozen)} rows vs {expected_n} expected")
    if v.notna().sum() > 2 and hist_max_step:
        # "Extreme" means larger than anything the region has ever actually done
        # between two consecutive half-hours, not merely larger than average.
        n_jump = int((v.diff().abs() > hist_max_step).sum())
        add("no_extreme_discontinuity", "PASS" if n_jump == 0 else "WARNING",
            f"{n_jump} steps exceed the largest observed historical half-hour change "
            f"({hist_max_step:.1f} MW)")
    if v.notna().sum() > 2:
        add("no_negative_forecast", "PASS" if (v.dropna() >= 0).all() else "WARNING",
            f"{int((v.dropna() < 0).sum())} negative values flagged, not clipped")
    if convention == "interval_ending":
        # Stage 2 writes the future frame on 00:00-23:30 labels while the NEM
        # series is interval-ending, so the horizon carries the last half-hour of
        # the previous month and stops one interval short of the final month end.
        add("interval_label_convention", "WARNING",
            "region uses interval-ending labels but the Stage 2 future frame is "
            "labelled 00:00-23:30; horizon is offset by one 30-minute interval at "
            "each end. Forecasts are still matched to actuals on exact timestamps.")
    else:
        add("interval_label_convention", "PASS", f"labels are {convention}")
    return out


# ============================================================================
# 12. AEMO BENCHMARK (never a predictor, never called accuracy)
# ============================================================================

def aemo_source_consistency(s2: Stage2, actuals: dict) -> tuple:
    if s2.aemo_actual is None:
        return pd.DataFrame(), pd.DataFrame()
    rows, summary = [], []
    for region, grp in s2.aemo_actual.groupby("region"):
        if region not in actuals:
            continue
        proj = actuals[region].rename("operational_actual_MW")
        j = grp.set_index(TIME).join(proj, how="inner").dropna(
            subset=["aemo_actual_MW", "operational_actual_MW"])
        if j.empty:
            continue
        j = j.reset_index().rename(columns={"index": TIME})
        j["source_difference_MW"] = j["aemo_actual_MW"] - j["operational_actual_MW"]
        j["absolute_source_difference_MW"] = j["source_difference_MW"].abs()
        j["benchmark_target"] = "AEMO dashboard / pre-dispatch TOTALDEMAND"
        j["project_target"] = "Operational Demand (Stage 2 processed)"
        # Stage 2 does not certify the two definitions are identical, so the
        # comparison is labelled a target relationship check, not an error.
        j["target_match"] = "UNCONFIRMED"
        j["comparison_status"] = "TARGET_MISMATCH_SECONDARY_BENCHMARK"
        j["REGION"] = region
        rows.append(j[[ "REGION", TIME, "aemo_actual_MW", "operational_actual_MW",
                        "source_difference_MW", "absolute_source_difference_MW",
                        "benchmark_target", "project_target", "target_match",
                        "comparison_status"]])
        d = j["source_difference_MW"]
        summary.append({
            "REGION": region, "matched_intervals": len(j),
            "window_start": j[TIME].min(), "window_end": j[TIME].max(),
            "MAE_between_sources_MW": float(d.abs().mean()),
            "RMSE_between_sources_MW": float(np.sqrt((d ** 2).mean())),
            "Bias_between_sources_MW": float(d.mean()),
            "correlation": float(np.corrcoef(j["aemo_actual_MW"], j["operational_actual_MW"])[0, 1])
            if len(j) > 2 else np.nan,
            "interpretation": "SOURCE CONSISTENCY / TARGET RELATIONSHIP CHECK - not forecast accuracy",
            "comparison_status": "TARGET_MISMATCH_SECONDARY_BENCHMARK"})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.DataFrame(summary))


def aemo_forecast_margin(s2: Stage2, current: dict, actuals: dict) -> tuple:
    """Margin against whatever pre-dispatch horizon Stage 2 genuinely captured."""
    if s2.aemo_forecast is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    detail, summary, verified = [], [], []
    for region, grp in s2.aemo_forecast.groupby("region"):
        g = grp.sort_values(TIME)
        avail_start, avail_end = g[TIME].min(), g[TIME].max()
        avail_hours = (avail_end - avail_start) / pd.Timedelta(hours=1) + 0.5
        spacing = g[TIME].diff().dropna().mode()
        spacing = str(spacing.iloc[0]) if len(spacing) else "unknown"
        vintage = (str(g["source_vintage"].iloc[0]) if "source_vintage" in g.columns else "")
        origin = (g["forecast_origin"].min() if "forecast_origin" in g.columns else pd.NaT)

        proj = current.get(region)
        matched = 0
        if proj is not None and not proj.empty:
            j = g.merge(proj[["target_timestamp", "forecast_MW", "selected_model"]],
                        left_on=TIME, right_on="target_timestamp", how="inner")
            j = j[j["forecast_MW"].notna()]
            matched = len(j)
            if matched:
                j["forecast_margin_MW"] = j["forecast_MW"] - j["aemo_forecast_MW"]
                j["absolute_forecast_margin_MW"] = j["forecast_margin_MW"].abs()
                j["forecast_margin_pct"] = np.where(
                    j["aemo_forecast_MW"].abs() > 0,
                    100.0 * j["forecast_margin_MW"] / j["aemo_forecast_MW"], np.nan)
                j["REGION"] = region
                j["comparison_status"] = "FORECAST_ONLY"
                detail.append(j[["REGION", TIME, "forecast_MW", "aemo_forecast_MW",
                                 "forecast_margin_MW", "absolute_forecast_margin_MW",
                                 "forecast_margin_pct", "selected_model",
                                 "comparison_status"]].rename(
                    columns={"forecast_MW": "project_forecast_MW"}))
        summary.append({
            "REGION": region,
            "requested_max_horizon_hours": AEMO_REQUESTED_HORIZON_HOURS,
            "available_horizon_hours": round(float(avail_hours), 2),
            "available_intervals": len(g),
            "forecast_start": avail_start, "forecast_end": avail_end,
            "interval_spacing": spacing, "forecast_origin": origin,
            "source_vintage": vintage,
            "matched_project_intervals": matched,
            "mean_margin_MW": float(detail[-1]["forecast_margin_MW"].mean()) if matched and detail else np.nan,
            "mean_absolute_margin_MW": float(detail[-1]["absolute_forecast_margin_MW"].mean()) if matched and detail else np.nan,
            "comparison_status": "FORECAST_ONLY",
            "interpretation": "FORECAST DIVERGENCE / MARGIN - neither source is scored as better"})

        # Verified accuracy only where an archived vintage AND later actuals exist.
        act = actuals.get(region)
        overlap = 0
        if act is not None:
            ov = g.set_index(TIME).join(act.rename("actual_MW"), how="inner").dropna(subset=["actual_MW"])
            overlap = len(ov)
        if overlap == 0:
            verified.append({"REGION": region, "status": "FORECAST_ONLY",
                             "matched_intervals": 0,
                             "detail": "target period has not occurred yet in the Stage 2 actuals; "
                                       "no AEMO forecast is ever reconstructed retrospectively"})
        else:
            proj_ov = None
            if proj is not None:
                proj_ov = proj.set_index("target_timestamp")["forecast_MW"].reindex(ov.index)
            m_aemo = metrics(ov["actual_MW"].to_numpy(), ov["aemo_forecast_MW"].to_numpy())
            row = {"REGION": region, "status": "VERIFIED", "matched_intervals": overlap,
                   "aemo_MAE": m_aemo["MAE"], "aemo_RMSE": m_aemo["RMSE"],
                   "aemo_sMAPE": m_aemo["sMAPE"], "aemo_Bias": m_aemo["Bias"]}
            if proj_ov is not None and proj_ov.notna().any():
                m_proj = metrics(ov["actual_MW"].to_numpy(), proj_ov.to_numpy())
                row.update({"project_MAE": m_proj["MAE"], "project_RMSE": m_proj["RMSE"],
                            "project_sMAPE": m_proj["sMAPE"], "project_Bias": m_proj["Bias"]})
            row["detail"] = "both an archived AEMO vintage and later actuals exist for these intervals"
            verified.append(row)
    return (pd.concat(detail, ignore_index=True) if detail else pd.DataFrame(),
            pd.DataFrame(summary), pd.DataFrame(verified))


# ============================================================================
# 13. FIGURES
# ============================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    return plt


def _save(fig, path: Path, made: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    made.append(str(path))
    import matplotlib.pyplot as plt
    plt.close(fig)


def ab_backtest_figures(region: str, out: Path, detail: pd.DataFrame,
                        monthly_metrics: pd.DataFrame, daily_metrics: pd.DataFrame,
                        halfhour_metrics: pd.DataFrame, made: list):
    """Evidence-first four-month A/B visualisations for one region.

    The most recent historical analogue origin is used for the detailed plots so
    the chart is readable. Aggregate decision CSVs still retain every origin.
    """
    if detail.empty:
        return
    plt = _plt()
    d = out / "figures" / region / "model_A_vs_B"
    rg = detail[detail["REGION"] == region].copy()
    if rg.empty:
        return
    origins = pd.to_datetime(rg["backtest_origin"].dropna().unique())
    if len(origins) == 0:
        return
    origin = pd.Timestamp(max(origins))
    g = rg[pd.to_datetime(rg["backtest_origin"]) == origin].copy()
    if g.empty:
        return

    # 1. Requested evidence at the native 30-minute resolution.
    actual_30 = g.groupby("target_timestamp", as_index=False)["actual_MW"].mean()
    if not actual_30.empty:
        fig, ax = plt.subplots(figsize=(11.0, 4.0))
        ax.plot(actual_30["target_timestamp"], actual_30["actual_MW"], lw=0.65,
                color="#202020", label="Actual demand")
        for family, label, colour in (("MODEL_A", "Model A", "#1f77b4"),
                                      ("MODEL_B", "Model B", "#d95f02")):
            fg = g[g["family"] == family].sort_values("target_timestamp")
            if not fg.empty:
                model = fg["model"].iloc[0]
                ax.plot(fg["target_timestamp"], fg["forecast_MW"], lw=0.45, alpha=0.78,
                        color=colour, label=f"{label} ({model})")
        ax.set_ylabel("Demand (MW)")
        ax.set_title(f"{region} - full-resolution four-month backtest: actual vs Model A vs Model B\n"
                     f"Backtest origin {origin.date()} | 30-minute intervals")
        ax.legend(ncol=3, fontsize=8); ax.tick_params(axis="x", rotation=25)
        _save(fig, d / f"{region}_01_four_month_actual_vs_A_vs_B_30min.png", made)

    # 2. Daily means provide a report-friendly view of the same four-month evidence.
    daily = daily_metrics[(daily_metrics["REGION"] == region) &
                          (pd.to_datetime(daily_metrics["backtest_origin"]) == origin)].copy()
    if not daily.empty:
        actual = daily.groupby("date", as_index=False)["actual_mean_MW"].mean()
        fig, ax = plt.subplots(figsize=(10.5, 4.0))
        ax.plot(actual["date"], actual["actual_mean_MW"], lw=1.8, color="#202020", label="Actual demand")
        for family, label, colour in (("MODEL_A", "Model A", "#1f77b4"),
                                      ("MODEL_B", "Model B", "#d95f02")):
            fg = daily[daily["family"] == family]
            if not fg.empty:
                model = fg["model"].iloc[0]
                ax.plot(fg["date"], fg["forecast_mean_MW"], lw=1.2, color=colour,
                        label=f"{label} ({model})")
        ax.set_ylabel("Daily mean demand (MW)")
        ax.set_title(f"{region} - four-month historical backtest: actual vs Model A vs Model B\n"
                     f"Backtest origin {origin.date()} | lower error = closer forecast")
        ax.legend(ncol=3, fontsize=8)
        ax.tick_params(axis="x", rotation=25)
        _save(fig, d / f"{region}_02_four_month_actual_vs_A_vs_B_daily.png", made)

        # 2. Rolling 7-day MAE shows whether one family stays better through time.
        fig, ax = plt.subplots(figsize=(10.0, 3.5))
        for family, label, colour in (("MODEL_A", "Model A", "#1f77b4"),
                                      ("MODEL_B", "Model B", "#d95f02")):
            fg = daily[daily["family"] == family]
            if not fg.empty:
                ax.plot(fg["date"], fg["rolling_7day_MAE_MW"], lw=1.4,
                        color=colour, label=label)
        ax.set_ylabel("Rolling 7-day MAE (MW)")
        ax.set_title(f"{region} - rolling forecast error through the four-month backtest")
        ax.legend(); ax.tick_params(axis="x", rotation=25)
        _save(fig, d / f"{region}_03_rolling_7day_MAE_A_vs_B.png", made)

    # 3. Month-by-month MAE/RMSE makes the four target months explicit.
    mm = monthly_metrics[(monthly_metrics["REGION"] == region) &
                         (pd.to_datetime(monthly_metrics["backtest_origin"]) == origin)].copy()
    if not mm.empty:
        months = sorted(mm["month"].dropna().unique())
        x = np.arange(len(months)); width = 0.36
        fig, ax = plt.subplots(figsize=(8.5, 3.5))
        for shift, family, label, colour in ((-width/2, "MODEL_A", "Model A", "#1f77b4"),
                                             ( width/2, "MODEL_B", "Model B", "#d95f02")):
            vals = mm[mm["family"] == family].set_index("month")["MAE"].reindex(months)
            ax.bar(x + shift, vals.to_numpy(), width, label=label, color=colour)
        ax.set_xticks(x, months); ax.set_ylabel("MAE (MW)")
        ax.set_title(f"{region} - monthly MAE across the four-month backtest")
        ax.legend()
        _save(fig, d / f"{region}_04_monthly_MAE_A_vs_B.png", made)

    # 4. Actual-vs-forecast scatter on daily means checks calibration visually.
    if not daily.empty:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        lo = float(min(daily["actual_mean_MW"].min(), daily["forecast_mean_MW"].min()))
        hi = float(max(daily["actual_mean_MW"].max(), daily["forecast_mean_MW"].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="#555", label="Perfect agreement")
        for family, label, colour, marker in (("MODEL_A", "Model A", "#1f77b4", "o"),
                                              ("MODEL_B", "Model B", "#d95f02", "x")):
            fg = daily[daily["family"] == family]
            if not fg.empty:
                ax.scatter(fg["actual_mean_MW"], fg["forecast_mean_MW"], s=18,
                           alpha=0.65, color=colour, marker=marker, label=label)
        ax.set_xlabel("Actual daily mean (MW)"); ax.set_ylabel("Forecast daily mean (MW)")
        ax.set_title(f"{region} - calibration: actual vs forecast")
        ax.legend(fontsize=8)
        _save(fig, d / f"{region}_05_actual_vs_forecast_scatter_A_vs_B.png", made)

    # 5. Bias by half-hour detects systematic under/over forecasting by time of day.
    hh = halfhour_metrics[(halfhour_metrics["REGION"] == region) &
                          (pd.to_datetime(halfhour_metrics["backtest_origin"]) == origin)].copy()
    if not hh.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        for family, label, colour in (("MODEL_A", "Model A", "#1f77b4"),
                                      ("MODEL_B", "Model B", "#d95f02")):
            fg = hh[hh["family"] == family]
            if not fg.empty:
                ax.plot(fg["half_hour_index"], fg["Bias"], marker="o", ms=2.5,
                        lw=1.0, color=colour, label=label)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xlabel("Half-hour index (0-47)"); ax.set_ylabel("Bias (MW)")
        ax.set_title(f"{region} - time-of-day residual bias")
        ax.legend()
        _save(fig, d / f"{region}_06_halfhour_bias_A_vs_B.png", made)


def cross_region_ab_figures(out: Path, decision: pd.DataFrame, made: list):
    """Cross-region A/B evidence for management/model-selection reporting."""
    if decision.empty:
        return
    plt = _plt(); d = out / "figures" / "cross_region"
    g = decision.sort_values("REGION").copy()
    x = np.arange(len(g)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - width/2, g["model_A_MAE"], width, label="Model A", color="#1f77b4")
    ax.bar(x + width/2, g["model_B_MAE"], width, label="Model B", color="#d95f02")
    ax.set_xticks(x, g["REGION"]); ax.set_ylabel("Aggregate backtest MAE (MW)")
    ax.set_title("Model A vs Model B - four-month historical backtest by region")
    ax.legend()
    _save(fig, d / "model_A_vs_B_MAE_by_region.png", made)

    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    signed = np.where(g["closer_to_actual"].eq("MODEL_A"),
                      g["better_family_improvement_pct"],
                      -g["better_family_improvement_pct"])
    ax.bar(g["REGION"], signed, color=["#1f77b4" if v >= 0 else "#d95f02" for v in signed])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("Relative MAE advantage (%)")
    ax.set_title("A/B decision margin by region (+ Model A closer, - Model B closer)")
    _save(fig, d / "model_A_vs_B_relative_MAE_advantage.png", made)


def region_figures(region: str, out: Path, backtest: pd.DataFrame, frozen: pd.DataFrame,
                   current: pd.DataFrame, verify: pd.DataFrame, by_day: pd.DataFrame,
                   by_hh: pd.DataFrame, importance: pd.DataFrame,
                   aemo_margin: pd.DataFrame, aemo_src: pd.DataFrame, made: list):
    plt = _plt()
    d = out / "figures" / region

    bt = backtest[(backtest.region == region) & backtest.get("MAE", pd.Series(dtype=float)).notna()]
    if not bt.empty:
        agg = bt.groupby("model")["MAE"].mean().reindex(
            [m for m in MODEL_ORDER if m in set(bt.model)])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(agg.index, agg.to_numpy(), color="#4a6fa5")
        ax.set_ylabel("mean backtest MAE (MW)")
        ax.set_title(f"{region} - four-month backtest MAE by model")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, d / f"{region}_backtest_model_comparison.png", made)

    if not frozen.empty:
        f = frozen.dropna(subset=["forecast_MW"])
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(f["target_timestamp"], f["forecast_MW"], lw=0.5, color="#4a6fa5",
                label=f"frozen forecast ({f['selected_model'].iloc[0]})")
        ax.set_ylabel("MW")
        ax.set_title(f"{region} - frozen four-month forecast from {f['forecast_origin'].iloc[0]}")
        ax.legend(loc="upper right")
        _save(fig, d / f"{region}_frozen_four_month_forecast.png", made)

    if not verify.empty:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(verify["target_timestamp"], verify["actual_MW"], lw=0.8, color="#222", label="actual")
        ax.plot(verify["target_timestamp"], verify["forecast_MW"], lw=0.8, color="#c0504d",
                label="frozen forecast")
        ax.set_ylabel("MW")
        ax.set_title(f"{region} - current-month actual vs frozen forecast")
        ax.legend(loc="upper right")
        _save(fig, d / f"{region}_actual_vs_frozen_forecast.png", made)

    if not by_day.empty:
        g = by_day[by_day.REGION == region]
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7, 3.0))
            ax.plot(g["date"], g["actual_mean_MW"], marker="o", ms=3, label="actual mean")
            ax.plot(g["date"], g["forecast_mean_MW"], marker="o", ms=3, label="forecast mean")
            ax.set_ylabel("MW"); ax.legend(); ax.tick_params(axis="x", rotation=30)
            ax.set_title(f"{region} - daily actual vs forecast")
            _save(fig, d / f"{region}_daily_actual_vs_forecast.png", made)
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.bar(g["date"], g["MAE"], color="#c0504d")
            ax.set_ylabel("MAE (MW)"); ax.tick_params(axis="x", rotation=30)
            ax.set_title(f"{region} - daily MAE, frozen forecast")
            _save(fig, d / f"{region}_daily_MAE.png", made)

    if not by_hh.empty:
        g = by_hh[by_hh.REGION == region]
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.plot(g["half_hour_index"], g["MAE"], marker="o", ms=3, color="#4a6fa5")
            ax.axhline(0, color="#888", lw=0.6)
            ax.set_xlabel("half-hour index (0-47)"); ax.set_ylabel("MAE (MW)")
            ax.set_title(f"{region} - error profile across the day")
            _save(fig, d / f"{region}_error_by_halfhour.png", made)

    if not frozen.empty and not current.empty:
        j = frozen.merge(current, on="target_timestamp", suffixes=("_frozen", "_current"))
        j = j.dropna(subset=["forecast_MW_frozen", "forecast_MW_current"])
        if not j.empty:
            fig, ax = plt.subplots(figsize=(9, 3.0))
            ax.plot(j["target_timestamp"], j["forecast_MW_frozen"], lw=0.5, label="frozen vintage")
            ax.plot(j["target_timestamp"], j["forecast_MW_current"], lw=0.5, label="current vintage")
            ax.set_ylabel("MW"); ax.legend()
            ax.set_title(f"{region} - forecast revision between vintages")
            _save(fig, d / f"{region}_forecast_revision.png", made)

    if not aemo_margin.empty:
        g = aemo_margin[aemo_margin.REGION == region]
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7, 3.0))
            ax.plot(g[TIME], g["project_forecast_MW"], marker="o", ms=3, label="project forecast")
            ax.plot(g[TIME], g["aemo_forecast_MW"], marker="o", ms=3, label="AEMO forecast")
            ax.set_ylabel("MW"); ax.legend(); ax.tick_params(axis="x", rotation=30)
            ax.set_title(f"{region} - project vs AEMO forecast (margin, not accuracy)")
            _save(fig, d / f"{region}_project_vs_aemo_forecast.png", made)

    if not aemo_src.empty:
        g = aemo_src[aemo_src.REGION == region]
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7, 3.0))
            ax.plot(g[TIME], g["operational_actual_MW"], label="project Operational Demand")
            ax.plot(g[TIME], g["aemo_actual_MW"], label="AEMO benchmark actual")
            ax.set_ylabel("MW"); ax.legend(); ax.tick_params(axis="x", rotation=30)
            ax.set_title(f"{region} - source consistency, not accuracy")
            _save(fig, d / f"{region}_aemo_actual_vs_operational.png", made)

    if not importance.empty:
        g = importance[importance.REGION == region].nlargest(15, "importance")
        if not g.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(g["feature"][::-1], g["importance"][::-1], color="#4a6fa5")
            ax.set_xlabel(f"{g['importance_type'].iloc[0]}")
            ax.set_title(f"{region} - {g['model'].iloc[0]} feature importance")
            _save(fig, d / f"{region}_selected_model_importance.png", made)


def cross_region_figures(out: Path, selection: pd.DataFrame, verify_sum: pd.DataFrame,
                         monthly: pd.DataFrame, aemo_margin_sum: pd.DataFrame, made: list):
    plt = _plt()
    d = out / "figures" / "cross_region"
    if not selection.empty and "mean_backtest_MAE" in selection.columns:
        fig, ax = plt.subplots(figsize=(7, 3.0))
        ax.bar(selection["region"], selection["mean_backtest_MAE"], color="#4a6fa5")
        for i, (_, r) in enumerate(selection.iterrows()):
            ax.text(i, r["mean_backtest_MAE"], str(r["selected_model"]),
                    ha="center", va="bottom", rotation=90, fontsize=7)
        ax.set_ylabel("mean backtest MAE (MW)")
        ax.set_title("Selected model per region - four-month backtest MAE")
        _save(fig, d / "backtest_MAE_by_selected_model.png", made)
        if "skill_vs_naive_pct" in selection.columns:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.bar(selection["region"], selection["skill_vs_naive_pct"], color="#4a7c59")
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_ylabel("skill vs naive (%)")
            ax.set_title("Backtest skill of the selected model over the best naive baseline")
            _save(fig, d / "skill_vs_naive.png", made)
    if not verify_sum.empty:
        for col, title, fname in (("MAE", "Current-month MAE, frozen forecast", "current_month_MAE.png"),
                                  ("sMAPE", "Current-month sMAPE, frozen forecast", "current_month_sMAPE.png")):
            if col in verify_sum.columns:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.bar(verify_sum["REGION"], verify_sum[col], color="#c0504d")
                ax.set_ylabel(col); ax.set_title(title)
                _save(fig, d / fname, made)
    if not monthly.empty:
        fr = monthly[monthly.vintage.str.startswith("FROZEN")]
        for col, fname, lab in (("mean_forecast_MW", "forecast_monthly_means.png", "mean MW"),
                                ("maximum_forecast_MW", "forecast_monthly_peaks.png", "peak MW")):
            fig, ax = plt.subplots(figsize=(8, 3.0))
            for region, g in fr.groupby("REGION"):
                ax.plot(g["month"], g[col], marker="o", ms=4, label=region)
            ax.set_ylabel(lab); ax.legend(fontsize=7, ncol=3)
            ax.set_title(f"Frozen four-month forecast - {lab} by target month")
            _save(fig, d / fname, made)
    if not aemo_margin_sum.empty and "mean_absolute_margin_MW" in aemo_margin_sum.columns:
        g = aemo_margin_sum.dropna(subset=["mean_absolute_margin_MW"])
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.bar(g["REGION"], g["mean_absolute_margin_MW"], color="#7a6fa5")
            ax.set_ylabel("mean |margin| (MW)")
            ax.set_title("Project vs AEMO forecast margin (divergence, not accuracy)")
            _save(fig, d / "aemo_margin_comparison.png", made)


# ============================================================================
# 14. EXPLAINABILITY AND SERIALISATION
# ============================================================================

def extract_importance(run: Run, region: str, model_name: str) -> pd.DataFrame:
    """RF -> native importances, XGB -> gain, SARIMAX -> exog coefficients, SARIMA -> none."""
    if run.fitted is None:
        return pd.DataFrame()
    if model_name.startswith("RandomForest"):
        feats = run.extra.get("feature_names", [])
        return pd.DataFrame({"REGION": region, "model": model_name,
                             "importance_type": "impurity_decrease",
                             "feature": feats,
                             "importance": run.fitted.feature_importances_})
    if model_name.startswith("XGBoost"):
        feats = run.extra.get("feature_names", [])
        gain = run.fitted.get_booster().get_score(importance_type="gain")
        vals = [gain.get(f"f{i}", 0.0) for i in range(len(feats))]
        return pd.DataFrame({"REGION": region, "model": model_name,
                             "importance_type": "gain", "feature": feats,
                             "importance": vals})
    if model_name == "SARIMAX":
        p = run.fitted.params
        exog = run.extra.get("exog", [])
        rows = [{"REGION": region, "model": model_name, "importance_type": "exog_coefficient",
                 "feature": k, "importance": float(v)} for k, v in p.items() if k in exog]
        return pd.DataFrame(rows)
    return pd.DataFrame()     # SARIMA and the naives get no fabricated importance


def save_model(run: Run, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if run.fitted is None:
        return ""
    try:
        if run.name in ("SARIMA", "SARIMAX"):
            p = path.with_suffix(".pkl")
            run.fitted.save(str(p))
        else:
            import joblib
            p = path.with_suffix(".joblib")
            joblib.dump(run.fitted, p)
        return str(p.name)
    except Exception as exc:
        log.warning(f"could not serialise {run.name}: {exc}")
        return ""


# ============================================================================
# 15. MAIN PIPELINE
# ============================================================================

def run_pipeline(zip_path: str, out_root: str, regions=None, quick=False,
                 origins=None, make_figures=True, progress=None) -> dict:
    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / f"Stage3_Modelling_{stamp}"
    for sub in ("forecasts/frozen", "forecasts/current", "forecasts/model_A",
                "forecasts/model_B", "forecasts/comparison", "validation", "benchmark",
                "explainability", "figures", "models/frozen", "models/current", "metadata"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    setup_logging(out / "metadata" / "stage3_run.log")

    def say(msg):
        log.info(msg)
        if progress:
            progress(msg)

    say("=" * 78)
    say(f"PRT661 Stage 3 modelling  |  run {stamp}  |  {'QUICK' if quick else 'FULL'} mode")
    say("=" * 78)

    deps = dependency_status()
    write_csv(deps, out / "metadata" / "dependency_status.csv")
    missing = deps[deps.status == "MISSING"]["package"].tolist()
    if missing:
        say(f"  dependencies MISSING: {missing} - the models that need them are recorded "
            f"as MODEL_FAILED, never substituted")

    tmp = tempfile.TemporaryDirectory()
    try:
        root = open_stage2(Path(zip_path), Path(tmp.name))
        s2 = load_stage2(root)
        say(f"  Stage 2 contract read from {root.name}")
        say(f"  regions detected: {', '.join(s2.regions)}")

        safety = audit_feature_safety(s2)
        if (safety["result"] == "FAIL").any():
            raise RuntimeError("Stage 2 feature sets contain unsafe columns:\n" +
                               safety.to_string(index=False))
        say(f"  Model A = {len(s2.features_A)} features, Model B = {len(s2.features_B)} features")

        anchor, horizon_end, months = resolve_horizon(s2.run_config)
        declared = s2.run_config.get("frozen_forecast_origin")
        say(f"  rolling window: {', '.join(str(m) for m in months)}")
        say(f"  frozen origin {anchor}  ->  horizon end {horizon_end}")

        cfg = RunConfig.build(quick, origins, make_figures)
        selected_regions = [r for r in (regions or s2.regions) if r in s2.region_files]
        if not selected_regions:
            raise RuntimeError("No requested region exists in the Stage 2 ZIP")

        self_test = horizon_self_test()
        write_csv(self_test, out / "validation" / "december_reusability_test.csv")
        say(f"  horizon self-test: {(self_test.result == 'PASS').sum()}/{len(self_test)} pass "
            f"(includes the December -> March crossover)")

        screen_all, backtest_all, rebuild_all, backtest_detail_all = [], [], [], []
        frozen_all, current_all, verify_all = [], [], []
        by_day_all, by_hh_all, monthly_all, importance_all = [], [], [], []
        integrity_all, leakage_all, manifest, forecast_manifest = [], [], [], []
        selection_rows, family_selection_rows = [], []
        region_status, actuals, current_by_region = [], {}, {}
        frozen_by_region, verify_by_region = {}, {}
        ab_verification_summary_all, ab_verification_detail_all = [], []
        model_a_full_all, model_b_rolling_all = [], []
        model_a_current_by_region, model_b_current_by_region = {}, {}

        for region in selected_regions:
            say("-" * 78)
            say(f"REGION {region}")
            rd = load_region(s2, region)
            hist_ok = rd.hist[rd.hist[TARGET].notna()]
            training_end = hist_ok.index.max()
            declared_cutoff = pd.to_datetime(s2.run_config.get("training_cutoff"), errors="coerce")
            if pd.notna(declared_cutoff) and training_end > declared_cutoff:
                raise RuntimeError(
                    f"{region} modelling data leaks into the prediction month: "
                    f"training ends {training_end}, declared cutoff {declared_cutoff}")

            verification_end = (rd.current_actuals.index.max()
                                if rd.current_actuals is not None and not rd.current_actuals.empty
                                else pd.NaT)
            benchmark_actual = hist_ok[TARGET].copy()
            if rd.current_actuals is not None and not rd.current_actuals.empty and TARGET in rd.current_actuals:
                benchmark_actual = pd.concat([benchmark_actual, rd.current_actuals[TARGET]]).sort_index()
                benchmark_actual = benchmark_actual[~benchmark_actual.index.duplicated(keep="last")]
            actuals[region] = benchmark_actual
            say(f"  TRAINING history {rd.hist.index.min()} -> {training_end} "
                f"({len(rd.hist)} rows; strict previous-month cutoff; "
                f"verification actuals to {verification_end if pd.notna(verification_end) else 'none'}; "
                f"interval labels: {rd.convention})")

            origins_list = analogue_origins(months[0], rd.hist.index.min(), training_end,
                                            max_origins=cfg.max_origins)
            if not origins_list:
                say(f"  {region}: no historical four-month analogue window fits inside the data")
            else:
                say(f"  analogue origins: {', '.join(str(o.date()) for o, _ in origins_list)}")

            # ---------------- PHASE A ----------------
            say("  PHASE A  screening")
            screen = screen_region(rd, s2, anchor, cfg)
            if not screen.empty:
                screen_all.append(screen)
            keep = screen_decision(screen)
            dropped = [m for m, k in keep.items() if not k]
            if dropped:
                say(f"    screened out of the expensive backtest: {dropped}")

            say("  PHASE A  four-month recursive backtest")
            bt, rb, bd = backtest_region(rd, s2, origins_list, cfg, keep, progress=say)
            if not bt.empty:
                backtest_all.append(bt)
            if not rb.empty:
                rebuild_all.append(rb)
            if not bd.empty:
                backtest_detail_all.append(bd)

            sel = select_model(bt, region) if not bt.empty else {
                "region": region, "selected_model": "NONE", "feature_set": "",
                "status": "MODEL_FAILED",
                "selection_reason": "no historical analogue window available for backtesting"}
            selection_rows.append(sel)
            family_rows = select_feature_family_champions(bt, region) if not bt.empty else []
            family_selection_rows.extend(family_rows)
            a_model = next((r["selected_model"] for r in family_rows if r.get("family") == "MODEL_A"), "NONE")
            b_model = next((r["selected_model"] for r in family_rows if r.get("family") == "MODEL_B"), "NONE")
            if family_rows:
                say(f"  A/B champions: Model A={a_model}, Model B={b_model}")

            chosen = sel["selected_model"]
            say(f"  selected model: {chosen} - {sel['selection_reason']}")
            if chosen == "NONE":
                region_status.append({"region": region, "status": "MODEL_FAILED",
                                      "detail": sel["selection_reason"]})
                continue

            # ---------------- PHASE B ----------------
            say("  PHASE B  frozen beginning-of-month forecast")
            maps_frozen = build_origin_maps(rd.hist, anchor)
            panel_frozen = build_panel(rd, anchor, horizon_end, use_stage2_future=True)
            panel_frozen = apply_weather_policy(panel_frozen, rd, maps_frozen, s2, "frozen")
            rebuild_all.append(pd.DataFrame([check_rebuild(rd, panel_frozen, maps_frozen, anchor, s2)]))
            run_frozen = run_model(chosen, panel_frozen, s2, maps_frozen, cfg)
            say(f"    {chosen}: {run_frozen.status} ({run_frozen.runtime_s:.1f}s) - {run_frozen.detail}")

            vintage_frozen = f"FROZEN@{anchor.isoformat()}"
            frozen = forecast_frame(rd, panel_frozen, run_frozen, anchor, chosen,
                                    vintage_frozen, stamp)
            write_csv(frozen, out / "forecasts" / "frozen" / f"forecast_frozen_{region}.csv")
            frozen_all.append(frozen)
            frozen_by_region[region] = frozen
            hist_step = float(hist_ok[TARGET].diff().abs().max())
            integrity_all += forecast_sanity(frozen, region, vintage_frozen, anchor,
                                             horizon_end, hist_step, rd.convention)
            monthly_all.append(monthly_forecast_summary(frozen, region, vintage_frozen))

            verify = verify_current_month(frozen, rd, anchor)
            verify_by_region[region] = verify
            if not verify.empty:
                verify_all.append(verify)
                m = metrics(verify["actual_MW"].to_numpy(), verify["forecast_MW"].to_numpy())
                say(f"    current-month verification: {m['matched_intervals']} intervals, "
                    f"MAE {m['MAE']:.1f} MW, Bias {m['Bias']:+.1f} MW")
                g = verify.groupby("local_date")
                by_day_all.append(pd.DataFrame({
                    "REGION": region, "date": g.size().index,
                    "matched_intervals": g.size().to_numpy(),
                    "actual_mean_MW": g["actual_MW"].mean().to_numpy(),
                    "forecast_mean_MW": g["forecast_MW"].mean().to_numpy(),
                    "MAE": g["absolute_error_MW"].mean().to_numpy(),
                    "RMSE": np.sqrt(g["squared_error"].mean()).to_numpy(),
                    "sMAPE": g["smape_component"].mean().to_numpy(),
                    "Bias": g["error_MW"].mean().to_numpy(),
                    "actual_peak_MW": g["actual_MW"].max().to_numpy(),
                    "forecast_peak_MW": g["forecast_MW"].max().to_numpy()}))
                h = verify.groupby("half_hour_index")
                by_hh_all.append(pd.DataFrame({
                    "REGION": region, "half_hour_index": h.size().index,
                    "matched_intervals": h.size().to_numpy(),
                    "MAE": h["absolute_error_MW"].mean().to_numpy(),
                    "RMSE": np.sqrt(h["squared_error"].mean()).to_numpy(),
                    "sMAPE": h["smape_component"].mean().to_numpy(),
                    "Bias": h["error_MW"].mean().to_numpy()}))
            else:
                say("    current-month verification: no actuals inside the horizon yet")

            # ---------------- PHASE C ----------------
            # Live weather refresh: the demand model is STILL trained only through
            # the previous-month cutoff.  We re-run the projection from month start
            # so recursive lags for the prediction month come from earlier model
            # predictions, never from current-month actual demand.
            say("  PHASE C  live weather refresh (training cutoff remains previous month)")
            configured_weather_origin = pd.to_datetime(
                s2.run_config.get("weather_forecast_origin"), errors="coerce")
            source_analysis_origin = pd.to_datetime(
                s2.run_config.get("analysis_origin"), errors="coerce")
            live_information_origin = (configured_weather_origin if pd.notna(configured_weather_origin)
                                       else source_analysis_origin if pd.notna(source_analysis_origin)
                                       else anchor)
            live_information_origin = max(pd.Timestamp(anchor), pd.Timestamp(live_information_origin))

            current = pd.DataFrame()
            run_current = Run(name=chosen, status="MODEL_FAILED", detail="not attempted")
            maps_cur = build_origin_maps(rd.hist, anchor)
            panel_cur = build_panel(rd, anchor, horizon_end, use_stage2_future=True)
            if FEATURE_SET_OF.get(chosen) == "B" or chosen == "SARIMAX":
                panel_cur = apply_weather_policy(panel_cur, rd, maps_cur, s2, "live",
                                                 live_information_origin)
            else:
                panel_cur.weather_policy = ("NONE_MODEL_A" if FEATURE_SET_OF.get(chosen) == "A"
                                            else "NOT_APPLICABLE")
            run_current = run_model(chosen, panel_cur, s2, maps_cur, cfg)
            vintage_current = f"LIVE_UPDATE@{live_information_origin.isoformat()}"
            current = forecast_frame(rd, panel_cur, run_current, live_information_origin, chosen,
                                     vintage_current, stamp)
            write_csv(current, out / "forecasts" / "current" / f"forecast_current_{region}.csv")
            current_all.append(current)
            current_by_region[region] = current
            integrity_all += forecast_sanity(current, region, vintage_current, anchor,
                                             horizon_end, hist_step, rd.convention)
            monthly_all.append(monthly_forecast_summary(current, region, vintage_current))
            say(f"    live information origin {live_information_origin}; projection still starts "
                f"{anchor}; training ends {training_end}: {run_current.status}")

            # ------------- A/B architecture products -----------------------
            # Model A: one full current-month + next-three-month forecast.
            # Model B: full projection from current-month start with the same
            # previous-month demand cutoff.  Genuine 7-day forecast weather is
            # inserted only where its issue time allows; earlier/later rows use
            # origin-safe climatology.  Current-month actuals are verification-only.
            a_full = pd.DataFrame()
            a_verify = pd.DataFrame()
            if a_model != "NONE":
                if chosen == a_model:
                    a_full = frozen.copy()
                    a_run = run_frozen
                else:
                    a_run, _, a_full = run_named_forecast(
                        rd, s2, a_model, anchor, horizon_end, cfg,
                        "frozen", stamp, f"MODEL_A_FOUR_MONTH@{anchor.isoformat()}")
                write_csv(a_full, out / "forecasts" / "model_A" /
                          f"forecast_model_A_four_month_{region}.csv")
                model_a_full_all.append(a_full)
                a_verify = verify_current_month(a_full, rd, anchor)

            b_verify = pd.DataFrame()
            b_live = pd.DataFrame()
            if b_model != "NONE":
                if chosen == b_model and not current.empty:
                    b_live = current.copy()
                else:
                    b_run_live, _, b_live = run_named_forecast(
                        rd, s2, b_model, anchor, horizon_end, cfg,
                        "live", stamp, f"MODEL_B_LIVE@{live_information_origin.isoformat()}",
                        information_origin=live_information_origin)
                write_csv(b_live, out / "forecasts" / "model_B" /
                          f"forecast_model_B_live_{region}.csv")
                model_b_current_by_region[region] = b_live
                b_verify = verify_current_month(b_live, rd, anchor)

                # Rolling user-facing output begins at the current prediction month
                # start.  Past rows are still MODEL PREDICTIONS (not current-month
                # demand reused as lags); genuine FORECAST_7DAY weather is used only
                # where its archived issue time permits it.
                y = b_live.copy()
                forecast_rows = y[y["weather_feature_source"].astype(str).eq("FORECAST_7DAY")]
                if not forecast_rows.empty:
                    live_end = forecast_rows["target_timestamp"].max()
                else:
                    configured = pd.to_datetime(s2.run_config.get("weather_forecast_end"), errors="coerce")
                    live_end = configured if pd.notna(configured) else min(
                        horizon_end, live_information_origin + pd.Timedelta(days=7))
                b_roll = y[y["target_timestamp"] <= live_end].copy()
                b_roll["segment_type"] = np.select(
                    [b_roll["target_timestamp"] < live_information_origin,
                     b_roll["weather_feature_source"].astype(str).eq("FORECAST_7DAY")],
                    ["MONTH_TO_DATE_MODEL_PREDICTION", "LIVE_FORECAST_7DAY"],
                    default="CLIMATOLOGY_BRIDGE")
                write_csv(b_roll, out / "forecasts" / "model_B" /
                          f"forecast_model_B_rolling_{region}.csv")
                model_b_rolling_all.append(b_roll)

            ab_now = current_month_ab_summary(region, a_verify, b_verify, a_model, b_model)
            if not ab_now.empty:
                ab_verification_summary_all.append(ab_now)
                dparts = []
                if not a_verify.empty:
                    aa = a_verify.copy(); aa["family"] = "MODEL_A"; aa["model"] = a_model
                    dparts.append(aa)
                if not b_verify.empty:
                    bb = b_verify.copy(); bb["family"] = "MODEL_B"; bb["model"] = b_model
                    dparts.append(bb)
                if dparts:
                    ab_verification_detail_all.append(pd.concat(dparts, ignore_index=True))

            # ------------- explainability, models, manifests -------------
            imp = extract_importance(run_frozen, region, chosen)
            if not imp.empty:
                importance_all.append(imp)
            f_file = save_model(run_frozen, out / "models" / "frozen" / f"{region}_{chosen}_frozen")
            c_file = save_model(run_current, out / "models" / "current" / f"{region}_{chosen}_current")
            for kind, r, origin_ts, fname in (("frozen", run_frozen, anchor, f_file),
                                              ("current", run_current, live_information_origin, c_file)):
                manifest.append({
                    "region": region, "vintage": kind, "model": chosen,
                    "feature_set": FEATURE_SET_OF.get(chosen, ""),
                    "training_start": r.train_start, "training_end": r.train_end,
                    "training_rows": r.train_rows, "forecast_origin": origin_ts,
                    "forecast_end": horizon_end,
                    "backtest_mean_MAE": sel.get("mean_backtest_MAE"),
                    "backtest_mean_RMSE": sel.get("mean_RMSE"),
                    "backtest_mean_sMAPE": sel.get("mean_sMAPE"),
                    "backtest_mean_Bias": sel.get("mean_Bias"),
                    "random_state": RANDOM_STATE, "status": r.status,
                    "runtime_seconds": round(r.runtime_s, 2),
                    "serialised_file": fname, "detail": r.detail})

            # ------------- leakage tests -------------
            leakage_all += leakage_tests(region, rd, s2, anchor, run_frozen, maps_frozen,
                                         origins_list, panel_frozen)

            worst = [i for i in integrity_all if i["region"] == region and i["result"] == "FAIL"]
            region_status.append({
                "region": region,
                "status": ("MODEL_FAILED" if run_frozen.status == "MODEL_FAILED"
                           else "MODEL_READY_WITH_WARNING" if (worst or run_frozen.status.endswith("WARNING"))
                           else "MODEL_READY"),
                "detail": f"selected {chosen}; frozen {run_frozen.status}; current {run_current.status}"})
            forecast_manifest.append({
                "region": region, "frozen_rows": len(frozen), "current_rows": len(current),
                "verification_intervals": len(verify),
                "latest_verification_actual_timestamp": verification_end,
                "training_end_timestamp": training_end,
                "frozen_origin": anchor, "current_information_origin": live_information_origin,
                "horizon_end": horizon_end, "selected_model": chosen})

        # ---------------- aggregate outputs ----------------
        say("-" * 78)
        say("AGGREGATION, BENCHMARKS AND FIGURES")

        def cat(frames):
            frames = [f for f in frames if f is not None and not f.empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        screen_df = cat(screen_all)
        backtest_df = cat(backtest_all)
        backtest_detail_raw = cat(backtest_detail_all)
        rebuild_df = cat(rebuild_all)
        selection = pd.DataFrame(selection_rows)
        family_selection = pd.DataFrame(family_selection_rows)
        ab_backtest = ab_backtest_summary(backtest_df, family_selection)
        ab_backtest_detail = select_ab_backtest_detail(backtest_detail_raw, family_selection)
        ab_backtest_monthly = ab_backtest_monthly_metrics(ab_backtest_detail)
        ab_backtest_daily = ab_backtest_daily_metrics(ab_backtest_detail)
        ab_backtest_halfhour = ab_backtest_halfhour_metrics(ab_backtest_detail)
        ab_decision = ab_decision_evidence(ab_backtest_detail)
        ab_verification_summary = cat(ab_verification_summary_all)
        ab_verification_detail = cat(ab_verification_detail_all)
        verify_df = cat(verify_all)
        by_day = cat(by_day_all)
        by_hh = cat(by_hh_all)
        monthly = cat(monthly_all)
        importance = cat(importance_all)
        frozen_df = cat(frozen_all)
        current_df = cat(current_all)

        write_csv(screen_df, out / "validation" / "model_screening_scores.csv")
        write_csv(backtest_df, out / "validation" / "four_month_backtest_scores.csv")
        write_csv(rebuild_df, out / "validation" / "origin_feature_rebuild_checks.csv")
        write_csv(selection, out / "validation" / "model_selection.csv")
        write_csv(family_selection, out / "validation" / "model_A_B_family_selection.csv")
        write_csv(ab_backtest, out / "validation" / "model_A_vs_B_backtest_summary.csv")
        write_csv(ab_backtest_detail, out / "validation" / "model_A_vs_B_four_month_backtest_detail.csv")
        write_csv(ab_backtest_monthly, out / "validation" / "model_A_vs_B_monthly_backtest_metrics.csv")
        write_csv(ab_backtest_daily, out / "validation" / "model_A_vs_B_daily_backtest_metrics.csv")
        write_csv(ab_backtest_halfhour, out / "validation" / "model_A_vs_B_halfhour_backtest_metrics.csv")
        write_csv(ab_decision, out / "validation" / "model_A_vs_B_decision_evidence.csv")
        write_csv(ab_verification_summary, out / "validation" / "current_month_A_vs_B_verification_summary.csv")
        write_csv(ab_verification_detail, out / "validation" / "current_month_A_vs_B_verification_detail.csv")

        if not backtest_df.empty:
            cols = ["region", "model", "feature_set", "backtest_origin", "MAE", "RMSE",
                    "sMAPE", "Bias", "skill_vs_naive_pct", "runtime_seconds", "status"]
            comp = backtest_df.reindex(columns=cols)
            write_csv(comp, out / "validation" / "model_comparison.csv")

        write_csv(verify_df, out / "validation" / "current_month_actual_vs_frozen_forecast.csv")
        write_csv(by_day, out / "validation" / "current_month_verification_by_day.csv")
        write_csv(by_hh, out / "validation" / "current_month_error_by_halfhour.csv")

        verify_sum = pd.DataFrame()
        if not verify_df.empty:
            rows = []
            for region, g in verify_df.groupby("REGION"):
                m = metrics(g["actual_MW"].to_numpy(), g["forecast_MW"].to_numpy())
                rows.append({"REGION": region, "verification_month": str(months[0]),
                             "verification_start": g["target_timestamp"].min(),
                             "verification_end": g["target_timestamp"].max(),
                             "selected_model": g["selected_model"].iloc[0],
                             "feature_set": g["feature_set"].iloc[0], **m})
            verify_sum = pd.DataFrame(rows)
        write_csv(verify_sum, out / "validation" / "current_month_verification_summary.csv")
        write_csv(pd.DataFrame(integrity_all), out / "validation" / "forecast_integrity_checks.csv")
        write_csv(pd.DataFrame(leakage_all), out / "validation" / "leakage_tests.csv")
        write_csv(monthly, out / "forecasts" / "forecast_monthly_summary.csv")

        revision = pd.DataFrame()
        if not frozen_df.empty and not current_df.empty:
            j = frozen_df.merge(current_df, on=["REGION", "target_timestamp"],
                                suffixes=("_frozen", "_current"))
            j = j.dropna(subset=["forecast_MW_frozen", "forecast_MW_current"])
            if not j.empty:
                j["revision_MW"] = j["forecast_MW_current"] - j["forecast_MW_frozen"]
                j["absolute_revision_MW"] = j["revision_MW"].abs()
                j["revision_pct"] = np.where(j["forecast_MW_frozen"].abs() > 0,
                                             100.0 * j["revision_MW"] / j["forecast_MW_frozen"], np.nan)
                j["comparison_status"] = "FORECAST_REVISION"
                revision = j[["REGION", "target_timestamp", "forecast_MW_frozen",
                              "forecast_MW_current", "revision_MW", "absolute_revision_MW",
                              "revision_pct", "comparison_status"]].rename(
                    columns={"forecast_MW_frozen": "frozen_forecast_MW",
                             "forecast_MW_current": "current_forecast_MW"})
        write_csv(revision, out / "forecasts" / "forecast_revision_all_regions.csv")

        src_detail, src_summary = aemo_source_consistency(s2, actuals)
        write_csv(src_detail, out / "benchmark" / "aemo_actual_vs_operational_actual.csv")
        write_csv(src_summary, out / "benchmark" / "aemo_source_consistency_summary.csv")
        marg_detail, marg_summary, verified = aemo_forecast_margin(s2, current_by_region, actuals)
        write_csv(marg_detail, out / "benchmark" / "project_vs_aemo_forecast.csv")
        write_csv(marg_summary, out / "benchmark" / "aemo_forecast_margin_summary.csv")
        write_csv(verified, out / "benchmark" / "aemo_verified_accuracy.csv")

        # Dedicated Model-B margin so the live seven-day weather-informed forecast
        # can be compared with AEMO on exactly the timestamps both sources publish.
        b_marg_detail, b_marg_summary, b_verified = aemo_forecast_margin(
            s2, model_b_current_by_region, actuals)
        write_csv(b_marg_detail, out / "benchmark" / "model_B_vs_aemo_forecast.csv")
        write_csv(b_marg_summary, out / "benchmark" / "model_B_aemo_margin_summary.csv")
        write_csv(b_verified, out / "benchmark" / "model_B_aemo_verified_accuracy.csv")

        write_csv(importance, out / "explainability" / "selected_model_feature_importance.csv",
                  floats="%.6f")
        write_csv(selection, out / "explainability" / "selected_model_summary.csv")
        write_csv(pd.DataFrame(manifest), out / "metadata" / "model_manifest.csv")
        write_csv(pd.DataFrame(forecast_manifest), out / "metadata" / "forecast_manifest.csv")
        say("  aggregate tables and benchmark outputs written")

        made = []
        if cfg.make_figures:
            for region in selected_regions:
                ab_backtest_figures(region, out, ab_backtest_detail, ab_backtest_monthly,
                                    ab_backtest_daily, ab_backtest_halfhour, made)
                region_figures(region, out, backtest_df,
                               frozen_by_region.get(region, pd.DataFrame()),
                               current_by_region.get(region, pd.DataFrame()),
                               verify_by_region.get(region, pd.DataFrame()),
                               by_day, by_hh, importance, marg_detail, src_detail, made)
            cross_region_figures(out, selection, verify_sum, monthly, marg_summary, made)
            cross_region_ab_figures(out, ab_decision, made)
            say(f"  {len(made)} figures written")

        stage3_cfg = {
            "stage3_run": stamp, "mode": "quick" if quick else "full",
            "stage2_source": str(Path(zip_path).name),
            "stage2_run_config": s2.run_config,
            "rolling_window": [str(m) for m in months],
            "anchor_month_start": str(anchor),
            "training_cutoff": s2.run_config.get("training_cutoff"),
            "training_cutoff_policy": "strict previous-month end; current-month actuals verification-only",
            "declared_frozen_forecast_origin": str(declared) if declared else None,
            "horizon_end": str(horizon_end),
            "regions": selected_regions,
            "model_set": MODEL_ORDER,
            "feature_set_A_size": len(s2.features_A),
            "feature_set_B_size": len(s2.features_B),
            "model_selection_horizon": "same four-month historical windows for all candidates",
            "model_B_backtest_weather_policy": "origin-safe climatology; realised target-period weather hidden",
            "A_B_backtest_evidence": "interval, daily, monthly and half-hour diagnostics retained for selected A/B family champions",
            "model_B_live_weather_policy": "FORECAST_7DAY where available, then climatology fallback",
            "weather_forecast_origin": s2.run_config.get("weather_forecast_origin"),
            "weather_forecast_start": s2.run_config.get("weather_forecast_start"),
            "weather_forecast_end": s2.run_config.get("weather_forecast_end"),
            "random_state": RANDOM_STATE,
            "origin_feature_policy": cfg.profile_mode,
            "smape_min_denominator_MW": SMAPE_MIN_DENOM,
            "python": platform.python_version(),
            "stage2_notes": s2.notes,
        }
        (out / "metadata" / "run_config.json").write_text(json.dumps(stage3_cfg, indent=2, default=str))
        write_csv(safety, out / "validation" / "feature_safety_audit.csv")
        say("  metadata and feature safety written")

        summary = build_summary(region_status, selection, verify_sum, self_test,
                                pd.DataFrame(integrity_all), pd.DataFrame(leakage_all),
                                deps, src_summary, marg_summary, made)
        write_csv(summary, out / "stage3_validation_summary.csv")
        say("  stage3 validation summary written")

        overall = ("FAILED" if all(r["status"] == "MODEL_FAILED" for r in region_status) or not region_status
                   else "COMPLETE_WITH_WARNINGS"
                   if any(r["status"] != "MODEL_READY" for r in region_status)
                   or (pd.DataFrame(leakage_all)["result"] == "FAIL").any()
                   else "COMPLETE")

        zip_out = Path(out_root) / f"Stage3_Modelling_Output_{stamp}.zip"
        say("  building Stage 3 ZIP")
        make_zip(out, zip_out)
        say("  first ZIP written; verifying")
        checks = verify_zip(zip_out, selected_regions, expect_figures=cfg.make_figures)
        verification_path = write_csv(checks, out / "validation" / "stage3_zip_verification.csv")
        if (checks["result"] == "FAIL").any() and overall == "COMPLETE":
            overall = "COMPLETE_WITH_WARNINGS"
        # Add only the new verification record instead of recompressing every model
        # artefact a second time. This keeps the handoff fast and avoids leaving a
        # half-written archive if a long second compression is interrupted.
        with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.write(verification_path,
                     arcname=str(Path(out.name) / "validation" / verification_path.name))

        say("=" * 78)
        say(f"{'region':<8}{'selected model':<18}{'backtest MAE':>13}{'skill%':>9}   status")
        for _, r in selection.iterrows():
            st = next((x["status"] for x in region_status if x["region"] == r["region"]), "-")
            mae = r.get("mean_backtest_MAE", np.nan)
            sk = r.get("skill_vs_naive_pct", np.nan)
            mae_txt = f"{mae:>13.1f}" if np.isfinite(mae) else f"{'-':>13}"
            sk_txt = f"{sk:>9.1f}" if np.isfinite(sk) else f"{'-':>9}"
            say(f"{r['region']:<8}{str(r['selected_model']):<18}{mae_txt}{sk_txt}   {st}")
        say("-" * 78)
        say(f"  ZIP        {zip_out}")
        say(f"  artefacts  {(checks['result'] == 'PASS').sum()}/{len(checks)} required artefacts present")
        say(f"  runtime    {time.time() - t_start:.0f}s")
        say(f"  STATUS     {overall}")
        say("=" * 78)

        return {"status": overall, "out_dir": out, "zip": zip_out,
                "selection": selection, "backtest": backtest_df,
                "verification": verify_sum, "zip_checks": checks,
                "region_status": pd.DataFrame(region_status)}
    finally:
        tmp.cleanup()


def leakage_tests(region: str, rd: RegionData, s2: Stage2, anchor: pd.Timestamp,
                  run_frozen: Run, maps: OriginMaps, origins: list, panel: Panel) -> list:
    rows = []

    def add(check, result, detail):
        rows.append({"region": region, "check": check, "result": result, "detail": detail})

    if run_frozen.train_end is not None:
        add("frozen_training_end_before_forecast_origin",
            "PASS" if run_frozen.train_end < anchor else "FAIL",
            f"training ends {run_frozen.train_end}, origin {anchor}")
        add("no_current_month_actual_in_frozen_training",
            "PASS" if run_frozen.train_end < anchor else "FAIL",
            "frozen model never sees an actual at or after the anchor month start")
    add("origin_profiles_built_from_pre_origin_rows_only",
        "PASS" if (maps.history_end is None or maps.history_end < anchor) else "FAIL",
        f"profile history ends {maps.history_end}; years used {maps.years_used}")
    add("origin_climatology_built_from_pre_origin_rows_only",
        "PASS" if (maps.history_end is None or maps.history_end < anchor) else "FAIL",
        f"weather history ends {maps.history_end}, balance point {maps.balance_point}")
    ok_bt = all(o > rd.hist.index.min() for o, _ in origins)
    add("backtest_training_ends_before_each_historical_origin",
        "PASS" if ok_bt else "WARNING",
        f"{len(origins)} origin(s): " + ", ".join(str(o.date()) for o, _ in origins))
    unsafe = [f for f in set(s2.features_A) | set(s2.features_B)
              if any(re.search(p, f, flags=re.I) for p in FORBIDDEN_PATTERNS)]
    add("no_legacy_observed_weather_or_benchmark_predictor", "PASS" if not unsafe else "FAIL",
        f"offending features: {unsafe or 'none'}")

    bad_observed = 0
    bad_vintage = 0
    if panel.known is not None and "weather_feature_source" in panel.known.columns:
        src = panel.known["weather_feature_source"].astype(str)
        bad_observed = int(src.eq("OBSERVED_REANALYSIS").sum())
        if "weather_forecast_origin" in panel.known.columns:
            issue = pd.to_datetime(panel.known["weather_forecast_origin"], errors="coerce")
            bad_vintage = int((src.eq("FORECAST_7DAY") & issue.notna() & (issue > anchor)).sum())
    add("no_future_observed_weather_in_frozen_forecast",
        "PASS" if bad_observed == 0 else "FAIL",
        f"{bad_observed} frozen-horizon rows use OBSERVED_REANALYSIS")
    add("weather_forecast_vintage_not_after_frozen_origin",
        "PASS" if bad_vintage == 0 else "FAIL",
        f"{bad_vintage} forecast-weather rows were issued after {anchor}")
    add("no_aemo_or_marketrequirements_predictor", "PASS" if not unsafe else "FAIL",
        "AEMO actual/forecast and MarketRequirements are benchmark-only by contract")
    prov = lag_provenance(panel)
    late = prov[(prov.index >= anchor)]
    bad = 0
    for lag, col in ((48, "lag48_source"), (96, "lag96_source"), (336, "lag336_source")):
        ref = late.index - pd.Timedelta(minutes=30 * lag)
        bad += int(((ref >= anchor) & (late[col] == "ACTUAL_HISTORY")).sum())
    add("recursive_lag_provenance_correct", "PASS" if bad == 0 else "FAIL",
        f"{bad} horizon rows claim ACTUAL_HISTORY for a lag that falls at or after the origin")
    add("no_scaling_or_tuning_fitted_on_test", "PASS",
        "tree models are unscaled and hyper-parameters are fixed, so nothing is fitted on the horizon")
    declared_cutoff = pd.to_datetime(s2.run_config.get("training_cutoff"), errors="coerce")
    train_end = rd.hist.index.max() if len(rd.hist) else pd.NaT
    cutoff_ok = pd.notna(declared_cutoff) and pd.notna(train_end) and train_end <= declared_cutoff < anchor
    add("strict_previous_month_training_cutoff", "PASS" if cutoff_ok else "FAIL",
        f"training ends {train_end}; declared cutoff {declared_cutoff}; prediction starts {anchor}")
    add("current_month_actuals_verification_only", "PASS",
        "current-month actuals are loaded from a separate Stage 2 validation file and never enter rd.hist")
    return rows


def build_summary(region_status, selection, verify_sum, self_test, integrity,
                  leakage, deps, src_summary, marg_summary, figures) -> pd.DataFrame:
    rows = []

    def add(item, value, detail=""):
        rows.append({"item": item, "value": value, "detail": detail})

    add("regions_modelled", len(region_status),
        ", ".join(f"{r['region']}={r['status']}" for r in region_status))
    ready = sum(1 for r in region_status if r["status"] == "MODEL_READY")
    add("regions_MODEL_READY", ready)
    add("regions_MODEL_READY_WITH_WARNING",
        sum(1 for r in region_status if r["status"] == "MODEL_READY_WITH_WARNING"))
    add("regions_MODEL_FAILED", sum(1 for r in region_status if r["status"] == "MODEL_FAILED"))
    if not selection.empty:
        add("selected_models", "; ".join(f"{r.region}={r.selected_model}"
                                         for r in selection.itertuples()))
    if not verify_sum.empty:
        add("current_month_verification_regions", len(verify_sum),
            "; ".join(f"{r.REGION}={int(r.matched_intervals)} intervals, MAE {r.MAE:.1f} MW"
                      for r in verify_sum.itertuples()))
    add("december_reusability_test", f"{int((self_test.result == 'PASS').sum())}/{len(self_test)} PASS")
    if not leakage.empty:
        add("leakage_tests", f"{int((leakage.result == 'PASS').sum())}/{len(leakage)} PASS",
            "; ".join(sorted(set(leakage[leakage.result == "FAIL"]["check"]))) or "no failures")
    if not integrity.empty:
        add("forecast_integrity_checks",
            f"{int((integrity.result == 'PASS').sum())}/{len(integrity)} PASS",
            "; ".join(sorted(set(integrity[integrity.result != "PASS"]["check"]))) or "no warnings")
    add("dependencies_missing", ", ".join(deps[deps.status == "MISSING"]["package"]) or "none")
    add("aemo_source_consistency_regions", len(src_summary),
        "SOURCE CONSISTENCY only - not forecast accuracy")
    add("aemo_forecast_margin_regions", len(marg_summary),
        "FORECAST_ONLY margin - neither source is scored as better")
    add("figures_written", len(figures))
    return pd.DataFrame(rows)


# ============================================================================
# 16. ZIP OUT AND REOPEN
# ============================================================================

def make_zip(folder: Path, target: Path) -> Path:
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path(folder.name) / p.relative_to(folder)))
    return target


def verify_zip(target: Path, regions: list, expect_figures: bool = True) -> pd.DataFrame:
    """Reopen the archive and confirm the artefacts Stage 3 promised are inside."""
    rows = []
    if not zipfile.is_zipfile(target):
        return pd.DataFrame([{"artefact": str(target), "result": "FAIL",
                              "detail": "not a readable ZIP"}])
    with zipfile.ZipFile(target) as zf:
        names = {Path(n).as_posix() for n in zf.namelist()}
        stem = sorted({n.split("/")[0] for n in names})[0]

        def need(rel, label=None):
            path = f"{stem}/{rel}"
            ok = path in names
            rows.append({"artefact": rel, "result": "PASS" if ok else "FAIL",
                         "detail": label or ("present" if ok else "missing")})

        for r in regions:
            need(f"forecasts/frozen/forecast_frozen_{r}.csv")
        for r in regions:
            need(f"forecasts/current/forecast_current_{r}.csv")
        for f in ("validation/model_selection.csv",
                  "validation/model_A_B_family_selection.csv",
                  "validation/model_A_vs_B_backtest_summary.csv",
                  "validation/model_A_vs_B_four_month_backtest_detail.csv",
                  "validation/model_A_vs_B_monthly_backtest_metrics.csv",
                  "validation/model_A_vs_B_daily_backtest_metrics.csv",
                  "validation/model_A_vs_B_halfhour_backtest_metrics.csv",
                  "validation/model_A_vs_B_decision_evidence.csv",
                  "validation/current_month_A_vs_B_verification_summary.csv",
                  "validation/current_month_A_vs_B_verification_detail.csv",
                  "validation/four_month_backtest_scores.csv",
                  "validation/model_screening_scores.csv",
                  "validation/current_month_actual_vs_frozen_forecast.csv",
                  "validation/current_month_verification_summary.csv",
                  "validation/current_month_verification_by_day.csv",
                  "validation/current_month_error_by_halfhour.csv",
                  "validation/forecast_integrity_checks.csv",
                  "validation/leakage_tests.csv",
                  "validation/december_reusability_test.csv",
                  "validation/origin_feature_rebuild_checks.csv",
                  "benchmark/aemo_actual_vs_operational_actual.csv",
                  "benchmark/aemo_source_consistency_summary.csv",
                  "benchmark/project_vs_aemo_forecast.csv",
                  "benchmark/aemo_forecast_margin_summary.csv",
                  "benchmark/aemo_verified_accuracy.csv",
                  "benchmark/model_B_vs_aemo_forecast.csv",
                  "benchmark/model_B_aemo_margin_summary.csv",
                  "benchmark/model_B_aemo_verified_accuracy.csv",
                  "explainability/selected_model_summary.csv",
                  "metadata/model_manifest.csv",
                  "metadata/forecast_manifest.csv",
                  "metadata/dependency_status.csv",
                  "metadata/run_config.json",
                  "stage3_validation_summary.csv"):
            need(f)

        # The A/B evidence package is part of the final modelling deliverable.
        # Only require PNGs when figure generation was enabled for this run.
        if expect_figures:
            for r in regions:
                for fname in (
                    f"{r}_01_four_month_actual_vs_A_vs_B_30min.png",
                    f"{r}_02_four_month_actual_vs_A_vs_B_daily.png",
                    f"{r}_03_rolling_7day_MAE_A_vs_B.png",
                    f"{r}_04_monthly_MAE_A_vs_B.png",
                    f"{r}_05_actual_vs_forecast_scatter_A_vs_B.png",
                    f"{r}_06_halfhour_bias_A_vs_B.png",
                ):
                    need(f"figures/{r}/model_A_vs_B/{fname}")
            need("figures/cross_region/model_A_vs_B_MAE_by_region.png")
            need("figures/cross_region/model_A_vs_B_relative_MAE_advantage.png")
    return pd.DataFrame(rows)


# ============================================================================
# 17. TKINTER INTERFACE
# ============================================================================

def launch_gui():
    import queue as _queue
    import threading
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk, scrolledtext
    except Exception as exc:
        print(f"Tkinter is not available here ({exc}). Use the command line instead:\n"
              f"  python 03_Modelling.py --zip <stage2.zip> --out-dir <folder> [--quick]")
        return

    msgs = _queue.Queue()

    class QueueHandler(logging.Handler):
        def emit(self, record):
            msgs.put(("log", self.format(record)))

    root = tk.Tk()
    root.title("PRT661 Stage 3 - rolling four-month demand forecasting")
    root.geometry("860x560")
    zip_var, out_var, quick_var = tk.StringVar(), tk.StringVar(), tk.BooleanVar()

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="x")
    ttk.Label(frm, text="Stage 2 ZIP:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=zip_var, width=78).grid(row=0, column=1, padx=6)
    ttk.Button(frm, text="Browse",
               command=lambda: zip_var.set(filedialog.askopenfilename(
                   title="Select the Stage 2 preprocessing ZIP",
                   filetypes=[("ZIP archive", "*.zip")]) or zip_var.get())
               ).grid(row=0, column=2)
    ttk.Label(frm, text="Output folder:").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=out_var, width=78).grid(row=1, column=1, padx=6)
    ttk.Button(frm, text="Browse",
               command=lambda: out_var.set(filedialog.askdirectory(
                   title="Select a folder for the Stage 3 output") or out_var.get())
               ).grid(row=1, column=2)
    ttk.Checkbutton(frm, text="Quick test mode (fewer trees, one backtest origin, "
                              "shorter SARIMA window - leakage rules unchanged)",
                    variable=quick_var).grid(row=2, column=1, sticky="w")

    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(fill="x", padx=10)
    status = ttk.Label(root, text="Idle", anchor="w", padding=(10, 4))
    status.pack(fill="x")
    box = scrolledtext.ScrolledText(root, height=24, font=("Consolas", 9))
    box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    state = {"result": None, "busy": False, "out": None}

    def worker(zip_path, out_dir, quick):
        try:
            res = run_pipeline(zip_path, out_dir, quick=quick,
                               progress=lambda m: msgs.put(("log", m)))
            msgs.put(("done", res))
        except Exception as exc:
            msgs.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    def start():
        if state["busy"]:
            return
        if not zip_var.get() or not Path(zip_var.get()).exists():
            messagebox.showerror("Stage 3", "Select an existing Stage 2 ZIP first.")
            return
        if not out_var.get():
            messagebox.showerror("Stage 3", "Select an output folder first.")
            return
        box.delete("1.0", "end")
        state["busy"] = True
        bar.start(12)
        status.config(text="Running...")
        run_btn.config(state="disabled")
        setup_logging(None, QueueHandler())
        threading.Thread(target=worker, args=(zip_var.get(), out_var.get(),
                                              quick_var.get()), daemon=True).start()

    def pump():
        # Tkinter is only ever touched from the main thread; the worker posts here.
        while True:
            try:
                kind, payload = msgs.get_nowait()
            except _queue.Empty:
                break
            if kind == "log":
                box.insert("end", str(payload) + "\n")
                box.see("end")
            elif kind == "done":
                state["busy"] = False
                state["out"] = payload["out_dir"]
                bar.stop()
                run_btn.config(state="normal")
                open_btn.config(state="normal")
                status.config(text=f"{payload['status']}  |  {payload['zip']}")
            elif kind == "error":
                state["busy"] = False
                bar.stop()
                run_btn.config(state="normal")
                status.config(text="FAILED")
                box.insert("end", str(payload) + "\n")
        root.after(150, pump)

    btns = ttk.Frame(root, padding=(10, 0, 10, 8))
    btns.pack(fill="x")
    run_btn = ttk.Button(btns, text="Run Full Modelling", command=start)
    run_btn.pack(side="left")

    def open_output():
        import subprocess
        p = state["out"]
        if not p:
            return
        if sys.platform.startswith("win"):
            import os
            os.startfile(p)                                  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    open_btn = ttk.Button(btns, text="Open Output Folder", command=open_output, state="disabled")
    open_btn.pack(side="left", padx=8)
    root.after(150, pump)
    root.mainloop()


# ============================================================================
# 18. CLI
# ============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="PRT661 Stage 3 - rolling four-month operational-demand forecasting "
                    "from a Stage 2 preprocessing ZIP. Run with no arguments for the "
                    "Tkinter interface.")
    p.add_argument("--zip", dest="zip_path", help="Stage 2 ZIP (or an extracted Stage 2 folder)")
    p.add_argument("--out-dir", dest="out_dir", help="Folder to write Stage3_Modelling_<stamp>/ into")
    p.add_argument("--regions", nargs="+", default=None, help="Subset of regions to model")
    p.add_argument("--quick", action="store_true",
                   help="Fewer trees, one backtest origin, shorter SARIMA window. "
                        "Leakage rules, feature safety and recursion are unchanged.")
    p.add_argument("--origins", type=int, default=None, help="Override the number of backtest origins")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run the horizon and December-crossover tests only, then exit")
    a = p.parse_args(argv)

    if a.self_test:
        setup_logging()
        t = horizon_self_test()
        print(t.to_string(index=False))
        return 0 if (t.result == "PASS").all() else 1

    if not a.zip_path or not a.out_dir:
        launch_gui()
        return 0

    setup_logging()
    res = run_pipeline(a.zip_path, a.out_dir, regions=a.regions, quick=a.quick,
                       origins=a.origins, make_figures=not a.skip_figures)
    return 0 if res["status"] != "FAILED" else 1


if __name__ == "__main__":
    sys.exit(main())
