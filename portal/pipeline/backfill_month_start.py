from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, PRODUCTION_MODELS, REGIONS
from .database import init_db, upsert_frame, utcnow_naive
from .live_inference import DEMAND_FEATURES, _demand_features, _grid_positions, resolve_stage2_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("month_start_baseline")


def _bundle(region: str, model_name: str):
    from .live_inference import _load_bundle
    return _load_bundle(region, model_name)


def _month_bounds(month: str | pd.Timestamp | None):
    if month is None:
        start = pd.Timestamp.now(tz="Australia/Adelaide").tz_localize(None).normalize().replace(day=1)
    elif isinstance(month, str):
        start = pd.Timestamp(f"{month}-01") if len(month) == 7 else pd.Timestamp(month).normalize().replace(day=1)
    else:
        start = pd.Timestamp(month).normalize().replace(day=1)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _training_history(bundle, region: str, month_start: pd.Timestamp) -> pd.Series:
    root = resolve_stage2_root(bundle["stage2_root"])
    source = root / "data" / f"modelling_{region}_30min.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing Stage 2 modelling file: {source}")
    d = pd.read_csv(source, usecols=["SETTLEMENTDATE", "TOTALDEMAND"], parse_dates=["SETTLEMENTDATE"])
    d["SETTLEMENTDATE"] = pd.to_datetime(d["SETTLEMENTDATE"], errors="coerce")
    d["TOTALDEMAND"] = pd.to_numeric(d["TOTALDEMAND"], errors="coerce")
    d = d.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
    d = d[d["SETTLEMENTDATE"] < month_start].copy()
    d = d.drop_duplicates("SETTLEMENTDATE", keep="last").sort_values("SETTLEMENTDATE")
    if d.empty:
        raise RuntimeError(f"{region}: no pre-month training history before {month_start}")
    return d.set_index("SETTLEMENTDATE")["TOTALDEMAND"].astype(float)


def _future_frame(bundle, region: str, month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    root = resolve_stage2_root(bundle["stage2_root"])
    path = root / "data" / "future_feature_frame_all_regions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing future feature frame: {path}")
    f = pd.read_csv(path, parse_dates=["SETTLEMENTDATE", "local_timestamp", "local_date"], low_memory=False)
    key = "region" if "region" in f.columns else "REGION"
    f = f[f[key].astype(str) == region].copy()
    f["SETTLEMENTDATE"] = pd.to_datetime(f["SETTLEMENTDATE"], errors="coerce")
    f = f.dropna(subset=["SETTLEMENTDATE"])
    f = f[(f["SETTLEMENTDATE"] >= month_start) & (f["SETTLEMENTDATE"] < month_end)].copy()
    return f.drop_duplicates("SETTLEMENTDATE", keep="last").sort_values("SETTLEMENTDATE")


def forecast_month_start(region: str, model_name: str, month: str | pd.Timestamp | None = None) -> pd.DataFrame:
    month_start, month_end = _month_bounds(month)
    bundle = _bundle(region, model_name)

    training_end = pd.Timestamp(bundle.get("training_end"))
    if pd.isna(training_end) or training_end >= month_start:
        raise RuntimeError(
            f"{region}/{model_name}: refusing historical baseline because model training_end={training_end} "
            f"is not strictly before month_start={month_start}. This prevents post-month leakage."
        )

    future = _future_frame(bundle, region, month_start, month_end)
    if future.empty:
        raise RuntimeError(f"{region}/{model_name}: Stage 2 has no future feature rows for {month_start:%Y-%m}")

    history = _training_history(bundle, region, month_start)
    grid_start = pd.Timestamp(history.index.min()).floor("30min")
    grid_end = pd.Timestamp(future["SETTLEMENTDATE"].max()).ceil("30min")
    idx = pd.date_range(grid_start, grid_end, freq="30min")

    y = pd.Series(np.nan, index=idx, dtype=float)
    common = history.index.intersection(idx)
    if len(common):
        y.loc[common] = history.loc[common].to_numpy(dtype=float)
    yv = np.array(y.to_numpy(dtype=float), dtype=float, copy=True)

    features = list(bundle["feature_names"])
    medians = pd.Series(bundle["feature_medians"], dtype=float)
    model = bundle["estimator"]
    forecasts: list[float] = []

    # No live weather overlay is used here. Model-B receives only the future fields
    # that existed in the Stage-2 month-start frame, preserving a forecast-time-safe baseline.
    for start in range(0, len(future), 48):
        block = future.iloc[start:start + 48].copy()
        positions = _grid_positions(idx, block["SETTLEMENTDATE"])
        dyn = _demand_features(yv, positions, set(features) & DEMAND_FEATURES)
        X = pd.DataFrame(index=block.index)
        for feature in features:
            if feature in dyn:
                X[feature] = dyn[feature]
            elif feature in block.columns:
                X[feature] = pd.to_numeric(block[feature], errors="coerce").to_numpy()
            else:
                X[feature] = np.nan
        X = X.reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(medians)
        pred = np.asarray(model.predict(X.to_numpy(dtype=np.float32)), dtype=float)
        yv[positions] = pred
        forecasts.extend(pred.tolist())

    created = utcnow_naive()
    return pd.DataFrame({
        "region": region,
        "model_name": model_name,
        "model_version": bundle["model_version"],
        "forecast_origin": month_start,
        "target_timestamp": future["SETTLEMENTDATE"].to_numpy(),
        "forecast_mw": forecasts,
        "created_at": created,
    })


def capture_month_start_for_all(month: str | pd.Timestamp | None = None, overwrite: bool = False) -> pd.DataFrame:
    month_start, _ = _month_bounds(month)
    frames = []
    for region in REGIONS:
        for model_name in PRODUCTION_MODELS[region]:
            if not overwrite:
                from .database import read_df
                existing = read_df(
                    """
                    SELECT COUNT(*) AS n FROM project_forecast
                    WHERE region=:r AND model_name=:m AND forecast_origin=:o
                    """,
                    {"r": region, "m": model_name, "o": month_start.to_pydatetime()},
                )
                if len(existing) and int(existing.iloc[0, 0] or 0) > 0:
                    log.info("%s/%s %s baseline already exists; skipping", region, model_name, month_start.strftime("%Y-%m"))
                    continue
            fc = forecast_month_start(region, model_name, month_start)
            upsert_frame(
                "project_forecast",
                fc,
                ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
            )
            frames.append(fc)
            log.info("%s/%s: stored %s month-start baseline rows for %s", region, model_name, f"{len(fc):,}", month_start.strftime("%Y-%m"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe month-start forecast baseline using a production bundle trained strictly before that month."
    )
    parser.add_argument("month", nargs="?", help="Month in YYYY-MM format. Defaults to current month.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute an existing baseline for the month.")
    args = parser.parse_args()
    init_db()
    out = capture_month_start_for_all(args.month, overwrite=args.overwrite)
    log.info("Month-start baseline complete; rows=%s", f"{len(out):,}")


if __name__ == "__main__":
    main()
