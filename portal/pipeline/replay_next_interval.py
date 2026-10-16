from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PRODUCTION_MODELS, REGIONS
from .database import init_db, read_df, upsert_frame, utcnow_naive
from .live_inference import DEMAND_FEATURES, _demand_features, _grid_positions, _history_for_region, _load_bundle, resolve_stage2_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("historical_next_interval_replay")


def _eligible_bundle(region: str, model_name: str, month_start: pd.Timestamp):
    """Load the newest stored model that was trained strictly before the replay month."""
    rows = read_df(
        """
        SELECT model_path, model_version, training_end, created_at
        FROM model_registry
        WHERE region=:r AND model_name=:m
          AND status='PRODUCTION'
          AND training_end < :month_start
        ORDER BY training_end DESC, created_at DESC
        LIMIT 1
        """,
        {"r": region, "m": model_name, "month_start": month_start.to_pydatetime()},
    )
    if not rows.empty:
        path = Path(str(rows.iloc[0]["model_path"]))
        if path.exists():
            return joblib.load(path)

    # Safe fallback for the current deployment: use CURRENT only when its training
    # cutoff is still strictly before the month being replayed.
    bundle = _load_bundle(region, model_name)
    training_end = pd.to_datetime(bundle.get("training_end"), errors="coerce")
    if pd.isna(training_end) or training_end >= month_start:
        raise RuntimeError(
            f"{region}/{model_name}: no leakage-safe model is available for {month_start:%Y-%m}; "
            f"current training_end={training_end}."
        )
    return bundle


def _replay_one(region: str, model_name: str, month: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)

    actual = read_df(
        """
        SELECT timestamp AS target_timestamp, demand_mw AS actual_mw
        FROM actual_demand
        WHERE region=:r AND timestamp>=:start AND timestamp<:end
        ORDER BY timestamp
        """,
        {"r": region, "start": start.to_pydatetime(), "end": end.to_pydatetime()},
    )
    if actual.empty:
        log.warning("%s/%s: no actual demand for %s", region, model_name, month)
        return pd.DataFrame()

    actual["target_timestamp"] = pd.to_datetime(actual["target_timestamp"], errors="coerce")
    actual["actual_mw"] = pd.to_numeric(actual["actual_mw"], errors="coerce")
    actual = actual.dropna(subset=["target_timestamp", "actual_mw"]).drop_duplicates("target_timestamp", keep="last")
    if actual.empty:
        return pd.DataFrame()

    # Do not manufacture replay points where a forecast was genuinely stored before
    # the target. Genuine stored evidence always wins on the dashboard.
    live = read_df(
        """
        SELECT DISTINCT target_timestamp
        FROM next_interval_forecast
        WHERE region=:r AND model_name=:m
          AND target_timestamp>=:start AND target_timestamp<:end
        """,
        {"r": region, "m": model_name, "start": start.to_pydatetime(), "end": end.to_pydatetime()},
    )
    if not live.empty:
        live_targets = set(pd.to_datetime(live["target_timestamp"], errors="coerce").dropna())
        actual = actual[~actual["target_timestamp"].isin(live_targets)].copy()
    if actual.empty:
        log.info("%s/%s: every %s actual already has a stored live forecast", region, model_name, month)
        return pd.DataFrame()

    bundle = _eligible_bundle(region, model_name, start)
    training_end = pd.to_datetime(bundle.get("training_end"), errors="coerce")
    log.info(
        "%s/%s: replay %s with model %s trained through %s",
        region,
        model_name,
        month,
        bundle.get("model_version", "unknown"),
        training_end,
    )

    root = resolve_stage2_root(bundle["stage2_root"])
    future_path = root / "data" / "future_feature_frame_all_regions.csv"
    if not future_path.exists():
        raise FileNotFoundError(f"Missing frozen future feature frame: {future_path}")

    future = pd.read_csv(
        future_path,
        parse_dates=["SETTLEMENTDATE", "local_timestamp", "local_date"],
        low_memory=False,
    )
    key = "region" if "region" in future.columns else "REGION"
    future = future[future[key].astype(str) == region].copy()
    future["SETTLEMENTDATE"] = pd.to_datetime(future["SETTLEMENTDATE"], errors="coerce")
    future = future.dropna(subset=["SETTLEMENTDATE"]).drop_duplicates("SETTLEMENTDATE", keep="last")

    block = actual.merge(
        future,
        left_on="target_timestamp",
        right_on="SETTLEMENTDATE",
        how="inner",
    ).sort_values("target_timestamp")
    if block.empty:
        log.warning("%s/%s: no frozen feature rows match actual targets for %s", region, model_name, month)
        return pd.DataFrame()

    # Demand-derived model inputs are all anchored at t-48 or earlier in this
    # project, so each reconstructed row uses only demand that was already known
    # before its target interval. Model-B exogenous columns come from the frozen
    # deployment feature frame; current live weather is deliberately NOT overlaid.
    history = _history_for_region(bundle, region)
    if history.empty:
        raise RuntimeError(f"{region}: no historical demand is available")

    grid_start = pd.Timestamp(history.index.min()).floor("30min")
    grid_end = pd.Timestamp(block["target_timestamp"].max()).ceil("30min")
    idx = pd.date_range(grid_start, grid_end, freq="30min")

    y = pd.Series(np.nan, index=idx, dtype=float)
    aligned = history.copy()
    if aligned.index.tz is not None and idx.tz is None:
        aligned.index = aligned.index.tz_localize(None)
    elif aligned.index.tz is None and idx.tz is not None:
        aligned.index = aligned.index.tz_localize(idx.tz)
    common = aligned.index.intersection(idx)
    if len(common):
        y.loc[common] = aligned.loc[common].to_numpy(dtype=float)
    yv = np.array(y.to_numpy(dtype=float), dtype=float, copy=True)

    positions = _grid_positions(idx, block["target_timestamp"])
    features = list(bundle["feature_names"])
    medians = pd.Series(bundle["feature_medians"], dtype=float)
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

    pred = np.asarray(bundle["estimator"].predict(X.to_numpy(dtype=np.float32)), dtype=float)
    out = pd.DataFrame({
        "region": region,
        "model_name": model_name,
        "model_version": bundle["model_version"],
        # Mirrors the production :05/:35 schedule: a 20:00 target would have been
        # issued at 19:35, giving a 25-minute operational lead.
        "forecast_origin": pd.to_datetime(block["target_timestamp"]) - pd.Timedelta(minutes=25),
        "target_timestamp": pd.to_datetime(block["target_timestamp"]),
        "forecast_mw": pred,
        "actual_mw": pd.to_numeric(block["actual_mw"], errors="coerce").to_numpy(dtype=float),
        "lead_minutes": 25.0,
        "replay_generated_at": utcnow_naive(),
    })
    out["error_mw"] = out["forecast_mw"] - out["actual_mw"]
    out["abs_error_mw"] = out["error_mw"].abs()
    out["squared_error"] = out["error_mw"] ** 2
    denom = out["forecast_mw"].abs() + out["actual_mw"].abs()
    out["smape_component"] = np.where(
        denom >= 1.0,
        200.0 * out["abs_error_mw"] / denom,
        np.nan,
    )
    out = out.dropna(subset=["target_timestamp", "forecast_mw", "actual_mw"])
    if out.empty:
        return out

    upsert_frame(
        "historical_replay_accuracy",
        out,
        ["region", "model_name", "model_version", "target_timestamp"],
    )
    log.info("%s/%s: stored %s historical replay intervals", region, model_name, f"{len(out):,}")
    return out


def replay_month(month: str) -> pd.DataFrame:
    init_db()
    # Validate YYYY-MM early and consistently.
    pd.Timestamp(f"{month}-01")
    frames: list[pd.DataFrame] = []
    for region in REGIONS:
        for model_name in PRODUCTION_MODELS[region]:
            try:
                out = _replay_one(region, model_name, month)
                if not out.empty:
                    frames.append(out)
            except Exception as exc:
                log.exception("%s/%s replay failed: %s", region, model_name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    month = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now().strftime("%Y-%m")
    out = replay_month(month)
    print(f"Historical next-interval replay complete for {month}; stored {len(out):,} interval rows.")
    print("Live stored forecasts remain authoritative and are never replaced by replay rows.")


if __name__ == "__main__":
    main()
