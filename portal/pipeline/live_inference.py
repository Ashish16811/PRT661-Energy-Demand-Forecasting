from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, REGIONS, STAGE2_DIR
from .model_evidence import pointer_path, production_config, resolve_pointer, validate_bundle_object
from .timezones import market_now_naive, next_market_half_hour
from .database import read_df, upsert_frame, utcnow_naive

log = logging.getLogger("live_inference")

DEMAND_FEATURES = {
    "lag_48", "lag_96", "lag_336",
    "previous_day_mean", "previous_day_peak", "previous_day_min",
    "same_half_hour_7day_mean", "same_half_hour_14day_mean",
    "rolling_mean_24h_at_t_minus_48", "rolling_std_24h_at_t_minus_48",
}


def _load_bundle(region: str, model_name: str):
    """Load a portal production bundle with portable pointer resolution and a
    structural check, so a WP4 bare estimator or a wrong-family bundle is refused."""
    pointer = pointer_path(region, model_name)
    if not pointer.exists():
        raise FileNotFoundError(
            f"No production bundle for {region}/{model_name}. "
            f"Run: python -m pipeline.monthly_retrain --region {region} --model {model_name}"
        )
    bundle_path = resolve_pointer(pointer)
    if bundle_path is None:
        raise FileNotFoundError(f"Production bundle pointer does not resolve: {pointer.read_text().strip()}")
    bundle = joblib.load(bundle_path)
    problems = validate_bundle_object(bundle, model_name)
    if problems:
        raise ValueError(f"{region}/{model_name} bundle invalid: {'; '.join(problems)}")
    return bundle


def resolve_stage2_root(value) -> Path:
    """Stage 2 roots stored inside bundles may be absolute Docker paths
    (/app/data/stage2/...). Rebase onto STAGE2_DIR when the path is absent locally."""
    p = Path(str(value))
    if p.exists():
        return p
    parts = str(value).replace("\\", "/").split("/")
    if "stage2" in parts:
        idx = len(parts) - 1 - parts[::-1].index("stage2")
        cand = Path(STAGE2_DIR).joinpath(*parts[idx + 1:])
        if cand.exists():
            return cand
    return p


def _demand_features(
    y: np.ndarray,
    positions: np.ndarray,
    wanted: set[str],
) -> dict[str, np.ndarray]:
    """Build leakage-safe lag and rolling demand features for forecast rows."""

    positions = np.asarray(positions, dtype=np.int64)

    def gather(lag: int) -> np.ndarray:
        out = np.full(len(positions), np.nan, dtype=float)
        idx = positions - int(lag)
        ok = (idx >= 0) & (idx < len(y))
        if np.any(ok):
            out[ok] = y[idx[ok]]
        return out

    out: dict[str, np.ndarray] = {}

    for lag in (48, 96, 336):
        name = f"lag_{lag}"
        if name in wanted:
            out[name] = gather(lag)

    rolling = wanted & {
        "previous_day_mean",
        "previous_day_peak",
        "previous_day_min",
        "rolling_mean_24h_at_t_minus_48",
        "rolling_std_24h_at_t_minus_48",
    }
    if rolling:
        mat = np.column_stack([gather(48 + j) for j in range(48)])
        with np.errstate(all="ignore"):
            mean = np.nanmean(mat, axis=1)
            peak = np.nanmax(mat, axis=1)
            minimum = np.nanmin(mat, axis=1)
            std = np.nanstd(mat, axis=1, ddof=1)

        if "previous_day_mean" in wanted:
            out["previous_day_mean"] = mean
        if "rolling_mean_24h_at_t_minus_48" in wanted:
            out["rolling_mean_24h_at_t_minus_48"] = mean
        if "previous_day_peak" in wanted:
            out["previous_day_peak"] = peak
        if "previous_day_min" in wanted:
            out["previous_day_min"] = minimum
        if "rolling_std_24h_at_t_minus_48" in wanted:
            out["rolling_std_24h_at_t_minus_48"] = std

    for days, name in (
        (7, "same_half_hour_7day_mean"),
        (14, "same_half_hour_14day_mean"),
    ):
        if name in wanted:
            vals = np.column_stack([gather(48 * k) for k in range(1, days + 1)])
            with np.errstate(all="ignore"):
                out[name] = np.nanmean(vals, axis=1)

    return out


def _latest_weather(region: str) -> pd.DataFrame:
    df = read_df(
        """
        SELECT * FROM weather_forecast
        WHERE region=:region
          AND forecast_origin=(
              SELECT MAX(forecast_origin)
              FROM weather_forecast
              WHERE region=:region
          )
        ORDER BY target_timestamp
        """,
        {"region": region},
    )
    if not df.empty:
        df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], errors="coerce")
        df = df.dropna(subset=["target_timestamp"])
    return df


def _overlay_live_weather(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    weather = _latest_weather(region)
    if weather.empty or "local_timestamp" not in frame.columns:
        return frame

    out = frame.copy()
    w = weather.rename(columns={"target_timestamp": "local_timestamp"}).copy()

    out["local_timestamp"] = pd.to_datetime(out["local_timestamp"], errors="coerce")
    w["local_timestamp"] = pd.to_datetime(w["local_timestamp"], errors="coerce")

    # Remove timezone information only for the merge if one side is timezone-aware
    # and the other is not. The model feature frame itself remains unchanged.
    try:
        out_tz = out["local_timestamp"].dt.tz
        w_tz = w["local_timestamp"].dt.tz
        if out_tz is None and w_tz is not None:
            w["local_timestamp"] = w["local_timestamp"].dt.tz_localize(None)
        elif out_tz is not None and w_tz is None:
            out["local_timestamp"] = out["local_timestamp"].dt.tz_localize(None)
        elif out_tz is not None and w_tz is not None and str(out_tz) != str(w_tz):
            w["local_timestamp"] = w["local_timestamp"].dt.tz_convert(out_tz)
    except (AttributeError, TypeError):
        pass

    keep = ["local_timestamp", "temp_c", "humidity_pct", "rainfall_mm"]
    keep = [c for c in keep if c in w.columns]
    out = out.merge(w[keep], on="local_timestamp", how="left", suffixes=("", "_live"))

    mapping = {
        "weather_temp_c": "temp_c",
        "weather_humidity_pct": "humidity_pct",
    }
    for dst, src in mapping.items():
        if dst in out.columns and src in out.columns:
            mask = out[src].notna()
            out.loc[mask, dst] = out.loc[mask, src]

    # Daily Model-B fields from the current weather forecast vintage.
    if "local_date" in out.columns and "temp_c" in w.columns:
        daily_source = w.copy()
        daily_source["local_date"] = pd.to_datetime(
            daily_source["local_timestamp"], errors="coerce"
        ).dt.normalize()

        agg_spec = {
            "_max": ("temp_c", "max"),
            "_min": ("temp_c", "min"),
            "_mean": ("temp_c", "mean"),
        }
        if "rainfall_mm" in daily_source.columns:
            agg_spec["_rain"] = ("rainfall_mm", "sum")

        d = daily_source.groupby("local_date").agg(**agg_spec)
        ld = pd.to_datetime(out["local_date"], errors="coerce").dt.normalize()

        field_map = [
            ("weather_daily_temp_max_c", "_max"),
            ("weather_daily_temp_min_c", "_min"),
            ("weather_daily_temp_mean_c", "_mean"),
            ("weather_daily_rainfall_mm", "_rain"),
        ]
        for dst, col in field_map:
            if dst in out.columns and col in d.columns:
                vals = ld.map(d[col])
                mask = vals.notna()
                out.loc[mask, dst] = vals.loc[mask].to_numpy()

    return out.drop(
        columns=[c for c in ("temp_c", "humidity_pct", "rainfall_mm") if c in out.columns],
        errors="ignore",
    )


def _history_for_region(bundle, region: str) -> pd.Series:
    root = resolve_stage2_root(bundle["stage2_root"])
    source = root / "data" / f"modelling_{region}_30min.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing Stage 2 modelling file: {source}")

    train = pd.read_csv(
        source,
        usecols=["SETTLEMENTDATE", "TOTALDEMAND"],
        parse_dates=["SETTLEMENTDATE"],
    )
    train["SETTLEMENTDATE"] = pd.to_datetime(train["SETTLEMENTDATE"], errors="coerce")
    train["TOTALDEMAND"] = pd.to_numeric(train["TOTALDEMAND"], errors="coerce")
    train = train.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
    train = train.drop_duplicates("SETTLEMENTDATE", keep="last")

    live = read_df(
        """
        SELECT timestamp, demand_mw
        FROM actual_demand
        WHERE region=:region
        ORDER BY timestamp
        """,
        {"region": region},
    )
    if not live.empty:
        live = live.rename(
            columns={"timestamp": "SETTLEMENTDATE", "demand_mw": "TOTALDEMAND"}
        )
        live["SETTLEMENTDATE"] = pd.to_datetime(live["SETTLEMENTDATE"], errors="coerce")
        live["TOTALDEMAND"] = pd.to_numeric(live["TOTALDEMAND"], errors="coerce")
        live = live.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
        train = pd.concat([train, live], ignore_index=True)

    train = train.drop_duplicates("SETTLEMENTDATE", keep="last")
    train = train.sort_values("SETTLEMENTDATE")

    series = train.set_index("SETTLEMENTDATE")["TOTALDEMAND"].astype(float)
    return series


def _grid_positions(index: pd.DatetimeIndex, timestamps: pd.Series) -> np.ndarray:
    """Return safe integer positions for timestamps on the 30-minute model grid.

    The previous implementation converted missing reindex results (NaN) directly to
    integer, producing values near 9.22e18. This function detects missing positions
    before any integer conversion and only accepts exact/rounded 30-minute matches.
    """

    target = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce"))
    if target.isna().any():
        raise RuntimeError("Future feature frame contains invalid SETTLEMENTDATE values")

    grid = pd.DatetimeIndex(index)

    # Align timezone representation without changing wall-clock values when only one
    # side carries tz metadata. Stage 2 SETTLEMENTDATE is normally timezone-naive.
    if grid.tz is None and target.tz is not None:
        target = target.tz_localize(None)
    elif grid.tz is not None and target.tz is None:
        target = target.tz_localize(grid.tz)
    elif grid.tz is not None and target.tz is not None and str(grid.tz) != str(target.tz):
        target = target.tz_convert(grid.tz)

    positions = grid.get_indexer(target)

    # Some source timestamps can differ by seconds/microseconds. Only snap to the
    # intended half-hour grid when that is enough to resolve the mismatch.
    missing = positions < 0
    if np.any(missing):
        rounded = target.round("30min")
        retry = grid.get_indexer(rounded)
        acceptable = np.abs((target - rounded).total_seconds()) <= 60
        use_retry = missing & (retry >= 0) & acceptable
        positions[use_retry] = retry[use_retry]

    if np.any(positions < 0):
        bad = target[positions < 0]
        preview = ", ".join(str(x) for x in bad[:5])
        raise RuntimeError(
            f"{len(bad)} forecast timestamp(s) are outside the continuous 30-minute grid. "
            f"First unmatched: {preview}"
        )

    return positions.astype(np.int64, copy=False)


def forecast_one(region: str, model_name: str) -> pd.DataFrame:
    bundle = _load_bundle(region, model_name)
    root = resolve_stage2_root(bundle["stage2_root"])
    future_path = root / "data" / "future_feature_frame_all_regions.csv"
    if not future_path.exists():
        raise FileNotFoundError(f"Missing future feature frame: {future_path}")

    future = pd.read_csv(
        future_path,
        parse_dates=["SETTLEMENTDATE", "local_timestamp", "local_date"],
        low_memory=False,
    )
    key = "region" if "region" in future.columns else "REGION"
    future = future[future[key].astype(str) == region].copy()
    future["SETTLEMENTDATE"] = pd.to_datetime(future["SETTLEMENTDATE"], errors="coerce")
    future = future.dropna(subset=["SETTLEMENTDATE"])
    future = future.drop_duplicates("SETTLEMENTDATE", keep="last")
    future = future.sort_values("SETTLEMENTDATE")

    history = _history_for_region(bundle, region)
    if history.empty:
        raise RuntimeError(f"{region}: no historical actual demand available")

    latest_actual = history.dropna().index.max()
    origin = pd.Timestamp(latest_actual).ceil("30min")
    if origin <= latest_actual:
        origin += pd.Timedelta(minutes=30)

    future = future[future["SETTLEMENTDATE"] >= origin].copy()
    if future.empty:
        raise RuntimeError(
            f"{region}: no future feature rows at/after {origin}; run monthly retraining"
        )

    if bundle["feature_set"] == "B":
        future = _overlay_live_weather(future, region)

    # Build one continuous half-hour grid from the oldest history point through the
    # final forecast timestamp. Reindex historical demand onto the same grid.
    grid_start = pd.Timestamp(history.index.min()).floor("30min")
    grid_end = pd.Timestamp(future["SETTLEMENTDATE"].max()).ceil("30min")
    idx = pd.date_range(grid_start, grid_end, freq="30min")

    y = pd.Series(np.nan, index=idx, dtype=float)
    aligned_history = history.copy()
    if aligned_history.index.tz is not None and idx.tz is None:
        aligned_history.index = aligned_history.index.tz_localize(None)
    elif aligned_history.index.tz is None and idx.tz is not None:
        aligned_history.index = aligned_history.index.tz_localize(idx.tz)

    common = aligned_history.index.intersection(idx)
    if len(common):
        y.loc[common] = aligned_history.loc[common].to_numpy(dtype=float)

    # Explicit copy is required because pandas may expose a read-only NumPy view.
    yv = np.array(y.to_numpy(dtype=float), dtype=float, copy=True)

    features = list(bundle["feature_names"])
    medians = pd.Series(bundle["feature_medians"], dtype=float)
    model = bundle["estimator"]

    forecast_values: list[float] = []
    stamps = pd.to_datetime(future["SETTLEMENTDATE"]).tolist()

    for start in range(0, len(future), 48):
        block = future.iloc[start:start + 48].copy()
        positions = _grid_positions(idx, block["SETTLEMENTDATE"])

        X = pd.DataFrame(index=block.index)
        dyn = _demand_features(yv, positions, set(features) & DEMAND_FEATURES)

        for feature in features:
            if feature in dyn:
                X[feature] = dyn[feature]
            elif feature in block.columns:
                X[feature] = pd.to_numeric(block[feature], errors="coerce").to_numpy()
            else:
                X[feature] = np.nan

        X = X.reindex(columns=features)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)

        pred = np.asarray(
            model.predict(X.to_numpy(dtype=np.float32)),
            dtype=float,
        )

        if len(pred) != len(positions):
            raise RuntimeError(
                f"{region}/{model_name}: model returned {len(pred)} predictions "
                f"for {len(positions)} rows"
            )

        # Recursive forecasting: predictions become history for later blocks so lag
        # features can be built safely across the full multi-month horizon.
        yv[positions] = pred
        forecast_values.extend(pred.tolist())

    if len(forecast_values) != len(stamps):
        raise RuntimeError(
            f"{region}/{model_name}: forecast length mismatch "
            f"({len(forecast_values)} values vs {len(stamps)} timestamps)"
        )

    return pd.DataFrame(
        {
            "region": region,
            "model_name": model_name,
            "model_version": bundle["model_version"],
            "forecast_origin": origin,
            "target_timestamp": pd.to_datetime(stamps),
            "forecast_mw": forecast_values,
            "created_at": utcnow_naive(),
        }
    )


def run_live_forecasts() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for region in REGIONS:
        for model_name in production_config(max_age_seconds=0).production_models[region]:
            try:
                fc = forecast_one(region, model_name)
                upsert_frame(
                    "project_forecast",
                    fc,
                    [
                        "region",
                        "model_name",
                        "model_version",
                        "forecast_origin",
                        "target_timestamp",
                    ],
                )
                frames.append(fc)
                log.info(
                    "%s/%s: generated %s forecast rows",
                    region,
                    model_name,
                    f"{len(fc):,}",
                )
            except Exception as exc:
                log.exception("%s/%s live forecast failed: %s", region, model_name, exc)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def forecast_next_interval(region: str, model_name: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Generate one immutable forecast for the next public 30-minute interval.

    This is intentionally separate from the multi-month rolling forecast. It uses
    the same promoted production model and latest actual/weather inputs, but only
    predicts the next interval users are about to see. That point forecast is
    persisted and later compared with the actual demand for the exact timestamp.
    """
    generated_at = market_now_naive(region, now)
    requested_target = next_market_half_hour(region, now)

    bundle = _load_bundle(region, model_name)
    root = resolve_stage2_root(bundle["stage2_root"])
    future_path = root / "data" / "future_feature_frame_all_regions.csv"
    if not future_path.exists():
        raise FileNotFoundError(f"Missing future feature frame: {future_path}")

    future = pd.read_csv(
        future_path,
        parse_dates=["SETTLEMENTDATE", "local_timestamp", "local_date"],
        low_memory=False,
    )
    key = "region" if "region" in future.columns else "REGION"
    future = future[future[key].astype(str) == region].copy()
    future["SETTLEMENTDATE"] = pd.to_datetime(future["SETTLEMENTDATE"], errors="coerce")
    future = future.dropna(subset=["SETTLEMENTDATE"])
    future = future.drop_duplicates("SETTLEMENTDATE", keep="last").sort_values("SETTLEMENTDATE")

    # The operational product must represent the exact next public half-hour.
    # Do not silently substitute a later target because that would make the card
    # and the later accuracy comparison misleading.
    block = future[future["SETTLEMENTDATE"] == requested_target].copy()
    if block.empty:
        raise RuntimeError(
            f"{region}: no exact future feature row for next interval {requested_target}; "
            "run monthly retraining or inspect the forecast-frame clock"
        )
    block = block.iloc[[0]].copy()
    target = requested_target

    if bundle["feature_set"] == "B":
        block = _overlay_live_weather(block, region)

    history = _history_for_region(bundle, region)
    if history.empty:
        raise RuntimeError(f"{region}: no historical actual demand available")

    grid_start = pd.Timestamp(history.index.min()).floor("30min")
    grid_end = target.ceil("30min")
    idx = pd.date_range(grid_start, grid_end, freq="30min")

    y = pd.Series(np.nan, index=idx, dtype=float)
    aligned_history = history.copy()
    if aligned_history.index.tz is not None and idx.tz is None:
        aligned_history.index = aligned_history.index.tz_localize(None)
    elif aligned_history.index.tz is None and idx.tz is not None:
        aligned_history.index = aligned_history.index.tz_localize(idx.tz)
    common = aligned_history.index.intersection(idx)
    if len(common):
        y.loc[common] = aligned_history.loc[common].to_numpy(dtype=float)
    yv = np.array(y.to_numpy(dtype=float), dtype=float, copy=True)

    positions = _grid_positions(idx, block["SETTLEMENTDATE"])
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

    pred = float(np.asarray(bundle["estimator"].predict(X.to_numpy(dtype=np.float32)), dtype=float)[0])
    lead_minutes = max(0.0, (target - generated_at).total_seconds() / 60.0)

    return pd.DataFrame([{
        "region": region,
        "model_name": model_name,
        "model_version": bundle["model_version"],
        "forecast_origin": generated_at.floor("min"),
        "target_timestamp": target,
        "forecast_mw": pred,
        "lead_minutes": lead_minutes,
        "created_at": utcnow_naive(),
    }])


def run_next_interval_forecasts(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Forecast and store exactly one next-interval point for every production model."""
    frames: list[pd.DataFrame] = []
    for region in REGIONS:
        for model_name in production_config(max_age_seconds=0).production_models[region]:
            try:
                fc = forecast_next_interval(region, model_name, now=now)
                upsert_frame(
                    "next_interval_forecast",
                    fc,
                    ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
                )
                frames.append(fc)
                r = fc.iloc[0]
                log.info(
                    "%s/%s next interval: %s -> %.1f MW (lead %.0f min)",
                    region,
                    model_name,
                    pd.Timestamp(r["target_timestamp"]).strftime("%d %b %H:%M"),
                    float(r["forecast_mw"]),
                    float(r["lead_minutes"]),
                )
            except Exception as exc:
                log.exception("%s/%s next-interval forecast failed: %s", region, model_name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
