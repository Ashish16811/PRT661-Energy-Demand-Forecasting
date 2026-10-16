from __future__ import annotations

import numpy as np
import pandas as pd

from .database import read_df, upsert_frame, utcnow_naive


def verify_available_forecasts() -> pd.DataFrame:
    joined = read_df("""
        SELECT f.region, f.model_name, f.model_version, f.forecast_origin,
               f.target_timestamp, f.forecast_mw, a.demand_mw AS actual_mw
        FROM project_forecast f
        JOIN actual_demand a
          ON a.region=f.region AND a.timestamp=f.target_timestamp
    """)
    if joined.empty:
        return joined
    for c in ("forecast_origin", "target_timestamp"):
        joined[c] = pd.to_datetime(joined[c])
    joined["error_mw"] = joined["forecast_mw"] - joined["actual_mw"]
    joined["abs_error_mw"] = joined["error_mw"].abs()
    joined["squared_error"] = joined["error_mw"] ** 2
    denom = joined["forecast_mw"].abs() + joined["actual_mw"].abs()
    joined["smape_component"] = np.where(
        denom >= 1.0, 200.0 * joined["abs_error_mw"] / denom, np.nan)
    joined["evaluated_at"] = utcnow_naive()
    upsert_frame("forecast_accuracy", joined,
                 ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"])
    return joined


def verify_next_interval_forecasts() -> pd.DataFrame:
    """Score stored next-interval forecasts once matching actual demand arrives.

    Every forecast remains preserved. Dashboard month-to-date metrics later select
    the latest forecast issued for each target interval, avoiding duplicate scoring
    when more than one live cycle refreshed the same upcoming interval.
    """
    joined = read_df("""
        SELECT f.region, f.model_name, f.model_version, f.forecast_origin,
               f.target_timestamp, f.forecast_mw, a.demand_mw AS actual_mw
        FROM next_interval_forecast f
        JOIN actual_demand a
          ON a.region=f.region
         AND a.timestamp=f.target_timestamp
    """)
    if joined.empty:
        return joined

    for c in ("forecast_origin", "target_timestamp"):
        joined[c] = pd.to_datetime(joined[c], errors="coerce")
    for c in ("forecast_mw", "actual_mw"):
        joined[c] = pd.to_numeric(joined[c], errors="coerce")
    joined = joined.dropna(subset=["forecast_origin", "target_timestamp", "forecast_mw", "actual_mw"])
    if joined.empty:
        return joined

    joined["error_mw"] = joined["forecast_mw"] - joined["actual_mw"]
    joined["abs_error_mw"] = joined["error_mw"].abs()
    joined["squared_error"] = joined["error_mw"] ** 2
    denom = joined["forecast_mw"].abs() + joined["actual_mw"].abs()
    joined["smape_component"] = np.where(
        denom >= 1.0,
        200.0 * joined["abs_error_mw"] / denom,
        np.nan,
    )
    joined["evaluated_at"] = utcnow_naive()
    upsert_frame(
        "next_interval_accuracy",
        joined,
        ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
    )
    return joined
