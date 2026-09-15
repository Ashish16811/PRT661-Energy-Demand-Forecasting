"""
PRT661 – Data Science Practice
WP2 Individual Contribution Module – Ashish Shrestha (S388084)
Role: Forecasting Lead
Project: Australian Electricity Demand Forecasting | Dan6

This module represents the forecasting-oriented feature-engineering workstream:
horizon-safe demand history, historical profiles, weather learning features,
forecast-weather integration, climatology fallback and the Model A / Model B
feature contract.

Feature engineering is collaborative by design:
- Suraj supplies the trustworthy 30-minute time grid and calendar/cyclical fields.
- Sudip supplies source-quality decisions and holiday features.
- Ashish leads demand/weather feature design and future feature-frame logic.
- Bishal independently validates leakage, weather provenance and readiness.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd

try:
    from WP2_Suraj_Raut_Preprocessing import add_calendar_features
    from WP2_Sudip_Lamichhane_Preprocessing import add_holiday_features
except ImportError:  # allows documentation/import inspection outside the shared folder
    add_calendar_features = None
    add_holiday_features = None

log = logging.getLogger("wp2_ashish")
TIME = "SETTLEMENTDATE"; TARGET = "TOTALDEMAND"; TARGET_FREQ_MIN = 30
NEM_REGIONS = ["NSW1","QLD1","VIC1","SA1","TAS1"]
REGION_TZ = {"NSW1":"Australia/Sydney","QLD1":"Australia/Brisbane","VIC1":"Australia/Melbourne",
             "SA1":"Australia/Adelaide","TAS1":"Australia/Hobart","WA":"Australia/Perth"}
REGION_SOURCE_TZ = {r:"Etc/GMT-10" for r in NEM_REGIONS} | {"WA":"Etc/GMT-8"}
HORIZON_SAFE_LAGS = {"lag_48":48,"lag_96":96,"lag_336":336}
CALENDAR_FEATURES = ["half_hour_index","hour","day_of_week","day_of_month","week_of_year","month","quarter",
                     "day_of_year","is_weekend","sin_half_hour","cos_half_hour","sin_day_of_week","cos_day_of_week",
                     "sin_month","cos_month","sin_day_of_year","cos_day_of_year"]
HOLIDAY_FEATURES = ["is_public_holiday","is_day_before_holiday","is_day_after_holiday"]
HISTORICAL_FEATURES = ["historical_month_halfhour_mean","historical_month_halfhour_median",
                       "historical_month_dow_halfhour_mean","historical_month_daily_mean",
                       "historical_month_peak_mean","historical_month_variability"]
SAFE_DEMAND_FEATURES = ["previous_day_mean","previous_day_peak","previous_day_min","same_half_hour_7day_mean",
                        "same_half_hour_14day_mean","rolling_mean_24h_at_t_minus_48","rolling_std_24h_at_t_minus_48"]
TREND_FEATURES = ["time_index","days_since_start","year"]
WEATHER_INPUT_FEATURES = ["weather_temp_c","weather_humidity_pct","weather_daily_temp_max_c",
                          "weather_daily_temp_min_c","weather_daily_temp_mean_c","weather_daily_rainfall_mm"]
WEATHER_CONTEXT_FEATURES = ["temp_mean_lag_1d","temp_mean_lag_7d","rolling_temp_mean_7d",
                            "expected_temp_climatology","historical_rain_probability","hdd_lag_1d","cdd_lag_1d",
                            "expected_hdd_climatology","expected_cdd_climatology"]
WEATHER_SAFE_FEATURES = WEATHER_INPUT_FEATURES + WEATHER_CONTEXT_FEATURES
FEATURE_SET_A = list(HORIZON_SAFE_LAGS) + CALENDAR_FEATURES + HOLIDAY_FEATURES + HISTORICAL_FEATURES + SAFE_DEMAND_FEATURES + TREND_FEATURES
FEATURE_SET_B = FEATURE_SET_A + WEATHER_SAFE_FEATURES
BALANCE_POINT_SEARCH = np.arange(10.0, 28.0, 0.5)


def derive_model_cutoff(analysis_origin):
    """Joint Stage-2 rule: training stops at the previous complete month end."""
    origin = pd.Timestamp(analysis_origin)
    forecast_start = origin.to_period("M").to_timestamp()
    return forecast_start - pd.Timedelta(minutes=TARGET_FREQ_MIN), forecast_start


def determine_rolling_window(analysis_origin):
    origin = pd.Timestamp(analysis_origin).to_period("M")
    return [(p.year, p.month) for p in (origin+i for i in range(4))]


def load_weather_daily(raw_dir: Path, region: str):
    p = raw_dir / "weather" / f"weather_daily_{region}.csv"
    if not p.exists() or not p.stat().st_size:
        return None
    df = pd.read_csv(p)
    cols = {c.strip().lower():c for c in df.columns}
    keys = {"date":("date",),"temp_max_c":("temp_max",),"temp_min_c":("temp_min",),
            "temp_mean_c":("temp_mean",),"rainfall_mm":("rainfall",)}
    resolved = {k:next((cols[c] for c in cols if any(x in c for x in v)),None) for k,v in keys.items()}
    if resolved["date"] is None or resolved["temp_max_c"] is None:
        return None
    out = pd.DataFrame({k:df[v] for k,v in resolved.items() if v})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for c in ("temp_max_c","temp_min_c","temp_mean_c","rainfall_mm"):
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "temp_mean_c" not in out or out["temp_mean_c"].isna().all():
        out["temp_mean_c"] = (out["temp_max_c"] + out["temp_min_c"])/2
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def load_weather_hourly(raw_dir: Path, region: str):
    p = raw_dir / "weather" / f"weather_hourly_{region}.csv"
    if not p.exists() or not p.stat().st_size: return None
    h = pd.read_csv(p)
    if "TIMESTAMP" not in h: return None
    h["TIMESTAMP"] = pd.to_datetime(h["TIMESTAMP"], errors="coerce")
    h = h.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
    cols = [c for c in ("temp_c","humidity_pct") if c in h]
    if not cols: return None
    for c in cols: h[c] = pd.to_numeric(h[c], errors="coerce")
    hh = h.set_index("TIMESTAMP")[cols].resample(f"{TARGET_FREQ_MIN}min").interpolate(method="time", limit=2)
    return hh.rename(columns={"temp_c":"temp_halfhourly_c"})


def load_weather_forecast(raw_dir: Path, region: str):
    """Load the latest seven-day forecast and map continuous variables to 30 minutes."""
    latest = raw_dir / "weather" / "forecast" / "latest"
    hp = latest / f"weather_forecast_hourly_{region}.csv"
    dp = latest / f"weather_forecast_daily_{region}.csv"
    meta = {"region":region,"forecast_available":False,"forecast_origin":pd.NaT,
            "forecast_start":pd.NaT,"forecast_end":pd.NaT,"forecast_days":0,
            "source":"","model":"","retrieved_at_utc":"","raw_hourly_rows":0}
    if not hp.exists() or not hp.stat().st_size: return None, None, meta
    h = pd.read_csv(hp)
    required = {"TIMESTAMP","temp_c","humidity_pct","SOURCE_TYPE","FORECAST_ORIGIN_DATE","LEAD_DAY"}
    if required-set(h.columns): return None, None, meta
    h["TIMESTAMP"] = pd.to_datetime(h["TIMESTAMP"], errors="coerce")
    for c in ("temp_c","humidity_pct","rainfall_mm","LEAD_DAY"):
        if c in h: h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").drop_duplicates("TIMESTAMP", keep="last")
    origin_vals = pd.to_datetime(h["FORECAST_ORIGIN_DATE"], errors="coerce").dropna()
    origin = origin_vals.iloc[0].normalize() if len(origin_vals) else pd.NaT
    meta.update({"forecast_available":len(h)>0,"forecast_origin":origin,"forecast_start":h["TIMESTAMP"].min(),
                 "forecast_end":h["TIMESTAMP"].max(),"forecast_days":int(h["TIMESTAMP"].dt.normalize().nunique()),
                 "source":str(h["SOURCE"].dropna().iloc[0]) if "SOURCE" in h and h["SOURCE"].notna().any() else "",
                 "model":str(h["MODEL"].dropna().iloc[0]) if "MODEL" in h and h["MODEL"].notna().any() else "",
                 "retrieved_at_utc":str(h["RETRIEVED_AT_UTC"].dropna().iloc[0]) if "RETRIEVED_AT_UTC" in h and h["RETRIEVED_AT_UTC"].notna().any() else "",
                 "raw_hourly_rows":len(h)})
    grid = pd.date_range(h["TIMESTAMP"].min(), h["TIMESTAMP"].max()+pd.Timedelta(minutes=30), freq="30min")
    state = h.set_index("TIMESTAMP")[["temp_c","humidity_pct"]].reindex(grid).interpolate(method="time", limit=2).ffill(limit=1)
    state.index.name = "local_timestamp"
    state = state.rename(columns={"temp_c":"forecast_temp_c","humidity_pct":"forecast_humidity_pct"})
    state["weather_forecast_origin"] = origin; state["weather_source_type"] = "FORECAST_7DAY"
    state["weather_forecast_source"] = meta["source"]; state["weather_forecast_model"] = meta["model"]
    state["weather_retrieved_at_utc"] = meta["retrieved_at_utc"]
    state["weather_lead_day"] = (state.index.normalize()-origin).days if pd.notna(origin) else np.nan
    daily = None
    if dp.exists() and dp.stat().st_size:
        d = pd.read_csv(dp); d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce").dt.normalize()
        for c in ("temp_max_c","temp_min_c","temp_mean_c","rainfall_mm","humidity_mean_pct"):
            if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
        daily = d.dropna(subset=["DATE"]).sort_values("DATE").drop_duplicates("DATE", keep="last")
    return state, daily, meta


def add_horizon_safe_demand_features(g):
    g = g.sort_values(TIME).copy(); demand = g[TARGET]
    for name, steps in HORIZON_SAFE_LAGS.items(): g[name] = demand.shift(steps)
    prior_day = demand.shift(48)
    g["previous_day_mean"] = prior_day.rolling(48).mean(); g["previous_day_peak"] = prior_day.rolling(48).max(); g["previous_day_min"] = prior_day.rolling(48).min()
    g["same_half_hour_7day_mean"] = pd.concat([demand.shift(48*k) for k in range(1,8)], axis=1).mean(axis=1)
    g["same_half_hour_14day_mean"] = pd.concat([demand.shift(48*k) for k in range(1,15)], axis=1).mean(axis=1)
    g["rolling_mean_24h_at_t_minus_48"] = prior_day.rolling(48).mean(); g["rolling_std_24h_at_t_minus_48"] = prior_day.rolling(48).std()
    return g


def add_trend_features(g):
    g = g.copy(); start = g[TIME].min(); g["time_index"] = np.arange(len(g)); g["days_since_start"] = (g[TIME]-start).dt.total_seconds()/86400
    return g


def build_historical_month_profiles(g, origin_year):
    hist = g[g["year"] < origin_year].dropna(subset=[TARGET])
    if hist.empty: return {}
    profiles = {"halfhour":hist.groupby(["month","half_hour_index"])[TARGET].agg(["mean","median"]),
                "dow_halfhour":hist.groupby(["month","day_of_week","half_hour_index"])[TARGET].mean()}
    daily = hist.groupby(["month","local_date"])[TARGET].agg(["mean","max"])
    profiles["daily"] = daily.groupby("month").agg(daily_mean=("mean","mean"), peak_mean=("max","mean"), variability=("mean","std"))
    return profiles


def apply_historical_month_profiles(g, profiles):
    g = g.copy()
    if not profiles:
        for c in HISTORICAL_FEATURES: g[c] = np.nan
        return g
    hh = profiles["halfhour"]; idx = pd.MultiIndex.from_arrays([g["month"],g["half_hour_index"]])
    g["historical_month_halfhour_mean"] = hh["mean"].reindex(idx).to_numpy(); g["historical_month_halfhour_median"] = hh["median"].reindex(idx).to_numpy()
    dow = profiles["dow_halfhour"]; idx3 = pd.MultiIndex.from_arrays([g["month"],g["day_of_week"],g["half_hour_index"]])
    g["historical_month_dow_halfhour_mean"] = dow.reindex(idx3).to_numpy()
    daily = profiles["daily"]
    g["historical_month_daily_mean"] = g["month"].map(daily["daily_mean"]); g["historical_month_peak_mean"] = g["month"].map(daily["peak_mean"]); g["historical_month_variability"] = g["month"].map(daily["variability"])
    return g


def derive_balance_point(daily_temp, demand_by_date, origin_year):
    hist = demand_by_date[demand_by_date.index.year < origin_year]
    joined = hist.to_frame("demand").join(daily_temp.rename("temp"), how="inner").dropna()
    if len(joined) < 365: return None, np.nan
    best = None; best_corr = -np.inf
    for base in BALANCE_POINT_SEARCH:
        corr = abs(joined["demand"].corr((joined["temp"]-base).abs()))
        if np.isfinite(corr) and corr > best_corr: best, best_corr = float(base), corr
    return best, round(best_corr,4)


def add_weather_features(g, raw_dir, region, origin_year):
    daily = load_weather_daily(raw_dir, region)
    if daily is None:
        for c in ["temp_max_c","temp_min_c","temp_mean_c","rainfall_mm","temp_halfhourly_c","humidity_pct","hdd","cdd"] + WEATHER_SAFE_FEATURES:
            g[c] = np.nan
        g["weather_available"] = 0; g["weather_feature_source"] = "MISSING"; g["balance_point_c"] = np.nan
        return g
    daily = daily.sort_values("date").copy(); daily["temp_mean_lag_1d"] = daily["temp_mean_c"].shift(1); daily["temp_mean_lag_7d"] = daily["temp_mean_c"].shift(7); daily["rolling_temp_mean_7d"] = daily["temp_mean_c"].shift(1).rolling(7).mean()
    demand_by_date = g.set_index("local_date")[TARGET].groupby(level=0).mean(); base, _ = derive_balance_point(daily.set_index("date")["temp_mean_c"], demand_by_date, origin_year)
    daily["hdd"] = (base-daily["temp_mean_c"]).clip(lower=0) if base is not None else np.nan; daily["cdd"] = (daily["temp_mean_c"]-base).clip(lower=0) if base is not None else np.nan
    daily["hdd_lag_1d"] = daily["hdd"].shift(1); daily["cdd_lag_1d"] = daily["cdd"].shift(1)
    keep = [c for c in ("date","temp_max_c","temp_min_c","temp_mean_c","rainfall_mm","temp_mean_lag_1d","temp_mean_lag_7d","rolling_temp_mean_7d","hdd","cdd","hdd_lag_1d","cdd_lag_1d") if c in daily]
    g = g.merge(daily[keep].rename(columns={"date":"local_date"}), on="local_date", how="left"); g["balance_point_c"] = base
    hourly = load_weather_hourly(raw_dir, region)
    if hourly is not None: g = g.merge(hourly, left_on="local_timestamp", right_index=True, how="left")
    else: g["temp_halfhourly_c"] = np.nan; g["humidity_pct"] = np.nan
    hist = g[g["year"] < origin_year]
    if len(hist) and hist["temp_mean_c"].notna().any():
        clim = hist.groupby("day_of_year")["temp_mean_c"].mean(); g["expected_temp_climatology"] = g["day_of_year"].map(clim)
        if base is not None: g["expected_hdd_climatology"] = (base-g["expected_temp_climatology"]).clip(lower=0); g["expected_cdd_climatology"] = (g["expected_temp_climatology"]-base).clip(lower=0)
        rain = hist.assign(wet=(hist["rainfall_mm"]>0.2)).groupby("month")["wet"].mean(); g["historical_rain_probability"] = g["month"].map(rain)
    g["weather_temp_c"] = g["temp_halfhourly_c"].combine_first(g["temp_mean_c"]); g["weather_humidity_pct"] = g["humidity_pct"]
    g["weather_daily_temp_max_c"] = g["temp_max_c"]; g["weather_daily_temp_min_c"] = g["temp_min_c"]; g["weather_daily_temp_mean_c"] = g["temp_mean_c"]; g["weather_daily_rainfall_mm"] = g["rainfall_mm"]
    g["weather_available"] = g["weather_temp_c"].notna().astype(int); g["weather_feature_source"] = np.where(g["weather_available"].eq(1),"OBSERVED_REANALYSIS","MISSING")
    return g


def _build_weather_climatology(g_hist, origin_year):
    """Climatology from rows available before the target year.

    It is a fallback only. It never masquerades as a numerical weather forecast.
    """
    hist = g_hist[(g_hist["year"] < origin_year) & g_hist[TARGET].notna()].copy()
    out = {}
    if hist.empty:
        return out

    if "temp_halfhourly_c" in hist and hist["temp_halfhourly_c"].notna().any():
        out["temp_hh"] = hist.groupby(["day_of_year", "half_hour_index"])["temp_halfhourly_c"].mean()
    if "humidity_pct" in hist and hist["humidity_pct"].notna().any():
        out["humidity_hh"] = hist.groupby(["month", "half_hour_index"])["humidity_pct"].mean()
    daily_cols = [c for c in ("temp_max_c", "temp_min_c", "temp_mean_c", "rainfall_mm")
                  if c in hist and hist[c].notna().any()]
    if daily_cols:
        by_date = hist.groupby(["local_date", "day_of_year", "month"])[daily_cols].first().reset_index()
        out["daily_doy"] = by_date.groupby("day_of_year")[[c for c in daily_cols if c != "rainfall_mm"]].mean()
        if "rainfall_mm" in by_date:
            out["rain_month"] = by_date.groupby("month")["rainfall_mm"].mean()
    return out


def _map_multiindex(series, *arrays):
    if series is None:
        return np.full(len(arrays[0]), np.nan)
    idx = pd.MultiIndex.from_arrays(arrays)
    return series.reindex(idx).to_numpy()


def build_future_feature_frame(region, raw_dir, window, profiles, g_hist,
                               holidays_table=None, analysis_origin=None):
    """Build the current-month + next-three-month modelling frame.

    Weather priority is explicit:
      1. past rows in the current month -> observed/reanalysis weather if present;
      2. future rows covered by the current Stage-1 forecast -> FORECAST_7DAY;
      3. remaining future rows -> prior-year climatology fallback.

    The provenance columns are never predictors. They exist so Stage 3 can prove
    what information was available for every timestamp.
    """
    first = pd.Timestamp(year=window[0][0], month=window[0][1], day=1)
    last_year, last_month = window[-1]
    last = (pd.Timestamp(year=last_year, month=last_month, day=1)
            + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=30))
    grid = pd.date_range(first, last, freq=f"{TARGET_FREQ_MIN}min")

    market = grid.tz_localize(REGION_SOURCE_TZ[region])
    local = market.tz_convert(REGION_TZ[region]).tz_localize(None)
    f = pd.DataFrame({TIME: grid, "REGION": region, "local_timestamp": local})
    f["local_date"] = f["local_timestamp"].dt.normalize()
    f = add_calendar_features(f)
    f = add_holiday_features(f, raw_dir, region, holidays_table)
    f = apply_historical_month_profiles(f, profiles)

    start = g_hist[TIME].min()
    # Continue trend by elapsed 30-minute intervals rather than by row count;
    # future frame begins at month start and can overlap historical rows.
    f["time_index"] = ((f[TIME] - start).dt.total_seconds() /
                       (TARGET_FREQ_MIN * 60)).round().astype(int)
    f["days_since_start"] = (f[TIME] - start).dt.total_seconds() / 86400

    # Carry climatology features into the whole frame.
    hist_for_clim = g_hist[g_hist["year"] < window[0][0]]
    if len(hist_for_clim) and hist_for_clim["temp_mean_c"].notna().any():
        temp_clim = hist_for_clim.groupby("day_of_year")["temp_mean_c"].mean()
        f["expected_temp_climatology"] = f["day_of_year"].map(temp_clim)
        rain_prob = (hist_for_clim.assign(wet=(hist_for_clim["rainfall_mm"] > 0.2))
                     .groupby("month")["wet"].mean())
        f["historical_rain_probability"] = f["month"].map(rain_prob)
    else:
        f["expected_temp_climatology"] = np.nan
        f["historical_rain_probability"] = np.nan

    base = g_hist["balance_point_c"].dropna().iloc[-1] \
        if "balance_point_c" in g_hist and g_hist["balance_point_c"].notna().any() else np.nan
    if base == base:
        f["expected_hdd_climatology"] = (base - f["expected_temp_climatology"]).clip(lower=0)
        f["expected_cdd_climatology"] = (f["expected_temp_climatology"] - base).clip(lower=0)
    else:
        f["expected_hdd_climatology"] = np.nan
        f["expected_cdd_climatology"] = np.nan

    # Dynamic demand features are rebuilt recursively in Stage 3. Keep the
    # columns present but empty so the schema is explicit and uniform.
    for c in list(HORIZON_SAFE_LAGS) + SAFE_DEMAND_FEATURES:
        f[c] = np.nan

    # Start with climatology as the long-horizon weather fallback.
    clim = _build_weather_climatology(g_hist, window[0][0])
    temp_hh = clim.get("temp_hh")
    humidity_hh = clim.get("humidity_hh")
    daily_doy = clim.get("daily_doy")
    rain_month = clim.get("rain_month")

    f["weather_temp_c"] = _map_multiindex(
        temp_hh, f["day_of_year"], f["half_hour_index"]) if temp_hh is not None \
        else f["expected_temp_climatology"].to_numpy()
    f["weather_humidity_pct"] = _map_multiindex(
        humidity_hh, f["month"], f["half_hour_index"]) if humidity_hh is not None \
        else np.nan
    for src, dst in (("temp_max_c", "weather_daily_temp_max_c"),
                     ("temp_min_c", "weather_daily_temp_min_c"),
                     ("temp_mean_c", "weather_daily_temp_mean_c")):
        f[dst] = f["day_of_year"].map(daily_doy[src]) \
            if daily_doy is not None and src in daily_doy else np.nan
    f["weather_daily_rainfall_mm"] = f["month"].map(rain_month) \
        if rain_month is not None else np.nan
    f["weather_feature_source"] = "CLIMATOLOGY_FALLBACK"
    f["weather_forecast_origin"] = pd.NaT
    f["weather_lead_day"] = np.nan
    f["weather_forecast_source"] = ""
    f["weather_forecast_model"] = ""
    f["weather_retrieved_at_utc"] = ""

    # Past/current-month rows use historical observed weather where Stage 1 has it.
    observed_cols = [TIME] + [c for c in WEATHER_INPUT_FEATURES if c in g_hist]
    observed = g_hist[observed_cols].drop_duplicates(TIME, keep="last")
    obs_names = {c: f"_obs_{c}" for c in observed_cols if c != TIME}
    observed = observed.rename(columns=obs_names)
    f = f.merge(observed, on=TIME, how="left")
    origin = pd.Timestamp(analysis_origin if analysis_origin is not None else g_hist[TIME].max())
    past_mask = f[TIME] <= origin
    obs_temp = f.get("_obs_weather_temp_c", pd.Series(np.nan, index=f.index))
    use_obs = past_mask & obs_temp.notna()
    for c in WEATHER_INPUT_FEATURES:
        oc = f"_obs_{c}"
        if oc in f:
            f.loc[use_obs, c] = f.loc[use_obs, oc]
    f.loc[use_obs, "weather_feature_source"] = "OBSERVED_REANALYSIS"
    f = f.drop(columns=[c for c in f.columns if c.startswith("_obs_")])

    # Genuine latest 7-day forecast overrides climatology only for future rows.
    forecast_hh, forecast_daily, forecast_meta = load_weather_forecast(raw_dir, region)
    if forecast_hh is not None and len(forecast_hh):
        fh = forecast_hh.reset_index().rename(columns={
            "weather_forecast_origin": "_fc_forecast_origin",
            "weather_lead_day": "_fc_lead_day",
            "weather_forecast_source": "_fc_forecast_source",
            "weather_forecast_model": "_fc_forecast_model",
            "weather_retrieved_at_utc": "_fc_retrieved_at_utc",
            "weather_source_type": "_fc_source_type",
        })
        f = f.merge(fh, on="local_timestamp", how="left")
        future_forecast = ((f[TIME] > origin) & f["forecast_temp_c"].notna())
        f.loc[future_forecast, "weather_temp_c"] = f.loc[future_forecast, "forecast_temp_c"]
        f.loc[future_forecast, "weather_humidity_pct"] = f.loc[future_forecast, "forecast_humidity_pct"]
        mapping = {
            "_fc_forecast_origin": "weather_forecast_origin",
            "_fc_lead_day": "weather_lead_day",
            "_fc_forecast_source": "weather_forecast_source",
            "_fc_forecast_model": "weather_forecast_model",
            "_fc_retrieved_at_utc": "weather_retrieved_at_utc",
        }
        for src, dst in mapping.items():
            if src in f:
                f.loc[future_forecast, dst] = f.loc[future_forecast, src]
        f.loc[future_forecast, "weather_feature_source"] = "FORECAST_7DAY"
        drop_cols = [c for c in ("forecast_temp_c", "forecast_humidity_pct",
                                 "_fc_source_type", *mapping.keys()) if c in f]
        f = f.drop(columns=drop_cols)

    if forecast_daily is not None and len(forecast_daily):
        d = forecast_daily.rename(columns={
            "DATE": "local_date",
            "temp_max_c": "_fc_temp_max_c",
            "temp_min_c": "_fc_temp_min_c",
            "temp_mean_c": "_fc_temp_mean_c",
            "rainfall_mm": "_fc_rainfall_mm",
        })
        keep = [c for c in ("local_date", "_fc_temp_max_c", "_fc_temp_min_c",
                            "_fc_temp_mean_c", "_fc_rainfall_mm") if c in d]
        f = f.merge(d[keep], on="local_date", how="left")
        fc_day = (f[TIME] > origin) & f["_fc_temp_mean_c"].notna() \
            if "_fc_temp_mean_c" in f else pd.Series(False, index=f.index)
        for src, dst in (("_fc_temp_max_c", "weather_daily_temp_max_c"),
                         ("_fc_temp_min_c", "weather_daily_temp_min_c"),
                         ("_fc_temp_mean_c", "weather_daily_temp_mean_c"),
                         ("_fc_rainfall_mm", "weather_daily_rainfall_mm")):
            if src in f:
                f.loc[fc_day, dst] = f.loc[fc_day, src]
        f = f.drop(columns=[c for c in f.columns if c.startswith("_fc_")])

    # Weather lag/context features are training/history constructs. Stage 3 must
    # rebuild them recursively or from the current future weather frame.
    for c in ("temp_mean_lag_1d", "temp_mean_lag_7d", "rolling_temp_mean_7d",
              "hdd_lag_1d", "cdd_lag_1d"):
        f[c] = np.nan

    return f, forecast_meta


def build_horizon_safe_feature_report(columns):
    """Document why each feature is or is not safe at operational horizons."""
    rows=[]
    def add(feature,hist,next7,four,rule,reason):
        rows.append({"feature":feature,"historical_training":hist,"next_7_day_forecast":next7,
                     "four_month_recursive":four,"backtest_rule":rule,"reason":reason})
    for f in HORIZON_SAFE_LAGS: add(f,"Yes","Yes, recursive","Yes, recursive","rebuild per fold origin","actual history initially, then earlier forecasts")
    for f in SAFE_DEMAND_FEATURES: add(f,"Yes","Yes, recursive","Yes, recursive","rebuild per fold origin","anchored at t-48 or earlier")
    for f in CALENDAR_FEATURES+HOLIDAY_FEATURES+TREND_FEATURES: add(f,"Yes","Yes","Yes","","known/deterministic from timestamp")
    for f in HISTORICAL_FEATURES: add(f,"Yes","Rebuild","Rebuild/fallback","recompute from rows before fold origin","prior-history demand profile")
    for f in WEATHER_INPUT_FEATURES: add(f,"Observed weather","Forecast weather","Forecast then climatology","observed target-period weather cannot be used as a simulated future forecast","same numerical feature space; provenance changes by target timestamp")
    report=pd.DataFrame(rows); report["present_in_dataset"] = report["feature"].isin(columns)
    return report


def build_weather_feature_provenance():
    rows=[]
    for f in WEATHER_INPUT_FEATURES:
        rows.append({"feature":f,"historical_training_source":"OBSERVED_REANALYSIS","live_future_source_priority":"FORECAST_7DAY > CLIMATOLOGY_FALLBACK","predictor":True,"historical_backtest_rule":"do not use realised target-period weather as if forecast"})
    for f in WEATHER_CONTEXT_FEATURES:
        rows.append({"feature":f,"historical_training_source":"historical weather","live_future_source_priority":"origin-safe reconstruction/climatology","predictor":True,"historical_backtest_rule":"recompute at each fold origin"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"WP2 Ashish Shrestha module loaded: Feature Set A={len(FEATURE_SET_A)}, Feature Set B={len(FEATURE_SET_B)}")
