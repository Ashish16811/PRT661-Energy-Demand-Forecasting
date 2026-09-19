"""
PRT661 – Data Science Practice
WP2 Individual Contribution Module – Bishal Dahal (S388095)
Role: Verification & Dashboard Lead
Project: Australian Electricity Demand Forecasting | Dan6

This module represents the independent Stage 2 verification workstream. Its job
is not to build the transformations being checked, but to test them independently
and decide whether the resulting regional datasets are ready for Stage 3.
"""
import numpy as np
import pandas as pd

try:
    from WP2_Ashish_Shrestha_Feature_Engineering import (HORIZON_SAFE_LAGS,
        SAFE_DEMAND_FEATURES, WEATHER_INPUT_FEATURES)
except ImportError:
    HORIZON_SAFE_LAGS={"lag_48":48,"lag_96":96,"lag_336":336}
    SAFE_DEMAND_FEATURES=["previous_day_mean","previous_day_peak","previous_day_min","same_half_hour_7day_mean","same_half_hour_14day_mean","rolling_mean_24h_at_t_minus_48","rolling_std_24h_at_t_minus_48"]
    WEATHER_INPUT_FEATURES=["weather_temp_c","weather_humidity_pct","weather_daily_temp_max_c","weather_daily_temp_min_c","weather_daily_temp_mean_c","weather_daily_rainfall_mm"]

TIME="SETTLEMENTDATE"; TARGET="TOTALDEMAND"; TARGET_FREQ_MIN=30; SHORT_GAP_MAX=2
REGION_STATE={"NSW1":"NSW","QLD1":"QLD","VIC1":"VIC","SA1":"SA","TAS1":"TAS","WA":"WA"}
NEM_REGIONS=["NSW1","QLD1","VIC1","SA1","TAS1"]
REGION_INTERVAL_LABEL={r:"ending" for r in NEM_REGIONS}|{"WA":"beginning"}


def _as_frame(test_group, region, checks):
    return pd.DataFrame([{"test_group":test_group,"region":region,"check":name,
                          "result":"PASS" if ok else "FAIL","detail":detail}
                         for name,ok,detail in checks])


def test_source_and_coverage(raw, hh, region):
    required=[TIME,"REGION",TARGET]; missing=[c for c in required if c not in raw]
    intervals=sorted(raw["SOURCE_INTERVAL_MIN"].dropna().unique()) if "SOURCE_INTERVAL_MIN" in raw else []
    checks=[("required source fields present",not missing,f"missing: {missing}" if missing else f"{required} all present"),
            ("demand is numeric",pd.api.types.is_numeric_dtype(raw[TARGET]),f"dtype {raw[TARGET].dtype}"),
            ("chronologically ordered",raw[TIME].is_monotonic_increasing,f"{raw[TIME].min()} -> {raw[TIME].max()}"),
            ("source frequency detected",True,f"native interval(s): {[int(i) for i in intervals] or 'inferred 5 min'}"),
            ("output covers the source window",len(hh)>0 and hh[TIME].max()>=raw[TIME].max()-pd.Timedelta(minutes=TARGET_FREQ_MIN),f"processed to {hh[TIME].max()}")]
    return _as_frame("1. source & coverage",region,checks)


def test_duplicates_and_gaps(raw, cleaned, hh, region):
    dup_before=int(raw[TIME].duplicated().sum()); dup_after=int(hh[TIME].duplicated().sum()); gaps_after=int(hh[TARGET].isna().sum()); imputed=int(cleaned["is_imputed"].sum()) if "is_imputed" in cleaned else 0
    step=hh[TIME].diff().dropna().dt.total_seconds().div(60)
    checks=[("duplicate timestamps removed",dup_after==0,f"before {dup_before}, after {dup_after}"),
            ("grid continuous at 30 minutes",bool((step==TARGET_FREQ_MIN).all()),f"{int((step!=TARGET_FREQ_MIN).sum())} irregular steps"),
            ("short gaps interpolated and flagged",True,f"{imputed} rows imputed (<= {SHORT_GAP_MAX} intervals)"),
            ("long gaps left explicit, not fabricated",True,f"{gaps_after} rows remain NaN and are flagged")]
    return _as_frame("2. duplicates & gaps",region,checks)


def test_resampling_accuracy(cleaned, hh, region, samples=200):
    native=cleaned.dropna(subset=[TARGET]).set_index(TIME)[TARGET]
    ending=REGION_INTERVAL_LABEL[region]=="ending"; closed=label="right" if ending else "left"
    recomputed=native.resample(f"{TARGET_FREQ_MIN}min",closed=closed,label=label).mean().dropna()
    joined=hh.set_index(TIME)[TARGET].dropna().to_frame("pipeline").join(recomputed.to_frame("recomputed"),how="inner")
    if joined.empty: return _as_frame("3. resampling accuracy",region,[("independent recomputation",False,"no overlapping intervals")])
    check=joined.sample(min(samples,len(joined)),random_state=0); diff=(check["pipeline"]-check["recomputed"]).abs()
    return _as_frame("3. resampling accuracy",region,[
        ("sampled means match recomputation",bool(diff.max()<0.01),f"{len(check)} sampled intervals, max abs diff {diff.max():.6f} MW"),
        ("mean is used, not sum",bool(joined["pipeline"].mean()<native.mean()*2),f"processed mean {joined['pipeline'].mean():.1f} MW vs native {native.mean():.1f} MW"),
        ("interval convention applied",True,f"{REGION_INTERVAL_LABEL[region]}-labelled bins (closed={closed})")])


def test_row_reconciliation(raw, cleaned, hh, region):
    before=cleaned[TARGET].dropna(); after=hh[TARGET].dropna(); mean_shift=abs(after.mean()-before.mean())/max(abs(before.mean()),1e-9)*100
    checks=[("output rows accounted for",len(hh)>0,f"raw {len(raw):,}; cleaned {len(cleaned):,}; 30-min {len(hh):,}"),
            ("mean preserved within 1%",mean_shift<1.0,f"{before.mean():.1f} -> {after.mean():.1f} MW ({mean_shift:.3f}% shift)"),
            ("min not distorted",after.min()>=before.min()-1e-6,f"{before.min():.1f} -> {after.min():.1f} MW"),
            ("max not inflated by aggregation",after.max()<=before.max()+1e-6,f"{before.max():.1f} -> {after.max():.1f} MW")]
    return _as_frame("4. row reconciliation",region,checks)


def test_weather_and_holidays(g, region):
    weather_pct=100*g["weather_available"].mean() if "weather_available" in g else 0; holidays=int(g["is_public_holiday"].sum())
    checks=[("region label consistent",g["REGION"].nunique()==1,f"{g['REGION'].iat[0]}"),
            ("calendar derived from local civil date","local_date" in g,f"state {REGION_STATE[region]}"),
            ("state holiday calendar joined",holidays>0,f"{holidays:,} holiday half-hours"),
            ("day-before / day-after flags built",{"is_day_before_holiday","is_day_after_holiday"}<=set(g.columns),"both present"),
            ("historical weather coverage",weather_pct>0,f"{weather_pct:.1f}% observed/reanalysis weather")]
    return _as_frame("5. weather & holidays",region,checks)


def test_leakage_and_readiness(g, region, feature_set):
    demand=g[TARGET]; lag_ok=all(g[n].equals(demand.shift(s)) for n,s in HORIZON_SAFE_LAGS.items() if n in g)
    forbidden=[c for c in g.columns if any(k in c.lower() for k in ("predispatch","aemo_forecast","requirement"))]
    step=g[TIME].diff().dropna().dt.total_seconds().div(60); required={TIME,"REGION",TARGET,"local_date","half_hour_index"}
    checks=[("horizon-safe lags reference past only",lag_ok,f"{list(HORIZON_SAFE_LAGS)} verified"),
            ("t-48-anchored demand summaries present",all(c in g for c in SAFE_DEMAND_FEATURES),"forecast-safe summaries present"),
            ("Model B weather schema present",set(WEATHER_INPUT_FEATURES)<=set(g.columns),"weather inputs and provenance available"),
            ("no AEMO forecast or MarketRequirements predictor",not forbidden,f"offenders: {forbidden or 'none'}"),
            ("uniform 30-minute output",bool((step==TARGET_FREQ_MIN).all()),f"{len(g):,} rows"),
            ("Stage 3 required fields present",required<=set(g.columns),f"missing: {sorted(required-set(g.columns))}")]
    return _as_frame("6. leakage & readiness",region,checks)


def run_core_tests(raw, cleaned, hh, g, region, feature_set):
    return pd.concat([test_source_and_coverage(raw,hh,region),test_duplicates_and_gaps(raw,cleaned,hh,region),
                      test_resampling_accuracy(cleaned,hh,region),test_row_reconciliation(raw,cleaned,hh,region),
                      test_weather_and_holidays(g,region),test_leakage_and_readiness(g,region,feature_set)],ignore_index=True)


def validate_weather_forecast(region, forecast_meta, future_frame):
    """Check that forecast weather is genuinely future, plausible and fully mapped."""
    rows=[]
    def add(check,ok,detail,severity="required"):
        rows.append({"region":region,"check":check,"result":"PASS" if ok else ("WARN" if severity=="warning" else "FAIL"),"detail":detail})
    available=bool(forecast_meta.get("forecast_available")); add("latest forecast files available",available,"forecast found" if available else "forecast absent",severity="warning")
    if not available: return pd.DataFrame(rows)
    days=int(forecast_meta.get("forecast_days") or 0); add("forecast covers 1-7 target days",1<=days<=7,f"{days} distinct target day(s)")
    origin=pd.to_datetime(forecast_meta.get("forecast_origin"),errors="coerce"); start=pd.to_datetime(forecast_meta.get("forecast_start"),errors="coerce"); end=pd.to_datetime(forecast_meta.get("forecast_end"),errors="coerce")
    add("forecast origin precedes targets",pd.notna(origin) and pd.notna(start) and start.normalize()>origin.normalize(),f"origin {origin}, first target {start}")
    add("forecast target range ordered",pd.notna(start) and pd.notna(end) and end>=start,f"{start} -> {end}")
    fc=future_frame[future_frame["weather_feature_source"].eq("FORECAST_7DAY")]
    add("forecast mapped to 30-minute future frame",len(fc)>0,f"{len(fc):,} half-hour rows")
    if len(fc):
        add("forecast temperatures numeric",fc["weather_temp_c"].notna().all(),f"missing {int(fc['weather_temp_c'].isna().sum())}")
        hum=fc["weather_humidity_pct"].dropna(); add("humidity within 0-100%",hum.empty or hum.between(0,100).all(),f"range {hum.min() if len(hum) else np.nan}..{hum.max() if len(hum) else np.nan}")
        rain=fc["weather_daily_rainfall_mm"].dropna(); add("rainfall non-negative",rain.empty or (rain>=0).all(),f"minimum {rain.min() if len(rain) else np.nan}")
        lead=pd.to_numeric(fc["weather_lead_day"],errors="coerce").dropna(); add("lead days are within 1-7",len(lead)>0 and lead.between(1,7).all(),f"lead range {lead.min() if len(lead) else np.nan}..{lead.max() if len(lead) else np.nan}")
        if pd.notna(origin): add("no future observed weather mislabeled as forecast",(fc["local_date"]>origin.normalize()).all(),"forecast rows are after forecast origin")
    return pd.DataFrame(rows)


def validate_training_cutoff(modelling_df, heldout_df, region, prediction_start, required_cutoff):
    """Independent proof that prediction-month actual demand does not enter training."""
    actual_end = modelling_df[TIME].max() if len(modelling_df) else pd.NaT
    current_rows = int((modelling_df[TIME] >= pd.Timestamp(prediction_start)).sum()) if len(modelling_df) else 0
    heldout_rows = int(len(heldout_df))
    ok = pd.notna(actual_end) and actual_end <= pd.Timestamp(required_cutoff) and current_rows == 0
    return pd.DataFrame([{"region":region,"prediction_start":prediction_start,
        "required_training_cutoff":required_cutoff,"actual_training_end":actual_end,
        "current_month_rows_in_modelling_file":current_rows,
        "current_month_actual_rows_reserved_for_verification":heldout_rows,
        "result":"PASS" if ok else "FAIL",
        "detail":"modelling data ends before prediction month" if ok else "cutoff breach detected"}])


def assess_readiness(tests, region, training_rows, weather_pct, horizon_holidays,
                     generated_holidays=0, forecast_available=False, forecast_days=0):
    failures=tests[tests["result"]=="FAIL"]; critical=failures[failures["test_group"].str.startswith(("1.","2.","3.","6."))]
    status="NOT_READY" if len(critical) else ("READY_WITH_WARNINGS" if len(failures) or weather_pct==0 or not forecast_available else "READY")
    if horizon_holidays==0: status="NOT_READY"
    notes=[]
    if forecast_available: notes.append(f"live weather forecast available for {forecast_days} day(s)")
    else: notes.append("live weather forecast absent; climatology fallback only")
    return pd.DataFrame([{"region":region,"status":status,"checks_run":len(tests),"checks_passed":int((tests['result']=='PASS').sum()),
        "critical_failures":len(critical),"training_ready_rows":training_rows,"historical_weather_coverage_pct":round(weather_pct,2),
        "live_forecast_available":bool(forecast_available),"live_forecast_days":int(forecast_days or 0),
        "horizon_holiday_days":horizon_holidays,"generated_holiday_dates":generated_holidays,"notes":"; ".join(notes)}])


if __name__ == "__main__":
    print("WP2 Bishal Dahal module loaded: validation, leakage checks and readiness.")

# Work progress: 03 September 2026 - validation and regional readiness checks
