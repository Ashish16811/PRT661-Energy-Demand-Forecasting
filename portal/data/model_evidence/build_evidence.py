"""Build the curated dashboard evidence pack from the WP4 run (read-only, deterministic)."""
import pandas as pd, numpy as np, json, hashlib, shutil
from pathlib import Path
R = Path("/home/claude/wp4/run"); O = Path("/home/claude/v2/model_evidence/current"); O.mkdir(parents=True, exist_ok=True)
CORE = ["Naive48","Naive336","RandomForest_A","RandomForest_B","XGBoost_A","XGBoost_B","SARIMA","SARIMAX","MLP","GRU"]
notes = {}
# 1. copied unchanged
for src in ["models/selected_model.json","metrics/regional_model_winners.csv","metrics/model_comparison_by_region.csv",
            "metrics/neural_vs_previous_comparison.csv","metrics/model_stability.csv","metrics/statistical_comparison.csv",
            "metrics/common_model_summary.csv","metrics/common_model_comparison.csv","metrics/model_comparison_overall.csv",
            "validation/model_failures.csv","validation/model_registry.csv","validation/backtest_origins.csv",
            "validation/peak_thresholds.csv","validation/leakage_audit.csv","summary/run_status.json","summary/selection_trace.txt",
            "metrics/production_month_to_date_verification.csv"]:
    shutil.copy(R/src, O/Path(src).name); notes[Path(src).name] = "copied unchanged from " + src
# selected_model.json: strip windows paths only in a derived companion (original kept)
# 2. trimmed copies (row subset / column subset only, values untouched)
y = pd.read_csv(R/"metrics/model_comparison_by_year.csv")
keep = [c for c in ["region","model","model_family","development_period","model_role","feature_set","backtest_year","origin_type",
        "used_for_selection","status","MAE","RMSE","sMAPE","nMAE_pct","Bias","R2","MASE","fit_seconds","predict_seconds"] if c in y.columns]
y[keep].to_csv(O/"model_comparison_by_year.csv", index=False); notes["model_comparison_by_year.csv"]="column subset of metrics/model_comparison_by_year.csv"
h = pd.read_csv(R/"metrics/model_comparison_by_horizon.csv"); h = h[h.origin_scope=="SELECTION_MEAN"]
h.to_csv(O/"model_comparison_by_horizon.csv", index=False); notes["model_comparison_by_horizon.csv"]="rows with origin_scope == SELECTION_MEAN (mean of 2023-2025 origins)"
a = pd.read_csv(R/"metrics/weather_ablation_metrics.csv"); a = a[a.origin_scope=="SELECTION_MEAN"]
a.to_csv(O/"weather_ablation_metrics.csv", index=False); notes["weather_ablation_metrics.csv"]="rows with origin_scope == SELECTION_MEAN"
p = pd.read_csv(R/"metrics/peak_demand_metrics.csv"); p.to_csv(O/"peak_demand_metrics.csv", index=False); notes["peak_demand_metrics.csv"]="copied unchanged"
fi = pd.read_csv(R/"explainability/feature_importance.csv"); fi.to_csv(O/"feature_importance.csv", index=False); notes["feature_importance.csv"]="copied unchanged from explainability/"
# 3. derived (deterministic aggregations; documented)
sel = json.load(open(R/"models/selected_model.json"))
selm = {r["region"]: r["selected_model"] for r in sel["regions"]}
def grp(x):
    if x.startswith("lag_") or x in ["previous_day_mean","previous_day_peak","previous_day_min","same_half_hour_7day_mean","same_half_hour_14day_mean","rolling_mean_24h_at_t_minus_48","rolling_std_24h_at_t_minus_48"]: return "Recent demand"
    if x.startswith("historical_month"): return "Typical pattern (historical profiles)"
    if x in ["is_public_holiday","is_day_before_holiday","is_day_after_holiday"]: return "Holidays"
    if x in ["time_index","days_since_start","year"]: return "Trend"
    if x.startswith(("weather","temp","rolling_temp","expected","historical_rain","hdd","cdd")): return "Weather"
    return "Calendar / season"
f2 = fi[fi.apply(lambda r: selm.get(r.region)==r.model, axis=1) & (pd.to_datetime(fi.backtest_origin)>="2023-01-01")].copy()
f2["driver_group"] = f2.feature.map(grp)
g = f2.groupby(["region","model","driver_group"]).importance.mean().groupby(level=[0,1,2]).sum()  # mean per feature then sum
per_feat = f2.groupby(["region","model","driver_group","feature"]).importance.mean().reset_index()
grouped = per_feat.groupby(["region","model","driver_group"]).importance.sum().reset_index().rename(columns={"importance":"importance_share"})
grouped.to_csv(O/"feature_importance_grouped.csv", index=False)
per_feat.sort_values(["region","importance"],ascending=[True,False]).groupby("region").head(10).to_csv(O/"feature_importance_top10_selected.csv", index=False)
notes["feature_importance_grouped.csv"]="selected model per region; importance averaged per feature over 2023-2025 origins, then summed into driver groups (mapping in build_evidence.py)"
notes["feature_importance_top10_selected.csv"]="top 10 features of the selected model per region (mean over 2023-2025 origins)"
# error bands from interval detail for selected models, selection origins only
d = pd.read_csv(R/"forecasts/all_models_backtest_detail.csv.gz", usecols=["region","model","used_for_selection","lead_band","local_hour","residual_MW","absolute_error_MW","actual_MW","peak_flag"])
d = d[d.used_for_selection & d.apply(lambda r: selm[r.region]==r.model, axis=1)] if False else d[d.used_for_selection]
d = d[[selm[r]==m for r,m in zip(d.region,d.model)]]
def q(s,k): return float(np.quantile(s,k))
rows=[]
for (r,m,b),s in d.groupby(["region","model","lead_band"]):
    rows.append({"region":r,"model":m,"lead_band":b,"hour":"ALL","n":len(s),"residual_p10":q(s.residual_MW,.1),"residual_p50":q(s.residual_MW,.5),
                 "residual_p90":q(s.residual_MW,.9),"abs_error_p80":q(s.absolute_error_MW,.8),"MAE":s.absolute_error_MW.mean(),"mean_actual":s.actual_MW.mean()})
for (r,m,b,hh),s in d.groupby(["region","model","lead_band","local_hour"]):
    rows.append({"region":r,"model":m,"lead_band":b,"hour":int(hh),"n":len(s),"residual_p10":q(s.residual_MW,.1),"residual_p50":q(s.residual_MW,.5),
                 "residual_p90":q(s.residual_MW,.9),"abs_error_p80":q(s.absolute_error_MW,.8),"MAE":s.absolute_error_MW.mean(),"mean_actual":s.actual_MW.mean()})
pd.DataFrame(rows).to_csv(O/"forecast_error_bands.csv", index=False)
notes["forecast_error_bands.csv"]="empirical residual quantiles (forecast - actual) of the SELECTED model per region, 2023-2025 selection origins only, by lead band and by local hour; a historical error range, NOT a probabilistic forecast"
# peak summary: normal vs peak intervals for selected + MLP + GRU
rows=[]
d2 = pd.read_csv(R/"forecasts/all_models_backtest_detail.csv.gz", usecols=["region","model","used_for_selection","absolute_error_MW","residual_MW","peak_flag"])
d2 = d2[d2.used_for_selection]
for r in selm:
    for m in [selm[r],"MLP","GRU"]:
        s = d2[(d2.region==r)&(d2.model==m)]
        for flag,label in [(0,"Normal intervals"),(1,"Peak intervals (top 5% threshold)")]:
            t = s[s.peak_flag==flag]
            rows.append({"region":r,"model":m,"is_selected":m==selm[r],"interval_type":label,"n":len(t),"MAE":t.absolute_error_MW.mean(),"Bias":t.residual_MW.mean()})
pd.DataFrame(rows).to_csv(O/"peak_normal_vs_high_summary.csv", index=False)
notes["peak_normal_vs_high_summary.csv"]="MAE and bias of normal vs peak-flagged intervals (peak_flag from training-only thresholds), selection origins only"
# daily-mean error bands (for 30D/4M daily-mean chart views)
dd = pd.read_csv(R/"forecasts/all_models_daily_detail.csv")
dd = dd[dd.used_for_selection & (dd.n_intervals == 48)]
dd = dd[[selm[r] == m for r, m in zip(dd.region, dd.model)]]
dd["lead_band"] = np.where(dd.lead_day <= 7, "Days 1-7", np.where(dd.lead_day <= 30, "Days 8-30", "Months 2-4"))
dd["daily_residual"] = dd.forecast_mean_MW - dd.actual_mean_MW
rows = []
for (r, m, b), s in dd.groupby(["region", "model", "lead_band"]):
    rows.append({"region": r, "model": m, "lead_band": b, "n_days": len(s), "residual_p10": q(s.daily_residual, .1),
                 "residual_p50": q(s.daily_residual, .5), "residual_p90": q(s.daily_residual, .9),
                 "daily_mean_abs_error": s.daily_residual.abs().mean()})
pd.DataFrame(rows).to_csv(O/"forecast_error_bands_daily.csv", index=False)
notes["forecast_error_bands_daily.csv"] = "empirical quantiles of daily-mean residual (forecast mean - actual mean) of the SELECTED model, complete days only, 2023-2025 selection origins, by lead band; used for daily-mean chart views"
# 4. manifest
env = json.load(open(R/"environment/run_environment.json"))
st = json.load(open(R/"summary/run_status.json"))
files = {}
for f in sorted(O.iterdir()):
    if f.name == "evidence_manifest.json": continue
    rows_n = None
    if f.suffix == ".csv": rows_n = int(sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1)
    files[f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size, "rows": rows_n, "provenance": notes.get(f.name, "")}
manifest = {
  "evidence_version": "wp4-2026-09-29",
  "pack_revision": 2,
  "source_run": st.get("run_id") or st.get("run_name") or "EnergyForecast_Run_20260929_145931",
  "pipeline_version": env.get("pipeline_version"),
  "selected_model_generated": sel.get("generated"),
  "selection_rule": "lowest mean four-month MAE over FULL_HISTORY origins 2023-09-01, 2024-09-01, 2025-09-01; 1% practical tie; sequential tie-breaks",
  "selection_origins": ["2023-09-01","2024-09-01","2025-09-01"],
  "limited_history_origin": "2022-09-01 (reported, not used for selection)",
  "core_models": CORE,
  "non_selection_experiments": {"GRU_L672":"SENSITIVITY_ONLY","XGBoost_A+CoreWeather":"WEATHER_ABLATION_ONLY"},
  "selected_models": selm,
  "weather_policy_note": "historical backtests used origin-safe climatology on the horizon, never realised future weather",
  "files": files,
}
json.dump(manifest, open(O/"evidence_manifest.json","w"), indent=2)
# 5. test expectations (derived from the files, for automated tests only - never displayed as hard-coded text)
w = pd.read_csv(O/"regional_model_winners.csv").set_index("region")
nv = pd.read_csv(O/"neural_vs_previous_comparison.csv").set_index("region")
exp = {"selected_model": selm,
       "feature_set": {r: w.loc[r,"feature_set"] for r in selm},
       "runner_up": {r: w.loc[r,"runner_up"] for r in selm},
       "mlp_material_improvement": {r: bool(nv.loc[r,"MLP_material_improvement"]) for r in selm},
       "gru_material_improvement": {r: bool(nv.loc[r,"GRU_material_improvement"]) for r in selm},
       "sarimax_failed_runs": int((pd.read_csv(O/"model_failures.csv").model=="SARIMAX").sum()),
       "core_model_count": 10}
json.dump(exp, open(O.parent/"dashboard_test_expectations.json","w"), indent=2, default=str)
print(json.dumps({k:(v["rows"],v["bytes"]) for k,v in files.items()}, indent=0)); print(exp)
