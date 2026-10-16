"""Deterministic plain-English insights (portal v3.0.0).

Every function is a pure transformation of evidence frames (or live values) into a
short sentence plus the small table a chart needs. No text is hard-coded about a
result: the direction, ranking and wording are computed from the data, so the
same code produces correct statements for all six regions (for example NSW1's
weaker 2024 year, TAS1's easier peaks and VIC1's non-monotonic horizon errors).
No external language model is used.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model_evidence import CORE_MODELS, MODEL_INFO, NON_SELECTION_EXPERIMENTS, model_feature_set

BANDS = ["Days 1-7", "Days 8-30", "Months 2-4"]
BAND_LABEL = {"Days 1-7": "the first week", "Days 8-30": "days 8–30", "Months 2-4": "months 2–4"}
MATERIAL_PCT = 1.0            # the project's practical-tie threshold
CLEAR_MARGIN_PCT = 3.0        # decision-confidence rule (declared in the design notes)
CLEAR_P_VALUE = 0.05


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def pct_words(pct: float, noun: str = "error") -> str:
    """'5.8% higher error' / '2.1% lower error' / 'about the same error' — never a bare sign."""
    if not _fin(pct):
        return "not available"
    if abs(pct) < 0.05:
        return f"about the same {noun}"
    return f"{abs(pct):.1f}% {'higher' if pct > 0 else 'lower'} {noun}"


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def forecast_direction_insight(current_mw, next_mw, flat_pct: float = 0.5) -> dict:
    if not (_fin(current_mw) and _fin(next_mw)) or float(current_mw) == 0:
        return {"direction": "unknown", "words": "Waiting for data", "pct": None,
                "sentence": "The next-interval comparison appears once both actual demand and the next forecast are available."}
    pct = 100.0 * (float(next_mw) - float(current_mw)) / float(current_mw)
    if abs(pct) < flat_pct:
        d, words = "flat", "Holding steady"
        sentence = "Demand is expected to stay about the same over the next 30 minutes."
    else:
        d = "up" if pct > 0 else "down"
        size = "sharply" if abs(pct) >= 5 else ("moderately" if abs(pct) >= 2 else "slightly")
        words = f"{'Rising' if d == 'up' else 'Falling'} {size}"
        sentence = f"Demand is expected to {'increase' if d == 'up' else 'decrease'} {size} over the next 30 minutes."
    return {"direction": d, "words": words, "pct": pct, "sentence": sentence}


def horizon_peak_insight(forecast: pd.DataFrame, horizon_label: str, time_col="target_timestamp",
                         value_col="forecast_mw") -> str:
    if forecast is None or forecast.empty:
        return ""
    i = forecast[value_col].idxmax()
    t = pd.Timestamp(forecast.loc[i, time_col])
    return f"Highest forecast demand in the {horizon_label} view is expected around {t:%a %d %b, %H:%M} (market time)."


def freshness(latest_actual, latest_origin, now) -> dict:
    """Stale-data guard: actual older than 60 min or forecast origin older than 2 h."""
    out = {"actual_stale": False, "forecast_stale": False, "messages": []}
    now = pd.Timestamp(now)
    if pd.notna(latest_actual):
        age = (now - pd.Timestamp(latest_actual)).total_seconds() / 60
        if age > 60:
            out["actual_stale"] = True
            out["messages"].append(f"Last available demand is {int(age)} minutes old.")
    else:
        out["actual_stale"] = True
        out["messages"].append("No actual demand is available yet.")
    if pd.notna(latest_origin):
        age_h = (now - pd.Timestamp(latest_origin)).total_seconds() / 3600
        if age_h > 2:
            out["forecast_stale"] = True
            out["messages"].append(f"Forecast update delayed — latest run {age_h:.1f} hours ago.")
    else:
        out["forecast_stale"] = True
        out["messages"].append("No rolling forecast has been stored yet.")
    return out


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def decision_confidence(margin_pct, p_value) -> dict:
    clear = _fin(margin_pct) and _fin(p_value) and float(margin_pct) >= CLEAR_MARGIN_PCT and float(p_value) < CLEAR_P_VALUE
    return {"label": "Clear margin" if clear else "Narrow margin", "clear": bool(clear),
            "rule": f"Clear margin = runner-up at least {CLEAR_MARGIN_PCT:.0f}% behind and Diebold–Mariano p < {CLEAR_P_VALUE}."}


def selected_model_insight(region: str, winners: pd.DataFrame) -> dict:
    w = winners[winners["region"] == region]
    if w.empty:
        return {}
    r = w.iloc[0]
    conf = decision_confidence(r.get("MAE_difference_pct"), r.get("dm_pvalue_vs_runner_up"))
    fs = r.get("feature_set")
    sentence = (f"{r.best_model} produced the lowest four-month forecast error for {region} across the matched "
                f"2023–2025 evaluation periods. The next-best model, {r.runner_up}, had "
                f"{pct_words(r.MAE_difference_pct)}"
                f"{' — a clear margin' if conf['clear'] else ' — a narrow margin'}.")
    return {"model": r.best_model, "feature_set": fs, "runner_up": r.runner_up,
            "margin_pct": float(r.MAE_difference_pct), "p_value": float(r.dm_pvalue_vs_runner_up)
            if _fin(r.get("dm_pvalue_vs_runner_up")) else None,
            "confidence": conf, "development_period": r.get("development_period"), "sentence": sentence,
            "MAE": float(r.MAE), "nMAE_pct": float(r.nMAE_pct)}


def ranking_frame(region: str, by_region: pd.DataFrame, failures: pd.DataFrame, selected: str | None) -> pd.DataFrame:
    """Ten core candidates only (sensitivity/ablation arms excluded), best first."""
    d = by_region[(by_region["region"] == region) & by_region["model"].isin(CORE_MODELS)].copy()
    if d.empty:
        return d
    fails = failures[failures["region"] == region].groupby("model").size() if not failures.empty else pd.Series(dtype=int)
    d["failed_runs"] = d["model"].map(fails).fillna(0).astype(int)
    d["complete"] = d["complete_coverage"].astype(str).str.lower().isin(["true", "1"]) if "complete_coverage" in d else True
    d["family_label"] = d["model"].map(lambda m: MODEL_INFO.get(m, ("", ""))[0])
    d["is_new"] = d["model"].map(lambda m: MODEL_INFO.get(m, ("", ""))[1] == "new")
    d["is_selected"] = d["model"] == selected
    d = d.sort_values(["complete", "mean_MAE"], ascending=[False, True]).reset_index(drop=True)
    return d


def ranking_sentence(rank: pd.DataFrame, region: str) -> str:
    if rank.empty:
        return "Model comparison evidence is not available for this region."
    complete = rank[rank["complete"]]
    top = complete.iloc[0]
    new_rows = complete[complete["is_new"]]
    best_new = new_rows.index.min() + 1 if not new_rows.empty else None
    inc = rank[~rank["complete"]]["model"].tolist()
    s = f"Of the ten candidate models, {top.model} ranked first for {region}."
    if best_new:
        s += f" The best new model after feedback ({complete.loc[best_new - 1, 'model']}) ranked {best_new}."
    if inc:
        s += f" {', '.join(inc)} did not complete every historical run and is shown separately."
    return s


def neural_comparison_insight(region: str, nvp: pd.DataFrame) -> dict:
    n = nvp[nvp["region"] == region]
    if n.empty:
        return {}
    r = n.iloc[0]
    champ = r.strongest_previous_model
    rows, improved = [], []
    for m in ("MLP", "GRU"):
        pct = float(r[f"{m}_MAE_change_vs_previous_pct"])
        material = str(r[f"{m}_material_improvement"]).lower() in ("true", "1")
        if material:
            improved.append(m)
        rows.append({"model": m, "change_pct": pct, "words": pct_words(pct), "material_improvement": material,
                     "p_value": r.get(f"{m}_dm_pvalue_vs_previous")})
    if not improved:
        closing = f"Neither neural model gave a material improvement over {champ} in {region}."
    elif len(improved) == 2:
        closing = f"Both neural models improved materially on {champ} in {region}."
    else:
        closing = f"{improved[0]} improved materially on {champ} in {region}; the other neural model did not."
    lines = [f"{x['model']}: {x['words']} than {champ}." for x in rows]
    return {"champion": champ, "rows": rows, "improved": improved, "sentence": " ".join(lines + [closing])}


def horizon_insight(region: str, horizon: pd.DataFrame, selected: str) -> dict:
    h = horizon[(horizon["region"] == region) & horizon["model"].isin([selected, "MLP", "GRU"]) & horizon["lead_band"].isin(BANDS)]
    if h.empty:
        return {"frame": h, "sentence": "Horizon evidence is not available."}
    piv = h.pivot_table(index="lead_band", columns="model", values="MAE").reindex(BANDS)
    s = piv[selected] if selected in piv else pd.Series(dtype=float)
    parts = []
    if s.notna().all():
        if s.is_monotonic_increasing:
            parts.append(f"{selected}'s error grows steadily as the forecast extends further ahead.")
        else:
            lo, hi = s.idxmin(), s.idxmax()
            parts.append(f"{selected}'s error is lowest in {BAND_LABEL[lo]} and highest in {BAND_LABEL[hi]}; "
                         f"it does not simply rise with every step further ahead.")
    beats = [b for b in BANDS if all(pd.notna(piv.loc[b, m]) and piv.loc[b, selected] <= piv.loc[b, m]
                                    for m in ("MLP", "GRU") if m in piv)]
    if len(beats) == 3:
        parts.append("It stays ahead of both neural models in every horizon.")
    elif beats:
        parts.append(f"It is ahead of both neural models in {', '.join(BAND_LABEL[b] for b in beats)}.")
    else:
        parts.append("A neural model is ahead in every horizon band.")
    gaps = {b: min((100 * (piv.loc[b, m] - piv.loc[b, selected]) / piv.loc[b, selected]) for m in ("MLP", "GRU") if m in piv)
            for b in BANDS}
    closest = min(gaps, key=gaps.get)
    parts.append(f"The neural models come closest in {BAND_LABEL[closest]}.")
    return {"frame": h, "pivot": piv, "sentence": " ".join(parts)}


def stability_insight(region: str, by_year: pd.DataFrame, selected: str) -> dict:
    ok = by_year["status"].isin(["MODEL_READY", "MODEL_READY_WITH_WARNING"])
    y = by_year[(by_year["region"] == region) & by_year["model"].isin(CORE_MODELS) & ok].copy()
    if y.empty:
        return {"frame": y, "sentence": "Year-by-year evidence is not available."}
    rows = []
    for yr, g in y.groupby("backtest_year"):
        if selected not in set(g["model"]):
            continue
        best = g["MAE"].min()
        mae = float(g.loc[g["model"] == selected, "MAE"].iloc[0])
        rows.append({"year": int(yr), "rank": int(g["MAE"].rank(method="min")[g["model"] == selected].iloc[0]),
                     "n_models": len(g), "gap_pct": 100 * (mae - best) / best, "MAE": mae,
                     "selection_year": bool(g["used_for_selection"].iloc[0]),
                     "origin_type": g["origin_type"].iloc[0]})
    f = pd.DataFrame(rows).sort_values("year")
    sel = f[f["selection_year"]]
    if sel.empty:
        return {"frame": f, "sentence": "No matched selection years are available."}
    wins = int((sel["rank"] == 1).sum())
    s = f"{selected} was the best model in {wins} of {len(sel)} matched years"
    weak = sel[sel["rank"] > 1]
    if weak.empty:
        s += "."
    else:
        details = "; ".join(f"in {int(r.year)} it ranked {int(r['rank'])} of {int(r.n_models)}, {r.gap_pct:.1f}% behind the best"
                            for _, r in weak.iterrows())
        s += f" ({details})."
    return {"frame": f, "sentence": s, "wins": wins}


def weather_insight(region: str, ablation: pd.DataFrame, by_region: pd.DataFrame, feature_set: str | None) -> dict:
    a = ablation[(ablation["region"] == region) & (ablation["variant"] == "XGBoost_B") & ablation["lead_band"].isin(BANDS)]
    parts = []
    for b in BANDS:
        v = a.loc[a["lead_band"] == b, "MAE_change_vs_set_A_pct"]
        if v.empty or not _fin(v.iloc[0]):
            continue
        pct = float(v.iloc[0])
        if abs(pct) < MATERIAL_PCT:
            parts.append((b, pct, f"{BAND_LABEL[b].capitalize()}: weather made little difference."))
        elif pct < 0:
            parts.append((b, pct, f"{BAND_LABEL[b].capitalize()}: weather reduced error by {abs(pct):.1f}%."))
        else:
            parts.append((b, pct, f"{BAND_LABEL[b].capitalize()}: weather added {pct:.1f}% error."))
    fs_sentence = feature_set_insight(region, by_region, feature_set)
    return {"bands": parts, "sentence": " ".join(p[2] for p in parts),
            "feature_set_sentence": fs_sentence,
            "caveat": ("Historical tests used typical (climatological) weather for the forecast period, "
                       "not the weather that actually happened. Weather tests compare XGBoost with and without weather features.")}


def feature_set_insight(region: str, by_region: pd.DataFrame, feature_set: str | None) -> str:
    d = by_region[(by_region["region"] == region) & by_region["model"].isin(["RandomForest_A", "RandomForest_B"])]
    if d.empty or feature_set not in ("A", "B"):
        return ""
    mae = d.set_index("model")["mean_MAE"]
    if not {"RandomForest_A", "RandomForest_B"} <= set(mae.index):
        return ""
    chosen, other = f"RandomForest_{feature_set}", f"RandomForest_{'B' if feature_set == 'A' else 'A'}"
    diff = 100 * (mae[other] - mae[chosen]) / mae[other]
    name = "weather-enhanced (B)" if feature_set == "B" else "core (A)"
    return (f"{region} uses the {name} feature set: in the 2023–2025 backtests it had "
            f"{abs(diff):.1f}% {'lower' if diff >= 0 else 'higher'} error than the other Random Forest configuration.")


DRIVER_WORDS = {"Recent demand": "recent demand", "Typical pattern (historical profiles)": "typical demand patterns", "Calendar / season": "calendar and season", "Weather": "weather", "Holidays": "holidays", "Trend": "the long-term trend"}
DRIVER_ORDER = ["Recent demand", "Typical pattern (historical profiles)", "Calendar / season", "Weather", "Holidays", "Trend"]


def drivers_insight(region: str, grouped: pd.DataFrame, top_n: int = 5) -> dict:
    g = grouped[grouped["region"] == region].sort_values("importance_share", ascending=False).head(top_n)
    if g.empty:
        return {"frame": g, "sentence": "Feature-importance evidence is not available."}
    first = g.iloc[0]
    s = (f"The model leans most on {DRIVER_WORDS.get(first.driver_group, first.driver_group.lower())} "
         f"(about {100 * first.importance_share:.0f}% of total importance)")
    w = g[g["driver_group"] == "Weather"]
    s += f"; weather accounts for about {100 * w.iloc[0].importance_share:.0f}%." if not w.empty else "; it uses no weather features."
    return {"frame": g, "sentence": s,
            "caveat": "Importance shows how much the model relied on a feature; it does not prove the feature causes demand to change."}


def peak_insight(region: str, summary: pd.DataFrame, selected: str) -> dict:
    s = summary[(summary["region"] == region) & (summary["model"] == selected)]
    if len(s) < 2:
        return {"frame": s, "sentence": "Peak evidence is not available.", "under_forecasts_peaks": None}
    normal = s[s["interval_type"].str.startswith("Normal")].iloc[0]
    peak = s[s["interval_type"].str.startswith("Peak")].iloc[0]
    ratio = peak.MAE / normal.MAE if normal.MAE else np.nan
    if ratio >= 1.1:
        sentence = f"Forecasting is harder in the highest-demand periods: error is about {ratio:.1f}× the normal level."
    elif ratio <= 0.9:
        sentence = "Forecasting was not harder in the highest-demand periods: peak-interval error was lower than normal."
    else:
        sentence = "Peak-interval error was similar to normal-interval error."
    under = bool(peak.Bias < 0)
    if under:
        sentence += " At peaks the model tended to forecast too low."
    return {"frame": s, "sentence": sentence, "under_forecasts_peaks": under, "ratio": ratio}


def region_difficulty(winners: pd.DataFrame) -> dict:
    if winners.empty:
        return {"frame": winners, "sentence": ""}
    f = winners[["region", "best_model", "nMAE_pct"]].sort_values("nMAE_pct")
    hi, lo = f.iloc[-1], f.iloc[0]
    return {"frame": f, "sentence": (f"Relative to typical demand, {lo.region} was the easiest region to forecast "
                                     f"and {hi.region} the hardest in the 2023–2025 backtests.")}


def additional_experiments(by_region: pd.DataFrame, region: str) -> pd.DataFrame:
    d = by_region[(by_region["region"] == region) & by_region["model"].isin(list(NON_SELECTION_EXPERIMENTS))]
    return d.assign(role=d["model"].map(NON_SELECTION_EXPERIMENTS))


def band_for_lead(lead: pd.Timedelta) -> str:
    days = lead / pd.Timedelta(days=1)
    return "Days 1-7" if days <= 7 else ("Days 8-30" if days <= 30 else "Months 2-4")


def feature_set_of(model: str) -> str | None:
    return model_feature_set(model)
