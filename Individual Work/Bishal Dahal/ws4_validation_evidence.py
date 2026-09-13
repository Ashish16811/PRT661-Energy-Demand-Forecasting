"""
ws4_validation_evidence.py - Validation, weather, calendar and evidence
PRT661 WP2 Preprocessing · Dan6: Theme 2
Workstream owner: Bishal Dahal (Verification & Dashboard Lead)
Peer reviewer: Ashish Shrestha

Weather source validation · holiday calendar validation and joins ·
output verification · evidence figures.

This workstream verifies the output of the other three. Nothing here trusts an
upstream claim: the holiday join is re-checked against the calendar files, the
processed grid is re-measured rather than assumed, and the weather source is
validated before any downstream code is allowed to read it.

WEATHER IS BLOCKED, AND STAYS VISIBLY BLOCKED.
load_weather() returns None while any file is unusable. No weather value is
imputed anywhere, and HDD/CDD are omitted entirely rather than left blank,
because no base temperature has been justified in the supplied material.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from common import REGION_STATE, TARGET_FREQ_MIN, _fig
from ws3_features_eda import seasonal_residual


def check_bom_weather(raw_dir: Path) -> pd.DataFrame:
    """Verify BOM files contain data before anything downstream trusts them.
    Detects an HTML page saved under a .csv name, and a ZIP saved as CSV -
    both of which the corrected acquisition code can now produce or prevent."""
    rows = []
    for f in sorted((raw_dir / "bom_weather").glob("bom_*.csv")):
        head = f.read_bytes()[:512]
        low = head.decode("latin-1", errors="replace").lstrip().lower()
        if head[:2] == b"PK":
            status, why = "ZIP_NOT_CSV", "BOM returned a zip archive; extract the inner CSV"
        elif low.startswith(("<!doctype", "<html")):
            status, why = "HTML_NOT_DATA", "BOM served the landing page, not the data file"
        elif "date" in low.split("\n")[0]:
            status, why = "ok", "CSV header present"
        else:
            status, why = "UNRECOGNISED", "neither HTML, ZIP, nor a CSV header"
        rows.append({"file": f.name, "bytes": f.stat().st_size,
                     "status": status, "diagnosis": why})
    out = pd.DataFrame(rows)
    bad = int((out["status"] != "ok").sum())
    if bad:
        log.error(f"[WS4] {bad}/{len(out)} BOM files unusable - weather features "
                  f"held back, NOT imputed. See WEATHER_STATUS.md")
    return out


def load_weather(raw_dir: Path, region: str, status: pd.DataFrame) -> pd.DataFrame | None:
    """Return None when the source is unusable. Never returns fabricated
    values - a blocked source must stay visibly blocked."""
    files = status[status["file"].str.contains(f"_{region}_")]
    if files.empty or (files["status"] != "ok").any():
        return None
    frames = {}
    for _, r in files.iterrows():
        kind = ("max_temperature" if "max_temperature" in r["file"]
                else "min_temperature" if "min_temperature" in r["file"] else "rainfall")
        d = pd.read_csv(raw_dir / "bom_weather" / r["file"])
        frames[kind] = d
    return frames or None


def validate_holidays(raw_dir: Path, regions) -> pd.DataFrame:
    """Each region joins its OWN state calendar. This is not cosmetic:
    Labour Day falls in March in VIC/TAS, May in QLD, June in WA and October
    in NSW/SA, and WA/QLD/NT observe no Queen's/King's Birthday on the same
    date as the south-east. Joining one national calendar would mislabel
    dozens of days per region, and a public holiday is one of the strongest
    single-day predictors of a demand shortfall."""
    rows = []
    for r in regions:
        state = REGION_STATE[r]
        p = raw_dir / "public_holidays" / f"public_holidays_{r}_{state}.csv"
        if not p.exists():
            rows.append({"region": r, "state": state, "file": p.name,
                         "status": "MISSING", "n_holidays": 0,
                         "date_min": "", "date_max": "", "unique_names": 0})
            continue
        h = pd.read_csv(p, parse_dates=["date"])
        hh = h[h["is_public_holiday"].astype(bool)]
        rows.append({"region": r, "state": state, "file": p.name, "status": "ok",
                     "n_holidays": len(hh), "date_min": hh["date"].min().date(),
                     "date_max": hh["date"].max().date(),
                     "unique_names": hh["holiday_name"].nunique()})
    return pd.DataFrame(rows)


def load_holidays(raw_dir: Path, region: str) -> pd.DataFrame:
    p = raw_dir / "public_holidays" / f"public_holidays_{region}_{REGION_STATE[region]}.csv"
    h = pd.read_csv(p, parse_dates=["date"])
    return h[["date", "is_public_holiday", "holiday_name"]]


def join_holidays(g: pd.DataFrame, raw_dir: Path, region: str) -> pd.DataFrame:
    """Joined on LOCAL_DATE, not the market-clock date. On the market clock a
    NEM region's late-evening intervals can carry the following civil date
    (and SA is 30 minutes off AEST all year), so joining on the raw stamp
    mislabels the boundary intervals of every holiday."""
    h = load_holidays(raw_dir, region).rename(columns={"date": "local_date"})
    g = g.merge(h, on="local_date", how="left")
    g["is_public_holiday"] = g["is_public_holiday"].fillna(0).astype(int)
    g["holiday_name"] = g["holiday_name"].fillna("")
    return g


def verify_processed_outputs(tbl: pd.DataFrame, region: str) -> pd.DataFrame:
    checks = []

    def add(name, ok, detail):
        checks.append({"region": region, "check": name,
                       "result": "PASS" if ok else "FAIL", "detail": detail})

    add("chronologically sorted", tbl["SETTLEMENTDATE"].is_monotonic_increasing,
        f"{len(tbl):,} rows")
    add("no duplicate timestamps", not tbl["SETTLEMENTDATE"].duplicated().any(), "")
    step = tbl["SETTLEMENTDATE"].diff().dropna().dt.total_seconds().div(60)
    add(f"uniform {TARGET_FREQ_MIN}-minute grid", bool((step == TARGET_FREQ_MIN).all()),
        f"{int((step != TARGET_FREQ_MIN).sum())} irregular steps")
    add("target present", tbl["TOTALDEMAND"].notna().any(),
        f"{int(tbl['TOTALDEMAND'].isna().sum())} NaN target rows (long gaps, retained)")
    add("48 intervals per civil day where a full day exists",
        tbl.groupby(tbl["local_date"]).size().mode().iat[0] == 48,
        "modal rows/day")
    add("holiday flag joined", tbl["is_public_holiday"].sum() > 0,
        f"{int(tbl['is_public_holiday'].sum()):,} holiday half-hours")
    add("weather explicitly unavailable, not imputed",
        bool((tbl["weather_available"] == 0).all()),
        "weather columns held back pending BOM re-download")
    return pd.DataFrame(checks)


def generate_quality_figures(cleaned: pd.DataFrame, model: dict, eda_dir: Path) -> list[str]:
    """Per-region and cross-region evidence figures, all from real data."""
    made = []
    reg_dir = eda_dir / "regional"
    reg_dir.mkdir(parents=True, exist_ok=True)

    for region, g in model.items():
        g = g.dropna(subset=["TOTALDEMAND"])
        if g.empty:
            continue
        d = g["TOTALDEMAND"]

        # trend (daily mean keeps the figure readable at 4 years)
        fig, ax = plt.subplots(figsize=(11, 3.4))
        g.set_index("SETTLEMENTDATE")["TOTALDEMAND"].resample("D").mean().plot(ax=ax, lw=0.7)
        ax.set(title=f"{region} - daily mean operational demand, 2022-2026",
               ylabel="MW", xlabel="")
        _fig(reg_dir / f"{region}_01_trend.png", fig); made.append(f"{region}_01_trend.png")

        # average daily profile + weekday/weekend
        fig, ax = plt.subplots(figsize=(7, 3.6))
        g.groupby("half_hour_index")["TOTALDEMAND"].mean().plot(ax=ax, label="all days")
        for lbl, sel in [("weekday", g["is_weekend"] == 0), ("weekend", g["is_weekend"] == 1)]:
            g[sel].groupby("half_hour_index")["TOTALDEMAND"].mean().plot(ax=ax, label=lbl, ls="--")
        ax.set(title=f"{region} - mean half-hourly profile", xlabel="half-hour index (local)",
               ylabel="MW"); ax.legend()
        _fig(reg_dir / f"{region}_02_daily_profile.png", fig); made.append(f"{region}_02_daily_profile.png")

        # day-of-week and monthly
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        g.groupby("day_of_week")["TOTALDEMAND"].mean().plot(kind="bar", ax=axes[0])
        axes[0].set(title=f"{region} - mean demand by day of week", xlabel="0=Mon", ylabel="MW")
        g.groupby("month")["TOTALDEMAND"].mean().plot(kind="bar", ax=axes[1], color="tab:orange")
        axes[1].set(title=f"{region} - mean demand by month", xlabel="month", ylabel="MW")
        _fig(reg_dir / f"{region}_03_dow_month.png", fig); made.append(f"{region}_03_dow_month.png")

        # seasonal profile
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for s in ["Summer", "Autumn", "Winter", "Spring"]:
            sub = g[g["season"] == s]
            if len(sub):
                sub.groupby("half_hour_index")["TOTALDEMAND"].mean().plot(ax=ax, label=s)
        ax.set(title=f"{region} - seasonal half-hourly profile", xlabel="half-hour index",
               ylabel="MW"); ax.legend()
        _fig(reg_dir / f"{region}_04_seasonal.png", fig); made.append(f"{region}_04_seasonal.png")

        # histogram / boxplot / Q-Q
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        axes[0].hist(d, bins=80, color="tab:blue"); axes[0].set(title=f"{region} - histogram", xlabel="MW")
        axes[1].boxplot(d, vert=True, widths=0.5); axes[1].set(title=f"{region} - boxplot", ylabel="MW")
        stats.probplot(d.sample(min(20000, len(d)), random_state=0), dist="norm", plot=axes[2])
        axes[2].set_title(f"{region} - Q-Q vs normal")
        _fig(reg_dir / f"{region}_05_distribution.png", fig); made.append(f"{region}_05_distribution.png")

        # raw vs log10 vs seasonal residual
        resid = seasonal_residual(g).dropna()
        pos = d[d > 0]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        axes[0].hist(pos, bins=80); axes[0].set_title(f"{region} - raw")
        axes[1].hist(np.log10(pos), bins=80, color="tab:green"); axes[1].set_title(f"{region} - log10")
        axes[2].hist(resid, bins=80, color="tab:purple"); axes[2].set_title(f"{region} - seasonal residual")
        _fig(reg_dir / f"{region}_06_transforms.png", fig); made.append(f"{region}_06_transforms.png")

        # ramp distribution
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(d.diff().dropna(), bins=100, color="tab:red")
        ax.set(title=f"{region} - 30-minute ramp distribution", xlabel="MW change", yscale="log")
        _fig(reg_dir / f"{region}_07_ramp.png", fig); made.append(f"{region}_07_ramp.png")

        # holiday vs ordinary
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for lbl, sel in [("ordinary weekday", (g["is_public_holiday"] == 0) & (g["is_weekend"] == 0)),
                         ("public holiday", g["is_public_holiday"] == 1)]:
            sub = g[sel]
            if len(sub):
                sub.groupby("half_hour_index")["TOTALDEMAND"].mean().plot(ax=ax, label=lbl)
        ax.set(title=f"{region} - public holiday vs ordinary weekday", ylabel="MW",
               xlabel="half-hour index"); ax.legend()
        _fig(reg_dir / f"{region}_08_holiday.png", fig); made.append(f"{region}_08_holiday.png")

        # flagged points
        c = cleaned[cleaned["REGION"] == region]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.plot(c["SETTLEMENTDATE"], c["TOTALDEMAND"], lw=0.2, color="0.75", label="demand")
        for flag, col, lbl in [("is_invalid", "red", "invalid (<=0 MW)"),
                               ("is_stat_outlier", "orange", "statistical outlier")]:
            sel = c[c[flag].astype(bool)]
            if len(sel):
                ax.scatter(sel["SETTLEMENTDATE"], sel["TOTALDEMAND"], s=3, c=col, label=lbl)
        ax.set(title=f"{region} - flagged values (retained, not deleted)", ylabel="MW")
        ax.legend(markerscale=3)
        _fig(reg_dir / f"{region}_09_flagged.png", fig); made.append(f"{region}_09_flagged.png")

    # ---- cross-region ----
    allm = pd.concat([g.assign(REGION=r) for r, g in model.items()], ignore_index=True)
    allm = allm.dropna(subset=["TOTALDEMAND"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    allm.groupby("REGION")["TOTALDEMAND"].mean().plot(kind="bar", ax=axes[0])
    axes[0].set(title="Mean demand by region", ylabel="MW")
    allm.groupby("REGION")["TOTALDEMAND"].std().plot(kind="bar", ax=axes[1], color="tab:orange")
    axes[1].set(title="Demand variability (SD) by region", ylabel="MW")
    allm.groupby("REGION")["TOTALDEMAND"].max().plot(kind="bar", ax=axes[2], color="tab:red")
    axes[2].set(title="Peak demand by region", ylabel="MW")
    _fig(eda_dir / "cross_region_01_levels.png", fig); made.append("cross_region_01_levels.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for r, g in allm.groupby("REGION"):
        p = g.groupby("half_hour_index")["TOTALDEMAND"].mean()
        ((p - p.min()) / (p.max() - p.min())).plot(ax=ax, label=r)
    ax.set(title="Normalised 24-hour demand shape by region (local civil time)",
           xlabel="half-hour index", ylabel="normalised 0-1"); ax.legend(ncol=3)
    _fig(eda_dir / "cross_region_02_shape.png", fig); made.append("cross_region_02_shape.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    piv = allm.pivot_table(index="month", columns="REGION", values="TOTALDEMAND", aggfunc="mean")
    (piv / piv.mean()).plot(ax=ax)
    ax.set(title="Monthly demand relative to each region's own mean", ylabel="ratio")
    _fig(eda_dir / "cross_region_03_seasonality.png", fig); made.append("cross_region_03_seasonality.png")
    return made

