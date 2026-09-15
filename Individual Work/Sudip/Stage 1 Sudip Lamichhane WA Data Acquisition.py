"""
PRT661 – Data Science Practice
WP1 Stage 1 contribution module – Sudip Lamichhane (S388085)
Primary scope: WA/WEM operational-demand acquisition, cutover handling and
AEMO MarketRequirements-2026 acquisition.

The key design rule is source preservation: legacy 30-minute WA observations are
not expanded to five-minute history during acquisition. SOURCE and
SOURCE_INTERVAL_MIN travel with every output row.
"""
from __future__ import annotations
import argparse
import logging
import os
import time
from io import StringIO
from pathlib import Path
import pandas as pd
import requests

log = logging.getLogger("sudip_wem")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
WA_OLD_BASE = "https://data.wa.aemo.com.au/datafiles/operational-demand"
WA_NEW_BASE = "https://data.wa.aemo.com.au/datafiles/operational-demand-withdrawal-csv"
WA_DAILY_BASE = "https://data.wa.aemo.com.au/public/market-data/wemde/operationalDemandWithdrawal/dailyFiles"
WA_MARKET_REQUIREMENTS_2026 = "https://data.wa.aemo.com.au/datafiles/market-requirements-csv/MarketRequirements-2026.csv"


def _parse_wa_datetime(series):
    s = series.astype(str).str.strip().str.strip('"')
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    out.loc[iso] = pd.to_datetime(s.loc[iso], errors="coerce", yearfirst=True)
    out.loc[~iso] = pd.to_datetime(s.loc[~iso], errors="coerce", dayfirst=True)
    return out


def _find_wa_timestamp(df, native_minutes):
    cols = list(df.columns); low = {c: str(c).strip().lower() for c in cols}
    ranked = sorted(cols, key=lambda c: (0 if any(k in low[c] for k in ("timestamp", "dispatch interval", "trading interval")) else 1,
                                        0 if any(k in low[c] for k in ("date", "time", "interval")) else 1))
    for col in ranked:
        if not any(k in low[col] for k in ("date", "time", "interval", "timestamp")):
            continue
        text = df[col].astype(str).str.strip()
        if text.str.contains(r"[-/:T]", regex=True, na=False).mean() < .80:
            continue
        parsed = _parse_wa_datetime(df[col]); valid = parsed.notna() & parsed.dt.year.between(2000, 2100)
        if valid.mean() >= .80 and parsed[valid].nunique() / valid.sum() >= .80:
            return parsed
    date_cols = [c for c in cols if "date" in low[c]]
    interval_cols = [c for c in cols if "interval" in low[c] or "period" in low[c]]
    for dcol in date_cols:
        base = _parse_wa_datetime(df[dcol]).dt.normalize()
        for icol in interval_cols:
            n = pd.to_numeric(df[icol], errors="coerce"); max_i = int(24*60/native_minutes)
            if n.notna().mean() >= .80 and n.dropna().between(1, max_i).all():
                return base + pd.Timedelta(hours=8) + pd.to_timedelta((n-1)*native_minutes, unit="m")
    raise ValueError(f"Could not identify WA timestamp column. Columns: {cols}")


def _find_wa_demand_column(df):
    candidates = []
    for col in df.columns:
        name = str(col).strip().lower().replace("_", " ")
        if "demand" not in name or "forecast" in name or "withdrawal" in name:
            continue
        score = (4 if "operational" in name else 0) + (2 if "mw" in name else 0)
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= .80:
            candidates.append((score, col))
    if not candidates:
        raise ValueError(f"Could not identify WA operational-demand column: {list(df.columns)}")
    return max(candidates)[1]


def _normalise_wa_frame(df, source, native_minutes):
    ts = _find_wa_timestamp(df, native_minutes)
    demand = pd.to_numeric(df[_find_wa_demand_column(df)], errors="coerce")
    out = pd.DataFrame({"SETTLEMENTDATE": ts, "TOTALDEMAND": demand}).dropna()
    out = out.drop_duplicates("SETTLEMENTDATE", keep="last").sort_values("SETTLEMENTDATE")
    out["REGION"]="WA"; out["PERIODTYPE"]="ACTUAL"; out["SOURCE"]=source
    out["SOURCE_INTERVAL_MIN"]=native_minutes
    return out[["REGION","SETTLEMENTDATE","TOTALDEMAND","PERIODTYPE","SOURCE","SOURCE_INTERVAL_MIN"]]


def _largest_record_list(obj):
    found=[]
    def walk(v):
        if isinstance(v,list):
            if v and all(isinstance(x,dict) for x in v): found.append(v)
            for x in v: walk(x)
        elif isinstance(v,dict):
            for x in v.values(): walk(x)
    walk(obj); return max(found,key=len) if found else []


def _download_csv(session,url,source,native_minutes):
    r=session.get(url,timeout=60); r.raise_for_status()
    return _normalise_wa_frame(pd.read_csv(StringIO(r.text)),source,native_minutes)


def _fetch_daily_tail(session,start_date,end_date):
    frames=[]
    for day in pd.date_range(start_date,end_date,freq="D"):
        stamp=day.strftime("%Y-%m-%d")
        url=f"{WA_DAILY_BASE}/OperationalDemandAndWithdrawal_{stamp}.json"
        try:
            r=session.get(url,timeout=30)
            if r.status_code==404: continue
            r.raise_for_status(); records=_largest_record_list(r.json())
            if records: frames.append(_normalise_wa_frame(pd.json_normalize(records),"AEMO_WEM_DAILY_5MIN",5))
        except Exception as exc: log.warning("WA daily %s: %s",stamp,exc)
        time.sleep(.05)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def fetch_wa_demand(start_date,end_date,demand_dir):
    out=Path(demand_dir); out.mkdir(parents=True,exist_ok=True)
    start=pd.Timestamp(start_date).normalize(); end=pd.Timestamp(end_date).normalize(); end_x=end+pd.Timedelta(days=1)
    session=requests.Session(); session.headers.update({"User-Agent":UA,"Accept":"*/*"})
    legacy=[]; modern=[]
    for year in (2022,2023):
        try: legacy.append(_download_csv(session,f"{WA_OLD_BASE}/operational-demand-{year}.csv",f"AEMO_WEM_LEGACY_{year}",30))
        except Exception as exc: log.error("Legacy %s failed: %s",year,exc)
    for year in (2023,2024,2025,2026):
        try: modern.append(_download_csv(session,f"{WA_NEW_BASE}/OperationalDemandWithdrawal-{year}.csv",f"AEMO_WEM_{year}",5))
        except Exception as exc: log.error("Modern %s failed: %s",year,exc)
    frames=[]
    if legacy:
        old=pd.concat(legacy,ignore_index=True); frames.append(old[old.SETTLEMENTDATE < pd.Timestamp("2023-10-01")])
    if modern:
        new=pd.concat(modern,ignore_index=True); new=new[new.SETTLEMENTDATE >= pd.Timestamp("2023-10-01")]; frames.append(new)
        y2026=new[new.SETTLEMENTDATE.dt.year==2026]
        tail_start=y2026.SETTLEMENTDATE.max().normalize() if not y2026.empty else pd.Timestamp("2026-01-01")
        tail_end=min(end,pd.Timestamp.today().normalize())
        if tail_start<=tail_end:
            tail=_fetch_daily_tail(session,tail_start,tail_end)
            if not tail.empty: frames.append(tail)
    if not frames: raise RuntimeError("No WA operational-demand source was retrieved")
    final=pd.concat(frames,ignore_index=True)
    final=final[(final.SETTLEMENTDATE>=start)&(final.SETTLEMENTDATE<end_x)].copy()
    final["_priority"]=final.SOURCE.map(lambda x:3 if "DAILY" in x else (2 if "WEM_20" in x else 1))
    final=(final.sort_values(["SETTLEMENTDATE","_priority"],kind="stable")
           .drop_duplicates("SETTLEMENTDATE",keep="last").drop(columns="_priority")
           .sort_values("SETTLEMENTDATE"))
    path=out/"WA_demand_2022_2026.csv"; final.to_csv(path,index=False,date_format="%Y-%m-%d %H:%M:%S")
    return str(path)


def fetch_wa_market_requirements_2026(output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); path=out/"MarketRequirements-2026.csv"
    r=requests.get(WA_MARKET_REQUIREMENTS_2026,headers={"User-Agent":UA,"Accept":"text/csv,text/plain,*/*"},timeout=60)
    r.raise_for_status()
    if r.content.lstrip().lower().startswith((b"<!doctype",b"<html")):
        raise ValueError("AEMO returned HTML instead of MarketRequirements CSV")
    path.write_bytes(r.content); return str(path)


def main():
    p=argparse.ArgumentParser(description="Sudip WP1 – WA/WEM acquisition")
    p.add_argument("--start",default="2022-01-01"); p.add_argument("--end",default=pd.Timestamp.today().date().isoformat())
    p.add_argument("--demand-dir",required=True); p.add_argument("--requirements-dir",required=True)
    a=p.parse_args(); fetch_wa_demand(a.start,a.end,a.demand_dir); fetch_wa_market_requirements_2026(a.requirements_dir)

if __name__=="__main__": main()
