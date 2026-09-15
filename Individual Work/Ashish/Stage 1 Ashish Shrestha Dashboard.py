"""
PRT661 – Data Science Practice
WP1 Stage 1 contribution module – Ashish Shrestha (S388084)
Primary scope: AEMO dashboard dispatch and pre-dispatch snapshot acquisition.

The two views are intentionally stored separately. PERIODTYPE is preserved because
both API views may contain different record types; pre-dispatch data is never folded
into observed demand during acquisition.
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

log=logging.getLogger("ashish_dashboard")
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
DASHBOARD_URL="https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
HEADERS={"User-Agent":UA,"Content-Type":"application/json","Accept":"application/json",
         "Origin":"https://www.aemo.com.au","Referer":"https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/data-dashboard-nem"}
NEM_REGIONS=["NSW1","QLD1","VIC1","SA1","TAS1"]
DASHBOARD_VIEWS={"dispatch":"5MIN","predispatch":"30MIN"}


def _dashboard_request(scale):
    payload={"timeScale":[scale]}
    r=requests.post(DASHBOARD_URL,headers=HEADERS,json=payload,timeout=60)
    r.raise_for_status(); data=r.json()
    if isinstance(data,list): rows=data
    elif isinstance(data,dict):
        # choose the largest list-of-dicts found in the response
        lists=[]
        def walk(v):
            if isinstance(v,list):
                if v and all(isinstance(x,dict) for x in v): lists.append(v)
                for x in v: walk(x)
            elif isinstance(v,dict):
                for x in v.values(): walk(x)
        walk(data); rows=max(lists,key=len) if lists else []
    else: rows=[]
    if not rows: raise ValueError("No tabular rows found in AEMO dashboard response")
    return pd.DataFrame(rows)


def fetch_dispatch_and_predispatch(regions,dispatch_dir,predispatch_dir,snapshot_dir=None):
    dirs={"dispatch":Path(dispatch_dir),"predispatch":Path(predispatch_dir)}
    for d in dirs.values(): d.mkdir(parents=True,exist_ok=True)
    pulled_at=datetime.now()
    for view,scale in DASHBOARD_VIEWS.items():
        try: df=_dashboard_request(scale)
        except Exception as exc:
            log.error("%s request failed: %s",view,exc); continue
        if "REGIONID" not in df.columns:
            path=dirs[view]/f"{view}_ALL_REGIONS.csv"; df.to_csv(path,index=False)
            log.error("%s response lacked REGIONID; unsplit file written",view); continue
        for region in regions:
            sub=df[df.REGIONID==region]
            if sub.empty:
                log.warning("%s/%s: no rows",view,region); continue
            path=dirs[view]/f"{view}_{region}.csv"; sub.to_csv(path,index=False)
            breakdown=sub["PERIODTYPE"].value_counts().to_dict() if "PERIODTYPE" in sub else {}
            log.info("%s/%s: %d rows %s",view,region,len(sub),breakdown)
            if snapshot_dir:
                sd=Path(snapshot_dir); sd.mkdir(parents=True,exist_ok=True)
                sub.to_csv(sd/f"{view}_{region}_{pulled_at:%Y%m%d_%H%M}.csv",index=False)


def main():
    p=argparse.ArgumentParser(description="Ashish WP1 – AEMO dashboard snapshot acquisition")
    p.add_argument("--dispatch-dir",required=True); p.add_argument("--predispatch-dir",required=True)
    p.add_argument("--snapshot-dir",default=None); p.add_argument("--regions",nargs="+",default=NEM_REGIONS)
    a=p.parse_args(); fetch_dispatch_and_predispatch(a.regions,a.dispatch_dir,a.predispatch_dir,a.snapshot_dir)

if __name__=="__main__": main()
