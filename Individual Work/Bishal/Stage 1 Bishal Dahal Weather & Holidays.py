"""
PRT661 – Data Science Practice
WP1 Stage 1 contribution module – Bishal Dahal (S388095)
Primary scope: external drivers – historical weather, rolling seven-day forecast
weather, forecast-vintage archiving and state public-holiday calendars.

Architecture evolution reflected here:
- Assessment 1 proposed BOM Climate Data Online.
- Early BOM downloads were not reliably usable in the project pipeline.
- The final reproducible implementation uses Open-Meteo ERA5 reanalysis for
  historical weather and ECMWF IFS HRES through Open-Meteo for the next seven
  full local calendar days, with a logged generic Open-Meteo fallback.
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

log=logging.getLogger("bishal_weather_holidays")
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
OPEN_METEO_ARCHIVE="https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_ECMWF="https://api.open-meteo.com/v1/ecmwf"
OPEN_METEO_FORECAST="https://api.open-meteo.com/v1/forecast"
ROLLING_FORECAST_DAYS=7; ERA5_LAG_DAYS=5
WEATHER_SITES={
 "NSW1":{"lat":-33.8688,"lon":151.2093,"tz":"Australia/Sydney","city":"Sydney","state":"NSW"},
 "QLD1":{"lat":-27.4698,"lon":153.0251,"tz":"Australia/Brisbane","city":"Brisbane","state":"QLD"},
 "VIC1":{"lat":-37.8136,"lon":144.9631,"tz":"Australia/Melbourne","city":"Melbourne","state":"VIC"},
 "SA1":{"lat":-34.9285,"lon":138.6007,"tz":"Australia/Adelaide","city":"Adelaide","state":"SA"},
 "TAS1":{"lat":-42.8821,"lon":147.3272,"tz":"Australia/Hobart","city":"Hobart","state":"TAS"},
 "WA":{"lat":-31.9523,"lon":115.8613,"tz":"Australia/Perth","city":"Perth","state":"WA"},
}
DAILY_VARS=["temperature_2m_max","temperature_2m_min","temperature_2m_mean","precipitation_sum"]
HOURLY_VARS=["temperature_2m","relative_humidity_2m","precipitation"]


def _block(payload,key,expected):
    b=payload.get(key)
    if not b or "time" not in b: raise ValueError(f"No {key} block")
    n=len(b["time"])
    if any(v not in b or len(b[v])!=n for v in expected): raise ValueError(f"Malformed {key} arrays")
    return pd.DataFrame({"time":b["time"],**{v:b[v] for v in expected}})


def fetch_weather_history(regions,output_dir,start,end):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    session=requests.Session(); session.headers.update({"User-Agent":UA,"Accept":"application/json"})
    cutoff=(pd.Timestamp.today().normalize()-pd.Timedelta(days=ERA5_LAG_DAYS)).date()
    end_req=min(pd.Timestamp(end).date(),cutoff)
    for region in regions:
        s=WEATHER_SITES[region]
        params={"latitude":s["lat"],"longitude":s["lon"],"start_date":str(pd.Timestamp(start).date()),
                "end_date":str(end_req),"timezone":s["tz"],"daily":",".join(DAILY_VARS),"hourly":",".join(HOURLY_VARS)}
        r=session.get(OPEN_METEO_ARCHIVE,params=params,timeout=120); r.raise_for_status(); payload=r.json()
        day=_block(payload,"daily",DAILY_VARS).rename(columns={"time":"DATE","temperature_2m_max":"temp_max_c",
            "temperature_2m_min":"temp_min_c","temperature_2m_mean":"temp_mean_c","precipitation_sum":"rainfall_mm"})
        day["DATE"]=pd.to_datetime(day.DATE); day.insert(1,"REGION",region); day["LOCATION"]=s["city"]; day["TIMEZONE"]=s["tz"]
        day["SOURCE_TYPE"]="OBSERVED_REANALYSIS"; day["SOURCE"]="Open-Meteo ERA5"
        hour=_block(payload,"hourly",HOURLY_VARS).rename(columns={"time":"TIMESTAMP","temperature_2m":"temp_c",
            "relative_humidity_2m":"humidity_pct","precipitation":"rainfall_mm"})
        hour["TIMESTAMP"]=pd.to_datetime(hour.TIMESTAMP); hour.insert(1,"REGION",region); hour["LOCATION"]=s["city"]; hour["TIMEZONE"]=s["tz"]
        hour["SOURCE_TYPE"]="OBSERVED_REANALYSIS"; hour["SOURCE"]="Open-Meteo ERA5"
        day.to_csv(out/f"weather_daily_{region}.csv",index=False); hour.to_csv(out/f"weather_hourly_{region}.csv",index=False)


def _request_forecast(session,site,params):
    attempts=[(OPEN_METEO_ECMWF,"Open-Meteo ECMWF Forecast API","ECMWF IFS HRES"),
              (OPEN_METEO_FORECAST,"Open-Meteo Forecast API","Open-Meteo best-match forecast")]
    failures=[]
    for endpoint,source,model in attempts:
        try:
            r=session.get(endpoint,params=params,timeout=120); r.raise_for_status(); payload=r.json()
            if "error" in payload: raise ValueError(payload.get("reason","API error"))
            return payload,source,model,endpoint
        except Exception as exc: failures.append(f"{model}: {exc}")
    raise RuntimeError("; ".join(failures))


def _daily_from_hourly(hourly):
    w=hourly.copy(); w["DATE"]=w.TIMESTAMP.dt.normalize(); g=w.groupby("DATE",as_index=False)
    d=g.agg(temp_max_c=("temp_c","max"),temp_min_c=("temp_c","min"),temp_mean_c=("temp_c","mean"),humidity_mean_pct=("humidity_pct","mean"))
    return d.merge(g["rainfall_mm"].agg(lambda s:s.sum(min_count=1)),on="DATE",how="left")


def fetch_weather_forecast(regions,output_dir,forecast_origin=None,forecast_days=7):
    origin=pd.Timestamp(forecast_origin or datetime.today().strftime("%Y-%m-%d")).normalize()
    start=(origin+pd.Timedelta(days=1)).date(); end=(origin+pd.Timedelta(days=forecast_days)).date()
    retrieved=pd.Timestamp.now(tz="UTC"); stamp=retrieved.strftime("%Y%m%d_%H%M%SZ")
    root=Path(output_dir)/"forecast"; latest=root/"latest"; archive=root/"archive"/stamp
    latest.mkdir(parents=True,exist_ok=True); archive.mkdir(parents=True,exist_ok=True)
    session=requests.Session(); session.headers.update({"User-Agent":UA,"Accept":"application/json"})
    expected=set(pd.date_range(start,end,freq="D").date); manifest=[]
    for region in regions:
        s=WEATHER_SITES[region]
        params={"latitude":s["lat"],"longitude":s["lon"],"timezone":s["tz"],"start_date":str(start),"end_date":str(end),"hourly":",".join(HOURLY_VARS)}
        payload,source,model,endpoint=_request_forecast(session,s,params)
        hr=_block(payload,"hourly",HOURLY_VARS).rename(columns={"time":"TIMESTAMP","temperature_2m":"temp_c","relative_humidity_2m":"humidity_pct","precipitation":"rainfall_mm"})
        hr.TIMESTAMP=pd.to_datetime(hr.TIMESTAMP); hr=hr[hr.TIMESTAMP.dt.date.isin(expected)].copy()
        if set(hr.TIMESTAMP.dt.date)!=expected: raise ValueError(f"{region}: incomplete seven-day forecast")
        hr.insert(1,"REGION",region); hr["LOCATION"]=s["city"]; hr["TIMEZONE"]=s["tz"]; hr["SOURCE_TYPE"]="FORECAST_7DAY"; hr["SOURCE"]=source; hr["MODEL"]=model
        hr["RETRIEVED_AT_UTC"]=retrieved.isoformat(); hr["FORECAST_ORIGIN_DATE"]=origin.date().isoformat(); hr["TARGET_DATE"]=hr.TIMESTAMP.dt.date.astype(str)
        hr["LEAD_DAY"]=(hr.TIMESTAMP.dt.normalize()-origin).dt.days.astype("Int64")
        day=_daily_from_hourly(hr); day.insert(1,"REGION",region); day["LOCATION"]=s["city"]; day["TIMEZONE"]=s["tz"]; day["SOURCE_TYPE"]="FORECAST_7DAY"; day["SOURCE"]=source; day["MODEL"]=model
        day["RETRIEVED_AT_UTC"]=retrieved.isoformat(); day["FORECAST_ORIGIN_DATE"]=origin.date().isoformat(); day["LEAD_DAY"]=(day.DATE-origin).dt.days.astype("Int64")
        for folder in (latest,archive):
            hr.to_csv(folder/f"weather_forecast_hourly_{region}.csv",index=False); day.to_csv(folder/f"weather_forecast_daily_{region}.csv",index=False)
        manifest.append({"REGION":region,"LOCATION":s["city"],"TIMEZONE":s["tz"],"FORECAST_ORIGIN_DATE":origin.date().isoformat(),
                         "TARGET_START":str(start),"TARGET_END":str(end),"HOURLY_ROWS":len(hr),"DAILY_ROWS":len(day),"SOURCE":source,"MODEL":model,
                         "ENDPOINT":endpoint,"RETRIEVED_AT_UTC":retrieved.isoformat(),"ARCHIVE_RUN":stamp})
    m=pd.DataFrame(manifest); m.to_csv(latest/"weather_forecast_manifest.csv",index=False); m.to_csv(archive/"weather_forecast_manifest.csv",index=False)
    run={"forecast_origin_date":origin.date().isoformat(),"target_start":str(start),"target_end":str(end),"forecast_days":forecast_days,
         "retrieved_at_utc":retrieved.isoformat(),"archive_run":stamp,"primary_model":"ECMWF IFS HRES","fallback":"Open-Meteo best-match forecast"}
    for folder in (latest,archive): (folder/"weather_forecast_run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")


def holiday_calendar_end(end_date):
    # Stage 1 must know holidays for the remaining forecast year, not only to the demand cutoff.
    end=pd.Timestamp(end_date); return pd.Timestamp(year=end.year,month=12,day=31)


def fetch_public_holidays(start_date,end_date,regions,output_dir):
    try: import holidays
    except ImportError as exc: raise RuntimeError("Install 'holidays' package") from exc
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); end=holiday_calendar_end(end_date)
    dates=pd.date_range(start_date,end,freq="D"); years=range(pd.Timestamp(start_date).year,end.year+1)
    for region in regions:
        s=WEATHER_SITES[region]; cal=holidays.Australia(subdiv=s["state"],years=years)
        df=pd.DataFrame({"date":dates,"state":s["state"]}); df["is_public_holiday"]=df.date.dt.date.isin(cal).astype(int)
        df["holiday_name"]=df.date.dt.date.map(lambda d:cal.get(d,"")); df.to_csv(out/f"public_holidays_{region}_{s['state']}.csv",index=False)


def main():
    p=argparse.ArgumentParser(description="Bishal WP1 – weather and holiday acquisition")
    p.add_argument("--start",default="2022-01-01"); p.add_argument("--end",default=datetime.today().strftime("%Y-%m-%d")); p.add_argument("--forecast-origin",default=datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--regions",nargs="+",default=list(WEATHER_SITES)); p.add_argument("--weather-dir",required=True); p.add_argument("--holiday-dir",required=True)
    a=p.parse_args(); fetch_weather_history(a.regions,a.weather_dir,a.start,a.end); fetch_weather_forecast(a.regions,a.weather_dir,a.forecast_origin); fetch_public_holidays(a.start,a.end,a.regions,a.holiday_dir)

if __name__=="__main__": main()

# Work progress: 19 August 2026 - weather and public holiday acquisition/validation
