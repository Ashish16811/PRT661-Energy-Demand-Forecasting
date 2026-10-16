from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .config import NEM_REGIONS, RAW_DIR, REGIONS, STAGE2_DIR
from .database import ENGINE, bulk_append_frame, init_db, utcnow_naive
from .loader import acquisition_module


def _demand_history_exists() -> bool:
    needed = [RAW_DIR / "price_and_demand" / f"price_and_demand_{r}.csv" for r in NEM_REGIONS]
    needed.append(RAW_DIR / "price_and_demand" / "WA_demand_2022_2026.csv")
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def _weather_history_exists() -> bool:
    needed = []
    for r in REGIONS:
        needed += [RAW_DIR / "weather" / f"weather_daily_{r}.csv",
                   RAW_DIR / "weather" / f"weather_hourly_{r}.csv"]
    return all(p.exists() and p.stat().st_size > 0 for p in needed)


def _holiday_history_exists() -> bool:
    return all(any((RAW_DIR / "public_holidays").glob(f"public_holidays_{r}_*.csv"))
               for r in REGIONS)



def _stage2_data_dir() -> Path | None:
    """Locate the current Stage 2 data directory without relying on machine-specific paths."""
    candidates: list[Path] = []
    pointer = STAGE2_DIR / "CURRENT_STAGE2.txt"
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw:
            p = Path(raw)
            if not p.exists():
                parts = raw.replace("\\", "/").split("/")
                if "stage2" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("stage2")
                    p = STAGE2_DIR.joinpath(*parts[idx + 1:])
            candidates.append(p)
    candidates += [STAGE2_DIR / "current", STAGE2_DIR]
    for root in candidates:
        data_dir = root / "data"
        if (data_dir / "modelling_NSW1_30min.csv").exists():
            return data_dir
    try:
        hit = next(STAGE2_DIR.rglob("modelling_NSW1_30min.csv"))
        return hit.parent
    except StopIteration:
        return None


def _stage2_30min(region: str, data_dir: Path) -> pd.DataFrame:
    p = data_dir / f"modelling_{region}_30min.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing Stage 2 modelling history: {p}")
    d = pd.read_csv(p, usecols=["SETTLEMENTDATE", "TOTALDEMAND"])
    d["SETTLEMENTDATE"] = pd.to_datetime(d["SETTLEMENTDATE"], errors="coerce")
    d["TOTALDEMAND"] = pd.to_numeric(d["TOTALDEMAND"], errors="coerce")
    d = (d.dropna(subset=["SETTLEMENTDATE", "TOTALDEMAND"])
           .drop_duplicates("SETTLEMENTDATE", keep="last")
           .sort_values("SETTLEMENTDATE"))
    return pd.DataFrame({
        "region": region,
        "timestamp": d["SETTLEMENTDATE"],
        "demand_mw": d["TOTALDEMAND"],
        "source": "STAGE2_MODEL_READY_HISTORY",
        "retrieved_at": utcnow_naive(),
    })


def seed_actual_database_from_stage2(clear: bool = True) -> int:
    """Seed historical 30-minute actual demand from the deployed Stage 2 snapshot.

    This is the cloud-friendly bootstrap path: the worker already needs Stage 2 for
    recursive inference, so Railway does not also need the full historical raw archive.
    """
    init_db()
    data_dir = _stage2_data_dir()
    if data_dir is None:
        raise FileNotFoundError(f"No Stage 2 modelling files found under {STAGE2_DIR}")
    if clear:
        with ENGINE.begin() as con:
            con.execute(text("DELETE FROM actual_demand"))
    total = 0
    for region in REGIONS:
        frame = _stage2_30min(region, data_dir)
        total += bulk_append_frame("actual_demand", frame)
        print(f"[Database] {region}: {len(frame):,} Stage 2 rows loaded", flush=True)
    return total


def bootstrap_database_if_empty() -> int:
    """Initialize tables and seed history once; safe to call at every worker start."""
    init_db()
    with ENGINE.connect() as con:
        existing = int(con.execute(text("SELECT COUNT(*) FROM actual_demand")).scalar() or 0)
    if existing:
        print(f"[Bootstrap] actual_demand already contains {existing:,} rows; seed skipped.", flush=True)
        return 0
    data_dir = _stage2_data_dir()
    if data_dir is not None:
        print(f"[Bootstrap] Seeding PostgreSQL from Stage 2 snapshot: {data_dir}", flush=True)
        return seed_actual_database_from_stage2(clear=False)
    if _demand_history_exists():
        print("[Bootstrap] Stage 2 snapshot unavailable; seeding from raw demand history.", flush=True)
        return _seed_actual_database()
    raise RuntimeError(
        "Database is empty and no Stage 2/raw historical demand is available. "
        "Upload the worker asset archive before starting the worker service."
    )

def _nem_30min(region: str) -> pd.DataFrame:
    p = RAW_DIR / "price_and_demand" / f"price_and_demand_{region}.csv"
    d = pd.read_csv(p, usecols=["SETTLEMENTDATE", "TOTALDEMAND"])
    d["SETTLEMENTDATE"] = pd.to_datetime(d["SETTLEMENTDATE"], errors="coerce")
    d["TOTALDEMAND"] = pd.to_numeric(d["TOTALDEMAND"], errors="coerce")
    d = d.dropna().set_index("SETTLEMENTDATE").sort_index()
    # NEM source is 5-minute demand. The production model and portal database
    # operate on the established 30-minute grid, using the same mean aggregation.
    d30 = d["TOTALDEMAND"].resample("30min", closed="right", label="right").mean().dropna().reset_index()
    return pd.DataFrame({
        "region": region,
        "timestamp": d30["SETTLEMENTDATE"],
        "demand_mw": d30["TOTALDEMAND"],
        "source": "AEMO_PRICE_AND_DEMAND_30MIN_MEAN",
        "retrieved_at": utcnow_naive(),
    })


def _wa_30min() -> pd.DataFrame:
    wa = pd.read_csv(RAW_DIR / "price_and_demand" / "WA_demand_2022_2026.csv",
                     usecols=["SETTLEMENTDATE", "TOTALDEMAND"])
    wa["SETTLEMENTDATE"] = pd.to_datetime(wa["SETTLEMENTDATE"], errors="coerce")
    wa["TOTALDEMAND"] = pd.to_numeric(wa["TOTALDEMAND"], errors="coerce")
    wa = wa.dropna().set_index("SETTLEMENTDATE").sort_index()
    wa30 = wa["TOTALDEMAND"].resample("30min", closed="left", label="left").mean().dropna().reset_index()
    return pd.DataFrame({
        "region": "WA",
        "timestamp": wa30["SETTLEMENTDATE"],
        "demand_mw": wa30["TOTALDEMAND"],
        "source": "AEMO_WEM_30MIN_MEAN",
        "retrieved_at": utcnow_naive(),
    })


def _seed_actual_database():
    # Bootstrap is authoritative for the first load, so clear only this table
    # and bulk append each region. Hourly operation later uses normal upserts.
    with ENGINE.begin() as con:
        con.execute(text("DELETE FROM actual_demand"))

    total = 0
    for region in NEM_REGIONS:
        frame = _nem_30min(region)
        total += bulk_append_frame("actual_demand", frame)
        print(f"[Database] {region}: {len(frame):,} 30-minute rows loaded", flush=True)

    wa = _wa_30min()
    total += bulk_append_frame("actual_demand", wa)
    print(f"[Database] WA: {len(wa):,} 30-minute rows loaded", flush=True)
    return total


def main():
    """Run once when the local/cloud environment is first created."""
    init_db()
    acq = acquisition_module()
    today = datetime.today().strftime("%Y-%m-%d")
    for folder in ("price_and_demand", "weather", "public_holidays", "predispatch", "snapshots"):
        (RAW_DIR / folder).mkdir(parents=True, exist_ok=True)

    if _demand_history_exists():
        print("[Bootstrap] Existing demand history found - reusing downloaded files.", flush=True)
    else:
        acq.fetch_price_and_demand("2022-01-01", today, NEM_REGIONS,
                                   str(RAW_DIR / "price_and_demand"))
        acq.fetch_wa_demand("2022-01-01", today, str(RAW_DIR / "price_and_demand"))

    if _weather_history_exists():
        print("[Bootstrap] Existing weather history found - reusing downloaded files.", flush=True)
    else:
        acq.fetch_weather(REGIONS, str(RAW_DIR / "weather"), "2022-01-01", today, hourly=True)

    if _holiday_history_exists():
        print("[Bootstrap] Existing holiday calendars found - reusing downloaded files.", flush=True)
    else:
        acq.fetch_public_holidays("2022-01-01", today, REGIONS,
                                  str(RAW_DIR / "public_holidays"))

    try:
        acq.fetch_predispatch(NEM_REGIONS, str(RAW_DIR / "predispatch"),
                              str(RAW_DIR / "snapshots"))
    except Exception as exc:
        print(f"Predispatch benchmark warning during bootstrap: {exc}")

    print("[Bootstrap] Loading 30-minute demand history into PostgreSQL...", flush=True)
    if _stage2_data_dir() is not None:
        rows = seed_actual_database_from_stage2(clear=True)
    else:
        rows = _seed_actual_database()
    print(f"Bootstrap complete; seeded {rows:,} 30-minute actual-demand rows.", flush=True)
    print("Next: python -m pipeline.monthly_retrain")
    print("Then: python -m pipeline.hourly_job")


if __name__ == "__main__":
    main()
