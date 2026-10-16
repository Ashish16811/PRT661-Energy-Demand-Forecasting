from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATABASE_URL

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
IS_SQLITE = ENGINE.dialect.name == "sqlite"

if IS_SQLITE:
    # pandas.to_sql stores SQLite timestamps as 'YYYY-MM-DD HH:MM:SS.ffffff' text.
    # Bind datetime parameters in the same format so equality and range filters
    # (e.g. forecast_origin = :o) behave as they do on PostgreSQL. (v3)
    import sqlite3
    from datetime import datetime as _dt

    def _adapt_ts(value):
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")

    sqlite3.register_adapter(_dt, _adapt_ts)
    sqlite3.register_adapter(pd.Timestamp, _adapt_ts)


# Dialect-portable SQL fragments (PostgreSQL is the deployment database; SQLite
# is the local/test default in .env.example). v3.
def sql_year(col: str) -> str:
    return f"CAST(strftime('%Y', {col}) AS INTEGER)" if IS_SQLITE else f"EXTRACT(YEAR FROM {col})::int"


def sql_month(col: str) -> str:
    return f"CAST(strftime('%m', {col}) AS INTEGER)" if IS_SQLITE else f"EXTRACT(MONTH FROM {col})::int"


def sql_half_hour(col: str) -> str:
    if IS_SQLITE:
        return (f"(CAST(strftime('%H', {col}) AS INTEGER) * 2 + "
                f"CASE WHEN CAST(strftime('%M', {col}) AS INTEGER) >= 30 THEN 1 ELSE 0 END)")
    return (f"(EXTRACT(HOUR FROM {col})::int * 2 + "
            f"CASE WHEN EXTRACT(MINUTE FROM {col})::int >= 30 THEN 1 ELSE 0 END)::int")


def sql_month_start(col: str) -> str:
    return f"strftime('%Y-%m-01 00:00:00', {col})" if IS_SQLITE else f"DATE_TRUNC('month', {col})"


def sql_minutes_between(later: str, earlier: str) -> str:
    if IS_SQLITE:
        return f"((julianday({later}) - julianday({earlier})) * 1440.0)"
    return f"(EXTRACT(EPOCH FROM ({later} - {earlier})) / 60.0)"

DDL = [
    """CREATE TABLE IF NOT EXISTS actual_demand (
        region TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        demand_mw REAL NOT NULL,
        source TEXT,
        retrieved_at TIMESTAMP,
        PRIMARY KEY (region, timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS weather_forecast (
        region TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        temp_c REAL,
        humidity_pct REAL,
        rainfall_mm REAL,
        source TEXT,
        retrieved_at TIMESTAMP,
        PRIMARY KEY (region, forecast_origin, target_timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS project_forecast (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        forecast_mw REAL,
        created_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version, forecast_origin, target_timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS aemo_forecast (
        region TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        poe10_mw REAL,
        poe50_mw REAL,
        poe90_mw REAL,
        source TEXT,
        retrieved_at TIMESTAMP,
        PRIMARY KEY (region, forecast_origin, target_timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS forecast_accuracy (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        forecast_mw REAL,
        actual_mw REAL,
        error_mw REAL,
        abs_error_mw REAL,
        squared_error REAL,
        smape_component REAL,
        evaluated_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version, forecast_origin, target_timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS next_interval_forecast (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        forecast_mw REAL NOT NULL,
        lead_minutes REAL,
        created_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version, forecast_origin, target_timestamp)
    )""",
    """CREATE TABLE IF NOT EXISTS next_interval_accuracy (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        forecast_mw REAL NOT NULL,
        actual_mw REAL NOT NULL,
        error_mw REAL NOT NULL,
        abs_error_mw REAL NOT NULL,
        squared_error REAL NOT NULL,
        smape_component REAL,
        evaluated_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version, forecast_origin, target_timestamp)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_next_interval_target
        ON next_interval_forecast (region, model_name, target_timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_next_accuracy_target
        ON next_interval_accuracy (region, model_name, target_timestamp)""",
    """CREATE TABLE IF NOT EXISTS historical_replay_accuracy (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        forecast_origin TIMESTAMP NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        forecast_mw REAL NOT NULL,
        actual_mw REAL NOT NULL,
        error_mw REAL NOT NULL,
        abs_error_mw REAL NOT NULL,
        squared_error REAL NOT NULL,
        smape_component REAL,
        lead_minutes REAL,
        replay_generated_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version, target_timestamp)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_replay_accuracy_target
        ON historical_replay_accuracy (region, model_name, target_timestamp)""",
    """CREATE TABLE IF NOT EXISTS model_registry (
        region TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        feature_set TEXT,
        training_start TIMESTAMP,
        training_end TIMESTAMP,
        model_path TEXT NOT NULL,
        stage2_snapshot TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP,
        PRIMARY KEY (region, model_name, model_version)
    )""",
    """CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        job_type TEXT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        status TEXT,
        detail TEXT
    )""",
]


def utcnow_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def init_db():
    with ENGINE.begin() as con:
        for stmt in DDL:
            con.execute(text(stmt))


def bulk_append_frame(table: str, frame: pd.DataFrame, chunksize: int = 5000):
    """Fast append used for one-time bootstrap after the target table is cleared."""
    if frame is None or frame.empty:
        return 0
    frame.to_sql(table, ENGINE, if_exists="append", index=False, chunksize=chunksize)
    return len(frame)


def upsert_frame(table: str, frame: pd.DataFrame, keys: list[str]):
    """Set-based PostgreSQL upsert.

    The original prototype deleted one database key at a time. That is fine for
    a handful of rows but becomes extremely slow for multi-year demand history.
    This version loads a staging table and lets PostgreSQL merge the rows in one
    set operation.
    """
    if frame is None or frame.empty:
        return 0

    out = frame.copy().drop_duplicates(subset=keys, keep="last")
    staging = f"_stg_{table}_{uuid4().hex[:10]}"
    out.to_sql(staging, ENGINE, if_exists="replace", index=False, chunksize=5000)

    cols = list(out.columns)
    qcols = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{k}"' for k in keys)
    nonkeys = [c for c in cols if c not in keys]
    if nonkeys:
        update = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in nonkeys)
        action = f"DO UPDATE SET {update}"
    else:
        action = "DO NOTHING"

    try:
        with ENGINE.begin() as con:
            con.execute(text(
                f'INSERT INTO "{table}" ({qcols}) '
                # "WHERE true" is required by SQLite's parser for INSERT ... SELECT ...
                # ON CONFLICT (otherwise ON is read as a join); harmless on PostgreSQL.
                f'SELECT {qcols} FROM "{staging}" WHERE true '
                f'ON CONFLICT ({conflict}) {action}'
            ))
    finally:
        with ENGINE.begin() as con:
            con.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))
    return len(out)


def read_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql(text(sql), ENGINE, params=params or {})


def log_run(run_id: str, job_type: str, started_at, status: str, detail: str = ""):
    row = pd.DataFrame([{
        "run_id": run_id,
        "job_type": job_type,
        "started_at": started_at,
        "finished_at": utcnow_naive(),
        "status": status,
        "detail": detail[:4000],
    }])
    upsert_frame("pipeline_runs", row, ["run_id"])
