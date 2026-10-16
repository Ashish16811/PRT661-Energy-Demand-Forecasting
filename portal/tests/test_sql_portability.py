"""Dialect helpers: PostgreSQL SQL must stay equivalent to v1; SQLite gets working equivalents."""
from pipeline import database as db


def test_postgres_fragments_equal_v1(monkeypatch):
    monkeypatch.setattr(db, "IS_SQLITE", False)
    assert db.sql_year("timestamp") == "EXTRACT(YEAR FROM timestamp)::int"
    assert db.sql_month("timestamp") == "EXTRACT(MONTH FROM timestamp)::int"
    assert db.sql_month_start("timestamp") == "DATE_TRUNC('month', timestamp)"
    assert db.sql_minutes_between("a", "b") == "(EXTRACT(EPOCH FROM (a - b)) / 60.0)"
    assert "EXTRACT(HOUR FROM timestamp)::int * 2" in db.sql_half_hour("timestamp")


def test_sqlite_fragments_execute():
    import pandas as pd
    from sqlalchemy import text
    q = (f"SELECT {db.sql_year(':t')} AS y, {db.sql_month(':t')} AS m, {db.sql_half_hour(':t')} AS hh, "
         f"{db.sql_month_start(':t')} AS ms, {db.sql_minutes_between(':u', ':t')} AS mins")
    with db.ENGINE.connect() as con:
        r = con.execute(text(q), {"t": pd.Timestamp("2026-09-30 13:30"), "u": pd.Timestamp("2026-09-30 14:00")}).one()
    assert (r.y, r.m, r.hh) == (2026, 9, 27) and r.ms.startswith("2026-09-01") and abs(r.mins - 30) < 1e-6
