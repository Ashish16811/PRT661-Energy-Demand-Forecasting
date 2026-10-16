from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODEL_BY_REGION
from .database import init_db, upsert_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("import_monthly_frozen")

REGION_ALIASES = {"NSW": "NSW1", "QLD": "QLD1", "VIC": "VIC1", "SA": "SA1", "TAS": "TAS1", "WEM": "WA"}


def _normalise_region(value: str) -> str:
    value = str(value).strip().upper()
    return REGION_ALIASES.get(value, value)


def _load_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    region_col = cols.get("region")
    ts_col = cols.get("target_timestamp") or cols.get("settlementdate")
    forecast_col = cols.get("forecast_mw") or cols.get("frozen_forecast_mw")
    origin_col = cols.get("forecast_origin")
    model_col = cols.get("selected_model") or cols.get("model_name") or cols.get("model")
    version_col = cols.get("model_version")

    if not ts_col or not forecast_col:
        raise ValueError(
            f"{path.name}: expected target_timestamp/SETTLEMENTDATE and forecast_MW columns; "
            f"found {list(df.columns)}"
        )

    if region_col:
        regions = df[region_col].map(_normalise_region)
    else:
        # Stage-3 frozen files are normally named forecast_frozen_<REGION>.csv.
        token = path.stem.replace("forecast_frozen_", "").split("_")[-1]
        regions = pd.Series([_normalise_region(token)] * len(df), index=df.index)

    target = pd.to_datetime(df[ts_col], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")

    if origin_col:
        origin = pd.to_datetime(df[origin_col], errors="coerce")
    else:
        inferred = target.dropna().min()
        if pd.isna(inferred):
            raise ValueError(f"{path.name}: cannot infer forecast month")
        inferred = pd.Timestamp(inferred).normalize().replace(day=1)
        origin = pd.Series([inferred] * len(df), index=df.index)

    if model_col:
        model = df[model_col].astype(str)
    else:
        model = regions.map(lambda r: DEFAULT_MODEL_BY_REGION.get(r, "RandomForest_A"))

    if version_col:
        version = df[version_col].astype(str)
    else:
        first_origin = pd.to_datetime(origin, errors="coerce").dropna().min()
        stamp = first_origin.strftime("%Y%m") if pd.notna(first_origin) else "UNKNOWN"
        version = pd.Series([f"ARCHIVE_FROZEN_{stamp}"] * len(df), index=df.index)

    out = pd.DataFrame({
        "region": regions,
        "model_name": model,
        "model_version": version,
        "forecast_origin": pd.to_datetime(origin, errors="coerce"),
        "target_timestamp": target,
        "forecast_mw": forecast,
        "created_at": pd.to_datetime(origin, errors="coerce"),
    }).dropna(subset=["region", "forecast_origin", "target_timestamp", "forecast_mw"])

    # A month-start verification vintage must be a genuine forecast made at the
    # beginning of the month. Do not silently relabel later forecasts as day-1.
    bad = out["forecast_origin"].dt.day.ne(1)
    if bad.any():
        examples = out.loc[bad, "forecast_origin"].drop_duplicates().head(3).tolist()
        raise ValueError(
            f"{path.name}: forecast_origin is not on day 1 ({examples}). "
            "Use the original Stage-3 frozen beginning-of-month forecast file."
        )

    return out


def import_path(source: Path) -> int:
    if source.is_dir():
        files = sorted(source.glob("forecast_frozen_*.csv"))
        if not files:
            files = sorted(source.glob("*.csv"))
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"No CSV files found at {source}")

    frames = []
    for path in files:
        frame = _load_file(path)
        if not frame.empty:
            frames.append(frame)
            log.info("Prepared %s rows from %s", len(frame), path.name)
    if not frames:
        return 0

    out = pd.concat(frames, ignore_index=True)
    n = upsert_frame(
        "project_forecast",
        out,
        ["region", "model_name", "model_version", "forecast_origin", "target_timestamp"],
    )
    log.info("Imported %s frozen monthly forecast rows into project_forecast", n)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Import genuine Stage-3 frozen beginning-of-month forecasts into the live portal database."
    )
    parser.add_argument("source", help="CSV file or folder containing forecast_frozen_<REGION>.csv files")
    args = parser.parse_args()
    init_db()
    import_path(Path(args.source))


if __name__ == "__main__":
    main()
