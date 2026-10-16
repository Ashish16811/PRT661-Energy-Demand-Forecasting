from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .bootstrap import _stage2_data_dir, bootstrap_database_if_empty
from .config import MODEL_DIR, REGIONS, STAGE2_DIR
from .database import init_db, upsert_frame, utcnow_naive
from .hourly_job import main as hourly_main
from .model_evidence import clear_caches, resolve_production_config, resolve_pointer
from .operational_job import main as operational_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("railway_worker")

EXPECTED = {
    "NSW1": "RandomForest_A",
    "QLD1": "RandomForest_A",
    "VIC1": "RandomForest_A",
    "SA1": "RandomForest_B",
    "TAS1": "RandomForest_A",
    "WA": "RandomForest_B",
}


def _storage_dir() -> Path:
    return Path(os.getenv("WORKER_STORAGE_DIR", os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/app/storage")))


def _assets_present() -> bool:
    if _stage2_data_dir() is None:
        return False
    for region, model in EXPECTED.items():
        pointer = Path(MODEL_DIR) / region / model / "CURRENT.txt"
        if not pointer.exists() or resolve_pointer(pointer) is None:
            return False
    return True


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download_asset_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    log.info("Downloading worker asset archive to %s", destination)
    with requests.get(url, stream=True, timeout=(30, 300), headers={"User-Agent": "PRT661-Railway-Worker/1.0"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(destination)
    expected = os.getenv("WORKER_ASSET_SHA256", "").strip().lower()
    if expected:
        actual = _sha256(destination)
        if actual != expected:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Worker asset SHA256 mismatch: expected {expected}, got {actual}")
        log.info("Worker asset SHA256 verified")


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise RuntimeError(f"Unsafe path in worker asset archive: {member.name}")
        # Python 3.13 supports the data filter, which blocks special-file surprises.
        tf.extractall(target, filter="data")


def ensure_worker_assets() -> None:
    """Expand the one-time Railway volume archive when the volume is still empty."""
    storage = _storage_dir()
    storage.mkdir(parents=True, exist_ok=True)
    if _assets_present():
        log.info("Worker model/Stage 2 assets are already present on %s", storage)
        return

    archive = Path(os.getenv("WORKER_ASSET_ARCHIVE", storage / "PRT661_WORKER_ASSETS.tar.gz"))
    if not archive.exists():
        url = os.getenv("WORKER_ASSET_URL", "").strip()
        if url:
            _download_asset_archive(url, archive)
        else:
            raise FileNotFoundError(
                f"Worker assets are not present and archive was not found at {archive}. "
                "Either upload PRT661_WORKER_ASSETS.tar.gz to the Railway worker volume "
                "or configure WORKER_ASSET_URL for a one-time download."
            )

    log.info("Extracting worker assets from %s ...", archive)
    _safe_extract(archive, storage)
    clear_caches()
    if not _assets_present():
        raise RuntimeError("Worker asset extraction finished, but required models/Stage 2 files are still missing")

    if os.getenv("DELETE_ASSET_ARCHIVE_AFTER_EXTRACT", "1").lower() not in {"0", "false", "no"}:
        try:
            archive.unlink()
            log.info("Deleted uploaded asset archive after successful extraction to free volume space")
        except OSError as exc:
            log.warning("Could not delete asset archive after extraction: %s", exc)


def validate_production_mapping() -> None:
    """Fail fast if the worker does not have exactly the approved six production bundles."""
    clear_caches()
    cfg = resolve_production_config()
    problems = []
    for region in REGIONS:
        expected = EXPECTED[region]
        st = cfg.status[region]
        if st.selected != expected or st.served != expected or st.state != "SERVING_SELECTED":
            problems.append(
                f"{region}: expected={expected}, selected={st.selected}, served={st.served}, "
                f"state={st.state}, available={st.available}"
            )
    if problems:
        raise RuntimeError("Production mapping validation failed: " + " | ".join(problems))
    log.info("Approved production mapping validated: %s", EXPECTED)


def sync_model_registry() -> None:
    """Seed lightweight model provenance in PostgreSQL from the asset manifest."""
    manifest_path = _storage_dir() / "worker_asset_manifest.json"
    generated = utcnow_naive()
    meta = {}
    if manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated = pd.to_datetime(meta.get("generated_at"), errors="coerce")
            if pd.isna(generated):
                generated = utcnow_naive()
            elif getattr(generated, "tzinfo", None) is not None:
                generated = generated.tz_convert("UTC").tz_localize(None)
        except Exception as exc:
            log.warning("Could not read worker asset manifest: %s", exc)
            meta = {}
            generated = utcnow_naive()

    by_key = {(x.get("region"), x.get("model_name")): x for x in meta.get("models", [])}
    rows = []
    for region, model in EXPECTED.items():
        pointer = Path(MODEL_DIR) / region / model / "CURRENT.txt"
        bundle_path = resolve_pointer(pointer)
        if bundle_path is None:
            raise FileNotFoundError(f"Cannot sync registry: unresolved pointer {pointer}")
        item = by_key.get((region, model), {})
        rows.append({
            "region": region,
            "model_name": model,
            "model_version": item.get("model_version") or bundle_path.parent.name,
            "feature_set": item.get("feature_set") or ("B" if model.endswith("_B") else "A"),
            "training_start": pd.to_datetime(item.get("training_start"), errors="coerce"),
            "training_end": pd.to_datetime(item.get("training_end"), errors="coerce"),
            "model_path": str(bundle_path),
            "stage2_snapshot": item.get("stage2_root") or str(STAGE2_DIR),
            "status": "PRODUCTION",
            "created_at": generated,
        })
    upsert_frame("model_registry", pd.DataFrame(rows), ["region", "model_name", "model_version"])
    log.info("Synchronized %d production model-registry rows", len(rows))


def _parse_operational_minutes() -> tuple[int, ...]:
    raw = os.getenv("OPERATIONAL_MINUTES", "5,35")
    vals = []
    for part in raw.split(","):
        try:
            minute = int(part.strip())
        except ValueError:
            continue
        if 0 <= minute <= 59:
            vals.append(minute)
    return tuple(sorted(set(vals))) or (5, 35)


def _next_minute_slot(now: datetime, minutes: tuple[int, ...]) -> datetime:
    now = now.astimezone(timezone.utc)
    for minute in minutes:
        candidate = now.replace(minute=minute, second=0, microsecond=0)
        if candidate > now:
            return candidate
    return (now + timedelta(hours=1)).replace(minute=minutes[0], second=0, microsecond=0)


def _run(name: str, fn) -> None:
    try:
        log.info("Starting scheduled %s", name)
        fn()
        log.info("Completed scheduled %s", name)
    except Exception:
        # A short upstream outage should not kill the persistent scheduler. The job
        # itself records FAILED in pipeline_runs; the next scheduled cycle retries.
        log.exception("Scheduled %s failed; worker will continue", name)


def scheduler_loop() -> None:
    op_minutes = _parse_operational_minutes()
    hourly_minute = int(os.getenv("HOURLY_MINUTE", "15"))
    if not 0 <= hourly_minute <= 59:
        hourly_minute = 15

    now = datetime.now(timezone.utc)
    next_op = _next_minute_slot(now, op_minutes)
    next_hourly = _next_minute_slot(now, (hourly_minute,))
    log.info("Scheduler active (UTC): operational minutes=%s, hourly minute=%02d", op_minutes, hourly_minute)

    while True:
        target = min(next_op, next_hourly)
        delay = (target - datetime.now(timezone.utc)).total_seconds()
        if delay > 0:
            time.sleep(min(delay, 60.0))
            continue

        now = datetime.now(timezone.utc)
        if next_op <= now:
            _run("operational_job", operational_main)
            next_op = _next_minute_slot(datetime.now(timezone.utc), op_minutes)
        now = datetime.now(timezone.utc)
        if next_hourly <= now:
            _run("hourly_job", hourly_main)
            next_hourly = _next_minute_slot(datetime.now(timezone.utc), (hourly_minute,))


def main() -> None:
    if os.getenv("PORTAL_ROLE", "worker").lower() == "web":
        raise RuntimeError("pipeline.railway_worker must not run with PORTAL_ROLE=web")

    ensure_worker_assets()
    init_db()
    validate_production_mapping()
    bootstrap_database_if_empty()
    sync_model_registry()

    if os.getenv("RUN_STARTUP_JOBS", "1").lower() not in {"0", "false", "no"}:
        _run("startup operational_job", operational_main)
        _run("startup hourly_job", hourly_main)

    scheduler_loop()


if __name__ == "__main__":
    main()
