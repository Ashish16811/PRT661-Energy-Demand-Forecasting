# Railway deployment — PRT661 portal

This deployment uses two Python services from the same GitHub repository plus one Railway PostgreSQL database. The worker owns the large model/Stage 2 files on a persistent volume; the web service stays lightweight.

## 1. Before GitHub

From the complete local project, confirm:

```powershell
python -m pytest -q
python -m pipeline.model_evidence validate-bundles
python -m pipeline.operational_job
```

Expected production mapping:

```text
NSW1 RandomForest_A
QLD1 RandomForest_A
VIC1 RandomForest_A
SA1  RandomForest_B
TAS1 RandomForest_A
WA   RandomForest_B
```

Then build the one-time worker asset archive:

```powershell
python tools\build_worker_assets.py
```

Do **not** commit `models/`, `data/stage2/`, `data/raw/`, the local SQLite database, or the generated asset archive to GitHub.

## 2. Push the code repository to GitHub

The repository should contain the application code, tests, requirements and `data/model_evidence/`. The supplied `.gitignore` protects heavy runtime assets.

## 3. Create Railway project + PostgreSQL

Create a Railway project, then add PostgreSQL. Keep PostgreSQL private; both Python services will use its Railway reference variable.

## 4. Create WEB service

Deploy the GitHub repository as a Railway service.

Variables:

```text
PORTAL_ROLE=web
DATABASE_URL=${{Postgres.DATABASE_URL}}
PORTAL_DEFAULT_REGION=NSW1
PORTAL_TIMEZONE=Australia/Adelaide
TZ=UTC
```

Start command:

```text
gunicorn app.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
```

Healthcheck path:

```text
/healthz
```

Generate a public domain only after the web deployment is healthy.

## 5. Create WORKER service

Create a second service from the same GitHub repository.

Variables:

```text
PORTAL_ROLE=worker
DATABASE_URL=${{Postgres.DATABASE_URL}}
DATA_DIR=/app/storage/data
MODEL_DIR=/app/storage/models
WORKER_STORAGE_DIR=/app/storage
WORKER_ASSET_ARCHIVE=/app/storage/PRT661_WORKER_ASSETS.tar.gz
# Optional alternative to CLI upload:
# WORKER_ASSET_URL=<public GitHub Release asset URL>
# WORKER_ASSET_SHA256=<SHA256 printed by the asset builder>
DELETE_ASSET_ARCHIVE_AFTER_EXTRACT=1
RUN_STARTUP_JOBS=1
OPERATIONAL_MINUTES=5,35
HOURLY_MINUTE=15
PREDISPATCH_CACHE_MAX_AGE_MINUTES=90
TZ=UTC
```

Start command:

```text
python -m pipeline.railway_worker
```

Attach one Railway Volume to this service at:

```text
/app/storage
```

A 5 GB volume is appropriate for the current six production bundles plus Stage 2 runtime data and live raw/cache files.

## 6. Make the worker asset archive available

**Option A — Railway Volume CLI (private and direct):** install/login to the Railway CLI, link the worker service, then upload:

```text
railway volume files upload ./PRT661_WORKER_ASSETS.tar.gz /PRT661_WORKER_ASSETS.tar.gz
```

**Option B — GitHub Release asset (often easier for a public academic repository):** create a GitHub Release, attach `PRT661_WORKER_ASSETS.tar.gz`, then set the worker variables `WORKER_ASSET_URL` to the direct public release-asset URL and `WORKER_ASSET_SHA256` to the hash printed by the builder. GitHub Release assets must be under 2 GiB each, so confirm the generated archive size first.

The first worker start obtains the archive, extracts it to `/app/storage/models` and `/app/storage/data`, validates the mapping, seeds PostgreSQL if required, then deletes the archive to recover volume space.

If the worker was already running before the archive became available, redeploy/restart the worker.

## 7. Verify worker logs

Look for:

```text
Approved production mapping validated
actual_demand ... rows loaded
Synchronized 6 production model-registry rows
next_interval_rows=6
```

and six forecast streams only:

```text
NSW1/RandomForest_A
QLD1/RandomForest_A
VIC1/RandomForest_A
SA1/RandomForest_B
TAS1/RandomForest_A
WA/RandomForest_B
```

## 8. Verify the public site

Check:

- Dashboard
- Forecast Performance
- AEMO Comparison
- Regional Analysis
- current regional model labels
- fresh/current timestamps
- live verification accumulation

## 9. First-release policy

Automatic monthly retraining is intentionally not enabled in the first public release. The six validated production bundles remain frozen while live data/verification accumulates. Retraining can be added later with explicit artifact promotion/storage controls.
