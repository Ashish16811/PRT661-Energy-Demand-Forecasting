# PRT661 Australian Electricity Demand Forecasting Portal

Public Dash portal for six Australian electricity-demand regions using the final PRT661 forecasting pipeline.

## Final production model mapping

| Region | Live production model |
|---|---|
| NSW1 | RandomForest_A |
| QLD1 | RandomForest_A |
| VIC1 | RandomForest_A |
| SA1 | RandomForest_B |
| TAS1 | RandomForest_A |
| WA | RandomForest_B |

Historical A/B, XGBoost, SARIMA/SARIMAX, naive, MLP and GRU results remain available as model-comparison evidence. They are not alternate live-serving models.

## Portal views

1. **Dashboard** — current demand, next interval, forecast outlook and operational status.
2. **Forecast Performance** — live verification and historical model comparison.
3. **AEMO Comparison** — project forecast versus AEMO NEM pre-dispatch where applicable.
4. **Regional Analysis** — regional demand history and analytical context.

## Forecast-horizon interpretation

Model B uses genuine live forecast weather only for the available near-term weather horizon (currently seven days). Beyond that, the four-month demand outlook uses forecast-safe climatology, calendar/history features and recursive demand lags. The long horizon is therefore an outlook, not four months of known future weather.

## Local run

```bash
pip install -r requirements.txt
python -m pytest -q
python -m pipeline.model_evidence validate-bundles
python -m pipeline.operational_job
python -m app.app
```

Local portal: `http://127.0.0.1:8050`

Validated local baseline before deployment: **102 passed, 2 skipped**.

## Public deployment architecture

```text
GitHub repository (code + model evidence only)
                 |
                 v
             Railway
       +---------+---------+
       |                   |
   Web service          Worker service
   Dash/Gunicorn        persistent scheduler
       |                   |
       +------ PostgreSQL --+
                           |
                     Railway Volume
                     /app/storage
                     - six model bundles
                     - Stage 2 runtime data
```

The public web service does not need its own 1.5+ GB copy of the model bundles. Forecast execution is performed by the worker and persisted to PostgreSQL; the web service reads the database and curated model evidence.

## Railway web service

Environment:

```text
PORTAL_ROLE=web
DATABASE_URL=${{Postgres.DATABASE_URL}}
PORTAL_DEFAULT_REGION=NSW1
PORTAL_TIMEZONE=Australia/Adelaide
TZ=UTC
```

Start command:

```bash
gunicorn app.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
```

Healthcheck: `/healthz`

## Railway worker service

Attach a persistent Railway Volume at `/app/storage`.

Environment:

```text
PORTAL_ROLE=worker
DATABASE_URL=${{Postgres.DATABASE_URL}}
DATA_DIR=/app/storage/data
MODEL_DIR=/app/storage/models
WORKER_STORAGE_DIR=/app/storage
WORKER_ASSET_ARCHIVE=/app/storage/PRT661_WORKER_ASSETS.tar.gz
# Optional: WORKER_ASSET_URL=<public release asset URL>
# Optional: WORKER_ASSET_SHA256=<builder output>
DELETE_ASSET_ARCHIVE_AFTER_EXTRACT=1
RUN_STARTUP_JOBS=1
OPERATIONAL_MINUTES=5,35
HOURLY_MINUTE=15
PREDISPATCH_CACHE_MAX_AGE_MINUTES=90
TZ=UTC
```

Start command:

```bash
python -m pipeline.railway_worker
```

The worker automatically:

- expands the one-time uploaded asset archive when the volume is empty;
- verifies the approved six-region production mapping;
- initializes PostgreSQL;
- seeds historical 30-minute demand from the deployed Stage 2 snapshot when the database is empty;
- records model provenance in `model_registry`;
- runs an initial operational and hourly refresh;
- continues the operational job at minutes 05/35 and the long-horizon refresh at minute 15 each hour.

## Build the one-time worker assets

Run from the complete local project that still contains `models/` and `data/stage2/`:

```bash
python tools/build_worker_assets.py
```

This creates:

```text
PRT661_WORKER_ASSETS.tar.gz
```

The archive contains only:

- the six selected production bundles;
- portable `CURRENT.txt` pointers;
- the Stage 2 runtime snapshot required by recursive inference;
- a lightweight provenance manifest.

It deliberately excludes the full raw archive, local SQLite database and non-production model bundles.

See `DEPLOYMENT_RAILWAY.md` for the full deployment sequence.
