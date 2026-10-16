# CHANGELOG — portal v3.0.0 ("Dashboard V2" feature release)

Naming note: the previous README contained an internal "V3.2 — Performance Continuity" milestone. This release is designated **portal v3.0.0** as agreed for the Dashboard V2 update; the earlier milestone text is kept for history.

## PRESERVED
- Dash architecture, visual identity (navy / warm white / gold / forecast blue / verification green, Manrope).
- Live forecasting engine, hourly rolling outlook, 30-minute operational job, verification, historical replay, month-start baselines.
- Database schema (no table changed), timestamp contract (naive market clock; NEM AEST, WA AWST; civil DST conversion).
- AEMO comparison logic, Regional Analysis charts, all existing downloads and the complete region ZIP, all existing component ids used by callbacks.
- 48H / 7D / 30D / 4M horizons, upcoming interval table, live-vs-replay provenance, Docker/compose, gunicorn entry point, monthly deployment gate.

## MODIFIED
- **Dashboard navigation tab removed; Live is the default and home view** (`NAV_ITEMS`, `DEFAULT_TAB`, clientside navigation, `portal_a11y.js` PANES).
- **Production model selection is evidence-driven**: `PRODUCTION_MODELS` / `DEFAULT_MODEL_BY_REGION` are resolved from `selected_model.json` + allow-list + existing bundle (lazy, backward-compatible names in `config.py`).
- **WA default corrected to RandomForest_B**; **SA1 fallback handling** (serves A with a visible warning until the SA1 B bundle exists).
- **Correct A/B methodology**: A · core (39, no weather), B · weather-enhanced (54); exact feature lists shown.
- Model switch: static label for single-bundle regions; "Compare A/B configuration" for SA1/WA (comparison only).
- **Numeric-budget reduction**: Live 3 headline numbers; Performance MAE/sMAPE/coverage with "More metrics"; AEMO 2 headline numbers.
- Region-aware clock (local + market time) replaces the fixed Adelaide clock.
- `live_inference`: portable bundle loading with structure validation; Stage 2 root rebasing; per-run production resolution.
- `model_training` / `monthly_retrain`: `--region`, `--model`, `--reuse-stage2`; relative `CURRENT.txt` pointers; research models refused.
- `database`: SQLite upsert fix; SQLite datetime binding; dialect-portable SQL helpers (PostgreSQL output unchanged).
- **Path portability** for legacy `/app/models/...` and `/app/data/stage2/...` paths.

## ADDED
- **Timezone-aware region initialisation** with **manual override persistence** (`region-preference` local store), "Use my timezone" reset, deep links `?region=&view=&mode=`.
- **Model comparison mode** in Forecast Performance: selected-model card with decision confidence, ten-model ranking (MAE/nMAE), **MLP + GRU research evidence** ("Did deep learning help?"), **horizon insight**, **stability insight**, **weather insight**, **feature importance** (grouped drivers), **peak insight**, **model journey**, technical drawer (DM tests, failures, additional experiments).
- **Generated insight system** (`pipeline/insights.py`) — deterministic sentences from evidence and live values.
- Typical error range band; forecast-revision sentence; high-demand note; data-freshness guard; production-model line with plotted-forecast provenance; mixed-model-version warning.
- Production model card (Markdown) and ten-model evidence CSV downloads; forecast difficulty by region.
- `pipeline/model_evidence.py`, `pipeline/insights.py`, `pipeline/region_prefs.py`, `pipeline/feature_contract.py`; `data/model_evidence/current/` (27 files, SHA-256 manifest); `tests/` (102 tests).
- Evidence version in footer and Methodology; `PORTAL_DEFAULT_REGION` env; `python -m pipeline.model_evidence status|validate-bundles`.
- **Accessibility updates**: sr-only chart summaries, aria-live region message, pattern + colour model families, 44 px targets.

## REMOVED / RELOCATED
- Dashboard pane, hero and capability strip (content relocated into Live / Methodology).
- National pulse KPIs, region share bar and six multi-value region cards → Other-regions strip.
- Live month-to-date MAE/sMAPE/peak/coverage cards → collapsed "Month-to-date monitoring" (same ids).
- `app/__pycache__/`, stray `dockerignore`; `VERSION.txt` now holds version metadata.
