# AUDIT_V3 — audit of the existing portal before the v3.0.0 changes

Scope: `app/app.py` (3,410 lines), `app/assets/dashboard.css`, `app/assets/portal_a11y.js`, `pipeline/*.py`, Docker files, model folders (`CURRENT.txt` pointers + bundles), WP4 evidence pack.

## A. Production-correctness findings (fixed first)

| # | Finding | Evidence | Fix in v3 |
|---|---|---|---|
| C1 | SA1 served RandomForest_A; evidence selects RandomForest_B; no SA1 B bundle | `config.PRODUCTION_MODELS["SA1"]=["RandomForest_A"]`; `selected_model.json` | Resolver + visible fallback; targeted training CLI |
| C2 | WA default was RandomForest_A (first list item); evidence selects B | `DEFAULT_MODEL_BY_REGION = models[0]` | Default comes from evidence + artefact check, never list order |
| C3 | Hand-coded production lists drifted from evidence | C1, C2 | `pipeline/model_evidence.resolve_production_config()` |
| C4 | Methodology listed "Weather" under Model A | `_methodology_panel()` | A · core (39, no weather) / B · weather-enhanced (54) with exact lists |
| C5 | `CURRENT.txt` holds absolute Docker paths (`/app/models/...`) | pointer files | Portable resolver; new pointers relative to `MODEL_DIR` |
| C5b | Bundles store absolute Stage 2 roots (`/app/data/stage2/...`) | `bundle["stage2_root"]` | `resolve_stage2_root()` rebases onto `STAGE2_DIR` |
| C6 | WP4 `.joblib` files are bare estimators, not portal bundles | WP4 pipeline `joblib.dump(run.fitted.model)` | Never loaded; bundle structure validated before use |
| C7 | Clock fixed to `PORTAL_TIMEZONE=Australia/Adelaide` for all users | `portal_clock()` | Selected-region civil time + market time where different |
| C8 | `"NSW1"` hard-coded in stores/dropdowns/maps | layout | Region resolver + `PORTAL_DEFAULT_REGION` |
| C9 | SQLite upserts failed (`INSERT … SELECT … ON CONFLICT` parse error) | test run | `WHERE true` added (portable) |
| C10 | Several queries PostgreSQL-only (`EXTRACT … ::int`, `DATE_TRUNC`, `EXTRACT(EPOCH …)`); SQLite text timestamps broke `forecast_origin = :o` | test run | Dialect helpers in `database.py`; SQLite datetime binding matches stored format |
| C11 | Packaging: `app/__pycache__/`, stray `dockerignore`, `VERSION.txt` contained a shell command | archive | Removed / rewritten |

## B. Feature audit

| Current feature | Keep | Modify | Remove / relocate | New dependency | Reason |
|---|---|---|---|---|---|
| Nav: Dashboard, Live, Performance, AEMO, Regional | | ✓ Live, Performance, AEMO, Regional; default `live` | Dashboard tab removed | — | Requested; duplicate home |
| Hero + capability strip | | | Relocated: region header in Live; capabilities in Methodology | — | Space and repetition |
| Portal clock (Adelaide) | | ✓ region-aware | | `timezones.market_now` | C7 |
| National pulse KPIs + share bar | | | Replaced by "Other regions" strip (one number each) | — | Numeric budget |
| Six region cards | | ✓ compact tiles, click = manual region | | — | Less density |
| Operational status | ✓ | collapsed in Live | | — | Transparency |
| Downloads + region ZIP | ✓ ids unchanged | + model card, + ten-model evidence CSV | | evidence loader | Traceability |
| Methodology | | ✓ corrected A/B, model journey, evidence version | | `feature_contract.py` | C4 |
| Team strip | ✓ | moved to Live footer | | — | Attribution |
| Live map picker | ✓ | behind "Change region" | | — | Maps secondary |
| Live lead KPIs | ✓ | direction as words (value on hover) | | `insights.py` | Numeric budget |
| Live MTD MAE/sMAPE/peak/coverage cards | ✓ ids unchanged | | Moved into collapsed "Month-to-date monitoring" | — | Duplicates Performance |
| Horizon chart | ✓ | + typical error range band | | `forecast_error_bands*.csv` | Meaning without numbers |
| Upcoming intervals table | ✓ | collapsed, default 10 rows | | — | Detail |
| Context bar | ✓ | on Live: only the model control (region field hidden) | | — | One region selector per view |
| Model switch | | ✓ static label when one bundle; "Compare A/B configuration" for SA1/WA | | resolver | Comparison ≠ production serving |
| Performance 5 KPIs | | ✓ MAE, sMAPE, Coverage; RMSE/Bias in "More metrics"; mixed-version warning | | — | Numeric budget; auditability |
| — | | | | `model_evidence.py`, `insights.py`, `region_prefs.py` | Model comparison mode, timezone region |
| AEMO 4 KPIs | ✓ logic | 2 headline + detail | | — | Numeric budget |
| Regional analysis | ✓ | + forecast difficulty by region | | evidence | Evidence-based context |
| `portal_a11y.js` | ✓ | PANES = live, performance, aemo, history | | — | Nav change |

## C. Verified evidence facts used by the portal

Selected models: NSW1/QLD1/VIC1/TAS1 RandomForest_A; SA1/WA RandomForest_B. Feature contracts: Stage 2 `feature_set_A_core.csv` (39) = WP4 `FEATURE_SET_A`; Stage 2 `feature_set_B_weather.csv` (54) = WP4 `FEATURE_SET_B` = real WA RandomForest_B bundle `feature_names`. Real WA B bundle loads, validates and produced a 5,856-row forecast end-to-end (`v2026_09_20_0536`).
