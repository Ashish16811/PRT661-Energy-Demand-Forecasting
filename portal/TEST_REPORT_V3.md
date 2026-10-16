# TEST_REPORT_V3 — portal v3.0.0

Environment: Linux, Python 3.12, dash 4.4.1, plotly 7.1.0, scikit-learn 1.8.0, SQLAlchemy 2.1.1, SQLite. Command: `python -m pytest -q`.
Result: **103 passed, 1 skipped** (default); **104 passed** with `REAL_MODEL_DIR` pointing at the supplied WA RandomForest_B bundle.
Database-backed tests use a temporary SQLite database seeded with synthetic rows created by the tests (never shipped or displayed).

## Required test list

| # | Test | Status | Evidence |
|---|---|---|---|
| 1 | selected_model mapping NSW1 A, QLD1 A, VIC1 A, SA1 B, TAS1 A, WA B | PASSED | `test_01_selected_model_mapping_from_evidence` |
| 2 | SA1 B missing → selected B, served A, visible warning | PASSED | `test_02_…`, `test_sa1_fallback_visible_on_live`, model card + comparison text |
| 3 | SA1 fake B bundle → served B | PASSED | `test_03_…`, `test_04_full_expected_production_config` |
| 4 | WA B bundle → served B | PASSED | fake: `test_04_…`; **real bundle**: `test_wa_real_bundle_loads_and_is_feature_set_b` (with REAL_MODEL_DIR) |
| 5 | Bundle structure validation | PASSED | `test_05_…` (bare WP4-style estimator refused, wrong family refused), `test_05b_…` recorded-invalid fallback |
| 6 | Feature Set A exact contract (39, no weather) | PASSED | `test_06_…` |
| 7 | Feature Set B exact contract (54 = A + 15 weather) | PASSED | `test_07_…`, `test_07b_…` (equals Stage 2 CSVs) |
| 8 | CURRENT.txt legacy Docker path fallback | PASSED | `test_08_…`; real `/app/models/WA/...` pointer resolved locally |
| 9 | Relative model path loading | PASSED | `test_09_…` |
| 10 | MLP never production | PASSED | `test_10_11_…[MLP]` |
| 11 | GRU never production | PASSED | `test_10_11_…[GRU]` (+ XGBoost_B, SARIMAX) |
| 12 | SARIMAX incomplete handled honestly | PASSED | `test_12_…` (14 failed runs from evidence; flagged incomplete) |
| 13 | Browser timezone mapping | PASSED | 18 parametrised aliases |
| 14 | Darwin unsupported | PASSED | `test_14_16_…`, `test_14_darwin_falls_back_with_message` |
| 15 | Eucla unsupported | PASSED | `test_14_16_…` |
| 16 | Lord Howe unsupported | PASSED | `test_14_16_…` (Lord_Howe, LHI) |
| 17 | Manual override persists | PASSED | `test_17_…`, `test_17b_…`, `test_17_18_choose_region_callback_manual_and_url` |
| 18 | URL region overrides local selection | PASSED | `test_18_…` (manual preference not overwritten) |
| 19 | Dashboard tab removed | PASSED | `test_19_…` |
| 20 | Live default | PASSED | `test_19_…_and_20_live_default`, `test_30_…` |
| 21 | All callback ids exist | PASSED | `test_21_…` (33 callbacks; only dynamic `mc-metric` whitelisted) |
| 22 | AEMO view works | PASSED | `test_22_aemo_view_runs` |
| 23 | Regional Analysis works | PASSED | `test_23_…` (history, profile, difficulty) |
| 24 | Downloads work | PASSED | `test_24_…` (model card, evidence CSV, upcoming forecast) |
| 25 | Live provenance matches UI label | PASSED | `test_25_…` — plotted WA forecasts labelled `RandomForest_B vB_TEST`; comparison view labelled A; band hidden for non-selected model |
| 26 | Mixed model versions not mislabelled | PASSED | `test_26_…` |
| 27 | Stale-data guard | PASSED | `test_27_…`, `test_direction_words_and_freshness` |
| 28 | Responsive CSS | PASSED (static) | `test_28_…` — breakpoints, 44 px targets, reduced motion |
| 29 | Keyboard accessibility | PASSED (logic) | `test_29_…` PANES; `test_29b_…` Arrow/Home/End index logic executed in Node |
| 30 | App import / server startup | PASSED | `test_30_…`; plus manual run below |
| 31 | Docker build | **NOT RUN** | Docker is not available in the build environment |

Additional: insight rules against evidence (NSW1 2024 weak year named, TAS1 peaks "not harder", VIC1 non-monotonic horizon, no "improved" wording when flags are false), manifest OK / missing / tampered, decision-confidence rule, targeted-training CLI, PostgreSQL SQL fragments identical to v1, SQLite fragments execute, `__main__` block last.

## Manual end-to-end checks (performed)

| Check | Result |
|---|---|
| `python -m pipeline.model_evidence validate-bundles` on the supplied models | WA/RandomForest_B **VALID**; resolver: WA serves B (other bundles were not extracted in the build environment) |
| Real WA RandomForest_B forecast through `live_inference.forecast_one("WA","RandomForest_B")` with Docker paths rebased | 5,856 rows, `model_version v2026_09_20_0536`, finite values (1,046–3,347 MW) |
| Dash server started; GET `/`, `/_dash-layout`, `/_dash-dependencies`, assets | all 200; layout has `pane-live`, no `pane-dashboard` |
| POST `/_dash-update-component` for Model comparison (SA1) | 200, cards rendered incl. "Currently serving" fallback |

## Not run / limitations
- Docker build and `docker compose up` (no Docker).
- PostgreSQL execution (package installation was blocked); PostgreSQL SQL is unchanged from v1 (asserted by test), SQLite paths are tested.
- Visual browser rendering / screenshots and screen-reader testing (no browser in the environment).
- SA1 RandomForest_B bundle was **not trained** (by instruction); SA1 remains PENDING PRODUCTION ARTEFACT and serves RandomForest_A with a visible warning.

## Go / no-go gate
| Gate | Status |
|---|---|
| A. Production selection map correct | GO |
| B. WA RandomForest_B loads | GO (real bundle validated and forecast end-to-end) |
| C. SA1 resolver/fallback works | GO (fallback visible) |
| D. A/B feature contracts | GO |
| E. Evidence manifest verifies | GO (27 files, SHA-256) |
| F. Provenance labels match served forecasts | GO |
| G. No callback references removed Dashboard components | GO |
| H. App starts | GO (Dash server); gunicorn/Docker not run |
| I. Live / Performance / AEMO / Regional function | GO on seeded SQLite; PostgreSQL + browser check pending |

**Release status: ready for deployment testing, with SA1 on fallback A until `python -m pipeline.monthly_retrain --region SA1 --model RandomForest_B` has been run, and Docker/PostgreSQL/browser checks to be done on the deployment host.**
