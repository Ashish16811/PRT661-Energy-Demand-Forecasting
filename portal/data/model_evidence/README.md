# Dashboard model evidence (WP4, 29 Sep 2026)

Curated from run EnergyForecast_Run_20260929_145931 (pipeline v4.0.0). Drop this folder into the portal as `data/model_evidence/`.

- `current/` — evidence files read by `pipeline/model_evidence.py`; `evidence_manifest.json` lists SHA-256, row counts and provenance for each file.
- Copied files are unchanged; trimmed files keep original values (row/column subsets, see manifest `provenance`).
- Derived files (`feature_importance_grouped.csv`, `feature_importance_top10_selected.csv`, `forecast_error_bands.csv`, `peak_normal_vs_high_summary.csv`) are deterministic aggregations written by `build_evidence.py` (paths inside it refer to the machine that built the pack).
- `dashboard_test_expectations.json` is for automated tests only; never display it as hard-coded text.
- `forecast_error_bands.csv` is a historical error range from four-month backtests, not a probability forecast.
