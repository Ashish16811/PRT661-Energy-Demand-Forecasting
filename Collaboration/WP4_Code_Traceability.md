# WP4 Code Traceability and Merge Specification

Each module maps onto an existing section of `Energy_Demand_Forecasting_Pipeline.py` v4.0.0. None of them is a blind concatenation; each carries executable checks proving it matches the section it will replace.

| Member module | Source functions in pipeline v4.0.0 | Final pipeline section | Dependencies | Possible conflict | Integration test |
|---|---|---|---|---|---|
| Ashish – MLP | `MLPLearner` (`_build`, `fit`, `predict`, `architecture`), `MLP_PARAMETERS`, `FEATURE_SET_B_NN`, `NN_EXCLUDED_FEATURES`, MLP branch of `run_model()` | P (scalers, callbacks), Q (MLP), T (`run_model`) | `wp4_contracts`, pipeline `run_ml`, `ColumnScaler`, `TargetScaler`, `_keras_callbacks` | Two copies of the MLP if Section Q is not removed when the import is added | 11 checks incl. identical-seed identical predictions |
| Bishal – MLP validation | `metrics`, `group_metrics`, `stability_table`, `peak_metrics`, `neural_vs_previous`, `neural_figures` (N01) | J, U–X, Y, Z (reads outputs only) | `wp4_contracts`, run folder | Column renames in output CSVs would break loading | self-test + detail-vs-published re-derivation (< 1e-3) |
| Suraj – GRU sequence | `gru_training_starts`, `build_gru_arrays`, `_nan_count_windows`, `GRU_DECODER_FEATURES`, `GRU_ENCODER_CALENDAR`, data path and recursion loop in `run_gru` | R, and the forecast half of S | `wp4_contracts`, pipeline `design_matrix` | Recursion loop currently inline in `run_gru`; must be replaced, not duplicated | 14 checks incl. identical arrays and horizon-actual sentinel |
| Sudip – GRU training | `_build_gru`, training/refit half of `run_gru`, `_keras_callbacks`, `_best_epoch`, weather-policy and leakage-audit checks | S, P, T (`_leakage_audit`) | `wp4_contracts`, Suraj's `SequenceBatch` | Scaler bundle fitted twice if both halves fit their own | 12 checks incl. identical-seed identical weights |
| Shared – contracts | constants `TIME`, `TARGET`, `FREQ`, `REGIONS`, `RANDOM_SEED`, `PIPELINE_VERSION`, `Run`, `Panel`, `OriginMaps` | A (configuration), T | none | A constant changed in one place only | `check_constants()` |

## What does NOT move into the modules

Preprocessing (Stage 2 functions), Feature Sets A/B, the tree models, SARIMA/SARIMAX, the selection rule and the production step stay where they are. The selected production models (Random Forest A/B) do not depend on any neural module.

## DESIGN_CONTROL rows of leakage_audit.csv mapped to executable checks (Issue 08)

| leakage_audit.csv row | Executable check |
|---|---|
| Neural scalers fitted on training rows only | `NeuralScalerBundle.assert_before()` in Ashish/Sudip modules |
| Neural early stopping uses only the 8 weeks before the origin | Ashish `chronological_validation_mask` check; Sudip `split_train_validation` check |
| GRU sequences end before the block they forecast | Suraj `training_starts` assertion, `SequenceBatch.validate()`, sentinel test |
| RF/XGB hyper-parameters fixed | pipeline parameters untouched by all modules (no module imports tree code) |
| Screening gate removed | not applicable to the neural modules; documented in the change log |
