# Baseline and Metric Contract

Every candidate model should be evaluated against the same forecast origin
and target observations.

## Core metrics

- MAE
- RMSE
- sMAPE
- Bias
- Median Absolute Error where required

## Baseline

The seasonal-naive baseline provides the minimum reference for determining
whether a machine-learning or statistical model contributes useful forecast
skill.

Metrics must use matched timestamps and identical target definitions.
