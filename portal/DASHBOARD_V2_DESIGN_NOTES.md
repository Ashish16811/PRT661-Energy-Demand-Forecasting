# Dashboard V2 design notes (portal v3.0.0)

## 1. Principles
**More explanation, less numerical clutter. More model evidence, without confusing research models with production models.**
Every card follows **Question → one generated sentence → small visual → optional technical detail**.

## 2. Numeric budget (prominent numbers visible by default)

| View | Budget | Everything else |
|---|---|---|
| Live | 3 (current MW, next 30-min MW, direction in words) + 1 per other region | insight sentence, chart, band, collapsibles |
| Performance · Live verification | 3 (MAE, sMAPE, coverage) | "More metrics" (RMSE, Bias); plain-English MAE translation |
| Performance · Model comparison | ≤ 1 per card | hover, technical-detail drawers, CSV download |
| AEMO | 2 (common timestamps, mean absolute gap) | detail row |

## 3. Information architecture
**Live:** region header (name, source message, Change region, Use my timezone, local + market clock) → freshness guard → headline trio → chart (48H/7D/30D/4M, typical error range) → insight → production-model line (served model, feature set, confidence, fallback / comparison chips, provenance of plotted forecast) → trust line (View performance / Why this model?) → other regions → upcoming intervals ▸ → month-to-date monitoring ▸ → system status ▸ → downloads / methodology → team.
**Forecast Performance:** segmented control *Live verification | Model comparison* (no new top-level tab).
**Context bar:** hidden region field on Live (Live has its own header); shown on the other views.

## 4. Region from browser timezone
`Intl.DateTimeFormat().resolvedOptions().timeZone` (no geolocation) → exact IANA mapping in `pipeline/region_prefs.py`. Unsupported by design: Australia/Darwin, North, Eucla, Lord_Howe, LHI, Antarctica/Macquarie and all non-Australian zones — never inferred from UTC offset. Priority: URL `?region=` → stored manual → mapped timezone → stored timezone → `PORTAL_DEFAULT_REGION` → NSW1. Stored in `dcc.Store(id="region-preference", storage_type="local")` as `{region, selection_source, browser_timezone, updated_at}`. Every user action writes `manual`; timezone never overwrites it; "Use my timezone" resets. A URL region is shown without overwriting the stored manual choice. UI only — database timestamps stay naive market-clock values.

## 5. Production vs research
Served model = selected by evidence ∧ allow-listed (RandomForest_A/B) ∧ valid portal bundle. Model switch is a **comparison view** for SA1/WA only; it never changes what scheduled jobs forecast. MLP/GRU appear only in Model comparison with "New after feedback · research candidate". GRU_L672 and XGBoost_A+CoreWeather appear only under "Additional experiments".

## 6. Insight rules (`pipeline/insights.py`, deterministic, no LLM)
| Insight | Rule |
|---|---|
| Direction | |Δ| < 0.5% → "Holding steady"; < 2% slightly; < 5% moderately; else sharply |
| Decision confidence | Clear margin if runner-up ≥ 3% behind **and** DM p < 0.05; else Narrow margin |
| Deep learning | % change in words ("x% higher/lower error"); closing sentence from `*_material_improvement` flags |
| Horizon | detects monotonic vs non-monotonic error; names band where neural models are closest |
| Stability | rank per matched year (2023–2025); names weak years; 2022 shown hollow as limited-history |
| Weather | XGBoost B vs A by band (|Δ| < 1% = "little difference"); climatology caveat always shown |
| Feature set | RandomForest A vs B mean MAE for the region |
| Drivers | ≤ 5 grouped importances; causation caveat always shown |
| Peaks | ratio peak/normal MAE ≥ 1.1 harder, ≤ 0.9 not harder, else similar; "too low" only if peak bias < 0 |
| Freshness | actual > 60 min or forecast origin > 2 h → amber status |

## 7. Typical error range
`forecast_error_bands.csv` (interval, by lead band × local hour) and `forecast_error_bands_daily.csv` (daily means, by lead band): p10–p90 residuals of the **selected** model on 2023–2025 backtests. Band = forecast − p90 … forecast − p10. Labelled "typical error range … not a probability forecast". Hidden when the viewed model is not the selected model. Default on (toggle).

## 8. Performance
Evidence CSVs are cached (keyed on manifest mtime/size) and never re-read by the 60-second refresh; model-comparison callbacks do not depend on `refresh-ui`. The other-regions strip uses two queries for all regions (was: one wide query plus per-card logic). The production resolver is cached for 5 minutes in the web process and re-resolved at every job run.

## 9. Accessibility and responsive
PANES updated; tabs have role/aria-selected/aria-controls/aria-labelledby, Enter/Space/Arrow/Home/End. Region auto-selection announced once (`aria-live="polite"`); every model-comparison chart and the live chart carry an sr-only generated summary. Model families use colour **and** bar pattern. `<details>` disclosures keep native keyboard support. Breakpoints at 1100/900/680/420 px; touch targets ≥ 44 px; tables scroll horizontally; reduced-motion respected. Palette unchanged.

## 10. Honesty guards
No example values in callbacks; empty states everywhere; evidence status OK / EVIDENCE_MISSING / EVIDENCE_UNVERIFIED shown quietly; served ≠ selected always visible; mixed model versions in a verification period flagged.
