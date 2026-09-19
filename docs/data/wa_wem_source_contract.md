# WA/WEM Acquisition Contract

WA is handled as a separate WEM acquisition path rather than being treated
as another NEM region.

## Native frequency

Historical WA demand contains two genuine source regimes:

- legacy 30-minute observations
- later 5-minute observations

The original 30-minute history must remain 30-minute data.

Synthetic 5-minute observations must not be generated from the older
30-minute series.

## Downstream integration

Stage 2 may aggregate genuine 5-minute MW observations to the common
30-minute modelling grid using the mean.
