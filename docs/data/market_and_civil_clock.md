# Market Clock and Civil Calendar Handling

The modelling timestamp and civil-calendar interpretation serve different
purposes.

## Market clock

Used for:

- continuous 30-minute modelling grid
- lag construction
- rolling features
- chronological model execution

## Civil calendar

Used for:

- weekday/weekend classification
- public holidays
- calendar interpretation

Keeping these concepts separate prevents daylight-saving changes from
silently shifting holiday and lag relationships.
