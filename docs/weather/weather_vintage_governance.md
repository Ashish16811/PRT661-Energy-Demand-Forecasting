# Weather Forecast Vintage Governance

Model B must distinguish between:

- historical observed/reanalysis weather
- genuine seven-day forecast weather
- origin-safe climatology fallback

Forecast weather should preserve:

- provider
- model
- forecast origin
- lead day
- retrieval timestamp
- archive vintage

Future realised weather must never be substituted into a historical forecast
evaluation because that would give the model information unavailable at the
forecast origin.
