from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("PORTAL_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = Path(os.getenv("RAW_DIR", DATA_DIR / "raw"))
STAGE2_DIR = Path(os.getenv("STAGE2_DIR", DATA_DIR / "stage2"))
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", DATA_DIR / "runtime"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{RUNTIME_DIR / 'portal.db'}")
PORTAL_ROLE = os.getenv("PORTAL_ROLE", "local").strip().lower() or "local"

REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1", "WA"]
NEM_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]

REGION_LABELS = {
    "NSW1": "New South Wales",
    "QLD1": "Queensland",
    "VIC1": "Victoria",
    "SA1": "South Australia",
    "TAS1": "Tasmania",
    "WA": "Western Australia / SWIS",
}
REGION_MARKET = {r: "NEM" for r in NEM_REGIONS} | {"WA": "WEM / SWIS"}
REGION_TZ = {
    "NSW1": "Australia/Sydney",
    "QLD1": "Australia/Brisbane",
    "VIC1": "Australia/Melbourne",
    "SA1": "Australia/Adelaide",
    "TAS1": "Australia/Hobart",
    "WA": "Australia/Perth",
}
REGION_MARKET_TZ = {
    **{r: "Australia/Brisbane" for r in NEM_REGIONS},  # NEM market clock: AEST year-round
    "WA": "Australia/Perth",
}
REGION_COORDS = {
    "NSW1": (-33.8688, 151.2093),
    "QLD1": (-27.4698, 153.0251),
    "VIC1": (-37.8136, 144.9631),
    "SA1": (-34.9285, 138.6007),
    "TAS1": (-42.8821, 147.3272),
    "WA": (-31.9523, 115.8613),
}

# ---------------------------------------------------------------------------
# Production models (portal v3.0.0)
#
# The SELECTION authority is data/model_evidence/current/selected_model.json
# (WP4 run 2026-09-29). pipeline.model_evidence.resolve_production_config()
# serves a model only if it is selected, allow-listed (RandomForest_A/B) and a
# valid portal bundle exists; otherwise it serves the fallback below and reports
# the gap. The lists below are the explicit FALLBACK / intended bundles that the
# monthly retraining maintains. List order never decides the served model.
# ---------------------------------------------------------------------------
FALLBACK_PRODUCTION_MODELS = {
    "NSW1": ["RandomForest_A"],
    "QLD1": ["RandomForest_A"],
    "VIC1": ["RandomForest_A"],
    # SA1 and WA: B is the selected production configuration.
    # Historical A/B evidence is retained separately, but live inference serves
    # one model per region to avoid production ambiguity.
    "SA1": ["RandomForest_B"],
    "TAS1": ["RandomForest_A"],
    "WA": ["RandomForest_B"],
}
# Served when selection evidence is missing or the selected bundle is absent.
FALLBACK_DEFAULT_MODEL = {
    "NSW1": "RandomForest_A",
    "QLD1": "RandomForest_A",
    "VIC1": "RandomForest_A",
    "SA1": "RandomForest_B",
    "TAS1": "RandomForest_A",
    "WA": "RandomForest_B",
}
PORTAL_DEFAULT_REGION = os.getenv("PORTAL_DEFAULT_REGION", "NSW1")
PORTAL_VERSION = "3.0.0"


def __getattr__(name):
    """Backward-compatible PRODUCTION_MODELS / DEFAULT_MODEL_BY_REGION / TRAINING_MODELS.

    Resolved lazily (PEP 562) through the evidence-driven resolver, so existing
    ``from .config import PRODUCTION_MODELS`` imports keep working.
    """
    if name in ("PRODUCTION_MODELS", "DEFAULT_MODEL_BY_REGION", "TRAINING_MODELS"):
        from .model_evidence import production_config
        cfg = production_config()
        return {"PRODUCTION_MODELS": cfg.production_models,
                "DEFAULT_MODEL_BY_REGION": cfg.default_model,
                "TRAINING_MODELS": cfg.training_models}[name]
    raise AttributeError(name)


FORECAST_FREQ = "30min"
FORECAST_HORIZON_MONTHS = 4
LIVE_REFRESH_MINUTES = 30
LONG_HORIZON_REFRESH_MINUTES = 60
WEATHER_FORECAST_DAYS = 7
MODEL_RETRAIN_DAY = 1

# Open-Meteo forecast feed used only for live future weather. Historical weather
# remains the archive/reanalysis source already used by Stage 1/2.
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

for p in (DATA_DIR, RAW_DIR, STAGE2_DIR, RUNTIME_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)
