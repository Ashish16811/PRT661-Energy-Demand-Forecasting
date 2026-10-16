from __future__ import annotations

import logging
import uuid

from .acquisition_live import run_hourly_acquisition
from .database import init_db, log_run, utcnow_naive
from .live_inference import run_live_forecasts
from .verification import verify_available_forecasts, verify_next_interval_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hourly_job")


def main():
    """Hourly rolling-outlook refresh.

    This heavier job refreshes the full current-month + next-three-month forecast.
    The dedicated pipeline.operational_job is responsible for the next 30-minute
    point forecast used by the primary live accuracy metric.
    """
    init_db()
    run_id = f"hourly-{uuid.uuid4().hex[:12]}"
    started = utcnow_naive()
    try:
        acq = run_hourly_acquisition()
        fc = run_live_forecasts()
        verified = verify_available_forecasts()
        one_step_verified = verify_next_interval_forecasts()
        detail = (
            f"acquisition={acq}; forecast_rows={len(fc)}; verified_rows={len(verified)}; "
            f"next_interval_verified={len(one_step_verified)}"
        )
        log_run(run_id, "HOURLY_REFRESH", started, "COMPLETE", detail)
        log.info(detail)
    except Exception as exc:
        log.exception("Hourly pipeline failed")
        log_run(run_id, "HOURLY_REFRESH", started, "FAILED", str(exc))
        raise


if __name__ == "__main__":
    main()
