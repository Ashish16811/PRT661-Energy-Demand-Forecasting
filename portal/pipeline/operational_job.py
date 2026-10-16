from __future__ import annotations

import logging
import uuid

from .acquisition_live import run_hourly_acquisition
from .database import init_db, log_run, utcnow_naive
from .live_inference import run_next_interval_forecasts
from .verification import verify_next_interval_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("operational_30min_job")


def main():
    """Fast 30-minute operational cycle.

    Refreshes current inputs, stores only the next-interval forecast that users see,
    and scores older next-interval forecasts whose actual demand has arrived. The
    heavier four-month rolling forecast remains in pipeline.hourly_job.
    """
    init_db()
    run_id = f"operational-{uuid.uuid4().hex[:12]}"
    started = utcnow_naive()
    try:
        acq = run_hourly_acquisition()
        one_step = run_next_interval_forecasts()
        verified = verify_next_interval_forecasts()
        detail = (
            f"acquisition={acq}; next_interval_rows={len(one_step)}; "
            f"next_interval_verified={len(verified)}"
        )
        log_run(run_id, "OPERATIONAL_30MIN", started, "COMPLETE", detail)
        log.info(detail)
    except Exception as exc:
        log.exception("30-minute operational pipeline failed")
        log_run(run_id, "OPERATIONAL_30MIN", started, "FAILED", str(exc))
        raise


if __name__ == "__main__":
    main()
