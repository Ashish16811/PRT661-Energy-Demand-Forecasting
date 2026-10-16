from __future__ import annotations

import argparse
import logging
import uuid

import pandas as pd

from .backfill_month_start import capture_month_start_for_all
from .database import init_db, log_run, utcnow_naive
from .model_training import run_monthly_retraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("monthly_retrain")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Monthly production retraining (Random Forest bundles only).")
    ap.add_argument("--region", action="append", help="Limit to region(s), e.g. --region SA1 (repeatable)")
    ap.add_argument("--model", action="append", help="Limit to model(s), e.g. --model RandomForest_B (repeatable)")
    ap.add_argument("--reuse-stage2", action="store_true",
                    help="Train from the current Stage 2 snapshot instead of rebuilding it (default for targeted runs)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    targeted = bool(args.region or args.model)
    init_db()
    run_id = f"monthly-{uuid.uuid4().hex[:12]}"
    started = utcnow_naive()
    job = "TARGETED_PRODUCTION_TRAINING" if targeted else "MONTHLY_RETRAIN"
    try:
        registry = run_monthly_retraining(args.region, args.model, reuse_stage2=args.reuse_stage2 or targeted)
        if registry is None or registry.empty:
            promoted = rejected = evaluated = 0
        else:
            status = registry.get("status", pd.Series(index=registry.index, dtype="object")).astype(str)
            promoted = int(status.eq("PRODUCTION").sum())
            rejected = int(status.str.startswith("REJECTED").sum())
            evaluated = int(len(registry))
        detail = (
            f"evaluated={evaluated} production bundle target(s); "
            f"promoted={promoted}; rejected={rejected}"
        )
        if targeted:
            # Targeted artefact generation (e.g. SA1 RandomForest_B) does not create a
            # month-start baseline: the new bundle was trained inside the month, so a
            # baseline from it would not be leakage-safe for this month.
            detail += f"; targets regions={args.region or 'all'} models={args.model or 'intended'}; no month-start baseline"
        else:
            month = pd.Timestamp.now(tz="Australia/Adelaide").strftime("%Y-%m")
            # Ensure the current month has one immutable month-start baseline. This is
            # leakage-safe even when the scheduled job was missed on day 1 because the
            # backfill helper refuses to run unless the production bundle was trained
            # strictly before the first day of the target month.
            baseline = capture_month_start_for_all(month, overwrite=False)
            detail += f"; month_start_baseline={month}; rows_created={len(baseline)}"
        log_run(run_id, job, started, "COMPLETE", detail)
        log.info(detail)
    except Exception as exc:
        log.exception("Production retraining failed")
        log_run(run_id, job, started, "FAILED", str(exc))
        raise


if __name__ == "__main__":
    main()
