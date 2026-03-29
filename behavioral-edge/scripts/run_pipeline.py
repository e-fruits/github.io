"""
scripts/run_pipeline.py

CLI to run the full historical data ingestion pipeline.

Usage examples
──────────────
  # Full backfill from 2020 to today
  python scripts/run_pipeline.py --from 2020-01-01

  # Single date refresh
  python scripts/run_pipeline.py --from 2024-01-15 --to 2024-01-15

  # Run only specific stages
  python scripts/run_pipeline.py --from 2024-01-01 --stages universe,prices,herd

  # Use custom database path
  python scripts/run_pipeline.py --from 2024-01-01 --db /path/to/custom.db
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure src/ is importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.utils.config import settings
from src.utils.db import init_db
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

ALL_STAGES = ["universe", "prices", "options", "wsb", "catalysts", "herd", "scanner"]


@click.command()
@click.option("--from", "from_date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--to",   "to_date",   default=None,  help="End date (YYYY-MM-DD), default=today")
@click.option(
    "--stages",
    default=",".join(ALL_STAGES),
    show_default=True,
    help="Comma-separated list of stages to run",
)
@click.option("--db", "db_path", default=None, help="Override database path")
@click.option("--log-level", default="INFO", show_default=True)
def main(from_date: str, to_date: str | None, stages: str, db_path: str | None, log_level: str) -> None:
    log_cfg = settings.get("logging", {})
    configure_logging(
        level=log_level,
        log_dir=log_cfg.get("log_dir"),
        json_logs=log_cfg.get("json_logs", False),
        console=True,
    )

    db = db_path or settings["database"]["path"]
    log.info("pipeline_start", from_date=from_date, to_date=to_date, stages=stages, db=db)

    # Initialise schema
    init_db(db)

    stage_list = [s.strip() for s in stages.split(",")]

    # ── Universe ────────────────────────────────────────────────────────────
    if "universe" in stage_list:
        log.info("stage_start", stage="universe")
        from src.pipeline.universe import build_universe_range
        build_universe_range(from_date, to_date, db_path=db)
        log.info("stage_done", stage="universe")

    # ── Prices ──────────────────────────────────────────────────────────────
    if "prices" in stage_list:
        log.info("stage_start", stage="prices")
        from src.pipeline.price_volume import ingest_date_range
        ingest_date_range(from_date, to_date, db_path=db)
        log.info("stage_done", stage="prices")

    # ── Options ─────────────────────────────────────────────────────────────
    if "options" in stage_list:
        log.info("stage_start", stage="options")
        from src.pipeline.options_flow import ingest_options_range
        ingest_options_range(from_date, to_date, db_path=db)
        log.info("stage_done", stage="options")

    # ── WSB sentiment ────────────────────────────────────────────────────────
    if "wsb" in stage_list:
        log.info("stage_start", stage="wsb")
        from src.pipeline.wsb_sentiment import ingest_apewisdom
        from datetime import date, timedelta
        start = date.fromisoformat(from_date)
        end   = date.fromisoformat(to_date) if to_date else date.today()
        cur   = start
        while cur <= end:
            if cur.weekday() < 5:
                ingest_apewisdom(cur.isoformat(), db_path=db)
            cur += timedelta(days=1)
        log.info("stage_done", stage="wsb")

    # ── Catalysts ────────────────────────────────────────────────────────────
    if "catalysts" in stage_list:
        log.info("stage_start", stage="catalysts")
        from src.pipeline.catalysts import ingest_all_catalysts
        ingest_all_catalysts(from_date, to_date, db_path=db)
        log.info("stage_done", stage="catalysts")

    # ── Herd scores ──────────────────────────────────────────────────────────
    if "herd" in stage_list:
        log.info("stage_start", stage="herd")
        from src.pipeline.herd_score import compute_herd_scores_range
        compute_herd_scores_range(from_date, to_date, db_path=db)
        log.info("stage_done", stage="herd")

    # ── Scanner watchlist ────────────────────────────────────────────────────
    if "scanner" in stage_list:
        log.info("stage_start", stage="scanner")
        from src.pipeline.scanner import generate_watchlist_range
        generate_watchlist_range(from_date, to_date, db_path=db)
        log.info("stage_done", stage="scanner")

    log.info("pipeline_complete")


if __name__ == "__main__":
    main()
