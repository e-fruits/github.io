"""
scripts/run_scanner.py

CLI to run the pre-market scanner — either live (today) or historically.

Usage examples
──────────────
  # Today's pre-market watchlist (live mode)
  python scripts/run_scanner.py

  # Historical watchlist for a specific date
  python scripts/run_scanner.py --date 2023-07-14

  # Generate historical watchlists for a range (writes to scanner_watchlist)
  python scripts/run_scanner.py --from 2023-01-01 --to 2023-12-31

  # Override filter thresholds
  python scripts/run_scanner.py --min-rvol 5 --min-herd 5.0 --min-gap 0.03
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.utils.config import settings
from src.utils.db import init_db
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


@click.command()
@click.option("--date",      "target_date", default=None, help="Single date (YYYY-MM-DD)")
@click.option("--from",      "from_date",   default=None, help="Range start")
@click.option("--to",        "to_date",     default=None, help="Range end")
@click.option("--min-rvol",  default=None,  type=float,   help="Min relative volume (override settings)")
@click.option("--min-herd",  default=None,  type=float,   help="Min herd score (override settings)")
@click.option("--min-gap",   default=None,  type=float,   help="Min gap %% (override settings, e.g. 0.02)")
@click.option("--db",        "db_path",     default=None, help="Override database path")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    target_date: str | None,
    from_date: str | None,
    to_date: str | None,
    min_rvol: float | None,
    min_herd: float | None,
    min_gap: float | None,
    db_path: str | None,
    log_level: str,
) -> None:
    log_cfg = settings.get("logging", {})
    configure_logging(level=log_level, console=True)

    db = db_path or settings["database"]["path"]
    init_db(db)

    from src.pipeline.scanner import generate_watchlist, generate_watchlist_range, print_watchlist

    if target_date:
        # Single date
        df = generate_watchlist(
            target_date,
            min_relative_volume=min_rvol,
            min_gap_pct=min_gap,
            min_herd_score=min_herd,
            db_path=db,
        )
        print_watchlist(df, target_date)

    elif from_date:
        # Date range
        results = generate_watchlist_range(from_date, to_date, db_path=db)
        for ds, df in sorted(results.items()):
            print_watchlist(df, ds)

    else:
        # Live mode: today
        today = date.today().isoformat()
        log.info("live_scanner", date=today)

        # Optionally refresh pre-market data for today's universe
        try:
            from src.utils.db import fetch_df, get_connection
            conn = get_connection(db)
            watchlist_tickers_df = fetch_df(
                conn,
                "SELECT DISTINCT ticker FROM universe WHERE date = ? AND is_active = 1",
                (today,),
            )
            if not watchlist_tickers_df.empty:
                from src.pipeline.price_volume import ingest_premarket
                ingest_premarket(watchlist_tickers_df["ticker"].tolist(), today, db_path=db)
        except Exception as exc:
            log.warning("premarket_refresh_failed", error=str(exc))

        df = generate_watchlist(
            today,
            min_relative_volume=min_rvol,
            min_gap_pct=min_gap,
            min_herd_score=min_herd,
            db_path=db,
        )
        print_watchlist(df, today)


if __name__ == "__main__":
    main()
