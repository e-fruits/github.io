"""
scripts/run_backtest.py

CLI to run the event-driven backtest.

Usage examples
──────────────
  # Full backtest with default MechanicalStrategy
  python scripts/run_backtest.py --from 2022-01-01 --to 2023-12-31

  # Custom capital and position type
  python scripts/run_backtest.py --from 2022-01-01 --capital 50000 --pos-type margin

  # Print per-day equity curve
  python scripts/run_backtest.py --from 2022-01-01 --verbose

  # Use a specific run ID (for comparing runs)
  python scripts/run_backtest.py --from 2022-01-01 --run-id experiment_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.utils.config import settings
from src.utils.db import init_db
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


@click.command()
@click.option("--from",     "from_date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--to",       "to_date",   default=None,  help="End date (YYYY-MM-DD)")
@click.option("--capital",  default=None, type=float,   help="Override starting capital")
@click.option(
    "--pos-type",
    default="equity",
    type=click.Choice(["equity", "margin", "debit_spread"]),
    show_default=True,
    help="Position type for all trades",
)
@click.option("--run-id",   default=None,  help="Run identifier (auto-generated if omitted)")
@click.option("--db",       "db_path", default=None, help="Override database path")
@click.option("--verbose",  is_flag=True, default=False)
@click.option("--output",   "output_path", default=None, help="Write metrics JSON to file")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    from_date: str,
    to_date: str | None,
    capital: float | None,
    pos_type: str,
    run_id: str | None,
    db_path: str | None,
    verbose: bool,
    output_path: str | None,
    log_level: str,
) -> None:
    log_cfg = settings.get("logging", {})
    configure_logging(level=log_level, log_dir=log_cfg.get("log_dir"), console=True)

    db = db_path or settings["database"]["path"]
    init_db(db)

    # Override capital and position type in settings for this run
    if capital is not None:
        settings["backtest"]["starting_capital"] = capital
    settings["backtest"]["position_type"] = pos_type

    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import print_metrics

    engine = BacktestEngine(db_path=db, run_id=run_id)
    metrics = engine.run(from_date=from_date, to_date=to_date)

    print_metrics(metrics)

    if verbose:
        from src.utils.db import fetch_df, get_connection
        conn = get_connection(db)
        daily_df = fetch_df(
            conn,
            "SELECT date, ending_capital, daily_pnl, num_trades FROM backtest_daily WHERE run_id = ? ORDER BY date",
            (engine.run_id,),
        )
        if not daily_df.empty:
            print("\nDaily equity curve:")
            print(daily_df.to_string(index=False))

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        log.info("metrics_written", path=output_path)


if __name__ == "__main__":
    main()
