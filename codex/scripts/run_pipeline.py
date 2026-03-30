"""CLI for initializing the database and running the universe builder scaffold."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.catalysts import load_catalysts_from_config
from src.pipeline.price_volume import load_daily_prices_from_config
from src.pipeline.universe import build_universe_from_config
from src.utils.config import load_settings
from src.utils.db import DatabaseManager
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 pipeline tasks.")
    parser.add_argument("--settings", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--build-universe", action="store_true")
    parser.add_argument("--load-daily-prices", action="store_true")
    parser.add_argument("--load-catalysts", action="store_true")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings)
    configure_logging(settings.pipeline.log_level)

    if args.init_db:
        db = DatabaseManager(settings.database.path)
        db.initialize()

    if args.build_universe:
        if args.start_date is None or args.end_date is None:
            raise SystemExit("--build-universe requires --start-date and --end-date")
        build_universe_from_config(args.start_date, args.end_date, args.settings)

    if args.load_daily_prices:
        if args.start_date is None or args.end_date is None:
            raise SystemExit("--load-daily-prices requires --start-date and --end-date")
        load_daily_prices_from_config(args.start_date, args.end_date, args.settings)

    if args.load_catalysts:
        if args.start_date is None or args.end_date is None:
            raise SystemExit("--load-catalysts requires --start-date and --end-date")
        load_catalysts_from_config(args.start_date, args.end_date, args.settings)


if __name__ == "__main__":
    main()
