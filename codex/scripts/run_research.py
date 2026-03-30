"""Run Phase 2 catalyst and crowding baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.research.catalyst_baseline import run_catalyst_baseline
from src.research.crowding_baseline import run_crowding_baseline
from src.utils.db import DatabaseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run catalyst and crowding research baselines.")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/catalyst_crowding.sqlite"))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--output-dir", type=Path, default=Path("data/research"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = DatabaseManager(args.db_path)
    catalyst_dir = args.output_dir / "catalyst_baseline"
    crowding_dir = args.output_dir / "crowding_baseline"

    catalyst_report = run_catalyst_baseline(
        db=db,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=catalyst_dir,
    )
    crowding_report = run_crowding_baseline(
        db=db,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=crowding_dir,
    )

    print(f"Catalyst baseline rows: {len(catalyst_report.dataset)}")
    print(f"Catalyst artifacts: {catalyst_dir}")
    print(f"Crowding baseline rows: {len(crowding_report.dataset)}")
    print(f"Crowding artifacts: {crowding_dir}")


if __name__ == "__main__":
    main()
