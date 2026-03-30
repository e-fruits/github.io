"""Scanner CLI wrapper."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

MIN_PYTHON = (3, 11)

if sys.version_info < MIN_PYTHON:
    required = ".".join(str(part) for part in MIN_PYTHON)
    current = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        f"Python {required}+ is required for catalyst-crowding. "
        f"Current interpreter: Python {current}. "
        "Create and activate a Python 3.11 virtualenv, then reinstall requirements."
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.scanner import run_scanner_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical watchlist generation.")
    parser.add_argument("--settings", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_scanner_from_config(args.date, args.settings)
    print(f"watchlist_rows={count}")


if __name__ == "__main__":
    main()
