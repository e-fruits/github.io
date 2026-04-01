# catalyst-crowding

Phase 1 scaffold for an intraday catalyst-and-crowding equities research platform.

This initial build includes:

- project directory structure
- configuration files and loaders
- SQLite schema with point-in-time `as_of_timestamp` enforcement
- compliance blocklist support
- a scaffolded point-in-time universe builder for historical daily universe construction

The code is intentionally conservative and audit-friendly. It is designed to make timestamp handling explicit and to keep the data pipeline modular before adding richer ingestion, signals, and backtesting logic.

## Current Scope

This phase implements only:

1. project scaffold
2. database schema
3. config loading
4. compliance blocklist handling
5. universe builder scaffold

Backtest and signal logic are intentionally not implemented yet in this pass.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --init-db
python scripts/run_pipeline.py --build-universe --start-date 2020-01-01 --end-date 2020-01-10
```

The default universe builder uses a stubbed reference-data client. Replace it with a real Massive.com (formerly Polygon.io) client before expecting production ingestion.

## Point-in-Time Principles

- Every time-varying table includes `as_of_timestamp`.
- `as_of_timestamp` means when the system could have known the data, not when the event occurred.
- The pipeline fails loudly on missing timestamps.
- The universe builder is structured to support delisted securities and historical reconstruction.

## Compliance

Tickers and sectors can be excluded through [config/blocked_tickers.yaml](/Users/eric/git_repo/codex/config/blocked_tickers.yaml). This blocklist is loaded at runtime and applied during universe construction before records are stored.
