"""SQLite connection management and schema creation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS universe (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    market_cap REAL NOT NULL,
    sector TEXT,
    exchange TEXT NOT NULL,
    avg_volume_20d REAL NOT NULL,
    avg_dollar_volume_20d REAL NOT NULL,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    vwap REAL,
    dollar_volume REAL,
    relative_volume_20d REAL,
    gap_pct REAL,
    intraday_range_pct REAL,
    pre_market_high REAL,
    pre_market_low REAL,
    pre_market_volume REAL,
    pre_market_dollar_volume REAL,
    pre_market_relative_volume REAL,
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS options_flow_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    total_call_volume REAL,
    total_put_volume REAL,
    call_put_ratio REAL,
    total_call_oi REAL,
    total_put_oi REAL,
    oi_change_calls REAL,
    oi_change_puts REAL,
    unusual_volume_ratio REAL,
    small_lot_call_volume REAL,
    small_lot_put_volume REAL,
    small_lot_call_pct REAL,
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS wsb_mentions (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    mention_count REAL,
    mention_count_prior_day REAL,
    mention_velocity_pct REAL,
    mention_vs_baseline REAL,
    sentiment_score REAL,
    sentiment_unanimity REAL,
    upvotes REAL,
    rank REAL,
    rank_change_24h REAL,
    as_of_timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (date, ticker, source)
);

CREATE TABLE IF NOT EXISTS catalysts (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    catalyst_bucket TEXT NOT NULL,
    catalyst_detail TEXT NOT NULL,
    catalyst_direction TEXT NOT NULL,
    pre_market_gap_pct REAL,
    source TEXT NOT NULL,
    source_timestamp TEXT NOT NULL,
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker, catalyst_bucket, catalyst_detail, source_timestamp)
);

CREATE TABLE IF NOT EXISTS short_interest (
    report_date TEXT NOT NULL,
    settlement_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    short_interest_shares REAL,
    shares_outstanding REAL,
    float_shares REAL,
    short_pct_float REAL,
    days_to_cover REAL,
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (report_date, ticker)
);

CREATE TABLE IF NOT EXISTS crowding_index (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    attention TEXT NOT NULL,
    positioning_pressure TEXT NOT NULL,
    crowd_trap_risk TEXT NOT NULL,
    attention_score REAL NOT NULL,
    positioning_score REAL NOT NULL,
    trap_score REAL NOT NULL,
    component_details TEXT NOT NULL,
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS watchlist (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    catalyst_bucket TEXT NOT NULL,
    catalyst_direction TEXT NOT NULL,
    catalyst_detail TEXT NOT NULL,
    pre_market_gap_pct REAL,
    pre_market_relative_volume REAL,
    pre_market_dollar_volume REAL,
    attention TEXT,
    positioning_pressure TEXT,
    crowd_trap_risk TEXT,
    short_interest_pct_float REAL,
    days_to_cover REAL,
    suggested_setup TEXT NOT NULL,
    setup_score REAL,
    short_tradable INTEGER NOT NULL CHECK (short_tradable IN (0, 1)),
    as_of_timestamp TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    catalyst_bucket TEXT,
    catalyst_direction TEXT,
    attention TEXT,
    positioning_pressure TEXT,
    crowd_trap_risk TEXT,
    direction TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_slippage REAL,
    exit_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_slippage REAL,
    stop_price REAL,
    stop_type TEXT,
    target_price REAL,
    shares REAL NOT NULL,
    position_value REAL NOT NULL,
    risk_dollars REAL,
    risk_pct_capital REAL,
    pnl_dollars REAL,
    pnl_pct REAL,
    exit_reason TEXT NOT NULL,
    market_regime TEXT,
    as_of_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_daily (
    date TEXT PRIMARY KEY,
    starting_capital REAL NOT NULL,
    ending_capital REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    daily_return_pct REAL NOT NULL,
    num_trades INTEGER NOT NULL,
    num_wins INTEGER NOT NULL,
    num_losses INTEGER NOT NULL,
    max_intraday_drawdown REAL,
    positions_open_at_close INTEGER NOT NULL,
    cumulative_return REAL,
    peak_capital REAL,
    drawdown_from_peak REAL,
    market_regime TEXT
);

CREATE INDEX IF NOT EXISTS idx_universe_ticker_date ON universe (ticker, date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices (ticker, date);
CREATE INDEX IF NOT EXISTS idx_catalysts_ticker_date ON catalysts (ticker, date);
CREATE INDEX IF NOT EXISTS idx_short_interest_ticker_report_date ON short_interest (ticker, report_date);
"""


class DatabaseManager:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.commit()

    def upsert_universe(self, rows: Iterable[dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO universe (
                    date, ticker, company_name, market_cap, sector, exchange,
                    avg_volume_20d, avg_dollar_volume_20d, is_active, as_of_timestamp
                ) VALUES (
                    :date, :ticker, :company_name, :market_cap, :sector, :exchange,
                    :avg_volume_20d, :avg_dollar_volume_20d, :is_active, :as_of_timestamp
                )
                ON CONFLICT(date, ticker) DO UPDATE SET
                    company_name = excluded.company_name,
                    market_cap = excluded.market_cap,
                    sector = excluded.sector,
                    exchange = excluded.exchange,
                    avg_volume_20d = excluded.avg_volume_20d,
                    avg_dollar_volume_20d = excluded.avg_dollar_volume_20d,
                    is_active = excluded.is_active,
                    as_of_timestamp = excluded.as_of_timestamp
                """,
                rows,
            )
            connection.commit()

    def fetch_universe_tickers(self, start_date: str, end_date: str) -> list[str]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT DISTINCT ticker
                FROM universe
                WHERE date BETWEEN ? AND ?
                ORDER BY ticker
                """,
                (start_date, end_date),
            )
            return [str(row["ticker"]) for row in cursor.fetchall()]

    def upsert_daily_prices(self, rows: Iterable[dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO daily_prices (
                    date, ticker, open, high, low, close, volume, vwap,
                    dollar_volume, relative_volume_20d, gap_pct, intraday_range_pct,
                    pre_market_high, pre_market_low, pre_market_volume,
                    pre_market_dollar_volume, pre_market_relative_volume, as_of_timestamp
                ) VALUES (
                    :date, :ticker, :open, :high, :low, :close, :volume, :vwap,
                    :dollar_volume, :relative_volume_20d, :gap_pct, :intraday_range_pct,
                    :pre_market_high, :pre_market_low, :pre_market_volume,
                    :pre_market_dollar_volume, :pre_market_relative_volume, :as_of_timestamp
                )
                ON CONFLICT(date, ticker) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    vwap = excluded.vwap,
                    dollar_volume = excluded.dollar_volume,
                    relative_volume_20d = excluded.relative_volume_20d,
                    gap_pct = excluded.gap_pct,
                    intraday_range_pct = excluded.intraday_range_pct,
                    pre_market_high = excluded.pre_market_high,
                    pre_market_low = excluded.pre_market_low,
                    pre_market_volume = excluded.pre_market_volume,
                    pre_market_dollar_volume = excluded.pre_market_dollar_volume,
                    pre_market_relative_volume = excluded.pre_market_relative_volume,
                    as_of_timestamp = excluded.as_of_timestamp
                """,
                rows,
            )
            connection.commit()

    def fetch_daily_gap_pct(self, ticker: str, trade_date: str) -> float | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT gap_pct
                FROM daily_prices
                WHERE ticker = ? AND date = ?
                """,
                (ticker, trade_date),
            ).fetchone()
            return None if row is None else row["gap_pct"]

    def upsert_catalysts(self, rows: Iterable[dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO catalysts (
                    date, ticker, catalyst_bucket, catalyst_detail, catalyst_direction,
                    pre_market_gap_pct, source, source_timestamp, as_of_timestamp
                ) VALUES (
                    :date, :ticker, :catalyst_bucket, :catalyst_detail, :catalyst_direction,
                    :pre_market_gap_pct, :source, :source_timestamp, :as_of_timestamp
                )
                ON CONFLICT(date, ticker, catalyst_bucket, catalyst_detail, source_timestamp) DO UPDATE SET
                    catalyst_direction = excluded.catalyst_direction,
                    pre_market_gap_pct = excluded.pre_market_gap_pct,
                    source = excluded.source,
                    as_of_timestamp = excluded.as_of_timestamp
                """,
                rows,
            )
            connection.commit()
