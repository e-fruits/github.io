"""
src/utils/db.py

SQLite database connection management and schema.
Designed for easy migration to PostgreSQL:
  - All SQL uses ANSI-compatible syntax where possible.
  - Upsert uses INSERT OR REPLACE / ON CONFLICT, which maps to
    INSERT ... ON CONFLICT DO UPDATE in PostgreSQL.
  - Replace sqlite3.connect() with psycopg2 / sqlalchemy engine
    and swap the pragma block for pg-level settings.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from src.utils.logging import get_logger

log = get_logger(__name__)

# Thread-local storage so each thread gets its own connection.
_local = threading.local()


def _get_db_path() -> Path:
    """Load database path from settings, lazily to avoid circular imports."""
    from src.utils.config import settings
    return Path(settings["database"]["path"])


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply performance and correctness pragmas."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MB
    conn.row_factory = sqlite3.Row


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Return the thread-local SQLite connection, creating it on first access.
    Caller is responsible for closing via close_connection() or using
    the get_db() context manager.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        path = Path(db_path) if db_path else _get_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
        _configure_connection(conn)
        _local.conn = conn
        log.debug("opened_db_connection", path=str(path))
    return _local.conn


def close_connection() -> None:
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


@contextmanager
def get_db(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on clean exit."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Schema ─────────────────────────────────────────────────────────────────────

DDL_STATEMENTS = [
    # ------------------------------------------------------------------
    # universe  — point-in-time stock universe (rebuilt daily)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS universe (
        date            TEXT    NOT NULL,   -- ISO-8601  YYYY-MM-DD
        ticker          TEXT    NOT NULL,
        company_name    TEXT,
        market_cap      REAL,               -- USD
        sector          TEXT,
        exchange        TEXT,
        avg_volume_20d  REAL,               -- shares
        is_active       INTEGER NOT NULL DEFAULT 1,  -- 0 = delisted on this date
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_universe_date    ON universe (date)",
    "CREATE INDEX IF NOT EXISTS idx_universe_ticker  ON universe (ticker)",

    # ------------------------------------------------------------------
    # daily_prices  — OHLCV + derived fields
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        date                TEXT    NOT NULL,
        ticker              TEXT    NOT NULL,
        open                REAL,
        high                REAL,
        low                 REAL,
        close               REAL,
        volume              INTEGER,
        vwap                REAL,
        avg_volume_20d      REAL,
        relative_volume     REAL,           -- volume / avg_volume_20d
        gap_pct             REAL,           -- (open - prior_close) / prior_close
        intraday_range_pct  REAL,           -- (high - low) / open
        prior_close         REAL,
        pre_market_high     REAL,
        pre_market_low      REAL,
        pre_market_volume   INTEGER,
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_date    ON daily_prices (date)",
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker  ON daily_prices (ticker)",
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_relvol  ON daily_prices (date, relative_volume)",

    # ------------------------------------------------------------------
    # options_snapshots  — raw per-contract end-of-day snapshot
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS options_snapshots (
        date            TEXT    NOT NULL,
        ticker          TEXT    NOT NULL,
        expiration      TEXT    NOT NULL,   -- YYYY-MM-DD
        strike          REAL    NOT NULL,
        option_type     TEXT    NOT NULL,   -- 'call' | 'put'
        open_interest   INTEGER,
        volume          INTEGER,
        bid             REAL,
        ask             REAL,
        implied_vol     REAL,
        delta           REAL,
        gamma           REAL,
        PRIMARY KEY (date, ticker, expiration, strike, option_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_opts_snap_date_ticker ON options_snapshots (date, ticker)",

    # ------------------------------------------------------------------
    # options_flow_daily  — aggregated options metrics per ticker per day
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS options_flow_daily (
        date                    TEXT    NOT NULL,
        ticker                  TEXT    NOT NULL,
        total_call_volume       INTEGER,
        total_put_volume        INTEGER,
        call_put_ratio          REAL,
        total_oi_calls          INTEGER,
        total_oi_puts           INTEGER,
        oi_change_calls         INTEGER,    -- vs prior day
        oi_change_puts          INTEGER,
        unusual_volume_ratio    REAL,       -- today total opts vol / 20-day avg
        max_oi_strike_call      REAL,
        max_oi_strike_put       REAL,
        small_lot_call_pct      REAL,       -- % volume from 1-10 contract trades
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ofd_date_ticker ON options_flow_daily (date, ticker)",

    # ------------------------------------------------------------------
    # wsb_mentions  — Reddit / WSB social data
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS wsb_mentions (
        date                    TEXT    NOT NULL,
        ticker                  TEXT    NOT NULL,
        mention_count           INTEGER,
        mention_count_24h_prior INTEGER,
        mention_velocity        REAL,       -- % change vs prior day
        sentiment_score         REAL,       -- -1.0 .. +1.0
        sentiment_label         TEXT,       -- 'bullish' | 'neutral' | 'bearish'
        upvotes                 INTEGER,
        rank                    INTEGER,    -- rank on ApeWisdom leaderboard
        rank_change_24h         INTEGER,    -- positive = moved up
        source                  TEXT,       -- 'apewisdom' | 'praw'
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wsb_date_ticker ON wsb_mentions (date, ticker)",

    # ------------------------------------------------------------------
    # catalysts  — earnings, FDA, analyst actions, news
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS catalysts (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        date                TEXT    NOT NULL,   -- date the catalyst became public
        ticker              TEXT    NOT NULL,
        catalyst_type       TEXT    NOT NULL,   -- see settings.yaml catalyst_types
        description         TEXT,
        pre_market_gap_pct  REAL,               -- filled in after open
        source              TEXT,               -- 'finnhub' | 'yfinance' | etc.
        UNIQUE (date, ticker, catalyst_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_catalysts_date_ticker ON catalysts (date, ticker)",

    # ------------------------------------------------------------------
    # herd_scores  — composite scores per ticker per day
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS herd_scores (
        date                    TEXT    NOT NULL,
        ticker                  TEXT    NOT NULL,
        herd_score              REAL,           -- 0–10
        herd_direction          TEXT,           -- 'bullish' | 'bearish' | 'neutral'
        herd_stage              TEXT,           -- 'forming' | 'saturated' | 'dispersing'
        social_component        REAL,           -- 0–1 normalized
        options_flow_component  REAL,           -- 0–1 normalized
        component_details       TEXT,           -- JSON blob
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_herd_date_ticker ON herd_scores (date, ticker)",
    "CREATE INDEX IF NOT EXISTS idx_herd_date_score  ON herd_scores (date, herd_score)",

    # ------------------------------------------------------------------
    # backtest_trades  — trade-level log
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS backtest_trades (
        trade_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              TEXT    NOT NULL,   -- links to a backtest run
        date                TEXT    NOT NULL,
        ticker              TEXT    NOT NULL,
        entry_time          TEXT,               -- HH:MM:SS
        exit_time           TEXT,
        direction           TEXT,               -- 'long' | 'short'
        entry_price         REAL,
        exit_price          REAL,
        position_type       TEXT,               -- 'equity' | 'margin' | 'debit_spread'
        leverage_multiple   REAL    DEFAULT 1.0,
        shares_or_contracts INTEGER,
        pnl_dollars         REAL,
        pnl_pct             REAL,
        stop_price          REAL,
        target_price        REAL,
        exit_reason         TEXT,   -- 'target'|'stop'|'time_stop'|'circuit_breaker'
        herd_score_at_entry REAL,
        catalyst_type       TEXT,
        strategy_signal     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bt_trades_run  ON backtest_trades (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_bt_trades_date ON backtest_trades (date, ticker)",

    # ------------------------------------------------------------------
    # backtest_daily  — daily portfolio summary
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS backtest_daily (
        run_id                  TEXT    NOT NULL,
        date                    TEXT    NOT NULL,
        starting_capital        REAL,
        ending_capital          REAL,
        daily_pnl               REAL,
        daily_return_pct        REAL,
        num_trades              INTEGER DEFAULT 0,
        num_wins                INTEGER DEFAULT 0,
        num_losses              INTEGER DEFAULT 0,
        max_drawdown_intraday   REAL,
        positions_open          INTEGER DEFAULT 0,
        PRIMARY KEY (run_id, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bt_daily_run ON backtest_daily (run_id)",

    # ------------------------------------------------------------------
    # scanner_watchlist  — pre-market watchlist per day (historical)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS scanner_watchlist (
        date                TEXT    NOT NULL,
        ticker              TEXT    NOT NULL,
        catalyst_type       TEXT,
        gap_pct             REAL,
        relative_volume     REAL,
        herd_score          REAL,
        herd_direction      TEXT,
        herd_stage          TEXT,
        market_cap          REAL,
        PRIMARY KEY (date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scan_date ON scanner_watchlist (date)",
]


def init_db(db_path: str | Path | None = None) -> None:
    """Create all tables and indexes if they don't exist."""
    with get_db(db_path) as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(stmt)
    log.info("db_initialized")


# ── Upsert helpers ─────────────────────────────────────────────────────────────

def upsert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: list[dict[str, Any]],
    *,
    serialize_json_fields: tuple[str, ...] = (),
) -> int:
    """
    INSERT OR REPLACE a list of dicts into *table*.
    Returns the number of rows affected.

    JSON fields listed in *serialize_json_fields* are automatically
    serialized to strings before insertion.
    """
    if not rows:
        return 0

    for row in rows:
        for field in serialize_json_fields:
            if field in row and not isinstance(row[field], str):
                row[field] = json.dumps(row[field])

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})"

    data = [tuple(row[c] for c in columns) for row in rows]
    cursor = conn.executemany(sql, data)
    return cursor.rowcount


def fetch_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> "pd.DataFrame":
    """Execute *sql* and return results as a pandas DataFrame."""
    import pandas as pd
    return pd.read_sql_query(sql, conn, params=params)
