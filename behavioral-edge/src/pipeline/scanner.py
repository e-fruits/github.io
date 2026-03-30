"""
src/pipeline/scanner.py

Generates the daily pre-market watchlist by joining universe,
catalysts, daily_prices, and herd_scores for each trading day.

Historical mode: iterates over every trading day in [start, end] and
writes the watchlist to scanner_watchlist.

Live mode (for actual pre-market use): operates on today's date,
using pre-market price data loaded by price_volume.ingest_premarket().

Filter criteria (all must pass):
  1. Ticker is in the universe for that date
  2. A catalyst exists on that date (or prior evening for AH earnings)
  3. relative_volume >= scanner.min_relative_volume  (default 3×)
  4. |gap_pct| >= scanner.min_gap_pct               (default 2%)
  5. herd_score >= scanner.min_herd_score            (default 4.0)

Output: scanner_watchlist table + optional DataFrame return.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.utils.config import settings
from src.utils.db import fetch_df, get_connection, get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_SCAN_CFG = settings["scanner"]


# ── Core query ─────────────────────────────────────────────────────────────────

_WATCHLIST_SQL = """
SELECT
    u.date,
    u.ticker,
    u.company_name,
    u.market_cap,
    u.sector,
    -- Catalyst (prefer same-day; also capture prior-evening AH earnings)
    c.catalyst_type,
    c.description      AS catalyst_description,
    -- Price
    dp.gap_pct,
    dp.relative_volume,
    dp.open,
    dp.high,
    dp.low,
    dp.close,
    dp.volume,
    dp.pre_market_high,
    dp.pre_market_low,
    dp.pre_market_volume,
    -- Herd score
    hs.herd_score,
    hs.herd_direction,
    hs.herd_stage,
    hs.social_component,
    hs.options_flow_component
FROM universe u
JOIN daily_prices dp
    ON  dp.date   = u.date
    AND dp.ticker = u.ticker
JOIN catalysts c
    ON  c.ticker  = u.ticker
    AND (
          c.date = u.date
          -- capture prior-evening after-hours earnings
          OR (
                c.date = date(u.date, '-1 day')
            AND c.catalyst_type = 'earnings'
            AND c.description LIKE '%after%'
          )
        )
LEFT JOIN herd_scores hs
    ON  hs.date   = u.date
    AND hs.ticker = u.ticker
WHERE
    u.date               = :target_date
    AND u.is_active      = 1
    AND ABS(dp.gap_pct)  >= :min_gap_pct
    AND dp.relative_volume >= :min_rel_vol
    AND COALESCE(hs.herd_score, 0) >= :min_herd_score
ORDER BY COALESCE(hs.herd_score, 0) DESC, ABS(dp.gap_pct) DESC
"""


def generate_watchlist(
    target_date: str,
    min_relative_volume: float | None = None,
    min_gap_pct: float | None = None,
    min_herd_score: float | None = None,
    db_path=None,
) -> pd.DataFrame:
    """
    Run the watchlist query for *target_date* and return a DataFrame.
    Writes results to scanner_watchlist as a side effect.
    """
    rel_vol    = min_relative_volume if min_relative_volume is not None else _SCAN_CFG["min_relative_volume"]
    gap_pct    = min_gap_pct         if min_gap_pct         is not None else _SCAN_CFG["min_gap_pct"]
    herd_score = min_herd_score      if min_herd_score      is not None else _SCAN_CFG["min_herd_score"]

    conn = get_connection(db_path)

    df = pd.read_sql_query(
        _WATCHLIST_SQL,
        conn,
        params={
            "target_date": target_date,
            "min_gap_pct": gap_pct,
            "min_rel_vol": rel_vol,
            "min_herd_score": herd_score,
        },
    )

    if df.empty:
        log.info("watchlist_empty", date=target_date)
        return df

    # Write to scanner_watchlist
    wl_rows = df[[
        "date", "ticker", "catalyst_type", "gap_pct",
        "relative_volume", "herd_score", "herd_direction",
        "herd_stage", "market_cap",
    ]].to_dict("records")

    with get_db(db_path) as conn:
        upsert_rows(conn, "scanner_watchlist", wl_rows)

    log.info("watchlist_generated", date=target_date, tickers=len(df))
    return df


def generate_watchlist_range(
    from_date: str,
    to_date: str | None = None,
    db_path=None,
) -> dict[str, pd.DataFrame]:
    """
    Generate and store watchlists for every trading day in [from_date, to_date].
    Returns a dict of {date_str: DataFrame}.
    """
    start = date.fromisoformat(from_date)
    end   = date.fromisoformat(to_date) if to_date else date.today()

    results: dict[str, pd.DataFrame] = {}
    current = start
    while current <= end:
        if current.weekday() < 5:
            ds = current.isoformat()
            try:
                df = generate_watchlist(ds, db_path=db_path)
                if not df.empty:
                    results[ds] = df
            except Exception as exc:
                log.error("watchlist_date_failed", date=ds, error=str(exc))
        current += timedelta(days=1)

    log.info("watchlist_range_done", dates=len(results))
    return results


def print_watchlist(df: pd.DataFrame, date_str: str) -> None:
    """Pretty-print a watchlist DataFrame to stdout."""
    if df.empty:
        print(f"\n[{date_str}] No watchlist candidates.\n")
        return

    print(f"\n{'='*72}")
    print(f"  WATCHLIST  {date_str}  ({len(df)} candidates)")
    print(f"{'='*72}")
    cols = [
        "ticker", "catalyst_type", "gap_pct", "relative_volume",
        "herd_score", "herd_direction", "herd_stage",
    ]
    display = df[[c for c in cols if c in df.columns]].copy()
    display["gap_pct"] = display["gap_pct"].map(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    display["relative_volume"] = display["relative_volume"].map(lambda x: f"{x:.1f}×" if pd.notna(x) else "—")
    display["herd_score"] = display["herd_score"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    print(display.to_string(index=False))
    print()


def get_live_watchlist(db_path=None) -> pd.DataFrame:
    """
    Convenience function for actual pre-market use.
    Generates watchlist for today using whatever data is already in the DB.
    Call after price_volume.ingest_premarket() has been run.
    """
    today = date.today().isoformat()
    return generate_watchlist(today, db_path=db_path)
