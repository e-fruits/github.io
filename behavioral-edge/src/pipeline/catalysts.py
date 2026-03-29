"""
src/pipeline/catalysts.py

Fetches and stores catalyst events: earnings dates, analyst actions,
and notable news.

Data sources (tried in order)
------------------------------
  1. Finnhub   — earnings calendar, analyst upgrades/downgrades
  2. yfinance  — earnings calendar fallback (free, no key required)

Once a catalyst is stored, the pre_market_gap_pct field is back-filled
from daily_prices after that session's open is known.

Output table: catalysts
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.db import fetch_df, get_connection, get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_FINNHUB_BASE = settings["api"]["finnhub"]["base_url"]
_FINNHUB_KEY  = settings["api"]["finnhub"]["api_key"]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _finnhub_get(endpoint: str, params: dict | None = None) -> Any:
    params = params or {}
    params["token"] = _FINNHUB_KEY
    resp = requests.get(
        f"{_FINNHUB_BASE}{endpoint}",
        params=params,
        timeout=settings["pipeline"]["request_timeout"],
    )
    resp.raise_for_status()
    time.sleep(0.5)  # Finnhub free tier: 60 req/min
    return resp.json()


# ── Finnhub: earnings ──────────────────────────────────────────────────────────

def fetch_finnhub_earnings(
    from_date: str,
    to_date: str,
    symbol: str | None = None,
) -> list[dict]:
    """
    Fetch earnings calendar from Finnhub.
    If *symbol* is given, fetch for that specific ticker only.
    Returns list of event dicts with at minimum: date, symbol, hour (bmo/amc).
    """
    params: dict[str, Any] = {"from": from_date, "to": to_date}
    if symbol:
        params["symbol"] = symbol
    try:
        data = _finnhub_get("/calendar/earnings", params)
        return data.get("earningsCalendar", [])
    except Exception as exc:
        log.warning("finnhub_earnings_failed", error=str(exc))
        return []


def fetch_finnhub_analyst_recommendations(ticker: str) -> list[dict]:
    """
    Fetch analyst upgrade/downgrade history from Finnhub.
    Returns list of dicts with: period, strongBuy, buy, hold, sell, strongSell.
    """
    try:
        return _finnhub_get("/stock/recommendation", {"symbol": ticker})
    except Exception as exc:
        log.debug("finnhub_recommendations_failed", ticker=ticker, error=str(exc))
        return []


def fetch_finnhub_upgrades(ticker: str) -> list[dict]:
    """
    Fetch analyst price-target upgrade/downgrade events from Finnhub.
    Returns list of dicts with: symbol, gradeTime, fromGrade, toGrade, action.
    """
    try:
        return _finnhub_get("/stock/upgrade-downgrade", {"symbol": ticker})
    except Exception as exc:
        log.debug("finnhub_upgrades_failed", ticker=ticker, error=str(exc))
        return []


# ── yfinance fallback ──────────────────────────────────────────────────────────

def fetch_yfinance_earnings(ticker: str) -> list[dict]:
    """
    Fetch earnings dates from yfinance.  Used as a fallback when Finnhub
    data is unavailable or the API key is invalid.
    Returns list of dicts: {date, ticker, catalyst_type, description, source}.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        cal = info.calendar
        if cal is None or cal.empty:
            return []
        # calendar has columns like 'Earnings Date', 'Earnings High', etc.
        results = []
        for col in cal.columns:
            if "earnings date" in col.lower():
                for dt in cal[col]:
                    if pd.isna(dt):
                        continue
                    results.append({
                        "date": pd.Timestamp(dt).date().isoformat(),
                        "ticker": ticker,
                        "catalyst_type": "earnings",
                        "description": f"Earnings (yfinance)",
                        "pre_market_gap_pct": None,
                        "source": "yfinance",
                    })
        return results
    except Exception as exc:
        log.debug("yfinance_earnings_failed", ticker=ticker, error=str(exc))
        return []


# ── Normalizers ────────────────────────────────────────────────────────────────

def _finnhub_earnings_to_rows(events: list[dict]) -> list[dict]:
    """Convert Finnhub earnings dicts to our catalysts schema."""
    rows = []
    for e in events:
        ticker = e.get("symbol")
        dt     = e.get("date")
        if not (ticker and dt):
            continue
        hour = e.get("hour", "").lower()  # "bmo" = before market open, "amc" = after market close
        desc = f"Earnings {'(pre-market)' if hour == 'bmo' else '(after-hours)' if hour == 'amc' else ''}"
        rows.append({
            "date": dt,
            "ticker": ticker,
            "catalyst_type": "earnings",
            "description": desc.strip(),
            "pre_market_gap_pct": None,
            "source": "finnhub",
        })
    return rows


def _finnhub_upgrades_to_rows(events: list[dict], ticker: str) -> list[dict]:
    """Convert Finnhub upgrade/downgrade events to our catalysts schema."""
    rows = []
    for e in events:
        action = e.get("action", "").lower()
        if action not in ("upgrade", "downgrade", "init", "reiterated"):
            continue
        dt_raw = e.get("gradeTime")
        if not dt_raw:
            continue
        # gradeTime can be epoch or ISO string
        try:
            if isinstance(dt_raw, (int, float)):
                dt = date.fromtimestamp(dt_raw).isoformat()
            else:
                dt = pd.Timestamp(dt_raw).date().isoformat()
        except Exception:
            continue

        catalyst_type = "analyst_upgrade" if action == "upgrade" else "analyst_downgrade"
        from_g = e.get("fromGrade", "")
        to_g   = e.get("toGrade", "")
        company = e.get("company", "")
        desc = f"{company}: {from_g} → {to_g}" if from_g and to_g else f"{company}: {to_g}"

        rows.append({
            "date": dt,
            "ticker": ticker,
            "catalyst_type": catalyst_type,
            "description": desc.strip(),
            "pre_market_gap_pct": None,
            "source": "finnhub",
        })
    return rows


# ── Gap back-fill ──────────────────────────────────────────────────────────────

def backfill_gap_pct(db_path=None) -> int:
    """
    For every row in *catalysts* where pre_market_gap_pct IS NULL,
    look up the daily_prices gap_pct on the same date and fill it in.
    Returns number of rows updated.
    """
    conn = get_connection(db_path)
    missing = fetch_df(
        conn,
        "SELECT id, date, ticker FROM catalysts WHERE pre_market_gap_pct IS NULL",
    )
    if missing.empty:
        return 0

    updated = 0
    with get_db(db_path) as conn:
        for _, row in missing.iterrows():
            price_row = conn.execute(
                "SELECT gap_pct FROM daily_prices WHERE date = ? AND ticker = ?",
                (row["date"], row["ticker"]),
            ).fetchone()
            if price_row and price_row["gap_pct"] is not None:
                conn.execute(
                    "UPDATE catalysts SET pre_market_gap_pct = ? WHERE id = ?",
                    (price_row["gap_pct"], row["id"]),
                )
                updated += 1

    log.info("gap_pct_backfilled", updated=updated)
    return updated


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_earnings_range(from_date: str, to_date: str, db_path=None) -> int:
    """
    Fetch all earnings events from Finnhub for [from_date, to_date]
    (single bulk call) and store them.

    For individual tickers not returned by the bulk call, fall back to yfinance.
    Returns total rows written.
    """
    log.info("earnings_ingest_start", from_date=from_date, to_date=to_date)

    # Finnhub bulk earnings call
    events = fetch_finnhub_earnings(from_date, to_date)
    rows = _finnhub_earnings_to_rows(events)

    # Filter to our universe tickers
    conn = get_connection(db_path)
    universe_tickers_df = fetch_df(
        conn,
        "SELECT DISTINCT ticker FROM universe WHERE is_active = 1",
    )
    if not universe_tickers_df.empty:
        universe_tickers: set[str] = set(universe_tickers_df["ticker"].tolist())
        rows = [r for r in rows if r["ticker"] in universe_tickers]

    with get_db(db_path) as conn:
        upsert_rows(conn, "catalysts", rows)

    log.info("earnings_ingest_done", rows=len(rows))
    backfill_gap_pct(db_path)
    return len(rows)


def ingest_analyst_actions(tickers: list[str], db_path=None) -> int:
    """
    Fetch analyst upgrade/downgrade history for each ticker in *tickers*
    and store in catalysts.  Returns total rows written.
    """
    log.info("analyst_actions_start", tickers=len(tickers))
    total = 0
    for ticker in tickers:
        events = fetch_finnhub_upgrades(ticker)
        rows = _finnhub_upgrades_to_rows(events, ticker)
        if rows:
            with get_db(db_path) as conn:
                upsert_rows(conn, "catalysts", rows)
            total += len(rows)
    log.info("analyst_actions_done", rows=total)
    return total


def ingest_all_catalysts(
    from_date: str,
    to_date: str | None = None,
    db_path=None,
) -> None:
    """
    Master entry point: run earnings + analyst actions for the full date range.
    """
    to = to_date or date.today().isoformat()
    ingest_earnings_range(from_date, to, db_path)

    # Analyst actions: pull for all current universe tickers
    conn = get_connection(db_path)
    tickers_df = fetch_df(conn, "SELECT DISTINCT ticker FROM universe WHERE is_active = 1")
    if not tickers_df.empty:
        ingest_analyst_actions(tickers_df["ticker"].tolist(), db_path)

    backfill_gap_pct(db_path)
