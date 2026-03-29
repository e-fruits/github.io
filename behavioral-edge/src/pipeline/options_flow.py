"""
src/pipeline/options_flow.py

Fetches end-of-day options chain snapshots from Polygon.io and
aggregates them into per-ticker daily metrics.

Two output tables
-----------------
  options_snapshots   — raw per-contract rows
  options_flow_daily  — aggregated metrics used by the herd score

Design note
-----------
Not every small/mid-cap name has listed options.  All callers must
handle None / empty gracefully.  The herd score degrades to
social-only when this table has no row for a given (date, ticker).
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

_POLYGON_BASE = settings["api"]["polygon"]["base_url"]
_API_KEY = settings["api"]["polygon"]["api_key"]
_RATE_LIMIT = settings["api"]["polygon"]["rate_limit_per_minute"]
_REQUEST_GAP = 60.0 / max(_RATE_LIMIT, 1)


# ── HTTP ───────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(endpoint: str, params: dict | None = None) -> dict:
    params = params or {}
    params["apiKey"] = _API_KEY
    resp = requests.get(
        f"{_POLYGON_BASE}{endpoint}",
        params=params,
        timeout=settings["pipeline"]["request_timeout"],
    )
    resp.raise_for_status()
    time.sleep(_REQUEST_GAP)
    return resp.json()


def _paginate(endpoint: str, params: dict | None = None) -> list[dict]:
    params = params or {}
    results: list[dict] = []
    while True:
        data = _get(endpoint, params)
        results.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        cursor = next_url.split("cursor=")[-1].split("&")[0]
        params = {"cursor": cursor}
    return results


# ── Polygon options endpoints ──────────────────────────────────────────────────

def fetch_options_chain_snapshot(ticker: str) -> list[dict]:
    """
    Fetch the current options chain snapshot for *ticker*.
    Returns raw contract-level dicts from Polygon /v3/snapshot/options/{ticker}.
    """
    try:
        results = _paginate(
            f"/v3/snapshot/options/{ticker}",
            {"limit": 250},
        )
        return results
    except Exception as exc:
        log.debug("options_snapshot_failed", ticker=ticker, error=str(exc))
        return []


def fetch_options_chain_historical(ticker: str, target_date: str) -> list[dict]:
    """
    Historical options chain via /v3/snapshot/options/{ticker} with
    Polygon's as_of parameter.  Available on Starter plan and above.
    """
    try:
        results = _paginate(
            f"/v3/snapshot/options/{ticker}",
            {"limit": 250, "as_of": target_date},
        )
        return results
    except Exception as exc:
        log.debug("options_historical_failed", ticker=ticker, date=target_date, error=str(exc))
        return []


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_contract(raw: dict, ticker: str, target_date: str) -> dict | None:
    """
    Flatten a Polygon options snapshot result into our schema.
    Returns None if the raw dict is missing essential fields.
    """
    details = raw.get("details", {})
    greeks = raw.get("greeks", {})
    day = raw.get("day", {})

    expiration = details.get("expiration_date")
    strike = details.get("strike_price")
    option_type = details.get("contract_type")  # "call" or "put"

    if not all([expiration, strike is not None, option_type]):
        return None

    return {
        "date": target_date,
        "ticker": ticker,
        "expiration": expiration,
        "strike": float(strike),
        "option_type": option_type.lower(),
        "open_interest": raw.get("open_interest"),
        "volume": day.get("volume"),
        "bid": raw.get("last_quote", {}).get("bid"),
        "ask": raw.get("last_quote", {}).get("ask"),
        "implied_vol": raw.get("implied_volatility"),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_options_flow(
    contracts: list[dict],
    ticker: str,
    target_date: str,
    prior_flow: dict | None = None,
) -> dict | None:
    """
    Aggregate contract-level data into the options_flow_daily schema.

    *prior_flow* is the row from the previous day (for OI change calculation).
    Returns None if *contracts* is empty (no options data available).
    """
    if not contracts:
        return None

    df = pd.DataFrame(contracts)
    if df.empty:
        return None

    calls = df[df["option_type"] == "call"]
    puts  = df[df["option_type"] == "put"]

    total_call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else None
    total_put_vol  = int(puts["volume"].sum())  if "volume" in puts.columns  else None
    total_oi_calls = int(calls["open_interest"].sum()) if "open_interest" in calls.columns else None
    total_oi_puts  = int(puts["open_interest"].sum())  if "open_interest" in puts.columns  else None

    call_put_ratio: float | None = None
    if total_call_vol and total_put_vol:
        call_put_ratio = total_call_vol / total_put_vol if total_put_vol > 0 else None

    # OI changes vs prior day
    oi_change_calls = None
    oi_change_puts  = None
    if prior_flow:
        if total_oi_calls is not None and prior_flow.get("total_oi_calls") is not None:
            oi_change_calls = total_oi_calls - prior_flow["total_oi_calls"]
        if total_oi_puts is not None and prior_flow.get("total_oi_puts") is not None:
            oi_change_puts  = total_oi_puts - prior_flow["total_oi_puts"]

    # Max OI strikes
    max_oi_call_strike: float | None = None
    max_oi_put_strike:  float | None = None
    if not calls.empty and "open_interest" in calls.columns:
        max_oi_call_strike = float(calls.loc[calls["open_interest"].idxmax(), "strike"])
    if not puts.empty and "open_interest" in puts.columns:
        max_oi_put_strike  = float(puts.loc[puts["open_interest"].idxmax(), "strike"])

    # Unusual volume ratio is computed in post-process (needs 20-day rolling avg)
    return {
        "date": target_date,
        "ticker": ticker,
        "total_call_volume": total_call_vol,
        "total_put_volume": total_put_vol,
        "call_put_ratio": call_put_ratio,
        "total_oi_calls": total_oi_calls,
        "total_oi_puts": total_oi_puts,
        "oi_change_calls": oi_change_calls,
        "oi_change_puts": oi_change_puts,
        "unusual_volume_ratio": None,   # filled in by compute_unusual_volume_ratio()
        "max_oi_strike_call": max_oi_call_strike,
        "max_oi_strike_put": max_oi_put_strike,
        "small_lot_call_pct": None,     # requires trade-level data (Polygon Business+)
    }


def compute_unusual_volume_ratio(db_path=None) -> None:
    """
    Post-process: for every row in options_flow_daily where unusual_volume_ratio
    is NULL, compute today's total options volume / 20-day rolling average and
    write back.
    """
    conn = get_connection(db_path)
    df = fetch_df(
        conn,
        """
        SELECT date, ticker,
               COALESCE(total_call_volume, 0) + COALESCE(total_put_volume, 0) AS total_vol
          FROM options_flow_daily
         ORDER BY ticker, date
        """,
    )
    if df.empty:
        return

    df["avg_vol_20d"] = (
        df.groupby("ticker")["total_vol"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    df["unusual_volume_ratio"] = df["total_vol"] / df["avg_vol_20d"].replace(0, float("nan"))

    rows = df[["date", "ticker", "unusual_volume_ratio"]].to_dict("records")
    with get_db(db_path) as conn:
        for row in rows:
            if row["unusual_volume_ratio"] is None:
                continue
            conn.execute(
                "UPDATE options_flow_daily SET unusual_volume_ratio = ? WHERE date = ? AND ticker = ?",
                (row["unusual_volume_ratio"], row["date"], row["ticker"]),
            )
    log.info("unusual_volume_ratio_computed", rows=len(rows))


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_options_for_date(target_date: str, tickers: list[str], db_path=None) -> int:
    """
    For each ticker in *tickers*, fetch the historical options chain snapshot
    for *target_date* and store raw contracts + aggregated flow.

    Returns the number of tickers with options data found.
    """
    log.info("options_ingest_start", date=target_date, tickers=len(tickers))

    # Load prior-day flow for OI-change computation
    conn = get_connection(db_path)
    prior_date = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    prior_df = fetch_df(
        conn,
        "SELECT * FROM options_flow_daily WHERE date = ?",
        (prior_date,),
    )
    prior_map: dict[str, dict] = {}
    if not prior_df.empty:
        prior_map = prior_df.set_index("ticker").to_dict("index")

    tickers_with_data = 0
    snapshot_rows: list[dict] = []
    flow_rows: list[dict] = []

    for ticker in tickers:
        raw_contracts = fetch_options_chain_historical(ticker, target_date)
        if not raw_contracts:
            continue

        tickers_with_data += 1
        parsed = [c for c in (_parse_contract(r, ticker, target_date) for r in raw_contracts) if c]
        snapshot_rows.extend(parsed)

        flow = aggregate_options_flow(
            parsed, ticker, target_date, prior_map.get(ticker)
        )
        if flow:
            flow_rows.append(flow)

    if snapshot_rows:
        with get_db(db_path) as conn:
            upsert_rows(conn, "options_snapshots", snapshot_rows)

    if flow_rows:
        with get_db(db_path) as conn:
            upsert_rows(conn, "options_flow_daily", flow_rows)

    log.info(
        "options_ingest_done",
        date=target_date,
        tickers_with_data=tickers_with_data,
        contracts=len(snapshot_rows),
    )
    return tickers_with_data


def ingest_options_range(from_date: str, to_date: str | None = None, db_path=None) -> None:
    """
    Iterate over trading days in [from_date, to_date] and ingest options for
    the universe tickers on each day.  Runs compute_unusual_volume_ratio() at end.
    """
    start = date.fromisoformat(from_date)
    end   = date.fromisoformat(to_date) if to_date else date.today()

    conn = get_connection(db_path)
    current = start
    while current <= end:
        if current.weekday() < 5:
            ds = current.isoformat()
            # Only pull options for tickers above the volume threshold
            min_vol = settings["pipeline"].get("options_min_volume", 1_000_000)
            universe_df = fetch_df(
                conn,
                """
                SELECT u.ticker
                  FROM universe u
                  JOIN daily_prices dp ON dp.date = u.date AND dp.ticker = u.ticker
                 WHERE u.date = ?
                   AND u.is_active = 1
                   AND COALESCE(dp.avg_volume_20d, 0) >= ?
                """,
                (ds, min_vol),
            )
            tickers = universe_df["ticker"].tolist() if not universe_df.empty else []
            try:
                ingest_options_for_date(ds, tickers, db_path)
            except Exception as exc:
                log.error("options_date_failed", date=ds, error=str(exc))
        current += timedelta(days=1)

    compute_unusual_volume_ratio(db_path)
