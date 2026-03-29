"""
src/pipeline/herd_score.py

Computes the composite herd score for each (date, ticker) pair.

Score architecture (V1 — weighted sum, no ML)
─────────────────────────────────────────────
All raw inputs are normalized to [0, 1] using a rolling
percentile window (*normalization_window* days, default 90).
This makes the score robust to structural shifts in the underlying
signal magnitudes over time.

  social_score = (
      0.40 × normalize(mention_velocity)
    + 0.30 × normalize(mention_count_vs_baseline)
    + 0.30 × normalize(|sentiment_score|)
  )

  options_score = (
      0.40 × normalize(unusual_volume_ratio)
    + 0.30 × normalize(call_put_ratio_deviation)   # abs deviation from 1.0
    + 0.30 × normalize(oi_accumulation_rate)        # oi_change_calls / total_oi_calls
  )

  # Options are weighted higher — actions > words
  herd_score = 10 × (weight_social × social_score + weight_options × options_score)

  # Default weights: social=0.40, options=0.60

When a ticker has no options data, the formula degrades cleanly:
  herd_score = 10 × social_score

Direction (bullish / bearish / neutral):
  sentiment_score > 0.05 AND call_put_ratio > 1.0 → bullish
  sentiment_score < -0.05 OR call_put_ratio < 0.8  → bearish
  else → neutral

Stage (forming / saturated / dispersing):
  mention_velocity > stage_forming_velocity_min     → forming
  mention_velocity < stage_saturated_velocity_max   → dispersing
  else                                               → saturated

Output table: herd_scores
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.db import fetch_df, get_connection, get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_CFG = settings["herd_score"]
_NORM_WINDOW = _CFG.get("normalization_window", 90)

# ── Normalization ──────────────────────────────────────────────────────────────

def rolling_percentile_normalize(series: pd.Series, window: int) -> pd.Series:
    """
    For each element x in *series*, compute its percentile rank within the
    preceding *window* values (exclusive of x itself).  Returns a [0, 1]
    series.  Uses shift(1) so there is no look-ahead contamination.
    """
    def _percentile_rank(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.all(np.isnan(arr)):
            return 0.5  # default to neutral when insufficient history
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return 0.5
        return float(np.mean(valid <= valid[-1]))

    # Use a rolling apply on the shifted series
    shifted = series.shift(1)
    result = shifted.rolling(window, min_periods=5).apply(
        lambda w: _percentile_rank(w.values), raw=True
    )
    # Fill early rows with 0.5 (neutral)
    return result.fillna(0.5)


# ── Direction and stage helpers ────────────────────────────────────────────────

def _compute_direction(
    sentiment_score: float | None,
    call_put_ratio: float | None,
) -> str:
    s = sentiment_score or 0.0
    cpr = call_put_ratio or 1.0

    if s > 0.05 and cpr >= 1.0:
        return "bullish"
    if s < -0.05 or cpr < 0.8:
        return "bearish"
    return "neutral"


def _compute_stage(mention_velocity: float | None, oi_change_calls: int | None) -> str:
    v = mention_velocity or 0.0
    forming_min = _CFG.get("stage_forming_velocity_min", 0.20) * 100  # stored as %, config as ratio
    saturated_max = _CFG.get("stage_saturated_velocity_max", -0.10) * 100

    if v > forming_min:
        return "forming"
    if v < saturated_max:
        return "dispersing"
    return "saturated"


# ── Core computation ───────────────────────────────────────────────────────────

def compute_herd_scores_for_date(target_date: str, db_path=None) -> int:
    """
    Compute herd scores for all universe tickers on *target_date*.
    Writes results to the herd_scores table.
    Returns the number of rows written.
    """
    log.info("herd_score_compute_start", date=target_date)

    conn = get_connection(db_path)

    # Load a rolling window of social + options data ending on target_date
    window_start = (date.fromisoformat(target_date) - timedelta(days=_NORM_WINDOW * 2)).isoformat()

    social_df = fetch_df(
        conn,
        """
        SELECT date, ticker, mention_count, mention_velocity,
               sentiment_score, upvotes
          FROM wsb_mentions
         WHERE date BETWEEN ? AND ?
         ORDER BY ticker, date
        """,
        (window_start, target_date),
    )

    options_df = fetch_df(
        conn,
        """
        SELECT date, ticker, call_put_ratio, unusual_volume_ratio,
               oi_change_calls, total_oi_calls, total_oi_puts
          FROM options_flow_daily
         WHERE date BETWEEN ? AND ?
         ORDER BY ticker, date
        """,
        (window_start, target_date),
    )

    # Universe tickers for the target date
    universe_df = fetch_df(
        conn,
        "SELECT ticker FROM universe WHERE date = ? AND is_active = 1",
        (target_date,),
    )
    if universe_df.empty:
        log.warning("no_universe", date=target_date)
        return 0

    universe_tickers: set[str] = set(universe_df["ticker"].tolist())

    # ── Precompute rolling normalizers on the full window data ──────────────

    # Social signals
    if not social_df.empty:
        social_df["mention_velocity"] = pd.to_numeric(social_df["mention_velocity"], errors="coerce").fillna(0.0)
        social_df["sentiment_score"]  = pd.to_numeric(social_df["sentiment_score"],  errors="coerce").fillna(0.0)
        social_df["mention_count"]    = pd.to_numeric(social_df["mention_count"],     errors="coerce").fillna(0.0)

        # Rolling baseline for mention_count (20-day avg of prior days)
        social_df = social_df.sort_values(["ticker", "date"])
        social_df["mention_baseline"] = (
            social_df.groupby("ticker")["mention_count"]
            .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean().fillna(s.mean()))
        )
        social_df["mention_vs_baseline"] = (
            social_df["mention_count"] / social_df["mention_baseline"].replace(0, float("nan"))
        ).fillna(1.0)

        for ticker, grp in social_df.groupby("ticker"):
            social_df.loc[grp.index, "norm_velocity"]       = rolling_percentile_normalize(grp["mention_velocity"], _NORM_WINDOW)
            social_df.loc[grp.index, "norm_vs_baseline"]    = rolling_percentile_normalize(grp["mention_vs_baseline"], _NORM_WINDOW)
            social_df.loc[grp.index, "norm_sentiment_mag"]  = rolling_percentile_normalize(grp["sentiment_score"].abs(), _NORM_WINDOW)

    # Options signals
    if not options_df.empty:
        options_df = options_df.sort_values(["ticker", "date"])
        options_df["cpr_deviation"] = (options_df["call_put_ratio"] - 1.0).abs().fillna(0.0)
        options_df["oi_accum_rate"] = (
            options_df["oi_change_calls"] / options_df["total_oi_calls"].replace(0, float("nan"))
        ).fillna(0.0)
        options_df["unusual_volume_ratio"] = pd.to_numeric(options_df["unusual_volume_ratio"], errors="coerce").fillna(0.0)

        for ticker, grp in options_df.groupby("ticker"):
            options_df.loc[grp.index, "norm_unusual_vol"]  = rolling_percentile_normalize(grp["unusual_volume_ratio"], _NORM_WINDOW)
            options_df.loc[grp.index, "norm_cpr_dev"]      = rolling_percentile_normalize(grp["cpr_deviation"], _NORM_WINDOW)
            options_df.loc[grp.index, "norm_oi_accum"]     = rolling_percentile_normalize(grp["oi_accum_rate"], _NORM_WINDOW)

    # ── Extract target-date rows ────────────────────────────────────────────

    social_today  = social_df[social_df["date"] == target_date].set_index("ticker") if not social_df.empty else pd.DataFrame()
    options_today = options_df[options_df["date"] == target_date].set_index("ticker") if not options_df.empty else pd.DataFrame()

    w_social  = _CFG.get("weight_social", 0.40)
    w_options = _CFG.get("weight_options", 0.60)

    sw = {
        "velocity":    _CFG.get("social_mention_velocity", 0.40),
        "vs_baseline": _CFG.get("social_mention_count_vs_baseline", 0.30),
        "sentiment":   _CFG.get("social_sentiment_magnitude", 0.30),
    }
    ow = {
        "unusual_vol": _CFG.get("options_unusual_volume", 0.40),
        "cpr_dev":     _CFG.get("options_call_put_deviation", 0.30),
        "oi_accum":    _CFG.get("options_oi_accumulation", 0.30),
    }

    rows: list[dict] = []

    for ticker in universe_tickers:
        s = social_today.loc[ticker] if ticker in social_today.index else None
        o = options_today.loc[ticker] if ticker in options_today.index else None

        # ── Social component ────────────────────────────────────────────────
        if s is not None:
            social_score = (
                sw["velocity"]    * float(s.get("norm_velocity", 0.5))
              + sw["vs_baseline"] * float(s.get("norm_vs_baseline", 0.5))
              + sw["sentiment"]   * float(s.get("norm_sentiment_mag", 0.5))
            )
            sentiment_score_raw = float(s.get("sentiment_score", 0.0))
            mention_velocity_raw = float(s.get("mention_velocity", 0.0))
        else:
            social_score = 0.0
            sentiment_score_raw = 0.0
            mention_velocity_raw = 0.0

        # ── Options component ────────────────────────────────────────────────
        has_options = o is not None
        if has_options:
            options_score = (
                ow["unusual_vol"] * float(o.get("norm_unusual_vol", 0.5))
              + ow["cpr_dev"]     * float(o.get("norm_cpr_dev", 0.5))
              + ow["oi_accum"]    * float(o.get("norm_oi_accum", 0.5))
            )
            cpr_raw        = float(o.get("call_put_ratio") or 1.0)
            oi_change_calls = o.get("oi_change_calls")
        else:
            options_score  = 0.0
            cpr_raw        = 1.0
            oi_change_calls = None

        # ── Composite score ─────────────────────────────────────────────────
        if has_options:
            herd_score = 10.0 * (w_social * social_score + w_options * options_score)
        else:
            # Degrade to social-only
            herd_score = 10.0 * social_score

        herd_score = round(min(max(herd_score, 0.0), 10.0), 3)

        direction = _compute_direction(sentiment_score_raw, cpr_raw if has_options else None)
        stage     = _compute_stage(mention_velocity_raw, oi_change_calls)

        details: dict[str, Any] = {
            "social_score": round(social_score, 4),
            "options_score": round(options_score, 4),
            "has_options": has_options,
            "norm_velocity": round(float(s.get("norm_velocity", 0.5)), 4) if s is not None else None,
            "norm_vs_baseline": round(float(s.get("norm_vs_baseline", 0.5)), 4) if s is not None else None,
            "norm_sentiment_mag": round(float(s.get("norm_sentiment_mag", 0.5)), 4) if s is not None else None,
            "norm_unusual_vol": round(float(o.get("norm_unusual_vol", 0.5)), 4) if has_options else None,
            "norm_cpr_dev": round(float(o.get("norm_cpr_dev", 0.5)), 4) if has_options else None,
            "norm_oi_accum": round(float(o.get("norm_oi_accum", 0.5)), 4) if has_options else None,
        }

        rows.append({
            "date": target_date,
            "ticker": ticker,
            "herd_score": herd_score,
            "herd_direction": direction,
            "herd_stage": stage,
            "social_component": round(social_score, 4),
            "options_flow_component": round(options_score, 4),
            "component_details": details,   # serialized by upsert_rows
        })

    with get_db(db_path) as conn:
        upsert_rows(conn, "herd_scores", rows, serialize_json_fields=("component_details",))

    log.info("herd_score_done", date=target_date, tickers=len(rows))
    return len(rows)


def compute_herd_scores_range(
    from_date: str,
    to_date: str | None = None,
    db_path=None,
) -> None:
    """Compute herd scores for every trading day in [from_date, to_date]."""
    start = date.fromisoformat(from_date)
    end   = date.fromisoformat(to_date) if to_date else date.today()

    current = start
    while current <= end:
        if current.weekday() < 5:
            try:
                compute_herd_scores_for_date(current.isoformat(), db_path)
            except Exception as exc:
                log.error("herd_score_date_failed", date=current.isoformat(), error=str(exc))
        current += timedelta(days=1)
