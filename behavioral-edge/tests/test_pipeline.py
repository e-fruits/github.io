"""
tests/test_pipeline.py

Unit tests for the data pipeline modules.
Uses an in-memory SQLite database — no API calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.utils.db import get_db, init_db, upsert_rows, fetch_df, get_connection


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "test.db")
    init_db(p)
    return p


# ── Database schema tests ──────────────────────────────────────────────────────

class TestSchema:
    def test_tables_created(self, db_path):
        conn = get_connection(db_path)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "universe", "daily_prices", "options_snapshots", "options_flow_daily",
            "wsb_mentions", "catalysts", "herd_scores", "backtest_trades",
            "backtest_daily", "scanner_watchlist",
        }
        assert expected.issubset(tables)

    def test_upsert_rows(self, db_path):
        rows = [
            {"date": "2024-01-02", "ticker": "AAPL", "company_name": "Apple", "market_cap": 3e12,
             "sector": "Tech", "exchange": "NASDAQ", "avg_volume_20d": 60_000_000, "is_active": 1},
        ]
        with get_db(db_path) as conn:
            n = upsert_rows(conn, "universe", rows)
        assert n >= 0  # sqlite3 rowcount can be -1 for executemany

        conn = get_connection(db_path)
        result = conn.execute("SELECT * FROM universe WHERE ticker = 'AAPL'").fetchone()
        assert result is not None
        assert result["ticker"] == "AAPL"

    def test_upsert_idempotent(self, db_path):
        rows = [
            {"date": "2024-01-02", "ticker": "MSFT", "company_name": "Microsoft",
             "market_cap": 3e12, "sector": "Tech", "exchange": "NASDAQ",
             "avg_volume_20d": 25_000_000, "is_active": 1},
        ]
        with get_db(db_path) as conn:
            upsert_rows(conn, "universe", rows)
        rows[0]["company_name"] = "Microsoft Corp"
        with get_db(db_path) as conn:
            upsert_rows(conn, "universe", rows)

        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM universe WHERE ticker='MSFT'").fetchone()[0]
        assert count == 1


# ── price_volume derived metrics ───────────────────────────────────────────────

class TestPriceVolumeDerived:
    def test_compute_derived(self):
        from src.pipeline.price_volume import compute_derived

        data = {
            "date":   ["2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["TEST", "TEST", "TEST"],
            "open":   [100.0, 102.0, 99.0],
            "high":   [105.0, 106.0, 103.0],
            "low":    [99.0,  101.0, 97.0],
            "close":  [103.0, 104.0, 101.0],
            "volume": [1_000_000, 1_500_000, 800_000],
            "vwap":   [102.0, 103.5, 100.0],
            "avg_volume_20d": [None, None, None],
            "relative_volume": [None, None, None],
            "gap_pct": [None, None, None],
            "intraday_range_pct": [None, None, None],
            "prior_close": [None, None, None],
        }
        df = pd.DataFrame(data)
        result = compute_derived(df)

        # Second row should have prior_close from first row
        assert result.iloc[1]["prior_close"] == pytest.approx(103.0)

        # gap_pct for second row: (102 - 103) / 103
        expected_gap = (102.0 - 103.0) / 103.0
        assert result.iloc[1]["gap_pct"] == pytest.approx(expected_gap, rel=1e-4)

        # intraday_range_pct: (high - low) / open
        expected_range = (106.0 - 101.0) / 102.0
        assert result.iloc[1]["intraday_range_pct"] == pytest.approx(expected_range, rel=1e-4)

    def test_relative_volume_requires_history(self):
        from src.pipeline.price_volume import compute_derived

        # With only 1 row, relative_volume should be NaN (insufficient history)
        data = {
            "date": ["2024-01-02"], "ticker": ["TEST"],
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1_000_000], "vwap": [102.0],
            "avg_volume_20d": [None], "relative_volume": [None],
            "gap_pct": [None], "intraday_range_pct": [None], "prior_close": [None],
        }
        df = compute_derived(pd.DataFrame(data))
        assert pd.isna(df.iloc[0]["relative_volume"])


# ── WSB sentiment tests ────────────────────────────────────────────────────────

class TestWSBSentiment:
    def test_extract_tickers_dollar_sign(self):
        from src.pipeline.wsb_sentiment import extract_tickers
        text = "bought $AAPL and $TSLA puts today"
        tickers = extract_tickers(text)
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_extract_tickers_filters_common_words(self):
        from src.pipeline.wsb_sentiment import extract_tickers
        text = "I AM SO BULLISH ON AAPL"
        tickers = extract_tickers(text)
        assert "I" not in tickers
        assert "AM" not in tickers
        assert "SO" not in tickers

    def test_extract_tickers_known_set(self):
        from src.pipeline.wsb_sentiment import extract_tickers
        known = {"AAPL", "TSLA"}
        text = "$AAPL and $MSFT both ripping"
        tickers = extract_tickers(text, known_tickers=known)
        assert "AAPL" in tickers
        assert "MSFT" not in tickers  # not in known set

    def test_vader_sentiment(self):
        from src.pipeline.wsb_sentiment import score_texts, sentiment_label
        scores = score_texts(["This stock is absolutely incredible and amazing!"])
        assert len(scores) == 1
        assert scores[0] > 0.05
        assert sentiment_label(scores[0]) == "bullish"

        scores_neg = score_texts(["This is terrible, going to zero, complete disaster"])
        assert scores_neg[0] < -0.05
        assert sentiment_label(scores_neg[0]) == "bearish"

    def test_sentiment_label_neutral(self):
        from src.pipeline.wsb_sentiment import sentiment_label
        assert sentiment_label(0.0) == "neutral"
        assert sentiment_label(0.04) == "neutral"
        assert sentiment_label(-0.04) == "neutral"


# ── Catalysts ──────────────────────────────────────────────────────────────────

class TestCatalysts:
    def test_finnhub_earnings_to_rows(self):
        from src.pipeline.catalysts import _finnhub_earnings_to_rows
        events = [
            {"symbol": "NVDA", "date": "2024-02-21", "hour": "amc"},
            {"symbol": "TSLA", "date": "2024-01-24", "hour": "bmo"},
            {"symbol": None,   "date": "2024-01-01", "hour": ""},  # should be skipped
        ]
        rows = _finnhub_earnings_to_rows(events)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "NVDA"
        assert rows[0]["catalyst_type"] == "earnings"
        assert "after" in rows[0]["description"].lower()
        assert "pre" in rows[1]["description"].lower()


# ── Herd score ─────────────────────────────────────────────────────────────────

class TestHerdScore:
    def test_rolling_percentile_normalize(self):
        from src.pipeline.herd_score import rolling_percentile_normalize
        import pandas as pd

        s = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 2.0])
        result = rolling_percentile_normalize(s, window=20)
        # Early values (< 5) should return 0.5 (neutral)
        assert result.iloc[0] == pytest.approx(0.5)
        # The spike at index 4 (10.0) should have a high percentile
        # (it is above all prior values in window)
        assert result.iloc[4] == pytest.approx(1.0)

    def test_direction_bullish(self):
        from src.pipeline.herd_score import _compute_direction
        assert _compute_direction(0.6, 1.5) == "bullish"

    def test_direction_bearish(self):
        from src.pipeline.herd_score import _compute_direction
        assert _compute_direction(-0.3, 0.7) == "bearish"

    def test_direction_neutral(self):
        from src.pipeline.herd_score import _compute_direction
        assert _compute_direction(0.0, 1.0) == "neutral"

    def test_stage_forming(self):
        from src.pipeline.herd_score import _compute_stage
        # 25% velocity (config threshold is 20% = 20.0 as stored pct)
        assert _compute_stage(25.0, None) == "forming"

    def test_stage_dispersing(self):
        from src.pipeline.herd_score import _compute_stage
        assert _compute_stage(-15.0, None) == "dispersing"

    def test_stage_saturated(self):
        from src.pipeline.herd_score import _compute_stage
        assert _compute_stage(5.0, None) == "saturated"
