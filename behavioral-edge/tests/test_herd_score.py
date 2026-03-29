"""
tests/test_herd_score.py

Integration tests for the herd score computation.
Uses an in-memory SQLite database seeded with minimal fixture data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.db import get_connection, get_db, init_db, upsert_rows


@pytest.fixture
def seeded_db(tmp_path):
    """Create a DB with one universe ticker, WSB data, and options flow data."""
    p = str(tmp_path / "herd_test.db")
    init_db(p)

    with get_db(p) as conn:
        # Universe
        upsert_rows(conn, "universe", [
            {"date": "2024-01-10", "ticker": "MEME", "company_name": "Meme Corp",
             "market_cap": 1e9, "sector": "Consumer", "exchange": "NASDAQ",
             "avg_volume_20d": 2_000_000, "is_active": 1},
        ])

        # WSB mentions for normalisation window + target date
        for i, (d, count, vel, sent) in enumerate([
            ("2024-01-02", 10, 0.0,  0.1),
            ("2024-01-03", 12, 20.0, 0.2),
            ("2024-01-04", 11, -8.3, 0.1),
            ("2024-01-05", 15, 36.4, 0.3),
            ("2024-01-08", 20, 33.3, 0.5),
            ("2024-01-09", 18, -10.0, 0.4),
            ("2024-01-10", 50, 177.8, 0.8),  # target date — spike
        ]):
            upsert_rows(conn, "wsb_mentions", [{
                "date": d, "ticker": "MEME",
                "mention_count": count,
                "mention_count_24h_prior": count - 2,
                "mention_velocity": vel,
                "sentiment_score": sent,
                "sentiment_label": "bullish",
                "upvotes": count * 10,
                "rank": 5,
                "rank_change_24h": 1,
                "source": "apewisdom",
            }])

        # Options flow for normalisation window + target date
        for d, cvol, pvol, uv, oi_calls, oi_change in [
            ("2024-01-02", 1000, 800, 1.0, 5000, 0),
            ("2024-01-03", 1200, 700, 1.1, 5200, 200),
            ("2024-01-04", 900,  900, 0.9, 5100, -100),
            ("2024-01-05", 1500, 600, 1.3, 5400, 300),
            ("2024-01-08", 2000, 500, 1.8, 5800, 400),
            ("2024-01-09", 1800, 700, 1.6, 5900, 100),
            ("2024-01-10", 8000, 500, 4.5, 8000, 2100),  # target date — spike
        ]:
            upsert_rows(conn, "options_flow_daily", [{
                "date": d, "ticker": "MEME",
                "total_call_volume": cvol,
                "total_put_volume": pvol,
                "call_put_ratio": round(cvol / pvol, 3),
                "total_oi_calls": oi_calls,
                "total_oi_puts": 3000,
                "oi_change_calls": oi_change,
                "oi_change_puts": 0,
                "unusual_volume_ratio": uv,
                "max_oi_strike_call": 50.0,
                "max_oi_strike_put": 45.0,
                "small_lot_call_pct": None,
            }])

    return p


class TestHerdScoreComputation:
    def test_score_written_to_db(self, seeded_db):
        from src.pipeline.herd_score import compute_herd_scores_for_date
        n = compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)
        assert n >= 1

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT * FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'MEME'"
        ).fetchone()
        assert row is not None
        assert row["herd_score"] is not None

    def test_score_range(self, seeded_db):
        from src.pipeline.herd_score import compute_herd_scores_for_date
        compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT herd_score FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'MEME'"
        ).fetchone()
        score = row["herd_score"]
        assert 0.0 <= score <= 10.0

    def test_spike_day_high_score(self, seeded_db):
        """On the spike date the score should be meaningfully above zero."""
        from src.pipeline.herd_score import compute_herd_scores_for_date
        compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT herd_score FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'MEME'"
        ).fetchone()
        assert row["herd_score"] > 3.0

    def test_direction_stored(self, seeded_db):
        from src.pipeline.herd_score import compute_herd_scores_for_date
        compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT herd_direction FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'MEME'"
        ).fetchone()
        assert row["herd_direction"] in ("bullish", "bearish", "neutral")

    def test_component_details_json(self, seeded_db):
        from src.pipeline.herd_score import compute_herd_scores_for_date
        compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT component_details FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'MEME'"
        ).fetchone()
        details = json.loads(row["component_details"])
        assert "social_score" in details
        assert "options_score" in details
        assert details["has_options"] is True

    def test_social_only_when_no_options(self, seeded_db):
        """A ticker with no options data should still get a score (social-only path)."""
        with get_db(seeded_db) as conn:
            upsert_rows(conn, "universe", [{
                "date": "2024-01-10", "ticker": "NOOPT", "company_name": "No Options Inc",
                "market_cap": 400e6, "sector": "Tech", "exchange": "NYSE",
                "avg_volume_20d": 600_000, "is_active": 1,
            }])
            upsert_rows(conn, "wsb_mentions", [{
                "date": "2024-01-10", "ticker": "NOOPT",
                "mention_count": 30, "mention_count_24h_prior": 5,
                "mention_velocity": 500.0,
                "sentiment_score": 0.7, "sentiment_label": "bullish",
                "upvotes": 300, "rank": 10, "rank_change_24h": 5,
                "source": "apewisdom",
            }])
            # Add some historical mentions for the normalization window
            for i, d in enumerate(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]):
                upsert_rows(conn, "wsb_mentions", [{
                    "date": d, "ticker": "NOOPT",
                    "mention_count": 3 + i, "mention_count_24h_prior": 2 + i,
                    "mention_velocity": float(i * 5),
                    "sentiment_score": 0.2, "sentiment_label": "neutral",
                    "upvotes": 20, "rank": 50, "rank_change_24h": 0,
                    "source": "apewisdom",
                }])

        from src.pipeline.herd_score import compute_herd_scores_for_date
        compute_herd_scores_for_date("2024-01-10", db_path=seeded_db)

        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT herd_score, component_details FROM herd_scores WHERE date = '2024-01-10' AND ticker = 'NOOPT'"
        ).fetchone()
        assert row is not None
        details = json.loads(row["component_details"])
        assert details["has_options"] is False
        assert row["herd_score"] >= 0.0
