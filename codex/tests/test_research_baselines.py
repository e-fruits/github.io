from pathlib import Path

from src.research.catalyst_baseline import load_catalyst_day_dataset, run_catalyst_baseline
from src.research.crowding_baseline import run_crowding_baseline
from src.utils.db import DatabaseManager


def test_catalyst_baseline_builds_dataset_and_exports(tmp_path: Path) -> None:
    db = _seed_research_db(tmp_path / "research.sqlite")

    report = run_catalyst_baseline(
        db=db,
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_dir=tmp_path / "artifacts",
    )

    assert len(report.dataset) == 5
    assert set(report.dataset["gap_size_bucket"]) == {"2-5%", "5-10%"}

    earnings_positive = report.tables["by_bucket_direction"]
    earnings_positive = earnings_positive[
        (earnings_positive["catalyst_bucket"] == "earnings")
        & (earnings_positive["catalyst_direction"] == "positive")
    ].iloc[0]

    assert earnings_positive["sample_size"] == 4
    assert earnings_positive["win_rate"] == 0.5
    assert "avg_directional_return_by_bucket" in report.charts
    assert report.exported_files["summary"].exists()


def test_crowding_baseline_compares_states_and_lagged_attention(tmp_path: Path) -> None:
    db = _seed_research_db(tmp_path / "research.sqlite")

    dataset = load_catalyst_day_dataset(db, start_date="2024-01-01", end_date="2024-12-31")
    assert list(dataset.loc[dataset["ticker"] == "AAA", "lagged_attention"].dropna()) == ["low", "high", "low"]

    report = run_crowding_baseline(
        db=db,
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_dir=tmp_path / "artifacts",
    )

    attention_tests = report.tables["attention_state_tests"]
    earnings_row = attention_tests[attention_tests["catalyst_bucket"] == "earnings"].iloc[0]
    assert earnings_row["positive_count"] == 2
    assert earnings_row["negative_count"] == 2
    assert earnings_row["mean_return_diff"] > 0

    classification = report.tables["classification_accuracy"].iloc[0]
    assert classification["sample_size"] == 5
    assert 0.0 <= classification["accuracy"] <= 1.0

    lagged = report.tables["lagged_attention_tests"]
    assert not lagged.empty
    assert report.exported_files["summary"].exists()


def _seed_research_db(db_path: Path) -> DatabaseManager:
    db = DatabaseManager(db_path)
    db.initialize()

    db.upsert_daily_prices(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "open": 10.0,
                "high": 10.8,
                "low": 9.8,
                "close": 10.6,
                "volume": 1_000_000,
                "vwap": 10.3,
                "dollar_volume": 10_300_000,
                "relative_volume_20d": 1.2,
                "gap_pct": 0.03,
                "intraday_range_pct": 0.10,
                "pre_market_high": 10.5,
                "pre_market_low": 10.1,
                "pre_market_volume": 50_000,
                "pre_market_dollar_volume": 500_000,
                "pre_market_relative_volume": 1.1,
                "as_of_timestamp": "2024-01-02T20:00:00+00:00",
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "open": 11.0,
                "high": 11.1,
                "low": 10.1,
                "close": 10.3,
                "volume": 900_000,
                "vwap": 10.6,
                "dollar_volume": 9_540_000,
                "relative_volume_20d": 1.1,
                "gap_pct": 0.03,
                "intraday_range_pct": 0.09,
                "pre_market_high": 11.2,
                "pre_market_low": 10.9,
                "pre_market_volume": 45_000,
                "pre_market_dollar_volume": 490_000,
                "pre_market_relative_volume": 1.0,
                "as_of_timestamp": "2024-01-03T20:00:00+00:00",
            },
            {
                "date": "2024-01-04",
                "ticker": "AAA",
                "open": 12.0,
                "high": 12.9,
                "low": 11.8,
                "close": 12.7,
                "volume": 1_200_000,
                "vwap": 12.4,
                "dollar_volume": 14_880_000,
                "relative_volume_20d": 1.4,
                "gap_pct": 0.06,
                "intraday_range_pct": 0.09,
                "pre_market_high": 12.5,
                "pre_market_low": 12.0,
                "pre_market_volume": 70_000,
                "pre_market_dollar_volume": 840_000,
                "pre_market_relative_volume": 1.4,
                "as_of_timestamp": "2024-01-04T20:00:00+00:00",
            },
            {
                "date": "2024-01-05",
                "ticker": "AAA",
                "open": 13.0,
                "high": 13.1,
                "low": 12.0,
                "close": 12.1,
                "volume": 1_050_000,
                "vwap": 12.5,
                "dollar_volume": 13_125_000,
                "relative_volume_20d": 1.3,
                "gap_pct": 0.06,
                "intraday_range_pct": 0.08,
                "pre_market_high": 13.2,
                "pre_market_low": 12.9,
                "pre_market_volume": 60_000,
                "pre_market_dollar_volume": 780_000,
                "pre_market_relative_volume": 1.2,
                "as_of_timestamp": "2024-01-05T20:00:00+00:00",
            },
            {
                "date": "2024-02-01",
                "ticker": "BBB",
                "open": 20.0,
                "high": 20.4,
                "low": 18.0,
                "close": 18.4,
                "volume": 1_500_000,
                "vwap": 19.1,
                "dollar_volume": 28_650_000,
                "relative_volume_20d": 1.6,
                "gap_pct": -0.07,
                "intraday_range_pct": 0.12,
                "pre_market_high": 19.8,
                "pre_market_low": 18.9,
                "pre_market_volume": 80_000,
                "pre_market_dollar_volume": 1_520_000,
                "pre_market_relative_volume": 1.7,
                "as_of_timestamp": "2024-02-01T20:00:00+00:00",
            },
        ]
    )

    db.upsert_catalysts(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_detail": "Q4 earnings beat",
                "catalyst_direction": "positive",
                "pre_market_gap_pct": 0.03,
                "source": "finnhub",
                "source_timestamp": "2024-01-02T11:00:00+00:00",
                "as_of_timestamp": "2024-01-02T11:05:00+00:00",
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_detail": "Q4 earnings follow-through",
                "catalyst_direction": "positive",
                "pre_market_gap_pct": 0.03,
                "source": "finnhub",
                "source_timestamp": "2024-01-03T11:00:00+00:00",
                "as_of_timestamp": "2024-01-03T11:05:00+00:00",
            },
            {
                "date": "2024-01-04",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_detail": "Analyst boost after results",
                "catalyst_direction": "positive",
                "pre_market_gap_pct": 0.06,
                "source": "finnhub",
                "source_timestamp": "2024-01-04T11:00:00+00:00",
                "as_of_timestamp": "2024-01-04T11:05:00+00:00",
            },
            {
                "date": "2024-01-05",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_detail": "Crowded fade day",
                "catalyst_direction": "positive",
                "pre_market_gap_pct": 0.06,
                "source": "finnhub",
                "source_timestamp": "2024-01-05T11:00:00+00:00",
                "as_of_timestamp": "2024-01-05T11:05:00+00:00",
            },
            {
                "date": "2024-02-01",
                "ticker": "BBB",
                "catalyst_bucket": "news",
                "catalyst_detail": "Secondary offering",
                "catalyst_direction": "negative",
                "pre_market_gap_pct": -0.07,
                "source": "newswire",
                "source_timestamp": "2024-02-01T11:30:00+00:00",
                "as_of_timestamp": "2024-02-01T11:35:00+00:00",
            },
        ]
    )

    db.upsert_crowding_index(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "attention": "low",
                "positioning_pressure": "low",
                "crowd_trap_risk": "absent",
                "attention_score": 0.15,
                "positioning_score": 0.20,
                "trap_score": 0.05,
                "component_details": "{}",
                "as_of_timestamp": "2024-01-02T13:00:00+00:00",
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "attention": "high",
                "positioning_pressure": "high",
                "crowd_trap_risk": "present",
                "attention_score": 0.90,
                "positioning_score": 0.85,
                "trap_score": 0.80,
                "component_details": "{}",
                "as_of_timestamp": "2024-01-03T13:00:00+00:00",
            },
            {
                "date": "2024-01-04",
                "ticker": "AAA",
                "attention": "low",
                "positioning_pressure": "low",
                "crowd_trap_risk": "absent",
                "attention_score": 0.20,
                "positioning_score": 0.25,
                "trap_score": 0.10,
                "component_details": "{}",
                "as_of_timestamp": "2024-01-04T13:00:00+00:00",
            },
            {
                "date": "2024-01-05",
                "ticker": "AAA",
                "attention": "high",
                "positioning_pressure": "high",
                "crowd_trap_risk": "present",
                "attention_score": 0.88,
                "positioning_score": 0.83,
                "trap_score": 0.76,
                "component_details": "{}",
                "as_of_timestamp": "2024-01-05T13:00:00+00:00",
            },
            {
                "date": "2024-02-01",
                "ticker": "BBB",
                "attention": "low",
                "positioning_pressure": "high",
                "crowd_trap_risk": "absent",
                "attention_score": 0.25,
                "positioning_score": 0.72,
                "trap_score": 0.15,
                "component_details": "{}",
                "as_of_timestamp": "2024-02-01T13:00:00+00:00",
            },
        ]
    )

    db.upsert_watchlist(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_direction": "positive",
                "catalyst_detail": "Q4 earnings beat",
                "pre_market_gap_pct": 0.03,
                "pre_market_relative_volume": 1.1,
                "pre_market_dollar_volume": 500_000,
                "attention": "low",
                "positioning_pressure": "low",
                "crowd_trap_risk": "absent",
                "short_interest_pct_float": 0.05,
                "days_to_cover": 1.5,
                "suggested_setup": "long_continuation",
                "setup_score": 0.61,
                "short_tradable": 1,
                "as_of_timestamp": "2024-01-02T13:15:00+00:00",
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_direction": "positive",
                "catalyst_detail": "Q4 earnings follow-through",
                "pre_market_gap_pct": 0.03,
                "pre_market_relative_volume": 1.0,
                "pre_market_dollar_volume": 490_000,
                "attention": "high",
                "positioning_pressure": "high",
                "crowd_trap_risk": "present",
                "short_interest_pct_float": 0.06,
                "days_to_cover": 1.6,
                "suggested_setup": "long_mean_reversion",
                "setup_score": 0.77,
                "short_tradable": 1,
                "as_of_timestamp": "2024-01-03T13:15:00+00:00",
            },
            {
                "date": "2024-01-04",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_direction": "positive",
                "catalyst_detail": "Analyst boost after results",
                "pre_market_gap_pct": 0.06,
                "pre_market_relative_volume": 1.4,
                "pre_market_dollar_volume": 840_000,
                "attention": "low",
                "positioning_pressure": "low",
                "crowd_trap_risk": "absent",
                "short_interest_pct_float": 0.05,
                "days_to_cover": 1.5,
                "suggested_setup": "long_continuation",
                "setup_score": 0.66,
                "short_tradable": 1,
                "as_of_timestamp": "2024-01-04T13:15:00+00:00",
            },
            {
                "date": "2024-01-05",
                "ticker": "AAA",
                "catalyst_bucket": "earnings",
                "catalyst_direction": "positive",
                "catalyst_detail": "Crowded fade day",
                "pre_market_gap_pct": 0.06,
                "pre_market_relative_volume": 1.2,
                "pre_market_dollar_volume": 780_000,
                "attention": "high",
                "positioning_pressure": "high",
                "crowd_trap_risk": "present",
                "short_interest_pct_float": 0.06,
                "days_to_cover": 1.6,
                "suggested_setup": "long_mean_reversion",
                "setup_score": 0.79,
                "short_tradable": 1,
                "as_of_timestamp": "2024-01-05T13:15:00+00:00",
            },
            {
                "date": "2024-02-01",
                "ticker": "BBB",
                "catalyst_bucket": "news",
                "catalyst_direction": "negative",
                "catalyst_detail": "Secondary offering",
                "pre_market_gap_pct": -0.07,
                "pre_market_relative_volume": 1.7,
                "pre_market_dollar_volume": 1_520_000,
                "attention": "low",
                "positioning_pressure": "high",
                "crowd_trap_risk": "absent",
                "short_interest_pct_float": 0.11,
                "days_to_cover": 3.1,
                "suggested_setup": "short_continuation",
                "setup_score": 0.73,
                "short_tradable": 1,
                "as_of_timestamp": "2024-02-01T13:15:00+00:00",
            },
        ]
    )

    db.upsert_wsb_mentions(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "mention_count": 5,
                "mention_count_prior_day": 4,
                "mention_velocity_pct": 0.25,
                "mention_vs_baseline": 0.40,
                "sentiment_score": 0.2,
                "sentiment_unanimity": 0.5,
                "upvotes": 100,
                "rank": 20,
                "rank_change_24h": 2,
                "as_of_timestamp": "2024-01-02T12:00:00+00:00",
                "source": "apewisdom",
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "mention_count": 50,
                "mention_count_prior_day": 10,
                "mention_velocity_pct": 2.50,
                "mention_vs_baseline": 3.00,
                "sentiment_score": 0.8,
                "sentiment_unanimity": 0.9,
                "upvotes": 5000,
                "rank": 1,
                "rank_change_24h": -10,
                "as_of_timestamp": "2024-01-03T12:00:00+00:00",
                "source": "apewisdom",
            },
            {
                "date": "2024-01-04",
                "ticker": "AAA",
                "mention_count": 6,
                "mention_count_prior_day": 5,
                "mention_velocity_pct": 0.20,
                "mention_vs_baseline": 0.50,
                "sentiment_score": 0.3,
                "sentiment_unanimity": 0.6,
                "upvotes": 120,
                "rank": 18,
                "rank_change_24h": 1,
                "as_of_timestamp": "2024-01-04T12:00:00+00:00",
                "source": "apewisdom",
            },
            {
                "date": "2024-01-05",
                "ticker": "AAA",
                "mention_count": 55,
                "mention_count_prior_day": 12,
                "mention_velocity_pct": 2.60,
                "mention_vs_baseline": 3.10,
                "sentiment_score": 0.7,
                "sentiment_unanimity": 0.9,
                "upvotes": 5100,
                "rank": 1,
                "rank_change_24h": -7,
                "as_of_timestamp": "2024-01-05T12:00:00+00:00",
                "source": "apewisdom",
            },
            {
                "date": "2024-02-01",
                "ticker": "BBB",
                "mention_count": 8,
                "mention_count_prior_day": 6,
                "mention_velocity_pct": 0.33,
                "mention_vs_baseline": 0.70,
                "sentiment_score": -0.2,
                "sentiment_unanimity": 0.4,
                "upvotes": 180,
                "rank": 15,
                "rank_change_24h": 1,
                "as_of_timestamp": "2024-02-01T12:00:00+00:00",
                "source": "apewisdom",
            },
        ]
    )

    return db
