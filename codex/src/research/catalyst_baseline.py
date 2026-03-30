"""Catalyst-only baseline research utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.db import DatabaseManager

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_DIR = Path("data/research")
DAY_OF_WEEK_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class CatalystBaselineReport:
    dataset: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    charts: dict[str, str]
    exported_files: dict[str, Path]


def load_catalyst_day_dataset(
    db: DatabaseManager,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Load point-in-time catalyst rows with realized intraday outcomes."""
    with db.connect() as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                c.date,
                c.ticker,
                c.catalyst_bucket,
                c.catalyst_detail,
                c.catalyst_direction,
                c.pre_market_gap_pct,
                c.source,
                c.source_timestamp,
                dp.open,
                dp.high,
                dp.low,
                dp.close,
                dp.gap_pct AS daily_gap_pct,
                ci.attention,
                ci.positioning_pressure,
                ci.crowd_trap_risk,
                ci.attention_score,
                ci.positioning_score,
                ci.trap_score,
                wl.suggested_setup,
                wl.setup_score,
                wl.short_tradable,
                wsb.wsb_as_of_timestamp
            FROM catalysts AS c
            INNER JOIN daily_prices AS dp
                ON dp.date = c.date
               AND dp.ticker = c.ticker
            LEFT JOIN crowding_index AS ci
                ON ci.date = c.date
               AND ci.ticker = c.ticker
            LEFT JOIN watchlist AS wl
                ON wl.date = c.date
               AND wl.ticker = c.ticker
            LEFT JOIN (
                SELECT date, ticker, MAX(as_of_timestamp) AS wsb_as_of_timestamp
                FROM wsb_mentions
                GROUP BY date, ticker
            ) AS wsb
                ON wsb.date = c.date
               AND wsb.ticker = c.ticker
            WHERE c.date BETWEEN ? AND ?
              AND c.catalyst_direction IN ('positive', 'negative')
            ORDER BY c.date, c.ticker, c.source_timestamp
            """,
            connection,
            params=(start_date, end_date),
        )

    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    numeric_columns = [
        "pre_market_gap_pct",
        "daily_gap_pct",
        "open",
        "high",
        "low",
        "close",
        "attention_score",
        "positioning_score",
        "trap_score",
        "setup_score",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["year"] = frame["date"].dt.year
    frame["day_of_week"] = pd.Categorical(frame["date"].dt.day_name(), categories=DAY_OF_WEEK_ORDER, ordered=True)
    frame["gap_pct_used"] = frame["pre_market_gap_pct"].fillna(frame["daily_gap_pct"])
    frame["gap_size_bucket"] = frame["gap_pct_used"].abs().map(_gap_bucket)
    frame["open_to_close_return"] = (frame["close"] - frame["open"]) / frame["open"]
    frame["max_adverse_excursion"] = (frame["low"] - frame["open"]) / frame["open"]
    frame["max_favorable_excursion"] = (frame["high"] - frame["open"]) / frame["open"]
    frame["direction_multiplier"] = frame["catalyst_direction"].map({"positive": 1.0, "negative": -1.0})
    frame["directional_return"] = frame["open_to_close_return"] * frame["direction_multiplier"]
    frame["directional_mae"] = frame.apply(_directional_mae, axis=1)
    frame["directional_mfe"] = frame.apply(_directional_mfe, axis=1)
    frame["win"] = frame["directional_return"] > 0
    frame["realized_class"] = frame["directional_return"].map(lambda value: "continuation" if value >= 0 else "mean_reversion")
    frame["opportunity_side"] = frame["catalyst_direction"].map({"positive": "long", "negative": "short"})
    frame["predicted_class"] = frame["suggested_setup"].map(_predicted_class)
    frame["predicted_side"] = frame["suggested_setup"].map(_predicted_side)
    frame = frame.sort_values(["ticker", "date", "source_timestamp"], kind="stable").reset_index(drop=True)
    # Gate 1 lag testing uses the prior available same-ticker attention state as a stale social-signal proxy.
    frame["lagged_attention"] = frame.groupby("ticker")["attention"].shift(1)
    frame["lagged_attention_score"] = frame.groupby("ticker")["attention_score"].shift(1)
    return frame


def summarize_catalyst_segments(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the catalyst-only baseline tables required for Gate 1."""
    if dataset.empty:
        return {
            "overall": _summary_table(dataset, []),
            "by_bucket_direction": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction"]),
            "by_bucket_direction_gap": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "gap_size_bucket"]),
            "by_bucket_direction_year": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "year"]),
            "by_bucket_direction_day_of_week": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "day_of_week"]),
        }

    return {
        "overall": _summary_table(dataset, []),
        "by_bucket_direction": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction"]),
        "by_bucket_direction_gap": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "gap_size_bucket"]),
        "by_bucket_direction_year": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "year"]),
        "by_bucket_direction_day_of_week": _summary_table(dataset, ["catalyst_bucket", "catalyst_direction", "day_of_week"]),
    }


def render_catalyst_baseline_charts(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    charts: dict[str, str] = {}
    by_bucket_direction = tables.get("by_bucket_direction", pd.DataFrame())
    if not by_bucket_direction.empty:
        charts["avg_directional_return_by_bucket"] = _ascii_bar_chart(
            by_bucket_direction,
            label_column="catalyst_bucket",
            value_column="avg_directional_return",
            title="Average directional return by catalyst bucket",
        )

    by_year = tables.get("by_bucket_direction_year", pd.DataFrame())
    if not by_year.empty:
        year_totals = (
            by_year.groupby("year", dropna=False)["avg_directional_return"]
            .mean()
            .reset_index()
            .sort_values("year", kind="stable")
        )
        charts["avg_directional_return_by_year"] = _ascii_bar_chart(
            year_totals,
            label_column="year",
            value_column="avg_directional_return",
            title="Average directional return by year",
        )
    return charts


def export_catalyst_baseline_report(
    report: CatalystBaselineReport,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}

    dataset_path = output_dir / "catalyst_baseline_dataset.csv"
    report.dataset.to_csv(dataset_path, index=False)
    artifact_paths["dataset"] = dataset_path

    for name, table in report.tables.items():
        table_path = output_dir / f"catalyst_baseline_{name}.csv"
        table.to_csv(table_path, index=False)
        artifact_paths[name] = table_path

    summary_path = output_dir / "catalyst_baseline_summary.md"
    summary_path.write_text(_render_markdown_summary("Catalyst Baseline", report.tables, report.charts), encoding="utf-8")
    artifact_paths["summary"] = summary_path
    return artifact_paths


def run_catalyst_baseline(
    db: DatabaseManager,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> CatalystBaselineReport:
    dataset = load_catalyst_day_dataset(db=db, start_date=start_date, end_date=end_date)
    tables = summarize_catalyst_segments(dataset)
    charts = render_catalyst_baseline_charts(tables)
    report = CatalystBaselineReport(dataset=dataset, tables=tables, charts=charts, exported_files={})
    if output_dir is not None:
        report.exported_files = export_catalyst_baseline_report(report, output_dir=output_dir)
    return report


def _summary_table(dataset: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(
            columns=group_columns
            + [
                "sample_size",
                "avg_open_to_close_return",
                "avg_directional_return",
                "avg_max_adverse_excursion",
                "avg_max_favorable_excursion",
                "avg_directional_mae",
                "avg_directional_mfe",
                "win_rate",
                "payoff_ratio",
            ]
        )

    if group_columns:
        summary = dataset.groupby(group_columns, dropna=False, observed=False).apply(_aggregate_performance).reset_index()
    else:
        summary = pd.DataFrame([_aggregate_performance(dataset)])

    if "day_of_week" in summary.columns:
        summary["day_of_week"] = summary["day_of_week"].astype(str)
    return summary.sort_values(group_columns, kind="stable") if group_columns else summary


def _aggregate_performance(group: pd.DataFrame) -> pd.Series:
    wins = group.loc[group["directional_return"] > 0, "directional_return"]
    losses = group.loc[group["directional_return"] <= 0, "directional_return"]
    loss_magnitude = abs(float(losses.mean())) if not losses.empty else None
    payoff_ratio = float(wins.mean()) / loss_magnitude if not wins.empty and loss_magnitude not in (None, 0.0) else None
    return pd.Series(
        {
            "sample_size": int(len(group)),
            "avg_open_to_close_return": group["open_to_close_return"].mean(),
            "avg_directional_return": group["directional_return"].mean(),
            "avg_max_adverse_excursion": group["max_adverse_excursion"].mean(),
            "avg_max_favorable_excursion": group["max_favorable_excursion"].mean(),
            "avg_directional_mae": group["directional_mae"].mean(),
            "avg_directional_mfe": group["directional_mfe"].mean(),
            "win_rate": group["win"].mean(),
            "payoff_ratio": payoff_ratio,
        }
    )


def _gap_bucket(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    absolute_gap = abs(float(value))
    if absolute_gap < 0.02:
        return "0-2%"
    if absolute_gap < 0.05:
        return "2-5%"
    if absolute_gap < 0.10:
        return "5-10%"
    return "10%+"


def _directional_mae(row: pd.Series) -> float:
    if row["catalyst_direction"] == "negative":
        return (row["open"] - row["high"]) / row["open"]
    return (row["low"] - row["open"]) / row["open"]


def _directional_mfe(row: pd.Series) -> float:
    if row["catalyst_direction"] == "negative":
        return (row["open"] - row["low"]) / row["open"]
    return (row["high"] - row["open"]) / row["open"]


def _predicted_class(setup: Any) -> str | None:
    if not isinstance(setup, str):
        return None
    if "continuation" in setup:
        return "continuation"
    if "mean_reversion" in setup:
        return "mean_reversion"
    return None


def _predicted_side(setup: Any) -> str | None:
    if not isinstance(setup, str):
        return None
    if setup.startswith("long_"):
        return "long"
    if setup.startswith("short_"):
        return "short"
    return None


def _ascii_bar_chart(frame: pd.DataFrame, label_column: str, value_column: str, title: str, width: int = 30) -> str:
    if frame.empty:
        return f"{title}\n(no data)"

    values = frame[value_column].fillna(0.0).astype(float)
    max_abs = max(abs(values).max(), 1e-9)
    lines = [title]
    for _, row in frame.iterrows():
        value = float(row[value_column]) if pd.notna(row[value_column]) else 0.0
        label = str(row[label_column])
        bar_size = int(round((abs(value) / max_abs) * width))
        bar = "#" * bar_size
        sign = "+" if value >= 0 else "-"
        lines.append(f"{label:>12} | {sign}{bar} {value:.2%}")
    return "\n".join(lines)


def _render_markdown_summary(title: str, tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> str:
    sections = [f"# {title}", ""]
    for name, chart in charts.items():
        sections.append(f"## {name.replace('_', ' ').title()}")
        sections.append("```text")
        sections.append(chart)
        sections.append("```")
        sections.append("")

    for name, table in tables.items():
        sections.append(f"## {name.replace('_', ' ').title()}")
        sections.append("```text")
        sections.append(_format_table(table))
        sections.append("```")
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def _format_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "(no rows)"

    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)
