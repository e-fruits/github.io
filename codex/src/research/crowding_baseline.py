"""Crowding uplift baseline research utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from src.research.catalyst_baseline import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    _aggregate_performance,
    _format_table,
    load_catalyst_day_dataset,
)
from src.utils.db import DatabaseManager

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 7


@dataclass
class CrowdingBaselineReport:
    dataset: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    exported_files: dict[str, Path]


def run_crowding_baseline(
    db: DatabaseManager,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> CrowdingBaselineReport:
    dataset = load_catalyst_day_dataset(db=db, start_date=start_date, end_date=end_date)
    tables = summarize_crowding_baseline(dataset)
    report = CrowdingBaselineReport(dataset=dataset, tables=tables, exported_files={})
    if output_dir is not None:
        report.exported_files = export_crowding_baseline_report(report, output_dir=output_dir)
    return report


def summarize_crowding_baseline(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if dataset.empty:
        empty = pd.DataFrame()
        return {
            "attention_state_tests": empty,
            "positioning_state_tests": empty,
            "trap_state_tests": empty,
            "attention_state_tests_by_year": empty,
            "interaction_effects": empty,
            "classification_accuracy": empty,
            "classification_accuracy_by_year": empty,
            "opportunity_distribution": empty,
            "opportunity_distribution_by_year": empty,
            "lagged_attention_tests": empty,
        }

    interaction_effects = _interaction_effects_table(dataset)
    classification = _classification_accuracy_table(dataset, [])
    classification_by_year = _classification_accuracy_table(dataset, ["year"])
    opportunity_distribution = _opportunity_distribution_table(dataset, [])
    opportunity_distribution_by_year = _opportunity_distribution_table(dataset, ["year"])

    return {
        "attention_state_tests": _binary_state_tests(dataset, "attention", "high", "low", ["catalyst_bucket"]),
        "positioning_state_tests": _binary_state_tests(dataset, "positioning_pressure", "high", "low", ["catalyst_bucket"]),
        "trap_state_tests": _binary_state_tests(dataset, "crowd_trap_risk", "present", "absent", ["catalyst_bucket"]),
        "attention_state_tests_by_year": _binary_state_tests(dataset, "attention", "high", "low", ["catalyst_bucket", "year"]),
        "interaction_effects": interaction_effects,
        "classification_accuracy": classification,
        "classification_accuracy_by_year": classification_by_year,
        "opportunity_distribution": opportunity_distribution,
        "opportunity_distribution_by_year": opportunity_distribution_by_year,
        "lagged_attention_tests": _binary_state_tests(dataset.dropna(subset=["lagged_attention"]), "lagged_attention", "high", "low", ["catalyst_bucket"]),
    }


def export_crowding_baseline_report(
    report: CrowdingBaselineReport,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}

    dataset_path = output_dir / "crowding_baseline_dataset.csv"
    report.dataset.to_csv(dataset_path, index=False)
    artifact_paths["dataset"] = dataset_path

    for name, table in report.tables.items():
        table_path = output_dir / f"crowding_baseline_{name}.csv"
        table.to_csv(table_path, index=False)
        artifact_paths[name] = table_path

    summary_path = output_dir / "crowding_baseline_summary.md"
    summary_path.write_text(_render_markdown_summary(report.tables), encoding="utf-8")
    artifact_paths["summary"] = summary_path
    return artifact_paths


def _binary_state_tests(
    dataset: pd.DataFrame,
    state_column: str,
    positive_state: str,
    negative_state: str,
    group_columns: list[str],
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = dataset.groupby(group_columns, dropna=False, observed=False) if group_columns else [((), dataset)]
    for group_key, group_frame in grouped:
        high_state = group_frame[group_frame[state_column] == positive_state]
        low_state = group_frame[group_frame[state_column] == negative_state]
        result = _compare_samples(high_state, low_state)
        base = _group_key_to_dict(group_columns, group_key)
        base.update(
            {
                "state_column": state_column,
                "positive_state": positive_state,
                "negative_state": negative_state,
                "positive_count": int(len(high_state)),
                "negative_count": int(len(low_state)),
            }
        )
        base.update(result)
        rows.append(base)

    table = pd.DataFrame(rows)
    return table.sort_values(group_columns, kind="stable") if group_columns and not table.empty else table


def _compare_samples(positive_frame: pd.DataFrame, negative_frame: pd.DataFrame) -> dict[str, float | None]:
    positive_returns = positive_frame["directional_return"].dropna().astype(float).tolist()
    negative_returns = negative_frame["directional_return"].dropna().astype(float).tolist()

    positive_mean = sum(positive_returns) / len(positive_returns) if positive_returns else None
    negative_mean = sum(negative_returns) / len(negative_returns) if negative_returns else None
    mean_diff = None if positive_mean is None or negative_mean is None else positive_mean - negative_mean

    positive_win_rate = float(positive_frame["win"].mean()) if len(positive_frame) else None
    negative_win_rate = float(negative_frame["win"].mean()) if len(negative_frame) else None

    t_statistic, t_p_value = _welch_t_test(positive_returns, negative_returns)
    chi_square, chi_square_p_value = _chi_square_win_rate_test(positive_frame, negative_frame)
    bootstrap_low, bootstrap_high = _bootstrap_mean_difference_ci(positive_returns, negative_returns)
    cohen_d = _cohen_d(positive_returns, negative_returns)

    return {
        "positive_mean_return": positive_mean,
        "negative_mean_return": negative_mean,
        "mean_return_diff": mean_diff,
        "positive_win_rate": positive_win_rate,
        "negative_win_rate": negative_win_rate,
        "welch_t_statistic": t_statistic,
        "welch_t_p_value": t_p_value,
        "chi_square_statistic": chi_square,
        "chi_square_p_value": chi_square_p_value,
        "cohen_d": cohen_d,
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
    }


def _interaction_effects_table(dataset: pd.DataFrame) -> pd.DataFrame:
    enriched = dataset.copy()
    enriched["attention_high"] = enriched["attention"] == "high"
    enriched["positioning_high"] = enriched["positioning_pressure"] == "high"
    enriched["trap_present"] = enriched["crowd_trap_risk"] == "present"
    grouped = enriched.groupby(
        ["catalyst_bucket", "catalyst_direction", "attention_high", "positioning_high", "trap_present"],
        dropna=False,
        observed=False,
    )
    table = grouped.apply(_aggregate_performance).reset_index()
    table["continuation_rate"] = grouped["realized_class"].apply(lambda series: float((series == "continuation").mean())).reset_index(drop=True)
    return table.sort_values(["catalyst_bucket", "catalyst_direction", "attention_high", "positioning_high", "trap_present"], kind="stable")


def _classification_accuracy_table(dataset: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    valid = dataset.dropna(subset=["predicted_class"])
    if valid.empty:
        return pd.DataFrame(columns=group_columns + ["sample_size", "accuracy"])

    grouped = valid.groupby(group_columns, dropna=False, observed=False) if group_columns else [((), valid)]
    rows: list[dict[str, object]] = []
    for group_key, group_frame in grouped:
        base = _group_key_to_dict(group_columns, group_key)
        base.update(
            {
                "sample_size": int(len(group_frame)),
                "accuracy": float((group_frame["predicted_class"] == group_frame["realized_class"]).mean()),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable") if group_columns else pd.DataFrame(rows)


def _opportunity_distribution_table(dataset: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    valid = dataset.dropna(subset=["predicted_side"])
    if valid.empty:
        return pd.DataFrame(columns=group_columns + ["predicted_side", "sample_size", "share"])

    grouped = valid.groupby(group_columns + ["predicted_side"], dropna=False, observed=False).size().reset_index(name="sample_size")
    denominator = grouped.groupby(group_columns, dropna=False, observed=False)["sample_size"].transform("sum") if group_columns else grouped["sample_size"].sum()
    grouped["share"] = grouped["sample_size"] / denominator
    return grouped.sort_values(group_columns + ["predicted_side"], kind="stable") if group_columns else grouped.sort_values(["predicted_side"], kind="stable")


def _welch_t_test(sample_a: list[float], sample_b: list[float]) -> tuple[float | None, float | None]:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None, None

    mean_a = sum(sample_a) / len(sample_a)
    mean_b = sum(sample_b) / len(sample_b)
    variance_a = _sample_variance(sample_a)
    variance_b = _sample_variance(sample_b)
    denominator = math.sqrt((variance_a / len(sample_a)) + (variance_b / len(sample_b)))
    if denominator == 0:
        return None, None

    t_statistic = (mean_a - mean_b) / denominator
    numerator = ((variance_a / len(sample_a)) + (variance_b / len(sample_b))) ** 2
    denominator_df = ((variance_a / len(sample_a)) ** 2) / (len(sample_a) - 1) + ((variance_b / len(sample_b)) ** 2) / (len(sample_b) - 1)
    if denominator_df == 0:
        return t_statistic, None

    degrees_of_freedom = numerator / denominator_df
    try:
        from scipy import stats  # type: ignore

        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom))
    except Exception:
        p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(t_statistic))))
    return t_statistic, p_value


def _chi_square_win_rate_test(positive_frame: pd.DataFrame, negative_frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if positive_frame.empty or negative_frame.empty:
        return None, None

    wins_positive = int(positive_frame["win"].sum())
    losses_positive = int(len(positive_frame) - wins_positive)
    wins_negative = int(negative_frame["win"].sum())
    losses_negative = int(len(negative_frame) - wins_negative)
    total = wins_positive + losses_positive + wins_negative + losses_negative
    if total == 0:
        return None, None

    row_totals = [wins_positive + losses_positive, wins_negative + losses_negative]
    column_totals = [wins_positive + wins_negative, losses_positive + losses_negative]
    observed = [
        [wins_positive, losses_positive],
        [wins_negative, losses_negative],
    ]

    chi_square = 0.0
    for row_index in range(2):
        for column_index in range(2):
            expected = (row_totals[row_index] * column_totals[column_index]) / total
            if expected == 0:
                return None, None
            chi_square += ((observed[row_index][column_index] - expected) ** 2) / expected

    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    return chi_square, p_value


def _cohen_d(sample_a: list[float], sample_b: list[float]) -> float | None:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None
    variance_a = _sample_variance(sample_a)
    variance_b = _sample_variance(sample_b)
    pooled_denominator = len(sample_a) + len(sample_b) - 2
    if pooled_denominator <= 0:
        return None
    pooled_std = math.sqrt((((len(sample_a) - 1) * variance_a) + ((len(sample_b) - 1) * variance_b)) / pooled_denominator)
    if pooled_std == 0:
        return None
    return ((sum(sample_a) / len(sample_a)) - (sum(sample_b) / len(sample_b))) / pooled_std


def _bootstrap_mean_difference_ci(
    sample_a: list[float],
    sample_b: list[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    if not sample_a or not sample_b:
        return None, None

    generator = random.Random(seed)
    bootstrap_diffs: list[float] = []
    for _ in range(iterations):
        resample_a = [sample_a[generator.randrange(len(sample_a))] for _ in range(len(sample_a))]
        resample_b = [sample_b[generator.randrange(len(sample_b))] for _ in range(len(sample_b))]
        bootstrap_diffs.append((sum(resample_a) / len(resample_a)) - (sum(resample_b) / len(resample_b)))

    bootstrap_diffs.sort()
    lower_index = int(0.025 * (iterations - 1))
    upper_index = int(0.975 * (iterations - 1))
    return bootstrap_diffs[lower_index], bootstrap_diffs[upper_index]


def _sample_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _group_key_to_dict(group_columns: list[str], group_key: object) -> dict[str, object]:
    if not group_columns:
        return {}
    if len(group_columns) == 1:
        return {group_columns[0]: group_key}
    return dict(zip(group_columns, group_key))


def _render_markdown_summary(tables: dict[str, pd.DataFrame]) -> str:
    sections = ["# Crowding Baseline", ""]
    for name, table in tables.items():
        sections.append(f"## {name.replace('_', ' ').title()}")
        sections.append("```text")
        sections.append(_format_table(table))
        sections.append("```")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"
