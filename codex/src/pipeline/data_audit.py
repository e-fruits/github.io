"""Point-in-time audit utility for decision-time data availability."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.utils.config import load_settings
from src.utils.db import DatabaseManager

ET = ZoneInfo("America/New_York")


@dataclass
class FieldAudit:
    dataset: str
    field_name: str
    value: Any
    as_of_timestamp: str | None
    available_at_decision_time: bool
    lookahead_violation: bool
    missing: bool
    notes: str | None = None


@dataclass
class AuditReport:
    ticker: str
    trade_date: str
    decision_time_et: str
    decision_time_utc: str
    available_fields: list[FieldAudit] = field(default_factory=list)
    missing_fields: list[FieldAudit] = field(default_factory=list)
    lookahead_violations: list[FieldAudit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "decision_time_et": self.decision_time_et,
            "decision_time_utc": self.decision_time_utc,
            "available_fields": [asdict(item) for item in self.available_fields],
            "missing_fields": [asdict(item) for item in self.missing_fields],
            "lookahead_violations": [asdict(item) for item in self.lookahead_violations],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class DataAuditService:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def audit(self, ticker: str, trade_date: date) -> AuditReport:
        decision_time_et = datetime.combine(trade_date, time(hour=9, minute=15), tzinfo=ET)
        decision_time_utc = decision_time_et.astimezone(timezone.utc)
        report = AuditReport(
            ticker=ticker,
            trade_date=trade_date.isoformat(),
            decision_time_et=decision_time_et.isoformat(),
            decision_time_utc=decision_time_utc.isoformat(),
        )

        self._audit_single_row(
            report=report,
            dataset="universe",
            row=self.db.fetch_row_by_date("universe", ticker, trade_date.isoformat()),
            fields=["company_name", "market_cap", "sector", "exchange", "avg_volume_20d", "avg_dollar_volume_20d", "is_active"],
            decision_time_utc=decision_time_utc,
        )
        self._audit_single_row(
            report=report,
            dataset="daily_prices",
            row=self.db.fetch_row_by_date("daily_prices", ticker, trade_date.isoformat()),
            fields=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "dollar_volume",
                "relative_volume_20d",
                "gap_pct",
                "intraday_range_pct",
                "pre_market_high",
                "pre_market_low",
                "pre_market_volume",
                "pre_market_dollar_volume",
                "pre_market_relative_volume",
            ],
            decision_time_utc=decision_time_utc,
        )
        self._audit_single_row(
            report=report,
            dataset="options_flow_daily",
            row=self.db.fetch_row_by_date("options_flow_daily", ticker, trade_date.isoformat()),
            fields=[
                "total_call_volume",
                "total_put_volume",
                "call_put_ratio",
                "total_call_oi",
                "total_put_oi",
                "oi_change_calls",
                "oi_change_puts",
                "unusual_volume_ratio",
                "small_lot_call_volume",
                "small_lot_put_volume",
                "small_lot_call_pct",
            ],
            decision_time_utc=decision_time_utc,
        )

        catalyst_rows = self.db.fetch_catalyst_rows(ticker, trade_date.isoformat())
        if not catalyst_rows:
            self._register_missing_dataset(report, "catalysts", ["catalyst_bucket", "catalyst_detail", "catalyst_direction", "source_timestamp"])
        else:
            for index, row in enumerate(catalyst_rows, start=1):
                source_timestamp = self._parse_timestamp(row["source_timestamp"])
                self._register_field(
                    report=report,
                    dataset=f"catalysts[{index}]",
                    field_name="catalyst_bucket",
                    value=row["catalyst_bucket"],
                    as_of_timestamp=row["source_timestamp"],
                    decision_time_utc=decision_time_utc,
                    notes="Uses source_timestamp because event availability matters more than scrape time.",
                )
                self._register_field(
                    report=report,
                    dataset=f"catalysts[{index}]",
                    field_name="catalyst_detail",
                    value=row["catalyst_detail"],
                    as_of_timestamp=row["source_timestamp"],
                    decision_time_utc=decision_time_utc,
                    notes="Uses source_timestamp because event availability matters more than scrape time.",
                )
                self._register_field(
                    report=report,
                    dataset=f"catalysts[{index}]",
                    field_name="catalyst_direction",
                    value=row["catalyst_direction"],
                    as_of_timestamp=row["source_timestamp"],
                    decision_time_utc=decision_time_utc,
                    notes="Uses source_timestamp because event availability matters more than scrape time.",
                )
                self._register_field(
                    report=report,
                    dataset=f"catalysts[{index}]",
                    field_name="source_timestamp",
                    value=row["source_timestamp"],
                    as_of_timestamp=row["source_timestamp"],
                    decision_time_utc=decision_time_utc,
                    notes=None if source_timestamp <= decision_time_utc else "Catalyst was not yet known by decision time.",
                )

        wsb_rows = self.db.fetch_wsb_rows(ticker, trade_date.isoformat())
        if not wsb_rows:
            self._register_missing_dataset(
                report,
                "wsb_mentions",
                ["mention_count", "mention_count_prior_day", "mention_velocity_pct", "mention_vs_baseline", "sentiment_score", "sentiment_unanimity", "upvotes", "rank"],
            )
        else:
            for row in wsb_rows:
                source = row["source"]
                self._audit_single_row(
                    report=report,
                    dataset=f"wsb_mentions[{source}]",
                    row=row,
                    fields=[
                        "mention_count",
                        "mention_count_prior_day",
                        "mention_velocity_pct",
                        "mention_vs_baseline",
                        "sentiment_score",
                        "sentiment_unanimity",
                        "upvotes",
                        "rank",
                        "rank_change_24h",
                    ],
                    decision_time_utc=decision_time_utc,
                )

        latest_short_interest = self.db.fetch_latest_short_interest_as_of(ticker, decision_time_utc)
        if latest_short_interest is None:
            self._register_missing_dataset(
                report,
                "short_interest_latest_available",
                ["report_date", "settlement_date", "short_pct_float", "days_to_cover"],
            )
        else:
            self._audit_single_row(
                report=report,
                dataset="short_interest_latest_available",
                row=latest_short_interest,
                fields=["report_date", "settlement_date", "short_interest_shares", "short_pct_float", "days_to_cover"],
                decision_time_utc=decision_time_utc,
            )

        return report

    def _audit_single_row(
        self,
        report: AuditReport,
        dataset: str,
        row: Any,
        fields: list[str],
        decision_time_utc: datetime,
    ) -> None:
        if row is None:
            self._register_missing_dataset(report, dataset, fields)
            return

        as_of_timestamp_raw = row["as_of_timestamp"] if "as_of_timestamp" in row.keys() else None
        for field_name in fields:
            self._register_field(
                report=report,
                dataset=dataset,
                field_name=field_name,
                value=row[field_name] if field_name in row.keys() else None,
                as_of_timestamp=as_of_timestamp_raw,
                decision_time_utc=decision_time_utc,
                notes=None,
            )

    def _register_missing_dataset(self, report: AuditReport, dataset: str, fields: list[str]) -> None:
        for field_name in fields:
            report.missing_fields.append(
                FieldAudit(
                    dataset=dataset,
                    field_name=field_name,
                    value=None,
                    as_of_timestamp=None,
                    available_at_decision_time=False,
                    lookahead_violation=False,
                    missing=True,
                    notes="No dataset row available for this ticker/date at audit time.",
                )
            )

    def _register_field(
        self,
        report: AuditReport,
        dataset: str,
        field_name: str,
        value: Any,
        as_of_timestamp: str | None,
        decision_time_utc: datetime,
        notes: str | None,
    ) -> None:
        parsed_as_of = self._parse_timestamp(as_of_timestamp) if as_of_timestamp else None
        available = parsed_as_of is not None and parsed_as_of <= decision_time_utc and value is not None
        violation = parsed_as_of is not None and parsed_as_of > decision_time_utc and value is not None
        audit = FieldAudit(
            dataset=dataset,
            field_name=field_name,
            value=value,
            as_of_timestamp=as_of_timestamp,
            available_at_decision_time=available,
            lookahead_violation=violation,
            missing=value is None,
            notes=notes,
        )
        if audit.missing:
            report.missing_fields.append(audit)
        elif audit.lookahead_violation:
            report.lookahead_violations.append(audit)
        else:
            report.available_fields.append(audit)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError(f"Timestamp must be timezone-aware: {value}")
        return parsed.astimezone(timezone.utc)


def audit_data_from_config(
    ticker: str,
    trade_date: date,
    settings_path: Path = Path("config/settings.yaml"),
) -> AuditReport:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    service = DataAuditService(db)
    return service.audit(ticker=ticker, trade_date=trade_date)
