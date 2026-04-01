"""Application config and compliance blocklist loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class DatabaseSettings(BaseModel):
    path: Path


class PipelineSettings(BaseModel):
    cache_dir: Path
    log_level: str = "INFO"
    start_date: str = "2020-01-01"
    request_timeout_seconds: int = 30
    use_stub_fallback: bool = True


class PriceVolumeSettings(BaseModel):
    adjusted: bool = True
    include_premarket: bool = True
    rolling_window_days: int = 20
    chunk_size: int = 250


class CatalystSettings(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["finnhub", "yahoo_finance"])
    include_earnings: bool = True
    include_analyst_actions: bool = True
    include_fda_events: bool = True
    include_company_news: bool = True
    chunk_size: int = 250


class WsbSettings(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["apewisdom", "kaggle"])
    rolling_window_days: int = 20
    apewisdom_top_n: int = 50
    sentiment_model: str = "finbert"
    fallback_sentiment_model: str = "vader"
    kaggle_dataset_path: Path = Path("data/raw/kaggle/wsb_historical.csv")


class OptionsFlowSettings(BaseModel):
    adjusted: bool = True
    rolling_window_days: int = 20
    use_trade_level_small_lot_estimate: bool = True


class ShortInterestSettings(BaseModel):
    report_publication_lag_business_days: int = 8
    use_latest_available_as_of_date: bool = True


class AttentionSignalSettings(BaseModel):
    medium_threshold: float = 0.4
    high_threshold: float = 0.7
    mention_velocity_cap: float = 3.0
    mention_vs_baseline_cap: float = 5.0
    unusual_options_volume_cap: float = 4.0
    wsb_weight: float = 0.75
    options_weight: float = 0.25


class PositioningSignalSettings(BaseModel):
    medium_threshold: float = 0.4
    high_threshold: float = 0.7
    small_lot_call_put_ratio_cap: float = 4.0
    unusual_options_volume_cap: float = 4.0
    short_pct_float_cap: float = 0.35
    days_to_cover_cap: float = 12.0


class TrapRiskSignalSettings(BaseModel):
    present_threshold: float = 0.65
    round_number_band_pct: float = 0.02
    yearly_extreme_band_pct: float = 0.03
    support_resistance_band_pct: float = 0.025
    round_numbers: list[float] = Field(default_factory=lambda: [10.0, 20.0, 50.0, 100.0])


class PremarketSignalSettings(BaseModel):
    meaningful_gap_pct: float = 0.04
    min_relative_volume: float = 1.0
    min_dollar_volume: float = 500_000.0
    max_spread_estimate_pct: float = 0.03


class SqueezeRiskSignalSettings(BaseModel):
    block_threshold: float = 0.7
    short_pct_float_cap: float = 0.35
    days_to_cover_cap: float = 12.0
    low_float_cap: float = 25_000_000.0
    upward_attention_cap: float = 1.0
    gap_magnitude_cap: float = 0.2


class ScannerSignalSettings(BaseModel):
    continuation_score_threshold: float = 0.62
    mean_reversion_score_threshold: float = 0.52


class SignalSettings(BaseModel):
    attention: AttentionSignalSettings = Field(default_factory=AttentionSignalSettings)
    positioning: PositioningSignalSettings = Field(default_factory=PositioningSignalSettings)
    trap_risk: TrapRiskSignalSettings = Field(default_factory=TrapRiskSignalSettings)
    premarket: PremarketSignalSettings = Field(default_factory=PremarketSignalSettings)
    squeeze_risk: SqueezeRiskSignalSettings = Field(default_factory=SqueezeRiskSignalSettings)
    scanner: ScannerSignalSettings = Field(default_factory=ScannerSignalSettings)


class UniverseSettings(BaseModel):
    min_market_cap: float
    max_market_cap: float
    min_price: float
    min_avg_volume_20d: float
    allowed_exchanges: list[str]
    include_delisted: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> "UniverseSettings":
        if self.min_market_cap >= self.max_market_cap:
            raise ValueError("min_market_cap must be smaller than max_market_cap")
        return self


class PolygonSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.polygon.io"


class SchwabSettings(BaseModel):
    app_key: str = ""
    app_secret: str = ""


class FinnhubSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://finnhub.io/api/v1"


class YahooFinanceSettings(BaseModel):
    enabled: bool = True


class ApeWisdomSettings(BaseModel):
    base_url: str = "https://apewisdom.io/api/v1.0"


class FinraSettings(BaseModel):
    base_url: str = "https://api.finra.org"


class ProviderSettings(BaseModel):
    polygon: PolygonSettings = Field(default_factory=PolygonSettings)
    schwab: SchwabSettings = Field(default_factory=SchwabSettings)
    finnhub: FinnhubSettings = Field(default_factory=FinnhubSettings)
    yahoo_finance: YahooFinanceSettings = Field(default_factory=YahooFinanceSettings)
    apewisdom: ApeWisdomSettings = Field(default_factory=ApeWisdomSettings)
    finra: FinraSettings = Field(default_factory=FinraSettings)


class ComplianceSettings(BaseModel):
    blocklist_path: Path
    fail_on_blocklist_conflicts: bool = True


class BlocklistEntry(BaseModel):
    value: str
    reason: str


class ComplianceBlocklist(BaseModel):
    blocked_tickers: set[str] = Field(default_factory=set)
    blocked_sectors: set[str] = Field(default_factory=set)
    blocked_industries: set[str] = Field(default_factory=set)


class AppSettings(BaseModel):
    database: DatabaseSettings
    pipeline: PipelineSettings
    price_volume: PriceVolumeSettings = Field(default_factory=PriceVolumeSettings)
    catalysts: CatalystSettings = Field(default_factory=CatalystSettings)
    wsb: WsbSettings = Field(default_factory=WsbSettings)
    options_flow: OptionsFlowSettings = Field(default_factory=OptionsFlowSettings)
    short_interest: ShortInterestSettings = Field(default_factory=ShortInterestSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    universe: UniverseSettings
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    compliance: ComplianceSettings
    compliance_blocklist: ComplianceBlocklist = Field(default_factory=ComplianceBlocklist)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def load_blocklist(path: Path) -> ComplianceBlocklist:
    payload = _read_yaml(path)

    blocked_tickers = {
        str(item["ticker"]).upper()
        for item in payload.get("tickers", [])
        if isinstance(item, dict) and item.get("ticker")
    }
    blocked_sectors = {
        str(item["sector"])
        for item in payload.get("sectors", [])
        if isinstance(item, dict) and item.get("sector")
    }
    blocked_industries = {
        str(item["industry"])
        for item in payload.get("industries", [])
        if isinstance(item, dict) and item.get("industry")
    }
    return ComplianceBlocklist(
        blocked_tickers=blocked_tickers,
        blocked_sectors=blocked_sectors,
        blocked_industries=blocked_industries,
    )


def load_settings(path: Path) -> AppSettings:
    raw = _read_yaml(path)
    base_dir = path.parent.parent

    database_path = base_dir / raw["database"]["path"]
    cache_dir = base_dir / raw["pipeline"]["cache_dir"]
    blocklist_path = base_dir / raw["compliance"]["blocklist_path"]
    wsb_dataset_path = base_dir / raw.get("wsb", {}).get("kaggle_dataset_path", "data/raw/kaggle/wsb_historical.csv")

    settings = AppSettings.model_validate(
        {
            **raw,
            "database": {**raw["database"], "path": database_path},
            "pipeline": {**raw["pipeline"], "cache_dir": cache_dir},
            "wsb": {**raw.get("wsb", {}), "kaggle_dataset_path": wsb_dataset_path},
            "compliance": {**raw["compliance"], "blocklist_path": blocklist_path},
        }
    )
    settings.compliance_blocklist = load_blocklist(settings.compliance.blocklist_path)
    return settings
