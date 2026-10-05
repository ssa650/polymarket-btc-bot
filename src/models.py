from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


RECORDER_VERSION = "2.0.0"
SCHEMA_VERSION = "2"



def utc_now() -> datetime:
    return datetime.now(timezone.utc)



def to_iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()



def parse_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def parse_exchange_timestamp(value: object) -> Optional[datetime]:
    """
    Parse exchange timestamps from Polymarket payloads.

    `parse_timestamp` intentionally treats numeric values as seconds. Market
    WebSocket payloads commonly send millisecond epoch strings, so raw event
    storage uses this helper to avoid turning ms timestamps into far-future
    datetimes.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return parse_timestamp(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        magnitude = abs(numeric)
        if magnitude >= 1_000_000_000_000_000_000:
            numeric /= 1_000_000_000.0
        elif magnitude >= 1_000_000_000_000_000:
            numeric /= 1_000_000.0
        elif magnitude >= 100_000_000_000:
            numeric /= 1_000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return parse_exchange_timestamp(float(raw))
        except ValueError:
            return parse_timestamp(raw)
    return None


def local_arrival_iso_from_ns(local_arrival_ns: int) -> str:
    return datetime.fromtimestamp(
        int(local_arrival_ns) / 1_000_000_000.0,
        tz=timezone.utc,
    ).isoformat()


@dataclass(slots=True, frozen=True)
class PriceLevel:
    price: float
    size: float


@dataclass(slots=True)
class MarketMetadata:
    market_id: str
    event_id: Optional[str]
    question: Optional[str]
    description: Optional[str]
    category: Optional[str]
    outcomes: List[str]
    resolution_source: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    close_time: Optional[datetime]
    platform_status: Optional[str] = None
    phase: Optional[str] = None
    tracking_state: Optional[str] = None
    # Legacy mirrors kept for backward compatibility with older SQLite files/code.
    status: Optional[str] = None
    market_phase: Optional[str] = None
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    condition_id: Optional[str] = None
    resolved: int = 0
    resolved_at: Optional[datetime] = None
    winning_asset_id: Optional[str] = None
    winning_outcome: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    last_updated: datetime = field(default_factory=utc_now)

    # Runtime mapping between outcomes and Polymarket CLOB token ids.
    token_ids: Dict[str, str] = field(default_factory=dict)
    initial_volume: Optional[float] = None
    initial_liquidity: Optional[float] = None
    run_id: str = ""
    strict_validation_passed: int = 0
    strict_rejection_reason: Optional[str] = None
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class OrderBookSnapshot:
    token_id: str
    timestamp: datetime
    bids: List[PriceLevel]
    asks: List[PriceLevel]


@dataclass(slots=True)
class TradeRecord:
    timestamp: datetime
    market_id: str
    trade_id: str
    price: float
    size: float
    side: Optional[str]
    maker: Optional[str]
    taker: Optional[str]
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class MarketSnapshotRecord:
    timestamp: datetime
    market_id: str
    yes_price: Optional[float]
    no_price: Optional[float]
    best_bid_yes: Optional[float]
    best_ask_yes: Optional[float]
    best_bid_no: Optional[float]
    best_ask_no: Optional[float]
    spread_yes: Optional[float]
    spread_no: Optional[float]
    mid_price_yes: Optional[float]
    mid_price_no: Optional[float]
    volume: Optional[float]
    liquidity: Optional[float]
    last_trade_price: Optional[float]
    last_trade_size: Optional[float]
    last_trade_time: Optional[datetime]
    has_orderbook: int
    has_trade_data: int
    run_id: str = ""
    strict_validation_passed: int = 0
    snapshot_quality_status: Optional[str] = None
    is_partial_orderbook: int = 0
    missing_level_count: int = 0
    time_gap_from_prev_snapshot_sec: Optional[float] = None
    is_gap_affected: int = 0
    feature_ready: int = 0
    book_checksum: Optional[str] = None
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class OrderBookLevelRecord:
    timestamp: datetime
    market_id: str
    outcome_side: str
    book_side: str
    level: int
    price: float
    size: float
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class FeatureRecord:
    timestamp: datetime
    market_id: str

    price_change_1s: Optional[float] = None
    price_change_10s: Optional[float] = None
    price_change_60s: Optional[float] = None
    velocity_5s: Optional[float] = None
    velocity_30s: Optional[float] = None
    acceleration_5s: Optional[float] = None
    rolling_mean_60s: Optional[float] = None
    distance_from_rolling_mean_60s: Optional[float] = None

    total_bid_liquidity_yes: Optional[float] = None
    total_ask_liquidity_yes: Optional[float] = None
    liquidity_change_bid_5s: Optional[float] = None
    orderbook_imbalance_yes: Optional[float] = None

    buy_volume_5s: Optional[float] = None
    sell_volume_5s: Optional[float] = None
    net_trade_flow_5s: Optional[float] = None
    trade_flow_ratio_5s: Optional[float] = None

    rolling_volatility_10s: Optional[float] = None
    rolling_volatility_60s: Optional[float] = None

    volume_delta_1s: Optional[float] = None
    avg_volume_60s: Optional[float] = None
    volume_spike_ratio: Optional[float] = None

    largest_bid_wall_size_yes: Optional[float] = None
    largest_ask_wall_size_yes: Optional[float] = None
    distance_to_bid_wall: Optional[float] = None

    time_since_market_created: Optional[float] = None
    time_until_resolution: Optional[float] = None

    is_price_jump: Optional[int] = None
    run_id: str = ""
    strict_validation_passed: int = 0
    feature_ready: int = 0
    is_gap_affected: int = 0
    snapshot_quality_status: Optional[str] = None
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class MarketEventRecord:
    timestamp: datetime
    market_id: str
    event_type: str
    details: Optional[str]
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class RawPolymarketEventRecord:
    local_arrival_ns: int
    local_arrival_iso: str
    exchange_timestamp: Optional[datetime]
    event_type: str
    market_id: Optional[str]
    condition_id: Optional[str]
    asset_id: Optional[str]
    slug: Optional[str]
    parse_status: str
    parse_error: Optional[str]
    raw_json: str
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class TickSizeChangeRecord:
    timestamp: datetime
    market_id: Optional[str]
    condition_id: Optional[str]
    asset_id: Optional[str]
    old_tick_size: Optional[float]
    new_tick_size: Optional[float]
    raw_json: str
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class BestBidAskRecord:
    timestamp: datetime
    market_id: Optional[str]
    condition_id: Optional[str]
    asset_id: Optional[str]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    raw_json: str
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class BTCPriceSampleRecord:
    source: str
    price: float
    local_arrival_ns: int
    local_arrival_iso: str
    exchange_timestamp: Optional[datetime]
    raw_json: str
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class RecorderMetricRecord:
    timestamp: datetime
    markets_polled: int
    successful_market_fetches: int
    failed_markets: int
    api_latency_ms: Optional[float]
    db_write_time_ms: Optional[float]
    cycle_duration_ms: Optional[float]
    rows_inserted: int
    duplicate_rows_skipped: int
    active_markets_snapshot_attempted: int = 0
    snapshots_written: int = 0
    snapshot_markets_skipped: int = 0
    discovery_candidates_seen: int = 0
    discovery_candidates_matched: int = 0
    discovery_strict_5m_candidates: int = 0
    discovery_broad_btc_candidates: int = 0
    discovery_fallback_used: int = 0
    tracked_markets_count: int = 0
    api_call_count: int = 0
    api_success_count: int = 0
    api_failure_count: int = 0
    fetched_trades: int = 0
    accepted_trades: int = 0
    rejected_before_start: int = 0
    rejected_after_close: int = 0
    skipped_already_seen: int = 0
    last_accepted_trade_ts: Optional[datetime] = None
    ws_reconnect_count: int = 0
    raw_ws_events_seen: int = 0
    raw_ws_events_written: int = 0
    malformed_ws_events: int = 0
    raw_ws_write_failures: int = 0
    last_ws_event_age_sec: Optional[float] = None
    subscribed_asset_count: int = 0
    subscribed_asset_ids_json: Optional[str] = None
    run_id: str = ""
    recorder_version: str = RECORDER_VERSION
    schema_version: str = SCHEMA_VERSION


@dataclass(slots=True)
class WSOrderBookUpdate:
    timestamp: datetime
    token_id: str
    bids: List[PriceLevel]
    asks: List[PriceLevel]


@dataclass(slots=True)
class WSPriceChange:
    timestamp: datetime
    token_id: str
    book_side: str
    price: float
    size: float


@dataclass(slots=True)
class WSStatusEvent:
    timestamp: datetime
    market_id: str
    event_type: str
    details: Optional[str]
