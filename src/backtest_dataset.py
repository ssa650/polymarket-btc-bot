from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .models import parse_timestamp


class BacktestDatasetError(RuntimeError):
    pass


FEATURE_COLUMNS: tuple[str, ...] = (
    "price_change_1s",
    "price_change_10s",
    "price_change_60s",
    "velocity_5s",
    "velocity_30s",
    "acceleration_5s",
    "rolling_mean_60s",
    "distance_from_rolling_mean_60s",
    "total_bid_liquidity_yes",
    "total_ask_liquidity_yes",
    "liquidity_change_bid_5s",
    "orderbook_imbalance_yes",
    "buy_volume_5s",
    "sell_volume_5s",
    "net_trade_flow_5s",
    "trade_flow_ratio_5s",
    "rolling_volatility_10s",
    "rolling_volatility_60s",
    "volume_delta_1s",
    "avg_volume_60s",
    "volume_spike_ratio",
    "largest_bid_wall_size_yes",
    "largest_ask_wall_size_yes",
    "distance_to_bid_wall",
    "time_since_market_created",
    "time_until_resolution",
    "is_price_jump",
)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "market_id",
    "timestamp",
    "market_start_time",
    "market_close_time",
    "seconds_since_market_start",
    "seconds_until_close",
    "best_bid_yes",
    "best_ask_yes",
    "best_bid_no",
    "best_ask_no",
    "spread_yes",
    "spread_no",
    "mid_price_yes",
    "mid_price_no",
    "last_trade_price",
    "last_trade_size",
    "last_trade_time",
    "has_btc_price",
    "btc_price",
    "btc_exchange_timestamp",
    "btc_local_arrival_ns",
    "btc_sample_age_sec",
    "btc_source",
    "latest_tick_size_yes",
    "latest_tick_size_no",
    "tick_size_age_sec",
    "resolved",
    "resolved_at",
    "winning_asset_id",
    "winning_outcome",
    "yes_won",
    "no_won",
    "label_available",
    "has_full_orderbook",
    "has_trade_data",
    "gap_affected",
    "snapshot_quality_status",
    *FEATURE_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class BacktestRecord:
    market_id: str
    timestamp: datetime
    run_id: str | None
    seconds_since_market_start: float | None
    seconds_until_close: float | None
    best_bid_yes: float | None
    best_ask_yes: float | None
    best_bid_no: float | None
    best_ask_no: float | None
    spread_yes: float | None
    spread_no: float | None
    mid_price_yes: float | None
    mid_price_no: float | None
    last_trade_price: float | None
    last_trade_size: float | None
    last_trade_time: datetime | None
    btc_price: float | None
    btc_exchange_timestamp: datetime | None
    btc_local_arrival_ns: int | None
    btc_sample_age_sec: float | None
    btc_source: str | None
    latest_tick_size_yes: float | None
    latest_tick_size_no: float | None
    tick_size_age_sec: float | None
    market_start_time: datetime | None
    market_close_time: datetime | None
    resolved: int
    resolved_at: datetime | None
    winning_asset_id: str | None
    winning_outcome: str | None
    yes_won: int | None
    no_won: int | None
    label_available: int
    has_btc_price: int
    has_full_orderbook: int
    has_trade_data: int
    gap_affected: int
    snapshot_quality_status: str | None
    features: Mapping[str, Any]
    raw: Mapping[str, Any]
    has_depth: int = 0


@dataclass(frozen=True, slots=True)
class MarketTimeline:
    market_id: str
    records: tuple[BacktestRecord, ...]

    @property
    def first_timestamp(self) -> datetime | None:
        return self.records[0].timestamp if self.records else None

    @property
    def last_timestamp(self) -> datetime | None:
        return self.records[-1].timestamp if self.records else None


@dataclass(frozen=True, slots=True)
class BacktestDatasetSummary:
    total_rows: int
    total_markets: int
    rows_per_market: dict[str, int]
    btc_coverage_pct: float
    label_coverage_pct: float
    full_orderbook_coverage_pct: float
    depth_coverage_pct: float
    trade_coverage_pct: float
    time_range_start: datetime | None
    time_range_end: datetime | None
    market_ranges: dict[str, tuple[datetime | None, datetime | None]]


@dataclass(frozen=True, slots=True)
class BacktestDataset:
    records: tuple[BacktestRecord, ...]
    timelines: Mapping[str, MarketTimeline]

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        market_ids: Sequence[str] | None = None,
        require_btc_price: bool = False,
        require_full_orderbook: bool = False,
        require_depth: bool = False,
        require_label_available: bool = False,
        require_trade_data: bool = False,
        time_window_before_close_sec: float | None = None,
    ) -> "BacktestDataset":
        rows = _read_parquet_rows(path)
        _validate_required_columns(rows)
        market_filter = {str(market_id) for market_id in market_ids or ()}
        records = [_record_from_row(row) for row in rows]
        dataset = cls.from_records(records)
        return dataset.filter(
            market_ids=market_filter or None,
            require_btc_price=require_btc_price,
            require_full_orderbook=require_full_orderbook,
            require_depth=require_depth,
            require_label_available=require_label_available,
            require_trade_data=require_trade_data,
            time_window_before_close_sec=time_window_before_close_sec,
        )

    @classmethod
    def from_records(cls, records: Iterable[BacktestRecord]) -> "BacktestDataset":
        ordered = sorted(records, key=lambda row: (row.market_id, row.timestamp))
        grouped: dict[str, list[BacktestRecord]] = {}
        for record in ordered:
            grouped.setdefault(record.market_id, []).append(record)

        timelines = {
            market_id: MarketTimeline(market_id=market_id, records=tuple(rows))
            for market_id, rows in sorted(grouped.items())
        }
        _validate_monotonic_timelines(timelines.values())
        return cls(records=tuple(ordered), timelines=timelines)

    def filter(
        self,
        *,
        market_ids: Sequence[str] | set[str] | None = None,
        require_btc_price: bool = False,
        require_full_orderbook: bool = False,
        require_depth: bool = False,
        require_label_available: bool = False,
        require_trade_data: bool = False,
        time_window_before_close_sec: float | None = None,
    ) -> "BacktestDataset":
        market_filter = {str(market_id) for market_id in market_ids or ()}
        filtered = []
        for record in self.records:
            if market_filter and record.market_id not in market_filter:
                continue
            if require_btc_price and record.has_btc_price != 1:
                continue
            if require_full_orderbook and record.has_full_orderbook != 1:
                continue
            if require_depth and record.has_depth != 1:
                continue
            if require_label_available and record.label_available != 1:
                continue
            if require_trade_data and record.has_trade_data != 1:
                continue
            if time_window_before_close_sec is not None:
                if record.seconds_until_close is None:
                    continue
                if record.seconds_until_close < 0:
                    continue
                if record.seconds_until_close > float(time_window_before_close_sec):
                    continue
            filtered.append(record)
        return BacktestDataset.from_records(filtered)

    def only_with_btc_price(self) -> "BacktestDataset":
        return self.filter(require_btc_price=True)

    def only_with_full_orderbook(self) -> "BacktestDataset":
        return self.filter(require_full_orderbook=True)

    def only_with_depth(self) -> "BacktestDataset":
        return self.filter(require_depth=True)

    def only_with_labels(self) -> "BacktestDataset":
        return self.filter(require_label_available=True)

    def only_with_trade_data(self) -> "BacktestDataset":
        return self.filter(require_trade_data=True)

    def within_seconds_before_close(self, seconds: float) -> "BacktestDataset":
        return self.filter(time_window_before_close_sec=seconds)

    def for_market_ids(self, market_ids: Sequence[str]) -> "BacktestDataset":
        return self.filter(market_ids=market_ids)

    def summary(self) -> BacktestDatasetSummary:
        total = len(self.records)
        rows_per_market = {
            market_id: len(timeline.records)
            for market_id, timeline in self.timelines.items()
        }
        return BacktestDatasetSummary(
            total_rows=total,
            total_markets=len(self.timelines),
            rows_per_market=rows_per_market,
            btc_coverage_pct=_pct(
                sum(1 for record in self.records if record.has_btc_price == 1),
                total,
            ),
            label_coverage_pct=_pct(
                sum(1 for record in self.records if record.label_available == 1),
                total,
            ),
            full_orderbook_coverage_pct=_pct(
                sum(1 for record in self.records if record.has_full_orderbook == 1),
                total,
            ),
            depth_coverage_pct=_pct(
                sum(1 for record in self.records if record.has_depth == 1),
                total,
            ),
            trade_coverage_pct=_pct(
                sum(1 for record in self.records if record.has_trade_data == 1),
                total,
            ),
            time_range_start=min((record.timestamp for record in self.records), default=None),
            time_range_end=max((record.timestamp for record in self.records), default=None),
            market_ranges={
                market_id: (timeline.first_timestamp, timeline.last_timestamp)
                for market_id, timeline in self.timelines.items()
            },
        )


def load_backtest_dataset(
    path: str | Path,
    *,
    market_ids: Sequence[str] | None = None,
    require_btc_price: bool = False,
    require_full_orderbook: bool = False,
    require_depth: bool = False,
    require_label_available: bool = False,
    require_trade_data: bool = False,
    time_window_before_close_sec: float | None = None,
) -> BacktestDataset:
    return BacktestDataset.from_parquet(
        path,
        market_ids=market_ids,
        require_btc_price=require_btc_price,
        require_full_orderbook=require_full_orderbook,
        require_depth=require_depth,
        require_label_available=require_label_available,
        require_trade_data=require_trade_data,
        time_window_before_close_sec=time_window_before_close_sec,
    )


def inspect_backtest_dataset(path: str | Path) -> str:
    dataset = load_backtest_dataset(path)
    summary = dataset.summary()
    rows_per_market = list(summary.rows_per_market.values())
    per_market_min = min(rows_per_market) if rows_per_market else 0
    per_market_max = max(rows_per_market) if rows_per_market else 0
    per_market_avg = mean(rows_per_market) if rows_per_market else 0.0

    lines = [
        "=== Backtest Dataset ===",
        f"total_rows={summary.total_rows}",
        f"total_markets={summary.total_markets}",
        f"rows_per_market_min={per_market_min}",
        f"rows_per_market_max={per_market_max}",
        f"rows_per_market_avg={per_market_avg:.2f}",
        f"btc_coverage_pct={summary.btc_coverage_pct:.2f}",
        f"label_coverage_pct={summary.label_coverage_pct:.2f}",
        f"full_orderbook_coverage_pct={summary.full_orderbook_coverage_pct:.2f}",
        f"depth_coverage_pct={summary.depth_coverage_pct:.2f}",
        f"trade_coverage_pct={summary.trade_coverage_pct:.2f}",
        f"time_range_start={_iso(summary.time_range_start)}",
        f"time_range_end={_iso(summary.time_range_end)}",
        "",
        "=== Markets ===",
    ]
    for market_id, timeline in summary.market_ranges.items():
        lines.append(
            " | ".join(
                [
                    f"market_id={market_id}",
                    f"rows={summary.rows_per_market[market_id]}",
                    f"first_timestamp={_iso(timeline[0])}",
                    f"last_timestamp={_iso(timeline[1])}",
                ]
            )
        )
    if summary.label_coverage_pct == 0.0:
        lines.extend(["", "warning=no_labels_available"])
    return "\n".join(lines)


def _read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BacktestDatasetError(
            "Backtest dataset reader requires pyarrow. Install project dependencies first."
        ) from exc

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise BacktestDatasetError(f"Backtest dataset not found: {parquet_path}")
    return pq.read_table(parquet_path).to_pylist()


def _validate_required_columns(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise BacktestDatasetError("Backtest dataset contains no rows")
    columns = set(rows[0].keys())
    missing = sorted(column for column in REQUIRED_COLUMNS if column not in columns)
    if missing:
        raise BacktestDatasetError(
            "Backtest dataset is missing required columns: " + ", ".join(missing)
        )


def _validate_monotonic_timelines(timelines: Iterable[MarketTimeline]) -> None:
    for timeline in timelines:
        previous: datetime | None = None
        for record in timeline.records:
            if previous is not None and record.timestamp < previous:
                raise BacktestDatasetError(
                    f"Timestamps are not monotonic for market_id={timeline.market_id}"
                )
            previous = record.timestamp


def _record_from_row(row: Mapping[str, Any]) -> BacktestRecord:
    timestamp = _required_timestamp(row.get("timestamp"), "timestamp")
    return BacktestRecord(
        run_id=_str_or_none(row.get("run_id")),
        market_id=_required_str(row.get("market_id"), "market_id"),
        timestamp=timestamp,
        seconds_since_market_start=_float_or_none(row.get("seconds_since_market_start")),
        seconds_until_close=_float_or_none(row.get("seconds_until_close")),
        best_bid_yes=_float_or_none(row.get("best_bid_yes")),
        best_ask_yes=_float_or_none(row.get("best_ask_yes")),
        best_bid_no=_float_or_none(row.get("best_bid_no")),
        best_ask_no=_float_or_none(row.get("best_ask_no")),
        spread_yes=_float_or_none(row.get("spread_yes")),
        spread_no=_float_or_none(row.get("spread_no")),
        mid_price_yes=_float_or_none(row.get("mid_price_yes")),
        mid_price_no=_float_or_none(row.get("mid_price_no")),
        last_trade_price=_float_or_none(row.get("last_trade_price")),
        last_trade_size=_float_or_none(row.get("last_trade_size")),
        last_trade_time=_timestamp_or_none(row.get("last_trade_time"), "last_trade_time"),
        btc_price=_float_or_none(row.get("btc_price")),
        btc_exchange_timestamp=_timestamp_or_none(
            row.get("btc_exchange_timestamp"),
            "btc_exchange_timestamp",
        ),
        btc_local_arrival_ns=_int_or_none(row.get("btc_local_arrival_ns")),
        btc_sample_age_sec=_float_or_none(row.get("btc_sample_age_sec")),
        btc_source=_str_or_none(row.get("btc_source")),
        latest_tick_size_yes=_float_or_none(row.get("latest_tick_size_yes")),
        latest_tick_size_no=_float_or_none(row.get("latest_tick_size_no")),
        tick_size_age_sec=_float_or_none(row.get("tick_size_age_sec")),
        market_start_time=_timestamp_or_none(row.get("market_start_time"), "market_start_time"),
        market_close_time=_timestamp_or_none(row.get("market_close_time"), "market_close_time"),
        resolved=_int_flag(row.get("resolved")),
        resolved_at=_timestamp_or_none(row.get("resolved_at"), "resolved_at"),
        winning_asset_id=_str_or_none(row.get("winning_asset_id")),
        winning_outcome=_str_or_none(row.get("winning_outcome")),
        yes_won=_optional_int_flag(row.get("yes_won")),
        no_won=_optional_int_flag(row.get("no_won")),
        label_available=_int_flag(row.get("label_available")),
        has_btc_price=_int_flag(row.get("has_btc_price")),
        has_full_orderbook=_int_flag(row.get("has_full_orderbook")),
        has_trade_data=_int_flag(row.get("has_trade_data")),
        gap_affected=_int_flag(row.get("gap_affected")),
        snapshot_quality_status=_str_or_none(row.get("snapshot_quality_status")),
        features={column: row.get(column) for column in FEATURE_COLUMNS},
        raw=dict(row),
        has_depth=_int_flag(row.get("has_depth", 0)),
    )


def _required_timestamp(value: object, column: str) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None:
        raise BacktestDatasetError(f"Invalid or missing timestamp column: {column}")
    return parsed


def _timestamp_or_none(value: object, column: str) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = parse_timestamp(value)
    if parsed is None:
        raise BacktestDatasetError(f"Invalid timestamp column: {column}")
    return parsed


def _required_str(value: object, column: str) -> str:
    text = _str_or_none(value)
    if text is None:
        raise BacktestDatasetError(f"Missing required string column: {column}")
    return text


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int_flag(value: object) -> int:
    if value is None or value == "":
        return 0
    return 1 if int(value) else 0


def _optional_int_flag(value: object) -> int | None:
    if value is None or value == "":
        return None
    return _int_flag(value)


def _pct(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator > 0 else 0.0


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "none"


__all__ = [
    "BacktestDataset",
    "BacktestDatasetError",
    "BacktestDatasetSummary",
    "BacktestRecord",
    "FEATURE_COLUMNS",
    "MarketTimeline",
    "REQUIRED_COLUMNS",
    "inspect_backtest_dataset",
    "load_backtest_dataset",
]
