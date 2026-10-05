from __future__ import annotations

import json
import logging
import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .features import MarketFeatureFrame, compute_feature_record, liquidity_sum, mid_and_spread
from .models import (
    FeatureRecord,
    MarketEventRecord,
    MarketMetadata,
    MarketSnapshotRecord,
    OrderBookLevelRecord,
    PriceLevel,
    TradeRecord,
    WSStatusEvent,
    utc_now,
)

_LOG = logging.getLogger(__name__)
_TERMINAL_PLATFORM_STATUSES = {"closed", "resolved", "finalized", "settled"}
_FEATURE_ROLLING_WINDOW_SECONDS = 60
_FEATURE_FRAME_RETENTION_SECONDS = _FEATURE_ROLLING_WINDOW_SECONDS + 5
_TRADE_SNAPSHOT_ALIGNMENT_SECONDS = 2
_FEATURE_HISTORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "price_change_60s",
    "rolling_mean_60s",
    "rolling_volatility_60s",
    "avg_volume_60s",
)


def _normalize_outcome_side(value: str) -> str:
    clean = value.strip().upper()
    if clean in {"Y", "YES", "TRUE"}:
        return "YES"
    if clean in {"N", "NO", "FALSE"}:
        return "NO"
    return clean


def _derive_phase(metadata: MarketMetadata, now: datetime) -> str:
    status = (metadata.platform_status or metadata.status or "").strip().lower()
    if status in _TERMINAL_PLATFORM_STATUSES:
        return "resolved"
    if metadata.start_time is not None and now < metadata.start_time:
        return "future"
    if metadata.close_time is not None and now >= metadata.close_time:
        return "expired_waiting_resolution"
    return "active"


def _missing_levels(levels: Sequence[PriceLevel], expected_depth: int) -> int:
    if expected_depth <= 0:
        return 0
    return max(0, expected_depth - len(levels))


def _orderbook_checksum(
    yes_bids: Sequence[PriceLevel],
    yes_asks: Sequence[PriceLevel],
    no_bids: Sequence[PriceLevel],
    no_asks: Sequence[PriceLevel],
    depth: int,
) -> str:
    payload = {
        "yes_bids": [(level.price, level.size) for level in yes_bids[:depth]],
        "yes_asks": [(level.price, level.size) for level in yes_asks[:depth]],
        "no_bids": [(level.price, level.size) for level in no_bids[:depth]],
        "no_asks": [(level.price, level.size) for level in no_asks[:depth]],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(slots=True)
class MarketRuntimeState:
    metadata: MarketMetadata
    yes_bids: List[PriceLevel] = field(default_factory=list)
    yes_asks: List[PriceLevel] = field(default_factory=list)
    no_bids: List[PriceLevel] = field(default_factory=list)
    no_asks: List[PriceLevel] = field(default_factory=list)
    last_trade: Optional[TradeRecord] = None

    cumulative_volume_delta: float = 0.0

    recent_trades: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=5000))
    seen_trade_ids: Set[str] = field(default_factory=set)
    seen_trade_order: Deque[str] = field(default_factory=lambda: deque(maxlen=20000))
    pending_trades: List[TradeRecord] = field(default_factory=list)
    pending_events: List[MarketEventRecord] = field(default_factory=list)
    feature_frames: Deque[MarketFeatureFrame] = field(default_factory=deque)
    last_snapshot_timestamp: Optional[datetime] = None
    last_snapshot_signature: Optional[
        tuple[
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ] = None
    trade_reject_warn_count: int = 0
    snapshot_trade_attachment_log_counts: Dict[str, int] = field(default_factory=dict)
    feature_readiness_log_counts: Dict[str, int] = field(default_factory=dict)

    def _expected_trade_token_ids(self) -> Set[str]:
        token_ids = {
            str(token)
            for token in (self.metadata.yes_token_id, self.metadata.no_token_id)
            if token
        }
        if token_ids:
            return token_ids
        for side in ("YES", "NO"):
            token = self.metadata.token_ids.get(side)
            if token:
                token_ids.add(str(token))
        return token_ids

    def _extract_trade_token_id(self, trade_id: str) -> Optional[str]:
        if ":" not in trade_id:
            return None
        return trade_id.rsplit(":", 1)[-1]

    def _validate_trade_for_market(self, trade: TradeRecord) -> Optional[str]:
        expected_tokens = self._expected_trade_token_ids()
        trade_token_id = self._extract_trade_token_id(trade.trade_id)
        if expected_tokens and trade_token_id and trade_token_id not in expected_tokens:
            return "token_mismatch"
        if self.metadata.start_time is not None and trade.timestamp < self.metadata.start_time:
            return "trade_before_market_start"
        if self.metadata.close_time is not None and trade.timestamp >= self.metadata.close_time:
            return "trade_after_market_close"
        return None

    def apply_orderbook(
        self,
        outcome_side: str,
        bids: Sequence[PriceLevel],
        asks: Sequence[PriceLevel],
    ) -> None:
        side = _normalize_outcome_side(outcome_side)
        if side == "YES":
            self.yes_bids = list(bids)
            self.yes_asks = list(asks)
        elif side == "NO":
            self.no_bids = list(bids)
            self.no_asks = list(asks)

    def apply_price_level(
        self,
        outcome_side: str,
        book_side: str,
        price: float,
        size: float,
    ) -> None:
        side = _normalize_outcome_side(outcome_side)
        book = book_side.strip().lower()

        if side == "YES" and book == "bid":
            levels = self.yes_bids
        elif side == "YES" and book == "ask":
            levels = self.yes_asks
        elif side == "NO" and book == "bid":
            levels = self.no_bids
        elif side == "NO" and book == "ask":
            levels = self.no_asks
        else:
            return

        updated = False
        for idx, level in enumerate(levels):
            if abs(level.price - price) < 1e-12:
                updated = True
                if size <= 0:
                    del levels[idx]
                else:
                    levels[idx] = PriceLevel(price=price, size=size)
                break

        if not updated and size > 0:
            levels.append(PriceLevel(price=price, size=size))

        if book == "bid":
            levels.sort(key=lambda level: level.price, reverse=True)
        else:
            levels.sort(key=lambda level: level.price)

    def apply_trade(self, trade: TradeRecord) -> bool:
        return self.apply_trade_with_reason(trade) == "accepted"

    def apply_trade_with_reason(self, trade: TradeRecord) -> str:
        if trade.timestamp.tzinfo is None:
            trade.timestamp = trade.timestamp.replace(tzinfo=timezone.utc)
        else:
            trade.timestamp = trade.timestamp.astimezone(timezone.utc)
        trade.run_id = self.metadata.run_id
        invalid_reason = self._validate_trade_for_market(trade)
        if invalid_reason is not None:
            self.trade_reject_warn_count += 1
            if self.trade_reject_warn_count <= 5 or self.trade_reject_warn_count % 100 == 0:
                _LOG.warning(
                    "trade_rejected_for_market",
                    extra={
                        "market_id": self.metadata.market_id,
                        "trade_id": trade.trade_id,
                        "trade_ts": trade.timestamp.isoformat(),
                        "start_time": self.metadata.start_time.isoformat()
                        if self.metadata.start_time
                        else None,
                        "close_time": self.metadata.close_time.isoformat()
                        if self.metadata.close_time
                        else None,
                        "reason": invalid_reason,
                        "rejected_count_for_market": self.trade_reject_warn_count,
                    },
                )
            return str(invalid_reason)
        if trade.trade_id in self.seen_trade_ids:
            # `seen_trade_ids` is only populated after accepted inserts for this
            # specific market runtime, so this verifies same-market duplicates.
            return "duplicate_trade_id"
        if len(self.seen_trade_order) == self.seen_trade_order.maxlen:
            oldest = self.seen_trade_order[0]
            self.seen_trade_ids.discard(oldest)
        self.seen_trade_order.append(trade.trade_id)
        self.seen_trade_ids.add(trade.trade_id)

        if self.last_trade is None or trade.timestamp >= self.last_trade.timestamp:
            self.last_trade = trade
        self.cumulative_volume_delta += trade.size
        self.recent_trades.append(trade)
        self.pending_trades.append(trade)
        return "accepted"

    def apply_status_event(self, status_event: WSStatusEvent) -> None:
        self.pending_events.append(
            MarketEventRecord(
                timestamp=status_event.timestamp,
                market_id=status_event.market_id,
                event_type=status_event.event_type,
                details=status_event.details,
                run_id=self.metadata.run_id,
            )
        )

    def trim_recent_trades(self, now: datetime, seconds: int = 120) -> None:
        cutoff = now - timedelta(seconds=seconds)
        while self.recent_trades and self.recent_trades[0].timestamp < cutoff:
            self.recent_trades.popleft()

    @staticmethod
    def _should_emit_rate_limited_log(
        counts: Dict[str, int], key: str, first_n: int = 5, every_n: int = 100
    ) -> bool:
        count = counts.get(key, 0) + 1
        counts[key] = count
        return count <= first_n or count % every_n == 0

    def _latest_trade_near_snapshot(
        self,
        timestamp: datetime,
        max_age_seconds: int = _TRADE_SNAPSHOT_ALIGNMENT_SECONDS,
    ) -> Optional[TradeRecord]:
        cutoff = timestamp - timedelta(seconds=max_age_seconds)
        candidate: Optional[TradeRecord] = None
        for trade in self.recent_trades:
            if trade.timestamp > timestamp:
                continue
            if trade.timestamp < cutoff:
                continue
            if candidate is None or trade.timestamp > candidate.timestamp:
                candidate = trade
        return candidate

    def _latest_trade_at_or_before_snapshot(
        self,
        timestamp: datetime,
    ) -> Optional[TradeRecord]:
        candidate: Optional[TradeRecord] = None
        for trade in self.recent_trades:
            if trade.timestamp > timestamp:
                continue
            if candidate is None or trade.timestamp > candidate.timestamp:
                candidate = trade
        return candidate

    def _select_trade_for_snapshot(
        self,
        timestamp: datetime,
        max_age_seconds: int,
    ) -> tuple[Optional[TradeRecord], str]:
        aligned_trade = self._latest_trade_near_snapshot(
            timestamp,
            max_age_seconds=max(1, max_age_seconds),
        )
        if aligned_trade is not None:
            return aligned_trade, "aligned_recent_trade"

        if self.last_trade is None:
            return None, "no_accepted_trade_for_market"

        if self.last_trade.timestamp <= timestamp:
            # Fallback: attach the latest accepted same-market trade even when no
            # trade landed inside the narrow alignment window.
            return self.last_trade, "latest_accepted_trade_fallback"

        older_recent_trade = self._latest_trade_at_or_before_snapshot(timestamp)
        if older_recent_trade is not None:
            return older_recent_trade, "recent_trade_before_snapshot_fallback"
        return None, "latest_trade_after_snapshot_timestamp"

    def _validate_trade_for_snapshot(
        self,
        trade: TradeRecord,
        snapshot_timestamp: datetime,
    ) -> Optional[str]:
        if (
            self.metadata.start_time is not None
            and trade.timestamp < self.metadata.start_time
        ):
            return "last_trade_before_market_start"
        if trade.timestamp > snapshot_timestamp:
            return "last_trade_after_snapshot_timestamp"
        if (
            self.metadata.close_time is not None
            and trade.timestamp >= self.metadata.close_time
        ):
            return "last_trade_after_market_close"
        return None

    def _log_snapshot_trade_attachment(
        self,
        *,
        timestamp: datetime,
        selected_mode: str,
        trade: Optional[TradeRecord],
        failure_reason: Optional[str],
    ) -> None:
        status = "attached" if trade is not None and failure_reason is None else "missing"
        reason = failure_reason or selected_mode
        key = f"{status}:{reason}"
        if not self._should_emit_rate_limited_log(
            self.snapshot_trade_attachment_log_counts,
            key,
        ):
            return
        payload = {
            "market_id": self.metadata.market_id,
            "snapshot_timestamp": timestamp.isoformat(),
            "attachment_mode": selected_mode,
            "attachment_status": status,
            "reason": reason,
            "trade_id": trade.trade_id if trade is not None else None,
            "trade_timestamp": trade.timestamp.isoformat() if trade is not None else None,
            "has_trade_data": 1 if trade is not None and failure_reason is None else 0,
        }
        if status == "attached":
            _LOG.info("snapshot_trade_attachment_succeeded", extra=payload)
        else:
            _LOG.info("snapshot_trade_attachment_failed", extra=payload)

    def _snapshot_signature(
        self, snapshot: MarketSnapshotRecord
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        return (
            snapshot.yes_price,
            snapshot.no_price,
            snapshot.best_bid_yes,
            snapshot.best_ask_yes,
            snapshot.best_bid_no,
            snapshot.best_ask_no,
            snapshot.volume,
        )

    def should_write_snapshot(self, snapshot: MarketSnapshotRecord) -> bool:
        signature = self._snapshot_signature(snapshot)
        if self.last_snapshot_signature is None:
            self.last_snapshot_signature = signature
            return True
        if signature != self.last_snapshot_signature:
            self.last_snapshot_signature = signature
            return True
        return False

    def _trim_feature_frames(self, now: datetime) -> None:
        # Keep a small buffer beyond the strict 60s lookback. With real-world
        # scheduler jitter (e.g. ~1.00-1.02s loop cadence), trimming exactly at
        # 60s can drop the only candidate frame for *_60s deltas.
        cutoff = now - timedelta(seconds=_FEATURE_FRAME_RETENTION_SECONDS)
        while self.feature_frames and self.feature_frames[0].timestamp < cutoff:
            self.feature_frames.popleft()

    def current_volume(self) -> Optional[float]:
        base = self.metadata.initial_volume
        if base is None and self.cumulative_volume_delta == 0:
            return None
        return float((base or 0.0) + self.cumulative_volume_delta)

    def has_usable_orderbook(self) -> bool:
        return bool(self.yes_bids and self.yes_asks)

    def is_snapshot_window_open(self, now: datetime) -> bool:
        phase = self.metadata.phase or self.metadata.market_phase
        if phase is not None and phase != "active":
            return False
        if self.metadata.start_time is not None and now < self.metadata.start_time:
            return False
        if self.metadata.close_time is not None and now >= self.metadata.close_time:
            return False
        return True

    def build_snapshot(
        self,
        timestamp: datetime,
        liquidity_depth: int,
        expected_depth: int,
        expected_snapshot_interval_sec: float,
        gap_threshold_multiplier: float,
        trade_snapshot_freshness_sec: int,
    ) -> tuple[MarketSnapshotRecord, Optional[dict[str, str]]]:
        best_bid_yes = self.yes_bids[0].price if self.yes_bids else None
        best_ask_yes = self.yes_asks[0].price if self.yes_asks else None
        best_bid_no = self.no_bids[0].price if self.no_bids else None
        best_ask_no = self.no_asks[0].price if self.no_asks else None

        mid_yes, spread_yes = mid_and_spread(best_bid_yes, best_ask_yes)
        mid_no, spread_no = mid_and_spread(best_bid_no, best_ask_no)

        yes_price = mid_yes
        no_price = mid_no

        liquidity = (
            liquidity_sum(self.yes_bids, liquidity_depth)
            + liquidity_sum(self.yes_asks, liquidity_depth)
            + liquidity_sum(self.no_bids, liquidity_depth)
            + liquidity_sum(self.no_asks, liquidity_depth)
        )

        if liquidity == 0 and self.metadata.initial_liquidity is not None:
            liquidity = self.metadata.initial_liquidity

        aligned_trade, trade_selection_mode = self._select_trade_for_snapshot(
            timestamp=timestamp,
            max_age_seconds=trade_snapshot_freshness_sec,
        )
        last_trade_price = aligned_trade.price if aligned_trade is not None else None
        last_trade_size = aligned_trade.size if aligned_trade is not None else None
        last_trade_time = aligned_trade.timestamp if aligned_trade is not None else None
        trade_warning: Optional[dict[str, str]] = None
        invalid_trade_reason: Optional[str] = (
            self._validate_trade_for_snapshot(aligned_trade, timestamp)
            if aligned_trade is not None
            else None
        )
        invalid_trade_ts: Optional[str] = (
            aligned_trade.timestamp.isoformat() if aligned_trade is not None else None
        )

        if invalid_trade_reason is not None:
            trade_warning = {
                "market_id": self.metadata.market_id,
                "snapshot_timestamp": timestamp.isoformat(),
                "invalid_trade_timestamp": invalid_trade_ts or "",
                "market_start_time": self.metadata.start_time.isoformat()
                if self.metadata.start_time
                else "",
                "reason": invalid_trade_reason,
            }
            last_trade_price = None
            last_trade_size = None
            last_trade_time = None

        has_orderbook = 1 if (best_bid_yes is not None and best_ask_yes is not None) else 0
        has_trade_data = (
            1
            if (
                aligned_trade is not None
                and last_trade_price is not None
                and last_trade_size is not None
                and last_trade_time is not None
            )
            else 0
        )
        if has_trade_data == 1 and (
            last_trade_price is None or last_trade_size is None or last_trade_time is None
        ):
            trade_warning = {
                "market_id": self.metadata.market_id,
                "snapshot_timestamp": timestamp.isoformat(),
                "invalid_trade_timestamp": last_trade_time.isoformat()
                if last_trade_time is not None
                else "",
                "market_start_time": self.metadata.start_time.isoformat()
                if self.metadata.start_time
                else "",
                "reason": "has_trade_data_inconsistent",
            }

        self._log_snapshot_trade_attachment(
            timestamp=timestamp,
            selected_mode=trade_selection_mode,
            trade=aligned_trade,
            failure_reason=invalid_trade_reason,
        )

        missing_level_count = (
            _missing_levels(self.yes_bids, expected_depth)
            + _missing_levels(self.yes_asks, expected_depth)
            + _missing_levels(self.no_bids, expected_depth)
            + _missing_levels(self.no_asks, expected_depth)
        )
        is_partial_orderbook = 1 if missing_level_count > 0 else 0

        time_gap_from_prev_snapshot_sec: Optional[float] = None
        is_gap_affected = 0
        if self.last_snapshot_timestamp is not None:
            time_gap_from_prev_snapshot_sec = max(
                0.0, (timestamp - self.last_snapshot_timestamp).total_seconds()
            )
            gap_threshold = max(0.0, expected_snapshot_interval_sec) * max(
                1.0, gap_threshold_multiplier
            )
            if gap_threshold > 0 and time_gap_from_prev_snapshot_sec > gap_threshold:
                is_gap_affected = 1

        snapshot_quality_status = "ok"
        if is_partial_orderbook and is_gap_affected:
            snapshot_quality_status = "partial_orderbook_and_gap"
        elif is_partial_orderbook:
            snapshot_quality_status = "partial_orderbook"
        elif is_gap_affected:
            snapshot_quality_status = "gap_affected"

        book_checksum = _orderbook_checksum(
            yes_bids=self.yes_bids,
            yes_asks=self.yes_asks,
            no_bids=self.no_bids,
            no_asks=self.no_asks,
            depth=max(1, expected_depth),
        )

        snapshot = MarketSnapshotRecord(
            timestamp=timestamp,
            market_id=self.metadata.market_id,
            run_id=self.metadata.run_id,
            yes_price=yes_price,
            no_price=no_price,
            best_bid_yes=best_bid_yes,
            best_ask_yes=best_ask_yes,
            best_bid_no=best_bid_no,
            best_ask_no=best_ask_no,
            spread_yes=spread_yes,
            spread_no=spread_no,
            mid_price_yes=mid_yes,
            mid_price_no=mid_no,
            volume=self.current_volume(),
            liquidity=liquidity,
            last_trade_price=last_trade_price,
            last_trade_size=last_trade_size,
            last_trade_time=last_trade_time,
            has_orderbook=has_orderbook,
            has_trade_data=has_trade_data,
            strict_validation_passed=self.metadata.strict_validation_passed,
            snapshot_quality_status=snapshot_quality_status,
            is_partial_orderbook=is_partial_orderbook,
            missing_level_count=missing_level_count,
            time_gap_from_prev_snapshot_sec=time_gap_from_prev_snapshot_sec,
            is_gap_affected=is_gap_affected,
            feature_ready=0,
            book_checksum=book_checksum,
            recorder_version=self.metadata.recorder_version,
            schema_version=self.metadata.schema_version,
        )
        self.last_snapshot_timestamp = timestamp
        return snapshot, trade_warning

    def build_orderbook_levels(
        self, timestamp: datetime, levels_to_store: int
    ) -> List[OrderBookLevelRecord]:
        records: List[OrderBookLevelRecord] = []

        def add_records(
            outcome_side: str,
            book_side: str,
            levels: Sequence[PriceLevel],
        ) -> None:
            for idx, level in enumerate(levels[:levels_to_store], start=1):
                records.append(
                    OrderBookLevelRecord(
                        timestamp=timestamp,
                        market_id=self.metadata.market_id,
                        outcome_side=outcome_side,
                        book_side=book_side,
                        level=idx,
                        price=level.price,
                        size=level.size,
                        run_id=self.metadata.run_id,
                        recorder_version=self.metadata.recorder_version,
                        schema_version=self.metadata.schema_version,
                    )
                )

        add_records("YES", "bid", self.yes_bids)
        add_records("YES", "ask", self.yes_asks)
        add_records("NO", "bid", self.no_bids)
        add_records("NO", "ask", self.no_asks)
        return records

    def _evaluate_feature_readiness(
        self,
        snapshot: MarketSnapshotRecord,
        feature_record: FeatureRecord,
    ) -> tuple[int, list[str], list[str], list[str]]:
        reasons: list[str] = []
        history_missing: list[str] = []
        trade_missing: list[str] = []

        if self.metadata.strict_validation_passed != 1:
            reasons.append("strict_validation_failed")
        if snapshot.is_partial_orderbook == 1:
            reasons.append("partial_orderbook")
        if snapshot.is_gap_affected == 1:
            reasons.append("gap_affected")

        for field_name in _FEATURE_HISTORY_REQUIRED_FIELDS:
            if getattr(feature_record, field_name) is None:
                history_missing.append(field_name)
        if history_missing:
            reasons.append("insufficient_history")

        if snapshot.has_trade_data != 1:
            trade_missing.append("snapshot_missing_last_trade")
        if feature_record.net_trade_flow_5s is None:
            trade_missing.append("trade_flow_window_empty_or_unrecognized")
        if trade_missing:
            reasons.append("missing_trade_inputs")

        feature_ready = 1 if not reasons else 0
        return feature_ready, reasons, history_missing, trade_missing

    def _log_feature_readiness(
        self,
        *,
        snapshot: MarketSnapshotRecord,
        feature_inserted: bool,
        feature_ready: int,
        reasons: Sequence[str],
        history_missing: Sequence[str],
        trade_missing: Sequence[str],
    ) -> None:
        if not feature_inserted:
            key = "not_inserted:missing_mid_price"
            if not self._should_emit_rate_limited_log(
                self.feature_readiness_log_counts,
                key,
            ):
                return
            _LOG.info(
                "feature_row_not_inserted",
                extra={
                    "market_id": self.metadata.market_id,
                    "snapshot_timestamp": snapshot.timestamp.isoformat(),
                    "reason": "missing_mid_price",
                },
            )
            return

        if feature_ready == 1:
            key = "ready"
            if not self._should_emit_rate_limited_log(
                self.feature_readiness_log_counts,
                key,
            ):
                return
            _LOG.info(
                "feature_row_ready",
                extra={
                    "market_id": self.metadata.market_id,
                    "snapshot_timestamp": snapshot.timestamp.isoformat(),
                },
            )
            return

        reason_key = ",".join(sorted(set(reasons))) if reasons else "unknown"
        if not self._should_emit_rate_limited_log(
            self.feature_readiness_log_counts,
            f"not_ready:{reason_key}",
        ):
            return
        _LOG.info(
            "feature_row_not_ready",
            extra={
                "market_id": self.metadata.market_id,
                "snapshot_timestamp": snapshot.timestamp.isoformat(),
                "reasons": list(reasons),
                "missing_history_fields": list(history_missing),
                "missing_trade_inputs": list(trade_missing),
                "strict_validation_passed": self.metadata.strict_validation_passed,
                "snapshot_is_partial_orderbook": snapshot.is_partial_orderbook,
                "snapshot_is_gap_affected": snapshot.is_gap_affected,
            },
        )

    def build_feature_record(
        self,
        snapshot: MarketSnapshotRecord,
        feature_orderbook_depth: int,
        jump_threshold: float,
    ) -> Optional[FeatureRecord]:
        self._trim_feature_frames(snapshot.timestamp)
        total_bid_liquidity_yes = liquidity_sum(self.yes_bids, feature_orderbook_depth)
        total_ask_liquidity_yes = liquidity_sum(self.yes_asks, feature_orderbook_depth)

        feature_record = compute_feature_record(
            snapshot=snapshot,
            frame_history=list(self.feature_frames),
            recent_trades=list(self.recent_trades),
            yes_bids=self.yes_bids,
            yes_asks=self.yes_asks,
            market_start_time=self.metadata.start_time,
            market_close_time=self.metadata.close_time,
            orderbook_depth=feature_orderbook_depth,
            jump_threshold=jump_threshold,
        )

        frame = MarketFeatureFrame(
            timestamp=snapshot.timestamp,
            mid_price_yes=snapshot.mid_price_yes,
            volume=snapshot.volume,
            liquidity=snapshot.liquidity,
            total_bid_liquidity_yes=total_bid_liquidity_yes,
            total_ask_liquidity_yes=total_ask_liquidity_yes,
        )
        self.feature_frames.append(frame)
        self._trim_feature_frames(snapshot.timestamp)

        if feature_record is not None:
            feature_record.run_id = self.metadata.run_id
            feature_record.strict_validation_passed = self.metadata.strict_validation_passed
            feature_record.is_gap_affected = snapshot.is_gap_affected
            feature_record.snapshot_quality_status = snapshot.snapshot_quality_status
            feature_record.recorder_version = self.metadata.recorder_version
            feature_record.schema_version = self.metadata.schema_version
            (
                feature_record.feature_ready,
                readiness_reasons,
                missing_history_fields,
                missing_trade_inputs,
            ) = self._evaluate_feature_readiness(
                snapshot=snapshot,
                feature_record=feature_record,
            )
            snapshot.feature_ready = feature_record.feature_ready
            self._log_feature_readiness(
                snapshot=snapshot,
                feature_inserted=True,
                feature_ready=feature_record.feature_ready,
                reasons=readiness_reasons,
                history_missing=missing_history_fields,
                trade_missing=missing_trade_inputs,
            )
        else:
            snapshot.feature_ready = 0
            self._log_feature_readiness(
                snapshot=snapshot,
                feature_inserted=False,
                feature_ready=0,
                reasons=["missing_mid_price"],
                history_missing=[],
                trade_missing=[],
            )

        return feature_record


class RecorderState:
    """In-memory state of tracked markets and token mappings."""

    def __init__(self) -> None:
        self.markets: Dict[str, MarketRuntimeState] = {}
        self.token_to_market: Dict[str, Tuple[str, str]] = {}
        self.last_cycle_snapshot_no_change: List[str] = []
        self.last_cycle_snapshot_generation_failed: List[str] = []

    def market_count(self) -> int:
        return len(self.markets)

    def all_token_ids(self) -> List[str]:
        return list(self.token_to_market.keys())

    def upsert_markets(self, markets: Iterable[MarketMetadata]) -> List[MarketEventRecord]:
        now = utc_now()
        new_events: List[MarketEventRecord] = []

        for metadata in markets:
            market_id = metadata.market_id
            existing = self.markets.get(market_id)

            # Remove stale token mapping for this market before remapping.
            for token_id, mapping in list(self.token_to_market.items()):
                if mapping[0] == market_id:
                    del self.token_to_market[token_id]

            if existing is None:
                runtime = MarketRuntimeState(metadata=metadata)
                self.markets[market_id] = runtime
            else:
                old_status = existing.metadata.platform_status or existing.metadata.status
                existing.metadata = metadata
                new_status = metadata.platform_status or metadata.status
                if old_status != new_status and new_status is not None:
                    event_type = _status_to_event_type(new_status)
                    if event_type == "market_opened":
                        # `market_opened` is emitted centrally by discovery-phase transitions.
                        event_type = "status_changed"
                    new_events.append(
                        MarketEventRecord(
                            timestamp=now,
                            market_id=market_id,
                            event_type=event_type,
                            details=json.dumps(
                                {
                                    "old_platform_status": old_status,
                                    "new_platform_status": new_status,
                                }
                            ),
                            run_id=metadata.run_id,
                            recorder_version=metadata.recorder_version,
                            schema_version=metadata.schema_version,
                        )
                    )

            if metadata.yes_token_id:
                self.token_to_market[metadata.yes_token_id] = (market_id, "YES")
            if metadata.no_token_id:
                self.token_to_market[metadata.no_token_id] = (market_id, "NO")

            # Backward-compatible fallback when YES/NO token ids are unavailable.
            if not metadata.yes_token_id or not metadata.no_token_id:
                for outcome_label, token_id in metadata.token_ids.items():
                    if not token_id:
                        continue
                    side = _normalize_outcome_side(outcome_label)
                    if side not in {"YES", "NO"}:
                        continue
                    self.token_to_market[token_id] = (market_id, side)

        return new_events

    def prune_inactive_markets(
        self,
        active_market_ids: Set[str],
        now: Optional[datetime] = None,
    ) -> tuple[List[MarketEventRecord], List[str], List[str], List[tuple[str, str]]]:
        now_ts = now or utc_now()
        events: List[MarketEventRecord] = []
        removed_markets: List[str] = []
        removed_tokens: List[str] = []
        status_updates: List[tuple[str, str]] = []

        for market_id, runtime in list(self.markets.items()):
            if market_id in active_market_ids:
                continue

            removed_markets.append(market_id)
            del self.markets[market_id]

            status = (
                runtime.metadata.platform_status
                or runtime.metadata.status
                or ""
            ).strip().lower()
            is_expired = (
                runtime.metadata.close_time is not None
                and runtime.metadata.close_time <= now_ts
            )
            is_terminal = status in {"closed", "resolved", "finalized", "settled"}

            previous_phase = runtime.metadata.phase or runtime.metadata.market_phase
            # Runtime deselection does not imply market closure. Terminal status is
            # emitted from raw platform status only; otherwise we emit phase changes.
            if is_expired or is_terminal:
                if status in {"resolved", "settled"}:
                    event_type = "market_resolved"
                    status_updates.append((market_id, "resolved"))
                    new_phase = "resolved"
                elif status in {"closed", "finalized"}:
                    event_type = "market_closed"
                    status_updates.append((market_id, "closed"))
                    new_phase = "resolved"
                else:
                    event_type = "status_changed"
                    new_phase = "expired_waiting_resolution"
                events.append(
                    MarketEventRecord(
                        timestamp=now_ts,
                        market_id=market_id,
                        event_type=event_type,
                        details=json.dumps(
                            {
                                "reason": "phase_transition_on_deselection",
                                "previous_phase": previous_phase,
                                "new_phase": new_phase,
                            }
                        ),
                        run_id=runtime.metadata.run_id,
                        recorder_version=runtime.metadata.recorder_version,
                        schema_version=runtime.metadata.schema_version,
                    )
                )

            for token_id, mapping in list(self.token_to_market.items()):
                if mapping[0] == market_id:
                    removed_tokens.append(token_id)
                    del self.token_to_market[token_id]

        return events, removed_markets, removed_tokens, status_updates

    def apply_orderbook_update(
        self,
        token_id: str,
        bids: Sequence[PriceLevel],
        asks: Sequence[PriceLevel],
    ) -> bool:
        mapping = self.token_to_market.get(token_id)
        if mapping is None:
            return False
        market_id, outcome_side = mapping
        market = self.markets.get(market_id)
        if market is None:
            return False

        market.apply_orderbook(outcome_side=outcome_side, bids=bids, asks=asks)
        return True

    def apply_trade(self, trade: TradeRecord) -> bool:
        market = self.markets.get(trade.market_id)
        if market is None:
            return False
        return market.apply_trade(trade)

    def apply_trade_with_reason(self, trade: TradeRecord) -> str:
        market = self.markets.get(trade.market_id)
        if market is None:
            return "unknown_market"
        return market.apply_trade_with_reason(trade)

    def apply_price_change(
        self,
        token_id: str,
        book_side: str,
        price: float,
        size: float,
    ) -> bool:
        mapping = self.token_to_market.get(token_id)
        if mapping is None:
            return False
        market_id, outcome_side = mapping
        market = self.markets.get(market_id)
        if market is None:
            return False
        market.apply_price_level(
            outcome_side=outcome_side,
            book_side=book_side,
            price=price,
            size=size,
        )
        return True

    def apply_status_events(self, events: Sequence[WSStatusEvent]) -> None:
        for event in events:
            if event.event_type == "market_opened":
                # Open events are managed from deterministic phase transitions.
                continue
            market = self.markets.get(event.market_id)
            if market is not None:
                market.apply_status_event(event)

    def reset_market_runtime(self, market_id: str) -> None:
        market = self.markets.get(market_id)
        if market is None:
            return
        # Rotation safety: this market becomes a fresh tracking target, so
        # dedupe and recent-trade caches must not leak historical context.
        market.last_trade = None
        market.cumulative_volume_delta = 0.0
        market.recent_trades.clear()
        market.seen_trade_ids.clear()
        market.seen_trade_order.clear()
        market.trade_reject_warn_count = 0
        market.snapshot_trade_attachment_log_counts.clear()
        market.feature_readiness_log_counts.clear()
        market.pending_trades = []
        market.pending_events = []
        market.feature_frames.clear()
        market.last_snapshot_signature = None
        market.last_snapshot_timestamp = None

    def build_cycle_records(
        self,
        timestamp: datetime,
        levels_to_store: int,
        feature_orderbook_depth: int,
        jump_threshold: float,
        expected_snapshot_interval_sec: float,
        gap_threshold_multiplier: float,
        trade_snapshot_freshness_sec: int,
    ) -> tuple[
        List[MarketSnapshotRecord],
        List[OrderBookLevelRecord],
        List[FeatureRecord],
        List[TradeRecord],
        List[MarketEventRecord],
        List[str],
        List[str],
        List[dict[str, str]],
        List[str],
    ]:
        snapshots: List[MarketSnapshotRecord] = []
        orderbook_levels: List[OrderBookLevelRecord] = []
        features: List[FeatureRecord] = []
        trades: List[TradeRecord] = []
        events: List[MarketEventRecord] = []
        skipped_outside_window: List[str] = []
        skipped_missing_orderbook: List[str] = []
        snapshot_trade_warnings: List[dict[str, str]] = []
        invalid_active_past_close: List[str] = []
        snapshot_no_change: List[str] = []
        snapshot_generation_failed: List[str] = []

        for market in self.markets.values():
            previous_phase = market.metadata.phase or market.metadata.market_phase
            derived_phase = _derive_phase(market.metadata, timestamp)
            if previous_phase != derived_phase:
                market.metadata.phase = derived_phase
                market.metadata.market_phase = derived_phase
            if (
                previous_phase == "active"
                and market.metadata.close_time is not None
                and timestamp >= market.metadata.close_time
            ):
                invalid_active_past_close.append(market.metadata.market_id)

            market.trim_recent_trades(timestamp)
            if not market.is_snapshot_window_open(timestamp):
                skipped_outside_window.append(market.metadata.market_id)
            elif market.has_usable_orderbook():
                try:
                    snapshot, trade_warning = market.build_snapshot(
                        timestamp,
                        liquidity_depth=levels_to_store,
                        expected_depth=levels_to_store,
                        expected_snapshot_interval_sec=expected_snapshot_interval_sec,
                        gap_threshold_multiplier=gap_threshold_multiplier,
                        trade_snapshot_freshness_sec=trade_snapshot_freshness_sec,
                    )
                    should_write_snapshot = market.should_write_snapshot(snapshot)
                    if should_write_snapshot:
                        snapshots.append(snapshot)
                        if trade_warning is not None:
                            snapshot_trade_warnings.append(trade_warning)
                        orderbook_levels.extend(
                            market.build_orderbook_levels(
                                timestamp=timestamp,
                                levels_to_store=levels_to_store,
                            )
                        )
                    else:
                        snapshot_no_change.append(market.metadata.market_id)

                    feature_record = market.build_feature_record(
                        snapshot=snapshot,
                        feature_orderbook_depth=feature_orderbook_depth,
                        jump_threshold=jump_threshold,
                    )
                    if feature_record is not None:
                        features.append(feature_record)
                except Exception:
                    snapshot_generation_failed.append(market.metadata.market_id)
                    _LOG.exception(
                        "snapshot_generation_failed",
                        extra={"market_id": market.metadata.market_id},
                    )
            else:
                skipped_missing_orderbook.append(market.metadata.market_id)

            if market.pending_trades:
                trades.extend(market.pending_trades)
                market.pending_trades = []

            if market.pending_events:
                events.extend(market.pending_events)
                market.pending_events = []

        self.last_cycle_snapshot_no_change = snapshot_no_change
        self.last_cycle_snapshot_generation_failed = snapshot_generation_failed

        return (
            snapshots,
            orderbook_levels,
            features,
            trades,
            events,
            skipped_outside_window,
            skipped_missing_orderbook,
            snapshot_trade_warnings,
            invalid_active_past_close,
        )



def _status_to_event_type(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"open", "active"}:
        return "market_opened"
    if normalized in {"closed", "finalized"}:
        return "market_closed"
    if normalized in {"resolved", "settled"}:
        return "market_resolved"
    if normalized in {"halted", "paused", "suspended"}:
        return "trading_halted"
    return "status_changed"
