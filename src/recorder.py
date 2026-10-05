from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence

from .btc_price_feed import (
    BTCPriceFeed,
    BTCPriceFeedError,
    build_btc_price_feed,
    build_btc_price_feeds,
    describe_btc_price_feed_config,
    is_btc_price_feed_source_supported,
)
from .config import Settings
from .db import SQLiteStore, SnapshotInsertShapeError
from .models import (
    BTCPriceSampleRecord,
    MarketEventRecord,
    MarketMetadata,
    RECORDER_VERSION,
    RecorderMetricRecord,
    SCHEMA_VERSION,
    local_arrival_iso_from_ns,
    parse_timestamp,
    to_iso,
    utc_now,
)
from .polymarket.clob_client import ClobClient
from .polymarket.gamma_client import GammaClient
from .polymarket.normalization import (
    normalize_raw_polymarket_events,
    normalize_ws_message,
)
from .polymarket.ws_client import WSClient
from .state import RecorderState
from .market_validation import validate_strict_btc_5m_market


_RESOLVED_STATUSES = {"closed", "resolved", "finalized", "settled"}


@dataclass(slots=True)
class TradeBackfillCursor:
    last_timestamp: datetime
    last_trade_id: str = ""


def _normalized_status(status: Optional[str]) -> str:
    return (status or "").strip().lower()


def validate_strict_market(market: MarketMetadata) -> tuple[bool, Optional[str]]:
    result = validate_strict_btc_5m_market(market)
    return result.is_valid, result.reason


def classify_market_phase(market: MarketMetadata, now: datetime) -> str:
    """
    Application-level phase model:
    - future: now < start_time
    - active: start_time <= now < close_time
    - expired_waiting_resolution: now >= close_time while not resolved by status
    - resolved: resolved/closed/finalized/settled status
    """
    status = _normalized_status(market.platform_status or market.status)
    if status in _RESOLVED_STATUSES:
        return "resolved"
    if market.start_time is not None and now < market.start_time:
        return "future"
    if market.close_time is not None and now >= market.close_time:
        return "expired_waiting_resolution"
    return "active"


def _market_start_sort_key(market: MarketMetadata, now: datetime) -> datetime:
    return market.start_time or market.close_time or now


def _market_close_sort_key(market: MarketMetadata, now: datetime) -> datetime:
    return market.close_time or market.start_time or now


def select_primary_market(
    candidates: Sequence[MarketMetadata],
    now: datetime,
) -> tuple[
    Optional[MarketMetadata],
    Optional[str],
    dict[str, str],
    dict[str, str],
    Counter[str],
    dict[str, str],
]:
    tracking_states: dict[str, str] = {}
    market_phases: dict[str, str] = {}
    skipped_counts: Counter[str] = Counter()
    selection_reasons: dict[str, str] = {}

    active_strict: list[MarketMetadata] = []
    future_strict: list[MarketMetadata] = []

    for market in candidates:
        market_id = market.market_id
        phase = classify_market_phase(market, now)
        market_phases[market_id] = phase

        strict = validate_strict_btc_5m_market(market)
        market.strict_validation_passed = 1 if strict.is_valid else 0
        market.strict_rejection_reason = strict.reason
        market.last_updated = now

        if not strict.is_valid:
            reason = f"strict_rejected_{strict.reason or 'unknown'}"
            tracking_states[market_id] = "inactive"
            selection_reasons[market_id] = reason
            skipped_counts[reason] += 1
            continue

        if phase == "active":
            active_strict.append(market)
            continue
        if phase == "future":
            future_strict.append(market)
            tracking_states[market_id] = "discovered"
            continue

        tracking_states[market_id] = "inactive"
        selection_reasons[market_id] = f"phase_{phase}"
        skipped_counts["resolved_not_selected"] += 1

    selected: Optional[MarketMetadata] = None
    selected_mode: Optional[str] = None
    if active_strict:
        selected = min(active_strict, key=lambda m: _market_close_sort_key(m, now))
        selected_mode = "selected_active"
    elif future_strict:
        selected = min(future_strict, key=lambda m: _market_start_sort_key(m, now))
        selected_mode = "future_standby"

    for market in active_strict:
        if selected is market and selected_mode == "selected_active":
            tracking_states[market.market_id] = "selected_active"
            selection_reasons[market.market_id] = "selected_active"
        else:
            tracking_states[market.market_id] = "inactive"
            selection_reasons[market.market_id] = "active_not_selected"
            skipped_counts["active_not_selected"] += 1

    for market in future_strict:
        tracking_states[market.market_id] = "discovered"
        if selected is market and selected_mode == "future_standby":
            selection_reasons[market.market_id] = "selected_future_standby"
        else:
            selection_reasons[market.market_id] = "future_not_selected"
            skipped_counts["future_not_selected"] += 1

    if selected is not None:
        selected.phase = market_phases.get(selected.market_id)
        selected.market_phase = selected.phase
        selected.tracking_state = tracking_states.get(selected.market_id)

    return (
        selected,
        selected_mode,
        tracking_states,
        market_phases,
        skipped_counts,
        selection_reasons,
    )


def _cursor_trade_id(value: Any) -> str:
    return str(value or "")


def _is_before_or_at_cursor(trade: Any, cursor: TradeBackfillCursor) -> bool:
    trade_ts = getattr(trade, "timestamp", None)
    cursor_ts = cursor.last_timestamp
    if trade_ts is None or cursor_ts is None:
        return False
    if trade_ts < cursor_ts:
        return True
    if trade_ts > cursor_ts:
        return False
    cursor_trade_id = _cursor_trade_id(cursor.last_trade_id)
    if not cursor_trade_id:
        return False
    return _cursor_trade_id(getattr(trade, "trade_id", "")) <= cursor_trade_id


def _advance_cursor_if_newer(trade: Any, cursor: TradeBackfillCursor) -> None:
    trade_ts = getattr(trade, "timestamp", None)
    cursor_ts = cursor.last_timestamp
    if trade_ts is None or cursor_ts is None:
        return
    trade_id = _cursor_trade_id(getattr(trade, "trade_id", ""))
    if trade_ts > cursor_ts or (
        trade_ts == cursor_ts and trade_id > _cursor_trade_id(cursor.last_trade_id)
    ):
        cursor.last_timestamp = trade_ts
        cursor.last_trade_id = trade_id


def _market_ws_asset_ids(market: MarketMetadata) -> set[str]:
    tokens = {str(token) for token in (market.yes_token_id, market.no_token_id) if token}
    for side in ("YES", "NO"):
        token = market.token_ids.get(side)
        if token:
            tokens.add(str(token))
    return tokens


class RecorderApp:
    """Orchestrates discovery, websocket ingestion, snapshots, and persistence."""

    def __init__(self, settings: Settings, run_id: str) -> None:
        self.settings = settings
        self.log = logging.getLogger(self.__class__.__name__)
        self.run_id = str(run_id)
        self.db = SQLiteStore(
            db_path=settings.db_path,
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            run_id=self.run_id,
            startup_integrity_check_mode=settings.startup_integrity_check_mode,
            startup_integrity_logger=self.log,
        )
        self.gamma = GammaClient(
            base_url=settings.gamma_api_url,
            timeout_sec=settings.request_timeout_sec,
            page_size=settings.discovery_page_size,
            max_pages=settings.discovery_max_pages,
            lookahead_sec=settings.discovery_lookahead_sec,
        )
        self.clob = ClobClient(
            clob_base_url=settings.clob_api_url,
            data_api_base_url=settings.data_api_url,
            timeout_sec=settings.request_timeout_sec,
        )
        self.ws = WSClient(
            url=settings.ws_url,
            ping_interval_sec=settings.ws_ping_interval_sec,
            ping_timeout_sec=settings.ws_ping_timeout_sec,
            reconnect_min_sec=settings.ws_reconnect_min_sec,
            reconnect_max_sec=settings.ws_reconnect_max_sec,
        )
        self.btc_price_feeds: dict[str, BTCPriceFeed] = (
            build_btc_price_feeds(settings)
            if self._btc_price_feed_enabled_for_settings(settings)
            else {}
        )
        self.btc_price_feed: Optional[BTCPriceFeed] = (
            next(iter(self.btc_price_feeds.values()), None)
            if self.btc_price_feeds
            else (
                build_btc_price_feed(settings)
                if self._btc_price_feed_enabled_for_settings(settings)
                else None
            )
        )
        self._btc_feed_health: dict[str, dict[str, Any]] = {}
        self._log_btc_price_feed_startup_config()
        self.state = RecorderState()
        self.stop_event = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[Any]] = []

        self._last_api_latency_ms: Optional[float] = None
        self._last_successful_market_fetches = 0
        self._last_failed_markets = 0
        self._last_discovery_candidates_seen = 0
        self._last_discovery_candidates_matched = 0
        self._last_discovery_strict_5m_candidates = 0
        self._last_discovery_broad_btc_candidates = 0
        self._last_discovery_fallback_used = 0
        self._last_heartbeat_monotonic = 0.0
        self._last_snapshot_quality_rollup: Optional[dict[str, Any]] = None
        self._backfill_cursor: dict[str, TradeBackfillCursor] = {}
        self._last_trade_backfill_summary: dict[str, Any] = {
            "fetched_trades": 0,
            "accepted_trades": 0,
            "rejected_before_start": 0,
            "rejected_after_close": 0,
            "skipped_already_seen": 0,
            "last_accepted_trade_ts": None,
        }
        self._active_primary_market_id: Optional[str] = None
        self._api_call_count_since_snapshot = 0
        self._api_success_count_since_snapshot = 0
        self._api_failure_count_since_snapshot = 0
        self._raw_ws_events_seen = 0
        self._raw_ws_events_written = 0
        self._raw_ws_malformed_events = 0
        self._raw_ws_write_failures = 0
        self._raw_ws_events_skipped_logged = False
        self._ws_reconnect_count = 0
        self._last_ws_event_monotonic: Optional[float] = None
        self._desired_ws_asset_ids: set[str] = set()
        self._stale_ws_asset_deadlines: dict[str, datetime] = {}
        self.log.info(
            "raw_ws_events_persistence_enabled",
            extra={
                "raw_ws_events_enabled": self._raw_ws_events_persistence_enabled(),
                "raw_ws_events_retention_sec": getattr(
                    self.settings,
                    "raw_ws_events_retention_sec",
                    0.0,
                ),
                "raw_ws_events_max_rows": getattr(
                    self.settings,
                    "raw_ws_events_max_rows",
                    0,
                ),
                "raw_ws_events_prune_batch_size": getattr(
                    self.settings,
                    "raw_ws_events_prune_batch_size",
                    50000,
                ),
            },
        )

    async def run(self) -> None:
        self.db.init_schema()
        self.log.info(
            "db_initialized",
            extra={"db_path": self.settings.db_path, "run_id": self.run_id},
        )
        self.log.info(
            "recorder_metrics_semantics",
            extra={
                "markets_polled": "legacy alias for active_markets_snapshot_attempted",
                "successful_market_fetches": "legacy alias for discovery_candidates_matched",
                "failed_markets": "legacy alias for api_failure_count",
            },
        )
        await self._discover_and_update(trigger_orderbook_sync=False)
        await self._sync_all_known_orderbooks(reason="startup")

        self._tasks = self._create_runtime_tasks()
        try:
            await self.stop_event.wait()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self.stop_event.set()

    def _create_runtime_tasks(self) -> list[asyncio.Task[Any]]:
        tasks = [
            asyncio.create_task(self._discovery_loop(), name="discovery_loop"),
            asyncio.create_task(self._snapshot_loop(), name="snapshot_loop"),
            asyncio.create_task(self._ws_loop(), name="ws_loop"),
            asyncio.create_task(self._trade_backfill_loop(), name="trade_backfill_loop"),
        ]
        if self._btc_price_feed_enabled():
            btc_feeds = getattr(self, "btc_price_feeds", None)
            if isinstance(btc_feeds, dict) and btc_feeds:
                for source in sorted(btc_feeds):
                    self.log.info(
                        "btc_price_feed_task_created",
                        extra=self._btc_price_feed_task_extra(
                            source,
                            btc_feeds.get(source),
                        ),
                    )
                    tasks.append(
                        asyncio.create_task(
                            self._btc_price_feed_loop(source=source),
                            name=f"btc_price_feed_loop:{source}",
                        )
                    )
            else:
                source, feed = self._resolve_btc_price_feed()
                self.log.info(
                    "btc_price_feed_task_created",
                    extra=self._btc_price_feed_task_extra(source, feed),
                )
                tasks.append(
                    asyncio.create_task(
                        self._btc_price_feed_loop(),
                        name="btc_price_feed_loop",
                    )
                )
        if self._auto_normalize_resolutions_enabled():
            tasks.append(
                asyncio.create_task(
                    self._auto_normalize_resolutions_loop(),
                    name="auto_normalize_resolutions_loop",
                )
            )
        if self._auto_repair_stale_active_enabled():
            tasks.append(
                asyncio.create_task(
                    self._auto_repair_stale_active_loop(),
                    name="auto_repair_stale_active_loop",
                )
            )
        if self._raw_ws_events_prune_enabled():
            tasks.append(
                asyncio.create_task(
                    self._raw_ws_events_prune_loop(),
                    name="raw_ws_events_prune_loop",
                )
            )
        if self._sqlite_wal_checkpoint_enabled():
            tasks.append(
                asyncio.create_task(
                    self._sqlite_wal_checkpoint_loop(),
                    name="sqlite_wal_checkpoint_loop",
                )
            )
        return tasks

    async def _discover_and_update(self, trigger_orderbook_sync: bool) -> None:
        start = time.monotonic()
        try:
            discovered = await self.gamma.fetch_active_markets()
            self._record_api_call(success=True)
        except Exception:
            self._record_api_call(success=False)
            raise

        if self.settings.tracked_markets_limit > 0:
            discovered = discovered[: self.settings.tracked_markets_limit]

        now = utc_now()
        (
            selected_market,
            selected_mode,
            tracking_states,
            market_phases,
            skipped_counts,
            selection_reasons,
        ) = select_primary_market(discovered, now=now)

        ws_tracked = [
            market
            for market in discovered
            if market_phases.get(market.market_id) in {"active", "future"}
            and validate_strict_btc_5m_market(market).is_valid
        ]

        tracked = (
            [selected_market]
            if selected_market is not None and selected_mode == "selected_active"
            else []
        )
        tracked_market_ids = {market.market_id for market in tracked}
        selected_active_market_id = (
            selected_market.market_id
            if selected_market is not None and selected_mode == "selected_active"
            else None
        )

        for market in discovered:
            if market.platform_status is None and market.status is not None:
                market.platform_status = market.status
            market.status = market.platform_status
            market.tracking_state = tracking_states.get(market.market_id, "discovered")
            market.phase = market_phases.get(market.market_id, "active")
            market.market_phase = market.phase
            market.run_id = self.run_id
            market.recorder_version = RECORDER_VERSION
            market.schema_version = SCHEMA_VERSION

        async with self._state_lock:
            before_tokens = set(self.state.all_token_ids())
            lifecycle_events = self.state.upsert_markets(tracked)
            (
                pruned_events,
                removed_markets,
                removed_tokens,
                status_updates,
            ) = self.state.prune_inactive_markets(
                active_market_ids=tracked_market_ids,
                now=now,
            )
            lifecycle_events.extend(pruned_events)

            previous_active_market_id = self._active_primary_market_id
            if selected_active_market_id != previous_active_market_id:
                self._active_primary_market_id = selected_active_market_id
                for market_id in list(self._backfill_cursor.keys()):
                    if market_id not in tracked_market_ids:
                        self._backfill_cursor.pop(market_id, None)
                if selected_active_market_id is not None:
                    self.state.reset_market_runtime(selected_active_market_id)
                    runtime = self.state.markets.get(selected_active_market_id)
                    if runtime is not None and runtime.metadata.start_time is not None:
                        self._backfill_cursor[selected_active_market_id] = (
                            TradeBackfillCursor(last_timestamp=runtime.metadata.start_time)
                        )
                    if not self.db.has_market_event(
                        selected_active_market_id,
                        "market_opened",
                        run_id=self.run_id,
                    ):
                        lifecycle_events.append(
                            MarketEventRecord(
                                timestamp=now,
                                market_id=selected_active_market_id,
                                event_type="market_opened",
                                details=json.dumps(
                                    {
                                        "reason": "selected_active_transition",
                                        "selection_mode": selected_mode,
                                        "previous_active_market_id": previous_active_market_id,
                                        "previous_phase": None,
                                        "new_phase": "active",
                                    },
                                    separators=(",", ":"),
                                ),
                                run_id=self.run_id,
                                recorder_version=RECORDER_VERSION,
                                schema_version=SCHEMA_VERSION,
                            )
                        )
            after_tokens = set(self.state.all_token_ids())

        state_new_tokens = sorted(after_tokens - before_tokens)
        ws_new_tokens: list[str] = []
        ws_unsubscribe_tokens: list[str] = []
        desired_ws_tokens: list[str] = []
        (
            ws_new_tokens,
            ws_unsubscribe_tokens,
            desired_ws_tokens,
        ) = self._update_desired_ws_assets(
            ws_tracked,
            now=now,
            discovered_markets=discovered,
            market_phases=market_phases,
        )
        stats = getattr(self.gamma, "last_discovery_stats", {}) or {}
        self._last_discovery_candidates_seen = int(stats.get("total_markets_from_api", 0) or 0)
        self._last_discovery_candidates_matched = int(
            stats.get("selected_btc_5m_markets", 0) or 0
        )
        self._last_discovery_strict_5m_candidates = self._last_discovery_candidates_matched
        self._last_discovery_broad_btc_candidates = int(
            stats.get("broad_btc_candidates", 0) or 0
        )
        self._last_discovery_fallback_used = int(stats.get("fallback_used", 0) or 0)

        phase_refresh_changes = 0
        stale_primary_demotions = 0
        stale_changes = 0
        market_events_inserted = 0
        market_events_duplicates = 0
        with_events = lifecycle_events
        try:
            db_changes = self.db.upsert_markets(discovered, run_id=self.run_id)
            phase_refresh_changes = self.db.refresh_market_phases(now=now, run_id=self.run_id)
            stale_primary_demotions = self.db.demote_stale_primary_markets(
                current_primary_market_ids=sorted(tracked_market_ids),
                run_id=self.run_id,
            )
            if status_updates:
                self.db.update_market_statuses(status_updates, run_id=self.run_id)
            if tracking_states:
                self.db.update_market_tracking_states(
                    sorted(tracking_states.items()),
                    run_id=self.run_id,
                )
            stale_changes = self.db.mark_non_target_open_markets_inactive(
                active_target_market_ids=sorted(tracked_market_ids),
                run_id=self.run_id,
            )
            market_events_inserted, market_events_duplicates = (
                self.db.insert_market_events(with_events)
            )
        except Exception:
            self.log.exception("market_discovery_db_write_error")
            db_changes = 0

        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._last_api_latency_ms = elapsed_ms
        self._last_successful_market_fetches = self._last_discovery_candidates_matched
        self._last_failed_markets = 0

        self.log.info(
            "market_discovery_cycle",
            extra={
                "discovery_candidates_seen": self._last_discovery_candidates_seen,
                "discovery_candidates_matched": self._last_discovery_candidates_matched,
                "discovery_strict_5m_candidates": self._last_discovery_strict_5m_candidates,
                "discovery_broad_btc_candidates": self._last_discovery_broad_btc_candidates,
                "discovery_fallback_used": self._last_discovery_fallback_used,
                "tracked_markets": len(tracked_market_ids),
                "new_tokens": len(ws_new_tokens),
                "removed_markets": len(removed_markets),
                "removed_tokens": len(ws_unsubscribe_tokens),
                "desired_ws_assets": len(desired_ws_tokens),
                "ws_tracked_markets": len(ws_tracked),
                "db_changes": db_changes,
                "stale_open_rows_marked_inactive": stale_changes,
                "phase_refresh_changes": phase_refresh_changes,
                "stale_primary_demotions": stale_primary_demotions,
                "market_events_inserted": market_events_inserted,
                "market_events_duplicates_skipped": market_events_duplicates,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        self.log.info(
            "primary_market_selection",
            extra={
                "candidate_count": len(discovered),
                "selected_market_id": selected_market.market_id if selected_market else None,
                "selected_question": selected_market.question if selected_market else None,
                "selected_start_time": to_iso(selected_market.start_time)
                if selected_market
                else None,
                "selected_close_time": to_iso(selected_market.close_time)
                if selected_market
                else None,
                "selected_mode": selected_mode,
                "selected_market_phase": market_phases.get(selected_market.market_id)
                if selected_market
                else None,
                "skipped_reasons": dict(skipped_counts),
                "market_selection_reasons": selection_reasons,
                "rejected_markets": [
                    {
                        "market_id": market.market_id,
                        "phase": market_phases.get(market.market_id),
                        "tracking_state": tracking_states.get(market.market_id),
                        "selection_reason": selection_reasons.get(market.market_id),
                        "strict_validation_passed": market.strict_validation_passed,
                        "strict_rejection_reason": market.strict_rejection_reason,
                    }
                    for market in discovered
                    if tracking_states.get(market.market_id) == "inactive"
                ],
            },
        )

        if ws_new_tokens:
            self.log.info(
                "subscribing_new_tokens",
                extra={
                    "new_token_count": len(ws_new_tokens),
                    "token_sample": ws_new_tokens[:10],
                },
            )
            try:
                await self.ws.subscribe_tokens(ws_new_tokens)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception(
                    "ws_subscribe_tokens_failed",
                    extra={"token_count": len(ws_new_tokens), "token_sample": ws_new_tokens[:10]},
                )
        if ws_unsubscribe_tokens:
            self.log.info(
                "unsubscribing_stale_tokens",
                extra={
                    "stale_token_count": len(ws_unsubscribe_tokens),
                    "token_sample": ws_unsubscribe_tokens[:10],
                },
            )
            try:
                await self.ws.unsubscribe_tokens(ws_unsubscribe_tokens)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception(
                    "ws_unsubscribe_tokens_failed",
                    extra={
                        "token_count": len(ws_unsubscribe_tokens),
                        "token_sample": ws_unsubscribe_tokens[:10],
                    },
                )
        if trigger_orderbook_sync and state_new_tokens:
            await self._sync_orderbooks_for_tokens(state_new_tokens, reason="rediscovery")

    async def _discovery_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.settings.discovery_interval_sec,
                )
                continue
            except asyncio.TimeoutError:
                pass
            try:
                await self._discover_and_update(trigger_orderbook_sync=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception("discovery_loop_error")

    async def _snapshot_loop(self) -> None:
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            timestamp = datetime.now(timezone.utc)
            async with self._state_lock:
                (
                    snapshots,
                    orderbook_levels,
                    features,
                    trades,
                    events,
                    skipped_outside_window,
                    skipped_missing_orderbook,
                    snapshot_trade_warnings,
                    invalid_active_past_close,
                ) = self.state.build_cycle_records(
                    timestamp=timestamp,
                    levels_to_store=self.settings.orderbook_levels_to_store,
                    feature_orderbook_depth=self.settings.feature_orderbook_depth,
                    jump_threshold=self.settings.jump_threshold,
                    expected_snapshot_interval_sec=self.settings.snapshot_interval_sec,
                    gap_threshold_multiplier=self.settings.snapshot_gap_threshold_multiplier,
                    trade_snapshot_freshness_sec=self.settings.trade_snapshot_freshness_sec,
                )
                skipped_no_state_change = list(self.state.last_cycle_snapshot_no_change)
                snapshot_generation_failed = list(
                    self.state.last_cycle_snapshot_generation_failed
                )
                tracked_markets = self.state.market_count()
                tracked_market_ids = list(self.state.markets.keys())
                subscribed_asset_ids = self._get_token_ids_for_ws()
                should_rotate = any(
                    runtime.metadata.close_time is not None
                    and timestamp >= runtime.metadata.close_time
                    for runtime in self.state.markets.values()
                )

            if skipped_missing_orderbook:
                self.log.debug(
                    "snapshot_skipped_missing_orderbook",
                    extra={
                        "count": len(skipped_missing_orderbook),
                        "market_sample": skipped_missing_orderbook[:10],
                    },
                )
            if skipped_no_state_change:
                self.log.debug(
                    "snapshot_skipped_no_state_change",
                    extra={
                        "count": len(skipped_no_state_change),
                        "market_sample": skipped_no_state_change[:10],
                    },
                )
            if snapshot_generation_failed:
                self.log.warning(
                    "snapshot_generation_failed_summary",
                    extra={
                        "count": len(snapshot_generation_failed),
                        "market_sample": snapshot_generation_failed[:10],
                    },
                )
            if skipped_outside_window:
                self.log.debug(
                    "snapshot_skipped_outside_active_window",
                    extra={
                        "count": len(skipped_outside_window),
                        "market_sample": skipped_outside_window[:10],
                    },
                )
            if snapshot_trade_warnings:
                reasons = Counter(row.get("reason", "unknown") for row in snapshot_trade_warnings)
                self.log.warning(
                    "snapshot_trade_state_invalid_summary",
                    extra={
                        "reasons": dict(reasons),
                        "sample": snapshot_trade_warnings[:10],
                    },
                )
            if invalid_active_past_close:
                self.log.warning(
                    "market_marked_active_past_close",
                    extra={"market_ids": invalid_active_past_close[:20]},
                )

            rows_inserted = 0
            duplicate_rows_skipped = 0
            db_write_start = time.monotonic()
            try:
                self.db.refresh_market_phases(now=timestamp, run_id=self.run_id)
            except Exception:
                self.log.exception("refresh_market_phases_error")
            try:
                for inserted, skipped in (
                    self.db.insert_market_snapshots(snapshots),
                    self.db.insert_order_book_levels(orderbook_levels),
                    self.db.insert_features(features),
                    self.db.insert_trades(trades),
                    self.db.insert_market_events(events),
                ):
                    rows_inserted += inserted
                    duplicate_rows_skipped += skipped
            except SnapshotInsertShapeError as exc:
                self.log.exception(
                    "snapshot_db_write_error",
                    extra={
                        "expected_column_count": exc.expected_column_count,
                        "actual_row_length": exc.actual_row_length,
                        "insert_columns": exc.columns,
                        "problem_row_index": exc.row_index,
                        "problem_row_preview": exc.row_preview,
                    },
                )
            except Exception:
                self.log.exception("snapshot_db_write_error")
            db_write_ms = (time.monotonic() - db_write_start) * 1000.0

            api_call_count, api_success_count, api_failure_count = self._drain_api_counters()
            trade_summary = self._last_trade_backfill_summary or {}
            health_fields = self._recorder_health_fields(
                subscribed_asset_ids=subscribed_asset_ids,
            )
            metrics = RecorderMetricRecord(
                timestamp=timestamp,
                markets_polled=len(snapshots),
                successful_market_fetches=self._last_successful_market_fetches,
                failed_markets=self._last_failed_markets,
                api_latency_ms=self._last_api_latency_ms,
                db_write_time_ms=db_write_ms,
                cycle_duration_ms=(time.monotonic() - cycle_start) * 1000.0,
                rows_inserted=rows_inserted,
                duplicate_rows_skipped=duplicate_rows_skipped,
                active_markets_snapshot_attempted=tracked_markets,
                snapshots_written=len(snapshots),
                snapshot_markets_skipped=(
                    len(skipped_outside_window)
                    + len(skipped_missing_orderbook)
                    + len(skipped_no_state_change)
                    + len(snapshot_generation_failed)
                ),
                discovery_candidates_seen=self._last_discovery_candidates_seen,
                discovery_candidates_matched=self._last_discovery_candidates_matched,
                discovery_strict_5m_candidates=self._last_discovery_strict_5m_candidates,
                discovery_broad_btc_candidates=self._last_discovery_broad_btc_candidates,
                discovery_fallback_used=self._last_discovery_fallback_used,
                tracked_markets_count=tracked_markets,
                api_call_count=api_call_count,
                api_success_count=api_success_count,
                api_failure_count=api_failure_count,
                fetched_trades=int(trade_summary.get("fetched_trades", 0) or 0),
                accepted_trades=int(trade_summary.get("accepted_trades", 0) or 0),
                rejected_before_start=int(trade_summary.get("rejected_before_start", 0) or 0),
                rejected_after_close=int(trade_summary.get("rejected_after_close", 0) or 0),
                skipped_already_seen=int(trade_summary.get("skipped_already_seen", 0) or 0),
                last_accepted_trade_ts=trade_summary.get("last_accepted_trade_ts"),
                ws_reconnect_count=health_fields["ws_reconnect_count"],
                raw_ws_events_seen=health_fields["raw_ws_events_seen"],
                raw_ws_events_written=health_fields["raw_ws_events_written"],
                malformed_ws_events=health_fields["malformed_ws_events"],
                raw_ws_write_failures=health_fields["raw_ws_write_failures"],
                last_ws_event_age_sec=health_fields["last_ws_event_age_sec"],
                subscribed_asset_count=health_fields["subscribed_asset_count"],
                subscribed_asset_ids_json=health_fields["subscribed_asset_ids_json"],
                run_id=self.run_id,
                recorder_version=RECORDER_VERSION,
                schema_version=SCHEMA_VERSION,
            )
            try:
                self.db.insert_recorder_metric(metrics)
            except Exception:
                self.log.exception("metrics_db_write_error")

            self._maybe_log_heartbeat(
                metrics=metrics,
                tracked_markets=tracked_markets,
                snapshots=snapshots,
                skipped_outside_window=skipped_outside_window,
                skipped_missing_orderbook=skipped_missing_orderbook,
                skipped_no_state_change=skipped_no_state_change,
                snapshot_generation_failed=snapshot_generation_failed,
                features=features,
            )

            if should_rotate:
                self.log.info(
                    "primary_market_rotation_triggered",
                    extra={
                        "reason": "tracked_market_close_time_reached",
                        "tracked_market_ids": tracked_market_ids,
                    },
                )
                try:
                    await self._discover_and_update(trigger_orderbook_sync=True)
                except Exception:
                    self.log.exception("rotation_discovery_error")

            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=self.settings.snapshot_interval_sec,
            )

    async def _ws_loop(self) -> None:
        await self.ws.run_forever(
            get_token_ids=self._get_token_ids_for_ws,
            on_message=self._handle_ws_message,
            stop_event=self.stop_event,
            on_reconnect=self._recover_from_ws_disconnect,
        )

    async def _trade_backfill_loop(self) -> None:
        interval_sec = min(
            float(self.settings.trade_backfill_interval_sec),
            max(1.0, float(self.settings.trade_snapshot_freshness_sec)),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                if self.settings.enable_trade_backfill:
                    await self._run_trade_backfill_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception("trade_backfill_loop_error")
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )

    def _auto_normalize_resolutions_enabled(self) -> bool:
        return bool(
            getattr(self.settings, "auto_normalize_resolutions_enabled", False)
        )

    def _auto_repair_stale_active_enabled(self) -> bool:
        return bool(
            getattr(self.settings, "auto_repair_stale_active_enabled", False)
        )

    def _raw_ws_events_persistence_enabled(self) -> bool:
        return bool(getattr(self.settings, "raw_ws_events_enabled", False))

    def _raw_ws_events_prune_enabled(self) -> bool:
        return (
            float(getattr(self.settings, "raw_ws_events_retention_sec", 0.0) or 0.0)
            > 0
            or int(getattr(self.settings, "raw_ws_events_max_rows", 0) or 0) > 0
        )

    def _sqlite_wal_checkpoint_enabled(self) -> bool:
        if not hasattr(self.settings, "sqlite_wal_checkpoint_interval_sec"):
            return False
        mode = str(
            getattr(self.settings, "sqlite_wal_checkpoint_mode", "PASSIVE")
            or "PASSIVE"
        ).strip().upper()
        interval_sec = float(
            getattr(self.settings, "sqlite_wal_checkpoint_interval_sec", 300.0)
            or 0.0
        )
        return mode != "OFF" and interval_sec > 0

    async def _auto_normalize_resolutions_loop(self) -> None:
        interval_sec = max(
            0.1,
            float(
                getattr(
                    self.settings,
                    "auto_normalize_resolutions_interval_sec",
                    60.0,
                )
                or 60.0
            ),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            await self._run_auto_normalize_resolutions_cycle()
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )

    async def _raw_ws_events_prune_loop(self) -> None:
        interval_sec = max(
            1.0,
            float(
                getattr(
                    self.settings,
                    "raw_ws_events_prune_interval_sec",
                    300.0,
                )
                or 300.0
            ),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )
            if self.stop_event.is_set():
                break
            await self._run_raw_ws_events_prune_cycle()

    async def _sqlite_wal_checkpoint_loop(self) -> None:
        interval_sec = max(
            1.0,
            float(
                getattr(
                    self.settings,
                    "sqlite_wal_checkpoint_interval_sec",
                    300.0,
                )
                or 300.0
            ),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )
            if self.stop_event.is_set():
                break
            await self._run_sqlite_wal_checkpoint_cycle()

    async def _auto_repair_stale_active_loop(self) -> None:
        interval_sec = max(
            0.1,
            float(
                getattr(
                    self.settings,
                    "auto_repair_stale_active_interval_sec",
                    300.0,
                )
                or 300.0
            ),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            await self._run_auto_repair_stale_active_cycle()
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )

    async def _run_auto_normalize_resolutions_cycle(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            report = self.db.normalize_market_resolutions(
                dry_run=False,
                run_id=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception("auto_normalize_resolutions_error")
            return {}

        self.log.info(
            "auto_normalize_resolutions_completed",
            extra={
                "duration_ms": round((time.monotonic() - start) * 1000.0, 2),
                "run_id_scope": report.get("run_id_scope"),
                "resolution_events_seen": report.get("resolution_events_seen"),
                "mapped_resolution_events": report.get("mapped_resolution_events"),
                "unmapped_resolution_events": report.get("unmapped_resolution_events"),
                "ambiguous_resolution_events": report.get("ambiguous_resolution_events"),
                "markets_updated": report.get("markets_updated"),
            },
        )
        return report

    async def _run_auto_repair_stale_active_cycle(self) -> dict[str, Any]:
        start = time.monotonic()
        grace_sec = max(
            0.0,
            float(
                getattr(
                    self.settings,
                    "auto_repair_stale_active_grace_sec",
                    60.0,
                )
                or 0.0
            ),
        )
        try:
            report = self.db.repair_stale_selected_active_markets(
                dry_run=False,
                grace_sec=grace_sec,
                run_id=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception("auto_repair_stale_active_error")
            return {}

        self.log.info(
            "auto_repair_stale_active_completed",
            extra={
                "duration_ms": round((time.monotonic() - start) * 1000.0, 2),
                "run_id_scope": report.get("run_id_scope"),
                "grace_sec": report.get("grace_sec"),
                "stale_selected_active_count": report.get(
                    "stale_selected_active_count"
                ),
                "demoted_count": report.get("demoted_count"),
                "market_events_inserted": report.get("market_events_inserted"),
            },
        )
        return report

    async def _run_raw_ws_events_prune_cycle(self) -> dict[str, Any]:
        try:
            report = self.db.prune_raw_polymarket_events(
                retention_sec=float(
                    getattr(self.settings, "raw_ws_events_retention_sec", 0.0)
                    or 0.0
                ),
                max_rows=int(getattr(self.settings, "raw_ws_events_max_rows", 0) or 0),
                batch_size=int(
                    getattr(
                        self.settings,
                        "raw_ws_events_prune_batch_size",
                        50000,
                    )
                    or 50000
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception("raw_ws_events_prune_error")
            return {}
        self.log.info("raw_ws_events_pruned", extra=report)
        await asyncio.sleep(0)
        return report

    async def _run_sqlite_wal_checkpoint_cycle(self) -> dict[str, Any]:
        mode = str(
            getattr(self.settings, "sqlite_wal_checkpoint_mode", "PASSIVE")
            or "PASSIVE"
        ).strip().upper()
        try:
            report = self.db.wal_checkpoint(mode=mode)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception(
                "sqlite_wal_checkpoint_error",
                extra={"sqlite_wal_checkpoint_mode": mode},
            )
            return {}
        self.log.info("sqlite_wal_checkpoint_completed", extra=report)
        await asyncio.sleep(0)
        return report

    @staticmethod
    def _btc_price_feed_enabled_for_settings(settings: object) -> bool:
        return bool(getattr(settings, "btc_price_feed_enabled", False))

    def _btc_price_feed_enabled(self) -> bool:
        if not self._btc_price_feed_enabled_for_settings(self.settings):
            return False
        feeds = getattr(self, "btc_price_feeds", None)
        if isinstance(feeds, dict) and feeds:
            return True
        return getattr(self, "btc_price_feed", None) is not None

    def _log_btc_price_feed_startup_config(self) -> None:
        config = describe_btc_price_feed_config(self.settings)
        self.log.info(
            "btc_price_feed_config_resolved",
            extra={
                "btc_price_feed_enabled": config["BTC_PRICE_FEED_ENABLED"],
                "btc_price_feed_source": config["BTC_PRICE_FEED_SOURCE"],
                "btc_price_feed_sources_raw": config["BTC_PRICE_FEED_SOURCES"],
                "btc_price_feed_parsed_sources": config["parsed_sources"],
                "btc_price_feed_source_symbol_map": config["source_symbol_map"],
                "btc_price_feed_source_ws_url_map": config["source_ws_url_map"],
                "btc_price_feed_stale_reconnect_sec": config[
                    "BTC_PRICE_FEED_STALE_RECONNECT_SEC"
                ],
                "btc_price_max_exchange_age_sec": config[
                    "BTC_PRICE_MAX_EXCHANGE_AGE_SEC"
                ],
                "btc_price_drop_stale": config["BTC_PRICE_DROP_STALE"],
                "btc_price_stale_reconnect_sec": config[
                    "BTC_PRICE_STALE_RECONNECT_SEC"
                ],
                "btc_price_feed_startup_timeout_sec": config[
                    "BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC"
                ],
                "btc_price_feed_sample_timeout_sec": config[
                    "BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC"
                ],
                "btc_price_binance_rest_url": config["BTC_PRICE_BINANCE_REST_URL"],
            },
        )
        for feed in config["planned_feeds"]:
            if not bool(feed.get("supported")):
                self.log.warning(
                    "btc_price_feed_source_unsupported",
                    extra={
                        "btc_feed_source": feed.get("source"),
                        "symbol": feed.get("symbol"),
                        "ws_url": feed.get("ws_url"),
                    },
                )

    def _btc_price_feed_task_extra(
        self,
        source: str,
        feed: BTCPriceFeed | None,
    ) -> dict[str, Any]:
        config = describe_btc_price_feed_config(self.settings)
        configured = {
            str(item["source"]): item
            for item in config.get("planned_feeds", [])
            if isinstance(item, dict)
        }
        planned = configured.get(str(source), {})
        return {
            "btc_feed_source": str(source),
            "symbol": getattr(feed, "symbol", planned.get("symbol", None)),
            "ws_url": getattr(feed, "ws_url", planned.get("ws_url", None)),
            "stale_reconnect_sec": getattr(
                feed,
                "stale_reconnect_sec",
                getattr(self.settings, "btc_price_feed_stale_reconnect_sec", None),
            ),
            "max_exchange_age_sec": getattr(
                feed,
                "max_exchange_age_sec",
                getattr(self.settings, "btc_price_max_exchange_age_sec", None),
            ),
            "drop_stale": getattr(
                feed,
                "drop_stale",
                getattr(self.settings, "btc_price_drop_stale", None),
            ),
            "stale_message_reconnect_sec": getattr(
                feed,
                "stale_message_reconnect_sec",
                getattr(self.settings, "btc_price_stale_reconnect_sec", None),
            ),
            "sample_timeout_sec": getattr(
                self.settings,
                "btc_price_feed_sample_timeout_sec",
                None,
            ),
            "startup_timeout_sec": getattr(
                self.settings,
                "btc_price_feed_startup_timeout_sec",
                None,
            ),
            "supported": is_btc_price_feed_source_supported(str(source)),
        }

    async def _btc_price_feed_loop(self, source: str | None = None) -> None:
        source_name, feed = self._resolve_btc_price_feed(source)
        state = self._btc_feed_health_state(source_name)
        state["task_started_monotonic"] = time.monotonic()
        self.log.info(
            "btc_price_feed_task_started",
            extra=self._btc_price_feed_task_extra(source_name, feed),
        )
        interval_sec = max(
            0.1,
            float(getattr(self.settings, "btc_price_feed_interval_sec", 1.0) or 1.0),
        )
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            await self._run_btc_price_feed_cycle(source=source)
            self._maybe_log_btc_price_feed_startup_timeout(source_name)
            await self._sleep_until_next_cycle(
                cycle_start=cycle_start,
                interval_sec=interval_sec,
            )

    async def _run_btc_price_feed_cycle(self, source: str | None = None) -> int:
        source_name, _feed = self._resolve_btc_price_feed(source)
        try:
            inserted = await self._sample_btc_price_once(source=source)
            self._log_btc_price_feed_health(source_name)
            return inserted
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_btc_price_feed_error(source_name, exc)
            self.log.exception(
                "btc_price_feed_sample_error",
                extra={
                    "source": source_name,
                    "symbol": getattr(self.settings, "btc_price_feed_symbol", None),
                    **self._btc_price_feed_health_extra(source_name),
                },
            )
            return 0

    def _resolve_btc_price_feed(
        self,
        source: str | None = None,
    ) -> tuple[str, BTCPriceFeed | None]:
        feeds = getattr(self, "btc_price_feeds", None)
        if isinstance(feeds, dict) and feeds:
            if source is not None:
                return str(source), feeds.get(str(source))
            configured_source = str(
                getattr(self.settings, "btc_price_feed_source", "") or ""
            ).strip().lower()
            if configured_source and configured_source in feeds:
                return configured_source, feeds[configured_source]
            first_source = next(iter(feeds))
            return str(first_source), feeds[first_source]

        feed = getattr(self, "btc_price_feed", None)
        source_name = str(
            source
            or getattr(feed, "source", None)
            or getattr(self.settings, "btc_price_feed_source", "unknown")
            or "unknown"
        )
        return source_name, feed

    async def _sample_btc_price_once(self, source: str | None = None) -> int:
        source_name, feed = self._resolve_btc_price_feed(source)
        if feed is None:
            return 0
        local_arrival_ns = time.time_ns()
        local_arrival_iso = local_arrival_iso_from_ns(local_arrival_ns)
        sample_timeout_sec = max(
            0.01,
            float(
                getattr(self.settings, "btc_price_feed_sample_timeout_sec", 2.0)
                or 2.0
            ),
        )
        try:
            sample = await asyncio.wait_for(
                feed.sample(
                    local_arrival_ns=local_arrival_ns,
                    local_arrival_iso=local_arrival_iso,
                ),
                timeout=sample_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise BTCPriceFeedError(
                f"BTC price feed sample timed out after {sample_timeout_sec:.3f}s"
            ) from exc
        if sample is None:
            return 0
        if not isinstance(sample, BTCPriceSampleRecord):
            raise TypeError(f"unexpected BTC price sample type: {type(sample)!r}")
        if not sample.run_id:
            sample.run_id = self.run_id
        stale = self._btc_sample_stale_result(sample)
        if stale is not None and bool(getattr(self.settings, "btc_price_drop_stale", True)):
            self._record_btc_price_feed_stale_drop(source_name, sample, stale)
            self.log.warning(
                "btc_price_sample_stale_dropped",
                extra={
                    "source": sample.source,
                    "price": sample.price,
                    "exchange_timestamp": to_iso(sample.exchange_timestamp),
                    "local_arrival_iso": sample.local_arrival_iso,
                    "age_sec": round(float(stale["age_sec"]), 3),
                    "timestamp_source": stale.get("timestamp_source"),
                    "max_exchange_age_sec": stale.get("max_exchange_age_sec"),
                    **self._btc_price_feed_health_extra(source_name),
                },
            )
            return 0
        inserted, _skipped = self.db.insert_btc_price_samples([sample])
        if inserted:
            self._record_btc_price_feed_sample(source_name, sample)
            self.log.debug(
                "btc_price_sample_written",
                extra={
                    "source": sample.source,
                    "price": sample.price,
                    "local_arrival_ns": sample.local_arrival_ns,
                    **self._btc_price_feed_health_extra(source_name),
                },
            )
        return inserted

    def _btc_sample_stale_result(
        self,
        sample: BTCPriceSampleRecord,
    ) -> dict[str, Any] | None:
        max_age = float(
            getattr(self.settings, "btc_price_max_exchange_age_sec", 15.0) or 0.0
        )
        if max_age <= 0:
            return None
        observed_at = sample.exchange_timestamp or parse_timestamp(sample.local_arrival_iso)
        timestamp_source = (
            "exchange_timestamp" if sample.exchange_timestamp else "local_arrival_iso"
        )
        if observed_at is None:
            return None
        age_sec = (
            datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)
        ).total_seconds()
        if age_sec <= max_age:
            return None
        return {
            "age_sec": age_sec,
            "timestamp_source": timestamp_source,
            "max_exchange_age_sec": max_age,
        }

    def _record_btc_price_feed_sample(
        self,
        source: str,
        sample: BTCPriceSampleRecord,
    ) -> None:
        state = self._btc_feed_health_state(source)
        state["sample_count"] = int(state.get("sample_count") or 0) + 1
        state["rows_inserted"] = int(state.get("rows_inserted") or 0) + 1
        state["latest_sample_local_arrival_iso"] = sample.local_arrival_iso
        state["latest_sample_monotonic"] = time.monotonic()
        state["latest_price"] = sample.price
        state["latest_exchange_timestamp"] = to_iso(sample.exchange_timestamp)
        state["startup_warning_logged"] = False

    def _record_btc_price_feed_error(self, source: str, exc: BaseException) -> None:
        state = self._btc_feed_health_state(source)
        state["error_count"] = int(state.get("error_count") or 0) + 1
        state["last_error"] = str(exc)

    def _record_btc_price_feed_stale_drop(
        self,
        source: str,
        sample: BTCPriceSampleRecord,
        stale: dict[str, Any],
    ) -> None:
        state = self._btc_feed_health_state(source)
        state["stale_messages_dropped"] = int(state.get("stale_messages_dropped") or 0) + 1
        state["stale_rows_dropped"] = int(state.get("stale_rows_dropped") or 0) + 1
        state["latest_stale_price"] = sample.price
        state["latest_stale_exchange_timestamp"] = to_iso(sample.exchange_timestamp)
        state["latest_stale_local_arrival_iso"] = sample.local_arrival_iso
        state["latest_stale_age_sec"] = float(stale.get("age_sec") or 0.0)

    def _btc_feed_health_state(self, source: str) -> dict[str, Any]:
        return self._btc_feed_health.setdefault(
            str(source),
            {
                "sample_count": 0,
                "error_count": 0,
                "last_error": None,
                "latest_sample_local_arrival_iso": None,
                "latest_sample_monotonic": None,
                "latest_price": None,
                "latest_exchange_timestamp": None,
                "rows_inserted": 0,
                "stale_messages_dropped": 0,
                "stale_rows_dropped": 0,
                "task_started_monotonic": None,
                "startup_warning_logged": False,
            },
        )

    def _btc_price_feed_health_extra(self, source: str) -> dict[str, Any]:
        state = self._btc_feed_health.get(str(source), {})
        latest_monotonic = state.get("latest_sample_monotonic")
        sample_age_sec = (
            round(time.monotonic() - float(latest_monotonic), 3)
            if latest_monotonic is not None
            else None
        )
        return {
            "btc_feed_source": str(source),
            "btc_feed_latest_sample_time": state.get(
                "latest_sample_local_arrival_iso"
            ),
            "btc_feed_sample_age_sec": sample_age_sec,
            "btc_feed_sample_count": int(state.get("sample_count") or 0),
            "btc_feed_error_count": int(state.get("error_count") or 0),
            "btc_feed_last_error": state.get("last_error"),
            "btc_feed_latest_price": state.get("latest_price"),
            "btc_feed_latest_exchange_timestamp": state.get("latest_exchange_timestamp"),
            "btc_feed_rows_inserted": int(state.get("rows_inserted") or 0),
            "btc_feed_stale_messages_dropped": int(
                state.get("stale_messages_dropped") or 0
            ),
            "btc_feed_stale_rows_dropped": int(state.get("stale_rows_dropped") or 0),
            "btc_feed_latest_stale_age_sec": state.get("latest_stale_age_sec"),
        }

    def _log_btc_price_feed_health(self, source: str) -> None:
        self.log.debug(
            "btc_price_feed_health",
            extra=self._btc_price_feed_health_extra(source),
        )

    def _btc_source_is_healthy(self, source: str) -> bool:
        state = self._btc_feed_health_state(source)
        max_age = float(
            getattr(self.settings, "btc_price_max_exchange_age_sec", 15.0) or 15.0
        )
        exchange_ts = parse_timestamp(state.get("latest_exchange_timestamp"))
        if exchange_ts is not None:
            age_sec = (datetime.now(timezone.utc) - exchange_ts).total_seconds()
            return age_sec <= max_age
        sample_mono = state.get("latest_sample_monotonic")
        if sample_mono is not None:
            age_sec = time.monotonic() - float(sample_mono)
            return age_sec <= max_age
        return False

    def _maybe_log_btc_price_feed_startup_timeout(self, source: str) -> None:
        state = self._btc_feed_health_state(source)
        if self._btc_source_is_healthy(source):
            return
        if bool(state.get("startup_warning_logged")):
            return
        started = state.get("task_started_monotonic")
        if started is None:
            return
        timeout_sec = max(
            0.0,
            float(
                getattr(self.settings, "btc_price_feed_startup_timeout_sec", 5.0)
                or 5.0
            ),
        )
        if time.monotonic() - float(started) < timeout_sec:
            return
        state["startup_warning_logged"] = True
        self.log.warning(
            "btc_price_feed_startup_unhealthy_timeout",
            extra={
                "source": source,
                "startup_timeout_sec": timeout_sec,
                **self._btc_price_feed_health_extra(source),
            },
        )

    async def _sync_all_known_orderbooks(self, reason: str) -> None:
        async with self._state_lock:
            token_ids = self.state.all_token_ids()
        await self._sync_orderbooks_for_tokens(token_ids, reason=reason)

    async def _sync_orderbooks_for_tokens(
        self,
        token_ids: Sequence[str],
        reason: str,
    ) -> None:
        if not token_ids:
            return

        start = time.monotonic()
        semaphore = asyncio.Semaphore(self.settings.startup_orderbook_concurrency)
        successful = 0
        failed = 0
        missing = 0

        async def _fetch_one(token_id: str) -> None:
            nonlocal successful, failed, missing
            async with semaphore:
                try:
                    snapshot = await self.clob.fetch_orderbook_snapshot(token_id)
                    self._record_api_call(success=True)
                    if snapshot is None:
                        missing += 1
                        return
                    async with self._state_lock:
                        updated = self.state.apply_orderbook_update(
                            token_id=token_id,
                            bids=snapshot.bids,
                            asks=snapshot.asks,
                        )
                    if updated:
                        successful += 1
                    else:
                        failed += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._record_api_call(success=False)
                    failed += 1
                    self.log.warning(
                        "orderbook_sync_failed",
                        extra={"token_id": token_id, "reason": reason},
                        exc_info=True,
                    )

        await asyncio.gather(*[_fetch_one(token_id) for token_id in token_ids])
        elapsed_ms = (time.monotonic() - start) * 1000.0
        self.log.info(
            "orderbook_sync_completed",
            extra={
                "reason": reason,
                "tokens": len(token_ids),
                "successful": successful,
                "missing_orderbook": missing,
                "failed": failed,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

    async def _recover_from_ws_disconnect(self) -> None:
        self._ws_reconnect_count += 1
        reconnect_count = self._ws_reconnect_count
        self.log.info(
            "ws_reconnect_recovery_started",
            extra={"reconnect_count": reconnect_count},
        )
        try:
            await self._sync_all_known_orderbooks(reason="ws_reconnect_recovery")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception(
                "ws_reconnect_recovery_error",
                extra={"reconnect_count": reconnect_count},
            )
            return
        self.log.info(
            "ws_reconnect_recovery_completed",
            extra={"reconnect_count": reconnect_count},
        )

    def _recorder_health_fields(
        self,
        *,
        subscribed_asset_ids: Sequence[str],
        now_mono: Optional[float] = None,
    ) -> dict[str, Any]:
        monotonic_now = time.monotonic() if now_mono is None else now_mono
        last_event_mono = getattr(self, "_last_ws_event_monotonic", None)
        last_event_age_sec = (
            None
            if last_event_mono is None
            else max(0.0, monotonic_now - float(last_event_mono))
        )
        asset_ids = sorted({str(token) for token in subscribed_asset_ids if token})
        ws_client = getattr(self, "ws", None)
        ws_client_reconnects = int(getattr(ws_client, "reconnect_count", 0) or 0)
        recorder_reconnects = int(getattr(self, "_ws_reconnect_count", 0) or 0)
        return {
            "ws_reconnect_count": max(recorder_reconnects, ws_client_reconnects),
            "raw_ws_events_seen": int(getattr(self, "_raw_ws_events_seen", 0) or 0),
            "raw_ws_events_written": int(getattr(self, "_raw_ws_events_written", 0) or 0),
            "malformed_ws_events": int(getattr(self, "_raw_ws_malformed_events", 0) or 0),
            "raw_ws_write_failures": int(getattr(self, "_raw_ws_write_failures", 0) or 0),
            "last_ws_event_age_sec": last_event_age_sec,
            "subscribed_asset_count": len(asset_ids),
            "subscribed_asset_ids_json": json.dumps(asset_ids, separators=(",", ":")),
        }

    def _update_desired_ws_assets(
        self,
        markets: Sequence[MarketMetadata],
        *,
        now: datetime,
        discovered_markets: Sequence[MarketMetadata] = (),
        market_phases: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[str], list[str]]:
        active_tokens: set[str] = set()
        for market in markets:
            active_tokens.update(_market_ws_asset_ids(market))

        grace_sec = max(
            0.0,
            float(getattr(self.settings, "market_unsubscribe_grace_sec", 60) or 0),
        )
        previous_tokens = set(getattr(self, "_desired_ws_asset_ids", set()))
        deadlines = dict(getattr(self, "_stale_ws_asset_deadlines", {}))
        phase_by_market = market_phases or {}
        stale_deadline_by_token: dict[str, datetime] = {}

        for market in discovered_markets:
            phase = phase_by_market.get(market.market_id) or classify_market_phase(market, now)
            if phase in {"active", "future"}:
                continue
            if phase == "resolved":
                deadline_base = now
            else:
                deadline_base = market.close_time or market.end_time or now
            token_deadline = deadline_base + timedelta(seconds=grace_sec)
            for token in _market_ws_asset_ids(market):
                stale_deadline_by_token[token] = token_deadline

        for token in active_tokens:
            deadlines.pop(token, None)

        stale_candidates = previous_tokens - active_tokens
        for token in stale_candidates:
            candidate_deadline = stale_deadline_by_token.get(
                token,
                now + timedelta(seconds=grace_sec),
            )
            existing_deadline = deadlines.get(token)
            if existing_deadline is None or candidate_deadline < existing_deadline:
                deadlines[token] = candidate_deadline

        expired_tokens = sorted(
            token for token, deadline in deadlines.items() if deadline <= now
        )
        for token in expired_tokens:
            deadlines.pop(token, None)

        grace_tokens = {token for token in deadlines if token not in expired_tokens}
        desired_tokens = active_tokens | grace_tokens
        new_tokens = sorted(active_tokens - previous_tokens)

        self._desired_ws_asset_ids = set(desired_tokens)
        self._stale_ws_asset_deadlines = deadlines
        return new_tokens, expired_tokens, sorted(desired_tokens)

    def _get_token_ids_for_ws(self) -> List[str]:
        desired = sorted(getattr(self, "_desired_ws_asset_ids", set()))
        if desired:
            return desired
        return self.state.all_token_ids()

    def _capture_raw_ws_message(
        self,
        raw_message: str | bytes,
        *,
        local_arrival_ns: int,
    ) -> None:
        if not self._raw_ws_events_persistence_enabled():
            self._raw_ws_events_seen += 1
            if not getattr(self, "_raw_ws_events_skipped_logged", False):
                self._raw_ws_events_skipped_logged = True
                self.log.info(
                    "raw_ws_events_skipped_due_to_config",
                    extra={
                        "raw_ws_events_enabled": False,
                        "local_arrival_ns": local_arrival_ns,
                    },
                )
            return
        try:
            raw_records = normalize_raw_polymarket_events(
                raw_message,
                local_arrival_ns=local_arrival_ns,
            )
            self._raw_ws_events_seen += len(raw_records)
            self._raw_ws_malformed_events += sum(
                1 for record in raw_records if record.parse_status != "ok"
            )
            for record in raw_records:
                record.run_id = self.run_id
            inserted, _skipped = self.db.insert_raw_polymarket_events(raw_records)
            self._raw_ws_events_written += inserted
        except Exception:
            self._raw_ws_write_failures += 1
            self.log.exception("raw_ws_event_write_error")

    def _resolved_market_placeholder(self, event: MarketEventRecord) -> MarketMetadata:
        details: dict[str, Any] = {}
        if event.details:
            try:
                decoded = json.loads(event.details)
                if isinstance(decoded, dict):
                    details = decoded
            except json.JSONDecodeError:
                details = {}

        condition_id = (
            details.get("condition_id")
            or details.get("conditionId")
            or details.get("market")
            or event.market_id
        )
        event_id = details.get("event_id") or details.get("eventId") or details.get("id")
        return MarketMetadata(
            market_id=event.market_id,
            event_id=str(event_id) if event_id is not None else None,
            question=None,
            description=None,
            category=None,
            outcomes=[],
            resolution_source=None,
            start_time=None,
            end_time=None,
            close_time=None,
            platform_status="resolved",
            phase="resolved",
            tracking_state="inactive",
            status="resolved",
            market_phase="resolved",
            condition_id=str(condition_id) if condition_id is not None else None,
            created_at=event.timestamp,
            last_updated=event.timestamp,
            run_id=self.run_id,
            recorder_version=RECORDER_VERSION,
            schema_version=SCHEMA_VERSION,
        )

    def _persist_normalized_ws_records(
        self,
        normalized: Any,
        *,
        accepted_last_trade_prices: Sequence[Any],
    ) -> None:
        try:
            for market in normalized.new_markets:
                market.run_id = self.run_id
                market.recorder_version = RECORDER_VERSION
                market.schema_version = SCHEMA_VERSION
            if normalized.new_markets:
                self.db.upsert_markets(normalized.new_markets, run_id=self.run_id)

            resolved_placeholders = [
                self._resolved_market_placeholder(event)
                for event in normalized.market_events
                if event.event_type == "market_resolved"
            ]
            if resolved_placeholders:
                self.db.insert_markets_ignore(resolved_placeholders, run_id=self.run_id)
                self.db.update_market_statuses(
                    [
                        (event.market_id, "resolved")
                        for event in normalized.market_events
                        if event.event_type == "market_resolved"
                    ],
                    run_id=self.run_id,
                )

            for event in normalized.market_events:
                event.run_id = self.run_id
                event.recorder_version = RECORDER_VERSION
                event.schema_version = SCHEMA_VERSION
            if normalized.market_events:
                self.db.insert_market_events(normalized.market_events)

            for trade in accepted_last_trade_prices:
                trade.run_id = self.run_id
                trade.recorder_version = RECORDER_VERSION
                trade.schema_version = SCHEMA_VERSION
            if accepted_last_trade_prices:
                self.db.insert_trades(accepted_last_trade_prices)

            for record in normalized.tick_size_changes:
                record.run_id = self.run_id
                record.recorder_version = RECORDER_VERSION
                record.schema_version = SCHEMA_VERSION
            if normalized.tick_size_changes:
                self.db.insert_tick_size_changes(normalized.tick_size_changes)

            for record in normalized.best_bid_ask_updates:
                record.run_id = self.run_id
                record.recorder_version = RECORDER_VERSION
                record.schema_version = SCHEMA_VERSION
            if normalized.best_bid_ask_updates:
                self.db.insert_best_bid_ask_updates(normalized.best_bid_ask_updates)
        except Exception:
            self.log.exception("normalized_ws_event_write_error")

    async def _handle_ws_message(self, raw_message: str | bytes) -> None:
        local_arrival_ns = time.time_ns()
        self._last_ws_event_monotonic = time.monotonic()
        self._capture_raw_ws_message(
            raw_message,
            local_arrival_ns=local_arrival_ns,
        )

        try:
            async with self._state_lock:
                token_map = dict(self.state.token_to_market)
            normalized = normalize_ws_message(
                raw_message,
                token_to_market=token_map,
            )
        except Exception:
            if not self._raw_ws_events_persistence_enabled():
                self._raw_ws_malformed_events += 1
            self.log.debug("ws_message_parse_error", exc_info=True)
            return

        accepted_last_trade_prices: list[Any] = []
        last_trade_price_ids = {id(trade) for trade in normalized.last_trade_prices}
        async with self._state_lock:
            for update in normalized.orderbook_updates:
                self.state.apply_orderbook_update(
                    token_id=update.token_id,
                    bids=update.bids,
                    asks=update.asks,
                )
            for change in normalized.price_changes:
                self.state.apply_price_change(
                    token_id=change.token_id,
                    book_side=change.book_side,
                    price=change.price,
                    size=change.size,
                )
            for trade in normalized.trades:
                accepted = self.state.apply_trade(trade)
                if accepted and id(trade) in last_trade_price_ids:
                    accepted_last_trade_prices.append(trade)
            self.state.apply_status_events(normalized.status_events)
        self._persist_normalized_ws_records(
            normalized,
            accepted_last_trade_prices=accepted_last_trade_prices,
        )

    async def _run_trade_backfill_cycle(self) -> None:
        async with self._state_lock:
            tracked = [
                (
                    market_id,
                    runtime.metadata.yes_token_id,
                    runtime.metadata.no_token_id,
                    dict(runtime.metadata.token_ids),
                    runtime.metadata.condition_id,
                    runtime.metadata.start_time,
                    runtime.metadata.close_time,
                )
                for market_id, runtime in self.state.markets.items()
            ]

        if not tracked:
            self._last_trade_backfill_summary = {
                "fetched_trades": 0,
                "accepted_trades": 0,
                "rejected_before_start": 0,
                "rejected_after_close": 0,
                "skipped_already_seen": 0,
                "skipped_before_cursor": 0,
                "rejected_invalid_market": 0,
                "market_cycle_summaries": [],
                "last_accepted_trade_ts": None,
            }
            return

        fetched = 0
        accepted = 0
        rejected_before_start = 0
        rejected_after_close = 0
        skipped_already_seen = 0
        skipped_before_cursor = 0
        rejected_invalid_market = 0
        last_accepted_trade_ts = None
        market_cycle_summaries: list[dict[str, Any]] = []

        for (
            market_id,
            yes_token_id,
            no_token_id,
            token_map,
            condition_id,
            market_start,
            market_close,
        ) in tracked:
            cursor = self._backfill_cursor.get(market_id)
            if cursor is None:
                cursor = TradeBackfillCursor(last_timestamp=market_start or utc_now())
                self._backfill_cursor[market_id] = cursor

            if market_start is not None and cursor.last_timestamp < market_start:
                cursor.last_timestamp = market_start
                cursor.last_trade_id = ""

            cursor_before = (cursor.last_timestamp, cursor.last_trade_id)
            since = cursor.last_timestamp
            if market_start is not None and since < market_start:
                since = market_start

            self.log.info(
                "trade_backfill_market_fetch_start",
                extra={
                    "market_id": market_id,
                    "cursor_before_timestamp": to_iso(cursor_before[0]),
                    "cursor_before_trade_id": cursor_before[1],
                    "fetch_since_timestamp": to_iso(since),
                    "fetch_since_operator": "trade.timestamp >= since",
                    "market_window_operator": (
                        "market_start_time <= trade.timestamp < market_close_time"
                    ),
                    "market_start_time": to_iso(market_start),
                    "market_close_time": to_iso(market_close),
                },
            )

            try:
                token_ids = [token for token in (yes_token_id, no_token_id) if token]
                if not token_ids and token_map:
                    for side in ("YES", "NO"):
                        token = token_map.get(side)
                        if token:
                            token_ids.append(str(token))
                if not token_ids and token_map:
                    token_ids.extend(str(token) for token in token_map.values() if token)
                token_ids = sorted(set(token_ids))
                trades = await self.clob.fetch_recent_trades(
                    market_id=market_id,
                    token_ids=token_ids,
                    condition_id=condition_id,
                    since=since,
                    limit=500,
                )
                self._record_api_call(success=True)
                fetched += len(trades)
            except Exception:
                self._record_api_call(success=False)
                self.log.warning(
                    "trade_backfill_failed",
                    extra={"market_id": market_id},
                    exc_info=True,
                )
                continue

            sorted_trades = sorted(trades, key=lambda trade: (trade.timestamp, trade.trade_id))
            market_fetched = len(sorted_trades)
            market_accepted = 0
            market_rejected_before_start = 0
            market_rejected_after_close = 0
            market_skipped_before_cursor = 0
            market_skipped_already_seen = 0
            market_rejected_invalid = 0
            skip_reason_counts: Counter[str] = Counter()
            duplicate_samples: list[dict[str, Any]] = []

            async with self._state_lock:
                for trade in sorted_trades:
                    if market_start is not None and trade.timestamp < market_start:
                        rejected_before_start += 1
                        market_rejected_before_start += 1
                        skip_reason_counts["before_market_start"] += 1
                        continue
                    if market_close is not None and trade.timestamp >= market_close:
                        rejected_after_close += 1
                        market_rejected_after_close += 1
                        skip_reason_counts["after_market_close"] += 1
                        continue
                    if _is_before_or_at_cursor(trade, cursor):
                        skipped_before_cursor += 1
                        market_skipped_before_cursor += 1
                        skip_reason_counts["before_or_at_cursor"] += 1
                        continue

                    trade.run_id = self.run_id
                    apply_reason = self.state.apply_trade_with_reason(trade)
                    if apply_reason == "accepted":
                        accepted += 1
                        market_accepted += 1
                        _advance_cursor_if_newer(trade, cursor)
                        if (
                            last_accepted_trade_ts is None
                            or trade.timestamp > last_accepted_trade_ts
                        ):
                            last_accepted_trade_ts = trade.timestamp
                        continue

                    if apply_reason == "duplicate_trade_id":
                        skipped_already_seen += 1
                        market_skipped_already_seen += 1
                        skip_reason_counts["duplicate_for_same_market"] += 1
                        _advance_cursor_if_newer(trade, cursor)
                        if len(duplicate_samples) < 5:
                            duplicate_samples.append(
                                {
                                    "trade_id": trade.trade_id,
                                    "timestamp": to_iso(trade.timestamp),
                                    "verified_previously_accepted_same_market": True,
                                }
                            )
                        continue

                    if apply_reason == "trade_before_market_start":
                        rejected_before_start += 1
                        market_rejected_before_start += 1
                        skip_reason_counts["before_market_start"] += 1
                        continue

                    if apply_reason == "trade_after_market_close":
                        rejected_after_close += 1
                        market_rejected_after_close += 1
                        skip_reason_counts["after_market_close"] += 1
                        continue

                    rejected_invalid_market += 1
                    market_rejected_invalid += 1
                    skip_reason_counts[f"invalid_{apply_reason}"] += 1

            self._backfill_cursor[market_id] = cursor
            market_summary = {
                "market_id": market_id,
                "cursor_before_timestamp": to_iso(cursor_before[0]),
                "cursor_before_trade_id": cursor_before[1],
                "cursor_after_timestamp": to_iso(cursor.last_timestamp),
                "cursor_after_trade_id": cursor.last_trade_id,
                "fetched": market_fetched,
                "accepted": market_accepted,
                "rejected_before_start": market_rejected_before_start,
                "rejected_after_close": market_rejected_after_close,
                "rejected_invalid_market": market_rejected_invalid,
                "skipped_before_cursor": market_skipped_before_cursor,
                "skipped_already_seen": market_skipped_already_seen,
                "skip_reason_counts": dict(skip_reason_counts),
                "duplicate_samples": duplicate_samples,
            }
            market_cycle_summaries.append(market_summary)
            self.log.info("trade_backfill_market_cycle", extra=market_summary)

        self._last_trade_backfill_summary = {
            "fetched_trades": fetched,
            "accepted_trades": accepted,
            "rejected_before_start": rejected_before_start,
            "rejected_after_close": rejected_after_close,
            "skipped_already_seen": skipped_already_seen,
            "skipped_before_cursor": skipped_before_cursor,
            "rejected_invalid_market": rejected_invalid_market,
            "market_cycle_summaries": market_cycle_summaries,
            "last_accepted_trade_ts": last_accepted_trade_ts,
        }
        self.log.info(
            "trade_backfill_cycle",
            extra={
                "market_ids": [market_id for market_id, *_ in tracked],
                "markets": len(tracked),
                "fetched": fetched,
                "accepted": accepted,
                "rejected_before_start": rejected_before_start,
                "rejected_after_close": rejected_after_close,
                "skipped_already_seen": skipped_already_seen,
                "skipped_before_cursor": skipped_before_cursor,
                "rejected_invalid_market": rejected_invalid_market,
                "last_accepted_trade_ts": to_iso(last_accepted_trade_ts),
            },
        )

    def _maybe_log_heartbeat(
        self,
        *,
        metrics: Any,
        tracked_markets: int,
        snapshots: Sequence[Any],
        skipped_outside_window: Sequence[str],
        skipped_missing_orderbook: Sequence[str],
        skipped_no_state_change: Sequence[str],
        snapshot_generation_failed: Sequence[str],
        features: Sequence[Any],
    ) -> None:
        now_mono = time.monotonic()
        if (
            now_mono - float(getattr(self, "_last_heartbeat_monotonic", 0.0))
            < float(getattr(self.settings, "heartbeat_log_interval_sec", 30))
        ):
            return
        self._last_heartbeat_monotonic = now_mono

        snapshot_count = len(snapshots)
        snapshot_attempted_with_orderbook = (
            snapshot_count + len(skipped_no_state_change) + len(snapshot_generation_failed)
        )

        if snapshot_count > 0:
            non_null_mid = sum(
                1 for s in snapshots if getattr(s, "mid_price_yes", None) is not None
            )
            has_trade_data = sum(
                1 for s in snapshots if int(getattr(s, "has_trade_data", 0) or 0) == 1
            )
            partial_books = sum(
                1
                for s in snapshots
                if int(getattr(s, "is_partial_orderbook", 0) or 0) == 1
            )
            gap_affected = sum(
                1 for s in snapshots if int(getattr(s, "is_gap_affected", 0) or 0) == 1
            )
            feature_ready = sum(
                1 for s in snapshots if int(getattr(s, "feature_ready", 0) or 0) == 1
            )
            mid_pct = 100.0 * non_null_mid / snapshot_count
            trade_data_pct = 100.0 * has_trade_data / snapshot_count
            snapshot_quality_metrics_source = "current_written_snapshots"
            snapshot_quality_sample_size = snapshot_count
            self._last_snapshot_quality_rollup = {
                "snapshot_count": snapshot_count,
                "non_null_mid": non_null_mid,
                "has_trade_data": has_trade_data,
                "partial_books": partial_books,
                "gap_affected": gap_affected,
                "feature_ready": feature_ready,
                "mid_pct": mid_pct,
                "trade_data_pct": trade_data_pct,
            }
        else:
            rollup = getattr(self, "_last_snapshot_quality_rollup", None)
            carry_forward_allowed = (
                rollup is not None
                and len(skipped_no_state_change) > 0
                and len(skipped_missing_orderbook) == 0
                and len(snapshot_generation_failed) == 0
            )
            if carry_forward_allowed:
                non_null_mid = int(rollup.get("non_null_mid", 0))
                has_trade_data = int(rollup.get("has_trade_data", 0))
                partial_books = int(rollup.get("partial_books", 0))
                gap_affected = int(rollup.get("gap_affected", 0))
                feature_ready = int(rollup.get("feature_ready", 0))
                mid_pct = float(rollup.get("mid_pct", 0.0))
                trade_data_pct = float(rollup.get("trade_data_pct", 0.0))
                snapshot_quality_metrics_source = "carried_forward_no_state_change"
                snapshot_quality_sample_size = int(rollup.get("snapshot_count", 0))
            else:
                non_null_mid = 0
                has_trade_data = 0
                partial_books = 0
                gap_affected = 0
                feature_ready = 0
                mid_pct = 0.0
                trade_data_pct = 0.0
                snapshot_quality_metrics_source = "current_cycle_no_written_snapshots"
                snapshot_quality_sample_size = 0

        btc_heartbeat = self._btc_price_feed_heartbeat_fields()
        self.log.info(
            "recorder_heartbeat",
            extra={
                "tracked_markets": tracked_markets,
                "active_markets_snapshot_attempted": metrics.active_markets_snapshot_attempted,
                "snapshots_written": metrics.snapshots_written,
                "snapshot_markets_skipped": metrics.snapshot_markets_skipped,
                "snapshot_no_write_needed_no_state_change": len(skipped_no_state_change),
                "snapshot_generation_failed": len(snapshot_generation_failed),
                "snapshot_attempted_with_orderbook": snapshot_attempted_with_orderbook,
                "discovery_candidates_seen": metrics.discovery_candidates_seen,
                "discovery_candidates_matched": metrics.discovery_candidates_matched,
                "discovery_strict_5m_candidates": metrics.discovery_strict_5m_candidates,
                "discovery_broad_btc_candidates": metrics.discovery_broad_btc_candidates,
                "discovery_fallback_used": metrics.discovery_fallback_used,
                "api_call_count": metrics.api_call_count,
                "api_success_count": metrics.api_success_count,
                "api_failure_count": metrics.api_failure_count,
                "ws_reconnect_count": metrics.ws_reconnect_count,
                "raw_ws_events_seen": metrics.raw_ws_events_seen,
                "raw_ws_events_written": metrics.raw_ws_events_written,
                "malformed_ws_events": metrics.malformed_ws_events,
                "raw_ws_write_failures": metrics.raw_ws_write_failures,
                "last_ws_event_age_sec": round(metrics.last_ws_event_age_sec, 3)
                if metrics.last_ws_event_age_sec is not None
                else None,
                "subscribed_asset_count": metrics.subscribed_asset_count,
                "subscribed_asset_ids_json": metrics.subscribed_asset_ids_json,
                "fetched_trades": metrics.fetched_trades,
                "accepted_trades": metrics.accepted_trades,
                "rejected_before_start": metrics.rejected_before_start,
                "rejected_after_close": metrics.rejected_after_close,
                "skipped_already_seen": metrics.skipped_already_seen,
                "snapshots_skipped_outside_window": len(skipped_outside_window),
                "snapshots_skipped_missing_orderbook": len(skipped_missing_orderbook),
                "snapshot_mid_price_yes_non_null_pct": round(mid_pct, 2),
                "snapshot_has_trade_data_pct": round(trade_data_pct, 2),
                "snapshot_partial_orderbook_count": partial_books,
                "snapshot_gap_affected_count": gap_affected,
                "snapshot_feature_ready_count": feature_ready,
                "snapshot_quality_metrics_source": snapshot_quality_metrics_source,
                "snapshot_quality_sample_size": snapshot_quality_sample_size,
                "rows_inserted": metrics.rows_inserted,
                "duplicates": metrics.duplicate_rows_skipped,
                "cycle_ms": round(metrics.cycle_duration_ms or 0.0, 2),
                "db_ms": round(metrics.db_write_time_ms or 0.0, 2),
                **btc_heartbeat,
            },
        )

        if int(metrics.accepted_trades or 0) > 0 and snapshot_count > 0 and has_trade_data == 0:
            self.log.warning(
                "accepted_trades_without_snapshot_trade_attachment",
                extra={
                    "accepted_trades": metrics.accepted_trades,
                    "snapshot_has_trade_data_count": has_trade_data,
                    "snapshot_has_trade_data_pct": round(trade_data_pct, 2),
                    "trade_snapshot_freshness_sec": self.settings.trade_snapshot_freshness_sec,
                },
            )
        if int(metrics.accepted_trades or 0) > 0 and metrics.last_accepted_trade_ts is None:
            self.log.warning(
                "accepted_trades_missing_last_accepted_trade_ts",
                extra={"accepted_trades": metrics.accepted_trades},
            )

        feature_rows_by_market: dict[str, int] = {}
        feature_ready_by_market: dict[str, int] = {}
        for feature in features:
            market_id = str(getattr(feature, "market_id", ""))
            feature_rows_by_market[market_id] = feature_rows_by_market.get(market_id, 0) + 1
            if int(getattr(feature, "feature_ready", 0) or 0) == 1:
                feature_ready_by_market[market_id] = (
                    feature_ready_by_market.get(market_id, 0) + 1
                )
        if feature_rows_by_market:
            self.log.info(
                "feature_generation_coverage",
                extra={
                    "feature_rows_by_market": feature_rows_by_market,
                    "feature_ready_by_market": feature_ready_by_market,
                },
            )

        negative_resolution = sum(
            1
            for feature in features
            if getattr(feature, "time_until_resolution", None) is not None
            and float(feature.time_until_resolution) < 0
        )
        if negative_resolution > 0:
            self.log.warning(
                "negative_time_until_resolution_detected",
                extra={"market_count": negative_resolution},
            )

        try:
            integrity = self.db.snapshot_trade_integrity_counts(run_id=self.run_id)
            if int(integrity.get("total_invalid", 0)) > 0:
                self.log.warning("snapshot_trade_integrity_violations", extra=integrity)
        except Exception:
            self.log.exception("snapshot_trade_integrity_check_failed")

    def _btc_price_feed_heartbeat_fields(self) -> dict[str, Any]:
        health = getattr(self, "_btc_feed_health", {}) or {}
        enabled = self._btc_price_feed_enabled_for_settings(self.settings)
        sources: set[str] = set(str(source) for source in health)
        feeds = getattr(self, "btc_price_feeds", None)
        if isinstance(feeds, dict):
            sources.update(str(source) for source in feeds)
        legacy_feed = getattr(self, "btc_price_feed", None)
        if legacy_feed is not None:
            sources.add(
                str(
                    getattr(legacy_feed, "source", None)
                    or getattr(self.settings, "btc_price_feed_source", "unknown")
                    or "unknown"
                )
            )
        latest_price_by_source: dict[str, float | None] = {}
        latest_exchange_timestamp_by_source: dict[str, str | None] = {}
        age_sec_by_source: dict[str, float | None] = {}
        rows_inserted_by_source: dict[str, int] = {}
        stale_dropped_by_source: dict[str, int] = {}
        reconnect_count_by_source: dict[str, int] = {}
        last_error_by_source: dict[str, str | None] = {}
        for source in sorted(source for source in sources if source):
            state = health.get(source, {})
            feed = feeds.get(source) if isinstance(feeds, dict) else None
            if feed is None and legacy_feed is not None:
                legacy_source = str(
                    getattr(legacy_feed, "source", None)
                    or getattr(self.settings, "btc_price_feed_source", "unknown")
                    or "unknown"
                )
                if legacy_source == source:
                    feed = legacy_feed
            latest_price_by_source[source] = state.get("latest_price")
            latest_exchange_timestamp_by_source[source] = state.get(
                "latest_exchange_timestamp"
            )
            exchange_ts = parse_timestamp(state.get("latest_exchange_timestamp"))
            if exchange_ts is not None:
                age = (datetime.now(timezone.utc) - exchange_ts).total_seconds()
                age_sec_by_source[source] = round(max(0.0, age), 3)
            elif state.get("latest_sample_monotonic") is not None:
                age_sec_by_source[source] = round(
                    time.monotonic() - float(state["latest_sample_monotonic"]),
                    3,
                )
            else:
                age_sec_by_source[source] = None
            rows_inserted_by_source[source] = int(state.get("rows_inserted") or 0)
            stale_dropped_by_source[source] = int(
                state.get("stale_rows_dropped")
                or state.get("stale_messages_dropped")
                or 0
            )
            reconnect_count_by_source[source] = int(
                getattr(feed, "reconnect_count", 0) or 0
            )
            last_error_by_source[source] = state.get("last_error")
        healthy = (
            any(
                age_sec is not None
                and age_sec
                <= float(
                    getattr(self.settings, "btc_price_max_exchange_age_sec", 15.0)
                    or 15.0
                )
                for age_sec in age_sec_by_source.values()
            )
            if enabled
            else False
        )
        return {
            "btc_price_feed_enabled": enabled,
            "btc_price_feed_healthy": healthy,
            "latest_btc_price_by_source": latest_price_by_source,
            "latest_btc_exchange_timestamp_by_source": latest_exchange_timestamp_by_source,
            "btc_price_age_sec_by_source": age_sec_by_source,
            "btc_price_latest_age_sec_by_source": age_sec_by_source,
            "btc_price_rows_inserted_by_source": rows_inserted_by_source,
            "btc_stale_messages_dropped_by_source": stale_dropped_by_source,
            "btc_price_stale_rows_dropped_by_source": stale_dropped_by_source,
            "btc_feed_reconnect_count_by_source": reconnect_count_by_source,
            "btc_price_reconnects_by_source": reconnect_count_by_source,
            "btc_price_last_error_by_source": last_error_by_source,
        }

    def _record_api_call(self, success: bool) -> None:
        self._api_call_count_since_snapshot += 1
        if success:
            self._api_success_count_since_snapshot += 1
        else:
            self._api_failure_count_since_snapshot += 1

    def _drain_api_counters(self) -> tuple[int, int, int]:
        call_count = self._api_call_count_since_snapshot
        success_count = self._api_success_count_since_snapshot
        failure_count = self._api_failure_count_since_snapshot
        self._api_call_count_since_snapshot = 0
        self._api_success_count_since_snapshot = 0
        self._api_failure_count_since_snapshot = 0
        return call_count, success_count, failure_count

    async def _sleep_until_next_cycle(
        self,
        cycle_start: float,
        interval_sec: float,
    ) -> None:
        elapsed = time.monotonic() - cycle_start
        remaining = max(0.0, interval_sec - elapsed)
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            return

    async def _shutdown(self) -> None:
        self.log.info(
            "shutdown_started",
            extra={
                "run_id": self.run_id,
                "task_count": len(self._tasks),
                "task_names": [task.get_name() for task in self._tasks],
            },
        )
        for task in self._tasks:
            task.cancel()
        task_results: list[Any] = []
        if self._tasks:
            task_results = list(await asyncio.gather(*self._tasks, return_exceptions=True))
        task_errors = [
            repr(result)
            for result in task_results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        self.log.info(
            "shutdown_task_summary",
            extra={
                "task_count": len(self._tasks),
                "cancelled_tasks": sum(1 for task in self._tasks if task.cancelled()),
                "task_errors": task_errors[:10],
            },
        )
        await self.gamma.close()
        await self.clob.close()
        btc_feeds = getattr(self, "btc_price_feeds", None)
        feeds_to_close: list[tuple[str, BTCPriceFeed]] = []
        if isinstance(btc_feeds, dict):
            feeds_to_close.extend(
                (str(source), feed) for source, feed in btc_feeds.items()
            )
        legacy_feed = getattr(self, "btc_price_feed", None)
        if legacy_feed is not None and all(
            id(feed) != id(legacy_feed) for _source, feed in feeds_to_close
        ):
            feeds_to_close.append(("legacy", legacy_feed))
        for source, feed in feeds_to_close:
            try:
                await feed.close()
            except Exception:
                self.log.exception(
                    "btc_price_feed_close_error",
                    extra={"source": source},
                )
        self.db.close()
        self.log.info(
            "shutdown_complete",
            extra={
                "run_id": self.run_id,
                "ws_reconnect_count": getattr(self, "_ws_reconnect_count", 0),
                "raw_ws_events_seen": getattr(self, "_raw_ws_events_seen", 0),
                "raw_ws_events_written": getattr(self, "_raw_ws_events_written", 0),
                "malformed_ws_events": getattr(self, "_raw_ws_malformed_events", 0),
                "raw_ws_write_failures": getattr(self, "_raw_ws_write_failures", 0),
            },
        )


__all__ = [
    "RecorderApp",
    "RecorderMetricRecord",
    "TradeBackfillCursor",
    "classify_market_phase",
    "select_primary_market",
    "validate_strict_market",
    "validate_strict_btc_5m_market",
]
