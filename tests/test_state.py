from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.models import MarketMetadata, PriceLevel, TradeRecord, WSStatusEvent
from src.state import RecorderState



def _market(market_id: str, yes_token: str, no_token: str) -> MarketMetadata:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event-{market_id}",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=ts,
        end_time=ts,
        close_time=ts,
        status="open",
        token_ids={"YES": yes_token, "NO": no_token},
    )


def test_prune_inactive_markets_removes_state_and_tokens() -> None:
    state = RecorderState()
    m1 = _market("m1", "m1_yes", "m1_no")
    m2 = _market("m2", "m2_yes", "m2_no")

    state.upsert_markets([m1, m2])
    assert set(state.all_token_ids()) == {"m1_yes", "m1_no", "m2_yes", "m2_no"}

    events, removed_markets, removed_tokens, status_updates = state.prune_inactive_markets(
        active_market_ids={"m1"},
        now=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
    )

    assert removed_markets == ["m2"]
    assert set(removed_tokens) == {"m2_yes", "m2_no"}
    assert set(state.markets.keys()) == {"m1"}
    assert set(state.all_token_ids()) == {"m1_yes", "m1_no"}
    assert len(events) == 1
    assert events[0].market_id == "m2"
    assert events[0].event_type == "status_changed"
    assert status_updates == []


def test_build_cycle_records_skips_future_selected_market_for_snapshots() -> None:
    state = RecorderState()
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="future_m",
        event_id="event-future",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        close_time=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        status="open",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )

    state.upsert_markets([market])
    runtime = state.markets["future_m"]
    runtime.yes_bids = [PriceLevel(price=0.45, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.46, size=11)]

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        skipped_outside_window,
        skipped_missing_orderbook,
        _snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=ts,
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )

    assert snapshots == []
    assert skipped_outside_window == ["future_m"]
    assert skipped_missing_orderbook == []


def test_apply_status_events_skips_market_opened_ws_events() -> None:
    state = RecorderState()
    market = _market("m1", "m1_yes", "m1_no")
    state.upsert_markets([market])

    opened = WSStatusEvent(
        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        market_id="m1",
        event_type="market_opened",
        details="{}",
    )
    halted = WSStatusEvent(
        timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        market_id="m1",
        event_type="trading_halted",
        details="{}",
    )
    state.apply_status_events([opened, halted])

    runtime = state.markets["m1"]
    assert len(runtime.pending_events) == 1
    assert runtime.pending_events[0].event_type == "trading_halted"


def test_trade_before_market_start_not_used_for_snapshot_trade_fields() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_trade_guard",
        event_id="event-guard",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_trade_guard"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=11)]

    stale_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        market_id="m_trade_guard",
        trade_id="trade123:tok_yes",
        price=0.5,
        size=3.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    # Stale trade should be rejected from runtime state.
    assert state.apply_trade(stale_trade) is False

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        _snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(snapshots) == 1
    assert snapshots[0].has_trade_data == 0
    assert snapshots[0].last_trade_time is None


def test_snapshot_trade_uses_latest_timestamp_even_when_trades_arrive_out_of_order() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_trade_order",
        event_id="event-order",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_trade_order"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]

    newer = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
        market_id="m_trade_order",
        trade_id="new:tok_yes",
        price=0.51,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    older = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
        market_id="m_trade_order",
        trade_id="old:tok_yes",
        price=0.49,
        size=1.0,
        side="SELL",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(newer) is True
    assert state.apply_trade(older) is True

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        _snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(snapshots) == 1
    assert snapshots[0].has_trade_data == 1
    assert snapshots[0].last_trade_time == newer.timestamp
    assert snapshots[0].last_trade_price == newer.price


def test_snapshot_ignores_cached_last_trade_when_not_in_recent_window() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_snapshot_guard_start",
        event_id="event-guard-start",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_snapshot_guard_start"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.last_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 9, 59, tzinfo=timezone.utc),
        market_id="m_snapshot_guard_start",
        trade_id="stale:tok_yes",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 10, 10, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(snapshots) == 1
    assert snapshots[0].last_trade_price is None
    assert snapshots[0].last_trade_size is None
    assert snapshots[0].last_trade_time is None
    assert snapshots[0].has_trade_data == 0
    assert len(snapshot_trade_warnings) == 1
    assert snapshot_trade_warnings[0]["reason"] == "last_trade_before_market_start"


def test_snapshot_ignores_future_cached_last_trade_when_not_in_recent_window() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_snapshot_guard_future",
        event_id="event-guard-future",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_snapshot_guard_future"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.last_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 10, 20, tzinfo=timezone.utc),
        market_id="m_snapshot_guard_future",
        trade_id="future:tok_yes",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 10, 10, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(snapshots) == 1
    assert snapshots[0].last_trade_price is None
    assert snapshots[0].last_trade_size is None
    assert snapshots[0].last_trade_time is None
    assert snapshots[0].has_trade_data == 0
    assert snapshot_trade_warnings == []


def test_snapshot_attaches_recent_trade_after_acceptance() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_attach",
        event_id="event-attach",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_attach"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]

    accepted_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        market_id="m_attach",
        trade_id="attach_trade:tok_yes",
        price=0.5,
        size=2.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(accepted_trade) is True

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        _snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )

    assert len(snapshots) == 1
    assert snapshots[0].has_trade_data == 1
    assert snapshots[0].last_trade_price == 0.5
    assert snapshots[0].last_trade_size == 2.0
    assert snapshots[0].last_trade_time == accepted_trade.timestamp


def test_snapshot_attaches_latest_trade_fallback_when_no_recent_trade() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_attach_fallback",
        event_id="event-attach-fallback",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_attach_fallback"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=10)]
    runtime.no_asks = [PriceLevel(price=0.51, size=10)]

    accepted_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        market_id="m_attach_fallback",
        trade_id="attach_fallback_trade:tok_yes",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(accepted_trade) is True

    (
        snapshots,
        _levels,
        _features,
        _trades,
        _events,
        _skipped_outside_window,
        _skipped_missing_orderbook,
        snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
        levels_to_store=3,
        feature_orderbook_depth=3,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(snapshots) == 1
    assert snapshots[0].has_trade_data == 1
    assert snapshots[0].last_trade_time == accepted_trade.timestamp
    assert snapshot_trade_warnings == []


def test_build_cycle_records_deduplicates_unchanged_snapshots() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_dedupe",
        event_id="event-dedupe",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    runtime = state.markets["m_dedupe"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]

    first = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    second = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )

    assert len(first[0]) == 1
    assert len(second[0]) == 0

    fresh_trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        market_id="m_dedupe",
        trade_id="dedupe-trade:tok_yes",
        price=0.5,
        size=1.5,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(fresh_trade) is True

    third = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
        levels_to_store=5,
        feature_orderbook_depth=5,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(third[0]) == 1


def test_reset_market_runtime_clears_trade_dedupe_state() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_reset",
        event_id="event-reset",
        question="BTC Up or Down - 00:00-00:05",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
    )
    state.upsert_markets([market])
    trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        market_id="m_reset",
        trade_id="dup_id",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(trade) is True
    assert state.apply_trade(trade) is False

    state.reset_market_runtime("m_reset")
    assert state.apply_trade(trade) is True


def test_snapshot_quality_flags_partial_and_gap_affected() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_quality",
        event_id="event-quality",
        question="BTC Up or Down - 00:00-00:05",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        strict_validation_passed=1,
        run_id="run_quality",
    )
    state.upsert_markets([market])
    runtime = state.markets["m_quality"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=8)]
    runtime.no_asks = [PriceLevel(price=0.51, size=9)]

    first = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        levels_to_store=3,
        feature_orderbook_depth=3,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(first[0]) == 1
    first_snapshot = first[0][0]
    assert first_snapshot.is_partial_orderbook == 1
    assert first_snapshot.missing_level_count == 8
    assert first_snapshot.is_gap_affected == 0

    # Trigger a changed snapshot with a large timing gap.
    runtime.apply_price_level("YES", "bid", 0.50, 11)
    second = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 6, tzinfo=timezone.utc),
        levels_to_store=3,
        feature_orderbook_depth=3,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(second[0]) == 1
    second_snapshot = second[0][0]
    assert second_snapshot.is_gap_affected == 1
    assert second_snapshot.time_gap_from_prev_snapshot_sec == 5.0


def test_feature_readiness_transitions_with_history_and_trade_inputs(caplog) -> None:
    caplog.set_level(logging.INFO)

    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 1, 5, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_feature_ready",
        event_id="event-feature-ready",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        strict_validation_passed=1,
        initial_volume=100.0,
    )
    state.upsert_markets([market])
    runtime = state.markets["m_feature_ready"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=9)]
    runtime.no_asks = [PriceLevel(price=0.51, size=9)]

    first = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        levels_to_store=1,
        feature_orderbook_depth=1,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(first[2]) == 1
    assert first[2][0].feature_ready == 0

    for offset in range(2, 61):
        state.build_cycle_records(
            timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            + timedelta(seconds=offset),
            levels_to_store=1,
            feature_orderbook_depth=1,
            jump_threshold=0.05,
            expected_snapshot_interval_sec=1.0,
            gap_threshold_multiplier=1.5,
            trade_snapshot_freshness_sec=2,
        )

    trade = TradeRecord(
        timestamp=datetime(2026, 1, 1, 0, 1, 1, tzinfo=timezone.utc),
        market_id="m_feature_ready",
        trade_id="feature-ready-trade:tok_yes",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(trade) is True

    final = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 1, 1, tzinfo=timezone.utc),
        levels_to_store=1,
        feature_orderbook_depth=1,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(final[2]) == 1
    assert final[2][0].feature_ready == 1
    assert len(final[0]) == 1
    assert final[0][0].has_trade_data == 1

    not_ready_records = [
        record
        for record in caplog.records
        if record.message == "feature_row_not_ready"
        and getattr(record, "market_id", None) == "m_feature_ready"
    ]
    assert any("insufficient_history" in getattr(record, "reasons", []) for record in not_ready_records)
    assert any("missing_trade_inputs" in getattr(record, "reasons", []) for record in not_ready_records)
    assert any(
        record.message == "feature_row_ready"
        and getattr(record, "market_id", None) == "m_feature_ready"
        for record in caplog.records
    )


def test_partial_orderbook_blocks_feature_ready(caplog) -> None:
    caplog.set_level(logging.INFO)

    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    market = MarketMetadata(
        market_id="m_partial_block",
        event_id="event-partial-block",
        question="BTC Up or Down - test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        strict_validation_passed=1,
    )
    state.upsert_markets([market])
    runtime = state.markets["m_partial_block"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=9)]
    runtime.no_asks = [PriceLevel(price=0.51, size=9)]

    record_set = state.build_cycle_records(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        levels_to_store=3,
        feature_orderbook_depth=3,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    assert len(record_set[0]) == 1
    assert record_set[0][0].is_partial_orderbook == 1
    assert len(record_set[2]) == 1
    assert record_set[2][0].feature_ready == 0
    partial_logs = [
        record
        for record in caplog.records
        if record.message == "feature_row_not_ready"
        and getattr(record, "market_id", None) == "m_partial_block"
    ]
    assert any("partial_orderbook" in getattr(record, "reasons", []) for record in partial_logs)


def test_feature_60s_fields_populate_with_jittered_snapshot_cadence() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = start + timedelta(minutes=10)
    market = MarketMetadata(
        market_id="m_feature_jitter_60s",
        event_id="event-feature-jitter-60s",
        question="BTC Up or Down - jitter test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        strict_validation_passed=1,
        initial_volume=100.0,
    )
    state.upsert_markets([market])
    runtime = state.markets["m_feature_jitter_60s"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=9)]
    runtime.no_asks = [PriceLevel(price=0.51, size=9)]

    first_non_null_60s: FeatureRecord | None = None
    # 1.001s cadence reproduces real scheduler jitter from production runs.
    for i in range(1, 130):
        timestamp = start + timedelta(milliseconds=1001 * i)
        features = state.build_cycle_records(
            timestamp=timestamp,
            levels_to_store=1,
            feature_orderbook_depth=1,
            jump_threshold=0.05,
            expected_snapshot_interval_sec=1.0,
            gap_threshold_multiplier=1.5,
            trade_snapshot_freshness_sec=2,
        )[2]
        if not features:
            continue
        candidate = features[0]
        if candidate.price_change_60s is not None:
            first_non_null_60s = candidate
            break

    assert first_non_null_60s is not None
    assert first_non_null_60s.rolling_mean_60s is not None
    assert first_non_null_60s.rolling_volatility_60s is not None
    assert first_non_null_60s.avg_volume_60s is not None


def test_trade_flow_uses_5s_window_not_2s() -> None:
    state = RecorderState()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = start + timedelta(minutes=10)
    market = MarketMetadata(
        market_id="m_trade_flow_5s",
        event_id="event-trade-flow-5s",
        question="BTC Up or Down - trade flow test",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        strict_validation_passed=1,
        initial_volume=100.0,
    )
    state.upsert_markets([market])
    runtime = state.markets["m_trade_flow_5s"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=10)]
    runtime.no_bids = [PriceLevel(price=0.49, size=9)]
    runtime.no_asks = [PriceLevel(price=0.51, size=9)]

    for i in range(1, 70):
        state.build_cycle_records(
            timestamp=start + timedelta(seconds=i),
            levels_to_store=1,
            feature_orderbook_depth=1,
            jump_threshold=0.05,
            expected_snapshot_interval_sec=1.0,
            gap_threshold_multiplier=1.5,
            trade_snapshot_freshness_sec=2,
        )

    # Trade is 4 seconds old at snapshot time: should still contribute to 5s flow.
    assert state.apply_trade(
        TradeRecord(
            timestamp=start + timedelta(seconds=71),
            market_id="m_trade_flow_5s",
            trade_id="trade-flow-5s:tok_yes",
            price=0.5,
            size=2.0,
            side="BUY",
            maker=None,
            taker=None,
        )
    )
    records = state.build_cycle_records(
        timestamp=start + timedelta(seconds=75),
        levels_to_store=1,
        feature_orderbook_depth=1,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )
    feature = records[2][0]
    assert feature.buy_volume_5s is not None
    assert feature.net_trade_flow_5s is not None
    assert feature.trade_flow_ratio_5s is not None
