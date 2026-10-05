from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models import MarketMetadata, PriceLevel
from src.recorder import select_primary_market
from src.state import RecorderState


def _market(market_id: str, question: str, start: datetime, close: datetime) -> MarketMetadata:
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event-{market_id}",
        question=question,
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        token_ids={"YES": f"{market_id}_yes", "NO": f"{market_id}_no"},
        yes_token_id=f"{market_id}_yes",
        no_token_id=f"{market_id}_no",
        run_id="run_integration",
    )


def test_active_primary_selection_flows_into_snapshot_writes() -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    active = _market(
        "m_active",
        "Bitcoin Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    future = _market(
        "m_future",
        "Bitcoin Up or Down - 00:05-00:10",
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
    )

    selected, mode, _tracking, _phases, _skipped, _selection_reasons = select_primary_market(
        [active, future],
        now=now,
    )

    assert selected is not None
    assert selected.market_id == "m_active"
    assert mode == "selected_active"

    state = RecorderState()
    state.upsert_markets([selected])

    runtime = state.markets["m_active"]
    runtime.yes_bids = [PriceLevel(price=0.49, size=10.0)]
    runtime.yes_asks = [PriceLevel(price=0.51, size=11.0)]
    runtime.no_bids = [PriceLevel(price=0.49, size=9.0)]
    runtime.no_asks = [PriceLevel(price=0.51, size=8.0)]

    (
        snapshots,
        orderbook_levels,
        _features,
        _trades,
        _events,
        skipped_outside_window,
        skipped_missing_orderbook,
        _snapshot_trade_warnings,
        _invalid_active_past_close,
    ) = state.build_cycle_records(
        timestamp=now + timedelta(seconds=1),
        levels_to_store=2,
        feature_orderbook_depth=2,
        jump_threshold=0.05,
        expected_snapshot_interval_sec=1.0,
        gap_threshold_multiplier=1.5,
        trade_snapshot_freshness_sec=2,
    )

    assert skipped_outside_window == []
    assert skipped_missing_orderbook == []
    assert len(snapshots) == 1
    assert snapshots[0].market_id == "m_active"
    assert len(orderbook_levels) == 4
