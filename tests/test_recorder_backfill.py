from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.models import MarketMetadata, TradeRecord
from src.recorder import RecorderApp, TradeBackfillCursor
from src.state import RecorderState


class _FakeClobClient:
    def __init__(self, by_market_id: dict[str, list[TradeRecord]]) -> None:
        self.by_market_id = by_market_id
        self.calls: list[dict[str, object]] = []

    async def fetch_recent_trades(
        self,
        market_id: str,
        token_ids=None,
        condition_id=None,
        since=None,
        limit: int = 500,
    ) -> list[TradeRecord]:
        self.calls.append(
            {
                "market_id": market_id,
                "token_ids": list(token_ids or []),
                "condition_id": condition_id,
                "since": since,
                "limit": limit,
            }
        )
        return list(self.by_market_id.get(market_id, []))


def _market(
    market_id: str,
    start: datetime,
    close: datetime,
    yes_token: str,
    no_token: str,
) -> MarketMetadata:
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event-{market_id}",
        question="Bitcoin Up or Down - 00:00-00:05",
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
        tracking_state="selected_active",
        yes_token_id=yes_token,
        no_token_id=no_token,
        token_ids={"YES": yes_token, "NO": no_token},
        run_id="run_test",
    )


def _build_app_stub(state: RecorderState, clob: _FakeClobClient) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.state = state
    app._state_lock = asyncio.Lock()
    app._backfill_cursor = {}
    app._last_trade_backfill_summary = {}
    app.run_id = "run_test"
    app.clob = clob
    app.log = logging.getLogger("test_recorder_backfill")
    app._api_call_count_since_snapshot = 0
    app._api_success_count_since_snapshot = 0
    app._api_failure_count_since_snapshot = 0
    return app


def test_backfill_after_market_rotation_accepts_fresh_market_trades() -> None:
    old_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    old_close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    new_start = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    new_close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)

    old_market = _market("old_market", old_start, old_close, "tok_old_yes", "tok_old_no")
    new_market = _market("new_market", new_start, new_close, "tok_new_yes", "tok_new_no")

    state = RecorderState()
    state.upsert_markets([old_market])

    old_trade = TradeRecord(
        timestamp=old_start + timedelta(seconds=3),
        market_id="old_market",
        trade_id="shared_trade_id",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    assert state.apply_trade(old_trade) is True

    state.upsert_markets([new_market])
    state.prune_inactive_markets(active_market_ids={"new_market"}, now=new_start)
    state.reset_market_runtime("new_market")

    new_trade = TradeRecord(
        timestamp=new_start + timedelta(seconds=1),
        market_id="new_market",
        trade_id="shared_trade_id",
        price=0.51,
        size=2.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    clob = _FakeClobClient({"new_market": [new_trade]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "new_market": TradeBackfillCursor(last_timestamp=new_start, last_trade_id="")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    summary = app._last_trade_backfill_summary
    assert summary["fetched_trades"] == 1
    assert summary["accepted_trades"] == 1
    assert summary["skipped_already_seen"] == 0
    assert summary["last_accepted_trade_ts"] == new_trade.timestamp


def test_backfill_stale_before_start_not_counted_as_already_seen() -> None:
    start = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    market = _market("m_stale", start, close, "tok_yes", "tok_no")

    state = RecorderState()
    state.upsert_markets([market])

    stale_trade = TradeRecord(
        timestamp=start - timedelta(seconds=10),
        market_id="m_stale",
        trade_id="stale_trade",
        price=0.4,
        size=1.0,
        side="SELL",
        maker=None,
        taker=None,
    )
    clob = _FakeClobClient({"m_stale": [stale_trade]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "m_stale": TradeBackfillCursor(last_timestamp=start, last_trade_id="")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    summary = app._last_trade_backfill_summary
    assert summary["fetched_trades"] == 1
    assert summary["accepted_trades"] == 0
    assert summary["rejected_before_start"] == 1
    assert summary["skipped_already_seen"] == 0
    market_summaries = summary["market_cycle_summaries"]
    assert market_summaries[0]["skip_reason_counts"]["before_market_start"] == 1


def test_backfill_duplicate_skips_are_same_market_verified() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = _market("m_dup", start, close, "tok_yes", "tok_no")

    state = RecorderState()
    state.upsert_markets([market])

    t1 = TradeRecord(
        timestamp=start + timedelta(seconds=1),
        market_id="m_dup",
        trade_id="dup_trade",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    t2 = TradeRecord(
        timestamp=start + timedelta(seconds=2),
        market_id="m_dup",
        trade_id="dup_trade",
        price=0.51,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    clob = _FakeClobClient({"m_dup": [t1, t2]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "m_dup": TradeBackfillCursor(last_timestamp=start, last_trade_id="")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    summary = app._last_trade_backfill_summary
    assert summary["accepted_trades"] == 1
    assert summary["skipped_already_seen"] == 1
    market_summaries = summary["market_cycle_summaries"]
    assert len(market_summaries) == 1
    duplicate_samples = market_summaries[0]["duplicate_samples"]
    assert len(duplicate_samples) == 1
    assert duplicate_samples[0]["verified_previously_accepted_same_market"] is True
    assert market_summaries[0]["skip_reason_counts"]["duplicate_for_same_market"] == 1


def test_backfill_cursor_advances_when_newer_valid_trade_exists() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = _market("m_cursor", start, close, "tok_yes", "tok_no")

    state = RecorderState()
    state.upsert_markets([market])

    newer = TradeRecord(
        timestamp=start + timedelta(seconds=20),
        market_id="m_cursor",
        trade_id="newer_trade",
        price=0.52,
        size=1.5,
        side="BUY",
        maker=None,
        taker=None,
    )
    clob = _FakeClobClient({"m_cursor": [newer]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "m_cursor": TradeBackfillCursor(last_timestamp=start, last_trade_id="")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    summary = app._last_trade_backfill_summary
    assert summary["accepted_trades"] == 1
    market_summary = summary["market_cycle_summaries"][0]
    assert market_summary["cursor_before_timestamp"] == start.isoformat()
    assert market_summary["cursor_after_timestamp"] == newer.timestamp.isoformat()
    assert market_summary["cursor_after_trade_id"] == "newer_trade"


def test_backfill_accepts_trade_exactly_at_market_start_boundary() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = _market("m_boundary", start, close, "tok_yes", "tok_no")

    state = RecorderState()
    state.upsert_markets([market])

    boundary_trade = TradeRecord(
        timestamp=start,
        market_id="m_boundary",
        trade_id="boundary_trade",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    clob = _FakeClobClient({"m_boundary": [boundary_trade]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "m_boundary": TradeBackfillCursor(last_timestamp=start, last_trade_id="")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    summary = app._last_trade_backfill_summary
    assert summary["accepted_trades"] == 1
    assert summary["rejected_before_start"] == 0
    assert summary["skipped_before_cursor"] == 0


def test_backfill_cursor_advances_across_same_timestamp_duplicates() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    market = _market("m_cursor_same_ts", start, close, "tok_yes", "tok_no")

    state = RecorderState()
    state.upsert_markets([market])

    ts = start + timedelta(seconds=20)
    t1 = TradeRecord(
        timestamp=ts,
        market_id="m_cursor_same_ts",
        trade_id="dup_a",
        price=0.5,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    t2 = TradeRecord(
        timestamp=ts,
        market_id="m_cursor_same_ts",
        trade_id="dup_b",
        price=0.51,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )
    t3 = TradeRecord(
        timestamp=ts,
        market_id="m_cursor_same_ts",
        trade_id="dup_c",
        price=0.52,
        size=1.0,
        side="BUY",
        maker=None,
        taker=None,
    )

    # Seed seen ids for this market so backfill sees all fetched trades as duplicates.
    assert state.apply_trade(t1) is True
    assert state.apply_trade(t2) is True
    assert state.apply_trade(t3) is True

    clob = _FakeClobClient({"m_cursor_same_ts": [t1, t2, t3]})
    app = _build_app_stub(state, clob)
    app._backfill_cursor = {
        "m_cursor_same_ts": TradeBackfillCursor(last_timestamp=ts, last_trade_id="dup_a")
    }

    asyncio.run(app._run_trade_backfill_cycle())

    first_summary = app._last_trade_backfill_summary
    first_market = first_summary["market_cycle_summaries"][0]
    assert first_summary["accepted_trades"] == 0
    assert first_summary["skipped_before_cursor"] == 1
    assert first_summary["skipped_already_seen"] == 2
    assert first_market["cursor_after_trade_id"] == "dup_c"

    asyncio.run(app._run_trade_backfill_cycle())

    second_summary = app._last_trade_backfill_summary
    second_market = second_summary["market_cycle_summaries"][0]
    assert second_summary["accepted_trades"] == 0
    assert second_summary["skipped_before_cursor"] == 3
    assert second_summary["skipped_already_seen"] == 0
    assert second_market["cursor_after_trade_id"] == "dup_c"
