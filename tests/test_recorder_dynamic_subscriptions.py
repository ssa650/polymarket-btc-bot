from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.models import MarketMetadata
from src.recorder import RecorderApp
from src.state import RecorderState


class _FakeGamma:
    def __init__(self, markets: list[MarketMetadata]) -> None:
        self.markets = markets
        self.last_discovery_stats = {
            "total_markets_from_api": len(markets),
            "selected_btc_5m_markets": len(markets),
            "broad_btc_candidates": len(markets),
            "fallback_used": 0,
        }

    async def fetch_active_markets(self) -> list[MarketMetadata]:
        return list(self.markets)


class _FakeDB:
    def has_market_event(self, *args, **kwargs) -> bool:
        return False

    def upsert_markets(self, markets, run_id=None) -> int:
        return len(markets)

    def refresh_market_phases(self, now=None, run_id=None) -> int:
        return 0

    def demote_stale_primary_markets(self, current_primary_market_ids, run_id=None) -> int:
        return 0

    def update_market_statuses(self, updates, run_id=None) -> int:
        return len(updates)

    def update_market_tracking_states(self, updates, run_id=None) -> int:
        return len(updates)

    def mark_non_target_open_markets_inactive(self, active_target_market_ids, run_id=None) -> int:
        return 0

    def insert_market_events(self, events):
        return len(events), 0


class _FakeWS:
    def __init__(
        self,
        *,
        fail_subscribe: bool = False,
        fail_unsubscribe: bool = False,
    ) -> None:
        self.subscribe_calls: list[list[str]] = []
        self.unsubscribe_calls: list[list[str]] = []
        self.reconnect_requests: list[str] = []
        self.fail_subscribe = fail_subscribe
        self.fail_unsubscribe = fail_unsubscribe

    async def subscribe_tokens(self, token_ids: list[str]) -> None:
        self.subscribe_calls.append(list(token_ids))
        if self.fail_subscribe:
            raise RuntimeError("offline subscribe failure")

    async def unsubscribe_tokens(self, token_ids: list[str]) -> None:
        self.unsubscribe_calls.append(list(token_ids))
        if self.fail_unsubscribe:
            raise RuntimeError("offline unsubscribe failure")

    async def request_reconnect(self, reason: str) -> None:
        self.reconnect_requests.append(reason)


def _market(
    market_id: str,
    start: datetime,
    close: datetime,
    yes_token: str,
    no_token: str,
    *,
    status: str = "open",
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
        status=status,
        platform_status=status,
        yes_token_id=yes_token,
        no_token_id=no_token,
        token_ids={"YES": yes_token, "NO": no_token},
        run_id="run_dynamic_subs",
    )


def _app(markets: list[MarketMetadata], ws: _FakeWS | None = None) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.settings = SimpleNamespace(
        tracked_markets_limit=0,
        market_unsubscribe_grace_sec=60,
    )
    app.gamma = _FakeGamma(markets)
    app.db = _FakeDB()
    app.ws = ws or _FakeWS()
    app.state = RecorderState()
    app._state_lock = asyncio.Lock()
    app._backfill_cursor = {}
    app._active_primary_market_id = None
    app._desired_ws_asset_ids = set()
    app._stale_ws_asset_deadlines = {}
    app._last_api_latency_ms = None
    app._last_successful_market_fetches = 0
    app._last_failed_markets = 0
    app._last_discovery_candidates_seen = 0
    app._last_discovery_candidates_matched = 0
    app._last_discovery_strict_5m_candidates = 0
    app._last_discovery_broad_btc_candidates = 0
    app._last_discovery_fallback_used = 0
    app._api_call_count_since_snapshot = 0
    app._api_success_count_since_snapshot = 0
    app._api_failure_count_since_snapshot = 0
    app.run_id = "run_dynamic_subs"
    app.log = logging.getLogger("test.recorder.dynamic_subscriptions")
    return app


def test_newly_discovered_current_and_upcoming_markets_trigger_subscribe() -> None:
    now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    active = _market(
        "active",
        now - timedelta(minutes=1),
        now + timedelta(minutes=4),
        "active_yes",
        "active_no",
    )
    upcoming = _market(
        "upcoming",
        now + timedelta(minutes=4),
        now + timedelta(minutes=9),
        "upcoming_yes",
        "upcoming_no",
    )
    app = _app([active, upcoming])

    with patch("src.recorder.utc_now", return_value=now):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert app.ws.subscribe_calls == [
        ["active_no", "active_yes", "upcoming_no", "upcoming_yes"]
    ]
    assert app.ws.unsubscribe_calls == []
    assert app._get_token_ids_for_ws() == [
        "active_no",
        "active_yes",
        "upcoming_no",
        "upcoming_yes",
    ]


def test_expired_market_does_not_unsubscribe_before_grace_period() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = start + timedelta(minutes=5)
    active_now = start + timedelta(minutes=1)
    expired_before_grace = close + timedelta(seconds=30)
    market = _market("m1", start, close, "old_yes", "old_no")
    app = _app([market])

    with patch("src.recorder.utc_now", return_value=active_now):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))
    app.gamma.markets = [market]

    with patch("src.recorder.utc_now", return_value=expired_before_grace):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert app.ws.unsubscribe_calls == []
    assert app._get_token_ids_for_ws() == ["old_no", "old_yes"]


def test_expired_market_unsubscribes_after_grace_period() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = start + timedelta(minutes=5)
    active_now = start + timedelta(minutes=1)
    expired_after_grace = close + timedelta(seconds=61)
    market = _market("m1", start, close, "old_yes", "old_no")
    app = _app([market])

    with patch("src.recorder.utc_now", return_value=active_now):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))
    app.gamma.markets = [market]

    with patch("src.recorder.utc_now", return_value=expired_after_grace):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert app.ws.unsubscribe_calls == [["old_no", "old_yes"]]
    assert app._get_token_ids_for_ws() == []


def test_reconnect_uses_current_rotated_desired_asset_set() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    old_market = _market(
        "old",
        start,
        start + timedelta(minutes=5),
        "old_yes",
        "old_no",
    )
    new_market = _market(
        "new",
        start + timedelta(minutes=5),
        start + timedelta(minutes=10),
        "new_yes",
        "new_no",
    )
    app = _app([old_market])

    with patch("src.recorder.utc_now", return_value=start + timedelta(minutes=1)):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    app.gamma.markets = [old_market, new_market]
    with patch("src.recorder.utc_now", return_value=start + timedelta(minutes=6, seconds=1)):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert app.ws.unsubscribe_calls == [["old_no", "old_yes"]]
    assert app._get_token_ids_for_ws() == ["new_no", "new_yes"]


def test_subscription_failures_are_logged_and_non_fatal(caplog) -> None:
    now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    market = _market(
        "m1",
        now - timedelta(minutes=1),
        now + timedelta(minutes=4),
        "yes",
        "no",
    )
    ws = _FakeWS(fail_subscribe=True)
    app = _app([market], ws=ws)
    caplog.set_level(logging.ERROR)

    with patch("src.recorder.utc_now", return_value=now):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert ws.subscribe_calls == [["no", "yes"]]
    assert "ws_subscribe_tokens_failed" in [record.getMessage() for record in caplog.records]


def test_unsubscribe_failures_are_logged_and_non_fatal(caplog) -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = start + timedelta(minutes=5)
    market = _market("m1", start, close, "yes", "no")
    ws = _FakeWS(fail_unsubscribe=True)
    app = _app([market], ws=ws)
    caplog.set_level(logging.ERROR)

    with patch("src.recorder.utc_now", return_value=start + timedelta(minutes=1)):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))
    app.gamma.markets = [market]

    with patch("src.recorder.utc_now", return_value=close + timedelta(seconds=61)):
        asyncio.run(app._discover_and_update(trigger_orderbook_sync=False))

    assert ws.unsubscribe_calls == [["no", "yes"]]
    assert "ws_unsubscribe_tokens_failed" in [
        record.getMessage() for record in caplog.records
    ]
