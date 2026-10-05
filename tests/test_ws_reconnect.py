from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.models import MarketMetadata, OrderBookSnapshot, PriceLevel
from src.polymarket.ws_client import WSClient
from src.recorder import RecorderApp
from src.state import RecorderState


def _expected_initial_payload(token_ids: list[str]) -> dict[str, object]:
    return {
        "assets_ids": token_ids,
        "type": "market",
        "custom_feature_enabled": True,
    }


class _FakeWebSocket:
    def __init__(self, stop_event: asyncio.Event, *, stop_on_recv: bool = False) -> None:
        self.stop_event = stop_event
        self.stop_on_recv = stop_on_recv
        self.sent_payloads: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.sent_payloads.append(json.loads(payload))

    async def recv(self) -> str:
        if self.stop_on_recv:
            self.stop_event.set()
        raise ConnectionError("offline test disconnect")


class _FakeWebSocketContext:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self) -> _FakeWebSocket:
        return self.ws

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _ConnectFactory:
    def __init__(self, sockets: list[_FakeWebSocket]) -> None:
        self.sockets = sockets
        self.calls = 0

    def __call__(self, *args, **kwargs) -> _FakeWebSocketContext:
        ws = self.sockets[self.calls]
        self.calls += 1
        return _FakeWebSocketContext(ws)


def test_ws_client_resubscribes_same_assets_after_reconnect() -> None:
    async def _run() -> tuple[WSClient, _FakeWebSocket, _FakeWebSocket, list[int]]:
        stop_event = asyncio.Event()
        first_ws = _FakeWebSocket(stop_event)
        second_ws = _FakeWebSocket(stop_event, stop_on_recv=True)
        connect = _ConnectFactory([first_ws, second_ws])
        reconnect_calls: list[int] = []

        client = WSClient(
            url="wss://example.invalid/ws",
            ping_interval_sec=1.0,
            ping_timeout_sec=1.0,
            reconnect_min_sec=0.0,
            reconnect_max_sec=0.0,
        )

        async def on_message(_raw: str | bytes) -> None:
            raise AssertionError("fake sockets disconnect before delivering messages")

        async def on_reconnect() -> None:
            reconnect_calls.append(client.reconnect_count)

        with patch("src.polymarket.ws_client.websockets.connect", connect), patch(
            "src.polymarket.ws_client.random.uniform",
            return_value=0.0,
        ):
            await client.run_forever(
                get_token_ids=lambda: ["tok_b", "tok_a"],
                on_message=on_message,
                stop_event=stop_event,
                on_reconnect=on_reconnect,
            )

        return client, first_ws, second_ws, reconnect_calls

    client, first_ws, second_ws, reconnect_calls = asyncio.run(_run())

    assert client.reconnect_count == 1
    assert reconnect_calls == [1]
    assert first_ws.sent_payloads == [_expected_initial_payload(["tok_a", "tok_b"])]
    assert second_ws.sent_payloads == [_expected_initial_payload(["tok_a", "tok_b"])]


def test_ws_client_initial_subscription_payload_enables_custom_features() -> None:
    async def _run() -> _FakeWebSocket:
        stop_event = asyncio.Event()
        ws = _FakeWebSocket(stop_event, stop_on_recv=True)
        connect = _ConnectFactory([ws])
        client = WSClient(
            url="wss://example.invalid/ws",
            ping_interval_sec=1.0,
            ping_timeout_sec=1.0,
            reconnect_min_sec=0.0,
            reconnect_max_sec=0.0,
        )

        async def on_message(_raw: str | bytes) -> None:
            raise AssertionError("fake socket disconnects before delivering messages")

        with patch("src.polymarket.ws_client.websockets.connect", connect), patch(
            "src.polymarket.ws_client.random.uniform",
            return_value=0.0,
        ):
            await client.run_forever(
                get_token_ids=lambda: ["tok_yes", "tok_no"],
                on_message=on_message,
                stop_event=stop_event,
            )
        return ws

    ws = asyncio.run(_run())

    assert ws.sent_payloads == [_expected_initial_payload(["tok_no", "tok_yes"])]


def test_ws_client_dynamic_subscribe_payload_enables_custom_features() -> None:
    async def _run() -> tuple[WSClient, _FakeWebSocket]:
        stop_event = asyncio.Event()
        ws = _FakeWebSocket(stop_event)
        client = WSClient(
            url="wss://example.invalid/ws",
            ping_interval_sec=1.0,
            ping_timeout_sec=1.0,
            reconnect_min_sec=0.0,
            reconnect_max_sec=0.0,
        )
        async with client._sub_lock:
            client._ws = ws
            client._subscribed_tokens = {"tok_existing"}

        await client.subscribe_tokens(["tok_new", "tok_existing"])
        return client, ws

    client, ws = asyncio.run(_run())

    assert ws.sent_payloads == [
        {
            "assets_ids": ["tok_new"],
            "operation": "subscribe",
            "custom_feature_enabled": True,
        }
    ]
    assert client._subscribed_tokens == {"tok_existing", "tok_new"}


def test_ws_client_unsubscribe_payload_support() -> None:
    async def _run() -> tuple[WSClient, _FakeWebSocket]:
        stop_event = asyncio.Event()
        ws = _FakeWebSocket(stop_event)
        client = WSClient(
            url="wss://example.invalid/ws",
            ping_interval_sec=1.0,
            ping_timeout_sec=1.0,
            reconnect_min_sec=0.0,
            reconnect_max_sec=0.0,
        )
        async with client._sub_lock:
            client._ws = ws
            client._subscribed_tokens = {"tok_a", "tok_b"}
            client._pending_tokens = {"tok_pending"}

        await client.unsubscribe_tokens(["tok_b", "tok_pending", "tok_missing"])
        return client, ws

    client, ws = asyncio.run(_run())

    assert ws.sent_payloads == [
        {
            "assets_ids": ["tok_b"],
            "operation": "unsubscribe",
        }
    ]
    assert client._subscribed_tokens == {"tok_a"}
    assert client._pending_tokens == set()


class _FakeClobClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch_orderbook_snapshot(self, token_id: str) -> OrderBookSnapshot | None:
        self.calls.append(token_id)
        if token_id == "tok_no":
            raise RuntimeError("offline book refresh failure")
        return OrderBookSnapshot(
            token_id=token_id,
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            bids=[PriceLevel(price=0.49, size=10.0)],
            asks=[PriceLevel(price=0.51, size=11.0)],
        )


def _market() -> MarketMetadata:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    return MarketMetadata(
        market_id="market-1",
        event_id="event-market-1",
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
        market_phase="active",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        token_ids={"YES": "tok_yes", "NO": "tok_no"},
        run_id="run_reconnect",
    )


def test_recorder_reconnect_recovery_refreshes_orderbooks_and_survives_failures(
    caplog,
) -> None:
    app = RecorderApp.__new__(RecorderApp)
    app.settings = SimpleNamespace(startup_orderbook_concurrency=2)
    app.log = logging.getLogger("test.recorder.reconnect")
    app.state = RecorderState()
    app.state.upsert_markets([_market()])
    app._state_lock = asyncio.Lock()
    app.clob = _FakeClobClient()
    app._api_call_count_since_snapshot = 0
    app._api_success_count_since_snapshot = 0
    app._api_failure_count_since_snapshot = 0
    app._ws_reconnect_count = 0

    caplog.set_level(logging.INFO)

    asyncio.run(app._recover_from_ws_disconnect())

    assert app._ws_reconnect_count == 1
    assert set(app.clob.calls) == {"tok_yes", "tok_no"}
    assert app._api_call_count_since_snapshot == 2
    assert app._api_success_count_since_snapshot == 1
    assert app._api_failure_count_since_snapshot == 1

    runtime = app.state.markets["market-1"]
    assert runtime.yes_bids == [PriceLevel(price=0.49, size=10.0)]
    assert runtime.yes_asks == [PriceLevel(price=0.51, size=11.0)]
    assert runtime.no_bids == []
    assert runtime.no_asks == []

    messages = [record.getMessage() for record in caplog.records]
    assert "orderbook_sync_failed" in messages
    assert "ws_reconnect_recovery_completed" in messages
