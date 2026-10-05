from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.db import SQLiteStore
from src.models import MarketMetadata, OrderBookSnapshot, PriceLevel, RecorderMetricRecord
from src.recorder import RecorderApp
from src.state import RecorderState


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
        run_id="run_health",
    )


class _FakeClobClient:
    async def fetch_orderbook_snapshot(self, token_id: str) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            token_id=token_id,
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            bids=[PriceLevel(price=0.49, size=10.0)],
            asks=[PriceLevel(price=0.51, size=11.0)],
        )


def _make_app(tmp_path) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.run_id = "run_health"
    app.log = logging.getLogger("test.recorder.health")
    app.settings = SimpleNamespace(
        startup_orderbook_concurrency=2,
        raw_ws_events_enabled=True,
    )
    app.db = SQLiteStore(str(tmp_path / "recorder.db"), run_id=app.run_id)
    app.db.init_schema()
    app.state = RecorderState()
    market = _market()
    app.state.upsert_markets([market])
    app.db.upsert_markets([market], run_id=app.run_id)
    app._state_lock = asyncio.Lock()
    app.clob = _FakeClobClient()
    app.ws = SimpleNamespace(reconnect_count=0)
    app._api_call_count_since_snapshot = 0
    app._api_success_count_since_snapshot = 0
    app._api_failure_count_since_snapshot = 0
    app._raw_ws_events_seen = 0
    app._raw_ws_events_written = 0
    app._raw_ws_malformed_events = 0
    app._raw_ws_write_failures = 0
    app._raw_ws_events_skipped_logged = False
    app._ws_reconnect_count = 0
    app._last_ws_event_monotonic = None
    return app


def _metric_from_health(app: RecorderApp, health: dict[str, object]) -> RecorderMetricRecord:
    return RecorderMetricRecord(
        timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        markets_polled=0,
        successful_market_fetches=0,
        failed_markets=0,
        api_latency_ms=None,
        db_write_time_ms=0.0,
        cycle_duration_ms=0.0,
        rows_inserted=0,
        duplicate_rows_skipped=0,
        ws_reconnect_count=int(health["ws_reconnect_count"]),
        raw_ws_events_seen=int(health["raw_ws_events_seen"]),
        raw_ws_events_written=int(health["raw_ws_events_written"]),
        malformed_ws_events=int(health["malformed_ws_events"]),
        raw_ws_write_failures=int(health["raw_ws_write_failures"]),
        last_ws_event_age_sec=health["last_ws_event_age_sec"],
        subscribed_asset_count=int(health["subscribed_asset_count"]),
        subscribed_asset_ids_json=str(health["subscribed_asset_ids_json"]),
        run_id=app.run_id,
    )


def test_health_metrics_capture_valid_malformed_ws_and_reconnect(tmp_path) -> None:
    app = _make_app(tmp_path)
    valid_message = json.dumps(
        {
            "event_type": "book",
            "token_id": "tok_yes",
            "bids": [[0.49, 10]],
            "asks": [[0.51, 11]],
            "timestamp": "2026-01-01T00:00:01Z",
        }
    )

    try:
        with patch("src.recorder.time.time_ns", return_value=1_775_286_600_000_000_000), patch(
            "src.recorder.time.monotonic",
            return_value=100.0,
        ):
            asyncio.run(app._handle_ws_message(valid_message))

        with patch("src.recorder.time.time_ns", return_value=1_775_286_601_000_000_000), patch(
            "src.recorder.time.monotonic",
            return_value=130.0,
        ):
            asyncio.run(app._handle_ws_message("{not-json"))

        asyncio.run(app._recover_from_ws_disconnect())

        health = app._recorder_health_fields(
            subscribed_asset_ids=app.state.all_token_ids(),
            now_mono=142.5,
        )
        app.db.insert_recorder_metric(_metric_from_health(app, health))

        row = app.db.conn.execute(
            """
            SELECT ws_reconnect_count, raw_ws_events_seen, raw_ws_events_written,
                   malformed_ws_events, raw_ws_write_failures, last_ws_event_age_sec,
                   subscribed_asset_count, subscribed_asset_ids_json
            FROM recorder_metrics
            WHERE run_id = ?
            """,
            (app.run_id,),
        ).fetchone()

        assert row == (
            1,
            2,
            2,
            1,
            0,
            12.5,
            2,
            '["tok_no","tok_yes"]',
        )
    finally:
        app.db.close()
