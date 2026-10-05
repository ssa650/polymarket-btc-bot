from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.db import SQLiteStore
from src.models import MarketMetadata, local_arrival_iso_from_ns
from src.recorder import RecorderApp
from src.state import RecorderState


LOCAL_ARRIVAL_NS = 1_775_286_600_987_654_321
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "polymarket_ws_events.jsonl"


def _market(
    market_id: str = "market-1",
    yes_token: str = "tok_yes",
    no_token: str = "tok_no",
    *,
    start: datetime | None = None,
    close: datetime | None = None,
) -> MarketMetadata:
    start = start or datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = close or datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event-{market_id}",
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
        yes_token_id=yes_token,
        no_token_id=no_token,
        condition_id=market_id,
        token_ids={"YES": yes_token, "NO": no_token},
        run_id="run_ws_raw",
    )


def _make_app(
    tmp_path: Path,
    markets: list[MarketMetadata] | None = None,
    *,
    raw_ws_events_enabled: bool = True,
) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.run_id = "run_ws_raw"
    app.log = logging.getLogger("test.recorder.raw_ws_capture")
    app.settings = SimpleNamespace(raw_ws_events_enabled=raw_ws_events_enabled)
    app.db = SQLiteStore(str(tmp_path / "recorder.db"), run_id=app.run_id)
    app.db.init_schema()
    app.state = RecorderState()
    tracked_markets = markets or [_market()]
    app.state.upsert_markets(tracked_markets)
    app.db.upsert_markets(tracked_markets, run_id=app.run_id)
    app._state_lock = asyncio.Lock()
    app._raw_ws_events_seen = 0
    app._raw_ws_events_written = 0
    app._raw_ws_malformed_events = 0
    app._raw_ws_write_failures = 0
    app._raw_ws_events_skipped_logged = False
    return app


def test_handle_ws_message_writes_raw_events_and_preserves_normalized_state(
    tmp_path,
) -> None:
    app = _make_app(tmp_path)
    raw_message = json.dumps(
        {
            "data": [
                {
                    "event_type": "book",
                    "token_id": "tok_yes",
                    "bids": [[0.51, 10]],
                    "asks": [[0.53, 11]],
                    "timestamp": "2026-01-01T00:00:01Z",
                },
                {
                    "event_type": "price_change",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "price_changes": [
                        {
                            "asset_id": "tok_yes",
                            "price": "0.52",
                            "size": "9",
                            "side": "BUY",
                        }
                    ],
                },
                {
                    "event_type": "trade",
                    "token_id": "tok_yes",
                    "trade_id": "trade-1",
                    "price": "0.52",
                    "size": "7",
                    "side": "buy",
                    "timestamp": "2026-01-01T00:00:03Z",
                },
            ]
        }
    )

    try:
        with patch("src.recorder.time.time_ns", return_value=LOCAL_ARRIVAL_NS):
            asyncio.run(app._handle_ws_message(raw_message))

        rows = app.db.conn.execute(
            """
            SELECT local_arrival_ns, local_arrival_iso, event_type, asset_id,
                   parse_status, parse_error, raw_json
            FROM raw_polymarket_events
            WHERE run_id = ?
            ORDER BY id
            """,
            (app.run_id,),
        ).fetchall()
        assert len(rows) == 3
        assert [row[2] for row in rows] == ["book", "price_change", "trade"]
        assert all(row[0] == LOCAL_ARRIVAL_NS for row in rows)
        assert all(row[1] == local_arrival_iso_from_ns(LOCAL_ARRIVAL_NS) for row in rows)
        assert [row[3] for row in rows] == ["tok_yes", None, "tok_yes"]
        assert all(row[4] == "ok" for row in rows)
        assert all(row[5] is None for row in rows)
        assert json.loads(rows[0][6])["event_type"] == "book"

        runtime = app.state.markets["market-1"]
        assert [level.price for level in runtime.yes_bids] == [0.52, 0.51]
        assert [level.size for level in runtime.yes_bids] == [9.0, 10.0]
        assert [level.price for level in runtime.yes_asks] == [0.53]
        assert runtime.last_trade is not None
        assert runtime.last_trade.trade_id == "trade-1:tok_yes"
        assert runtime.last_trade.price == 0.52
        assert app._raw_ws_events_seen == 3
        assert app._raw_ws_events_written == 3
        assert app._raw_ws_malformed_events == 0
        assert app._raw_ws_write_failures == 0
    finally:
        app.db.close()


def test_handle_ws_message_skips_raw_events_when_disabled_but_keeps_normalized_state(
    tmp_path,
) -> None:
    app = _make_app(tmp_path, raw_ws_events_enabled=False)
    raw_message = json.dumps(
        {
            "data": [
                {
                    "event_type": "book",
                    "token_id": "tok_yes",
                    "bids": [[0.51, 10]],
                    "asks": [[0.53, 11]],
                    "timestamp": "2026-01-01T00:00:01Z",
                },
                {
                    "event_type": "trade",
                    "token_id": "tok_yes",
                    "trade_id": "trade-1",
                    "price": "0.52",
                    "size": "7",
                    "side": "buy",
                    "timestamp": "2026-01-01T00:00:03Z",
                },
            ]
        }
    )

    try:
        with patch("src.recorder.time.time_ns", return_value=LOCAL_ARRIVAL_NS):
            asyncio.run(app._handle_ws_message(raw_message))

        raw_count = app.db.conn.execute(
            "SELECT COUNT(*) FROM raw_polymarket_events WHERE run_id = ?",
            (app.run_id,),
        ).fetchone()[0]
        assert raw_count == 0

        runtime = app.state.markets["market-1"]
        assert [level.price for level in runtime.yes_bids] == [0.51]
        assert [level.price for level in runtime.yes_asks] == [0.53]
        assert runtime.last_trade is not None
        assert runtime.last_trade.trade_id == "trade-1:tok_yes"
        assert app._raw_ws_events_seen == 1
        assert app._raw_ws_events_written == 0
        assert app._raw_ws_malformed_events == 0
        assert app._raw_ws_write_failures == 0
    finally:
        app.db.close()


def test_handle_ws_message_normalizes_remaining_official_event_types(tmp_path) -> None:
    fixture_market = _market(
        market_id="0xcondition",
        yes_token="asset_yes",
        no_token="asset_no",
        start=datetime(2025, 9, 15, 4, 0, tzinfo=timezone.utc),
        close=datetime(2025, 9, 15, 4, 5, tzinfo=timezone.utc),
    )
    app = _make_app(tmp_path, markets=[fixture_market])
    fixture_events = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_message = json.dumps({"data": fixture_events})

    try:
        with patch("src.recorder.time.time_ns", return_value=LOCAL_ARRIVAL_NS):
            asyncio.run(app._handle_ws_message(raw_message))

        raw_count = app.db.conn.execute(
            "SELECT COUNT(*) FROM raw_polymarket_events WHERE run_id = ?",
            (app.run_id,),
        ).fetchone()[0]
        assert raw_count == 7

        runtime = app.state.markets["0xcondition"]
        assert [level.price for level in runtime.yes_bids] == [0.5, 0.48]
        assert [level.size for level in runtime.yes_bids] == [200.0, 30.0]
        assert [level.price for level in runtime.yes_asks] == [0.52]
        assert runtime.last_trade is not None
        assert runtime.last_trade.trade_id == "0xtradehash:asset_yes"
        assert runtime.last_trade.price == 0.456
        assert runtime.last_trade.size == 219.217767

        trade_row = app.db.conn.execute(
            """
            SELECT market_id, trade_id, price, size, side, timestamp
            FROM trades
            WHERE run_id = ? AND market_id = '0xcondition'
            """,
            (app.run_id,),
        ).fetchone()
        assert trade_row == (
            "0xcondition",
            "0xtradehash:asset_yes",
            0.456,
            219.217767,
            "BUY",
            "2025-09-15T04:01:32.353000+00:00",
        )

        tick_row = app.db.conn.execute(
            """
            SELECT market_id, condition_id, asset_id, old_tick_size, new_tick_size
            FROM tick_size_changes
            WHERE run_id = ?
            """,
            (app.run_id,),
        ).fetchone()
        assert tick_row == ("0xcondition", "0xcondition", "asset_yes", 0.01, 0.001)

        quote_row = app.db.conn.execute(
            """
            SELECT market_id, condition_id, asset_id, best_bid, best_ask, spread
            FROM best_bid_ask_updates
            WHERE run_id = ?
            """,
            (app.run_id,),
        ).fetchone()
        assert quote_row == ("0xcondition", "0xcondition", "asset_yes", 0.73, 0.77, 0.04)

        new_market_row = app.db.conn.execute(
            """
            SELECT market_id, event_id, question, yes_token_id, no_token_id,
                   condition_id, status, phase, tracking_state
            FROM markets
            WHERE run_id = ? AND market_id = '0xcondition_new'
            """,
            (app.run_id,),
        ).fetchone()
        assert new_market_row == (
            "0xcondition_new",
            "1031769",
            "Will Bitcoin be up or down?",
            "asset_new_yes",
            "asset_new_no",
            "0xcondition_new",
            "resolved",
            "resolved",
            "inactive",
        )

        event_rows = app.db.conn.execute(
            """
            SELECT market_id, event_type
            FROM market_events
            WHERE run_id = ? AND market_id = '0xcondition_new'
            ORDER BY event_type
            """,
            (app.run_id,),
        ).fetchall()
        assert event_rows == [
            ("0xcondition_new", "market_resolved"),
            ("0xcondition_new", "new_market"),
        ]
    finally:
        app.db.close()


def test_handle_ws_message_stores_malformed_payload_without_crashing(tmp_path) -> None:
    app = _make_app(tmp_path)

    try:
        with patch("src.recorder.time.time_ns", return_value=LOCAL_ARRIVAL_NS):
            asyncio.run(app._handle_ws_message("{not-json"))

        row = app.db.conn.execute(
            """
            SELECT local_arrival_ns, event_type, parse_status, parse_error, raw_json
            FROM raw_polymarket_events
            WHERE run_id = ?
            LIMIT 1
            """,
            (app.run_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == LOCAL_ARRIVAL_NS
        assert row[1] == "unknown"
        assert row[2] == "error"
        assert row[3]
        assert json.loads(row[4])["raw_message"] == "{not-json"
        assert app._raw_ws_events_seen == 1
        assert app._raw_ws_events_written == 1
        assert app._raw_ws_malformed_events == 1
        assert app._raw_ws_write_failures == 0
    finally:
        app.db.close()
