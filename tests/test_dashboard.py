from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.dashboard import connect_read_only, fetch_dashboard_data, render_dashboard
from src.db import SQLiteStore
from src.models import to_iso


RUN_ID = "run_dashboard"
MARKET_ID = "market_dash"
YES = "yes_asset"
NO = "no_asset"


def _init_db(db_path) -> SQLiteStore:
    store = SQLiteStore(str(db_path), run_id=RUN_ID, quarantine_on_startup=False)
    store.init_schema()
    return store


def _insert_active_market(conn: sqlite3.Connection, *, now: datetime) -> None:
    start = now - timedelta(minutes=1)
    close = now + timedelta(minutes=4)
    conn.execute(
        """
        INSERT INTO markets (
            run_id, market_id, question, start_time, end_time, close_time,
            platform_status, status, phase, market_phase, tracking_state,
            yes_token_id, no_token_id, created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            MARKET_ID,
            "Bitcoin Up or Down - dashboard",
            to_iso(start),
            to_iso(close),
            to_iso(close),
            "open",
            "open",
            "active",
            "active",
            "selected_active",
            YES,
            NO,
            to_iso(start),
            to_iso(now),
        ),
    )


def _insert_snapshot(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    conn.execute(
        """
        INSERT INTO market_snapshots (
            run_id, timestamp, market_id,
            best_bid_yes, best_ask_yes, spread_yes,
            best_bid_no, best_ask_no, spread_no,
            mid_price_yes, mid_price_no,
            last_trade_price, last_trade_size, last_trade_time,
            has_orderbook, has_trade_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            MARKET_ID,
            0.49,
            0.51,
            0.02,
            0.48,
            0.52,
            0.04,
            0.5,
            0.5,
            0.5,
            2.0,
            to_iso(timestamp),
            1,
            1,
        ),
    )


def _insert_btc(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    conn.execute(
        """
        INSERT INTO btc_prices (
            run_id, source, price, exchange_timestamp,
            local_arrival_ns, local_arrival_iso, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            "polymarket_rtds_chainlink",
            50000.0,
            to_iso(timestamp - timedelta(seconds=1)),
            1_767_225_660_000_000_000,
            to_iso(timestamp),
            "{}",
        ),
    )


def _insert_best_bid_ask(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    for asset_id, bid, ask in ((YES, 0.49, 0.51), (NO, 0.48, 0.52)):
        conn.execute(
            """
            INSERT INTO best_bid_ask_updates (
                run_id, timestamp, market_id, condition_id, asset_id,
                best_bid, best_ask, spread, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                RUN_ID,
                to_iso(timestamp),
                MARKET_ID,
                "condition_dash",
                asset_id,
                bid,
                ask,
                ask - bid,
                "{}",
            ),
        )


def _insert_trade(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    conn.execute(
        """
        INSERT INTO trades (
            run_id, timestamp, market_id, trade_id, price, size, side
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            MARKET_ID,
            f"trade_hash:{YES}",
            0.5,
            3.0,
            "BUY",
        ),
    )


def _insert_metric(
    conn: sqlite3.Connection,
    *,
    timestamp: datetime,
    last_ws_event_age_sec: float = 0.5,
    malformed_ws_events: int = 0,
    raw_ws_write_failures: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO recorder_metrics (
            run_id, timestamp, markets_polled, successful_market_fetches,
            failed_markets, rows_inserted, duplicate_rows_skipped,
            raw_ws_events_seen, raw_ws_events_written, malformed_ws_events,
            raw_ws_write_failures, last_ws_event_age_sec,
            subscribed_asset_count, ws_reconnect_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            1,
            1,
            0,
            10,
            0,
            100,
            100,
            malformed_ws_events,
            raw_ws_write_failures,
            last_ws_event_age_sec,
            2,
            1,
        ),
    )


def _insert_raw_events(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    for idx, event_type in enumerate(
        ("price_change", "best_bid_ask", "book", "last_trade_price")
    ):
        conn.execute(
            """
            INSERT INTO raw_polymarket_events (
                run_id, local_arrival_ns, local_arrival_iso,
                event_type, parse_status, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                RUN_ID,
                1_767_225_660_000_000_000 + idx,
                to_iso(timestamp),
                event_type,
                "ok",
                "{}",
            ),
        )


def _seed_live_db(db_path, *, now: datetime) -> None:
    store = _init_db(db_path)
    try:
        with store.conn:
            _insert_active_market(store.conn, now=now)
            _insert_snapshot(store.conn, timestamp=now - timedelta(seconds=1))
            _insert_btc(store.conn, timestamp=now - timedelta(seconds=1))
            _insert_best_bid_ask(store.conn, timestamp=now - timedelta(seconds=1))
            _insert_trade(store.conn, timestamp=now - timedelta(seconds=1))
            _insert_metric(store.conn, timestamp=now - timedelta(seconds=1))
            _insert_raw_events(store.conn, timestamp=now - timedelta(seconds=1))
    finally:
        store.close()


def _table_counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "markets",
                "market_snapshots",
                "btc_prices",
                "best_bid_ask_updates",
                "trades",
                "recorder_metrics",
                "raw_polymarket_events",
            )
        }
    finally:
        conn.close()


def test_dashboard_reads_active_market_live_sections_without_writing(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    before = _table_counts(db_path)

    data = fetch_dashboard_data(str(db_path), now=now)
    output = render_dashboard(data)

    assert _table_counts(db_path) == before
    assert data.active_market is not None
    assert data.active_market["market_id"] == MARKET_ID
    assert data.btc_price is not None
    assert data.latest_snapshot is not None
    assert data.best_bid_ask["YES"] is not None
    assert data.best_bid_ask["NO"] is not None
    assert data.recent_trades[0]["asset_id"] == YES
    assert data.recent_trades[0]["outcome"] == "YES"
    assert data.event_counts["price_change"] == 1
    assert "market_id=market_dash" in output
    assert "price=50000.0" in output
    assert "best_bid_yes=0.49" in output
    assert "raw_ws_events_seen=100" in output
    assert "warning=" not in output


def test_dashboard_connection_is_sqlite_read_only(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    store.close()

    conn = connect_read_only(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                """
                INSERT INTO markets (run_id, market_id, created_at, last_updated)
                VALUES ('x', 'y', 'now', 'now')
                """
            )
    finally:
        conn.close()


def test_dashboard_handles_no_active_market_and_missing_btc(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    store.close()

    data = fetch_dashboard_data(str(db_path), now=now)
    output = render_dashboard(data)

    assert data.active_market is None
    assert data.btc_price is None
    assert "=== Current Selected Active Market ===\nnone" in output
    assert "warning=no_selected_active_market" in output
    assert "warning=missing_btc_price" in output
    assert "warning=no_recent_snapshots" in output


def test_dashboard_reports_stale_metrics_and_error_warnings(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    try:
        with store.conn:
            _insert_active_market(store.conn, now=now)
            _insert_snapshot(store.conn, timestamp=now - timedelta(seconds=30))
            _insert_best_bid_ask(store.conn, timestamp=now - timedelta(seconds=30))
            _insert_metric(
                store.conn,
                timestamp=now - timedelta(seconds=30),
                last_ws_event_age_sec=9.0,
                malformed_ws_events=1,
                raw_ws_write_failures=1,
            )
    finally:
        store.close()

    data = fetch_dashboard_data(str(db_path), now=now)

    assert "missing_btc_price" in data.warnings
    assert "no_recent_snapshots" in data.warnings
    assert "stale_recorder_metrics" in data.warnings
    assert "stale_ws_events" in data.warnings
    assert "malformed_ws_events_gt_zero" in data.warnings
    assert "raw_ws_write_failures_gt_zero" in data.warnings


def test_dashboard_cli_renders_once_without_network(monkeypatch, tmp_path, capsys) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)

    import src.main as main_module

    settings = SimpleNamespace(
        db_path=str(db_path),
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dashboard must not start network or recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "dashboard", "--dashboard-once"],
    )
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    out = capsys.readouterr().out

    assert "=== Polymarket BTC 5m Recorder Dashboard ===" in out
    assert "market_id=market_dash" in out
    assert "=== Recorder Health ===" in out
