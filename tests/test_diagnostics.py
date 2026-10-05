from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from src.db import SQLiteStore
from src.diagnostics import print_diagnostics
from src.models import MarketMetadata, to_iso


def _seed_inspection_db(db_path) -> None:
    store = SQLiteStore(str(db_path), run_id="run_diag")
    try:
        store.init_schema()
        now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
        store.upsert_markets(
            [
                MarketMetadata(
                    market_id="market_diag",
                    event_id="event_diag",
                    question="Bitcoin Up or Down - diagnostic",
                    description=None,
                    category="crypto",
                    outcomes=["Yes", "No"],
                    resolution_source=None,
                    start_time=start,
                    end_time=close,
                    close_time=close,
                    platform_status="open",
                    phase="active",
                    tracking_state="selected_active",
                    status="open",
                    market_phase="active",
                    yes_token_id="yes_token",
                    no_token_id="no_token",
                    condition_id="condition_diag",
                    run_id="run_diag",
                    strict_validation_passed=1,
                )
            ],
            run_id="run_diag",
        )
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id, best_bid_yes, best_ask_yes,
                    spread_yes, last_trade_price, has_orderbook, has_trade_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("run_diag", to_iso(now), "market_diag", 0.49, 0.51, 0.02, 0.50, 1, 1),
            )
            store.conn.execute(
                """
                INSERT INTO trades (
                    run_id, timestamp, market_id, trade_id, price, size, side
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("run_diag", to_iso(now), "market_diag", "trade_diag", 0.50, 12.0, "BUY"),
            )
            store.conn.execute(
                """
                INSERT INTO market_events (
                    run_id, timestamp, market_id, event_type, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("run_diag", to_iso(now), "market_diag", "market_resolved", "{}"),
            )
            store.conn.execute(
                """
                INSERT INTO raw_polymarket_events (
                    run_id, local_arrival_ns, local_arrival_iso, exchange_timestamp,
                    event_type, market_id, condition_id, asset_id, slug,
                    parse_status, parse_error, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    1_767_225_660_000_000_000,
                    "2026-01-01T00:01:00+00:00",
                    "2026-01-01T00:01:00+00:00",
                    "book",
                    "market_diag",
                    "condition_diag",
                    "yes_token",
                    "btc-up-down",
                    "ok",
                    None,
                    '{"event_type":"book"}',
                ),
            )
            store.conn.execute(
                """
                INSERT INTO raw_polymarket_events (
                    run_id, local_arrival_ns, local_arrival_iso, exchange_timestamp,
                    event_type, market_id, condition_id, asset_id, slug,
                    parse_status, parse_error, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    1_767_225_661_000_000_000,
                    "2026-01-01T00:01:01+00:00",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "malformed",
                    "not json",
                    '"not-json"',
                ),
            )
            store.conn.execute(
                """
                INSERT INTO tick_size_changes (
                    run_id, timestamp, market_id, condition_id, asset_id,
                    old_tick_size, new_tick_size, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    to_iso(now),
                    "market_diag",
                    "condition_diag",
                    "yes_token",
                    0.001,
                    0.01,
                    '{"event_type":"tick_size_change"}',
                ),
            )
            store.conn.execute(
                """
                INSERT INTO best_bid_ask_updates (
                    run_id, timestamp, market_id, condition_id, asset_id,
                    best_bid, best_ask, spread, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    to_iso(now),
                    "market_diag",
                    "condition_diag",
                    "yes_token",
                    0.49,
                    0.51,
                    0.02,
                    '{"event_type":"best_bid_ask"}',
                ),
            )
            store.conn.execute(
                """
                INSERT INTO recorder_metrics (
                    run_id, timestamp, markets_polled, successful_market_fetches,
                    failed_markets, rows_inserted, duplicate_rows_skipped,
                    tracked_markets_count, snapshots_written, ws_reconnect_count,
                    raw_ws_events_seen, raw_ws_events_written, malformed_ws_events,
                    raw_ws_write_failures, last_ws_event_age_sec,
                    subscribed_asset_count, subscribed_asset_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    to_iso(now),
                    1,
                    1,
                    0,
                    8,
                    0,
                    1,
                    1,
                    2,
                    12,
                    11,
                    1,
                    0,
                    3.5,
                    2,
                    json.dumps(["no_token", "yes_token"]),
                ),
            )
            store.conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, exchange_timestamp,
                    local_arrival_ns, local_arrival_iso, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_diag",
                    "offline_fixture",
                    42123.45,
                    "2026-01-01T00:00:59+00:00",
                    1_767_225_662_000_000_000,
                    "2026-01-01T00:01:02+00:00",
                    '{"price":42123.45}',
                ),
            )
    finally:
        store.close()


def test_print_diagnostics_reports_recorder_health_sections(tmp_path, capsys) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_inspection_db(db_path)

    print_diagnostics(str(db_path))

    out = capsys.readouterr().out
    assert "=== Database Files ===" in out
    assert "db_file_size_bytes=" in out
    assert "wal_file_size_bytes=" in out
    assert "shm_file_size_bytes=" in out
    assert "=== Raw WS Persistence Config ===" in out
    assert "raw_ws_events_persistence_enabled=unknown" in out
    assert "=== Row Counts ===" in out
    assert "markets=1" in out
    assert "=== Recent Markets ===" in out
    assert "market_id=market_diag" in out
    assert "=== Last Tracked Assets ===" in out
    assert 'subscribed_asset_ids_json=["no_token", "yes_token"]' in out
    assert "=== Raw Polymarket Events ===" in out
    assert "event_type=book | parse_status=ok | count=1" in out
    assert "event_type=unknown | parse_status=malformed | count=1" in out
    assert "=== Latest Recorder Metrics ===" in out
    assert "ws_reconnect_count=2" in out
    assert "raw_ws_events_seen=12" in out
    assert "malformed_ws_events=1" in out
    assert "btc_prices=1" in out
    assert "=== Latest BTC Price ===" in out
    assert "source=offline_fixture" in out
    assert "price=42123.45" in out
    assert "=== Latest BTC Price By Source ===" in out
    assert "sample_age_sec=" in out
    assert "warning btc_price_source_stale source=offline_fixture" in out
    assert "=== Recent Trades ===" in out
    assert "trade_id=trade_diag" in out
    assert "=== Recent Snapshots ===" in out
    assert "last_trade_price=0.5" in out
    assert "=== Recent Tick Size Changes ===" in out
    assert "new_tick_size=0.01" in out
    assert "=== Recent Best Bid Ask Updates ===" in out
    assert "best_bid=0.49" in out
    assert "=== Recent Market Events ===" in out
    assert "event_type=market_resolved" in out


def test_print_diagnostics_missing_db_is_read_only_and_does_not_create_file(tmp_path, capsys) -> None:
    db_path = tmp_path / "missing.db"

    print_diagnostics(str(db_path))

    out = capsys.readouterr().out
    assert f"database_missing path={db_path}" in out
    assert not db_path.exists()


def test_print_diagnostics_reports_raw_persistence_config(tmp_path, capsys) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_inspection_db(db_path)

    print_diagnostics(str(db_path), raw_ws_events_enabled=False)

    out = capsys.readouterr().out
    assert "raw_ws_events_persistence_enabled=false" in out


def test_main_diagnostics_does_not_open_write_store(monkeypatch, tmp_path, capsys) -> None:
    import src.main as main_module

    db_path = tmp_path / "recorder.db"
    _seed_inspection_db(db_path)

    settings = SimpleNamespace(
        db_path=str(db_path),
        discovery_lookahead_sec=7200,
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden_sqlite_store(*_args, **_kwargs):
        raise AssertionError("diagnostics must not open SQLiteStore")

    monkeypatch.setattr(sys, "argv", ["prog", "--diagnostics"])
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "SQLiteStore", forbidden_sqlite_store)

    main_module.main()

    out = capsys.readouterr().out
    assert "=== Row Counts ===" in out
    assert "raw_polymarket_events=2" in out
    assert "btc_prices=1" in out
