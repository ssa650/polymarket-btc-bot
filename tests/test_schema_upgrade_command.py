from __future__ import annotations

import json
import sqlite3
import sys
from types import SimpleNamespace

from src.db import SQLiteStore, _wal_size_warnings
from src.diagnostics import print_diagnostics


TARGET_TABLES = {
    "raw_polymarket_events",
    "tick_size_changes",
    "best_bid_ask_updates",
    "btc_prices",
}

PRESERVE_TABLES = (
    "markets",
    "market_snapshots",
    "order_book_levels",
    "trades",
    "features",
    "market_events",
    "recorder_metrics",
)


def _table_names(db_path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def _row_counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in PRESERVE_TABLES
        }
    finally:
        conn.close()


def _seed_old_db_missing_stage_tables(db_path) -> None:
    store = SQLiteStore(str(db_path), run_id="legacy")
    try:
        store.init_schema()
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO markets (
                    run_id, market_id, question, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "old_market",
                    "Bitcoin Up or Down - old",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:01+00:00",
                ),
            )
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id
                ) VALUES (?, ?, ?)
                """,
                ("legacy", "2026-01-01T00:00:01+00:00", "old_market"),
            )
            store.conn.execute(
                """
                INSERT INTO order_book_levels (
                    run_id, timestamp, market_id, outcome_side, book_side,
                    level, price, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "2026-01-01T00:00:01+00:00",
                    "old_market",
                    "YES",
                    "bid",
                    0,
                    0.49,
                    10.0,
                ),
            )
            store.conn.execute(
                """
                INSERT INTO trades (
                    run_id, timestamp, market_id, trade_id, price, size
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "2026-01-01T00:00:01+00:00",
                    "old_market",
                    "old_trade",
                    0.50,
                    5.0,
                ),
            )
            store.conn.execute(
                """
                INSERT INTO features (
                    run_id, timestamp, market_id
                ) VALUES (?, ?, ?)
                """,
                ("legacy", "2026-01-01T00:00:01+00:00", "old_market"),
            )
            store.conn.execute(
                """
                INSERT INTO market_events (
                    run_id, timestamp, market_id, event_type
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "2026-01-01T00:00:01+00:00",
                    "old_market",
                    "market_opened",
                ),
            )
            store.conn.execute(
                """
                INSERT INTO recorder_metrics (
                    run_id, timestamp, markets_polled, successful_market_fetches,
                    failed_markets, rows_inserted, duplicate_rows_skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("legacy", "2026-01-01T00:00:01+00:00", 1, 1, 0, 6, 0),
            )
    finally:
        store.close()
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            for table in TARGET_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        conn.close()


def test_upgrade_db_command_adds_missing_tables_and_preserves_rows(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import src.main as main_module

    db_path = tmp_path / "recorder.db"
    _seed_old_db_missing_stage_tables(db_path)
    assert TARGET_TABLES.isdisjoint(_table_names(db_path))
    before_counts = _row_counts(db_path)

    settings = SimpleNamespace(
        db_path=str(db_path),
        sqlite_busy_timeout_ms=5000,
        discovery_lookahead_sec=7200,
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("upgrade-db must not start network or recorder code")

    monkeypatch.setattr(sys, "argv", ["prog", "upgrade-db"])
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["action"] == "schema_upgraded"
    assert report["schema_only"] is True
    assert set(report["created_tables"]) >= TARGET_TABLES
    assert all(report["target_tables_present"].values())
    assert TARGET_TABLES.issubset(_table_names(db_path))
    assert _row_counts(db_path) == before_counts
    for table, counts in report["preserved_row_counts"].items():
        assert counts["before"] == before_counts[table]
        assert counts["after"] == before_counts[table]


def test_diagnostics_does_not_upgrade_missing_tables(tmp_path, capsys) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_old_db_missing_stage_tables(db_path)
    assert TARGET_TABLES.isdisjoint(_table_names(db_path))

    print_diagnostics(str(db_path))

    out = capsys.readouterr().out
    assert "raw_polymarket_events=missing" in out
    assert "tick_size_changes=missing" in out
    assert "best_bid_ask_updates=missing" in out
    assert "btc_prices=missing" in out
    assert TARGET_TABLES.isdisjoint(_table_names(db_path))


def test_wal_checkpoint_command_invokes_pragma_without_recorder(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import src.main as main_module

    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_checkpoint")
    try:
        store.init_schema()
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO recorder_metrics (
                    run_id, timestamp, markets_polled, successful_market_fetches,
                    failed_markets, rows_inserted, duplicate_rows_skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("run_checkpoint", "2026-01-01T00:00:01+00:00", 1, 1, 0, 1, 0),
            )
    finally:
        store.close()

    settings = SimpleNamespace(
        db_path=str(db_path),
        sqlite_busy_timeout_ms=5000,
        discovery_lookahead_sec=7200,
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("wal-checkpoint must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "wal-checkpoint",
            "--db",
            str(db_path),
            "--checkpoint-mode",
            "TRUNCATE",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["mode"] == "TRUNCATE"
    assert report["db_path"] == str(db_path)
    assert report["truncate_requested"] is True
    assert report["passive_checkpoint"]["busy"] in (0, 1)
    assert report["truncate_checkpoint"] is not None
    assert "checkpointed_frames" in report
    assert "sizes_before" in report
    assert "sizes_after" in report
    assert "wal_size_before_bytes" in report
    assert "wal_size_after_bytes" in report
    assert "Long-lived SQLite readers" in report["note"]


def test_wal_checkpoint_size_warnings() -> None:
    gib = 1024 ** 3
    assert _wal_size_warnings(0) == []
    assert _wal_size_warnings(10 * gib) == ["recorder.db-wal exceeds 10GB"]
    assert _wal_size_warnings(25 * gib) == [
        "recorder.db-wal exceeds 10GB",
        "recorder.db-wal exceeds 25GB",
    ]
    assert _wal_size_warnings(50 * gib) == [
        "recorder.db-wal exceeds 10GB",
        "recorder.db-wal exceeds 25GB",
        "recorder.db-wal exceeds 50GB",
    ]
