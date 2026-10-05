from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from src.db import SQLiteStore
from src.diagnostics import print_diagnostics
from src.models import to_iso


RUN_ID = "run_resolution"
RESOLVED_AT = datetime(2026, 1, 1, 0, 5, 1, tzinfo=timezone.utc)
CONDITION_HASH = "0x" + ("a" * 64)


def _seed_store(db_path) -> SQLiteStore:
    store = SQLiteStore(str(db_path), run_id=RUN_ID, quarantine_on_startup=False)
    store.init_schema()
    return store


def _insert_market(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    condition_id: str,
    yes_token_id: str = "yes_asset",
    no_token_id: str = "no_asset",
    question: str | None = "Bitcoin Up or Down - fixture",
) -> None:
    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    conn.execute(
        """
        INSERT INTO markets (
            run_id, market_id, question, start_time, end_time, close_time,
            platform_status, status, phase, market_phase, tracking_state,
            yes_token_id, no_token_id, condition_id, created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            market_id,
            question,
            to_iso(now),
            to_iso(close),
            to_iso(close),
            "open",
            "open",
            "expired_waiting_resolution",
            "expired_waiting_resolution",
            "inactive",
            yes_token_id,
            no_token_id,
            condition_id,
            to_iso(now),
            to_iso(now),
        ),
    )


def _insert_resolution_event(
    conn: sqlite3.Connection,
    *,
    event_market_id: str,
    details: dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT INTO market_events (
            run_id, timestamp, market_id, event_type, details
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(RESOLVED_AT),
            event_market_id,
            "market_resolved",
            json.dumps(details, sort_keys=True),
        ),
    )


def _market_row(db_path, market_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
                market_id, condition_id, resolved, resolved_at,
                winning_asset_id, winning_outcome, platform_status,
                phase, tracking_state
            FROM markets
            WHERE run_id = ? AND market_id = ?
            """,
            (RUN_ID, market_id),
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def _table_counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "market_snapshots",
                "order_book_levels",
                "trades",
                "features",
                "raw_polymarket_events",
                "btc_prices",
            )
        }
    finally:
        conn.close()


def _run_normalize_command(monkeypatch, db_path, capsys, *extra_args: str) -> dict:
    import src.main as main_module

    settings = SimpleNamespace(
        db_path=str(db_path),
        sqlite_busy_timeout_ms=5000,
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normalize-resolutions must not start network or recorder code")

    monkeypatch.setattr(sys, "argv", ["prog", "normalize-resolutions", *extra_args])
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    return json.loads(capsys.readouterr().out)


def test_normalize_resolutions_maps_numeric_market_id(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076290", condition_id="0xcond")
            _insert_resolution_event(
                store.conn,
                event_market_id="2076290",
                details={
                    "id": "2076290",
                    "market": "0xcond",
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "yes_asset",
                    "winning_outcome": "Up",
                },
            )
    finally:
        store.close()

    report = _run_normalize_command(monkeypatch, db_path, capsys)

    assert report["dry_run"] is False
    assert report["mapped_resolution_events"] == 1
    assert report["markets_updated"] == 1
    row = _market_row(db_path, "2076290")
    assert row["resolved"] == 1
    assert row["resolved_at"] == "2026-01-01T00:05:01+00:00"
    assert row["winning_asset_id"] == "yes_asset"
    assert row["winning_outcome"] == "Up"
    assert row["platform_status"] == "resolved"
    assert row["phase"] == "resolved"
    assert row["tracking_state"] == "inactive"


def test_normalize_resolutions_maps_condition_hash_to_numeric_market(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076290", condition_id=CONDITION_HASH)
            _insert_market(
                store.conn,
                market_id=CONDITION_HASH,
                condition_id=CONDITION_HASH,
                yes_token_id="",
                no_token_id="",
                question=None,
            )
            _insert_resolution_event(
                store.conn,
                event_market_id=CONDITION_HASH,
                details={
                    "id": "2076290",
                    "market": CONDITION_HASH,
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "yes_asset",
                    "winning_outcome": "Up",
                },
            )
    finally:
        store.close()

    report = _run_normalize_command(monkeypatch, db_path, capsys)

    assert report["mapped_market_ids"] == ["2076290"]
    assert _market_row(db_path, "2076290")["resolved"] == 1
    assert _market_row(db_path, CONDITION_HASH)["resolved"] == 0


def test_normalize_resolutions_infers_yes_no_outcome_from_winning_asset_id(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076291", condition_id="0xcond-no")
            _insert_resolution_event(
                store.conn,
                event_market_id="2076291",
                details={
                    "id": "2076291",
                    "market": "0xcond-no",
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "no_asset",
                },
            )
    finally:
        store.close()

    _run_normalize_command(monkeypatch, db_path, capsys)

    row = _market_row(db_path, "2076291")
    assert row["winning_asset_id"] == "no_asset"
    assert row["winning_outcome"] == "NO"


def test_normalize_resolutions_preserves_existing_data_rows(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076290", condition_id="0xcond")
            _insert_resolution_event(
                store.conn,
                event_market_id="2076290",
                details={
                    "id": "2076290",
                    "market": "0xcond",
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "yes_asset",
                },
            )
            ts = "2026-01-01T00:04:59+00:00"
            store.conn.execute(
                "INSERT INTO market_snapshots (run_id, timestamp, market_id) VALUES (?, ?, ?)",
                (RUN_ID, ts, "2076290"),
            )
            store.conn.execute(
                """
                INSERT INTO order_book_levels (
                    run_id, timestamp, market_id, outcome_side, book_side, level, price, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, ts, "2076290", "YES", "bid", 1, 0.5, 1.0),
            )
            store.conn.execute(
                """
                INSERT INTO trades (run_id, timestamp, market_id, trade_id, price, size)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, ts, "2076290", "trade1", 0.5, 1.0),
            )
            store.conn.execute(
                "INSERT INTO features (run_id, timestamp, market_id) VALUES (?, ?, ?)",
                (RUN_ID, ts, "2076290"),
            )
            store.conn.execute(
                """
                INSERT INTO raw_polymarket_events (
                    run_id, local_arrival_ns, local_arrival_iso, event_type, parse_status, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, 1, ts, "market_resolved", "ok", "{}"),
            )
            store.conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, local_arrival_ns, local_arrival_iso, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, "mock", 50000.0, 1, ts, "{}"),
            )
    finally:
        store.close()

    before = _table_counts(db_path)

    _run_normalize_command(monkeypatch, db_path, capsys)

    assert _table_counts(db_path) == before


def test_normalize_resolutions_dry_run_makes_no_changes(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076290", condition_id="0xcond")
            _insert_resolution_event(
                store.conn,
                event_market_id="2076290",
                details={
                    "id": "2076290",
                    "market": "0xcond",
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "yes_asset",
                },
            )
    finally:
        store.close()

    report = _run_normalize_command(monkeypatch, db_path, capsys, "--dry-run")

    assert report["dry_run"] is True
    assert report["mapped_resolution_events"] == 1
    assert report["markets_updated"] == 0
    assert _market_row(db_path, "2076290")["resolved"] == 0


def test_diagnostics_reports_resolution_mapping(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = _seed_store(db_path)
    try:
        with store.conn:
            _insert_market(store.conn, market_id="2076290", condition_id=CONDITION_HASH)
            _insert_market(
                store.conn,
                market_id=CONDITION_HASH,
                condition_id=CONDITION_HASH,
                yes_token_id="",
                no_token_id="",
                question=None,
            )
            _insert_resolution_event(
                store.conn,
                event_market_id=CONDITION_HASH,
                details={
                    "id": "2076290",
                    "market": CONDITION_HASH,
                    "timestamp": "2026-01-01T00:05:01+00:00",
                    "winning_asset_id": "yes_asset",
                },
            )
    finally:
        store.close()

    _run_normalize_command(monkeypatch, db_path, capsys)
    print_diagnostics(str(db_path))
    out = capsys.readouterr().out

    assert "=== Resolution Mapping ===" in out
    assert "resolution_columns_available=1" in out
    assert "market_resolved_events=1" in out
    assert "mapped_resolution_events=1" in out
    assert "unmapped_resolution_events=0" in out
    assert "condition_hash_events_mapped_to_numeric_market=1" in out
    assert "resolved_market_rows=1" in out
    assert "resolved_market_rows_with_winner=1" in out
