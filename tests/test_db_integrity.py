from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.db import (
    MARKET_SNAPSHOT_INSERT_COLUMNS,
    SQLiteStore,
    SnapshotInsertShapeError,
    ensure_startup_db_integrity,
    run_sqlite_integrity_checks,
    serialize_market_snapshot_row,
)
from src.models import MarketMetadata, MarketSnapshotRecord


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _record_sqlite_pragmas(monkeypatch) -> list[str]:
    real_connect = sqlite3.connect
    observed: list[str] = []

    class RecordingConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if str(sql).strip().upper().startswith("PRAGMA "):
                observed.append(str(sql).strip())
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    def fake_connect(*args, **kwargs):
        return RecordingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr("src.db.sqlite3.connect", fake_connect)
    return observed


def _create_valid_recorder_db(db_path) -> None:
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
    finally:
        store.close()


def test_snapshot_trade_integrity_counts_and_repair(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
        market = MarketMetadata(
            market_id="m_db_guard",
            event_id="e_db_guard",
            question="BTC Up or Down - test",
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
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            condition_id="cond_db_guard",
        )
        store.upsert_markets([market])

        ts_valid = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
        ts_invalid_before_start = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
        ts_invalid_after_snapshot = datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc)

        with store.conn:
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id, last_trade_price, last_trade_size, last_trade_time, has_trade_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    _iso(ts_valid),
                    "m_db_guard",
                    0.5,
                    1.0,
                    _iso(start + timedelta(seconds=30)),
                    1,
                ),
            )
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id, last_trade_price, last_trade_size, last_trade_time, has_trade_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    _iso(ts_invalid_before_start),
                    "m_db_guard",
                    0.5,
                    1.0,
                    _iso(start - timedelta(seconds=1)),
                    1,
                ),
            )
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id, last_trade_price, last_trade_size, last_trade_time, has_trade_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    _iso(ts_invalid_after_snapshot),
                    "m_db_guard",
                    0.5,
                    1.0,
                    _iso(ts_invalid_after_snapshot + timedelta(seconds=5)),
                    1,
                ),
            )

        counts_before = store.snapshot_trade_integrity_counts()
        assert counts_before["last_trade_before_market_start"] == 1
        assert counts_before["last_trade_after_snapshot_time"] == 1
        assert counts_before["total_invalid"] == 2

        repaired = store.repair_invalid_snapshot_trade_fields()
        assert repaired == 2

        counts_after = store.snapshot_trade_integrity_counts()
        assert counts_after["total_invalid"] == 0

        rows = store.conn.execute(
            """
            SELECT timestamp, last_trade_price, last_trade_size, last_trade_time, has_trade_data
            FROM market_snapshots
            WHERE market_id = 'm_db_guard'
            ORDER BY timestamp ASC
            """
        ).fetchall()
        assert rows[0][1] is not None
        assert rows[1][1] is None
        assert rows[1][2] is None
        assert rows[1][3] is None
        assert rows[1][4] == 0
        assert rows[2][1] is None
        assert rows[2][2] is None
        assert rows[2][3] is None
        assert rows[2][4] == 0
    finally:
        store.close()


def test_run_sqlite_integrity_checks_detects_corrupt_file(tmp_path) -> None:
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"this-is-not-sqlite")

    report = run_sqlite_integrity_checks(str(db_path))

    assert report["db_exists"] is True
    assert report["ok"] is False
    assert report["error"] is not None


def test_run_sqlite_integrity_checks_flags_orphan_sidecars(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    (tmp_path / "recorder.db-wal").write_bytes(b"orphan-wal-content")
    (tmp_path / "recorder.db-shm").write_bytes(b"orphan-shm-content")

    report = run_sqlite_integrity_checks(str(db_path))

    assert report["db_exists"] is False
    assert report["ok"] is False
    assert "orphan_sqlite_sidecars_without_base_db" in str(report["error"])
    assert len(report["sidecars_present"]) == 2


def test_run_sqlite_integrity_checks_quick_mode_skips_full_integrity_check(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "recorder.db"
    _create_valid_recorder_db(db_path)
    observed = _record_sqlite_pragmas(monkeypatch)

    report = run_sqlite_integrity_checks(str(db_path), mode="quick")

    assert report["ok"] is True
    assert report["integrity_check_mode"] == "quick"
    assert str(report["quick_check"]).lower() == "ok"
    assert report["integrity_check"] is None
    assert "PRAGMA quick_check" in observed
    assert "PRAGMA integrity_check" not in observed


def test_run_sqlite_integrity_checks_full_mode_runs_full_integrity_check(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "recorder.db"
    _create_valid_recorder_db(db_path)
    observed = _record_sqlite_pragmas(monkeypatch)

    report = run_sqlite_integrity_checks(str(db_path), mode="full")

    assert report["ok"] is True
    assert report["integrity_check_mode"] == "full"
    assert str(report["quick_check"]).lower() == "ok"
    assert str(report["integrity_check"]).lower() == "ok"
    assert "PRAGMA quick_check" in observed
    assert "PRAGMA integrity_check" in observed


def test_startup_integrity_check_off_mode_skips_sqlite_connection(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    db_path = tmp_path / "recorder.db"
    _create_valid_recorder_db(db_path)

    def fail_connect(*_args, **_kwargs):
        raise AssertionError("startup integrity off mode should not open SQLite")

    monkeypatch.setattr("src.db.sqlite3.connect", fail_connect)

    with caplog.at_level(logging.WARNING):
        report = ensure_startup_db_integrity(str(db_path), mode="off")

    assert report["ok"] is True
    assert report["integrity_check_mode"] == "off"
    assert report["integrity_check_skipped"] is True
    assert "startup_integrity_check_skipped" in [
        record.getMessage() for record in caplog.records
    ]


def test_startup_integrity_check_logs_mode_and_duration(tmp_path, caplog) -> None:
    db_path = tmp_path / "recorder.db"
    _create_valid_recorder_db(db_path)

    with caplog.at_level(logging.INFO):
        report = ensure_startup_db_integrity(str(db_path), mode="quick")

    messages = [record.getMessage() for record in caplog.records]
    assert report["ok"] is True
    assert report["integrity_check_mode"] == "quick"
    assert isinstance(report["elapsed_sec"], float)
    assert "startup_integrity_check_starting" in messages
    assert "startup_integrity_check_finished" in messages


def test_startup_integrity_check_quarantines_orphan_sidecars(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    (tmp_path / "recorder.db-wal").write_bytes(b"orphan-wal-content")
    (tmp_path / "recorder.db-shm").write_bytes(b"orphan-shm-content")

    ensure_startup_db_integrity(str(db_path))

    assert not (tmp_path / "recorder.db-wal").exists()
    assert not (tmp_path / "recorder.db-shm").exists()
    quarantined_wal = sorted(tmp_path.glob("recorder.corrupt.*.db-wal"))
    quarantined_shm = sorted(tmp_path.glob("recorder.corrupt.*.db-shm"))
    assert len(quarantined_wal) == 1
    assert len(quarantined_shm) == 1


def test_startup_integrity_check_quarantines_corrupt_db(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    db_path.write_bytes(b"corrupt-content")
    (tmp_path / "recorder.db-wal").write_bytes(b"wal-corrupt-content")
    (tmp_path / "recorder.db-shm").write_bytes(b"shm-corrupt-content")

    ensure_startup_db_integrity(str(db_path))

    assert not db_path.exists()
    quarantined = sorted(tmp_path.glob("recorder.corrupt.*.db"))
    assert len(quarantined) == 1


def test_sqlite_store_constructor_rotates_corrupt_db_before_init(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    db_path.write_bytes(b"broken-db-content")
    (tmp_path / "recorder.db-wal").write_bytes(b"broken-wal")
    (tmp_path / "recorder.db-shm").write_bytes(b"broken-shm")

    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        report = run_sqlite_integrity_checks(str(db_path))
        assert report["ok"] is True
    finally:
        store.close()

    quarantined = sorted(tmp_path.glob("recorder.corrupt.*.db"))
    assert len(quarantined) == 1


def test_sqlite_store_constructor_recovers_from_orphan_sidecars(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    (tmp_path / "recorder.db-wal").write_bytes(b"orphan-wal")
    (tmp_path / "recorder.db-shm").write_bytes(b"orphan-shm")

    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        report = run_sqlite_integrity_checks(str(db_path))
        assert report["ok"] is True
        assert report["sqlite_master_issues"] == []
    finally:
        store.close()

    quarantined_wal = sorted(tmp_path.glob("recorder.corrupt.*.db-wal"))
    quarantined_shm = sorted(tmp_path.glob("recorder.corrupt.*.db-shm"))
    assert len(quarantined_wal) == 1
    assert len(quarantined_shm) == 1


def _snapshot_record(ts: datetime, market_id: str = "m_snapshot") -> MarketSnapshotRecord:
    return MarketSnapshotRecord(
        timestamp=ts,
        market_id=market_id,
        yes_price=0.5,
        no_price=0.5,
        best_bid_yes=0.49,
        best_ask_yes=0.51,
        best_bid_no=0.49,
        best_ask_no=0.51,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.5,
        mid_price_no=0.5,
        volume=100.0,
        liquidity=2000.0,
        last_trade_price=0.5,
        last_trade_size=2.0,
        last_trade_time=ts,
        has_orderbook=1,
        has_trade_data=1,
        run_id="legacy",
        strict_validation_passed=1,
        snapshot_quality_status="ok",
        is_partial_orderbook=0,
        missing_level_count=0,
        time_gap_from_prev_snapshot_sec=1.0,
        is_gap_affected=0,
        feature_ready=1,
        book_checksum="abc123",
    )


def test_snapshot_row_length_matches_insert_columns() -> None:
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    row = serialize_market_snapshot_row(_snapshot_record(ts), fallback_run_id="legacy")
    assert len(row) == len(MARKET_SNAPSHOT_INSERT_COLUMNS)


def test_init_schema_creates_expected_tables(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        rows = store.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
        table_names = {str(row[0]) for row in rows}
        expected = {
            "markets",
            "market_snapshots",
            "order_book_levels",
            "trades",
            "features",
            "market_events",
            "recorder_metrics",
        }
        missing = expected - table_names
        assert not missing, f"missing tables: {sorted(missing)}"

        integrity = run_sqlite_integrity_checks(str(db_path))
        assert integrity["ok"] is True
        assert str(integrity["quick_check"]).lower() == "ok"
        assert str(integrity["integrity_check"]).lower() == "ok"
        assert integrity["sqlite_master_issues"] == []
    finally:
        store.close()


def test_run_sqlite_integrity_checks_detects_polluted_sqlite_master(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("PRAGMA writable_schema=ON")
            # Reproduces observed corruption signature where timestamps leak into schema rows.
            conn.execute(
                """
                INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
                VALUES (
                    'table',
                    '2026-04-06T12:00:00Z',
                    '2026-04-06T12:00:00Z',
                    0,
                    '2026-04-06T12:00:00Z'
                )
                """
            )
    finally:
        conn.close()

    report = run_sqlite_integrity_checks(str(db_path))
    assert report["ok"] is False
    assert report["error"] is not None


def test_manual_writable_schema_can_mask_pragmas_but_integrity_check_still_fails(
    tmp_path,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute(
                """
                INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
                VALUES ('table', '2026-04-06T12:34:56Z', '2026-04-06T12:34:56Z', 0, '2026-04-06T12:34:56Z')
                """
            )
    finally:
        conn.close()

    # Demonstrate SQLite edge case: writable_schema can make pragmas report "ok".
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA writable_schema=ON")
        assert str(conn.execute("PRAGMA quick_check").fetchone()[0]).lower() == "ok"
        assert str(conn.execute("PRAGMA integrity_check").fetchone()[0]).lower() == "ok"
    finally:
        conn.close()

    report = run_sqlite_integrity_checks(str(db_path))
    assert report["writable_schema"] == 0
    assert report["ok"] is False
    assert report["error"] is not None


def test_init_schema_recovers_if_startup_check_is_bypassed_and_db_is_malformed(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "recorder.db"

    seed = SQLiteStore(str(db_path))
    try:
        seed.init_schema()
    finally:
        seed.close()

    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute(
                """
                INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
                VALUES ('table', '2026-04-06T00:00:00Z', '2026-04-06T00:00:00Z', 0, '2026-04-06T00:00:00Z')
                """
            )
    finally:
        conn.close()

    monkeypatch.setattr(
        "src.db.ensure_startup_db_integrity",
        lambda *_args, **_kwargs: {"ok": True, "db_exists": True},
    )

    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        report = run_sqlite_integrity_checks(str(db_path))
        assert report["ok"] is True
    finally:
        store.close()

    quarantined = sorted(tmp_path.glob("recorder.corrupt.*.db"))
    assert len(quarantined) == 1


def test_schema_write_guard_blocks_sqlite_master_mutation(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        with pytest.raises(sqlite3.DatabaseError):
            with store.conn:
                store.conn.execute("PRAGMA writable_schema=ON")
                store.conn.execute(
                    """
                    INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
                    VALUES ('table', 'oops', 'oops', 0, 'CREATE TABLE oops(x INTEGER)')
                    """
                )
    finally:
        store.close()


def test_schema_write_guard_is_active_before_init_schema(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        with pytest.raises(sqlite3.DatabaseError):
            with store.conn:
                store.conn.execute("PRAGMA writable_schema=ON")
    finally:
        store.close()


def test_insert_market_snapshots_succeeds_with_matching_shape(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        market = MarketMetadata(
            market_id="m_snapshot",
            event_id="e_snapshot",
            question="Bitcoin Up or Down - 00:00-00:05",
            description=None,
            category="crypto",
            outcomes=["Yes", "No"],
            resolution_source=None,
            start_time=ts,
            end_time=ts + timedelta(minutes=5),
            close_time=ts + timedelta(minutes=5),
            status="open",
            platform_status="open",
            phase="active",
            tracking_state="selected_active",
            yes_token_id="tok_yes",
            no_token_id="tok_no",
        )
        store.upsert_markets([market])

        inserted, skipped = store.insert_market_snapshots([_snapshot_record(ts)])
        assert inserted == 1
        assert skipped == 0
    finally:
        store.close()


def test_insert_market_snapshots_raises_clear_error_on_shape_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path))
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        market = MarketMetadata(
            market_id="m_snapshot",
            event_id="e_snapshot",
            question="Bitcoin Up or Down - 00:00-00:05",
            description=None,
            category="crypto",
            outcomes=["Yes", "No"],
            resolution_source=None,
            start_time=ts,
            end_time=ts + timedelta(minutes=5),
            close_time=ts + timedelta(minutes=5),
            status="open",
            platform_status="open",
            phase="active",
            tracking_state="selected_active",
            yes_token_id="tok_yes",
            no_token_id="tok_no",
        )
        store.upsert_markets([market])

        original = serialize_market_snapshot_row

        def bad_serializer(record: MarketSnapshotRecord, fallback_run_id: str) -> tuple[object, ...]:
            return original(record, fallback_run_id)[:-1]

        monkeypatch.setattr("src.db.serialize_market_snapshot_row", bad_serializer)
        with pytest.raises(SnapshotInsertShapeError) as exc:
            store.insert_market_snapshots([_snapshot_record(ts)])
        err = exc.value
        assert err.expected_column_count == len(MARKET_SNAPSHOT_INSERT_COLUMNS)
        assert err.actual_row_length == len(MARKET_SNAPSHOT_INSERT_COLUMNS) - 1
        assert "schema_version" in err.columns
    finally:
        store.close()
