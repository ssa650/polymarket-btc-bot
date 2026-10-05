from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.db import SQLiteStore
from src.diagnostics import print_diagnostics
from src.models import to_iso


def _insert_market(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    start: datetime,
    close: datetime,
    tracking_state: str = "selected_active",
    phase: str = "active",
) -> None:
    conn.execute(
        """
        INSERT INTO markets (
            run_id, market_id, question, start_time, end_time, close_time,
            platform_status, status, phase, market_phase, tracking_state,
            yes_token_id, no_token_id, created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            market_id,
            f"Bitcoin Up or Down - {market_id}",
            to_iso(start),
            to_iso(close),
            to_iso(close),
            "open",
            "open",
            phase,
            phase,
            tracking_state,
            f"{market_id}_yes",
            f"{market_id}_no",
            to_iso(start),
            to_iso(start),
        ),
    )


def _seed_db(db_path, *, now: datetime) -> None:
    store = SQLiteStore(str(db_path), run_id="legacy")
    try:
        store.init_schema()
        stale_close_1 = now - timedelta(days=2)
        stale_close_2 = now - timedelta(days=1)
        current_start = now - timedelta(seconds=60)
        current_close = now + timedelta(seconds=240)
        with store.conn:
            _insert_market(
                store.conn,
                run_id="legacy",
                market_id="stale_legacy",
                start=stale_close_1 - timedelta(minutes=5),
                close=stale_close_1,
            )
            _insert_market(
                store.conn,
                run_id="old_run",
                market_id="stale_old_run",
                start=stale_close_2 - timedelta(minutes=5),
                close=stale_close_2,
            )
            _insert_market(
                store.conn,
                run_id="live_run",
                market_id="current_live",
                start=current_start,
                close=current_close,
            )
            store.conn.execute(
                """
                INSERT INTO market_snapshots (
                    run_id, timestamp, market_id, has_orderbook, has_trade_data
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("legacy", to_iso(stale_close_1), "stale_legacy", 1, 1),
            )
            store.conn.execute(
                """
                INSERT INTO order_book_levels (
                    run_id, timestamp, market_id, outcome_side, book_side,
                    level, price, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("legacy", to_iso(stale_close_1), "stale_legacy", "YES", "bid", 0, 0.49, 5.0),
            )
            store.conn.execute(
                """
                INSERT INTO trades (
                    run_id, timestamp, market_id, trade_id, price, size
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("legacy", to_iso(stale_close_1), "stale_legacy", "trade_stale", 0.5, 2.0),
            )
            store.conn.execute(
                """
                INSERT INTO features (
                    run_id, timestamp, market_id
                ) VALUES (?, ?, ?)
                """,
                ("legacy", to_iso(stale_close_1), "stale_legacy"),
            )
            store.conn.execute(
                """
                INSERT INTO raw_polymarket_events (
                    run_id, local_arrival_ns, local_arrival_iso, event_type,
                    parse_status, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    1_767_225_660_000_000_000,
                    "2026-01-01T00:01:00+00:00",
                    "book",
                    "ok",
                    '{"event_type":"book"}',
                ),
            )
    finally:
        store.close()


def _counts(db_path) -> dict[str, int]:
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
                "market_events",
            )
        }
    finally:
        conn.close()


def _selected_rows(db_path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT market_id, tracking_state, COALESCE(phase, market_phase)
            FROM markets
            ORDER BY market_id
            """
        ).fetchall()
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
    finally:
        conn.close()


def _run_repair_command(monkeypatch, db_path, capsys, *extra_args: str) -> dict:
    import src.main as main_module

    settings = SimpleNamespace(
        db_path=str(db_path),
        sqlite_busy_timeout_ms=5000,
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("repair command must not start network or recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "repair-stale-active-markets", "--stale-active-grace-sec", "0", *extra_args],
    )
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    return json.loads(capsys.readouterr().out)


def test_repair_demotes_stale_selected_active_and_preserves_current_and_data(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    now = datetime.now(timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_db(db_path, now=now)
    before_counts = _counts(db_path)

    report = _run_repair_command(monkeypatch, db_path, capsys)

    assert report["dry_run"] is False
    assert report["stale_selected_active_count"] == 2
    assert report["demoted_count"] == 2
    assert report["market_events_inserted"] == 2
    assert set(report["candidate_market_ids"]) == {"stale_legacy", "stale_old_run"}
    assert report["before"]["selected_active_integrity_alert"] == 1
    assert report["after"]["selected_active_integrity_alert"] == 0

    rows = _selected_rows(db_path)
    assert rows["stale_legacy"] == ("inactive", "expired_waiting_resolution")
    assert rows["stale_old_run"] == ("inactive", "expired_waiting_resolution")
    assert rows["current_live"] == ("selected_active", "active")

    after_counts = _counts(db_path)
    for table in (
        "market_snapshots",
        "order_book_levels",
        "trades",
        "features",
        "raw_polymarket_events",
    ):
        assert after_counts[table] == before_counts[table]
    assert after_counts["market_events"] == before_counts["market_events"] + 2

    conn = sqlite3.connect(str(db_path))
    try:
        event_rows = conn.execute(
            """
            SELECT market_id, event_type, details
            FROM market_events
            ORDER BY market_id
            """
        ).fetchall()
    finally:
        conn.close()
    assert [row[0] for row in event_rows] == ["stale_legacy", "stale_old_run"]
    assert {row[1] for row in event_rows} == {"stale_selected_active_demoted"}
    assert all("stale_selected_active_past_close" in row[2] for row in event_rows)


def test_repair_dry_run_makes_no_changes(monkeypatch, tmp_path, capsys) -> None:
    now = datetime.now(timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_db(db_path, now=now)
    before_counts = _counts(db_path)
    before_rows = _selected_rows(db_path)

    report = _run_repair_command(monkeypatch, db_path, capsys, "--dry-run")

    assert report["dry_run"] is True
    assert report["stale_selected_active_count"] == 2
    assert report["demoted_count"] == 0
    assert report["market_events_inserted"] == 0
    assert _selected_rows(db_path) == before_rows
    assert _counts(db_path) == before_counts


def test_diagnostics_alert_clears_after_stale_active_repair(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    now = datetime.now(timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_db(db_path, now=now)

    print_diagnostics(str(db_path))
    before = capsys.readouterr().out
    assert "selected_active_integrity_alert=1" in before
    assert "invalid_active_markets_past_close=2" in before

    _run_repair_command(monkeypatch, db_path, capsys)

    print_diagnostics(str(db_path))
    after = capsys.readouterr().out
    assert "selected_active_total=1" in after
    assert "invalid_active_markets_past_close=0" in after
    assert "selected_active_integrity_alert=0" in after
