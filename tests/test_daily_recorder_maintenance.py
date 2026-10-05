from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.daily_recorder_maintenance import run_daily_recorder_maintenance
from src.models import to_iso


NOW = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(hours=24)
RECENT = NOW - timedelta(hours=1)


def _create_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE markets (
                run_id TEXT,
                market_id TEXT,
                start_time TEXT,
                close_time TEXT,
                created_at TEXT,
                last_updated TEXT
            );
            CREATE TABLE market_snapshots (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT
            );
            CREATE TABLE features (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                feature_ready INTEGER
            );
            CREATE TABLE trades (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                trade_id TEXT,
                price REAL,
                size REAL
            );
            CREATE TABLE btc_prices (
                run_id TEXT,
                source TEXT,
                price REAL,
                local_arrival_ns INTEGER,
                local_arrival_iso TEXT,
                raw_json TEXT
            );
            CREATE TABLE order_book_levels (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                outcome_side TEXT,
                book_side TEXT,
                level INTEGER,
                price REAL,
                size REAL
            );
            CREATE TABLE best_bid_ask_updates (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                raw_json TEXT
            );
            """
        )
        for market_id, ts in (("old", OLD), ("recent", RECENT)):
            conn.execute(
                """
                INSERT INTO markets (
                    run_id, market_id, start_time, close_time, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "run1",
                    market_id,
                    to_iso(ts - timedelta(minutes=5)),
                    to_iso(ts),
                    to_iso(ts - timedelta(minutes=6)),
                    to_iso(ts),
                ),
            )
            conn.execute(
                "INSERT INTO market_snapshots VALUES (?, ?, ?)",
                ("run1", to_iso(ts), market_id),
            )
            conn.execute(
                "INSERT INTO features VALUES (?, ?, ?, ?)",
                ("run1", to_iso(ts), market_id, 1),
            )
            conn.execute(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
                ("run1", to_iso(ts), market_id, f"trade_{market_id}", 0.5, 1.0),
            )
            conn.execute(
                "INSERT INTO btc_prices VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "run1",
                    "polymarket_rtds_chainlink",
                    100.0,
                    1_000_000,
                    to_iso(ts),
                    "{}",
                ),
            )
            conn.execute(
                "INSERT INTO order_book_levels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("run1", to_iso(ts), market_id, "YES", "bid", 1, 0.5, 10.0),
            )
            conn.execute(
                "INSERT INTO best_bid_ask_updates VALUES (?, ?, ?, ?)",
                ("run1", to_iso(ts), market_id, "{}"),
            )
        conn.commit()
    finally:
        conn.close()


def _count(path: Path, table: str) -> int:
    conn = sqlite3.connect(str(path))
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
    finally:
        conn.close()


def _market_ids(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [
            str(row[0])
            for row in conn.execute("SELECT market_id FROM markets ORDER BY market_id")
        ]
    finally:
        conn.close()


def test_daily_maintenance_dry_run_does_not_write_or_delete(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    export_dir = tmp_path / "exports"
    archive_dir = tmp_path / "archive"
    _create_db(db_path)

    report = run_daily_recorder_maintenance(
        db_path=str(db_path),
        export_dir=str(export_dir),
        archive_dir=str(archive_dir),
        keep_recent_hours=12,
        dry_run=True,
        now=NOW,
    )

    assert report["status"] == "dry_run"
    assert report["manifest_path"] is None
    assert report["archive_path"] is None
    assert not export_dir.exists()
    assert not archive_dir.exists()
    assert _count(db_path, "market_snapshots") == 2


def test_daily_maintenance_writes_manifest_archive_and_cleans_old_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    export_dir = tmp_path / "exports"
    archive_dir = tmp_path / "archive"
    _create_db(db_path)

    report = run_daily_recorder_maintenance(
        db_path=str(db_path),
        export_dir=str(export_dir),
        archive_dir=str(archive_dir),
        keep_recent_hours=12,
        checkpoint=True,
        now=NOW,
    )

    manifest_path = Path(str(report["manifest_path"]))
    archive_path = Path(str(report["archive_path"]))

    assert report["status"] == "ok"
    assert manifest_path.exists()
    assert archive_path.exists()
    assert report["checkpoint_before_archive"]["status"] == "ok"
    assert report["checkpoint_after_cleanup"]["status"] == "ok"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_table = {row["table"]: row for row in manifest["tables"]}
    assert by_table["market_snapshots"]["export_count"] == 1
    assert Path(by_table["market_snapshots"]["path"]).exists()
    assert _market_ids(db_path) == ["recent"]
    assert _market_ids(archive_path) == ["old", "recent"]
    assert _count(db_path, "market_snapshots") == 1
    assert _count(db_path, "features") == 1
    assert _count(db_path, "trades") == 1
    assert _count(db_path, "btc_prices") == 1
    assert _count(db_path, "order_book_levels") == 1
    assert _count(db_path, "best_bid_ask_updates") == 1
    assert report["deleted_rows"]["market_snapshots"] == 1


def test_daily_maintenance_does_not_clean_when_export_verification_fails(
    tmp_path,
    monkeypatch,
) -> None:
    from src import daily_recorder_maintenance

    db_path = tmp_path / "recorder.db"
    _create_db(db_path)

    monkeypatch.setattr(
        daily_recorder_maintenance,
        "_verify_exported_counts",
        lambda _plan: {
            "ok": False,
            "checked_row_counts": {},
            "mismatches": [{"table": "market_snapshots", "expected": 1, "actual": 0}],
        },
    )

    report = daily_recorder_maintenance.run_daily_recorder_maintenance(
        db_path=str(db_path),
        export_dir=str(tmp_path / "exports"),
        archive_dir=str(tmp_path / "archive"),
        keep_recent_hours=12,
        now=NOW,
    )

    assert report["status"] == "export_verification_failed"
    assert report["archive_path"] is None
    assert _market_ids(db_path) == ["old", "recent"]
    assert _count(db_path, "market_snapshots") == 2
