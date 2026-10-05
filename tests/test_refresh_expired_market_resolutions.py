from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from src.baseline_paper_trader import (
    connect_paper_output_db,
    connect_recorder_read_only,
    ensure_paper_schema,
    settle_open_paper_trades,
)
from src.db import SQLiteStore
from src.expired_resolution_refresh import refresh_expired_market_resolutions
from src.models import to_iso


RUN_ID = "run_refresh"
NOW = datetime(2026, 4, 28, 22, 0, tzinfo=timezone.utc)


class FakeGamma:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def fetch_market_by_id(self, market_id, *, condition_id=None):
        self.calls.append((market_id, condition_id))
        return self.payload


def _seed_recorder(db_path) -> None:
    store = SQLiteStore(str(db_path), run_id=RUN_ID, quarantine_on_startup=False)
    try:
        store.init_schema()
        close_time = NOW - timedelta(minutes=5)
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO markets (
                    run_id, market_id, question, start_time, end_time, close_time,
                    platform_status, status, phase, market_phase, tracking_state,
                    yes_token_id, no_token_id, condition_id, resolved,
                    created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    RUN_ID,
                    "m1",
                    "BTC Up or Down",
                    to_iso(close_time - timedelta(minutes=5)),
                    to_iso(close_time),
                    to_iso(close_time),
                    "open",
                    "open",
                    "expired_waiting_resolution",
                    "expired_waiting_resolution",
                    "inactive",
                    "yes_asset",
                    "no_asset",
                    "0xcond",
                    to_iso(close_time - timedelta(minutes=5)),
                    to_iso(close_time - timedelta(minutes=5)),
                ),
            )
    finally:
        store.close()


def _market_row(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM markets WHERE run_id = ? AND market_id = 'm1'",
            (RUN_ID,),
        ).fetchone()
    finally:
        conn.close()


def _market_event_count(db_path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM market_events WHERE event_type = 'market_resolution_backfilled'"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_refresh_expired_market_resolutions_dry_run_does_not_update(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_recorder(db_path)
    gamma = FakeGamma(
        {
            "id": "m1",
            "resolved": True,
            "winningAssetId": "yes_asset",
            "winningOutcome": "YES",
        }
    )

    report = refresh_expired_market_resolutions(
        db_path=str(db_path),
        gamma_client=gamma,
        older_than_minutes=2,
        limit=100,
        apply=False,
        now=NOW,
    )
    row = _market_row(db_path)

    assert report["dry_run"] is True
    assert report["resolved_found"] == 1
    assert report["markets_updated"] == 0
    assert row["resolved"] == 0
    assert _market_event_count(db_path) == 0


def test_refresh_expired_market_resolutions_apply_updates_market_and_event(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_recorder(db_path)
    gamma = FakeGamma(
        {
            "id": "m1",
            "status": "resolved",
            "winningAssetId": "no_asset",
        }
    )

    report = refresh_expired_market_resolutions(
        db_path=str(db_path),
        gamma_client=gamma,
        older_than_minutes=2,
        limit=100,
        apply=True,
        now=NOW,
    )
    row = _market_row(db_path)

    assert report["dry_run"] is False
    assert report["markets_updated"] == 1
    assert row["resolved"] == 1
    assert row["winning_asset_id"] == "no_asset"
    assert row["winning_outcome"] == "NO"
    assert row["phase"] == "resolved"
    assert _market_event_count(db_path) == 1


def test_refreshed_resolution_settles_awaiting_paper_trade(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _seed_recorder(recorder_db)
    paper = connect_paper_output_db(str(paper_db))
    try:
        ensure_paper_schema(paper)
        paper.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp,
                market_close_time, predicted_probability_yes,
                probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd,
                entry_price, adjusted_entry_price, status
            ) VALUES (?, ?, 'm1', ?, ?, 0.8, 0.8, 0.3, 'YES', 0.65, 1.0, 0.5, 0.5, 'awaiting_resolution')
            """,
            (
                to_iso(NOW - timedelta(minutes=10)),
                RUN_ID,
                to_iso(NOW - timedelta(minutes=6)),
                to_iso(NOW - timedelta(minutes=5)),
            ),
        )
        paper.commit()
    finally:
        paper.close()
    refresh_expired_market_resolutions(
        db_path=str(recorder_db),
        gamma_client=FakeGamma(
            {
                "id": "m1",
                "resolved": True,
                "winningOutcome": "YES",
            }
        ),
        older_than_minutes=2,
        limit=100,
        apply=True,
        now=NOW,
    )

    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(paper_db))
    try:
        settled = settle_open_paper_trades(recorder, output, now=NOW, emit_logs=False)
    finally:
        recorder.close()
        output.close()
    paper = sqlite3.connect(str(paper_db))
    paper.row_factory = sqlite3.Row
    try:
        trade = paper.execute("SELECT * FROM paper_trades").fetchone()
    finally:
        paper.close()

    assert settled == 1
    assert trade["status"] == "settled"
    assert trade["resolved_label"] == "YES"
    assert trade["pnl_usd"] == 1.0
