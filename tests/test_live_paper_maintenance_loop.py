from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.baseline_paper_trader import connect_paper_output_db, ensure_paper_schema
from src.live_paper_maintenance_loop import (
    LivePaperMaintenanceConfig,
    run_live_paper_maintenance_loop,
)
from tests.test_baseline_paper_trader import _fixture, _insert_candidate, _iso


NOW = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)


def _config(tmp_path, recorder_db, paper_db, **overrides) -> LivePaperMaintenanceConfig:
    values = {
        "recorder_db_path": str(recorder_db),
        "paper_db_path": str(paper_db),
        "export_dir": str(tmp_path / "exports"),
        "analytics_dir": str(tmp_path / "analytics"),
        "poll_sec": 0.0,
        "resolution_older_than_minutes": 2.0,
        "analytics_every_minutes": 5.0,
        "export_every_minutes": 15.0,
        "once": True,
    }
    values.update(overrides)
    return LivePaperMaintenanceConfig(**values)


def _fake_analytics(call_log: list[dict]):
    def _inner(*, paper_db_path, output_dir, strategy_id=None, now=None):
        call_log.append(
            {
                "paper_db_path": paper_db_path,
                "output_dir": output_dir,
                "strategy_id": strategy_id,
                "now": now,
            }
        )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "paper_trader_report.json"
        txt_path = output / "paper_trader_report.txt"
        json_path.write_text(
            json.dumps(
                {
                    "call": len(call_log),
                    "strategy_id": strategy_id,
                }
            ),
            encoding="utf-8",
        )
        txt_path.write_text(f"analytics {len(call_log)}\n", encoding="utf-8")
        return {
            "summary": {
                "output_files": {
                    "json": str(json_path),
                    "txt": str(txt_path),
                }
            }
        }

    return _inner


def _fake_export(call_log: list[dict]):
    def _inner(*, paper_db_path, output_path, output_csv_path, include_unresolved, strategy_id):
        call_log.append(
            {
                "paper_db_path": paper_db_path,
                "output_path": output_path,
                "output_csv_path": output_csv_path,
                "include_unresolved": include_unresolved,
                "strategy_id": strategy_id,
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(f"parquet {len(call_log)}\n", encoding="utf-8")
        Path(output_csv_path).write_text(f"csv {len(call_log)}\n", encoding="utf-8")
        return {"status": "ok", "rows_exported": len(call_log)}

    return _inner


def _paper_status_counts(paper_db) -> dict[str, int]:
    conn = sqlite3.connect(str(paper_db))
    conn.row_factory = sqlite3.Row
    try:
        return {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM paper_trades GROUP BY status"
            ).fetchall()
        }
    finally:
        conn.close()


def test_one_shot_mode_runs_refresh_settlement_analytics_and_export(tmp_path) -> None:
    recorder_db, paper_db, _model_path, _features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="m1",
        close_offset_sec=-60,
        resolved=1,
        winning_outcome="YES",
    )
    output = connect_paper_output_db(str(paper_db))
    try:
        ensure_paper_schema(output)
        output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, signal_direction,
                stake_usd, adjusted_entry_price, market_close_time, status
            ) VALUES (?, 'run_live', 'm1', ?, 'YES', 1.0, 0.5, ?, 'open')
            """,
            (_iso(NOW - timedelta(minutes=4)), _iso(NOW - timedelta(minutes=4)), _iso(NOW - timedelta(minutes=1))),
        )
        output.commit()
    finally:
        output.close()
    refresh_calls: list[dict] = []
    analytics_calls: list[dict] = []
    export_calls: list[dict] = []

    def refresh(**kwargs):
        refresh_calls.append(kwargs)
        return {"status": "ok", "markets_updated": 1}

    report = run_live_paper_maintenance_loop(
        _config(tmp_path, recorder_db, paper_db, strategy_id="s1"),
        resolution_refresh_fn=refresh,
        analytics_fn=_fake_analytics(analytics_calls),
        export_fn=_fake_export(export_calls),
        now_fn=lambda: NOW,
        emit_logs=False,
    )

    assert report["iterations"] == 1
    assert refresh_calls[0]["apply"] is True
    assert analytics_calls[0]["strategy_id"] == "s1"
    assert export_calls[0]["strategy_id"] == "s1"
    assert _paper_status_counts(paper_db)["settled"] == 1
    assert (tmp_path / "analytics" / "latest_analytics.json").exists()
    assert (tmp_path / "exports" / "latest_meta_strategy_dataset.csv").exists()


def test_dry_run_does_not_write_files_or_apply_updates(tmp_path) -> None:
    recorder_db, paper_db, _model_path, _features_path = _fixture(tmp_path)
    refresh_calls: list[dict] = []

    def refresh(**kwargs):
        refresh_calls.append(kwargs)
        return {"status": "ok", "dry_run": not kwargs["apply"]}

    report = run_live_paper_maintenance_loop(
        _config(tmp_path, recorder_db, paper_db, dry_run=True),
        resolution_refresh_fn=refresh,
        analytics_fn=_fake_analytics([]),
        export_fn=_fake_export([]),
        now_fn=lambda: NOW,
        emit_logs=False,
    )

    assert refresh_calls[0]["apply"] is False
    assert report["last_poll"]["settlement"]["status"] == "skipped"
    assert not Path(paper_db).exists()
    assert not (tmp_path / "analytics").exists()
    assert not (tmp_path / "exports").exists()


def test_loop_handles_one_failure_and_continues(tmp_path) -> None:
    recorder_db, paper_db, _model_path, _features_path = _fixture(tmp_path)
    analytics_calls = 0
    export_calls: list[dict] = []

    def refresh(**kwargs):
        return {"status": "ok", "apply": kwargs["apply"]}

    def flaky_analytics(**kwargs):
        nonlocal analytics_calls
        analytics_calls += 1
        if analytics_calls == 1:
            raise RuntimeError("temporary analytics failure")
        return _fake_analytics([])(**kwargs)

    report = run_live_paper_maintenance_loop(
        _config(
            tmp_path,
            recorder_db,
            paper_db,
            once=False,
            analytics_every_minutes=0.0,
            export_every_minutes=0.0,
        ),
        resolution_refresh_fn=refresh,
        analytics_fn=flaky_analytics,
        export_fn=_fake_export(export_calls),
        max_iterations=2,
        sleep_fn=lambda _seconds: None,
        now_fn=lambda: NOW,
        emit_logs=False,
    )

    assert report["iterations"] == 2
    assert analytics_calls == 2
    assert report["last_poll"]["errors"] == []
    assert (tmp_path / "analytics" / "latest_analytics.json").exists()


def test_latest_files_are_replaced(tmp_path) -> None:
    recorder_db, paper_db, _model_path, _features_path = _fixture(tmp_path)
    analytics_calls: list[dict] = []
    export_calls: list[dict] = []
    times = iter([NOW, NOW + timedelta(minutes=1)])

    def refresh(**kwargs):
        return {"status": "ok"}

    run_live_paper_maintenance_loop(
        _config(
            tmp_path,
            recorder_db,
            paper_db,
            once=False,
            analytics_every_minutes=0.0,
            export_every_minutes=0.0,
        ),
        resolution_refresh_fn=refresh,
        analytics_fn=_fake_analytics(analytics_calls),
        export_fn=_fake_export(export_calls),
        max_iterations=2,
        sleep_fn=lambda _seconds: None,
        now_fn=lambda: next(times),
        emit_logs=False,
    )

    latest_analytics = (tmp_path / "analytics" / "latest_analytics.txt").read_text(
        encoding="utf-8"
    )
    latest_export = (tmp_path / "exports" / "latest_meta_strategy_dataset.csv").read_text(
        encoding="utf-8"
    )

    assert latest_analytics == "analytics 2\n"
    assert latest_export == "csv 2\n"


def test_live_paper_maintenance_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "live-paper-maintenance-loop",
            "--recorder-db",
            "data/recorder.db",
            "--paper-db",
            "data/paper_trades_multi_strategy_expanded.db",
            "--export-dir",
            "data/exports/live_paper",
            "--analytics-dir",
            "data/analytics/live_paper",
            "--once",
        ],
    )

    args = main.parse_args()

    assert args.command == "live-paper-maintenance-loop"
    assert args.recorder_db == "data/recorder.db"
    assert args.paper_db == "data/paper_trades_multi_strategy_expanded.db"
    assert args.once is True
