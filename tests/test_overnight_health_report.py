from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.overnight_health_report import (
    build_overnight_health_report,
    render_overnight_health_report,
    summarize_disk_wal,
)


NOW = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def _create_recorder_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE features (
            run_id TEXT,
            market_id TEXT,
            timestamp TEXT,
            feature_ready INTEGER
        );
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT
        );
        INSERT INTO markets (run_id, market_id) VALUES ('run', 'm1');
        INSERT INTO features (run_id, market_id, timestamp, feature_ready)
        VALUES
            ('run', 'm1', '2026-01-01T00:08:00+00:00', 1),
            ('run', 'm1', '2026-01-01T00:09:00+00:00', 0),
            ('run', 'm1', '2026-01-01T00:09:30+00:00', 1);
        """
    )
    conn.commit()
    conn.close()


def _create_prediction_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE transformer_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            run_id TEXT,
            market_id TEXT,
            model_path TEXT,
            sequence_length INTEGER,
            transformer_sequence_ready INTEGER,
            probability_yes REAL,
            reason TEXT,
            latest_feature_timestamp TEXT
        );
        INSERT INTO transformer_predictions (
            timestamp, run_id, market_id, model_path, sequence_length,
            transformer_sequence_ready, probability_yes, reason, latest_feature_timestamp
        ) VALUES
            ('2026-01-01T00:07:00+00:00', 'r1', 'm1', 'model.pt', 60, 1, 0.76, 'ok', '2026-01-01T00:07:00+00:00'),
            ('2026-01-01T00:08:00+00:00', 'r1', 'm2', 'model.pt', 60, 1, 0.81, 'ok', '2026-01-01T00:08:00+00:00'),
            ('2026-01-01T00:09:00+00:00', 'r1', 'm3', 'model.pt', 60, 1, 0.20, 'ok', '2026-01-01T00:09:00+00:00'),
            ('2026-01-01T00:09:30+00:00', 'r1', 'm3', 'model.pt', 60, 0, NULL, 'insufficient_sequence_rows', '2026-01-01T00:09:30+00:00');
        """
    )
    conn.commit()
    conn.close()


def _create_paper_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            strategy_id TEXT,
            market_id TEXT,
            status TEXT,
            realized_pnl_usd REAL,
            realized_roi REAL,
            pnl_usd REAL,
            roi REAL,
            skip_reason TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rejection_reason TEXT
        );
        INSERT INTO paper_trades (
            created_at, strategy_id, market_id, status, realized_pnl_usd,
            realized_roi, pnl_usd, roi, skip_reason
        ) VALUES
            ('2026-01-01T00:06:00+00:00', 's1', 'm1', 'closed', -0.25, -0.25, NULL, NULL, NULL),
            ('2026-01-01T00:06:20+00:00', 's1', 'm1', 'closed', 0.05, 0.05, NULL, NULL, NULL),
            ('2026-01-01T00:07:00+00:00', 's2', 'm2', 'open', NULL, NULL, NULL, NULL, NULL),
            ('2026-01-01T00:07:30+00:00', 's2', 'm3', 'skipped', NULL, NULL, NULL, NULL, 'threshold');
        INSERT INTO paper_trade_candidates (rejection_reason)
        VALUES ('threshold'), ('threshold'), ('time_window');
        """
    )
    conn.commit()
    conn.close()


def test_overnight_health_report_summarizes_recorder_predictions_and_paper(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    paper_db = tmp_path / "paper_trades.db"
    _create_recorder_db(recorder_db)
    _create_prediction_db(prediction_db)
    _create_paper_db(paper_db)

    report = build_overnight_health_report(
        recorder_db_path=str(recorder_db),
        prediction_db_inputs=[str(tmp_path / "transformer_*.db")],
        paper_db_globs=[str(tmp_path / "paper_*.db")],
        now=NOW,
    )

    assert report["recorder"]["feature_rows"] == 3
    assert report["recorder"]["ready_rows"] == 2
    assert report["recorder"]["ready_pct"] == 66.666667
    predictions = report["transformer_predictions"]["aggregate"]
    assert predictions["total_rows"] == 4
    assert predictions["probability_rows"] == 3
    assert predictions["market_count"] == 3
    assert predictions["count_probability_gte_075"] == 2
    assert predictions["count_probability_gte_080"] == 1
    assert predictions["count_probability_lte_030"] == 1
    assert predictions["count_probability_lte_025"] == 1
    paper_db_report = report["paper_dbs"]["databases"][0]
    assert paper_db_report["summary"]["pnl"] == -0.2
    assert paper_db_report["summary"]["avg_roi"] == -0.1
    assert paper_db_report["summary"]["wins"] == 1
    assert paper_db_report["summary"]["losses"] == 1
    assert paper_db_report["top_rejection_reasons"][0] == {"reason": "threshold", "count": 3}
    assert paper_db_report["duplicate_strategy_market_groups"][0]["strategy_id"] == "s1"
    assert any(flag["code"] == "negative_pnl" for flag in report["red_flags"])
    assert any(flag["code"] == "duplicate_strategy_market_trades" for flag in report["red_flags"])

    rendered = render_overnight_health_report(report)
    assert "Recorder" in rendered
    assert "Transformer Predictions" in rendered
    assert "Paper DBs" in rendered
    assert "negative_pnl" in rendered


def test_overnight_health_report_handles_missing_inputs(tmp_path) -> None:
    report = build_overnight_health_report(
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        prediction_db_inputs=[str(tmp_path / "missing_predictions_*.db")],
        paper_db_globs=[str(tmp_path / "missing_paper_*.db")],
        now=NOW,
    )

    assert report["recorder"]["status"] == "missing"
    assert report["transformer_predictions"]["unmatched_inputs"]
    assert report["paper_dbs"]["unmatched_inputs"]
    assert any(flag["code"] == "recorder_unavailable" for flag in report["red_flags"])
    assert any(
        flag["code"] == "transformer_prediction_inputs_unmatched"
        for flag in report["red_flags"]
    )
    assert any(flag["code"] == "paper_db_inputs_unmatched" for flag in report["red_flags"])


def test_overnight_health_report_red_flags_stale_btc_price(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    _create_recorder_db(recorder_db)
    conn = sqlite3.connect(recorder_db)
    conn.executescript(
        """
        CREATE TABLE btc_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            price REAL,
            exchange_timestamp TEXT,
            local_arrival_iso TEXT,
            local_arrival_ns INTEGER
        );
        INSERT INTO btc_prices (
            source, price, exchange_timestamp, local_arrival_iso, local_arrival_ns
        ) VALUES (
            'polymarket_rtds_chainlink',
            42000.0,
            '2026-01-01T00:09:00+00:00',
            '2026-01-01T00:09:00+00:00',
            1767226140000000000
        );
        """
    )
    conn.commit()
    conn.close()

    report = build_overnight_health_report(
        recorder_db_path=str(recorder_db),
        prediction_db_inputs=[],
        paper_db_globs=[],
        now=NOW,
    )

    btc_summary = report["recorder"]["btc_prices"]
    assert btc_summary["sources"]["polymarket_rtds_chainlink"]["age_sec"] == 60.0
    assert btc_summary["stale_sources"] == ["polymarket_rtds_chainlink"]
    assert any(flag["code"] == "stale_btc_price" for flag in report["red_flags"])
    rendered = render_overnight_health_report(report)
    assert "btc_prices" in rendered
    assert "polymarket_rtds_chainlink=42000" in rendered


def test_overnight_health_report_cli_writes_json(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "prediction.db"
    paper_db = tmp_path / "paper.db"
    output_json = tmp_path / "health.json"
    _create_recorder_db(recorder_db)
    _create_prediction_db(prediction_db)
    _create_paper_db(paper_db)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "overnight-health-report",
            "--recorder-db",
            str(recorder_db),
            "--prediction-db",
            str(prediction_db),
            "--paper-db-glob",
            str(paper_db),
            "--output-json",
            str(output_json),
        ],
    )

    main_module.main()

    stdout = capsys.readouterr().out
    assert "Overnight Health Report" in stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["recorder"]["feature_rows"] == 3
    assert payload["transformer_predictions"]["aggregate"]["probability_rows"] == 3


def test_summarize_disk_wal_reports_sidecar_sizes(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    recorder_db.write_bytes(b"db")
    Path(f"{recorder_db}-wal").write_bytes(b"wal")
    Path(f"{recorder_db}-shm").write_bytes(b"shm")

    summary = summarize_disk_wal(str(recorder_db))

    assert summary["db_bytes"] == 2
    assert summary["wal_bytes"] == 3
    assert summary["shm_bytes"] == 3
    assert summary["wal_size_warnings"] == []
