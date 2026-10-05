from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from src.recorder_archive_export import export_recorder_archive_to_parquet


NOW = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


def _create_archive_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            question TEXT,
            start_time TEXT,
            close_time TEXT
        );
        CREATE TABLE market_snapshots (
            run_id TEXT,
            timestamp TEXT,
            market_id TEXT,
            yes_price REAL,
            no_price REAL
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
            price REAL
        );
        CREATE TABLE btc_prices (
            run_id TEXT,
            source TEXT,
            price REAL,
            local_arrival_iso TEXT
        );
        INSERT INTO markets VALUES
            ('run_keep', 'm1', 'BTC up?', '2026-01-01T00:00:00+00:00', '2026-01-01T00:05:00+00:00'),
            ('run_skip', 'm2', 'BTC up?', '2026-01-01T00:05:00+00:00', '2026-01-01T00:10:00+00:00');
        INSERT INTO market_snapshots VALUES
            ('run_keep', '2026-01-01T00:01:00+00:00', 'm1', 0.51, 0.49),
            ('run_skip', '2026-01-01T00:06:00+00:00', 'm2', 0.52, 0.48);
        INSERT INTO features VALUES
            ('run_keep', '2026-01-01T00:01:00+00:00', 'm1', 1),
            ('run_skip', '2026-01-01T00:06:00+00:00', 'm2', 1);
        INSERT INTO trades VALUES
            ('run_keep', '2026-01-01T00:01:05+00:00', 'm1', 't1', 0.51),
            ('run_skip', '2026-01-01T00:06:05+00:00', 'm2', 't2', 0.52);
        INSERT INTO btc_prices VALUES
            ('run_keep', 'chainlink', 100.0, '2026-01-01T00:01:00+00:00'),
            ('run_skip', 'chainlink', 101.0, '2026-01-01T00:06:00+00:00');
        """
    )
    conn.commit()
    conn.close()


def test_export_recorder_archive_to_parquet_writes_manifest_and_tables(tmp_path) -> None:
    archive_db = tmp_path / "archive" / "recorder.db"
    archive_db.parent.mkdir()
    _create_archive_db(archive_db)
    output_dir = tmp_path / "parquet"

    report = export_recorder_archive_to_parquet(
        archive_db_inputs=[str(archive_db)],
        output_dir=str(output_dir),
        run_id="run_keep",
        append=True,
        now=NOW,
    )

    assert report["status"] == "ok"
    assert report["manifest_path"]
    manifest = json.loads(Path(str(report["manifest_path"])).read_text(encoding="utf-8"))
    assert manifest["safety"]["live_trading_enabled"] is False
    assert manifest["safety"]["models_promoted_automatically"] is False
    assert manifest["aggregate_row_counts"] == {
        "btc_prices": 1,
        "features": 1,
        "market_snapshots": 1,
        "markets": 1,
        "trades": 1,
    }
    source = report["sources"][0]
    by_table = {item["table"]: item for item in source["tables"]}
    assert by_table["features"]["row_count"] == 1
    assert by_table["features"]["first_timestamp"] == "2026-01-01T00:01:00+00:00"
    features_path = Path(by_table["features"]["path"])
    assert features_path.exists()
    rows = pq.read_table(features_path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_keep"
    assert rows[0]["timestamp"] == "2026-01-01T00:01:00+00:00"
    assert rows[0]["market_id"] == "m1"
    assert rows[0]["feature_ready"] == 1
    assert rows[0]["source_id"] == source["source_id"]
    assert any("train-baseline-model" in command for command in report["next_commands"])
    assert any("train-transformer-sequence-model" in command for command in report["next_commands"])
    assert not any(command.strip().startswith("promote-research-model-candidate") for command in report["next_commands"])


def test_export_recorder_archive_to_parquet_is_idempotent(tmp_path) -> None:
    archive_db = tmp_path / "recorder.db"
    _create_archive_db(archive_db)
    output_dir = tmp_path / "parquet"

    first = export_recorder_archive_to_parquet(
        archive_db_inputs=[str(archive_db)],
        output_dir=str(output_dir),
        append=True,
        now=NOW,
    )
    second = export_recorder_archive_to_parquet(
        archive_db_inputs=[str(archive_db)],
        output_dir=str(output_dir),
        append=True,
        now=NOW,
    )

    assert first["sources"][0]["status"] == "exported"
    assert second["sources"][0]["status"] == "skipped_existing"
    first_features = {
        item["table"]: item for item in first["sources"][0]["tables"]
    }["features"]["path"]
    second_features = {
        item["table"]: item for item in second["sources"][0]["tables"]
    }["features"]["path"]
    assert first_features == second_features
    assert pq.read_metadata(first_features).num_rows == 2


def test_export_recorder_archive_to_parquet_cli(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    archive_db = tmp_path / "recorder.db"
    output_dir = tmp_path / "parquet"
    output_json = tmp_path / "report.json"
    _create_archive_db(archive_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "export-recorder-archive-to-parquet",
            "--archive-db",
            str(archive_db),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--append",
        ],
    )

    main_module.main()

    stdout = capsys.readouterr().out
    assert "Recorder Archive Parquet Export" in stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["sources"][0]["status"] == "exported"
