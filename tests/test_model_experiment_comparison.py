from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.model_experiment_comparison import compare_model_experiments
from src.transformer_live_inference import ensure_transformer_prediction_schema


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _setup_recorder(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            close_time TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO markets (run_id, market_id, close_time) VALUES (?, ?, ?)",
        ("run_live", "market_1", (BASE + timedelta(minutes=5)).isoformat()),
    )
    conn.commit()
    conn.close()


def _setup_paper_experiment(path: Path) -> Path:
    paper_dir = path / "paper_dbs"
    paper_dir.mkdir(parents=True)
    paper_db = paper_dir / "worker.db"
    conn = _connect(paper_db)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            market_id TEXT,
            strategy_id TEXT,
            status TEXT,
            skip_reason TEXT,
            stake_usd REAL,
            pnl_usd REAL,
            roi REAL,
            realized_pnl_usd REAL,
            realized_roi REAL,
            exit_type TEXT,
            exit_reason TEXT,
            fixed_horizon_exit_sec INTEGER,
            threshold_used REAL,
            time_until_resolution REAL,
            time_regime TEXT,
            liquidity_regime TEXT,
            spread_regime TEXT,
            bankroll_status TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT,
            strategy_id TEXT,
            decision TEXT,
            rejection_reason TEXT,
            threshold_used REAL,
            min_time_until_resolution_sec REAL,
            max_time_until_resolution_sec REAL
        );
        """
    )
    trades = [
        (
            BASE.isoformat(), "m1", "s1", "closed", None, 1.0, None, None,
            0.2, 0.2, "FIXED_HORIZON_EXIT", "fixed_horizon", 15, 0.85,
            45.0, "30-60s", "normal", "tight", "ACTIVE",
        ),
        (
            (BASE + timedelta(seconds=1)).isoformat(), "m2", "s1", "closed", None,
            1.0, None, None, -0.1, -0.1, "FIXED_HORIZON_EXIT",
            "fixed_horizon", 15, 0.85, 50.0, "30-60s", "thin", "wide", "ACTIVE",
        ),
        (
            (BASE + timedelta(seconds=2)).isoformat(), "m3", "s2", "settled", None,
            1.0, 0.5, 0.5, None, None, "HOLD_TO_RESOLUTION", None, None,
            0.90, 80.0, "60-120s", "deep", "tight", "ACTIVE",
        ),
        (
            (BASE + timedelta(seconds=3)).isoformat(), "m4", "s2", "skipped",
            "threshold", 1.0, None, None, None, None, None, None, None,
            0.90, 75.0, "60-120s", "deep", "tight", "ACTIVE",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, market_id, strategy_id, status, skip_reason, stake_usd,
            pnl_usd, roi, realized_pnl_usd, realized_roi, exit_type, exit_reason,
            fixed_horizon_exit_sec, threshold_used, time_until_resolution, time_regime,
            liquidity_regime, spread_regime, bankroll_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        trades,
    )
    conn.executemany(
        """
        INSERT INTO paper_trade_candidates (
            market_id, strategy_id, decision, rejection_reason, threshold_used,
            min_time_until_resolution_sec, max_time_until_resolution_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("m1", "s1", "TRADE", None, 0.85, 30.0, 60.0),
            ("m2", "s1", "TRADE", None, 0.85, 30.0, 60.0),
            ("m4", "s2", "SKIP", "threshold", 0.90, 60.0, 120.0),
        ],
    )
    conn.commit()
    conn.close()
    manifest = {
        "experiment_id": "exp_test",
        "recorder_db_path": "recorder.db",
        "model_path": "model.joblib",
        "feature_columns_path": "feature_columns.json",
        "workers": [
            {
                "paper_db_path": str(paper_db),
                "config_path": "config.json",
                "run_id": "exp_test_worker",
            }
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return paper_db


def _setup_prediction_db(path: Path) -> None:
    conn = _connect(path)
    ensure_transformer_prediction_schema(conn)
    conn.execute(
        """
        INSERT INTO transformer_predictions (
            created_at, run_id, market_id, timestamp, signal_timestamp, model_path,
            sequence_length, probability_yes, probability_no, latest_feature_timestamp,
            market_start_time, market_close_time, time_until_resolution,
            sequence_rows_available
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            BASE.isoformat(), "run_live", "m1", BASE.isoformat(), BASE.isoformat(),
            "model.pt", 60, 0.2, 0.8, BASE.isoformat(), BASE.isoformat(),
            (BASE + timedelta(minutes=5)).isoformat(), 45.0, 60,
        ),
    )
    conn.commit()
    conn.close()


def _write_autopsy_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "prediction_id": 1,
            "market_id": "m1",
            "side": "NO",
            "threshold": 0.75,
            "horizon_sec": 15,
            "simulated_pnl_usd": 0.4,
            "simulated_roi": 0.4,
            "hit_reason": "fixed_horizon",
            "time_regime": "30-60s",
            "liquidity_regime": "normal",
            "spread_regime": "tight",
            "signal_timestamp": BASE.isoformat(),
        },
        {
            "prediction_id": 2,
            "market_id": "m2",
            "side": "NO",
            "threshold": 0.75,
            "horizon_sec": 15,
            "simulated_pnl_usd": -0.2,
            "simulated_roi": -0.2,
            "hit_reason": "fixed_horizon",
            "time_regime": "30-60s",
            "liquidity_regime": "thin",
            "spread_regime": "wide",
            "signal_timestamp": (BASE + timedelta(seconds=1)).isoformat(),
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_compare_model_experiments_writes_reports_and_compares(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    experiment_dir = tmp_path / "experiment"
    prediction_db = tmp_path / "predictions.db"
    autopsy = tmp_path / "autopsy.parquet"
    output = tmp_path / "comparison.json"
    output_txt = tmp_path / "comparison.txt"
    output_csv = tmp_path / "comparison.csv"
    _setup_recorder(recorder_db)
    _setup_paper_experiment(experiment_dir)
    _setup_prediction_db(prediction_db)
    _write_autopsy_parquet(autopsy)

    report = compare_model_experiments(
        recorder_db_path=str(recorder_db),
        paper_experiment_dir=str(experiment_dir),
        transformer_prediction_db_paths=[str(prediction_db)],
        transformer_autopsy_path=str(autopsy),
        output_path=str(output),
        output_txt_path=str(output_txt),
        output_csv_path=str(output_csv),
        now=BASE + timedelta(seconds=60),
    )

    assert report["status"] == "ok"
    assert report["input_health"]["recorder_db"]["query_only"] == 1
    assert report["input_health"]["paper_experiment"]["status"] == "ok"
    assert report["input_health"]["transformer_prediction_dbs"][0]["status"] == "ok"
    assert report["gb_paper_strategy"]["overall"]["total_pnl"] == 0.6
    assert report["gb_paper_strategy"]["fixed_horizon"]["total_pnl"] == 0.1
    assert report["transformer_autopsy"]["overall"]["total_pnl"] == 0.2
    assert report["comparison"]["pnl_difference_transformer_minus_gb_fixed_horizon"] == 0.1
    by_strategy = {
        row["strategy_id"]: row
        for row in report["gb_paper_strategy"]["by_strategy"]
    }
    assert by_strategy["s1"]["total_pnl"] == 0.1
    assert by_strategy["s2"]["total_pnl"] == 0.5
    by_transformer = report["transformer_autopsy"]["by_side_threshold_horizon"][0]
    assert by_transformer["side"] == "NO"
    assert by_transformer["threshold"] == 0.75
    assert by_transformer["horizon_sec"] == 15
    assert output.exists()
    assert output_txt.exists()
    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["section"] == "gb_paper_strategy" for row in rows)
    assert any(row["section"] == "transformer_autopsy" for row in rows)


def test_compare_model_experiments_reports_empty_and_stale_inputs(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    experiment_dir = tmp_path / "experiment"
    prediction_db = tmp_path / "empty_predictions.db"
    autopsy_csv = tmp_path / "autopsy.csv"
    _setup_recorder(recorder_db)
    _setup_paper_experiment(experiment_dir)
    conn = _connect(prediction_db)
    ensure_transformer_prediction_schema(conn)
    conn.close()
    autopsy_csv.write_text(
        "market_id,side,threshold,horizon_sec,simulated_pnl_usd,simulated_roi,signal_timestamp\n"
        "old,YES,0.7,15,0.1,0.1,2020-01-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    report = compare_model_experiments(
        recorder_db_path=str(recorder_db),
        paper_experiment_dir=str(experiment_dir),
        transformer_prediction_db_paths=[str(prediction_db)],
        transformer_autopsy_path=str(autopsy_csv),
        now=BASE + timedelta(days=1),
    )

    assert report["input_health"]["transformer_prediction_dbs"][0]["status"] == "empty"
    assert report["input_health"]["transformer_autopsy"]["status"] == "stale"
    assert report["transformer_autopsy"]["overall"]["count"] == 1


def test_compare_model_experiments_cli_smoke(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    recorder_db = tmp_path / "recorder.db"
    experiment_dir = tmp_path / "experiment"
    prediction_db = tmp_path / "predictions.db"
    autopsy = tmp_path / "autopsy.parquet"
    output = tmp_path / "comparison.json"
    _setup_recorder(recorder_db)
    _setup_paper_experiment(experiment_dir)
    _setup_prediction_db(prediction_db)
    _write_autopsy_parquet(autopsy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "compare-model-experiments",
            "--recorder-db",
            str(recorder_db),
            "--paper-experiment-dir",
            str(experiment_dir),
            "--transformer-prediction-db",
            str(prediction_db),
            "--transformer-autopsy-parquet",
            str(autopsy),
            "--output",
            str(output),
            "--output-txt",
            str(tmp_path / "comparison.txt"),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["output_path"] == str(output)
    assert output.exists()
