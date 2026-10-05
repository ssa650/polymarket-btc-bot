from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.trade_autopsy import build_trade_autopsy_report, export_meta_trade_dataset


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _iso(seconds: int) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _setup_recorder(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE market_snapshots (
            run_id TEXT,
            market_id TEXT,
            timestamp TEXT,
            best_bid_yes REAL,
            best_bid_no REAL,
            mid_price_yes REAL,
            mid_price_no REAL,
            last_trade_price REAL
        );
        """
    )
    rows = [
        ("run_meta", "m1", _iso(0), 0.50, 0.50, 0.51, 0.49, 0.50),
        ("run_meta", "m1", _iso(5), 0.67, 0.33, 0.68, 0.34, 0.67),
        ("run_meta", "m2", _iso(0), 0.50, 0.50, 0.51, 0.49, 0.50),
        ("run_meta", "m2", _iso(5), 0.40, 0.60, 0.41, 0.61, 0.40),
    ]
    conn.executemany(
        """
        INSERT INTO market_snapshots (
            run_id, market_id, timestamp, best_bid_yes, best_bid_no,
            mid_price_yes, mid_price_no, last_trade_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _setup_paper(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            run_id TEXT,
            market_id TEXT,
            signal_timestamp TEXT,
            market_start_time TEXT,
            market_close_time TEXT,
            status TEXT,
            signal_direction TEXT,
            strategy_id TEXT,
            model_name TEXT,
            predicted_probability_yes REAL,
            probability_for_direction REAL,
            estimated_edge REAL,
            time_until_resolution REAL,
            time_regime TEXT,
            btc_trend_regime TEXT,
            volatility_regime TEXT,
            spread_regime TEXT,
            liquidity_regime TEXT,
            entry_price REAL,
            adjusted_entry_price REAL,
            stake_usd REAL,
            feature_ready INTEGER,
            strict_validation_passed INTEGER,
            snapshot_quality_status TEXT,
            requested_shares REAL,
            max_fillable_shares REAL,
            liquidity_fill_fraction_used REAL,
            spread_cents_at_entry REAL,
            entry_latency_sec REAL,
            entry_price_drift REAL
        );
        """
    )
    rows = [
        (
            _iso(0), "run_meta", "m1", _iso(0), _iso(0), _iso(30), "closed", "YES",
            "strategy_a", "gb", 0.8, 0.8, 0.3, 30.0, "0-30s", "weak_up", "low",
            "tight", "normal", 0.52, 0.50, 1.0, 1, 1, "ok", 2.0, 10.0, 0.2, 1.0, 1.0, 0.0,
        ),
        (
            _iso(0), "run_meta", "m2", _iso(0), _iso(0), _iso(30), "closed", "YES",
            "strategy_a", "gb", 0.7, 0.7, 0.2, 30.0, "0-30s", "weak_down", "low",
            "tight", "normal", 0.52, 0.50, 1.0, 1, 1, "ok", 2.0, 10.0, 0.2, 1.0, 1.0, 0.0,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, run_id, market_id, signal_timestamp, market_start_time,
            market_close_time, status, signal_direction, strategy_id, model_name,
            predicted_probability_yes, probability_for_direction, estimated_edge,
            time_until_resolution, time_regime, btc_trend_regime, volatility_regime,
            spread_regime, liquidity_regime, entry_price, adjusted_entry_price,
            stake_usd, feature_ready, strict_validation_passed, snapshot_quality_status,
            requested_shares, max_fillable_shares, liquidity_fill_fraction_used,
            spread_cents_at_entry, entry_latency_sec, entry_price_drift
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _read_parquet(path):
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(path).to_pylist()]


def test_export_meta_trade_dataset_labels_from_autopsy_parquet(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    autopsy = tmp_path / "autopsy.parquet"
    output = tmp_path / "meta.parquet"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    build_trade_autopsy_report(
        recorder_db_path=str(recorder_db),
        paper_db_paths=[str(paper_db)],
        output_path=str(autopsy),
        fixed_horizons_sec=[],
        stop_loss_take_profit=[(-0.2, 0.3)],
        near_close_sec=3,
    )

    report = export_meta_trade_dataset(
        autopsy_input_path=str(autopsy),
        output_path=str(output),
        label_policy="stop_loss_20_take_profit_30",
        label_column="label_positive_roi",
        roi_threshold=0.0,
    )

    assert report["status"] == "ok"
    assert report["rows_exported"] == 2
    rows = _read_parquet(output)
    labels = {row["market_id"]: row["label_value"] for row in rows}
    assert labels == {"m1": 1, "m2": 0}
    assert rows[0]["strategy_id"] == "strategy_a"
    assert "simulated_roi" in rows[0]


def test_export_meta_trade_dataset_can_run_autopsy_internally(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    output = tmp_path / "meta_internal.parquet"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)

    report = export_meta_trade_dataset(
        autopsy_input_path=None,
        recorder_db_path=str(recorder_db),
        paper_db_paths=[str(paper_db)],
        output_path=str(output),
        label_policy="stop_loss_20_take_profit_30",
        label_column="label_roi_above_threshold",
        roi_threshold=0.1,
        fixed_horizons_sec=[],
        stop_loss_take_profit=[(-0.2, 0.3)],
        near_close_sec=3,
    )

    rows = _read_parquet(output)
    assert report["rows_exported"] == 2
    assert {row["label_column"] for row in rows} == {"label_roi_above_threshold"}
    assert {row["label_value"] for row in rows} == {0, 1}

