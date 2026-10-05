from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from src.transformer_live_inference import ensure_transformer_prediction_schema
from src.transformer_prediction_autopsy import (
    audit_transformer_label_semantics,
    build_transformer_prediction_autopsy_report,
    transformer_calibration_bins,
    load_transformer_prediction_rows,
    replay_transformer_predictions,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _setup_recorder(path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE market_snapshots (
            run_id TEXT,
            market_id TEXT,
            timestamp TEXT,
            best_bid_yes REAL,
            best_ask_yes REAL,
            best_bid_no REAL,
            best_ask_no REAL,
            mid_price_yes REAL,
            mid_price_no REAL,
            spread_yes REAL,
            spread_no REAL,
            last_trade_price REAL,
            total_liquidity REAL
        );
        """
    )
    points = [
        (0, 0.49, 0.51, 0.49, 0.51, 90.0),
        (5, 0.60, 0.62, 0.38, 0.40, 90.0),
        (15, 0.65, 0.67, 0.33, 0.35, 90.0),
    ]
    for seconds, bid_yes, ask_yes, bid_no, ask_no, liquidity in points:
        conn.execute(
            """
            INSERT INTO market_snapshots (
                run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, mid_price_yes, mid_price_no,
                spread_yes, spread_no, last_trade_price, total_liquidity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                "market_1",
                (BASE + timedelta(seconds=seconds)).isoformat(),
                bid_yes,
                ask_yes,
                bid_no,
                ask_no,
                (bid_yes + ask_yes) / 2,
                (bid_no + ask_no) / 2,
                ask_yes - bid_yes,
                ask_no - bid_no,
                bid_yes,
                liquidity,
            ),
        )
    conn.commit()
    conn.close()


def _setup_prediction_db(path) -> None:
    conn = _connect(path)
    ensure_transformer_prediction_schema(conn)
    conn.execute(
        """
        INSERT INTO transformer_predictions (
            created_at, run_id, market_id, timestamp, signal_timestamp, model_path,
            sequence_length, probability_yes, probability_no, latest_feature_timestamp,
            market_start_time, market_close_time, time_until_resolution, yes_price,
            no_price, best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
            mid_price_yes, mid_price_no, spread_yes, spread_no, feature_ready,
            snapshot_quality_status, strict_validation_passed, ready_ratio,
            sequence_rows_available, diagnostics_reason, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            BASE.isoformat(),
            "run_live",
            "market_1",
            BASE.isoformat(),
            BASE.isoformat(),
            "model.pt",
            120,
            0.80,
            0.20,
            BASE.isoformat(),
            BASE.isoformat(),
            (BASE + timedelta(minutes=5)).isoformat(),
            55.0,
            0.50,
            0.50,
            0.49,
            0.50,
            0.49,
            0.50,
            0.50,
            0.50,
            0.01,
            0.01,
            1,
            "ok",
            1,
            1.0,
            120,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()


def test_transformer_prediction_autopsy_replays_share_price_path(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "predictions.db"
    _setup_recorder(recorder_db)
    _setup_prediction_db(prediction_db)
    recorder = _connect(recorder_db)
    predictions = load_transformer_prediction_rows([str(prediction_db)])

    try:
        rows = replay_transformer_predictions(
            recorder,
            predictions,
            horizons_sec=[5],
            near_close_sec=3.0,
            thresholds=[0.75],
            side_policy="follow",
        )
    finally:
        recorder.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["side"] == "YES"
    assert row["side_policy"] == "follow"
    assert row["signal_side"] == "YES"
    assert row["executed_side"] == "YES"
    assert row["threshold"] == 0.75
    assert row["probability_for_signal_side"] == 0.8
    assert row["probability_for_executed_side"] == 0.8
    assert row["entry_price_used"] == 0.50
    assert row["exit_price_used"] == 0.60
    assert row["simulated_pnl_usd"] == 0.2
    assert row["simulated_roi"] == 0.2
    assert row["hit_reason"] == "fixed_horizon"
    assert row["price_path_rows"] == 3
    assert row["time_regime"] == "30-60s"
    assert row["spread_regime"] == "tight"
    assert row["liquidity_regime"] == "thin"


def test_transformer_prediction_autopsy_writes_parquet_and_csv(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "predictions.db"
    output = tmp_path / "autopsy.parquet"
    output_csv = tmp_path / "autopsy.csv"
    _setup_recorder(recorder_db)
    _setup_prediction_db(prediction_db)

    report = build_transformer_prediction_autopsy_report(
        recorder_db_path=str(recorder_db),
        prediction_db_paths=[str(prediction_db)],
        output_path=str(output),
        output_csv_path=str(output_csv),
        horizons_sec=[5],
        near_close_sec=3.0,
        thresholds=[0.75],
        min_summary_n=1,
    )

    assert report["status"] == "ok"
    assert report["prediction_rows_loaded"] == 1
    assert report["rows_evaluated"] == 1
    assert output.exists()
    assert output_csv.exists()
    assert report["summary"]["by_threshold_side_horizon"][0]["total_simulated_pnl"] == 0.2


def test_transformer_prediction_autopsy_supports_fade_policy(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "predictions.db"
    _setup_recorder(recorder_db)
    _setup_prediction_db(prediction_db)
    recorder = _connect(recorder_db)
    predictions = load_transformer_prediction_rows([str(prediction_db)])

    try:
        rows = replay_transformer_predictions(
            recorder,
            predictions,
            horizons_sec=[5],
            near_close_sec=3.0,
            thresholds=[0.75],
            side_policy="fade",
        )
    finally:
        recorder.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["side_policy"] == "fade"
    assert row["signal_side"] == "YES"
    assert row["executed_side"] == "NO"
    assert row["side"] == "NO"
    assert row["probability_for_signal_side"] == 0.8
    assert row["probability_for_executed_side"] == 0.2
    assert row["entry_price_used"] == 0.50
    assert row["exit_price_used"] == 0.38
    assert row["simulated_roi"] == -0.24


def test_transformer_prediction_autopsy_summary_has_concentration_and_min_n(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "predictions.db"
    _setup_recorder(recorder_db)
    _setup_prediction_db(prediction_db)

    report = build_transformer_prediction_autopsy_report(
        recorder_db_path=str(recorder_db),
        prediction_db_paths=[str(prediction_db)],
        output_path=None,
        horizons_sec=[5],
        near_close_sec=3.0,
        thresholds=[0.75],
        side_policy="both",
        min_summary_n=1,
    )

    summary = report["summary"]["by_policy_signal_executed_threshold_horizon"]
    assert len(summary) == 2
    first = summary[0]
    assert "distinct_markets" in first
    assert "top_market_pnl" in first
    assert "top_market_pnl_share_abs" in first
    assert "top_3_market_pnl_share_abs" in first
    filtered = build_transformer_prediction_autopsy_report(
        recorder_db_path=str(recorder_db),
        prediction_db_paths=[str(prediction_db)],
        output_path=None,
        horizons_sec=[5],
        thresholds=[0.75],
        side_policy="both",
        min_summary_n=2,
    )
    assert filtered["summary"]["by_policy_signal_executed_threshold_horizon"] == []


def test_transformer_prediction_autopsy_calibration_bins(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "predictions.db"
    _setup_recorder(recorder_db)
    _setup_prediction_db(prediction_db)
    recorder = _connect(recorder_db)
    predictions = load_transformer_prediction_rows([str(prediction_db)])

    try:
        rows = replay_transformer_predictions(
            recorder,
            predictions,
            horizons_sec=[5],
            near_close_sec=3.0,
            thresholds=[0.75],
            side_policy="both",
        )
    finally:
        recorder.close()

    bins = transformer_calibration_bins(predictions, rows)
    bucket = next(item for item in bins if item["probability_yes_bin"] == "0.8-0.9")
    assert bucket["count"] == 1
    assert bucket["distinct_markets"] == 1
    assert bucket["avg_probability_yes"] == 0.8
    assert bucket["horizons"]["5"]["follow_yes_count"] == 1
    assert bucket["horizons"]["5"]["avg_follow_yes_roi"] == 0.2
    assert bucket["horizons"]["5"]["fade_yes_count"] == 1
    assert bucket["horizons"]["5"]["avg_fade_yes_roi"] == -0.24


def test_transformer_label_semantics_proven_from_training_config(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.pt"
    model_path.write_bytes(b"not-used")
    (model_dir / "training_config.json").write_text(
        '{"label_column": "label_yes_win"}',
        encoding="utf-8",
    )

    report = audit_transformer_label_semantics([{"model_path": str(model_path)}])

    assert report["status"] == "ok"
    assert report["label_yes_win_positive_class"] == "YES/UP outcome"
    assert report["warnings"] == []
