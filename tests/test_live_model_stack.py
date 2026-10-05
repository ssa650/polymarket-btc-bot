from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import joblib

from src import main
from src.live_model_stack import (
    LoadedTransformer,
    _open_recorder_readonly,
    build_live_model_dashboard_report,
    connect_prediction_db,
    load_rf_live_model,
    load_transformer_live_artifacts,
    render_live_model_dashboard,
    run_live_model_predictions_once,
    score_live_model_predictions,
)


BASE = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)


class ConstantProbModel:
    def __init__(self, probability_yes: float) -> None:
        self.probability_yes = float(probability_yes)

    def predict_proba(self, matrix):
        return [[1.0 - self.probability_yes, self.probability_yes] for _ in matrix]


def _setup_recorder_db(path, *, now: datetime = BASE, btc_age_sec: float = 1.0, feature_age_sec: float = 1.0, rows: int = 1) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            question TEXT,
            start_time TEXT,
            close_time TEXT,
            phase TEXT,
            resolved INTEGER DEFAULT 0,
            winning_outcome TEXT
        );
        CREATE TABLE features (
            run_id TEXT,
            market_id TEXT,
            timestamp TEXT,
            signal REAL,
            feature_ready INTEGER,
            is_gap_affected INTEGER,
            snapshot_quality_status TEXT,
            strict_validation_passed INTEGER,
            time_until_resolution REAL
        );
        CREATE TABLE market_snapshots (
            run_id TEXT,
            market_id TEXT,
            timestamp TEXT,
            best_bid_yes REAL,
            best_ask_yes REAL,
            best_bid_no REAL,
            best_ask_no REAL,
            mid_price_yes REAL,
            mid_price_no REAL
        );
        CREATE TABLE btc_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            source TEXT,
            price REAL,
            exchange_timestamp TEXT,
            local_arrival_ns INTEGER,
            local_arrival_iso TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO markets (
            run_id, market_id, question, start_time, close_time, phase, resolved, winning_outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run_live",
            "m1",
            "BTC up?",
            (now - timedelta(minutes=1)).isoformat(),
            (now + timedelta(minutes=1)).isoformat(),
            "active",
            0,
            None,
        ),
    )
    start_ts = now - timedelta(seconds=feature_age_sec + max(0, rows - 1))
    for index in range(rows):
        ts = start_ts + timedelta(seconds=index)
        conn.execute(
            """
            INSERT INTO features (
                run_id, market_id, timestamp, signal, feature_ready, is_gap_affected,
                snapshot_quality_status, strict_validation_passed, time_until_resolution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                "m1",
                ts.isoformat(),
                float(index + 1),
                1,
                0,
                "ok",
                1,
                60.0 - index,
            ),
        )
        conn.execute(
            """
            INSERT INTO market_snapshots (
                run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, mid_price_yes, mid_price_no
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                "m1",
                ts.isoformat(),
                0.58,
                0.60,
                0.40,
                0.42,
                0.59,
                0.41,
            ),
        )
    btc_ts = now - timedelta(seconds=btc_age_sec)
    conn.execute(
        """
        INSERT INTO btc_prices (
            run_id, source, price, exchange_timestamp, local_arrival_ns, local_arrival_iso
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "run_live",
            "polymarket_rtds_chainlink",
            100000.0,
            btc_ts.isoformat(),
            int(btc_ts.timestamp() * 1_000_000_000),
            btc_ts.isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def _write_rf_artifacts(tmp_path, probability_yes: float = 0.8):
    model_path = tmp_path / "rf.joblib"
    feature_path = tmp_path / "feature_columns.json"
    joblib.dump(ConstantProbModel(probability_yes), model_path)
    feature_path.write_text(json.dumps(["signal"]), encoding="utf-8")
    return model_path, feature_path


def _write_transformer_artifacts(tmp_path):
    model_dir = tmp_path / "transformer"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"unused")
    (model_dir / "feature_columns.json").write_text(json.dumps(["signal"]), encoding="utf-8")
    (model_dir / "scaler_stats.json").write_text(
        json.dumps({"mean": [0.0], "std": [1.0]}),
        encoding="utf-8",
    )
    (model_dir / "training_config.json").write_text(
        json.dumps({"sequence_length": 3}),
        encoding="utf-8",
    )
    return model_dir


def test_prediction_rows_are_written_for_rf(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db)
    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)

    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    rf_model = load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path))
    try:
        heartbeat = run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=rf_model,
            transformer=None,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()

    conn = sqlite3.connect(output_db)
    row = conn.execute("SELECT * FROM live_model_predictions").fetchone()
    conn.close()
    assert heartbeat["status"] == "healthy"
    assert row is not None
    assert row[2] == "random_forest"


def test_transformer_gracefully_skips_when_sequence_unavailable(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db, rows=1)
    transformer_dir = _write_transformer_artifacts(tmp_path)
    artifacts = load_transformer_live_artifacts(
        model_path=str(transformer_dir / "model.pt"),
        feature_columns_path=str(transformer_dir / "feature_columns.json"),
        scaler_stats_path=str(transformer_dir / "scaler_stats.json"),
        training_config_path=str(transformer_dir / "training_config.json"),
    )
    transformer = LoadedTransformer(
        artifacts=artifacts,
        sequence_length=3,
        inference_fn=lambda _sequence, _artifacts: 0.7,
    )

    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        heartbeat = run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=None,
            transformer=transformer,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()

    assert heartbeat["status"] == "model_unavailable"
    assert heartbeat["model_errors"][0]["reason"] == "insufficient_clean_sequence_rows"
    conn = sqlite3.connect(output_db)
    count = conn.execute("SELECT COUNT(*) FROM live_model_predictions").fetchone()[0]
    conn.close()
    assert count == 0


def test_stale_btc_blocks_healthy_prediction(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db, btc_age_sec=30.0)
    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)

    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        heartbeat = run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path)),
            transformer=None,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()

    assert heartbeat["status"] == "stale_or_not_ready"
    assert "btc_stale" in heartbeat["stale_flags"]
    conn = sqlite3.connect(output_db)
    assert conn.execute("SELECT COUNT(*) FROM live_model_predictions").fetchone()[0] == 0
    conn.close()


def test_stale_recorder_blocks_healthy_prediction(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db, feature_age_sec=30.0)
    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)

    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        heartbeat = run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path)),
            transformer=None,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()

    assert heartbeat["status"] == "stale_or_not_ready"
    assert "feature_stale" in heartbeat["stale_flags"]


def test_scoring_resolves_prediction_accuracy(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db)
    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)
    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path)),
            transformer=None,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()
    conn = sqlite3.connect(recorder_db)
    conn.execute("UPDATE markets SET resolved = 1, winning_outcome = 'Up'")
    conn.commit()
    conn.close()

    report = score_live_model_predictions(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(output_db),
        lookback_hours=48,
        now=BASE,
    )

    conn = sqlite3.connect(output_db)
    row = conn.execute("SELECT was_correct FROM live_model_accuracy").fetchone()
    conn.close()
    assert report["predictions_scored"] == 1
    assert row[0] == 1


def test_dashboard_renders_without_predictions_and_with_predictions(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db)
    connect_prediction_db(str(output_db)).close()

    empty_report = build_live_model_dashboard_report(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(output_db),
        now=BASE,
    )
    empty_text = render_live_model_dashboard(empty_report)
    assert "Live Model Dashboard [SHADOW]" in empty_text
    assert "no_model_predictions" in empty_text

    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)
    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        run_live_model_predictions_once(
            recorder_conn=recorder_conn,
            output_conn=output_conn,
            rf_model=load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path)),
            transformer=None,
            poll_sec=1.0,
            max_feature_age_sec=5.0,
            max_btc_age_sec=15.0,
            min_confidence=0.55,
            run_id="test_run",
            mode="SHADOW",
            now=BASE,
        )
    finally:
        recorder_conn.close()
        output_conn.close()
    text = render_live_model_dashboard(
        build_live_model_dashboard_report(
            recorder_db_path=str(recorder_db),
            prediction_db_path=str(output_db),
            now=BASE,
        )
    )
    assert "random_forest" in text
    assert "YES=0.800" in text


def test_no_duplicate_predictions_for_same_model_market_feature(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "live_predictions.db"
    _setup_recorder_db(recorder_db)
    model_path, feature_path = _write_rf_artifacts(tmp_path, 0.8)
    rf_model = load_rf_live_model(model_path=str(model_path), feature_columns_path=str(feature_path))

    recorder_conn = _open_recorder_readonly(str(recorder_db))
    output_conn = connect_prediction_db(str(output_db))
    try:
        for _ in range(2):
            run_live_model_predictions_once(
                recorder_conn=recorder_conn,
                output_conn=output_conn,
                rf_model=rf_model,
                transformer=None,
                poll_sec=1.0,
                max_feature_age_sec=5.0,
                max_btc_age_sec=15.0,
                min_confidence=0.55,
                run_id="test_run",
                mode="SHADOW",
                now=BASE,
            )
    finally:
        recorder_conn.close()
        output_conn.close()

    conn = sqlite3.connect(output_db)
    assert conn.execute("SELECT COUNT(*) FROM live_model_predictions").fetchone()[0] == 1
    conn.close()


def test_live_model_stack_cli_defaults_to_shadow_and_live_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        main.sys,
        "argv",
        [
            "prog",
            "run-live-model-predictions",
            "--recorder-db",
            "data/recorder.db",
            "--output-db",
            "data/live_predictions.db",
        ],
    )

    args = main.parse_args()

    assert args.command == "run-live-model-predictions"
    assert args.shadow_only is True
    assert args.enable_live_trading is False
