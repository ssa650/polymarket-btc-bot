from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.transformer_live_inference import (
    analyze_transformer_calibration,
    apply_scaler_to_sequence,
    backtest_transformer_threshold_strategy,
    build_latest_live_sequence,
    ensure_transformer_prediction_schema,
    load_transformer_live_artifacts,
    report_transformer_predictions,
    run_transformer_paper_trader,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _setup_db(
    path,
    *,
    rows: int = 3,
    gap_index: int | None = None,
    not_ready_indices: set[int] | None = None,
    null_signal_indices: set[int] | None = None,
    strict_failed_indices: set[int] | None = None,
) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            question TEXT,
            start_time TEXT,
            close_time TEXT,
            phase TEXT,
            yes_token_id TEXT,
            no_token_id TEXT
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
            has_orderbook INTEGER,
            has_trade_data INTEGER
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
            run_id, market_id, question, start_time, close_time, phase, yes_token_id, no_token_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run_live",
            "market_1",
            "BTC up?",
            BASE.isoformat(),
            (BASE + timedelta(minutes=5)).isoformat(),
            "active",
            "yes",
            "no",
        ),
    )
    for index in range(rows):
        timestamp = BASE + timedelta(seconds=index)
        conn.execute(
            """
            INSERT INTO features (
                run_id, market_id, timestamp, signal, feature_ready, is_gap_affected,
                snapshot_quality_status, strict_validation_passed, time_until_resolution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                "market_1",
                timestamp.isoformat(),
                None if null_signal_indices and index in null_signal_indices else float(index),
                0 if not_ready_indices and index in not_ready_indices else 1,
                1 if gap_index == index else 0,
                "ok",
                0 if strict_failed_indices and index in strict_failed_indices else 1,
                300.0 - index,
            ),
        )
        conn.execute(
            """
            INSERT INTO market_snapshots (
                run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, has_orderbook, has_trade_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                "market_1",
                timestamp.isoformat(),
                0.50 + index * 0.01,
                0.52 + index * 0.01,
                0.48 - index * 0.01,
                0.50 - index * 0.01,
                1,
                1,
            ),
        )
        for source, price in (
            ("polymarket_rtds_chainlink", 100.0 + index),
            ("polymarket_rtds_binance", 100.5 + index),
        ):
            conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, exchange_timestamp, local_arrival_ns, local_arrival_iso
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_live",
                    source,
                    price,
                    timestamp.isoformat(),
                    index + 1,
                    timestamp.isoformat(),
                ),
            )
    conn.commit()
    conn.close()


def _write_artifacts(tmp_path, *, feature_columns: list[str] | None = None):
    model_path = tmp_path / "model.pt"
    feature_path = tmp_path / "feature_columns.json"
    scaler_path = tmp_path / "scaler_stats.json"
    config_path = tmp_path / "training_config.json"
    features = feature_columns or ["signal"]
    model_path.write_bytes(b"not-used-by-mocked-inference")
    feature_path.write_text(json.dumps(features), encoding="utf-8")
    scaler_path.write_text(
        json.dumps({"mean": [1.0] * len(features), "std": [1.0] * len(features)}),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps({"sequence_length": 3}), encoding="utf-8")
    return model_path, feature_path, scaler_path, config_path


def _setup_resolution_db(path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            resolved INTEGER,
            winning_outcome TEXT,
            winning_asset_id TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            close_time TEXT
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO markets (
            run_id, market_id, resolved, winning_outcome,
            winning_asset_id, yes_token_id, no_token_id, close_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "run_live",
                "market_1",
                1,
                "Up",
                None,
                "yes_1",
                "no_1",
                (BASE + timedelta(minutes=5)).isoformat(),
            ),
            (
                "run_live",
                "market_2",
                1,
                "Down",
                None,
                "yes_2",
                "no_2",
                (BASE + timedelta(minutes=5)).isoformat(),
            ),
        ],
    )
    conn.commit()
    conn.close()


def _setup_threshold_backtest_recorder(path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            resolved INTEGER,
            winning_outcome TEXT,
            winning_asset_id TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            start_time TEXT,
            close_time TEXT
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
        """
    )
    markets = [
        ("run_live", "m_yes", 1, "Up", None, "yes_1", "no_1", BASE.isoformat(), (BASE + timedelta(minutes=5)).isoformat()),
        ("run_live", "m_no", 1, "Down", None, "yes_2", "no_2", BASE.isoformat(), (BASE + timedelta(minutes=5)).isoformat()),
        ("run_live", "m_settle", 1, "Up", None, "yes_3", "no_3", BASE.isoformat(), (BASE + timedelta(minutes=5)).isoformat()),
    ]
    conn.executemany(
        """
        INSERT INTO markets (
            run_id, market_id, resolved, winning_outcome, winning_asset_id,
            yes_token_id, no_token_id, start_time, close_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        markets,
    )
    snapshots = [
        ("run_live", "m_yes", BASE.isoformat(), 0.73, 0.75, 0.25, 0.27, 0.74, 0.26),
        ("run_live", "m_yes", (BASE + timedelta(seconds=15)).isoformat(), 0.825, 0.84, 0.16, 0.18, 0.8325, 0.17),
        ("run_live", "m_no", BASE.isoformat(), 0.30, 0.32, 0.68, 0.70, 0.31, 0.69),
        ("run_live", "m_no", (BASE + timedelta(seconds=15)).isoformat(), 0.21, 0.23, 0.77, 0.79, 0.22, 0.78),
    ]
    conn.executemany(
        """
        INSERT INTO market_snapshots (
            run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
            best_bid_no, best_ask_no, mid_price_yes, mid_price_no
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snapshots,
    )
    conn.commit()
    conn.close()


def _setup_threshold_prediction_db(path) -> None:
    conn = _connect(path)
    try:
        ensure_transformer_prediction_schema(conn)
        rows = [
            (
                BASE.isoformat(),
                "run_live",
                "m_yes",
                BASE.isoformat(),
                BASE.isoformat(),
                "model.pt",
                60,
                1,
                0.78,
                0.22,
                BASE.isoformat(),
                (BASE + timedelta(minutes=5)).isoformat(),
                90.0,
                0.74,
                0.26,
                0.73,
                0.75,
                0.25,
                0.27,
                0.74,
                0.26,
                "ok",
            ),
            (
                (BASE + timedelta(seconds=1)).isoformat(),
                "run_live",
                "m_no",
                (BASE + timedelta(seconds=1)).isoformat(),
                (BASE + timedelta(seconds=1)).isoformat(),
                "model.pt",
                60,
                1,
                0.22,
                0.78,
                BASE.isoformat(),
                (BASE + timedelta(minutes=5)).isoformat(),
                90.0,
                0.31,
                0.69,
                0.30,
                0.32,
                0.68,
                0.70,
                0.31,
                0.69,
                "ok",
            ),
            (
                (BASE + timedelta(seconds=2)).isoformat(),
                "run_live",
                "m_settle",
                (BASE + timedelta(seconds=2)).isoformat(),
                (BASE + timedelta(seconds=2)).isoformat(),
                "model.pt",
                60,
                1,
                0.79,
                0.21,
                BASE.isoformat(),
                (BASE + timedelta(minutes=5)).isoformat(),
                90.0,
                0.74,
                0.26,
                0.73,
                0.75,
                0.25,
                0.27,
                0.74,
                0.26,
                "ok",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, signal_timestamp,
                model_path, sequence_length, transformer_sequence_ready,
                probability_yes, probability_no, market_start_time,
                market_close_time, time_until_resolution, yes_price, no_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                mid_price_yes, mid_price_no, snapshot_quality_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _setup_threshold_split_inputs(recorder_path, prediction_path) -> None:
    recorder = _connect(recorder_path)
    recorder.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            resolved INTEGER,
            winning_outcome TEXT,
            winning_asset_id TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            start_time TEXT,
            close_time TEXT
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
        """
    )
    train_time = BASE
    test_time = BASE + timedelta(minutes=10)
    recorder.executemany(
        """
        INSERT INTO markets (
            run_id, market_id, resolved, winning_outcome, winning_asset_id,
            yes_token_id, no_token_id, start_time, close_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "run_split",
                "m_train",
                1,
                "Up",
                None,
                "yes_train",
                "no_train",
                (train_time - timedelta(minutes=4)).isoformat(),
                (train_time + timedelta(seconds=80)).isoformat(),
            ),
            (
                "run_split",
                "m_test",
                1,
                "Down",
                None,
                "yes_test",
                "no_test",
                (test_time - timedelta(minutes=4)).isoformat(),
                (test_time + timedelta(seconds=80)).isoformat(),
            ),
        ],
    )
    recorder.executemany(
        """
        INSERT INTO market_snapshots (
            run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
            best_bid_no, best_ask_no, mid_price_yes, mid_price_no
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("run_split", "m_train", train_time.isoformat(), 0.73, 0.75, 0.25, 0.27, 0.74, 0.26),
            ("run_split", "m_train", (train_time + timedelta(seconds=60)).isoformat(), 0.90, 0.92, 0.08, 0.10, 0.91, 0.09),
            ("run_split", "m_test", test_time.isoformat(), 0.73, 0.75, 0.25, 0.27, 0.74, 0.26),
            ("run_split", "m_test", (test_time + timedelta(seconds=60)).isoformat(), 0.50, 0.52, 0.48, 0.50, 0.51, 0.49),
        ],
    )
    recorder.commit()
    recorder.close()

    predictions = _connect(prediction_path)
    try:
        ensure_transformer_prediction_schema(predictions)
        rows = [
            (
                train_time.isoformat(),
                "run_split",
                "m_train",
                train_time.isoformat(),
                train_time.isoformat(),
                "model.pt",
                60,
                1,
                0.76,
                0.24,
                (train_time - timedelta(minutes=4)).isoformat(),
                (train_time + timedelta(seconds=80)).isoformat(),
                80.0,
                0.74,
                0.26,
                0.73,
                0.75,
                0.25,
                0.27,
                0.74,
                0.26,
                "ok",
            ),
            (
                test_time.isoformat(),
                "run_split",
                "m_test",
                test_time.isoformat(),
                test_time.isoformat(),
                "model.pt",
                60,
                1,
                0.76,
                0.24,
                (test_time - timedelta(minutes=4)).isoformat(),
                (test_time + timedelta(seconds=80)).isoformat(),
                80.0,
                0.74,
                0.26,
                0.73,
                0.75,
                0.25,
                0.27,
                0.74,
                0.26,
                "ok",
            ),
        ]
        predictions.executemany(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, signal_timestamp,
                model_path, sequence_length, transformer_sequence_ready,
                probability_yes, probability_no, market_start_time,
                market_close_time, time_until_resolution, yes_price, no_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                mid_price_yes, mid_price_no, snapshot_quality_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        predictions.commit()
    finally:
        predictions.close()


def test_build_latest_live_sequence_requires_sequence_length_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=2)
    conn = _connect(db_path)

    sequence = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        now=BASE + timedelta(seconds=3),
    )

    assert sequence["ready"] is False
    assert sequence["reason"] == "insufficient_sequence_rows"
    assert sequence["sequence_rows_available"] == 2
    assert sequence["sequence_row_policy"] == "latest_rows"
    conn.close()


def test_build_latest_live_sequence_blocks_gap_rows_unless_allowed(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=3, gap_index=1)
    conn = _connect(db_path)

    blocked = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        now=BASE + timedelta(seconds=3),
    )
    allowed = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=True,
        now=BASE + timedelta(seconds=3),
    )

    assert blocked["ready"] is False
    assert blocked["reason"] == "gap_affected"
    assert blocked["sequence_row_policy"] == "latest_rows"
    assert allowed["ready"] is True
    conn.close()


def test_latest_clean_rows_skips_warmup_null_and_gap_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(
        db_path,
        rows=6,
        not_ready_indices={0},
        null_signal_indices={1},
        gap_index=2,
    )
    conn = _connect(db_path)

    latest_rows = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=6,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        now=BASE + timedelta(seconds=6),
    )
    clean_rows = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        sequence_row_policy="latest_clean_rows",
        now=BASE + timedelta(seconds=6),
    )

    assert latest_rows["ready"] is False
    assert latest_rows["reason"] == "not_feature_ready"
    assert clean_rows["ready"] is True
    assert clean_rows["sequence_row_policy"] == "latest_clean_rows"
    assert clean_rows["clean_sequence_rows_available"] == 3
    assert clean_rows["latest_clean_sequence_timestamp"] == (BASE + timedelta(seconds=5)).isoformat()
    assert clean_rows["latest_feature_timestamp"] == (BASE + timedelta(seconds=5)).isoformat()
    assert clean_rows["sequence_values"] == [[3.0], [4.0], [5.0]]
    assert clean_rows["rejected_rows_by_reason"] == {
        "gap_affected": 1,
        "missing_or_null_model_feature": 1,
        "not_feature_ready": 1,
    }
    conn.close()


def test_latest_clean_rows_reports_insufficient_clean_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=4, not_ready_indices={0, 1}, gap_index=2)
    conn = _connect(db_path)

    sequence = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        sequence_row_policy="latest_clean_rows",
        now=BASE + timedelta(seconds=4),
    )

    assert sequence["ready"] is False
    assert sequence["reason"] == "insufficient_clean_sequence_rows"
    assert sequence["clean_sequence_rows_available"] == 1
    assert sequence["rejected_rows_by_reason"] == {
        "gap_affected": 1,
        "not_feature_ready": 2,
    }
    conn.close()


def test_strict_mode_blocks_not_ready_rows_with_diagnostics(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=3, not_ready_indices={0, 1})
    conn = _connect(db_path)

    sequence = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        now=BASE + timedelta(seconds=3),
    )

    assert sequence["ready"] is False
    assert sequence["reason"] == "not_feature_ready"
    assert sequence["total_sequence_rows_checked"] == 3
    assert sequence["feature_ready_false_count"] == 2
    assert sequence["latest_row_feature_ready"] == 1
    assert sequence["earliest_sequence_timestamp"] == BASE.isoformat()
    assert sequence["latest_sequence_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()
    assert sequence["bad_row_samples"][0]["timestamp"] == BASE.isoformat()
    conn.close()


def test_relaxed_mode_allows_not_ready_rows_when_min_ready_ratio_is_met(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=5, not_ready_indices={0})
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)
    config_path.write_text(json.dumps({"sequence_length": 5}), encoding="utf-8")

    report = run_transformer_paper_trader(
        recorder_db_path=str(db_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        max_iterations=1,
        max_feature_age_sec=5.0,
        allow_not_ready_sequence_rows=True,
        min_ready_ratio=0.8,
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.61,
        now=BASE + timedelta(seconds=5),
    )

    heartbeat = report["last_heartbeat"]
    assert heartbeat["transformer_sequence_ready"] is True
    assert heartbeat["probability_yes"] == 0.61
    assert heartbeat["feature_ready_false_count"] == 1
    assert heartbeat["not_ready_rows_allowed"] == 1
    assert heartbeat["ready_ratio"] == 0.8


def test_min_ready_ratio_blocks_when_not_met(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=5, not_ready_indices={0, 1})
    conn = _connect(db_path)

    sequence = build_latest_live_sequence(
        conn,
        feature_columns=["signal"],
        sequence_length=5,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        min_ready_ratio=0.8,
        now=BASE + timedelta(seconds=5),
    )

    assert sequence["ready"] is False
    assert sequence["reason"] == "min_ready_ratio_not_met"
    assert sequence["ready_ratio"] == 0.6
    conn.close()


def test_missing_feature_columns_report_available_columns(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=3)
    conn = _connect(db_path)

    sequence = build_latest_live_sequence(
        conn,
        feature_columns=["signal", "missing_model_feature"],
        sequence_length=3,
        max_feature_age_sec=5.0,
        allow_gap_affected=False,
        now=BASE + timedelta(seconds=3),
    )

    assert sequence["ready"] is False
    assert sequence["reason"] == "missing_feature_columns"
    assert sequence["missing_feature_columns"] == ["missing_model_feature"]
    assert "signal" in sequence["available_feature_table_columns"]
    assert sequence["model_feature_columns_count"] == 2
    assert "best_bid_yes" in sequence["available_sequence_columns"]
    conn.close()


def test_run_transformer_paper_trader_outputs_mocked_probability_and_scaled_sequence(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=3)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)
    seen_sequences: list[list[list[float]]] = []

    def fake_inference(sequence, _artifacts):
        seen_sequences.append(sequence)
        return 0.73

    report = run_transformer_paper_trader(
        recorder_db_path=str(db_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        max_iterations=1,
        max_feature_age_sec=5.0,
        emit_logs=False,
        inference_fn=fake_inference,
        now=BASE + timedelta(seconds=3),
    )

    heartbeat = report["last_heartbeat"]
    assert heartbeat["transformer_sequence_ready"] is True
    assert heartbeat["sequence_rows_available"] == 3
    assert heartbeat["probability_yes"] == 0.73
    assert heartbeat["latest_market_id"] == "market_1"
    assert heartbeat["latest_feature_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()
    assert seen_sequences == [[[-1.0], [0.0], [1.0]]]


def test_run_transformer_paper_trader_records_prediction_rows_per_heartbeat(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    output_db = tmp_path / "transformer_predictions.db"
    _setup_db(db_path, rows=3)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    for _ in range(2):
        run_transformer_paper_trader(
            recorder_db_path=str(db_path),
            model_path=str(model_path),
            feature_columns_path=str(feature_path),
            scaler_stats_path=str(scaler_path),
            training_config_path=str(config_path),
            max_iterations=1,
            output_db_path=str(output_db),
            max_feature_age_sec=5.0,
            emit_logs=False,
            inference_fn=lambda _sequence, _artifacts: 0.73,
            now=BASE + timedelta(seconds=3),
        )

    conn = _connect(output_db)
    try:
        ensure_transformer_prediction_schema(conn)
        rows = conn.execute("SELECT * FROM transformer_predictions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    row = dict(rows[0])
    assert row["market_id"] == "market_1"
    assert row["run_id"] == "run_live"
    assert row["timestamp"] == (BASE + timedelta(seconds=3)).isoformat()
    assert row["latest_feature_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()
    assert row["latest_sequence_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()
    assert row["sequence_row_policy"] == "latest_rows"
    assert row["transformer_sequence_ready"] == 1
    assert row["probability_yes"] == 0.73
    assert row["probability_no"] == 0.27
    assert row["best_bid_yes"] == 0.52
    assert row["best_ask_no"] == 0.48
    assert row["feature_ready"] == 1
    assert row["snapshot_quality_status"] == "ok"


def test_latest_clean_rows_records_prediction_rows_per_heartbeat(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    output_db = tmp_path / "transformer_predictions.db"
    _setup_db(
        db_path,
        rows=6,
        not_ready_indices={0},
        null_signal_indices={1},
        gap_index=2,
    )
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    for _ in range(2):
        run_transformer_paper_trader(
            recorder_db_path=str(db_path),
            model_path=str(model_path),
            feature_columns_path=str(feature_path),
            scaler_stats_path=str(scaler_path),
            training_config_path=str(config_path),
            max_iterations=1,
            output_db_path=str(output_db),
            max_feature_age_sec=5.0,
            sequence_row_policy="latest_clean_rows",
            emit_logs=False,
            inference_fn=lambda _sequence, _artifacts: 0.82,
            now=BASE + timedelta(seconds=6),
        )

    conn = _connect(output_db)
    try:
        rows = conn.execute("SELECT * FROM transformer_predictions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    row = dict(rows[0])
    assert row["latest_feature_timestamp"] == (BASE + timedelta(seconds=5)).isoformat()
    assert row["probability_yes"] == 0.82
    assert row["sequence_rows_available"] == 3
    assert row["sequence_row_policy"] == "latest_clean_rows"


def test_transformer_prediction_logging_records_not_ready_attempts(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    output_db = tmp_path / "transformer_predictions.db"
    _setup_db(db_path, rows=2)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    report = run_transformer_paper_trader(
        recorder_db_path=str(db_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        max_iterations=1,
        output_db_path=str(output_db),
        max_feature_age_sec=5.0,
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.82,
        now=BASE + timedelta(seconds=3),
    )

    conn = _connect(output_db)
    try:
        rows = conn.execute("SELECT * FROM transformer_predictions").fetchall()
    finally:
        conn.close()

    assert report["last_heartbeat"]["transformer_sequence_ready"] is False
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["transformer_sequence_ready"] == 0
    assert row["probability_yes"] is None
    assert row["reason"] == "insufficient_sequence_rows"
    assert row["sequence_rows_available"] == 2


def test_transformer_jsonl_log_file_writes_valid_compact_heartbeat_lines(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    log_path = tmp_path / "logs" / "transformer.jsonl"
    _setup_db(db_path, rows=3)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    run_transformer_paper_trader(
        recorder_db_path=str(db_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        max_iterations=1,
        max_feature_age_sec=5.0,
        jsonl_log_file_path=str(log_path),
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.76,
        now=BASE + timedelta(seconds=3),
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert "\n" not in lines[0]
    payload = json.loads(lines[0])
    assert payload["event"] == "transformer_paper_trader_heartbeat"
    assert payload["timestamp"] == (BASE + timedelta(seconds=3)).isoformat()
    assert payload["market_id"] == "market_1"
    assert payload["sequence_length"] == 3
    assert payload["transformer_sequence_ready"] is True
    assert payload["readiness_status"] == "ready"
    assert payload["probability_yes"] == 0.76
    assert payload["reason"] is None
    assert payload["feature_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()
    assert payload["latest_sequence_timestamp"] == (BASE + timedelta(seconds=2)).isoformat()


def test_transformer_heartbeat_log_file_appends_and_flushes(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    log_path = tmp_path / "transformer_heartbeat.jsonl"
    _setup_db(db_path, rows=3)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    for _ in range(2):
        run_transformer_paper_trader(
            recorder_db_path=str(db_path),
            model_path=str(model_path),
            feature_columns_path=str(feature_path),
            scaler_stats_path=str(scaler_path),
            training_config_path=str(config_path),
            max_iterations=1,
            max_feature_age_sec=5.0,
            heartbeat_log_file_path=str(log_path),
            emit_logs=False,
            inference_fn=lambda _sequence, _artifacts: 0.64,
            now=BASE + timedelta(seconds=3),
        )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    assert len(payloads) == 2
    assert all(payload["probability_yes"] == 0.64 for payload in payloads)
    assert all(payload["market_id"] == "market_1" for payload in payloads)


def test_transformer_prediction_schema_has_required_columns_and_indexes(tmp_path) -> None:
    output_db = tmp_path / "transformer_predictions.db"
    conn = _connect(output_db)
    try:
        ensure_transformer_prediction_schema(conn)
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(transformer_predictions)").fetchall()
        }
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(transformer_predictions)").fetchall()
        }
    finally:
        conn.close()

    assert {
        "timestamp",
        "run_id",
        "market_id",
        "model_path",
        "sequence_length",
        "sequence_row_policy",
        "transformer_sequence_ready",
        "probability_yes",
        "reason",
        "feature_timestamp",
        "latest_feature_timestamp",
        "latest_sequence_timestamp",
        "missing_feature_columns_json",
        "rejected_rows_by_reason_json",
        "ready_ratio",
        "snapshot_quality_status",
    }.issubset(columns)
    assert "idx_transformer_predictions_timestamp" in indexes
    assert "idx_transformer_predictions_market_id" in indexes
    assert "idx_transformer_predictions_run_id" in indexes
    assert "idx_transformer_predictions_sequence_length" in indexes
    assert "idx_transformer_predictions_probability_yes" in indexes


def test_run_transformer_paper_trader_does_not_write_to_recorder_db(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _setup_db(db_path, rows=3)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)
    before = _connect(db_path)
    before_feature_count = before.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    before_tables = {
        row[0]
        for row in before.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    before.close()

    run_transformer_paper_trader(
        recorder_db_path=str(db_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        max_iterations=1,
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.5,
        now=BASE + timedelta(seconds=3),
    )

    after = _connect(db_path)
    after_feature_count = after.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    after_tables = {
        row[0]
        for row in after.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    after.close()
    assert after_feature_count == before_feature_count
    assert after_tables == before_tables


def test_apply_scaler_fills_missing_values_with_train_mean() -> None:
    scaled = apply_scaler_to_sequence(
        [[1.0, None], [3.0, 10.0]],
        {"mean": [1.0, 5.0], "std": [2.0, 5.0]},
    )

    assert scaled == [[0.0, 0.0], [1.0, 1.0]]


def test_transformer_paper_trader_cli_is_registered_and_offline(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import src.main as main_module
    import src.transformer_live_inference as live_module

    def fake_run(**kwargs):
        return {
            "status": "ok",
            "recorder_db_path": kwargs["recorder_db_path"],
            "model_path": kwargs["model_path"],
            "feature_columns_path": kwargs["feature_columns_path"],
            "scaler_stats_path": kwargs["scaler_stats_path"],
            "training_config_path": kwargs["training_config_path"],
            "sequence_length": kwargs["sequence_length"],
            "allow_gap_affected": kwargs["allow_gap_affected"],
            "allow_not_ready_sequence_rows": kwargs["allow_not_ready_sequence_rows"],
            "min_ready_ratio": kwargs["min_ready_ratio"],
            "sequence_row_policy": kwargs["sequence_row_policy"],
            "diagnostics": kwargs["diagnostics"],
            "output_db_path": kwargs["output_db_path"],
            "log_file_path": kwargs["log_file_path"],
            "jsonl_log_file_path": kwargs["jsonl_log_file_path"],
            "heartbeat_log_file_path": kwargs["heartbeat_log_file_path"],
        }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run-transformer-paper-trader must not start recorder")

    monkeypatch.setattr(live_module, "run_transformer_paper_trader", fake_run)
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-transformer-paper-trader",
            "--db",
            str(tmp_path / "recorder.db"),
            "--model-path",
            str(tmp_path / "model.pt"),
            "--feature-columns",
            str(tmp_path / "feature_columns.json"),
            "--scaler-stats",
            str(tmp_path / "scaler_stats.json"),
            "--training-config",
            str(tmp_path / "training_config.json"),
            "--sequence-length",
            "8",
            "--allow-gap-affected",
            "--allow-not-ready-sequence-rows",
            "--min-ready-ratio",
            "0.8",
            "--sequence-row-policy",
            "latest_clean_rows",
            "--diagnostics",
            "--log-file",
            str(tmp_path / "transformer.log.jsonl"),
            "--jsonl-log-file",
            str(tmp_path / "transformer.jsonl"),
            "--heartbeat-log-file",
            str(tmp_path / "heartbeats.jsonl"),
            "--paper-max-iterations",
            "1",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["sequence_length"] == 8
    assert payload["allow_gap_affected"] is True
    assert payload["allow_not_ready_sequence_rows"] is True
    assert payload["min_ready_ratio"] == 0.8
    assert payload["sequence_row_policy"] == "latest_clean_rows"
    assert payload["diagnostics"] is True
    assert payload["output_db_path"] is None
    assert payload["log_file_path"].endswith("transformer.log.jsonl")
    assert payload["jsonl_log_file_path"].endswith("transformer.jsonl")
    assert payload["heartbeat_log_file_path"].endswith("heartbeats.jsonl")
    assert payload["model_path"].endswith("model.pt")


def test_report_transformer_predictions_counts_buckets_and_reasons(tmp_path) -> None:
    output_db = tmp_path / "transformer_predictions.db"
    conn = _connect(output_db)
    try:
        ensure_transformer_prediction_schema(conn)
        conn.execute(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, signal_timestamp,
                model_path, sequence_length, sequence_row_policy,
                transformer_sequence_ready, probability_yes, probability_no,
                feature_timestamp, latest_feature_timestamp, latest_sequence_timestamp,
                ready_ratio, sequence_rows_available, diagnostics_reason, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (BASE + timedelta(seconds=10)).isoformat(),
                "run_live",
                "market_1",
                (BASE + timedelta(seconds=10)).isoformat(),
                (BASE + timedelta(seconds=8)).isoformat(),
                "model.pt",
                60,
                "latest_clean_rows",
                1,
                0.91,
                0.09,
                (BASE + timedelta(seconds=8)).isoformat(),
                (BASE + timedelta(seconds=8)).isoformat(),
                (BASE + timedelta(seconds=8)).isoformat(),
                1.0,
                60,
                None,
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, signal_timestamp,
                model_path, sequence_length, sequence_row_policy,
                transformer_sequence_ready, probability_yes, probability_no,
                feature_timestamp, latest_feature_timestamp, latest_sequence_timestamp,
                ready_ratio, sequence_rows_available, diagnostics_reason, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (BASE + timedelta(seconds=11)).isoformat(),
                "run_live",
                "market_1",
                (BASE + timedelta(seconds=11)).isoformat(),
                (BASE + timedelta(seconds=9)).isoformat(),
                "model.pt",
                60,
                "latest_clean_rows",
                0,
                None,
                None,
                (BASE + timedelta(seconds=9)).isoformat(),
                (BASE + timedelta(seconds=9)).isoformat(),
                (BASE + timedelta(seconds=9)).isoformat(),
                0.5,
                30,
                "insufficient_clean_sequence_rows",
                "insufficient_clean_sequence_rows",
            ),
        )
        conn.execute(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, signal_timestamp,
                model_path, sequence_length, sequence_row_policy,
                transformer_sequence_ready, probability_yes, probability_no,
                feature_timestamp, latest_feature_timestamp, latest_sequence_timestamp,
                ready_ratio, sequence_rows_available, diagnostics_reason, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (BASE + timedelta(seconds=12)).isoformat(),
                "run_live",
                "market_2",
                (BASE + timedelta(seconds=12)).isoformat(),
                (BASE + timedelta(seconds=10)).isoformat(),
                "model.pt",
                60,
                "latest_clean_rows",
                1,
                0.78,
                0.22,
                (BASE + timedelta(seconds=10)).isoformat(),
                (BASE + timedelta(seconds=10)).isoformat(),
                (BASE + timedelta(seconds=10)).isoformat(),
                1.0,
                60,
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    report = report_transformer_predictions([str(output_db)], now=BASE)

    assert report["total_rows"] == 3
    assert report["ready_rows"] == 2
    assert report["non_null_probability_rows"] == 2
    assert report["market_count"] == 2
    assert report["markets_with_probability_count"] == 2
    assert report["min_probability_yes"] == 0.78
    assert report["average_probability_yes"] == 0.845
    assert report["max_probability_yes"] == 0.91
    assert report["threshold_counts"]["gte_0_75"] == 2
    assert report["threshold_counts"]["gte_0_77"] == 2
    assert report["threshold_counts"]["gte_0_79"] == 1
    assert report["threshold_counts"]["gte_0_80"] == 1
    assert report["threshold_counts"]["gte_0_90"] == 1
    assert report["average_prediction_feature_latency_sec"] == 2.0
    assert report["max_prediction_feature_latency_sec"] == 2.0
    assert report["readiness_failure_reasons"] == {
        "insufficient_clean_sequence_rows": 1
    }
    assert report["top_rejection_reasons"] == [
        {"reason": "insufficient_clean_sequence_rows", "count": 1}
    ]
    assert report["top_markets_by_max_probability"][0]["market_id"] == "market_1"
    assert report["top_markets_by_max_probability"][0]["max_probability_yes"] == 0.91


def test_report_transformer_predictions_cli(monkeypatch, tmp_path, capsys) -> None:
    import src.main as main_module

    output_db = tmp_path / "transformer_predictions.db"
    conn = _connect(output_db)
    try:
        ensure_transformer_prediction_schema(conn)
        conn.execute(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, model_path,
                sequence_length, transformer_sequence_ready, probability_yes,
                probability_no
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                BASE.isoformat(),
                "run_live",
                "market_1",
                BASE.isoformat(),
                "model.pt",
                120,
                1,
                0.8,
                0.2,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "report-transformer-predictions",
            "--prediction-db",
            str(output_db),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["total_rows"] == 1
    assert payload["ready_rows"] == 1
    assert payload["market_count"] == 1
    assert payload["min_probability_yes"] == 0.8
    assert payload["average_probability_yes"] == 0.8
    assert payload["max_probability_yes"] == 0.8
    assert payload["threshold_counts"]["gte_0_77"] == 1
    assert payload["threshold_counts"]["gte_0_79"] == 1
    assert payload["threshold_counts"]["gte_0_80"] == 1


def test_backtest_transformer_threshold_strategy_replays_yes_no_and_settlement(tmp_path) -> None:
    prediction_db = tmp_path / "transformer_predictions.db"
    recorder_db = tmp_path / "recorder.db"
    output_json = tmp_path / "threshold_backtest.json"
    output_txt = tmp_path / "threshold_backtest.txt"
    _setup_threshold_backtest_recorder(recorder_db)
    _setup_threshold_prediction_db(prediction_db)

    report = backtest_transformer_threshold_strategy(
        prediction_db_paths=[str(prediction_db)],
        recorder_db_path=str(recorder_db),
        output_json_path=str(output_json),
        output_txt_path=str(output_txt),
        thresholds=[0.77],
        min_time_until_resolution_options=[30.0],
        max_time_until_resolution_options=[120.0],
        exit_horizon_sec_options=[15],
        side="BOTH",
        slippage_cents=0.0,
        fee_cents=0.0,
        min_done_trades=1,
        now=BASE,
    )

    assert report["status"] == "ok"
    assert report["probability_rows_loaded"] == 3
    assert report["trades"] == 3
    assert report["done_trades"] == 3
    assert report["markets_covered"] == 3
    side_rows = {row["side"]: row for row in report["side_comparison"]}
    assert side_rows["YES"]["done_trades"] == 2
    assert side_rows["NO"]["done_trades"] == 1
    assert side_rows["NO"]["pnl"] > 0
    assert report["best_by_avg_roi_min_done"]
    assert output_json.exists()
    text = output_txt.read_text(encoding="utf-8")
    assert "Transformer Threshold Strategy Backtest" in text
    assert "do not go live" in text


def test_backtest_transformer_threshold_strategy_split_by_time_validates_out_of_sample(tmp_path) -> None:
    prediction_db = tmp_path / "transformer_predictions_split.db"
    recorder_db = tmp_path / "recorder_split.db"
    output_txt = tmp_path / "threshold_split.txt"
    _setup_threshold_split_inputs(recorder_db, prediction_db)

    report = backtest_transformer_threshold_strategy(
        prediction_db_paths=[str(prediction_db)],
        recorder_db_path=str(recorder_db),
        output_txt_path=str(output_txt),
        thresholds=[0.75],
        min_time_until_resolution_options=[60.0],
        max_time_until_resolution_options=[90.0],
        exit_horizon_sec_options=[60],
        side="YES",
        slippage_cents=0.0,
        fee_cents=0.0,
        min_done_trades=1,
        split_by_time=True,
        train_ratio=0.5,
        min_test_done_trades=2,
        strategy_filter="transformer_yes_t075_60_90_fh60",
        now=BASE,
    )

    validation = report["validation"]
    split = validation["split"]
    assert split["train_market_ids"] == ["m_train"]
    assert split["test_market_ids"] == ["m_test"]
    assert not (set(split["train_market_keys"]) & set(split["test_market_keys"]))
    comparison = validation["strategy_comparison"][0]
    assert comparison["strategy_id"] == "transformer_yes_t075_60_90_fh60"
    assert comparison["train"]["pnl"] > 0
    assert comparison["test"]["pnl"] < 0
    assert "train_positive_test_negative:transformer_yes_t075_60_90_fh60" in validation["warnings"]
    assert "test_sample_too_small:transformer_yes_t075_60_90_fh60" in validation["warnings"]
    text = output_txt.read_text(encoding="utf-8")
    assert "Out-of-Sample Time Split Validation" in text
    assert "train_positive_test_negative" in text
    assert "paper/shadow only; do not go live" in text


def test_backtest_transformer_threshold_strategy_filter_focuses_one_strategy(tmp_path) -> None:
    prediction_db = tmp_path / "transformer_predictions.db"
    recorder_db = tmp_path / "recorder.db"
    _setup_threshold_backtest_recorder(recorder_db)
    _setup_threshold_prediction_db(prediction_db)

    report = backtest_transformer_threshold_strategy(
        prediction_db_paths=[str(prediction_db)],
        recorder_db_path=str(recorder_db),
        thresholds=[0.77],
        min_time_until_resolution_options=[30.0],
        max_time_until_resolution_options=[120.0],
        exit_horizon_sec_options=[15],
        side="BOTH",
        slippage_cents=0.0,
        fee_cents=0.0,
        min_done_trades=1,
        strategy_filter="transformer_no_t077_30_120_fh15",
        now=BASE,
    )

    assert report["strategy_filter"] == ["transformer_no_t077_30_120_fh15"]
    assert report["done_trades"] == 1
    assert {row["strategy_id"] for row in report["best_by_pnl"]} == {
        "transformer_no_t077_30_120_fh15"
    }


def test_backtest_transformer_threshold_strategy_cli(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    prediction_db = tmp_path / "transformer_predictions.db"
    recorder_db = tmp_path / "recorder.db"
    output_json = tmp_path / "threshold_backtest_cli.json"
    output_txt = tmp_path / "threshold_backtest_cli.txt"
    _setup_threshold_backtest_recorder(recorder_db)
    _setup_threshold_prediction_db(prediction_db)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "backtest-transformer-threshold-strategy",
            "--prediction-db",
            str(prediction_db),
            "--recorder-db",
            str(recorder_db),
            "--output-json",
            str(output_json),
            "--output-txt",
            str(output_txt),
            "--thresholds",
            "0.77,0.80",
            "--min-time-until-resolution-sec",
            "30",
            "--max-time-until-resolution-sec",
            "120",
            "--exit-horizon-sec",
            "15",
            "--side",
            "BOTH",
            "--entry-slippage-cents",
            "0",
            "--fee-cents",
            "0",
            "--min-done-trades",
            "1",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["done_trades"] == 3
    assert payload["thresholds"] == [0.77, 0.8]
    assert output_json.exists()
    assert output_txt.exists()


def test_backtest_transformer_threshold_strategy_split_cli(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    prediction_db = tmp_path / "transformer_predictions_split.db"
    recorder_db = tmp_path / "recorder_split.db"
    output_json = tmp_path / "threshold_split_cli.json"
    _setup_threshold_split_inputs(recorder_db, prediction_db)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "backtest-transformer-threshold-strategy",
            "--prediction-db",
            str(prediction_db),
            "--recorder-db",
            str(recorder_db),
            "--output-json",
            str(output_json),
            "--thresholds",
            "0.75",
            "--min-time-until-resolution-sec",
            "60",
            "--max-time-until-resolution-sec",
            "90",
            "--exit-horizon-sec",
            "60",
            "--side",
            "YES",
            "--entry-slippage-cents",
            "0",
            "--fee-cents",
            "0",
            "--min-done-trades",
            "1",
            "--split-by-time",
            "--train-ratio",
            "0.5",
            "--min-test-done-trades",
            "2",
            "--strategy-filter",
            "transformer_yes_t075_60_90_fh60",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["split_by_time"] is True
    assert payload["validation"]["split"]["test_market_ids"] == ["m_test"]
    assert "train_positive_test_negative:transformer_yes_t075_60_90_fh60" in payload["warnings"]
    assert output_json.exists()


def test_transformer_calibration_analysis_joins_outcomes_and_writes_reports(tmp_path) -> None:
    prediction_db = tmp_path / "transformer_predictions.db"
    recorder_db = tmp_path / "recorder.db"
    output_json = tmp_path / "calibration.json"
    output_txt = tmp_path / "calibration.txt"
    _setup_resolution_db(recorder_db)
    conn = _connect(prediction_db)
    try:
        ensure_transformer_prediction_schema(conn)
        conn.executemany(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, model_path,
                sequence_length, transformer_sequence_ready, probability_yes,
                probability_no, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    BASE.isoformat(),
                    "run_live",
                    "market_1",
                    BASE.isoformat(),
                    "model.pt",
                    120,
                    1,
                    0.81,
                    0.19,
                    None,
                ),
                (
                    (BASE + timedelta(seconds=1)).isoformat(),
                    "run_live",
                    "market_2",
                    (BASE + timedelta(seconds=1)).isoformat(),
                    "model.pt",
                    120,
                    1,
                    0.78,
                    0.22,
                    None,
                ),
                (
                    (BASE + timedelta(seconds=2)).isoformat(),
                    "run_live",
                    "market_2",
                    (BASE + timedelta(seconds=2)).isoformat(),
                    "model.pt",
                    120,
                    0,
                    None,
                    None,
                    "insufficient_sequence_rows",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    report = analyze_transformer_calibration(
        prediction_db_paths=[str(prediction_db)],
        recorder_db_path=str(recorder_db),
        output_json_path=str(output_json),
        output_txt_path=str(output_txt),
        now=BASE,
    )

    assert report["rows"] == 3
    assert report["markets"] == 2
    assert report["probability_rows"] == 2
    assert report["ready_rows"] == 2
    assert report["threshold_hit_counts"] == {
        "gte_0_75": 2,
        "gte_0_77": 2,
        "gte_0_79": 1,
        "gte_0_80": 1,
    }
    threshold_080 = {
        row["threshold"]: row for row in report["threshold_outcome_calibration"]
    }["gte_0_80"]
    assert threshold_080["resolved_rows"] == 1
    assert threshold_080["win_rate"] == 1.0
    threshold_075 = {
        row["threshold"]: row for row in report["threshold_outcome_calibration"]
    }["gte_0_75"]
    assert threshold_075["resolved_rows"] == 2
    assert threshold_075["win_rate"] == 0.5
    assert report["outcome_join"]["available"] is True
    assert "sample is too small" in report["warnings"]
    assert output_json.exists()
    text = output_txt.read_text(encoding="utf-8")
    assert "Transformer Calibration Analysis" in text
    assert "Recommendation: paper/shadow only" in text


def test_transformer_calibration_analysis_loads_jsonl_and_warns_without_outcomes(tmp_path) -> None:
    jsonl_path = tmp_path / "heartbeats.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": BASE.isoformat(),
                        "run_id": "run_live",
                        "market_id": "market_1",
                        "sequence_length": 120,
                        "transformer_sequence_ready": True,
                        "probability_yes": 0.80,
                        "reason": None,
                        "latest_feature_timestamp": BASE.isoformat(),
                    }
                ),
                json.dumps(
                    {
                        "timestamp": (BASE + timedelta(seconds=1)).isoformat(),
                        "run_id": "run_live",
                        "market_id": "market_1",
                        "sequence_length": 120,
                        "transformer_sequence_ready": False,
                        "probability_yes": None,
                        "reason": "not_feature_ready",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_transformer_calibration(
        jsonl_log_paths=[str(jsonl_path)],
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        now=BASE,
    )

    assert report["rows"] == 2
    assert report["probability_rows"] == 1
    assert report["ready_rows"] == 1
    assert report["threshold_hit_counts"]["gte_0_80"] == 1
    assert report["input_jsonl_logs"][0]["rows"] == 2
    assert "outcome join unavailable" in report["warnings"]


def test_transformer_calibration_analysis_cli(monkeypatch, tmp_path, capsys) -> None:
    import src.main as main_module

    prediction_db = tmp_path / "transformer_predictions.db"
    recorder_db = tmp_path / "recorder.db"
    output_json = tmp_path / "calibration_cli.json"
    output_txt = tmp_path / "calibration_cli.txt"
    _setup_resolution_db(recorder_db)
    conn = _connect(prediction_db)
    try:
        ensure_transformer_prediction_schema(conn)
        conn.execute(
            """
            INSERT INTO transformer_predictions (
                created_at, run_id, market_id, timestamp, model_path,
                sequence_length, transformer_sequence_ready, probability_yes,
                probability_no
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                BASE.isoformat(),
                "run_live",
                "market_1",
                BASE.isoformat(),
                "model.pt",
                120,
                1,
                0.8,
                0.2,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "transformer-calibration-analysis",
            "--prediction-db",
            str(prediction_db),
            "--recorder-db",
            str(recorder_db),
            "--output-json",
            str(output_json),
            "--output-txt",
            str(output_txt),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["rows"] == 1
    assert payload["threshold_hit_counts"]["gte_0_80"] == 1
    assert output_json.exists()
    assert output_txt.exists()
