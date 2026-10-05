from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.transformer_shadow_paper_trader import (
    TransformerShadowPaperTraderError,
    run_transformer_shadow_paper_strategy,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _setup_recorder(path, *, time_until: float = 60.0) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            question TEXT,
            start_time TEXT,
            close_time TEXT,
            phase TEXT
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
        INSERT INTO markets (run_id, market_id, question, start_time, close_time, phase)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "run_live",
            "market_1",
            "BTC up?",
            BASE.isoformat(),
            (BASE + timedelta(minutes=5)).isoformat(),
            "active",
        ),
    )
    for index in range(3):
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
                float(index),
                1,
                0,
                "ok",
                1,
                time_until,
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
                "market_1",
                timestamp.isoformat(),
                0.50,
                0.52,
                0.48,
                0.50,
                0.51,
                0.49,
            ),
        )
        for source in ("polymarket_rtds_chainlink", "polymarket_rtds_binance"):
            conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, exchange_timestamp, local_arrival_ns, local_arrival_iso
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("run_live", source, 100.0, timestamp.isoformat(), index + 1, timestamp.isoformat()),
            )
    conn.commit()
    conn.close()


def _write_artifacts(tmp_path):
    model_path = tmp_path / "model.pt"
    feature_path = tmp_path / "feature_columns.json"
    scaler_path = tmp_path / "scaler_stats.json"
    config_path = tmp_path / "training_config.json"
    model_path.write_bytes(b"not-used")
    feature_path.write_text(json.dumps(["signal"]), encoding="utf-8")
    scaler_path.write_text(json.dumps({"mean": [0.0], "std": [1.0]}), encoding="utf-8")
    config_path.write_text(
        json.dumps({"sequence_length": 3, "label_column": "label_yes_win"}),
        encoding="utf-8",
    )
    return model_path, feature_path, scaler_path, config_path


def test_transformer_shadow_paper_inserts_fixed_horizon_trades(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    report = run_transformer_shadow_paper_strategy(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="shadow_test",
        sequence_length=3,
        fixed_horizons_sec=[15, 30, 60],
        max_iterations=1,
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.91,
        now=BASE + timedelta(seconds=3),
    )

    assert report["status"] == "ok"
    assert report["trades_opened"] == 3
    conn = _connect(output_db)
    try:
        rows = [dict(row) for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC")]
    finally:
        conn.close()
    assert [row["fixed_horizon_exit_sec"] for row in rows] == [15.0, 30.0, 60.0]
    assert all(row["status"] == "open" for row in rows)
    assert all(row["signal_direction"] == "YES" for row in rows)
    assert all(row["entry_price"] == 0.52 for row in rows)
    assert all(row["predicted_probability_yes"] == 0.91 for row in rows)


def test_transformer_shadow_paper_refuses_live_trading(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    _setup_recorder(recorder_db)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)

    try:
        run_transformer_shadow_paper_strategy(
            recorder_db_path=str(recorder_db),
            model_path=str(model_path),
            feature_columns_path=str(feature_path),
            scaler_stats_path=str(scaler_path),
            training_config_path=str(config_path),
            output_db_path=str(tmp_path / "paper.db"),
            live_trading_enabled=True,
            max_iterations=1,
            emit_logs=False,
            inference_fn=lambda _sequence, _artifacts: 0.91,
            now=BASE + timedelta(seconds=3),
        )
    except TransformerShadowPaperTraderError as exc:
        assert "live_trading_enabled" in str(exc)
    else:
        raise AssertionError("expected live trading to be refused")


def test_transformer_shadow_paper_does_not_write_to_recorder(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    output_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    model_path, feature_path, scaler_path, config_path = _write_artifacts(tmp_path)
    before = _connect(recorder_db)
    before_count = before.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    before.close()

    run_transformer_shadow_paper_strategy(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="shadow_test",
        sequence_length=3,
        max_iterations=1,
        emit_logs=False,
        inference_fn=lambda _sequence, _artifacts: 0.91,
        now=BASE + timedelta(seconds=3),
    )

    after = _connect(recorder_db)
    after_count = after.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    after.close()
    assert after_count == before_count
