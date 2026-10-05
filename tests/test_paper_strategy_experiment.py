from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from src.paper_strategy_experiment import (
    build_paper_strategy_experiment_report,
    monitor_paper_strategy_experiment,
    render_paper_strategy_experiment_monitor,
    render_paper_strategy_experiment_report,
    run_paper_strategy_experiment,
)
from src.paper_strategy_experiment_analysis import analyze_paper_strategy_experiment
from src.transformer_live_inference import ensure_transformer_prediction_schema
from src.transformer_shadow_paper_trader import (
    backfill_transformer_prediction_paper_pnl,
    run_transformer_prediction_paper_strategy,
)


def _write_config(path: Path, *, live_trading: bool = False) -> None:
    payload = {
        "live_trading": {
            "live_trading_enabled": live_trading,
            "dry_run_orders": True,
        },
        "strategies": [
            {
                "strategy_id": path.stem,
                "enabled": True,
                "direction_mode": "NO_ONLY",
                "long_threshold": 2.0,
                "short_threshold": 0.15,
                "min_estimated_edge": 0.05,
                "entry_slippage_cents": 0.01,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 1,
                "one_trade_per_market": True,
                "require_positive_edge": True,
                "min_time_until_resolution_sec": 30,
                "max_time_until_resolution_sec": 60,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_transformer_prediction_config(path: Path, prediction_db: Path) -> None:
    payload = {
        "prediction_source": "transformer_predictions",
        "transformer_prediction_db": str(prediction_db),
        "live_trading": {
            "live_trading_enabled": False,
            "dry_run_orders": True,
        },
        "strategies": [
            {
                "strategy_id": "tf_yes_t077_fh15",
                "strategy_name": "Transformer YES 0.77 fh15",
                "enabled": True,
                "direction_mode": "YES_ONLY",
                "long_threshold": 0.77,
                "short_threshold": -1.0,
                "min_estimated_edge": 0.0,
                "entry_slippage_cents": 0.0,
                "exit_slippage_cents": 0.0,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 1,
                "one_trade_per_market": True,
                "require_positive_edge": True,
                "min_time_until_resolution_sec": 30,
                "max_time_until_resolution_sec": 120,
                "exit_type": "FIXED_HORIZON_EXIT",
                "fixed_horizon_exit_sec": 15,
            },
            {
                "strategy_id": "tf_no_t077_fh15",
                "strategy_name": "Transformer NO 0.77 fh15",
                "enabled": True,
                "direction_mode": "NO_ONLY",
                "long_threshold": 2.0,
                "short_threshold": 0.23,
                "min_estimated_edge": 0.0,
                "entry_slippage_cents": 0.0,
                "exit_slippage_cents": 0.0,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 1,
                "one_trade_per_market": True,
                "require_positive_edge": True,
                "min_time_until_resolution_sec": 30,
                "max_time_until_resolution_sec": 120,
                "exit_type": "FIXED_HORIZON_EXIT",
                "fixed_horizon_exit_sec": 15,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_transformer_fade_config(
    path: Path,
    prediction_db: Path,
    *,
    live_trading: bool = False,
    threshold: float = 0.95,
) -> None:
    payload = {
        "prediction_source": "transformer_predictions",
        "transformer_prediction_db": str(prediction_db),
        "live_trading": {
            "live_trading_enabled": live_trading,
            "dry_run_orders": True,
        },
        "strategies": [
            {
                "strategy_id": "tf_fade_yes_t095_no_fh15",
                "strategy_name": "Fade transformer YES 0.95 with NO fh15",
                "enabled": True,
                "direction_mode": "FADE_YES_WITH_NO",
                "fade_yes_threshold": threshold,
                "long_threshold": 2.0,
                "short_threshold": -1.0,
                "min_estimated_edge": 0.0,
                "entry_slippage_cents": 0.0,
                "exit_slippage_cents": 0.0,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 1,
                "one_trade_per_market": True,
                "min_time_until_resolution_sec": 30,
                "max_time_until_resolution_sec": 120,
                "exit_type": "FIXED_HORIZON_EXIT",
                "fixed_horizon_exit_sec": 15,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_transformer_prediction_inputs(recorder_db: Path, prediction_db: Path) -> None:
    conn = sqlite3.connect(recorder_db)
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
    conn.executemany(
        "INSERT INTO markets VALUES (?, ?, ?, ?, ?)",
        [
            ("run_live", "m_yes", "BTC up?", "2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"),
            ("run_live", "m_no", "BTC down?", "2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO market_snapshots (
            run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
            best_bid_no, best_ask_no, mid_price_yes, mid_price_no
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("run_live", "m_yes", "2026-01-01T00:00:00+00:00", 0.48, 0.50, 0.50, 0.52, 0.49, 0.51),
            ("run_live", "m_yes", "2026-01-01T00:00:15+00:00", 0.60, 0.62, 0.38, 0.40, 0.61, 0.39),
            ("run_live", "m_no", "2026-01-01T00:00:01+00:00", 0.50, 0.52, 0.48, 0.50, 0.51, 0.49),
            ("run_live", "m_no", "2026-01-01T00:00:16+00:00", 0.38, 0.40, 0.60, 0.62, 0.39, 0.61),
        ],
    )
    conn.commit()
    conn.close()

    pred = sqlite3.connect(prediction_db)
    ensure_transformer_prediction_schema(pred)
    pred.executemany(
        """
        INSERT INTO transformer_predictions (
            created_at, run_id, market_id, timestamp, signal_timestamp,
            model_path, sequence_length, transformer_sequence_ready,
            probability_yes, probability_no, latest_feature_timestamp,
            market_start_time, market_close_time, time_until_resolution,
            best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
            mid_price_yes, mid_price_no, snapshot_quality_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-01-01T00:00:00+00:00",
                "run_live",
                "m_yes",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "transformer.pt",
                60,
                1,
                0.78,
                0.22,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:05:00+00:00",
                90.0,
                0.48,
                0.50,
                0.50,
                0.52,
                0.49,
                0.51,
                "ok",
            ),
            (
                "2026-01-01T00:00:01+00:00",
                "run_live",
                "m_no",
                "2026-01-01T00:00:01+00:00",
                "2026-01-01T00:00:01+00:00",
                "transformer.pt",
                60,
                1,
                0.22,
                0.78,
                "2026-01-01T00:00:01+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:05:00+00:00",
                90.0,
                0.50,
                0.52,
                0.48,
                0.50,
                0.51,
                0.49,
                "ok",
            ),
        ],
    )
    pred.commit()
    pred.close()


def _set_transformer_probability(prediction_db: Path, market_id: str, probability_yes: float) -> None:
    pred = sqlite3.connect(prediction_db)
    pred.execute(
        """
        UPDATE transformer_predictions
        SET probability_yes = ?,
            probability_no = ?
        WHERE market_id = ?
        """,
        (probability_yes, 1.0 - probability_yes, market_id),
    )
    pred.commit()
    pred.close()


def test_run_paper_strategy_experiment_writes_manifest(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_config(config_dir / "a.json")
    _write_config(config_dir / "b.json")
    output_dir = tmp_path / "experiment"

    report = run_paper_strategy_experiment(
        recorder_db_path="data/recorder.db",
        model_path="data/models/model.joblib",
        feature_columns_path="data/models/feature_columns.json",
        config_glob=str(config_dir / "*.json"),
        experiment_id="exp_test",
        output_dir=str(output_dir),
        dry_run=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert manifest["experiment_id"] == "exp_test"
    assert len(manifest["workers"]) == 2
    assert len(manifest["command_lines"]) == 2
    assert all("--live-trading" not in command for command in manifest["command_lines"])
    assert all("--heartbeat-detail compact" in command for command in manifest["command_lines"])
    assert all(worker["launch_status"] == "dry_run" for worker in manifest["workers"])
    assert all(str(output_dir / "paper_dbs") in path for path in manifest["paper_db_paths"])
    assert all(str(output_dir / "logs") in path for path in manifest["log_paths"])


def test_run_paper_strategy_experiment_ignores_metadata_json_from_recommendations(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_config(config_dir / "valid.json")
    (config_dir / "next_sweep_recommendations.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "recommendations": {"suggested_strategies": []},
                "generated_config_paths": [str(config_dir / "valid.json")],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "experiment"

    report = run_paper_strategy_experiment(
        recorder_db_path="data/recorder.db",
        model_path="data/models/model.joblib",
        feature_columns_path="data/models/feature_columns.json",
        config_glob=str(config_dir / "*.json"),
        experiment_id="exp_test",
        output_dir=str(output_dir),
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    assert len(report["workers"]) == 1
    assert report["workers"][0]["config_path"].endswith("valid.json")


def test_run_paper_strategy_experiment_refuses_live_trading_config(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _write_config(config_dir / "danger.json", live_trading=True)

    report = run_paper_strategy_experiment(
        recorder_db_path="data/recorder.db",
        model_path="model.joblib",
        feature_columns_path="feature_columns.json",
        config_glob=str(config_dir / "*.json"),
        experiment_id="exp_test",
        output_dir=str(tmp_path / "experiment"),
        dry_run=True,
    )

    assert report["status"] == "error"
    assert any("live_trading_enabled is true" in error for error in report["errors"])


def test_run_paper_strategy_experiment_launches_transformer_prediction_runner(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    prediction_db = tmp_path / "transformer_predictions.db"
    _write_transformer_prediction_config(config_dir / "transformer.json", prediction_db)

    report = run_paper_strategy_experiment(
        recorder_db_path="data/recorder.db",
        model_path="unused_model.joblib",
        feature_columns_path="unused_features.json",
        config_glob=str(config_dir / "*.json"),
        experiment_id="exp_transformer",
        output_dir=str(tmp_path / "experiment"),
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    worker = report["workers"][0]
    assert worker["runner"] == "transformer_predictions"
    assert worker["transformer_prediction_db"] == str(prediction_db)
    assert "run-transformer-prediction-paper-strategy" in worker["command_line"]
    assert "--prediction-db" in worker["command_line"]


def test_transformer_prediction_paper_strategy_writes_trades_and_candidates(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)

    report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert report["status"] == "ok"
    assert report["trades_opened"] == 2
    assert report["candidates_logged"] == 4
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    trades = [dict(row) for row in conn.execute("SELECT * FROM paper_trades").fetchall()]
    candidates = [dict(row) for row in conn.execute("SELECT * FROM paper_trade_candidates").fetchall()]
    conn.close()

    assert {row["strategy_id"] for row in trades} == {"tf_yes_t077_fh15", "tf_no_t077_fh15"}
    assert {row["signal_direction"] for row in trades} == {"YES", "NO"}
    assert all(row["status"] == "closed" for row in trades)
    assert all(row["exit_price"] is not None for row in trades)
    assert all(row["exit_time"] is not None for row in trades)
    assert all(row["exit_reason"] == "fixed_horizon" for row in trades)
    assert all(row["pnl_usd"] is not None for row in trades)
    assert all(row["roi"] is not None for row in trades)
    assert all(row["realized_pnl_usd"] > 0 for row in trades)
    assert all(row["pnl_usd"] == row["realized_pnl_usd"] for row in trades)
    assert len(candidates) == 4
    decision_counts = {}
    for row in candidates:
        decision_counts[row["decision"]] = decision_counts.get(row["decision"], 0) + 1
    assert decision_counts == {"TRADE": 2, "SKIP": 2}
    trade_candidates = [row for row in candidates if row["decision"] == "TRADE"]
    assert all(row["candidate_direction"] in {"YES", "NO"} for row in trade_candidates)
    assert all(row["probability_for_direction"] == 0.78 for row in trade_candidates)


def test_transformer_prediction_fade_yes_opens_no_trade(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_fade_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _set_transformer_probability(prediction_db, "m_yes", 0.97)
    _write_transformer_fade_config(config_path, prediction_db, threshold=0.95)

    report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert report["trades_opened"] == 1
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    trade = dict(conn.execute("SELECT * FROM paper_trades").fetchone())
    candidate = dict(
        conn.execute(
            "SELECT * FROM paper_trade_candidates WHERE decision = 'TRADE'"
        ).fetchone()
    )
    conn.close()
    assert trade["signal_direction"] == "NO"
    assert trade["entry_price"] == 0.52
    assert trade["threshold_used"] == 0.95
    assert trade["predicted_probability_yes"] == 0.97
    assert round(float(trade["probability_for_direction"]), 10) == 0.03
    assert candidate["candidate_direction"] == "NO"
    assert candidate["rejection_reason"] is None


def test_transformer_prediction_fade_yes_skips_below_threshold(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_fade_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _set_transformer_probability(prediction_db, "m_yes", 0.90)
    _write_transformer_fade_config(config_path, prediction_db, threshold=0.95)

    report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert report["trades_opened"] == 0
    assert report["last_poll"]["top_rejection_reasons"]["fade_threshold"] == 2
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    skips = [
        dict(row)
        for row in conn.execute("SELECT * FROM paper_trade_candidates").fetchall()
    ]
    conn.close()
    assert trades == 0
    assert {row["rejection_reason"] for row in skips} == {"fade_threshold"}
    assert {row["candidate_direction"] for row in skips} == {"NO"}


def test_transformer_prediction_fade_config_refuses_live_trading(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_fade_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_fade_config(config_path, prediction_db, live_trading=True)

    try:
        run_transformer_prediction_paper_strategy(
            recorder_db_path=str(recorder_db),
            prediction_db_path=str(prediction_db),
            config_path=str(config_path),
            output_db_path=str(output_db),
            run_id="run_transformer_predictions",
            poll_sec=0.0,
            max_iterations=1,
            emit_logs=False,
        )
    except Exception as exc:
        assert "live_trading_enabled is not supported" in str(exc)
    else:
        raise AssertionError("live trading config should be refused")


def test_transformer_prediction_fade_dedupes_repeated_market_predictions(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_fade_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _set_transformer_probability(prediction_db, "m_yes", 0.97)
    _write_transformer_fade_config(config_path, prediction_db, threshold=0.95)
    pred = sqlite3.connect(prediction_db)
    pred.execute(
        """
        INSERT INTO transformer_predictions (
            created_at, run_id, market_id, timestamp, signal_timestamp,
            model_path, sequence_length, transformer_sequence_ready,
            probability_yes, probability_no, latest_feature_timestamp,
            market_start_time, market_close_time, time_until_resolution,
            best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
            mid_price_yes, mid_price_no, snapshot_quality_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-01-01T00:00:02+00:00",
            "run_live",
            "m_yes",
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:02+00:00",
            "transformer.pt",
            60,
            1,
            0.98,
            0.02,
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:05:00+00:00",
            88.0,
            0.49,
            0.51,
            0.49,
            0.51,
            0.50,
            0.50,
            "ok",
        ),
    )
    pred.commit()
    pred.close()

    first_report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    second_report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    conn = sqlite3.connect(output_db)
    trade_count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    duplicate_skip_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM paper_trade_candidates
        WHERE rejection_reason = 'one_trade_per_market'
        """
    ).fetchone()[0]
    conn.close()
    assert first_report["trades_opened"] == 1
    assert second_report["trades_opened"] == 0
    assert trade_count == 1
    assert duplicate_skip_count >= 1


def test_transformer_prediction_missing_exit_snapshot_leaves_trade_open_without_fake_pnl(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    conn = sqlite3.connect(recorder_db)
    conn.execute("DELETE FROM market_snapshots WHERE timestamp >= ?", ("2026-01-01T00:00:15+00:00",))
    conn.commit()
    conn.close()
    _write_transformer_prediction_config(config_path, prediction_db)

    report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert report["trades_opened"] == 2
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    trades = [dict(row) for row in conn.execute("SELECT * FROM paper_trades").fetchall()]
    conn.close()
    assert all(row["status"] == "open" for row in trades)
    assert all(row["pnl_usd"] is None for row in trades)
    assert all(row["realized_pnl_usd"] is None for row in trades)
    assert all(row["exit_price"] is None for row in trades)


def test_backfill_transformer_prediction_paper_pnl_fills_old_closed_trade(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="paper_run_not_recorder_run",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    conn = sqlite3.connect(output_db)
    conn.execute(
        """
        UPDATE paper_trades
        SET pnl_usd = NULL,
            roi = NULL,
            realized_pnl_usd = NULL,
            realized_roi = NULL,
            exit_price = NULL,
            adjusted_exit_price = NULL,
            exit_time = NULL
        WHERE signal_direction = 'YES'
        """
    )
    conn.commit()
    conn.close()

    report = backfill_transformer_prediction_paper_pnl(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(output_db),
    )

    assert report["rows_updated"] >= 1
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    yes_trade = dict(
        conn.execute(
            "SELECT * FROM paper_trades WHERE signal_direction = 'YES' LIMIT 1"
        ).fetchone()
    )
    conn.close()
    assert yes_trade["status"] == "closed"
    assert yes_trade["exit_time"] == "2026-01-01T00:00:15+00:00"
    assert yes_trade["exit_price"] == 0.60
    assert yes_trade["pnl_usd"] is not None
    assert yes_trade["roi"] is not None
    assert yes_trade["realized_pnl_usd"] == yes_trade["pnl_usd"]


def test_backfill_transformer_prediction_paper_pnl_fills_aliases_from_realized(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="paper_run",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    conn = sqlite3.connect(output_db)
    conn.execute(
        """
        UPDATE paper_trades
        SET pnl_usd = NULL,
            roi = NULL
        WHERE signal_direction = 'NO'
        """
    )
    conn.commit()
    conn.close()

    report = backfill_transformer_prediction_paper_pnl(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(output_db),
    )

    assert report["aliases_filled"] >= 1
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    no_trade = dict(
        conn.execute(
            "SELECT * FROM paper_trades WHERE signal_direction = 'NO' LIMIT 1"
        ).fetchone()
    )
    conn.close()
    assert no_trade["pnl_usd"] == no_trade["realized_pnl_usd"]
    assert no_trade["roi"] == no_trade["realized_roi"]


def test_backfill_transformer_prediction_paper_pnl_reports_duplicate_market_groups(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="paper_run",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    conn = sqlite3.connect(output_db)
    conn.execute(
        """
        INSERT INTO paper_trades (
            created_at, run_id, market_id, signal_timestamp, model_name,
            strategy_id, signal_direction, stake_usd, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-01-01T00:00:03+00:00",
            "other_run",
            "m_yes",
            "2026-01-01T00:00:03+00:00",
            "transformer_predictions",
            "tf_yes_t077_fh15",
            "YES",
            1.0,
            "closed",
        ),
    )
    conn.commit()
    conn.close()

    report = backfill_transformer_prediction_paper_pnl(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(output_db),
        dry_run=True,
    )

    assert report["duplicate_strategy_market_group_count"] == 1
    group = report["duplicate_strategy_market_groups"][0]
    assert group["strategy_id"] == "tf_yes_t077_fh15"
    assert group["market_id"] == "m_yes"
    assert group["trade_count"] == 2


def test_transformer_prediction_paper_strategy_dedupes_prediction_rows(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)

    first_report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    second_report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert first_report["trades_opened"] == 2
    assert first_report["candidates_logged"] == 4
    assert second_report["trades_opened"] == 0
    assert second_report["candidates_logged"] == 0
    conn = sqlite3.connect(output_db)
    trade_count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    candidate_count = conn.execute("SELECT COUNT(*) FROM paper_trade_candidates").fetchone()[0]
    conn.close()
    assert trade_count == 2
    assert candidate_count == 4


def test_transformer_prediction_one_trade_per_market_blocks_repeated_market_predictions(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    conn = sqlite3.connect(prediction_db)
    conn.execute(
        """
        INSERT INTO transformer_predictions (
            created_at, run_id, market_id, timestamp, signal_timestamp,
            model_path, sequence_length, transformer_sequence_ready,
            probability_yes, probability_no, latest_feature_timestamp,
            market_start_time, market_close_time, time_until_resolution,
            best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
            mid_price_yes, mid_price_no, snapshot_quality_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-01-01T00:00:02+00:00",
            "run_live",
            "m_yes",
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:02+00:00",
            "transformer.pt",
            60,
            1,
            0.79,
            0.21,
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:05:00+00:00",
            88.0,
            0.49,
            0.51,
            0.49,
            0.51,
            0.50,
            0.50,
            "ok",
        ),
    )
    conn.commit()
    conn.close()

    report = run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )

    assert report["trades_opened"] == 2
    conn = sqlite3.connect(output_db)
    conn.row_factory = sqlite3.Row
    yes_trades = [
        dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM paper_trades
            WHERE strategy_id = 'tf_yes_t077_fh15'
              AND market_id = 'm_yes'
              AND status != 'skipped'
            """
        ).fetchall()
    ]
    duplicate_skip = conn.execute(
        """
        SELECT *
        FROM paper_trade_candidates
        WHERE strategy_id = 'tf_yes_t077_fh15'
          AND market_id = 'm_yes'
          AND rejection_reason = 'one_trade_per_market'
        """
    ).fetchone()
    conn.close()
    assert len(yes_trades) == 1
    assert duplicate_skip is not None


def test_analyze_experiment_ranks_transformer_threshold_strategies(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "configs" / "transformer_config.json"
    output_db = tmp_path / "experiment" / "paper_dbs" / "worker.db"
    config_path.parent.mkdir(parents=True)
    output_db.parent.mkdir(parents=True)
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    run_transformer_prediction_paper_strategy(
        recorder_db_path=str(recorder_db),
        prediction_db_path=str(prediction_db),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="run_transformer_predictions",
        poll_sec=0.0,
        max_iterations=1,
        emit_logs=False,
    )
    manifest = {
        "experiment_id": "exp_transformer_predictions",
        "model_path": "transformer.pt",
        "feature_columns_path": "feature_columns.json",
        "workers": [
            {
                "worker_id": "worker",
                "config_path": str(config_path),
                "paper_db_path": str(output_db),
                "run_id": "run_transformer_predictions",
            }
        ],
    }
    (tmp_path / "experiment" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze_paper_strategy_experiment(
        experiment_dir=str(tmp_path / "experiment"),
        min_done_thresholds=(1,),
    )

    ranked = report["strategy_rankings"]["min_done_1"]["sorted_by_pnl"]
    assert {row["strategy_id"] for row in ranked} == {"tf_yes_t077_fh15", "tf_no_t077_fh15"}
    assert all(row["pnl"] > 0 for row in ranked)


def test_transformer_prediction_paper_strategy_cli(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    recorder_db = tmp_path / "recorder.db"
    prediction_db = tmp_path / "transformer_predictions.db"
    config_path = tmp_path / "transformer_config.json"
    output_db = tmp_path / "paper.db"
    _create_transformer_prediction_inputs(recorder_db, prediction_db)
    _write_transformer_prediction_config(config_path, prediction_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-transformer-prediction-paper-strategy",
            "--db",
            str(recorder_db),
            "--prediction-db",
            str(prediction_db),
            "--config",
            str(config_path),
            "--output-db",
            str(output_db),
            "--run-id",
            "run_cli",
            "--poll-sec",
            "0",
            "--paper-max-iterations",
            "1",
        ],
    )

    main_module.main()

    output = capsys.readouterr().out
    payload = json.loads(output[output.rfind("\n{") + 1 :])
    assert payload["status"] == "ok"
    assert payload["trades_opened"] == 2
    assert output_db.exists()


def _create_paper_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            strategy_id TEXT,
            signal_direction TEXT,
            status TEXT,
            skip_reason TEXT,
            stake_usd REAL,
            pnl_usd REAL,
            roi REAL,
            realized_pnl_usd REAL,
            realized_roi REAL,
            time_regime TEXT,
            liquidity_regime TEXT,
            btc_trend_regime TEXT,
            spread_regime TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            strategy_id TEXT,
            decision TEXT,
            rejection_reason TEXT,
            threshold_used REAL,
            min_time_until_resolution_sec REAL,
            max_time_until_resolution_sec REAL,
            max_probability_for_direction REAL,
            starting_bankroll_usd REAL
        );
        CREATE TABLE multi_strategy_paper_trader_heartbeats (
            timestamp TEXT PRIMARY KEY,
            run_id TEXT,
            candidates_logged INTEGER,
            candidate_rows_total INTEGER,
            candidate_rows_written_this_poll INTEGER,
            candidate_decisions_evaluated_this_poll INTEGER,
            trades_opened INTEGER,
            skips_logged INTEGER
        );
        """
    )
    rows = [
        ("2026-01-01T00:00:00+00:00", "s1", "NO", "closed", None, 1.0, None, None, 0.2, 0.2, "30-60s", "normal", "weak_down", "tight"),
        ("2026-01-01T00:00:01+00:00", "s1", "NO", "closed", None, 1.0, None, None, -0.1, -0.1, "30-60s", "thin", "strong_down", "wide"),
        ("2026-01-01T00:00:02+00:00", "s2", "YES", "settled", None, 1.0, 0.5, 0.5, None, None, "0-30s", "deep", "weak_up", "tight"),
        ("2026-01-01T00:00:03+00:00", "s2", "YES", "skipped", "threshold", 1.0, None, None, None, None, "0-30s", "deep", "weak_up", "tight"),
    ]
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, strategy_id, signal_direction, status, skip_reason,
            stake_usd, pnl_usd, roi, realized_pnl_usd, realized_roi,
            time_regime, liquidity_regime, btc_trend_regime, spread_regime
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.executemany(
        """
        INSERT INTO paper_trade_candidates (
            created_at, strategy_id, decision, rejection_reason, threshold_used,
            min_time_until_resolution_sec, max_time_until_resolution_sec,
            max_probability_for_direction, starting_bankroll_usd
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("2026-01-01T00:00:00+00:00", "s1", "TRADE", None, 0.85, 30.0, 60.0, 0.9, 100.0),
            ("2026-01-01T00:00:01+00:00", "s1", "SKIP", "min_estimated_edge", 0.85, 30.0, 60.0, 0.9, 100.0),
            ("2026-01-01T00:00:02+00:00", "s2", "SKIP", "threshold", 0.90, 60.0, 120.0, 0.95, 1000.0),
        ],
    )
    conn.execute(
        """
        INSERT INTO multi_strategy_paper_trader_heartbeats (
            timestamp, run_id, candidates_logged, candidate_rows_total,
            candidate_rows_written_this_poll, candidate_decisions_evaluated_this_poll,
            trades_opened, skips_logged
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("2026-01-01T00:00:04+00:00", "run_worker", 2, 3, 2, 4, 1, 2),
    )
    conn.commit()
    conn.close()


def test_report_paper_strategy_experiment_aggregates_synthetic_paper_db(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    paper_dir = experiment_dir / "paper_dbs"
    paper_dir.mkdir(parents=True)
    paper_db = paper_dir / "worker.db"
    _create_paper_db(paper_db)
    manifest = {
        "experiment_id": "exp_report",
        "recorder_db_path": "data/recorder.db",
        "model_path": "model.joblib",
        "feature_columns_path": "feature_columns.json",
        "workers": [
            {
                "config_path": "config.json",
                "paper_db_path": str(paper_db),
                "run_id": "run_worker",
            }
        ],
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_paper_strategy_experiment_report(
        experiment_dir=str(experiment_dir),
        recorder_db_path="data/recorder.db",
        output_path=str(experiment_dir / "report.json"),
        output_txt_path=str(experiment_dir / "report.txt"),
    )

    summary = report["summary"]
    assert summary["total_trades"] == 4
    assert summary["closed_trades"] == 2
    assert summary["settled_trades"] == 1
    assert summary["skipped_trades"] == 1
    assert summary["pnl"] == 0.6
    assert summary["roi"] == 0.2
    assert summary["win_rate"] == round(2 / 3, 10)
    assert summary["skip_reason_counts"] == {"threshold": 1}
    assert summary["candidate_decision_counts"] == {"SKIP": 2, "TRADE": 1}
    assert summary["candidate_total"] == 3
    assert summary["candidate_rejection_counts"] == {
        "min_estimated_edge": 1,
        "threshold": 1,
    }
    assert summary["candidate_rejection_groups"]["time_window"][0]["time_window"] in {
        "30-60s",
        "60-120s",
    }
    assert summary["candidate_rejection_groups"]["threshold"][0]["count"] == 1
    assert summary["warnings"]
    by_strategy = {
        row["strategy_id"]: row
        for row in summary["grouped_performance"]["strategy_id"]
    }
    assert by_strategy["s1"]["pnl"] == 0.1
    assert by_strategy["s2"]["pnl"] == 0.5
    assert (experiment_dir / "report.json").exists()
    assert (experiment_dir / "report.txt").exists()
    text = render_paper_strategy_experiment_report(report)
    assert "Compact Summary" in text
    assert "Candidate Rejections By Reason" in text
    assert "  threshold: count=1" in text
    assert "  None: count=" not in text
    assert "Candidate Rejections By Time Window" in text
    assert "sample size is too small" in text
    assert '{"' not in text


def test_monitor_paper_strategy_experiment_reports_worker_activity(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    paper_dir = experiment_dir / "paper_dbs"
    log_dir = experiment_dir / "logs"
    paper_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    paper_db = paper_dir / "worker.db"
    log_path = log_dir / "worker.log"
    _create_paper_db(paper_db)
    log_path.write_text(
        json.dumps(
            {
                "event": "multi_strategy_paper_trader_heartbeat",
                "timestamp": "2026-01-01T00:00:04+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": "exp_monitor",
        "workers": [
            {
                "worker_id": "worker",
                "config_path": "config.json",
                "paper_db_path": str(paper_db),
                "log_path": str(log_path),
                "run_id": "run_worker",
                "tmux_session": "paper_worker",
            }
        ],
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = monitor_paper_strategy_experiment(
        experiment_dir=str(experiment_dir),
        now=datetime.fromisoformat("2026-01-01T00:00:05+00:00"),
        active_tmux_sessions={"paper_worker"},
    )

    assert report["summary"]["active_tmux_workers"] == 1
    assert report["summary"]["total_trades"] == 4
    assert report["summary"]["total_candidates"] == 3
    assert report["summary"]["top_candidate_rejection_reasons"]["threshold"] == 1
    assert report["summary"]["top_strategy_trade_candidate_counts"]["s1"] == 1
    worker = report["workers"][0]
    assert worker["latest_poll_candidate_delta"] == 2
    assert worker["latest_poll_trade_delta"] == 1
    assert worker["warnings"] == []
    text = render_paper_strategy_experiment_monitor(report)
    assert "Paper Strategy Experiment Monitor" in text
    assert "delta_candidates=2" in text
    assert "Top Strategy IDs By Closed PnL" in text


def test_monitor_paper_strategy_experiment_warns_for_stale_worker(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    paper_dir = experiment_dir / "paper_dbs"
    log_dir = experiment_dir / "logs"
    paper_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    paper_db = paper_dir / "worker.db"
    log_path = log_dir / "worker.log"
    _create_paper_db(paper_db)
    log_path.write_text(
        '{"event":"multi_strategy_paper_trader_heartbeat","timestamp":"2026-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": "exp_stale",
        "workers": [
            {
                "worker_id": "worker",
                "paper_db_path": str(paper_db),
                "log_path": str(log_path),
                "tmux_session": "paper_worker",
            }
        ],
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = monitor_paper_strategy_experiment(
        experiment_dir=str(experiment_dir),
        stale_after_sec=1.0,
        now=datetime.fromisoformat("2026-01-01T00:05:00+00:00"),
        active_tmux_sessions=set(),
    )

    assert "tmux_session_inactive" in report["workers"][0]["warnings"]
    assert "worker_stale" in report["workers"][0]["warnings"]
