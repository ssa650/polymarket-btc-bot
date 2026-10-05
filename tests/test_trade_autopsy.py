from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.trade_autopsy import (
    build_policy_specs,
    build_trade_autopsy_report,
    load_paper_trade_rows,
    replay_trade_rows,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _iso(seconds: int) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _setup_recorder(path, *, include_path: bool = True) -> None:
    conn = sqlite3.connect(path)
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
            last_trade_price REAL
        );
        """
    )
    if include_path:
        rows = [
            ("run_a", "m_yes", _iso(0), 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50),
            ("run_a", "m_yes", _iso(5), 0.56, 0.58, 0.42, 0.44, 0.57, 0.43, 0.56),
            ("run_a", "m_yes", _iso(15), 0.65, 0.67, 0.35, 0.37, 0.66, 0.36, 0.65),
            ("run_a", "m_yes", _iso(30), 0.45, 0.47, 0.55, 0.57, 0.46, 0.56, 0.45),
            ("run_a", "m_no", _iso(0), 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.48),
            ("run_a", "m_no", _iso(5), 0.44, 0.46, 0.56, 0.58, 0.45, 0.57, 0.56),
            ("run_a", "m_no", _iso(15), 0.36, 0.38, 0.64, 0.66, 0.37, 0.65, 0.64),
            ("run_a", "m_no", _iso(30), 0.55, 0.57, 0.45, 0.47, 0.56, 0.46, 0.45),
        ]
        conn.executemany(
            """
            INSERT INTO market_snapshots (
                run_id, market_id, timestamp, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, mid_price_yes, mid_price_no, last_trade_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            exit_reason TEXT,
            realized_pnl_usd REAL,
            realized_roi REAL,
            spread_cents_at_entry REAL,
            entry_latency_sec REAL,
            entry_price_drift REAL,
            liquidity_fill_fraction_used REAL,
            max_fillable_shares REAL,
            requested_shares REAL,
            feature_ready INTEGER,
            strict_validation_passed INTEGER,
            snapshot_quality_status TEXT
        );
        """
    )
    rows = [
        (
            _iso(0), "run_a", "m_yes", _iso(0), _iso(0), _iso(35), "closed", "YES",
            "s_yes", "gb", 0.8, 0.8, 0.3, 35.0, "0-30s", "weak_up", "low",
            "tight", "normal", 0.52, 0.50, 1.0, "fixed_horizon", 0.3, 0.3,
            1.0, 1.0, 0.0, 0.5, 10.0, 2.0, 1, 1, "ok",
        ),
        (
            _iso(0), "run_a", "m_no", _iso(0), _iso(0), _iso(35), "closed", "NO",
            "s_no", "gb", 0.2, 0.8, 0.32, 35.0, "30-60s", "weak_down", "medium",
            "normal", "thin", 0.50, 0.50, 1.0, "fixed_horizon", 0.2, 0.2,
            1.0, 1.0, 0.0, 0.5, 10.0, 2.0, 1, 1, "ok",
        ),
        (
            _iso(0), "run_a", "m_missing", _iso(0), _iso(0), _iso(35), "closed", "YES",
            "s_missing", "gb", 0.7, 0.7, 0.2, 35.0, "60-120s", "flat", "low",
            "wide", "thin", 0.50, 0.50, 1.0, "fixed_horizon", None, None,
            None, None, None, None, None, None, 1, 1, "ok",
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
            stake_usd, exit_reason, realized_pnl_usd, realized_roi,
            spread_cents_at_entry, entry_latency_sec, entry_price_drift,
            liquidity_fill_fraction_used, max_fillable_shares, requested_shares,
            feature_ready, strict_validation_passed, snapshot_quality_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_fixed_horizon_pnl_for_yes_and_no_uses_side_specific_bid_path(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    recorder = sqlite3.connect(recorder_db)
    recorder.row_factory = sqlite3.Row
    paper_rows = load_paper_trade_rows(
        paper_db_paths=[str(paper_db)],
        status_filter=("closed",),
        strategy_ids=None,
        market_ids=None,
        limit=2,
        include_skipped=False,
    )
    rows = replay_trade_rows(
        recorder,
        paper_rows,
        policies=[{"type": "fixed_horizon", "name": "fixed_horizon_15s", "seconds": 15}],
        near_close_sec=3,
    )
    recorder.close()

    by_market = {row["market_id"]: row for row in rows}
    assert by_market["m_yes"]["simulated_exit_price"] == 0.65
    assert by_market["m_yes"]["simulated_roi"] == 0.3
    assert by_market["m_no"]["simulated_exit_price"] == 0.64
    assert by_market["m_no"]["simulated_roi"] == 0.28


def test_stop_loss_and_take_profit_policies_hit_first_threshold(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    recorder = sqlite3.connect(recorder_db)
    recorder.row_factory = sqlite3.Row
    paper_rows = load_paper_trade_rows(
        paper_db_paths=[str(paper_db)],
        status_filter=("closed",),
        strategy_ids=None,
        market_ids=None,
        limit=2,
        include_skipped=False,
    )
    policies = [
        {"type": "stop_loss_take_profit", "name": "tp", "stop_loss": -0.2, "take_profit": 0.25},
    ]
    rows = replay_trade_rows(recorder, paper_rows, policies=policies, near_close_sec=3)
    recorder.close()

    by_market = {row["market_id"]: row for row in rows}
    assert by_market["m_yes"]["hit_reason"] == "take_profit"
    assert by_market["m_yes"]["simulated_exit_time"] == _iso(15)
    assert by_market["m_no"]["hit_reason"] == "take_profit"
    assert by_market["m_no"]["simulated_exit_time"] == _iso(15)


def test_stop_loss_hit_before_take_profit(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    recorder = sqlite3.connect(recorder_db)
    recorder.row_factory = sqlite3.Row
    paper_rows = load_paper_trade_rows(
        paper_db_paths=[str(paper_db)],
        status_filter=("closed",),
        strategy_ids=["s_yes"],
        market_ids=None,
        limit=None,
        include_skipped=False,
    )
    rows = replay_trade_rows(
        recorder,
        paper_rows,
        policies=[{"type": "stop_loss_take_profit", "name": "sl", "stop_loss": -0.05, "take_profit": 0.5}],
        near_close_sec=3,
    )
    recorder.close()

    assert rows[0]["hit_reason"] == "stop_loss"
    assert rows[0]["simulated_exit_time"] == _iso(30)


def test_missing_price_path_returns_no_price_path(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    recorder = sqlite3.connect(recorder_db)
    recorder.row_factory = sqlite3.Row
    paper_rows = load_paper_trade_rows(
        paper_db_paths=[str(paper_db)],
        status_filter=("closed",),
        strategy_ids=["s_missing"],
        market_ids=None,
        limit=None,
        include_skipped=False,
    )
    rows = replay_trade_rows(
        recorder,
        paper_rows,
        policies=[{"type": "fixed_horizon", "name": "fixed_horizon_5s", "seconds": 5}],
        near_close_sec=3,
    )
    recorder.close()

    assert rows[0]["hit_reason"] == "no_price_path"
    assert rows[0]["simulated_pnl_usd"] is None


def test_trade_autopsy_cli_writes_parquet_csv_and_prints_summary(monkeypatch, tmp_path, capsys) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    output = tmp_path / "autopsy.parquet"
    output_csv = tmp_path / "autopsy.csv"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)
    import src.main as main_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "trade-autopsy",
            "--recorder-db",
            str(recorder_db),
            "--paper-db",
            str(paper_db),
            "--output",
            str(output),
            "--output-csv",
            str(output_csv),
            "--fixed-horizons-sec",
            "15",
            "--stop-loss-take-profit",
            "-0.2:0.3",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["rows_loaded"] == 3
    assert output.exists()
    assert output_csv.exists()
    csv_rows = list(csv.DictReader(output_csv.open()))
    assert {row["policy_name"] for row in csv_rows} >= {"fixed_horizon_15s", "stop_loss_20_take_profit_30"}

