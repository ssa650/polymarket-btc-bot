from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.baseline_paper_trader import connect_paper_output_db, ensure_paper_schema
from src.btc_price_feed import POLYMARKET_RTDS_BINANCE_SOURCE, POLYMARKET_RTDS_CHAINLINK_SOURCE
from src.paper_trader_eligibility import build_paper_trader_eligibility_report


NOW = datetime(2026, 4, 28, 22, 0, tzinfo=timezone.utc)


class ProbabilityBySignalModel:
    def predict_proba(self, matrix):
        return [[1.0 - float(row[0]), float(row[0])] for row in matrix]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _write_model(path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ProbabilityBySignalModel(), path)


def _create_recorder_db(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE markets (
                run_id TEXT,
                market_id TEXT,
                question TEXT,
                start_time TEXT,
                close_time TEXT,
                phase TEXT,
                tracking_state TEXT
            );
            CREATE TABLE market_snapshots (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                best_bid_yes REAL,
                best_ask_yes REAL,
                best_bid_no REAL,
                best_ask_no REAL,
                has_orderbook INTEGER,
                has_trade_data INTEGER,
                strict_validation_passed INTEGER
            );
            CREATE TABLE features (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                signal REAL,
                feature_ready INTEGER,
                strict_validation_passed INTEGER,
                is_gap_affected INTEGER,
                snapshot_quality_status TEXT,
                time_until_resolution REAL
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
        conn.commit()
    finally:
        conn.close()


def _insert_row(
    path,
    *,
    run_id: str,
    market_id: str,
    timestamp: datetime,
    signal: float = 0.8,
    feature_ready: int = 1,
    strict_validation_passed: int = 1,
    is_gap_affected: int = 0,
    snapshot_quality_status: str = "ok",
    has_orderbook: int = 1,
    has_trade_data: int = 1,
    btc_age_sec: float = 1.0,
    binance_age_sec: float = 1.0,
) -> None:
    start = timestamp - timedelta(seconds=60)
    close = timestamp + timedelta(seconds=120)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO markets VALUES (?, ?, ?, ?, ?, 'active', 'selected_active')",
            (run_id, market_id, "BTC Up or Down", _iso(start), _iso(close)),
        )
        conn.execute(
            """
            INSERT INTO market_snapshots VALUES (
                ?, ?, ?, 0.49, 0.50, 0.49, 0.50, ?, ?, 1
            )
            """,
            (run_id, _iso(timestamp), market_id, has_orderbook, has_trade_data),
        )
        conn.execute(
            """
            INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _iso(timestamp),
                market_id,
                signal,
                feature_ready,
                strict_validation_passed,
                is_gap_affected,
                snapshot_quality_status,
                120.0,
            ),
        )
        for idx, (source, age) in enumerate(
            (
                (POLYMARKET_RTDS_CHAINLINK_SOURCE, btc_age_sec),
                (POLYMARKET_RTDS_BINANCE_SOURCE, binance_age_sec),
            ),
            start=1,
        ):
            btc_ts = timestamp - timedelta(seconds=age)
            conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, exchange_timestamp, local_arrival_ns, local_arrival_iso
                ) VALUES (?, ?, 100.0, ?, ?, ?)
                """,
                (run_id, source, _iso(btc_ts), idx, _iso(btc_ts)),
            )
        conn.commit()
    finally:
        conn.close()


def _create_paper_db(path) -> None:
    conn = connect_paper_output_db(str(path))
    try:
        ensure_paper_schema(conn)
    finally:
        conn.close()


def test_eligibility_report_counts_recent_gate_failures(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "feature_columns.json"
    _create_recorder_db(recorder_db)
    _create_paper_db(paper_db)
    _write_model(model_path)
    features_path.write_text(json.dumps(["signal"]), encoding="utf-8")
    _insert_row(recorder_db, run_id="ok", market_id="ok", timestamp=NOW)
    _insert_row(recorder_db, run_id="not_ready", market_id="not_ready", timestamp=NOW, feature_ready=0)
    _insert_row(recorder_db, run_id="stale", market_id="stale", timestamp=NOW, btc_age_sec=30.0)
    _insert_row(recorder_db, run_id="nosignal", market_id="nosignal", timestamp=NOW, signal=0.5)

    report = build_paper_trader_eligibility_report(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(paper_db),
        feature_columns_path=str(features_path),
        model_path=str(model_path),
        recent_minutes=30,
        now=NOW,
    )

    assert report["rows_considered"] == 4
    assert report["gate_failures"]["not_feature_ready"] == 1
    assert report["gate_failures"]["stale_canonical_btc"] == 1
    assert report["gate_failures"]["threshold"] == 1
    assert report["latest_seen_feature_timestamp"] == _iso(NOW)
    assert report["latest_eligible_feature_timestamp"] == _iso(NOW)
    assert report["markets_considered"] == 4
    assert report["live_active_markets_considered"] == 4
    assert report["open_trades"] == 0


def test_eligibility_report_command_is_offline(tmp_path, monkeypatch, capsys) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "feature_columns.json"
    _create_recorder_db(recorder_db)
    _create_paper_db(paper_db)
    _write_model(model_path)
    features_path.write_text(json.dumps(["signal"]), encoding="utf-8")
    _insert_row(recorder_db, run_id="ok", market_id="ok", timestamp=NOW)

    import src.main as main_module
    import src.paper_trader_eligibility as eligibility_module
    original_report = eligibility_module.build_paper_trader_eligibility_report
    monkeypatch.setattr(
        eligibility_module, 'build_paper_trader_eligibility_report',
        lambda **kwargs: original_report(**kwargs, now=NOW),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("paper-trader-eligibility-report must not start recorder/network")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "paper-trader-eligibility-report",
            "--db",
            str(recorder_db),
            "--paper-db",
            str(paper_db),
            "--model-path",
            str(model_path),
            "--feature-columns",
            str(features_path),
            "--recent-minutes",
            "100000",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    output = capsys.readouterr().out

    assert "Paper Trader Eligibility Report" in output
    assert "rows_considered=1" in output
