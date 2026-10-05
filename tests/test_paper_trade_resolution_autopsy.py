from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.paper_trade_resolution_autopsy import (
    build_paper_trade_resolution_autopsy,
    render_paper_trade_resolution_autopsy,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _iso(offset_sec: int) -> str:
    return (BASE + timedelta(seconds=offset_sec)).isoformat()


def _setup_recorder(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE markets (
            run_id TEXT,
            market_id TEXT,
            start_time TEXT,
            close_time TEXT,
            resolved INTEGER,
            winning_asset_id TEXT,
            winning_outcome TEXT,
            yes_token_id TEXT,
            no_token_id TEXT
        );
        CREATE TABLE btc_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            source TEXT,
            price REAL,
            local_arrival_iso TEXT
        );
        """
    )
    markets = [
        ("run_autopsy", "m_yes", _iso(0), _iso(300), 1, None, "Up", "yes1", "no1"),
        ("run_autopsy", "m_btc", _iso(300), _iso(600), 0, None, None, "yes2", "no2"),
    ]
    conn.executemany(
        """
        INSERT INTO markets (
            run_id, market_id, start_time, close_time, resolved,
            winning_asset_id, winning_outcome, yes_token_id, no_token_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        markets,
    )
    btc_rows = [
        ("run_autopsy", "polymarket_rtds_chainlink", 100.0, _iso(0)),
        ("run_autopsy", "polymarket_rtds_chainlink", 103.0, _iso(300)),
        ("run_autopsy", "polymarket_rtds_chainlink", 105.0, _iso(301)),
        ("run_autopsy", "polymarket_rtds_chainlink", 101.0, _iso(600)),
    ]
    conn.executemany(
        "INSERT INTO btc_prices (run_id, source, price, local_arrival_iso) VALUES (?, ?, ?, ?)",
        btc_rows,
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
            status TEXT,
            signal_direction TEXT,
            stake_usd REAL,
            adjusted_entry_price REAL,
            exit_price REAL,
            adjusted_exit_price REAL,
            realized_pnl_usd REAL,
            pnl_usd REAL,
            roi REAL,
            probability_for_direction REAL,
            estimated_edge REAL,
            time_until_resolution REAL,
            market_start_time TEXT,
            market_close_time TEXT,
            time_regime TEXT,
            btc_trend_regime TEXT,
            liquidity_regime TEXT,
            spread_regime TEXT,
            volatility_regime TEXT
        );
        """
    )
    rows = [
        (
            _iso(10),
            "run_autopsy",
            "m_yes",
            "closed",
            "YES",
            1.0,
            0.50,
            0.60,
            0.60,
            0.20,
            None,
            None,
            0.76,
            0.26,
            45.0,
            _iso(0),
            _iso(300),
            "30-60s",
            "weak_up",
            "normal",
            "tight",
            "low",
        ),
        (
            _iso(320),
            "run_autopsy",
            "m_btc",
            "closed",
            "NO",
            1.0,
            0.40,
            0.50,
            0.50,
            0.25,
            None,
            None,
            0.72,
            0.32,
            90.0,
            _iso(300),
            _iso(600),
            "60-120s",
            "weak_down",
            "thin",
            "normal",
            "medium",
        ),
        (
            _iso(330),
            "run_autopsy",
            "m_btc",
            "skipped",
            "NO",
            1.0,
            0.40,
            None,
            None,
            None,
            None,
            None,
            0.72,
            0.32,
            90.0,
            _iso(300),
            _iso(600),
            "60-120s",
            "weak_down",
            "thin",
            "normal",
            "medium",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, run_id, market_id, status, signal_direction, stake_usd,
            adjusted_entry_price, exit_price, adjusted_exit_price, realized_pnl_usd,
            pnl_usd, roi, probability_for_direction, estimated_edge,
            time_until_resolution, market_start_time, market_close_time,
            time_regime, btc_trend_regime, liquidity_regime, spread_regime, volatility_regime
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_resolution_autopsy_compares_cashout_to_resolution_and_writes_csv(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    output_root = tmp_path / "analytics"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)

    report = build_paper_trade_resolution_autopsy(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(paper_db),
        output_root=str(output_root),
        now=BASE,
    )

    assert report["status"] == "ok"
    assert report["trade_count"] == 2
    csv_path = output_root / "run_autopsy" / "resolution_autopsy.csv"
    assert report["csv_path"] == str(csv_path)
    rows = list(csv.DictReader(csv_path.open()))
    by_market = {row["market_id"]: row for row in rows}
    assert by_market["m_yes"]["actual_outcome"] == "YES"
    assert by_market["m_yes"]["outcome_source"] == "markets_resolution"
    assert float(by_market["m_yes"]["fixed_horizon_pnl"]) == 0.2
    assert float(by_market["m_yes"]["hold_to_resolution_pnl"]) == 1.0
    assert by_market["m_yes"]["would_win_at_resolution"] == "True"
    assert by_market["m_btc"]["actual_outcome"] == "NO"
    assert by_market["m_btc"]["outcome_source"] == "btc_price_estimate"
    assert float(by_market["m_btc"]["btc_move"]) == -2.0
    assert float(by_market["m_btc"]["hold_to_resolution_pnl"]) == 1.5


def test_resolution_autopsy_grouped_summary_and_render(tmp_path) -> None:
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper.db"
    _setup_recorder(recorder_db)
    _setup_paper(paper_db)

    report = build_paper_trade_resolution_autopsy(
        recorder_db_path=str(recorder_db),
        paper_db_path=str(paper_db),
        output_root=str(tmp_path / "analytics"),
        now=BASE,
    )

    by_direction = {
        row["group"]: row
        for row in report["grouped_summaries"]["signal_direction"]
    }
    assert by_direction["YES"]["trades"] == 1
    assert by_direction["NO"]["resolution_win_rate"] == 1.0
    by_probability = {
        row["group"]: row
        for row in report["grouped_summaries"]["probability_bucket"]
    }
    assert by_probability["0.7-0.8"]["trades"] == 2
    rendered = render_paper_trade_resolution_autopsy(report)
    assert "Paper Trade Resolution Autopsy" in rendered
    assert "signal_direction" in rendered


def test_resolution_autopsy_cli_is_registered(monkeypatch, tmp_path, capsys) -> None:
    import src.main as main_module
    import src.paper_trade_resolution_autopsy as autopsy_module

    def fake_build(**kwargs):
        return {
            "status": "ok",
            "recorder_db_path": kwargs["recorder_db_path"],
            "paper_db_path": kwargs["paper_db_path"],
            "trade_count": 0,
            "csv_path": "data/analytics/run/resolution_autopsy.csv",
            "grouped_summaries": {},
        }

    monkeypatch.setattr(autopsy_module, "build_paper_trade_resolution_autopsy", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "paper-trade-resolution-autopsy",
            "--recorder-db",
            str(tmp_path / "recorder.db"),
            "--paper-db",
            str(tmp_path / "paper.db"),
        ],
    )

    main_module.main()

    output = capsys.readouterr().out
    assert "Paper Trade Resolution Autopsy" in output
    assert str(tmp_path / "paper.db") in output
