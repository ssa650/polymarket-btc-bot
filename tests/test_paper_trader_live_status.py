from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from src.paper_trader_live_status import (
    build_paper_trader_live_status,
    render_paper_trader_live_status,
)


def _create_new_paper_db(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                market_id TEXT,
                strategy_id TEXT,
                signal_direction TEXT,
                status TEXT,
                skip_reason TEXT,
                stake_usd REAL,
                entry_price REAL,
                adjusted_entry_price REAL,
                payout_usd REAL,
                pnl_usd REAL,
                roi REAL,
                realized_pnl_usd REAL,
                realized_roi REAL,
                bankroll_after_trade REAL,
                starting_bankroll_usd REAL,
                open_exposure_before_trade REAL,
                probability_for_direction REAL,
                estimated_edge REAL,
                time_until_resolution REAL,
                realistic_execution_skip_reason TEXT,
                liquidity_skip_reason TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO paper_trades (
                created_at, market_id, strategy_id, signal_direction, status,
                skip_reason, stake_usd, entry_price, adjusted_entry_price,
                payout_usd, pnl_usd, roi, realized_pnl_usd, realized_roi,
                bankroll_after_trade, starting_bankroll_usd, open_exposure_before_trade,
                probability_for_direction, estimated_edge, time_until_resolution,
                realistic_execution_skip_reason, liquidity_skip_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "2026-01-01T00:00:01+00:00",
                    "m_open",
                    "s_open",
                    "YES",
                    "open",
                    None,
                    1.0,
                    0.7,
                    0.71,
                    None,
                    None,
                    None,
                    None,
                    None,
                    100.0,
                    100.0,
                    1.0,
                    0.78,
                    0.07,
                    45.0,
                    None,
                    None,
                ),
                (
                    "2026-01-01T00:00:02+00:00",
                    "m_win",
                    "s_win",
                    "YES",
                    "settled",
                    None,
                    1.0,
                    0.6,
                    0.61,
                    1.6393,
                    0.6393,
                    0.6393,
                    None,
                    None,
                    100.64,
                    100.0,
                    0.0,
                    0.82,
                    0.21,
                    25.0,
                    None,
                    None,
                ),
                (
                    "2026-01-01T00:00:03+00:00",
                    "m_loss",
                    "s_loss",
                    "NO",
                    "closed",
                    None,
                    1.0,
                    0.55,
                    0.56,
                    None,
                    None,
                    None,
                    -0.2,
                    -0.2,
                    100.44,
                    100.0,
                    0.0,
                    0.71,
                    0.15,
                    30.0,
                    None,
                    None,
                ),
                (
                    "2026-01-01T00:00:04+00:00",
                    "m_skip",
                    "s_skip",
                    "YES",
                    "skipped",
                    "quote_after_latency_missing",
                    1.0,
                    None,
                    None,
                    None,
                    0.0,
                    None,
                    None,
                    None,
                    100.44,
                    100.0,
                    0.0,
                    0.8,
                    0.1,
                    35.0,
                    "quote_after_latency_missing",
                    "liquidity_missing",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _create_old_paper_db(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                market_id TEXT,
                status TEXT,
                stake_usd REAL,
                pnl_usd REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO paper_trades (created_at, market_id, status, stake_usd, pnl_usd)
            VALUES ('2026-01-01T00:00:00+00:00', 'old_market', 'settled', 1.0, 0.25)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _create_empty_paper_db(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, created_at TEXT, status TEXT)")
        conn.commit()
    finally:
        conn.close()


def test_live_status_handles_new_schema(tmp_path) -> None:
    paper_db = tmp_path / "paper_new.db"
    _create_new_paper_db(paper_db)

    report = build_paper_trader_live_status(
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        paper_db_paths=[str(paper_db)],
        now=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    trader = report["paper_traders"]["traders"][0]
    rendered = render_paper_trader_live_status(report, show_skips=True, show_open=True, show_closed=True)

    assert trader["total_trades"] == 4
    assert trader["open_awaiting_trades"] == 1
    assert trader["settled_trades"] == 1
    assert trader["closed_trades"] == 1
    assert trader["skipped_trades"] == 1
    assert trader["realized_pnl"] == 0.4393
    assert trader["current_bankroll"] == 100.44
    assert "quote_after_latency_missing" in rendered
    assert "s_win" in rendered
    assert "s_loss" in rendered


def test_live_status_handles_old_schema(tmp_path) -> None:
    paper_db = tmp_path / "paper_old.db"
    _create_old_paper_db(paper_db)

    report = build_paper_trader_live_status(
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        paper_db_paths=[str(paper_db)],
        now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )
    trader = report["paper_traders"]["traders"][0]

    assert trader["status"] == "STALE"
    assert trader["settled_trades"] == 1
    assert trader["realized_pnl"] == 0.25
    assert trader["current_bankroll"] is None


def test_live_status_handles_no_trade_db(tmp_path) -> None:
    paper_db = tmp_path / "paper_empty.db"
    _create_empty_paper_db(paper_db)

    report = build_paper_trader_live_status(
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        paper_db_paths=[str(paper_db)],
    )
    trader = report["paper_traders"]["traders"][0]

    assert trader["status"] == "NO_TRADES"
    assert trader["total_trades"] == 0
    assert "no paper DBs configured" not in render_paper_trader_live_status(report)


def test_live_status_resolves_repeated_paths_and_globs(tmp_path) -> None:
    paper_a = tmp_path / "paper_a.db"
    paper_b = tmp_path / "paper_b.db"
    _create_empty_paper_db(paper_a)
    _create_empty_paper_db(paper_b)

    report = build_paper_trader_live_status(
        recorder_db_path=str(tmp_path / "missing_recorder.db"),
        paper_db_paths=[str(paper_a), str(paper_a)],
        paper_db_globs=[str(tmp_path / "paper_*.db")],
    )

    assert report["paper_db_paths"] == [str(paper_a), str(paper_b)]
    assert report["paper_traders"]["totals"]["paper_db_count"] == 2


def test_live_status_cli_registration(monkeypatch, tmp_path) -> None:
    import src.main as main_module
    import src.paper_trader_live_status as live_status_module

    calls: list[dict[str, object]] = []
    settings = SimpleNamespace(db_path=str(tmp_path / "recorder.db"), log_level="CRITICAL", log_json=False)

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "paper-trader-live-status",
        "--recorder-db",
        str(tmp_path / "recorder.db"),
        "--paper-db",
        str(tmp_path / "paper_a.db"),
        "--paper-db",
        str(tmp_path / "paper_b.db"),
        "--paper-db-glob",
        str(tmp_path / "paper_*.db"),
        "--limit-activity",
        "7",
        "--show-skips",
        "--show-open",
        "--show-closed",
    ])
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(live_status_module, "run_paper_trader_live_status", fake_run)

    args = main_module.parse_args()
    assert args.command == "paper-trader-live-status"
    assert args.paper_db == [str(tmp_path / "paper_a.db"), str(tmp_path / "paper_b.db")]

    main_module.main()

    assert calls == [
        {
            "recorder_db_path": str(tmp_path / "recorder.db"),
            "paper_db_paths": [str(tmp_path / "paper_a.db"), str(tmp_path / "paper_b.db")],
            "paper_db_globs": [str(tmp_path / "paper_*.db")],
            "limit_activity": 7,
            "show_skips": True,
            "show_open": True,
            "show_closed": True,
            "watch_sec": None,
        }
    ]
