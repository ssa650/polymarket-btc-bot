from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.paper_trader_promotion_report import (
    build_paper_trader_promotion_report,
    render_paper_trader_promotion_report,
)


NOW = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _create_paper_db(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                run_id TEXT,
                market_id TEXT,
                signal_timestamp TEXT,
                strategy_id TEXT,
                strategy_name TEXT,
                signal_direction TEXT,
                status TEXT,
                stake_usd REAL,
                pnl_usd REAL,
                roi REAL,
                settled_at TEXT,
                realized_pnl_usd REAL,
                realized_roi REAL,
                exit_time TEXT,
                starting_bankroll_usd REAL,
                bankroll_after_trade REAL,
                open_exposure_before_trade REAL,
                max_open_exposure_usd REAL,
                liquidity_check_passed INTEGER,
                liquidity_skip_reason TEXT,
                realistic_execution_enabled INTEGER,
                realistic_execution_skip_reason TEXT,
                btc_trend_regime TEXT,
                volatility_regime TEXT,
                spread_regime TEXT,
                liquidity_regime TEXT,
                time_regime TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_trade(
    path,
    *,
    idx: int,
    strategy_id: str = "s1",
    direction: str = "YES",
    status: str = "settled",
    pnl: float = 0.2,
    roi: float = 0.2,
    created_at: datetime | None = None,
    stake: float = 1.0,
    liquidity_skip: str | None = None,
    realistic_skip: str | None = None,
    trend: str = "strong_up",
) -> None:
    created = created_at or (NOW - timedelta(hours=2) + timedelta(minutes=idx))
    settled = created + timedelta(minutes=5)
    conn = sqlite3.connect(str(path))
    try:
        if status == "closed":
            pnl_usd = None
            row_roi = None
            realized_pnl = pnl
            realized_roi = roi
            exit_time = _iso(settled)
            settled_at = None
        elif status == "settled":
            pnl_usd = pnl
            row_roi = roi
            realized_pnl = None
            realized_roi = None
            exit_time = None
            settled_at = _iso(settled)
        else:
            pnl_usd = None
            row_roi = None
            realized_pnl = None
            realized_roi = None
            exit_time = None
            settled_at = None
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp,
                strategy_id, strategy_name, signal_direction, status,
                stake_usd, pnl_usd, roi, settled_at,
                realized_pnl_usd, realized_roi, exit_time,
                starting_bankroll_usd, bankroll_after_trade,
                open_exposure_before_trade, max_open_exposure_usd,
                liquidity_check_passed, liquidity_skip_reason,
                realistic_execution_enabled, realistic_execution_skip_reason,
                btc_trend_regime, volatility_regime, spread_regime,
                liquidity_regime, time_regime
            ) VALUES (
                ?, 'run1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                100.0, ?, ?, 20.0, ?, ?, ?, ?, ?, 'medium', 'tight', 'deep', '30-60s'
            )
            """,
            (
                _iso(created),
                f"m{idx}",
                _iso(created + timedelta(seconds=1)),
                strategy_id,
                f"Strategy {strategy_id}",
                direction,
                status,
                stake,
                pnl_usd,
                row_roi,
                settled_at,
                realized_pnl,
                realized_roi,
                exit_time,
                100.0 + pnl if status in {"settled", "closed"} else None,
                float(idx % 5),
                None if liquidity_skip is None else 0,
                liquidity_skip,
                1 if realistic_skip is not None else None,
                realistic_skip,
                trend,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_promotion_report_ranks_multiple_dbs_and_writes_outputs(tmp_path) -> None:
    better = tmp_path / "better.db"
    worse = tmp_path / "worse.db"
    output = tmp_path / "promotion.json"
    output_txt = tmp_path / "promotion.txt"
    _create_paper_db(better)
    _create_paper_db(worse)
    for idx in range(120):
        _insert_trade(
            better,
            idx=idx,
            strategy_id="winner" if idx < 80 else "ok",
            direction="YES" if idx % 2 == 0 else "NO",
            pnl=0.2,
            roi=0.2,
        )
        _insert_trade(
            worse,
            idx=idx,
            strategy_id="loser",
            direction="YES" if idx % 2 == 0 else "NO",
            pnl=-0.1,
            roi=-0.1,
        )

    report = build_paper_trader_promotion_report(
        paper_db_paths=[str(worse), str(better)],
        output_path=str(output),
        output_txt_path=str(output_txt),
        now=NOW,
    )
    ranked = sorted(
        report["paper_dbs"],
        key=lambda item: int(item.get("deployability_rank") or 999),
    )

    assert ranked[0]["paper_db_path"] == str(better)
    assert ranked[0]["done_trades"] == 120
    assert ranked[0]["combined_realized_pnl"] == 24.0
    assert ranked[0]["done_win_rate"] == 1.0
    assert ranked[0]["bankroll_roi"] == 0.24
    assert ranked[0]["top_10_strategy_ids_by_pnl"][0]["strategy_id"] == "winner"
    assert "negative_pnl" in ranked[1]["warning_flags"]
    assert output.exists()
    assert output_txt.exists()
    assert "Paper Trader Promotion Report" in render_paper_trader_promotion_report(report)


def test_promotion_report_counts_blocks_and_regime_pnl(tmp_path) -> None:
    db = tmp_path / "blocked.db"
    _create_paper_db(db)
    _insert_trade(db, idx=1, pnl=0.5, roi=0.5, trend="strong_up")
    _insert_trade(db, idx=2, status="closed", pnl=-0.2, roi=-0.2, trend="weak_down")
    _insert_trade(
        db,
        idx=3,
        status="skipped",
        pnl=0.0,
        roi=0.0,
        liquidity_skip="liquidity_too_low",
    )
    _insert_trade(
        db,
        idx=4,
        status="skipped",
        pnl=0.0,
        roi=0.0,
        realistic_skip="spread_too_wide",
    )

    report = build_paper_trader_promotion_report(
        paper_db_paths=[str(db)],
        now=NOW,
    )
    item = report["paper_dbs"][0]

    assert item["combined_realized_pnl"] == 0.3
    assert item["liquidity_blocked_trades"] == 1
    assert item["realistic_execution_blocked_trades"] == 1
    assert item["pnl_by_regime_tag"]["btc_trend_regime"]["strong_up"] == 0.5
    assert item["pnl_by_regime_tag"]["btc_trend_regime"]["weak_down"] == -0.2


def test_promotion_report_warning_flags(tmp_path) -> None:
    db = tmp_path / "warnings.db"
    _create_paper_db(db)
    for idx in range(12):
        _insert_trade(db, idx=idx, direction="YES", pnl=-0.5, roi=-0.5)
    for idx in range(12, 25):
        _insert_trade(db, idx=idx, direction="YES", status="awaiting_resolution", pnl=0.0, roi=0.0)

    report = build_paper_trader_promotion_report(
        paper_db_paths=[str(db)],
        now=NOW,
    )
    flags = set(report["paper_dbs"][0]["warning_flags"])

    assert "fewer_than_100_done_trades" in flags
    assert "negative_pnl" in flags
    assert "win_rate_below_55_percent" in flags
    assert "avg_roi_below_0" in flags
    assert "too_many_awaiting_resolution" in flags
    assert "one_direction_dominates" in flags
    assert "high_drawdown" in flags


def test_promotion_report_handles_missing_table(tmp_path) -> None:
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()

    report = build_paper_trader_promotion_report(
        paper_db_paths=[str(db)],
        now=NOW,
    )

    assert report["paper_dbs"][0]["status"] == "error"
    assert report["paper_dbs"][0]["error"] == "paper_trades_table_missing"


def test_paper_trader_promotion_report_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "paper-trader-promotion-report",
            "--paper-db-path",
            "one.db",
            "--paper-db-path",
            "two.db",
            "--output",
            "promotion.json",
            "--output-txt",
            "promotion.txt",
        ],
    )

    args = main.parse_args()

    assert args.command == "paper-trader-promotion-report"
    assert args.paper_db_path == ["one.db", "two.db"]
    assert args.output == "promotion.json"
    assert args.output_txt == "promotion.txt"
