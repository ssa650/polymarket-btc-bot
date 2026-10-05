from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.paper_strategy_leaderboard import (
    build_paper_strategy_leaderboard,
    render_paper_strategy_leaderboard,
)


NOW = datetime(2026, 4, 29, 15, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _create_db(path) -> None:
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
                signal_direction TEXT,
                probability_for_direction REAL,
                estimated_edge REAL,
                entry_price REAL,
                adjusted_entry_price REAL,
                time_until_resolution REAL,
                status TEXT,
                pnl_usd REAL,
                roi REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_trade(
    path,
    *,
    strategy_id: str,
    direction: str,
    status: str = "settled",
    pnl: float | None = 0.5,
    roi: float | None = 0.5,
    probability: float = 0.7,
    edge: float = 0.1,
    entry: float = 0.55,
    adjusted: float = 0.56,
    seconds_until_close: float = 90.0,
    index: int = 0,
) -> None:
    created = NOW + timedelta(seconds=index)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, strategy_id,
                signal_direction, probability_for_direction, estimated_edge,
                entry_price, adjusted_entry_price, time_until_resolution,
                status, pnl_usd, roi
            ) VALUES (?, 'run', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(created),
                f"{strategy_id}_{direction}_{index}",
                _iso(created),
                strategy_id,
                direction,
                probability,
                edge,
                entry,
                adjusted,
                seconds_until_close,
                status,
                pnl,
                roi,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_group(
    path,
    *,
    strategy_id: str,
    direction: str,
    settled: int,
    wins: int,
    open_count: int = 0,
    awaiting_count: int = 0,
    edge: float = 0.1,
    start_index: int = 0,
) -> None:
    idx = start_index
    for _ in range(wins):
        _insert_trade(
            path,
            strategy_id=strategy_id,
            direction=direction,
            pnl=0.8,
            roi=0.8,
            edge=edge,
            index=idx,
        )
        idx += 1
    for _ in range(settled - wins):
        _insert_trade(
            path,
            strategy_id=strategy_id,
            direction=direction,
            pnl=-1.0,
            roi=-1.0,
            edge=edge,
            index=idx,
        )
        idx += 1
    for _ in range(open_count):
        _insert_trade(
            path,
            strategy_id=strategy_id,
            direction=direction,
            status="open",
            pnl=None,
            roi=None,
            edge=edge,
            index=idx,
        )
        idx += 1
    for _ in range(awaiting_count):
        _insert_trade(
            path,
            strategy_id=strategy_id,
            direction=direction,
            status="awaiting_resolution",
            pnl=None,
            roi=None,
            edge=edge,
            index=idx,
        )
        idx += 1


def test_groups_by_strategy_id_and_direction(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _seed_group(db, strategy_id="s1", direction="YES", settled=2, wins=1)
    _seed_group(db, strategy_id="s1", direction="NO", settled=1, wins=1, start_index=10)
    _insert_trade(db, strategy_id="s1", direction="YES", status="skipped", pnl=None, roi=None)

    report = build_paper_strategy_leaderboard(
        paper_db_path=str(db),
        min_settled=0,
    )
    rows = {
        (row["strategy_id"], row["signal_direction"]): row
        for row in report["rows"]
    }

    assert set(rows) == {("s1", "YES"), ("s1", "NO")}
    assert rows[("s1", "YES")]["total_trades"] == 2
    assert rows[("s1", "YES")]["settled_trades"] == 2
    assert rows[("s1", "NO")]["win_rate_settled"] == 1.0


def test_min_settled_and_direction_filters(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _seed_group(db, strategy_id="s1", direction="YES", settled=2, wins=2)
    _seed_group(db, strategy_id="s2", direction="NO", settled=4, wins=3, start_index=10)

    report = build_paper_strategy_leaderboard(
        paper_db_path=str(db),
        min_settled=3,
        direction="NO",
    )

    assert [(row["strategy_id"], row["signal_direction"]) for row in report["rows"]] == [
        ("s2", "NO")
    ]


def test_strategy_id_filter_accepts_multiple_ids(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _seed_group(db, strategy_id="s1", direction="YES", settled=1, wins=1)
    _seed_group(db, strategy_id="s2", direction="YES", settled=1, wins=1, start_index=10)
    _seed_group(db, strategy_id="s3", direction="YES", settled=1, wins=1, start_index=20)

    report = build_paper_strategy_leaderboard(
        paper_db_path=str(db),
        min_settled=0,
        strategy_ids=["s1", "s3"],
    )

    assert {row["strategy_id"] for row in report["rows"]} == {"s1", "s3"}


def test_reliability_tiers_and_warnings(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _seed_group(
        db,
        strategy_id="too_early_bad",
        direction="YES",
        settled=2,
        wins=0,
        awaiting_count=2,
        edge=-0.1,
    )
    _seed_group(db, strategy_id="early", direction="YES", settled=30, wins=20, start_index=20)
    _seed_group(db, strategy_id="usable", direction="YES", settled=100, wins=60, start_index=100)
    _seed_group(db, strategy_id="strong", direction="YES", settled=300, wins=200, start_index=300)

    report = build_paper_strategy_leaderboard(
        paper_db_path=str(db),
        min_settled=0,
    )
    rows = {row["strategy_id"]: row for row in report["rows"]}

    assert rows["too_early_bad"]["reliability_tier"] == "too_early"
    assert rows["early"]["reliability_tier"] == "early"
    assert rows["usable"]["reliability_tier"] == "usable_sample"
    assert rows["strong"]["reliability_tier"] == "strong_sample"
    assert set(rows["too_early_bad"]["warnings"]) == {
        "low_sample",
        "negative_pnl",
        "win_rate_below_50",
        "avg_roi_below_zero",
        "high_awaiting_resolution_ratio",
    }


def test_json_and_csv_outputs_and_render(tmp_path) -> None:
    db = tmp_path / "paper.db"
    json_path = tmp_path / "leaderboard.json"
    csv_path = tmp_path / "leaderboard.csv"
    _create_db(db)
    _seed_group(db, strategy_id="s1", direction="YES", settled=1, wins=1)

    report = build_paper_strategy_leaderboard(
        paper_db_path=str(db),
        min_settled=0,
        output_path=str(json_path),
        output_csv_path=str(csv_path),
    )
    rendered = render_paper_strategy_leaderboard(report)

    assert json.loads(json_path.read_text(encoding="utf-8"))["row_count"] == 1
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["strategy_id"] == "s1"
    assert "Paper Strategy Leaderboard" in rendered
    assert "s1" in rendered


def test_paper_strategy_leaderboard_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "paper-strategy-leaderboard",
            "--paper-db",
            "data/paper_trades_multi_strategy_guarded.db",
            "--min-settled",
            "20",
            "--strategy-id",
            "s1",
            "--strategy-id",
            "s2",
        ],
    )

    args = main.parse_args()

    assert args.command == "paper-strategy-leaderboard"
    assert args.min_settled == 20
    assert args.strategy_id == ["s1", "s2"]
