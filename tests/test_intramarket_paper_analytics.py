from __future__ import annotations

import sqlite3

from src.intramarket_paper_trader import (
    build_intramarket_paper_analytics_report,
    build_intramarket_strategy_leaderboard,
    ensure_intramarket_schema,
    render_intramarket_strategy_leaderboard,
)
from src.models import to_iso
from tests.test_baseline_paper_trader import NOW


def _db(tmp_path):
    path = tmp_path / "paper_intramarket.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_intramarket_schema(conn)
    conn.close()
    return path


def _insert_position(
    db_path,
    *,
    strategy_id: str = "s1",
    side: str = "NO",
    exit_type: str = "FIXED_HORIZON_EXIT",
    status: str = "closed",
    pnl: float = 0.25,
    roi: float = 0.25,
    exit_reason: str = "fixed_horizon",
    probability: float = 0.8,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO paper_positions (
                created_at, updated_at, run_id, strategy_id, strategy_name,
                market_id, side, status, entry_type, exit_type, entry_time,
                entry_feature_timestamp, entry_price, adjusted_entry_price,
                shares, stake_usd, predicted_probability_yes_at_entry,
                probability_for_direction_at_entry, estimated_edge_at_entry,
                time_until_resolution_at_entry, market_start_time, market_close_time,
                exit_time, exit_reason, exit_price, adjusted_exit_price,
                realized_pnl_usd, realized_roi
            ) VALUES (
                ?, ?, 'run_live', ?, ?, ?, ?, ?, 'MODEL_EDGE', ?, ?, ?, 0.4,
                0.4, 2.5, 1.0, ?, ?, 0.4, 120.0, ?, ?, ?, ?, 0.5,
                0.5, ?, ?
            )
            """,
            (
                to_iso(NOW),
                to_iso(NOW),
                strategy_id,
                f"Strategy {strategy_id}",
                f"{strategy_id}_{side}_{exit_type}",
                side,
                status,
                exit_type,
                to_iso(NOW),
                to_iso(NOW),
                1.0 - probability if side == "NO" else probability,
                probability,
                to_iso(NOW),
                to_iso(NOW),
                to_iso(NOW),
                exit_reason,
                pnl,
                roi,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_intramarket_analytics_groups_by_strategy_id_and_exit_type(tmp_path) -> None:
    db_path = _db(tmp_path)
    _insert_position(db_path, strategy_id="s_win", exit_type="FIXED_HORIZON_EXIT", pnl=0.25, roi=0.25)
    _insert_position(db_path, strategy_id="s_loss", exit_type="TAKE_PROFIT_STOP_LOSS", pnl=-0.2, roi=-0.2)

    report = build_intramarket_paper_analytics_report(
        paper_db_path=str(db_path),
        output_dir=str(tmp_path / "analytics"),
    )

    summary = report["summary"]
    assert summary["total_positions"] == 2
    assert summary["realized_closed_pnl"] == 0.05
    assert summary["pnl_by_strategy_id"]["s_win"]["pnl"] == 0.25
    assert summary["pnl_by_exit_type"]["TAKE_PROFIT_STOP_LOSS"]["pnl"] == -0.2
    assert (tmp_path / "analytics" / "intramarket_report.json").exists()
    assert (tmp_path / "analytics" / "intramarket_report.txt").exists()


def test_intramarket_analytics_probability_and_time_buckets(tmp_path) -> None:
    db_path = _db(tmp_path)
    _insert_position(db_path, strategy_id="s1", probability=0.75, pnl=0.2)
    _insert_position(db_path, strategy_id="s1", probability=0.85, pnl=0.3)

    report = build_intramarket_paper_analytics_report(paper_db_path=str(db_path))

    buckets = report["summary"]["pnl_by_probability_bucket"]
    assert buckets["0.7-0.8"]["count"] == 1
    assert buckets["0.8-0.9"]["count"] == 1
    assert report["summary"]["pnl_by_time_until_resolution_bucket"]["120-180s"]["count"] == 2


def test_intramarket_strategy_leaderboard(tmp_path) -> None:
    db_path = _db(tmp_path)
    _insert_position(db_path, strategy_id="s1", side="YES", exit_type="FIXED_HORIZON_EXIT", pnl=0.2, roi=0.2)
    _insert_position(db_path, strategy_id="s1", side="YES", exit_type="FIXED_HORIZON_EXIT", pnl=0.1, roi=0.1)
    _insert_position(db_path, strategy_id="s2", side="NO", exit_type="TAKE_PROFIT_STOP_LOSS", pnl=-0.2, roi=-0.2)

    report = build_intramarket_strategy_leaderboard(
        paper_db_path=str(db_path),
        min_closed=1,
    )
    rendered = render_intramarket_strategy_leaderboard(report)

    assert report["status"] == "ok"
    assert report["rows"][0]["strategy_id"] == "s1"
    assert report["rows"][0]["side"] == "YES"
    assert report["rows"][0]["exit_type"] == "FIXED_HORIZON_EXIT"
    assert "strategy_id side exit_type" in rendered


def test_intramarket_analytics_missing_table_is_clean_error(tmp_path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()

    report = build_intramarket_paper_analytics_report(paper_db_path=str(empty))

    assert report["status"] == "error"
    assert report["error"] == "paper_positions_table_missing"
