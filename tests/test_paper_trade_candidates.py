from __future__ import annotations

import sqlite3
from datetime import timedelta

from src.baseline_paper_trader import (
    connect_paper_output_db,
    ensure_paper_schema,
    run_baseline_paper_trader,
    settle_open_paper_trades,
)
from tests.test_baseline_paper_trader import (
    NOW,
    _config,
    _fixture,
    _insert_candidate,
    _iso,
)


def _candidate_rows(paper_db) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(paper_db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM paper_trade_candidates ORDER BY id").fetchall()
    finally:
        conn.close()


def _trade_rows(paper_db) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(paper_db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM paper_trades ORDER BY id").fetchall()
    finally:
        conn.close()


def _resolve_market(recorder_db, *, winning_outcome: str, close_time=None) -> None:
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute(
            """
            UPDATE markets
            SET resolved = 1,
                winning_outcome = ?,
                close_time = ?
            WHERE run_id = 'run_live' AND market_id = 'm1'
            """,
            (winning_outcome, _iso(close_time or (NOW - timedelta(seconds=1)))),
        )
        conn.commit()
    finally:
        conn.close()


def test_candidate_table_created_additively(tmp_path) -> None:
    paper_db = tmp_path / "paper.db"
    conn = connect_paper_output_db(str(paper_db))
    try:
        ensure_paper_schema(conn)
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "paper_trade_candidates" in names
    assert "idx_paper_trade_candidates_run_feature" in indexes
    assert "ux_paper_trade_candidates_dedupe" in indexes


def test_candidate_row_logged_for_taken_trade(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)

    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            strategy_id="s1",
            strategy_name="Strategy One",
        ),
        max_iterations=1,
        emit_logs=False,
    )
    trade = _trade_rows(paper_db)[0]
    candidate = _candidate_rows(paper_db)[0]

    assert candidate["decision"] == "TRADE"
    assert candidate["strategy_id"] == "s1"
    assert candidate["strategy_name"] == "Strategy One"
    assert candidate["candidate_direction"] == "YES"
    assert candidate["linked_paper_trade_id"] == trade["id"]
    assert candidate["estimated_edge"] == 0.3


def test_candidate_row_logged_for_skipped_trade(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.66, best_ask_yes=0.65)

    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            min_estimated_edge=0.05,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    candidate = _candidate_rows(paper_db)[0]

    assert candidate["decision"] == "SKIP"
    assert candidate["rejection_reason"] == "min_estimated_edge"
    assert candidate["linked_paper_trade_id"] is None
    assert candidate["candidate_direction"] == "YES"


def test_duplicate_candidate_rows_are_not_inserted(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.66, best_ask_yes=0.65)
    config = _config(
        recorder_db,
        paper_db,
        model_path,
        features_path,
        min_estimated_edge=0.05,
        strategy_id="dedupe",
    )

    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)
    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)

    assert len(_candidate_rows(paper_db)) == 1


def test_settling_paper_trade_updates_linked_candidate_result(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    _resolve_market(recorder_db, winning_outcome="YES")
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = connect_paper_output_db(str(paper_db))
    try:
        settled = settle_open_paper_trades(
            recorder,
            output,
            now=NOW + timedelta(minutes=10),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()
    candidate = _candidate_rows(paper_db)[0]

    assert settled == 1
    assert candidate["actual_resolved_label"] == "YES"
    assert candidate["would_have_payout_usd"] == 2.0
    assert candidate["would_have_pnl_usd"] == 1.0
    assert candidate["would_have_roi"] == 1.0


def test_skipped_candidate_gets_would_have_pnl_after_resolution(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.66, best_ask_yes=0.65)

    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            min_estimated_edge=0.05,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    _resolve_market(recorder_db, winning_outcome="NO")
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = connect_paper_output_db(str(paper_db))
    try:
        settle_open_paper_trades(
            recorder,
            output,
            now=NOW + timedelta(minutes=10),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()
    candidate = _candidate_rows(paper_db)[0]

    assert candidate["decision"] == "SKIP"
    assert candidate["actual_resolved_label"] == "NO"
    assert candidate["would_have_payout_usd"] == 0.0
    assert candidate["would_have_pnl_usd"] == -1.0
    assert candidate["would_have_roi"] == -1.0
