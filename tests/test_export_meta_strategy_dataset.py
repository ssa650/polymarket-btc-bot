from __future__ import annotations

import csv
import sqlite3
from datetime import timedelta

from src.baseline_paper_trader import (
    connect_paper_output_db,
    run_baseline_paper_trader,
    settle_open_paper_trades,
)
from src.export_meta_strategy_dataset import export_meta_strategy_dataset
from tests.test_baseline_paper_trader import (
    NOW,
    _config,
    _fixture,
    _insert_candidate,
    _iso,
)


def _read_parquet(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _resolve_market(recorder_db, *, winning_outcome: str = "YES") -> None:
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
            (winning_outcome, _iso(NOW - timedelta(seconds=1))),
        )
        conn.commit()
    finally:
        conn.close()


def _settle(recorder_db, paper_db) -> None:
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


def test_export_meta_strategy_dataset_creates_parquet_and_csv(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    out_path = tmp_path / "meta.parquet"
    csv_path = tmp_path / "meta.csv"
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path, strategy_id="s1"),
        max_iterations=1,
        emit_logs=False,
    )
    _resolve_market(recorder_db, winning_outcome="YES")
    _settle(recorder_db, paper_db)

    report = export_meta_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(out_path),
        output_csv_path=str(csv_path),
    )
    rows = _read_parquet(out_path)

    assert report["status"] == "ok"
    assert report["rows_exported"] == 1
    assert out_path.exists()
    assert csv_path.exists()
    assert rows[0]["decision"] == "TRADE"
    assert rows[0]["traded"] == 1
    assert rows[0]["profitable"] == 1
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["strategy_id"] == "s1"


def test_export_meta_strategy_dataset_filters_and_include_unresolved(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    out_path = tmp_path / "meta.parquet"
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path, strategy_id="s1"),
        max_iterations=1,
        emit_logs=False,
    )

    unresolved = export_meta_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(out_path),
    )
    included = export_meta_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(out_path),
        include_unresolved=True,
        strategy_id="s1",
        min_created_at=_iso(NOW - timedelta(days=1)),
        max_created_at=_iso(NOW + timedelta(days=1)),
    )

    assert unresolved["rows_exported"] == 0
    assert included["rows_exported"] == 1
    assert _read_parquet(out_path)[0]["traded"] == 1


def test_export_meta_strategy_dataset_profitable_and_traded_labels_for_skips(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    out_path = tmp_path / "meta.parquet"
    _insert_candidate(recorder_db, signal=0.66, best_ask_yes=0.65)
    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            min_estimated_edge=0.05,
            strategy_id="skip_strategy",
        ),
        max_iterations=1,
        emit_logs=False,
    )
    _resolve_market(recorder_db, winning_outcome="NO")
    _settle(recorder_db, paper_db)

    export_meta_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(out_path),
    )
    row = _read_parquet(out_path)[0]

    assert row["decision"] == "SKIP"
    assert row["rejection_reason"] == "min_estimated_edge"
    assert row["traded"] == 0
    assert row["profitable"] == 0
    assert row["would_have_roi"] == -1.0


def test_export_meta_strategy_dataset_missing_table_is_clean_error(tmp_path) -> None:
    paper_db = tmp_path / "paper.db"
    sqlite3.connect(str(paper_db)).close()

    report = export_meta_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(tmp_path / "meta.parquet"),
    )

    assert report == {
        "status": "error",
        "error": "paper_trade_candidates_table_missing",
        "paper_db_path": str(paper_db),
    }
