from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.export_paper_trader_dataset import export_paper_trader_dataset


NOW = datetime(2026, 4, 29, 8, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _read_parquet(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _create_paper_db(path, *, trades: bool = True, heartbeat: bool = True) -> None:
    conn = sqlite3.connect(str(path))
    try:
        if trades:
            conn.execute(
                """
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    run_id TEXT,
                    market_id TEXT,
                    signal_timestamp TEXT,
                    question TEXT,
                    market_start_time TEXT,
                    market_close_time TEXT,
                    time_until_resolution REAL,
                    model_path TEXT,
                    model_name TEXT,
                    predicted_probability_yes REAL,
                    probability_for_direction REAL,
                    estimated_edge REAL,
                    signal_direction TEXT,
                    threshold_used REAL,
                    stake_usd REAL,
                    entry_price REAL,
                    adjusted_entry_price REAL,
                    best_bid_yes REAL,
                    best_ask_yes REAL,
                    best_bid_no REAL,
                    best_ask_no REAL,
                    btc_chainlink_price REAL,
                    btc_binance_price REAL,
                    btc_chainlink_age_sec_at_feature REAL,
                    btc_binance_age_sec_at_feature REAL,
                    feature_ready INTEGER,
                    strict_validation_passed INTEGER,
                    snapshot_quality_status TEXT,
                    status TEXT,
                    skip_reason TEXT,
                    resolved_label TEXT,
                    payout_usd REAL,
                    pnl_usd REAL,
                    roi REAL,
                    settled_at TEXT
                )
                """
            )
        if heartbeat:
            conn.execute(
                """
                CREATE TABLE paper_trader_heartbeats (
                    timestamp TEXT PRIMARY KEY,
                    latest_seen_feature_timestamp TEXT,
                    latest_eligible_feature_timestamp TEXT,
                    open_trades INTEGER,
                    awaiting_resolution_trades INTEGER,
                    settled_trades INTEGER,
                    total_trades INTEGER,
                    loop_error TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO paper_trader_heartbeats (
                    timestamp, latest_seen_feature_timestamp,
                    latest_eligible_feature_timestamp, open_trades,
                    awaiting_resolution_trades, settled_trades, total_trades,
                    loop_error
                ) VALUES (?, ?, ?, 1, 1, 1, 3, NULL)
                """,
                (_iso(NOW), _iso(NOW - timedelta(seconds=2)), _iso(NOW - timedelta(seconds=2))),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_trade(
    path,
    *,
    market_id: str,
    status: str = "settled",
    direction: str = "YES",
    probability_yes: float = 0.8,
    adjusted_entry_price: float = 0.55,
    pnl_usd: float | None = 0.8,
    roi: float | None = 0.8,
    skip_reason: str | None = None,
) -> None:
    created = NOW - timedelta(minutes=5)
    close = NOW - timedelta(minutes=1)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, predicted_probability_yes,
                probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd,
                entry_price, adjusted_entry_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                btc_chainlink_price, btc_binance_price,
                btc_chainlink_age_sec_at_feature, btc_binance_age_sec_at_feature,
                feature_ready, strict_validation_passed, snapshot_quality_status,
                status, skip_reason, resolved_label, payout_usd, pnl_usd, roi, settled_at
            ) VALUES (
                ?, 'run1', ?, ?, 'BTC Up or Down',
                ?, ?, 45.0,
                'model.joblib', 'RandomForestClassifier', ?,
                NULL, NULL,
                ?, 0.65, 1.0,
                ?, ?, 0.50, 0.55, 0.44, 0.45,
                100.0, 100.2, 1.0, 1.0,
                1, 1, 'ok',
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                _iso(created),
                market_id,
                _iso(created + timedelta(seconds=1)),
                _iso(created - timedelta(minutes=1)),
                _iso(close),
                probability_yes,
                direction,
                adjusted_entry_price,
                adjusted_entry_price,
                status,
                skip_reason,
                direction if status == "settled" and pnl_usd and pnl_usd > 0 else None,
                (1.0 + (pnl_usd or 0.0)) if status == "settled" and pnl_usd is not None else None,
                pnl_usd,
                roi,
                _iso(NOW - timedelta(seconds=10)) if status == "settled" else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_paper_db(path) -> None:
    _create_paper_db(path)
    _insert_trade(path, market_id="settled_yes", status="settled", pnl_usd=0.8)
    _insert_trade(path, market_id="awaiting_no", status="awaiting_resolution", direction="NO", pnl_usd=None, roi=None)
    _insert_trade(
        path,
        market_id="skipped_yes",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="min_estimated_edge",
    )


def test_export_creates_expected_strategy_files_and_metadata(tmp_path) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "dataset"
    _seed_paper_db(db)

    report = export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="guarded_v1",
        strategy_name="Guarded V1",
        output_dir=str(output),
        notes="test notes",
        now=NOW,
    )

    strategy_dir = output / "strategies" / "guarded_v1"
    assert report["status"] == "ok"
    assert (strategy_dir / "trades.parquet").exists()
    assert (strategy_dir / "trades.csv").exists()
    assert (strategy_dir / "summary.json").exists()
    assert (strategy_dir / "bucket_metrics.parquet").exists()
    assert (strategy_dir / "llm_strategy_report.md").exists()

    trades = _read_parquet(strategy_dir / "trades.parquet")
    assert {row["strategy_id"] for row in trades} == {"guarded_v1"}
    assert {row["strategy_name"] for row in trades} == {"Guarded V1"}
    assert {row["notes"] for row in trades} == {"test notes"}
    assert {row["status"] for row in trades} == {"settled"}

    summary = json.loads((strategy_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary_row"]["strategy_id"] == "guarded_v1"
    assert summary["summary_row"]["skipped_trades"] == 1
    assert "Recommendation" in (strategy_dir / "llm_strategy_report.md").read_text(encoding="utf-8")


def test_append_mode_creates_master_parquet_files(tmp_path) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "dataset"
    _seed_paper_db(db)

    export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="s1",
        strategy_name="Strategy 1",
        output_dir=str(output),
        append=True,
        now=NOW,
    )
    export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="s2",
        strategy_name="Strategy 2",
        output_dir=str(output),
        append=True,
        now=NOW,
    )

    trades = _read_parquet(output / "all_trades.parquet")
    summaries = _read_parquet(output / "all_strategy_summaries.parquet")
    buckets = _read_parquet(output / "all_bucket_metrics.parquet")
    assert {row["strategy_id"] for row in trades} == {"s1", "s2"}
    assert {row["strategy_id"] for row in summaries} == {"s1", "s2"}
    assert {row["strategy_id"] for row in buckets} == {"s1", "s2"}


def test_skipped_and_awaiting_trades_can_be_included_or_excluded(tmp_path) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "dataset"
    _seed_paper_db(db)

    export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="default",
        strategy_name="Default",
        output_dir=str(output),
        now=NOW,
    )
    default_statuses = {
        row["status"]
        for row in _read_parquet(output / "strategies" / "default" / "trades.parquet")
    }

    export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="all",
        strategy_name="All Rows",
        output_dir=str(output),
        include_skipped=True,
        include_awaiting=True,
        now=NOW,
    )
    all_statuses = {
        row["status"]
        for row in _read_parquet(output / "strategies" / "all" / "trades.parquet")
    }

    assert default_statuses == {"settled"}
    assert all_statuses == {"settled", "awaiting_resolution", "skipped"}


def test_bucket_metrics_are_exported(tmp_path) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "dataset"
    _seed_paper_db(db)

    report = export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="bucketed",
        strategy_name="Bucketed",
        output_dir=str(output),
        now=NOW,
    )
    buckets = _read_parquet(output / "strategies" / "bucketed" / "bucket_metrics.parquet")

    assert report["bucket_rows_exported"] > 0
    assert {"probability", "edge", "time_until_resolution"}.issubset(
        {row["bucket_type"] for row in buckets}
    )
    assert all(row["strategy_id"] == "bucketed" for row in buckets)


def test_missing_paper_trades_table_returns_clean_error(tmp_path) -> None:
    db = tmp_path / "paper_missing.db"
    output = tmp_path / "dataset"
    _create_paper_db(db, trades=False)

    report = export_paper_trader_dataset(
        paper_db_path=str(db),
        strategy_id="missing",
        strategy_name="Missing",
        output_dir=str(output),
        now=NOW,
    )

    assert report == {
        "status": "error",
        "error": "paper_trades_table_missing",
        "paper_db_path": str(db),
        "strategy_id": "missing",
        "strategy_name": "Missing",
    }
