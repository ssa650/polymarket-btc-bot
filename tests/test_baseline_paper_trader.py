from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

from src.baseline_paper_trader import (
    BaselinePaperTraderConfig,
    build_paper_trader_summary,
    connect_paper_output_db,
    connect_recorder_read_only,
    ensure_paper_schema,
    fetch_latest_candidate_rows,
    run_baseline_paper_trader,
    settle_open_paper_trades,
)
from src.btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)


NOW = datetime.now(timezone.utc).replace(microsecond=0)
BASE = NOW - timedelta(seconds=120)


class ProbabilityBySignalModel:
    def predict_proba(self, matrix):
        return [[1.0 - float(row[0]), float(row[0])] for row in matrix]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _write_model(path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ProbabilityBySignalModel(), path)


def _write_features(path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(columns or ["signal"]), encoding="utf-8")


def _create_recorder_db(db_path) -> None:
    conn = sqlite3.connect(str(db_path))
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
                yes_token_id TEXT,
                no_token_id TEXT,
                resolved INTEGER,
                winning_asset_id TEXT,
                winning_outcome TEXT
            );
            CREATE TABLE market_snapshots (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                best_bid_yes REAL,
                best_ask_yes REAL,
                best_bid_no REAL,
                best_ask_no REAL,
                spread_yes REAL,
                spread_no REAL,
                mid_price_yes REAL,
                mid_price_no REAL,
                last_trade_price REAL,
                last_trade_size REAL,
                last_trade_time TEXT,
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


def _insert_candidate(
    db_path,
    *,
    market_id: str = "m1",
    signal: float = 0.8,
    best_ask_yes: float | None = 0.50,
    best_ask_no: float | None = 0.40,
    close_offset_sec: int = 300,
    start_time: datetime | None = BASE,
    feature_time: datetime | None = None,
    time_until_resolution: float | None = None,
    resolved: int = 0,
    winning_asset_id: str | None = None,
    winning_outcome: str | None = None,
) -> None:
    ts = feature_time or (BASE + timedelta(seconds=120))
    close = BASE + timedelta(seconds=close_offset_sec)
    if time_until_resolution is None:
        time_until_resolution = (close - ts).total_seconds()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO markets (
                run_id, market_id, question, start_time, close_time, phase,
                yes_token_id, no_token_id, resolved, winning_asset_id, winning_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                market_id,
                f"Bitcoin Up or Down - {market_id}",
                _iso(start_time) if start_time is not None else None,
                _iso(close),
                "active",
                f"{market_id}_yes",
                f"{market_id}_no",
                resolved,
                winning_asset_id,
                winning_outcome,
            ),
        )
        conn.execute(
            """
            INSERT INTO market_snapshots (
                run_id, timestamp, market_id, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, spread_yes, spread_no,
                mid_price_yes, mid_price_no, last_trade_price, last_trade_size,
                last_trade_time, has_orderbook, has_trade_data, strict_validation_passed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 1)
            """,
            (
                "run_live",
                _iso(ts),
                market_id,
                0.49,
                best_ask_yes,
                0.59,
                best_ask_no,
                0.02,
                0.02,
                0.50,
                0.50,
                0.51,
                5.0,
                _iso(ts),
            ),
        )
        conn.execute(
            """
            INSERT INTO features (
                run_id, timestamp, market_id, signal, feature_ready,
                strict_validation_passed, is_gap_affected,
                snapshot_quality_status, time_until_resolution
            ) VALUES (?, ?, ?, ?, 1, 1, 0, 'ok', ?)
            """,
            ("run_live", _iso(ts), market_id, signal, time_until_resolution),
        )
        for idx, (source, price) in enumerate(
            (
                (POLYMARKET_RTDS_CHAINLINK_SOURCE, 100.0),
                (POLYMARKET_RTDS_BINANCE_SOURCE, 100.5),
            ),
            start=1,
        ):
            btc_ts = ts - timedelta(seconds=1)
            conn.execute(
                """
                INSERT INTO btc_prices (
                    run_id, source, price, exchange_timestamp,
                    local_arrival_ns, local_arrival_iso
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_live",
                    source,
                    price,
                    _iso(btc_ts),
                    1_767_225_600_000_000_000 + idx,
                    _iso(btc_ts),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _fixture(tmp_path, *, feature_columns: list[str] | None = None):
    recorder_db = tmp_path / "recorder.db"
    paper_db = tmp_path / "paper_trades.db"
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "feature_columns.json"
    _create_recorder_db(recorder_db)
    _write_model(model_path)
    _write_features(features_path, feature_columns)
    return recorder_db, paper_db, model_path, features_path


def _config(recorder_db, paper_db, model_path, features_path, **overrides):
    values = {
        "recorder_db_path": str(recorder_db),
        "output_db_path": str(paper_db),
        "model_path": str(model_path),
        "feature_columns_path": str(features_path),
        "poll_sec": 0.0,
        "entry_slippage_cents": 0.0,
        "fee_cents": 0.0,
        "max_feature_age_sec": 100000.0,
    }
    values.update(overrides)
    return BaselinePaperTraderConfig(**values)


def _paper_rows(paper_db) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(paper_db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM paper_trades ORDER BY id").fetchall()
    finally:
        conn.close()


def _recorder_counts(recorder_db) -> dict[str, int]:
    conn = sqlite3.connect(str(recorder_db))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
            for table in ("markets", "market_snapshots", "features", "btc_prices")
        }
    finally:
        conn.close()


def _latest_heartbeat(paper_db) -> sqlite3.Row:
    conn = sqlite3.connect(str(paper_db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT *
            FROM paper_trader_heartbeats
            ORDER BY datetime(timestamp) DESC, timestamp DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()


def _update_market_resolution(
    db_path,
    *,
    market_id: str = "m1",
    resolved: int = 1,
    winning_outcome: str | None = None,
    winning_asset_id: str | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            UPDATE markets
            SET resolved = ?,
                winning_outcome = ?,
                winning_asset_id = ?
            WHERE run_id = 'run_live'
              AND market_id = ?
            """,
            (resolved, winning_outcome, winning_asset_id, market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _update_feature(
    db_path,
    *,
    market_id: str = "m1",
    **fields,
) -> None:
    assignments = ", ".join(f"{column} = ?" for column in fields)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            f"""
            UPDATE features
            SET {assignments}
            WHERE run_id = 'run_live'
              AND market_id = ?
            """,
            (*fields.values(), market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _update_snapshot(
    db_path,
    *,
    market_id: str = "m1",
    **fields,
) -> None:
    assignments = ", ".join(f"{column} = ?" for column in fields)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            f"""
            UPDATE market_snapshots
            SET {assignments}
            WHERE run_id = 'run_live'
              AND market_id = ?
            """,
            (*fields.values(), market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _settle_now(recorder_db, paper_db, now: datetime, *, emit_logs: bool = False) -> int:
    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(paper_db))
    try:
        return settle_open_paper_trades(
            recorder,
            output,
            now=now,
            emit_logs=emit_logs,
        )
    finally:
        recorder.close()
        output.close()


def test_connect_recorder_read_only_rejects_writes(tmp_path) -> None:
    recorder_db, _paper_db, _model_path, _features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db)

    conn = connect_recorder_read_only(str(recorder_db))
    try:
        assert int(conn.execute("PRAGMA query_only").fetchone()[0]) == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO markets (run_id, market_id) VALUES ('x', 'y')")
    finally:
        conn.close()


def test_paper_trader_creates_schema_without_real_order_path(tmp_path) -> None:
    _recorder_db, paper_db, _model_path, _features_path = _fixture(tmp_path)
    conn = connect_paper_output_db(str(paper_db))
    try:
        ensure_paper_schema(conn)
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert {"paper_trades", "paper_trader_heartbeats"}.issubset(names)


def test_paper_trader_skips_when_feature_column_missing(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(
        tmp_path,
        feature_columns=["missing_signal"],
    )
    _insert_candidate(recorder_db)

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    summary = build_paper_trader_summary(str(paper_db))

    assert report["last_poll"]["reason"] == "missing_feature_columns"
    assert summary["status_counts"] == {}
    assert summary["latest_heartbeat"]["loop_error"].startswith("missing_feature_columns")


def test_paper_trader_opens_yes_trade_above_threshold(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    before_counts = _recorder_counts(recorder_db)

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "open"
    assert row["signal_direction"] == "YES"
    assert row["entry_price"] == 0.50
    assert row["adjusted_entry_price"] == 0.50
    assert _recorder_counts(recorder_db) == before_counts


def test_paper_trader_stale_feature_is_skipped(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        feature_time=NOW - timedelta(seconds=30),
        close_offset_sec=300,
    )

    report = run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            max_feature_age_sec=5.0,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["prediction_rows"] == 0
    assert report["last_poll"]["stale_feature_skips"] == 1
    assert heartbeat["stale_feature_skips"] == 1
    assert _paper_rows(paper_db) == []


def test_paper_trader_heartbeat_eligibility_skip_counters_and_latest_rejection(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="not_ready", signal=0.8)
    _update_feature(recorder_db, market_id="not_ready", feature_ready=0)
    _insert_candidate(recorder_db, market_id="gap", signal=0.8)
    _update_feature(recorder_db, market_id="gap", is_gap_affected=1)
    _insert_candidate(recorder_db, market_id="strict", signal=0.8)
    _update_feature(recorder_db, market_id="strict", strict_validation_passed=0)
    _insert_candidate(recorder_db, market_id="quality", signal=0.8)
    _update_feature(recorder_db, market_id="quality", snapshot_quality_status="partial")
    _insert_candidate(recorder_db, market_id="orderbook", signal=0.8)
    _update_snapshot(recorder_db, market_id="orderbook", has_orderbook=0)
    _insert_candidate(recorder_db, market_id="trade", signal=0.8)
    _update_snapshot(recorder_db, market_id="trade", has_trade_data=0)

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["prediction_rows"] == 0
    assert heartbeat["not_feature_ready_skips"] == 1
    assert heartbeat["gap_affected_skips"] == 1
    assert heartbeat["strict_validation_failed_skips"] == 1
    assert heartbeat["snapshot_quality_not_ok_skips"] == 1
    assert heartbeat["missing_orderbook_skips"] == 1
    assert heartbeat["missing_trade_data_skips"] == 1
    assert heartbeat["latest_rejection_reason"] in {
        "not_feature_ready",
        "gap_affected",
        "strict_validation_failed",
        "snapshot_quality_not_ok",
        "missing_orderbook",
        "missing_trade_data",
    }
    assert heartbeat["latest_rejection_market_id"] is not None
    assert heartbeat["latest_rejection_feature_timestamp"] is not None


def test_paper_trader_stale_canonical_btc_skip_counter(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8)
    conn = sqlite3.connect(str(recorder_db))
    try:
        stale_ts = NOW - timedelta(seconds=30)
        conn.execute(
            "UPDATE btc_prices SET local_arrival_iso = ?, exchange_timestamp = ?",
            (_iso(stale_ts), _iso(stale_ts)),
        )
        conn.commit()
    finally:
        conn.close()

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path, max_btc_age_sec=3.0),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert heartbeat["canonical_btc_stale_skips"] == 1
    assert heartbeat["latest_rejection_reason"] == "stale_canonical_btc"
    assert heartbeat["latest_canonical_btc_age_sec"] is not None


def test_paper_trader_stale_binance_does_not_block_by_default(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    conn = sqlite3.connect(str(recorder_db))
    try:
        stale_ts = NOW - timedelta(seconds=30)
        conn.execute(
            "UPDATE btc_prices SET local_arrival_iso = ?, exchange_timestamp = ? WHERE source = ?",
            (_iso(stale_ts), _iso(stale_ts), POLYMARKET_RTDS_BINANCE_SOURCE),
        )
        conn.commit()
    finally:
        conn.close()

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path, max_btc_age_sec=3.0),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]
    heartbeat = _latest_heartbeat(paper_db)

    assert row["status"] == "open"
    assert heartbeat["comparison_btc_stale_warnings"] == 1
    assert heartbeat["comparison_btc_stale_skips"] == 0


def test_paper_trader_require_fresh_comparison_btc_restores_strict_behavior(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    conn = sqlite3.connect(str(recorder_db))
    try:
        stale_ts = NOW - timedelta(seconds=30)
        conn.execute(
            "UPDATE btc_prices SET local_arrival_iso = ?, exchange_timestamp = ? WHERE source = ?",
            (_iso(stale_ts), _iso(stale_ts), POLYMARKET_RTDS_BINANCE_SOURCE),
        )
        conn.commit()
    finally:
        conn.close()

    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            max_btc_age_sec=3.0,
            require_fresh_comparison_btc=True,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert _paper_rows(paper_db) == []
    assert heartbeat["comparison_btc_stale_warnings"] == 1
    assert heartbeat["comparison_btc_stale_skips"] == 1
    assert heartbeat["latest_rejection_reason"] == "stale_comparison_btc"


def test_paper_trader_closed_market_is_skipped(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        close_offset_sec=100,
        feature_time=NOW,
        time_until_resolution=60.0,
    )

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["prediction_rows"] == 0
    assert report["last_poll"]["market_closed_skips"] == 1
    assert heartbeat["market_closed_skips"] == 1
    assert _paper_rows(paper_db) == []


def test_paper_trader_expired_feature_is_skipped(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        feature_time=NOW,
        close_offset_sec=300,
        time_until_resolution=0.0,
    )

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["prediction_rows"] == 0
    assert report["last_poll"]["expired_feature_skips"] == 1
    assert heartbeat["expired_feature_skips"] == 1
    assert _paper_rows(paper_db) == []


def test_paper_trader_builds_export_feature_parity_fields(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(
        tmp_path,
        feature_columns=["signal", "btc_price_at_market_start", "seconds_after_start"],
    )
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    recorder = connect_recorder_read_only(str(recorder_db))
    try:
        candidate = fetch_latest_candidate_rows(
            recorder,
            max_btc_age_sec=3.0,
            min_time_until_resolution_sec=0.0,
            max_time_until_resolution_sec=300.0,
        )[0]
    finally:
        recorder.close()

    assert candidate["btc_price_at_market_start"] == 100.0
    assert candidate["seconds_after_start"] == 120.0

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]

    assert report["last_poll"]["prediction_rows"] == 1
    assert row["status"] == "open"
    assert row["signal_direction"] == "YES"


def test_paper_trader_prefers_latest_prior_chainlink_start_price(tmp_path) -> None:
    recorder_db, _paper_db, _model_path, _features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute(
            """
            INSERT INTO btc_prices (
                run_id, source, price, exchange_timestamp,
                local_arrival_ns, local_arrival_iso
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "run_live",
                POLYMARKET_RTDS_CHAINLINK_SOURCE,
                99.0,
                _iso(BASE - timedelta(seconds=1)),
                1_767_225_599_000_000_000,
                _iso(BASE - timedelta(seconds=1)),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    recorder = connect_recorder_read_only(str(recorder_db))
    try:
        candidate = fetch_latest_candidate_rows(
            recorder,
            max_btc_age_sec=3.0,
            min_time_until_resolution_sec=0.0,
            max_time_until_resolution_sec=300.0,
        )[0]
    finally:
        recorder.close()

    assert candidate["btc_price_at_market_start"] == 99.0


def test_paper_trader_missing_chainlink_start_price_skips_cleanly(tmp_path, capsys) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(
        tmp_path,
        feature_columns=["signal", "btc_price_at_market_start"],
    )
    _insert_candidate(recorder_db, signal=0.8, start_time=None)

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=True,
    )
    output = capsys.readouterr().out

    assert report["last_poll"]["prediction_rows"] == 0
    assert report["last_poll"]["skipped_trades"] == 1
    assert "missing_btc_price_at_market_start" in output
    assert _paper_rows(paper_db) == []


def test_paper_trader_missing_feature_columns_log_once_without_crashing(
    tmp_path,
    capsys,
) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(
        tmp_path,
        feature_columns=["definitely_missing_feature"],
    )
    _insert_candidate(recorder_db)

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=2,
        emit_logs=True,
    )
    output = capsys.readouterr().out

    assert report["status"] == "ok"
    assert output.count("missing_feature_columns") == 1
    assert _paper_rows(paper_db) == []


def test_paper_trader_opens_no_trade_below_threshold_using_best_ask_no_only(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.2, best_ask_no=0.30)

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]

    assert row["signal_direction"] == "NO"
    assert row["entry_price"] == 0.30


def test_paper_trades_store_probability_for_direction_and_estimated_edge(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50)

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]

    assert row["probability_for_direction"] == 0.8
    assert row["estimated_edge"] == 0.3


def test_paper_trader_min_estimated_edge_blocks_bad_trade(tmp_path) -> None:
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
    row = _paper_rows(paper_db)[0]
    heartbeat = _latest_heartbeat(paper_db)

    assert row["status"] == "skipped"
    assert row["skip_reason"] == "min_estimated_edge"
    assert row["probability_for_direction"] == 0.66
    assert row["estimated_edge"] == 0.01
    assert heartbeat["min_estimated_edge_skips"] == 1


def test_paper_trader_max_open_trades_blocks_new_trade(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="m1", signal=0.8)
    _insert_candidate(recorder_db, market_id="m2", signal=0.8)

    report = run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            max_open_trades=1,
            one_trade_per_market=False,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["opened_trades"] == 1
    assert heartbeat["max_open_trades_skips"] == 1
    assert len([row for row in _paper_rows(paper_db) if row["status"] == "open"]) == 1


def test_paper_trader_time_window_blocks_trade(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="too_soon",
        signal=0.8,
        feature_time=NOW,
        close_offset_sec=300,
        time_until_resolution=20.0,
    )
    _insert_candidate(
        recorder_db,
        market_id="too_far",
        signal=0.8,
        feature_time=NOW,
        close_offset_sec=300,
        time_until_resolution=250.0,
    )

    report = run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            min_time_until_resolution_sec=30.0,
            max_time_until_resolution_sec=120.0,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert report["last_poll"]["prediction_rows"] == 0
    assert heartbeat["time_window_skips"] == 2
    assert _paper_rows(paper_db) == []


def test_paper_trader_probability_block_options_work(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.95, best_ask_yes=0.50)

    run_baseline_paper_trader(
        _config(
            recorder_db,
            paper_db,
            model_path,
            features_path,
            block_probability_above=0.9,
        ),
        max_iterations=1,
        emit_logs=False,
    )
    heartbeat = _latest_heartbeat(paper_db)

    assert _paper_rows(paper_db) == []
    assert heartbeat["probability_block_skips"] == 1


def test_paper_trader_one_trade_per_market_prevents_duplicates(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8)
    config = _config(recorder_db, paper_db, model_path, features_path)

    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)
    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)

    rows = _paper_rows(paper_db)
    assert len(rows) == 1
    assert rows[0]["status"] == "open"


def test_paper_trader_skipped_diagnostics_do_not_block_first_real_trade(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8)
    output = connect_paper_output_db(str(paper_db))
    try:
        ensure_paper_schema(output)
        with output:
            output.execute(
                """
                INSERT INTO paper_trades (
                    created_at, run_id, market_id, signal_timestamp,
                    signal_direction, stake_usd, status, skip_reason
                ) VALUES (?, 'run_live', 'm1', ?, 'YES', 1.0, 'skipped', 'min_estimated_edge')
                """,
                (_iso(NOW), _iso(NOW)),
            )
    finally:
        output.close()

    report = run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    rows = _paper_rows(paper_db)

    assert report["last_poll"]["opened_trades"] == 1
    assert len([row for row in rows if row["status"] == "open"]) == 1
    assert len([row for row in rows if row["status"] == "skipped"]) == 1


def test_paper_trader_missing_entry_price_records_skip(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.2, best_ask_no=None)

    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "skipped"
    assert row["signal_direction"] == "NO"
    assert row["skip_reason"] == "missing_entry_price"


def test_paper_trader_settlement_computes_pnl(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        best_ask_yes=0.50,
        close_offset_sec=300,
        resolved=1,
        winning_outcome="YES",
    )
    config = _config(recorder_db, paper_db, model_path, features_path)

    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "settled"
    assert row["resolved_label"] == "YES"
    assert row["payout_usd"] == 2.0
    assert row["pnl_usd"] == 1.0
    assert row["roi"] == 1.0


def test_paper_trade_before_close_remains_open(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, close_offset_sec=300, resolved=0)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=200))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "open"


def test_paper_trade_after_close_unresolved_becomes_awaiting_resolution(
    tmp_path,
    capsys,
) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, close_offset_sec=300, resolved=0)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )

    _settle_now(
        recorder_db,
        paper_db,
        BASE + timedelta(seconds=301),
        emit_logs=True,
    )
    output = capsys.readouterr().out
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "awaiting_resolution"
    assert "paper_trade_awaiting_resolution" in output


def test_awaiting_resolution_trade_settles_yes_winner(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50, close_offset_sec=300)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    _update_market_resolution(recorder_db, winning_outcome="YES")

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=302))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "settled"
    assert row["resolved_label"] == "YES"
    assert row["payout_usd"] == 2.0
    assert row["pnl_usd"] == 1.0
    assert row["roi"] == 1.0


def test_awaiting_resolution_trade_settles_no_winner(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.2, best_ask_no=0.25, close_offset_sec=300)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    _update_market_resolution(recorder_db, winning_outcome="down")

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=302))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "settled"
    assert row["resolved_label"] == "NO"
    assert row["payout_usd"] == 4.0
    assert row["pnl_usd"] == 3.0
    assert row["roi"] == 3.0


def test_winning_asset_id_fallback_settles_correctly(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50, close_offset_sec=300)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    _update_market_resolution(recorder_db, winning_asset_id="m1_yes")

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=302))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "settled"
    assert row["resolved_label"] == "YES"
    assert row["payout_usd"] == 2.0


def test_losing_trade_settlement_payout_and_pnl(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        best_ask_yes=0.67,
        close_offset_sec=300,
        resolved=1,
        winning_outcome="NO",
    )
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "settled"
    assert row["resolved_label"] == "NO"
    assert row["payout_usd"] == 0.0
    assert row["pnl_usd"] == -1.0
    assert row["roi"] == -1.0


def test_unnormalizable_resolved_winner_remains_awaiting_and_logs(
    tmp_path,
    capsys,
) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        signal=0.8,
        close_offset_sec=300,
        resolved=1,
        winning_outcome="MAYBE",
    )
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301), emit_logs=True)
    output = capsys.readouterr().out
    row = _paper_rows(paper_db)[0]

    assert row["status"] == "awaiting_resolution"
    assert row["payout_usd"] is None
    assert "paper_trade_resolution_unavailable" in output


def test_paper_trader_summary_includes_awaiting_resolution_count(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8, close_offset_sec=300, resolved=0)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))

    summary = build_paper_trader_summary(str(paper_db))

    assert summary["open_trades"] == 0
    assert summary["awaiting_resolution_trades"] == 1
    assert summary["settled_trades"] == 0
    assert summary["status_counts"]["awaiting_resolution"] == 1
    assert summary["recent_trades"][0]["status"] == "awaiting_resolution"
    assert "older_than_10_minutes" in summary["awaiting_resolution_age_buckets"]
    assert summary["oldest_awaiting_resolution_trades"][0]["market_id"] == "m1"


def test_paper_trader_heartbeat_includes_awaiting_and_settled_counts(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    config = _config(recorder_db, paper_db, model_path, features_path)
    _insert_candidate(recorder_db, signal=0.8, best_ask_yes=0.50, close_offset_sec=300)
    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)

    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=301))
    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)
    heartbeat = _latest_heartbeat(paper_db)

    assert heartbeat["open_trades"] == 0
    assert heartbeat["awaiting_resolution_trades"] == 1
    assert heartbeat["settled_trades"] == 0
    assert heartbeat["total_trades"] == 1

    _update_market_resolution(recorder_db, winning_outcome="YES")
    _settle_now(recorder_db, paper_db, BASE + timedelta(seconds=302))
    run_baseline_paper_trader(config, max_iterations=1, emit_logs=False)
    heartbeat = _latest_heartbeat(paper_db)

    assert heartbeat["open_trades"] == 0
    assert heartbeat["awaiting_resolution_trades"] == 0
    assert heartbeat["settled_trades"] == 1
    assert heartbeat["total_trades"] == 1


def test_paper_trader_summary_command(tmp_path, monkeypatch, capsys) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8)
    run_baseline_paper_trader(
        _config(recorder_db, paper_db, model_path, features_path),
        max_iterations=1,
        emit_logs=False,
    )

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("paper-trader-summary must not start recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "paper-trader-summary", "--paper-db", str(paper_db)],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["open_trades"] == 1
    assert payload["settled_trades"] == 0
    assert payload["recent_trades"][0]["signal_direction"] == "YES"


def test_run_baseline_paper_trader_command_is_offline(tmp_path, monkeypatch, capsys) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, signal=0.8)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run-baseline-paper-trader must not start recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-baseline-paper-trader",
            "--db",
            str(recorder_db),
            "--output-db",
            str(paper_db),
            "--model-path",
            str(model_path),
            "--feature-columns",
            str(features_path),
            "--paper-max-iterations",
            "1",
            "--max-feature-age-sec",
            "100000",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)

    main_module.main()
    output = capsys.readouterr().out

    assert "paper_trader_started" in output
    assert '"status": "ok"' in output
    assert _paper_rows(paper_db)[0]["signal_direction"] == "YES"
