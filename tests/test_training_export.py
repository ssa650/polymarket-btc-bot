from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

from src.db import SQLiteStore
from src.models import (
    FeatureRecord,
    MarketMetadata,
    MarketSnapshotRecord,
    OrderBookLevelRecord,
)
from src.training_export import export_training_parquet


def _market(run_id: str, market_id: str) -> MarketMetadata:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event-{market_id}",
        question="BTC Up or Down - 00:00-00:05",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        platform_status="open",
        phase="active",
        tracking_state="selected_active",
        yes_token_id=f"{market_id}_yes",
        no_token_id=f"{market_id}_no",
        strict_validation_passed=1,
        run_id=run_id,
    )


def _snapshot(run_id: str, market_id: str, ts: datetime) -> MarketSnapshotRecord:
    return MarketSnapshotRecord(
        timestamp=ts,
        market_id=market_id,
        yes_price=0.5,
        no_price=0.5,
        best_bid_yes=0.49,
        best_ask_yes=0.51,
        best_bid_no=0.49,
        best_ask_no=0.51,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.5,
        mid_price_no=0.5,
        volume=100.0,
        liquidity=1000.0,
        last_trade_price=0.5,
        last_trade_size=1.0,
        last_trade_time=ts,
        has_orderbook=1,
        has_trade_data=1,
        run_id=run_id,
        strict_validation_passed=1,
        snapshot_quality_status="ok",
        is_partial_orderbook=0,
        missing_level_count=0,
        time_gap_from_prev_snapshot_sec=1.0,
        is_gap_affected=0,
        feature_ready=1,
        book_checksum="abc",
    )


def _feature(run_id: str, market_id: str, ts: datetime, ready: int) -> FeatureRecord:
    return FeatureRecord(
        timestamp=ts,
        market_id=market_id,
        price_change_1s=0.01,
        run_id=run_id,
        strict_validation_passed=1,
        feature_ready=ready,
        is_gap_affected=0,
        snapshot_quality_status="ok",
    )


def _read_export_rows(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _insert_btc_price(
    store: SQLiteStore,
    *,
    run_id: str,
    source: str,
    price: float,
    local_arrival_ns: int,
    local_arrival_iso: str,
    exchange_timestamp: str | None = None,
) -> None:
    store.conn.execute(
        """
        INSERT INTO btc_prices (
            run_id, source, price, exchange_timestamp,
            local_arrival_ns, local_arrival_iso, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            source,
            price,
            exchange_timestamp,
            local_arrival_ns,
            local_arrival_iso,
            json.dumps({"price": price}),
        ),
    )


def _insert_tick_size(
    store: SQLiteStore,
    *,
    run_id: str,
    timestamp: str,
    market_id: str,
    asset_id: str,
    new_tick_size: float,
) -> None:
    store.conn.execute(
        """
        INSERT INTO tick_size_changes (
            run_id, timestamp, market_id, asset_id,
            old_tick_size, new_tick_size, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            timestamp,
            market_id,
            asset_id,
            0.01,
            new_tick_size,
            json.dumps({"asset_id": asset_id, "new_tick_size": new_tick_size}),
        ),
    )


def _insert_order_book_levels(
    store: SQLiteStore,
    *,
    run_id: str,
    market_id: str,
    ts: datetime,
) -> None:
    store.insert_order_book_levels(
        [
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="YES",
                book_side="bid",
                level=1,
                price=0.48,
                size=11.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="YES",
                book_side="bid",
                level=2,
                price=0.49,
                size=12.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="YES",
                book_side="ask",
                level=1,
                price=0.53,
                size=13.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="YES",
                book_side="ask",
                level=2,
                price=0.52,
                size=14.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="NO",
                book_side="bid",
                level=1,
                price=0.47,
                size=21.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="NO",
                book_side="bid",
                level=2,
                price=0.46,
                size=22.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="NO",
                book_side="ask",
                level=1,
                price=0.54,
                size=23.0,
                run_id=run_id,
            ),
            OrderBookLevelRecord(
                timestamp=ts,
                market_id=market_id,
                outcome_side="NO",
                book_side="ask",
                level=2,
                price=0.55,
                size=24.0,
                run_id=run_id,
            ),
        ]
    )


def test_run_id_isolation_allows_same_market_id_across_runs(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        m_a = _market("run_a", "same_market")
        m_b = _market("run_b", "same_market")

        assert store.upsert_markets([m_a], run_id="run_a") == 1
        assert store.upsert_markets([m_b], run_id="run_b") == 1

        count = int(
            store.conn.execute(
                "SELECT COUNT(*) FROM markets WHERE market_id = 'same_market'"
            ).fetchone()[0]
            or 0
        )
        assert count == 2
    finally:
        store.close()


def test_export_training_parquet_includes_resolution_btc_and_tick_context(tmp_path) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        return

    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"

    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
        store.upsert_markets([_market("run_a", "m_a")], run_id="run_a")
        store.insert_market_snapshots([_snapshot("run_a", "m_a", ts)])
        store.insert_features([_feature("run_a", "m_a", ts, ready=1)])

        with store.conn:
            store.conn.execute(
                """
                UPDATE markets
                SET resolved = 1,
                    resolved_at = ?,
                    winning_asset_id = ?,
                    winning_outcome = ?
                WHERE run_id = ? AND market_id = ?
                """,
                (
                    "2026-01-01T00:05:01+00:00",
                    "m_a_yes",
                    "Up",
                    "run_a",
                    "m_a",
                ),
            )
            _insert_btc_price(
                store,
                run_id="run_a",
                source="polymarket_rtds_chainlink",
                price=50000.0,
                local_arrival_ns=1_767_225_660_000_000_000,
                local_arrival_iso="2026-01-01T00:01:50+00:00",
                exchange_timestamp="2026-01-01T00:01:49+00:00",
            )
            _insert_btc_price(
                store,
                run_id="run_a",
                source="polymarket_rtds_chainlink",
                price=51000.0,
                local_arrival_ns=1_767_225_670_000_000_000,
                local_arrival_iso="2026-01-01T00:02:10+00:00",
                exchange_timestamp="2026-01-01T00:02:09+00:00",
            )
            _insert_tick_size(
                store,
                run_id="run_a",
                timestamp="2026-01-01T00:01:55+00:00",
                market_id="m_a",
                asset_id="m_a_yes",
                new_tick_size=0.001,
            )
            _insert_tick_size(
                store,
                run_id="run_a",
                timestamp="2026-01-01T00:01:40+00:00",
                market_id="m_a",
                asset_id="m_a_no",
                new_tick_size=0.01,
            )

        report = export_training_parquet(
            db_path=str(db_path),
            output_path=str(out_path),
            run_ids=["run_a"],
        )
        rows = _read_export_rows(out_path)

        assert report["snapshots_kept"] == 1
        assert report["btc_coverage_pct"] == 100.0
        assert report["label_coverage_pct"] == 100.0
        row = rows[0]
        assert row["resolved"] == 1
        assert row["resolved_at"] == "2026-01-01T00:05:01+00:00"
        assert row["winning_asset_id"] == "m_a_yes"
        assert row["winning_outcome"] == "Up"
        assert row["label_available"] == 1
        assert row["yes_won"] == 1
        assert row["no_won"] == 0
        assert row["market_start_time"] == "2026-01-01T00:00:00+00:00"
        assert row["market_close_time"] == "2026-01-01T00:05:00+00:00"
        assert round(row["seconds_since_market_start"]) == 120
        assert round(row["seconds_until_close"]) == 180
        assert row["btc_price"] == 50000.0
        assert row["btc_exchange_timestamp"] == "2026-01-01T00:01:49+00:00"
        assert row["btc_source"] == "polymarket_rtds_chainlink"
        assert row["has_btc_price"] == 1
        assert round(row["btc_sample_age_sec"]) == 10
        assert row["latest_tick_size_yes"] == 0.001
        assert row["latest_tick_size_no"] == 0.01
        assert round(row["tick_size_age_sec"]) == 5
        assert row["has_full_orderbook"] == 1
        assert row["gap_affected"] == 0
        assert row["has_depth"] == 0
        assert row["yes_bids_json"] == "[]"
        assert row["yes_asks_json"] == "[]"
        assert row["no_bids_json"] == "[]"
        assert row["no_asks_json"] == "[]"
    finally:
        store.close()


def test_export_training_parquet_includes_sorted_depth_arrays(tmp_path) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        return

    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"

    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
        store.upsert_markets([_market("run_a", "m_depth")], run_id="run_a")
        store.insert_market_snapshots([_snapshot("run_a", "m_depth", ts)])
        store.insert_features([_feature("run_a", "m_depth", ts, ready=1)])
        _insert_order_book_levels(
            store,
            run_id="run_a",
            market_id="m_depth",
            ts=ts,
        )

        report = export_training_parquet(
            db_path=str(db_path),
            output_path=str(out_path),
            run_ids=["run_a"],
        )
        row = _read_export_rows(out_path)[0]

        assert report["depth_coverage_pct"] == 100.0
        assert row["has_depth"] == 1
        assert json.loads(row["yes_bids_json"]) == [
            {"price": 0.49, "size": 12.0},
            {"price": 0.48, "size": 11.0},
        ]
        assert json.loads(row["yes_asks_json"]) == [
            {"price": 0.52, "size": 14.0},
            {"price": 0.53, "size": 13.0},
        ]
        assert json.loads(row["no_bids_json"]) == [
            {"price": 0.47, "size": 21.0},
            {"price": 0.46, "size": 22.0},
        ]
        assert json.loads(row["no_asks_json"]) == [
            {"price": 0.54, "size": 23.0},
            {"price": 0.55, "size": 24.0},
        ]
    finally:
        store.close()


def test_export_training_parquet_uses_condition_hash_resolution_normalized_to_market(
    tmp_path,
) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        return

    condition_hash = "0x" + ("b" * 64)
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"

    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
        numeric = _market("run_a", "2076290")
        numeric.condition_id = condition_hash
        placeholder = _market("run_a", condition_hash)
        placeholder.condition_id = condition_hash
        placeholder.yes_token_id = None
        placeholder.no_token_id = None
        placeholder.question = None
        store.upsert_markets([numeric, placeholder], run_id="run_a")
        store.insert_market_snapshots([_snapshot("run_a", "2076290", ts)])
        store.insert_features([_feature("run_a", "2076290", ts, ready=1)])
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO market_events (
                    run_id, timestamp, market_id, event_type, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "run_a",
                    "2026-01-01T00:05:01+00:00",
                    condition_hash,
                    "market_resolved",
                    json.dumps(
                        {
                            "id": "2076290",
                            "market": condition_hash,
                            "winning_asset_id": "2076290_no",
                        }
                    ),
                ),
            )
        normalize_report = store.normalize_market_resolutions(dry_run=False, run_id="run_a")

        assert normalize_report["mapped_market_ids"] == ["2076290"]

        export_training_parquet(
            db_path=str(db_path),
            output_path=str(out_path),
            run_ids=["run_a"],
        )
        row = _read_export_rows(out_path)[0]
        assert row["market_id"] == "2076290"
        assert row["resolved"] == 1
        assert row["winning_asset_id"] == "2076290_no"
        assert row["winning_outcome"] == "NO"
        assert row["label_available"] == 1
        assert row["yes_won"] == 0
        assert row["no_won"] == 1
    finally:
        store.close()


def test_export_training_parquet_marks_missing_btc_and_unresolved_labels(
    tmp_path,
) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        return

    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"

    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
        store.upsert_markets([_market("run_a", "m_a")], run_id="run_a")
        store.insert_market_snapshots([_snapshot("run_a", "m_a", ts)])
        store.insert_features([_feature("run_a", "m_a", ts, ready=1)])

        export_training_parquet(
            db_path=str(db_path),
            output_path=str(out_path),
            run_ids=["run_a"],
        )
        row = _read_export_rows(out_path)[0]

        assert row["has_btc_price"] == 0
        assert row["btc_price"] is None
        assert row["btc_sample_age_sec"] is None
        assert row["resolved"] == 0
        assert row["label_available"] == 0
        assert row["yes_won"] is None
        assert row["no_won"] is None
    finally:
        store.close()


def test_export_training_parquet_filters_to_requested_run(tmp_path) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        return

    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"

    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        ts = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        store.upsert_markets([_market("run_a", "m_a")], run_id="run_a")
        store.upsert_markets([_market("run_b", "m_b")], run_id="run_b")

        store.insert_market_snapshots([_snapshot("run_a", "m_a", ts)])
        store.insert_market_snapshots([_snapshot("run_b", "m_b", ts)])

        store.insert_features([_feature("run_a", "m_a", ts, ready=1)])
        store.insert_features([_feature("run_b", "m_b", ts, ready=1)])

        report = export_training_parquet(
            db_path=str(db_path),
            output_path=str(out_path),
            run_ids=["run_a"],
            merge_runs=False,
        )
        assert report["run_ids"] == ["run_a"]
        assert report["snapshots_kept"] == 1
        assert out_path.exists()
        assert (tmp_path / "training.parquet.report.json").exists()
    finally:
        store.close()
