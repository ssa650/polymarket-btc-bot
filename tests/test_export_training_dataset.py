from __future__ import annotations

import csv
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from src.db import SQLiteStore
from src.models import FeatureRecord, MarketMetadata, MarketSnapshotRecord, to_iso
from src.training_dataset_export import (
    EXPORT_COLUMNS,
    _connect_read_only,
    export_training_dataset,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _market(run_id: str, market_id: str, *, question: str | None = None) -> MarketMetadata:
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event_{run_id}_{market_id}",
        question=question or f"Bitcoin Up or Down - {run_id}/{market_id}",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=BASE,
        end_time=BASE + timedelta(minutes=5),
        close_time=BASE + timedelta(minutes=5),
        platform_status="resolved",
        phase="resolved",
        tracking_state="inactive",
        status="resolved",
        market_phase="resolved",
        yes_token_id=f"{market_id}_yes",
        no_token_id=f"{market_id}_no",
        condition_id=f"condition_{run_id}_{market_id}",
        run_id=run_id,
        strict_validation_passed=1,
    )


def _snapshot(
    run_id: str,
    market_id: str,
    ts: datetime,
    *,
    has_orderbook: int = 1,
    has_trade_data: int = 1,
) -> MarketSnapshotRecord:
    return MarketSnapshotRecord(
        timestamp=ts,
        market_id=market_id,
        yes_price=0.52,
        no_price=0.48,
        best_bid_yes=0.51,
        best_ask_yes=0.53,
        best_bid_no=0.47,
        best_ask_no=0.49,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.52,
        mid_price_no=0.48,
        volume=100.0,
        liquidity=200.0,
        last_trade_price=0.52 if has_trade_data else None,
        last_trade_size=4.0 if has_trade_data else None,
        last_trade_time=ts if has_trade_data else None,
        has_orderbook=has_orderbook,
        has_trade_data=has_trade_data,
        run_id=run_id,
        strict_validation_passed=1,
        snapshot_quality_status="ok",
        is_partial_orderbook=0,
        missing_level_count=0,
        time_gap_from_prev_snapshot_sec=1.0,
        is_gap_affected=0,
        feature_ready=1,
    )


def _feature(
    run_id: str,
    market_id: str,
    ts: datetime,
    *,
    ready: int = 1,
    gap: int = 0,
    strict: int = 1,
    quality: str = "ok",
) -> FeatureRecord:
    return FeatureRecord(
        timestamp=ts,
        market_id=market_id,
        price_change_1s=0.01,
        price_change_10s=0.02,
        price_change_60s=0.03,
        velocity_5s=0.04,
        velocity_30s=0.05,
        acceleration_5s=0.06,
        rolling_mean_60s=0.52,
        distance_from_rolling_mean_60s=0.01,
        total_bid_liquidity_yes=10.0,
        total_ask_liquidity_yes=12.0,
        liquidity_change_bid_5s=1.0,
        orderbook_imbalance_yes=0.1,
        buy_volume_5s=3.0,
        sell_volume_5s=2.0,
        net_trade_flow_5s=1.0,
        trade_flow_ratio_5s=0.6,
        rolling_volatility_10s=0.001,
        rolling_volatility_60s=0.002,
        volume_delta_1s=1.0,
        avg_volume_60s=5.0,
        volume_spike_ratio=1.2,
        largest_bid_wall_size_yes=20.0,
        largest_ask_wall_size_yes=22.0,
        distance_to_bid_wall=0.01,
        time_since_market_created=(ts - BASE).total_seconds(),
        time_until_resolution=((BASE + timedelta(minutes=5)) - ts).total_seconds(),
        is_price_jump=0,
        run_id=run_id,
        strict_validation_passed=strict,
        feature_ready=ready,
        is_gap_affected=gap,
        snapshot_quality_status=quality,
    )


def _insert_btc(
    store: SQLiteStore,
    *,
    run_id: str,
    source: str,
    ts: datetime,
    price: float,
    idx: int,
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
            to_iso(ts - timedelta(milliseconds=100)),
            1_767_225_600_000_000_000 + idx,
            to_iso(ts),
            json.dumps({"source": source, "price": price}),
        ),
    )


def _insert_trade(
    store: SQLiteStore,
    *,
    run_id: str,
    market_id: str,
    ts: datetime,
    side: str = "BUY",
) -> None:
    store.conn.execute(
        """
        INSERT INTO trades (
            run_id, timestamp, market_id, trade_id, price, size, side
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, to_iso(ts), market_id, f"trade_{run_id}_{market_id}", 0.52, 4.0, side),
    )


def _read_parquet_rows(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _table_counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
            for table in (
                "markets",
                "market_snapshots",
                "features",
                "trades",
                "btc_prices",
            )
        }
    finally:
        conn.close()


def _seed_filter_db(db_path) -> None:
    run_id = "run_filter"
    market_id = "m_filter"
    store = SQLiteStore(str(db_path), run_id=run_id)
    try:
        store.init_schema()
        store.upsert_markets([_market(run_id, market_id)], run_id=run_id)
        timestamps = [BASE + timedelta(minutes=2, seconds=i) for i in range(7)]
        store.insert_market_snapshots(
            [
                _snapshot(run_id, market_id, timestamps[0]),
                _snapshot(run_id, market_id, timestamps[1]),
                _snapshot(run_id, market_id, timestamps[2]),
                _snapshot(run_id, market_id, timestamps[3]),
                _snapshot(run_id, market_id, timestamps[4]),
                _snapshot(run_id, market_id, timestamps[5], has_orderbook=0),
                _snapshot(run_id, market_id, timestamps[6], has_trade_data=0),
            ]
        )
        store.insert_features(
            [
                _feature(run_id, market_id, timestamps[0]),
                _feature(run_id, market_id, timestamps[1], ready=0),
                _feature(run_id, market_id, timestamps[2], gap=1),
                _feature(run_id, market_id, timestamps[3], strict=0),
                _feature(run_id, market_id, timestamps[4], quality="partial_orderbook"),
                _feature(run_id, market_id, timestamps[5]),
                _feature(run_id, market_id, timestamps[6]),
            ]
        )
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
                    to_iso(BASE + timedelta(minutes=5, seconds=1)),
                    f"{market_id}_yes",
                    "Up",
                    run_id,
                    market_id,
                ),
            )
            _insert_trade(store, run_id=run_id, market_id=market_id, ts=timestamps[0])
            _insert_btc(
                store,
                run_id=run_id,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                ts=BASE,
                price=100.0,
                idx=1,
            )
            _insert_btc(
                store,
                run_id=run_id,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                ts=timestamps[0] - timedelta(seconds=10),
                price=110.0,
                idx=2,
            )
            _insert_btc(
                store,
                run_id=run_id,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                ts=BASE + timedelta(minutes=5),
                price=120.0,
                idx=3,
            )
            _insert_btc(
                store,
                run_id=run_id,
                source=POLYMARKET_RTDS_BINANCE_SOURCE,
                ts=timestamps[0] - timedelta(seconds=5),
                price=111.0,
                idx=4,
            )
    finally:
        store.close()


def test_export_training_dataset_default_filters_and_labels(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "exports" / "training.parquet"
    _seed_filter_db(db_path)

    report = export_training_dataset(
        db_path=str(db_path),
        output_path=str(out_path),
    )
    rows = _read_parquet_rows(out_path)

    assert report["exported_rows"] == 1
    assert report["usable_rows"] == 1
    assert report["skipped_rows_by_reason"]["unusable_rows"] == 6
    assert report["skipped_rows_by_reason"]["feature_not_ready"] == 1
    assert report["skipped_rows_by_reason"]["gap_affected"] == 1
    assert report["skipped_rows_by_reason"]["feature_strict_validation_failed"] == 1
    assert report["skipped_rows_by_reason"]["snapshot_quality_not_ok"] == 1
    assert report["skipped_rows_by_reason"]["missing_orderbook"] == 1
    assert report["skipped_rows_by_reason"]["missing_trade_data"] == 1

    row = rows[0]
    assert row["export_row_usable"] == 1
    assert row["feature_ready"] == 1
    assert row["is_gap_affected"] == 0
    assert row["snapshot_quality_status"] == "ok"
    assert row["strict_validation_passed"] == 1
    assert row["last_trade_side"] == "BUY"
    assert row["btc_chainlink_price"] == 110.0
    assert round(row["btc_chainlink_age_sec_at_feature"]) == 10
    assert row["btc_binance_price"] == 111.0
    assert round(row["btc_binance_age_sec_at_feature"]) == 5
    assert row["btc_price_diff_binance_minus_chainlink"] == 1.0
    assert row["label_resolved_up_down"] == "UP"
    assert row["label_yes_win"] == 1
    assert row["btc_price_at_market_start"] == 100.0
    assert row["btc_price_at_resolution"] == 120.0
    assert row["label_btc_up_at_resolution"] == 1
    assert row["label_btc_return_to_resolution"] == 0.2
    assert round(row["future_btc_return_to_resolution_from_feature"], 6) == round(
        (120.0 - 110.0) / 110.0,
        6,
    )
    assert round(row["seconds_after_start"]) == 120
    assert round(row["seconds_before_close"]) == 180
    assert out_path.exists()
    assert report["output_file_size_bytes"] > 0


def test_export_training_dataset_include_not_ready_marks_usability_and_writes_csv(
    tmp_path,
) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    csv_path = tmp_path / "training.csv"
    _seed_filter_db(db_path)

    report = export_training_dataset(
        db_path=str(db_path),
        output_path=str(out_path),
        output_csv_path=str(csv_path),
        include_not_ready=True,
    )
    rows = _read_parquet_rows(out_path)

    assert report["exported_rows"] == 7
    assert report["usable_rows"] == 1
    assert sorted(row["export_row_usable"] for row in rows) == [0, 0, 0, 0, 0, 0, 1]
    assert csv_path.exists()
    assert report["output_csv_file_size_bytes"] and report["output_csv_file_size_bytes"] > 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 7
    assert "btc_chainlink_price" in csv_rows[0]


def test_export_training_dataset_limit_and_output_schema(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    _seed_filter_db(db_path)

    report = export_training_dataset(
        db_path=str(db_path),
        output_path=str(out_path),
        include_not_ready=True,
        limit=3,
        chunk_size=1,
        progress_every_rows=0,
        progress_every_sec=0,
    )
    rows = _read_parquet_rows(out_path)

    assert report["exported_rows"] == 3
    assert report["limit"] == 3
    assert report["chunk_size"] == 1
    assert len(rows) == 3
    assert list(rows[0].keys()) == list(EXPORT_COLUMNS)


def test_export_training_dataset_chunked_export_matches_large_chunk_export(
    tmp_path,
) -> None:
    db_path = tmp_path / "recorder.db"
    chunked_path = tmp_path / "chunked.parquet"
    regular_path = tmp_path / "regular.parquet"
    _seed_filter_db(db_path)

    chunked = export_training_dataset(
        db_path=str(db_path),
        output_path=str(chunked_path),
        include_not_ready=True,
        chunk_size=1,
        progress_every_rows=0,
        progress_every_sec=0,
    )
    regular = export_training_dataset(
        db_path=str(db_path),
        output_path=str(regular_path),
        include_not_ready=True,
        chunk_size=100,
        progress_every_rows=0,
        progress_every_sec=0,
    )

    assert _read_parquet_rows(chunked_path) == _read_parquet_rows(regular_path)
    assert chunked["exported_rows"] == regular["exported_rows"] == 7
    assert chunked["usable_rows"] == regular["usable_rows"] == 1


def test_export_training_dataset_progress_logging(caplog, tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    _seed_filter_db(db_path)

    with caplog.at_level(logging.INFO, logger="src.training_dataset_export"):
        export_training_dataset(
            db_path=str(db_path),
            output_path=str(out_path),
            include_not_ready=True,
            chunk_size=2,
            progress_every_rows=2,
            progress_every_sec=0,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("export_training_dataset_started" in message for message in messages)
    assert any("export_training_dataset_progress" in message for message in messages)
    assert any("export_training_dataset_completed" in message for message in messages)


def test_export_training_dataset_run_id_market_id_join_isolated(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    market_id = "same_market"
    ts = BASE + timedelta(minutes=2)
    store = SQLiteStore(str(db_path), run_id="run_a")
    try:
        store.init_schema()
        store.upsert_markets(
            [_market("run_a", market_id, question="run a market")],
            run_id="run_a",
        )
        store.upsert_markets(
            [_market("run_b", market_id, question="run b market")],
            run_id="run_b",
        )
        store.insert_market_snapshots(
            [_snapshot("run_a", market_id, ts), _snapshot("run_b", market_id, ts)]
        )
        store.insert_features(
            [_feature("run_a", market_id, ts), _feature("run_b", market_id, ts)]
        )
        with store.conn:
            for idx, (run_id, start_price, feature_price, close_price) in enumerate(
                (("run_a", 100.0, 110.0, 120.0), ("run_b", 200.0, 210.0, 220.0)),
                start=10,
            ):
                store.conn.execute(
                    """
                    UPDATE markets
                    SET resolved = 1,
                        winning_asset_id = ?,
                        winning_outcome = ?
                    WHERE run_id = ? AND market_id = ?
                    """,
                    (f"{market_id}_no", "Down", run_id, market_id),
                )
                _insert_btc(
                    store,
                    run_id=run_id,
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                    ts=BASE,
                    price=start_price,
                    idx=idx,
                )
                _insert_btc(
                    store,
                    run_id=run_id,
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                    ts=ts,
                    price=feature_price,
                    idx=idx + 10,
                )
                _insert_btc(
                    store,
                    run_id=run_id,
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                    ts=BASE + timedelta(minutes=5),
                    price=close_price,
                    idx=idx + 20,
                )
    finally:
        store.close()

    report = export_training_dataset(db_path=str(db_path), output_path=str(out_path))
    rows = sorted(_read_parquet_rows(out_path), key=lambda row: row["run_id"])

    assert report["exported_rows"] == 2
    assert report["unique_markets"] == 2
    assert [row["question"] for row in rows] == ["run a market", "run b market"]
    assert [row["btc_chainlink_price"] for row in rows] == [110.0, 210.0]
    assert [row["label_yes_win"] for row in rows] == [0, 0]


def test_export_training_dataset_recent_minutes_filter(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    run_id = "run_recent"
    market_id = "m_recent"
    now = BASE + timedelta(hours=2)
    recent_ts = now - timedelta(seconds=30)
    old_ts = now - timedelta(minutes=10)
    store = SQLiteStore(str(db_path), run_id=run_id)
    try:
        store.init_schema()
        market = _market(run_id, market_id)
        market.start_time = now - timedelta(minutes=4)
        market.close_time = now + timedelta(minutes=1)
        market.end_time = market.close_time
        store.upsert_markets([market], run_id=run_id)
        store.insert_market_snapshots(
            [_snapshot(run_id, market_id, old_ts), _snapshot(run_id, market_id, recent_ts)]
        )
        store.insert_features(
            [_feature(run_id, market_id, old_ts), _feature(run_id, market_id, recent_ts)]
        )
    finally:
        store.close()

    report = export_training_dataset(
        db_path=str(db_path),
        output_path=str(out_path),
        recent_minutes=1,
        now=now,
    )
    rows = _read_parquet_rows(out_path)

    assert report["exported_rows"] == 1
    assert rows[0]["timestamp"] == to_iso(recent_ts)
    assert report["recent_minutes"] == 1


def test_export_training_dataset_time_until_resolution_filter(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    _seed_filter_db(db_path)

    report = export_training_dataset(
        db_path=str(db_path),
        output_path=str(out_path),
        max_time_until_resolution_sec=179,
    )

    assert report["exported_rows"] == 0
    assert report["skipped_rows_by_reason"]["candidate_rows"] == 6


def test_export_training_dataset_read_only_connection_rejects_writes(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_filter_db(db_path)

    conn = _connect_read_only(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE should_not_write (id INTEGER)")
    finally:
        conn.close()


def test_export_training_dataset_cli_is_offline_and_prints_summary(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    out_path = tmp_path / "training.parquet"
    _seed_filter_db(db_path)
    before_counts = _table_counts(db_path)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("export-training-dataset must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "export-training-dataset", "--db", str(db_path), "--output", str(out_path)],
    )
    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda env_file=None: SimpleNamespace(
            db_path=str(db_path),
            log_level="CRITICAL",
            log_json=False,
        ),
    )
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "SQLiteStore", forbidden)

    main_module.main()
    after_counts = _table_counts(db_path)

    payload = json.loads(capsys.readouterr().out)
    assert after_counts == before_counts
    assert payload["exported_rows"] == 1
    assert payload["usable_rows"] == 1
    assert payload["output_path"] == str(out_path)
    assert out_path.exists()


def test_export_training_dataset_cli_uses_db_argument(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "explicit_recorder.db"
    wrong_db_path = tmp_path / "wrong_recorder.db"
    out_path = tmp_path / "training.parquet"
    _seed_filter_db(db_path)

    import src.main as main_module

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "export-training-dataset", "--db", str(db_path), "--output", str(out_path)],
    )
    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda env_file=None: SimpleNamespace(
            db_path=str(wrong_db_path),
            log_level="CRITICAL",
            log_json=False,
        ),
    )
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["exported_rows"] == 1
    assert payload["output_path"] == str(out_path)
    assert out_path.exists()
    assert not wrong_db_path.exists()
