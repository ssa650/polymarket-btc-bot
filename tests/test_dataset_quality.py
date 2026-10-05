from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from src.dataset_quality import (
    build_dataset_quality_report,
    render_dataset_quality_report,
)
from src.db import SQLiteStore
from src.models import (
    FeatureRecord,
    MarketMetadata,
    MarketSnapshotRecord,
    OrderBookLevelRecord,
    to_iso,
)


RUN_ID = "run_quality"
NOW = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def _market(market_id: str, *, start: datetime, close: datetime) -> MarketMetadata:
    return MarketMetadata(
        market_id=market_id,
        event_id=f"event_{market_id}",
        question=f"Bitcoin Up or Down - {market_id}",
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        platform_status="open",
        phase="active",
        tracking_state="selected_active",
        status="open",
        market_phase="active",
        yes_token_id=f"{market_id}_yes",
        no_token_id=f"{market_id}_no",
        condition_id=f"condition_{market_id}",
        run_id=RUN_ID,
        strict_validation_passed=1,
    )


def _snapshot(market_id: str, ts: datetime, *, trade: int = 1) -> MarketSnapshotRecord:
    return MarketSnapshotRecord(
        timestamp=ts,
        market_id=market_id,
        yes_price=0.51,
        no_price=0.49,
        best_bid_yes=0.50,
        best_ask_yes=0.52,
        best_bid_no=0.48,
        best_ask_no=0.50,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.51,
        mid_price_no=0.49,
        volume=100.0,
        liquidity=200.0,
        last_trade_price=0.51 if trade else None,
        last_trade_size=5.0 if trade else None,
        last_trade_time=ts if trade else None,
        has_orderbook=1,
        has_trade_data=trade,
        run_id=RUN_ID,
        strict_validation_passed=1,
        snapshot_quality_status="ok",
        feature_ready=1,
    )


def _feature(
    market_id: str,
    ts: datetime,
    *,
    ready: int,
    gap: int = 0,
    quality: str = "ok",
) -> FeatureRecord:
    return FeatureRecord(
        timestamp=ts,
        market_id=market_id,
        run_id=RUN_ID,
        feature_ready=ready,
        is_gap_affected=gap,
        snapshot_quality_status=quality,
        strict_validation_passed=1,
    )


def _insert_btc(
    store: SQLiteStore,
    *,
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
            RUN_ID,
            source,
            price,
            to_iso(ts - timedelta(milliseconds=100)),
            1_767_226_200_000_000_000 + idx,
            to_iso(ts),
            json.dumps({"source": source, "price": price}),
        ),
    )


def _seed_quality_db(db_path) -> None:
    store = SQLiteStore(str(db_path), run_id=RUN_ID)
    try:
        store.init_schema()
        recent_ts = NOW - timedelta(seconds=1)
        older_ts = NOW - timedelta(hours=2)
        store.upsert_markets(
            [
                _market(
                    "m_recent",
                    start=NOW - timedelta(minutes=4),
                    close=NOW + timedelta(minutes=1),
                ),
                _market(
                    "m_old",
                    start=older_ts - timedelta(minutes=4),
                    close=older_ts + timedelta(minutes=1),
                ),
            ],
            run_id=RUN_ID,
        )
        store.insert_market_snapshots(
            [
                _snapshot("m_recent", NOW - timedelta(seconds=30), trade=1),
                _snapshot("m_recent", recent_ts, trade=0),
                _snapshot("m_old", older_ts, trade=1),
            ]
        )
        store.insert_features(
            [
                _feature("m_recent", NOW - timedelta(seconds=30), ready=1),
                _feature("m_recent", recent_ts, ready=0, gap=1, quality="gap"),
                _feature("m_old", older_ts, ready=0),
            ]
        )
        store.insert_order_book_levels(
            [
                OrderBookLevelRecord(
                    timestamp=recent_ts,
                    market_id="m_recent",
                    outcome_side="YES",
                    book_side="bid",
                    level=1,
                    price=0.50,
                    size=10.0,
                    run_id=RUN_ID,
                )
            ]
        )
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO trades (
                    run_id, timestamp, market_id, trade_id, price, size, side
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, to_iso(recent_ts), "m_recent", "t1", 0.51, 5.0, "BUY"),
            )
            store.conn.execute(
                """
                INSERT INTO market_events (
                    run_id, timestamp, market_id, event_type, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (RUN_ID, to_iso(recent_ts), "m_recent", "new_market", "{}"),
            )
            store.conn.execute(
                """
                INSERT INTO best_bid_ask_updates (
                    run_id, timestamp, market_id, condition_id, asset_id,
                    best_bid, best_ask, spread, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    RUN_ID,
                    to_iso(recent_ts),
                    "m_recent",
                    "condition_m_recent",
                    "m_recent_yes",
                    0.50,
                    0.52,
                    0.02,
                    "{}",
                ),
            )
            store.conn.execute(
                """
                INSERT INTO raw_polymarket_events (
                    run_id, local_arrival_ns, local_arrival_iso, exchange_timestamp,
                    event_type, market_id, condition_id, asset_id, slug,
                    parse_status, parse_error, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    RUN_ID,
                    1_767_226_200_000_000_001,
                    to_iso(recent_ts),
                    to_iso(recent_ts),
                    "book",
                    "m_recent",
                    "condition_m_recent",
                    "m_recent_yes",
                    "btc-up-down",
                    "ok",
                    None,
                    '{"event_type":"book"}',
                ),
            )
            _insert_btc(
                store,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                ts=NOW - timedelta(seconds=2),
                price=50_000.0,
                idx=1,
            )
            _insert_btc(
                store,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                ts=NOW - timedelta(seconds=1),
                price=50_001.0,
                idx=2,
            )
            _insert_btc(
                store,
                source=POLYMARKET_RTDS_BINANCE_SOURCE,
                ts=NOW - timedelta(seconds=10),
                price=49_999.0,
                idx=3,
            )
            store.conn.execute(
                """
                INSERT INTO recorder_metrics (
                    run_id, timestamp, markets_polled, successful_market_fetches,
                    failed_markets, rows_inserted, duplicate_rows_skipped,
                    raw_ws_events_seen, raw_ws_events_written
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    RUN_ID,
                    to_iso(NOW - timedelta(seconds=5)),
                    1,
                    1,
                    0,
                    10,
                    0,
                    20,
                    9,
                ),
            )
            store.conn.execute(
                """
                INSERT INTO recorder_metrics (
                    run_id, timestamp, markets_polled, successful_market_fetches,
                    failed_markets, rows_inserted, duplicate_rows_skipped,
                    raw_ws_events_seen, raw_ws_events_written
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (RUN_ID, to_iso(recent_ts), 1, 1, 0, 10, 0, 25, 7),
            )
    finally:
        store.close()


def test_dataset_quality_reports_btc_health_and_feature_ready(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_quality_db(db_path)

    report = build_dataset_quality_report(str(db_path), now=NOW)

    chainlink = next(
        row
        for row in report["btc_source_health"]
        if row["source"] == POLYMARKET_RTDS_CHAINLINK_SOURCE
    )
    assert chainlink["canonical"] is True
    assert chainlink["row_count"] == 2
    assert chainlink["sample_age_sec"] == 1.0
    assert chainlink["rows_per_sec"] == 2.0
    readiness = report["feature_readiness"]
    assert readiness["total_feature_rows"] == 3
    assert readiness["feature_ready_rows"] == 1
    assert readiness["feature_ready_pct"] == 33.33
    assert readiness["gap_affected_rows"] == 1
    assert report["raw_ws_events_written"] == 7


def test_dataset_quality_warnings_for_stale_btc_raw_events_and_low_ready(
    tmp_path,
) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_quality_db(db_path)

    report = build_dataset_quality_report(str(db_path), now=NOW)

    warnings = report["warnings"]
    assert f"btc_source_stale source={POLYMARKET_RTDS_BINANCE_SOURCE}" in warnings
    assert "raw_ws_events_written_gt_zero value=7" in warnings
    assert "feature_ready_pct_below_50 value=33.33" in warnings
    assert "gap_pct_above_5 value=33.33" in warnings


def test_dataset_quality_render_includes_per_market_coverage(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_quality_db(db_path)

    output = render_dataset_quality_report(
        build_dataset_quality_report(str(db_path), now=NOW)
    )

    assert "=== Per Market Coverage ===" in output
    assert "market_id=m_recent" in output
    assert "question=Bitcoin Up or Down - m_recent" in output
    assert "snapshot_count=2" in output
    assert "feature_count=2" in output
    assert "ready_feature_count=1" in output
    assert "ready_pct=50.0" in output


def test_dataset_quality_recent_minutes_filter(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_quality_db(db_path)

    report = build_dataset_quality_report(
        str(db_path),
        recent_minutes=1,
        now=NOW,
    )
    output = render_dataset_quality_report(report)

    assert report["mode"] == "recent_only"
    assert report["row_counts"]["market_snapshots"] == 2
    assert report["row_counts"]["features"] == 2
    assert report["feature_readiness"]["total_feature_rows"] == 2
    assert "mode=recent_only" in output
    assert "recent_minutes=1" in output
    assert "market_id=m_old" not in output


def test_dataset_quality_cli_does_not_start_recorder_or_network(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    db_path = tmp_path / "recorder.db"
    _seed_quality_db(db_path)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dataset-quality must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "dataset-quality", "--recent-minutes", "1"],
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

    out = capsys.readouterr().out
    assert "=== Dataset Quality ===" in out
    assert "mode=recent_only" in out
    assert "recent_minutes=1.0" in out
