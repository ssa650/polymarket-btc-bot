from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.db import SQLiteStore
from src.models import (
    local_arrival_iso_from_ns,
    parse_exchange_timestamp,
)
from src.polymarket.normalization import normalize_raw_polymarket_events


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "polymarket_ws_events.jsonl"
LOCAL_ARRIVAL_NS = 1_775_286_600_123_456_789


def _fixture_lines() -> list[str]:
    return [
        line
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parse_exchange_timestamp_handles_seconds_millis_micros_nanos_and_iso() -> None:
    expected = datetime(2025, 9, 15, 4, 1, 32, 351000, tzinfo=timezone.utc)

    assert parse_exchange_timestamp("1757908892351") == expected
    assert parse_exchange_timestamp(1757908892351) == expected
    assert parse_exchange_timestamp(1757908892.351) == expected
    assert parse_exchange_timestamp(1757908892351000) == expected
    assert parse_exchange_timestamp(1757908892351000000) == expected
    assert parse_exchange_timestamp("2025-09-15T04:01:32.351Z") == expected


def test_local_arrival_iso_from_ns_is_utc_iso_string() -> None:
    assert local_arrival_iso_from_ns(LOCAL_ARRIVAL_NS) == (
        "2026-04-04T07:10:00.123457+00:00"
    )


def test_normalize_raw_polymarket_events_from_fixture_preserves_replay_fields() -> None:
    records = []
    for line in _fixture_lines():
        records.extend(
            normalize_raw_polymarket_events(
                line,
                local_arrival_ns=LOCAL_ARRIVAL_NS,
            )
        )

    event_types = [record.event_type for record in records]
    assert event_types == [
        "book",
        "price_change",
        "last_trade_price",
        "tick_size_change",
        "best_bid_ask",
        "new_market",
        "market_resolved",
    ]
    assert all(record.local_arrival_ns == LOCAL_ARRIVAL_NS for record in records)
    assert all(record.local_arrival_iso.endswith("+00:00") for record in records)
    assert all(record.parse_status == "ok" for record in records)
    assert records[0].asset_id == "asset_yes"
    assert records[0].condition_id == "0xcondition"
    assert records[0].exchange_timestamp == datetime(
        2025, 9, 15, 4, 1, 32, 351000, tzinfo=timezone.utc
    )
    assert records[5].slug == "btc-updown-5m-1766790300"
    assert records[5].market_id == "0xcondition_new"
    assert records[5].condition_id == "0xcondition_new"

    decoded = json.loads(records[2].raw_json)
    assert decoded["event_type"] == "last_trade_price"
    assert decoded["transaction_hash"] == "0xtradehash"


def test_normalize_raw_polymarket_events_records_malformed_payload() -> None:
    records = normalize_raw_polymarket_events(
        "{not-json",
        local_arrival_ns=LOCAL_ARRIVAL_NS,
    )

    assert len(records) == 1
    record = records[0]
    assert record.event_type == "unknown"
    assert record.parse_status == "error"
    assert record.parse_error
    assert json.loads(record.raw_json)["raw_message"] == "{not-json"


def test_insert_raw_polymarket_events_writes_append_only_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_raw")
    try:
        store.init_schema()
        records = []
        for line in _fixture_lines():
            records.extend(
                normalize_raw_polymarket_events(
                    line,
                    local_arrival_ns=LOCAL_ARRIVAL_NS,
                )
            )
        for record in records:
            record.run_id = "run_raw"

        inserted, skipped = store.insert_raw_polymarket_events(records)

        assert inserted == len(records)
        assert skipped == 0

        rows = store.conn.execute(
            """
            SELECT event_type, COUNT(*)
            FROM raw_polymarket_events
            WHERE run_id = 'run_raw'
            GROUP BY event_type
            ORDER BY event_type
            """
        ).fetchall()
        assert dict(rows) == {
            "best_bid_ask": 1,
            "book": 1,
            "last_trade_price": 1,
            "market_resolved": 1,
            "new_market": 1,
            "price_change": 1,
            "tick_size_change": 1,
        }

        row = store.conn.execute(
            """
            SELECT local_arrival_ns, local_arrival_iso, exchange_timestamp,
                   market_id, condition_id, asset_id, slug, parse_status,
                   parse_error, raw_json
            FROM raw_polymarket_events
            WHERE event_type = 'new_market'
            LIMIT 1
            """
        ).fetchone()
        assert row[0] == LOCAL_ARRIVAL_NS
        assert row[1] == local_arrival_iso_from_ns(LOCAL_ARRIVAL_NS)
        assert row[2] == "2025-12-26T23:06:55.550000+00:00"
        assert row[3] == "0xcondition_new"
        assert row[4] == "0xcondition_new"
        assert row[5] is None
        assert row[6] == "btc-updown-5m-1766790300"
        assert row[7] == "ok"
        assert row[8] is None
        assert json.loads(row[9])["event_type"] == "new_market"

        # No legacy table is rewritten as part of raw event insertion.
        assert int(store.conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]) == 0
    finally:
        store.close()


def test_prune_raw_polymarket_events_deletes_bounded_old_rows(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_raw")
    try:
        store.init_schema()
        records = []
        for index, line in enumerate(_fixture_lines()[:5]):
            records.extend(
                normalize_raw_polymarket_events(
                    line,
                    local_arrival_ns=LOCAL_ARRIVAL_NS + index,
                )
            )
        for record in records:
            record.run_id = "run_raw"
        inserted, _skipped = store.insert_raw_polymarket_events(records)
        assert inserted == 5

        report = store.prune_raw_polymarket_events(
            retention_sec=1.0,
            batch_size=2,
            now_ns=LOCAL_ARRIVAL_NS + 2_000_000_000,
        )

        assert report["deleted_rows"] == 2
        assert report["deleted_by_retention"] == 2
        remaining = store.conn.execute(
            "SELECT COUNT(*) FROM raw_polymarket_events"
        ).fetchone()[0]
        assert remaining == 3

        report = store.prune_raw_polymarket_events(max_rows=1, batch_size=1)

        assert report["deleted_rows"] == 1
        assert report["deleted_by_max_rows"] == 1
        remaining = store.conn.execute(
            "SELECT COUNT(*) FROM raw_polymarket_events"
        ).fetchone()[0]
        assert remaining == 2
    finally:
        store.close()
