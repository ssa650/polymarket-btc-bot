from __future__ import annotations

import json
from datetime import datetime, timezone

from src.db import SQLiteStore
from src.models import BTCPriceSampleRecord, local_arrival_iso_from_ns


def test_btc_price_samples_insert_and_read_latest(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_btc")
    try:
        store.init_schema()
        first_arrival_ns = 1_767_225_660_000_000_000
        second_arrival_ns = 1_767_225_661_000_000_000
        inserted, skipped = store.insert_btc_price_samples(
            [
                BTCPriceSampleRecord(
                    source="offline_fixture",
                    price=42100.25,
                    exchange_timestamp=datetime(2026, 1, 1, 0, 0, 58, tzinfo=timezone.utc),
                    local_arrival_ns=first_arrival_ns,
                    local_arrival_iso=local_arrival_iso_from_ns(first_arrival_ns),
                    raw_json=json.dumps({"price": 42100.25, "source": "offline_fixture"}),
                ),
                BTCPriceSampleRecord(
                    source="offline_fixture",
                    price=42105.50,
                    exchange_timestamp=datetime(2026, 1, 1, 0, 0, 59, tzinfo=timezone.utc),
                    local_arrival_ns=second_arrival_ns,
                    local_arrival_iso=local_arrival_iso_from_ns(second_arrival_ns),
                    raw_json=json.dumps({"price": 42105.50, "source": "offline_fixture"}),
                ),
            ]
        )

        assert inserted == 2
        assert skipped == 0

        latest = store.get_latest_btc_price_sample(source="offline_fixture")
        assert latest is not None
        assert latest.run_id == "run_btc"
        assert latest.source == "offline_fixture"
        assert latest.price == 42105.50
        assert latest.local_arrival_ns == second_arrival_ns
        assert latest.local_arrival_iso == local_arrival_iso_from_ns(second_arrival_ns)
        assert latest.exchange_timestamp == datetime(
            2026, 1, 1, 0, 0, 59, tzinfo=timezone.utc
        )
        assert json.loads(latest.raw_json)["price"] == 42105.50

        row = store.conn.execute(
            """
            SELECT run_id, source, price, exchange_timestamp, local_arrival_ns,
                   local_arrival_iso, raw_json, recorder_version, schema_version
            FROM btc_prices
            ORDER BY local_arrival_ns DESC
            LIMIT 1
            """
        ).fetchone()
        assert row[0] == "run_btc"
        assert row[1] == "offline_fixture"
        assert row[2] == 42105.50
        assert row[3] == "2026-01-01T00:00:59+00:00"
        assert row[4] == second_arrival_ns
        assert row[5] == local_arrival_iso_from_ns(second_arrival_ns)
        assert json.loads(row[6])["source"] == "offline_fixture"
        assert row[7]
        assert row[8]
    finally:
        store.close()


def test_btc_price_samples_are_append_only(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = SQLiteStore(str(db_path), run_id="run_btc")
    try:
        store.init_schema()
        arrival_ns = 1_767_225_660_000_000_000
        sample = BTCPriceSampleRecord(
            source="offline_fixture",
            price=42100.25,
            exchange_timestamp=None,
            local_arrival_ns=arrival_ns,
            local_arrival_iso=local_arrival_iso_from_ns(arrival_ns),
            raw_json=json.dumps({"price": 42100.25}),
        )

        first_inserted, first_skipped = store.insert_btc_price_samples([sample])
        second_inserted, second_skipped = store.insert_btc_price_samples([sample])

        assert (first_inserted, first_skipped) == (1, 0)
        assert (second_inserted, second_skipped) == (1, 0)
        count = store.conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        assert count == 2
    finally:
        store.close()
