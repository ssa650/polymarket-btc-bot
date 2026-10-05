from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from src.recorder import RecorderApp, RecorderMetricRecord


class _CapturingLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, dict[str, object]]] = []

    def info(self, message: str, *, extra: dict[str, object] | None = None) -> None:
        self.entries.append(("info", message, dict(extra or {})))

    def warning(self, message: str, *, extra: dict[str, object] | None = None) -> None:
        self.entries.append(("warning", message, dict(extra or {})))

    def exception(self, message: str, *, extra: dict[str, object] | None = None) -> None:
        self.entries.append(("exception", message, dict(extra or {})))


class _FakeDB:
    def snapshot_trade_integrity_counts(self, run_id: str) -> dict[str, int]:
        return {"total_invalid": 0}


def _metric(**overrides: object) -> RecorderMetricRecord:
    payload: dict[str, object] = {
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "markets_polled": 0,
        "successful_market_fetches": 0,
        "failed_markets": 0,
        "api_latency_ms": None,
        "db_write_time_ms": 0.0,
        "cycle_duration_ms": 0.0,
        "rows_inserted": 0,
        "duplicate_rows_skipped": 0,
        "active_markets_snapshot_attempted": 0,
        "snapshots_written": 0,
        "snapshot_markets_skipped": 0,
        "discovery_candidates_seen": 0,
        "discovery_candidates_matched": 0,
        "discovery_strict_5m_candidates": 0,
        "discovery_broad_btc_candidates": 0,
        "discovery_fallback_used": False,
        "tracked_markets_count": 0,
        "api_call_count": 0,
        "api_success_count": 0,
        "api_failure_count": 0,
        "fetched_trades": 0,
        "accepted_trades": 0,
        "rejected_before_start": 0,
        "rejected_after_close": 0,
        "skipped_already_seen": 0,
        "last_accepted_trade_ts": None,
        "run_id": "run_test",
        "recorder_version": "test",
        "schema_version": 1,
    }
    payload.update(overrides)
    return RecorderMetricRecord(**payload)


def _build_app_stub() -> tuple[RecorderApp, _CapturingLog]:
    app = RecorderApp.__new__(RecorderApp)
    log = _CapturingLog()
    app.log = log
    app.db = _FakeDB()
    app.run_id = "run_test"
    app.settings = SimpleNamespace(
        heartbeat_log_interval_sec=0,
        trade_snapshot_freshness_sec=2,
        btc_price_feed_enabled=True,
        btc_price_feed_source="mock",
        btc_price_max_exchange_age_sec=15.0,
    )
    app._last_heartbeat_monotonic = 0.0
    app._btc_feed_health = {}
    app.btc_price_feeds = {}
    app.btc_price_feed = None
    return app, log


def _last_heartbeat(log: _CapturingLog) -> dict[str, object]:
    heartbeats = [
        extra
        for level, message, extra in log.entries
        if level == "info" and message == "recorder_heartbeat"
    ]
    assert heartbeats, "expected at least one recorder_heartbeat log entry"
    return heartbeats[-1]


def test_heartbeat_carries_forward_snapshot_quality_on_no_state_change() -> None:
    app, log = _build_app_stub()

    app._maybe_log_heartbeat(
        metrics=_metric(snapshots_written=1, active_markets_snapshot_attempted=1),
        tracked_markets=1,
        snapshots=[
            SimpleNamespace(
                mid_price_yes=0.5,
                has_trade_data=1,
                is_partial_orderbook=0,
                is_gap_affected=0,
                feature_ready=1,
            )
        ],
        skipped_outside_window=[],
        skipped_missing_orderbook=[],
        skipped_no_state_change=[],
        snapshot_generation_failed=[],
        features=[],
    )

    first = _last_heartbeat(log)
    assert first["snapshot_has_trade_data_pct"] == 100.0
    assert first["snapshot_quality_metrics_source"] == "current_written_snapshots"
    assert first["snapshot_quality_sample_size"] == 1

    app._maybe_log_heartbeat(
        metrics=_metric(accepted_trades=2),
        tracked_markets=1,
        snapshots=[],
        skipped_outside_window=[],
        skipped_missing_orderbook=[],
        skipped_no_state_change=["m1"],
        snapshot_generation_failed=[],
        features=[],
    )

    second = _last_heartbeat(log)
    assert second["snapshot_mid_price_yes_non_null_pct"] == 100.0
    assert second["snapshot_has_trade_data_pct"] == 100.0
    assert second["snapshot_feature_ready_count"] == 1
    assert second["snapshot_quality_metrics_source"] == "carried_forward_no_state_change"
    assert second["snapshot_quality_sample_size"] == 1

    false_positive_warning = [
        entry
        for entry in log.entries
        if entry[0] == "warning"
        and entry[1] == "accepted_trades_without_snapshot_trade_attachment"
    ]
    assert false_positive_warning == []


def test_heartbeat_uses_zero_quality_for_non_no_state_change_empty_cycle() -> None:
    app, log = _build_app_stub()

    app._maybe_log_heartbeat(
        metrics=_metric(accepted_trades=1),
        tracked_markets=1,
        snapshots=[],
        skipped_outside_window=[],
        skipped_missing_orderbook=["m1"],
        skipped_no_state_change=[],
        snapshot_generation_failed=[],
        features=[],
    )

    heartbeat = _last_heartbeat(log)
    assert heartbeat["snapshot_mid_price_yes_non_null_pct"] == 0.0
    assert heartbeat["snapshot_has_trade_data_pct"] == 0.0
    assert heartbeat["snapshot_quality_metrics_source"] == "current_cycle_no_written_snapshots"
    assert heartbeat["snapshot_quality_sample_size"] == 0


def test_heartbeat_does_not_carry_forward_for_mixed_empty_cycle() -> None:
    app, log = _build_app_stub()

    app._maybe_log_heartbeat(
        metrics=_metric(snapshots_written=1, active_markets_snapshot_attempted=1),
        tracked_markets=1,
        snapshots=[
            SimpleNamespace(
                mid_price_yes=0.5,
                has_trade_data=1,
                is_partial_orderbook=0,
                is_gap_affected=0,
                feature_ready=1,
            )
        ],
        skipped_outside_window=[],
        skipped_missing_orderbook=[],
        skipped_no_state_change=[],
        snapshot_generation_failed=[],
        features=[],
    )

    app._maybe_log_heartbeat(
        metrics=_metric(),
        tracked_markets=1,
        snapshots=[],
        skipped_outside_window=[],
        skipped_missing_orderbook=["m1"],
        skipped_no_state_change=["m2"],
        snapshot_generation_failed=[],
        features=[],
    )

    heartbeat = _last_heartbeat(log)
    assert heartbeat["snapshot_has_trade_data_pct"] == 0.0
    assert heartbeat["snapshot_quality_metrics_source"] == "current_cycle_no_written_snapshots"


def test_heartbeat_includes_ws_health_fields() -> None:
    app, log = _build_app_stub()

    app._maybe_log_heartbeat(
        metrics=_metric(
            ws_reconnect_count=2,
            raw_ws_events_seen=10,
            raw_ws_events_written=9,
            malformed_ws_events=1,
            raw_ws_write_failures=1,
            last_ws_event_age_sec=4.3219,
            subscribed_asset_count=2,
            subscribed_asset_ids_json='["tok_no","tok_yes"]',
        ),
        tracked_markets=1,
        snapshots=[],
        skipped_outside_window=[],
        skipped_missing_orderbook=[],
        skipped_no_state_change=[],
        snapshot_generation_failed=[],
        features=[],
    )

    heartbeat = _last_heartbeat(log)
    assert heartbeat["ws_reconnect_count"] == 2
    assert heartbeat["raw_ws_events_seen"] == 10
    assert heartbeat["raw_ws_events_written"] == 9
    assert heartbeat["malformed_ws_events"] == 1
    assert heartbeat["raw_ws_write_failures"] == 1
    assert heartbeat["last_ws_event_age_sec"] == 4.322
    assert heartbeat["subscribed_asset_count"] == 2
    assert heartbeat["subscribed_asset_ids_json"] == '["tok_no","tok_yes"]'


def test_heartbeat_includes_btc_price_stale_drop_fields() -> None:
    app, log = _build_app_stub()
    latest_exchange_timestamp = datetime.now(timezone.utc).isoformat()
    app._btc_feed_health = {
        "polymarket_rtds_chainlink": {
            "latest_price": 42_000.0,
            "latest_exchange_timestamp": latest_exchange_timestamp,
            "rows_inserted": 4,
            "stale_messages_dropped": 3,
            "stale_rows_dropped": 3,
        }
    }
    app.btc_price_feeds = {
        "polymarket_rtds_chainlink": SimpleNamespace(reconnect_count=2)
    }

    app._maybe_log_heartbeat(
        metrics=_metric(),
        tracked_markets=1,
        snapshots=[],
        skipped_outside_window=[],
        skipped_missing_orderbook=[],
        skipped_no_state_change=[],
        snapshot_generation_failed=[],
        features=[],
    )

    heartbeat = _last_heartbeat(log)
    assert heartbeat["btc_price_feed_enabled"] is True
    assert heartbeat["btc_price_feed_healthy"] is True
    assert heartbeat["latest_btc_price_by_source"] == {
        "polymarket_rtds_chainlink": 42_000.0
    }
    assert heartbeat["latest_btc_exchange_timestamp_by_source"] == {
        "polymarket_rtds_chainlink": latest_exchange_timestamp
    }
    assert heartbeat["btc_price_age_sec_by_source"]["polymarket_rtds_chainlink"] >= 0
    assert heartbeat["btc_price_latest_age_sec_by_source"] == heartbeat[
        "btc_price_age_sec_by_source"
    ]
    assert heartbeat["btc_price_rows_inserted_by_source"] == {
        "polymarket_rtds_chainlink": 4
    }
    assert heartbeat["btc_stale_messages_dropped_by_source"] == {
        "polymarket_rtds_chainlink": 3
    }
    assert heartbeat["btc_price_stale_rows_dropped_by_source"] == {
        "polymarket_rtds_chainlink": 3
    }
    assert heartbeat["btc_feed_reconnect_count_by_source"] == {
        "polymarket_rtds_chainlink": 2
    }
    assert heartbeat["btc_price_reconnects_by_source"] == {
        "polymarket_rtds_chainlink": 2
    }
    assert heartbeat["btc_price_last_error_by_source"] == {
        "polymarket_rtds_chainlink": None
    }


def test_btc_feed_startup_does_not_block_recorder_heartbeat() -> None:
    app, log = _build_app_stub()
    app.stop_event = None
    app._btc_feed_health = {}
    app.btc_price_feeds = {"blocking": SimpleNamespace(source="blocking")}
    app.btc_price_feed = app.btc_price_feeds["blocking"]
    app._btc_price_feed_enabled = lambda: True
    app._auto_normalize_resolutions_enabled = lambda: False
    app._auto_repair_stale_active_enabled = lambda: False
    app._raw_ws_events_prune_enabled = lambda: False
    app._sqlite_wal_checkpoint_enabled = lambda: False

    async def _idle() -> None:
        await asyncio.Event().wait()

    async def _snapshot_loop() -> None:
        app._maybe_log_heartbeat(
            metrics=_metric(),
            tracked_markets=0,
            snapshots=[],
            skipped_outside_window=[],
            skipped_missing_orderbook=[],
            skipped_no_state_change=[],
            snapshot_generation_failed=[],
            features=[],
        )
        app.stop_event.set()
        await asyncio.Event().wait()

    async def _blocking_btc_loop(source: str | None = None) -> None:
        _ = source
        await asyncio.Event().wait()

    async def _run() -> None:
        app.stop_event = asyncio.Event()
        app._discovery_loop = _idle
        app._snapshot_loop = _snapshot_loop
        app._ws_loop = _idle
        app._trade_backfill_loop = _idle
        app._btc_price_feed_loop = _blocking_btc_loop
        tasks = app._create_runtime_tasks()
        try:
            await asyncio.wait_for(app.stop_event.wait(), timeout=0.2)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run())

    heartbeat = _last_heartbeat(log)
    assert heartbeat["btc_price_feed_enabled"] is True
    assert heartbeat["btc_price_feed_healthy"] is False
