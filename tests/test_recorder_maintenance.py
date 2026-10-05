from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_settings
from src.recorder import RecorderApp


class _FakeDB:
    def __init__(self) -> None:
        self.normalize_calls: list[dict[str, object]] = []
        self.repair_calls: list[dict[str, object]] = []
        self.fail_normalize = False
        self.fail_repair = False
        self.closed = False

    def normalize_market_resolutions(
        self,
        *,
        dry_run: bool,
        run_id: str | None,
    ) -> dict[str, object]:
        self.normalize_calls.append({"dry_run": dry_run, "run_id": run_id})
        if self.fail_normalize:
            raise RuntimeError("offline normalize failure")
        return {
            "run_id_scope": "all",
            "resolution_events_seen": 2,
            "mapped_resolution_events": 2,
            "unmapped_resolution_events": 0,
            "ambiguous_resolution_events": 0,
            "markets_updated": 1,
        }

    def repair_stale_selected_active_markets(
        self,
        *,
        dry_run: bool,
        grace_sec: float,
        run_id: str | None,
    ) -> dict[str, object]:
        self.repair_calls.append(
            {"dry_run": dry_run, "grace_sec": grace_sec, "run_id": run_id}
        )
        if self.fail_repair:
            raise RuntimeError("offline repair failure")
        return {
            "run_id_scope": "all",
            "grace_sec": grace_sec,
            "stale_selected_active_count": 1,
            "demoted_count": 1,
            "market_events_inserted": 1,
        }

    def close(self) -> None:
        self.closed = True


class _AsyncClosable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _app(*, settings: SimpleNamespace | None = None, db: _FakeDB | None = None) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.run_id = "run_maintenance"
    app.settings = settings or SimpleNamespace()
    app.db = db or _FakeDB()
    app.log = logging.getLogger("test.recorder.maintenance")
    app.stop_event = asyncio.Event()
    app._tasks = []
    app._ws_reconnect_count = 0
    app._raw_ws_events_seen = 0
    app._raw_ws_events_written = 0
    app._raw_ws_malformed_events = 0
    app._raw_ws_write_failures = 0
    return app


def test_auto_normalize_task_calls_resolution_normalization_on_interval() -> None:
    async def _run() -> tuple[list[dict[str, object]], list[float]]:
        db = _FakeDB()
        app = _app(
            db=db,
            settings=SimpleNamespace(auto_normalize_resolutions_interval_sec=42.0),
        )
        intervals: list[float] = []

        async def _sleep_once(*, cycle_start: float, interval_sec: float) -> None:
            _ = cycle_start
            intervals.append(interval_sec)
            app.stop_event.set()

        app._sleep_until_next_cycle = _sleep_once
        await app._auto_normalize_resolutions_loop()
        return db.normalize_calls, intervals

    calls, intervals = asyncio.run(_run())

    assert calls == [{"dry_run": False, "run_id": None}]
    assert intervals == [42.0]


def test_auto_repair_task_calls_stale_active_repair_on_interval() -> None:
    async def _run() -> tuple[list[dict[str, object]], list[float]]:
        db = _FakeDB()
        app = _app(
            db=db,
            settings=SimpleNamespace(
                auto_repair_stale_active_interval_sec=17.0,
                auto_repair_stale_active_grace_sec=123.0,
            ),
        )
        intervals: list[float] = []

        async def _sleep_once(*, cycle_start: float, interval_sec: float) -> None:
            _ = cycle_start
            intervals.append(interval_sec)
            app.stop_event.set()

        app._sleep_until_next_cycle = _sleep_once
        await app._auto_repair_stale_active_loop()
        return db.repair_calls, intervals

    calls, intervals = asyncio.run(_run())

    assert calls == [{"dry_run": False, "grace_sec": 123.0, "run_id": None}]
    assert intervals == [17.0]


def test_auto_maintenance_errors_are_logged_and_non_fatal(caplog) -> None:
    async def _run() -> list[dict[str, object]]:
        db = _FakeDB()
        db.fail_normalize = True
        app = _app(
            db=db,
            settings=SimpleNamespace(auto_normalize_resolutions_interval_sec=1.0),
        )

        async def _sleep_once(*, cycle_start: float, interval_sec: float) -> None:
            _ = (cycle_start, interval_sec)
            app.stop_event.set()

        app._sleep_until_next_cycle = _sleep_once
        await app._auto_normalize_resolutions_loop()
        return db.normalize_calls

    caplog.set_level(logging.ERROR)
    calls = asyncio.run(_run())

    assert calls == [{"dry_run": False, "run_id": None}]
    assert "auto_normalize_resolutions_error" in [
        record.getMessage() for record in caplog.records
    ]


def test_disabled_maintenance_tasks_are_not_started() -> None:
    async def _run() -> list[str]:
        app = _app(
            settings=SimpleNamespace(
                auto_normalize_resolutions_enabled=False,
                auto_repair_stale_active_enabled=False,
            )
        )

        async def _never() -> None:
            await asyncio.Event().wait()

        app._discovery_loop = _never
        app._snapshot_loop = _never
        app._ws_loop = _never
        app._trade_backfill_loop = _never
        app._btc_price_feed_enabled = lambda: False

        tasks = app._create_runtime_tasks()
        names = [task.get_name() for task in tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return names

    assert asyncio.run(_run()) == [
        "discovery_loop",
        "snapshot_loop",
        "ws_loop",
        "trade_backfill_loop",
    ]


def test_graceful_shutdown_cancels_maintenance_tasks_cleanly(caplog) -> None:
    async def _run() -> tuple[bool, bool, bool, bool, bool]:
        app = _app(db=_FakeDB())
        app.gamma = _AsyncClosable()
        app.clob = _AsyncClosable()
        app.btc_price_feed = _AsyncClosable()

        async def _never() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(_never(), name="auto_normalize_resolutions_loop")
        app._tasks = [task]

        await app._shutdown()
        return (
            task.cancelled(),
            app.gamma.closed,
            app.clob.closed,
            app.btc_price_feed.closed,
            app.db.closed,
        )

    caplog.set_level(logging.INFO)
    assert asyncio.run(_run()) == (True, True, True, True, True)
    assert "shutdown_task_summary" in [record.getMessage() for record in caplog.records]
    assert "shutdown_complete" in [record.getMessage() for record in caplog.records]


def test_maintenance_config_defaults_enabled_and_heartbeat_alias(
    monkeypatch,
    tmp_path,
) -> None:
    for name in (
        "AUTO_NORMALIZE_RESOLUTIONS_ENABLED",
        "AUTO_NORMALIZE_RESOLUTIONS_INTERVAL_SEC",
        "AUTO_REPAIR_STALE_ACTIVE_ENABLED",
        "AUTO_REPAIR_STALE_ACTIVE_INTERVAL_SEC",
        "AUTO_REPAIR_STALE_ACTIVE_GRACE_SEC",
        "HEARTBEAT_LOG_INTERVAL_SEC",
        "RECORDER_HEARTBEAT_INTERVAL_SEC",
        "STARTUP_INTEGRITY_CHECK_MODE",
        "BTC_PRICE_FEED_STALE_RECONNECT_SEC",
        "RAW_WS_EVENTS_ENABLED",
        "RAW_WS_EVENTS_RETENTION_SEC",
        "RAW_WS_EVENTS_MAX_ROWS",
        "RAW_WS_EVENTS_PRUNE_INTERVAL_SEC",
        "RAW_WS_EVENTS_PRUNE_BATCH_SIZE",
        "SQLITE_WAL_CHECKPOINT_INTERVAL_SEC",
        "SQLITE_WAL_CHECKPOINT_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(env_file=str(tmp_path / "missing.env"))

    assert settings.auto_normalize_resolutions_enabled is True
    assert settings.auto_normalize_resolutions_interval_sec == 60.0
    assert settings.auto_repair_stale_active_enabled is True
    assert settings.auto_repair_stale_active_interval_sec == 300.0
    assert settings.auto_repair_stale_active_grace_sec == 60.0
    assert settings.heartbeat_log_interval_sec == 30
    assert settings.startup_integrity_check_mode == "quick"
    assert settings.btc_price_feed_stale_reconnect_sec == 10.0
    assert settings.raw_ws_events_enabled is False
    assert settings.raw_ws_events_retention_sec == 0.0
    assert settings.raw_ws_events_max_rows == 0
    assert settings.raw_ws_events_prune_interval_sec == 300.0
    assert settings.raw_ws_events_prune_batch_size == 50000
    assert settings.sqlite_wal_checkpoint_interval_sec == 300.0
    assert settings.sqlite_wal_checkpoint_mode == "PASSIVE"

    monkeypatch.setenv("RECORDER_HEARTBEAT_INTERVAL_SEC", "9")
    heartbeat_settings = load_settings(env_file=str(tmp_path / "missing.env"))
    assert heartbeat_settings.heartbeat_log_interval_sec == 9
    monkeypatch.setenv("BTC_PRICE_FEED_STALE_RECONNECT_SEC", "7.5")
    stale_settings = load_settings(env_file=str(tmp_path / "missing.env"))
    assert stale_settings.btc_price_feed_stale_reconnect_sec == 7.5
    monkeypatch.setenv("RAW_WS_EVENTS_ENABLED", "true")
    monkeypatch.setenv("RAW_WS_EVENTS_RETENTION_SEC", "3600")
    monkeypatch.setenv("RAW_WS_EVENTS_MAX_ROWS", "123")
    monkeypatch.setenv("RAW_WS_EVENTS_PRUNE_INTERVAL_SEC", "17")
    monkeypatch.setenv("RAW_WS_EVENTS_PRUNE_BATCH_SIZE", "456")
    monkeypatch.setenv("SQLITE_WAL_CHECKPOINT_INTERVAL_SEC", "19")
    monkeypatch.setenv("SQLITE_WAL_CHECKPOINT_MODE", "restart")
    raw_settings = load_settings(env_file=str(tmp_path / "missing.env"))
    assert raw_settings.raw_ws_events_enabled is True
    assert raw_settings.raw_ws_events_retention_sec == 3600.0
    assert raw_settings.raw_ws_events_max_rows == 123
    assert raw_settings.raw_ws_events_prune_interval_sec == 17.0
    assert raw_settings.raw_ws_events_prune_batch_size == 456
    assert raw_settings.sqlite_wal_checkpoint_interval_sec == 19.0
    assert raw_settings.sqlite_wal_checkpoint_mode == "RESTART"
    monkeypatch.setenv("STARTUP_INTEGRITY_CHECK_MODE", "full")
    full_settings = load_settings(env_file=str(tmp_path / "missing.env"))
    assert full_settings.startup_integrity_check_mode == "full"
    monkeypatch.setenv("STARTUP_INTEGRITY_CHECK_MODE", "off")
    off_settings = load_settings(env_file=str(tmp_path / "missing.env"))
    assert off_settings.startup_integrity_check_mode == "off"
    monkeypatch.setenv("STARTUP_INTEGRITY_CHECK_MODE", "invalid")
    with pytest.raises(ValueError):
        load_settings(env_file=str(tmp_path / "missing.env"))


def test_run_recorder_script_uses_project_venv_and_btc_feed_defaults() -> None:
    text = Path("scripts/run_recorder.sh").read_text(encoding="utf-8")

    assert ".venv/bin/python" in text
    assert "BTC_PRICE_FEED_ENABLED" in text
    assert "BTC_PRICE_FEED_SOURCES:-" not in text
    assert "polymarket_rtds_chainlink" in text
    assert "btc/usd" in text
    assert "wss://ws-live-data.polymarket.com" in text
    assert "STARTUP_INTEGRITY_CHECK_MODE" in text
    assert "quick" in text
    assert "BTC_PRICE_FEED_STALE_RECONNECT_SEC" in text
    assert "RAW_WS_EVENTS_ENABLED" in text
    assert "false" in text
    assert "SQLITE_WAL_CHECKPOINT_INTERVAL_SEC" in text
    assert "SQLITE_WAL_CHECKPOINT_MODE" in text
    assert "-m src.main" in text
