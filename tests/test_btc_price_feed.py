from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.btc_price_feed import (
    BINANCE_REST_SOURCE,
    BTCPriceFeedStaleError,
    BinanceRESTBTCPriceFeed,
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
    MockBTCPriceFeed,
    MockBTCPriceFeedSample,
    PolymarketRTDSBTCPriceFeed,
    analyze_polymarket_rtds_btc_price_message,
    build_btc_price_feed,
    build_btc_price_feeds,
    describe_btc_price_feed_config,
    parse_btc_price_feed_sources,
    parse_polymarket_rtds_btc_price_message,
    probe_polymarket_rtds_btc_feed,
)
from src.config import load_settings
from src.db import SQLiteStore
from src.diagnostics import print_diagnostics
from src.models import BTCPriceSampleRecord
from src.recorder import RecorderApp


def _settings(**overrides):
    base = {
        "btc_price_feed_enabled": True,
        "btc_price_feed_sources_raw": "",
        "btc_price_feed_sources": (),
        "btc_price_feed_source": "mock",
        "btc_price_feed_symbol": "btc/usd",
        "btc_price_feed_interval_sec": 0.1,
        "btc_price_feed_ws_url": "wss://ws-live-data.polymarket.com",
        "btc_price_feed_stale_reconnect_sec": 10.0,
        "btc_price_max_exchange_age_sec": 15.0,
        "btc_price_drop_stale": True,
        "btc_price_stale_reconnect_sec": 30.0,
        "btc_price_feed_startup_timeout_sec": 5.0,
        "btc_price_feed_sample_timeout_sec": 2.0,
        "btc_price_binance_rest_url": "https://api.binance.com/api/v3/ticker/24hr",
        "ws_reconnect_min_sec": 0.25,
        "ws_reconnect_max_sec": 2.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_app(tmp_path, feed) -> RecorderApp:
    app = RecorderApp.__new__(RecorderApp)
    app.run_id = "run_btc_feed"
    app.log = logging.getLogger("test.recorder.btc_price_feed")
    app.settings = _settings(btc_price_drop_stale=False)
    app.db = SQLiteStore(str(tmp_path / "recorder.db"), run_id=app.run_id)
    app.db.init_schema()
    app.btc_price_feed = feed
    app.btc_price_feeds = {"mock": feed} if feed is not None else {}
    app._btc_feed_health = {}
    return app


class _FakeRTDSWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> object:
        if not self.messages:
            await asyncio.sleep(3600)
        return self.messages.pop(0)


class _FakeRTDSConnect:
    def __init__(self, messages: list[object]) -> None:
        self.ws = _FakeRTDSWebSocket(messages)
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.ws


class _DelayedRTDSWebSocket:
    def __init__(self, messages: list[tuple[float, object]]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> object:
        if not self.messages:
            await asyncio.sleep(3600)
        delay_sec, message = self.messages.pop(0)
        await asyncio.sleep(delay_sec)
        return message


class _DelayedRTDSConnect:
    def __init__(self, messages: list[tuple[float, object]]) -> None:
        self.ws = _DelayedRTDSWebSocket(messages)
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.ws


def test_btc_price_feed_config_defaults_disabled(monkeypatch, tmp_path) -> None:
    for name in (
        "BTC_PRICE_FEED_ENABLED",
        "BTC_PRICE_FEED_SOURCES",
        "BTC_PRICE_FEED_SOURCE",
        "BTC_PRICE_FEED_SYMBOL",
        "BTC_PRICE_FEED_INTERVAL_SEC",
        "BTC_PRICE_FEED_WS_URL",
        "BTC_PRICE_FEED_STALE_RECONNECT_SEC",
        "BTC_PRICE_MAX_EXCHANGE_AGE_SEC",
        "BTC_PRICE_DROP_STALE",
        "BTC_PRICE_STALE_RECONNECT_SEC",
        "BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC",
        "BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC",
        "BTC_PRICE_BINANCE_REST_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(env_file=str(tmp_path / "missing.env"))

    assert settings.btc_price_feed_enabled is False
    assert settings.btc_price_feed_sources == ()
    assert settings.btc_price_feed_source == "mock"
    assert settings.btc_price_feed_symbol == "btc/usd"
    assert settings.btc_price_feed_interval_sec == 1.0
    assert settings.btc_price_feed_ws_url == "wss://ws-live-data.polymarket.com"
    assert settings.btc_price_feed_stale_reconnect_sec == 10.0
    assert settings.btc_price_max_exchange_age_sec == 15.0
    assert settings.btc_price_drop_stale is True
    assert settings.btc_price_stale_reconnect_sec == 30.0
    assert settings.btc_price_feed_startup_timeout_sec == 5.0
    assert settings.btc_price_feed_sample_timeout_sec == 2.0
    assert (
        settings.btc_price_binance_rest_url
        == "https://api.binance.com/api/v3/ticker/24hr"
    )

    monkeypatch.setenv("BTC_PRICE_FEED_STALE_RECONNECT_SEC", "3.5")
    monkeypatch.setenv("BTC_PRICE_MAX_EXCHANGE_AGE_SEC", "7.0")
    monkeypatch.setenv("BTC_PRICE_DROP_STALE", "false")
    monkeypatch.setenv("BTC_PRICE_STALE_RECONNECT_SEC", "11.0")
    monkeypatch.setenv("BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC", "1.5")
    monkeypatch.setenv("BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC", "0.75")
    monkeypatch.setenv("BTC_PRICE_BINANCE_REST_URL", "https://example.test/ticker")
    overridden = load_settings(env_file=str(tmp_path / "missing.env"))
    assert overridden.btc_price_feed_stale_reconnect_sec == 3.5
    assert overridden.btc_price_max_exchange_age_sec == 7.0
    assert overridden.btc_price_drop_stale is False
    assert overridden.btc_price_stale_reconnect_sec == 11.0
    assert overridden.btc_price_feed_startup_timeout_sec == 1.5
    assert overridden.btc_price_feed_sample_timeout_sec == 0.75
    assert overridden.btc_price_binance_rest_url == "https://example.test/ticker"


def test_btc_price_feed_config_parses_multi_source_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BTC_PRICE_FEED_SOURCES", " polymarket_rtds_chainlink, polymarket_rtds_binance ")

    settings = load_settings(env_file=str(tmp_path / "missing.env"))

    assert settings.btc_price_feed_sources == (
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    )
    assert settings.btc_price_feed_sources_raw == (
        " polymarket_rtds_chainlink, polymarket_rtds_binance "
    )


def test_parse_btc_price_feed_sources_dedupes_and_normalizes() -> None:
    assert parse_btc_price_feed_sources(
        "POLYMARKET_RTDS_CHAINLINK,polymarket_rtds_chainlink,,polymarket_rtds_binance,binance_rest"
    ) == (
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
        BINANCE_REST_SOURCE,
    )


def test_build_btc_price_feed_selects_polymarket_rtds_provider() -> None:
    settings = SimpleNamespace(
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_feed_symbol="btc/usd",
        btc_price_feed_ws_url="wss://example.test",
        btc_price_feed_stale_reconnect_sec=4.0,
        ws_reconnect_min_sec=0.25,
        ws_reconnect_max_sec=2.0,
    )

    feed = build_btc_price_feed(settings)

    assert isinstance(feed, PolymarketRTDSBTCPriceFeed)
    assert feed.source == POLYMARKET_RTDS_CHAINLINK_SOURCE
    assert feed.symbol == "btc/usd"
    assert feed.ws_url == "wss://example.test"
    assert feed.stale_reconnect_sec == 4.0


def test_build_btc_price_feeds_keeps_single_source_backward_compatibility() -> None:
    settings = SimpleNamespace(
        btc_price_feed_sources=(),
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_feed_symbol="btc/usd",
        btc_price_feed_ws_url="wss://example.test",
        ws_reconnect_min_sec=0.25,
        ws_reconnect_max_sec=2.0,
    )

    feeds = build_btc_price_feeds(settings)

    assert list(feeds) == [POLYMARKET_RTDS_CHAINLINK_SOURCE]
    assert isinstance(feeds[POLYMARKET_RTDS_CHAINLINK_SOURCE], PolymarketRTDSBTCPriceFeed)


def test_build_btc_price_feeds_configures_chainlink_and_binance_symbols() -> None:
    settings = _settings(
        btc_price_feed_sources=(
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
        ),
        btc_price_feed_source="ignored",
        btc_price_feed_symbol="ignored",
        btc_price_feed_ws_url="wss://example.test",
    )

    feeds = build_btc_price_feeds(settings)

    assert set(feeds) == {
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    }
    assert feeds[POLYMARKET_RTDS_CHAINLINK_SOURCE].symbol == "btc/usd"
    assert feeds[POLYMARKET_RTDS_BINANCE_SOURCE].symbol == "btcusdt"


def test_build_btc_price_feeds_configures_binance_rest_fallback() -> None:
    settings = _settings(
        btc_price_feed_sources=(BINANCE_REST_SOURCE,),
        btc_price_feed_source="ignored",
        btc_price_feed_symbol="ignored",
        btc_price_binance_rest_url="https://example.test/ticker",
    )

    feeds = build_btc_price_feeds(settings)

    feed = feeds[BINANCE_REST_SOURCE]
    assert isinstance(feed, BinanceRESTBTCPriceFeed)
    assert feed.symbol == "BTCUSDT"
    assert feed.url == "https://example.test/ticker"


def test_describe_btc_price_feed_config_reports_planned_feeds() -> None:
    settings = _settings(
        btc_price_feed_sources_raw=(
            f"{POLYMARKET_RTDS_CHAINLINK_SOURCE},{POLYMARKET_RTDS_BINANCE_SOURCE}"
        ),
        btc_price_feed_sources=(
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
        ),
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_feed_symbol="ignored",
        btc_price_feed_ws_url="wss://example.test",
    )

    config = describe_btc_price_feed_config(settings)

    assert config["BTC_PRICE_FEED_ENABLED"] is True
    assert config["BTC_PRICE_FEED_SOURCE"] == POLYMARKET_RTDS_CHAINLINK_SOURCE
    assert config["BTC_PRICE_FEED_SOURCES"] == (
        f"{POLYMARKET_RTDS_CHAINLINK_SOURCE},{POLYMARKET_RTDS_BINANCE_SOURCE}"
    )
    assert config["parsed_sources"] == [
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    ]
    assert config["source_symbol_map"] == {
        POLYMARKET_RTDS_CHAINLINK_SOURCE: "btc/usd",
        POLYMARKET_RTDS_BINANCE_SOURCE: "btcusdt",
    }
    assert config["source_ws_url_map"] == {
        POLYMARKET_RTDS_CHAINLINK_SOURCE: "wss://example.test",
        POLYMARKET_RTDS_BINANCE_SOURCE: "wss://example.test",
    }
    assert config["BTC_PRICE_FEED_STALE_RECONNECT_SEC"] == 10.0
    assert config["BTC_PRICE_MAX_EXCHANGE_AGE_SEC"] == 15.0
    assert config["BTC_PRICE_DROP_STALE"] is True
    assert config["BTC_PRICE_STALE_RECONNECT_SEC"] == 30.0
    assert config["BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC"] == 5.0
    assert config["BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC"] == 2.0
    assert config["BTC_PRICE_BINANCE_REST_URL"] == (
        "https://api.binance.com/api/v3/ticker/24hr"
    )
    assert all(feed["supported"] for feed in config["planned_feeds"])


def test_btc_feed_config_cli_prints_planned_feeds(monkeypatch, tmp_path, capsys) -> None:
    import src.main as main_module

    monkeypatch.setenv("BTC_PRICE_FEED_ENABLED", "true")
    monkeypatch.setenv(
        "BTC_PRICE_FEED_SOURCES",
        f"{POLYMARKET_RTDS_CHAINLINK_SOURCE},{POLYMARKET_RTDS_BINANCE_SOURCE}",
    )
    monkeypatch.setenv("BTC_PRICE_FEED_SOURCE", POLYMARKET_RTDS_CHAINLINK_SOURCE)
    monkeypatch.setenv("BTC_PRICE_FEED_SYMBOL", "btc/usd")
    monkeypatch.setenv("BTC_PRICE_FEED_WS_URL", "wss://example.test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "btc-feed-config", "--env-file", str(tmp_path / "missing.env")],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["BTC_PRICE_FEED_ENABLED"] is True
    assert payload["parsed_sources"] == [
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    ]
    assert payload["source_symbol_map"][POLYMARKET_RTDS_BINANCE_SOURCE] == "btcusdt"


def test_probe_btc_rtds_prints_subscription_raw_match_and_ignore_reason() -> None:
    connector = _FakeRTDSConnect(
        [
            {
                "topic": "crypto_prices",
                "type": "update",
                "payload": {
                    "symbol": "BTCUSDT",
                    "timestamp": 1_753_314_088_395,
                    "value": "67235.25",
                },
            },
            {
                "topic": "crypto_prices",
                "type": "update",
                "payload": {
                    "symbol": "ETHUSDT",
                    "timestamp": 1_753_314_088_395,
                    "value": "3000.0",
                },
            },
        ]
    )
    out = io.StringIO()

    summary = asyncio.run(
        probe_polymarket_rtds_btc_feed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            ws_url="wss://example.test",
            max_messages=2,
            timeout_sec=1.0,
            connect=connector,
            output=out,
        )
    )

    text = out.getvalue()
    assert summary["connection_ok"] is True
    assert summary["messages_seen"] == 2
    assert summary["parser_matches"] == 1
    assert summary["parser_ignored"] == 1
    assert connector.ws.sent
    assert '"filters": "{\\"symbol\\":\\"btcusdt\\"}"' in text
    assert "subscription_payload=" in text
    assert "raw_message=" in text
    assert "parser_result=" in text
    assert '"matched": true' in text
    assert '"ignore_reason": "symbol_mismatch"' in text


def test_probe_btc_rtds_raw_only_prints_raw_messages_without_parser_details() -> None:
    connector = _FakeRTDSConnect(
        [
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "payload": {"symbol": "BTCUSDT", "value": "67235.25"},
                }
            )
        ]
    )
    out = io.StringIO()

    summary = asyncio.run(
        probe_polymarket_rtds_btc_feed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            ws_url="wss://example.test",
            max_messages=1,
            timeout_sec=1.0,
            raw_only=True,
            connect=connector,
            output=out,
        )
    )

    text = out.getvalue()
    assert summary["messages_seen"] == 1
    assert '"symbol": "BTCUSDT"' in text
    assert "parser_result=" not in text


def test_probe_btc_rtds_cli_does_not_open_db(monkeypatch, capsys) -> None:
    import src.btc_price_feed as btc_price_feed_module
    import src.main as main_module

    connector = _FakeRTDSConnect(
        [
            {
                "topic": "crypto_prices",
                "type": "update",
                "payload": {"symbol": "BTCUSDT", "value": "67235.25"},
            }
        ]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("probe-btc-rtds must not load settings or open SQLite")

    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module.sqlite3, "connect", forbidden)
    monkeypatch.setattr(btc_price_feed_module.websockets, "connect", connector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "probe-btc-rtds",
            "--btc-source",
            POLYMARKET_RTDS_BINANCE_SOURCE,
            "--btc-symbol",
            "btcusdt",
            "--btc-rtds-url",
            "wss://example.test",
            "--max-messages",
            "1",
            "--timeout-sec",
            "1",
        ],
    )

    main_module.main()

    text = capsys.readouterr().out
    assert "subscription_payload=" in text
    assert "connection=ok" in text
    assert "parser_result=" in text
    assert connector.calls[0]["url"] == "wss://example.test"


def test_recorder_init_uses_btc_price_feed_sources_in_live_path(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    monkeypatch.setenv("RECORDER_DB_PATH", str(tmp_path / "recorder.db"))
    monkeypatch.setenv("BTC_PRICE_FEED_ENABLED", "true")
    monkeypatch.setenv(
        "BTC_PRICE_FEED_SOURCES",
        f"{POLYMARKET_RTDS_CHAINLINK_SOURCE},{POLYMARKET_RTDS_BINANCE_SOURCE}",
    )
    monkeypatch.setenv("BTC_PRICE_FEED_SOURCE", POLYMARKET_RTDS_CHAINLINK_SOURCE)
    monkeypatch.setenv("BTC_PRICE_FEED_SYMBOL", "ignored")
    monkeypatch.setenv("BTC_PRICE_FEED_WS_URL", "wss://example.test")
    settings = load_settings(env_file=str(tmp_path / "missing.env"))

    caplog.set_level(logging.INFO)
    app = RecorderApp(settings, run_id="run_multi_source_init")
    try:
        assert set(app.btc_price_feeds) == {
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
        }
        assert app.btc_price_feeds[POLYMARKET_RTDS_CHAINLINK_SOURCE].symbol == "btc/usd"
        assert app.btc_price_feeds[POLYMARKET_RTDS_BINANCE_SOURCE].symbol == "btcusdt"
        assert "btc_price_feed_config_resolved" in [
            record.getMessage() for record in caplog.records
        ]
    finally:
        app.db.close()


def test_recorder_logs_unsupported_btc_feed_source(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    monkeypatch.setenv("RECORDER_DB_PATH", str(tmp_path / "recorder.db"))
    monkeypatch.setenv("BTC_PRICE_FEED_ENABLED", "true")
    monkeypatch.setenv("BTC_PRICE_FEED_SOURCES", "bad_source")
    settings = load_settings(env_file=str(tmp_path / "missing.env"))

    caplog.set_level(logging.WARNING)
    app = RecorderApp(settings, run_id="run_bad_source")
    try:
        assert "bad_source" in app.btc_price_feeds
        assert "btc_price_feed_source_unsupported" in [
            record.getMessage() for record in caplog.records
        ]
    finally:
        app.db.close()


def test_disabled_btc_price_feed_is_not_enabled_even_if_feed_object_exists() -> None:
    app = RecorderApp.__new__(RecorderApp)
    app.settings = SimpleNamespace(btc_price_feed_enabled=False)
    app.btc_price_feed = object()

    assert app._btc_price_feed_enabled() is False


def test_rtds_chainlink_subscription_payload() -> None:
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        symbol="btc/usd",
    )

    assert feed.subscription_payload() == {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": '{"symbol":"btc/usd"}',
            }
        ],
    }


def test_rtds_binance_subscription_payload() -> None:
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
    )

    assert feed.subscription_payload() == {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices",
                "type": "update",
                "filters": '{"symbol":"btcusdt"}',
            }
        ],
    }


def test_rtds_chainlink_btc_usd_message_parses_to_sample() -> None:
    message = {
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": 1_753_314_088_421,
        "payload": {
            "symbol": "btc/usd",
            "timestamp": 1_753_314_088_395,
            "value": 67_234.50,
        },
    }

    sample = parse_polymarket_rtds_btc_price_message(
        message,
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        symbol="btc/usd",
        local_arrival_ns=1_767_225_660_000_000_000,
        local_arrival_iso="2026-01-01T00:01:00+00:00",
    )

    assert sample is not None
    assert sample.source == POLYMARKET_RTDS_CHAINLINK_SOURCE
    assert sample.price == 67_234.50
    assert sample.exchange_timestamp == datetime.fromtimestamp(
        1_753_314_088_395 / 1000,
        tz=timezone.utc,
    )
    assert sample.local_arrival_ns == 1_767_225_660_000_000_000
    assert json.loads(sample.raw_json)["payload"]["symbol"] == "btc/usd"


def test_rtds_binance_btcusdt_message_parses_to_sample() -> None:
    raw_message = json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 1_753_314_088_421,
            "payload": {
                "symbol": "btcusdt",
                "timestamp": 1_753_314_088_395,
                "value": "67234.50",
            },
        }
    )

    sample = parse_polymarket_rtds_btc_price_message(
        raw_message,
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        local_arrival_ns=1_767_225_660_000_000_000,
        local_arrival_iso="2026-01-01T00:01:00+00:00",
    )

    assert sample is not None
    assert sample.source == POLYMARKET_RTDS_BINANCE_SOURCE
    assert sample.price == 67_234.50
    assert sample.exchange_timestamp == datetime.fromtimestamp(
        1_753_314_088_395 / 1000,
        tz=timezone.utc,
    )
    assert json.loads(sample.raw_json)["payload"]["symbol"] == "btcusdt"


def test_rtds_binance_uppercase_btcusdt_message_parses_to_sample() -> None:
    sample = parse_polymarket_rtds_btc_price_message(
        {
            "topic": "crypto_prices",
            "type": "update",
            "payload": {
                "symbol": "BTCUSDT",
                "timestamp": 1_753_314_088_395,
                "value": "67235.25",
            },
        },
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        local_arrival_ns=1_767_225_660_000_000_000,
        local_arrival_iso="2026-01-01T00:01:00+00:00",
    )

    assert sample is not None
    assert sample.source == POLYMARKET_RTDS_BINANCE_SOURCE
    assert sample.price == 67_235.25
    assert sample.exchange_timestamp == datetime.fromtimestamp(
        1_753_314_088_395 / 1000,
        tz=timezone.utc,
    )


def test_rtds_binance_nested_payload_message_parses_to_sample() -> None:
    sample = parse_polymarket_rtds_btc_price_message(
        {
            "topic": "crypto_prices",
            "type": "update",
            "payload": {
                "data": {
                    "s": "BTCUSDT",
                    "E": 1_753_314_088_395,
                    "c": "67236.75",
                },
            },
        },
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        local_arrival_ns=1_767_225_660_000_000_000,
        local_arrival_iso="2026-01-01T00:01:00+00:00",
    )

    assert sample is not None
    assert sample.price == 67_236.75
    assert sample.exchange_timestamp == datetime.fromtimestamp(
        1_753_314_088_395 / 1000,
        tz=timezone.utc,
    )


def test_rtds_binance_ignored_non_btc_message_reports_reason() -> None:
    result = analyze_polymarket_rtds_btc_price_message(
        {
            "topic": "crypto_prices",
            "type": "update",
            "payload": {
                "symbol": "ETHUSDT",
                "timestamp": 1_753_314_088_395,
                "value": "3000.0",
            },
        },
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        local_arrival_ns=1_767_225_660_000_000_000,
        local_arrival_iso="2026-01-01T00:01:00+00:00",
    )

    assert result.sample is None
    assert result.reason == "symbol_mismatch"
    assert result.topic == "crypto_prices"
    assert result.message_type == "update"
    assert result.payload_symbol == "ETHUSDT"
    assert result.matching_topic is True
    assert result.matching_symbol is False


def test_rtds_malformed_message_is_ignored_non_fatally(caplog) -> None:
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        symbol="btc/usd",
    )

    caplog.set_level(logging.DEBUG)
    with patch(
        "src.btc_price_feed.time.time_ns",
        return_value=1_767_225_660_000_000_000,
    ):
        sample = feed._handle_raw_message("not-json")

    assert sample is None
    assert "btc_price_feed_rtds_message_ignored" in [
        record.getMessage() for record in caplog.records
    ]
    assert feed.parse_counters == {
        "raw_messages_seen": 1,
        "matching_topic_messages": 0,
        "matching_symbol_messages": 0,
        "parsed_samples": 0,
        "ignored_messages": 1,
        "parse_errors": 1,
        "stale_messages_dropped": 0,
        "stale_reconnect_requested": 0,
    }


def test_rtds_binance_feed_counters_track_matching_and_parsed_messages() -> None:
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        drop_stale=False,
    )

    with patch(
        "src.btc_price_feed.time.time_ns",
        return_value=1_767_225_660_000_000_000,
    ):
        sample = feed._handle_raw_message(
            {
                "topic": "crypto_prices",
                "type": "update",
                "payload": {
                    "symbol": "BTCUSDT",
                    "timestamp": 1_753_314_088_395,
                    "value": "67234.50",
                },
            }
        )

    assert sample is not None
    assert feed.parse_counters == {
        "raw_messages_seen": 1,
        "matching_topic_messages": 1,
        "matching_symbol_messages": 1,
        "parsed_samples": 1,
        "ignored_messages": 0,
        "parse_errors": 0,
        "stale_messages_dropped": 0,
        "stale_reconnect_requested": 0,
    }


def test_rtds_stale_messages_are_dropped_and_request_reconnect(caplog) -> None:
    now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    stale_timestamp_ms = int((now - timedelta(seconds=60)).timestamp() * 1000)
    message = {
        "topic": "crypto_prices",
        "type": "update",
        "payload": {
            "symbol": "BTCUSDT",
            "timestamp": stale_timestamp_ms,
            "value": "67234.50",
        },
    }
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        max_exchange_age_sec=15.0,
        drop_stale=True,
        stale_message_reconnect_sec=0.05,
        now=lambda: now,
    )

    caplog.set_level(logging.WARNING)
    with patch(
        "src.btc_price_feed.time.time_ns",
        return_value=int(now.timestamp() * 1_000_000_000),
    ), patch(
        "src.btc_price_feed.time.monotonic",
        side_effect=[100.0, 100.1],
    ):
        assert feed._handle_raw_message(message) is None
        try:
            feed._handle_raw_message(message)
        except BTCPriceFeedStaleError:
            pass
        else:
            raise AssertionError("expected stale message reconnect request")

    assert feed.parse_counters["parsed_samples"] == 0
    assert feed.parse_counters["stale_messages_dropped"] == 2
    assert feed.parse_counters["stale_reconnect_requested"] == 1
    assert feed.stale_reconnect_requested_count == 1
    stale_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "btc_price_feed_stale_message_dropped"
    ]
    assert len(stale_logs) == 2
    assert stale_logs[0].source == POLYMARKET_RTDS_BINANCE_SOURCE
    assert stale_logs[0].age_sec > 15.0


def test_btc_stale_check_uses_local_arrival_when_exchange_timestamp_missing() -> None:
    now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        symbol="btc/usd",
        max_exchange_age_sec=15.0,
        now=lambda: now,
    )
    sample = BTCPriceSampleRecord(
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        price=42_000.0,
        exchange_timestamp=None,
        local_arrival_ns=0,
        local_arrival_iso="2026-01-01T00:00:00+00:00",
        raw_json="{}",
    )

    stale = feed._stale_sample_result(sample)

    assert stale is not None
    assert stale["timestamp_source"] == "local_arrival_iso"
    assert stale["age_sec"] == 60.0


def test_binance_rest_fallback_provides_fresh_btc_sample() -> None:
    exchange_ts = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    feed = BinanceRESTBTCPriceFeed(
        fetch_json=lambda: {
            "symbol": "BTCUSDT",
            "lastPrice": "67234.50",
            "closeTime": int(exchange_ts.timestamp() * 1000),
        }
    )

    sample = asyncio.run(
        feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
    )

    assert sample.source == BINANCE_REST_SOURCE
    assert sample.price == 67_234.50
    assert sample.exchange_timestamp == exchange_ts
    assert json.loads(sample.raw_json)["symbol"] == "BTCUSDT"


def test_rtds_ignored_message_debug_log_is_throttled(caplog) -> None:
    feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
    )

    caplog.set_level(logging.DEBUG)
    with patch(
        "src.btc_price_feed.time.time_ns",
        return_value=1_767_225_660_000_000_000,
    ), patch("src.btc_price_feed.time.monotonic", return_value=100.0):
        for _ in range(5):
            feed._handle_raw_message(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "payload": {
                        "symbol": "ETHUSDT",
                        "timestamp": 1_753_314_088_395,
                        "value": "3000.0",
                    },
                }
            )

    ignored_logs = [
        record for record in caplog.records
        if record.getMessage() == "btc_price_feed_rtds_message_ignored"
    ]
    assert len(ignored_logs) == 3
    assert feed.parse_counters["ignored_messages"] == 5


def test_rtds_provider_close_cancels_background_reader(caplog) -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def recv(self) -> str:
            await asyncio.Event().wait()
            return ""

    fake_ws = _FakeWebSocket()

    async def _run() -> None:
        caplog.set_level(logging.INFO)
        feed = PolymarketRTDSBTCPriceFeed(
            source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
            symbol="btc/usd",
            connect=lambda *args, **kwargs: fake_ws,
        )
        await feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
        await asyncio.sleep(0)
        assert feed._reader_task is not None
        assert fake_ws.sent
        assert "btc_price_feed_rtds_subscribing" in [
            record.getMessage() for record in caplog.records
        ]
        await feed.close()
        assert feed._reader_task is None

    asyncio.run(_run())


def test_rtds_stale_watchdog_reconnects_when_no_parsed_samples(caplog) -> None:
    connector = _FakeRTDSConnect([])

    async def _run() -> tuple[int, int]:
        feed = PolymarketRTDSBTCPriceFeed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            reconnect_min_sec=0.01,
            reconnect_max_sec=0.01,
            ping_interval_sec=3600.0,
            stale_reconnect_sec=0.05,
            drop_stale=False,
            connect=connector,
        )
        await feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 1.0
        while len(connector.calls) < 2 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        reconnect_count = feed.reconnect_count
        await feed.close()
        return len(connector.calls), reconnect_count

    caplog.set_level(logging.INFO)
    with patch("src.btc_price_feed.random.uniform", return_value=0.0):
        connect_calls, reconnect_count = asyncio.run(_run())

    messages = [record.getMessage() for record in caplog.records]
    assert connect_calls >= 2
    assert reconnect_count >= 1
    assert "btc_price_feed_stale_detected" in messages
    assert "btc_price_feed_reconnecting" in messages
    assert "btc_price_feed_reconnected" in messages
    stale_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "btc_price_feed_stale_detected"
    )
    assert stale_record.source == POLYMARKET_RTDS_BINANCE_SOURCE
    assert stale_record.symbol == "btcusdt"
    assert stale_record.last_sample_time is None
    assert stale_record.sample_age_sec is not None


def test_rtds_stale_watchdog_reconnects_after_initial_sample_then_silence(
    caplog,
) -> None:
    btc_message = {
        "topic": "crypto_prices",
        "type": "update",
        "payload": {
            "symbol": "BTCUSDT",
            "timestamp": 1_753_314_088_395,
            "value": "67234.50",
        },
    }
    connector = _FakeRTDSConnect([btc_message])

    async def _run() -> tuple[int, int, dict[str, int]]:
        feed = PolymarketRTDSBTCPriceFeed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            reconnect_min_sec=0.01,
            reconnect_max_sec=0.01,
            ping_interval_sec=3600.0,
            stale_reconnect_sec=0.05,
            drop_stale=False,
            connect=connector,
        )
        await feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 1.0
        while len(connector.calls) < 2 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        reconnect_count = feed.reconnect_count
        counters = feed.parse_counters
        await feed.close()
        return len(connector.calls), reconnect_count, counters

    caplog.set_level(logging.INFO)
    with patch("src.btc_price_feed.random.uniform", return_value=0.0):
        connect_calls, reconnect_count, counters = asyncio.run(_run())

    assert counters["parsed_samples"] == 1
    assert connect_calls >= 2
    assert reconnect_count >= 1
    stale_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "btc_price_feed_stale_detected"
    )
    assert stale_record.last_sample_time is not None
    assert stale_record.sample_age_sec is not None


def test_rtds_recv_timeout_while_fresh_does_not_reconnect() -> None:
    connector = _FakeRTDSConnect([])

    async def _run() -> tuple[int, int]:
        feed = PolymarketRTDSBTCPriceFeed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            reconnect_min_sec=0.01,
            reconnect_max_sec=0.01,
            ping_interval_sec=3600.0,
            stale_reconnect_sec=0.2,
            connect=connector,
        )
        await feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
        await asyncio.sleep(0.12)
        reconnect_count = feed.reconnect_count
        await feed.close()
        return len(connector.calls), reconnect_count

    connect_calls, reconnect_count = asyncio.run(_run())

    assert connect_calls == 1
    assert reconnect_count == 0


def test_rtds_fresh_parsed_samples_prevent_reconnect() -> None:
    btc_message = {
        "topic": "crypto_prices",
        "type": "update",
        "payload": {
            "symbol": "BTCUSDT",
            "timestamp": 1_753_314_088_395,
            "value": "67234.50",
        },
    }
    connector = _DelayedRTDSConnect(
        [
            (0.01, btc_message),
            (0.02, btc_message),
            (0.02, btc_message),
            (0.02, btc_message),
            (0.02, btc_message),
        ]
    )

    async def _run() -> tuple[int, dict[str, int]]:
        feed = PolymarketRTDSBTCPriceFeed(
            source=POLYMARKET_RTDS_BINANCE_SOURCE,
            symbol="btcusdt",
            reconnect_min_sec=0.01,
            reconnect_max_sec=0.01,
            ping_interval_sec=3600.0,
            stale_reconnect_sec=0.08,
            drop_stale=False,
            connect=connector,
        )
        await feed.sample(
            local_arrival_ns=1_767_225_660_000_000_000,
            local_arrival_iso="2026-01-01T00:01:00+00:00",
        )
        await asyncio.sleep(0.13)
        counters = feed.parse_counters
        await feed.close()
        return len(connector.calls), counters

    connect_calls, counters = asyncio.run(_run())

    assert connect_calls == 1
    assert counters["parsed_samples"] >= 4


def test_mock_btc_price_feed_sample_is_inserted(tmp_path) -> None:
    exchange_ts = datetime(2026, 1, 1, 0, 0, 59, tzinfo=timezone.utc)
    feed = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source="mock",
                price=42123.45,
                exchange_timestamp=exchange_ts,
                raw_json=json.dumps({"symbol": "btc/usd", "price": 42123.45}),
            )
        ]
    )
    app = _make_app(tmp_path, feed)
    try:
        with patch("src.recorder.time.time_ns", return_value=1_767_225_660_000_000_000):
            inserted = asyncio.run(app._run_btc_price_feed_cycle())

        assert inserted == 1
        row = app.db.conn.execute(
            """
            SELECT run_id, source, price, exchange_timestamp, local_arrival_ns,
                   local_arrival_iso, raw_json
            FROM btc_prices
            WHERE run_id = ?
            """,
            (app.run_id,),
        ).fetchone()
        assert row[0] == app.run_id
        assert row[1] == "mock"
        assert row[2] == 42123.45
        assert row[3] == "2026-01-01T00:00:59+00:00"
        assert row[4] == 1_767_225_660_000_000_000
        assert row[5] == "2026-01-01T00:01:00+00:00"
        assert json.loads(row[6])["symbol"] == "btc/usd"
    finally:
        app.db.close()


def test_btc_price_feed_fresh_sample_passes_stale_guard(tmp_path) -> None:
    feed = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                price=50_001.0,
                exchange_timestamp=datetime.now(timezone.utc),
                raw_json=json.dumps({"source": "chainlink"}),
            )
        ]
    )
    app = _make_app(tmp_path, feed)
    app.btc_price_feeds = {POLYMARKET_RTDS_CHAINLINK_SOURCE: feed}
    app.settings = _settings(
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_drop_stale=True,
        btc_price_max_exchange_age_sec=15.0,
    )
    try:
        inserted = asyncio.run(
            app._run_btc_price_feed_cycle(source=POLYMARKET_RTDS_CHAINLINK_SOURCE)
        )

        assert inserted == 1
        count = app.db.conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        assert count == 1
        assert (
            app._btc_feed_health[POLYMARKET_RTDS_CHAINLINK_SOURCE][
                "stale_messages_dropped"
            ]
            == 0
        )
    finally:
        app.db.close()


def test_btc_price_feed_stale_sample_is_dropped_before_insert(tmp_path, caplog) -> None:
    feed = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                price=50_001.0,
                exchange_timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
                raw_json=json.dumps({"source": "chainlink"}),
            )
        ]
    )
    app = _make_app(tmp_path, feed)
    app.btc_price_feeds = {POLYMARKET_RTDS_CHAINLINK_SOURCE: feed}
    app.settings = _settings(
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_drop_stale=True,
        btc_price_max_exchange_age_sec=15.0,
    )
    try:
        caplog.set_level(logging.WARNING)
        inserted = asyncio.run(
            app._run_btc_price_feed_cycle(source=POLYMARKET_RTDS_CHAINLINK_SOURCE)
        )

        assert inserted == 0
        count = app.db.conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        assert count == 0
        health = app._btc_feed_health[POLYMARKET_RTDS_CHAINLINK_SOURCE]
        assert health["stale_messages_dropped"] == 1
        assert health["latest_stale_age_sec"] > 15.0
        assert "btc_price_sample_stale_dropped" in [
            record.getMessage() for record in caplog.records
        ]
    finally:
        app.db.close()


def test_btc_price_feed_sample_timeout_does_not_block_recorder(tmp_path, caplog) -> None:
    class _BlockingFeed:
        source = "blocking"

        async def sample(self, *, local_arrival_ns: int, local_arrival_iso: str):
            _ = (local_arrival_ns, local_arrival_iso)
            await asyncio.sleep(3600)

        async def close(self) -> None:
            return None

    app = _make_app(tmp_path, _BlockingFeed())
    app.btc_price_feeds = {"blocking": app.btc_price_feed}
    app.settings = _settings(
        btc_price_feed_source="blocking",
        btc_price_feed_sample_timeout_sec=0.01,
    )
    try:
        caplog.set_level(logging.ERROR)
        inserted = asyncio.run(app._run_btc_price_feed_cycle(source="blocking"))

        assert inserted == 0
        health = app._btc_feed_health["blocking"]
        assert health["error_count"] == 1
        assert "timed out" in health["last_error"]
        assert "btc_price_feed_sample_error" in [
            record.getMessage() for record in caplog.records
        ]
    finally:
        app.db.close()


def test_btc_price_feed_startup_timeout_logs_warning_without_stopping(tmp_path, caplog) -> None:
    app = _make_app(tmp_path, MockBTCPriceFeed())
    app.settings = _settings(
        btc_price_feed_startup_timeout_sec=0.01,
        btc_price_feed_sample_timeout_sec=0.01,
    )
    state = app._btc_feed_health_state("mock")
    state["task_started_monotonic"] = time.monotonic() - 1.0
    try:
        caplog.set_level(logging.WARNING)
        app._maybe_log_btc_price_feed_startup_timeout("mock")

        assert state["startup_warning_logged"] is True
        assert "btc_price_feed_startup_unhealthy_timeout" in [
            record.getMessage() for record in caplog.records
        ]
        assert app._btc_price_feed_heartbeat_fields()["btc_price_feed_enabled"] is True
    finally:
        app.db.close()


def test_all_stale_btc_sources_are_unhealthy_but_non_fatal(tmp_path) -> None:
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
    chainlink = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                price=50_001.0,
                exchange_timestamp=stale_ts,
            )
        ]
    )
    binance = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_BINANCE_SOURCE,
                price=50_002.0,
                exchange_timestamp=stale_ts,
            )
        ]
    )
    app = _make_app(tmp_path, chainlink)
    app.btc_price_feeds = {
        POLYMARKET_RTDS_CHAINLINK_SOURCE: chainlink,
        POLYMARKET_RTDS_BINANCE_SOURCE: binance,
    }
    app.settings = _settings(
        btc_price_feed_sources=(
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
        ),
        btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        btc_price_drop_stale=True,
        btc_price_max_exchange_age_sec=15.0,
    )
    try:
        inserted_chainlink = asyncio.run(
            app._run_btc_price_feed_cycle(source=POLYMARKET_RTDS_CHAINLINK_SOURCE)
        )
        inserted_binance = asyncio.run(
            app._run_btc_price_feed_cycle(source=POLYMARKET_RTDS_BINANCE_SOURCE)
        )

        assert inserted_chainlink == 0
        assert inserted_binance == 0
        heartbeat = app._btc_price_feed_heartbeat_fields()
        assert heartbeat["btc_price_feed_enabled"] is True
        assert heartbeat["btc_price_feed_healthy"] is False
        assert heartbeat["btc_price_stale_rows_dropped_by_source"] == {
            POLYMARKET_RTDS_BINANCE_SOURCE: 1,
            POLYMARKET_RTDS_CHAINLINK_SOURCE: 1,
        }
    finally:
        app.db.close()


def test_multi_source_btc_feeds_insert_samples(tmp_path) -> None:
    chainlink = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                price=50_001.0,
                raw_json=json.dumps({"source": "chainlink"}),
            )
        ]
    )
    binance = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_BINANCE_SOURCE,
                price=50_003.0,
                raw_json=json.dumps({"source": "binance"}),
            )
        ]
    )
    app = _make_app(tmp_path, chainlink)
    app.btc_price_feeds = {
        POLYMARKET_RTDS_CHAINLINK_SOURCE: chainlink,
        POLYMARKET_RTDS_BINANCE_SOURCE: binance,
    }
    try:
        with patch(
            "src.recorder.time.time_ns",
            side_effect=[
                1_767_225_660_000_000_000,
                1_767_225_661_000_000_000,
            ],
        ):
            inserted_chainlink = asyncio.run(
                app._run_btc_price_feed_cycle(
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE
                )
            )
            inserted_binance = asyncio.run(
                app._run_btc_price_feed_cycle(
                    source=POLYMARKET_RTDS_BINANCE_SOURCE
                )
            )

        assert inserted_chainlink == 1
        assert inserted_binance == 1
        rows = app.db.conn.execute(
            """
            SELECT source, price
            FROM btc_prices
            ORDER BY local_arrival_ns ASC
            """
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [
            (POLYMARKET_RTDS_CHAINLINK_SOURCE, 50_001.0),
            (POLYMARKET_RTDS_BINANCE_SOURCE, 50_003.0),
        ]
        assert app._btc_feed_health[POLYMARKET_RTDS_CHAINLINK_SOURCE]["sample_count"] == 1
        assert app._btc_feed_health[POLYMARKET_RTDS_BINANCE_SOURCE]["sample_count"] == 1
    finally:
        app.db.close()


def test_one_btc_feed_failure_does_not_stop_other_source(tmp_path, caplog) -> None:
    class _FailingFeed:
        async def sample(self, *, local_arrival_ns: int, local_arrival_iso: str):
            _ = (local_arrival_ns, local_arrival_iso)
            raise RuntimeError("chainlink stalled")

        async def close(self) -> None:
            return None

    binance = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_BINANCE_SOURCE,
                price=50_003.0,
                raw_json=json.dumps({"source": "binance"}),
            )
        ]
    )
    app = _make_app(tmp_path, _FailingFeed())
    app.btc_price_feeds = {
        POLYMARKET_RTDS_CHAINLINK_SOURCE: _FailingFeed(),
        POLYMARKET_RTDS_BINANCE_SOURCE: binance,
    }
    try:
        caplog.set_level(logging.ERROR)
        with patch(
            "src.recorder.time.time_ns",
            return_value=1_767_225_661_000_000_000,
        ):
            failed_inserted = asyncio.run(
                app._run_btc_price_feed_cycle(
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE
                )
            )
            good_inserted = asyncio.run(
                app._run_btc_price_feed_cycle(
                    source=POLYMARKET_RTDS_BINANCE_SOURCE
                )
            )

        assert failed_inserted == 0
        assert good_inserted == 1
        assert app._btc_feed_health[POLYMARKET_RTDS_CHAINLINK_SOURCE]["error_count"] == 1
        assert app._btc_feed_health[POLYMARKET_RTDS_BINANCE_SOURCE]["sample_count"] == 1
        assert "btc_price_feed_sample_error" in [
            record.getMessage() for record in caplog.records
        ]
        count = app.db.conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        assert count == 1
    finally:
        app.db.close()


def test_one_stale_rtds_feed_does_not_stop_other_source(tmp_path) -> None:
    stale_connector = _FakeRTDSConnect([])
    stale_feed = PolymarketRTDSBTCPriceFeed(
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        symbol="btcusdt",
        reconnect_min_sec=0.01,
        reconnect_max_sec=0.01,
        ping_interval_sec=3600.0,
        stale_reconnect_sec=0.05,
        connect=stale_connector,
    )
    good_feed = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                price=50_001.0,
                raw_json=json.dumps({"source": "chainlink"}),
            )
        ]
    )
    app = _make_app(tmp_path, good_feed)
    app.btc_price_feeds = {
        POLYMARKET_RTDS_BINANCE_SOURCE: stale_feed,
        POLYMARKET_RTDS_CHAINLINK_SOURCE: good_feed,
    }
    try:
        async def _run() -> tuple[int, int, int]:
            stale_inserted = await app._run_btc_price_feed_cycle(
                source=POLYMARKET_RTDS_BINANCE_SOURCE
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 1.0
            while len(stale_connector.calls) < 2 and loop.time() < deadline:
                await asyncio.sleep(0.01)
            with patch(
                "src.recorder.time.time_ns",
                return_value=1_767_225_661_000_000_000,
            ):
                good_inserted = await app._run_btc_price_feed_cycle(
                    source=POLYMARKET_RTDS_CHAINLINK_SOURCE
                )
            await stale_feed.close()
            return stale_inserted, good_inserted, len(stale_connector.calls)

        with patch("src.btc_price_feed.random.uniform", return_value=0.0):
            stale_inserted, good_inserted, stale_connect_calls = asyncio.run(_run())

        assert stale_inserted == 0
        assert good_inserted == 1
        assert stale_connect_calls >= 2
        count = app.db.conn.execute(
            "SELECT COUNT(*) FROM btc_prices WHERE source = ?",
            (POLYMARKET_RTDS_CHAINLINK_SOURCE,),
        ).fetchone()[0]
        assert count == 1
    finally:
        app.db.close()


def test_create_runtime_tasks_starts_one_btc_task_per_source(caplog) -> None:
    async def _idle() -> None:
        await asyncio.Event().wait()

    async def _run() -> list[str]:
        app = RecorderApp.__new__(RecorderApp)
        app.log = logging.getLogger("test.recorder.btc_price_tasks")
        app.settings = _settings(
            btc_price_feed_sources_raw=(
                f"{POLYMARKET_RTDS_CHAINLINK_SOURCE},{POLYMARKET_RTDS_BINANCE_SOURCE}"
            ),
            btc_price_feed_sources=(
                POLYMARKET_RTDS_CHAINLINK_SOURCE,
                POLYMARKET_RTDS_BINANCE_SOURCE,
            ),
            btc_price_feed_source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        )
        app._discovery_loop = _idle
        app._snapshot_loop = _idle
        app._ws_loop = _idle
        app._trade_backfill_loop = _idle
        app._btc_price_feed_loop = lambda source=None: _idle()
        app._btc_price_feed_enabled = lambda: True
        app._auto_normalize_resolutions_enabled = lambda: False
        app._auto_repair_stale_active_enabled = lambda: False
        app.btc_price_feeds = {
            POLYMARKET_RTDS_CHAINLINK_SOURCE: MockBTCPriceFeed(),
            POLYMARKET_RTDS_BINANCE_SOURCE: MockBTCPriceFeed(),
        }
        caplog.set_level(logging.INFO)
        tasks = app._create_runtime_tasks()
        names = [task.get_name() for task in tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return names

    names = asyncio.run(_run())

    assert f"btc_price_feed_loop:{POLYMARKET_RTDS_CHAINLINK_SOURCE}" in names
    assert f"btc_price_feed_loop:{POLYMARKET_RTDS_BINANCE_SOURCE}" in names
    assert [record.getMessage() for record in caplog.records].count(
        "btc_price_feed_task_created"
    ) == 2


def test_btc_feed_loop_logs_task_started() -> None:
    async def _run() -> list[str]:
        app = RecorderApp.__new__(RecorderApp)
        app.log = logging.getLogger("test.recorder.btc_price_loop")
        app.settings = _settings(
            btc_price_feed_sources=(POLYMARKET_RTDS_BINANCE_SOURCE,),
            btc_price_feed_source=POLYMARKET_RTDS_BINANCE_SOURCE,
        )
        app.btc_price_feeds = {POLYMARKET_RTDS_BINANCE_SOURCE: MockBTCPriceFeed()}
        app.btc_price_feed = app.btc_price_feeds[POLYMARKET_RTDS_BINANCE_SOURCE]
        app.stop_event = asyncio.Event()
        app.stop_event.set()
        app._btc_feed_health = {}
        records: list[str] = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _ListHandler()
        app.log.addHandler(handler)
        app.log.setLevel(logging.INFO)
        try:
            await app._btc_price_feed_loop(source=POLYMARKET_RTDS_BINANCE_SOURCE)
        finally:
            app.log.removeHandler(handler)
        return records

    messages = asyncio.run(_run())

    assert "btc_price_feed_task_started" in messages


def test_btc_price_feed_failure_is_logged_and_non_fatal(tmp_path, caplog) -> None:
    class _FailingFeed:
        async def sample(self, *, local_arrival_ns: int, local_arrival_iso: str):
            _ = (local_arrival_ns, local_arrival_iso)
            raise RuntimeError("offline provider failure")

        async def close(self) -> None:
            return None

    app = _make_app(tmp_path, _FailingFeed())
    try:
        caplog.set_level(logging.ERROR)
        inserted = asyncio.run(app._run_btc_price_feed_cycle())

        assert inserted == 0
        assert "btc_price_feed_sample_error" in [
            record.getMessage() for record in caplog.records
        ]
        count = app.db.conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        assert count == 0
    finally:
        app.db.close()


def test_diagnostics_shows_latest_btc_price_after_mock_sample(tmp_path, capsys) -> None:
    feed = MockBTCPriceFeed(
        [
            MockBTCPriceFeedSample(
                source="mock",
                price=42123.45,
                raw_json=json.dumps({"price": 42123.45}),
            )
        ]
    )
    app = _make_app(tmp_path, feed)
    try:
        with patch("src.recorder.time.time_ns", return_value=1_767_225_660_000_000_000):
            asyncio.run(app._run_btc_price_feed_cycle())

        print_diagnostics(app.db.db_path)
        out = capsys.readouterr().out
        assert "btc_prices=1" in out
        assert "=== Latest BTC Price ===" in out
        assert "source=mock" in out
        assert "price=42123.45" in out
    finally:
        app.db.close()
