from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Protocol, Sequence, TextIO

import websockets

from .models import (
    BTCPriceSampleRecord,
    local_arrival_iso_from_ns,
    parse_exchange_timestamp,
    parse_timestamp,
)


POLYMARKET_RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
POLYMARKET_RTDS_CHAINLINK_SOURCE = "polymarket_rtds_chainlink"
POLYMARKET_RTDS_BINANCE_SOURCE = "polymarket_rtds_binance"
BINANCE_REST_SOURCE = "binance_rest"
BINANCE_REST_URL = "https://api.binance.com/api/v3/ticker/24hr"
SUPPORTED_BTC_PRICE_FEED_SOURCES = {
    "mock",
    BINANCE_REST_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
    POLYMARKET_RTDS_BINANCE_SOURCE,
}

_RTDS_CHAINLINK_TOPIC = "crypto_prices_chainlink"
_RTDS_BINANCE_TOPIC = "crypto_prices"
_RTDS_PRICE_TYPE = "update"


class BTCPriceFeedError(RuntimeError):
    pass


class BTCPriceFeedStaleError(BTCPriceFeedError):
    pass


class BTCPriceFeed(Protocol):
    async def sample(
        self,
        *,
        local_arrival_ns: int,
        local_arrival_iso: str,
    ) -> BTCPriceSampleRecord | None:
        ...

    async def close(self) -> None:
        ...


@dataclass(slots=True, frozen=True)
class MockBTCPriceFeedSample:
    price: float
    source: str = "mock"
    exchange_timestamp: datetime | None = None
    raw_json: str = "{}"


@dataclass(slots=True, frozen=True)
class RTDSBTCParseResult:
    sample: BTCPriceSampleRecord | None
    reason: str | None
    topic: str | None
    message_type: str | None
    payload_symbol: str | None
    matching_topic: bool = False
    matching_symbol: bool = False
    parse_error: str | None = None


class MockBTCPriceFeed:
    def __init__(
        self,
        samples: Sequence[MockBTCPriceFeedSample] = (),
        *,
        cycle: bool = False,
    ) -> None:
        self._samples = list(samples)
        self._cycle = bool(cycle)
        self._index = 0

    async def sample(
        self,
        *,
        local_arrival_ns: int,
        local_arrival_iso: str,
    ) -> BTCPriceSampleRecord | None:
        if not self._samples:
            return None
        if self._index >= len(self._samples):
            if not self._cycle:
                return None
            self._index = 0
        sample = self._samples[self._index]
        self._index += 1
        return BTCPriceSampleRecord(
            source=sample.source,
            price=float(sample.price),
            exchange_timestamp=sample.exchange_timestamp,
            local_arrival_ns=int(local_arrival_ns),
            local_arrival_iso=local_arrival_iso,
            raw_json=sample.raw_json,
        )

    async def close(self) -> None:
        return None


class BinanceRESTBTCPriceFeed:
    """Simple public Binance REST poller used as an independent BTC fallback."""

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        url: str = BINANCE_REST_URL,
        timeout_sec: float = 2.0,
        fetch_json: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.source = BINANCE_REST_SOURCE
        self.symbol = symbol.upper()
        self.url = url
        self.timeout_sec = max(0.1, float(timeout_sec or 2.0))
        self._fetch_json_override = fetch_json

    async def sample(
        self,
        *,
        local_arrival_ns: int,
        local_arrival_iso: str,
    ) -> BTCPriceSampleRecord | None:
        payload = await asyncio.to_thread(self._fetch_json)
        price = self._extract_price(payload)
        if price is None:
            raise BTCPriceFeedError("Binance REST BTC response did not include price")
        exchange_timestamp = parse_exchange_timestamp(
            payload.get("closeTime")
            or payload.get("time")
            or payload.get("eventTime")
            or payload.get("E")
        )
        return BTCPriceSampleRecord(
            source=self.source,
            price=float(price),
            exchange_timestamp=exchange_timestamp,
            local_arrival_ns=int(local_arrival_ns),
            local_arrival_iso=local_arrival_iso,
            raw_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    async def close(self) -> None:
        return None

    def _fetch_json(self) -> dict[str, Any]:
        if self._fetch_json_override is not None:
            return dict(self._fetch_json_override())
        separator = "&" if "?" in self.url else "?"
        url = f"{self.url}{separator}symbol={self.symbol}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "polymarket-bot-btc-price-feed/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise BTCPriceFeedError(f"Binance REST BTC request failed: {exc}") from exc
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BTCPriceFeedError("Binance REST BTC response was not JSON") from exc
        if not isinstance(payload, dict):
            raise BTCPriceFeedError("Binance REST BTC response was not an object")
        return payload

    @staticmethod
    def _extract_price(payload: dict[str, Any]) -> float | None:
        for key in ("lastPrice", "price", "weightedAvgPrice", "bidPrice", "askPrice"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None


def _coerce_json_payload(
    raw_message: str | bytes | dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if isinstance(raw_message, bytes):
        raw_text = raw_message.decode("utf-8", errors="replace")
    elif isinstance(raw_message, str):
        raw_text = raw_message
    elif isinstance(raw_message, dict):
        raw_text = json.dumps(raw_message, sort_keys=True, separators=(",", ":"))
        return raw_message, raw_text
    else:
        raw_text = json.dumps(raw_message, sort_keys=True, separators=(",", ":"))

    try:
        payload = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return None, raw_text
    if not isinstance(payload, dict):
        return None, raw_text
    return payload, raw_text


def _normalise_symbol(symbol: object) -> str:
    return "".join(
        char for char in str(symbol or "").strip().lower() if char.isalnum()
    )


def _extract_rtds_price(payload: dict[str, Any]) -> float | None:
    for key in ("value", "price", "full_accuracy_value", "c", "p", "last_price"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_rtds_symbol(payload: dict[str, Any]) -> str | None:
    for key in ("symbol", "s", "pair", "ticker"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _extract_rtds_timestamp(
    payload: dict[str, Any],
    message: dict[str, Any],
) -> Any:
    for key in ("timestamp", "time", "event_time", "E"):
        value = payload.get(key)
        if value is not None:
            return value
    return message.get("timestamp")


def extract_rtds_debug_fields(
    raw_message: str | bytes | dict[str, Any],
) -> dict[str, Any]:
    message, raw_text = _coerce_json_payload(raw_message)
    if message is None:
        return {
            "json_object": False,
            "raw_length": len(raw_text),
            "topic": None,
            "type": None,
            "message_type": None,
            "symbol_candidates": [],
            "price_candidates": [],
            "timestamp_candidates": [],
        }

    candidates = _iter_rtds_price_payloads(message.get("payload"))
    symbol_candidates: list[str] = []
    price_candidates: list[dict[str, Any]] = []
    timestamp_candidates: list[Any] = []
    for candidate in candidates:
        symbol = _extract_rtds_symbol(candidate)
        if symbol is not None:
            symbol_candidates.append(symbol)
        for key in ("value", "price", "full_accuracy_value", "c", "p", "last_price"):
            if candidate.get(key) is not None:
                price_candidates.append({key: candidate.get(key)})
        timestamp = _extract_rtds_timestamp(candidate, message)
        if timestamp is not None:
            timestamp_candidates.append(timestamp)
    if message.get("timestamp") is not None:
        timestamp_candidates.append(message.get("timestamp"))

    return {
        "json_object": True,
        "topic": message.get("topic"),
        "type": message.get("type"),
        "message_type": message.get("message_type", message.get("type")),
        "symbol_candidates": symbol_candidates,
        "price_candidates": price_candidates,
        "timestamp_candidates": timestamp_candidates,
    }


def _iter_rtds_price_payloads(payload: object) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(value: object) -> None:
        if isinstance(value, dict):
            candidates.append(value)
            for key in ("payload", "data", "price", "prices", "result"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested)
                elif isinstance(nested, list):
                    candidates.extend(item for item in nested if isinstance(item, dict))
        elif isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))

    add(payload)
    return candidates


def _expected_rtds_topic(source: str) -> str | None:
    if source == POLYMARKET_RTDS_CHAINLINK_SOURCE:
        return _RTDS_CHAINLINK_TOPIC
    if source == POLYMARKET_RTDS_BINANCE_SOURCE:
        return _RTDS_BINANCE_TOPIC
    return None


def parse_polymarket_rtds_btc_price_message(
    raw_message: str | bytes | dict[str, Any],
    *,
    source: str,
    symbol: str,
    local_arrival_ns: int,
    local_arrival_iso: str | None = None,
) -> BTCPriceSampleRecord | None:
    """Parse one Polymarket RTDS crypto price message into a BTC sample."""
    return analyze_polymarket_rtds_btc_price_message(
        raw_message,
        source=source,
        symbol=symbol,
        local_arrival_ns=local_arrival_ns,
        local_arrival_iso=local_arrival_iso,
    ).sample


def analyze_polymarket_rtds_btc_price_message(
    raw_message: str | bytes | dict[str, Any],
    *,
    source: str,
    symbol: str,
    local_arrival_ns: int,
    local_arrival_iso: str | None = None,
) -> RTDSBTCParseResult:
    message, raw_json = _coerce_json_payload(raw_message)
    if message is None:
        return RTDSBTCParseResult(
            sample=None,
            reason="invalid_json",
            topic=None,
            message_type=None,
            payload_symbol=None,
            parse_error="message is not a JSON object",
        )

    expected_topic = _expected_rtds_topic(source)
    if expected_topic is not None and message.get("topic") != expected_topic:
        return RTDSBTCParseResult(
            sample=None,
            reason="topic_mismatch",
            topic=str(message.get("topic") or ""),
            message_type=str(message.get("type") or ""),
            payload_symbol=None,
            matching_topic=False,
        )
    message_type = message.get("type", message.get("message_type"))
    if message_type not in {_RTDS_PRICE_TYPE, "subscribe"}:
        return RTDSBTCParseResult(
            sample=None,
            reason="type_mismatch",
            topic=str(message.get("topic") or ""),
            message_type=str(message_type or ""),
            payload_symbol=None,
            matching_topic=True,
        )

    payload = message.get("payload")
    candidates = _iter_rtds_price_payloads(payload)
    if not candidates:
        return RTDSBTCParseResult(
            sample=None,
            reason="missing_payload",
            topic=str(message.get("topic") or ""),
            message_type=str(message_type or ""),
            payload_symbol=None,
            matching_topic=True,
        )

    expected_symbol = _normalise_symbol(symbol)
    first_symbol: str | None = None
    saw_expected_symbol = False
    missing_value_for_symbol = False
    for candidate in candidates:
        actual_symbol_raw = _extract_rtds_symbol(candidate)
        if first_symbol is None and actual_symbol_raw is not None:
            first_symbol = actual_symbol_raw
        actual_symbol = _normalise_symbol(actual_symbol_raw)
        if expected_symbol and not actual_symbol:
            continue
        if expected_symbol and actual_symbol != expected_symbol:
            continue
        saw_expected_symbol = True
        price = _extract_rtds_price(candidate)
        if price is None:
            missing_value_for_symbol = True
            continue

        exchange_timestamp = parse_exchange_timestamp(
            _extract_rtds_timestamp(candidate, message)
        )
        return RTDSBTCParseResult(
            sample=BTCPriceSampleRecord(
                source=source,
                price=price,
                exchange_timestamp=exchange_timestamp,
                local_arrival_ns=int(local_arrival_ns),
                local_arrival_iso=(
                    local_arrival_iso or local_arrival_iso_from_ns(local_arrival_ns)
                ),
                raw_json=raw_json,
            ),
            reason=None,
            topic=str(message.get("topic") or ""),
            message_type=str(message_type or ""),
            payload_symbol=actual_symbol_raw,
            matching_topic=True,
            matching_symbol=True,
        )

    reason = "missing_value" if missing_value_for_symbol else "symbol_mismatch"
    if first_symbol is None:
        reason = "missing_symbol"
    return RTDSBTCParseResult(
        sample=None,
        reason=reason,
        topic=str(message.get("topic") or ""),
        message_type=str(message_type or ""),
        payload_symbol=first_symbol,
        matching_topic=True,
        matching_symbol=saw_expected_symbol,
    )


class PolymarketRTDSBTCPriceFeed:
    """Background RTDS reader that exposes parsed samples through sample()."""

    def __init__(
        self,
        *,
        source: str,
        symbol: str,
        ws_url: str = POLYMARKET_RTDS_WS_URL,
        reconnect_min_sec: float = 1.0,
        reconnect_max_sec: float = 30.0,
        ping_interval_sec: float = 5.0,
        stale_reconnect_sec: float = 10.0,
        max_exchange_age_sec: float = 15.0,
        drop_stale: bool = True,
        stale_message_reconnect_sec: float = 30.0,
        max_pending_samples: int = 1000,
        connect: Callable[..., Any] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.source = source
        self.symbol = symbol
        self.ws_url = ws_url
        self.reconnect_min_sec = reconnect_min_sec
        self.reconnect_max_sec = reconnect_max_sec
        self.ping_interval_sec = ping_interval_sec
        self.stale_reconnect_sec = max(0.01, float(stale_reconnect_sec or 10.0))
        self.max_exchange_age_sec = max(0.0, float(max_exchange_age_sec or 0.0))
        self.drop_stale = bool(drop_stale)
        self.stale_message_reconnect_sec = max(
            0.01,
            float(stale_message_reconnect_sec or 30.0),
        )
        self.reconnect_count = 0
        self.stale_reconnect_requested_count = 0
        self._connect = connect or websockets.connect
        self._sleep = sleep or asyncio.sleep
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._samples: Deque[BTCPriceSampleRecord] = deque(
            maxlen=max_pending_samples
        )
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False
        self._log = logging.getLogger(self.__class__.__name__)
        self._parse_counters: dict[str, int] = {
            "raw_messages_seen": 0,
            "matching_topic_messages": 0,
            "matching_symbol_messages": 0,
            "parsed_samples": 0,
            "ignored_messages": 0,
            "parse_errors": 0,
            "stale_messages_dropped": 0,
            "stale_reconnect_requested": 0,
        }
        self._ignored_log_counts: dict[str, int] = {}
        self._ignored_log_last_monotonic: dict[str, float] = {}
        self._current_connection_started_monotonic: float | None = None
        self._connection_id = 0
        self._last_sample_monotonic: float | None = None
        self._last_sample_local_arrival_iso: str | None = None
        self._stale_message_streak_started_monotonic: float | None = None

    @property
    def parse_counters(self) -> dict[str, int]:
        return dict(self._parse_counters)

    def subscription_payload(self, *, action: str = "subscribe") -> dict[str, object]:
        if self.source == POLYMARKET_RTDS_CHAINLINK_SOURCE:
            filters = json.dumps({"symbol": self.symbol}, separators=(",", ":"))
            return {
                "action": action,
                "subscriptions": [
                    {
                        "topic": _RTDS_CHAINLINK_TOPIC,
                        "type": "*",
                        "filters": filters,
                    }
                ],
            }
        if self.source == POLYMARKET_RTDS_BINANCE_SOURCE:
            filters = json.dumps({"symbol": self.symbol}, separators=(",", ":"))
            return {
                "action": action,
                "subscriptions": [
                    {
                        "topic": _RTDS_BINANCE_TOPIC,
                        "type": _RTDS_PRICE_TYPE,
                        "filters": filters,
                    }
                ],
            }
        raise BTCPriceFeedError(
            f"Unsupported Polymarket RTDS BTC source: {self.source}"
        )

    async def sample(
        self,
        *,
        local_arrival_ns: int,
        local_arrival_iso: str,
    ) -> BTCPriceSampleRecord | None:
        _ = (local_arrival_ns, local_arrival_iso)
        self._ensure_reader_started()
        if not self._samples:
            return None
        return self._samples.popleft()

    async def close(self) -> None:
        self._closed = True
        task = self._reader_task
        if task is None:
            return None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._reader_task = None
        return None

    def _ensure_reader_started(self) -> None:
        if self._closed:
            return
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.create_task(
                self._run_forever(),
                name="polymarket_rtds_btc_price_feed",
            )

    async def _run_forever(self) -> None:
        backoff = self.reconnect_min_sec
        has_connected_once = False
        while not self._closed:
            try:
                async with self._connect(
                    self.ws_url,
                    max_queue=None,
                    max_size=None,
                ) as ws:
                    if has_connected_once:
                        self.reconnect_count += 1
                    self._current_connection_started_monotonic = time.monotonic()
                    self._connection_id += 1
                    self._stale_message_streak_started_monotonic = None
                    subscription_payload = self.subscription_payload()
                    self._log.info(
                        "btc_price_feed_rtds_subscribing",
                        extra={
                            "source": self.source,
                            "symbol": self.symbol,
                            "ws_url": self.ws_url,
                            "connection_id": self._connection_id,
                            "subscription_payload": subscription_payload,
                        },
                    )
                    await ws.send(json.dumps(subscription_payload))
                    self._log.info(
                        "btc_price_feed_rtds_connected",
                        extra={
                            "source": self.source,
                            "symbol": self.symbol,
                            "ws_url": self.ws_url,
                            "reconnect_count": self.reconnect_count,
                            "connection_id": self._connection_id,
                        },
                    )
                    if has_connected_once:
                        self._log.info(
                            "btc_price_feed_reconnected",
                            extra=self._stale_watchdog_log_extra(),
                        )
                    has_connected_once = True
                    backoff = self.reconnect_min_sec
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        while not self._closed:
                            self._raise_if_stale()
                            raw_message = await self._recv_next_message(ws)
                            if raw_message is None:
                                continue
                            self._handle_raw_message(raw_message)
                            self._raise_if_stale()
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except BTCPriceFeedStaleError as exc:
                if self._closed:
                    return
                wait_for = self._next_reconnect_delay(backoff)
                self._log.warning(
                    "btc_price_feed_reconnecting",
                    extra={
                        **self._stale_watchdog_log_extra(),
                        "reason": "stale",
                        "error": str(exc),
                        "reconnect_in_sec": round(wait_for, 3),
                    },
                )
                await self._sleep(wait_for)
                backoff = self._next_backoff(backoff)
            except Exception as exc:
                if self._closed:
                    return
                wait_for = self._next_reconnect_delay(backoff)
                self._log.warning(
                    "btc_price_feed_rtds_disconnected",
                    extra={
                        "source": self.source,
                        "symbol": self.symbol,
                        "error": str(exc),
                        "reconnect_in_sec": round(wait_for, 3),
                        "reconnect_count": self.reconnect_count,
                    },
                    exc_info=True,
                )
                self._log.warning(
                    "btc_price_feed_reconnect_failed",
                    extra={
                        **self._stale_watchdog_log_extra(),
                        "error": str(exc),
                        "reconnect_in_sec": round(wait_for, 3),
                    },
                    exc_info=True,
                )
                self._log.warning(
                    "btc_price_feed_reconnecting",
                    extra={
                        **self._stale_watchdog_log_extra(),
                        "reason": "error",
                        "error": str(exc),
                        "reconnect_in_sec": round(wait_for, 3),
                    },
                )
                await self._sleep(wait_for)
                backoff = self._next_backoff(backoff)

    def _next_reconnect_delay(self, backoff: float) -> float:
        wait_for = min(backoff, self.reconnect_max_sec)
        wait_for += random.uniform(0.0, wait_for * 0.25)
        return wait_for

    def _next_backoff(self, backoff: float) -> float:
        return min(
            self.reconnect_max_sec,
            max(self.reconnect_min_sec, backoff * 2),
        )

    def _recv_watchdog_timeout_sec(self) -> float:
        return min(
            self.stale_reconnect_sec,
            1.0,
            max(0.05, self.stale_reconnect_sec / 2.0),
        )

    async def _recv_next_message(self, ws: object) -> object | None:
        try:
            return await asyncio.wait_for(
                ws.recv(),
                timeout=self._recv_watchdog_timeout_sec(),
            )
        except asyncio.TimeoutError:
            self._raise_if_stale()
            return None

    def _watchdog_age_sec(self) -> float | None:
        candidates = [
            value
            for value in (
                self._current_connection_started_monotonic,
                self._last_sample_monotonic,
            )
            if value is not None
        ]
        if not candidates:
            return None
        return time.monotonic() - max(candidates)

    def _sample_age_sec(self) -> float | None:
        if self._last_sample_monotonic is None:
            return self._watchdog_age_sec()
        return time.monotonic() - self._last_sample_monotonic

    def _stale_watchdog_log_extra(self) -> dict[str, Any]:
        sample_age = self._sample_age_sec()
        return {
            "source": self.source,
            "symbol": self.symbol,
            "last_sample_time": self._last_sample_local_arrival_iso,
            "sample_age_sec": (
                round(sample_age, 3) if sample_age is not None else None
            ),
            "reconnect_count": self.reconnect_count,
            "connection_id": self._connection_id,
        }

    def _raise_if_stale(self) -> None:
        age = self._watchdog_age_sec()
        if age is None or age < self.stale_reconnect_sec:
            return
        self._log.warning(
            "btc_price_feed_stale_detected",
            extra={
                **self._stale_watchdog_log_extra(),
                "stale_reconnect_sec": self.stale_reconnect_sec,
            },
        )
        raise BTCPriceFeedStaleError(
            f"BTC RTDS feed stale for {age:.3f}s "
            f"(threshold={self.stale_reconnect_sec:.3f}s)"
        )

    async def _ping_loop(self, ws: object) -> None:
        while not self._closed:
            await self._sleep(self.ping_interval_sec)
            await ws.send("PING")

    def _handle_raw_message(
        self,
        raw_message: str | bytes | dict[str, Any],
    ) -> BTCPriceSampleRecord | None:
        self._parse_counters["raw_messages_seen"] += 1
        local_arrival_ns = time.time_ns()
        result = analyze_polymarket_rtds_btc_price_message(
            raw_message,
            source=self.source,
            symbol=self.symbol,
            local_arrival_ns=local_arrival_ns,
            local_arrival_iso=local_arrival_iso_from_ns(local_arrival_ns),
        )
        if result.matching_topic:
            self._parse_counters["matching_topic_messages"] += 1
        if result.matching_symbol:
            self._parse_counters["matching_symbol_messages"] += 1
        if result.parse_error:
            self._parse_counters["parse_errors"] += 1
        if result.sample is None:
            self._parse_counters["ignored_messages"] += 1
            self._log_ignored_message(result)
            return None
        stale = self._stale_sample_result(result.sample)
        if stale is not None and self.drop_stale:
            return self._drop_stale_sample(result.sample, stale)
        self._parse_counters["parsed_samples"] += 1
        self._stale_message_streak_started_monotonic = None
        self._last_sample_monotonic = time.monotonic()
        self._last_sample_local_arrival_iso = result.sample.local_arrival_iso
        self._samples.append(result.sample)
        return result.sample

    def _stale_sample_result(
        self,
        sample: BTCPriceSampleRecord,
    ) -> dict[str, Any] | None:
        if self.max_exchange_age_sec <= 0:
            return None
        observed_at = sample.exchange_timestamp or parse_timestamp(sample.local_arrival_iso)
        timestamp_source = "exchange_timestamp" if sample.exchange_timestamp else "local_arrival_iso"
        if observed_at is None:
            return None
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        age_sec = (now - observed_at.astimezone(timezone.utc)).total_seconds()
        if age_sec <= self.max_exchange_age_sec:
            return None
        return {
            "age_sec": age_sec,
            "timestamp_source": timestamp_source,
            "observed_timestamp": observed_at.astimezone(timezone.utc).isoformat(),
        }

    def _drop_stale_sample(
        self,
        sample: BTCPriceSampleRecord,
        stale: dict[str, Any],
    ) -> None:
        self._parse_counters["stale_messages_dropped"] += 1
        now_mono = time.monotonic()
        if self._stale_message_streak_started_monotonic is None:
            self._stale_message_streak_started_monotonic = now_mono
        streak_sec = now_mono - self._stale_message_streak_started_monotonic
        self._log.warning(
            "btc_price_feed_stale_message_dropped",
            extra={
                "source": self.source,
                "symbol": self.symbol,
                "connection_id": self._connection_id,
                "exchange_timestamp": (
                    sample.exchange_timestamp.isoformat()
                    if sample.exchange_timestamp is not None
                    else None
                ),
                "local_arrival_iso": sample.local_arrival_iso,
                "timestamp_source": stale.get("timestamp_source"),
                "age_sec": round(float(stale["age_sec"]), 3),
                "max_exchange_age_sec": self.max_exchange_age_sec,
                "stale_message_streak_sec": round(streak_sec, 3),
                "stale_messages_dropped": self._parse_counters["stale_messages_dropped"],
            },
        )
        if streak_sec >= self.stale_message_reconnect_sec:
            self.stale_reconnect_requested_count += 1
            self._parse_counters["stale_reconnect_requested"] += 1
            raise BTCPriceFeedStaleError(
                "BTC RTDS feed received stale messages for "
                f"{streak_sec:.3f}s (threshold={self.stale_message_reconnect_sec:.3f}s)"
            )
        return None

    def _log_ignored_message(self, result: RTDSBTCParseResult) -> None:
        reason = result.reason or "unknown"
        key = "|".join(
            [
                reason,
                result.topic or "",
                result.message_type or "",
                result.payload_symbol or "",
            ]
        )
        count = self._ignored_log_counts.get(key, 0) + 1
        self._ignored_log_counts[key] = count
        now = time.monotonic()
        last_logged = self._ignored_log_last_monotonic.get(key, 0.0)
        if count > 3 and now - last_logged < 30.0:
            return
        self._ignored_log_last_monotonic[key] = now
        self._log.debug(
            "btc_price_feed_rtds_message_ignored",
            extra={
                "source": self.source,
                "symbol": self.symbol,
                "topic": result.topic,
                "message_type": result.message_type,
                "payload_symbol": result.payload_symbol,
                "ignore_reason": reason,
                "parse_error": result.parse_error,
                "parse_counters": self.parse_counters,
            },
        )


def _raw_message_text(raw_message: str | bytes | dict[str, Any]) -> str:
    if isinstance(raw_message, bytes):
        return raw_message.decode("utf-8", errors="replace")
    if isinstance(raw_message, str):
        return raw_message
    return json.dumps(raw_message, sort_keys=True, separators=(",", ":"))


def _pretty_raw_message(raw_message: str | bytes | dict[str, Any]) -> str:
    message, raw_text = _coerce_json_payload(raw_message)
    if message is None:
        return raw_text
    return json.dumps(message, indent=2, sort_keys=True)


def _parser_result_payload(result: RTDSBTCParseResult) -> dict[str, Any]:
    return {
        "matched": result.sample is not None,
        "ignore_reason": result.reason,
        "topic": result.topic,
        "message_type": result.message_type,
        "payload_symbol": result.payload_symbol,
        "matching_topic": result.matching_topic,
        "matching_symbol": result.matching_symbol,
        "parse_error": result.parse_error,
        "sample": (
            {
                "source": result.sample.source,
                "price": result.sample.price,
                "exchange_timestamp": (
                    result.sample.exchange_timestamp.isoformat()
                    if result.sample.exchange_timestamp is not None
                    else None
                ),
                "local_arrival_ns": result.sample.local_arrival_ns,
                "local_arrival_iso": result.sample.local_arrival_iso,
            }
            if result.sample is not None
            else None
        ),
    }


async def probe_polymarket_rtds_btc_feed(
    *,
    source: str,
    symbol: str,
    ws_url: str = POLYMARKET_RTDS_WS_URL,
    max_messages: int = 20,
    timeout_sec: float = 30.0,
    raw_only: bool = False,
    connect: Callable[..., Any] | None = None,
    output: TextIO | None = None,
) -> dict[str, Any]:
    out = output or sys.stdout
    safe_max = max(1, int(max_messages))
    safe_timeout = max(0.1, float(timeout_sec))
    feed = PolymarketRTDSBTCPriceFeed(
        source=str(source).strip().lower(),
        symbol=str(symbol).strip(),
        ws_url=str(ws_url).strip() or POLYMARKET_RTDS_WS_URL,
        connect=connect,
    )
    subscription_payload = feed.subscription_payload()
    summary: dict[str, Any] = {
        "source": feed.source,
        "symbol": feed.symbol,
        "ws_url": feed.ws_url,
        "max_messages": safe_max,
        "timeout_sec": safe_timeout,
        "messages_seen": 0,
        "parser_matches": 0,
        "parser_ignored": 0,
        "connection_ok": False,
        "error": None,
    }

    print("=== RTDS BTC Probe ===", file=out)
    print(f"source={feed.source}", file=out)
    print(f"symbol={feed.symbol}", file=out)
    print(f"ws_url={feed.ws_url}", file=out)
    print("subscription_payload=", file=out)
    print(json.dumps(subscription_payload, indent=2, sort_keys=True), file=out)

    connector = connect or websockets.connect
    try:
        async with connector(feed.ws_url, max_queue=None, max_size=None) as ws:
            summary["connection_ok"] = True
            print("connection=ok", file=out)
            await ws.send(json.dumps(subscription_payload))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + safe_timeout
            while summary["messages_seen"] < safe_max:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    print("timeout=reached", file=out)
                    break
                try:
                    raw_message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    print("timeout=reached", file=out)
                    break

                summary["messages_seen"] += 1
                message_index = summary["messages_seen"]
                print(f"--- message {message_index} ---", file=out)
                if raw_only:
                    print(_raw_message_text(raw_message), file=out)
                    continue

                print("raw_message=", file=out)
                print(_pretty_raw_message(raw_message), file=out)
                extracted = extract_rtds_debug_fields(raw_message)
                local_arrival_ns = time.time_ns()
                parser_result = analyze_polymarket_rtds_btc_price_message(
                    raw_message,
                    source=feed.source,
                    symbol=feed.symbol,
                    local_arrival_ns=local_arrival_ns,
                    local_arrival_iso=local_arrival_iso_from_ns(local_arrival_ns),
                )
                if parser_result.sample is not None:
                    summary["parser_matches"] += 1
                else:
                    summary["parser_ignored"] += 1
                print("extracted=", file=out)
                print(json.dumps(extracted, indent=2, sort_keys=True), file=out)
                print("parser_result=", file=out)
                print(
                    json.dumps(
                        _parser_result_payload(parser_result),
                        indent=2,
                        sort_keys=True,
                    ),
                    file=out,
                )
    except Exception as exc:
        summary["error"] = str(exc)
        print("connection=failed", file=out)
        print(f"error={exc}", file=out)

    print("summary=", file=out)
    print(json.dumps(summary, indent=2, sort_keys=True), file=out)
    return summary


class UnsupportedBTCPriceFeed:
    def __init__(self, source: str, symbol: str, ws_url: str) -> None:
        self.source = source
        self.symbol = symbol
        self.ws_url = ws_url

    async def sample(
        self,
        *,
        local_arrival_ns: int,
        local_arrival_iso: str,
    ) -> BTCPriceSampleRecord | None:
        _ = (local_arrival_ns, local_arrival_iso)
        raise BTCPriceFeedError(
            f"BTC price feed source is not implemented yet: {self.source}"
        )

    async def close(self) -> None:
        return None


def parse_btc_price_feed_sources(raw_sources: object) -> tuple[str, ...]:
    if raw_sources is None:
        return ()
    if isinstance(raw_sources, str):
        values = raw_sources.split(",")
    else:
        try:
            values = list(raw_sources)  # type: ignore[arg-type]
        except TypeError:
            values = [raw_sources]

    sources: list[str] = []
    seen: set[str] = set()
    for value in values:
        source = str(value or "").strip().lower()
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return tuple(sources)


def resolve_btc_price_feed_sources(settings: object) -> tuple[str, ...]:
    configured = parse_btc_price_feed_sources(
        getattr(settings, "btc_price_feed_sources", ())
    )
    if configured:
        return configured
    legacy_source = (
        str(getattr(settings, "btc_price_feed_source", "mock") or "mock")
        .strip()
        .lower()
    )
    return (legacy_source,)


def btc_price_feed_symbol_for_source(source: str, fallback_symbol: str) -> str:
    if source == POLYMARKET_RTDS_CHAINLINK_SOURCE:
        return "btc/usd"
    if source == POLYMARKET_RTDS_BINANCE_SOURCE:
        return "btcusdt"
    if source == BINANCE_REST_SOURCE:
        return "BTCUSDT"
    return fallback_symbol


def is_btc_price_feed_source_supported(source: str) -> bool:
    return str(source or "").strip().lower() in SUPPORTED_BTC_PRICE_FEED_SOURCES


def btc_price_feed_ws_url(settings: object) -> str:
    return str(
        getattr(
            settings,
            "btc_price_feed_ws_url",
            POLYMARKET_RTDS_WS_URL,
        )
        or POLYMARKET_RTDS_WS_URL
    ).strip()


def btc_price_feed_url_for_source(settings: object, source: str) -> str:
    if source == BINANCE_REST_SOURCE:
        return str(
            getattr(settings, "btc_price_binance_rest_url", BINANCE_REST_URL)
            or BINANCE_REST_URL
        ).strip()
    return btc_price_feed_ws_url(settings)


def describe_btc_price_feed_config(settings: object) -> dict[str, Any]:
    sources = resolve_btc_price_feed_sources(settings)
    fallback_symbol = str(
        getattr(settings, "btc_price_feed_symbol", "btc/usd") or "btc/usd"
    ).strip()
    planned_feeds = [
        {
            "source": source,
            "symbol": btc_price_feed_symbol_for_source(source, fallback_symbol),
            "ws_url": btc_price_feed_url_for_source(settings, source),
            "supported": is_btc_price_feed_source_supported(source),
        }
        for source in sources
    ]
    ws_url = btc_price_feed_ws_url(settings)
    return {
        "BTC_PRICE_FEED_ENABLED": bool(
            getattr(settings, "btc_price_feed_enabled", False)
        ),
        "BTC_PRICE_FEED_SOURCE": str(
            getattr(settings, "btc_price_feed_source", "mock") or "mock"
        ),
        "BTC_PRICE_FEED_SOURCES": getattr(
            settings,
            "btc_price_feed_sources_raw",
            ",".join(str(source) for source in getattr(settings, "btc_price_feed_sources", ()) or ()),
        ),
        "BTC_PRICE_FEED_SYMBOL": fallback_symbol,
        "BTC_PRICE_FEED_WS_URL": ws_url,
        "BTC_PRICE_FEED_STALE_RECONNECT_SEC": float(
            getattr(settings, "btc_price_feed_stale_reconnect_sec", 10.0) or 10.0
        ),
        "BTC_PRICE_MAX_EXCHANGE_AGE_SEC": float(
            getattr(settings, "btc_price_max_exchange_age_sec", 15.0) or 15.0
        ),
        "BTC_PRICE_DROP_STALE": bool(
            getattr(settings, "btc_price_drop_stale", True)
        ),
        "BTC_PRICE_STALE_RECONNECT_SEC": float(
            getattr(settings, "btc_price_stale_reconnect_sec", 30.0) or 30.0
        ),
        "BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC": float(
            getattr(settings, "btc_price_feed_startup_timeout_sec", 5.0) or 5.0
        ),
        "BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC": float(
            getattr(settings, "btc_price_feed_sample_timeout_sec", 2.0) or 2.0
        ),
        "BTC_PRICE_BINANCE_REST_URL": str(
            getattr(settings, "btc_price_binance_rest_url", BINANCE_REST_URL)
            or BINANCE_REST_URL
        ),
        "parsed_sources": list(sources),
        "source_symbol_map": {
            feed["source"]: feed["symbol"] for feed in planned_feeds
        },
        "source_ws_url_map": {
            feed["source"]: feed["ws_url"] for feed in planned_feeds
        },
        "planned_feeds": planned_feeds,
    }


def build_btc_price_feed_for_source(settings: object, source: str) -> BTCPriceFeed:
    source = str(source or "mock").strip().lower()
    fallback_symbol = str(
        getattr(settings, "btc_price_feed_symbol", "btc/usd") or "btc/usd"
    ).strip()
    symbol = btc_price_feed_symbol_for_source(source, fallback_symbol)
    url = btc_price_feed_url_for_source(settings, source)
    if source == "mock":
        return MockBTCPriceFeed()
    if source == BINANCE_REST_SOURCE:
        return BinanceRESTBTCPriceFeed(
            symbol=symbol,
            url=url,
            timeout_sec=float(
                getattr(settings, "btc_price_feed_sample_timeout_sec", 2.0) or 2.0
            ),
        )
    if source in {POLYMARKET_RTDS_CHAINLINK_SOURCE, POLYMARKET_RTDS_BINANCE_SOURCE}:
        return PolymarketRTDSBTCPriceFeed(
            source=source,
            symbol=symbol,
            ws_url=url,
            reconnect_min_sec=float(
                getattr(settings, "ws_reconnect_min_sec", 1.0) or 1.0
            ),
            reconnect_max_sec=float(
                getattr(settings, "ws_reconnect_max_sec", 30.0) or 30.0
            ),
            stale_reconnect_sec=float(
                getattr(settings, "btc_price_feed_stale_reconnect_sec", 10.0)
                or 10.0
            ),
            max_exchange_age_sec=float(
                getattr(settings, "btc_price_max_exchange_age_sec", 15.0) or 15.0
            ),
            drop_stale=bool(getattr(settings, "btc_price_drop_stale", True)),
            stale_message_reconnect_sec=float(
                getattr(settings, "btc_price_stale_reconnect_sec", 30.0) or 30.0
            ),
        )
    return UnsupportedBTCPriceFeed(source=source, symbol=symbol, ws_url=url)


def build_btc_price_feeds(settings: object) -> dict[str, BTCPriceFeed]:
    feeds: dict[str, BTCPriceFeed] = {}
    for source in resolve_btc_price_feed_sources(settings):
        feeds[source] = build_btc_price_feed_for_source(settings, source)
    return feeds


def build_btc_price_feed(settings: object) -> BTCPriceFeed:
    source = (
        str(getattr(settings, "btc_price_feed_source", "mock") or "mock")
        .strip()
        .lower()
    )
    return build_btc_price_feed_for_source(settings, source)
