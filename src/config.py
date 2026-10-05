from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_ENV_PATH = ".env"


def _load_dotenv_if_present(path: str) -> None:
    """Load simple KEY=VALUE pairs into environment if not already present."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name)
    value = (raw if raw is not None else default).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _get_csv(name: str) -> tuple[str, ...]:
    return _parse_csv(os.getenv(name))


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Settings:
    db_path: str
    log_level: str
    log_json: bool

    gamma_api_url: str
    clob_api_url: str
    data_api_url: str
    ws_url: str

    request_timeout_sec: float
    sqlite_busy_timeout_ms: int
    startup_integrity_check_mode: str

    discovery_interval_sec: int
    discovery_page_size: int
    discovery_max_pages: int
    discovery_lookahead_sec: int
    snapshot_interval_sec: float
    startup_orderbook_concurrency: int
    tracked_markets_limit: int
    market_unsubscribe_grace_sec: int

    orderbook_levels_to_store: int
    feature_orderbook_depth: int

    enable_trade_backfill: bool
    trade_backfill_interval_sec: int

    ws_ping_interval_sec: float
    ws_ping_timeout_sec: float
    ws_reconnect_min_sec: float
    ws_reconnect_max_sec: float

    jump_threshold: float
    heartbeat_log_interval_sec: int
    snapshot_gap_threshold_multiplier: float
    trade_snapshot_freshness_sec: int

    btc_price_feed_enabled: bool
    btc_price_feed_sources_raw: str | None
    btc_price_feed_sources: tuple[str, ...]
    btc_price_feed_source: str
    btc_price_feed_symbol: str
    btc_price_feed_interval_sec: float
    btc_price_feed_ws_url: str
    btc_price_feed_stale_reconnect_sec: float
    btc_price_max_exchange_age_sec: float
    btc_price_drop_stale: bool
    btc_price_stale_reconnect_sec: float
    btc_price_feed_startup_timeout_sec: float
    btc_price_feed_sample_timeout_sec: float
    btc_price_binance_rest_url: str

    raw_ws_events_enabled: bool
    raw_ws_events_retention_sec: float
    raw_ws_events_max_rows: int
    raw_ws_events_prune_interval_sec: float
    raw_ws_events_prune_batch_size: int

    sqlite_wal_checkpoint_interval_sec: float
    sqlite_wal_checkpoint_mode: str

    auto_normalize_resolutions_enabled: bool
    auto_normalize_resolutions_interval_sec: float
    auto_repair_stale_active_enabled: bool
    auto_repair_stale_active_interval_sec: float
    auto_repair_stale_active_grace_sec: float


def load_settings(env_file: Optional[str] = None) -> Settings:
    _load_dotenv_if_present(env_file or DEFAULT_ENV_PATH)
    btc_price_feed_sources_raw = os.getenv("BTC_PRICE_FEED_SOURCES")

    return Settings(
        db_path=os.getenv("RECORDER_DB_PATH", "data/recorder.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_json=_get_bool("LOG_JSON", True),
        gamma_api_url=os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com"),
        clob_api_url=os.getenv("CLOB_API_URL", "https://clob.polymarket.com"),
        data_api_url=os.getenv("DATA_API_URL", "https://data-api.polymarket.com"),
        ws_url=os.getenv(
            "CLOB_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        ),
        request_timeout_sec=_get_float("REQUEST_TIMEOUT_SEC", 10.0),
        sqlite_busy_timeout_ms=_get_int("SQLITE_BUSY_TIMEOUT_MS", 5000),
        startup_integrity_check_mode=_get_choice(
            "STARTUP_INTEGRITY_CHECK_MODE",
            "quick",
            {"off", "quick", "full"},
        ),
        discovery_interval_sec=_get_int("DISCOVERY_INTERVAL_SEC", 120),
        discovery_page_size=_get_int("DISCOVERY_PAGE_SIZE", 500),
        discovery_max_pages=_get_int("DISCOVERY_MAX_PAGES", 60),
        discovery_lookahead_sec=_get_int("DISCOVERY_LOOKAHEAD_SEC", 7200),
        snapshot_interval_sec=_get_float("SNAPSHOT_INTERVAL_SEC", 1.0),
        startup_orderbook_concurrency=_get_int("STARTUP_ORDERBOOK_CONCURRENCY", 20),
        tracked_markets_limit=_get_int("TRACKED_MARKETS_LIMIT", 0),
        market_unsubscribe_grace_sec=_get_int("MARKET_UNSUBSCRIBE_GRACE_SEC", 60),
        orderbook_levels_to_store=_get_int("ORDERBOOK_LEVELS_TO_STORE", 10),
        feature_orderbook_depth=_get_int("FEATURE_ORDERBOOK_DEPTH", 5),
        enable_trade_backfill=_get_bool("ENABLE_TRADE_BACKFILL", False),
        trade_backfill_interval_sec=_get_int("TRADE_BACKFILL_INTERVAL_SEC", 120),
        ws_ping_interval_sec=_get_float("WS_PING_INTERVAL_SEC", 20.0),
        ws_ping_timeout_sec=_get_float("WS_PING_TIMEOUT_SEC", 20.0),
        ws_reconnect_min_sec=_get_float("WS_RECONNECT_MIN_SEC", 1.0),
        ws_reconnect_max_sec=_get_float("WS_RECONNECT_MAX_SEC", 30.0),
        jump_threshold=_get_float("JUMP_THRESHOLD", 0.05),
        heartbeat_log_interval_sec=_get_int(
            "RECORDER_HEARTBEAT_INTERVAL_SEC",
            _get_int("HEARTBEAT_LOG_INTERVAL_SEC", 30),
        ),
        snapshot_gap_threshold_multiplier=_get_float(
            "SNAPSHOT_GAP_THRESHOLD_MULTIPLIER", 1.5
        ),
        trade_snapshot_freshness_sec=_get_int("TRADE_SNAPSHOT_FRESHNESS_SEC", 2),
        btc_price_feed_enabled=_get_bool("BTC_PRICE_FEED_ENABLED", False),
        btc_price_feed_sources_raw=btc_price_feed_sources_raw,
        btc_price_feed_sources=_parse_csv(btc_price_feed_sources_raw),
        btc_price_feed_source=os.getenv("BTC_PRICE_FEED_SOURCE", "mock"),
        btc_price_feed_symbol=os.getenv("BTC_PRICE_FEED_SYMBOL", "btc/usd"),
        btc_price_feed_interval_sec=_get_float("BTC_PRICE_FEED_INTERVAL_SEC", 1.0),
        btc_price_feed_ws_url=os.getenv(
            "BTC_PRICE_FEED_WS_URL",
            "wss://ws-live-data.polymarket.com",
        ),
        btc_price_feed_stale_reconnect_sec=_get_float(
            "BTC_PRICE_FEED_STALE_RECONNECT_SEC",
            10.0,
        ),
        btc_price_max_exchange_age_sec=_get_float(
            "BTC_PRICE_MAX_EXCHANGE_AGE_SEC",
            15.0,
        ),
        btc_price_drop_stale=_get_bool("BTC_PRICE_DROP_STALE", True),
        btc_price_stale_reconnect_sec=_get_float(
            "BTC_PRICE_STALE_RECONNECT_SEC",
            30.0,
        ),
        btc_price_feed_startup_timeout_sec=_get_float(
            "BTC_PRICE_FEED_STARTUP_TIMEOUT_SEC",
            5.0,
        ),
        btc_price_feed_sample_timeout_sec=_get_float(
            "BTC_PRICE_FEED_SAMPLE_TIMEOUT_SEC",
            2.0,
        ),
        btc_price_binance_rest_url=os.getenv(
            "BTC_PRICE_BINANCE_REST_URL",
            "https://api.binance.com/api/v3/ticker/24hr",
        ),
        raw_ws_events_enabled=_get_bool("RAW_WS_EVENTS_ENABLED", False),
        raw_ws_events_retention_sec=_get_float("RAW_WS_EVENTS_RETENTION_SEC", 0.0),
        raw_ws_events_max_rows=_get_int("RAW_WS_EVENTS_MAX_ROWS", 0),
        raw_ws_events_prune_interval_sec=_get_float(
            "RAW_WS_EVENTS_PRUNE_INTERVAL_SEC",
            300.0,
        ),
        raw_ws_events_prune_batch_size=_get_int(
            "RAW_WS_EVENTS_PRUNE_BATCH_SIZE",
            50000,
        ),
        sqlite_wal_checkpoint_interval_sec=_get_float(
            "SQLITE_WAL_CHECKPOINT_INTERVAL_SEC",
            300.0,
        ),
        sqlite_wal_checkpoint_mode=_get_choice(
            "SQLITE_WAL_CHECKPOINT_MODE",
            "passive",
            {"off", "passive", "full", "restart", "truncate"},
        ).upper(),
        auto_normalize_resolutions_enabled=_get_bool(
            "AUTO_NORMALIZE_RESOLUTIONS_ENABLED",
            True,
        ),
        auto_normalize_resolutions_interval_sec=_get_float(
            "AUTO_NORMALIZE_RESOLUTIONS_INTERVAL_SEC",
            60.0,
        ),
        auto_repair_stale_active_enabled=_get_bool(
            "AUTO_REPAIR_STALE_ACTIVE_ENABLED",
            True,
        ),
        auto_repair_stale_active_interval_sec=_get_float(
            "AUTO_REPAIR_STALE_ACTIVE_INTERVAL_SEC",
            300.0,
        ),
        auto_repair_stale_active_grace_sec=_get_float(
            "AUTO_REPAIR_STALE_ACTIVE_GRACE_SEC",
            60.0,
        ),
    )
