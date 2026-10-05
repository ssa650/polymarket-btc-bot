#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python venv not found at $PYTHON_BIN" >&2
  echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

export BTC_PRICE_FEED_ENABLED="${BTC_PRICE_FEED_ENABLED:-true}"
export BTC_PRICE_FEED_SOURCE="${BTC_PRICE_FEED_SOURCE:-polymarket_rtds_chainlink}"
export BTC_PRICE_FEED_SYMBOL="${BTC_PRICE_FEED_SYMBOL:-btc/usd}"
export BTC_PRICE_FEED_WS_URL="${BTC_PRICE_FEED_WS_URL:-wss://ws-live-data.polymarket.com}"
export BTC_PRICE_FEED_STALE_RECONNECT_SEC="${BTC_PRICE_FEED_STALE_RECONNECT_SEC:-10}"
export RAW_WS_EVENTS_ENABLED="${RAW_WS_EVENTS_ENABLED:-false}"
export RAW_WS_EVENTS_PRUNE_INTERVAL_SEC="${RAW_WS_EVENTS_PRUNE_INTERVAL_SEC:-300}"
export RAW_WS_EVENTS_PRUNE_BATCH_SIZE="${RAW_WS_EVENTS_PRUNE_BATCH_SIZE:-50000}"
export SQLITE_WAL_CHECKPOINT_INTERVAL_SEC="${SQLITE_WAL_CHECKPOINT_INTERVAL_SEC:-300}"
export SQLITE_WAL_CHECKPOINT_MODE="${SQLITE_WAL_CHECKPOINT_MODE:-PASSIVE}"
export STARTUP_INTEGRITY_CHECK_MODE="${STARTUP_INTEGRITY_CHECK_MODE:-quick}"

exec "$PYTHON_BIN" -m src.main "$@"
