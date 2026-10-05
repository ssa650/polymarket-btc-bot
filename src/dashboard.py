from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .models import parse_timestamp


EVENT_TYPES: tuple[str, ...] = (
    "price_change",
    "best_bid_ask",
    "book",
    "last_trade_price",
    "new_market",
    "tick_size_change",
    "market_resolved",
)


@dataclass(frozen=True, slots=True)
class DashboardData:
    now: datetime
    db_path: str
    active_market: dict[str, Any] | None
    btc_price: dict[str, Any] | None
    latest_snapshot: dict[str, Any] | None
    best_bid_ask: dict[str, dict[str, Any] | None]
    recent_trades: tuple[dict[str, Any], ...]
    health: dict[str, Any] | None
    event_counts: dict[str, int]
    warnings: tuple[str, ...]


def connect_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def fetch_dashboard_data(
    db_path: str,
    *,
    now: datetime | None = None,
    preferred_btc_source: str | None = None,
) -> DashboardData:
    now_dt = _utc(now or datetime.now(timezone.utc))
    conn = connect_read_only(db_path)
    try:
        active_market = _fetch_active_market(conn)
        run_id = _current_run_id(conn, active_market)
        market_id = _value(active_market, "market_id")
        yes_token_id = _value(active_market, "yes_token_id")
        no_token_id = _value(active_market, "no_token_id")
        btc_price = _fetch_latest_btc_price(
            conn,
            preferred_source=preferred_btc_source,
        )
        latest_snapshot = (
            _fetch_latest_snapshot(conn, run_id=run_id, market_id=market_id)
            if market_id is not None
            else None
        )
        best_bid_ask = {
            "YES": _fetch_latest_best_bid_ask(
                conn,
                run_id=run_id,
                asset_id=yes_token_id,
            ),
            "NO": _fetch_latest_best_bid_ask(
                conn,
                run_id=run_id,
                asset_id=no_token_id,
            ),
        }
        recent_trades = (
            _fetch_recent_trades(
                conn,
                run_id=run_id,
                market_id=market_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
            if market_id is not None
            else ()
        )
        health = _fetch_latest_health(conn)
        event_counts = _fetch_event_counts(conn, run_id=run_id)
        warnings = _dashboard_warnings(
            now=now_dt,
            active_market=active_market,
            btc_price=btc_price,
            latest_snapshot=latest_snapshot,
            best_bid_ask=best_bid_ask,
            health=health,
        )
        return DashboardData(
            now=now_dt,
            db_path=db_path,
            active_market=active_market,
            btc_price=btc_price,
            latest_snapshot=latest_snapshot,
            best_bid_ask=best_bid_ask,
            recent_trades=recent_trades,
            health=health,
            event_counts=event_counts,
            warnings=warnings,
        )
    finally:
        conn.close()


def render_dashboard(data: DashboardData) -> str:
    lines: list[str] = [
        "=== Polymarket BTC 5m Recorder Dashboard ===",
        f"db_path={data.db_path}",
        f"refreshed_at={data.now.isoformat()}",
    ]
    lines.extend(_render_active_market(data.active_market, now=data.now))
    lines.extend(_render_btc_price(data.btc_price, now=data.now))
    lines.extend(_render_snapshot(data.latest_snapshot, now=data.now))
    lines.extend(_render_best_bid_ask(data.best_bid_ask, now=data.now))
    lines.extend(_render_recent_trades(data.recent_trades))
    lines.extend(_render_health(data.health))
    lines.extend(_render_event_counts(data.event_counts))
    lines.extend(_render_warnings(data.warnings))
    return "\n".join(lines)


def run_dashboard(
    db_path: str,
    *,
    refresh_sec: float = 1.0,
    once: bool = False,
) -> None:
    interval = max(0.1, float(refresh_sec))
    while True:
        data = fetch_dashboard_data(db_path)
        output = render_dashboard(data)
        if once:
            print(output)
            return
        print("\033[2J\033[H" + output, end="\n", flush=True)
        time.sleep(interval)


def _fetch_active_market(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not _table_exists(conn, "markets"):
        return None
    row = conn.execute(
        """
        SELECT
          run_id,
          market_id,
          question,
          start_time,
          close_time,
          COALESCE(phase, market_phase) AS phase,
          tracking_state,
          yes_token_id,
          no_token_id
        FROM markets
        WHERE tracking_state = 'selected_active'
        ORDER BY datetime(start_time) DESC, datetime(last_updated) DESC
        LIMIT 1
        """
    ).fetchone()
    return _dict(row)


def _current_run_id(
    conn: sqlite3.Connection,
    active_market: dict[str, Any] | None,
) -> str | None:
    run_id = _value(active_market, "run_id")
    if run_id:
        return str(run_id)
    if not _table_exists(conn, "recorder_metrics"):
        return None
    row = conn.execute(
        """
        SELECT run_id
        FROM recorder_metrics
        WHERE run_id IS NOT NULL AND run_id <> ''
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row is not None and row[0] is not None else None


def _fetch_latest_btc_price(
    conn: sqlite3.Connection,
    *,
    preferred_source: str | None = None,
) -> dict[str, Any] | None:
    if not _table_exists(conn, "btc_prices"):
        return None
    params: list[object] = []
    source_sql = ""
    if preferred_source:
        source_sql = "WHERE source = ?"
        params.append(preferred_source)
    row = conn.execute(
        f"""
        SELECT source, price, exchange_timestamp, local_arrival_ns, local_arrival_iso
        FROM btc_prices
        {source_sql}
        ORDER BY local_arrival_ns DESC, datetime(local_arrival_iso) DESC, id DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    return _dict(row)


def _fetch_latest_snapshot(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    market_id: str | None,
) -> dict[str, Any] | None:
    if not _table_exists(conn, "market_snapshots") or market_id is None:
        return None
    params: list[object] = [market_id]
    run_sql = ""
    if run_id is not None:
        run_sql = "AND run_id = ?"
        params.append(run_id)
    row = conn.execute(
        f"""
        SELECT
          timestamp,
          best_bid_yes,
          best_ask_yes,
          spread_yes,
          best_bid_no,
          best_ask_no,
          spread_no,
          mid_price_yes,
          mid_price_no,
          last_trade_price,
          last_trade_size,
          last_trade_time,
          has_orderbook,
          has_trade_data
        FROM market_snapshots
        WHERE market_id = ?
          {run_sql}
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    return _dict(row)


def _fetch_latest_best_bid_ask(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    asset_id: str | None,
) -> dict[str, Any] | None:
    if not _table_exists(conn, "best_bid_ask_updates") or asset_id is None:
        return None
    params: list[object] = [asset_id]
    run_sql = ""
    if run_id is not None:
        run_sql = "AND run_id = ?"
        params.append(run_id)
    row = conn.execute(
        f"""
        SELECT timestamp, asset_id, best_bid, best_ask, spread
        FROM best_bid_ask_updates
        WHERE asset_id = ?
          {run_sql}
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    return _dict(row)


def _fetch_recent_trades(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    market_id: str | None,
    yes_token_id: str | None,
    no_token_id: str | None,
    limit: int = 10,
) -> tuple[dict[str, Any], ...]:
    if not _table_exists(conn, "trades") or market_id is None:
        return ()
    params: list[object] = [market_id]
    run_sql = ""
    if run_id is not None:
        run_sql = "AND run_id = ?"
        params.append(run_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT timestamp, side, price, size, trade_id
        FROM trades
        WHERE market_id = ?
          {run_sql}
        ORDER BY datetime(timestamp) DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    trades = []
    for row in rows:
        item = _dict(row) or {}
        asset_id = _asset_id_from_trade_id(str(item.get("trade_id") or ""))
        item["asset_id"] = asset_id
        item["outcome"] = _outcome_from_asset(
            asset_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        )
        trades.append(item)
    return tuple(trades)


def _fetch_latest_health(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not _table_exists(conn, "recorder_metrics"):
        return None
    columns = (
        "timestamp",
        "run_id",
        "raw_ws_events_seen",
        "raw_ws_events_written",
        "malformed_ws_events",
        "raw_ws_write_failures",
        "last_ws_event_age_sec",
        "subscribed_asset_count",
        "ws_reconnect_count",
    )
    selected = [column for column in columns if _has_column(conn, "recorder_metrics", column)]
    if not selected:
        return None
    row = conn.execute(
        f"""
        SELECT {", ".join(selected)}
        FROM recorder_metrics
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
        """
    ).fetchone()
    return _dict(row)


def _fetch_event_counts(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
) -> dict[str, int]:
    counts = {event_type: 0 for event_type in EVENT_TYPES}
    if not _table_exists(conn, "raw_polymarket_events"):
        return counts
    params: list[object] = []
    where = ""
    if run_id is not None:
        where = "WHERE run_id = ?"
        params.append(run_id)
    rows = conn.execute(
        f"""
        SELECT event_type, COUNT(*) AS cnt
        FROM raw_polymarket_events
        {where}
        GROUP BY event_type
        """,
        tuple(params),
    ).fetchall()
    for row in rows:
        event_type = str(row["event_type"] or "")
        if event_type in counts:
            counts[event_type] = int(row["cnt"] or 0)
    return counts


def _dashboard_warnings(
    *,
    now: datetime,
    active_market: dict[str, Any] | None,
    btc_price: dict[str, Any] | None,
    latest_snapshot: dict[str, Any] | None,
    best_bid_ask: dict[str, dict[str, Any] | None],
    health: dict[str, Any] | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if active_market is None:
        warnings.append("no_selected_active_market")
    else:
        close_time = parse_timestamp(active_market.get("close_time"))
        if close_time is not None and close_time <= now:
            warnings.append("selected_active_market_past_close_time")

    btc_ts = parse_timestamp(_value(btc_price, "local_arrival_iso"))
    if btc_price is None:
        warnings.append("missing_btc_price")
    elif btc_ts is not None and _age_sec(now, btc_ts) > 3:
        warnings.append("stale_btc_price")

    snapshot_ts = parse_timestamp(_value(latest_snapshot, "timestamp"))
    if latest_snapshot is None:
        warnings.append("no_recent_snapshots")
    elif snapshot_ts is not None and _age_sec(now, snapshot_ts) > 5:
        warnings.append("no_recent_snapshots")

    if best_bid_ask.get("YES") is None:
        warnings.append("missing_yes_best_bid_ask")
    if best_bid_ask.get("NO") is None:
        warnings.append("missing_no_best_bid_ask")

    metric_ts = parse_timestamp(_value(health, "timestamp"))
    if health is None:
        warnings.append("missing_recorder_metrics")
    elif metric_ts is not None and _age_sec(now, metric_ts) > 10:
        warnings.append("stale_recorder_metrics")

    last_ws_age = _float_or_none(_value(health, "last_ws_event_age_sec"))
    if last_ws_age is not None and last_ws_age > 5:
        warnings.append("stale_ws_events")
    if int(_value(health, "malformed_ws_events") or 0) > 0:
        warnings.append("malformed_ws_events_gt_zero")
    if int(_value(health, "raw_ws_write_failures") or 0) > 0:
        warnings.append("raw_ws_write_failures_gt_zero")
    return tuple(warnings)


def _render_active_market(market: dict[str, Any] | None, *, now: datetime) -> list[str]:
    lines = ["", "=== Current Selected Active Market ==="]
    if market is None:
        return [*lines, "none"]
    close_time = parse_timestamp(market.get("close_time"))
    remaining = (
        f"{max(0.0, (close_time - now).total_seconds()):.1f}s"
        if close_time is not None
        else "unknown"
    )
    for key in (
        "market_id",
        "question",
        "start_time",
        "close_time",
        "phase",
        "tracking_state",
        "yes_token_id",
        "no_token_id",
    ):
        lines.append(f"{key}={market.get(key)}")
    lines.append(f"time_remaining={remaining}")
    return lines


def _render_btc_price(row: dict[str, Any] | None, *, now: datetime) -> list[str]:
    lines = ["", "=== BTC Price ==="]
    if row is None:
        return [*lines, "none"]
    ts = parse_timestamp(row.get("local_arrival_iso"))
    lines.extend(
        [
            f"price={row.get('price')}",
            f"source={row.get('source')}",
            f"exchange_timestamp={row.get('exchange_timestamp')}",
            f"local_arrival_iso={row.get('local_arrival_iso')}",
            f"sample_age_sec={_age_text(now, ts)}",
        ]
    )
    return lines


def _render_snapshot(row: dict[str, Any] | None, *, now: datetime) -> list[str]:
    lines = ["", "=== Latest Market Snapshot ==="]
    if row is None:
        return [*lines, "none"]
    ts = parse_timestamp(row.get("timestamp"))
    lines.append(f"timestamp={row.get('timestamp')} age_sec={_age_text(now, ts)}")
    for key in (
        "best_bid_yes",
        "best_ask_yes",
        "spread_yes",
        "best_bid_no",
        "best_ask_no",
        "spread_no",
        "mid_price_yes",
        "mid_price_no",
        "last_trade_price",
        "last_trade_size",
        "last_trade_time",
        "has_orderbook",
        "has_trade_data",
    ):
        lines.append(f"{key}={row.get(key)}")
    return lines


def _render_best_bid_ask(
    rows: dict[str, dict[str, Any] | None],
    *,
    now: datetime,
) -> list[str]:
    lines = ["", "=== Latest Best Bid Ask Updates ==="]
    for outcome in ("YES", "NO"):
        row = rows.get(outcome)
        if row is None:
            lines.append(f"{outcome}: none")
            continue
        ts = parse_timestamp(row.get("timestamp"))
        lines.append(
            f"{outcome}: asset_id={row.get('asset_id')} "
            f"best_bid={row.get('best_bid')} best_ask={row.get('best_ask')} "
            f"spread={row.get('spread')} timestamp={row.get('timestamp')} "
            f"age_sec={_age_text(now, ts)}"
        )
    return lines


def _render_recent_trades(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = ["", "=== Recent Trades ==="]
    if not rows:
        return [*lines, "none"]
    for row in rows:
        lines.append(
            f"timestamp={row.get('timestamp')} side={row.get('side')} "
            f"price={row.get('price')} size={row.get('size')} "
            f"asset_id={row.get('asset_id')} outcome={row.get('outcome')}"
        )
    return lines


def _render_health(row: dict[str, Any] | None) -> list[str]:
    lines = ["", "=== Recorder Health ==="]
    if row is None:
        return [*lines, "none"]
    for key in (
        "timestamp",
        "raw_ws_events_seen",
        "raw_ws_events_written",
        "malformed_ws_events",
        "raw_ws_write_failures",
        "last_ws_event_age_sec",
        "subscribed_asset_count",
        "ws_reconnect_count",
    ):
        lines.append(f"{key}={row.get(key)}")
    return lines


def _render_event_counts(counts: dict[str, int]) -> list[str]:
    lines = ["", "=== Raw Event Counts ==="]
    for event_type in EVENT_TYPES:
        lines.append(f"{event_type}={counts.get(event_type, 0)}")
    return lines


def _render_warnings(warnings: Sequence[str]) -> list[str]:
    lines = ["", "=== Warnings ==="]
    if not warnings:
        return [*lines, "none"]
    lines.extend(f"warning={warning}" for warning in warnings)
    return lines


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _value(row: dict[str, Any] | None, key: str) -> Any:
    return row.get(key) if row is not None else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_sec(now: datetime, timestamp: datetime) -> float:
    return max(0.0, (now - _utc(timestamp)).total_seconds())


def _age_text(now: datetime, timestamp: datetime | None) -> str:
    if timestamp is None:
        return "unknown"
    return f"{_age_sec(now, timestamp):.3f}"


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _asset_id_from_trade_id(trade_id: str) -> str | None:
    if ":" not in trade_id:
        return None
    asset_id = trade_id.rsplit(":", 1)[-1].strip()
    return asset_id or None


def _outcome_from_asset(
    asset_id: str | None,
    *,
    yes_token_id: str | None,
    no_token_id: str | None,
) -> str | None:
    if asset_id is None:
        return None
    if yes_token_id is not None and asset_id == yes_token_id:
        return "YES"
    if no_token_id is not None and asset_id == no_token_id:
        return "NO"
    return None

__all__ = [
    "DashboardData",
    "connect_read_only",
    "fetch_dashboard_data",
    "render_dashboard",
    "run_dashboard",
]
