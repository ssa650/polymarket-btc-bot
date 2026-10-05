from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Tuple
from urllib.parse import quote

from .models import parse_timestamp


def _is_open_status(status: str | None) -> bool:
    if status is None:
        return True
    return status.strip().lower() in {"open", "active"}


def _is_not_expired(close_time: str | None, now: datetime) -> bool:
    close = parse_timestamp(close_time)
    if close is None:
        return True
    return close > now.replace(tzinfo=timezone.utc)


def _is_started(start_time: str | None, now: datetime) -> bool:
    start = parse_timestamp(start_time)
    if start is None:
        return False
    return start <= now.replace(tzinfo=timezone.utc)


def _is_trade_time_within_market_window(
    trade_time: str | None,
    start_time: str | None,
    close_time: str | None,
) -> int:
    trade_ts = parse_timestamp(trade_time)
    if trade_ts is None:
        return 0
    start_ts = parse_timestamp(start_time)
    close_ts = parse_timestamp(close_time)
    if start_ts is not None and trade_ts < start_ts:
        return 0
    if close_ts is not None and trade_ts >= close_ts:
        return 0
    return 1


def _in_clause(values: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    placeholders = ", ".join("?" for _ in values)
    return f"market_id IN ({placeholders})", tuple(values)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


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


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _print_database_files(db_path: str) -> None:
    path = Path(db_path)
    print("=== Database Files ===")
    print(f"db_path={db_path}")
    print(f"db_file_size_bytes={_file_size(path)}")
    print(f"wal_file_size_bytes={_file_size(Path(f'{db_path}-wal'))}")
    print(f"shm_file_size_bytes={_file_size(Path(f'{db_path}-shm'))}")


def _print_raw_ws_persistence_config(raw_ws_events_enabled: bool | None) -> None:
    print("\n=== Raw WS Persistence Config ===")
    value = "unknown" if raw_ws_events_enabled is None else str(bool(raw_ws_events_enabled)).lower()
    print(f"raw_ws_events_persistence_enabled={value}")


def _table_count(conn: sqlite3.Connection, table: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int((row[0] if row is not None else 0) or 0)


def _row_text(row: sqlite3.Row, columns: Sequence[str]) -> str:
    parts = []
    for column in columns:
        value = row[column]
        parts.append(f"{column}={value}")
    return " | ".join(parts)


def _print_row_count_summary(conn: sqlite3.Connection) -> dict[str, int | None]:
    tables = (
        "markets",
        "market_snapshots",
        "order_book_levels",
        "features",
        "trades",
        "market_events",
        "raw_polymarket_events",
        "tick_size_changes",
        "best_bid_ask_updates",
        "btc_prices",
        "recorder_metrics",
    )
    counts = {table: _table_count(conn, table) for table in tables}
    print("=== Row Counts ===")
    for table, count in counts.items():
        value = "missing" if count is None else str(count)
        print(f"{table}={value}")
    return counts


def _print_empty_table_warnings(counts: dict[str, int | None]) -> None:
    print("\n=== Warnings ===")
    warnings = []
    for table, count in counts.items():
        if count is None:
            warnings.append(f"warning table_missing table={table}")
        elif count == 0:
            warnings.append(f"warning table_empty table={table}")
    if not warnings:
        print("none")
        return
    for warning in warnings:
        print(warning)


def _print_recent_markets(conn: sqlite3.Connection, limit: int = 8) -> None:
    print("\n=== Recent Markets ===")
    if not _table_exists(conn, "markets"):
        print("markets table missing")
        return
    rows = conn.execute(
        """
        SELECT
          market_id,
          tracking_state,
          COALESCE(phase, market_phase) AS phase,
          platform_status,
          yes_token_id,
          no_token_id,
          start_time,
          close_time,
          question
        FROM markets
        ORDER BY COALESCE(last_updated, created_at, close_time, start_time) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("none")
        return
    for row in rows:
        print(
            " | ".join(
                [
                    f"market_id={row['market_id']}",
                    f"tracking_state={row['tracking_state']}",
                    f"phase={row['phase']}",
                    f"platform_status={row['platform_status']}",
                    f"yes_token_id={row['yes_token_id']}",
                    f"no_token_id={row['no_token_id']}",
                    f"start_time={row['start_time']}",
                    f"close_time={row['close_time']}",
                    f"question={row['question']}",
                ]
            )
        )


def _print_latest_tracked_assets(conn: sqlite3.Connection) -> None:
    print("\n=== Last Tracked Assets ===")
    if not _table_exists(conn, "recorder_metrics"):
        print("recorder_metrics table missing")
        return
    required = ("subscribed_asset_count", "subscribed_asset_ids_json")
    if not all(_has_column(conn, "recorder_metrics", column) for column in required):
        print("subscribed asset metrics unavailable")
        return
    row = conn.execute(
        """
        SELECT
          timestamp,
          run_id,
          subscribed_asset_count,
          subscribed_asset_ids_json
        FROM recorder_metrics
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("none")
        return
    print(
        " | ".join(
            [
                f"timestamp={row['timestamp']}",
                f"run_id={row['run_id']}",
                f"subscribed_asset_count={row['subscribed_asset_count']}",
                f"subscribed_asset_ids_json={row['subscribed_asset_ids_json']}",
            ]
        )
    )


def _print_raw_event_summary(conn: sqlite3.Connection) -> None:
    print("\n=== Raw Polymarket Events ===")
    if not _table_exists(conn, "raw_polymarket_events"):
        print("raw_polymarket_events table missing")
        return
    rows = conn.execute(
        """
        SELECT
          COALESCE(event_type, 'unknown') AS event_type,
          COALESCE(parse_status, 'unknown') AS parse_status,
          COUNT(*) AS cnt
        FROM raw_polymarket_events
        GROUP BY COALESCE(event_type, 'unknown'), COALESCE(parse_status, 'unknown')
        ORDER BY cnt DESC, event_type ASC, parse_status ASC
        LIMIT 30
        """
    ).fetchall()
    if not rows:
        print("none")
        return
    for row in rows:
        print(
            f"event_type={row['event_type']} | "
            f"parse_status={row['parse_status']} | "
            f"count={int(row['cnt'] or 0)}"
        )


def _print_latest_recorder_metrics(conn: sqlite3.Connection, limit: int = 5) -> None:
    print("\n=== Latest Recorder Metrics ===")
    if not _table_exists(conn, "recorder_metrics"):
        print("recorder_metrics table missing")
        return
    columns = [
        "timestamp",
        "run_id",
        "tracked_markets_count",
        "snapshots_written",
        "rows_inserted",
        "ws_reconnect_count",
        "raw_ws_events_seen",
        "raw_ws_events_written",
        "malformed_ws_events",
        "raw_ws_write_failures",
        "last_ws_event_age_sec",
        "subscribed_asset_count",
        "subscribed_asset_ids_json",
    ]
    selected_columns = [
        column
        for column in columns
        if column in {"timestamp", "run_id"} or _has_column(conn, "recorder_metrics", column)
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(selected_columns)}
        FROM recorder_metrics
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("none")
        return
    for row in rows:
        print(_row_text(row, selected_columns))


def _print_latest_btc_price(conn: sqlite3.Connection) -> None:
    print("\n=== Latest BTC Price ===")
    if not _table_exists(conn, "btc_prices"):
        print("btc_prices table missing")
        return
    columns = (
        "run_id",
        "source",
        "price",
        "exchange_timestamp",
        "local_arrival_ns",
        "local_arrival_iso",
    )
    if not all(_has_column(conn, "btc_prices", column) for column in columns):
        print("btc_prices columns unavailable")
        return
    row = conn.execute(
        """
        SELECT
          run_id,
          source,
          price,
          exchange_timestamp,
          local_arrival_ns,
          local_arrival_iso
        FROM btc_prices
        ORDER BY local_arrival_ns DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("none")
        return
    print(_row_text(row, columns))


def _print_latest_btc_prices_by_source(conn: sqlite3.Connection) -> None:
    print("\n=== Latest BTC Price By Source ===")
    if not _table_exists(conn, "btc_prices"):
        print("btc_prices table missing")
        return
    columns = (
        "source",
        "price",
        "exchange_timestamp",
        "local_arrival_ns",
        "local_arrival_iso",
    )
    if not all(_has_column(conn, "btc_prices", column) for column in columns):
        print("btc_prices columns unavailable")
        return
    rows = conn.execute(
        """
        SELECT
          p.source,
          p.price,
          p.exchange_timestamp,
          p.local_arrival_ns,
          p.local_arrival_iso
        FROM btc_prices p
        JOIN (
          SELECT source, MAX(local_arrival_ns) AS max_arrival_ns
          FROM btc_prices
          GROUP BY source
        ) latest
          ON p.source = latest.source
         AND p.local_arrival_ns = latest.max_arrival_ns
        ORDER BY p.source ASC, p.id DESC
        """
    ).fetchall()
    if not rows:
        print("none")
        return
    now = datetime.now(timezone.utc)
    seen_sources: set[str] = set()
    stale_sources: list[tuple[str, float]] = []
    for row in rows:
        source = str(row["source"])
        if source in seen_sources:
            continue
        seen_sources.add(source)
        arrival = parse_timestamp(row["local_arrival_iso"])
        sample_age_sec = (
            round((now - arrival).total_seconds(), 3)
            if arrival is not None
            else None
        )
        if sample_age_sec is not None and sample_age_sec > 3.0:
            stale_sources.append((source, sample_age_sec))
        print(
            " | ".join(
                [
                    f"source={source}",
                    f"price={row['price']}",
                    f"exchange_timestamp={row['exchange_timestamp']}",
                    f"local_arrival_ns={row['local_arrival_ns']}",
                    f"local_arrival_iso={row['local_arrival_iso']}",
                    f"sample_age_sec={sample_age_sec}",
                ]
            )
        )
    for source, sample_age_sec in stale_sources:
        print(
            "warning btc_price_source_stale "
            f"source={source} sample_age_sec={sample_age_sec}"
        )
    expected_sources = {
        "polymarket_rtds_chainlink",
        "polymarket_rtds_binance",
    }
    if expected_sources.issubset({source for source, _age in stale_sources}):
        print("warning both_btc_sources_stale")


def _print_recent_table(
    conn: sqlite3.Connection,
    *,
    title: str,
    table: str,
    columns: Sequence[str],
    order_by: str,
    limit: int = 5,
) -> None:
    print(f"\n=== {title} ===")
    if not _table_exists(conn, table):
        print(f"{table} table missing")
        return
    selected_columns = [column for column in columns if _has_column(conn, table, column)]
    if not selected_columns:
        print("no compatible columns")
        return
    rows = conn.execute(
        f"""
        SELECT {", ".join(selected_columns)}
        FROM {table}
        ORDER BY {order_by} DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("none")
        return
    for row in rows:
        print(_row_text(row, selected_columns))


def _print_resolution_mapping_status(conn: sqlite3.Connection) -> None:
    print("\n=== Resolution Mapping ===")
    if not _table_exists(conn, "market_events") or not _table_exists(conn, "markets"):
        print("resolution mapping unavailable")
        return

    resolution_columns = (
        "resolved",
        "resolved_at",
        "winning_asset_id",
        "winning_outcome",
    )
    columns_available = all(
        _has_column(conn, "markets", column) for column in resolution_columns
    )
    print(f"resolution_columns_available={1 if columns_available else 0}")

    event_rows = conn.execute(
        """
        SELECT run_id, timestamp, market_id, details
        FROM market_events
        WHERE event_type = 'market_resolved'
        ORDER BY run_id ASC, timestamp ASC, market_id ASC
        """
    ).fetchall()
    print(f"market_resolved_events={len(event_rows)}")

    market_rows = conn.execute(
        """
        SELECT run_id, market_id, condition_id, yes_token_id, no_token_id
        FROM markets
        """
    ).fetchall()
    by_run: dict[str, list[sqlite3.Row]] = {}
    for row in market_rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)

    def _json_obj(value: object) -> dict[str, object]:
        if not value:
            return {}
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _as_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _list_str(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            value = decoded
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    def _looks_like_condition_hash(value: str | None) -> bool:
        return bool(value and value.startswith("0x") and len(value) > 20)

    mapped = 0
    unmapped = 0
    condition_hash_events = 0
    condition_hash_to_numeric = 0
    for event in event_rows:
        details = _json_obj(event["details"])
        event_market_id = _as_str(event["market_id"])
        numeric_id = _as_str(
            details.get("id")
            or details.get("market_id")
            or details.get("marketId")
        )
        condition_id = _as_str(
            details.get("condition_id")
            or details.get("conditionId")
            or details.get("market")
        )
        winning_asset_id = _as_str(
            details.get("winning_asset_id")
            or details.get("winningAssetId")
            or details.get("winning_asset")
        )
        asset_ids = set(
            _list_str(details.get("assets_ids") or details.get("asset_ids"))
        )
        if winning_asset_id is not None:
            asset_ids.add(winning_asset_id)
        identifiers = {
            value
            for value in (event_market_id, numeric_id, condition_id)
            if value is not None
        }
        if _looks_like_condition_hash(event_market_id):
            condition_hash_events += 1

        candidates = []
        for market in by_run.get(str(event["run_id"]), []):
            if market["market_id"] in identifiers:
                candidates.append(market)
                continue
            if market["condition_id"] in identifiers:
                candidates.append(market)
                continue
            if asset_ids and (
                market["yes_token_id"] in asset_ids
                or market["no_token_id"] in asset_ids
            ):
                candidates.append(market)
        if not candidates:
            unmapped += 1
            continue
        mapped += 1
        if _looks_like_condition_hash(event_market_id) and any(
            not _looks_like_condition_hash(_as_str(candidate["market_id"]))
            for candidate in candidates
        ):
            condition_hash_to_numeric += 1

    print(f"mapped_resolution_events={mapped}")
    print(f"unmapped_resolution_events={unmapped}")
    print(f"condition_hash_resolution_events={condition_hash_events}")
    print(f"condition_hash_events_mapped_to_numeric_market={condition_hash_to_numeric}")

    if not columns_available:
        print("resolved_market_rows=unavailable")
        print("resolved_market_rows_with_winner=unavailable")
        return
    row = conn.execute(
        """
        SELECT
          COUNT(*) AS resolved_count,
          SUM(
            CASE
              WHEN winning_asset_id IS NOT NULL OR winning_outcome IS NOT NULL
              THEN 1 ELSE 0
            END
          ) AS with_winner
        FROM markets
        WHERE COALESCE(resolved, 0) = 1
        """
    ).fetchone()
    print(f"resolved_market_rows={int(row['resolved_count'] or 0)}")
    print(f"resolved_market_rows_with_winner={int(row['with_winner'] or 0)}")



def _nonnull_stats(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    where_sql: str = "",
    params: tuple[object, ...] = (),
) -> Tuple[int, int, float]:
    query = f"""
    SELECT
      COUNT(*) AS total,
      SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END) AS non_null
    FROM {table}
    """
    if where_sql:
        query = f"{query} WHERE {where_sql}"
    total, non_null = conn.execute(query, params).fetchone()
    total = int(total or 0)
    non_null = int(non_null or 0)
    pct = (100.0 * non_null / total) if total > 0 else 0.0
    return total, non_null, pct



def print_diagnostics(
    db_path: str,
    lookahead_sec: int = 7200,
    raw_ws_events_enabled: bool | None = None,
) -> None:
    try:
        conn = _connect_read_only(db_path)
    except FileNotFoundError:
        print("=== Database ===")
        print(f"database_missing path={db_path}")
        return

    now = datetime.now(timezone.utc)
    _ = lookahead_sec  # kept for CLI compatibility

    _print_database_files(db_path)
    _print_raw_ws_persistence_config(raw_ws_events_enabled)
    counts = _print_row_count_summary(conn)
    _print_empty_table_warnings(counts)
    _print_recent_markets(conn)
    _print_latest_tracked_assets(conn)
    _print_raw_event_summary(conn)
    _print_latest_recorder_metrics(conn)
    _print_latest_btc_price(conn)
    _print_latest_btc_prices_by_source(conn)
    _print_recent_table(
        conn,
        title="Recent Trades",
        table="trades",
        columns=("timestamp", "market_id", "trade_id", "price", "size", "side"),
        order_by="timestamp",
    )
    _print_recent_table(
        conn,
        title="Recent Snapshots",
        table="market_snapshots",
        columns=(
            "timestamp",
            "market_id",
            "best_bid_yes",
            "best_ask_yes",
            "spread_yes",
            "last_trade_price",
            "has_orderbook",
            "has_trade_data",
        ),
        order_by="timestamp",
    )
    _print_recent_table(
        conn,
        title="Recent Tick Size Changes",
        table="tick_size_changes",
        columns=(
            "timestamp",
            "market_id",
            "condition_id",
            "asset_id",
            "old_tick_size",
            "new_tick_size",
        ),
        order_by="timestamp",
    )
    _print_recent_table(
        conn,
        title="Recent Best Bid Ask Updates",
        table="best_bid_ask_updates",
        columns=("timestamp", "market_id", "condition_id", "asset_id", "best_bid", "best_ask", "spread"),
        order_by="timestamp",
    )
    _print_recent_table(
        conn,
        title="Recent Market Events",
        table="market_events",
        columns=("timestamp", "market_id", "event_type", "details"),
        order_by="timestamp",
    )
    _print_resolution_mapping_status(conn)

    required_existing_diagnostic_tables = (
        "markets",
        "market_snapshots",
        "trades",
        "features",
        "market_events",
        "recorder_metrics",
    )
    missing_required = [
        table for table in required_existing_diagnostic_tables if counts.get(table) is None
    ]
    if missing_required:
        print("\n=== Legacy Diagnostics ===")
        print(f"skipped_missing_tables={','.join(missing_required)}")
        conn.close()
        return

    selected_rows = conn.execute(
        """
        SELECT
          market_id,
          question,
          yes_token_id,
          no_token_id,
          platform_status,
          tracking_state,
          phase,
          start_time,
          close_time
        FROM markets
        WHERE tracking_state = 'selected_active'
        ORDER BY start_time ASC
        """
    ).fetchall()

    active_selected_rows = [
        row
        for row in selected_rows
        if (row["phase"] or "").strip().lower() == "active"
        and _is_open_status(row["platform_status"])
        and _is_started(row["start_time"], now)
    ]

    print("=== Current Selected Active Market ===")
    print(f"selected_active_count={len(active_selected_rows)}")
    print(f"exactly_one_selected_active={1 if len(active_selected_rows) == 1 else 0}")
    rows_to_show = active_selected_rows[:1] if active_selected_rows else selected_rows[:1]
    selected_effective_duration_sec: float | None = None
    selected_passes_strict_5m = 0
    for row in rows_to_show:
        start_ts = parse_timestamp(row["start_time"])
        close_ts = parse_timestamp(row["close_time"])
        if start_ts is not None and close_ts is not None:
            selected_effective_duration_sec = (close_ts - start_ts).total_seconds()
            selected_passes_strict_5m = 1 if 240 <= selected_effective_duration_sec <= 360 else 0
        print(
            " | ".join(
                [
                    f"market_id={row['market_id']}",
                    f"platform_status={row['platform_status']}",
                    f"tracking_state={row['tracking_state']}",
                    f"phase={row['phase']}",
                    f"yes_token_id={row['yes_token_id']}",
                    f"no_token_id={row['no_token_id']}",
                    f"start_time={row['start_time']}",
                    f"close_time={row['close_time']}",
                    f"question={row['question']}",
                ]
            )
        )
    print(f"selected_effective_duration_sec={selected_effective_duration_sec}")
    print(f"selected_passes_strict_5m_validation={selected_passes_strict_5m}")

    tracked_market_ids = [str(row["market_id"]) for row in rows_to_show]

    print("\n=== Snapshot Completeness ===")
    if tracked_market_ids:
        where_sql, where_params = _in_clause(tracked_market_ids)
        snapshot_total = (
            conn.execute(
                f"SELECT COUNT(*) FROM market_snapshots WHERE {where_sql}", where_params
            ).fetchone()[0]
            or 0
        )
    else:
        where_sql, where_params = "1=0", ()
        snapshot_total = 0
    print(f"total_snapshots={snapshot_total}")
    for column in ("best_bid_yes", "mid_price_yes", "last_trade_price"):
        total, non_null, pct = _nonnull_stats(
            conn,
            "market_snapshots",
            column,
            where_sql=where_sql,
            params=where_params,
        )
        print(f"{column}_non_null={non_null}/{total} ({pct:.2f}%)")
    if tracked_market_ids:
        where_sql, where_params = _in_clause(tracked_market_ids)
        total, has_orderbook, pct_orderbook = _nonnull_stats(
            conn,
            "market_snapshots",
            "CASE WHEN has_orderbook = 1 THEN 1 END",
            where_sql=where_sql,
            params=where_params,
        )
        _total, has_trade, pct_trade = _nonnull_stats(
            conn,
            "market_snapshots",
            "CASE WHEN has_trade_data = 1 THEN 1 END",
            where_sql=where_sql,
            params=where_params,
        )
    else:
        total, has_orderbook, pct_orderbook = (0, 0, 0.0)
        _total, has_trade, pct_trade = (0, 0, 0.0)
    print(f"has_orderbook_eq_1={has_orderbook}/{total} ({pct_orderbook:.2f}%)")
    print(f"has_trade_data_eq_1={has_trade}/{_total} ({pct_trade:.2f}%)")

    print("\n=== Snapshot Trade Integrity ===")
    integrity_counts = conn.execute(
        """
        SELECT
            SUM(
                CASE
                    WHEN s.last_trade_time IS NOT NULL
                     AND m.start_time IS NOT NULL
                     AND datetime(s.last_trade_time) < datetime(m.start_time)
                    THEN 1 ELSE 0
                END
            ) AS before_start,
            SUM(
                CASE
                    WHEN s.last_trade_time IS NOT NULL
                     AND datetime(s.last_trade_time) > datetime(s.timestamp)
                    THEN 1 ELSE 0
                END
            ) AS after_snapshot
        FROM market_snapshots s
        LEFT JOIN markets m ON m.market_id = s.market_id
        """
    ).fetchone()
    before_start = int((integrity_counts["before_start"] if integrity_counts is not None else 0) or 0)
    after_snapshot = int((integrity_counts["after_snapshot"] if integrity_counts is not None else 0) or 0)
    print(f"rows_last_trade_before_market_start={before_start}")
    print(f"rows_last_trade_after_snapshot_timestamp={after_snapshot}")

    by_market = conn.execute(
        """
        SELECT
            s.market_id,
            SUM(
                CASE
                    WHEN s.last_trade_time IS NOT NULL
                     AND m.start_time IS NOT NULL
                     AND datetime(s.last_trade_time) < datetime(m.start_time)
                    THEN 1 ELSE 0
                END
            ) AS before_start,
            SUM(
                CASE
                    WHEN s.last_trade_time IS NOT NULL
                     AND datetime(s.last_trade_time) > datetime(s.timestamp)
                    THEN 1 ELSE 0
                END
            ) AS after_snapshot
        FROM market_snapshots s
        LEFT JOIN markets m ON m.market_id = s.market_id
        GROUP BY s.market_id
        HAVING before_start > 0 OR after_snapshot > 0
        ORDER BY (before_start + after_snapshot) DESC, s.market_id ASC
        LIMIT 20
        """
    ).fetchall()
    for row in by_market:
        print(
            f"invalid_snapshot_trade_rows market_id={row['market_id']} "
            f"before_start={int(row['before_start'] or 0)} "
            f"after_snapshot={int(row['after_snapshot'] or 0)}"
        )

    print("\n=== Latest Snapshot Trade State ===")
    if tracked_market_ids:
        for market_id in tracked_market_ids:
            snapshot = conn.execute(
                """
                SELECT timestamp, last_trade_time, last_trade_price, last_trade_size, has_trade_data
                FROM market_snapshots
                WHERE market_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (market_id,),
            ).fetchone()
            market_row = conn.execute(
                """
                SELECT start_time, close_time
                FROM markets
                WHERE market_id = ?
                LIMIT 1
                """,
                (market_id,),
            ).fetchone()
            if snapshot is None or market_row is None:
                print(f"latest_snapshot_trade_state market_id={market_id} none")
                continue
            within_window = _is_trade_time_within_market_window(
                snapshot["last_trade_time"],
                market_row["start_time"],
                market_row["close_time"],
            )
            print(
                "latest_snapshot_trade_state "
                f"market_id={market_id} "
                f"snapshot_ts={snapshot['timestamp']} "
                f"last_trade_time={snapshot['last_trade_time']} "
                f"last_trade_price={snapshot['last_trade_price']} "
                f"last_trade_size={snapshot['last_trade_size']} "
                f"has_trade_data={snapshot['has_trade_data']} "
                f"trade_time_within_market_window={within_window}"
            )
    else:
        print("latest_snapshot_trade_state none")

    print("\n=== Trade Ingestion ===")
    total_trades = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] or 0)
    print(f"total_trade_rows_ingested={total_trades}")
    if tracked_market_ids:
        for market_id in tracked_market_ids:
            row = conn.execute(
                """
                SELECT timestamp, trade_id, price, size, side
                FROM trades
                WHERE market_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (market_id,),
            ).fetchone()
            if row is None:
                print(f"latest_trade market_id={market_id} none")
            else:
                print(
                    "latest_trade "
                    f"market_id={market_id} "
                    f"timestamp={row['timestamp']} "
                    f"trade_id={row['trade_id']} "
                    f"price={row['price']} "
                    f"size={row['size']} "
                    f"side={row['side']}"
                )
            recent_counts = conn.execute(
                """
                SELECT LOWER(COALESCE(side, 'unknown')) AS side, COUNT(*) AS cnt
                FROM trades
                WHERE market_id = ?
                  AND timestamp >= datetime('now', '-300 seconds')
                GROUP BY LOWER(COALESCE(side, 'unknown'))
                ORDER BY cnt DESC
                """,
                (market_id,),
            ).fetchall()
            counts_text = ", ".join(f"{r['side']}={r['cnt']}" for r in recent_counts) or "none"
            print(f"recent_5m_trade_side_counts market_id={market_id} {counts_text}")

    print("\n=== Feature Completeness ===")
    if tracked_market_ids:
        where_sql, where_params = _in_clause(tracked_market_ids)
        feature_total = (
            conn.execute(
                f"SELECT COUNT(*) FROM features WHERE {where_sql}", where_params
            ).fetchone()[0]
            or 0
        )
    else:
        where_sql, where_params = "1=0", ()
        feature_total = 0
    print(f"total_features={feature_total}")
    for column in (
        "price_change_1s",
        "price_change_10s",
        "velocity_5s",
        "velocity_30s",
        "rolling_mean_60s",
        "orderbook_imbalance_yes",
        "buy_volume_5s",
        "sell_volume_5s",
        "volume_spike_ratio",
    ):
        total, non_null, pct = _nonnull_stats(
            conn,
            "features",
            column,
            where_sql=where_sql,
            params=where_params,
        )
        print(f"{column}_non_null={non_null}/{total} ({pct:.2f}%)")
    trade_feature_non_null_rows = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM features
            WHERE {where_sql}
              AND (
                  buy_volume_5s IS NOT NULL OR
                  sell_volume_5s IS NOT NULL OR
                  net_trade_flow_5s IS NOT NULL OR
                  trade_flow_ratio_5s IS NOT NULL
              )
            """,
            where_params,
        ).fetchone()[0]
        or 0
    )
    print(f"trade_feature_rows_any_non_null={trade_feature_non_null_rows}")

    print("\n=== Event Integrity ===")
    duplicate_rows = conn.execute(
        """
        SELECT market_id, event_type, COUNT(*) AS cnt
        FROM market_events
        GROUP BY market_id, event_type
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC, market_id ASC, event_type ASC
        LIMIT 200
        """
    ).fetchall()
    print(f"duplicate_market_event_groups={len(duplicate_rows)}")
    for row in duplicate_rows:
        print(
            f"market_id={row['market_id']} "
            f"event_type={row['event_type']} "
            f"count={int(row['cnt'])}"
        )
    if tracked_market_ids:
        market_id = tracked_market_ids[0]
        open_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM market_events
                WHERE market_id = ? AND event_type = 'market_opened'
                """,
                (market_id,),
            ).fetchone()[0]
            or 0
        )
        print(f"active_market_open_event_count market_id={market_id} count={open_count}")
    terminal_events_resolved = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM market_events e
            JOIN markets m ON m.market_id = e.market_id
            WHERE e.event_type IN ('market_closed', 'market_resolved')
              AND COALESCE(m.phase, m.market_phase) = 'resolved'
            """
        ).fetchone()[0]
        or 0
    )
    print(f"terminal_event_count_for_resolved_markets={terminal_events_resolved}")

    print("\n=== Market Integrity ===")
    by_tracking = conn.execute(
        """
        SELECT tracking_state, COUNT(*) AS cnt
        FROM markets
        GROUP BY tracking_state
        ORDER BY cnt DESC
        """
    ).fetchall()
    for row in by_tracking:
        print(f"tracking_state_count {row['tracking_state']}={row['cnt']}")
    by_phase = conn.execute(
        """
        SELECT COALESCE(phase, market_phase) AS phase_value, COUNT(*) AS cnt
        FROM markets
        GROUP BY COALESCE(phase, market_phase)
        ORDER BY cnt DESC
        """
    ).fetchall()
    for row in by_phase:
        print(f"phase_count {row['phase_value']}={row['cnt']}")
    selected_active_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM markets WHERE tracking_state = 'selected_active'"
        ).fetchone()[0]
        or 0
    )
    active_phase_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM markets WHERE COALESCE(phase, market_phase) = 'active'"
        ).fetchone()[0]
        or 0
    )
    print(f"selected_active_total={selected_active_count}")
    invalid_active_past_close = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM markets
            WHERE COALESCE(phase, market_phase) = 'active'
              AND close_time IS NOT NULL
              AND datetime(close_time) <= datetime('now')
            """
        ).fetchone()[0]
        or 0
    )
    print(f"invalid_active_markets_past_close={invalid_active_past_close}")
    alert = 1 if active_phase_count > 0 and selected_active_count != 1 else 0
    print(f"selected_active_integrity_alert={alert}")

    print("\n=== Discovery Target Validation ===")
    has_strict_col = _has_column(conn, "recorder_metrics", "discovery_strict_5m_candidates")
    has_broad_col = _has_column(conn, "recorder_metrics", "discovery_broad_btc_candidates")
    has_fallback_col = _has_column(conn, "recorder_metrics", "discovery_fallback_used")
    if has_strict_col and has_broad_col and has_fallback_col:
        latest_metrics = conn.execute(
            """
            SELECT
              discovery_strict_5m_candidates,
              discovery_broad_btc_candidates,
              discovery_fallback_used
            FROM recorder_metrics
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_metrics is None:
            print("strict_5m_candidate_count=0")
            print("broad_btc_candidate_count=0")
            print("fallback_used=0")
        else:
            print(f"strict_5m_candidate_count={int(latest_metrics[0] or 0)}")
            print(f"broad_btc_candidate_count={int(latest_metrics[1] or 0)}")
            print(f"fallback_used={int(latest_metrics[2] or 0)}")
    else:
        print("strict_5m_candidate_count=unavailable")
        print("broad_btc_candidate_count=unavailable")
        print("fallback_used=unavailable")

    conn.close()
