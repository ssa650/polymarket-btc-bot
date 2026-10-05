from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .db import run_sqlite_integrity_checks


def _fetch_available_run_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT run_id
        FROM recorder_metrics
        WHERE run_id IS NOT NULL AND run_id <> ''
        ORDER BY timestamp DESC
        """
    ).fetchall()
    run_ids = [str(row[0]) for row in rows if row[0] is not None]
    if run_ids:
        return run_ids

    rows = conn.execute(
        """
        SELECT DISTINCT run_id
        FROM markets
        WHERE run_id IS NOT NULL AND run_id <> ''
        ORDER BY created_at DESC
        """
    ).fetchall()
    return [str(row[0]) for row in rows if row[0] is not None]


def _resolve_run_ids(
    conn: sqlite3.Connection,
    run_ids: Optional[Sequence[str]],
    merge_runs: bool,
) -> list[str]:
    if run_ids:
        return [str(run_id) for run_id in run_ids]

    available = _fetch_available_run_ids(conn)
    available = [run_id for run_id in available if run_id != "legacy"]
    if not available:
        available = _fetch_available_run_ids(conn)

    if not available:
        return []

    if merge_runs:
        return available
    return [available[0]]


def _placeholders(items: Sequence[object]) -> str:
    return ", ".join("?" for _ in items)


def _to_pylist(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


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


def _feature_columns() -> list[str]:
    return [
        "price_change_1s",
        "price_change_10s",
        "price_change_60s",
        "velocity_5s",
        "velocity_30s",
        "acceleration_5s",
        "rolling_mean_60s",
        "distance_from_rolling_mean_60s",
        "total_bid_liquidity_yes",
        "total_ask_liquidity_yes",
        "liquidity_change_bid_5s",
        "orderbook_imbalance_yes",
        "buy_volume_5s",
        "sell_volume_5s",
        "net_trade_flow_5s",
        "trade_flow_ratio_5s",
        "rolling_volatility_10s",
        "rolling_volatility_60s",
        "volume_delta_1s",
        "avg_volume_60s",
        "volume_spike_ratio",
        "largest_bid_wall_size_yes",
        "largest_ask_wall_size_yes",
        "distance_to_bid_wall",
        "time_since_market_created",
        "time_until_resolution",
        "is_price_jump",
    ]


def export_training_parquet(
    db_path: str,
    output_path: str,
    run_ids: Optional[Sequence[str]] = None,
    merge_runs: bool = False,
    require_trades: bool = False,
    require_full_orderbook_depth: bool = False,
    exclude_gap_affected: bool = True,
    label_horizon_sec: int = 60,
) -> dict[str, Any]:
    integrity = run_sqlite_integrity_checks(db_path)
    if not integrity.get("ok", False):
        raise RuntimeError(
            "Refusing training export from corrupted SQLite file. "
            f"Integrity report: {json.dumps(integrity, sort_keys=True)}"
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        resolved_run_ids = _resolve_run_ids(conn, run_ids=run_ids, merge_runs=merge_runs)
        if not resolved_run_ids:
            raise RuntimeError("No run_id found for export")

        placeholders = _placeholders(resolved_run_ids)
        params: list[object] = list(resolved_run_ids)

        where_clauses = [
            f"s.run_id IN ({placeholders})",
            "m.strict_validation_passed = 1",
            "s.strict_validation_passed = 1",
            "f.feature_ready = 1",
        ]
        if require_trades:
            where_clauses.append("s.has_trade_data = 1")
        if require_full_orderbook_depth:
            where_clauses.append("COALESCE(s.is_partial_orderbook, 0) = 0")
        if exclude_gap_affected:
            where_clauses.append("COALESCE(s.is_gap_affected, 0) = 0")
            where_clauses.append("COALESCE(f.is_gap_affected, 0) = 0")

        where_sql = " AND ".join(where_clauses)

        feature_cols = _feature_columns()
        feature_select_sql = ",\n                ".join(f"f.{column}" for column in feature_cols)
        has_btc_prices = _table_exists(conn, "btc_prices")
        has_tick_size_changes = _table_exists(conn, "tick_size_changes")
        has_order_book_levels = _table_exists(conn, "order_book_levels")
        has_market_resolution_columns = all(
            _has_column(conn, "markets", column)
            for column in (
                "resolved",
                "resolved_at",
                "winning_asset_id",
                "winning_outcome",
            )
        )

        if has_btc_prices:
            btc_join_sql = """
            LEFT JOIN btc_prices btc
              ON btc.id = (
                    SELECT b.id
                    FROM btc_prices b
                    WHERE b.run_id = s.run_id
                      AND datetime(b.local_arrival_iso) <= datetime(s.timestamp)
                    ORDER BY b.local_arrival_ns DESC, b.id DESC
                    LIMIT 1
                 )
            """
            btc_select_sql = """
                btc.price AS btc_price,
                btc.exchange_timestamp AS btc_exchange_timestamp,
                btc.local_arrival_ns AS btc_local_arrival_ns,
                CASE
                    WHEN btc.local_arrival_iso IS NULL THEN NULL
                    ELSE MAX(
                        0.0,
                        (julianday(s.timestamp) - julianday(btc.local_arrival_iso)) * 86400.0
                    )
                END AS btc_sample_age_sec,
                btc.source AS btc_source,
                CASE WHEN btc.id IS NULL THEN 0 ELSE 1 END AS has_btc_price,
            """
        else:
            btc_join_sql = ""
            btc_select_sql = """
                NULL AS btc_price,
                NULL AS btc_exchange_timestamp,
                NULL AS btc_local_arrival_ns,
                NULL AS btc_sample_age_sec,
                NULL AS btc_source,
                0 AS has_btc_price,
            """

        if has_tick_size_changes:
            tick_join_sql = """
            LEFT JOIN tick_size_changes tick_yes
              ON tick_yes.id = (
                    SELECT t.id
                    FROM tick_size_changes t
                    WHERE t.run_id = s.run_id
                      AND t.asset_id = m.yes_token_id
                      AND datetime(t.timestamp) <= datetime(s.timestamp)
                    ORDER BY datetime(t.timestamp) DESC, t.id DESC
                    LIMIT 1
                 )
            LEFT JOIN tick_size_changes tick_no
              ON tick_no.id = (
                    SELECT t.id
                    FROM tick_size_changes t
                    WHERE t.run_id = s.run_id
                      AND t.asset_id = m.no_token_id
                      AND datetime(t.timestamp) <= datetime(s.timestamp)
                    ORDER BY datetime(t.timestamp) DESC, t.id DESC
                    LIMIT 1
                 )
            LEFT JOIN tick_size_changes tick_any
              ON tick_any.id = (
                    SELECT t.id
                    FROM tick_size_changes t
                    WHERE t.run_id = s.run_id
                      AND t.asset_id IN (m.yes_token_id, m.no_token_id)
                      AND datetime(t.timestamp) <= datetime(s.timestamp)
                    ORDER BY datetime(t.timestamp) DESC, t.id DESC
                    LIMIT 1
                 )
            """
            tick_select_sql = """
                tick_yes.new_tick_size AS latest_tick_size_yes,
                tick_no.new_tick_size AS latest_tick_size_no,
                CASE
                    WHEN tick_any.timestamp IS NULL THEN NULL
                    ELSE MAX(
                        0.0,
                        (julianday(s.timestamp) - julianday(tick_any.timestamp)) * 86400.0
                    )
                END AS tick_size_age_sec,
            """
        else:
            tick_join_sql = ""
            tick_select_sql = """
                NULL AS latest_tick_size_yes,
                NULL AS latest_tick_size_no,
                NULL AS tick_size_age_sec,
            """

        if has_order_book_levels:
            depth_select_sql = """
                COALESCE((
                    SELECT json_group_array(json_object('price', price, 'size', size))
                    FROM (
                        SELECT obl.price AS price, obl.size AS size
                        FROM order_book_levels obl
                        WHERE obl.run_id = s.run_id
                          AND obl.market_id = s.market_id
                          AND obl.timestamp = s.timestamp
                          AND upper(obl.outcome_side) = 'YES'
                          AND lower(obl.book_side) IN ('bid', 'bids')
                        ORDER BY obl.price DESC, obl.level ASC
                    )
                ), '[]') AS yes_bids_json,
                COALESCE((
                    SELECT json_group_array(json_object('price', price, 'size', size))
                    FROM (
                        SELECT obl.price AS price, obl.size AS size
                        FROM order_book_levels obl
                        WHERE obl.run_id = s.run_id
                          AND obl.market_id = s.market_id
                          AND obl.timestamp = s.timestamp
                          AND upper(obl.outcome_side) = 'YES'
                          AND lower(obl.book_side) IN ('ask', 'asks')
                        ORDER BY obl.price ASC, obl.level ASC
                    )
                ), '[]') AS yes_asks_json,
                COALESCE((
                    SELECT json_group_array(json_object('price', price, 'size', size))
                    FROM (
                        SELECT obl.price AS price, obl.size AS size
                        FROM order_book_levels obl
                        WHERE obl.run_id = s.run_id
                          AND obl.market_id = s.market_id
                          AND obl.timestamp = s.timestamp
                          AND upper(obl.outcome_side) = 'NO'
                          AND lower(obl.book_side) IN ('bid', 'bids')
                        ORDER BY obl.price DESC, obl.level ASC
                    )
                ), '[]') AS no_bids_json,
                COALESCE((
                    SELECT json_group_array(json_object('price', price, 'size', size))
                    FROM (
                        SELECT obl.price AS price, obl.size AS size
                        FROM order_book_levels obl
                        WHERE obl.run_id = s.run_id
                          AND obl.market_id = s.market_id
                          AND obl.timestamp = s.timestamp
                          AND upper(obl.outcome_side) = 'NO'
                          AND lower(obl.book_side) IN ('ask', 'asks')
                        ORDER BY obl.price ASC, obl.level ASC
                    )
                ), '[]') AS no_asks_json,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM order_book_levels obl
                        WHERE obl.run_id = s.run_id
                          AND obl.market_id = s.market_id
                          AND obl.timestamp = s.timestamp
                        LIMIT 1
                    )
                    THEN 1 ELSE 0
                END AS has_depth,
            """
        else:
            depth_select_sql = """
                '[]' AS yes_bids_json,
                '[]' AS yes_asks_json,
                '[]' AS no_bids_json,
                '[]' AS no_asks_json,
                0 AS has_depth,
            """

        if has_market_resolution_columns:
            resolution_select_sql = """
                COALESCE(m.resolved, 0) AS resolved,
                m.resolved_at AS resolved_at,
                m.winning_asset_id,
                m.winning_outcome,
                CASE
                    WHEN COALESCE(m.resolved, 0) = 1
                     AND (
                            m.winning_asset_id IS NOT NULL
                         OR m.winning_outcome IS NOT NULL
                         )
                    THEN 1 ELSE 0
                END AS label_available,
                CASE
                    WHEN COALESCE(m.resolved, 0) <> 1
                      OR (m.winning_asset_id IS NULL AND m.winning_outcome IS NULL)
                    THEN NULL
                    WHEN m.winning_asset_id = m.yes_token_id
                      OR upper(COALESCE(m.winning_outcome, '')) IN ('YES', 'Y', 'UP', 'TRUE')
                    THEN 1 ELSE 0
                END AS yes_won,
                CASE
                    WHEN COALESCE(m.resolved, 0) <> 1
                      OR (m.winning_asset_id IS NULL AND m.winning_outcome IS NULL)
                    THEN NULL
                    WHEN m.winning_asset_id = m.no_token_id
                      OR upper(COALESCE(m.winning_outcome, '')) IN ('NO', 'N', 'DOWN', 'FALSE')
                    THEN 1 ELSE 0
                END AS no_won,
            """
        else:
            resolution_select_sql = """
                CASE
                    WHEN COALESCE(m.phase, m.market_phase) = 'resolved' THEN 1 ELSE 0
                END AS resolved,
                NULL AS resolved_at,
                NULL AS winning_asset_id,
                NULL AS winning_outcome,
                0 AS label_available,
                NULL AS yes_won,
                NULL AS no_won,
            """

        rows = conn.execute(
            f"""
            SELECT
                (s.run_id || ':' || s.market_id) AS sequence_id,
                s.run_id,
                s.market_id,
                s.timestamp,
                {int(label_horizon_sec)} AS label_horizon_sec,
                'mid_price_yes' AS label_target,
                m.start_time AS market_start_time,
                m.close_time AS market_close_time,
                CASE
                    WHEN m.start_time IS NULL THEN NULL
                    ELSE (julianday(s.timestamp) - julianday(m.start_time)) * 86400.0
                END AS seconds_since_market_start,
                CASE
                    WHEN m.close_time IS NULL THEN NULL
                    ELSE (julianday(m.close_time) - julianday(s.timestamp)) * 86400.0
                END AS seconds_until_close,
                {resolution_select_sql}
                s.snapshot_quality_status,
                s.yes_price,
                s.no_price,
                s.best_bid_yes,
                s.best_ask_yes,
                s.best_bid_no,
                s.best_ask_no,
                s.spread_yes,
                s.spread_no,
                s.mid_price_yes,
                s.mid_price_no,
                s.volume,
                s.liquidity,
                s.last_trade_price,
                s.last_trade_size,
                s.last_trade_time,
                s.is_partial_orderbook,
                s.missing_level_count,
                s.time_gap_from_prev_snapshot_sec,
                s.is_gap_affected,
                s.is_gap_affected AS gap_affected,
                s.feature_ready,
                s.has_trade_data,
                CASE WHEN COALESCE(s.is_partial_orderbook, 0) = 0 THEN 1 ELSE 0 END AS has_full_orderbook,
                s.book_checksum,
                {depth_select_sql}
                {btc_select_sql}
                {tick_select_sql}
                {feature_select_sql}
            FROM market_snapshots s
            JOIN markets m
              ON m.run_id = s.run_id
             AND m.market_id = s.market_id
            JOIN features f
              ON f.run_id = s.run_id
             AND f.market_id = s.market_id
             AND f.timestamp = s.timestamp
            {btc_join_sql}
            {tick_join_sql}
            WHERE {where_sql}
            ORDER BY s.run_id ASC, s.market_id ASC, s.timestamp ASC
            """,
            tuple(params),
        ).fetchall()

        rows_py = _to_pylist(rows)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        table = pa.Table.from_pylist(rows_py)
        pq.write_table(table, output)

        # Validation report
        total_snapshots = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM market_snapshots s
                JOIN markets m
                  ON m.run_id = s.run_id
                 AND m.market_id = s.market_id
                WHERE s.run_id IN ({placeholders})
                  AND m.strict_validation_passed = 1
                  AND s.strict_validation_passed = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )
        kept = len(rows_py)
        dropped = max(0, total_snapshots - kept)

        partial_books = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM market_snapshots s
                WHERE s.run_id IN ({placeholders})
                  AND COALESCE(s.is_partial_orderbook, 0) = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )
        gap_affected = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM market_snapshots s
                WHERE s.run_id IN ({placeholders})
                  AND COALESCE(s.is_gap_affected, 0) = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )
        trade_rows = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM market_snapshots s
                WHERE s.run_id IN ({placeholders})
                  AND COALESCE(s.has_trade_data, 0) = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )
        feature_rows = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM features f
                WHERE f.run_id IN ({placeholders})
                  AND COALESCE(f.feature_ready, 0) = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )

        markets_recorded = int(
            conn.execute(
                f"""
                SELECT COUNT(DISTINCT market_id)
                FROM markets
                WHERE run_id IN ({placeholders})
                  AND strict_validation_passed = 1
                """,
                tuple(params),
            ).fetchone()[0]
            or 0
        )

        null_percentages: dict[str, float] = {}
        if rows_py:
            for column in feature_cols:
                null_count = sum(1 for row in rows_py if row.get(column) is None)
                null_percentages[column] = (100.0 * null_count / len(rows_py))

        btc_rows = sum(1 for row in rows_py if int(row.get("has_btc_price") or 0) == 1)
        label_rows = sum(
            1 for row in rows_py if int(row.get("label_available") or 0) == 1
        )
        full_orderbook_rows = sum(
            1 for row in rows_py if int(row.get("has_full_orderbook") or 0) == 1
        )
        depth_rows = sum(1 for row in rows_py if int(row.get("has_depth") or 0) == 1)

        report = {
            "run_ids": resolved_run_ids,
            "markets_recorded": markets_recorded,
            "snapshots_kept": kept,
            "snapshots_dropped": dropped,
            "partial_books": partial_books,
            "gap_affected_rows": gap_affected,
            "trade_coverage_pct": (100.0 * trade_rows / total_snapshots)
            if total_snapshots > 0
            else 0.0,
            "feature_coverage_pct": (100.0 * feature_rows / total_snapshots)
            if total_snapshots > 0
            else 0.0,
            "btc_coverage_pct": (100.0 * btc_rows / kept) if kept > 0 else 0.0,
            "label_coverage_pct": (100.0 * label_rows / kept) if kept > 0 else 0.0,
            "full_orderbook_coverage_pct": (100.0 * full_orderbook_rows / kept)
            if kept > 0
            else 0.0,
            "depth_coverage_pct": (100.0 * depth_rows / kept) if kept > 0 else 0.0,
            "null_percentages": null_percentages,
            "output_path": str(output),
        }

        report_path = output.with_suffix(output.suffix + ".report.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    finally:
        conn.close()
