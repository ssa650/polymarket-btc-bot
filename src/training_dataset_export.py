from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from .models import to_iso


LOGGER = logging.getLogger(__name__)

DEFAULT_EXPORT_CHUNK_SIZE = 10_000
DEFAULT_PROGRESS_EVERY_ROWS = 50_000
DEFAULT_PROGRESS_EVERY_SEC = 10.0

FEATURE_COLUMNS: tuple[str, ...] = (
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
    "is_price_jump",
    "feature_ready",
    "is_gap_affected",
    "snapshot_quality_status",
    "strict_validation_passed",
    "recorder_version",
    "schema_version",
)

EXPORT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "market_id",
    "question",
    "start_time",
    "close_time",
    "timestamp",
    "time_until_resolution",
    "time_since_market_created",
    "market_phase",
    "yes_token_id",
    "no_token_id",
    "seconds_after_start",
    "seconds_before_close",
    "export_row_usable",
    *FEATURE_COLUMNS,
    "best_bid_yes",
    "best_ask_yes",
    "best_bid_no",
    "best_ask_no",
    "spread_yes",
    "spread_no",
    "mid_price_yes",
    "mid_price_no",
    "last_trade_price",
    "last_trade_size",
    "last_trade_side",
    "has_orderbook",
    "has_trade_data",
    "btc_chainlink_price",
    "btc_chainlink_exchange_timestamp",
    "btc_chainlink_local_arrival_iso",
    "btc_chainlink_age_sec_at_feature",
    "btc_binance_price",
    "btc_binance_exchange_timestamp",
    "btc_binance_local_arrival_iso",
    "btc_binance_age_sec_at_feature",
    "btc_price_diff_binance_minus_chainlink",
    "btc_price_diff_pct_binance_minus_chainlink",
    "label_resolved_up_down",
    "label_yes_win",
    "btc_price_at_market_start",
    "btc_price_at_resolution",
    "label_btc_up_at_resolution",
    "label_btc_return_to_resolution",
    "future_btc_return_to_resolution_from_feature",
)

_STRING_COLUMNS = {
    "run_id",
    "market_id",
    "question",
    "start_time",
    "close_time",
    "timestamp",
    "market_phase",
    "yes_token_id",
    "no_token_id",
    "snapshot_quality_status",
    "recorder_version",
    "schema_version",
    "last_trade_side",
    "btc_chainlink_exchange_timestamp",
    "btc_chainlink_local_arrival_iso",
    "btc_binance_exchange_timestamp",
    "btc_binance_local_arrival_iso",
    "label_resolved_up_down",
}

_INTEGER_COLUMNS = {
    "export_row_usable",
    "feature_ready",
    "is_gap_affected",
    "strict_validation_passed",
    "is_price_jump",
    "has_orderbook",
    "has_trade_data",
    "label_yes_win",
    "label_btc_up_at_resolution",
}

_SECONDS_BEFORE_CLOSE_SQL = """
CASE
  WHEN m.close_time IS NOT NULL
  THEN (julianday(m.close_time) - julianday(f.timestamp)) * 86400.0
  ELSE f.time_until_resolution
END
"""

_USABLE_SQL = """
CASE
  WHEN COALESCE(f.feature_ready, 0) = 1
   AND COALESCE(f.is_gap_affected, 0) = 0
   AND COALESCE(f.strict_validation_passed, 0) = 1
   AND COALESCE(f.snapshot_quality_status, '') = 'ok'
   AND COALESCE(s.has_orderbook, 0) = 1
   AND COALESCE(s.has_trade_data, 0) = 1
  THEN 1 ELSE 0
END
"""

_LABEL_YES_WIN_SQL = """
CASE
  WHEN COALESCE(m.resolved, 0) <> 1
    OR (m.winning_asset_id IS NULL AND m.winning_outcome IS NULL)
  THEN NULL
  WHEN m.winning_asset_id = m.yes_token_id
    OR upper(COALESCE(m.winning_outcome, '')) IN ('YES', 'Y', 'UP', 'TRUE')
  THEN 1
  WHEN m.winning_asset_id = m.no_token_id
    OR upper(COALESCE(m.winning_outcome, '')) IN ('NO', 'N', 'DOWN', 'FALSE')
  THEN 0
  ELSE NULL
END
"""


def export_training_dataset(
    *,
    db_path: str,
    output_path: str,
    output_csv_path: str | None = None,
    recent_minutes: float | None = None,
    min_time_until_resolution_sec: float | None = None,
    max_time_until_resolution_sec: float | None = None,
    include_not_ready: bool = False,
    canonical_btc_source: str = POLYMARKET_RTDS_CHAINLINK_SOURCE,
    comparison_btc_source: str = POLYMARKET_RTDS_BINANCE_SOURCE,
    limit: int | None = None,
    chunk_size: int = DEFAULT_EXPORT_CHUNK_SIZE,
    progress_every_rows: int = DEFAULT_PROGRESS_EVERY_ROWS,
    progress_every_sec: float = DEFAULT_PROGRESS_EVERY_SEC,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    cutoff = (
        now_dt - timedelta(minutes=max(0.0, float(recent_minutes)))
        if recent_minutes is not None
        else None
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = Path(output_csv_path) if output_csv_path else None
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)

    bounded_limit = None if limit is None else max(0, int(limit))
    effective_chunk_size = max(1, int(chunk_size))
    accumulator = _ExportSummaryAccumulator()
    progress = _ExportProgressLogger(
        every_rows=max(0, int(progress_every_rows)),
        every_sec=max(0.0, float(progress_every_sec)),
    )
    parquet_writer = _ParquetChunkWriter(output)
    csv_writer = _CSVChunkWriter(csv_output) if csv_output is not None else None

    LOGGER.info(
        "export_training_dataset_started db_path=%s output_path=%s recent_minutes=%s "
        "limit=%s chunk_size=%s include_not_ready=%s",
        db_path,
        output_path,
        recent_minutes,
        bounded_limit,
        effective_chunk_size,
        include_not_ready,
    )

    conn = _connect_read_only(db_path)
    try:
        for chunk in _iter_export_row_chunks(
            conn,
            cutoff=cutoff,
            min_time_until_resolution_sec=min_time_until_resolution_sec,
            max_time_until_resolution_sec=max_time_until_resolution_sec,
            include_not_ready=include_not_ready,
            canonical_btc_source=canonical_btc_source,
            comparison_btc_source=comparison_btc_source,
            limit=bounded_limit,
            chunk_size=effective_chunk_size,
        ):
            parquet_writer.write(chunk)
            if csv_writer is not None:
                csv_writer.write(chunk)
            accumulator.add_rows(chunk)
            progress.maybe_log(accumulator.exported_rows)

        parquet_writer.close()
        if csv_writer is not None:
            csv_writer.close()
        progress.log_final(accumulator.exported_rows)

        skipped = (
            accumulator.skipped_rows_by_reason()
            if bounded_limit is not None
            else _skipped_rows_by_reason(
                conn,
                cutoff=cutoff,
                min_time_until_resolution_sec=min_time_until_resolution_sec,
                max_time_until_resolution_sec=max_time_until_resolution_sec,
            )
        )
    finally:
        parquet_writer.close()
        if csv_writer is not None:
            csv_writer.close()
        conn.close()

    summary = accumulator.build_summary(
        output_path=str(output),
        output_csv_path=output_csv_path,
        output_file_size_bytes=_file_size(output),
        output_csv_file_size_bytes=_file_size(csv_output) if csv_output is not None else None,
        skipped_rows_by_reason=skipped,
        recent_minutes=recent_minutes,
        cutoff_iso=to_iso(cutoff) if cutoff is not None else None,
        include_not_ready=include_not_ready,
        canonical_btc_source=canonical_btc_source,
        comparison_btc_source=comparison_btc_source,
    )
    summary["limit"] = bounded_limit
    summary["chunk_size"] = effective_chunk_size
    summary["progress_every_rows"] = max(0, int(progress_every_rows))
    summary["progress_every_sec"] = max(0.0, float(progress_every_sec))
    LOGGER.info(
        "export_training_dataset_completed exported_rows=%s output_path=%s output_file_size_bytes=%s",
        summary["exported_rows"],
        output_path,
        summary["output_file_size_bytes"],
    )
    return summary


def _fetch_export_rows(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
    include_not_ready: bool,
    canonical_btc_source: str,
    comparison_btc_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in _iter_export_row_chunks(
        conn,
        cutoff=cutoff,
        min_time_until_resolution_sec=min_time_until_resolution_sec,
        max_time_until_resolution_sec=max_time_until_resolution_sec,
        include_not_ready=include_not_ready,
        canonical_btc_source=canonical_btc_source,
        comparison_btc_source=comparison_btc_source,
        limit=None,
        chunk_size=DEFAULT_EXPORT_CHUNK_SIZE,
    ):
        rows.extend(chunk)
    return rows


def _iter_export_row_chunks(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
    include_not_ready: bool,
    canonical_btc_source: str,
    comparison_btc_source: str,
    limit: int | None,
    chunk_size: int,
):
    _require_tables(conn)
    btc_cache = _BTCLookupCache(conn)
    trade_cache = _TradeSideCache(conn)
    remaining = limit
    last_key: tuple[str, str, str, int, int] | None = None
    while remaining is None or remaining > 0:
        current_limit = chunk_size if remaining is None else min(chunk_size, remaining)
        raw_rows = _fetch_export_row_chunk(
            conn,
            cutoff=cutoff,
            min_time_until_resolution_sec=min_time_until_resolution_sec,
            max_time_until_resolution_sec=max_time_until_resolution_sec,
            include_not_ready=include_not_ready,
            last_key=last_key,
            limit=current_limit,
        )
        if not raw_rows:
            break
        chunk = [
            _enrich_export_row(
                row,
                btc_cache=btc_cache,
                trade_cache=trade_cache,
                canonical_btc_source=canonical_btc_source,
                comparison_btc_source=comparison_btc_source,
            )
            for row in raw_rows
        ]
        yield chunk
        last_raw = raw_rows[-1]
        last_key = (
            str(last_raw.get("_run_id_key") or ""),
            str(last_raw.get("_market_id_key") or ""),
            str(last_raw.get("_timestamp_key") or ""),
            int(last_raw.get("_feature_rowid") or 0),
            int(last_raw.get("_snapshot_rowid") or 0),
        )
        if remaining is not None:
            remaining -= len(raw_rows)


def _fetch_export_row_chunk(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
    include_not_ready: bool,
    last_key: tuple[str, str, str, int, int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    feature_select = ",\n          ".join(f"f.{column} AS {column}" for column in FEATURE_COLUMNS)
    where_clauses, params = _base_export_where(
        cutoff=cutoff,
        min_time_until_resolution_sec=min_time_until_resolution_sec,
        max_time_until_resolution_sec=max_time_until_resolution_sec,
        include_not_ready=include_not_ready,
    )
    if last_key is not None:
        where_clauses.append(
            """
            (
              f.run_id > :last_run_id
              OR (f.run_id = :last_run_id AND f.market_id > :last_market_id)
              OR (
                f.run_id = :last_run_id
                AND f.market_id = :last_market_id
                AND f.timestamp > :last_timestamp
              )
              OR (
                f.run_id = :last_run_id
                AND f.market_id = :last_market_id
                AND f.timestamp = :last_timestamp
                AND f.rowid > :last_feature_rowid
              )
              OR (
                f.run_id = :last_run_id
                AND f.market_id = :last_market_id
                AND f.timestamp = :last_timestamp
                AND f.rowid = :last_feature_rowid
                AND s.rowid > :last_snapshot_rowid
              )
            )
            """
        )
        params.update(
            {
                "last_run_id": last_key[0],
                "last_market_id": last_key[1],
                "last_timestamp": last_key[2],
                "last_feature_rowid": last_key[3],
                "last_snapshot_rowid": last_key[4],
            }
        )
    params["chunk_limit"] = int(limit)
    where_sql = "\n          AND ".join(where_clauses)

    rows = conn.execute(
        f"""
        SELECT
          f.run_id AS _run_id_key,
          f.market_id AS _market_id_key,
          f.timestamp AS _timestamp_key,
          f.rowid AS _feature_rowid,
          s.rowid AS _snapshot_rowid,
          f.run_id,
          f.market_id,
          m.question,
          m.start_time,
          m.close_time,
          f.timestamp,
          f.time_until_resolution,
          f.time_since_market_created,
          COALESCE(m.phase, m.market_phase) AS market_phase,
          m.yes_token_id,
          m.no_token_id,
          CASE
            WHEN m.start_time IS NULL THEN NULL
            ELSE (julianday(f.timestamp) - julianday(m.start_time)) * 86400.0
          END AS seconds_after_start,
          {_SECONDS_BEFORE_CLOSE_SQL} AS seconds_before_close,
          {_USABLE_SQL} AS export_row_usable,
          {feature_select},
          s.best_bid_yes,
          s.best_ask_yes,
          s.best_bid_no,
          s.best_ask_no,
          s.spread_yes,
          s.spread_no,
          s.mid_price_yes,
          s.mid_price_no,
          s.last_trade_price,
          s.last_trade_size,
          s.has_orderbook,
          s.has_trade_data,
          CASE
            WHEN ({_LABEL_YES_WIN_SQL}) = 1 THEN 'UP'
            WHEN ({_LABEL_YES_WIN_SQL}) = 0 THEN 'DOWN'
            ELSE NULL
          END AS label_resolved_up_down,
          {_LABEL_YES_WIN_SQL} AS label_yes_win
        FROM features f
        JOIN market_snapshots s
          ON s.run_id = f.run_id
         AND s.market_id = f.market_id
         AND s.timestamp = f.timestamp
        JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        WHERE {where_sql}
        ORDER BY
          f.run_id ASC,
          f.market_id ASC,
          f.timestamp ASC,
          f.rowid ASC,
          s.rowid ASC
        LIMIT :chunk_limit
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _base_export_where(
    *,
    cutoff: datetime | None,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
    include_not_ready: bool,
) -> tuple[list[str], dict[str, Any]]:
    where_clauses = ["1 = 1"]
    params: dict[str, Any] = {}
    if cutoff is not None:
        where_clauses.append("f.timestamp >= :cutoff")
        params["cutoff"] = to_iso(cutoff)
    if min_time_until_resolution_sec is not None:
        where_clauses.append(f"({_SECONDS_BEFORE_CLOSE_SQL}) >= :min_time_until_resolution_sec")
        params["min_time_until_resolution_sec"] = float(min_time_until_resolution_sec)
    if max_time_until_resolution_sec is not None:
        where_clauses.append(f"({_SECONDS_BEFORE_CLOSE_SQL}) <= :max_time_until_resolution_sec")
        params["max_time_until_resolution_sec"] = float(max_time_until_resolution_sec)
    if not include_not_ready:
        where_clauses.extend(
            [
                "COALESCE(f.feature_ready, 0) = 1",
                "COALESCE(f.is_gap_affected, 0) = 0",
                "COALESCE(f.strict_validation_passed, 0) = 1",
                "COALESCE(f.snapshot_quality_status, '') = 'ok'",
                "COALESCE(s.has_orderbook, 0) = 1",
                "COALESCE(s.has_trade_data, 0) = 1",
            ]
        )
    return where_clauses, params


def _enrich_export_row(
    row: dict[str, Any],
    *,
    btc_cache: "_BTCLookupCache",
    trade_cache: "_TradeSideCache",
    canonical_btc_source: str,
    comparison_btc_source: str,
) -> dict[str, Any]:
    run_id = str(row.get("run_id") or "")
    market_id = str(row.get("market_id") or "")
    timestamp = row.get("timestamp")
    start_time = row.get("start_time")
    close_time = row.get("close_time")

    canonical = btc_cache.lookup_at_or_before(run_id, canonical_btc_source, timestamp)
    comparison = btc_cache.lookup_at_or_before(run_id, comparison_btc_source, timestamp)
    start_btc = btc_cache.lookup_at_or_before(run_id, canonical_btc_source, start_time)
    close_btc = btc_cache.lookup_at_or_before(run_id, canonical_btc_source, close_time)

    canonical_price = _float_or_none((canonical or {}).get("price"))
    comparison_price = _float_or_none((comparison or {}).get("price"))
    start_price = _float_or_none((start_btc or {}).get("price"))
    close_price = _float_or_none((close_btc or {}).get("price"))

    row["last_trade_side"] = trade_cache.lookup_side(run_id, market_id, timestamp)
    row["btc_chainlink_price"] = canonical_price
    row["btc_chainlink_exchange_timestamp"] = (canonical or {}).get("exchange_timestamp")
    row["btc_chainlink_local_arrival_iso"] = (canonical or {}).get("local_arrival_iso")
    row["btc_chainlink_age_sec_at_feature"] = _age_seconds(
        timestamp,
        (canonical or {}).get("local_arrival_iso"),
    )
    row["btc_binance_price"] = comparison_price
    row["btc_binance_exchange_timestamp"] = (comparison or {}).get("exchange_timestamp")
    row["btc_binance_local_arrival_iso"] = (comparison or {}).get("local_arrival_iso")
    row["btc_binance_age_sec_at_feature"] = _age_seconds(
        timestamp,
        (comparison or {}).get("local_arrival_iso"),
    )
    row["btc_price_diff_binance_minus_chainlink"] = (
        None
        if canonical_price is None or comparison_price is None
        else comparison_price - canonical_price
    )
    row["btc_price_diff_pct_binance_minus_chainlink"] = (
        None
        if canonical_price in (None, 0.0) or comparison_price is None
        else (comparison_price - canonical_price) / canonical_price
    )
    row["btc_price_at_market_start"] = start_price
    row["btc_price_at_resolution"] = close_price
    row["label_btc_up_at_resolution"] = (
        None
        if start_price is None or close_price is None
        else int(close_price > start_price)
    )
    row["label_btc_return_to_resolution"] = (
        None
        if start_price in (None, 0.0) or close_price is None
        else (close_price - start_price) / start_price
    )
    row["future_btc_return_to_resolution_from_feature"] = (
        None
        if close_price is None or canonical_price in (None, 0.0)
        else (close_price - canonical_price) / canonical_price
    )
    return _normalize_row(row)


class _BTCLookupCache:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: dict[tuple[str, str], tuple[list[str], list[dict[str, Any]]]] = {}

    def lookup_at_or_before(
        self,
        run_id: str,
        source: str,
        timestamp: Any,
    ) -> dict[str, Any] | None:
        if not run_id or not source or timestamp is None:
            return None
        keys, rows = self._load(run_id, source)
        if not keys:
            return None
        idx = bisect_right(keys, str(timestamp)) - 1
        if idx < 0:
            return None
        return rows[idx]

    def _load(self, run_id: str, source: str) -> tuple[list[str], list[dict[str, Any]]]:
        cache_key = (run_id, source)
        if cache_key not in self._cache:
            rows = [
                dict(row)
                for row in self._conn.execute(
                    """
                    SELECT price, exchange_timestamp, local_arrival_iso, local_arrival_ns, id
                    FROM btc_prices
                    WHERE run_id = ?
                      AND source = ?
                      AND local_arrival_iso IS NOT NULL
                    ORDER BY local_arrival_iso ASC, local_arrival_ns ASC, id ASC
                    """,
                    (run_id, source),
                ).fetchall()
            ]
            self._cache[cache_key] = (
                [str(row["local_arrival_iso"]) for row in rows],
                rows,
            )
        return self._cache[cache_key]


class _TradeSideCache:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: dict[tuple[str, str], tuple[list[str], list[str | None]]] = {}

    def lookup_side(self, run_id: str, market_id: str, timestamp: Any) -> str | None:
        if not run_id or not market_id or timestamp is None:
            return None
        keys, sides = self._load(run_id, market_id)
        if not keys:
            return None
        idx = bisect_right(keys, str(timestamp)) - 1
        if idx < 0:
            return None
        return sides[idx]

    def _load(self, run_id: str, market_id: str) -> tuple[list[str], list[str | None]]:
        cache_key = (run_id, market_id)
        if cache_key not in self._cache:
            rows = self._conn.execute(
                """
                SELECT timestamp, side, trade_id
                FROM trades
                WHERE run_id = ?
                  AND market_id = ?
                  AND timestamp IS NOT NULL
                ORDER BY timestamp ASC, trade_id ASC
                """,
                (run_id, market_id),
            ).fetchall()
            self._cache[cache_key] = (
                [str(row["timestamp"]) for row in rows],
                [None if row["side"] is None else str(row["side"]) for row in rows],
            )
        return self._cache[cache_key]


class _ParquetChunkWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._writer: Any = None
        self._closed = False
        self._schema: Any = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        pa, pq = _load_pyarrow()
        schema = _pyarrow_schema(pa)
        table = pa.Table.from_pylist(rows, schema=schema)
        if self._writer is None:
            self._schema = schema
            self._writer = pq.ParquetWriter(self._path, schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._closed:
            return
        pa, pq = _load_pyarrow()
        if self._writer is None:
            schema = _pyarrow_schema(pa)
            empty_table = pa.table(
                {field.name: pa.array([], type=field.type) for field in schema}
            )
            pq.write_table(empty_table, self._path)
        else:
            self._writer.close()
        self._closed = True


class _CSVChunkWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(EXPORT_COLUMNS))
        self._writer.writeheader()
        self._closed = False

    def write(self, rows: list[dict[str, Any]]) -> None:
        if self._closed:
            return
        for row in rows:
            self._writer.writerow({column: row.get(column) for column in EXPORT_COLUMNS})

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


class _ExportSummaryAccumulator:
    def __init__(self) -> None:
        self.exported_rows = 0
        self.usable_rows = 0
        self.ready_rows = 0
        self.run_ids: set[str] = set()
        self.markets: set[tuple[str, str]] = set()
        self.timestamp_min: str | None = None
        self.timestamp_max: str | None = None
        self.canonical_btc_null_count = 0
        self.comparison_btc_null_count = 0
        self.label_null_counts = {
            "label_resolved_up_down": 0,
            "label_yes_win": 0,
            "label_btc_up_at_resolution": 0,
            "label_btc_return_to_resolution": 0,
            "future_btc_return_to_resolution_from_feature": 0,
        }
        self._limited_skipped = {
            "candidate_rows": 0,
            "unusable_rows": 0,
            "feature_not_ready": 0,
            "gap_affected": 0,
            "feature_strict_validation_failed": 0,
            "snapshot_quality_not_ok": 0,
            "missing_orderbook": 0,
            "missing_trade_data": 0,
        }

    def add_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.exported_rows += 1
            if int(row.get("export_row_usable") or 0) == 1:
                self.usable_rows += 1
            if int(row.get("feature_ready") or 0) == 1:
                self.ready_rows += 1
            run_id = row.get("run_id")
            market_id = row.get("market_id")
            if run_id:
                self.run_ids.add(str(run_id))
            if run_id and market_id:
                self.markets.add((str(run_id), str(market_id)))
            timestamp = row.get("timestamp")
            if timestamp:
                timestamp_str = str(timestamp)
                self.timestamp_min = (
                    timestamp_str
                    if self.timestamp_min is None
                    else min(self.timestamp_min, timestamp_str)
                )
                self.timestamp_max = (
                    timestamp_str
                    if self.timestamp_max is None
                    else max(self.timestamp_max, timestamp_str)
                )
            if row.get("btc_chainlink_price") is None:
                self.canonical_btc_null_count += 1
            if row.get("btc_binance_price") is None:
                self.comparison_btc_null_count += 1
            for column in self.label_null_counts:
                if row.get(column) is None:
                    self.label_null_counts[column] += 1
            self._add_limited_skip_counts(row)

    def _add_limited_skip_counts(self, row: dict[str, Any]) -> None:
        self._limited_skipped["candidate_rows"] += 1
        failed = False
        if int(row.get("feature_ready") or 0) != 1:
            self._limited_skipped["feature_not_ready"] += 1
            failed = True
        if int(row.get("is_gap_affected") or 0) != 0:
            self._limited_skipped["gap_affected"] += 1
            failed = True
        if int(row.get("strict_validation_passed") or 0) != 1:
            self._limited_skipped["feature_strict_validation_failed"] += 1
            failed = True
        if str(row.get("snapshot_quality_status") or "") != "ok":
            self._limited_skipped["snapshot_quality_not_ok"] += 1
            failed = True
        if int(row.get("has_orderbook") or 0) != 1:
            self._limited_skipped["missing_orderbook"] += 1
            failed = True
        if int(row.get("has_trade_data") or 0) != 1:
            self._limited_skipped["missing_trade_data"] += 1
            failed = True
        if failed:
            self._limited_skipped["unusable_rows"] += 1

    def skipped_rows_by_reason(self) -> dict[str, int]:
        return dict(self._limited_skipped)

    def build_summary(
        self,
        *,
        output_path: str,
        output_csv_path: str | None,
        output_file_size_bytes: int,
        output_csv_file_size_bytes: int | None,
        skipped_rows_by_reason: dict[str, int],
        recent_minutes: float | None,
        cutoff_iso: str | None,
        include_not_ready: bool,
        canonical_btc_source: str,
        comparison_btc_source: str,
    ) -> dict[str, Any]:
        return {
            "exported_rows": self.exported_rows,
            "usable_rows": self.usable_rows,
            "skipped_rows_by_reason": skipped_rows_by_reason,
            "unique_run_ids": sorted(self.run_ids),
            "unique_run_id_count": len(self.run_ids),
            "unique_markets": len(self.markets),
            "timestamp_min": self.timestamp_min,
            "timestamp_max": self.timestamp_max,
            "feature_ready_pct": _pct(self.ready_rows, self.exported_rows),
            "canonical_btc_source": canonical_btc_source,
            "comparison_btc_source": comparison_btc_source,
            "canonical_btc_null_count": self.canonical_btc_null_count,
            "comparison_btc_null_count": self.comparison_btc_null_count,
            "label_null_counts": dict(self.label_null_counts),
            "recent_minutes": recent_minutes,
            "recent_cutoff": cutoff_iso,
            "include_not_ready": include_not_ready,
            "output_path": output_path,
            "output_file_size_bytes": output_file_size_bytes,
            "output_csv_path": output_csv_path,
            "output_csv_file_size_bytes": output_csv_file_size_bytes,
        }


class _ExportProgressLogger:
    def __init__(self, *, every_rows: int, every_sec: float) -> None:
        self._every_rows = every_rows
        self._every_sec = every_sec
        self._start = time.monotonic()
        self._last_log_time = self._start
        self._last_log_rows = 0

    def maybe_log(self, rows: int) -> None:
        now = time.monotonic()
        row_due = self._every_rows > 0 and rows - self._last_log_rows >= self._every_rows
        time_due = self._every_sec > 0 and now - self._last_log_time >= self._every_sec
        if row_due or time_due:
            self._log(rows, now, "export_training_dataset_progress")

    def log_final(self, rows: int) -> None:
        self._log(rows, time.monotonic(), "export_training_dataset_progress_final")

    def _log(self, rows: int, now: float, event: str) -> None:
        elapsed = max(0.000001, now - self._start)
        LOGGER.info(
            "%s rows=%s elapsed_sec=%.2f rows_per_sec=%.2f",
            event,
            rows,
            elapsed,
            rows / elapsed,
        )
        self._last_log_time = now
        self._last_log_rows = rows


def _skipped_rows_by_reason(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
) -> dict[str, int]:
    where_clauses = ["1 = 1"]
    params: dict[str, Any] = {}
    if cutoff is not None:
        where_clauses.append("f.timestamp >= :cutoff")
        params["cutoff"] = to_iso(cutoff)
    if min_time_until_resolution_sec is not None:
        where_clauses.append(f"({_SECONDS_BEFORE_CLOSE_SQL}) >= :min_time_until_resolution_sec")
        params["min_time_until_resolution_sec"] = float(min_time_until_resolution_sec)
    if max_time_until_resolution_sec is not None:
        where_clauses.append(f"({_SECONDS_BEFORE_CLOSE_SQL}) <= :max_time_until_resolution_sec")
        params["max_time_until_resolution_sec"] = float(max_time_until_resolution_sec)
    where_sql = "\n          AND ".join(where_clauses)
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS candidate_rows,
          SUM(CASE WHEN COALESCE(f.feature_ready, 0) <> 1 THEN 1 ELSE 0 END) AS feature_not_ready,
          SUM(CASE WHEN COALESCE(f.is_gap_affected, 0) <> 0 THEN 1 ELSE 0 END) AS gap_affected,
          SUM(CASE WHEN COALESCE(f.strict_validation_passed, 0) <> 1 THEN 1 ELSE 0 END) AS feature_strict_validation_failed,
          SUM(CASE WHEN COALESCE(f.snapshot_quality_status, '') <> 'ok' THEN 1 ELSE 0 END) AS snapshot_quality_not_ok,
          SUM(CASE WHEN COALESCE(s.has_orderbook, 0) <> 1 THEN 1 ELSE 0 END) AS missing_orderbook,
          SUM(CASE WHEN COALESCE(s.has_trade_data, 0) <> 1 THEN 1 ELSE 0 END) AS missing_trade_data,
          SUM(CASE WHEN ({_USABLE_SQL}) = 0 THEN 1 ELSE 0 END) AS unusable_rows
        FROM features f
        JOIN market_snapshots s
          ON s.run_id = f.run_id
         AND s.market_id = f.market_id
         AND s.timestamp = f.timestamp
        JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        WHERE {where_sql}
        """,
        params,
    ).fetchone()
    if row is None:
        return {}
    return {
        "candidate_rows": int(row["candidate_rows"] or 0),
        "unusable_rows": int(row["unusable_rows"] or 0),
        "feature_not_ready": int(row["feature_not_ready"] or 0),
        "gap_affected": int(row["gap_affected"] or 0),
        "feature_strict_validation_failed": int(
            row["feature_strict_validation_failed"] or 0
        ),
        "snapshot_quality_not_ok": int(row["snapshot_quality_not_ok"] or 0),
        "missing_orderbook": int(row["missing_orderbook"] or 0),
        "missing_trade_data": int(row["missing_trade_data"] or 0),
    }


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    output_path: str,
    output_csv_path: str | None,
    output_file_size_bytes: int,
    output_csv_file_size_bytes: int | None,
    skipped_rows_by_reason: dict[str, int],
    recent_minutes: float | None,
    cutoff_iso: str | None,
    include_not_ready: bool,
    canonical_btc_source: str,
    comparison_btc_source: str,
) -> dict[str, Any]:
    exported_rows = len(rows)
    usable_rows = sum(1 for row in rows if int(row.get("export_row_usable") or 0) == 1)
    timestamps = [str(row["timestamp"]) for row in rows if row.get("timestamp")]
    ready_rows = sum(1 for row in rows if int(row.get("feature_ready") or 0) == 1)
    run_ids = sorted({str(row.get("run_id")) for row in rows if row.get("run_id")})
    markets = {
        (str(row.get("run_id")), str(row.get("market_id")))
        for row in rows
        if row.get("run_id") and row.get("market_id")
    }
    label_columns = (
        "label_resolved_up_down",
        "label_yes_win",
        "label_btc_up_at_resolution",
        "label_btc_return_to_resolution",
        "future_btc_return_to_resolution_from_feature",
    )
    return {
        "exported_rows": exported_rows,
        "usable_rows": usable_rows,
        "skipped_rows_by_reason": skipped_rows_by_reason,
        "unique_run_ids": run_ids,
        "unique_run_id_count": len(run_ids),
        "unique_markets": len(markets),
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "feature_ready_pct": _pct(ready_rows, exported_rows),
        "canonical_btc_source": canonical_btc_source,
        "comparison_btc_source": comparison_btc_source,
        "canonical_btc_null_count": sum(
            1 for row in rows if row.get("btc_chainlink_price") is None
        ),
        "comparison_btc_null_count": sum(
            1 for row in rows if row.get("btc_binance_price") is None
        ),
        "label_null_counts": {
            column: sum(1 for row in rows if row.get(column) is None)
            for column in label_columns
        },
        "recent_minutes": recent_minutes,
        "recent_cutoff": cutoff_iso,
        "include_not_ready": include_not_ready,
        "output_path": output_path,
        "output_file_size_bytes": output_file_size_bytes,
        "output_csv_path": output_csv_path,
        "output_csv_file_size_bytes": output_csv_file_size_bytes,
    }


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _require_tables(conn: sqlite3.Connection) -> None:
    missing = [
        table
        for table in ("features", "market_snapshots", "markets", "btc_prices", "trades")
        if not _table_exists(conn, table)
    ]
    if missing:
        raise RuntimeError(f"Cannot export training dataset; missing tables: {missing}")


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


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    pa, pq = _load_pyarrow()

    if rows:
        table = pa.Table.from_pylist(rows, schema=_pyarrow_schema(pa))
    else:
        schema = _pyarrow_schema(pa)
        table = pa.table({field.name: pa.array([], type=field.type) for field in schema})
    pq.write_table(table, path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPORT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in EXPORT_COLUMNS})


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column) for column in EXPORT_COLUMNS}
    for column in _STRING_COLUMNS:
        if normalized.get(column) is not None:
            normalized[column] = str(normalized[column])
    for column in _INTEGER_COLUMNS:
        if normalized.get(column) is not None:
            normalized[column] = int(normalized[column])
    for column in (
        "seconds_after_start",
        "seconds_before_close",
        "btc_chainlink_age_sec_at_feature",
        "btc_binance_age_sec_at_feature",
    ):
        if normalized.get(column) is not None:
            normalized[column] = round(float(normalized[column]), 6)
    return normalized


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires pyarrow. Install with: pip install pyarrow"
        ) from exc
    return pa, pq


def _pyarrow_schema(pa: Any) -> Any:
    fields = []
    for column in EXPORT_COLUMNS:
        if column in _STRING_COLUMNS:
            arrow_type = pa.string()
        elif column in _INTEGER_COLUMNS:
            arrow_type = pa.int64()
        else:
            arrow_type = pa.float64()
        fields.append(pa.field(column, arrow_type))
    return pa.schema(fields)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_seconds(later: Any, earlier: Any) -> float | None:
    later_dt = _parse_iso_datetime(later)
    earlier_dt = _parse_iso_datetime(earlier)
    if later_dt is None or earlier_dt is None:
        return None
    return round(max(0.0, (later_dt - earlier_dt).total_seconds()), 6)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 2)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "EXPORT_COLUMNS",
    "FEATURE_COLUMNS",
    "export_training_dataset",
]
