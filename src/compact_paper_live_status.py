from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote


COMPACT_HEARTBEAT_FIELDS = (
    "timestamp",
    "latest_feature_timestamp",
    "candidate_rows_seen",
    "candidate_rows_total",
    "candidates_logged",
    "trades_opened",
    "trades_closed",
    "settled_trades",
    "skips_logged",
    "trades",
    "open_trades",
    "closed_trades",
    "pnl",
    "avg_roi",
    "wins",
    "losses",
    "latest_created_at",
    "latest_heartbeat_timestamp",
    "status",
    "top_rejection_reasons",
)


def build_compact_paper_live_status(
    paper_db_path: str,
    *,
    limit: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    db_path = Path(paper_db_path)
    if not db_path.exists():
        return {
            "status": "missing_db",
            "paper_db_path": str(paper_db_path),
            "limit": int(limit),
            "heartbeats": [],
            "error": "paper_db_missing",
        }
    conn = _open_readonly(db_path)
    try:
        heartbeat_table = _heartbeat_table(conn)
        heartbeat_rows = (
            _load_heartbeat_rows(conn, heartbeat_table, limit=max(0, int(limit)))
            if heartbeat_table
            else []
        )
        trade_summary = _trade_summary(conn)
        fallback_rejections = _candidate_rejection_reasons(conn)
        compact_rows = [
            _compact_heartbeat_row(
                row,
                trade_summary=trade_summary,
                fallback_rejections=fallback_rejections,
                verbose=bool(verbose),
            )
            for row in heartbeat_rows
        ]
        if not compact_rows:
            compact_rows = [
                _compact_heartbeat_row(
                    _fallback_heartbeat_row(trade_summary),
                    trade_summary=trade_summary,
                    fallback_rejections=fallback_rejections,
                    verbose=bool(verbose),
                )
            ]
    finally:
        conn.close()
    return {
        "status": "ok",
        "paper_db_path": str(paper_db_path),
        "limit": int(limit),
        "verbose": bool(verbose),
        "heartbeat_source": heartbeat_table or "paper_trades_fallback",
        "summary": trade_summary,
        "heartbeats": compact_rows,
    }


def render_compact_paper_live_status(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _heartbeat_table(conn: sqlite3.Connection) -> str | None:
    for table in ("multi_strategy_paper_trader_heartbeats", "paper_trader_heartbeats"):
        if _table_exists(conn, table):
            return table
    return None


def _load_heartbeat_rows(conn: sqlite3.Connection, table: str, *, limit: int) -> list[dict[str, Any]]:
    columns = set(_table_columns(conn, table))
    selected = [
        _column_or_null(columns, "timestamp"),
        _column_or_null(columns, "status"),
        _column_or_null(columns, "latest_feature_timestamp"),
        _column_or_null(columns, "latest_seen_feature_timestamp"),
        _column_or_null(columns, "latest_eligible_feature_timestamp"),
        _column_or_null(columns, "candidate_decisions_evaluated_this_poll"),
        _column_or_null(columns, "candidate_rows_seen"),
        _column_or_null(columns, "candidate_rows_total"),
        _column_or_null(columns, "candidate_rows_written_this_poll"),
        _column_or_null(columns, "candidates_logged"),
        _column_or_null(columns, "trades_opened"),
        _column_or_null(columns, "open_trades"),
        _column_or_null(columns, "awaiting_resolution_trades"),
        _column_or_null(columns, "settled_trades"),
        _column_or_null(columns, "total_trades"),
        _column_or_null(columns, "skips_logged"),
        _column_or_null(columns, "per_strategy_json"),
        *[
            column
            for column in sorted(columns)
            if column.endswith("_skips") and column not in {"skips_logged"}
        ],
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(selected)}
        FROM {table}
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def _compact_heartbeat_row(
    row: dict[str, Any],
    *,
    trade_summary: dict[str, Any],
    fallback_rejections: list[dict[str, Any]],
    verbose: bool,
) -> dict[str, Any]:
    strategy_stats = _parse_strategy_stats(row.get("per_strategy_json"))
    top_rejections = _top_rejection_reasons_from_strategy_stats(strategy_stats)
    if not top_rejections:
        top_rejections = _top_rejection_reasons_from_heartbeat_row(row)
    if not top_rejections:
        top_rejections = fallback_rejections[:5]
    latest_feature_timestamp = (
        row.get("latest_feature_timestamp")
        or row.get("latest_eligible_feature_timestamp")
        or row.get("latest_seen_feature_timestamp")
    )
    payload = {
        "timestamp": row.get("timestamp"),
        "latest_feature_timestamp": latest_feature_timestamp,
        "candidate_rows_seen": _int_value(
            row.get("candidate_decisions_evaluated_this_poll"),
            row.get("candidate_rows_seen"),
            row.get("candidate_rows_written_this_poll"),
            row.get("candidates_logged"),
        ),
        "candidate_rows_total": _int_value(row.get("candidate_rows_total")),
        "candidates_logged": _int_value(row.get("candidates_logged")),
        "trades_opened": _int_value(row.get("trades_opened")),
        "trades_closed": int(trade_summary.get("closed_trades") or 0),
        "settled_trades": int(trade_summary.get("settled_trades") or 0),
        "skips_logged": _int_value(row.get("skips_logged")),
        "trades": int(trade_summary.get("trades") or 0),
        "open_trades": int(trade_summary.get("open_trades") or 0),
        "closed_trades": int(trade_summary.get("closed_trades") or 0),
        "pnl": trade_summary.get("pnl"),
        "avg_roi": trade_summary.get("avg_roi"),
        "wins": int(trade_summary.get("wins") or 0),
        "losses": int(trade_summary.get("losses") or 0),
        "latest_created_at": trade_summary.get("latest_created_at"),
        "latest_heartbeat_timestamp": row.get("timestamp"),
        "status": row.get("status") or "ok",
        "top_rejection_reasons": top_rejections,
    }
    if verbose:
        payload["strategy_stats"] = strategy_stats
    return payload


def _fallback_heartbeat_row(trade_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": trade_summary.get("latest_created_at"),
        "latest_feature_timestamp": trade_summary.get("latest_created_at"),
        "candidate_rows_seen": 0,
        "candidate_rows_total": 0,
        "candidates_logged": 0,
        "trades_opened": 0,
        "skips_logged": 0,
        "per_strategy_json": None,
    }


def _parse_strategy_stats(raw: Any) -> Any:
    if raw in (None, ""):
        return []
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return []


def _top_rejection_reasons_from_strategy_stats(strategy_stats: Any) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    if isinstance(strategy_stats, dict):
        raw = strategy_stats.get("top_skip_reasons")
        if isinstance(raw, dict):
            for reason, count in raw.items():
                if reason:
                    counts[str(reason)] += int(float(count or 0))
        _collect_strategy_counter_fields(strategy_stats, counts)
    elif isinstance(strategy_stats, list):
        for item in strategy_stats:
            if not isinstance(item, dict):
                continue
            reason = item.get("latest_rejection_reason")
            if reason:
                counts[str(reason)] += 1
            _collect_strategy_counter_fields(item, counts)
    return [
        {"reason": reason, "count": count}
        for reason, count in counts.most_common(5)
        if count > 0
    ]


def _top_rejection_reasons_from_heartbeat_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for key, value in row.items():
        if not str(key).endswith("_skips"):
            continue
        count = _int_value(value)
        if count <= 0:
            continue
        counts[str(key).removesuffix("_skips")] += count
    return [
        {"reason": reason, "count": count}
        for reason, count in counts.most_common(5)
        if count > 0
    ]


def _collect_strategy_counter_fields(item: dict[str, Any], counts: Counter[str]) -> None:
    ignored = {
        "skipped_candidates",
        "skipped_candidate_count",
        "skipped_diagnostic_rows",
    }
    for key, value in item.items():
        if not str(key).startswith("skipped_") or key in ignored:
            continue
        count = _int_value(value)
        if count <= 0:
            continue
        reason = str(key).removeprefix("skipped_")
        counts[reason] += count


def _candidate_rejection_reasons(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    if _table_exists(conn, "paper_trade_candidates"):
        columns = set(_table_columns(conn, "paper_trade_candidates"))
        if "rejection_reason" in columns:
            rows = conn.execute(
                """
                SELECT rejection_reason, COUNT(*) AS count
                FROM paper_trade_candidates
                WHERE rejection_reason IS NOT NULL
                  AND rejection_reason != ''
                GROUP BY rejection_reason
                """
            ).fetchall()
            for row in rows:
                counts[str(row["rejection_reason"])] += int(row["count"])
    if _table_exists(conn, "paper_trades"):
        columns = set(_table_columns(conn, "paper_trades"))
        for column in ("skip_reason", "liquidity_skip_reason", "realistic_execution_skip_reason"):
            if column not in columns:
                continue
            rows = conn.execute(
                f"""
                SELECT {column} AS reason, COUNT(*) AS count
                FROM paper_trades
                WHERE {column} IS NOT NULL
                  AND {column} != ''
                GROUP BY {column}
                """
            ).fetchall()
            for row in rows:
                counts[str(row["reason"])] += int(row["count"])
    return [
        {"reason": reason, "count": count}
        for reason, count in counts.most_common(5)
    ]


def _trade_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "paper_trades"):
        return {
            "trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "settled_trades": 0,
            "pnl": 0.0,
            "avg_roi": None,
            "wins": 0,
            "losses": 0,
            "latest_created_at": None,
        }
    rows = _paper_trade_rows(conn)
    done_rows = [
        row for row in rows
        if str(row.get("status") or "").lower() in {"closed", "settled"}
    ]
    pnls = [value for value in (_result_pnl(row) for row in done_rows) if value is not None]
    rois = [value for value in (_result_roi(row) for row in done_rows) if value is not None]
    status_counts = Counter(str(row.get("status") or "").lower() for row in rows)
    return {
        "trades": len(rows),
        "open_trades": int(status_counts.get("open", 0)),
        "awaiting_resolution_trades": int(status_counts.get("awaiting_resolution", 0)),
        "closed_trades": int(status_counts.get("closed", 0)),
        "settled_trades": int(status_counts.get("settled", 0)),
        "skipped_trades": int(status_counts.get("skipped", 0)),
        "pnl": _round(sum(pnls)) if pnls else 0.0,
        "avg_roi": _round(sum(rois) / len(rois)) if rois else None,
        "wins": sum(1 for value in pnls if value > 0),
        "losses": sum(1 for value in pnls if value < 0),
        "latest_created_at": _latest_trade_timestamp(rows),
    }


def _paper_trade_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    columns = set(_table_columns(conn, "paper_trades"))
    if not columns:
        return []
    selected = [
        _column_or_null(columns, "created_at"),
        _column_or_null(columns, "signal_timestamp"),
        _column_or_null(columns, "exit_time"),
        _column_or_null(columns, "settled_at"),
        _column_or_null(columns, "status"),
        _column_or_null(columns, "pnl_usd"),
        _column_or_null(columns, "realized_pnl_usd"),
        _column_or_null(columns, "roi"),
        _column_or_null(columns, "realized_roi"),
    ]
    return [
        dict(row)
        for row in conn.execute(f"SELECT {', '.join(selected)} FROM paper_trades").fetchall()
    ]


def _result_pnl(row: dict[str, Any]) -> float | None:
    return _float_value(row.get("realized_pnl_usd"), row.get("pnl_usd"))


def _result_roi(row: dict[str, Any]) -> float | None:
    return _float_value(row.get("realized_roi"), row.get("roi"))


def _latest_trade_timestamp(rows: Sequence[dict[str, Any]]) -> str | None:
    values = [
        str(value)
        for row in rows
        for value in (
            row.get("created_at"),
            row.get("signal_timestamp"),
            row.get("exit_time"),
            row.get("settled_at"),
        )
        if value not in (None, "")
    ]
    return max(values) if values else None


def _column_or_null(columns: set[str], column: str) -> str:
    if column in columns:
        return column
    return f"NULL AS {column}"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    return [str(row[1]) for row in rows]


def _int_value(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _float_value(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _round(value: Any) -> float | None:
    parsed = _float_value(value)
    return round(parsed, 10) if parsed is not None else None


__all__ = [
    "COMPACT_HEARTBEAT_FIELDS",
    "build_compact_paper_live_status",
    "render_compact_paper_live_status",
]
