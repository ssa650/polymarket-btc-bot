from __future__ import annotations

import glob
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from .compact_paper_live_status import build_compact_paper_live_status
from .db import _wal_size_warnings


STALE_RECORDER_SEC = 10 * 60
STALE_PREDICTION_SEC = 10 * 60
STALE_PAPER_ACTIVITY_SEC = 2 * 60 * 60


def build_overnight_health_report(
    *,
    recorder_db_path: str,
    prediction_db_inputs: Sequence[str] | None = None,
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prediction_paths, prediction_unmatched = _resolve_path_inputs(prediction_db_inputs or [])
    paper_paths, paper_unmatched = _resolve_path_inputs([
        *(paper_db_paths or []),
        *(paper_db_globs or []),
    ])
    report: dict[str, Any] = {
        "status": "ok",
        "generated_at": _iso(generated_at),
        "recorder": summarize_recorder_db(recorder_db_path, now=generated_at),
        "disk_wal": summarize_disk_wal(recorder_db_path),
        "transformer_predictions": summarize_transformer_predictions(
            prediction_paths,
            unmatched_inputs=prediction_unmatched,
            now=generated_at,
        ),
        "paper_dbs": summarize_paper_dbs(
            paper_paths,
            unmatched_inputs=paper_unmatched,
            now=generated_at,
        ),
        "red_flags": [],
    }
    report["red_flags"] = _collect_red_flags(report, now=generated_at)
    if report["red_flags"]:
        report["status"] = "red_flags"
    return report


def summarize_recorder_db(recorder_db_path: str, *, now: datetime | None = None) -> dict[str, Any]:
    path = Path(recorder_db_path)
    if not path.exists():
        return {"status": "missing", "db_path": str(path), "error": "recorder_db_missing"}
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        return _sqlite_error_summary("recorder", path, exc)
    try:
        tables = _table_names(conn)
        table_row_counts = _table_row_counts(conn, tables)
        features = _feature_summary(conn)
        btc_prices = _btc_price_summary(conn, now=now)
    except sqlite3.Error as exc:
        return _sqlite_error_summary("recorder", path, exc)
    finally:
        conn.close()
    feature_max_age_sec = _age_sec(features.get("max_timestamp"), now=now)
    return {
        "status": "ok",
        "db_path": str(path),
        "feature_rows": features.get("feature_rows", 0),
        "ready_rows": features.get("ready_rows", 0),
        "ready_pct": features.get("ready_pct"),
        "min_timestamp": features.get("min_timestamp"),
        "max_timestamp": features.get("max_timestamp"),
        "feature_max_age_sec": feature_max_age_sec,
        "btc_prices": btc_prices,
        "table_row_counts": table_row_counts,
    }


def summarize_disk_wal(recorder_db_path: str) -> dict[str, Any]:
    path = Path(recorder_db_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    try:
        disk_usage = shutil.disk_usage(parent)
    except FileNotFoundError:
        disk_usage = None
    db_bytes = _file_size(path)
    wal_bytes = _file_size(Path(f"{path}-wal"))
    shm_bytes = _file_size(Path(f"{path}-shm"))
    return {
        "status": "ok" if path.exists() else "missing",
        "db_path": str(path),
        "db_bytes": db_bytes,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "db_human": _human_bytes(db_bytes),
        "wal_human": _human_bytes(wal_bytes),
        "shm_human": _human_bytes(shm_bytes),
        "disk_total_bytes": disk_usage.total if disk_usage else None,
        "disk_used_bytes": disk_usage.used if disk_usage else None,
        "disk_free_bytes": disk_usage.free if disk_usage else None,
        "disk_used_human": _human_bytes(disk_usage.used) if disk_usage else None,
        "disk_free_human": _human_bytes(disk_usage.free) if disk_usage else None,
        "wal_size_warnings": _wal_size_warnings(wal_bytes),
        "note": "Long-lived SQLite readers can prevent WAL truncation.",
    }


def summarize_transformer_predictions(
    prediction_db_paths: Sequence[str],
    *,
    unmatched_inputs: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    db_reports = [_prediction_db_summary(path, now=now) for path in prediction_db_paths]
    aggregate = {
        "total_rows": 0,
        "probability_rows": 0,
        "market_count": 0,
        "min_probability_yes": None,
        "avg_probability_yes": None,
        "max_probability_yes": None,
        "count_probability_gte_075": 0,
        "count_probability_gte_080": 0,
        "count_probability_lte_030": 0,
        "count_probability_lte_025": 0,
        "min_timestamp": None,
        "max_timestamp": None,
        "latest_probability_timestamp": None,
        "latest_probability_age_sec": None,
        "top_reasons": [],
    }
    market_ids: set[str] = set()
    probability_sum = 0.0
    probability_min: float | None = None
    probability_max: float | None = None
    reasons: Counter[str] = Counter()
    for db_report in db_reports:
        if db_report.get("status") != "ok":
            continue
        aggregate["total_rows"] += int(db_report.get("total_rows") or 0)
        aggregate["probability_rows"] += int(db_report.get("probability_rows") or 0)
        aggregate["count_probability_gte_075"] += int(db_report.get("count_probability_gte_075") or 0)
        aggregate["count_probability_gte_080"] += int(db_report.get("count_probability_gte_080") or 0)
        aggregate["count_probability_lte_030"] += int(db_report.get("count_probability_lte_030") or 0)
        aggregate["count_probability_lte_025"] += int(db_report.get("count_probability_lte_025") or 0)
        market_ids.update(str(item) for item in db_report.get("market_ids", []) if item)
        probability_sum += float(db_report.get("probability_sum") or 0.0)
        probability_min = _min_optional(probability_min, db_report.get("min_probability_yes"))
        probability_max = _max_optional(probability_max, db_report.get("max_probability_yes"))
        aggregate["min_timestamp"] = _min_text_timestamp(
            aggregate["min_timestamp"],
            db_report.get("min_timestamp"),
        )
        aggregate["max_timestamp"] = _max_text_timestamp(
            aggregate["max_timestamp"],
            db_report.get("max_timestamp"),
        )
        aggregate["latest_probability_timestamp"] = _max_text_timestamp(
            aggregate["latest_probability_timestamp"],
            db_report.get("latest_probability_timestamp"),
        )
        for item in db_report.get("top_reasons", []):
            reason = item.get("reason")
            if reason:
                reasons[str(reason)] += int(item.get("count") or 0)
    aggregate["market_count"] = len(market_ids)
    aggregate["min_probability_yes"] = _round(probability_min)
    aggregate["max_probability_yes"] = _round(probability_max)
    aggregate["avg_probability_yes"] = (
        _round(probability_sum / aggregate["probability_rows"])
        if aggregate["probability_rows"]
        else None
    )
    aggregate["latest_probability_age_sec"] = _age_sec(
        aggregate.get("latest_probability_timestamp"),
        now=now,
    )
    aggregate["top_reasons"] = [
        {"reason": reason, "count": count}
        for reason, count in reasons.most_common(10)
    ]
    return {
        "status": "ok" if db_reports or not unmatched_inputs else "missing_inputs",
        "db_count": len(db_reports),
        "unmatched_inputs": list(unmatched_inputs or []),
        "aggregate": aggregate,
        "databases": [
            {key: value for key, value in item.items() if key not in {"market_ids", "probability_sum"}}
            for item in db_reports
        ],
    }


def summarize_paper_dbs(
    paper_db_paths: Sequence[str],
    *,
    unmatched_inputs: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    db_reports = [_paper_db_summary(path, now=now) for path in paper_db_paths]
    aggregate = {
        "db_count": len(db_reports),
        "total_rows": sum(int(item.get("summary", {}).get("trades") or 0) for item in db_reports),
        "open_trades": sum(int(item.get("summary", {}).get("open_trades") or 0) for item in db_reports),
        "closed_trades": sum(int(item.get("summary", {}).get("closed_trades") or 0) for item in db_reports),
        "settled_trades": sum(int(item.get("summary", {}).get("settled_trades") or 0) for item in db_reports),
        "skipped_trades": sum(int(item.get("summary", {}).get("skipped_trades") or 0) for item in db_reports),
        "pnl": _round(sum(float(item.get("summary", {}).get("pnl") or 0.0) for item in db_reports)),
    }
    return {
        "status": "ok" if db_reports or not unmatched_inputs else "missing_inputs",
        "unmatched_inputs": list(unmatched_inputs or []),
        "aggregate": aggregate,
        "databases": db_reports,
    }


def render_overnight_health_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Overnight Health Report")
    lines.append(f"generated_at: {report.get('generated_at')}")
    lines.append(f"status: {report.get('status')}")
    lines.append("")
    _render_recorder(lines, report.get("recorder") or {})
    lines.append("")
    _render_disk_wal(lines, report.get("disk_wal") or {})
    lines.append("")
    _render_transformer_predictions(lines, report.get("transformer_predictions") or {})
    lines.append("")
    _render_paper_dbs(lines, report.get("paper_dbs") or {})
    lines.append("")
    lines.append("Red Flags")
    red_flags = report.get("red_flags") or []
    if not red_flags:
        lines.append("  none")
    else:
        for flag in red_flags:
            lines.append(f"  - {flag.get('code')}: {flag.get('message')}")
    return "\n".join(lines) + "\n"


def write_overnight_health_report_json(report: dict[str, Any], output_json_path: str | Path) -> None:
    path = Path(output_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _prediction_db_summary(path_value: str, *, now: datetime | None = None) -> dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {"status": "missing", "db_path": str(path), "error": "prediction_db_missing"}
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        return _sqlite_error_summary("prediction", path, exc)
    try:
        if not _table_exists(conn, "transformer_predictions"):
            return {
                "status": "missing_table",
                "db_path": str(path),
                "error": "transformer_predictions_missing",
            }
        columns = set(_table_columns(conn, "transformer_predictions"))
        if "probability_yes" not in columns:
            return {
                "status": "missing_column",
                "db_path": str(path),
                "error": "probability_yes_missing",
                "available_columns": sorted(columns),
            }
        timestamp_expr = _timestamp_expr(columns)
        reason_expr = _reason_expr(columns)
        total_rows = int(conn.execute("SELECT COUNT(*) FROM transformer_predictions").fetchone()[0] or 0)
        prob_stats = conn.execute(
            f"""
            SELECT
              COUNT(probability_yes) AS probability_rows,
              MIN(probability_yes) AS min_probability_yes,
              AVG(probability_yes) AS avg_probability_yes,
              MAX(probability_yes) AS max_probability_yes,
              SUM(CASE WHEN probability_yes >= 0.75 THEN 1 ELSE 0 END) AS count_probability_gte_075,
              SUM(CASE WHEN probability_yes >= 0.80 THEN 1 ELSE 0 END) AS count_probability_gte_080,
              SUM(CASE WHEN probability_yes <= 0.30 THEN 1 ELSE 0 END) AS count_probability_lte_030,
              SUM(CASE WHEN probability_yes <= 0.25 THEN 1 ELSE 0 END) AS count_probability_lte_025,
              SUM(CASE WHEN probability_yes IS NOT NULL THEN probability_yes ELSE 0 END) AS probability_sum
            FROM transformer_predictions
            """
        ).fetchone()
        timestamps = conn.execute(
            f"""
            SELECT
              MIN({timestamp_expr}) AS min_timestamp,
              MAX({timestamp_expr}) AS max_timestamp,
              MAX(CASE WHEN probability_yes IS NOT NULL THEN {timestamp_expr} ELSE NULL END)
                AS latest_probability_timestamp
            FROM transformer_predictions
            """
        ).fetchone()
        market_ids = _distinct_values(conn, "transformer_predictions", "market_id", columns)
        top_reasons = []
        if reason_expr != "NULL":
            reason_rows = conn.execute(
                f"""
                SELECT {reason_expr} AS reason, COUNT(*) AS count
                FROM transformer_predictions
                WHERE {reason_expr} IS NOT NULL
                  AND {reason_expr} != ''
                GROUP BY {reason_expr}
                ORDER BY count DESC, reason
                LIMIT 10
                """
            ).fetchall()
            top_reasons = [
                {"reason": str(row["reason"]), "count": int(row["count"])}
                for row in reason_rows
            ]
    except sqlite3.Error as exc:
        return _sqlite_error_summary("prediction", path, exc)
    finally:
        conn.close()
    return {
        "status": "ok",
        "db_path": str(path),
        "total_rows": total_rows,
        "probability_rows": int(prob_stats["probability_rows"] or 0),
        "market_count": len(market_ids),
        "market_ids": market_ids,
        "min_probability_yes": _round(prob_stats["min_probability_yes"]),
        "avg_probability_yes": _round(prob_stats["avg_probability_yes"]),
        "max_probability_yes": _round(prob_stats["max_probability_yes"]),
        "probability_sum": float(prob_stats["probability_sum"] or 0.0),
        "count_probability_gte_075": int(prob_stats["count_probability_gte_075"] or 0),
        "count_probability_gte_080": int(prob_stats["count_probability_gte_080"] or 0),
        "count_probability_lte_030": int(prob_stats["count_probability_lte_030"] or 0),
        "count_probability_lte_025": int(prob_stats["count_probability_lte_025"] or 0),
        "min_timestamp": timestamps["min_timestamp"],
        "max_timestamp": timestamps["max_timestamp"],
        "latest_probability_timestamp": timestamps["latest_probability_timestamp"],
        "latest_probability_age_sec": _age_sec(timestamps["latest_probability_timestamp"], now=now),
        "top_reasons": top_reasons,
    }


def _paper_db_summary(path_value: str, *, now: datetime | None = None) -> dict[str, Any]:
    path = Path(path_value)
    compact_report = build_compact_paper_live_status(str(path), limit=1, verbose=False)
    if compact_report.get("status") != "ok":
        return {
            "status": compact_report.get("status"),
            "db_path": str(path),
            "summary": compact_report.get("summary", {}),
            "strategies": [],
            "top_rejection_reasons": compact_report.get("heartbeats", [{}])[0].get(
                "top_rejection_reasons",
                [],
            ) if compact_report.get("heartbeats") else [],
            "error": compact_report.get("error"),
        }
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        return _sqlite_error_summary("paper", path, exc)
    try:
        rows = _paper_trade_rows(conn)
        strategies = _strategy_summaries(rows)
        duplicates = _duplicate_strategy_market_groups(rows)
    except sqlite3.Error as exc:
        return _sqlite_error_summary("paper", path, exc)
    finally:
        conn.close()
    summary = compact_report.get("summary", {})
    top_rejections = []
    if compact_report.get("heartbeats"):
        top_rejections = compact_report["heartbeats"][0].get("top_rejection_reasons", [])
    return {
        "status": "ok",
        "db_path": str(path),
        "db_name": path.name,
        "heartbeat_source": compact_report.get("heartbeat_source"),
        "summary": summary,
        "latest_activity_age_sec": _age_sec(summary.get("latest_created_at"), now=now),
        "top_rejection_reasons": top_rejections,
        "strategies": strategies,
        "duplicate_strategy_market_groups": duplicates,
    }


def _paper_trade_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "paper_trades"):
        return []
    columns = set(_table_columns(conn, "paper_trades"))
    selected = [
        _column_or_null(columns, "strategy_id"),
        _column_or_null(columns, "market_id"),
        _column_or_null(columns, "created_at"),
        _column_or_null(columns, "signal_timestamp"),
        _column_or_null(columns, "status"),
        _column_or_null(columns, "realized_pnl_usd"),
        _column_or_null(columns, "pnl_usd"),
        _column_or_null(columns, "realized_roi"),
        _column_or_null(columns, "roi"),
        _column_or_null(columns, "skip_reason"),
        _column_or_null(columns, "liquidity_skip_reason"),
        _column_or_null(columns, "realistic_execution_skip_reason"),
    ]
    return [
        dict(row)
        for row in conn.execute(f"SELECT {', '.join(selected)} FROM paper_trades").fetchall()
    ]


def _strategy_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strategy_id = str(row.get("strategy_id") or "unknown")
        grouped[strategy_id].append(row)
    summaries = [_strategy_summary(strategy_id, group_rows) for strategy_id, group_rows in grouped.items()]
    summaries.sort(key=lambda item: (float(item.get("pnl") or 0.0), item.get("strategy_id") or ""), reverse=True)
    return summaries


def _strategy_summary(strategy_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status") or "").lower() for row in rows)
    done_rows = [row for row in rows if str(row.get("status") or "").lower() in {"closed", "settled"}]
    pnls = [_result_pnl(row) for row in done_rows]
    pnls = [value for value in pnls if value is not None]
    rois = [_result_roi(row) for row in done_rows]
    rois = [value for value in rois if value is not None]
    return {
        "strategy_id": strategy_id,
        "rows": len(rows),
        "trades": sum(1 for row in rows if str(row.get("status") or "").lower() != "skipped"),
        "open_trades": int(statuses.get("open", 0) + statuses.get("awaiting_resolution", 0)),
        "closed_trades": int(statuses.get("closed", 0)),
        "settled_trades": int(statuses.get("settled", 0)),
        "skipped_trades": int(statuses.get("skipped", 0)),
        "pnl": _round(sum(pnls)) if pnls else 0.0,
        "avg_roi": _round(sum(rois) / len(rois)) if rois else None,
        "wins": sum(1 for value in pnls if value > 0),
        "losses": sum(1 for value in pnls if value < 0),
    }


def _duplicate_strategy_market_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        status = str(row.get("status") or "").lower()
        if status == "skipped":
            continue
        strategy_id = str(row.get("strategy_id") or "")
        market_id = str(row.get("market_id") or "")
        if not strategy_id or not market_id:
            continue
        counts[(strategy_id, market_id)] += 1
    return [
        {"strategy_id": strategy_id, "market_id": market_id, "count": count}
        for (strategy_id, market_id), count in counts.most_common(20)
        if count > 1
    ]


def _feature_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "features"):
        return {
            "feature_rows": 0,
            "ready_rows": 0,
            "ready_pct": None,
            "min_timestamp": None,
            "max_timestamp": None,
        }
    columns = set(_table_columns(conn, "features"))
    timestamp_col = _first_existing(columns, ("timestamp", "feature_timestamp", "created_at"))
    ready_expr = "feature_ready" if "feature_ready" in columns else "0"
    timestamp_select = timestamp_col if timestamp_col else "NULL"
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS feature_rows,
          SUM(CASE WHEN {ready_expr} = 1 THEN 1 ELSE 0 END) AS ready_rows,
          MIN({timestamp_select}) AS min_timestamp,
          MAX({timestamp_select}) AS max_timestamp
        FROM features
        """
    ).fetchone()
    feature_rows = int(row["feature_rows"] or 0)
    ready_rows = int(row["ready_rows"] or 0)
    return {
        "feature_rows": feature_rows,
        "ready_rows": ready_rows,
        "ready_pct": _round((ready_rows / feature_rows) * 100.0) if feature_rows else None,
        "min_timestamp": row["min_timestamp"],
        "max_timestamp": row["max_timestamp"],
    }


def _btc_price_summary(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not _table_exists(conn, "btc_prices"):
        return {"status": "missing_table", "sources": {}, "stale_sources": []}
    columns = set(_table_columns(conn, "btc_prices"))
    required = {"source", "price"}
    if not required.issubset(columns):
        return {
            "status": "missing_columns",
            "sources": {},
            "stale_sources": [],
            "available_columns": sorted(columns),
        }
    exchange_expr = "exchange_timestamp" if "exchange_timestamp" in columns else "NULL"
    local_expr = "local_arrival_iso" if "local_arrival_iso" in columns else "NULL"
    id_expr = "id" if "id" in columns else "rowid"
    order_expr = "local_arrival_ns" if "local_arrival_ns" in columns else id_expr
    rows = conn.execute(
        f"""
        SELECT source, price, {exchange_expr} AS exchange_timestamp,
               {local_expr} AS local_arrival_iso
        FROM btc_prices p
        WHERE {id_expr} = (
            SELECT {id_expr}
            FROM btc_prices p2
            WHERE p2.source = p.source
            ORDER BY {order_expr} DESC, {id_expr} DESC
            LIMIT 1
        )
        ORDER BY source
        """
    ).fetchall()
    sources: dict[str, dict[str, Any]] = {}
    stale_sources: list[str] = []
    for row in rows:
        timestamp = row["exchange_timestamp"] or row["local_arrival_iso"]
        age_sec = _age_sec(timestamp, now=now)
        source = str(row["source"] or "")
        sources[source] = {
            "price": _round(row["price"]),
            "exchange_timestamp": row["exchange_timestamp"],
            "local_arrival_iso": row["local_arrival_iso"],
            "age_sec": age_sec,
            "stale": age_sec is None or age_sec > 15.0,
        }
        if sources[source]["stale"]:
            stale_sources.append(source)
    return {"status": "ok", "sources": sources, "stale_sources": stale_sources}


def _table_row_counts(conn: sqlite3.Connection, tables: Sequence[str]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for table in tables:
        if table.startswith("sqlite_"):
            continue
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0] or 0)
        except sqlite3.Error:
            counts[table] = None
    return counts


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _collect_red_flags(report: dict[str, Any], *, now: datetime | None = None) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    recorder = report.get("recorder") or {}
    if recorder.get("status") != "ok":
        flags.append({"code": "recorder_unavailable", "message": str(recorder.get("error") or recorder.get("status"))})
    elif _is_stale(recorder.get("max_timestamp"), STALE_RECORDER_SEC, now=now):
        flags.append({
            "code": "stale_recorder",
            "message": f"latest feature timestamp is older than {STALE_RECORDER_SEC // 60} minutes",
        })
    btc_prices = recorder.get("btc_prices") or {}
    for source, item in (btc_prices.get("sources") or {}).items():
        age_sec = item.get("age_sec")
        if age_sec is None or float(age_sec) > 15.0:
            flags.append({
                "code": "stale_btc_price",
                "message": f"{source}: BTC price age is {age_sec}s (> 15s)",
            })
    for warning in (report.get("disk_wal") or {}).get("wal_size_warnings") or []:
        flags.append({"code": "large_wal", "message": str(warning)})
    predictions = (report.get("transformer_predictions") or {}).get("aggregate") or {}
    unmatched_prediction_inputs = (report.get("transformer_predictions") or {}).get("unmatched_inputs") or []
    if unmatched_prediction_inputs:
        flags.append({
            "code": "transformer_prediction_inputs_unmatched",
            "message": "no transformer prediction DBs matched: " + ", ".join(map(str, unmatched_prediction_inputs[:3])),
        })
    if predictions.get("total_rows") or (report.get("transformer_predictions") or {}).get("db_count"):
        if not predictions.get("probability_rows"):
            flags.append({"code": "no_transformer_probability_rows", "message": "no transformer probability rows found"})
        elif _is_stale(predictions.get("latest_probability_timestamp"), STALE_PREDICTION_SEC, now=now):
            flags.append({
                "code": "stale_transformer_predictions",
                "message": "no transformer probability rows in the last 10 minutes",
            })
    unmatched_paper_inputs = (report.get("paper_dbs") or {}).get("unmatched_inputs") or []
    if unmatched_paper_inputs:
        flags.append({
            "code": "paper_db_inputs_unmatched",
            "message": "no paper DBs matched: " + ", ".join(map(str, unmatched_paper_inputs[:3])),
        })
    for db_report in (report.get("paper_dbs") or {}).get("databases") or []:
        if db_report.get("status") != "ok":
            if _is_locked_error(db_report.get("error")):
                flags.append({"code": "db_locked", "message": f"{db_report.get('db_path')}: {db_report.get('error')}"})
            continue
        summary = db_report.get("summary") or {}
        db_name = db_report.get("db_name") or db_report.get("db_path")
        if not int(summary.get("trades") or 0):
            flags.append({"code": "no_paper_trades", "message": f"{db_name}: no paper trades recorded"})
        elif _is_stale(summary.get("latest_created_at"), STALE_PAPER_ACTIVITY_SEC, now=now):
            flags.append({"code": "stale_paper_trades", "message": f"{db_name}: latest paper activity older than 2 hours"})
        if float(summary.get("pnl") or 0.0) < 0:
            flags.append({"code": "negative_pnl", "message": f"{db_name}: PnL is negative ({summary.get('pnl')})"})
        duplicates = db_report.get("duplicate_strategy_market_groups") or []
        if duplicates:
            top = duplicates[0]
            flags.append({
                "code": "duplicate_strategy_market_trades",
                "message": (
                    f"{db_name}: duplicate strategy/market trades, "
                    f"{top.get('strategy_id')} {top.get('market_id')} count={top.get('count')}"
                ),
            })
    for section_name in ("recorder", "transformer_predictions", "paper_dbs"):
        section = report.get(section_name) or {}
        for db_report in section.get("databases", []) if isinstance(section.get("databases"), list) else []:
            if _is_locked_error(db_report.get("error")):
                flags.append({"code": "db_locked", "message": f"{db_report.get('db_path')}: {db_report.get('error')}"})
    return flags


def _render_recorder(lines: list[str], recorder: dict[str, Any]) -> None:
    lines.append("Recorder")
    if recorder.get("status") != "ok":
        lines.append(f"  status={recorder.get('status')} error={recorder.get('error')}")
        return
    lines.append(
        "  features={feature_rows} ready={ready_rows} ({ready_pct}%) range={min_ts} -> {max_ts}".format(
            feature_rows=recorder.get("feature_rows"),
            ready_rows=recorder.get("ready_rows"),
            ready_pct=_fmt_number(recorder.get("ready_pct")),
            min_ts=recorder.get("min_timestamp"),
            max_ts=recorder.get("max_timestamp"),
        )
    )
    lines.append(f"  latest_feature_age_sec={_fmt_number(recorder.get('feature_max_age_sec'))}")
    btc_sources = (recorder.get("btc_prices") or {}).get("sources") or {}
    if btc_sources:
        rendered_btc = ", ".join(
            f"{source}={_fmt_number(item.get('price'))} age={_fmt_number(item.get('age_sec'))}s"
            for source, item in btc_sources.items()
        )
        lines.append(f"  btc_prices: {rendered_btc}")
    counts = recorder.get("table_row_counts") or {}
    if counts:
        rendered = ", ".join(f"{key}={value}" for key, value in list(counts.items())[:12])
        if len(counts) > 12:
            rendered += f", ... {len(counts) - 12} more"
        lines.append(f"  table_rows: {rendered}")


def _render_disk_wal(lines: list[str], disk_wal: dict[str, Any]) -> None:
    lines.append("Disk/WAL")
    lines.append(
        "  db={db} wal={wal} shm={shm} disk_used={used} disk_free={free}".format(
            db=disk_wal.get("db_human"),
            wal=disk_wal.get("wal_human"),
            shm=disk_wal.get("shm_human"),
            used=disk_wal.get("disk_used_human"),
            free=disk_wal.get("disk_free_human"),
        )
    )
    warnings = disk_wal.get("wal_size_warnings") or []
    lines.append(f"  wal_warnings: {', '.join(warnings) if warnings else 'none'}")


def _render_transformer_predictions(lines: list[str], predictions_section: dict[str, Any]) -> None:
    lines.append("Transformer Predictions")
    unmatched = predictions_section.get("unmatched_inputs") or []
    if unmatched:
        lines.append(f"  unmatched_inputs: {', '.join(unmatched)}")
    aggregate = predictions_section.get("aggregate") or {}
    lines.append(
        "  dbs={db_count} rows={rows} probability_rows={prob_rows} markets={markets}".format(
            db_count=predictions_section.get("db_count", 0),
            rows=aggregate.get("total_rows", 0),
            prob_rows=aggregate.get("probability_rows", 0),
            markets=aggregate.get("market_count", 0),
        )
    )
    lines.append(
        "  probability_yes min/avg/max={min_p}/{avg_p}/{max_p} >=0.75={gte75} >=0.80={gte80} <=0.30={lte30} <=0.25={lte25}".format(
            min_p=_fmt_number(aggregate.get("min_probability_yes")),
            avg_p=_fmt_number(aggregate.get("avg_probability_yes")),
            max_p=_fmt_number(aggregate.get("max_probability_yes")),
            gte75=aggregate.get("count_probability_gte_075", 0),
            gte80=aggregate.get("count_probability_gte_080", 0),
            lte30=aggregate.get("count_probability_lte_030", 0),
            lte25=aggregate.get("count_probability_lte_025", 0),
        )
    )
    lines.append(
        f"  latest_probability={aggregate.get('latest_probability_timestamp')} age_sec={_fmt_number(aggregate.get('latest_probability_age_sec'))}"
    )
    top_reasons = aggregate.get("top_reasons") or []
    if top_reasons:
        lines.append("  top_reasons: " + _render_counts(top_reasons, key="reason"))


def _render_paper_dbs(lines: list[str], paper_section: dict[str, Any]) -> None:
    lines.append("Paper DBs")
    unmatched = paper_section.get("unmatched_inputs") or []
    if unmatched:
        lines.append(f"  unmatched_inputs: {', '.join(unmatched)}")
    aggregate = paper_section.get("aggregate") or {}
    lines.append(
        "  dbs={db_count} rows={rows} open={open_trades} closed={closed} settled={settled} skipped={skipped} pnl={pnl}".format(
            db_count=aggregate.get("db_count", 0),
            rows=aggregate.get("total_rows", 0),
            open_trades=aggregate.get("open_trades", 0),
            closed=aggregate.get("closed_trades", 0),
            settled=aggregate.get("settled_trades", 0),
            skipped=aggregate.get("skipped_trades", 0),
            pnl=_fmt_number(aggregate.get("pnl")),
        )
    )
    for db_report in paper_section.get("databases", [])[:20]:
        if db_report.get("status") != "ok":
            lines.append(f"  {Path(str(db_report.get('db_path'))).name}: status={db_report.get('status')} error={db_report.get('error')}")
            continue
        summary = db_report.get("summary") or {}
        lines.append(
            "  {name}: rows={rows} open={open_trades} closed={closed} settled={settled} skipped={skipped} pnl={pnl} avg_roi={avg_roi} wins/losses={wins}/{losses} latest={latest}".format(
                name=db_report.get("db_name") or Path(str(db_report.get("db_path"))).name,
                rows=summary.get("trades", 0),
                open_trades=summary.get("open_trades", 0),
                closed=summary.get("closed_trades", 0),
                settled=summary.get("settled_trades", 0),
                skipped=summary.get("skipped_trades", 0),
                pnl=_fmt_number(summary.get("pnl")),
                avg_roi=_fmt_number(summary.get("avg_roi")),
                wins=summary.get("wins", 0),
                losses=summary.get("losses", 0),
                latest=summary.get("latest_created_at"),
            )
        )
        rejections = db_report.get("top_rejection_reasons") or []
        if rejections:
            lines.append(f"    top_rejections: {_render_counts(rejections, key='reason')}")
        strategies = db_report.get("strategies") or []
        if strategies:
            top = strategies[:3]
            lines.append(
                "    top_strategies: "
                + "; ".join(
                    f"{item.get('strategy_id')} pnl={_fmt_number(item.get('pnl'))} rows={item.get('rows')} closed={item.get('closed_trades')}"
                    for item in top
                )
            )
        duplicates = db_report.get("duplicate_strategy_market_groups") or []
        if duplicates:
            lines.append(f"    duplicate_strategy_market_groups={len(duplicates)} top={duplicates[0]}")


def _resolve_path_inputs(inputs: Iterable[str]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for raw in inputs:
        text = str(raw).strip()
        if not text:
            continue
        matches = sorted(glob.glob(text)) if glob.has_magic(text) else [text]
        if glob.has_magic(text) and not matches:
            unmatched.append(text)
        for match in matches:
            if match not in seen:
                seen.add(match)
                resolved.append(match)
    return resolved, unmatched


def _open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    return [str(row[1]) for row in rows]


def _column_or_null(columns: set[str], column: str) -> str:
    if column in columns:
        return _quote_ident(column)
    return f"NULL AS {_quote_ident(column)}"


def _first_existing(columns: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return _quote_ident(candidate)
    return None


def _timestamp_expr(columns: set[str]) -> str:
    for column in ("timestamp", "created_at", "latest_feature_timestamp", "latest_sequence_timestamp", "signal_timestamp"):
        if column in columns:
            return _quote_ident(column)
    return "NULL"


def _reason_expr(columns: set[str]) -> str:
    for column in ("reason", "diagnostics_reason", "rejection_reason"):
        if column in columns:
            return _quote_ident(column)
    return "NULL"


def _distinct_values(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    columns: set[str],
) -> list[str]:
    if column not in columns:
        return []
    rows = conn.execute(
        f"SELECT DISTINCT {_quote_ident(column)} AS value FROM {_quote_ident(table)} WHERE {_quote_ident(column)} IS NOT NULL"
    ).fetchall()
    return [str(row["value"]) for row in rows if row["value"] not in (None, "")]


def _result_pnl(row: dict[str, Any]) -> float | None:
    return _float_value(row.get("realized_pnl_usd"), row.get("pnl_usd"))


def _result_roi(row: dict[str, Any]) -> float | None:
    return _float_value(row.get("realized_roi"), row.get("roi"))


def _float_value(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _min_optional(current: float | None, value: Any) -> float | None:
    parsed = _float_value(value)
    if parsed is None:
        return current
    if current is None:
        return parsed
    return min(current, parsed)


def _max_optional(current: float | None, value: Any) -> float | None:
    parsed = _float_value(value)
    if parsed is None:
        return current
    if current is None:
        return parsed
    return max(current, parsed)


def _min_text_timestamp(current: str | None, value: Any) -> str | None:
    text = str(value) if value not in (None, "") else None
    if text is None:
        return current
    return text if current is None or text < current else current


def _max_text_timestamp(current: str | None, value: Any) -> str | None:
    text = str(value) if value not in (None, "") else None
    if text is None:
        return current
    return text if current is None or text > current else current


def _age_sec(timestamp: Any, *, now: datetime | None = None) -> float | None:
    parsed = _parse_datetime(timestamp)
    if parsed is None:
        return None
    base = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return _round((base - parsed).total_seconds())


def _is_stale(timestamp: Any, threshold_sec: float, *, now: datetime | None = None) -> bool:
    age = _age_sec(timestamp, now=now)
    return age is None or age > threshold_sec


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except FileNotFoundError:
        return 0


def _human_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    size = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or suffix == "TB":
            return f"{size:.1f}{suffix}"
        size /= 1024.0
    return f"{size:.1f}TB"


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _round(value: Any) -> float | None:
    parsed = _float_value(value)
    return round(parsed, 6) if parsed is not None else None


def _fmt_number(value: Any) -> str:
    parsed = _float_value(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _render_counts(items: Sequence[dict[str, Any]], *, key: str) -> str:
    return ", ".join(f"{item.get(key)}={item.get('count')}" for item in items[:5])


def _sqlite_error_summary(kind: str, path: Path, exc: sqlite3.Error) -> dict[str, Any]:
    error = str(exc)
    status = "locked" if _is_locked_error(error) else "error"
    return {
        "status": status,
        "db_path": str(path),
        "error": f"{kind}_db_{status}: {error}",
    }


def _is_locked_error(error: Any) -> bool:
    return "locked" in str(error or "").lower()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "build_overnight_health_report",
    "render_overnight_health_report",
    "summarize_disk_wal",
    "summarize_paper_dbs",
    "summarize_recorder_db",
    "summarize_transformer_predictions",
    "write_overnight_health_report_json",
]
