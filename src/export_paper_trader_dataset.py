from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paper_trader_analytics import (
    _build_summary,
    _enrich_trade,
    _mean,
    _table_exists,
    connect_paper_db_read_only,
)


TRADE_METADATA_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "export_generated_at",
    "paper_db_path",
    "notes",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "paper_db_path",
    "generated_at",
    "total_trades",
    "settled_trades",
    "open_trades",
    "awaiting_resolution_trades",
    "skipped_trades",
    "total_staked",
    "settled_pnl",
    "average_roi_settled",
    "win_rate_settled",
    "average_adjusted_entry_price",
    "average_estimated_edge",
    "average_predicted_probability",
    "warnings_json",
    "recommendation",
    "unresolved_exposure_json",
    "status_counts_json",
    "duplicate_market_keys_json",
    "notes",
)

BUCKET_TYPES: tuple[tuple[str, str], ...] = (
    ("probability", "probability_buckets"),
    ("edge", "edge_buckets"),
    ("time_until_resolution", "time_until_resolution_buckets"),
)


def export_paper_trader_dataset(
    *,
    paper_db_path: str,
    strategy_id: str,
    strategy_name: str,
    output_dir: str,
    append: bool = False,
    include_skipped: bool = False,
    include_awaiting: bool = False,
    notes: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = _iso(now or datetime.now(timezone.utc))
    strategy_id = _validate_label("strategy_id", strategy_id)
    strategy_name = _validate_label("strategy_name", strategy_name)
    output_root = Path(output_dir)
    strategy_dir = output_root / "strategies" / strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)

    try:
        analytics = _load_analytics_snapshot(
            paper_db_path=paper_db_path,
            now=now,
        )
    except FileNotFoundError:
        return _error(
            "paper_db_missing",
            paper_db_path=paper_db_path,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
        )
    if analytics["table_missing"]:
        return _error(
            "paper_trades_table_missing",
            paper_db_path=paper_db_path,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
        )

    summary = dict(analytics["summary"])
    all_trades = list(analytics["trades"])
    exported_trades = _filter_export_trades(
        all_trades,
        include_skipped=include_skipped,
        include_awaiting=include_awaiting,
    )
    trade_rows = [
        {
            **_metadata(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                generated_at=generated_at,
                paper_db_path=paper_db_path,
                notes=notes,
            ),
            **row,
        }
        for row in exported_trades
    ]
    summary_rows = [
        _summary_row(
            summary,
            trades=all_trades,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            paper_db_path=paper_db_path,
            generated_at=generated_at,
            notes=notes,
        )
    ]
    bucket_rows = _bucket_rows(
        summary,
        strategy_id=strategy_id,
    )

    strategy_files = {
        "trades_parquet": strategy_dir / "trades.parquet",
        "trades_csv": strategy_dir / "trades.csv",
        "summary_json": strategy_dir / "summary.json",
        "bucket_metrics_parquet": strategy_dir / "bucket_metrics.parquet",
        "llm_strategy_report_md": strategy_dir / "llm_strategy_report.md",
    }
    _write_parquet(trade_rows, strategy_files["trades_parquet"])
    _write_csv(trade_rows, strategy_files["trades_csv"])
    _write_json(
        strategy_files["summary_json"],
        {
            "status": "ok",
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "paper_db_path": paper_db_path,
            "generated_at": generated_at,
            "notes": notes,
            "summary": summary,
            "summary_row": summary_rows[0],
            "latest_heartbeat": analytics.get("latest_heartbeat"),
        },
    )
    _write_parquet(bucket_rows, strategy_files["bucket_metrics_parquet"])
    _write_text(
        strategy_files["llm_strategy_report_md"],
        _render_llm_strategy_report(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            paper_db_path=paper_db_path,
            generated_at=generated_at,
            notes=notes,
            summary=summary,
        ),
    )

    master_files: dict[str, str] = {}
    if append:
        master_specs = (
            ("all_trades_parquet", output_root / "all_trades.parquet", trade_rows),
            (
                "all_strategy_summaries_parquet",
                output_root / "all_strategy_summaries.parquet",
                summary_rows,
            ),
            (
                "all_bucket_metrics_parquet",
                output_root / "all_bucket_metrics.parquet",
                bucket_rows,
            ),
        )
        for key, path, rows in master_specs:
            _append_parquet(path, rows)
            master_files[key] = str(path)

    return {
        "status": "ok",
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "paper_db_path": paper_db_path,
        "output_dir": str(output_root),
        "strategy_dir": str(strategy_dir),
        "generated_at": generated_at,
        "notes": notes,
        "trades_loaded": len(all_trades),
        "trades_exported": len(trade_rows),
        "bucket_rows_exported": len(bucket_rows),
        "append": bool(append),
        "include_skipped": bool(include_skipped),
        "include_awaiting": bool(include_awaiting),
        "strategy_files": {key: str(path) for key, path in strategy_files.items()},
        "master_files": master_files,
        "summary": summary_rows[0],
    }


def _load_analytics_snapshot(
    *,
    paper_db_path: str,
    now: datetime | None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    conn = connect_paper_db_read_only(paper_db_path)
    try:
        if not _table_exists(conn, "paper_trades"):
            return {"table_missing": True, "trades": [], "summary": {}, "latest_heartbeat": None}
        raw_rows = [
            dict(row)
            for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
        ]
        latest_heartbeat = None
        heartbeat_missing = True
        if _table_exists(conn, "paper_trader_heartbeats"):
            heartbeat_missing = False
            heartbeat = conn.execute(
                """
                SELECT *
                FROM paper_trader_heartbeats
                ORDER BY datetime(timestamp) DESC, timestamp DESC
                LIMIT 1
                """
            ).fetchone()
            latest_heartbeat = dict(heartbeat) if heartbeat is not None else None
    finally:
        conn.close()

    enriched = [_enrich_trade(row, now=now_dt) for row in raw_rows]
    summary = _build_summary(
        enriched,
        paper_db_path=paper_db_path,
        output_dir="",
        latest_heartbeat=latest_heartbeat,
        base_warnings=[],
        table_missing=False,
        heartbeat_missing=heartbeat_missing,
        now=now_dt,
    )
    return {
        "table_missing": False,
        "trades": enriched,
        "summary": summary,
        "latest_heartbeat": latest_heartbeat,
    }


def _filter_export_trades(
    rows: Sequence[dict[str, Any]],
    *,
    include_skipped: bool,
    include_awaiting: bool,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "")
        if status == "skipped" and not include_skipped:
            continue
        if status == "awaiting_resolution" and not include_awaiting:
            continue
        filtered.append(dict(row))
    return filtered


def _metadata(
    *,
    strategy_id: str,
    strategy_name: str,
    generated_at: str,
    paper_db_path: str,
    notes: str | None,
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "export_generated_at": generated_at,
        "paper_db_path": paper_db_path,
        "notes": notes,
    }


def _summary_row(
    summary: dict[str, Any],
    *,
    trades: Sequence[dict[str, Any]],
    strategy_id: str,
    strategy_name: str,
    paper_db_path: str,
    generated_at: str,
    notes: str | None,
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "paper_db_path": paper_db_path,
        "generated_at": generated_at,
        "total_trades": int(summary.get("total_trades") or 0),
        "settled_trades": int(summary.get("settled_trades") or 0),
        "open_trades": int(summary.get("open_trades") or 0),
        "awaiting_resolution_trades": int(summary.get("awaiting_resolution_trades") or 0),
        "skipped_trades": int((summary.get("status_counts") or {}).get("skipped") or 0),
        "total_staked": summary.get("total_staked"),
        "settled_pnl": summary.get("settled_pnl"),
        "average_roi_settled": summary.get("average_roi_settled"),
        "win_rate_settled": summary.get("win_rate_settled"),
        "average_adjusted_entry_price": _mean(
            row.get("adjusted_entry_price") for row in trades
        ),
        "average_estimated_edge": summary.get("average_estimated_edge"),
        "average_predicted_probability": _mean(
            row.get("probability_for_direction") for row in trades
        ),
        "warnings_json": _json_dumps(summary.get("warnings") or []),
        "recommendation": summary.get("recommendation"),
        "unresolved_exposure_json": _json_dumps(summary.get("unresolved_exposure") or {}),
        "status_counts_json": _json_dumps(summary.get("status_counts") or {}),
        "duplicate_market_keys_json": _json_dumps(summary.get("duplicate_market_keys") or []),
        "notes": notes,
    }


def _bucket_rows(summary: dict[str, Any], *, strategy_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_type, summary_key in BUCKET_TYPES:
        for bucket_name, metrics in dict(summary.get(summary_key) or {}).items():
            metric_dict = dict(metrics or {})
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "bucket_type": bucket_type,
                    "bucket_name": bucket_name,
                    "trade_count": int(metric_dict.get("trade_count") or 0),
                    "settled_trades": int(metric_dict.get("settled_trades") or 0),
                    "win_rate": metric_dict.get("win_rate"),
                    "pnl": metric_dict.get("pnl"),
                    "average_roi": metric_dict.get("average_roi"),
                    "average_adjusted_entry_price": metric_dict.get(
                        "average_adjusted_entry_price"
                    ),
                }
            )
    return rows


def _render_llm_strategy_report(
    *,
    strategy_id: str,
    strategy_name: str,
    paper_db_path: str,
    generated_at: str,
    notes: str | None,
    summary: dict[str, Any],
) -> str:
    lines = [
        f"# Paper Trader Strategy Report: {strategy_name}",
        "",
        "## Strategy Metadata",
        f"- strategy_id: `{strategy_id}`",
        f"- strategy_name: {strategy_name}",
        f"- paper_db_path: `{paper_db_path}`",
        f"- generated_at: {generated_at}",
        f"- notes: {notes or ''}",
        "",
        "## Headline Results",
        f"- total_trades: {summary.get('total_trades', 0)}",
        f"- settled_trades: {summary.get('settled_trades', 0)}",
        f"- open_trades: {summary.get('open_trades', 0)}",
        f"- awaiting_resolution_trades: {summary.get('awaiting_resolution_trades', 0)}",
        f"- settled_pnl: {summary.get('settled_pnl')}",
        f"- win_rate_settled: {summary.get('win_rate_settled')}",
        f"- average_roi_settled: {summary.get('average_roi_settled')}",
        f"- average_estimated_edge: {summary.get('average_estimated_edge')}",
        "",
        "## Risk Notes",
        f"- unresolved_exposure: `{_json_dumps(summary.get('unresolved_exposure') or {})}`",
        f"- duplicate_market_keys: `{_json_dumps(summary.get('duplicate_market_keys') or [])}`",
        "",
        "## Probability Bucket Performance",
        *_render_bucket_lines(summary.get("probability_buckets") or {}),
        "",
        "## Edge Bucket Performance",
        *_render_bucket_lines(summary.get("edge_buckets") or {}),
        "",
        "## Time-Until-Resolution Performance",
        *_render_bucket_lines(summary.get("time_until_resolution_buckets") or {}),
        "",
        "## Warnings",
    ]
    warnings = list(summary.get("warnings") or [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Recommendation", str(summary.get("recommendation") or "")])
    return "\n".join(lines) + "\n"


def _render_bucket_lines(bucket_metrics: dict[str, Any]) -> list[str]:
    if not bucket_metrics:
        return ["- none"]
    lines: list[str] = []
    for bucket_name, metrics in bucket_metrics.items():
        metric_dict = dict(metrics or {})
        lines.append(
            "- "
            f"{bucket_name}: trades={metric_dict.get('trade_count', 0)}, "
            f"settled={metric_dict.get('settled_trades', 0)}, "
            f"win_rate={metric_dict.get('win_rate')}, "
            f"pnl={metric_dict.get('pnl')}, "
            f"avg_roi={metric_dict.get('average_roi')}"
        )
    return lines


def _append_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    existing = _read_parquet_rows(path) if path.exists() else []
    _write_parquet([*existing, *rows], path)


def _write_parquet(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pa, pq = _load_pyarrow()
    normalized = _normalize_rows(rows)
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table({})
    pq.write_table(table, path)


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    _pa, pq = _load_pyarrow()
    return pq.read_table(path).to_pylist()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(rows)
    fieldnames = list(normalized[0].keys()) if normalized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(normalized)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    preferred = [
        *TRADE_METADATA_COLUMNS,
        *SUMMARY_COLUMNS,
        "bucket_type",
        "bucket_name",
    ]
    columns = []
    seen: set[str] = set()
    for column in preferred:
        if any(column in row for row in rows) and column not in seen:
            columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return [{column: _scalar(row.get(column)) for column in columns} for row in rows]


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Paper trader dataset export requires pyarrow. Install project dependencies first."
        ) from exc
    return pa, pq


def _validate_label(name: str, value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    return value


def _error(
    error: str,
    *,
    paper_db_path: str,
    strategy_id: str,
    strategy_name: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error,
        "paper_db_path": paper_db_path,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
    }


__all__ = ["export_paper_trader_dataset"]
