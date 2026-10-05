from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .paper_trader_analytics import _table_exists, connect_paper_db_read_only


def export_meta_strategy_dataset(
    *,
    paper_db_path: str,
    output_path: str,
    output_csv_path: str | None = None,
    include_unresolved: bool = False,
    strategy_id: str | None = None,
    min_created_at: str | None = None,
    max_created_at: str | None = None,
) -> dict[str, Any]:
    try:
        conn = connect_paper_db_read_only(paper_db_path)
    except FileNotFoundError:
        return {"status": "error", "error": "paper_db_missing", "paper_db_path": paper_db_path}
    try:
        if not _table_exists(conn, "paper_trade_candidates"):
            return {
                "status": "error",
                "error": "paper_trade_candidates_table_missing",
                "paper_db_path": paper_db_path,
            }
        rows = _fetch_candidate_rows(
            conn,
            include_unresolved=include_unresolved,
            strategy_id=strategy_id,
            min_created_at=min_created_at,
            max_created_at=max_created_at,
        )
    finally:
        conn.close()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(rows, output)
    csv_size = None
    if output_csv_path:
        csv_output = Path(output_csv_path)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, csv_output)
        csv_size = _file_size(csv_output)

    return {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "output_path": str(output),
        "output_file_size_bytes": _file_size(output),
        "output_csv_path": output_csv_path,
        "output_csv_file_size_bytes": csv_size,
        "rows_exported": len(rows),
        "include_unresolved": bool(include_unresolved),
        "strategy_id": strategy_id,
        "min_created_at": min_created_at,
        "max_created_at": max_created_at,
    }


def _fetch_candidate_rows(
    conn: Any,
    *,
    include_unresolved: bool,
    strategy_id: str | None,
    min_created_at: str | None,
    max_created_at: str | None,
) -> list[dict[str, Any]]:
    where = ["1 = 1"]
    params: dict[str, Any] = {}
    if not include_unresolved:
        where.append("actual_resolved_label IS NOT NULL")
    if strategy_id:
        where.append("strategy_id = :strategy_id")
        params["strategy_id"] = strategy_id
    if min_created_at:
        _validate_timestamp_arg("min_created_at", min_created_at)
        where.append("datetime(created_at) >= datetime(:min_created_at)")
        params["min_created_at"] = min_created_at
    if max_created_at:
        _validate_timestamp_arg("max_created_at", max_created_at)
        where.append("datetime(created_at) <= datetime(:max_created_at)")
        params["max_created_at"] = max_created_at
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM paper_trade_candidates
            WHERE {" AND ".join(where)}
            ORDER BY datetime(created_at) ASC, id ASC
            """,
            params,
        ).fetchall()
    ]
    return [_with_labels(row) for row in rows]


def _with_labels(row: dict[str, Any]) -> dict[str, Any]:
    roi = _float_or_none(row.get("would_have_roi"))
    decision = str(row.get("decision") or "").upper()
    result = dict(row)
    result["profitable"] = None if roi is None else int(roi > 0)
    result["traded"] = int(decision == "TRADE")
    return result


def _write_parquet(rows: Sequence[dict[str, Any]], path: Path) -> None:
    pa, pq = _load_pyarrow()
    normalized = _normalize_rows(rows)
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table({})
    pq.write_table(table, path)


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    normalized = _normalize_rows(rows)
    fieldnames = list(normalized[0].keys()) if normalized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(normalized)


def _normalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    columns: list[str] = []
    seen: set[str] = set()
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
            "Meta-strategy dataset export requires pyarrow. Install project dependencies first."
        ) from exc
    return pa, pq


def _validate_timestamp_arg(name: str, value: str) -> None:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601 parseable") from exc


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


__all__ = ["export_meta_strategy_dataset"]
