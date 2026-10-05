from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote
import glob

from .models import to_iso


ARCHIVE_EXPORT_TABLES: tuple[str, ...] = (
    "markets",
    "market_snapshots",
    "features",
    "trades",
    "btc_prices",
)

_TIME_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "markets": ("close_time", "end_time", "last_updated", "created_at", "start_time"),
    "market_snapshots": ("timestamp",),
    "features": ("timestamp",),
    "trades": ("timestamp",),
    "btc_prices": ("local_arrival_iso", "exchange_timestamp"),
}


class RecorderArchiveExportError(RuntimeError):
    pass


def export_recorder_archive_to_parquet(
    *,
    archive_db_inputs: Sequence[str],
    output_dir: str,
    run_id: str | None = None,
    append: bool = False,
    tables: Sequence[str] = ARCHIVE_EXPORT_TABLES,
    chunk_size: int = 50_000,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    archive_paths, unmatched_inputs = resolve_archive_db_paths(archive_db_inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_reports: list[dict[str, Any]] = []
    for archive_path in archive_paths:
        source_reports.append(
            _export_one_archive(
                archive_path=Path(archive_path),
                output_dir=output,
                run_id=run_id,
                append=bool(append),
                tables=tables,
                chunk_size=max(1, int(chunk_size)),
                now=now_dt,
            )
        )

    report = {
        "status": "ok" if archive_paths else "no_archives_matched",
        "created_at": to_iso(now_dt),
        "archive_db_inputs": list(archive_db_inputs),
        "archive_db_paths": archive_paths,
        "unmatched_inputs": unmatched_inputs,
        "output_dir": str(output),
        "run_id": run_id,
        "append": bool(append),
        "sources": source_reports,
        "aggregate_row_counts": _aggregate_row_counts(source_reports),
        "manifest_path": None,
        "next_commands": build_retrain_pipeline_commands(
            archive_db_paths=archive_paths,
            output_dir=str(output),
            timestamp=now_dt.strftime("%Y%m%dT%H%M%SZ"),
        ),
        "safety": {
            "live_trading_enabled": False,
            "models_promoted_automatically": False,
            "recorder_dbs_opened_read_only": True,
        },
    }
    manifest_path = _write_run_manifest(output, report, now=now_dt)
    report["manifest_path"] = str(manifest_path)
    return report


def render_recorder_archive_export_report(report: dict[str, Any]) -> str:
    lines = [
        "Recorder Archive Parquet Export",
        f"status={report.get('status')}",
        f"created_at={report.get('created_at')}",
        f"output_dir={report.get('output_dir')}",
        f"manifest_path={report.get('manifest_path')}",
        f"run_id={report.get('run_id')}",
        f"append={str(bool(report.get('append'))).lower()}",
        "",
        "Sources",
    ]
    if not report.get("sources"):
        lines.append("  none")
    for source in report.get("sources") or []:
        lines.append(
            "  {source} status={status} source_id={source_id} manifest={manifest}".format(
                source=source.get("source_db_path"),
                status=source.get("status"),
                source_id=source.get("source_id"),
                manifest=source.get("source_manifest_path"),
            )
        )
        for table in source.get("tables") or []:
            lines.append(
                "    table={table} rows={rows} first={first} last={last} path={path} status={status}".format(
                    table=table.get("table"),
                    rows=table.get("row_count"),
                    first=table.get("first_timestamp"),
                    last=table.get("last_timestamp"),
                    path=table.get("path"),
                    status=table.get("status"),
                )
            )
    lines.extend(["", "Aggregate Row Counts"])
    for table, count in sorted(dict(report.get("aggregate_row_counts") or {}).items()):
        lines.append(f"  {table}={count}")
    lines.extend(["", "Next Offline Commands"])
    for command in report.get("next_commands") or []:
        lines.append(f"  {command}")
    lines.append("")
    lines.append("Safety: no live trading, no automatic model promotion.")
    return "\n".join(lines) + "\n"


def resolve_archive_db_paths(inputs: Sequence[str]) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for raw in inputs:
        text = str(raw).strip()
        if not text:
            continue
        matches = sorted(glob.glob(text)) if glob.has_magic(text) else [text]
        if glob.has_magic(text) and not matches:
            unmatched.append(text)
            continue
        for match in matches:
            path = str(Path(match))
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths, unmatched


def build_retrain_pipeline_commands(
    *,
    archive_db_paths: Sequence[str],
    output_dir: str,
    timestamp: str,
) -> list[str]:
    first_archive = archive_db_paths[0] if archive_db_paths else "<archive-recorder.db>"
    training_dataset = f"data/exports/training_dataset_archive_{timestamp}.parquet"
    sequence_dataset = f"data/exports/transformer_sequences_archive_{timestamp}.parquet"
    baseline_output = f"data/models/baseline_retrain_{timestamp}"
    research_output = f"data/models/research_retrain_{timestamp}"
    transformer_output = f"data/models/transformer_sequence_retrain_{timestamp}"
    audit_output = f"data/analytics/model_audit/retrain_{timestamp}"
    return [
        (
            ".venv/bin/python -m src.main export-training-dataset "
            f"--db {first_archive} --output {training_dataset}"
        ),
        (
            ".venv/bin/python -m src.main train-baseline-model "
            f"--input {training_dataset} --output-dir {baseline_output}"
        ),
        (
            ".venv/bin/python -m src.main research-model-candidates "
            f"--db {first_archive} --output-dir {research_output} --test-market-fraction 0.25"
        ),
        (
            ".venv/bin/python -m src.main audit-baseline-model "
            f"--input {training_dataset} --model-dir {baseline_output} "
            f"--output-dir {audit_output} --split-by-market"
        ),
        (
            ".venv/bin/python -m src.main export-transformer-sequence-dataset "
            f"--input {training_dataset} --output {sequence_dataset} --sequence-length 120"
        ),
        (
            ".venv/bin/python -m src.main train-transformer-sequence-model "
            f"--input {sequence_dataset} --output-dir {transformer_output} "
            "--test-market-fraction 0.25 --validation-market-fraction 0.2"
        ),
        "# Review metrics before any promote-research-model-candidate command. No promotion is automatic.",
        f"# Raw archive table Parquet partitions are under {output_dir}.",
    ]


def _export_one_archive(
    *,
    archive_path: Path,
    output_dir: Path,
    run_id: str | None,
    append: bool,
    tables: Sequence[str],
    chunk_size: int,
    now: datetime,
) -> dict[str, Any]:
    if not archive_path.exists():
        return {
            "status": "missing",
            "source_db_path": str(archive_path),
            "error": "archive_db_missing",
            "tables": [],
        }
    source_id = _source_id(archive_path)
    source_manifest_path = output_dir / "manifests" / "sources" / f"{source_id}.json"
    if source_manifest_path.exists() and _source_outputs_exist(source_manifest_path):
        return _existing_source_report(source_manifest_path)
    if source_manifest_path.exists() and not append:
        raise RecorderArchiveExportError(
            f"archive source {archive_path} already has a manifest but one or more outputs "
            "are missing; pass --append to repair/rewrite the source partition"
        )
    conn = _connect_read_only(archive_path)
    try:
        table_reports = [
            _export_table(
                conn,
                table=str(table),
                source_id=source_id,
                output_dir=output_dir,
                run_id=run_id,
                chunk_size=chunk_size,
            )
            for table in tables
        ]
    finally:
        conn.close()
    report = {
        "status": "exported",
        "source_db_path": str(archive_path),
        "source_id": source_id,
        "source_size_bytes": archive_path.stat().st_size,
        "created_at": to_iso(now),
        "run_id": run_id,
        "source_manifest_path": str(source_manifest_path),
        "tables": table_reports,
    }
    _write_json(source_manifest_path, report)
    return report


def _export_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    source_id: str,
    output_dir: Path,
    run_id: str | None,
    chunk_size: int,
) -> dict[str, Any]:
    if not _table_exists(conn, table):
        return {
            "status": "missing_table",
            "table": table,
            "row_count": 0,
            "path": None,
            "first_timestamp": None,
            "last_timestamp": None,
        }
    columns = _table_columns(conn, table)
    where_sql, params = _where_clause(columns, run_id)
    row_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(table)} {where_sql}",
            params,
        ).fetchone()[0]
        or 0
    )
    first_timestamp, last_timestamp = _timestamp_range(conn, table, columns, where_sql, params)
    path = output_dir / table / f"source_id={source_id}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_table_parquet(
        conn,
        table=table,
        columns=columns,
        where_sql=where_sql,
        params=params,
        path=path,
        chunk_size=chunk_size,
    )
    return {
        "status": "exported",
        "table": table,
        "row_count": row_count,
        "path": str(path),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "run_id_filtered": bool(run_id and "run_id" in columns),
    }


def _write_table_parquet(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: Sequence[str],
    where_sql: str,
    params: Sequence[Any],
    path: Path,
    chunk_size: int,
) -> None:
    pa, pq = _load_pyarrow()
    writer = None
    wrote_rows = False
    offset = 0
    quoted_columns = ", ".join(_quote_ident(column) for column in columns)
    order_sql = "ORDER BY rowid ASC" if _has_rowid(conn, table) else ""
    try:
        while True:
            rows = conn.execute(
                f"""
                SELECT {quoted_columns}
                FROM {_quote_ident(table)}
                {where_sql}
                {order_sql}
                LIMIT ? OFFSET ?
                """,
                (*params, int(chunk_size), offset),
            ).fetchall()
            if not rows:
                break
            offset += len(rows)
            records = [_row_to_record(row, columns) for row in rows]
            arrow_table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(path, arrow_table.schema)
            writer.write_table(arrow_table)
            wrote_rows = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote_rows:
        pq.write_table(pa.table({column: [] for column in columns}), path)


def _timestamp_range(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    where_sql: str,
    params: Sequence[Any],
) -> tuple[str | None, str | None]:
    expr = _timestamp_expression(table, columns)
    if expr is None:
        return None, None
    row = conn.execute(
        f"""
        SELECT MIN({expr}) AS first_timestamp, MAX({expr}) AS last_timestamp
        FROM {_quote_ident(table)}
        {where_sql}
        """,
        params,
    ).fetchone()
    return row["first_timestamp"], row["last_timestamp"]


def _timestamp_expression(table: str, columns: Sequence[str]) -> str | None:
    available = set(columns)
    parts = [_quote_ident(column) for column in _TIME_EXPRESSIONS.get(table, ()) if column in available]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return "COALESCE(" + ", ".join(parts) + ")"


def _where_clause(columns: Sequence[str], run_id: str | None) -> tuple[str, tuple[Any, ...]]:
    if run_id and "run_id" in set(columns):
        return "WHERE run_id = ?", (run_id,)
    return "", ()


def _source_outputs_exist(source_manifest_path: Path) -> bool:
    try:
        report = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for table in report.get("tables") or []:
        path = table.get("path")
        if path and not Path(str(path)).exists():
            return False
    return True


def _existing_source_report(source_manifest_path: Path) -> dict[str, Any]:
    report = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    report = dict(report)
    report["status"] = "skipped_existing"
    return report


def _aggregate_row_counts(source_reports: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in source_reports:
        for table in source.get("tables") or []:
            table_name = str(table.get("table") or "")
            if not table_name:
                continue
            counts[table_name] = counts.get(table_name, 0) + int(table.get("row_count") or 0)
    return counts


def _write_run_manifest(output_dir: Path, report: dict[str, Any], *, now: datetime) -> Path:
    path = output_dir / "manifests" / f"{now.strftime('%Y%m%dT%H%M%SZ')}_manifest.json"
    payload = dict(report)
    payload["manifest_path"] = str(path)
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_id(path: Path) -> str:
    resolved = str(path.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    return digest[:16]


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    encoded = quote(str(db_path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
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


def _has_rowid(conn: sqlite3.Connection, table: str) -> bool:
    try:
        conn.execute(f"SELECT rowid FROM {_quote_ident(table)} LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def _row_to_record(row: sqlite3.Row, columns: Sequence[str]) -> dict[str, Any]:
    return {column: row[column] for column in columns}


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RecorderArchiveExportError(
            "export-recorder-archive-to-parquet requires pyarrow. "
            "Install project dependencies in the venv before running it."
        ) from exc
    return pa, pq


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "ARCHIVE_EXPORT_TABLES",
    "RecorderArchiveExportError",
    "build_retrain_pipeline_commands",
    "export_recorder_archive_to_parquet",
    "render_recorder_archive_export_report",
    "resolve_archive_db_paths",
]
