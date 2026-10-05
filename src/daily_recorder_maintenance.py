from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .db import run_sqlite_wal_checkpoint
from .models import to_iso


@dataclass(frozen=True, slots=True)
class MaintenanceTable:
    name: str
    time_expr: str


MAINTENANCE_TABLES: tuple[MaintenanceTable, ...] = (
    MaintenanceTable("market_snapshots", "timestamp"),
    MaintenanceTable("features", "timestamp"),
    MaintenanceTable("trades", "timestamp"),
    MaintenanceTable("btc_prices", "local_arrival_iso"),
    MaintenanceTable("order_book_levels", "timestamp"),
    MaintenanceTable("best_bid_ask_updates", "timestamp"),
    MaintenanceTable("tick_size_changes", "timestamp"),
    MaintenanceTable("market_events", "timestamp"),
    MaintenanceTable("raw_polymarket_events", "local_arrival_iso"),
    MaintenanceTable("recorder_metrics", "timestamp"),
    MaintenanceTable("markets", "COALESCE(close_time, end_time, last_updated, created_at, start_time)"),
)


def run_daily_recorder_maintenance(
    *,
    db_path: str,
    export_dir: str,
    archive_dir: str,
    keep_recent_hours: float = 12.0,
    checkpoint: bool = False,
    dry_run: bool = False,
    chunk_size: int = 50000,
    delete_batch_size: int = 50000,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    cutoff = now_dt - timedelta(hours=max(0.0, float(keep_recent_hours)))
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(db_path)

    before_sizes = _database_file_sizes(db)
    plan = _build_export_plan(db, cutoff=cutoff, now=now_dt, export_dir=Path(export_dir))
    report: dict[str, Any] = {
        "status": "dry_run" if dry_run else "ok",
        "dry_run": bool(dry_run),
        "db_path": str(db),
        "export_dir": str(export_dir),
        "archive_dir": str(archive_dir),
        "keep_recent_hours": float(keep_recent_hours),
        "cutoff_iso": to_iso(cutoff),
        "generated_at": to_iso(now_dt),
        "before_file_sizes": before_sizes,
        "after_file_sizes": before_sizes,
        "tables": plan,
        "manifest_path": None,
        "archive_path": None,
        "checkpoint_before_archive": None,
        "checkpoint_after_cleanup": None,
        "deleted_rows": {},
    }
    if dry_run:
        return report

    export_root = Path(export_dir)
    archive_root = Path(archive_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    _export_tables(db, plan, chunk_size=max(1, int(chunk_size)))
    verification = _verify_exported_counts(plan)
    report["export_verification"] = verification
    if not verification["ok"]:
        report["status"] = "export_verification_failed"
        report["after_file_sizes"] = _database_file_sizes(db)
        return report

    manifest_path = _write_manifest(
        export_root,
        db_path=db,
        plan=plan,
        report=report,
        now=now_dt,
    )
    report["manifest_path"] = str(manifest_path)

    if checkpoint:
        report["checkpoint_before_archive"] = run_sqlite_wal_checkpoint(
            str(db),
            mode="PASSIVE",
        )
    archive_path = _archive_sqlite_db(db, archive_root, now=now_dt)
    report["archive_path"] = str(archive_path)

    deleted_rows = _delete_exported_rows(
        db,
        plan,
        batch_size=max(1, int(delete_batch_size)),
    )
    report["deleted_rows"] = deleted_rows
    if checkpoint:
        report["checkpoint_after_cleanup"] = run_sqlite_wal_checkpoint(
            str(db),
            mode="TRUNCATE",
        )
    report["after_file_sizes"] = _database_file_sizes(db)
    return report


def render_daily_recorder_maintenance_report(report: dict[str, Any]) -> str:
    lines = [
        "Daily Recorder Maintenance",
        f"status={report.get('status')}",
        f"dry_run={str(bool(report.get('dry_run'))).lower()}",
        f"db_path={report.get('db_path')}",
        f"cutoff_iso={report.get('cutoff_iso')}",
        f"manifest_path={report.get('manifest_path')}",
        f"archive_path={report.get('archive_path')}",
        "",
        "Before Sizes",
    ]
    for key, value in dict(report.get("before_file_sizes") or {}).items():
        lines.append(f"{key}={value}")
    lines.extend(["", "After Sizes"])
    for key, value in dict(report.get("after_file_sizes") or {}).items():
        lines.append(f"{key}={value}")
    lines.extend(["", "Tables"])
    for table in report.get("tables") or []:
        lines.append(
            " | ".join(
                [
                    f"table={table.get('table')}",
                    f"exists={table.get('exists')}",
                    f"export_count={table.get('export_count')}",
                    f"deleted={dict(report.get('deleted_rows') or {}).get(table.get('table'), 0)}",
                    f"first={table.get('first_timestamp')}",
                    f"last={table.get('last_timestamp')}",
                    f"path={table.get('path')}",
                ]
            )
        )
    verification = report.get("export_verification")
    if verification:
        lines.extend(["", "Export Verification", json.dumps(verification, sort_keys=True)])
    if report.get("checkpoint_before_archive"):
        lines.extend(
            [
                "",
                "Checkpoint Before Archive",
                json.dumps(report["checkpoint_before_archive"], sort_keys=True),
            ]
        )
    if report.get("checkpoint_after_cleanup"):
        lines.extend(
            [
                "",
                "Checkpoint After Cleanup",
                json.dumps(report["checkpoint_after_cleanup"], sort_keys=True),
            ]
        )
    return "\n".join(lines) + "\n"


def _build_export_plan(
    db_path: Path,
    *,
    cutoff: datetime,
    now: datetime,
    export_dir: Path,
) -> list[dict[str, Any]]:
    conn = _connect_read_only(db_path)
    date_dir = export_dir / now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    try:
        plan: list[dict[str, Any]] = []
        for spec in MAINTENANCE_TABLES:
            if not _table_exists(conn, spec.name):
                plan.append(
                    {
                        "table": spec.name,
                        "exists": False,
                        "reason": "missing_table",
                        "export_count": 0,
                        "max_rowid": None,
                        "first_timestamp": None,
                        "last_timestamp": None,
                        "cutoff_iso": to_iso(cutoff),
                        "path": None,
                    }
                )
                continue
            time_expr = _resolve_time_expr(conn, spec)
            if time_expr is None:
                plan.append(
                    {
                        "table": spec.name,
                        "exists": False,
                        "reason": "missing_time_column",
                        "export_count": 0,
                        "max_rowid": None,
                        "first_timestamp": None,
                        "last_timestamp": None,
                        "cutoff_iso": to_iso(cutoff),
                        "path": None,
                    }
                )
                continue
            resolved_spec = MaintenanceTable(spec.name, time_expr)
            max_rowid = _max_rowid(conn, spec.name)
            export_count = _export_count(
                conn,
                resolved_spec,
                cutoff=cutoff,
                max_rowid=max_rowid,
            )
            timestamp_range = _timestamp_range(
                conn,
                resolved_spec,
                cutoff=cutoff,
                max_rowid=max_rowid,
            )
            plan.append(
                {
                    "table": spec.name,
                    "exists": True,
                    "time_expr": time_expr,
                    "export_count": export_count,
                    "max_rowid": max_rowid,
                    "first_timestamp": timestamp_range["first_timestamp"],
                    "last_timestamp": timestamp_range["last_timestamp"],
                    "cutoff_iso": to_iso(cutoff),
                    "path": str(date_dir / f"{stamp}_{spec.name}.parquet"),
                }
            )
        return plan
    finally:
        conn.close()


def _export_tables(db_path: Path, plan: Sequence[dict[str, Any]], *, chunk_size: int) -> None:
    conn = _connect_read_only(db_path)
    try:
        for table_plan in plan:
            if not table_plan.get("exists"):
                continue
            spec = MaintenanceTable(
                str(table_plan["table"]),
                str(table_plan.get("time_expr") or _table_spec(str(table_plan["table"])).time_expr),
            )
            path = Path(str(table_plan["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            _export_table_to_parquet(
                conn,
                spec,
                path=path,
                cutoff_iso=str(_plan_cutoff_marker(plan)),
                max_rowid=table_plan.get("max_rowid"),
                chunk_size=chunk_size,
            )
    finally:
        conn.close()


def _export_table_to_parquet(
    conn: sqlite3.Connection,
    spec: MaintenanceTable,
    *,
    path: Path,
    cutoff_iso: str,
    max_rowid: int | None,
    chunk_size: int,
) -> None:
    pa, pq = _load_pyarrow()
    columns = _table_columns(conn, spec.name)
    if max_rowid is None:
        table = pa.table({column: [] for column in columns})
        pq.write_table(table, path)
        return
    writer = None
    last_rowid = 0
    wrote_rows = False
    try:
        while True:
            rows = conn.execute(
                f"""
                SELECT rowid AS __maintenance_rowid__, *
                FROM {spec.name}
                WHERE rowid > ?
                  AND rowid <= ?
                  AND datetime({spec.time_expr}) < datetime(?)
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (last_rowid, int(max_rowid), cutoff_iso, int(chunk_size)),
            ).fetchall()
            if not rows:
                break
            last_rowid = int(rows[-1]["__maintenance_rowid__"])
            records = [
                {
                    key: row[key]
                    for key in row.keys()
                    if key != "__maintenance_rowid__"
                }
                for row in rows
            ]
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            wrote_rows = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote_rows:
        table = pa.table({column: [] for column in columns})
        pq.write_table(table, path)


def _verify_exported_counts(plan: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pq = _load_pyarrow()[1]
    mismatches: list[dict[str, Any]] = []
    checked: dict[str, int] = {}
    for table_plan in plan:
        if not table_plan.get("exists"):
            continue
        path = Path(str(table_plan["path"]))
        try:
            actual = int(pq.read_metadata(path).num_rows)
        except Exception as exc:  # pragma: no cover - defensive path.
            mismatches.append(
                {
                    "table": table_plan["table"],
                    "expected": int(table_plan.get("export_count") or 0),
                    "actual": None,
                    "error": str(exc),
                }
            )
            continue
        expected = int(table_plan.get("export_count") or 0)
        checked[str(table_plan["table"])] = actual
        if actual != expected:
            mismatches.append(
                {
                    "table": table_plan["table"],
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "ok": not mismatches,
        "checked_row_counts": checked,
        "mismatches": mismatches,
    }


def _write_manifest(
    export_dir: Path,
    *,
    db_path: Path,
    plan: Sequence[dict[str, Any]],
    report: dict[str, Any],
    now: datetime,
) -> Path:
    date_dir = export_dir / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / f"{now.strftime('%Y%m%dT%H%M%SZ')}_manifest.json"
    payload = {
        "generated_at": report["generated_at"],
        "db_path": str(db_path),
        "cutoff_iso": report["cutoff_iso"],
        "keep_recent_hours": report["keep_recent_hours"],
        "before_file_sizes": report["before_file_sizes"],
        "tables": list(plan),
        "export_verification": report.get("export_verification"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _archive_sqlite_db(db_path: Path, archive_dir: Path, *, now: datetime) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{db_path.stem}_maintenance_{now.strftime('%Y%m%dT%H%M%SZ')}{db_path.suffix}"
    source = _connect_read_only(db_path)
    dest = sqlite3.connect(str(archive_path))
    try:
        source.backup(dest)
        dest.execute("PRAGMA quick_check").fetchone()
    finally:
        dest.close()
        source.close()
    return archive_path


def _delete_exported_rows(
    db_path: Path,
    plan: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    deleted: dict[str, int] = {}
    cutoff_iso = str(_plan_cutoff_marker(plan))
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys=OFF")
        for table_plan in plan:
            if not table_plan.get("exists"):
                continue
            spec = MaintenanceTable(
                str(table_plan["table"]),
                str(table_plan.get("time_expr") or _table_spec(str(table_plan["table"])).time_expr),
            )
            max_rowid = table_plan.get("max_rowid")
            if max_rowid is None:
                deleted[spec.name] = 0
                continue
            table_deleted = 0
            while True:
                with conn:
                    cursor = conn.execute(
                        f"""
                        DELETE FROM {spec.name}
                        WHERE rowid IN (
                            SELECT rowid
                            FROM {spec.name}
                            WHERE rowid <= ?
                              AND datetime({spec.time_expr}) < datetime(?)
                            LIMIT ?
                        )
                        """,
                        (int(max_rowid), cutoff_iso, int(batch_size)),
                    )
                batch_deleted = int(cursor.rowcount or 0)
                table_deleted += batch_deleted
                if batch_deleted < int(batch_size):
                    break
            deleted[spec.name] = table_deleted
        return deleted
    finally:
        conn.close()


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
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _resolve_time_expr(
    conn: sqlite3.Connection,
    spec: MaintenanceTable,
) -> str | None:
    columns = set(_table_columns(conn, spec.name))
    if spec.name == "markets":
        candidates = [
            column
            for column in ("close_time", "end_time", "last_updated", "created_at", "start_time")
            if column in columns
        ]
        return f"COALESCE({', '.join(candidates)})" if candidates else None
    if spec.time_expr in columns:
        return spec.time_expr
    return None


def _max_rowid(conn: sqlite3.Connection, table: str) -> int | None:
    row = conn.execute(f"SELECT MAX(rowid) FROM {table}").fetchone()
    value = row[0] if row is not None else None
    return int(value) if value is not None else None


def _export_count(
    conn: sqlite3.Connection,
    spec: MaintenanceTable,
    *,
    cutoff: datetime,
    max_rowid: int | None,
) -> int:
    if max_rowid is None:
        return 0
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {spec.name}
        WHERE rowid <= ?
          AND datetime({spec.time_expr}) < datetime(?)
        """,
        (int(max_rowid), to_iso(cutoff)),
    ).fetchone()
    return int(row[0] or 0)


def _timestamp_range(
    conn: sqlite3.Connection,
    spec: MaintenanceTable,
    *,
    cutoff: datetime,
    max_rowid: int | None,
) -> dict[str, str | None]:
    if max_rowid is None:
        return {"first_timestamp": None, "last_timestamp": None}
    row = conn.execute(
        f"""
        SELECT MIN({spec.time_expr}), MAX({spec.time_expr})
        FROM {spec.name}
        WHERE rowid <= ?
          AND datetime({spec.time_expr}) < datetime(?)
        """,
        (int(max_rowid), to_iso(cutoff)),
    ).fetchone()
    return {
        "first_timestamp": row[0] if row is not None else None,
        "last_timestamp": row[1] if row is not None else None,
    }


def _table_spec(table: str) -> MaintenanceTable:
    for spec in MAINTENANCE_TABLES:
        if spec.name == table:
            return spec
    raise KeyError(table)


def _plan_cutoff_marker(plan: Sequence[dict[str, Any]]) -> str:
    # Every table in the plan uses the same cutoff; encode it once via the first table path metadata.
    # The caller writes the real cutoff into the plan before this helper is used.
    for item in plan:
        cutoff = item.get("cutoff_iso")
        if cutoff:
            return str(cutoff)
    raise ValueError("maintenance plan missing cutoff_iso")


def _database_file_sizes(db_path: Path) -> dict[str, int]:
    return {
        "db_file_size_bytes": _file_size(db_path),
        "wal_file_size_bytes": _file_size(Path(f"{db_path}-wal")),
        "shm_file_size_bytes": _file_size(Path(f"{db_path}-shm")),
    }


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Daily recorder maintenance requires pyarrow. Install project dependencies first."
        ) from exc
    return pa, pq


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "MAINTENANCE_TABLES",
    "run_daily_recorder_maintenance",
    "render_daily_recorder_maintenance_report",
]
