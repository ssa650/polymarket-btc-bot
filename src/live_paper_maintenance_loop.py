from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .baseline_paper_trader import (
    connect_paper_output_db,
    connect_recorder_read_only,
    ensure_paper_schema,
    settle_open_paper_trades,
)
from .export_meta_strategy_dataset import export_meta_strategy_dataset
from .models import to_iso
from .paper_trader_analytics import build_paper_trader_analytics_report


RefreshFn = Callable[..., dict[str, Any]]
AnalyticsFn = Callable[..., dict[str, Any]]
ExportFn = Callable[..., dict[str, Any]]
SleepFn = Callable[[float], None]
NowFn = Callable[[], datetime]


@dataclass(slots=True)
class LivePaperMaintenanceConfig:
    recorder_db_path: str
    paper_db_path: str
    export_dir: str
    analytics_dir: str
    poll_sec: float = 60.0
    resolution_older_than_minutes: float = 2.0
    export_every_minutes: float = 15.0
    analytics_every_minutes: float = 5.0
    resolution_limit: int = 100
    strategy_id: str | None = None
    dry_run: bool = False
    once: bool = False


@dataclass(slots=True)
class LivePaperMaintenanceState:
    last_analytics_at: datetime | None = None
    last_export_at: datetime | None = None
    last_analytics_paths: dict[str, str] | None = None
    last_export_paths: dict[str, str] | None = None


def run_live_paper_maintenance_loop(
    config: LivePaperMaintenanceConfig,
    *,
    resolution_refresh_fn: RefreshFn,
    analytics_fn: AnalyticsFn = build_paper_trader_analytics_report,
    export_fn: ExportFn = export_meta_strategy_dataset,
    max_iterations: int | None = None,
    sleep_fn: SleepFn = time.sleep,
    now_fn: NowFn | None = None,
    emit_logs: bool = True,
) -> dict[str, Any]:
    state = LivePaperMaintenanceState()
    iterations = 0
    last_report: dict[str, Any] | None = None
    while True:
        iterations += 1
        last_report = run_live_paper_maintenance_once(
            config,
            state=state,
            resolution_refresh_fn=resolution_refresh_fn,
            analytics_fn=analytics_fn,
            export_fn=export_fn,
            force_intervals=config.once,
            now_fn=now_fn,
            emit_logs=emit_logs,
        )
        if config.once or (max_iterations is not None and iterations >= int(max_iterations)):
            return {
                "status": "ok",
                "iterations": iterations,
                "last_poll": last_report,
            }
        sleep_fn(max(0.0, float(config.poll_sec)))


def run_live_paper_maintenance_once(
    config: LivePaperMaintenanceConfig,
    *,
    state: LivePaperMaintenanceState | None,
    resolution_refresh_fn: RefreshFn,
    analytics_fn: AnalyticsFn = build_paper_trader_analytics_report,
    export_fn: ExportFn = export_meta_strategy_dataset,
    force_intervals: bool = False,
    now_fn: NowFn | None = None,
    emit_logs: bool = True,
) -> dict[str, Any]:
    state = state or LivePaperMaintenanceState()
    now_dt = _utc_now(now_fn)
    errors: list[dict[str, Any]] = []
    last_resolution_refresh: dict[str, Any] | None = None
    settlement_report: dict[str, Any] = {"status": "skipped", "reason": "dry_run"}

    try:
        last_resolution_refresh = resolution_refresh_fn(
            apply=not bool(config.dry_run),
            now=now_dt,
            older_than_minutes=float(config.resolution_older_than_minutes),
            limit=int(config.resolution_limit),
        )
    except Exception as exc:
        error = _error_payload("resolution_refresh", exc)
        errors.append(error)
        _emit(emit_logs, "live_paper_maintenance_error", **error)

    if not config.dry_run:
        try:
            settlement_report = _settle_paper_state(
                recorder_db_path=config.recorder_db_path,
                paper_db_path=config.paper_db_path,
                now=now_dt,
            )
        except Exception as exc:
            error = _error_payload("paper_settlement", exc)
            errors.append(error)
            _emit(emit_logs, "live_paper_maintenance_error", **error)

    analytics_paths: dict[str, str] | None = None
    if _interval_due(
        state.last_analytics_at,
        every_minutes=float(config.analytics_every_minutes),
        now=now_dt,
        force=force_intervals,
    ):
        if config.dry_run:
            analytics_paths = {"status": "dry_run"}
        else:
            try:
                analytics_paths = _run_analytics_export(
                    config=config,
                    now=now_dt,
                    analytics_fn=analytics_fn,
                )
                state.last_analytics_at = now_dt
                state.last_analytics_paths = analytics_paths
            except Exception as exc:
                error = _error_payload("paper_analytics", exc)
                errors.append(error)
                _emit(emit_logs, "live_paper_maintenance_error", **error)

    export_paths: dict[str, str] | None = None
    if _interval_due(
        state.last_export_at,
        every_minutes=float(config.export_every_minutes),
        now=now_dt,
        force=force_intervals,
    ):
        if config.dry_run:
            export_paths = {"status": "dry_run"}
        else:
            try:
                export_paths = _run_meta_strategy_export(
                    config=config,
                    now=now_dt,
                    export_fn=export_fn,
                )
                state.last_export_at = now_dt
                state.last_export_paths = export_paths
            except Exception as exc:
                error = _error_payload("meta_strategy_export", exc)
                errors.append(error)
                _emit(emit_logs, "live_paper_maintenance_error", **error)

    counts = _paper_lifecycle_counts(config.paper_db_path)
    heartbeat = {
        "timestamp": to_iso(now_dt),
        "recorder_db_path": config.recorder_db_path,
        "paper_db_path": config.paper_db_path,
        "dry_run": bool(config.dry_run),
        **counts,
        "last_resolution_refresh": last_resolution_refresh,
        "settlement": settlement_report,
        "last_analytics_paths": analytics_paths or state.last_analytics_paths,
        "last_export_paths": export_paths or state.last_export_paths,
        "errors": errors,
    }
    _emit(emit_logs, "live_paper_maintenance_heartbeat", **heartbeat)
    return heartbeat


def _settle_paper_state(
    *,
    recorder_db_path: str,
    paper_db_path: str,
    now: datetime,
) -> dict[str, Any]:
    recorder = connect_recorder_read_only(recorder_db_path)
    output = connect_paper_output_db(paper_db_path)
    try:
        ensure_paper_schema(output)
        before = _paper_lifecycle_counts(paper_db_path)
        settled = settle_open_paper_trades(
            recorder,
            output,
            now=now,
            emit_logs=False,
        )
        after = _paper_lifecycle_counts(paper_db_path)
    finally:
        recorder.close()
        output.close()
    return {
        "status": "ok",
        "settled_trades": settled,
        "before": before,
        "after": after,
    }


def _run_analytics_export(
    *,
    config: LivePaperMaintenanceConfig,
    now: datetime,
    analytics_fn: AnalyticsFn,
) -> dict[str, str]:
    analytics_dir = Path(config.analytics_dir)
    analytics_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(now)
    temp_dir = analytics_dir / f".tmp_analytics_{stamp}_{uuid.uuid4().hex}"
    report = analytics_fn(
        paper_db_path=config.paper_db_path,
        output_dir=str(temp_dir),
        strategy_id=config.strategy_id,
        now=now,
    )
    timestamped_json = analytics_dir / f"{stamp}_analytics.json"
    timestamped_txt = analytics_dir / f"{stamp}_analytics.txt"
    source_json = Path(report["summary"]["output_files"]["json"])
    source_txt = Path(report["summary"]["output_files"]["txt"])
    _atomic_copy(source_json, timestamped_json)
    _atomic_copy(source_txt, timestamped_txt)
    latest_json = analytics_dir / "latest_analytics.json"
    latest_txt = analytics_dir / "latest_analytics.txt"
    _atomic_copy(timestamped_json, latest_json)
    _atomic_copy(timestamped_txt, latest_txt)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {
        "timestamped_json": str(timestamped_json),
        "timestamped_txt": str(timestamped_txt),
        "latest_json": str(latest_json),
        "latest_txt": str(latest_txt),
    }


def _run_meta_strategy_export(
    *,
    config: LivePaperMaintenanceConfig,
    now: datetime,
    export_fn: ExportFn,
) -> dict[str, str]:
    export_dir = Path(config.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(now)
    parquet_path = export_dir / f"{stamp}_meta_strategy_dataset.parquet"
    csv_path = export_dir / f"{stamp}_meta_strategy_dataset.csv"
    report = export_fn(
        paper_db_path=config.paper_db_path,
        output_path=str(parquet_path),
        output_csv_path=str(csv_path),
        include_unresolved=True,
        strategy_id=config.strategy_id,
    )
    if report.get("status") != "ok":
        raise RuntimeError(json.dumps(report, sort_keys=True))
    latest_parquet = export_dir / "latest_meta_strategy_dataset.parquet"
    latest_csv = export_dir / "latest_meta_strategy_dataset.csv"
    _atomic_copy(parquet_path, latest_parquet)
    _atomic_copy(csv_path, latest_csv)
    return {
        "timestamped_parquet": str(parquet_path),
        "timestamped_csv": str(csv_path),
        "latest_parquet": str(latest_parquet),
        "latest_csv": str(latest_csv),
        "rows_exported": str(report.get("rows_exported", 0)),
    }


def _paper_lifecycle_counts(paper_db_path: str) -> dict[str, Any]:
    path = Path(paper_db_path)
    if not path.exists():
        return {
            "open_trades": 0,
            "awaiting_resolution_trades": 0,
            "settled_trades": 0,
            "total_candidates": 0,
            "settled_candidates": 0,
            "skipped_candidates_with_would_have_roi": 0,
        }
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        trades = _status_counts(conn)
        return {
            "open_trades": trades.get("open", 0),
            "awaiting_resolution_trades": trades.get("awaiting_resolution", 0),
            "settled_trades": trades.get("settled", 0),
            "total_candidates": _table_count(conn, "paper_trade_candidates"),
            "settled_candidates": _candidate_count(
                conn,
                "actual_resolved_label IS NOT NULL AND actual_resolved_label != ''",
            ),
            "skipped_candidates_with_would_have_roi": _candidate_count(
                conn,
                "UPPER(COALESCE(decision, '')) != 'TRADE' AND would_have_roi IS NOT NULL",
            ),
        }
    finally:
        conn.close()


def _status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(conn, "paper_trades"):
        return {}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM paper_trades
        GROUP BY status
        """
    ).fetchall()
    return {str(row["status"] or ""): int(row["count"] or 0) for row in rows}


def _candidate_count(conn: sqlite3.Connection, where: str) -> int:
    if not _table_exists(conn, "paper_trade_candidates"):
        return 0
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM paper_trade_candidates WHERE {where}"
    ).fetchone()
    return int(row["count"] or 0)


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"] or 0)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table,),
    ).fetchone()
    return row is not None


def _interval_due(
    last_run_at: datetime | None,
    *,
    every_minutes: float,
    now: datetime,
    force: bool,
) -> bool:
    if force or last_run_at is None:
        return True
    return (now - last_run_at).total_seconds() >= float(every_minutes) * 60.0


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def _error_payload(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _utc_now(now_fn: NowFn | None) -> datetime:
    now = now_fn() if now_fn is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _stamp(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _emit(enabled: bool, event: str, **fields: Any) -> None:
    if not enabled:
        return
    payload = {
        "event": event,
        "timestamp": to_iso(datetime.now(timezone.utc)),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


__all__ = [
    "LivePaperMaintenanceConfig",
    "LivePaperMaintenanceState",
    "run_live_paper_maintenance_loop",
    "run_live_paper_maintenance_once",
]
