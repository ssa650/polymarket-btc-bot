from __future__ import annotations

import glob
import json
import shlex
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paper_trader_analytics import connect_paper_db_read_only
from .strategy_config import validate_strategy_config


DONE_STATUSES = {"closed", "settled"}
REGIME_FIELDS = (
    "strategy_id",
    "time_regime",
    "liquidity_regime",
    "btc_trend_regime",
    "spread_regime",
)
MONITOR_STALE_AFTER_SEC = 120.0
TOP_N = 10


class PaperStrategyExperimentError(RuntimeError):
    pass


def run_paper_strategy_experiment(
    *,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    config_glob: str,
    experiment_id: str | None = None,
    output_dir: str | None = None,
    poll_sec: float = 1.0,
    max_feature_age_sec: float = 8.0,
    heartbeat_detail: str = "compact",
    prediction_db_path: str | None = None,
    dry_run: bool = False,
    use_tmux: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    exp_id = str(experiment_id or f"paper_exp_{now_dt.strftime('%Y%m%dT%H%M%SZ')}")
    root = Path(output_dir) if output_dir else Path("data/experiments") / exp_id
    root.mkdir(parents=True, exist_ok=True)
    configs = _resolve_config_glob(config_glob)
    if not configs:
        raise PaperStrategyExperimentError(f"no configs matched: {config_glob}")

    workers: list[dict[str, Any]] = []
    errors: list[str] = []
    worker_id_counts: Counter[str] = Counter()
    for config_path in configs:
        base_worker_id = _safe_name(Path(config_path).stem)
        worker_id_counts[base_worker_id] += 1
        worker_id = (
            base_worker_id
            if worker_id_counts[base_worker_id] == 1
            else f"{base_worker_id}_{worker_id_counts[base_worker_id]}"
        )
        validation = validate_strategy_config(config_path)
        live_error = _live_trading_error(config_path)
        if validation.get("status") != "ok":
            errors.append(f"{config_path}: validation failed: {validation.get('errors')}")
        if live_error:
            errors.append(f"{config_path}: {live_error}")
        worker = _worker_spec(
            experiment_id=exp_id,
            root=root,
            worker_id=worker_id,
            config_path=config_path,
            recorder_db_path=recorder_db_path,
            model_path=model_path,
            feature_columns_path=feature_columns_path,
            poll_sec=poll_sec,
            max_feature_age_sec=max_feature_age_sec,
            heartbeat_detail=heartbeat_detail,
            prediction_db_path=prediction_db_path,
        )
        worker["validation"] = validation
        worker["live_trading_enabled"] = bool(live_error)
        if worker.get("runner") == "transformer_predictions" and not worker.get("transformer_prediction_db"):
            errors.append(f"{config_path}: transformer_prediction_db is required for transformer_predictions configs")
        workers.append(worker)
    if errors:
        manifest = _manifest(
            experiment_id=exp_id,
            started_at=now_dt,
            recorder_db_path=recorder_db_path,
            model_path=model_path,
            feature_columns_path=feature_columns_path,
            config_glob=config_glob,
            output_dir=root,
            dry_run=dry_run,
            use_tmux=use_tmux,
            heartbeat_detail=heartbeat_detail,
            prediction_db_path=prediction_db_path,
            workers=workers,
            status="error",
            errors=errors,
        )
        _write_manifest(root, manifest)
        return manifest

    launched: list[dict[str, Any]] = []
    for worker in workers:
        if dry_run:
            worker["launch_status"] = "dry_run"
            continue
        if use_tmux:
            _launch_tmux(worker)
            worker["launch_status"] = "tmux_launched"
        else:
            _launch_subprocess(worker)
            worker["launch_status"] = "subprocess_launched"
        launched.append(worker)

    manifest = _manifest(
        experiment_id=exp_id,
        started_at=now_dt,
        recorder_db_path=recorder_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        config_glob=config_glob,
        output_dir=root,
        dry_run=dry_run,
        use_tmux=use_tmux,
        heartbeat_detail=heartbeat_detail,
        prediction_db_path=prediction_db_path,
        workers=workers,
        status="dry_run" if dry_run else "launched",
        errors=[],
    )
    _write_manifest(root, manifest)
    return manifest


def build_paper_strategy_experiment_report(
    *,
    experiment_dir: str,
    recorder_db_path: str | None = None,
    output_path: str | None = None,
    output_txt_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(experiment_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise PaperStrategyExperimentError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers = list(manifest.get("workers") or [])
    db_reports = [
        _paper_db_report(worker.get("paper_db_path"), worker=worker)
        for worker in workers
    ]
    aggregate_rows = [
        row
        for report in db_reports
        for row in list(report.get("rows_for_aggregation") or [])
    ]
    candidate_rows = [
        row
        for report in db_reports
        for row in list(report.get("candidate_rows") or [])
    ]
    report = {
        "status": "ok",
        "generated_at": _iso(now or datetime.now(timezone.utc)),
        "experiment_dir": str(root),
        "experiment_id": manifest.get("experiment_id"),
        "manifest_path": str(manifest_path),
        "recorder_db_path": recorder_db_path or manifest.get("recorder_db_path"),
        "model_path": manifest.get("model_path"),
        "feature_columns_path": manifest.get("feature_columns_path"),
        "config_count": len(workers),
        "paper_dbs": [_strip_internal(report) for report in db_reports],
        "summary": _aggregate_summary(aggregate_rows, candidate_rows),
        "trade_autopsy": {
            "available": True,
            "invoked": False,
            "note": "Use trade-autopsy separately for full market share-price replay.",
        },
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["output_path"] = str(path)
    if output_txt_path:
        txt = Path(output_txt_path)
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(render_paper_strategy_experiment_report(report), encoding="utf-8")
        report["output_txt_path"] = str(txt)
    return report


def monitor_paper_strategy_experiment(
    *,
    experiment_dir: str,
    stale_after_sec: float = MONITOR_STALE_AFTER_SEC,
    now: datetime | None = None,
    active_tmux_sessions: set[str] | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    root = Path(experiment_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise PaperStrategyExperimentError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers = list(manifest.get("workers") or [])
    sessions = active_tmux_sessions if active_tmux_sessions is not None else _active_tmux_sessions()
    worker_reports = [
        _monitor_worker(
            worker,
            active_tmux_sessions=sessions,
            stale_after_sec=stale_after_sec,
            now=now_dt,
        )
        for worker in workers
    ]
    return {
        "status": "ok",
        "generated_at": _iso(now_dt),
        "experiment_dir": str(root),
        "experiment_id": manifest.get("experiment_id"),
        "manifest_path": str(manifest_path),
        "worker_count": len(workers),
        "active_tmux_sessions": sorted(sessions),
        "summary": _monitor_summary(worker_reports),
        "workers": worker_reports,
    }


def render_paper_strategy_experiment_monitor(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary") or {})
    lines = [
        "Paper Strategy Experiment Monitor",
        f"experiment_id: {report.get('experiment_id')}",
        f"generated_at: {report.get('generated_at')}",
        (
            f"workers={report.get('worker_count')} "
            f"active_tmux={summary.get('active_tmux_workers')} "
            f"stale_workers={summary.get('stale_workers')}"
        ),
        (
            f"trades total={summary.get('total_trades')} open={summary.get('open_trades')} "
            f"closed={summary.get('closed_trades')} settled={summary.get('settled_trades')} "
            f"skipped={summary.get('skipped_trades')}"
        ),
        (
            f"candidates total={summary.get('total_candidates')} "
            f"TRADE={summary.get('candidate_trade_decisions')} "
            f"SKIP={summary.get('candidate_skip_decisions')}"
        ),
        "",
        "Workers",
    ]
    for worker in list(report.get("workers") or []):
        warnings = ",".join(worker.get("warnings") or []) or "none"
        lines.append(
            "  "
            + " ".join(
                [
                    str(worker.get("worker_id") or "worker"),
                    f"tmux={worker.get('tmux_status')}",
                    f"log={worker.get('latest_log_timestamp') or 'none'}",
                    f"trade_ts={worker.get('latest_trade_timestamp') or 'none'}",
                    f"candidate_ts={worker.get('latest_candidate_timestamp') or 'none'}",
                    f"delta_candidates={worker.get('latest_poll_candidate_delta')}",
                    f"delta_trades={worker.get('latest_poll_trade_delta')}",
                    f"warnings={warnings}",
                ]
            )
        )
    lines.extend(["", "Top Candidate Rejection Reasons"])
    lines.extend(_format_count_lines(summary.get("top_candidate_rejection_reasons") or {}))
    lines.extend(["", "Top Strategy IDs By TRADE Candidate Count"])
    lines.extend(_format_count_lines(summary.get("top_strategy_trade_candidate_counts") or {}))
    lines.extend(["", "Top Strategy IDs By Closed PnL"])
    for row in list(summary.get("top_strategy_closed_pnl") or [])[:TOP_N]:
        lines.append(
            f"  {row.get('strategy_id')}: closed={row.get('closed_trades')} pnl={row.get('pnl')}"
        )
    return "\n".join(lines) + "\n"


def render_paper_strategy_experiment_report(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary") or {})
    done = int(summary.get("closed_trades") or 0) + int(summary.get("settled_trades") or 0)
    lines = [
        "Paper Strategy Experiment Report",
        f"experiment_id: {report.get('experiment_id')}",
        f"generated_at: {report.get('generated_at')}",
        f"paper_dbs: {len(report.get('paper_dbs') or [])}",
        "",
        "Compact Summary",
        (
            f"trades total={summary.get('total_trades')} closed={summary.get('closed_trades')} "
            f"settled={summary.get('settled_trades')} open={summary.get('open_trades')} "
            f"awaiting={summary.get('awaiting_resolution_trades')} skipped={summary.get('skipped_trades')}"
        ),
        (
            f"done={done} pnl={summary.get('pnl')} roi={summary.get('roi')} "
            f"win_rate={summary.get('win_rate')} avg_win={summary.get('average_win')} "
            f"avg_loss={summary.get('average_loss')}"
        ),
        (
            f"candidates total={summary.get('candidate_total')} "
            f"TRADE={dict(summary.get('candidate_decision_counts') or {}).get('TRADE', 0)} "
            f"SKIP={dict(summary.get('candidate_decision_counts') or {}).get('SKIP', 0)}"
        ),
    ]
    for warning in list(summary.get("warnings") or []):
        lines.append(f"note: {warning}")
    lines.extend(
        [
            "",
            "Skip Reasons",
        ]
    )
    lines.extend(_format_count_lines(summary.get("skip_reason_counts") or {}))
    lines.extend(
        [
            "",
            "Candidate Rejections By Reason",
        ]
    )
    lines.extend(_format_candidate_group_lines(summary, "reason"))
    for title, key in (
        ("Candidate Rejections By Strategy", "strategy_id"),
        ("Candidate Rejections By Time Window", "time_window"),
        ("Candidate Rejections By Threshold", "threshold"),
        ("Candidate Rejections By Max Probability", "max_probability"),
        ("Candidate Rejections By Bankroll", "bankroll"),
    ):
        lines.extend(["", title])
        lines.extend(_format_candidate_group_lines(summary, key))
    lines.extend(["", "Top Strategy Performance"])
    for row in list((summary.get("grouped_performance") or {}).get("strategy_id") or [])[:TOP_N]:
        lines.append(
            f"  {row.get('strategy_id')}: done={row.get('done_trades')} pnl={row.get('pnl')} "
            f"roi={row.get('roi')} win_rate={row.get('win_rate')}"
        )
    return "\n".join(lines) + "\n"


def _format_count_lines(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  none"]
    return [f"  {key}: {value}" for key, value in _top_count_items(counts)]


def _format_candidate_group_lines(summary: dict[str, Any], key: str) -> list[str]:
    rows = list((summary.get("candidate_rejection_groups") or {}).get(key) or [])
    if not rows:
        return ["  none"]
    lines: list[str] = []
    for row in rows[:TOP_N]:
        label = row.get("rejection_reason") if key == "reason" else row.get(key)
        reason_counts = ", ".join(
            f"{reason}={count}"
            for reason, count in _top_count_items(row.get("rejection_reason_counts") or {})
        )
        lines.append(f"  {label}: count={row.get('count')} reasons={reason_counts or 'none'}")
    return lines


def _top_count_items(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(
        ((str(key), int(value)) for key, value in counts.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[:TOP_N]


def _resolve_config_glob(config_glob: str) -> list[str]:
    paths = sorted(glob.glob(str(config_glob)))
    if not paths and Path(config_glob).exists():
        paths = [str(config_glob)]
    return [path for path in paths if _looks_like_strategy_config(path)]


def _looks_like_strategy_config(path: str) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    strategies = payload.get("strategies") if isinstance(payload, dict) else None
    return isinstance(strategies, list) and bool(strategies)


def _read_config_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_transformer_prediction_config(payload: dict[str, Any]) -> bool:
    source = str(payload.get("prediction_source") or "").strip().lower()
    return source == "transformer_predictions" or bool(
        _transformer_prediction_db_from_config(payload)
    )


def _transformer_prediction_db_from_config(payload: dict[str, Any]) -> str | None:
    for key in (
        "transformer_prediction_db",
        "transformer_predictions_db",
        "prediction_db",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    nested = payload.get("transformer_predictions")
    if isinstance(nested, dict):
        value = nested.get("db") or nested.get("db_path") or nested.get("path")
        if value not in (None, ""):
            return str(value)
    return None


def _worker_spec(
    *,
    experiment_id: str,
    root: Path,
    worker_id: str,
    config_path: str,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    poll_sec: float,
    max_feature_age_sec: float,
    heartbeat_detail: str,
    prediction_db_path: str | None = None,
) -> dict[str, Any]:
    config = Path(config_path)
    paper_db = root / "paper_dbs" / f"{worker_id}.db"
    log_path = root / "logs" / f"{worker_id}.log"
    run_id = f"{experiment_id}_{worker_id}"
    config_payload = _read_config_payload(config_path)
    transformer_prediction_db = (
        _transformer_prediction_db_from_config(config_payload)
        or prediction_db_path
    )
    if _is_transformer_prediction_config(config_payload):
        if not transformer_prediction_db:
            command = [
                sys.executable,
                "-m",
                "src.main",
                "run-transformer-prediction-paper-strategy",
                "--db",
                recorder_db_path,
                "--prediction-db",
                "__missing_transformer_prediction_db__",
                "--config",
                str(config),
                "--output-db",
                str(paper_db),
                "--run-id",
                run_id,
                "--poll-sec",
                str(float(poll_sec)),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "src.main",
                "run-transformer-prediction-paper-strategy",
                "--db",
                recorder_db_path,
                "--prediction-db",
                str(transformer_prediction_db),
                "--config",
                str(config),
                "--output-db",
                str(paper_db),
                "--run-id",
                run_id,
                "--poll-sec",
                str(float(poll_sec)),
            ]
    else:
        command = [
            sys.executable,
            "-m",
            "src.main",
            "run-multi-strategy-paper-trader",
            "--db",
            recorder_db_path,
            "--model-path",
            model_path,
            "--feature-columns",
            feature_columns_path,
            "--config",
            str(config),
            "--output-db",
            str(paper_db),
            "--run-id",
            run_id,
            "--poll-sec",
            str(float(poll_sec)),
            "--max-feature-age-sec",
            str(float(max_feature_age_sec)),
            "--heartbeat-detail",
            str(heartbeat_detail),
        ]
    return {
        "worker_id": worker_id,
        "config_path": str(config),
        "paper_db_path": str(paper_db),
        "log_path": str(log_path),
        "run_id": run_id,
        "runner": "transformer_predictions" if _is_transformer_prediction_config(config_payload) else "multi_strategy",
        "transformer_prediction_db": transformer_prediction_db,
        "command": command,
        "command_line": shlex.join(command),
    }


def _manifest(
    *,
    experiment_id: str,
    started_at: datetime,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    config_glob: str,
    output_dir: Path,
    dry_run: bool,
    use_tmux: bool,
    heartbeat_detail: str,
    prediction_db_path: str | None,
    workers: Sequence[dict[str, Any]],
    status: str,
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "experiment_id": experiment_id,
        "started_at": _iso(started_at),
        "recorder_db_path": recorder_db_path,
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "config_glob": config_glob,
        "output_dir": str(output_dir),
        "dry_run": bool(dry_run),
        "tmux": bool(use_tmux),
        "heartbeat_detail": str(heartbeat_detail),
        "prediction_db_path": prediction_db_path,
        "command_lines": [worker["command_line"] for worker in workers],
        "config_paths": [worker["config_path"] for worker in workers],
        "paper_db_paths": [worker["paper_db_path"] for worker in workers],
        "log_paths": [worker["log_path"] for worker in workers],
        "workers": list(workers),
        "errors": list(errors),
    }


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("paper_dbs", "logs"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _launch_subprocess(worker: dict[str, Any]) -> None:
    log_path = Path(str(worker["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(worker["paper_db_path"])).parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    subprocess.Popen(  # noqa: S603
        list(worker["command"]),
        cwd=str(Path.cwd()),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _launch_tmux(worker: dict[str, Any]) -> None:
    log_path = Path(str(worker["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(worker["paper_db_path"])).parent.mkdir(parents=True, exist_ok=True)
    session = f"paper_{_safe_name(str(worker['worker_id']))}"[:80]
    shell_command = f"{worker['command_line']} >> {shlex.quote(str(log_path))} 2>&1"
    subprocess.run(  # noqa: S603
        ["tmux", "new-session", "-d", "-s", session, shell_command],
        check=True,
    )
    worker["tmux_session"] = session


def _live_trading_error(config_path: str) -> str | None:
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return f"config_read_error: {exc}"
    if _contains_live_trading_enabled(payload):
        return "live_trading_enabled is true; refusing to launch paper experiment"
    return None


def _contains_live_trading_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("live_trading_enabled") is True:
            return True
        return any(_contains_live_trading_enabled(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_live_trading_enabled(item) for item in value)
    return False


def _paper_db_report(paper_db_path: Any, *, worker: dict[str, Any]) -> dict[str, Any]:
    path = str(paper_db_path or "")
    if not path:
        return {"status": "error", "error": "missing_paper_db_path", "worker": worker}
    try:
        conn = connect_paper_db_read_only(path)
    except FileNotFoundError:
        return {
            "status": "missing",
            "paper_db_path": path,
            "config_path": worker.get("config_path"),
            "rows_for_aggregation": [],
            "candidate_rows": [],
        }
    try:
        trade_rows = _read_table(conn, "paper_trades")
        candidate_rows = _read_table(conn, "paper_trade_candidates")
    finally:
        conn.close()
    metrics = _metrics_for_rows(trade_rows, candidate_rows)
    return {
        "status": "ok",
        "paper_db_path": path,
        "config_path": worker.get("config_path"),
        "run_id": worker.get("run_id"),
        **metrics,
        "rows_for_aggregation": trade_rows,
        "candidate_rows": candidate_rows,
    }


def _monitor_worker(
    worker: dict[str, Any],
    *,
    active_tmux_sessions: set[str],
    stale_after_sec: float,
    now: datetime,
) -> dict[str, Any]:
    worker_id = str(worker.get("worker_id") or Path(str(worker.get("config_path") or "worker")).stem)
    tmux_session = str(worker.get("tmux_session") or f"paper_{_safe_name(worker_id)}"[:80])
    log_info = _latest_log_info(worker.get("log_path"))
    db_info = _paper_db_monitor(worker.get("paper_db_path"), now=now)
    latest_activity = _latest_timestamp_value(
        [
            log_info.get("latest_log_timestamp"),
            db_info.get("latest_trade_timestamp"),
            db_info.get("latest_candidate_timestamp"),
        ]
    )
    latest_activity_age = _age_seconds(latest_activity, now)
    warnings: list[str] = []
    if worker.get("tmux_session") and tmux_session not in active_tmux_sessions:
        warnings.append("tmux_session_inactive")
    if latest_activity_age is None:
        warnings.append("no_activity_seen")
    elif latest_activity_age > stale_after_sec:
        warnings.append("worker_stale")
    return {
        "worker_id": worker_id,
        "run_id": worker.get("run_id"),
        "config_path": worker.get("config_path"),
        "paper_db_path": worker.get("paper_db_path"),
        "log_path": worker.get("log_path"),
        "tmux_session": tmux_session,
        "tmux_status": "active" if tmux_session in active_tmux_sessions else "inactive",
        "latest_log_timestamp": log_info.get("latest_log_timestamp"),
        "latest_log_event": log_info.get("latest_log_event"),
        "latest_log_age_sec": _round(_age_seconds(log_info.get("latest_log_timestamp"), now)),
        "latest_trade_timestamp": db_info.get("latest_trade_timestamp"),
        "latest_candidate_timestamp": db_info.get("latest_candidate_timestamp"),
        "latest_db_activity_age_sec": _round(
            _age_seconds(
                _latest_timestamp_value(
                    [db_info.get("latest_trade_timestamp"), db_info.get("latest_candidate_timestamp")]
                ),
                now,
            )
        ),
        "latest_poll_candidate_delta": db_info.get("latest_poll_candidate_delta"),
        "latest_poll_trade_delta": db_info.get("latest_poll_trade_delta"),
        "warnings": warnings,
        **db_info,
    }


def _paper_db_monitor(path_value: Any, *, now: datetime) -> dict[str, Any]:
    path = str(path_value or "")
    if not path:
        return {"db_status": "missing_path"}
    db_path = Path(path)
    if not db_path.exists():
        return {"db_status": "missing", "total_trades": 0, "total_candidates": 0}
    try:
        conn = connect_paper_db_read_only(path)
    except FileNotFoundError:
        return {"db_status": "missing", "total_trades": 0, "total_candidates": 0}
    try:
        trade_rows = _read_table(conn, "paper_trades")
        candidate_rows = _read_table(conn, "paper_trade_candidates")
        heartbeat = _latest_multi_strategy_heartbeat(conn)
    finally:
        conn.close()
    metrics = _metrics_for_rows(trade_rows, candidate_rows)
    latest_trade = _latest_row_timestamp(
        trade_rows,
        ("created_at", "signal_timestamp", "exit_time", "settled_at"),
    )
    latest_candidate = _latest_row_timestamp(
        candidate_rows,
        ("created_at", "feature_timestamp", "settled_at"),
    )
    strategy_trade_candidates = _counts(
        row.get("strategy_id")
        for row in candidate_rows
        if str(row.get("decision") or "").upper() == "TRADE"
    )
    strategy_closed_pnl = _strategy_closed_pnl(trade_rows)
    latest_poll_candidate_delta = None
    latest_poll_trade_delta = None
    if heartbeat:
        latest_poll_candidate_delta = _int_or_zero(
            heartbeat.get("candidate_rows_written_this_poll", heartbeat.get("candidates_logged"))
        )
        latest_poll_trade_delta = _int_or_zero(heartbeat.get("trades_opened"))
    return {
        "db_status": "ok",
        "latest_trade_timestamp": latest_trade,
        "latest_candidate_timestamp": latest_candidate,
        "latest_trade_age_sec": _round(_age_seconds(latest_trade, now)),
        "latest_candidate_age_sec": _round(_age_seconds(latest_candidate, now)),
        "latest_poll_candidate_delta": latest_poll_candidate_delta,
        "latest_poll_trade_delta": latest_poll_trade_delta,
        "latest_heartbeat": heartbeat,
        "top_candidate_rejection_reasons": _top_count_dict(metrics.get("candidate_rejection_counts") or {}),
        "top_strategy_trade_candidate_counts": _top_count_dict(strategy_trade_candidates),
        "top_strategy_closed_pnl": strategy_closed_pnl[:TOP_N],
        **metrics,
    }


def _read_table(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()]


def _metrics_for_rows(
    rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    done = [row for row in rows if _status(row) in DONE_STATUSES]
    wins = [row for row in done if (_result_pnl(row) or 0.0) > 0]
    losses = [row for row in done if (_result_pnl(row) or 0.0) < 0]
    skipped = [row for row in rows if _status(row) == "skipped"]
    total_stake = _sum(row.get("stake_usd") for row in done)
    pnl = _sum(_result_pnl(row) for row in done)
    return {
        "total_trades": len(rows),
        "closed_trades": sum(1 for row in rows if _status(row) == "closed"),
        "settled_trades": sum(1 for row in rows if _status(row) == "settled"),
        "open_trades": sum(1 for row in rows if _status(row) == "open"),
        "awaiting_resolution_trades": sum(1 for row in rows if _status(row) == "awaiting_resolution"),
        "skipped_trades": len(skipped),
        "pnl": _round(pnl),
        "roi": _round(pnl / total_stake) if total_stake else None,
        "win_rate": _round(len(wins) / len(done)) if done else None,
        "average_win": _mean(_result_pnl(row) for row in wins),
        "average_loss": _mean(_result_pnl(row) for row in losses),
        "skip_reason_counts": _counts(row.get("skip_reason") for row in skipped),
        "candidate_decision_counts": _counts(row.get("decision") for row in candidate_rows),
        "candidate_rejection_counts": _counts(row.get("rejection_reason") for row in candidate_rows),
        "candidate_total": len(candidate_rows),
    }


def _aggregate_summary(
    rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _metrics_for_rows(rows, candidate_rows)
    metrics["grouped_performance"] = {
        field: _grouped_performance(rows, field)
        for field in REGIME_FIELDS
    }
    metrics["candidate_rejection_groups"] = {
        "reason": _candidate_rejection_group(candidate_rows, "rejection_reason"),
        "strategy_id": _candidate_rejection_group(candidate_rows, "strategy_id"),
        "time_window": _candidate_rejection_group(candidate_rows, "time_window"),
        "threshold": _candidate_rejection_group(candidate_rows, "threshold"),
        "max_probability": _candidate_rejection_group(candidate_rows, "max_probability"),
        "bankroll": _candidate_rejection_group(candidate_rows, "bankroll"),
    }
    done = int(metrics.get("closed_trades") or 0) + int(metrics.get("settled_trades") or 0)
    warnings: list[str] = []
    if done < 30:
        warnings.append("sample size is too small for deployment-quality conclusions")
    elif done < 100:
        warnings.append("early sample; use caution before ranking strategies")
    metrics["warnings"] = warnings
    return metrics


def _grouped_performance(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _status(row) in DONE_STATUSES:
            grouped[str(row.get(field) or "unknown")].append(row)
    result: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        wins = [row for row in group_rows if (_result_pnl(row) or 0.0) > 0]
        pnl = _sum(_result_pnl(row) for row in group_rows)
        stake = _sum(row.get("stake_usd") for row in group_rows)
        result.append(
            {
                field: value,
                "done_trades": len(group_rows),
                "pnl": _round(pnl),
                "roi": _round(pnl / stake) if stake else None,
                "win_rate": _round(len(wins) / len(group_rows)) if group_rows else None,
                "average_roi": _mean(_result_roi(row) for row in group_rows),
            }
        )
    result.sort(
        key=lambda item: (
            int(item.get("done_trades") or 0),
            float(item.get("pnl") or 0.0),
            str(item),
        ),
        reverse=True,
    )
    return result


def _candidate_rejection_group(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    rejected = [
        row
        for row in rows
        if str(row.get("decision") or "").upper() != "TRADE"
        and row.get("rejection_reason") not in (None, "")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rejected:
        grouped[_candidate_group_value(row, field)].append(row)
    result: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        result.append(
            {
                field: value,
                "count": len(group_rows),
                "rejection_reason_counts": _counts(row.get("rejection_reason") for row in group_rows),
            }
        )
    result.sort(key=lambda item: (int(item.get("count") or 0), str(item.get(field))), reverse=True)
    return result


def _candidate_group_value(row: dict[str, Any], field: str) -> str:
    if field == "rejection_reason":
        return str(row.get("rejection_reason") or "unknown")
    if field == "time_window":
        return _time_window(row)
    if field == "threshold":
        return _format_group_number(row.get("threshold_used"))
    if field == "max_probability":
        for key in ("max_probability_for_direction", "block_probability_above"):
            if row.get(key) not in (None, ""):
                return _format_group_number(row.get(key))
        return "unknown"
    if field == "bankroll":
        for key in ("starting_bankroll_usd", "bankroll_before_trade", "bankroll_status"):
            if row.get(key) not in (None, ""):
                return _format_group_number(row.get(key))
        return "unknown"
    return str(row.get(field) or "unknown")


def _time_window(row: dict[str, Any]) -> str:
    min_value = _float_or_none(row.get("min_time_until_resolution_sec"))
    max_value = _float_or_none(row.get("max_time_until_resolution_sec"))
    if min_value is not None or max_value is not None:
        return f"{_format_group_number(min_value)}-{_format_group_number(max_value)}s"
    regime = row.get("time_regime")
    if regime not in (None, ""):
        return str(regime)
    time_until = _float_or_none(row.get("time_until_resolution"))
    if time_until is None:
        return "unknown"
    if time_until < 30:
        return "0-30s"
    if time_until < 60:
        return "30-60s"
    if time_until < 120:
        return "60-120s"
    return "120s+"


def _strip_internal(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"rows_for_aggregation", "candidate_rows"}
    }


def _active_tmux_sessions() -> set[str]:
    try:
        result = subprocess.run(  # noqa: S603
            ["tmux", "list-sessions", "-F", "#S"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _latest_log_info(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    if not path.exists():
        return {"log_status": "missing", "latest_log_timestamp": None, "latest_log_event": None}
    latest_timestamp: str | None = None
    latest_event: str | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    except OSError as exc:
        return {"log_status": "error", "latest_log_timestamp": None, "latest_log_event": str(exc)}
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = payload.get("timestamp")
        if _parse_timestamp(timestamp) is None:
            continue
        if _latest_timestamp_value([latest_timestamp, timestamp]) == timestamp:
            latest_timestamp = str(timestamp)
            latest_event = str(payload.get("event") or "")
    if latest_timestamp is None:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return {"log_status": "empty", "latest_log_timestamp": None, "latest_log_event": None}
        latest_timestamp = _iso(mtime)
        latest_event = "file_mtime"
    return {
        "log_status": "ok",
        "latest_log_timestamp": latest_timestamp,
        "latest_log_event": latest_event,
    }


def _latest_multi_strategy_heartbeat(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not _table_exists(conn, "multi_strategy_paper_trader_heartbeats"):
        return None
    row = conn.execute(
        """
        SELECT *
        FROM multi_strategy_paper_trader_heartbeats
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row is not None else None


def _monitor_summary(workers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rejection_counts: Counter[str] = Counter()
    strategy_trade_candidate_counts: Counter[str] = Counter()
    strategy_pnl: dict[str, dict[str, Any]] = defaultdict(lambda: {"strategy_id": "", "closed_trades": 0, "pnl": 0.0})
    for worker in workers:
        rejection_counts.update(worker.get("top_candidate_rejection_reasons") or {})
        strategy_trade_candidate_counts.update(worker.get("top_strategy_trade_candidate_counts") or {})
        for row in list(worker.get("top_strategy_closed_pnl") or []):
            strategy_id = str(row.get("strategy_id") or "unknown")
            item = strategy_pnl[strategy_id]
            item["strategy_id"] = strategy_id
            item["closed_trades"] = int(item.get("closed_trades") or 0) + int(row.get("closed_trades") or 0)
            item["pnl"] = float(item.get("pnl") or 0.0) + float(row.get("pnl") or 0.0)
    top_strategy_closed_pnl = [
        {"strategy_id": value["strategy_id"], "closed_trades": value["closed_trades"], "pnl": _round(value["pnl"])}
        for value in strategy_pnl.values()
    ]
    top_strategy_closed_pnl.sort(
        key=lambda item: (float(item.get("pnl") or 0.0), int(item.get("closed_trades") or 0)),
        reverse=True,
    )
    return {
        "active_tmux_workers": sum(1 for worker in workers if worker.get("tmux_status") == "active"),
        "stale_workers": sum(1 for worker in workers if "worker_stale" in set(worker.get("warnings") or [])),
        "total_trades": sum(int(worker.get("total_trades") or 0) for worker in workers),
        "open_trades": sum(int(worker.get("open_trades") or 0) for worker in workers),
        "closed_trades": sum(int(worker.get("closed_trades") or 0) for worker in workers),
        "settled_trades": sum(int(worker.get("settled_trades") or 0) for worker in workers),
        "skipped_trades": sum(int(worker.get("skipped_trades") or 0) for worker in workers),
        "total_candidates": sum(int(worker.get("candidate_total") or 0) for worker in workers),
        "candidate_trade_decisions": sum(
            int(dict(worker.get("candidate_decision_counts") or {}).get("TRADE", 0))
            for worker in workers
        ),
        "candidate_skip_decisions": sum(
            int(dict(worker.get("candidate_decision_counts") or {}).get("SKIP", 0))
            for worker in workers
        ),
        "top_candidate_rejection_reasons": _top_count_dict(rejection_counts),
        "top_strategy_trade_candidate_counts": _top_count_dict(strategy_trade_candidate_counts),
        "top_strategy_closed_pnl": top_strategy_closed_pnl[:TOP_N],
    }


def _strategy_closed_pnl(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _status(row) == "closed":
            grouped[str(row.get("strategy_id") or "unknown")].append(row)
    result: list[dict[str, Any]] = []
    for strategy_id, group_rows in grouped.items():
        result.append(
            {
                "strategy_id": strategy_id,
                "closed_trades": len(group_rows),
                "pnl": _round(_sum(_result_pnl(row) for row in group_rows)),
            }
        )
    result.sort(key=lambda item: (float(item.get("pnl") or 0.0), int(item.get("closed_trades") or 0)), reverse=True)
    return result


def _latest_row_timestamp(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str | None:
    return _latest_timestamp_value(row.get(field) for row in rows for field in fields)


def _latest_timestamp_value(values: Iterable[Any]) -> str | None:
    latest: datetime | None = None
    latest_raw: str | None = None
    for value in values:
        parsed = _parse_timestamp(value)
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
            latest_raw = str(value)
    return latest_raw


def _age_seconds(value: Any, now: datetime) -> float | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _top_count_dict(counts: dict[str, int] | Counter[str]) -> dict[str, int]:
    return dict(_top_count_items(dict(counts)))


def _int_or_zero(value: Any) -> int:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else 0


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "").lower()


def _result_pnl(row: dict[str, Any]) -> float | None:
    status = _status(row)
    if status == "closed":
        return _first_float(row, ("realized_pnl_usd", "pnl_usd"))
    if status == "settled":
        return _first_float(row, ("realized_pnl_usd", "pnl_usd"))
    return None


def _result_roi(row: dict[str, Any]) -> float | None:
    status = _status(row)
    if status == "closed":
        return _first_float(row, ("realized_roi", "roi"))
    if status == "settled":
        return _first_float(row, ("realized_roi", "roi"))
    return None


def _first_float(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return dict(sorted(counter.items()))


def _sum(values: Iterable[Any]) -> float:
    return sum(value for value in (_float_or_none(item) for item in values) if value is not None)


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return _round(sum(numeric) / len(numeric)) if numeric else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _round(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, 10) if parsed is not None else None


def _format_group_number(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "unknown"
    return str(int(parsed)) if float(parsed).is_integer() else str(_round(parsed))


def _safe_name(value: str) -> str:
    result = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return result.strip("._-") or "strategy"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "PaperStrategyExperimentError",
    "build_paper_strategy_experiment_report",
    "monitor_paper_strategy_experiment",
    "render_paper_strategy_experiment_monitor",
    "render_paper_strategy_experiment_report",
    "run_paper_strategy_experiment",
]
