from __future__ import annotations

import csv
import glob
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from .train_baseline_model import _float_or_none


STALE_AFTER_SEC = 10 * 60
DONE_STATUSES = {"closed", "settled"}


class ModelExperimentComparisonError(RuntimeError):
    pass


def compare_model_experiments(
    *,
    recorder_db_path: str,
    paper_experiment_dir: str,
    transformer_prediction_db_paths: Sequence[str] | str | None,
    transformer_autopsy_path: str | None,
    output_path: str | None = None,
    output_txt_path: str | None = None,
    output_csv_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    recorder = _connect_read_only(recorder_db_path)
    try:
        recorder_health = _recorder_health(recorder, recorder_db_path)
    finally:
        recorder.close()

    experiment = _load_paper_experiment(paper_experiment_dir, now=now_dt)
    prediction_paths = _resolve_paths(transformer_prediction_db_paths)
    prediction_health = [
        _transformer_prediction_db_health(path, now=now_dt)
        for path in prediction_paths
    ]
    autopsy_rows = _load_autopsy_rows(transformer_autopsy_path) if transformer_autopsy_path else []
    autopsy_health = _autopsy_health(
        transformer_autopsy_path,
        rows=autopsy_rows,
        now=now_dt,
    )

    paper_summary = _paper_summary(
        experiment["trade_rows"],
        experiment["candidate_rows"],
    )
    transformer_summary = _transformer_summary(autopsy_rows)
    comparison = _comparison_summary(
        paper_rows=experiment["trade_rows"],
        transformer_rows=autopsy_rows,
    )
    report = {
        "status": "ok",
        "generated_at": _iso(now_dt),
        "recorder_db_path": recorder_db_path,
        "paper_experiment_dir": str(Path(paper_experiment_dir)),
        "transformer_prediction_db_paths": prediction_paths,
        "transformer_autopsy_path": transformer_autopsy_path,
        "input_health": {
            "recorder_db": recorder_health,
            "paper_experiment": experiment["health"],
            "transformer_prediction_dbs": prediction_health,
            "transformer_autopsy": autopsy_health,
        },
        "gb_paper_strategy": paper_summary,
        "transformer_autopsy": transformer_summary,
        "comparison": comparison,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["output_path"] = str(path)
    if output_txt_path:
        txt = Path(output_txt_path)
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(render_model_experiment_comparison(report), encoding="utf-8")
        report["output_txt_path"] = str(txt)
    if output_csv_path:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_comparison_csv(report, csv_path)
        report["output_csv_path"] = str(csv_path)
    return report


def render_model_experiment_comparison(report: dict[str, Any]) -> str:
    paper = dict(report.get("gb_paper_strategy", {}).get("overall") or {})
    transformer = dict(report.get("transformer_autopsy", {}).get("overall") or {})
    comparison = dict(report.get("comparison") or {})
    lines = [
        "Model Experiment Comparison",
        f"generated_at: {report.get('generated_at')}",
        f"paper_experiment_dir: {report.get('paper_experiment_dir')}",
        f"transformer_autopsy_path: {report.get('transformer_autopsy_path')}",
        "",
        "Input Health",
        json.dumps(report.get("input_health") or {}, sort_keys=True, default=str),
        "",
        "GB Paper Strategy",
        f"trades: {paper.get('count')}",
        f"done_trades: {paper.get('done_count')}",
        f"total_pnl: {paper.get('total_pnl')}",
        f"average_roi: {paper.get('average_roi')}",
        f"win_rate: {paper.get('win_rate')}",
        f"markets_covered: {paper.get('markets_covered')}",
        "",
        "Transformer Autopsy",
        f"simulations: {transformer.get('count')}",
        f"total_simulated_pnl: {transformer.get('total_pnl')}",
        f"average_simulated_roi: {transformer.get('average_roi')}",
        f"win_rate: {transformer.get('win_rate')}",
        f"markets_covered: {transformer.get('markets_covered')}",
        "",
        "Comparison",
        json.dumps(comparison, sort_keys=True, default=str),
    ]
    return "\n".join(lines) + "\n"


def _load_paper_experiment(experiment_dir: str, *, now: datetime) -> dict[str, Any]:
    root = Path(experiment_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ModelExperimentComparisonError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    worker_paths = [
        str(worker.get("paper_db_path"))
        for worker in list(manifest.get("workers") or [])
        if worker.get("paper_db_path")
    ]
    if not worker_paths:
        worker_paths = [str(path) for path in manifest.get("paper_db_paths") or []]
    trade_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    db_health: list[dict[str, Any]] = []
    for path in worker_paths:
        health, trades, candidates = _load_paper_db(path, now=now)
        db_health.append(health)
        trade_rows.extend(trades)
        candidate_rows.extend(candidates)
    return {
        "manifest": manifest,
        "trade_rows": trade_rows,
        "candidate_rows": candidate_rows,
        "health": {
            "status": _combined_status(db_health),
            "manifest_path": str(manifest_path),
            "paper_db_count": len(worker_paths),
            "trade_rows_loaded": len(trade_rows),
            "candidate_rows_loaded": len(candidate_rows),
            "paper_dbs": db_health,
        },
    }


def _load_paper_db(
    path: str,
    *,
    now: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    db_path = Path(path)
    if not db_path.exists():
        return (
            {"path": path, "status": "missing", "trade_rows": 0, "candidate_rows": 0},
            [],
            [],
        )
    conn = _connect_read_only(path)
    try:
        trades = _read_table(conn, "paper_trades")
        candidates = _read_table(conn, "paper_trade_candidates")
    finally:
        conn.close()
    for row in trades:
        row["source_paper_db"] = path
    for row in candidates:
        row["source_paper_db"] = path
    latest = _latest_timestamp(trades, ("created_at", "signal_timestamp", "exit_time", "settled_at"))
    age = _age_sec(latest, now)
    status = "ok"
    if not trades and not candidates:
        status = "empty"
    elif age is not None and age > STALE_AFTER_SEC:
        status = "stale"
    return (
        {
            "path": path,
            "status": status,
            "trade_rows": len(trades),
            "candidate_rows": len(candidates),
            "latest_activity_timestamp": latest,
            "latest_activity_age_sec": _round(age),
        },
        trades,
        candidates,
    )


def _paper_summary(
    trade_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "overall": _trade_metrics(trade_rows),
        "fixed_horizon": _trade_metrics(_fixed_horizon_trade_rows(trade_rows)),
        "by_strategy": _group_metrics(trade_rows, ("strategy_id",), metric_fn=_trade_metrics),
        "by_bankroll": _group_metrics(trade_rows, ("bankroll_status",), metric_fn=_trade_metrics),
        "by_threshold": _group_metrics(trade_rows, ("threshold_used",), metric_fn=_trade_metrics),
        "by_time_window": _group_metrics(trade_rows, ("time_window",), metric_fn=_trade_metrics),
        "by_liquidity_regime": _group_metrics(trade_rows, ("liquidity_regime",), metric_fn=_trade_metrics),
        "by_spread_regime": _group_metrics(trade_rows, ("spread_regime",), metric_fn=_trade_metrics),
        "by_rejection_reason": _group_metrics(trade_rows, ("rejection_reason",), metric_fn=_trade_metrics),
        "candidate_summary": _candidate_summary(candidate_rows),
    }


def _transformer_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _simulation_metrics(rows),
        "by_side": _group_metrics(rows, ("side",), metric_fn=_simulation_metrics),
        "by_threshold": _group_metrics(rows, ("threshold",), metric_fn=_simulation_metrics),
        "by_horizon_sec": _group_metrics(rows, ("horizon_sec",), metric_fn=_simulation_metrics),
        "by_side_threshold_horizon": _group_metrics(
            rows,
            ("side", "threshold", "horizon_sec"),
            metric_fn=_simulation_metrics,
        ),
        "by_time_regime": _group_metrics(rows, ("time_regime",), metric_fn=_simulation_metrics),
        "by_liquidity_regime": _group_metrics(rows, ("liquidity_regime",), metric_fn=_simulation_metrics),
        "by_spread_regime": _group_metrics(rows, ("spread_regime",), metric_fn=_simulation_metrics),
    }


def _comparison_summary(
    *,
    paper_rows: Sequence[dict[str, Any]],
    transformer_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gb_done = [row for row in paper_rows if _status(row) in DONE_STATUSES]
    gb_fixed = _fixed_horizon_trade_rows(paper_rows)
    transformer_fixed = [
        row for row in transformer_rows
        if str(row.get("hit_reason") or "").lower() == "fixed_horizon"
    ]
    by_horizon: list[dict[str, Any]] = []
    horizons = sorted({_int_or_none(row.get("horizon_sec")) for row in transformer_fixed if _int_or_none(row.get("horizon_sec")) is not None})
    for horizon in horizons:
        transformer_group = [
            row for row in transformer_fixed
            if _int_or_none(row.get("horizon_sec")) == horizon
        ]
        paper_group = [
            row for row in gb_fixed
            if _int_or_none(row.get("fixed_horizon_exit_sec")) == horizon
        ]
        if not paper_group:
            paper_group = gb_fixed if horizon == 15 else []
        by_horizon.append(
            {
                "horizon_sec": horizon,
                "gb_fixed_horizon": _trade_metrics(paper_group),
                "transformer_fixed_horizon": _simulation_metrics(transformer_group),
                "pnl_difference_transformer_minus_gb": _round(
                    _sum(_simulation_pnl(row) for row in transformer_group)
                    - _sum(_trade_pnl(row) for row in paper_group)
                ),
            }
        )
    return {
        "gb_done": _trade_metrics(gb_done),
        "gb_fixed_horizon": _trade_metrics(gb_fixed),
        "transformer_fixed_horizon": _simulation_metrics(transformer_fixed),
        "pnl_difference_transformer_minus_gb_done": _round(
            _sum(_simulation_pnl(row) for row in transformer_fixed)
            - _sum(_trade_pnl(row) for row in gb_done)
        ),
        "pnl_difference_transformer_minus_gb_fixed_horizon": _round(
            _sum(_simulation_pnl(row) for row in transformer_fixed)
            - _sum(_trade_pnl(row) for row in gb_fixed)
        ),
        "by_horizon_sec": by_horizon,
    }


def _candidate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_candidates": len(rows),
        "decision_counts": _counts(row.get("decision") for row in rows),
        "rejection_reason_counts": _counts(row.get("rejection_reason") for row in rows),
        "by_strategy": _group_metrics(rows, ("strategy_id",), metric_fn=_candidate_metrics),
        "by_rejection_reason": _group_metrics(rows, ("rejection_reason",), metric_fn=_candidate_metrics),
        "by_time_window": _group_metrics(rows, ("time_window",), metric_fn=_candidate_metrics),
    }


def _candidate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "markets_covered": _market_count(rows),
        "decision_counts": _counts(row.get("decision") for row in rows),
        "rejection_reason_counts": _counts(row.get("rejection_reason") for row in rows),
    }


def _trade_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if _status(row) in DONE_STATUSES]
    pnls = [_trade_pnl(row) for row in done]
    rois = [_trade_roi(row) for row in done]
    wins = [value for value in pnls if value is not None and value > 0]
    losses = [value for value in pnls if value is not None and value < 0]
    return {
        "count": len(rows),
        "done_count": len(done),
        "closed_count": sum(1 for row in rows if _status(row) == "closed"),
        "settled_count": sum(1 for row in rows if _status(row) == "settled"),
        "open_count": sum(1 for row in rows if _status(row) == "open"),
        "awaiting_resolution_count": sum(1 for row in rows if _status(row) == "awaiting_resolution"),
        "skipped_count": sum(1 for row in rows if _status(row) == "skipped"),
        "total_pnl": _round(_sum(pnls)),
        "average_roi": _mean(rois),
        "win_rate": _round(len(wins) / len(done)) if done else None,
        "max_loss": _round(min(losses)) if losses else None,
        "max_win": _round(max(wins)) if wins else None,
        "average_win": _mean(wins),
        "average_loss": _mean(losses),
        "markets_covered": _market_count(rows),
        "skip_reason_counts": _counts(_rejection_reason(row) for row in rows if _status(row) == "skipped"),
    }


def _simulation_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_simulation_pnl(row) for row in rows]
    rois = [_simulation_roi(row) for row in rows]
    numeric_pnls = [value for value in pnls if value is not None]
    wins = [value for value in numeric_pnls if value > 0]
    losses = [value for value in numeric_pnls if value < 0]
    return {
        "count": len(rows),
        "evaluated_count": len(numeric_pnls),
        "total_pnl": _round(_sum(numeric_pnls)),
        "average_roi": _mean(rois),
        "win_rate": _round(len(wins) / len(numeric_pnls)) if numeric_pnls else None,
        "max_loss": _round(min(losses)) if losses else None,
        "max_win": _round(max(wins)) if wins else None,
        "average_win": _mean(wins),
        "average_loss": _mean(losses),
        "markets_covered": _market_count(rows),
    }


def _group_metrics(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    metric_fn: Any,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_group_value(row, field) for field in fields)
        groups[key].append(row)
    results: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        item = {field: value for field, value in zip(fields, key)}
        item.update(metric_fn(group_rows))
        results.append(item)
    results.sort(
        key=lambda item: (
            float(item.get("total_pnl") or 0.0),
            int(item.get("count") or 0),
            json.dumps({field: item.get(field) for field in fields}, sort_keys=True),
        ),
        reverse=True,
    )
    return results


def _group_value(row: dict[str, Any], field: str) -> Any:
    if field == "rejection_reason":
        return _rejection_reason(row) or "none"
    if field == "time_window":
        return _time_window(row)
    value = row.get(field)
    if value in (None, ""):
        return "unknown"
    if field in {"threshold", "threshold_used"}:
        parsed = _float_or_none(value)
        return _round(parsed) if parsed is not None else str(value)
    return value


def _fixed_horizon_trade_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if _status(row) not in DONE_STATUSES:
            continue
        exit_type = str(row.get("exit_type") or "").upper()
        exit_reason = str(row.get("exit_reason") or "").lower()
        if (
            "FIXED_HORIZON" in exit_type
            or exit_reason == "fixed_horizon"
            or row.get("fixed_horizon_exit_sec") not in (None, "")
        ):
            result.append(row)
    return result


def _recorder_health(conn: sqlite3.Connection, path: str) -> dict[str, Any]:
    tables = _table_names(conn)
    return {
        "path": path,
        "status": "ok",
        "query_only": _pragma_value(conn, "query_only"),
        "tables_present": sorted(tables),
    }


def _transformer_prediction_db_health(path: str, *, now: datetime) -> dict[str, Any]:
    db_path = Path(path)
    if not db_path.exists():
        return {"path": path, "status": "missing", "rows": 0}
    conn = _connect_read_only(path)
    try:
        if not _table_exists(conn, "transformer_predictions"):
            return {"path": path, "status": "missing_table", "rows": 0}
        row = conn.execute(
            """
            SELECT COUNT(*) AS count,
                   MAX(COALESCE(created_at, signal_timestamp, latest_feature_timestamp, timestamp)) AS latest
            FROM transformer_predictions
            """
        ).fetchone()
    finally:
        conn.close()
    count = int(row["count"] or 0)
    latest = row["latest"]
    age = _age_sec(latest, now)
    status = "ok"
    if count == 0:
        status = "empty"
    elif age is not None and age > STALE_AFTER_SEC:
        status = "stale"
    return {
        "path": path,
        "status": status,
        "rows": count,
        "latest_prediction_timestamp": latest,
        "latest_prediction_age_sec": _round(age),
    }


def _autopsy_health(path: str | None, *, rows: Sequence[dict[str, Any]], now: datetime) -> dict[str, Any]:
    if not path:
        return {"path": None, "status": "not_provided", "rows": 0}
    autopsy_path = Path(path)
    if not autopsy_path.exists():
        return {"path": path, "status": "missing", "rows": 0}
    latest = _latest_timestamp(rows, ("signal_timestamp", "latest_feature_timestamp"))
    age = _age_sec(latest, now)
    status = "ok"
    if not rows:
        status = "empty"
    elif age is not None and age > STALE_AFTER_SEC:
        status = "stale"
    return {
        "path": path,
        "status": status,
        "rows": len(rows),
        "latest_signal_timestamp": latest,
        "latest_signal_age_sec": _round(age),
    }


def _load_autopsy_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ModelExperimentComparisonError(
            "pyarrow is required to read transformer autopsy parquet files"
        ) from exc
    return [dict(row) for row in pq.read_table(source).to_pylist()]


def _read_table(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()]


def _write_comparison_csv(report: dict[str, Any], path: Path) -> None:
    rows = _csv_rows(report)
    columns = [
        "section",
        "group",
        "key",
        "count",
        "done_count",
        "evaluated_count",
        "total_pnl",
        "average_roi",
        "win_rate",
        "max_loss",
        "max_win",
        "markets_covered",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, payload in (
        ("gb_paper_strategy", report.get("gb_paper_strategy") or {}),
        ("transformer_autopsy", report.get("transformer_autopsy") or {}),
    ):
        for group, value in dict(payload).items():
            if group == "candidate_summary":
                continue
            if isinstance(value, dict):
                rows.append(_csv_metric_row(section, group, "overall", value))
            elif isinstance(value, list):
                for item in value:
                    rows.append(_csv_metric_row(section, group, _group_key(item), item))
    for item in list((report.get("comparison") or {}).get("by_horizon_sec") or []):
        rows.append(
            _csv_metric_row(
                "comparison",
                "by_horizon_sec",
                str(item.get("horizon_sec")),
                item.get("transformer_fixed_horizon") or {},
            )
        )
    return rows


def _csv_metric_row(section: str, group: str, key: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": section,
        "group": group,
        "key": key,
        "count": metrics.get("count"),
        "done_count": metrics.get("done_count"),
        "evaluated_count": metrics.get("evaluated_count"),
        "total_pnl": metrics.get("total_pnl"),
        "average_roi": metrics.get("average_roi"),
        "win_rate": metrics.get("win_rate"),
        "max_loss": metrics.get("max_loss"),
        "max_win": metrics.get("max_win"),
        "markets_covered": metrics.get("markets_covered"),
    }


def _group_key(item: dict[str, Any]) -> str:
    ignored = {
        "count",
        "done_count",
        "closed_count",
        "settled_count",
        "open_count",
        "awaiting_resolution_count",
        "skipped_count",
        "evaluated_count",
        "total_pnl",
        "average_roi",
        "win_rate",
        "max_loss",
        "max_win",
        "average_win",
        "average_loss",
        "markets_covered",
        "skip_reason_counts",
        "decision_counts",
        "rejection_reason_counts",
    }
    keys = {key: value for key, value in item.items() if key not in ignored}
    return json.dumps(keys, sort_keys=True, default=str)


def _connect_read_only(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise ModelExperimentComparisonError(f"SQLite DB not found: {path}")
    encoded = quote(str(db_path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _resolve_paths(values: Sequence[str] | str | None) -> list[str]:
    items: list[str]
    if values is None:
        items = []
    elif isinstance(values, str):
        items = [values]
    else:
        items = [str(value) for value in values]
    resolved: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        resolved.extend(matches if matches else [item])
    deduped: list[str] = []
    seen: set[str] = set()
    for item in resolved:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "").lower()


def _trade_pnl(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("realized_pnl_usd", "pnl_usd", "simulated_pnl_usd"))


def _trade_roi(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("realized_roi", "roi", "simulated_roi"))


def _simulation_pnl(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("simulated_pnl_usd", "pnl_usd", "realized_pnl_usd"))


def _simulation_roi(row: dict[str, Any]) -> float | None:
    return _first_number(row, ("simulated_roi", "roi", "realized_roi"))


def _first_number(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _rejection_reason(row: dict[str, Any]) -> str | None:
    for field in ("rejection_reason", "skip_reason", "realistic_execution_skip_reason", "liquidity_skip_reason"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _time_window(row: dict[str, Any]) -> str:
    min_value = _float_or_none(row.get("min_time_until_resolution_sec"))
    max_value = _float_or_none(row.get("max_time_until_resolution_sec"))
    if min_value is not None or max_value is not None:
        return f"{_format_number(min_value)}-{_format_number(max_value)}s"
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


def _format_number(value: float | None) -> str:
    if value is None:
        return "none"
    return str(int(value)) if float(value).is_integer() else str(value)


def _market_count(rows: Sequence[dict[str, Any]]) -> int:
    return len({str(row.get("market_id")) for row in rows if row.get("market_id") not in (None, "")})


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return dict(sorted(counter.items()))


def _sum(values: Iterable[Any]) -> float:
    return sum(value for value in (_float_or_none(item) for item in values) if value is not None)


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return _round(sum(numeric) / len(numeric)) if numeric else None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None


def _round(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, 10) if parsed is not None else None


def _latest_timestamp(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str | None:
    latest: datetime | None = None
    latest_raw: str | None = None
    for row in rows:
        for field in fields:
            raw = row.get(field)
            parsed = _parse_timestamp(raw)
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
                latest_raw = str(raw)
    return latest_raw


def _age_sec(raw_timestamp: Any, now: datetime) -> float | None:
    parsed = _parse_timestamp(raw_timestamp)
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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone() is not None


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }


def _pragma_value(conn: sqlite3.Connection, name: str) -> Any:
    try:
        row = conn.execute(f"PRAGMA {name}").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row is not None else None


def _combined_status(items: Sequence[dict[str, Any]]) -> str:
    statuses = {str(item.get("status")) for item in items}
    if not items:
        return "empty"
    if "ok" in statuses:
        return "ok"
    if "stale" in statuses:
        return "stale"
    if "empty" in statuses:
        return "empty"
    if "missing" in statuses:
        return "missing"
    return "error"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "ModelExperimentComparisonError",
    "compare_model_experiments",
    "render_model_experiment_comparison",
]
