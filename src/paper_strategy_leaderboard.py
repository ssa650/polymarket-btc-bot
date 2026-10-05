from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paper_trader_analytics import connect_paper_db_read_only


LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "signal_direction",
    "total_trades",
    "settled_trades",
    "open_trades",
    "awaiting_resolution_trades",
    "settled_pnl",
    "average_roi_settled",
    "win_rate_settled",
    "average_estimated_edge",
    "average_probability_for_direction",
    "average_entry_price",
    "average_adjusted_entry_price",
    "average_time_until_resolution",
    "first_trade_time",
    "latest_trade_time",
    "reliability_tier",
    "warnings",
)


def build_paper_strategy_leaderboard(
    *,
    paper_db_path: str,
    min_settled: int = 30,
    direction: str | None = None,
    strategy_ids: Sequence[str] | None = None,
    output_path: str | None = None,
    output_csv_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    normalized_direction = _normalize_direction(direction)
    strategy_filter = {
        str(strategy_id)
        for strategy_id in (strategy_ids or [])
        if str(strategy_id).strip()
    }
    try:
        conn = connect_paper_db_read_only(paper_db_path)
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "paper_db_missing",
            "paper_db_path": paper_db_path,
            "rows": [],
        }
    try:
        if not _table_exists(conn, "paper_trades"):
            return {
                "status": "error",
                "error": "paper_trades_table_missing",
                "paper_db_path": paper_db_path,
                "rows": [],
            }
        rows = [dict(row) for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC")]
    finally:
        conn.close()

    trade_rows = [
        row for row in rows
        if str(row.get("status") or "").lower() != "skipped"
        and str(row.get("signal_direction") or "").upper() in {"YES", "NO"}
    ]
    if normalized_direction:
        trade_rows = [
            row for row in trade_rows
            if str(row.get("signal_direction") or "").upper() == normalized_direction
        ]
    if strategy_filter:
        trade_rows = [
            row for row in trade_rows
            if _strategy_id(row) in strategy_filter
        ]

    leaderboard_rows = [
        item for item in _group_rows(trade_rows)
        if int(item["settled_trades"]) >= int(min_settled)
    ]
    leaderboard_rows.sort(
        key=lambda item: (
            -float(item["settled_pnl"] or 0.0),
            -float(item["average_roi_settled"] if item["average_roi_settled"] is not None else -1e9),
            str(item["strategy_id"]),
            str(item["signal_direction"]),
        )
    )
    report = {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "generated_at": _iso(now_dt),
        "min_settled": int(min_settled),
        "direction": normalized_direction,
        "strategy_ids": sorted(strategy_filter),
        "row_count": len(leaderboard_rows),
        "rows": leaderboard_rows,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["output_path"] = str(path)
    if output_csv_path:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_path, leaderboard_rows)
        report["output_csv_path"] = str(csv_path)
    return report


def render_paper_strategy_leaderboard(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return json.dumps(report, indent=2, sort_keys=True) + "\n"
    rows = list(report.get("rows") or [])
    lines = [
        "Paper Strategy Leaderboard",
        f"paper_db: {report.get('paper_db_path')}",
        f"min_settled: {report.get('min_settled')}",
        f"rows: {len(rows)}",
        "",
    ]
    if not rows:
        lines.append("No strategy/direction groups matched the filters.")
        return "\n".join(lines) + "\n"
    header = (
        f"{'strategy_id':38} {'dir':3} {'settled':>7} {'pnl':>10} "
        f"{'roi':>8} {'win':>8} {'edge':>8} {'tier':>13} warnings"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        warnings = ",".join(row.get("warnings") or [])
        lines.append(
            f"{str(row.get('strategy_id'))[:38]:38} "
            f"{str(row.get('signal_direction'))[:3]:3} "
            f"{int(row.get('settled_trades') or 0):7d} "
            f"{float(row.get('settled_pnl') or 0.0):10.4f} "
            f"{_fmt(row.get('average_roi_settled')):>8} "
            f"{_fmt(row.get('win_rate_settled')):>8} "
            f"{_fmt(row.get('average_estimated_edge')):>8} "
            f"{str(row.get('reliability_tier')):>13} "
            f"{warnings}"
        )
    return "\n".join(lines) + "\n"


def _group_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (_strategy_id(row), str(row.get("signal_direction") or "").upper())
        grouped.setdefault(key, []).append(row)
    return [
        _metrics_for_group(strategy_id, direction, group_rows)
        for (strategy_id, direction), group_rows in sorted(grouped.items())
    ]


def _metrics_for_group(
    strategy_id: str,
    direction: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    settled = [row for row in rows if str(row.get("status") or "").lower() == "settled"]
    open_rows = [row for row in rows if str(row.get("status") or "").lower() == "open"]
    awaiting = [
        row for row in rows
        if str(row.get("status") or "").lower() == "awaiting_resolution"
    ]
    settled_pnl = _round(_sum(row.get("pnl_usd") for row in settled))
    average_roi = _mean(row.get("roi") for row in settled)
    win_rate = _win_rate(settled)
    first_trade_time = min((_trade_time(row) for row in rows if _trade_time(row)), default=None)
    latest_trade_time = max((_trade_time(row) for row in rows if _trade_time(row)), default=None)
    item = {
        "strategy_id": strategy_id,
        "signal_direction": direction,
        "total_trades": len(rows),
        "settled_trades": len(settled),
        "open_trades": len(open_rows),
        "awaiting_resolution_trades": len(awaiting),
        "settled_pnl": settled_pnl,
        "average_roi_settled": average_roi,
        "win_rate_settled": win_rate,
        "average_estimated_edge": _mean(row.get("estimated_edge") for row in rows),
        "average_probability_for_direction": _mean(
            row.get("probability_for_direction") for row in rows
        ),
        "average_entry_price": _mean(row.get("entry_price") for row in rows),
        "average_adjusted_entry_price": _mean(row.get("adjusted_entry_price") for row in rows),
        "average_time_until_resolution": _mean(row.get("time_until_resolution") for row in rows),
        "first_trade_time": first_trade_time,
        "latest_trade_time": latest_trade_time,
        "reliability_tier": _reliability_tier(len(settled)),
    }
    item["warnings"] = _warnings_for_row(item)
    return item


def _warnings_for_row(item: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    settled = int(item.get("settled_trades") or 0)
    total = int(item.get("total_trades") or 0)
    awaiting = int(item.get("awaiting_resolution_trades") or 0)
    if settled < 30:
        warnings.append("low_sample")
    if float(item.get("settled_pnl") or 0.0) < 0:
        warnings.append("negative_pnl")
    win_rate = _float_or_none(item.get("win_rate_settled"))
    if win_rate is not None and win_rate < 0.5:
        warnings.append("win_rate_below_50")
    roi = _float_or_none(item.get("average_roi_settled"))
    if roi is not None and roi < 0:
        warnings.append("avg_roi_below_zero")
    if total > 0 and (awaiting / total) > 0.25:
        warnings.append("high_awaiting_resolution_ratio")
    return warnings


def _reliability_tier(settled_trades: int) -> str:
    if settled_trades < 30:
        return "too_early"
    if settled_trades < 100:
        return "early"
    if settled_trades < 300:
        return "usable_sample"
    return "strong_sample"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["warnings"] = json.dumps(record.get("warnings") or [])
            writer.writerow({column: record.get(column) for column in LEADERBOARD_COLUMNS})


def _normalize_direction(direction: str | None) -> str | None:
    if direction is None or str(direction).strip() == "":
        return None
    normalized = str(direction).strip().upper()
    if normalized not in {"YES", "NO"}:
        raise ValueError("--direction must be YES or NO")
    return normalized


def _strategy_id(row: dict[str, Any]) -> str:
    return str(row.get("strategy_id") or "baseline_default")


def _trade_time(row: dict[str, Any]) -> str | None:
    value = row.get("created_at") or row.get("signal_timestamp")
    return str(value) if value not in (None, "") else None


def _win_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    wins = sum(1 for row in rows if (_float_or_none(row.get("pnl_usd")) or 0.0) > 0)
    return _round(wins / len(rows))


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in (_float_or_none(value) for value in values)
        if value is not None
    ]
    if not numeric:
        return None
    return _round(sum(numeric) / len(numeric))


def _sum(values: Iterable[Any]) -> float:
    return sum(
        float(value)
        for value in (_float_or_none(value) for value in values)
        if value is not None
    )


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(float(value), 10) if value is not None else None


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "null"
    return f"{numeric:.4f}"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _table_exists(conn: Any, table: str) -> bool:
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


__all__ = [
    "build_paper_strategy_leaderboard",
    "render_paper_strategy_leaderboard",
]
