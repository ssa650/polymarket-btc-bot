from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .paper_trader_analytics import connect_paper_db_read_only


REAL_TRADE_STATUSES = {"open", "awaiting_resolution", "settled", "closed"}
DONE_STATUSES = {"settled", "closed"}
REGIME_FIELDS = (
    "btc_trend_regime",
    "volatility_regime",
    "spread_regime",
    "liquidity_regime",
    "time_regime",
)


def build_paper_trader_promotion_report(
    *,
    paper_db_paths: Sequence[str],
    output_path: str | None = None,
    output_txt_path: str | None = None,
    recorder_db_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    db_paths = [str(path) for path in paper_db_paths if str(path).strip()]
    if not db_paths:
        raise ValueError("at least one paper DB path is required")

    db_reports = [
        _build_single_db_report(path, recorder_db_path=recorder_db_path)
        for path in db_paths
    ]
    ranked = sorted(
        [item for item in db_reports if item.get("status") == "ok"],
        key=lambda item: (
            -float(item.get("deployability_score") or 0.0),
            -float(item.get("combined_realized_pnl") or 0.0),
            str(item.get("paper_db_path") or ""),
        ),
    )
    for index, item in enumerate(ranked, start=1):
        item["deployability_rank"] = index

    report = {
        "status": "ok",
        "generated_at": _iso(now_dt),
        "recorder_db_path": recorder_db_path,
        "recorder_db": _recorder_db_metadata(recorder_db_path),
        "paper_db_count": len(db_paths),
        "ranked_db_paths": [item["paper_db_path"] for item in ranked],
        "paper_dbs": db_reports,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        report["output_path"] = str(path)
    if output_txt_path:
        txt_path = Path(output_txt_path)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(render_paper_trader_promotion_report(report), encoding="utf-8")
        report["output_txt_path"] = str(txt_path)
    return report


def render_paper_trader_promotion_report(report: dict[str, Any]) -> str:
    lines = [
        "Paper Trader Promotion Report",
        f"generated_at: {report.get('generated_at')}",
        f"recorder_db: {report.get('recorder_db_path') or 'not provided'}",
        f"paper_db_count: {report.get('paper_db_count')}",
        "",
    ]
    dbs = list(report.get("paper_dbs") or [])
    ok_rows = [row for row in dbs if row.get("status") == "ok"]
    if not ok_rows:
        lines.append("No readable paper DBs with paper_trades were found.")
        for row in dbs:
            lines.append(
                f"- {row.get('paper_db_path')}: {row.get('status')} {row.get('error') or ''}"
            )
        return "\n".join(lines) + "\n"

    header = (
        f"{'rank':>4} {'db':34} {'done':>6} {'pnl':>10} {'win':>8} "
        f"{'avg_roi':>8} {'dd':>9} {'bankroll_roi':>12} {'score':>8} warnings"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in sorted(ok_rows, key=lambda item: int(item.get("deployability_rank") or 9999)):
        warnings = ",".join(row.get("warning_flags") or [])
        lines.append(
            f"{int(row.get('deployability_rank') or 0):4d} "
            f"{Path(str(row.get('paper_db_path') or '')).name[:34]:34} "
            f"{int(row.get('done_trades') or 0):6d} "
            f"{float(row.get('combined_realized_pnl') or 0.0):10.4f} "
            f"{_fmt(row.get('done_win_rate')):>8} "
            f"{_fmt(row.get('average_roi')):>8} "
            f"{_fmt(row.get('max_drawdown')):>9} "
            f"{_fmt(row.get('bankroll_roi')):>12} "
            f"{_fmt(row.get('deployability_score')):>8} "
            f"{warnings}"
        )

    lines.extend(["", "Details"])
    for row in sorted(ok_rows, key=lambda item: int(item.get("deployability_rank") or 9999)):
        lines.extend(
            [
                "",
                f"{row.get('deployability_rank')}. {row.get('paper_db_path')}",
                f"total_trades: {row.get('total_trades')}",
                f"closed_trades: {row.get('closed_trades')}",
                f"settled_trades: {row.get('settled_trades')}",
                f"awaiting_trades: {row.get('awaiting_trades')}",
                f"combined_realized_pnl: {row.get('combined_realized_pnl')}",
                f"ending_bankroll: {row.get('ending_bankroll')}",
                f"profit_per_dollar_staked: {row.get('profit_per_dollar_staked')}",
                f"trades_per_hour: {row.get('trades_per_hour')}",
                f"liquidity_blocked_trades: {row.get('liquidity_blocked_trades')}",
                f"realistic_execution_blocked_trades: {row.get('realistic_execution_blocked_trades')}",
                f"pnl_by_signal_direction: {json.dumps(row.get('pnl_by_signal_direction') or {}, sort_keys=True)}",
                f"top_strategy_ids_by_pnl: {json.dumps(row.get('top_10_strategy_ids_by_pnl') or [], sort_keys=True)}",
                f"bottom_strategy_ids_by_pnl: {json.dumps(row.get('bottom_10_strategy_ids_by_pnl') or [], sort_keys=True)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_single_db_report(
    paper_db_path: str,
    *,
    recorder_db_path: str | None,
) -> dict[str, Any]:
    try:
        conn = connect_paper_db_read_only(paper_db_path)
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "paper_db_missing",
            "paper_db_path": paper_db_path,
            "recorder_db_path": recorder_db_path,
        }
    try:
        if not _table_exists(conn, "paper_trades"):
            return {
                "status": "error",
                "error": "paper_trades_table_missing",
                "paper_db_path": paper_db_path,
                "recorder_db_path": recorder_db_path,
            }
        rows = [
            dict(row)
            for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
        ]
    finally:
        conn.close()
    return _metrics_for_db(
        paper_db_path=paper_db_path,
        recorder_db_path=recorder_db_path,
        rows=rows,
    )


def _metrics_for_db(
    *,
    paper_db_path: str,
    recorder_db_path: str | None,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    real_rows = [
        row for row in rows
        if str(row.get("status") or "").lower() in REAL_TRADE_STATUSES
    ]
    closed = [row for row in real_rows if str(row.get("status") or "").lower() == "closed"]
    settled = [row for row in real_rows if str(row.get("status") or "").lower() == "settled"]
    awaiting = [
        row for row in real_rows
        if str(row.get("status") or "").lower() == "awaiting_resolution"
    ]
    done = closed + settled
    combined_pnl = _sum_numeric(_result_pnl(row) for row in done)
    total_staked = _sum_numeric(row.get("stake_usd") for row in real_rows)
    starting_bankroll = _first_numeric(rows, "starting_bankroll_usd")
    ending_bankroll = (
        starting_bankroll + combined_pnl
        if starting_bankroll is not None
        else _last_numeric(rows, "bankroll_after_trade")
    )
    max_drawdown = _max_drawdown(done)
    max_drawdown_pct = (
        _round(max_drawdown / starting_bankroll)
        if max_drawdown is not None and starting_bankroll not in (None, 0)
        else None
    )
    row = {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "recorder_db_path": recorder_db_path,
        "total_trades": len(rows),
        "real_trades": len(real_rows),
        "closed_trades": len(closed),
        "settled_trades": len(settled),
        "awaiting_trades": len(awaiting),
        "open_trades": len(
            [item for item in real_rows if str(item.get("status") or "").lower() == "open"]
        ),
        "done_trades": len(done),
        "combined_realized_pnl": _round(combined_pnl),
        "done_win_rate": _result_win_rate(done),
        "average_roi": _mean(_result_roi(item) for item in done),
        "median_roi": _median(_result_roi(item) for item in done),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "starting_bankroll": _round(starting_bankroll),
        "ending_bankroll": _round(ending_bankroll),
        "bankroll_roi": _round(
            (ending_bankroll - starting_bankroll) / starting_bankroll
            if ending_bankroll is not None and starting_bankroll not in (None, 0)
            else None
        ),
        "max_exposure_used": _max_exposure_used(real_rows),
        "average_stake": _mean(row.get("stake_usd") for row in real_rows),
        "profit_per_dollar_staked": _round(
            combined_pnl / total_staked if total_staked else None
        ),
        "trades_per_hour": _trades_per_hour(real_rows),
        "pnl_by_signal_direction": _pnl_by_signal_direction(done),
        "pnl_by_regime_tag": _pnl_by_regime_tag(done),
        "top_10_strategy_ids_by_pnl": _strategy_rankings(done, reverse=True)[:10],
        "bottom_10_strategy_ids_by_pnl": _strategy_rankings(done, reverse=False)[:10],
        "liquidity_blocked_trades": _liquidity_blocked_trades(rows),
        "realistic_execution_blocked_trades": _realistic_execution_blocked_trades(rows),
        "status_counts": dict(sorted(Counter(str(row.get("status") or "") for row in rows).items())),
        "direction_counts": dict(
            sorted(Counter(str(row.get("signal_direction") or "") for row in real_rows).items())
        ),
    }
    row["warning_flags"] = _warning_flags(row)
    row["deployability_score"] = _deployability_score(row)
    return row


def _warning_flags(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    done = int(row.get("done_trades") or 0)
    if done < 100:
        warnings.append("fewer_than_100_done_trades")
    if float(row.get("combined_realized_pnl") or 0.0) < 0:
        warnings.append("negative_pnl")
    win_rate = _float_or_none(row.get("done_win_rate"))
    if win_rate is not None and win_rate < 0.55:
        warnings.append("win_rate_below_55_percent")
    avg_roi = _float_or_none(row.get("average_roi"))
    if avg_roi is not None and avg_roi < 0:
        warnings.append("avg_roi_below_0")
    awaiting = int(row.get("awaiting_trades") or 0)
    if awaiting > 10 or (done > 0 and awaiting / done > 0.25):
        warnings.append("too_many_awaiting_resolution")
    if _one_direction_dominates(row):
        warnings.append("one_direction_dominates")
    drawdown_pct = _float_or_none(row.get("max_drawdown_pct"))
    drawdown = _float_or_none(row.get("max_drawdown"))
    pnl = abs(float(row.get("combined_realized_pnl") or 0.0))
    if (drawdown_pct is not None and drawdown_pct >= 0.20) or (
        drawdown is not None and pnl > 0 and drawdown >= pnl
    ):
        warnings.append("high_drawdown")
    return warnings


def _deployability_score(row: dict[str, Any]) -> float:
    done = int(row.get("done_trades") or 0)
    sample_score = min(done, 300) / 300.0 * 30.0
    pnl = float(row.get("combined_realized_pnl") or 0.0)
    pnl_score = max(min(pnl, 25.0), -25.0)
    win_rate = _float_or_none(row.get("done_win_rate"))
    win_score = 0.0 if win_rate is None else max(min((win_rate - 0.55) * 100.0, 20.0), -20.0)
    avg_roi = _float_or_none(row.get("average_roi"))
    roi_score = 0.0 if avg_roi is None else max(min(avg_roi * 50.0, 15.0), -15.0)
    drawdown_pct = _float_or_none(row.get("max_drawdown_pct"))
    drawdown_penalty = 0.0 if drawdown_pct is None else min(drawdown_pct * 75.0, 25.0)
    awaiting = int(row.get("awaiting_trades") or 0)
    awaiting_penalty = min(awaiting * 0.5, 10.0)
    warning_penalty = len(row.get("warning_flags") or []) * 3.0
    score = 50.0 + sample_score + pnl_score + win_score + roi_score
    score -= drawdown_penalty + awaiting_penalty + warning_penalty
    return _round(max(0.0, min(100.0, score))) or 0.0


def _pnl_by_signal_direction(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, float] = {"YES": 0.0, "NO": 0.0}
    for row in rows:
        direction = str(row.get("signal_direction") or "UNKNOWN").upper()
        grouped.setdefault(direction, 0.0)
        grouped[direction] += float(_result_pnl(row) or 0.0)
    return {key: _round(value) or 0.0 for key, value in sorted(grouped.items())}


def _pnl_by_regime_tag(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in REGIME_FIELDS:
        grouped: dict[str, float] = {}
        for row in rows:
            regime = str(row.get(field) or "unknown")
            grouped.setdefault(regime, 0.0)
            grouped[regime] += float(_result_pnl(row) or 0.0)
        result[field] = {
            regime: _round(value) or 0.0
            for regime, value in sorted(grouped.items())
        }
    return result


def _strategy_rankings(rows: Sequence[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_strategy_id(row), []).append(row)
    items = []
    for strategy_id, strategy_rows in grouped.items():
        pnl = _sum_numeric(_result_pnl(row) for row in strategy_rows)
        items.append(
            {
                "strategy_id": strategy_id,
                "done_trades": len(strategy_rows),
                "pnl": _round(pnl),
                "win_rate": _result_win_rate(strategy_rows),
                "average_roi": _mean(_result_roi(row) for row in strategy_rows),
            }
        )
    items.sort(
        key=lambda item: (
            float(item.get("pnl") or 0.0),
            float(item.get("average_roi") or 0.0),
            str(item.get("strategy_id") or ""),
        ),
        reverse=reverse,
    )
    return items


def _liquidity_blocked_trades(rows: Sequence[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("liquidity_skip_reason") or row.get("skip_reason") or "")
        in {"liquidity_missing", "liquidity_too_low", "invalid_adjusted_entry_price"}
        or _int_or_none(row.get("liquidity_check_passed")) == 0
    )


def _realistic_execution_blocked_trades(rows: Sequence[dict[str, Any]]) -> int:
    reasons = {
        "quote_after_latency_missing",
        "entry_price_drift_too_high",
        "spread_too_wide",
        "btc_stale",
        "feature_stale",
    }
    return sum(
        1
        for row in rows
        if str(row.get("realistic_execution_skip_reason") or row.get("skip_reason") or "")
        in reasons
    )


def _max_exposure_used(rows: Sequence[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        before = _float_or_none(row.get("open_exposure_before_trade"))
        stake = _float_or_none(row.get("stake_usd")) or 0.0
        if before is not None:
            values.append(before + stake)
        configured = _float_or_none(row.get("max_open_exposure_usd"))
        if configured is not None:
            values.append(min(configured, before + stake if before is not None else configured))
    return _round(max(values)) if values else None


def _trades_per_hour(rows: Sequence[dict[str, Any]]) -> float | None:
    times = [
        timestamp for timestamp in (_trade_time(row) for row in rows)
        if timestamp is not None
    ]
    if not times:
        return None
    span_hours = (max(times) - min(times)).total_seconds() / 3600.0
    if span_hours <= 0:
        return float(len(rows)) if rows else None
    return _round(len(rows) / span_hours)


def _one_direction_dominates(row: dict[str, Any]) -> bool:
    counts = {
        key: int(value)
        for key, value in dict(row.get("direction_counts") or {}).items()
        if key in {"YES", "NO"}
    }
    total = sum(counts.values())
    return bool(total >= 10 and counts and max(counts.values()) / total >= 0.80)


def _max_drawdown(rows: Sequence[dict[str, Any]]) -> float | None:
    ordered = sorted(
        [row for row in rows if _result_pnl(row) is not None],
        key=lambda row: (
            _parse_timestamp(row.get("exit_time"))
            or _parse_timestamp(row.get("settled_at"))
            or _parse_timestamp(row.get("created_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            int(row.get("id") or 0),
        ),
    )
    if not ordered:
        return None
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in ordered:
        cumulative += float(_result_pnl(row) or 0.0)
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return _round(abs(max_drawdown))


def _result_pnl(row: dict[str, Any]) -> float | None:
    status = str(row.get("status") or "").lower()
    if status == "closed":
        return _float_or_none(row.get("realized_pnl_usd"))
    if status == "settled":
        return _float_or_none(row.get("pnl_usd"))
    return None


def _result_roi(row: dict[str, Any]) -> float | None:
    status = str(row.get("status") or "").lower()
    if status == "closed":
        return _float_or_none(row.get("realized_roi"))
    if status == "settled":
        return _float_or_none(row.get("roi"))
    return None


def _result_win_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    values = [_result_pnl(row) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    wins = sum(1 for value in numeric if value > 0)
    return _round(wins / len(numeric))


def _trade_time(row: dict[str, Any]) -> datetime | None:
    for field in ("created_at", "signal_timestamp", "exit_time", "settled_at"):
        parsed = _parse_timestamp(row.get(field))
        if parsed is not None:
            return parsed
    return None


def _strategy_id(row: dict[str, Any]) -> str:
    return str(row.get("strategy_id") or "baseline_default")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _recorder_db_metadata(recorder_db_path: str | None) -> dict[str, Any] | None:
    if not recorder_db_path:
        return None
    path = Path(recorder_db_path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _first_numeric(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _last_numeric(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _sum_numeric(values: Iterable[Any]) -> float:
    return sum(
        float(value)
        for value in (_float_or_none(value) for value in values)
        if value is not None
    )


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in (_float_or_none(value) for value in values)
        if value is not None
    ]
    if not numeric:
        return None
    return _round(sum(numeric) / len(numeric))


def _median(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in (_float_or_none(value) for value in values)
        if value is not None
    ]
    if not numeric:
        return None
    return _round(float(median(numeric)))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _round(value: Any, digits: int = 10) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, digits) if parsed is not None else None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _fmt(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.4f}"


__all__ = [
    "build_paper_trader_promotion_report",
    "render_paper_trader_promotion_report",
]
