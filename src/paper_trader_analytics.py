from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote


EDGE_BUCKETS: tuple[str, ...] = (
    "edge < 0",
    "0 to 0.02",
    "0.02 to 0.05",
    "0.05 to 0.10",
    "0.10 to 0.20",
    "edge >= 0.20",
)

TIME_BUCKETS: tuple[str, ...] = ("0-30s", "30-60s", "60-120s", "120-180s", "180s+")
PROBABILITY_BUCKETS: tuple[str, ...] = tuple(
    f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}" for idx in range(10)
)

COMPUTED_TRADE_COLUMNS: tuple[str, ...] = (
    "probability_for_direction",
    "estimated_edge",
    "estimated_edge_cents",
    "settled_win",
    "unresolved_age_sec",
)

REAL_TRADE_STATUSES = {"open", "awaiting_resolution", "settled", "closed"}


def build_paper_trader_analytics_report(
    *,
    paper_db_path: str,
    output_dir: str,
    strategy_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    table_missing = False
    heartbeat_missing = False
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    latest_heartbeat: dict[str, Any] | None = None
    latest_multi_strategy_heartbeat: dict[str, Any] | None = None

    try:
        conn = connect_paper_db_read_only(paper_db_path)
    except FileNotFoundError:
        table_missing = True
        warnings.append("paper_db_missing")
        conn = None

    if conn is not None:
        try:
            if _table_exists(conn, "paper_trades"):
                rows = [dict(row) for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC")]
                if strategy_id:
                    rows = [
                        row for row in rows
                        if _strategy_id_for_row(row) == strategy_id
                    ]
            else:
                table_missing = True
                warnings.append("paper_trades_table_missing")
            if _table_exists(conn, "paper_trader_heartbeats"):
                heartbeat = conn.execute(
                    """
                    SELECT *
                    FROM paper_trader_heartbeats
                    ORDER BY datetime(timestamp) DESC, timestamp DESC
                    LIMIT 1
                    """
                ).fetchone()
                latest_heartbeat = dict(heartbeat) if heartbeat is not None else None
            else:
                heartbeat_missing = True
                warnings.append("paper_trader_heartbeats_table_missing")
            if _table_exists(conn, "paper_trade_candidates"):
                candidate_rows = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM paper_trade_candidates ORDER BY id ASC"
                    ).fetchall()
                ]
                if strategy_id:
                    candidate_rows = [
                        row for row in candidate_rows
                        if _strategy_id_for_row(row) == strategy_id
                    ]
            if _table_exists(conn, "multi_strategy_paper_trader_heartbeats"):
                heartbeat = conn.execute(
                    """
                    SELECT *
                    FROM multi_strategy_paper_trader_heartbeats
                    ORDER BY datetime(timestamp) DESC, timestamp DESC
                    LIMIT 1
                    """
                ).fetchone()
                latest_multi_strategy_heartbeat = (
                    dict(heartbeat) if heartbeat is not None else None
                )
        finally:
            conn.close()

    enriched = [_enrich_trade(row, now=now_dt) for row in rows]
    summary = _build_summary(
        enriched,
        paper_db_path=paper_db_path,
        output_dir=str(output),
        latest_heartbeat=latest_heartbeat,
        base_warnings=warnings,
        table_missing=table_missing,
        heartbeat_missing=heartbeat_missing,
        now=now_dt,
        candidate_rows=candidate_rows,
        latest_multi_strategy_heartbeat=latest_multi_strategy_heartbeat,
    )
    report = {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "output_dir": str(output),
        "strategy_id_filter": strategy_id,
        "generated_at": _iso(now_dt),
        "summary": summary,
        "trades": enriched,
    }

    json_path = output / "paper_trader_report.json"
    txt_path = output / "paper_trader_report.txt"
    csv_path = output / "trades_enriched.csv"
    _write_json(json_path, report)
    _write_text(txt_path, render_paper_trader_analytics_report(report))
    _write_trades_csv(csv_path, enriched)
    summary["output_files"] = {
        "json": str(json_path),
        "txt": str(txt_path),
        "trades_enriched_csv": str(csv_path),
    }
    _write_json(json_path, report)
    _write_text(txt_path, render_paper_trader_analytics_report(report))
    return report


def render_paper_trader_analytics_report(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary") or {})
    lines = [
        "Paper Trader Analytics",
        f"paper_db: {report.get('paper_db_path')}",
        f"generated_at: {report.get('generated_at')}",
        "",
        "Headline",
        f"total_trades: {summary.get('total_trades', 0)}",
        f"settled_trades: {summary.get('settled_trades', 0)}",
        f"closed_trades: {summary.get('closed_trades', 0)}",
        f"open_trades: {summary.get('open_trades', 0)}",
        f"awaiting_resolution_trades: {summary.get('awaiting_resolution_trades', 0)}",
        f"settled_pnl: {summary.get('settled_pnl', 0.0)}",
        f"realized_cashout_pnl: {summary.get('realized_cashout_pnl', 0.0)}",
        f"combined_realized_pnl: {summary.get('combined_realized_pnl', 0.0)}",
        f"win_rate_settled: {summary.get('win_rate_settled')}",
        f"cashout_win_rate: {summary.get('cashout_win_rate')}",
        "",
        "Settled Performance",
        f"total_staked: {summary.get('total_staked', 0.0)}",
        f"average_roi_settled: {summary.get('average_roi_settled')}",
        f"average_roi_closed: {summary.get('average_roi_closed')}",
        f"average_result_win: {summary.get('average_result_win')}",
        f"average_result_loss: {summary.get('average_result_loss')}",
        f"average_estimated_edge: {summary.get('average_estimated_edge')}",
        "",
        "Bankroll",
        f"starting_bankroll_usd: {summary.get('starting_bankroll_usd')}",
        f"ending_bankroll_usd: {summary.get('ending_bankroll_usd')}",
        f"bankroll_roi: {summary.get('bankroll_roi')}",
        f"survived: {summary.get('survived')}",
        f"blown_up: {summary.get('blown_up')}",
        f"max_drawdown_usd: {summary.get('max_drawdown_usd')}",
        f"max_drawdown_pct: {summary.get('max_drawdown_pct')}",
        f"total_exposure_peak: {summary.get('total_exposure_peak')}",
        f"profit_per_dollar_staked: {summary.get('profit_per_dollar_staked')}",
        "",
        "Liquidity Fill Checks",
        f"liquidity_checked_trades: {summary.get('liquidity_checked_trades')}",
        f"liquidity_blocked_trades: {summary.get('liquidity_blocked_trades')}",
        f"liquidity_missing_skips: {summary.get('liquidity_missing_skips')}",
        f"liquidity_too_low_skips: {summary.get('liquidity_too_low_skips')}",
        f"liquidity_columns_populated_pct: {summary.get('liquidity_columns_populated_pct')}",
        f"liquidity_check_enabled_detected: {summary.get('liquidity_check_enabled_detected')}",
        f"average_requested_shares: {summary.get('average_requested_shares')}",
        f"average_max_fillable_shares: {summary.get('average_max_fillable_shares')}",
        "",
        "Realistic Execution",
        f"realistic_execution_checked_trades: {summary.get('realistic_execution_checked_trades')}",
        f"realistic_execution_blocked_trades: {summary.get('realistic_execution_blocked_trades')}",
        f"skips_by_realistic_execution_reason: {summary.get('skips_by_realistic_execution_reason')}",
        f"average_entry_price_drift: {summary.get('average_entry_price_drift')}",
        f"average_spread_cents_at_entry: {summary.get('average_spread_cents_at_entry')}",
        "",
        "Unresolved Exposure",
    ]
    exposure = dict(summary.get("unresolved_exposure") or {})
    for key in (
        "open_stake",
        "awaiting_resolution_stake",
        "total_unresolved_stake",
        "worst_case_total_pnl_if_all_unresolved_lose",
        "best_case_total_pnl_if_all_unresolved_win",
    ):
        lines.append(f"{key}: {exposure.get(key)}")

    lines.extend(["", "Direction Breakdown"])
    direction_breakdown = dict(summary.get("direction_breakdown") or {})
    for direction in ("YES", "NO"):
        lines.append(f"{direction}: {json.dumps(direction_breakdown.get(direction, {}), sort_keys=True)}")

    lines.extend(["", "Strategy Breakdown"])
    for strategy_id, metrics in dict(summary.get("strategy_breakdown") or {}).items():
        compact = {
            key: metrics.get(key)
            for key in (
                "total_trades",
                "settled_trades",
                "closed_trades",
                "hold_to_resolution_pnl",
                "realized_cashout_pnl",
                "combined_realized_pnl",
                "win_rate",
                "cashout_win_rate",
                "average_win",
                "average_loss",
                "max_drawdown",
                "market_level_majority_signal_accuracy",
            )
        }
        lines.append(f"{strategy_id}: {json.dumps(compact, sort_keys=True)}")

    lines.extend(["", "Edge Buckets"])
    for bucket, metrics in dict(summary.get("edge_buckets") or {}).items():
        lines.append(f"{bucket}: {json.dumps(metrics, sort_keys=True)}")

    lines.extend(["", "Time Until Resolution Buckets"])
    for bucket, metrics in dict(summary.get("time_until_resolution_buckets") or {}).items():
        lines.append(f"{bucket}: {json.dumps(metrics, sort_keys=True)}")

    lines.extend(["", "Warnings"])
    warnings = list(summary.get("warnings") or [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(["", "Recommendation", str(summary.get("recommendation") or "")])
    return "\n".join(lines) + "\n"


def connect_paper_db_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    paper_db_path: str,
    output_dir: str,
    latest_heartbeat: dict[str, Any] | None,
    base_warnings: Sequence[str],
    table_missing: bool,
    heartbeat_missing: bool,
    now: datetime,
    candidate_rows: Sequence[dict[str, Any]] | None = None,
    latest_multi_strategy_heartbeat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del output_dir, heartbeat_missing
    settled = [row for row in rows if row.get("status") == "settled"]
    closed = [row for row in rows if row.get("status") == "closed"]
    open_rows = [row for row in rows if row.get("status") == "open"]
    awaiting = [row for row in rows if row.get("status") == "awaiting_resolution"]
    unresolved = open_rows + awaiting
    settled_pnl = _sum_numeric(row.get("pnl_usd") for row in settled)
    cashout_pnl = _sum_numeric(row.get("realized_pnl_usd") for row in closed)
    result_rows = settled + closed
    bankroll_start = _first_numeric(rows, "starting_bankroll_usd")
    combined_pnl = settled_pnl + cashout_pnl
    ending_bankroll = bankroll_start + combined_pnl if bankroll_start is not None else None
    bankroll_blown_rows = [
        row for row in rows
        if str(row.get("bankroll_status") or "").upper() == "BLOWN_UP"
    ]
    drawdown_usd = _max_drawdown(result_rows)
    direction_breakdown = _direction_breakdown(rows)
    warnings = list(dict.fromkeys(base_warnings))
    warnings.extend(
        _build_warnings(
            rows,
            settled=settled,
            awaiting=awaiting,
            direction_breakdown=direction_breakdown,
            latest_heartbeat=latest_heartbeat,
            now=now,
            settled_pnl=settled_pnl,
        )
    )
    warnings.extend(_strategy_non_positive_edge_warnings(rows))
    warnings = list(dict.fromkeys(warnings))

    edge_buckets = _bucket_metrics(rows, _edge_bucket)
    time_buckets = _bucket_metrics(rows, _time_bucket)
    probability_buckets = _bucket_metrics(rows, _probability_bucket)
    candidate_stats = _candidate_stats(candidate_rows or [])
    strategy_breakdown = _strategy_breakdown(rows, candidate_rows or [])
    regime_performance = _regime_performance(rows)
    liquidity_summary = _liquidity_summary(
        rows,
        latest_multi_strategy_heartbeat=latest_multi_strategy_heartbeat,
    )
    realistic_execution_summary = _realistic_execution_summary(rows)
    summary = {
        "paper_db_path": paper_db_path,
        "paper_trades_table_missing": table_missing,
        "total_trades": len(rows),
        "settled_trades": len(settled),
        "closed_trades": len(closed),
        "open_trades": len(open_rows),
        "awaiting_resolution_trades": len(awaiting),
        "status_counts": dict(Counter(str(row.get("status") or "") for row in rows)),
        "total_staked": _round(_sum_numeric(row.get("stake_usd") for row in rows if row.get("status") != "skipped")),
        "settled_pnl": _round(settled_pnl),
        "hold_to_resolution_pnl": _round(settled_pnl),
        "realized_cashout_pnl": _round(cashout_pnl),
        "combined_realized_pnl": _round(combined_pnl),
        "starting_bankroll_usd": _round(bankroll_start),
        "ending_bankroll_usd": _round(ending_bankroll),
        "bankroll_roi": _round(
            (ending_bankroll - bankroll_start) / bankroll_start
            if bankroll_start not in (None, 0) and ending_bankroll is not None
            else None
        ),
        "survived": (
            bool(ending_bankroll is not None and ending_bankroll > 0 and not bankroll_blown_rows)
            if bankroll_start is not None
            else None
        ),
        "blown_up": bool(bankroll_blown_rows or (ending_bankroll is not None and ending_bankroll <= 0)),
        "blown_up_at": _earliest_value(row.get("blown_up_at") for row in bankroll_blown_rows),
        "max_drawdown_usd": drawdown_usd,
        "max_drawdown_pct": _round(
            drawdown_usd / bankroll_start
            if drawdown_usd is not None and bankroll_start not in (None, 0)
            else None
        ),
        "average_stake_usd": _mean(row.get("stake_usd") for row in rows if row.get("status") != "skipped"),
        "largest_stake_usd": _max_numeric(row.get("stake_usd") for row in rows if row.get("status") != "skipped"),
        "total_exposure_peak": _total_exposure_peak(rows),
        "profit_per_dollar_staked": _round(
            combined_pnl / _sum_numeric(row.get("stake_usd") for row in rows if row.get("status") != "skipped")
            if _sum_numeric(row.get("stake_usd") for row in rows if row.get("status") != "skipped")
            else None
        ),
        "average_roi_settled": _mean(row.get("roi") for row in settled),
        "average_roi_closed": _mean(row.get("realized_roi") for row in closed),
        "cashout_win_rate": _result_win_rate(closed),
        "average_cashout_win": _average_result_pnl(closed, wins=True),
        "average_cashout_loss": _average_result_pnl(closed, wins=False),
        "average_result_win": _average_result_pnl(result_rows, wins=True),
        "average_result_loss": _average_result_pnl(result_rows, wins=False),
        "win_rate_settled": _win_rate(settled),
        "pnl_by_direction": {
            direction: direction_breakdown[direction]["pnl"]
            for direction in ("YES", "NO")
        },
        "direction_breakdown": direction_breakdown,
        "win_rate_by_direction": {
            direction: direction_breakdown[direction]["win_rate"]
            for direction in ("YES", "NO")
        },
        "average_entry_price_by_direction": {
            direction: direction_breakdown[direction]["average_entry_price"]
            for direction in ("YES", "NO")
        },
        "average_adjusted_entry_price_by_direction": {
            direction: direction_breakdown[direction]["average_adjusted_entry_price"]
            for direction in ("YES", "NO")
        },
        "average_predicted_probability_by_direction": {
            direction: direction_breakdown[direction]["average_predicted_probability"]
            for direction in ("YES", "NO")
        },
        "average_estimated_edge_by_direction": {
            direction: direction_breakdown[direction]["average_estimated_edge"]
            for direction in ("YES", "NO")
        },
        "average_estimated_edge": _mean(row.get("estimated_edge") for row in rows),
        "edge_buckets": edge_buckets,
        "pnl_by_estimated_edge_bucket": _metric_by_bucket(edge_buckets, "pnl"),
        "win_rate_by_estimated_edge_bucket": _metric_by_bucket(edge_buckets, "win_rate"),
        "average_roi_by_estimated_edge_bucket": _metric_by_bucket(edge_buckets, "average_roi"),
        "trade_count_by_estimated_edge_bucket": _metric_by_bucket(edge_buckets, "trade_count"),
        "time_until_resolution_buckets": time_buckets,
        "pnl_by_time_until_resolution_bucket": _metric_by_bucket(time_buckets, "pnl"),
        "win_rate_by_time_until_resolution_bucket": _metric_by_bucket(time_buckets, "win_rate"),
        "probability_buckets": probability_buckets,
        "pnl_by_probability_bucket": _metric_by_bucket(probability_buckets, "pnl"),
        "win_rate_by_probability_bucket": _metric_by_bucket(probability_buckets, "win_rate"),
        "duplicate_market_keys": _duplicate_market_keys(rows),
        "unresolved_exposure": _unresolved_exposure(unresolved, settled_pnl=settled_pnl),
        "unresolved_exposure_by_direction": _unresolved_exposure_by_direction(unresolved),
        "unresolved_exposure_by_estimated_edge_bucket": _unresolved_exposure_by_edge_bucket(unresolved),
        "strategy_breakdown": strategy_breakdown,
        "regime_performance": regime_performance,
        **liquidity_summary,
        **realistic_execution_summary,
        "latest_heartbeat": latest_heartbeat,
        "latest_multi_strategy_heartbeat": latest_multi_strategy_heartbeat,
        "warnings": warnings,
        "recommendation": _recommendation(
            settled_count=len(settled),
            settled_pnl=settled_pnl,
            win_rate=_win_rate(settled),
            average_edge=_mean(row.get("estimated_edge") for row in rows),
            direction_breakdown=direction_breakdown,
            awaiting_count=len(awaiting),
            warnings=warnings,
        ),
        **candidate_stats,
    }
    return summary


def _enrich_trade(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    result = dict(row)
    direction = str(row.get("signal_direction") or "").upper()
    predicted_yes = _float_or_none(row.get("predicted_probability_yes"))
    adjusted_entry = _float_or_none(row.get("adjusted_entry_price"))
    probability_for_direction: float | None = _float_or_none(row.get("probability_for_direction"))
    if probability_for_direction is None and predicted_yes is not None:
        if direction == "YES":
            probability_for_direction = predicted_yes
        elif direction == "NO":
            probability_for_direction = 1.0 - predicted_yes
    edge = _float_or_none(row.get("estimated_edge"))
    if edge is None:
        edge = (
        probability_for_direction - adjusted_entry
        if probability_for_direction is not None and adjusted_entry is not None
        else None
        )
    status = str(row.get("status") or "")
    pnl = _float_or_none(row.get("pnl_usd"))
    result["probability_for_direction"] = _round(probability_for_direction)
    result["estimated_edge"] = _round(edge)
    result["estimated_edge_cents"] = _round(edge * 100.0 if edge is not None else None)
    result["settled_win"] = (
        1 if status == "settled" and pnl is not None and pnl > 0 else
        0 if status == "settled" and pnl is not None else
        None
    )
    if status in {"open", "awaiting_resolution"}:
        created_at = _parse_timestamp(row.get("created_at"))
        result["unresolved_age_sec"] = (
            _round((now - created_at).total_seconds()) if created_at is not None else None
        )
    else:
        result["unresolved_age_sec"] = None
    return result


def _direction_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for direction in ("YES", "NO"):
        direction_rows = [
            row for row in rows if str(row.get("signal_direction") or "").upper() == direction
        ]
        settled = [row for row in direction_rows if row.get("status") == "settled"]
        closed = [row for row in direction_rows if row.get("status") == "closed"]
        report[direction] = {
            "trades": len(direction_rows),
            "settled_trades": len(settled),
            "closed_trades": len(closed),
            "pnl": _round(_sum_numeric(row.get("pnl_usd") for row in settled)),
            "hold_to_resolution_pnl": _round(
                _sum_numeric(row.get("pnl_usd") for row in settled)
            ),
            "realized_cashout_pnl": _round(
                _sum_numeric(row.get("realized_pnl_usd") for row in closed)
            ),
            "combined_realized_pnl": _round(
                _sum_numeric(row.get("pnl_usd") for row in settled)
                + _sum_numeric(row.get("realized_pnl_usd") for row in closed)
            ),
            "win_rate": _win_rate(settled),
            "cashout_win_rate": _result_win_rate(closed),
            "average_entry_price": _mean(row.get("entry_price") for row in direction_rows),
            "average_adjusted_entry_price": _mean(
                row.get("adjusted_entry_price") for row in direction_rows
            ),
            "average_predicted_probability": _mean(
                row.get("probability_for_direction") for row in direction_rows
            ),
            "average_estimated_edge": _mean(row.get("estimated_edge") for row in direction_rows),
        }
    return report


def _bucket_metrics(
    rows: Sequence[dict[str, Any]],
    bucket_fn: Any,
) -> dict[str, dict[str, Any]]:
    ordered_buckets = (
        EDGE_BUCKETS
        if bucket_fn is _edge_bucket
        else TIME_BUCKETS
        if bucket_fn is _time_bucket
        else PROBABILITY_BUCKETS
    )
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in ordered_buckets}
    for row in rows:
        bucket = bucket_fn(row)
        if bucket is not None:
            grouped.setdefault(bucket, []).append(row)
    return {
        bucket: {
            "trade_count": len(bucket_rows),
            "settled_trades": len([row for row in bucket_rows if row.get("status") == "settled"]),
            "pnl": _round(
                _sum_numeric(
                    row.get("pnl_usd")
                    for row in bucket_rows
                    if row.get("status") == "settled"
                )
            ),
            "win_rate": _win_rate(
                [row for row in bucket_rows if row.get("status") == "settled"]
            ),
            "average_roi": _mean(
                row.get("roi") for row in bucket_rows if row.get("status") == "settled"
            ),
            "average_adjusted_entry_price": _mean(
                row.get("adjusted_entry_price") for row in bucket_rows
            ),
        }
        for bucket, bucket_rows in grouped.items()
    }


def _metric_by_bucket(
    bucket_metrics: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    return {bucket: values.get(metric) for bucket, values in bucket_metrics.items()}


def _edge_bucket(row: dict[str, Any]) -> str | None:
    edge = _float_or_none(row.get("estimated_edge"))
    if edge is None:
        return None
    if edge < 0:
        return "edge < 0"
    if edge < 0.02:
        return "0 to 0.02"
    if edge < 0.05:
        return "0.02 to 0.05"
    if edge < 0.10:
        return "0.05 to 0.10"
    if edge < 0.20:
        return "0.10 to 0.20"
    return "edge >= 0.20"


def _time_bucket(row: dict[str, Any]) -> str | None:
    seconds = _float_or_none(row.get("time_until_resolution"))
    if seconds is None:
        return None
    if seconds < 30:
        return "0-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    if seconds < 180:
        return "120-180s"
    return "180s+"


def _probability_bucket(row: dict[str, Any]) -> str | None:
    probability = _float_or_none(row.get("probability_for_direction"))
    if probability is None:
        return None
    clamped = min(0.999999, max(0.0, probability))
    idx = int(clamped * 10)
    return PROBABILITY_BUCKETS[idx]


def _unresolved_exposure(
    rows: Sequence[dict[str, Any]],
    *,
    settled_pnl: float,
) -> dict[str, Any]:
    open_stake = _sum_numeric(
        row.get("stake_usd") for row in rows if row.get("status") == "open"
    )
    awaiting_stake = _sum_numeric(
        row.get("stake_usd") for row in rows if row.get("status") == "awaiting_resolution"
    )
    unresolved_stake = open_stake + awaiting_stake
    best_case_pnl = settled_pnl
    for row in rows:
        stake = _float_or_none(row.get("stake_usd")) or 0.0
        adjusted_entry = _float_or_none(row.get("adjusted_entry_price"))
        if adjusted_entry is not None and adjusted_entry > 0:
            best_case_pnl += (stake / adjusted_entry) - stake
    return {
        "open_stake": _round(open_stake),
        "awaiting_resolution_stake": _round(awaiting_stake),
        "total_unresolved_stake": _round(unresolved_stake),
        "worst_case_total_pnl_if_all_unresolved_lose": _round(settled_pnl - unresolved_stake),
        "best_case_total_pnl_if_all_unresolved_win": _round(best_case_pnl),
    }


def _unresolved_exposure_by_direction(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        direction: {
            "trade_count": len(
                [
                    row
                    for row in rows
                    if str(row.get("signal_direction") or "").upper() == direction
                ]
            ),
            "stake": _round(
                _sum_numeric(
                    row.get("stake_usd")
                    for row in rows
                    if str(row.get("signal_direction") or "").upper() == direction
                )
            ),
        }
        for direction in ("YES", "NO")
    }


def _unresolved_exposure_by_edge_bucket(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        bucket: {"trade_count": 0, "stake": 0.0}
        for bucket in EDGE_BUCKETS
    }
    for row in rows:
        bucket = _edge_bucket(row)
        if bucket is None:
            continue
        report[bucket]["trade_count"] += 1
        report[bucket]["stake"] = _round(
            float(report[bucket]["stake"] or 0.0)
            + float(_float_or_none(row.get("stake_usd")) or 0.0)
        )
    return report


def _candidate_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    settled = [
        row for row in rows
        if row.get("actual_resolved_label") not in (None, "")
    ]
    skipped = [
        row for row in rows
        if str(row.get("decision") or "").upper() != "TRADE"
    ]
    trade_count = sum(1 for row in rows if str(row.get("decision") or "").upper() == "TRADE")
    positive_skipped = [
        row for row in skipped
        if (_float_or_none(row.get("would_have_roi")) or 0.0) > 0
    ]
    return {
        "total_candidates": total,
        "settled_candidates": len(settled),
        "candidate_trade_rate": _round(trade_count / total) if total else None,
        "skipped_candidate_count": len(skipped),
        "skipped_candidate_would_have_won_count": len(positive_skipped),
        "skipped_candidate_would_have_positive_roi_count": len(positive_skipped),
        "average_would_have_roi_by_decision": _average_candidate_roi_by(rows, "decision"),
        "average_would_have_roi_by_rejection_reason": _average_candidate_roi_by(
            rows,
            "rejection_reason",
        ),
    }


def _average_candidate_roi_by(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        group = str(row.get(key) or "")
        grouped.setdefault(group, []).append(row.get("would_have_roi"))
    return {group: _mean(values) for group, values in sorted(grouped.items())}


def _strategy_breakdown(
    trade_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    strategy_ids = sorted(
        {
            _strategy_id_for_row(row)
            for row in [*trade_rows, *candidate_rows]
        }
    )
    report_rows: list[tuple[str, dict[str, Any]]] = []
    for strategy_id in strategy_ids:
        trades = [
            row for row in trade_rows
            if _strategy_id_for_row(row) == strategy_id
            and str(row.get("status") or "") in REAL_TRADE_STATUSES
        ]
        settled = [row for row in trades if row.get("status") == "settled"]
        closed = [row for row in trades if row.get("status") == "closed"]
        result_rows = settled + closed
        hold_pnl = _sum_numeric(row.get("pnl_usd") for row in settled)
        cashout_pnl = _sum_numeric(row.get("realized_pnl_usd") for row in closed)
        candidates = [
            row for row in candidate_rows
            if _strategy_id_for_row(row) == strategy_id
        ]
        skipped_candidates = [
            row for row in candidates
            if str(row.get("decision") or "").upper() != "TRADE"
        ]
        candidate_trade_count = sum(
            1 for row in candidates
            if str(row.get("decision") or "").upper() == "TRADE"
        )
        item = {
            "trade_count": len(trades),
            "settled_count": len(settled),
            "closed_count": len(closed),
            "total_trades": len(trades),
            "settled_trades": len(settled),
            "closed_trades": len(closed),
            "open_trades": len([row for row in trades if row.get("status") == "open"]),
            "awaiting_resolution_trades": len(
                [row for row in trades if row.get("status") == "awaiting_resolution"]
            ),
            "settled_pnl": _round(hold_pnl),
            "hold_to_resolution_pnl": _round(hold_pnl),
            "realized_cashout_pnl": _round(cashout_pnl),
            "combined_realized_pnl": _round(hold_pnl + cashout_pnl),
            "win_rate": _win_rate(settled),
            "cashout_win_rate": _result_win_rate(closed),
            "result_win_rate": _result_win_rate(result_rows),
            "average_roi": _mean(row.get("roi") for row in settled),
            "avg_roi": _mean(row.get("roi") for row in settled),
            "average_cashout_roi": _mean(row.get("realized_roi") for row in closed),
            "average_result_roi": _mean(_result_roi(row) for row in result_rows),
            "average_win": _average_result_pnl(result_rows, wins=True),
            "average_loss": _average_result_pnl(result_rows, wins=False),
            "max_drawdown": _max_drawdown(result_rows),
            "market_level_majority_signal_accuracy": _market_majority_signal_accuracy(
                result_rows
            ),
            "average_estimated_edge": _mean(row.get("estimated_edge") for row in trades),
            "total_candidates": len(candidates),
            "candidate_trade_rate": (
                _round(candidate_trade_count / len(candidates)) if candidates else None
            ),
            "skipped_candidate_count": len(skipped_candidates),
            "skipped_would_have_roi": _mean(
                row.get("would_have_roi") for row in skipped_candidates
            ),
            "best_rejection_reasons": _ranked_rejection_reasons(
                skipped_candidates,
                reverse=True,
            ),
            "worst_rejection_reasons": _ranked_rejection_reasons(
                skipped_candidates,
                reverse=False,
            ),
        }
        report_rows.append((strategy_id, item))
    report_rows.sort(
        key=lambda pair: (
            -float(_float_or_none(pair[1].get("settled_pnl")) or 0.0),
            pair[0],
        ),
    )
    return {strategy_id: item for strategy_id, item in report_rows}


REGIME_FIELDS: dict[str, tuple[str, ...]] = {
    "btc_trend_regime": (
        "strong_up",
        "weak_up",
        "flat",
        "weak_down",
        "strong_down",
        "unknown",
    ),
    "volatility_regime": ("low", "medium", "high", "unknown"),
    "spread_regime": ("tight", "normal", "wide", "unknown"),
    "liquidity_regime": ("thin", "normal", "deep", "unknown"),
    "time_regime": ("0-30s", "30-60s", "60-120s", "120s+", "unknown"),
}


def _regime_performance(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    real_rows = [
        row for row in rows
        if str(row.get("status") or "") in REAL_TRADE_STATUSES
    ]
    performance: dict[str, dict[str, Any]] = {}
    for field, ordered_values in REGIME_FIELDS.items():
        grouped: dict[str, list[dict[str, Any]]] = {
            value: [] for value in ordered_values
        }
        for row in real_rows:
            regime = str(row.get(field) or "unknown")
            if not regime or regime.lower() == "none":
                regime = "unknown"
            grouped.setdefault(regime, []).append(row)
        performance[field] = {
            regime: _regime_group_metrics(group_rows)
            for regime, group_rows in grouped.items()
        }
    return performance


def _regime_group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("status") == "settled"]
    closed = [row for row in rows if row.get("status") == "closed"]
    result_rows = settled + closed
    pnl = _sum_numeric(_result_pnl(row) for row in result_rows)
    return {
        "trades": len(rows),
        "settled_trades": len(settled),
        "closed_trades": len(closed),
        "pnl": _round(pnl),
        "roi": _mean(_result_roi(row) for row in result_rows),
        "win_rate": _result_win_rate(result_rows),
        "average_stake": _mean(row.get("stake_usd") for row in rows),
        "max_drawdown": _max_drawdown(result_rows),
    }


def _liquidity_summary(
    rows: Sequence[dict[str, Any]],
    *,
    latest_multi_strategy_heartbeat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checked = [
        row for row in rows
        if row.get("liquidity_check_passed") is not None
        or row.get("liquidity_skip_reason") not in (None, "")
    ]
    blocked = [
        row for row in checked
        if str(row.get("liquidity_skip_reason") or row.get("skip_reason") or "")
        in {"liquidity_missing", "liquidity_too_low", "invalid_adjusted_entry_price"}
        or _int_or_none(row.get("liquidity_check_passed")) == 0
    ]
    missing = [
        row for row in checked
        if str(row.get("liquidity_skip_reason") or row.get("skip_reason") or "")
        == "liquidity_missing"
    ]
    too_low = [
        row for row in checked
        if str(row.get("liquidity_skip_reason") or row.get("skip_reason") or "")
        == "liquidity_too_low"
    ]
    real_trade_rows = [
        row for row in rows
        if str(row.get("status") or "") in REAL_TRADE_STATUSES
    ]
    populated = [
        row for row in real_trade_rows
        if row.get("liquidity_check_passed") is not None
        or row.get("liquidity_skip_reason") not in (None, "")
        or row.get("requested_shares") is not None
        or row.get("max_fillable_shares") is not None
        or row.get("liquidity_fill_fraction_used") is not None
    ]
    heartbeat_enabled = _heartbeat_liquidity_enabled(latest_multi_strategy_heartbeat)
    return {
        "liquidity_checked_trades": len(checked),
        "liquidity_blocked_trades": len(blocked),
        "liquidity_missing_skips": len(missing),
        "liquidity_too_low_skips": len(too_low),
        "liquidity_columns_populated_pct": _round(
            len(populated) / len(real_trade_rows) * 100.0 if real_trade_rows else None
        ),
        "liquidity_check_enabled_detected": bool(checked or heartbeat_enabled),
        "average_requested_shares": _mean(row.get("requested_shares") for row in checked),
        "average_max_fillable_shares": _mean(
            row.get("max_fillable_shares") for row in checked
        ),
    }


def _heartbeat_liquidity_enabled(
    heartbeat: dict[str, Any] | None,
) -> bool:
    if not heartbeat:
        return False
    if _int_or_none(heartbeat.get("liquidity_fill_check_enabled")) == 1:
        return True
    raw = heartbeat.get("per_strategy_json")
    if not raw:
        return False
    try:
        strategies = json.loads(str(raw))
    except json.JSONDecodeError:
        return False
    if not isinstance(strategies, list):
        return False
    return any(
        bool(item.get("liquidity_fill_check_enabled"))
        for item in strategies
        if isinstance(item, dict)
    )


def _realistic_execution_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checked = [
        row for row in rows
        if _int_or_none(row.get("realistic_execution_enabled")) == 1
        or row.get("realistic_execution_skip_reason") not in (None, "")
    ]
    blocked = [
        row for row in checked
        if row.get("realistic_execution_skip_reason") not in (None, "")
        or str(row.get("skip_reason") or "") in {
            "quote_after_latency_missing",
            "entry_price_drift_too_high",
            "spread_too_wide",
            "btc_stale",
            "feature_stale",
        }
    ]
    reasons = Counter(
        str(row.get("realistic_execution_skip_reason") or row.get("skip_reason") or "")
        for row in blocked
        if str(row.get("realistic_execution_skip_reason") or row.get("skip_reason") or "")
    )
    return {
        "realistic_execution_checked_trades": len(checked),
        "realistic_execution_blocked_trades": len(blocked),
        "skips_by_realistic_execution_reason": dict(sorted(reasons.items())),
        "average_entry_price_drift": _mean(row.get("entry_price_drift") for row in checked),
        "average_spread_cents_at_entry": _mean(
            row.get("spread_cents_at_entry") for row in checked
        ),
    }


def _strategy_non_positive_edge_warnings(rows: Sequence[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        if str(row.get("status") or "") not in REAL_TRADE_STATUSES:
            continue
        grouped.setdefault(_strategy_id_for_row(row), []).append(row.get("estimated_edge"))
    warnings: list[str] = []
    for strategy_id, values in sorted(grouped.items()):
        average_edge = _mean(values)
        if average_edge is not None and average_edge <= 0:
            warnings.append("strategy_with_non_positive_average_estimated_edge")
            warnings.append(f"strategy_non_positive_average_estimated_edge:{strategy_id}")
    return warnings


def _result_pnl(row: dict[str, Any]) -> float | None:
    status = str(row.get("status") or "")
    if status == "closed":
        return _float_or_none(row.get("realized_pnl_usd"))
    if status == "settled":
        return _float_or_none(row.get("pnl_usd"))
    return None


def _result_roi(row: dict[str, Any]) -> float | None:
    status = str(row.get("status") or "")
    if status == "closed":
        return _float_or_none(row.get("realized_roi"))
    if status == "settled":
        return _float_or_none(row.get("roi"))
    return None


def _result_win_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    results = [_result_pnl(row) for row in rows]
    numeric = [float(value) for value in results if value is not None]
    if not numeric:
        return None
    wins = sum(1 for value in numeric if value > 0)
    return _round(wins / len(numeric))


def _average_result_pnl(
    rows: Sequence[dict[str, Any]],
    *,
    wins: bool,
) -> float | None:
    values = [
        value
        for value in (_result_pnl(row) for row in rows)
        if value is not None and ((value > 0) if wins else (value <= 0))
    ]
    return _mean(values)


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


def _first_numeric(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _max_numeric(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in (_float_or_none(value) for value in values) if value is not None]
    return _round(max(numeric)) if numeric else None


def _earliest_value(values: Iterable[Any]) -> Any:
    present = [value for value in values if value not in (None, "")]
    return sorted(str(value) for value in present)[0] if present else None


def _total_exposure_peak(rows: Sequence[dict[str, Any]]) -> float | None:
    peaks: list[float] = []
    for row in rows:
        before = _float_or_none(row.get("open_exposure_before_trade"))
        stake = _float_or_none(row.get("stake_usd"))
        if before is None:
            continue
        peaks.append(float(before or 0.0) + float(stake or 0.0))
    return _round(max(peaks)) if peaks else None


def _market_majority_signal_accuracy(rows: Sequence[dict[str, Any]]) -> float | None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get("resolved_label") or "").upper()
        if label not in {"YES", "NO"}:
            continue
        grouped.setdefault(
            (str(row.get("run_id") or ""), str(row.get("market_id") or "")),
            [],
        ).append(row)
    if not grouped:
        return None
    correct = 0
    total = 0
    for group in grouped.values():
        labels = [str(row.get("resolved_label") or "").upper() for row in group]
        actual = labels[0]
        votes = Counter(str(row.get("signal_direction") or "").upper() for row in group)
        if not votes:
            continue
        majority = votes.most_common(1)[0][0]
        total += 1
        if majority == actual:
            correct += 1
    return _round(correct / total) if total else None


def _ranked_rejection_reasons(
    rows: Sequence[dict[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        reason = str(row.get("rejection_reason") or "")
        if not reason:
            continue
        grouped.setdefault(reason, []).append(row.get("would_have_roi"))
    ranked = [
        {
            "rejection_reason": reason,
            "candidate_count": len(values),
            "average_would_have_roi": _mean(values),
        }
        for reason, values in grouped.items()
    ]
    ranked.sort(
        key=lambda item: (
            item["average_would_have_roi"]
            if item["average_would_have_roi"] is not None
            else (-1_000_000.0 if reverse else 1_000_000.0)
        ),
        reverse=reverse,
    )
    return ranked[: int(limit)]


def _strategy_id_for_row(row: dict[str, Any]) -> str:
    return str(row.get("strategy_id") or "baseline_default")


def _build_warnings(
    rows: Sequence[dict[str, Any]],
    *,
    settled: Sequence[dict[str, Any]],
    awaiting: Sequence[dict[str, Any]],
    direction_breakdown: dict[str, dict[str, Any]],
    latest_heartbeat: dict[str, Any] | None,
    now: datetime,
    settled_pnl: float,
) -> list[str]:
    warnings: list[str] = []
    if len(settled) < 100:
        warnings.append("fewer_than_100_settled_trades")
    if len(settled) < 30:
        warnings.append("fewer_than_30_settled_trades")
    if settled_pnl < 0:
        warnings.append("negative_settled_pnl")
    win_rate = _win_rate(settled)
    if win_rate is not None and win_rate < 0.55:
        warnings.append("win_rate_below_55_percent")
    if float(direction_breakdown["YES"].get("pnl") or 0.0) < 0:
        warnings.append("yes_pnl_negative")
    if float(direction_breakdown["NO"].get("pnl") or 0.0) < 0:
        warnings.append("no_pnl_negative")
    if any(
        row.get("status") == "settled"
        and (_float_or_none(row.get("estimated_edge")) or 0.0) < 0
        for row in rows
    ):
        warnings.append("settled_trade_with_negative_estimated_edge")
    average_edge = _mean(row.get("estimated_edge") for row in rows)
    if average_edge is not None and average_edge <= 0:
        warnings.append("average_estimated_edge_non_positive")
    if len(awaiting) > 10:
        warnings.append("more_than_10_awaiting_resolution_trades")
    if any((_float_or_none(row.get("unresolved_age_sec")) or 0.0) > 1800 for row in awaiting):
        warnings.append("awaiting_resolution_trade_older_than_30_minutes")
    if any(_open_trade_past_close(row, now=now) for row in rows if row.get("status") == "open"):
        warnings.append("open_trade_past_market_close_time")
    average_tur = _mean(row.get("time_until_resolution") for row in rows)
    if average_tur is not None and average_tur > 180:
        warnings.append("average_time_until_resolution_above_180_seconds")
    if _duplicate_market_keys(rows):
        warnings.append("duplicate_trades_in_same_run_id_market_id")
    if _heartbeat_has_recent_seen_but_no_eligible(latest_heartbeat, now=now):
        warnings.append("latest_eligible_feature_timestamp_null_while_seen_feature_recent")
    if _high_probability_underperforms(rows):
        warnings.append("high_probability_bucket_underperforms_lower_probability_bucket")
    if _high_edge_bucket_underperforms(rows):
        warnings.append("edge_ge_0_20_win_rate_below_55_percent")
    return warnings


def _recommendation(
    *,
    settled_count: int,
    settled_pnl: float,
    win_rate: float | None,
    average_edge: float | None,
    direction_breakdown: dict[str, dict[str, Any]],
    awaiting_count: int,
    warnings: Sequence[str],
) -> str:
    parts: list[str] = []
    if settled_count < 100:
        parts.append("Keep paper trading and do not go live; fewer than 100 trades are settled.")
    if average_edge is not None and average_edge <= 0:
        parts.append("Thresholds need EV filtering; average estimated edge is non-positive.")
    if awaiting_count > 10:
        parts.append("Wait for settlement before trusting PnL; more than 10 trades are awaiting resolution.")
    if (
        "high_probability_bucket_underperforms_lower_probability_bucket" in warnings
        or "edge_ge_0_20_win_rate_below_55_percent" in warnings
    ):
        parts.append("Calibration looks inverted in at least one bucket; consider stricter min edge or probability block guardrails.")
    yes_pnl = float(direction_breakdown["YES"].get("pnl") or 0.0)
    no_pnl = float(direction_breakdown["NO"].get("pnl") or 0.0)
    if settled_pnl > 0 and win_rate is not None and win_rate > 0.55 and yes_pnl > 0 and no_pnl > 0:
        parts.append("Signal is promising but not validated; both directions are positive in settled paper results.")
    if not parts:
        parts.append("Continue paper trading and review edge, direction, and settlement cohorts before changing thresholds.")
    return " ".join(parts)


def _high_probability_underperforms(rows: Sequence[dict[str, Any]]) -> bool:
    buckets = _bucket_metrics(rows, _probability_bucket)
    high = buckets.get("0.9-1.0") or {}
    high_wr = _float_or_none(high.get("win_rate"))
    high_count = int(high.get("settled_trades") or 0)
    if high_wr is None or high_count == 0:
        return False
    for bucket, metrics in buckets.items():
        if bucket == "0.9-1.0":
            continue
        wr = _float_or_none(metrics.get("win_rate"))
        count = int(metrics.get("settled_trades") or 0)
        if wr is not None and count > 0 and high_wr < wr:
            return True
    return False


def _high_edge_bucket_underperforms(rows: Sequence[dict[str, Any]]) -> bool:
    buckets = _bucket_metrics(rows, _edge_bucket)
    high = buckets.get("edge >= 0.20") or {}
    return int(high.get("settled_trades") or 0) >= 10 and (
        _float_or_none(high.get("win_rate")) or 0.0
    ) < 0.55


def _duplicate_market_keys(rows: Sequence[dict[str, Any]]) -> list[str]:
    counts: Counter[tuple[str, str]] = Counter(
        (str(row.get("run_id") or ""), str(row.get("market_id") or ""))
        for row in rows
        if str(row.get("status") or "") in REAL_TRADE_STATUSES
        if row.get("run_id") is not None and row.get("market_id") is not None
    )
    return [
        f"{run_id}:{market_id}"
        for (run_id, market_id), count in sorted(counts.items())
        if count > 1
    ]


def _open_trade_past_close(row: dict[str, Any], *, now: datetime) -> bool:
    close_time = _parse_timestamp(row.get("market_close_time"))
    return close_time is not None and now >= close_time


def _heartbeat_has_recent_seen_but_no_eligible(
    heartbeat: dict[str, Any] | None,
    *,
    now: datetime,
) -> bool:
    if not heartbeat:
        return False
    if heartbeat.get("latest_eligible_feature_timestamp") not in (None, ""):
        return False
    seen = _parse_timestamp(heartbeat.get("latest_seen_feature_timestamp"))
    return seen is not None and (now - seen).total_seconds() <= 60


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _write_trades_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if rows:
        original_columns = sorted(
            column
            for column in set().union(*(row.keys() for row in rows))
            if column not in COMPUTED_TRADE_COLUMNS
        )
    else:
        original_columns = []
    fieldnames = tuple(original_columns) + COMPUTED_TRADE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _sum_numeric(values: Iterable[Any]) -> float:
    return sum(float(value) for value in (_float_or_none(value) for value in values) if value is not None)


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in (_float_or_none(value) for value in values) if value is not None]
    if not numeric:
        return None
    return _round(sum(numeric) / len(numeric))


def _win_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    if not settled_rows:
        return None
    wins = sum(1 for row in settled_rows if int(row.get("settled_win") or 0) == 1)
    return _round(wins / len(settled_rows))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(float(value), 10) if value is not None else None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "build_paper_trader_analytics_report",
    "connect_paper_db_read_only",
    "render_paper_trader_analytics_report",
]
