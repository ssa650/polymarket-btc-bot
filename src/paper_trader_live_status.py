"""Compact terminal status for recorder and paper trader DBs."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Sequence

from .models import parse_timestamp
from .web_dashboard import (
    fetch_paper_activity_payload,
    fetch_paper_closed_payload,
    fetch_paper_open_payload,
    fetch_paper_skips_payload,
    fetch_paper_traders_summary,
    fetch_recorder_summary_payload,
    resolve_paper_db_paths,
)


def build_paper_trader_live_status(
    *,
    recorder_db_path: str,
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
    limit_activity: int = 20,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only terminal status payload."""

    now_dt = now or datetime.now(timezone.utc)
    resolved_paper_paths = resolve_paper_db_paths(paper_db_paths, paper_db_globs)
    try:
        recorder = fetch_recorder_summary_payload(recorder_db_path, now=now_dt)
    except Exception as exc:
        recorder = {
            "now": now_dt.isoformat(),
            "db_path": recorder_db_path,
            "health_status": "ERROR",
            "error": str(exc),
            "active_market": None,
            "btc": {},
            "yes": {},
            "no": {},
            "recorder_health": {"status": "ERROR"},
            "warnings": ["recorder_status_unavailable"],
        }

    return {
        "now": now_dt.isoformat(),
        "recorder": recorder,
        "paper_db_paths": resolved_paper_paths,
        "paper_traders": fetch_paper_traders_summary(resolved_paper_paths, now=now_dt),
        "activity": fetch_paper_activity_payload(
            resolved_paper_paths,
            limit=limit_activity,
            now=now_dt,
        ),
        "skips": fetch_paper_skips_payload(
            resolved_paper_paths,
            limit=limit_activity,
            now=now_dt,
        ),
        "open": fetch_paper_open_payload(resolved_paper_paths, now=now_dt),
        "closed": fetch_paper_closed_payload(
            resolved_paper_paths,
            limit=limit_activity,
            now=now_dt,
        ),
    }


def render_paper_trader_live_status(
    report: dict[str, Any],
    *,
    show_skips: bool = False,
    show_open: bool = False,
    show_closed: bool = False,
) -> str:
    lines: list[str] = []
    lines.append(f"Paper Trader Live Status | {report.get('now')}")
    lines.append("")
    lines.extend(_render_recorder(report.get("recorder") or {}))
    lines.append("")
    lines.extend(_render_paper_traders(report.get("paper_traders") or {}))
    lines.append("")
    lines.extend(_render_activity(report.get("activity") or {}))
    if show_skips:
        lines.append("")
        lines.extend(_render_skips(report.get("skips") or {}))
    if show_open:
        lines.append("")
        lines.extend(_render_rows("Open / Awaiting", report.get("open") or {}))
    if show_closed:
        lines.append("")
        lines.extend(_render_rows("Closed / Settled", report.get("closed") or {}))
    return "\n".join(lines).rstrip() + "\n"


def run_paper_trader_live_status(
    *,
    recorder_db_path: str,
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
    limit_activity: int = 20,
    show_skips: bool = False,
    show_open: bool = False,
    show_closed: bool = False,
    watch_sec: float | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    iteration = 0
    latest_report: dict[str, Any] = {}
    while True:
        latest_report = build_paper_trader_live_status(
            recorder_db_path=recorder_db_path,
            paper_db_paths=paper_db_paths,
            paper_db_globs=paper_db_globs,
            limit_activity=limit_activity,
        )
        rendered = render_paper_trader_live_status(
            latest_report,
            show_skips=show_skips,
            show_open=show_open,
            show_closed=show_closed,
        )
        if watch_sec is not None:
            print("\033[2J\033[H", end="")
        print(rendered, end="", flush=True)
        iteration += 1
        if watch_sec is None:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break
        time.sleep(max(0.1, float(watch_sec)))
    return latest_report


def _render_recorder(recorder: dict[str, Any]) -> list[str]:
    health = recorder.get("recorder_health") or {}
    active = recorder.get("active_market") or {}
    btc = recorder.get("btc") or {}
    yes = recorder.get("yes") or {}
    no = recorder.get("no") or {}
    lines = ["Recorder"]
    if recorder.get("error"):
        lines.append(f"  status={recorder.get('health_status', 'ERROR')} error={recorder.get('error')}")
        return lines
    lines.append(
        "  "
        f"status={health.get('status')} "
        f"snapshot_age={_fmt_num(health.get('snapshot_age_sec'))}s "
        f"feature_age={_fmt_num(health.get('feature_age_sec'))}s "
        f"btc_age={_fmt_num(health.get('btc_age_sec'))}s "
        f"ws_age={_fmt_num(health.get('ws_event_age_sec'))}s"
    )
    lines.append(
        "  "
        f"active_market={active.get('market_id') or '-'} "
        f"time_left={_fmt_num(active.get('time_left_sec'))}s "
        f"phase={active.get('phase') or '-'} "
        f"tracking={active.get('tracking_state') or '-'}"
    )
    lines.append(
        "  "
        f"BTC={_fmt_money(btc.get('price'))} "
        f"source={btc.get('source') or '-'} "
        f"age={_fmt_num(btc.get('sample_age_sec'))}s"
    )
    lines.append(
        "  "
        f"YES bid={_fmt_price(yes.get('best_bid'))} "
        f"ask={_fmt_price(yes.get('best_ask'))} "
        f"spread={_fmt_price(yes.get('spread'))}"
    )
    lines.append(
        "  "
        f"NO  bid={_fmt_price(no.get('best_bid'))} "
        f"ask={_fmt_price(no.get('best_ask'))} "
        f"spread={_fmt_price(no.get('spread'))}"
    )
    return lines


def _render_paper_traders(payload: dict[str, Any]) -> list[str]:
    traders = payload.get("traders") or []
    totals = payload.get("totals") or {}
    lines = [
        "Paper Traders",
        (
            "  "
            f"dbs={totals.get('paper_db_count', 0)} "
            f"trades={totals.get('total_trades', 0)} "
            f"open/await={totals.get('open_awaiting_trades', 0)} "
            f"closed={totals.get('closed_trades', 0)} "
            f"settled={totals.get('settled_trades', 0)} "
            f"skipped={totals.get('skipped_trades', 0)} "
            f"pnl={_fmt_signed(totals.get('realized_pnl'))}"
        ),
    ]
    if not traders:
        lines.append("  no paper DBs configured")
        return lines
    for trader in traders:
        top_skip = _top_count(trader.get("skip_reason_counts") or {})
        top_strategy = _top_strategy(trader.get("top_strategies_by_pnl") or [])
        bottom_strategy = _top_strategy(trader.get("bottom_strategies_by_pnl") or [])
        lines.append(
            "  "
            f"{trader.get('label')} "
            f"{trader.get('status')} "
            f"total={trader.get('total_trades', 0)} "
            f"open/await={trader.get('open_awaiting_trades', 0)} "
            f"closed={trader.get('closed_trades', 0)} "
            f"settled={trader.get('settled_trades', 0)} "
            f"skipped={trader.get('skipped_trades', 0)} "
            f"pnl={_fmt_signed(trader.get('realized_pnl'))} "
            f"avg_roi={_fmt_pct(trader.get('avg_roi'))} "
            f"win={_fmt_pct(trader.get('win_rate'))} "
            f"avg_stake={_fmt_money(trader.get('avg_stake'))} "
            f"bankroll={_fmt_money(trader.get('current_bankroll'))} "
            f"exposure={_fmt_money(trader.get('open_exposure'))} "
            f"latest={_short_time(trader.get('latest_activity_time'))} "
            f"top_skip={top_skip} "
            f"top={top_strategy} "
            f"bottom={bottom_strategy}"
        )
        if trader.get("error"):
            lines.append(f"    error={trader.get('error')}")
    return lines


def _render_activity(payload: dict[str, Any]) -> list[str]:
    return _render_rows("Recent Activity", payload)


def _render_skips(payload: dict[str, Any]) -> list[str]:
    lines = ["Skips"]
    groups = payload.get("groups") or {}
    for name in ("skip_reason", "realistic_execution_skip_reason", "liquidity_skip_reason", "strategy_id", "trader_label"):
        values = groups.get(name) or {}
        if values:
            lines.append(f"  {name}: {_format_counts(values)}")
    rows = payload.get("rows") or []
    if rows:
        lines.append("  recent:")
        lines.extend(_format_row(row) for row in rows[:10])
    return lines


def _render_rows(title: str, payload: dict[str, Any]) -> list[str]:
    rows = payload.get("rows") or []
    lines = [title]
    if not rows:
        lines.append("  none")
        return lines
    for row in rows[:20]:
        lines.append(_format_row(row))
    return lines


def _format_row(row: dict[str, Any]) -> str:
    reason = row.get("skip_reason") or row.get("realistic_execution_skip_reason") or row.get("liquidity_skip_reason") or "-"
    return (
        "  "
        f"{_short_time(row.get('activity_time') or row.get('created_at'))} "
        f"{row.get('trader_label') or '-'} "
        f"market={row.get('market_id') or '-'} "
        f"strategy={row.get('strategy_id') or '-'} "
        f"dir={row.get('signal_direction') or '-'} "
        f"status={row.get('status') or '-'} "
        f"stake={_fmt_money(row.get('stake_usd'))} "
        f"entry={_fmt_price(row.get('entry_price'))}/{_fmt_price(row.get('adjusted_entry_price'))} "
        f"exit={_fmt_price(row.get('cashout_exit_price'))} "
        f"pnl={_fmt_signed(row.get('pnl'))} "
        f"roi={_fmt_pct(row.get('roi_combined'))} "
        f"prob={_fmt_price(row.get('probability_for_direction'))} "
        f"edge={_fmt_price(row.get('estimated_edge'))} "
        f"ttr={_fmt_num(row.get('time_until_resolution'))}s "
        f"bankroll={_fmt_money(row.get('bankroll_after_trade'))} "
        f"reason={reason}"
    )


def _top_count(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    key, value = next(iter(counts.items()))
    return f"{key}({value})"


def _top_strategy(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    row = rows[0]
    return f"{row.get('strategy_id')}({_fmt_signed(row.get('pnl'))})"


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in list(counts.items())[:8])


def _short_time(value: Any) -> str:
    parsed = parse_timestamp(value)
    if parsed is None:
        return "-"
    return parsed.astimezone(timezone.utc).strftime("%H:%M:%S")


def _fmt_num(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number:.1f}"


def _fmt_money(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"${number:.2f}"


def _fmt_signed(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number:+.2f}"


def _fmt_pct(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number * 100:.1f}%"


def _fmt_price(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number:.4f}"


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_paper_trader_live_status",
    "render_paper_trader_live_status",
    "run_paper_trader_live_status",
]
