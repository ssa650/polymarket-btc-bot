from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .models import parse_timestamp


AUTOPSY_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "run_id",
    "market_id",
    "status",
    "signal_direction",
    "adjusted_entry_price",
    "fixed_horizon_exit_price",
    "fixed_horizon_pnl",
    "hold_to_resolution_pnl",
    "hold_to_resolution_roi",
    "would_win_at_resolution",
    "actual_outcome",
    "outcome_source",
    "btc_start",
    "btc_close",
    "btc_move",
    "probability_for_direction",
    "estimated_edge",
    "time_until_resolution",
    "time_regime",
    "btc_trend_regime",
    "liquidity_regime",
    "spread_regime",
    "volatility_regime",
)


def build_paper_trade_resolution_autopsy(
    *,
    recorder_db_path: str,
    paper_db_path: str,
    output_root: str = "data/analytics",
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    recorder = _connect_read_only(recorder_db_path)
    paper = _connect_read_only(paper_db_path)
    try:
        if not _table_exists(paper, "paper_trades"):
            return {
                "status": "error",
                "error": "paper_trades_table_missing",
                "paper_db_path": paper_db_path,
                "recorder_db_path": recorder_db_path,
            }
        trades = _closed_or_settled_trades(paper)
        rows = [
            _autopsy_row(recorder, trade)
            for trade in trades
        ]
    finally:
        recorder.close()
        paper.close()

    run_id = _output_run_id(rows, paper_db_path)
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "resolution_autopsy.csv"
    _write_csv(csv_path, rows)
    report = {
        "status": "ok",
        "generated_at": _iso(now_dt),
        "recorder_db_path": recorder_db_path,
        "paper_db_path": paper_db_path,
        "run_id": run_id,
        "trade_count": len(rows),
        "csv_path": str(csv_path),
        "grouped_summaries": {
            "time_regime": _group_summary(rows, "time_regime"),
            "btc_trend_regime": _group_summary(rows, "btc_trend_regime"),
            "liquidity_regime": _group_summary(rows, "liquidity_regime"),
            "probability_bucket": _group_summary(rows, "probability_bucket"),
            "signal_direction": _group_summary(rows, "signal_direction"),
        },
    }
    return report


def render_paper_trade_resolution_autopsy(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return json.dumps(report, indent=2, sort_keys=True) + "\n"
    lines = [
        "Paper Trade Resolution Autopsy",
        f"paper_db: {report.get('paper_db_path')}",
        f"recorder_db: {report.get('recorder_db_path')}",
        f"trades: {report.get('trade_count')}",
        f"csv: {report.get('csv_path')}",
        "",
    ]
    grouped = dict(report.get("grouped_summaries") or {})
    for group_name in (
        "time_regime",
        "btc_trend_regime",
        "liquidity_regime",
        "probability_bucket",
        "signal_direction",
    ):
        lines.append(group_name)
        rows = list(grouped.get(group_name) or [])
        if not rows:
            lines.append("  no rows")
            lines.append("")
            continue
        for row in rows:
            lines.append(
                "  "
                f"{row.get('group')}: "
                f"trades={row.get('trades')} "
                f"resolution_win_rate={_fmt(row.get('resolution_win_rate'))} "
                f"fixed_pnl={_fmt(row.get('fixed_horizon_pnl'))} "
                f"hold_pnl={_fmt(row.get('hold_to_resolution_pnl'))} "
                f"avg_edge={_fmt(row.get('average_estimated_edge'))}"
            )
        lines.append("")
    return "\n".join(lines)


def _closed_or_settled_trades(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in conn.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
    ]
    filtered = [
        row for row in rows
        if str(row.get("status") or "").lower() in {"closed", "settled"}
    ]
    filtered.sort(key=lambda row: (_trade_sort_time(row) or "", int(row.get("id") or 0)))
    return filtered


def _trade_sort_time(row: dict[str, Any]) -> str | None:
    for field in ("exit_time", "settled_at", "created_at", "signal_timestamp"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _autopsy_row(recorder: sqlite3.Connection, trade: dict[str, Any]) -> dict[str, Any]:
    market = _market_row(
        recorder,
        run_id=str(trade.get("run_id") or ""),
        market_id=str(trade.get("market_id") or ""),
    )
    outcome, outcome_source = _resolution_outcome(market)
    btc_start = None
    btc_close = None
    btc_move = None
    if outcome is None:
        btc_start, btc_close = _btc_start_close(recorder, trade=trade, market=market)
        if btc_start is not None and btc_close is not None:
            btc_move = btc_close - btc_start
            outcome = "YES" if btc_move > 0 else "NO"
            outcome_source = "btc_price_estimate"
    else:
        btc_start, btc_close = _btc_start_close(recorder, trade=trade, market=market)
        if btc_start is not None and btc_close is not None:
            btc_move = btc_close - btc_start

    direction = _direction(trade.get("signal_direction"))
    adjusted_entry = _float_or_none(trade.get("adjusted_entry_price"))
    stake = _float_or_none(trade.get("stake_usd")) or 0.0
    exit_price = _first_float(
        trade,
        ("adjusted_exit_price", "exit_price", "fixed_horizon_exit_price"),
    )
    fixed_horizon_pnl = _first_float(
        trade,
        ("realized_pnl_usd", "fixed_horizon_pnl"),
    )
    if fixed_horizon_pnl is None and exit_price is not None:
        fixed_horizon_pnl = _pnl_at_price(stake, adjusted_entry, exit_price)
    would_win = bool(direction and outcome and direction == outcome)
    hold_pnl = _hold_to_resolution_pnl(
        stake_usd=stake,
        adjusted_entry_price=adjusted_entry,
        would_win=would_win if outcome is not None and direction is not None else None,
    )
    hold_roi = hold_pnl / stake if hold_pnl is not None and stake else None
    row = {
        "trade_id": trade.get("id"),
        "run_id": trade.get("run_id"),
        "market_id": trade.get("market_id"),
        "status": trade.get("status"),
        "signal_direction": direction,
        "adjusted_entry_price": _round(adjusted_entry),
        "fixed_horizon_exit_price": _round(exit_price),
        "fixed_horizon_pnl": _round(fixed_horizon_pnl),
        "hold_to_resolution_pnl": _round(hold_pnl),
        "hold_to_resolution_roi": _round(hold_roi),
        "would_win_at_resolution": would_win if outcome is not None and direction is not None else None,
        "actual_outcome": outcome,
        "outcome_source": outcome_source,
        "btc_start": _round(btc_start),
        "btc_close": _round(btc_close),
        "btc_move": _round(btc_move),
        "probability_for_direction": _round(_float_or_none(trade.get("probability_for_direction"))),
        "estimated_edge": _round(_float_or_none(trade.get("estimated_edge"))),
        "time_until_resolution": _round(_float_or_none(trade.get("time_until_resolution"))),
        "time_regime": _value_or_unknown(trade.get("time_regime")),
        "btc_trend_regime": _value_or_unknown(trade.get("btc_trend_regime")),
        "liquidity_regime": _value_or_unknown(trade.get("liquidity_regime")),
        "spread_regime": _value_or_unknown(trade.get("spread_regime")),
        "volatility_regime": _value_or_unknown(trade.get("volatility_regime")),
    }
    row["probability_bucket"] = _probability_bucket(row.get("probability_for_direction"))
    return row


def _market_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
) -> dict[str, Any]:
    if not _table_exists(conn, "markets"):
        return {}
    clauses = ["market_id = ?"]
    params: list[Any] = [market_id]
    if run_id:
        clauses.insert(0, "run_id = ?")
        params.insert(0, run_id)
    row = conn.execute(
        f"""
        SELECT *
        FROM markets
        WHERE {' AND '.join(clauses)}
        LIMIT 1
        """,
        params,
    ).fetchone()
    return dict(row) if row is not None else {}


def _resolution_outcome(market: dict[str, Any]) -> tuple[str | None, str | None]:
    if not market:
        return None, None
    resolved = _truthy(market.get("resolved"))
    outcome = _normalise_winning_label(
        winning_outcome=market.get("winning_outcome"),
        winning_asset_id=market.get("winning_asset_id"),
        yes_token_id=market.get("yes_token_id"),
        no_token_id=market.get("no_token_id"),
    )
    if resolved and outcome is not None:
        return outcome, "markets_resolution"
    if outcome is not None:
        return outcome, "markets_winner_fields"
    return None, None


def _normalise_winning_label(
    *,
    winning_outcome: Any,
    winning_asset_id: Any,
    yes_token_id: Any,
    no_token_id: Any,
) -> str | None:
    normalized_outcome = str(winning_outcome or "").strip().upper()
    if normalized_outcome in {"YES", "Y", "UP", "LONG", "TRUE"}:
        return "YES"
    if normalized_outcome in {"NO", "N", "DOWN", "SHORT", "FALSE"}:
        return "NO"
    normalized_asset = str(winning_asset_id or "")
    if normalized_asset and normalized_asset == str(yes_token_id or ""):
        return "YES"
    if normalized_asset and normalized_asset == str(no_token_id or ""):
        return "NO"
    return None


def _btc_start_close(
    conn: sqlite3.Connection,
    *,
    trade: dict[str, Any],
    market: dict[str, Any],
) -> tuple[float | None, float | None]:
    if not _table_exists(conn, "btc_prices"):
        return None, None
    run_id = str(trade.get("run_id") or market.get("run_id") or "")
    start_time = (
        trade.get("market_start_time")
        or market.get("start_time")
        or trade.get("start_time")
    )
    close_time = (
        trade.get("market_close_time")
        or market.get("close_time")
        or trade.get("close_time")
    )
    return (
        _nearest_btc_price(conn, run_id=run_id, timestamp=start_time),
        _nearest_btc_price(conn, run_id=run_id, timestamp=close_time),
    )


def _nearest_btc_price(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    timestamp: Any,
) -> float | None:
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return None
    params: list[Any] = []
    run_filter = ""
    if run_id:
        run_filter = "AND run_id = ?"
        params.append(run_id)
    params.append(parsed.isoformat())
    row = conn.execute(
        f"""
        SELECT price
        FROM btc_prices
        WHERE local_arrival_iso IS NOT NULL
          AND price IS NOT NULL
          {run_filter}
        ORDER BY
          CASE WHEN source = 'polymarket_rtds_chainlink' THEN 0 ELSE 1 END ASC,
          ABS((julianday(local_arrival_iso) - julianday(?)) * 86400.0) ASC,
          id ASC
        LIMIT 1
        """,
        params,
    ).fetchone()
    return _float_or_none(row["price"]) if row is not None else None


def _hold_to_resolution_pnl(
    *,
    stake_usd: float,
    adjusted_entry_price: float | None,
    would_win: bool | None,
) -> float | None:
    if adjusted_entry_price is None or adjusted_entry_price <= 0 or stake_usd <= 0:
        return None
    if would_win is None:
        return None
    if would_win:
        return stake_usd / adjusted_entry_price - stake_usd
    return -stake_usd


def _pnl_at_price(
    stake_usd: float,
    adjusted_entry_price: float | None,
    exit_price: float | None,
) -> float | None:
    if adjusted_entry_price is None or adjusted_entry_price <= 0 or exit_price is None:
        return None
    return stake_usd / adjusted_entry_price * exit_price - stake_usd


def _group_summary(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    summary: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        wins = [
            row for row in group_rows
            if row.get("would_win_at_resolution") is not None
        ]
        summary.append(
            {
                "group": group,
                "trades": len(group_rows),
                "resolution_win_rate": (
                    sum(1 for row in wins if row.get("would_win_at_resolution")) / len(wins)
                    if wins else None
                ),
                "fixed_horizon_pnl": _round(_sum(row.get("fixed_horizon_pnl") for row in group_rows)),
                "hold_to_resolution_pnl": _round(
                    _sum(row.get("hold_to_resolution_pnl") for row in group_rows)
                ),
                "average_hold_to_resolution_roi": _round(
                    _mean(row.get("hold_to_resolution_roi") for row in group_rows)
                ),
                "average_estimated_edge": _round(
                    _mean(row.get("estimated_edge") for row in group_rows)
                ),
                "average_probability_for_direction": _round(
                    _mean(row.get("probability_for_direction") for row in group_rows)
                ),
            }
        )
    summary.sort(
        key=lambda item: (
            -int(item["trades"]),
            str(item["group"]),
        )
    )
    return summary


def _connect_read_only(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(AUTOPSY_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _output_run_id(rows: Sequence[dict[str, Any]], paper_db_path: str) -> str:
    run_ids = {
        str(row.get("run_id"))
        for row in rows
        if row.get("run_id") not in (None, "")
    }
    if len(run_ids) == 1:
        return next(iter(run_ids))
    return Path(paper_db_path).stem


def _probability_bucket(value: Any) -> str:
    probability = _float_or_none(value)
    if probability is None:
        return "unknown"
    if probability >= 1.0:
        return "0.9-1.0"
    if probability <= 0.0:
        return "0.0-0.1"
    index = int(probability * 10)
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def _first_float(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _direction(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"YES", "NO"} else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _sum(values: Sequence[Any]) -> float:
    return sum(value for value in (_float_or_none(item) for item in values) if value is not None)


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _round(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return round(numeric, 10) if numeric is not None else None


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    return "None" if numeric is None else f"{numeric:.4f}"


def _value_or_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
