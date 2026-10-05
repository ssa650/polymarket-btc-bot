from __future__ import annotations

import csv
import glob
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .models import parse_timestamp


DEFAULT_FIXED_HORIZONS_SEC: tuple[int, ...] = (5, 15, 30, 60)
DEFAULT_STOP_LOSS_TAKE_PROFIT: tuple[tuple[float, float], ...] = (
    (-0.20, 0.30),
    (-0.30, 0.50),
)
DEFAULT_TRAILING_POLICIES: tuple[tuple[float, float], ...] = ((0.20, 0.15),)

ORIGINAL_COLUMNS: tuple[str, ...] = (
    "source_paper_db",
    "paper_trade_id",
    "run_id",
    "market_id",
    "signal_timestamp",
    "market_start_time",
    "market_close_time",
    "signal_direction",
    "strategy_id",
    "predicted_probability_yes",
    "probability_for_direction",
    "estimated_edge",
    "time_until_resolution",
    "time_regime",
    "btc_trend_regime",
    "volatility_regime",
    "spread_regime",
    "liquidity_regime",
    "entry_price",
    "adjusted_entry_price",
    "stake_usd",
    "original_status",
    "original_exit_reason",
    "original_pnl_usd",
    "original_roi",
)

SIMULATED_COLUMNS: tuple[str, ...] = (
    "policy_name",
    "simulated_exit_time",
    "simulated_exit_price",
    "simulated_adjusted_exit_price",
    "simulated_pnl_usd",
    "simulated_roi",
    "hit_reason",
    "max_favorable_price",
    "max_adverse_price",
    "max_favorable_roi",
    "max_adverse_roi",
    "price_path_rows",
    "entry_price_used",
    "adjusted_entry_price_used",
    "probability_bucket",
)

META_FEATURE_COLUMNS: tuple[str, ...] = (
    "predicted_probability_yes",
    "probability_for_direction",
    "estimated_edge",
    "model_name",
    "strategy_id",
    "signal_direction",
    "entry_price",
    "adjusted_entry_price",
    "spread_cents_at_entry",
    "entry_latency_sec",
    "entry_price_drift",
    "liquidity_fill_fraction_used",
    "max_fillable_shares",
    "requested_shares",
    "time_until_resolution",
    "time_regime",
    "btc_trend_regime",
    "volatility_regime",
    "spread_regime",
    "liquidity_regime",
    "feature_ready",
    "strict_validation_passed",
    "snapshot_quality_status",
)


class TradeAutopsyError(RuntimeError):
    pass


def build_trade_autopsy_report(
    *,
    recorder_db_path: str,
    paper_db_paths: Sequence[str],
    output_path: str | None,
    output_csv_path: str | None = None,
    status_filter: Sequence[str] | str = ("closed", "settled"),
    strategy_ids: Sequence[str] | None = None,
    market_ids: Sequence[str] | None = None,
    limit: int | None = None,
    fixed_horizons_sec: Sequence[int] = DEFAULT_FIXED_HORIZONS_SEC,
    stop_loss_take_profit: Sequence[tuple[float, float]] = DEFAULT_STOP_LOSS_TAKE_PROFIT,
    near_close_sec: float = 3.0,
    include_skipped: bool = False,
    summary_only: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    recorder = _connect_read_only(recorder_db_path)
    try:
        paper_rows = load_paper_trade_rows(
            paper_db_paths=paper_db_paths,
            status_filter=status_filter,
            strategy_ids=strategy_ids,
            market_ids=market_ids,
            limit=limit,
            include_skipped=include_skipped,
        )
        policies = build_policy_specs(
            fixed_horizons_sec=fixed_horizons_sec,
            stop_loss_take_profit=stop_loss_take_profit,
            near_close_sec=near_close_sec,
        )
        autopsy_rows = replay_trade_rows(
            recorder,
            paper_rows,
            policies=policies,
            near_close_sec=near_close_sec,
        )
    finally:
        recorder.close()

    summary = summarize_autopsy_rows(
        autopsy_rows,
        rows_loaded=len(paper_rows),
        policies_evaluated=len(policies),
        output_path=output_path,
    )
    report = {
        "status": "ok",
        "generated_at": _iso(now or datetime.now(timezone.utc)),
        "recorder_db_path": recorder_db_path,
        "paper_db_paths": list(paper_db_paths),
        "rows_loaded": len(paper_rows),
        "rows_evaluated": len(autopsy_rows),
        "policies_evaluated": len(policies),
        "output_path": output_path,
        "output_csv_path": output_csv_path,
        "summary": summary,
    }
    if not summary_only:
        if output_path:
            _write_parquet(autopsy_rows, Path(output_path))
            report["output_file_size_bytes"] = _file_size(Path(output_path))
        if output_csv_path:
            _write_csv(autopsy_rows, Path(output_csv_path))
            report["output_csv_file_size_bytes"] = _file_size(Path(output_csv_path))
    return report


def load_paper_trade_rows(
    *,
    paper_db_paths: Sequence[str],
    status_filter: Sequence[str] | str,
    strategy_ids: Sequence[str] | None,
    market_ids: Sequence[str] | None,
    limit: int | None,
    include_skipped: bool,
) -> list[dict[str, Any]]:
    statuses = _status_filter_set(status_filter)
    if include_skipped:
        statuses.add("skipped")
    strategy_filter = {str(item) for item in (strategy_ids or []) if str(item).strip()}
    market_filter = {str(item) for item in (market_ids or []) if str(item).strip()}
    rows: list[dict[str, Any]] = []
    for path in paper_db_paths:
        conn = _connect_read_only(path)
        try:
            if not _table_exists(conn, "paper_trades"):
                trade_rows: list[sqlite3.Row] = []
            else:
                trade_rows = conn.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
            for row in trade_rows:
                item = dict(row)
                item["source_paper_db"] = str(path)
                status = str(item.get("status") or "").lower()
                if statuses and status not in statuses:
                    continue
                if strategy_filter and str(item.get("strategy_id") or "") not in strategy_filter:
                    continue
                if market_filter and str(item.get("market_id") or "") not in market_filter:
                    continue
                rows.append(item)
                if limit is not None and len(rows) >= int(limit):
                    return rows
            if "candidate" in statuses and _table_exists(conn, "paper_trade_candidates"):
                for row in conn.execute(
                    "SELECT * FROM paper_trade_candidates ORDER BY id ASC"
                ).fetchall():
                    item = _candidate_row_as_trade(dict(row), source_paper_db=str(path))
                    if strategy_filter and str(item.get("strategy_id") or "") not in strategy_filter:
                        continue
                    if market_filter and str(item.get("market_id") or "") not in market_filter:
                        continue
                    rows.append(item)
                    if limit is not None and len(rows) >= int(limit):
                        return rows
        finally:
            conn.close()
    return rows


def _candidate_row_as_trade(row: dict[str, Any], *, source_paper_db: str) -> dict[str, Any]:
    item = dict(row)
    item["source_paper_db"] = source_paper_db
    item["status"] = "candidate"
    item["id"] = f"candidate_{row.get('id')}"
    item["signal_timestamp"] = row.get("feature_timestamp")
    item["signal_direction"] = row.get("candidate_direction")
    item["entry_price"] = row.get("candidate_entry_price")
    item["adjusted_entry_price"] = row.get("candidate_adjusted_entry_price")
    item["exit_reason"] = row.get("rejection_reason")
    return item


def build_policy_specs(
    *,
    fixed_horizons_sec: Sequence[int],
    stop_loss_take_profit: Sequence[tuple[float, float]],
    near_close_sec: float,
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for seconds in fixed_horizons_sec:
        policies.append(
            {
                "type": "fixed_horizon",
                "name": f"fixed_horizon_{int(seconds)}s",
                "seconds": int(seconds),
            }
        )
    for stop_loss, take_profit in stop_loss_take_profit:
        policies.append(
            {
                "type": "stop_loss_take_profit",
                "name": _stop_loss_take_profit_name(stop_loss, take_profit),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
            }
        )
    for activate_after_profit, trailing_stop in DEFAULT_TRAILING_POLICIES:
        policies.append(
            {
                "type": "trailing_stop",
                "name": _trailing_name(activate_after_profit, trailing_stop),
                "activate_after_profit": float(activate_after_profit),
                "trailing_stop": float(trailing_stop),
            }
        )
    policies.append(
        {
            "type": "near_close",
            "name": f"near_close_{int(float(near_close_sec))}s",
            "near_close_sec": float(near_close_sec),
        }
    )
    return policies


def replay_trade_rows(
    recorder: sqlite3.Connection,
    paper_rows: Sequence[dict[str, Any]],
    *,
    policies: Sequence[dict[str, Any]],
    near_close_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in paper_rows:
        path = load_share_price_path(recorder, trade)
        original = _original_fields(trade)
        for policy in policies:
            result = simulate_policy(
                trade,
                path,
                policy=policy,
                near_close_sec=near_close_sec,
            )
            rows.append({**original, **result})
    return rows


def load_share_price_path(
    recorder: sqlite3.Connection,
    trade: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _table_exists(recorder, "market_snapshots"):
        return []
    direction = _direction(trade.get("signal_direction"))
    if direction is None:
        return []
    run_id = str(trade.get("run_id") or "")
    market_id = str(trade.get("market_id") or "")
    start = _entry_time(trade)
    close = parse_timestamp(trade.get("market_close_time") or trade.get("close_time"))
    if start is None or not market_id:
        return []
    where = ["market_id = ?", "datetime(timestamp) >= datetime(?)"]
    params: list[Any] = [market_id, start.isoformat()]
    if run_id:
        where.insert(0, "run_id = ?")
        params.insert(0, run_id)
    if close is not None:
        where.append("datetime(timestamp) <= datetime(?)")
        params.append(close.isoformat())
    rows = recorder.execute(
        f"""
        SELECT *
        FROM market_snapshots
        WHERE {' AND '.join(where)}
        ORDER BY datetime(timestamp) ASC, timestamp ASC
        """,
        params,
    ).fetchall()
    path: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        price = _exit_price_for_direction(item, direction=direction)
        if price is None:
            continue
        path.append(
            {
                "timestamp": item.get("timestamp"),
                "price": price,
                "raw_price_source": _exit_price_source_for_direction(item, direction=direction),
            }
        )
    return path


def simulate_policy(
    trade: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    policy: dict[str, Any],
    near_close_sec: float,
) -> dict[str, Any]:
    entry = _entry_price(trade, path)
    stake = _float_or_none(trade.get("stake_usd")) or 1.0
    base = {
        "policy_name": policy["name"],
        "price_path_rows": len(path),
        "entry_price_used": _round(entry),
        "adjusted_entry_price_used": _round(entry),
        "probability_bucket": _probability_bucket(trade.get("probability_for_direction")),
    }
    if not path or entry is None or entry <= 0:
        return {
            **base,
            "simulated_exit_time": None,
            "simulated_exit_price": None,
            "simulated_adjusted_exit_price": None,
            "simulated_pnl_usd": None,
            "simulated_roi": None,
            "hit_reason": "no_price_path" if not path else "missing_entry_price",
            "max_favorable_price": None,
            "max_adverse_price": None,
            "max_favorable_roi": None,
            "max_adverse_roi": None,
        }
    selected = _select_exit_point(trade, path, entry=entry, policy=policy, near_close_sec=near_close_sec)
    stats = _path_stats(path, entry=entry)
    exit_price = _float_or_none(selected.get("price"))
    pnl, roi = _pnl_and_roi(stake, entry_price=entry, exit_price=exit_price)
    return {
        **base,
        "simulated_exit_time": selected.get("timestamp"),
        "simulated_exit_price": _round(exit_price),
        "simulated_adjusted_exit_price": _round(exit_price),
        "simulated_pnl_usd": _round(pnl),
        "simulated_roi": _round(roi),
        "hit_reason": selected.get("hit_reason"),
        "max_favorable_price": _round(stats["max_favorable_price"]),
        "max_adverse_price": _round(stats["max_adverse_price"]),
        "max_favorable_roi": _round(stats["max_favorable_roi"]),
        "max_adverse_roi": _round(stats["max_adverse_roi"]),
    }


def _select_exit_point(
    trade: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    entry: float,
    policy: dict[str, Any],
    near_close_sec: float,
) -> dict[str, Any]:
    policy_type = str(policy.get("type") or "")
    if policy_type == "fixed_horizon":
        start = _entry_time(trade)
        target = None if start is None else start + timedelta(seconds=float(policy["seconds"]))
        if target is not None:
            for point in path:
                timestamp = parse_timestamp(point.get("timestamp"))
                if timestamp is not None and timestamp >= target:
                    return {**point, "hit_reason": "fixed_horizon"}
        return {**path[-1], "hit_reason": "fixed_horizon_last_available"}
    if policy_type == "stop_loss_take_profit":
        stop_loss = float(policy["stop_loss"])
        take_profit = float(policy["take_profit"])
        for point in path:
            roi = _roi_for_price(entry, _float_or_none(point.get("price")))
            if roi is None:
                continue
            if roi <= stop_loss:
                return {**point, "hit_reason": "stop_loss"}
            if roi >= take_profit:
                return {**point, "hit_reason": "take_profit"}
        return {**_near_close_point(trade, path, near_close_sec=near_close_sec), "hit_reason": "near_close"}
    if policy_type == "trailing_stop":
        activate = float(policy["activate_after_profit"])
        trailing = float(policy["trailing_stop"])
        max_roi: float | None = None
        active = False
        for point in path:
            roi = _roi_for_price(entry, _float_or_none(point.get("price")))
            if roi is None:
                continue
            max_roi = roi if max_roi is None else max(max_roi, roi)
            if max_roi >= activate:
                active = True
            if active and roi <= max_roi - trailing:
                return {**point, "hit_reason": "trailing_stop"}
        return {**_near_close_point(trade, path, near_close_sec=near_close_sec), "hit_reason": "near_close"}
    return {**_near_close_point(trade, path, near_close_sec=near_close_sec), "hit_reason": "near_close"}


def _near_close_point(
    trade: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    near_close_sec: float,
) -> dict[str, Any]:
    close = parse_timestamp(trade.get("market_close_time") or trade.get("close_time"))
    if close is None:
        return dict(path[-1])
    target = close - timedelta(seconds=max(0.0, float(near_close_sec)))
    selected = dict(path[0])
    for point in path:
        timestamp = parse_timestamp(point.get("timestamp"))
        if timestamp is not None and timestamp <= target:
            selected = dict(point)
        elif timestamp is not None and timestamp > target:
            break
    return selected


def summarize_autopsy_rows(
    rows: Sequence[dict[str, Any]],
    *,
    rows_loaded: int,
    policies_evaluated: int,
    output_path: str | None,
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row.get("policy_name") or "unknown")].append(row)
    return {
        "rows_loaded": rows_loaded,
        "rows_evaluated": len(rows),
        "policies_evaluated": policies_evaluated,
        "output_path": output_path,
        "per_policy": {
            policy: _policy_summary(policy_rows)
            for policy, policy_rows in sorted(by_policy.items())
        },
        "top_loss_cohorts": {
            field: _top_loss_cohorts(rows, field)
            for field in (
                "time_regime",
                "btc_trend_regime",
                "liquidity_regime",
                "probability_bucket",
                "strategy_id",
            )
        },
    }


def export_meta_trade_dataset(
    *,
    autopsy_input_path: str | None,
    output_path: str,
    label_policy: str = "stop_loss_20_take_profit_30",
    label_column: str = "label_positive_roi",
    roi_threshold: float = 0.0,
    output_csv_path: str | None = None,
    recorder_db_path: str | None = None,
    paper_db_paths: Sequence[str] | None = None,
    status_filter: Sequence[str] | str = ("closed", "settled"),
    fixed_horizons_sec: Sequence[int] = DEFAULT_FIXED_HORIZONS_SEC,
    stop_loss_take_profit: Sequence[tuple[float, float]] = DEFAULT_STOP_LOSS_TAKE_PROFIT,
    near_close_sec: float = 3.0,
) -> dict[str, Any]:
    if autopsy_input_path:
        autopsy_rows = _read_parquet(Path(autopsy_input_path))
    else:
        if not recorder_db_path or not paper_db_paths:
            raise TradeAutopsyError(
                "export-meta-trade-dataset requires --autopsy-input or --recorder-db/--paper-db"
            )
        recorder = _connect_read_only(recorder_db_path)
        try:
            paper_rows = load_paper_trade_rows(
                paper_db_paths=paper_db_paths,
                status_filter=status_filter,
                strategy_ids=None,
                market_ids=None,
                limit=None,
                include_skipped=False,
            )
            autopsy_rows = replay_trade_rows(
                recorder,
                paper_rows,
                policies=build_policy_specs(
                    fixed_horizons_sec=fixed_horizons_sec,
                    stop_loss_take_profit=stop_loss_take_profit,
                    near_close_sec=near_close_sec,
                ),
                near_close_sec=near_close_sec,
            )
        finally:
            recorder.close()
    selected = [row for row in autopsy_rows if str(row.get("policy_name")) == str(label_policy)]
    meta_rows = [
        _meta_trade_row(row, label_column=label_column, roi_threshold=roi_threshold)
        for row in selected
    ]
    output = Path(output_path)
    _write_parquet(meta_rows, output)
    if output_csv_path:
        _write_csv(meta_rows, Path(output_csv_path))
    return {
        "status": "ok",
        "autopsy_input_path": autopsy_input_path,
        "output_path": str(output),
        "output_file_size_bytes": _file_size(output),
        "output_csv_path": output_csv_path,
        "rows_exported": len(meta_rows),
        "label_policy": label_policy,
        "label_column": label_column,
        "roi_threshold": float(roi_threshold),
    }


def _meta_trade_row(
    row: dict[str, Any],
    *,
    label_column: str,
    roi_threshold: float,
) -> dict[str, Any]:
    roi = _float_or_none(row.get("simulated_roi"))
    result = {
        column: row.get(column)
        for column in (
            "source_paper_db",
            "paper_trade_id",
            "run_id",
            "market_id",
            "signal_timestamp",
            *META_FEATURE_COLUMNS,
            "policy_name",
            "simulated_pnl_usd",
            "simulated_roi",
            "hit_reason",
            "max_favorable_roi",
            "max_adverse_roi",
        )
    }
    result["label_positive_roi"] = None if roi is None else int(roi > 0)
    result["label_roi_above_threshold"] = None if roi is None else int(roi > float(roi_threshold))
    result["label_take_profit_before_stop_loss"] = int(str(row.get("hit_reason")) == "take_profit")
    result["label_column"] = label_column
    result["label_value"] = result.get(label_column)
    return result


def resolve_paper_db_paths(
    *,
    paper_db_values: Sequence[str] | str | None,
    paper_db_globs: Sequence[str] | str | None,
) -> list[str]:
    paths: list[str] = []
    for value in _as_list(paper_db_values):
        if value:
            paths.append(value)
    for pattern in _as_list(paper_db_globs):
        if not pattern:
            continue
        paths.extend(sorted(glob.glob(pattern)))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def parse_fixed_horizons(value: str | Sequence[int] | None) -> list[int]:
    if value is None:
        return list(DEFAULT_FIXED_HORIZONS_SEC)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [int(float(part)) for part in parts]
    return [int(item) for item in value]


def parse_stop_loss_take_profit(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return list(DEFAULT_STOP_LOSS_TAKE_PROFIT)
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        pairs: list[tuple[float, float]] = []
        for item in parsed:
            if isinstance(item, dict):
                pairs.append((float(item["stop_loss"]), float(item["take_profit"])))
            else:
                pairs.append((float(item[0]), float(item[1])))
        return pairs
    pairs = []
    for part in text.split(","):
        if not part.strip():
            continue
        left, right = part.split(":", 1)
        pairs.append((float(left), float(right)))
    return pairs


def _policy_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if (_float_or_none(row.get("simulated_pnl_usd")) or 0.0) > 0]
    losses = [row for row in rows if (_float_or_none(row.get("simulated_pnl_usd")) or 0.0) < 0]
    return {
        "n": len(rows),
        "total_simulated_pnl": _round(_sum(row.get("simulated_pnl_usd") for row in rows)),
        "avg_simulated_roi": _round(_mean(row.get("simulated_roi") for row in rows)),
        "win_rate": len(wins) / len(rows) if rows else None,
        "avg_win": _round(_mean(row.get("simulated_pnl_usd") for row in wins)),
        "avg_loss": _round(_mean(row.get("simulated_pnl_usd") for row in losses)),
    }


def _top_loss_cohorts(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get(field) or "unknown"), str(row.get("policy_name") or "unknown"))].append(row)
    cohort_rows = [
        {
            "group": group,
            "policy_name": policy,
            "n": len(group_rows),
            "total_simulated_pnl": _round(_sum(row.get("simulated_pnl_usd") for row in group_rows)),
            "avg_simulated_roi": _round(_mean(row.get("simulated_roi") for row in group_rows)),
        }
        for (group, policy), group_rows in groups.items()
    ]
    cohort_rows.sort(key=lambda item: (float(item["total_simulated_pnl"] or 0.0), -int(item["n"])))
    return cohort_rows[:limit]


def _path_stats(path: Sequence[dict[str, Any]], *, entry: float) -> dict[str, float | None]:
    prices = [_float_or_none(point.get("price")) for point in path]
    numeric = [price for price in prices if price is not None]
    if not numeric:
        return {
            "max_favorable_price": None,
            "max_adverse_price": None,
            "max_favorable_roi": None,
            "max_adverse_roi": None,
        }
    favorable = max(numeric)
    adverse = min(numeric)
    return {
        "max_favorable_price": favorable,
        "max_adverse_price": adverse,
        "max_favorable_roi": _roi_for_price(entry, favorable),
        "max_adverse_roi": _roi_for_price(entry, adverse),
    }


def _original_fields(trade: dict[str, Any]) -> dict[str, Any]:
    original_pnl = _first_float(trade, ("realized_pnl_usd", "pnl_usd"))
    original_roi = _first_float(trade, ("realized_roi", "roi"))
    return {
        "source_paper_db": trade.get("source_paper_db"),
        "paper_trade_id": trade.get("id"),
        "run_id": trade.get("run_id"),
        "market_id": trade.get("market_id"),
        "signal_timestamp": trade.get("signal_timestamp") or trade.get("created_at"),
        "market_start_time": trade.get("market_start_time"),
        "market_close_time": trade.get("market_close_time"),
        "signal_direction": _direction(trade.get("signal_direction")),
        "strategy_id": trade.get("strategy_id"),
        "model_name": trade.get("model_name"),
        "predicted_probability_yes": _round(_float_or_none(trade.get("predicted_probability_yes"))),
        "probability_for_direction": _round(_float_or_none(trade.get("probability_for_direction"))),
        "estimated_edge": _round(_float_or_none(trade.get("estimated_edge"))),
        "time_until_resolution": _round(_float_or_none(trade.get("time_until_resolution"))),
        "time_regime": _value_or_unknown(trade.get("time_regime")),
        "btc_trend_regime": _value_or_unknown(trade.get("btc_trend_regime")),
        "volatility_regime": _value_or_unknown(trade.get("volatility_regime")),
        "spread_regime": _value_or_unknown(trade.get("spread_regime")),
        "liquidity_regime": _value_or_unknown(trade.get("liquidity_regime")),
        "entry_price": _round(_float_or_none(trade.get("entry_price"))),
        "adjusted_entry_price": _round(_float_or_none(trade.get("adjusted_entry_price"))),
        "stake_usd": _round(_float_or_none(trade.get("stake_usd"))),
        "original_status": trade.get("status"),
        "original_exit_reason": trade.get("exit_reason"),
        "original_pnl_usd": _round(original_pnl),
        "original_roi": _round(original_roi),
        "spread_cents_at_entry": _round(_float_or_none(trade.get("spread_cents_at_entry"))),
        "entry_latency_sec": _round(_float_or_none(trade.get("entry_latency_sec"))),
        "entry_price_drift": _round(_float_or_none(trade.get("entry_price_drift"))),
        "liquidity_fill_fraction_used": _round(_float_or_none(trade.get("liquidity_fill_fraction_used"))),
        "max_fillable_shares": _round(_float_or_none(trade.get("max_fillable_shares"))),
        "requested_shares": _round(_float_or_none(trade.get("requested_shares"))),
        "feature_ready": trade.get("feature_ready"),
        "strict_validation_passed": trade.get("strict_validation_passed"),
        "snapshot_quality_status": trade.get("snapshot_quality_status"),
    }


def _exit_price_for_direction(row: dict[str, Any], *, direction: str) -> float | None:
    if direction == "YES":
        fields = ("best_bid_yes", "mid_price_yes", "last_trade_price")
    else:
        fields = ("best_bid_no", "mid_price_no", "last_trade_price")
    return _first_float(row, fields)


def _exit_price_source_for_direction(row: dict[str, Any], *, direction: str) -> str | None:
    fields = (
        ("best_bid_yes", "mid_price_yes", "last_trade_price")
        if direction == "YES"
        else ("best_bid_no", "mid_price_no", "last_trade_price")
    )
    for field in fields:
        if _float_or_none(row.get(field)) is not None:
            return field
    return None


def _entry_price(trade: dict[str, Any], path: Sequence[dict[str, Any]]) -> float | None:
    price = _first_float(trade, ("adjusted_entry_price", "entry_price"))
    if price is not None:
        return price
    return _float_or_none(path[0].get("price")) if path else None


def _entry_time(trade: dict[str, Any]) -> datetime | None:
    return parse_timestamp(
        trade.get("signal_timestamp")
        or trade.get("created_at")
        or trade.get("entry_time")
    )


def _pnl_and_roi(
    stake_usd: float,
    *,
    entry_price: float,
    exit_price: float | None,
) -> tuple[float | None, float | None]:
    if exit_price is None or entry_price <= 0:
        return None, None
    pnl = stake_usd / entry_price * exit_price - stake_usd
    roi = pnl / stake_usd if stake_usd else None
    return pnl, roi


def _roi_for_price(entry: float, price: float | None) -> float | None:
    if price is None or entry <= 0:
        return None
    return price / entry - 1.0


def _connect_read_only(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(path)
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


def _write_parquet(rows: Sequence[dict[str, Any]], path: Path) -> None:
    pa, pq = _load_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(rows)
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({})
    pq.write_table(table, path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    _pa, pq = _load_pyarrow()
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(rows)
    fieldnames = list(normalized[0].keys()) if normalized else list(ORIGINAL_COLUMNS + SIMULATED_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


def _normalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return [{column: _scalar(row.get(column)) for column in columns} for row in rows]


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Trade autopsy exports require pyarrow. Install project dependencies first."
        ) from exc
    return pa, pq


def _status_filter_set(value: Sequence[str] | str) -> set[str]:
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _stop_loss_take_profit_name(stop_loss: float, take_profit: float) -> str:
    return (
        f"stop_loss_{int(abs(float(stop_loss)) * 100)}"
        f"_take_profit_{int(abs(float(take_profit)) * 100)}"
    )


def _trailing_name(activate_after_profit: float, trailing_stop: float) -> str:
    return (
        f"trailing_stop_{int(abs(float(activate_after_profit)) * 100)}"
        f"_{int(abs(float(trailing_stop)) * 100)}"
    )


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


def _direction(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"YES", "NO"} else None


def _first_float(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
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


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _value_or_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
