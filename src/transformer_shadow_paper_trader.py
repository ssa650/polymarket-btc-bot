from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .baseline_paper_trader import connect_paper_output_db, ensure_paper_schema
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none
from .transformer_live_inference import (
    InferenceFn,
    TransformerLiveArtifacts,
    load_transformer_live_artifacts,
    transformer_live_inference_heartbeat,
)


class TransformerShadowPaperTraderError(RuntimeError):
    pass


_TRANSFORMER_PAPER_TRADE_SQL = (
    "("
    "COALESCE(model_name, '') IN ('transformer_predictions', 'transformer_sequence') "
    "OR COALESCE(strategy_id, '') LIKE 'transformer_%'"
    ")"
)


def run_transformer_shadow_paper_strategy(
    *,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    output_db_path: str,
    scaler_stats_path: str | None = None,
    training_config_path: str | None = None,
    run_id: str | None = None,
    sequence_length: int = 120,
    sequence_row_policy: str = "latest_clean_rows",
    poll_sec: float = 1.0,
    max_feature_age_sec: float = 5.0,
    probability_threshold: float = 0.90,
    min_time_until_resolution_sec: float = 30.0,
    max_time_until_resolution_sec: float = 120.0,
    fixed_horizons_sec: Sequence[int] = (15, 30, 60),
    live_trading_enabled: bool = False,
    max_iterations: int | None = None,
    emit_logs: bool = True,
    inference_fn: InferenceFn | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if live_trading_enabled:
        raise TransformerShadowPaperTraderError(
            "live_trading_enabled is not supported by transformer shadow paper strategy"
        )
    artifacts = load_transformer_live_artifacts(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        scaler_stats_path=scaler_stats_path,
        training_config_path=training_config_path,
    )
    effective_run_id = run_id or f"transformer_shadow_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    recorder = _open_readonly_sqlite(recorder_db_path)
    output = connect_paper_output_db(output_db_path)
    ensure_paper_schema(output)
    try:
        iterations = 0
        opened = 0
        skipped = 0
        last_heartbeat: dict[str, Any] = {}
        while True:
            heartbeat = transformer_live_inference_heartbeat(
                recorder,
                artifacts=artifacts,
                inference_fn=inference_fn,
                sequence_length=int(sequence_length),
                max_feature_age_sec=float(max_feature_age_sec),
                allow_gap_affected=False,
                allow_not_ready_sequence_rows=False,
                min_ready_ratio=1.0,
                sequence_row_policy=sequence_row_policy,
                diagnostics=True,
                now=now,
            )
            decisions = _evaluate_shadow_heartbeat(
                output,
                heartbeat,
                artifacts=artifacts,
                run_id=effective_run_id,
                probability_threshold=float(probability_threshold),
                min_time_until_resolution_sec=float(min_time_until_resolution_sec),
                max_time_until_resolution_sec=float(max_time_until_resolution_sec),
                fixed_horizons_sec=fixed_horizons_sec,
            )
            opened += decisions["opened"]
            skipped += decisions["skipped"]
            last_heartbeat = {
                **heartbeat,
                "shadow_trades_opened_this_iteration": decisions["opened"],
                "shadow_skips_this_iteration": decisions["skipped"],
                "shadow_skip_reason": decisions.get("skip_reason"),
                "output_db_path": output_db_path,
                "run_id": effective_run_id,
            }
            _emit(emit_logs, "transformer_shadow_paper_heartbeat", **last_heartbeat)
            iterations += 1
            if max_iterations is not None and iterations >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iterations,
                    "trades_opened": opened,
                    "skipped": skipped,
                    "last_heartbeat": last_heartbeat,
                    "output_db_path": output_db_path,
                    "run_id": effective_run_id,
                    "live_trading_enabled": False,
                }
            time.sleep(max(0.0, float(poll_sec)))
    finally:
        output.close()
        recorder.close()


def run_transformer_prediction_paper_strategy(
    *,
    recorder_db_path: str,
    prediction_db_path: str,
    config_path: str,
    output_db_path: str,
    run_id: str | None = None,
    poll_sec: float = 1.0,
    live_trading_enabled: bool = False,
    max_iterations: int | None = None,
    emit_logs: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    if live_trading_enabled:
        raise TransformerShadowPaperTraderError(
            "live_trading_enabled is not supported by transformer prediction paper strategy"
        )
    config = _load_prediction_strategy_config(config_path)
    effective_run_id = run_id or f"transformer_predictions_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    recorder = _open_readonly_sqlite(recorder_db_path)
    predictions = _open_readonly_sqlite(prediction_db_path)
    output = connect_paper_output_db(output_db_path)
    ensure_paper_schema(output)
    try:
        iterations = 0
        total_opened = 0
        total_candidates = 0
        last_report: dict[str, Any] = {}
        while True:
            poll_report = _process_transformer_prediction_rows(
                recorder=recorder,
                predictions=predictions,
                output=output,
                config=config,
                run_id=effective_run_id,
            )
            total_opened += int(poll_report.get("trades_opened") or 0)
            total_candidates += int(poll_report.get("candidates_logged") or 0)
            last_report = {
                **poll_report,
                "output_db_path": output_db_path,
                "prediction_db_path": prediction_db_path,
                "recorder_db_path": recorder_db_path,
                "run_id": effective_run_id,
            }
            _emit(
                emit_logs,
                "transformer_prediction_paper_heartbeat",
                **last_report,
            )
            iterations += 1
            if max_iterations is not None and iterations >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iterations,
                    "trades_opened": total_opened,
                    "candidates_logged": total_candidates,
                    "last_poll": last_report,
                    "output_db_path": output_db_path,
                    "prediction_db_path": prediction_db_path,
                    "run_id": effective_run_id,
                    "live_trading_enabled": False,
                }
            time.sleep(max(0.0, float(poll_sec)))
    finally:
        output.close()
        predictions.close()
        recorder.close()


def backfill_transformer_prediction_paper_pnl(
    *,
    recorder_db_path: str,
    paper_db_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    recorder = _open_readonly_sqlite(recorder_db_path)
    output = connect_paper_output_db(paper_db_path)
    ensure_paper_schema(output)
    try:
        duplicate_groups = _transformer_duplicate_market_groups(output)
        rows = [
            dict(row)
            for row in output.execute(
                """
                SELECT *
                FROM paper_trades
                WHERE _is_transformer_paper_trade_sql
                  AND COALESCE(status, '') IN ('open', 'closed')
                  AND COALESCE(signal_direction, '') IN ('YES', 'NO')
                  AND (
                    pnl_usd IS NULL
                    OR roi IS NULL
                    OR realized_pnl_usd IS NULL
                    OR realized_roi IS NULL
                    OR exit_price IS NULL
                    OR adjusted_exit_price IS NULL
                    OR exit_time IS NULL
                  )
                ORDER BY id ASC
                """
                .replace("_is_transformer_paper_trade_sql", _TRANSFORMER_PAPER_TRADE_SQL)
            ).fetchall()
        ]
        updated = 0
        aliases_filled = 0
        no_exit_snapshot = 0
        invalid_entry_price = 0
        for row in rows:
            result = _backfill_transformer_prediction_trade(recorder, output, row, dry_run=dry_run)
            if result == "updated":
                updated += 1
            elif result == "aliases_filled":
                updated += 1
                aliases_filled += 1
            elif result == "no_exit_snapshot":
                no_exit_snapshot += 1
            elif result == "invalid_entry_price":
                invalid_entry_price += 1
        return {
            "status": "ok",
            "paper_db_path": paper_db_path,
            "recorder_db_path": recorder_db_path,
            "dry_run": bool(dry_run),
            "rows_seen": len(rows),
            "rows_updated": updated,
            "aliases_filled": aliases_filled,
            "no_exit_snapshot": no_exit_snapshot,
            "invalid_entry_price": invalid_entry_price,
            "duplicate_strategy_market_group_count": len(duplicate_groups),
            "duplicate_strategy_market_groups": duplicate_groups,
        }
    finally:
        output.close()
        recorder.close()


def _evaluate_shadow_heartbeat(
    output: sqlite3.Connection,
    heartbeat: dict[str, Any],
    *,
    artifacts: TransformerLiveArtifacts,
    run_id: str,
    probability_threshold: float,
    min_time_until_resolution_sec: float,
    max_time_until_resolution_sec: float,
    fixed_horizons_sec: Sequence[int],
) -> dict[str, Any]:
    probability = _float_or_none(heartbeat.get("probability_yes"))
    time_until = _float_or_none(heartbeat.get("time_until_resolution"))
    if not heartbeat.get("transformer_sequence_ready") or probability is None:
        return {"opened": 0, "skipped": 0, "skip_reason": heartbeat.get("reason") or "not_ready"}
    if probability < probability_threshold:
        _insert_shadow_skip(
            output,
            heartbeat,
            artifacts=artifacts,
            run_id=run_id,
            probability_threshold=probability_threshold,
            skip_reason="threshold",
        )
        return {"opened": 0, "skipped": 1, "skip_reason": "threshold"}
    if time_until is None or time_until < min_time_until_resolution_sec or time_until > max_time_until_resolution_sec:
        _insert_shadow_skip(
            output,
            heartbeat,
            artifacts=artifacts,
            run_id=run_id,
            probability_threshold=probability_threshold,
            skip_reason="time_window",
        )
        return {"opened": 0, "skipped": 1, "skip_reason": "time_window"}
    entry_price = _entry_price_yes(heartbeat)
    if entry_price is None or entry_price <= 0 or entry_price >= 1:
        _insert_shadow_skip(
            output,
            heartbeat,
            artifacts=artifacts,
            run_id=run_id,
            probability_threshold=probability_threshold,
            skip_reason="missing_entry_price",
        )
        return {"opened": 0, "skipped": 1, "skip_reason": "missing_entry_price"}
    opened = 0
    for horizon in fixed_horizons_sec:
        if _insert_shadow_trade(
            output,
            heartbeat,
            artifacts=artifacts,
            run_id=run_id,
            probability_threshold=probability_threshold,
            fixed_horizon_sec=int(horizon),
            entry_price=float(entry_price),
        ):
            opened += 1
    return {"opened": opened, "skipped": 0, "skip_reason": None}


def _process_transformer_prediction_rows(
    *,
    recorder: sqlite3.Connection,
    predictions: sqlite3.Connection,
    output: sqlite3.Connection,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    rows = _load_ready_prediction_rows(predictions)
    strategies = list(config.get("strategies") or [])
    opened = 0
    candidates_logged = 0
    skipped = 0
    latest_ts = None
    rejection_counts: dict[str, int] = {}
    for prediction in rows:
        latest_ts = prediction.get("latest_feature_timestamp") or prediction.get("timestamp") or latest_ts
        context = _prediction_context(recorder, prediction)
        for strategy in strategies:
            for direction in _strategy_directions(strategy):
                decision = _evaluate_prediction_strategy(
                    recorder,
                    output,
                    context,
                    strategy,
                    direction=direction,
                    run_id=run_id,
                )
                if decision.get("candidate_logged"):
                    candidates_logged += 1
                if decision.get("opened"):
                    opened += 1
                elif decision.get("rejection_reason"):
                    skipped += 1
                    reason = str(decision["rejection_reason"])
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return {
        "timestamp": to_iso(datetime.now(timezone.utc)),
        "prediction_rows_seen": len(rows),
        "strategy_count": len(strategies),
        "candidates_logged": candidates_logged,
        "trades_opened": opened,
        "skips_logged": skipped,
        "latest_feature_timestamp": latest_ts,
        "top_rejection_reasons": dict(
            sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ),
    }


def _evaluate_prediction_strategy(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    row: dict[str, Any],
    strategy: dict[str, Any],
    *,
    direction: str,
    run_id: str,
) -> dict[str, Any]:
    probability_yes = _float_or_none(row.get("probability_yes"))
    if probability_yes is None:
        return {"candidate_logged": False, "opened": False, "rejection_reason": "missing_probability"}
    probability_for_direction = probability_yes if direction == "YES" else 1.0 - probability_yes
    threshold = _threshold_for_direction(strategy, direction=direction)
    entry_price = _entry_price(row, direction)
    adjusted_entry = _adjusted_entry_price(entry_price, strategy)
    estimated_edge = (
        probability_for_direction - adjusted_entry
        if adjusted_entry is not None
        else None
    )
    rejection_reason = _prediction_rejection_reason(
        row,
        strategy,
        direction=direction,
        threshold=threshold,
        probability_for_direction=probability_for_direction,
        adjusted_entry=adjusted_entry,
        estimated_edge=estimated_edge,
    )
    if rejection_reason is not None:
        candidate_id = _insert_prediction_candidate(
            output,
            row,
            strategy,
            direction=direction,
            run_id=run_id,
            decision="SKIP",
            rejection_reason=rejection_reason,
            threshold=threshold,
            entry_price=entry_price,
            adjusted_entry_price=adjusted_entry,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            linked_trade_id=None,
        )
        return {
            "candidate_logged": candidate_id is not None,
            "opened": False,
            "rejection_reason": rejection_reason,
        }
    if _one_trade_per_market_enabled(strategy) and _has_existing_prediction_market_trade(
        output,
        strategy_id=str(strategy.get("strategy_id") or ""),
        market_id=str(row.get("market_id") or ""),
    ):
        candidate_id = _insert_prediction_candidate(
            output,
            row,
            strategy,
            direction=direction,
            run_id=run_id,
            decision="SKIP",
            rejection_reason="one_trade_per_market",
            threshold=threshold,
            entry_price=entry_price,
            adjusted_entry_price=adjusted_entry,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            linked_trade_id=None,
        )
        return {
            "candidate_logged": candidate_id is not None,
            "opened": False,
            "rejection_reason": "one_trade_per_market",
        }
    trade_id = _insert_prediction_trade(
        output,
        _row_with_exit_context(recorder, row, strategy),
        strategy,
        direction=direction,
        run_id=run_id,
        threshold=threshold,
        entry_price=float(entry_price),
        adjusted_entry_price=float(adjusted_entry),
        probability_for_direction=float(probability_for_direction),
        estimated_edge=float(estimated_edge),
    )
    candidate_id = _insert_prediction_candidate(
        output,
        row,
        strategy,
        direction=direction,
        run_id=run_id,
        decision="TRADE",
        rejection_reason=None,
        threshold=threshold,
        entry_price=entry_price,
        adjusted_entry_price=adjusted_entry,
        probability_for_direction=probability_for_direction,
        estimated_edge=estimated_edge,
        linked_trade_id=trade_id,
    )
    return {
        "candidate_logged": candidate_id is not None,
        "opened": trade_id is not None,
        "rejection_reason": None,
    }


def _load_prediction_strategy_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TransformerShadowPaperTraderError(f"config could not be read: {exc}") from exc
    if _contains_live_trading_enabled(payload):
        raise TransformerShadowPaperTraderError(
            "live_trading_enabled is not supported by transformer prediction paper strategy"
        )
    strategies = [
        strategy
        for strategy in list(payload.get("strategies") or [])
        if isinstance(strategy, dict) and strategy.get("enabled", True)
    ]
    if not strategies:
        raise TransformerShadowPaperTraderError("config must contain at least one enabled strategy")
    return {**payload, "strategies": strategies, "config_path": str(path)}


def _contains_live_trading_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("live_trading_enabled") is True:
            return True
        return any(_contains_live_trading_enabled(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_live_trading_enabled(item) for item in value)
    return False


def _load_ready_prediction_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "transformer_predictions"):
        return []
    rows = conn.execute(
        """
        SELECT *
        FROM transformer_predictions
        WHERE transformer_sequence_ready = 1
          AND probability_yes IS NOT NULL
        ORDER BY datetime(COALESCE(latest_feature_timestamp, feature_timestamp, timestamp, created_at)) ASC,
                 id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _prediction_context(
    recorder: sqlite3.Connection,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    row = dict(prediction)
    row["timestamp"] = _prediction_feature_timestamp(row)
    market = _market_row(recorder, run_id=str(row.get("run_id") or ""), market_id=str(row.get("market_id") or ""))
    for key, value in market.items():
        if key in {"run_id", "market_id"}:
            continue
        if key == "start_time":
            row.setdefault("market_start_time", value)
        elif key == "close_time":
            row.setdefault("market_close_time", value)
        else:
            row.setdefault(key, value)
    snapshot = _snapshot_at_or_after(
        recorder,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        timestamp=str(row.get("timestamp") or ""),
    )
    for key, value in snapshot.items():
        if key in {"run_id", "market_id", "timestamp"}:
            continue
        if row.get(key) in (None, ""):
            row[key] = value
    if row.get("time_until_resolution") in (None, ""):
        feature_time = parse_timestamp(row.get("timestamp"))
        close_time = parse_timestamp(row.get("market_close_time"))
        if feature_time is not None and close_time is not None:
            row["time_until_resolution"] = (close_time - feature_time).total_seconds()
    return row


def _market_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
) -> dict[str, Any]:
    if not _table_exists(conn, "markets") or not market_id:
        return {}
    where = ["market_id = ?"]
    params: list[Any] = [market_id]
    if run_id:
        where.insert(0, "run_id = ?")
        params.insert(0, run_id)
    row = conn.execute(
        f"SELECT * FROM markets WHERE {' AND '.join(where)} LIMIT 1",
        params,
    ).fetchone()
    return dict(row) if row is not None else {}


def _snapshot_at_or_after(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    timestamp: str,
) -> dict[str, Any]:
    if not _table_exists(conn, "market_snapshots") or not market_id or not timestamp:
        return {}
    where = ["market_id = ?", "datetime(timestamp) >= datetime(?)"]
    params: list[Any] = [market_id, timestamp]
    if run_id:
        where.insert(0, "run_id = ?")
        params.insert(0, run_id)
    query = f"""
        SELECT *
        FROM market_snapshots
        WHERE {{where}}
        ORDER BY datetime(timestamp) ASC, timestamp ASC
        LIMIT 1
        """
    row = conn.execute(
        query.format(where=" AND ".join(where)),
        params,
    ).fetchone()
    if row is None and run_id:
        row = conn.execute(
            query.format(where="market_id = ? AND datetime(timestamp) >= datetime(?)"),
            [market_id, timestamp],
        ).fetchone()
    return dict(row) if row is not None else {}


def _snapshot_at_or_after_exact_or_fallback(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    timestamp: str,
) -> dict[str, Any]:
    if not _table_exists(conn, "market_snapshots") or not market_id or not timestamp:
        return {}
    return _snapshot_at_or_after(
        conn,
        run_id=run_id,
        market_id=market_id,
        timestamp=timestamp,
    )


def _snapshot_at_or_after_target(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    horizon_sec: float,
) -> dict[str, Any]:
    signal_time = parse_timestamp(row.get("timestamp") or row.get("signal_timestamp"))
    if signal_time is None:
        return {}
    target = signal_time + timedelta(seconds=float(horizon_sec))
    return _snapshot_at_or_after_exact_or_fallback(
        conn,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        timestamp=target.isoformat(),
    )

def _row_with_exit_context(
    recorder: sqlite3.Connection,
    row: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    horizon = _float_or_none(strategy.get("fixed_horizon_exit_sec")) or 15.0
    snapshot = _snapshot_at_or_after_target(recorder, row, horizon_sec=horizon)
    if not snapshot:
        return dict(row)
    result = dict(row)
    result["exit_timestamp"] = snapshot.get("timestamp")
    result["exit_price_yes"] = _first_price(snapshot, ("best_bid_yes", "mid_price_yes", "last_trade_price"))
    result["exit_price_no"] = _first_price(snapshot, ("best_bid_no", "mid_price_no", "last_trade_price"))
    return result


def _strategy_directions(strategy: dict[str, Any]) -> list[str]:
    mode = str(strategy.get("direction_mode") or "").upper()
    if mode == "YES_ONLY":
        return ["YES"]
    if mode == "NO_ONLY":
        return ["NO"]
    if mode == "BOTH":
        return ["YES", "NO"]
    if mode == "FADE_YES_WITH_NO":
        return ["NO"]
    return []


def _threshold_for_direction(strategy: dict[str, Any], *, direction: str) -> float:
    if _is_fade_yes_with_no_strategy(strategy):
        threshold = _float_or_none(strategy.get("fade_yes_threshold"))
        return 1.0 if threshold is None else threshold
    shared = _float_or_none(strategy.get("probability_threshold"))
    if shared is not None:
        return shared
    if direction == "YES":
        threshold = _float_or_none(strategy.get("long_threshold"))
        return 1.0 if threshold is None else threshold
    short_threshold = _float_or_none(strategy.get("short_threshold"))
    if short_threshold is None:
        return 1.0
    if 0.0 <= short_threshold <= 1.0:
        return 1.0 - short_threshold
    return short_threshold


def _is_fade_yes_with_no_strategy(strategy: dict[str, Any]) -> bool:
    return str(strategy.get("direction_mode") or "").upper() == "FADE_YES_WITH_NO"


def _prediction_rejection_reason(
    row: dict[str, Any],
    strategy: dict[str, Any],
    *,
    direction: str,
    threshold: float,
    probability_for_direction: float,
    adjusted_entry: float | None,
    estimated_edge: float | None,
) -> str | None:
    time_until = _float_or_none(row.get("time_until_resolution"))
    min_time = _float_or_none(strategy.get("min_time_until_resolution_sec")) or 0.0
    max_time = _float_or_none(strategy.get("max_time_until_resolution_sec"))
    if max_time is None:
        max_time = 999999.0
    if time_until is None or time_until < min_time or time_until > max_time:
        return "time_window"
    if _is_fade_yes_with_no_strategy(strategy):
        probability_yes = _float_or_none(row.get("probability_yes"))
        if probability_yes is None or probability_yes < threshold:
            return "fade_threshold"
        if adjusted_entry is None or adjusted_entry <= 0 or adjusted_entry >= 1:
            return "missing_entry_price"
        return None
    min_probability = _float_or_none(strategy.get("min_probability_for_direction"))
    max_probability = _float_or_none(strategy.get("max_probability_for_direction"))
    if min_probability is not None and probability_for_direction < min_probability:
        return "probability_below_min"
    if max_probability is not None and probability_for_direction > max_probability:
        return "probability_above_max"
    if probability_for_direction < threshold:
        return "threshold"
    if adjusted_entry is None or adjusted_entry <= 0 or adjusted_entry >= 1:
        return "missing_entry_price"
    if bool(strategy.get("require_positive_edge", True)) and (estimated_edge is None or estimated_edge <= 0):
        return "non_positive_edge"
    min_edge = _float_or_none(strategy.get("min_estimated_edge")) or 0.0
    if estimated_edge is None or estimated_edge < min_edge:
        return "min_estimated_edge"
    return None


def _insert_prediction_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    strategy: dict[str, Any],
    *,
    direction: str,
    run_id: str,
    threshold: float,
    entry_price: float,
    adjusted_entry_price: float,
    probability_for_direction: float,
    estimated_edge: float,
) -> int | None:
    strategy_id = str(strategy.get("strategy_id") or "")
    signal_timestamp = str(row.get("timestamp") or "")
    market_id = str(row.get("market_id") or "")
    if not strategy_id or not market_id or not signal_timestamp:
        return None
    if _one_trade_per_market_enabled(strategy) and _has_existing_prediction_market_trade(
        output,
        strategy_id=strategy_id,
        market_id=market_id,
    ):
        return None
    exists = output.execute(
        """
        SELECT id
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND strategy_id = ?
        LIMIT 1
        """,
        (run_id, market_id, signal_timestamp, strategy_id),
    ).fetchone()
    if exists is not None:
        return None
    horizon = _float_or_none(strategy.get("fixed_horizon_exit_sec")) or 15.0
    # The recorder connection is not available here; exit fields are injected by row context if present.
    exit_price = _float_or_none(row.get(f"exit_price_{direction.lower()}"))
    adjusted_exit = _adjusted_exit_price(exit_price, strategy)
    stake_usd = _float_or_none(strategy.get("stake_usd")) or 1.0
    pnl, roi = _pnl_and_roi(adjusted_entry_price, adjusted_exit, stake_usd=stake_usd)
    status = "closed" if pnl is not None else "open"
    exit_time = row.get("exit_timestamp") if pnl is not None else None
    exit_reason = "fixed_horizon" if pnl is not None else None
    created_at = to_iso(datetime.now(timezone.utc))
    with output:
        cursor = output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes, probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd, entry_price,
                adjusted_entry_price, best_bid_yes, best_ask_yes, best_bid_no,
                best_ask_no, feature_ready, strict_validation_passed,
                snapshot_quality_status, status, skip_reason, exit_type,
                exit_time, exit_reason, exit_price, adjusted_exit_price,
                realized_pnl_usd, realized_roi, exit_slippage_cents,
                fixed_horizon_exit_sec, time_regime, pnl_usd, roi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                run_id,
                market_id,
                signal_timestamp,
                row.get("question"),
                row.get("market_start_time"),
                row.get("market_close_time"),
                _float_or_none(row.get("time_until_resolution")),
                row.get("model_path"),
                "transformer_predictions",
                strategy_id,
                strategy.get("strategy_name") or strategy_id,
                _float_or_none(row.get("probability_yes")),
                probability_for_direction,
                estimated_edge,
                direction,
                threshold,
                stake_usd,
                entry_price,
                adjusted_entry_price,
                _float_or_none(row.get("best_bid_yes")),
                _float_or_none(row.get("best_ask_yes")),
                _float_or_none(row.get("best_bid_no")),
                _float_or_none(row.get("best_ask_no")),
                _int_or_none(row.get("feature_ready")),
                _int_or_none(row.get("strict_validation_passed")),
                row.get("snapshot_quality_status"),
                status,
                None,
                "FIXED_HORIZON_EXIT",
                exit_time,
                exit_reason,
                exit_price,
                adjusted_exit,
                pnl,
                roi,
                _float_or_none(strategy.get("exit_slippage_cents")),
                horizon,
                _time_regime(_float_or_none(row.get("time_until_resolution"))),
                pnl,
                roi,
            ),
        )
    return int(cursor.lastrowid)


def _one_trade_per_market_enabled(strategy: dict[str, Any]) -> bool:
    return bool(strategy.get("one_trade_per_market", True))


def _has_existing_prediction_market_trade(
    output: sqlite3.Connection,
    *,
    strategy_id: str,
    market_id: str,
) -> bool:
    if not strategy_id or not market_id:
        return False
    row = output.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE market_id = ?
          AND COALESCE(strategy_id, '') = ?
          AND COALESCE(status, '') != 'skipped'
        LIMIT 1
        """,
        (market_id, strategy_id),
    ).fetchone()
    return row is not None


def _transformer_duplicate_market_groups(output: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = output.execute(
        f"""
        SELECT
            COALESCE(strategy_id, '') AS strategy_id,
            market_id,
            COUNT(*) AS trade_count,
            MIN(id) AS first_id,
            MAX(id) AS last_id,
            MIN(created_at) AS first_created_at,
            MAX(created_at) AS latest_created_at
        FROM paper_trades
        WHERE {_TRANSFORMER_PAPER_TRADE_SQL}
          AND COALESCE(status, '') != 'skipped'
          AND COALESCE(strategy_id, '') != ''
          AND COALESCE(market_id, '') != ''
        GROUP BY COALESCE(strategy_id, ''), market_id
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC, strategy_id ASC, market_id ASC
        LIMIT 100
        """
    ).fetchall()
    return [
        {
            "strategy_id": row["strategy_id"],
            "market_id": row["market_id"],
            "trade_count": int(row["trade_count"] or 0),
            "first_id": row["first_id"],
            "last_id": row["last_id"],
            "first_created_at": row["first_created_at"],
            "latest_created_at": row["latest_created_at"],
        }
        for row in rows
    ]


def _insert_prediction_candidate(
    output: sqlite3.Connection,
    row: dict[str, Any],
    strategy: dict[str, Any],
    *,
    direction: str,
    run_id: str,
    decision: str,
    rejection_reason: str | None,
    threshold: float,
    entry_price: float | None,
    adjusted_entry_price: float | None,
    probability_for_direction: float,
    estimated_edge: float | None,
    linked_trade_id: int | None,
) -> int | None:
    strategy_id = str(strategy.get("strategy_id") or "")
    market_id = str(row.get("market_id") or "")
    feature_timestamp = str(row.get("timestamp") or "")
    if not strategy_id or not market_id or not feature_timestamp:
        return None
    with output:
        cursor = output.execute(
            """
            INSERT OR IGNORE INTO paper_trade_candidates (
                created_at, run_id, market_id, question, feature_timestamp,
                market_start_time, market_close_time, strategy_id, strategy_name,
                candidate_direction, candidate_entry_price,
                candidate_adjusted_entry_price, stake_usd, base_model_name,
                base_model_path, predicted_probability_yes,
                probability_for_direction, estimated_edge, threshold_used,
                min_estimated_edge_used, time_until_resolution,
                min_time_until_resolution_sec, max_time_until_resolution_sec,
                min_probability_for_direction, max_probability_for_direction,
                max_open_trades, best_bid_yes, best_ask_yes, best_bid_no,
                best_ask_no, feature_ready, strict_validation_passed,
                snapshot_quality_status, decision, rejection_reason,
                linked_paper_trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                to_iso(datetime.now(timezone.utc)),
                run_id,
                market_id,
                row.get("question"),
                feature_timestamp,
                row.get("market_start_time"),
                row.get("market_close_time"),
                strategy_id,
                strategy.get("strategy_name") or strategy_id,
                direction,
                entry_price,
                adjusted_entry_price,
                _float_or_none(strategy.get("stake_usd")) or 1.0,
                "transformer_predictions",
                row.get("model_path"),
                _float_or_none(row.get("probability_yes")),
                probability_for_direction,
                estimated_edge,
                threshold,
                _float_or_none(strategy.get("min_estimated_edge")) or 0.0,
                _float_or_none(row.get("time_until_resolution")),
                _float_or_none(strategy.get("min_time_until_resolution_sec")),
                _float_or_none(strategy.get("max_time_until_resolution_sec")),
                _float_or_none(strategy.get("min_probability_for_direction")),
                _float_or_none(strategy.get("max_probability_for_direction")),
                _int_or_none(strategy.get("max_open_trades")),
                _float_or_none(row.get("best_bid_yes")),
                _float_or_none(row.get("best_ask_yes")),
                _float_or_none(row.get("best_bid_no")),
                _float_or_none(row.get("best_ask_no")),
                _int_or_none(row.get("feature_ready")),
                _int_or_none(row.get("strict_validation_passed")),
                row.get("snapshot_quality_status"),
                decision,
                rejection_reason,
                linked_trade_id,
            ),
        )
    if cursor.rowcount <= 0 or not cursor.lastrowid:
        return None
    return int(cursor.lastrowid)


def _backfill_transformer_prediction_trade(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    direction = str(row.get("signal_direction") or "").upper()
    alias_result = _backfill_aliases_from_realized(output, row, dry_run=dry_run)
    if alias_result is not None:
        return alias_result
    adjusted_entry = _float_or_none(row.get("adjusted_entry_price"))
    if adjusted_entry is None:
        entry_price = _float_or_none(row.get("entry_price"))
        adjusted_entry = entry_price
    stake_usd = _float_or_none(row.get("stake_usd")) or 1.0
    if adjusted_entry is None or adjusted_entry <= 0 or adjusted_entry >= 1:
        return "invalid_entry_price"
    exit_price = _float_or_none(row.get("exit_price"))
    exit_time = row.get("exit_time")
    if exit_price is None or not exit_time:
        horizon = _float_or_none(row.get("fixed_horizon_exit_sec")) or 15.0
        snapshot = _snapshot_at_or_after_target(recorder, _paper_trade_row_for_snapshot(row), horizon_sec=horizon)
        if not snapshot:
            return "no_exit_snapshot"
        exit_time = snapshot.get("timestamp")
        exit_price = _first_price(
            snapshot,
            ("best_bid_yes", "mid_price_yes", "last_trade_price")
            if direction == "YES"
            else ("best_bid_no", "mid_price_no", "last_trade_price"),
        )
    adjusted_exit = _float_or_none(row.get("adjusted_exit_price"))
    if adjusted_exit is None:
        adjusted_exit = _adjusted_exit_price_from_row(exit_price, row)
    pnl, roi = _pnl_and_roi(adjusted_entry, adjusted_exit, stake_usd=stake_usd)
    if pnl is None or roi is None:
        return "no_exit_snapshot"
    if dry_run:
        return "updated"
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET status = 'closed',
                exit_time = ?,
                exit_reason = COALESCE(exit_reason, 'fixed_horizon'),
                exit_price = ?,
                adjusted_exit_price = ?,
                realized_pnl_usd = ?,
                realized_roi = ?,
                pnl_usd = ?,
                roi = ?,
                exit_slippage_cents = COALESCE(exit_slippage_cents, ?),
                fixed_horizon_exit_sec = COALESCE(fixed_horizon_exit_sec, ?)
            WHERE id = ?
            """,
            (
                exit_time,
                exit_price,
                adjusted_exit,
                pnl,
                roi,
                pnl,
                roi,
                _float_or_none(row.get("exit_slippage_cents")) or 0.0,
                _float_or_none(row.get("fixed_horizon_exit_sec")) or 15.0,
                row.get("id"),
            ),
        )
    return "updated"


def _backfill_aliases_from_realized(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    dry_run: bool,
) -> str | None:
    realized_pnl = _float_or_none(row.get("realized_pnl_usd"))
    realized_roi = _float_or_none(row.get("realized_roi"))
    pnl = _float_or_none(row.get("pnl_usd"))
    roi = _float_or_none(row.get("roi"))
    if realized_pnl is None and realized_roi is None:
        return None
    next_pnl = pnl if pnl is not None else realized_pnl
    next_roi = roi if roi is not None else realized_roi
    if next_pnl is None or next_roi is None:
        return None
    if pnl is not None and roi is not None:
        return None
    if dry_run:
        return "aliases_filled"
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET pnl_usd = COALESCE(pnl_usd, ?),
                roi = COALESCE(roi, ?),
                realized_pnl_usd = COALESCE(realized_pnl_usd, ?),
                realized_roi = COALESCE(realized_roi, ?)
            WHERE id = ?
            """,
            (
                next_pnl,
                next_roi,
                next_pnl,
                next_roi,
                row.get("id"),
            ),
        )
    return "aliases_filled"


def _paper_trade_row_for_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "market_id": row.get("market_id"),
        "timestamp": row.get("signal_timestamp"),
        "signal_timestamp": row.get("signal_timestamp"),
    }


def _adjusted_exit_price_from_row(exit_price: float | None, row: dict[str, Any]) -> float | None:
    if exit_price is None:
        return None
    slippage = (_float_or_none(row.get("exit_slippage_cents")) or 0.0) / 100.0
    return max(0.0, float(exit_price) - slippage)


def _insert_shadow_trade(
    output: sqlite3.Connection,
    heartbeat: dict[str, Any],
    *,
    artifacts: TransformerLiveArtifacts,
    run_id: str,
    probability_threshold: float,
    fixed_horizon_sec: int,
    entry_price: float,
) -> bool:
    market_id = str(heartbeat.get("latest_market_id") or "")
    signal_timestamp = str(heartbeat.get("latest_feature_timestamp") or "")
    strategy_id = (
        f"transformer_clean{int(heartbeat.get('sequence_length') or 120)}"
        f"_yes_t{_threshold_slug(probability_threshold)}_fh{int(fixed_horizon_sec)}"
    )
    if not market_id or not signal_timestamp:
        return False
    exists = output.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND strategy_id = ?
        LIMIT 1
        """,
        (run_id, market_id, signal_timestamp, strategy_id),
    ).fetchone()
    if exists is not None:
        return False
    probability = float(heartbeat["probability_yes"])
    with output:
        output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes, probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd, entry_price,
                adjusted_entry_price, best_bid_yes, best_ask_yes, best_bid_no,
                best_ask_no, feature_ready, strict_validation_passed,
                snapshot_quality_status, status, skip_reason, exit_type,
                fixed_horizon_exit_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                to_iso(datetime.now(timezone.utc)),
                run_id,
                market_id,
                signal_timestamp,
                None,
                heartbeat.get("market_start_time"),
                heartbeat.get("market_close_time"),
                _float_or_none(heartbeat.get("time_until_resolution")),
                artifacts.model_path,
                "transformer_sequence",
                strategy_id,
                f"Transformer clean120 YES >= {probability_threshold:.2f} fh{fixed_horizon_sec}s",
                probability,
                probability,
                probability - entry_price,
                "YES",
                probability_threshold,
                1.0,
                entry_price,
                entry_price,
                _float_or_none(heartbeat.get("best_bid_yes")),
                _float_or_none(heartbeat.get("best_ask_yes")),
                _float_or_none(heartbeat.get("best_bid_no")),
                _float_or_none(heartbeat.get("best_ask_no")),
                _int_or_none(heartbeat.get("feature_ready")),
                _int_or_none(heartbeat.get("strict_validation_passed")),
                heartbeat.get("snapshot_quality_status"),
                "open",
                None,
                "FIXED_HORIZON_EXIT",
                float(fixed_horizon_sec),
            ),
        )
    return True


def _insert_shadow_skip(
    output: sqlite3.Connection,
    heartbeat: dict[str, Any],
    *,
    artifacts: TransformerLiveArtifacts,
    run_id: str,
    probability_threshold: float,
    skip_reason: str,
) -> None:
    market_id = str(heartbeat.get("latest_market_id") or "")
    signal_timestamp = str(heartbeat.get("latest_feature_timestamp") or "")
    if not market_id or not signal_timestamp:
        return
    strategy_id = (
        f"transformer_clean{int(heartbeat.get('sequence_length') or 120)}"
        f"_yes_t{_threshold_slug(probability_threshold)}_shadow"
    )
    exists = output.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND strategy_id = ?
          AND status = 'skipped'
          AND skip_reason = ?
        LIMIT 1
        """,
        (run_id, market_id, signal_timestamp, strategy_id, skip_reason),
    ).fetchone()
    if exists is not None:
        return
    probability = _float_or_none(heartbeat.get("probability_yes"))
    with output:
        output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes, probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd, entry_price,
                adjusted_entry_price, best_bid_yes, best_ask_yes, best_bid_no,
                best_ask_no, feature_ready, strict_validation_passed,
                snapshot_quality_status, status, skip_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                to_iso(datetime.now(timezone.utc)),
                run_id,
                market_id,
                signal_timestamp,
                heartbeat.get("market_start_time"),
                heartbeat.get("market_close_time"),
                _float_or_none(heartbeat.get("time_until_resolution")),
                artifacts.model_path,
                "transformer_sequence",
                strategy_id,
                f"Transformer clean120 YES >= {probability_threshold:.2f} shadow",
                probability,
                probability,
                None,
                "YES",
                probability_threshold,
                1.0,
                _entry_price_yes(heartbeat),
                _entry_price_yes(heartbeat),
                _float_or_none(heartbeat.get("best_bid_yes")),
                _float_or_none(heartbeat.get("best_ask_yes")),
                _float_or_none(heartbeat.get("best_bid_no")),
                _float_or_none(heartbeat.get("best_ask_no")),
                _int_or_none(heartbeat.get("feature_ready")),
                _int_or_none(heartbeat.get("strict_validation_passed")),
                heartbeat.get("snapshot_quality_status"),
                "skipped",
                skip_reason,
            ),
        )


def _entry_price_yes(heartbeat: dict[str, Any]) -> float | None:
    for field in ("best_ask_yes", "yes_price", "mid_price_yes"):
        value = _float_or_none(heartbeat.get(field))
        if value is not None:
            return value
    return None


def _entry_price(row: dict[str, Any], direction: str) -> float | None:
    if direction == "YES":
        return _first_price(row, ("best_ask_yes", "yes_price", "mid_price_yes"))
    return _first_price(row, ("best_ask_no", "no_price", "mid_price_no"))


def _first_price(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _adjusted_entry_price(entry_price: float | None, strategy: dict[str, Any]) -> float | None:
    if entry_price is None:
        return None
    slippage = (_float_or_none(strategy.get("entry_slippage_cents")) or 0.0) / 100.0
    fee = (_float_or_none(strategy.get("fee_cents")) or 0.0) / 100.0
    return min(1.0, float(entry_price) + slippage + fee)


def _adjusted_exit_price(exit_price: float | None, strategy: dict[str, Any]) -> float | None:
    if exit_price is None:
        return None
    slippage = (_float_or_none(strategy.get("exit_slippage_cents")) or 0.0) / 100.0
    fee = (_float_or_none(strategy.get("fee_cents")) or 0.0) / 100.0
    return max(0.0, float(exit_price) - slippage - fee)


def _pnl_and_roi(
    adjusted_entry_price: float | None,
    adjusted_exit_price: float | None,
    *,
    stake_usd: float,
) -> tuple[float | None, float | None]:
    if adjusted_entry_price is None or adjusted_exit_price is None:
        return None, None
    if adjusted_entry_price <= 0 or adjusted_entry_price >= 1:
        return None, None
    stake = max(0.0, float(stake_usd))
    if stake <= 0:
        return None, None
    exit_value = stake / adjusted_entry_price * adjusted_exit_price
    pnl = exit_value - stake
    return pnl, pnl / stake


def _time_regime(time_until_resolution: float | None) -> str:
    if time_until_resolution is None:
        return "unknown"
    if time_until_resolution < 30:
        return "0-30s"
    if time_until_resolution < 60:
        return "30-60s"
    if time_until_resolution < 120:
        return "60-120s"
    return "120s+"


def _threshold_slug(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def _open_readonly_sqlite(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise TransformerShadowPaperTraderError(f"recorder DB not found: {path}")
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


def _prediction_feature_timestamp(row: dict[str, Any]) -> str:
    return str(
        row.get("latest_feature_timestamp")
        or row.get("feature_timestamp")
        or row.get("signal_timestamp")
        or row.get("timestamp")
        or ""
    )


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None


def _emit(enabled: bool, event: str, **payload: Any) -> None:
    if not enabled:
        return
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


__all__ = [
    "TransformerShadowPaperTraderError",
    "backfill_transformer_prediction_paper_pnl",
    "run_transformer_prediction_paper_strategy",
    "run_transformer_shadow_paper_strategy",
]
