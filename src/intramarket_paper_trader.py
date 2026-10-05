from __future__ import annotations

import csv
import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .backtest_baseline_strategy import _load_feature_columns, _load_model, _predict_probabilities
from .baseline_paper_trader import (
    BaselinePaperTraderConfig,
    connect_paper_output_db,
    connect_recorder_read_only,
    fetch_latest_candidate_rows,
)
from .baseline_paper_trader import (
    _filter_candidate_eligibility,
    _filter_prediction_prerequisites,
    _float_or_none,
    _market_resolution_for_trade,
    _model_name,
    _probability_for_direction,
    _rate,
    _round_or_none,
)
from .btc_price_feed import POLYMARKET_RTDS_BINANCE_SOURCE, POLYMARKET_RTDS_CHAINLINK_SOURCE
from .models import parse_timestamp, to_iso


DIRECTION_MODES = {"YES_ONLY", "NO_ONLY", "BOTH"}
EXIT_TYPES = {
    "HOLD_TO_RESOLUTION",
    "FIXED_HORIZON_EXIT",
    "TAKE_PROFIT_STOP_LOSS",
    "TIME_OR_SIGNAL_EXIT",
}
ENTRY_TYPES = {"MODEL_EDGE"}


class IntramarketPaperTraderError(ValueError):
    pass


@dataclass(slots=True)
class IntramarketStrategyDefinition:
    strategy_id: str
    strategy_name: str
    enabled: bool
    direction_mode: str
    entry_type: str
    exit_type: str
    min_probability_for_direction: float | None
    max_probability_for_direction: float | None
    min_estimated_edge: float
    require_positive_edge: bool
    min_time_until_resolution_sec: float
    max_time_until_resolution_sec: float
    stake_usd: float
    entry_slippage_cents: float
    exit_slippage_cents: float
    fee_cents: float
    max_open_positions_per_strategy: int
    max_open_positions_per_market: int
    max_total_open_stake: float | None
    cooldown_after_exit_sec: float
    one_position_per_market_per_strategy: bool
    allow_reentry_per_market: bool
    allow_mid_price_fallback: bool
    long_threshold: float | None = None
    short_threshold: float | None = None
    exit_after_sec: float | None = None
    take_profit_cents: float | None = None
    stop_loss_cents: float | None = None
    max_hold_sec: float | None = None
    take_profit_roi: float | None = None
    stop_loss_roi: float | None = None
    max_hold_seconds: float | None = None
    min_hold_seconds: float = 0.0
    exit_probability_below: float | None = None
    exit_probability_drop_from_entry: float | None = None


def load_intramarket_strategy_config(config_path: str | Path) -> list[IntramarketStrategyDefinition]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = payload.get("strategies") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise IntramarketPaperTraderError("config must contain a non-empty strategies list")
    strategies = [_parse_strategy(item, index=index) for index, item in enumerate(raw)]
    return [strategy for strategy in strategies if strategy.enabled]


def run_intramarket_paper_trader(
    *,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    config_path: str,
    output_db_path: str,
    run_id: str | None = None,
    poll_sec: float = 1.0,
    max_btc_age_sec: float = 3.0,
    max_feature_age_sec: float = 5.0,
    max_iterations: int | None = None,
    emit_logs: bool = True,
) -> dict[str, Any]:
    strategies = load_intramarket_strategy_config(config_path)
    if not strategies:
        raise IntramarketPaperTraderError("no enabled intramarket strategies found")
    model = _load_model(model_path)
    feature_columns = _load_feature_columns(feature_columns_path)
    recorder = connect_recorder_read_only(recorder_db_path)
    output = connect_paper_output_db(output_db_path)
    try:
        ensure_intramarket_schema(output)
        iterations = 0
        last_report: dict[str, Any] | None = None
        _emit(
            emit_logs,
            "intramarket_paper_trader_started",
            run_id=run_id,
            recorder_db_path=recorder_db_path,
            output_db_path=output_db_path,
            strategy_count=len(strategies),
        )
        while True:
            iterations += 1
            last_report = run_intramarket_paper_trader_once(
                recorder,
                output,
                model=model,
                feature_columns=feature_columns,
                strategies=strategies,
                recorder_db_path=recorder_db_path,
                output_db_path=output_db_path,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                run_id=run_id,
                max_btc_age_sec=max_btc_age_sec,
                max_feature_age_sec=max_feature_age_sec,
                emit_logs=emit_logs,
            )
            if max_iterations is not None and iterations >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iterations,
                    "output_db_path": output_db_path,
                    "last_poll": last_report,
                }
            time.sleep(max(0.0, float(poll_sec)))
    finally:
        recorder.close()
        output.close()


def run_intramarket_paper_trader_once(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    model: Any,
    feature_columns: Sequence[str],
    strategies: Sequence[IntramarketStrategyDefinition],
    recorder_db_path: str,
    output_db_path: str,
    model_path: str,
    feature_columns_path: str,
    run_id: str | None,
    max_btc_age_sec: float,
    max_feature_age_sec: float,
    emit_logs: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    ensure_intramarket_schema(output)
    now_dt = now or datetime.now(timezone.utc)
    by_strategy = {strategy.strategy_id: strategy for strategy in strategies}
    exit_counts = update_intramarket_positions(
        recorder,
        output,
        strategies_by_id=by_strategy,
        model=model,
        feature_columns=feature_columns,
        now=now_dt,
        emit_logs=emit_logs,
    )
    settled_candidates = settle_intramarket_candidates(recorder, output, now=now_dt)
    max_time = max([strategy.max_time_until_resolution_sec for strategy in strategies] + [300.0])
    candidates = fetch_latest_candidate_rows(
        recorder,
        max_btc_age_sec=max_btc_age_sec,
        min_time_until_resolution_sec=0.0,
        max_time_until_resolution_sec=max_time,
        now=now_dt,
        max_feature_age_sec=max_feature_age_sec,
    )
    base_config = BaselinePaperTraderConfig(
        recorder_db_path=recorder_db_path,
        output_db_path=output_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        max_btc_age_sec=max_btc_age_sec,
        max_feature_age_sec=max_feature_age_sec,
        min_time_until_resolution_sec=0.0,
        max_time_until_resolution_sec=max_time,
    )
    live_rows, live_counts, _latest_rejection = _filter_candidate_eligibility(
        candidates,
        config=base_config,
        now=now_dt,
        emit_logs=False,
    )
    prediction_rows, prereq_counts, _prereq_rejection = _filter_prediction_prerequisites(
        live_rows,
        feature_columns,
        emit_logs=False,
    )
    probabilities = _predict_once(model, prediction_rows, feature_columns)
    model_name = _model_name(model)
    per_strategy: list[dict[str, Any]] = []
    total_candidates = 0
    trades_opened = 0
    skips_logged = 0
    for strategy in strategies:
        stats = Counter()
        strategy_candidates_logged = 0
        strategy_positions_opened = 0
        strategy_skips_logged = 0
        latest_decision = "NONE"
        latest_rejection: str | None = None
        latest_feature_timestamp: str | None = None
        for row in prediction_rows:
            key = _feature_key(row)
            if key not in probabilities:
                continue
            latest_feature_timestamp = str(row.get("timestamp") or "")
            result = evaluate_intramarket_strategy_row(
                output,
                row,
                probability_yes=probabilities[key],
                strategy=strategy,
                run_id=run_id,
                model_path=model_path,
                model_name=model_name,
                now=now_dt,
                emit_logs=emit_logs,
            )
            stats.update(result["rejection_counts"])
            total_candidates += int(result["candidates_logged"])
            strategy_candidates_logged += int(result["candidates_logged"])
            trades_opened += int(result["positions_opened"])
            strategy_positions_opened += int(result["positions_opened"])
            skips_logged += int(result["skips_logged"])
            strategy_skips_logged += int(result["skips_logged"])
            if result["latest_decision"]:
                latest_decision = str(result["latest_decision"])
            if result["latest_rejection_reason"]:
                latest_rejection = str(result["latest_rejection_reason"])
        per_strategy.append(
            {
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.strategy_name,
                "exit_type": strategy.exit_type,
                "candidates_logged": strategy_candidates_logged,
                "positions_opened": strategy_positions_opened,
                "skips_logged": strategy_skips_logged,
                "open_positions": _position_count(output, strategy.strategy_id, "open"),
                "positions_closed": _position_count(output, strategy.strategy_id, "closed"),
                "positions_settled": _position_count(output, strategy.strategy_id, "settled"),
                "skipped_threshold": int(stats.get("threshold", 0)),
                "skipped_probability_below_min": int(stats.get("probability_below_min", 0)),
                "skipped_probability_above_max": int(stats.get("probability_above_max", 0)),
                "skipped_non_positive_edge": int(stats.get("non_positive_edge", 0)),
                "skipped_min_estimated_edge": int(stats.get("min_estimated_edge", 0)),
                "skipped_time_window": int(stats.get("time_window", 0)),
                "skipped_max_open_positions": int(stats.get("max_open_positions", 0)),
                "skipped_cooldown": int(stats.get("cooldown", 0)),
                "latest_decision": latest_decision,
                "latest_rejection_reason": latest_rejection,
                "latest_feature_timestamp": latest_feature_timestamp,
            }
        )
    heartbeat = {
        "timestamp": to_iso(now_dt),
        "run_id": run_id,
        "active_strategies": len(strategies),
        "latest_feature_timestamp": _latest_feature_timestamp(prediction_rows),
        "candidates_logged": total_candidates,
        "positions_opened": trades_opened,
        "skips_logged": skips_logged,
        "open_positions": _status_count(output, "open"),
        "closed_positions": _status_count(output, "closed"),
        "awaiting_resolution_positions": _status_count(output, "awaiting_resolution"),
        "settled_positions": _status_count(output, "settled"),
        "exit_counts": exit_counts,
        "settled_candidates": settled_candidates,
        "eligibility_skips": dict(live_counts + prereq_counts),
        "strategy_stats": per_strategy,
    }
    _insert_intramarket_heartbeat(output, heartbeat)
    _emit(emit_logs, "intramarket_paper_trader_heartbeat", **heartbeat)
    return heartbeat


def ensure_intramarket_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                run_id TEXT,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_type TEXT,
                exit_type TEXT,
                entry_time TEXT NOT NULL,
                entry_feature_timestamp TEXT NOT NULL,
                entry_price REAL,
                adjusted_entry_price REAL,
                shares REAL,
                stake_usd REAL,
                predicted_probability_yes_at_entry REAL,
                probability_for_direction_at_entry REAL,
                estimated_edge_at_entry REAL,
                time_until_resolution_at_entry REAL,
                market_start_time TEXT,
                market_close_time TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                exit_price REAL,
                adjusted_exit_price REAL,
                realized_pnl_usd REAL,
                realized_roi REAL,
                resolved_label TEXT,
                settlement_pnl_usd REAL,
                settlement_roi REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_position_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                position_id INTEGER,
                run_id TEXT,
                strategy_id TEXT,
                market_id TEXT,
                event_type TEXT NOT NULL,
                details_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_intramarket_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_id TEXT,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT,
                market_id TEXT NOT NULL,
                question TEXT,
                feature_timestamp TEXT NOT NULL,
                market_start_time TEXT,
                market_close_time TEXT,
                entry_type TEXT,
                exit_type TEXT,
                candidate_direction TEXT NOT NULL,
                candidate_entry_price REAL,
                candidate_adjusted_entry_price REAL,
                stake_usd REAL,
                base_model_name TEXT,
                base_model_path TEXT,
                predicted_probability_yes REAL,
                probability_for_direction REAL,
                estimated_edge REAL,
                threshold_used REAL,
                min_probability_for_direction REAL,
                max_probability_for_direction REAL,
                min_estimated_edge_used REAL,
                time_until_resolution REAL,
                best_bid_yes REAL,
                best_ask_yes REAL,
                best_bid_no REAL,
                best_ask_no REAL,
                btc_chainlink_price REAL,
                btc_binance_price REAL,
                decision TEXT NOT NULL,
                rejection_reason TEXT,
                linked_position_id INTEGER,
                actual_resolved_label TEXT,
                would_have_payout_usd REAL,
                would_have_pnl_usd REAL,
                would_have_roi REAL,
                settled_at TEXT,
                config_snapshot_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_intramarket_heartbeats (
                timestamp TEXT PRIMARY KEY,
                run_id TEXT,
                active_strategies INTEGER,
                latest_feature_timestamp TEXT,
                open_positions INTEGER,
                closed_positions INTEGER,
                awaiting_resolution_positions INTEGER,
                settled_positions INTEGER,
                candidates_logged INTEGER,
                positions_opened INTEGER,
                skips_logged INTEGER,
                strategy_stats_json TEXT,
                details_json TEXT
            )
            """
        )
        _ensure_columns(
            conn,
            "paper_positions",
            {
                "strategy_name": "TEXT",
                "entry_type": "TEXT",
                "exit_type": "TEXT",
                "market_start_time": "TEXT",
                "market_close_time": "TEXT",
            },
        )
        _ensure_columns(
            conn,
            "paper_intramarket_candidates",
            {
                "strategy_name": "TEXT",
                "entry_type": "TEXT",
                "exit_type": "TEXT",
                "config_snapshot_json": "TEXT",
            },
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intramarket_positions_strategy_status
            ON paper_positions (strategy_id, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intramarket_positions_market_status
            ON paper_positions (run_id, market_id, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intramarket_candidates_key
            ON paper_intramarket_candidates (
                strategy_id, run_id, market_id, feature_timestamp, candidate_direction
            )
            """
        )


def evaluate_intramarket_strategy_row(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability_yes: float,
    strategy: IntramarketStrategyDefinition,
    run_id: str | None,
    model_path: str,
    model_name: str,
    now: datetime,
    emit_logs: bool = True,
) -> dict[str, Any]:
    side_results: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for direction in _strategy_directions(strategy):
        candidate = _candidate_context(
            row,
            probability_yes=probability_yes,
            direction=direction,
            strategy=strategy,
        )
        reason = _entry_rejection_reason(
            output,
            row,
            candidate=candidate,
            strategy=strategy,
            now=now,
        )
        side_results.append({"direction": direction, "candidate": candidate, "reason": reason})
    passing = [item for item in side_results if item["reason"] is None]
    selected = max(
        passing,
        key=lambda item: float(item["candidate"].get("estimated_edge") or -999.0),
        default=None,
    )
    candidates_logged = 0
    skips_logged = 0
    positions_opened = 0
    latest_decision: str | None = None
    latest_rejection: str | None = None
    for item in side_results:
        direction = str(item["direction"])
        candidate = item["candidate"]
        reason = item["reason"]
        if selected is not None and item is selected:
            position_id = _open_intramarket_position(
                output,
                row,
                candidate=candidate,
                strategy=strategy,
                run_id=run_id,
                model_path=model_path,
                model_name=model_name,
                now=now,
            )
            if position_id is None:
                reason = "duplicate_candidate"
            else:
                inserted = _insert_intramarket_candidate(
                    output,
                    row,
                    candidate=candidate,
                    strategy=strategy,
                    run_id=run_id,
                    model_path=model_path,
                    model_name=model_name,
                    decision="TRADE",
                    rejection_reason=None,
                    linked_position_id=position_id,
                    now=now,
                )
                candidates_logged += int(inserted)
                positions_opened += 1
                latest_decision = "TRADE"
                _emit(
                    emit_logs,
                    "intramarket_position_opened",
                    position_id=position_id,
                    strategy_id=strategy.strategy_id,
                    market_id=row.get("market_id"),
                    side=direction,
                    probability_for_direction=candidate.get("probability_for_direction"),
                    estimated_edge=candidate.get("estimated_edge"),
                )
                continue
        elif reason is None:
            reason = "not_selected"
        inserted = _insert_intramarket_candidate(
            output,
            row,
            candidate=candidate,
            strategy=strategy,
            run_id=run_id,
            model_path=model_path,
            model_name=model_name,
            decision="SKIP",
            rejection_reason=reason,
            linked_position_id=None,
            now=now,
        )
        candidates_logged += int(inserted)
        skips_logged += int(inserted)
        if reason:
            rejection_counts[str(reason)] += int(inserted)
            latest_rejection = str(reason)
        latest_decision = "SKIP"
    return {
        "candidates_logged": candidates_logged,
        "positions_opened": positions_opened,
        "skips_logged": skips_logged,
        "rejection_counts": rejection_counts,
        "latest_decision": latest_decision,
        "latest_rejection_reason": latest_rejection,
    }


def update_intramarket_positions(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    strategies_by_id: dict[str, IntramarketStrategyDefinition],
    model: Any | None = None,
    feature_columns: Sequence[str] | None = None,
    now: datetime,
    emit_logs: bool = True,
) -> dict[str, int]:
    ensure_intramarket_schema(output)
    counts: Counter[str] = Counter()
    rows = output.execute(
        """
        SELECT *
        FROM paper_positions
        WHERE status IN ('open', 'awaiting_resolution')
        ORDER BY id ASC
        """
    ).fetchall()
    for row in rows:
        strategy = strategies_by_id.get(str(row["strategy_id"]))
        status = str(row["status"])
        resolution = _market_resolution_for_trade(
            recorder,
            run_id=str(row["run_id"] or ""),
            market_id=str(row["market_id"] or ""),
        )
        if resolution["resolved"] and resolution["label"] is not None:
            if _settle_position(output, row, label=int(resolution["label"]), now=now):
                counts["settled"] += 1
            continue
        close_time = parse_timestamp(row["market_close_time"])
        if status == "open" and close_time is not None and now >= close_time:
            _mark_position_awaiting_resolution(output, row, now=now)
            counts["awaiting_resolution"] += 1
            continue
        if status != "open":
            continue
        if str(row["exit_type"]) == "HOLD_TO_RESOLUTION":
            continue
        exit_decision = _position_exit_decision(
            recorder,
            row,
            strategy=strategy,
            model=model,
            feature_columns=feature_columns,
            now=now,
        )
        if exit_decision is None:
            continue
        if _close_position(output, row, exit_decision=exit_decision, now=now):
            counts[str(exit_decision["reason"])] += 1
            _emit(
                emit_logs,
                "intramarket_position_closed",
                position_id=int(row["id"]),
                strategy_id=row["strategy_id"],
                market_id=row["market_id"],
                side=row["side"],
                exit_reason=exit_decision["reason"],
                realized_pnl_usd=exit_decision.get("realized_pnl_usd"),
            )
    return dict(counts)


def settle_intramarket_candidates(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    now: datetime,
    limit: int = 2000,
) -> int:
    if not _table_exists(output, "paper_intramarket_candidates"):
        return 0
    rows = output.execute(
        """
        SELECT *
        FROM paper_intramarket_candidates
        WHERE actual_resolved_label IS NULL
          AND market_close_time IS NOT NULL
          AND datetime(market_close_time) <= datetime(?)
        ORDER BY id ASC
        LIMIT ?
        """,
        (to_iso(now), int(limit)),
    ).fetchall()
    updated = 0
    for row in rows:
        resolution = _market_resolution_for_trade(
            recorder,
            run_id=str(row["run_id"] or ""),
            market_id=str(row["market_id"] or ""),
        )
        if not resolution["resolved"] or resolution["label"] is None:
            continue
        payout, pnl, roi = _candidate_settlement(row, label=int(resolution["label"]))
        with output:
            output.execute(
                """
                UPDATE paper_intramarket_candidates
                SET actual_resolved_label = ?,
                    would_have_payout_usd = ?,
                    would_have_pnl_usd = ?,
                    would_have_roi = ?,
                    settled_at = ?
                WHERE id = ?
                """,
                (
                    "YES" if int(resolution["label"]) == 1 else "NO",
                    payout,
                    pnl,
                    roi,
                    to_iso(now),
                    int(row["id"]),
                ),
            )
        updated += 1
    return updated


def build_intramarket_paper_analytics_report(
    *,
    paper_db_path: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    conn = _connect_read_only_if_exists(paper_db_path)
    try:
        if conn is None or not _table_exists(conn, "paper_positions"):
            report = {
                "status": "error",
                "error": "paper_positions_table_missing",
                "paper_db_path": paper_db_path,
                "summary": {},
            }
        else:
            positions = [dict(row) for row in conn.execute("SELECT * FROM paper_positions ORDER BY id").fetchall()]
            candidates = (
                [dict(row) for row in conn.execute("SELECT * FROM paper_intramarket_candidates ORDER BY id").fetchall()]
                if _table_exists(conn, "paper_intramarket_candidates")
                else []
            )
            report = {
                "status": "ok",
                "paper_db_path": paper_db_path,
                "summary": _intramarket_summary(positions, candidates),
                "positions": [_enrich_position(row) for row in positions],
            }
    finally:
        if conn is not None:
            conn.close()
    if output_dir:
        _write_intramarket_report_files(report, Path(output_dir))
    return report


def build_intramarket_strategy_leaderboard(
    *,
    paper_db_path: str,
    min_closed: int = 30,
) -> dict[str, Any]:
    conn = _connect_read_only_if_exists(paper_db_path)
    try:
        if conn is None or not _table_exists(conn, "paper_positions"):
            return {
                "status": "error",
                "error": "paper_positions_table_missing",
                "paper_db_path": paper_db_path,
                "rows": [],
            }
        rows = [dict(row) for row in conn.execute("SELECT * FROM paper_positions ORDER BY id").fetchall()]
    finally:
        if conn is not None:
            conn.close()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("strategy_id")), str(row.get("side")), str(row.get("exit_type")))].append(row)
    leaderboard = []
    for (strategy_id, side, exit_type), group in groups.items():
        closed = [row for row in group if row.get("status") in {"closed", "settled"}]
        settled = [row for row in group if row.get("status") == "settled"]
        if len(closed) < int(min_closed):
            continue
        pnl_values = [_combined_pnl(row) for row in closed]
        wins = [pnl for pnl in pnl_values if pnl is not None and pnl > 0]
        leaderboard.append(
            {
                "strategy_id": strategy_id,
                "side": side,
                "exit_type": exit_type,
                "closed_positions": len(closed),
                "settled_positions": len(settled),
                "realized_pnl": _round(sum(float(row.get("realized_pnl_usd") or 0.0) for row in closed)),
                "combined_pnl": _round(sum(float(pnl or 0.0) for pnl in pnl_values)),
                "average_roi": _mean([_combined_roi(row) for row in closed if _combined_roi(row) is not None]),
                "win_rate": _rate(len(wins), len(closed)),
                "average_hold_seconds": _mean([_hold_seconds(row) for row in closed if _hold_seconds(row) is not None]),
                "max_drawdown": _max_drawdown([float(pnl or 0.0) for pnl in pnl_values]),
                "reliability_tier": _reliability_tier(len(closed)),
            }
        )
    leaderboard.sort(key=lambda row: (float(row.get("combined_pnl") or 0.0), float(row.get("average_roi") or 0.0)), reverse=True)
    return {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "min_closed": int(min_closed),
        "rows": leaderboard,
    }


def render_intramarket_strategy_leaderboard(report: dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return json.dumps(report, indent=2, sort_keys=True)
    lines = ["strategy_id side exit_type closed pnl avg_roi win_rate tier"]
    for row in report.get("rows", []):
        lines.append(
            f"{row['strategy_id']} {row['side']} {row['exit_type']} "
            f"{row['closed_positions']} {row['combined_pnl']} {row['average_roi']} "
            f"{row['win_rate']} {row['reliability_tier']}"
        )
    if len(lines) == 1:
        lines.append("no rows met the minimum sample filter")
    return "\n".join(lines)


def export_intramarket_strategy_dataset(
    *,
    paper_db_path: str,
    output_path: str,
    output_csv_path: str | None = None,
    include_unresolved: bool = False,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    conn = _connect_read_only_if_exists(paper_db_path)
    try:
        if conn is None or not _table_exists(conn, "paper_intramarket_candidates"):
            return {
                "status": "error",
                "error": "paper_intramarket_candidates_table_missing",
                "paper_db_path": paper_db_path,
            }
        where = ["1 = 1"]
        params: dict[str, Any] = {}
        if not include_unresolved:
            where.append("c.actual_resolved_label IS NOT NULL OR p.status IN ('closed', 'settled')")
        if strategy_id:
            where.append("c.strategy_id = :strategy_id")
            params["strategy_id"] = strategy_id
        rows = [
            _intramarket_export_row(dict(row))
            for row in conn.execute(
                f"""
                SELECT c.*, p.status AS position_status, p.exit_reason, p.exit_time,
                       p.realized_pnl_usd, p.realized_roi, p.settlement_pnl_usd,
                       p.settlement_roi
                FROM paper_intramarket_candidates c
                LEFT JOIN paper_positions p ON p.id = c.linked_position_id
                WHERE {" AND ".join(where)}
                ORDER BY datetime(c.created_at) ASC, c.id ASC
                """,
                params,
            ).fetchall()
        ]
    finally:
        if conn is not None:
            conn.close()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(rows, output)
    csv_size = None
    if output_csv_path:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, csv_path)
        csv_size = _file_size(csv_path)
    return {
        "status": "ok",
        "paper_db_path": paper_db_path,
        "output_path": str(output),
        "output_file_size_bytes": _file_size(output),
        "output_csv_path": output_csv_path,
        "output_csv_file_size_bytes": csv_size,
        "rows_exported": len(rows),
        "include_unresolved": bool(include_unresolved),
        "strategy_id": strategy_id,
    }


def _parse_strategy(item: Any, *, index: int) -> IntramarketStrategyDefinition:
    if not isinstance(item, dict):
        raise IntramarketPaperTraderError(f"strategy at index {index} must be an object")
    strategy_id = _required_string(item, "strategy_id", index=index)
    direction_mode = _string(item.get("direction_mode"), "BOTH").upper()
    if direction_mode not in DIRECTION_MODES:
        raise IntramarketPaperTraderError(f"{strategy_id}.direction_mode is invalid")
    entry_type = _string(item.get("entry_type"), "MODEL_EDGE").upper()
    if entry_type not in ENTRY_TYPES:
        raise IntramarketPaperTraderError(f"{strategy_id}.entry_type is invalid")
    exit_type = _string(item.get("exit_type"), "TAKE_PROFIT_STOP_LOSS").upper()
    if exit_type not in EXIT_TYPES:
        raise IntramarketPaperTraderError(f"{strategy_id}.exit_type is invalid")
    return IntramarketStrategyDefinition(
        strategy_id=strategy_id,
        strategy_name=str(item.get("strategy_name") or strategy_id),
        enabled=_bool(item.get("enabled"), True),
        direction_mode=direction_mode,
        entry_type=entry_type,
        exit_type=exit_type,
        min_probability_for_direction=_optional_float(item.get("min_probability_for_direction")),
        max_probability_for_direction=_optional_float(item.get("max_probability_for_direction")),
        min_estimated_edge=_float(item.get("min_estimated_edge"), 0.0),
        require_positive_edge=_bool(item.get("require_positive_edge"), True),
        min_time_until_resolution_sec=_float(item.get("min_time_until_resolution_sec"), 0.0),
        max_time_until_resolution_sec=_float(item.get("max_time_until_resolution_sec"), 300.0),
        stake_usd=_float(item.get("stake_usd"), 1.0),
        entry_slippage_cents=_float(item.get("entry_slippage_cents"), 0.0),
        exit_slippage_cents=_float(item.get("exit_slippage_cents"), 0.0),
        fee_cents=_float(item.get("fee_cents"), 0.0),
        max_open_positions_per_strategy=int(_float(item.get("max_open_positions_per_strategy"), 1.0)),
        max_open_positions_per_market=int(_float(item.get("max_open_positions_per_market"), 1.0)),
        max_total_open_stake=_optional_float(item.get("max_total_open_stake")),
        cooldown_after_exit_sec=_float(item.get("cooldown_after_exit_sec"), 0.0),
        one_position_per_market_per_strategy=_bool(
            item.get("one_position_per_market_per_strategy"),
            True,
        ),
        allow_reentry_per_market=_bool(item.get("allow_reentry_per_market"), False),
        allow_mid_price_fallback=_bool(item.get("allow_mid_price_fallback"), False),
        long_threshold=_optional_float(item.get("long_threshold")),
        short_threshold=_optional_float(item.get("short_threshold")),
        exit_after_sec=_optional_float(item.get("exit_after_sec")),
        take_profit_cents=_optional_float(item.get("take_profit_cents")),
        stop_loss_cents=_optional_float(item.get("stop_loss_cents")),
        max_hold_sec=_optional_float(item.get("max_hold_sec")),
        take_profit_roi=_optional_float(item.get("take_profit_roi")),
        stop_loss_roi=_optional_float(item.get("stop_loss_roi")),
        max_hold_seconds=_optional_float(item.get("max_hold_seconds")),
        min_hold_seconds=_float(item.get("min_hold_seconds"), 0.0),
        exit_probability_below=_optional_float(item.get("exit_probability_below")),
        exit_probability_drop_from_entry=_optional_float(
            item.get("exit_probability_drop_from_entry")
        ),
    )


def _candidate_context(
    row: dict[str, Any],
    *,
    probability_yes: float,
    direction: str,
    strategy: IntramarketStrategyDefinition,
) -> dict[str, Any]:
    entry_price = _entry_price(row, direction=direction, allow_mid=strategy.allow_mid_price_fallback)
    adjusted = (
        min(
            1.0,
            float(entry_price)
            + float(strategy.entry_slippage_cents) / 100.0
            + float(strategy.fee_cents) / 100.0,
        )
        if entry_price is not None
        else None
    )
    probability_for_direction = _probability_for_direction(probability_yes, direction=direction)
    estimated_edge = (
        probability_for_direction - adjusted if adjusted is not None else None
    )
    threshold = strategy.long_threshold if direction == "YES" else strategy.short_threshold
    if threshold is None:
        threshold = strategy.min_probability_for_direction
    return {
        "direction": direction,
        "entry_price": entry_price,
        "adjusted_entry_price": adjusted,
        "probability_yes": float(probability_yes),
        "probability_for_direction": probability_for_direction,
        "estimated_edge": estimated_edge,
        "threshold_used": threshold,
    }


def _entry_rejection_reason(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    strategy: IntramarketStrategyDefinition,
    now: datetime,
) -> str | None:
    time_until = _float_or_none(row.get("time_until_resolution"))
    if time_until is None or time_until <= 0:
        return "expired_feature"
    if (
        time_until < float(strategy.min_time_until_resolution_sec)
        or time_until > float(strategy.max_time_until_resolution_sec)
    ):
        return "time_window"
    pdir = _float_or_none(candidate.get("probability_for_direction"))
    if pdir is None:
        return "no_signal"
    if (
        strategy.min_probability_for_direction is not None
        and pdir < float(strategy.min_probability_for_direction)
    ):
        return "probability_below_min"
    if (
        strategy.max_probability_for_direction is not None
        and pdir > float(strategy.max_probability_for_direction)
    ):
        return "probability_above_max"
    direction = str(candidate["direction"])
    if direction == "YES" and strategy.long_threshold is not None and pdir < strategy.long_threshold:
        return "threshold"
    if direction == "NO" and strategy.short_threshold is not None and pdir < strategy.short_threshold:
        return "threshold"
    entry = _float_or_none(candidate.get("entry_price"))
    adjusted = _float_or_none(candidate.get("adjusted_entry_price"))
    if entry is None or adjusted is None or adjusted <= 0 or adjusted >= 1:
        return "missing_entry_price"
    edge = _float_or_none(candidate.get("estimated_edge"))
    if edge is None:
        return "min_estimated_edge"
    if strategy.require_positive_edge and edge <= 0:
        return "non_positive_edge"
    if edge < float(strategy.min_estimated_edge):
        return "min_estimated_edge"
    if _open_position_count(output, strategy_id=strategy.strategy_id) >= int(strategy.max_open_positions_per_strategy):
        return "max_open_positions"
    if (
        _open_position_count(
            output,
            strategy_id=strategy.strategy_id,
            run_id=str(row.get("run_id") or ""),
            market_id=str(row.get("market_id") or ""),
        )
        >= int(strategy.max_open_positions_per_market)
    ):
        return "max_open_positions"
    if strategy.max_total_open_stake is not None and (
        _open_stake(output, strategy_id=strategy.strategy_id) + float(strategy.stake_usd)
        > float(strategy.max_total_open_stake)
    ):
        return "max_total_open_stake"
    if strategy.one_position_per_market_per_strategy and not strategy.allow_reentry_per_market:
        if _any_position_for_market(
            output,
            strategy_id=strategy.strategy_id,
            run_id=str(row.get("run_id") or ""),
            market_id=str(row.get("market_id") or ""),
        ):
            return "one_position_per_market"
    if strategy.cooldown_after_exit_sec > 0 and _within_cooldown(
        output,
        strategy_id=strategy.strategy_id,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        now=now,
        cooldown_sec=float(strategy.cooldown_after_exit_sec),
    ):
        return "cooldown"
    if _candidate_exists(
        output,
        strategy_id=strategy.strategy_id,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        feature_timestamp=str(row.get("timestamp") or ""),
        direction=direction,
    ):
        return "duplicate_candidate"
    return None


def _open_intramarket_position(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    strategy: IntramarketStrategyDefinition,
    run_id: str | None,
    model_path: str,
    model_name: str,
    now: datetime,
) -> int | None:
    if _candidate_exists(
        output,
        strategy_id=strategy.strategy_id,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        feature_timestamp=str(row.get("timestamp") or ""),
        direction=str(candidate["direction"]),
    ):
        return None
    adjusted = float(candidate["adjusted_entry_price"])
    stake = float(strategy.stake_usd)
    shares = stake / adjusted if adjusted > 0 else 0.0
    created = to_iso(now)
    market_run_id = row.get("run_id") or run_id
    entry_time = row.get("timestamp") or created
    values = (
        created,
        created,
        market_run_id,
        strategy.strategy_id,
        strategy.strategy_name,
        row.get("market_id"),
        candidate["direction"],
        "open",
        strategy.entry_type,
        strategy.exit_type,
        entry_time,
        row.get("timestamp"),
        candidate["entry_price"],
        adjusted,
        shares,
        stake,
        candidate["probability_yes"],
        candidate["probability_for_direction"],
        candidate["estimated_edge"],
        _float_or_none(row.get("time_until_resolution")),
        row.get("start_time"),
        row.get("close_time"),
    )
    with output:
        cursor = output.execute(
            """
            INSERT INTO paper_positions (
                created_at, updated_at, run_id, strategy_id, strategy_name,
                market_id, side, status, entry_type, exit_type, entry_time,
                entry_feature_timestamp, entry_price, adjusted_entry_price,
                shares, stake_usd, predicted_probability_yes_at_entry,
                probability_for_direction_at_entry, estimated_edge_at_entry,
                time_until_resolution_at_entry, market_start_time, market_close_time
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            values,
        )
        position_id = int(cursor.lastrowid)
        _insert_position_event(
            output,
            position_id=position_id,
            event_type="opened",
            details={
                "model_path": model_path,
                "model_name": model_name,
                "probability_for_direction": candidate["probability_for_direction"],
                "estimated_edge": candidate["estimated_edge"],
            },
        )
    return position_id


def _insert_intramarket_candidate(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    strategy: IntramarketStrategyDefinition,
    run_id: str | None,
    model_path: str,
    model_name: str,
    decision: str,
    rejection_reason: str | None,
    linked_position_id: int | None,
    now: datetime,
) -> bool:
    key = (
        strategy.strategy_id,
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(row.get("timestamp") or ""),
        str(candidate["direction"]),
    )
    existing = output.execute(
        """
        SELECT id
        FROM paper_intramarket_candidates
        WHERE strategy_id = ?
          AND COALESCE(run_id, '') = ?
          AND market_id = ?
          AND feature_timestamp = ?
          AND candidate_direction = ?
        LIMIT 1
        """,
        key,
    ).fetchone()
    if existing is not None:
        return False
    with output:
        output.execute(
            """
            INSERT INTO paper_intramarket_candidates (
                created_at, run_id, strategy_id, strategy_name, market_id,
                question, feature_timestamp, market_start_time, market_close_time,
                entry_type, exit_type, candidate_direction, candidate_entry_price,
                candidate_adjusted_entry_price, stake_usd, base_model_name,
                base_model_path, predicted_probability_yes,
                probability_for_direction, estimated_edge, threshold_used,
                min_probability_for_direction, max_probability_for_direction,
                min_estimated_edge_used, time_until_resolution, best_bid_yes,
                best_ask_yes, best_bid_no, best_ask_no, btc_chainlink_price,
                btc_binance_price, decision, rejection_reason, linked_position_id,
                config_snapshot_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                to_iso(now),
                row.get("run_id") or run_id,
                strategy.strategy_id,
                strategy.strategy_name,
                row.get("market_id"),
                row.get("question"),
                row.get("timestamp"),
                row.get("start_time"),
                row.get("close_time"),
                strategy.entry_type,
                strategy.exit_type,
                candidate["direction"],
                _round_or_none(candidate.get("entry_price")),
                _round_or_none(candidate.get("adjusted_entry_price")),
                float(strategy.stake_usd),
                model_name,
                model_path,
                _round_or_none(candidate.get("probability_yes")),
                _round_or_none(candidate.get("probability_for_direction")),
                _round_or_none(candidate.get("estimated_edge")),
                _round_or_none(candidate.get("threshold_used")),
                _round_or_none(strategy.min_probability_for_direction),
                _round_or_none(strategy.max_probability_for_direction),
                float(strategy.min_estimated_edge),
                _float_or_none(row.get("time_until_resolution")),
                _float_or_none(row.get("best_bid_yes")),
                _float_or_none(row.get("best_ask_yes")),
                _float_or_none(row.get("best_bid_no")),
                _float_or_none(row.get("best_ask_no")),
                _float_or_none(row.get("btc_chainlink_price")),
                _float_or_none(row.get("btc_binance_price")),
                decision,
                rejection_reason,
                linked_position_id,
                json.dumps(_strategy_snapshot(strategy), sort_keys=True),
            ),
        )
    return True


def _position_exit_decision(
    recorder: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    strategy: IntramarketStrategyDefinition | None,
    model: Any | None = None,
    feature_columns: Sequence[str] | None = None,
    now: datetime,
) -> dict[str, Any] | None:
    entry_time = parse_timestamp(row["entry_time"])
    if entry_time is None:
        return None
    elapsed = max(0.0, (now - entry_time).total_seconds())
    exit_type = str(row["exit_type"])
    if exit_type == "HOLD_TO_RESOLUTION":
        return None
    snapshot = _latest_snapshot_for_position(recorder, row, now=now)
    if snapshot is None:
        return None
    exit_price = _exit_price(snapshot, side=str(row["side"]))
    if exit_price is None:
        return None
    adjusted_exit, pnl, roi = _exit_roi(row, exit_price=exit_price, strategy=strategy)
    entry = _float_or_none(row["adjusted_entry_price"])
    reason: str | None = None
    max_hold = _max_hold_seconds(strategy, exit_type=exit_type)
    fixed_horizon = _float_or_none(getattr(strategy, "exit_after_sec", None))
    if exit_type == "FIXED_HORIZON_EXIT" and fixed_horizon is not None and elapsed >= fixed_horizon:
        reason = "fixed_horizon"
    else:
        min_hold = float(getattr(strategy, "min_hold_seconds", 0.0) or 0.0)
        if elapsed >= min_hold:
            take_profit_roi = _float_or_none(getattr(strategy, "take_profit_roi", None))
            stop_loss_roi = _float_or_none(getattr(strategy, "stop_loss_roi", None))
            if take_profit_roi is not None and roi >= take_profit_roi:
                reason = "take_profit"
            elif stop_loss_roi is not None and roi <= stop_loss_roi:
                reason = "stop_loss"
            else:
                take_profit_cents = _float_or_none(getattr(strategy, "take_profit_cents", None))
                stop_loss_cents = _float_or_none(getattr(strategy, "stop_loss_cents", None))
                if entry is not None and take_profit_cents is not None and adjusted_exit - entry >= take_profit_cents:
                    reason = "take_profit"
                elif entry is not None and stop_loss_cents is not None and entry - adjusted_exit >= stop_loss_cents:
                    reason = "stop_loss"
        if reason is None and max_hold is not None and elapsed >= max_hold:
            reason = "max_hold"
        if reason is None and elapsed >= min_hold:
            current_probability = _current_probability_for_position(
                recorder,
                row,
                model=model,
                feature_columns=feature_columns,
                now=now,
            )
            if current_probability is not None:
                probability_below = _float_or_none(getattr(strategy, "exit_probability_below", None))
                probability_drop = _float_or_none(
                    getattr(strategy, "exit_probability_drop_from_entry", None)
                )
                entry_probability = _float_or_none(row["probability_for_direction_at_entry"])
                if probability_below is not None and current_probability < probability_below:
                    reason = "probability_below"
                elif (
                    probability_drop is not None
                    and entry_probability is not None
                    and current_probability <= entry_probability - probability_drop
                ):
                    reason = "probability_drop"
    if reason is None:
        return None
    return {
        "reason": reason,
        "exit_price": exit_price,
        "adjusted_exit_price": adjusted_exit,
        "realized_pnl_usd": round(pnl, 10),
        "realized_roi": round(roi, 10),
        "held_seconds": round(elapsed, 10),
    }


def _exit_roi(
    row: sqlite3.Row,
    *,
    exit_price: float,
    strategy: IntramarketStrategyDefinition | None,
) -> tuple[float, float, float]:
    exit_slippage = float(getattr(strategy, "exit_slippage_cents", 0.0) or 0.0)
    fee = float(getattr(strategy, "fee_cents", 0.0) or 0.0)
    adjusted_exit = max(0.0, float(exit_price) - exit_slippage / 100.0 - fee / 100.0)
    stake = float(row["stake_usd"] or 0.0)
    shares = float(row["shares"] or 0.0)
    pnl = shares * adjusted_exit - stake
    roi = pnl / stake if stake else 0.0
    return adjusted_exit, pnl, roi


def _max_hold_seconds(
    strategy: IntramarketStrategyDefinition | None,
    *,
    exit_type: str,
) -> float | None:
    if strategy is None:
        return None
    explicit = _float_or_none(getattr(strategy, "max_hold_seconds", None))
    if explicit is not None:
        return explicit
    legacy = _float_or_none(getattr(strategy, "max_hold_sec", None))
    if legacy is not None:
        return legacy
    if exit_type == "FIXED_HORIZON_EXIT":
        return _float_or_none(getattr(strategy, "exit_after_sec", None))
    return None


def _current_probability_for_position(
    recorder: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    model: Any | None,
    feature_columns: Sequence[str] | None,
    now: datetime,
) -> float | None:
    if model is None or not feature_columns:
        return None
    feature_row = _latest_feature_row_for_position(recorder, row, now=now)
    if feature_row is None:
        return None
    try:
        probability_yes = _predict_probabilities(model, [feature_row], feature_columns)[0]
    except Exception:
        return None
    return _probability_for_direction(float(probability_yes), direction=str(row["side"]))


def _latest_feature_row_for_position(
    recorder: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    result = recorder.execute(
        """
        SELECT
          f.*,
          m.question,
          m.start_time,
          m.close_time,
          m.phase AS market_phase,
          m.yes_token_id,
          m.no_token_id,
          s.best_bid_yes,
          s.best_ask_yes,
          s.best_bid_no,
          s.best_ask_no,
          s.spread_yes,
          s.spread_no,
          s.mid_price_yes,
          s.mid_price_no,
          s.last_trade_price,
          s.last_trade_size,
          s.last_trade_time,
          s.has_orderbook,
          s.has_trade_data,
          s.strict_validation_passed AS snapshot_strict_validation_passed,
          CASE
            WHEN m.close_time IS NULL THEN f.time_until_resolution
            ELSE (julianday(m.close_time) - julianday(f.timestamp)) * 86400.0
          END AS seconds_before_close,
          btc_c.price AS btc_chainlink_price,
          btc_c.exchange_timestamp AS btc_chainlink_exchange_timestamp,
          btc_c.local_arrival_iso AS btc_chainlink_local_arrival_iso,
          CASE
            WHEN btc_c.local_arrival_iso IS NULL THEN NULL
            ELSE MAX(0.0, (julianday(f.timestamp) - julianday(btc_c.local_arrival_iso)) * 86400.0)
          END AS btc_chainlink_age_sec_at_feature,
          btc_cmp.price AS btc_binance_price,
          btc_cmp.exchange_timestamp AS btc_binance_exchange_timestamp,
          btc_cmp.local_arrival_iso AS btc_binance_local_arrival_iso,
          CASE
            WHEN btc_cmp.local_arrival_iso IS NULL THEN NULL
            ELSE MAX(0.0, (julianday(f.timestamp) - julianday(btc_cmp.local_arrival_iso)) * 86400.0)
          END AS btc_binance_age_sec_at_feature,
          CASE
            WHEN btc_c.price IS NULL OR btc_cmp.price IS NULL THEN NULL
            ELSE btc_cmp.price - btc_c.price
          END AS btc_price_diff_binance_minus_chainlink,
          CASE
            WHEN btc_c.price IS NULL OR btc_cmp.price IS NULL OR btc_c.price = 0 THEN NULL
            ELSE (btc_cmp.price - btc_c.price) / btc_c.price
          END AS btc_price_diff_pct_binance_minus_chainlink,
          COALESCE(btc_start_prior.price, btc_start_after.price)
            AS btc_price_at_market_start,
          COALESCE(btc_start_prior.local_arrival_iso, btc_start_after.local_arrival_iso)
            AS btc_price_at_market_start_local_arrival_iso,
          (julianday(?) - julianday(f.timestamp)) * 86400.0 AS feature_age_sec
        FROM features f
        JOIN market_snapshots s
          ON s.run_id = f.run_id
         AND s.market_id = f.market_id
         AND s.timestamp = f.timestamp
        JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        LEFT JOIN btc_prices btc_c
          ON btc_c.id = (
            SELECT b.id
            FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND datetime(b.local_arrival_iso) <= datetime(f.timestamp)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_cmp
          ON btc_cmp.id = (
            SELECT b.id
            FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND datetime(b.local_arrival_iso) <= datetime(f.timestamp)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_start_prior
          ON btc_start_prior.id = (
            SELECT b.id
            FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND m.start_time IS NOT NULL
              AND datetime(b.local_arrival_iso) <= datetime(m.start_time)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_start_after
          ON btc_start_after.id = (
            SELECT b.id
            FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND m.start_time IS NOT NULL
              AND datetime(b.local_arrival_iso) > datetime(m.start_time)
            ORDER BY b.local_arrival_ns ASC, b.id ASC
            LIMIT 1
          )
        WHERE f.run_id = ?
          AND f.market_id = ?
          AND datetime(f.timestamp) <= datetime(?)
        ORDER BY datetime(f.timestamp) DESC, f.timestamp DESC
        LIMIT 1
        """,
        (
            to_iso(now),
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            row["run_id"],
            row["market_id"],
            to_iso(now),
        ),
    ).fetchone()
    if result is None:
        return None
    item = dict(result)
    if item.get("time_until_resolution") is None and item.get("seconds_before_close") is not None:
        item["time_until_resolution"] = item.get("seconds_before_close")
    item["seconds_after_start"] = _seconds_after_start(item)
    return item


def _seconds_after_start(row: dict[str, Any]) -> float | None:
    start = parse_timestamp(row.get("start_time"))
    timestamp = parse_timestamp(row.get("timestamp"))
    if start is None or timestamp is None:
        return None
    return max(0.0, (timestamp - start).total_seconds())


def _close_position(
    output: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    exit_decision: dict[str, Any],
    now: datetime,
) -> bool:
    with output:
        cursor = output.execute(
            """
            UPDATE paper_positions
            SET status = 'closed',
                updated_at = ?,
                exit_time = ?,
                exit_reason = ?,
                exit_price = ?,
                adjusted_exit_price = ?,
                realized_pnl_usd = ?,
                realized_roi = ?
            WHERE id = ?
              AND status = 'open'
            """,
            (
                to_iso(now),
                to_iso(now),
                exit_decision["reason"],
                _round_or_none(exit_decision.get("exit_price")),
                _round_or_none(exit_decision.get("adjusted_exit_price")),
                _round_or_none(exit_decision.get("realized_pnl_usd")),
                _round_or_none(exit_decision.get("realized_roi")),
                int(row["id"]),
            ),
        )
        if int(cursor.rowcount or 0) <= 0:
            return False
        _insert_position_event(
            output,
            position_id=int(row["id"]),
            event_type="closed",
            details=exit_decision,
        )
    return True


def _settle_position(
    output: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    label: int,
    now: datetime,
) -> bool:
    side = str(row["side"])
    stake = float(row["stake_usd"] or 0.0)
    adjusted_entry = float(row["adjusted_entry_price"] or 0.0)
    shares = float(row["shares"] or 0.0)
    payout_per_share = 1.0 if (side == "YES" and label == 1) or (side == "NO" and label == 0) else 0.0
    payout = shares * payout_per_share
    pnl = payout - stake if adjusted_entry > 0 else 0.0
    roi = pnl / stake if stake else 0.0
    with output:
        cursor = output.execute(
            """
            UPDATE paper_positions
            SET status = 'settled',
                updated_at = ?,
                exit_time = COALESCE(exit_time, ?),
                exit_reason = COALESCE(exit_reason, 'settlement'),
                resolved_label = ?,
                settlement_pnl_usd = ?,
                settlement_roi = ?
            WHERE id = ?
              AND status IN ('open', 'awaiting_resolution')
            """,
            (
                to_iso(now),
                to_iso(now),
                "YES" if label == 1 else "NO",
                round(pnl, 10),
                round(roi, 10),
                int(row["id"]),
            ),
        )
        if int(cursor.rowcount or 0) <= 0:
            return False
        _insert_position_event(
            output,
            position_id=int(row["id"]),
            event_type="settled",
            details={"label": "YES" if label == 1 else "NO", "settlement_pnl_usd": pnl},
        )
    return True


def _mark_position_awaiting_resolution(
    output: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: datetime,
) -> None:
    with output:
        output.execute(
            """
            UPDATE paper_positions
            SET status = 'awaiting_resolution',
                updated_at = ?
            WHERE id = ?
              AND status = 'open'
            """,
            (to_iso(now), int(row["id"])),
        )
        _insert_position_event(
            output,
            position_id=int(row["id"]),
            event_type="awaiting_resolution",
            details={},
        )


def _insert_position_event(
    conn: sqlite3.Connection,
    *,
    position_id: int,
    event_type: str,
    details: dict[str, Any],
) -> None:
    row = conn.execute(
        "SELECT run_id, strategy_id, market_id FROM paper_positions WHERE id = ?",
        (int(position_id),),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO paper_position_events (
            created_at, position_id, run_id, strategy_id, market_id, event_type, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_iso(datetime.now(timezone.utc)),
            int(position_id),
            row["run_id"] if row is not None else None,
            row["strategy_id"] if row is not None else None,
            row["market_id"] if row is not None else None,
            event_type,
            json.dumps(details, sort_keys=True, default=str),
        ),
    )


def _latest_snapshot_for_position(
    recorder: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: datetime,
) -> sqlite3.Row | None:
    return recorder.execute(
        """
        SELECT *
        FROM market_snapshots
        WHERE run_id = ?
          AND market_id = ?
          AND datetime(timestamp) <= datetime(?)
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        LIMIT 1
        """,
        (row["run_id"], row["market_id"], to_iso(now)),
    ).fetchone()


def _entry_price(row: dict[str, Any], *, direction: str, allow_mid: bool) -> float | None:
    ask_key = "best_ask_yes" if direction == "YES" else "best_ask_no"
    price = _valid_price(row.get(ask_key))
    if price is not None or not allow_mid:
        return price
    mid_key = "mid_price_yes" if direction == "YES" else "mid_price_no"
    return _valid_price(row.get(mid_key))


def _exit_price(row: sqlite3.Row | dict[str, Any], *, side: str) -> float | None:
    key = "best_bid_yes" if side == "YES" else "best_bid_no"
    return _valid_price(row[key] if isinstance(row, sqlite3.Row) and key in row.keys() else row.get(key))


def _valid_price(value: Any) -> float | None:
    price = _float_or_none(value)
    if price is None or price <= 0 or price >= 1:
        return None
    return price


def _strategy_directions(strategy: IntramarketStrategyDefinition) -> list[str]:
    if strategy.direction_mode == "YES_ONLY":
        return ["YES"]
    if strategy.direction_mode == "NO_ONLY":
        return ["NO"]
    return ["YES", "NO"]


def _predict_once(
    model: Any,
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[tuple[str, str, str], float]:
    if not rows:
        return {}
    probabilities = _predict_probabilities(model, rows, feature_columns)
    return {_feature_key(row): float(probability) for row, probability in zip(rows, probabilities)}


def _feature_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(row.get("timestamp") or ""),
    )


def _latest_feature_timestamp(rows: Sequence[dict[str, Any]]) -> str | None:
    timestamps = [str(row.get("timestamp") or "") for row in rows if row.get("timestamp")]
    return max(timestamps) if timestamps else None


def _candidate_exists(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    run_id: str,
    market_id: str,
    feature_timestamp: str,
    direction: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM paper_intramarket_candidates
        WHERE strategy_id = ?
          AND COALESCE(run_id, '') = ?
          AND market_id = ?
          AND feature_timestamp = ?
          AND candidate_direction = ?
        LIMIT 1
        """,
        (strategy_id, run_id, market_id, feature_timestamp, direction),
    ).fetchone()
    return row is not None


def _open_position_count(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    run_id: str | None = None,
    market_id: str | None = None,
) -> int:
    where = ["strategy_id = ?", "status = 'open'"]
    params: list[Any] = [strategy_id]
    if run_id is not None:
        where.append("COALESCE(run_id, '') = ?")
        params.append(run_id)
    if market_id is not None:
        where.append("market_id = ?")
        params.append(market_id)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM paper_positions WHERE {' AND '.join(where)}",
            params,
        ).fetchone()[0]
        or 0
    )


def _open_stake(conn: sqlite3.Connection, *, strategy_id: str) -> float:
    return float(
        conn.execute(
            """
            SELECT COALESCE(SUM(stake_usd), 0)
            FROM paper_positions
            WHERE strategy_id = ?
              AND status = 'open'
            """,
            (strategy_id,),
        ).fetchone()[0]
        or 0.0
    )


def _any_position_for_market(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    run_id: str,
    market_id: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM paper_positions
        WHERE strategy_id = ?
          AND COALESCE(run_id, '') = ?
          AND market_id = ?
        LIMIT 1
        """,
        (strategy_id, run_id, market_id),
    ).fetchone()
    return row is not None


def _within_cooldown(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    run_id: str,
    market_id: str,
    now: datetime,
    cooldown_sec: float,
) -> bool:
    row = conn.execute(
        """
        SELECT exit_time
        FROM paper_positions
        WHERE strategy_id = ?
          AND COALESCE(run_id, '') = ?
          AND market_id = ?
          AND exit_time IS NOT NULL
        ORDER BY datetime(exit_time) DESC, exit_time DESC
        LIMIT 1
        """,
        (strategy_id, run_id, market_id),
    ).fetchone()
    if row is None:
        return False
    exit_time = parse_timestamp(row["exit_time"])
    return exit_time is not None and (now - exit_time).total_seconds() < cooldown_sec


def _position_count(conn: sqlite3.Connection, strategy_id: str, status: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE strategy_id = ? AND status = ?",
            (strategy_id, status),
        ).fetchone()[0]
        or 0
    )


def _status_count(conn: sqlite3.Connection, status: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE status = ?",
            (status,),
        ).fetchone()[0]
        or 0
    )


def _insert_intramarket_heartbeat(conn: sqlite3.Connection, heartbeat: dict[str, Any]) -> None:
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_intramarket_heartbeats (
                timestamp, run_id, active_strategies, latest_feature_timestamp,
                open_positions, closed_positions, awaiting_resolution_positions,
                settled_positions, candidates_logged, positions_opened,
                skips_logged, strategy_stats_json, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                heartbeat["timestamp"],
                heartbeat.get("run_id"),
                heartbeat.get("active_strategies"),
                heartbeat.get("latest_feature_timestamp"),
                heartbeat.get("open_positions"),
                heartbeat.get("closed_positions"),
                heartbeat.get("awaiting_resolution_positions"),
                heartbeat.get("settled_positions"),
                heartbeat.get("candidates_logged"),
                heartbeat.get("positions_opened"),
                heartbeat.get("skips_logged"),
                json.dumps(heartbeat.get("strategy_stats") or [], sort_keys=True),
                json.dumps(heartbeat, sort_keys=True, default=str),
            ),
        )


def _intramarket_summary(
    positions: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    closed_or_settled = [row for row in positions if row.get("status") in {"closed", "settled"}]
    combined_pnls = [_combined_pnl(row) for row in closed_or_settled]
    wins = [pnl for pnl in combined_pnls if pnl is not None and pnl > 0]
    return {
        "total_positions": len(positions),
        "open_positions": sum(1 for row in positions if row.get("status") == "open"),
        "closed_positions": sum(1 for row in positions if row.get("status") == "closed"),
        "awaiting_resolution_positions": sum(1 for row in positions if row.get("status") == "awaiting_resolution"),
        "settled_positions": sum(1 for row in positions if row.get("status") == "settled"),
        "total_candidates": len(candidates),
        "realized_closed_pnl": _round(sum(float(row.get("realized_pnl_usd") or 0.0) for row in positions)),
        "settlement_pnl": _round(sum(float(row.get("settlement_pnl_usd") or 0.0) for row in positions)),
        "combined_pnl": _round(sum(float(pnl or 0.0) for pnl in combined_pnls)),
        "win_rate": _rate(len(wins), len(closed_or_settled)),
        "average_roi": _mean([_combined_roi(row) for row in closed_or_settled if _combined_roi(row) is not None]),
        "average_hold_seconds": _mean([_hold_seconds(row) for row in closed_or_settled if _hold_seconds(row) is not None]),
        "trades_by_exit_reason": dict(Counter(str(row.get("exit_reason") or "open") for row in positions)),
        "pnl_by_exit_type": _pnl_breakdown(positions, "exit_type"),
        "pnl_by_strategy_id": _pnl_breakdown(positions, "strategy_id"),
        "pnl_by_time_until_resolution_bucket": _bucket_breakdown(positions, _time_bucket),
        "pnl_by_probability_bucket": _bucket_breakdown(positions, _probability_bucket),
        "pnl_by_entry_price_bucket": _bucket_breakdown(positions, _entry_price_bucket),
        "open_exposure": _round(sum(float(row.get("stake_usd") or 0.0) for row in positions if row.get("status") == "open")),
        "stale_open_positions": _stale_open_positions(positions),
    }


def _pnl_breakdown(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: _group_metrics(group) for name, group in sorted(groups.items())}


def _bucket_breakdown(rows: Sequence[dict[str, Any]], bucket_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[bucket_fn(row)].append(row)
    return {name: _group_metrics(group) for name, group in sorted(groups.items())}


def _group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if row.get("status") in {"closed", "settled"}]
    pnls = [_combined_pnl(row) for row in closed]
    wins = [pnl for pnl in pnls if pnl is not None and pnl > 0]
    return {
        "count": len(rows),
        "closed": len(closed),
        "pnl": _round(sum(float(pnl or 0.0) for pnl in pnls)),
        "average_roi": _mean([_combined_roi(row) for row in closed if _combined_roi(row) is not None]),
        "win_rate": _rate(len(wins), len(closed)),
    }


def _enrich_position(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["combined_pnl"] = _combined_pnl(row)
    enriched["combined_roi"] = _combined_roi(row)
    enriched["hold_seconds"] = _hold_seconds(row)
    return enriched


def _intramarket_export_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    pnl = _float_or_none(row.get("realized_pnl_usd"))
    if pnl is None:
        pnl = _float_or_none(row.get("settlement_pnl_usd"))
    roi = _float_or_none(row.get("realized_roi"))
    if roi is None:
        roi = _float_or_none(row.get("settlement_roi"))
    if roi is None:
        roi = _float_or_none(row.get("would_have_roi"))
    result["realized_or_would_have_pnl_usd"] = pnl if pnl is not None else row.get("would_have_pnl_usd")
    result["realized_or_would_have_roi"] = roi
    result["profitable"] = None if roi is None else int(roi > 0)
    result["traded"] = int(str(row.get("decision") or "").upper() == "TRADE")
    return result


def _candidate_settlement(
    candidate: sqlite3.Row | dict[str, Any],
    *,
    label: int,
) -> tuple[float | None, float | None, float | None]:
    direction = str(candidate["candidate_direction"] or "").upper()
    adjusted_entry = _float_or_none(candidate["candidate_adjusted_entry_price"])
    stake = _float_or_none(candidate["stake_usd"])
    if direction not in {"YES", "NO"} or adjusted_entry is None or adjusted_entry <= 0 or stake is None:
        return None, None, None
    shares = stake / adjusted_entry
    payout_per_share = 1.0 if (direction == "YES" and label == 1) or (direction == "NO" and label == 0) else 0.0
    payout = shares * payout_per_share
    pnl = payout - stake
    roi = pnl / stake if stake else 0.0
    return round(payout, 10), round(pnl, 10), round(roi, 10)


def _combined_pnl(row: dict[str, Any]) -> float | None:
    realized = _float_or_none(row.get("realized_pnl_usd"))
    settlement = _float_or_none(row.get("settlement_pnl_usd"))
    if realized is None and settlement is None:
        return None
    return _round(float(realized or 0.0) + float(settlement or 0.0))


def _combined_roi(row: dict[str, Any]) -> float | None:
    realized = _float_or_none(row.get("realized_roi"))
    settlement = _float_or_none(row.get("settlement_roi"))
    if realized is not None:
        return realized
    return settlement


def _hold_seconds(row: dict[str, Any]) -> float | None:
    entry = parse_timestamp(row.get("entry_time"))
    exit_time = parse_timestamp(row.get("exit_time"))
    if entry is None or exit_time is None:
        return None
    return max(0.0, (exit_time - entry).total_seconds())


def _time_bucket(row: dict[str, Any]) -> str:
    value = _float_or_none(row.get("time_until_resolution_at_entry"))
    if value is None:
        return "unknown"
    if value < 30:
        return "0-30s"
    if value < 60:
        return "30-60s"
    if value < 120:
        return "60-120s"
    if value < 180:
        return "120-180s"
    return "180s+"


def _probability_bucket(row: dict[str, Any]) -> str:
    value = _float_or_none(row.get("probability_for_direction_at_entry"))
    if value is None:
        value = _float_or_none(row.get("probability_for_direction"))
    if value is None:
        return "unknown"
    lower = int(max(0, min(9, int(value * 10)))) / 10
    return f"{lower:.1f}-{lower + 0.1:.1f}"


def _entry_price_bucket(row: dict[str, Any]) -> str:
    value = _float_or_none(row.get("adjusted_entry_price"))
    if value is None:
        value = _float_or_none(row.get("candidate_adjusted_entry_price"))
    if value is None:
        return "unknown"
    lower = int(max(0, min(9, int(value * 10)))) / 10
    return f"{lower:.1f}-{lower + 0.1:.1f}"


def _stale_open_positions(rows: Sequence[dict[str, Any]], *, threshold_sec: float = 300.0) -> int:
    now = datetime.now(timezone.utc)
    count = 0
    for row in rows:
        if row.get("status") != "open":
            continue
        entry = parse_timestamp(row.get("entry_time"))
        if entry is not None and (now - entry).total_seconds() > threshold_sec:
            count += 1
    return count


def _max_drawdown(pnls: Sequence[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        cumulative += float(pnl)
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return round(max_drawdown, 10)


def _reliability_tier(count: int) -> str:
    if count < 30:
        return "too_early"
    if count < 100:
        return "early"
    if count < 300:
        return "usable_sample"
    return "strong_sample"


def _write_intramarket_report_files(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intramarket_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary = report.get("summary") or {}
    lines = [
        f"status={report.get('status')}",
        f"total_positions={summary.get('total_positions', 0)}",
        f"combined_pnl={summary.get('combined_pnl')}",
        f"win_rate={summary.get('win_rate')}",
        f"open_exposure={summary.get('open_exposure')}",
    ]
    (output_dir / "intramarket_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_parquet(rows: Sequence[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = _normalize_rows(rows)
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({})
    pq.write_table(table, path)


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    normalized = _normalize_rows(rows)
    fieldnames = list(normalized[0].keys()) if normalized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(normalized)


def _normalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return [{column: _scalar(row.get(column)) for column in columns} for row in rows]


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _connect_read_only_if_exists(db_path: str) -> sqlite3.Connection | None:
    path = Path(db_path)
    if not path.exists():
        return None
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _required_string(item: dict[str, Any], key: str, *, index: int) -> str:
    value = item.get(key)
    if value is None or not str(value).strip():
        raise IntramarketPaperTraderError(f"strategy at index {index} missing required {key}")
    return str(value).strip()


def _float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _round(value: Any) -> float | None:
    value = _float_or_none(value)
    return None if value is None else round(value, 10)


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 10)


def _strategy_snapshot(strategy: IntramarketStrategyDefinition) -> dict[str, Any]:
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_name": strategy.strategy_name,
        "direction_mode": strategy.direction_mode,
        "entry_type": strategy.entry_type,
        "exit_type": strategy.exit_type,
        "min_probability_for_direction": strategy.min_probability_for_direction,
        "max_probability_for_direction": strategy.max_probability_for_direction,
        "min_estimated_edge": strategy.min_estimated_edge,
        "stake_usd": strategy.stake_usd,
        "entry_slippage_cents": strategy.entry_slippage_cents,
        "exit_slippage_cents": strategy.exit_slippage_cents,
        "fee_cents": strategy.fee_cents,
        "take_profit_roi": strategy.take_profit_roi,
        "stop_loss_roi": strategy.stop_loss_roi,
        "max_hold_seconds": strategy.max_hold_seconds,
        "min_hold_seconds": strategy.min_hold_seconds,
        "exit_probability_below": strategy.exit_probability_below,
        "exit_probability_drop_from_entry": strategy.exit_probability_drop_from_entry,
        "take_profit_cents": strategy.take_profit_cents,
        "stop_loss_cents": strategy.stop_loss_cents,
        "max_hold_sec": strategy.max_hold_sec,
    }


def _emit(enabled: bool, event: str, **fields: Any) -> None:
    if not enabled:
        return
    print(json.dumps({"event": event, **fields}, sort_keys=True, default=str))


__all__ = [
    "IntramarketPaperTraderError",
    "IntramarketStrategyDefinition",
    "build_intramarket_paper_analytics_report",
    "build_intramarket_strategy_leaderboard",
    "ensure_intramarket_schema",
    "export_intramarket_strategy_dataset",
    "load_intramarket_strategy_config",
    "render_intramarket_strategy_leaderboard",
    "run_intramarket_paper_trader",
    "run_intramarket_paper_trader_once",
    "settle_intramarket_candidates",
    "update_intramarket_positions",
]
