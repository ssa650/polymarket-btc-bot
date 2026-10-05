from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .backtest_baseline_strategy import (
    _direction_for_probability,
    _load_feature_columns,
    _load_model,
    _predict_probabilities,
)
from .baseline_paper_trader import (
    BaselinePaperTraderConfig,
    connect_paper_output_db,
    connect_recorder_read_only,
    ensure_paper_schema,
    fetch_latest_candidate_rows,
    settle_open_paper_trades,
)
from .baseline_paper_trader import (
    _filter_candidate_eligibility,
    _filter_prediction_prerequisites,
    _live_entry_price,
    _model_name,
    _paper_status_count,
    _probability_for_direction,
    _process_signal_row,
    _record_skipped_trade,
    _record_trade_candidate,
    _round_or_none,
)
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none


_STRATEGY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_DIRECTION_MODES = {"YES_ONLY", "NO_ONLY", "BOTH"}
_EXIT_TYPES = {
    "HOLD_TO_RESOLUTION",
    "FIXED_HORIZON_EXIT",
    "TAKE_PROFIT_STOP_LOSS",
    "TRAILING_STOP",
}
_YES_TOP_ASK_SIZE_FIELDS = (
    "best_ask_size_yes",
    "yes_best_ask_size",
    "top_ask_size_yes",
    "top_of_book_ask_size_yes",
    "ask_size_yes",
    "yes_ask_size",
)
_NO_TOP_ASK_SIZE_FIELDS = (
    "best_ask_size_no",
    "no_best_ask_size",
    "top_ask_size_no",
    "top_of_book_ask_size_no",
    "ask_size_no",
    "no_ask_size",
)
_REALISTIC_EXECUTION_SKIP_REASONS = {
    "quote_after_latency_missing",
    "entry_price_drift_too_high",
    "spread_too_wide",
    "btc_stale",
    "feature_stale",
}
_LIVE_ORDER_ACTIVE_STATUSES = {"dry_run", "submitted"}
_LIQUIDITY_REGIMES = {"thin", "normal", "deep", "unknown"}


class MultiStrategyPaperTraderError(ValueError):
    pass


@dataclass(slots=True)
class MultiStrategyDefinition:
    strategy_id: str
    strategy_name: str
    enabled: bool
    direction_mode: str
    long_threshold: float
    short_threshold: float
    min_estimated_edge: float
    entry_slippage_cents: float
    fee_cents: float
    stake_usd: float
    max_open_trades: int
    one_trade_per_market: bool
    min_time_until_resolution_sec: float
    max_time_until_resolution_sec: float
    block_probability_above: float | None = None
    block_probability_below: float | None = None
    require_positive_edge: bool = True
    min_probability_for_direction: float | None = None
    max_probability_for_direction: float | None = None
    exit_type: str = "HOLD_TO_RESOLUTION"
    exit_slippage_cents: float = 0.0
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    fixed_horizon_exit_sec: float | None = None
    trailing_stop_pct: float | None = None
    liquidity_fill_check_enabled: bool = False
    max_top_of_book_fill_fraction: float = 1.0
    min_top_of_book_shares: float = 0.0
    skip_if_liquidity_missing: bool = True
    blocked_liquidity_regimes: tuple[str, ...] = ()


@dataclass(slots=True)
class MultiStrategyBankrollSettings:
    bankroll_enabled: bool = False
    starting_bankroll_usd: float = 100.0
    base_risk_fraction: float = 0.01
    max_risk_fraction: float = 0.05
    max_total_exposure_fraction: float = 0.20
    min_stake_usd: float = 0.25
    max_stake_usd: float = 5.0
    stop_trading_on_bankroll_depleted: bool = True
    reset_bankroll_on_new_run_id: bool = True


@dataclass(slots=True)
class _BankrollSizingDecision:
    allowed: bool
    stake_usd: float | None
    rejection_reason: str | None
    starting_bankroll_usd: float
    bankroll_before_trade: float
    available_cash_before_trade: float
    open_exposure_before_trade: float
    stake_fraction_of_bankroll: float | None
    max_open_exposure_usd: float
    bankroll_after_trade: float
    bankroll_status: str
    blown_up_at: str | None
    risk_sizing_reason: str


@dataclass(slots=True)
class _LiquidityFillDecision:
    allowed: bool
    rejection_reason: str | None
    requested_shares: float | None
    top_of_book_shares: float | None
    max_fillable_shares: float | None
    liquidity_fill_fraction_used: float
    liquidity_check_passed: int | None


@dataclass(slots=True)
class MultiStrategyRealisticExecutionSettings:
    realistic_execution_enabled: bool = False
    entry_latency_sec: float = 1.0
    exit_latency_sec: float = 1.0
    max_quote_wait_sec: float = 3.0
    max_entry_price_drift_cents: float = 2.0
    max_exit_price_drift_cents: float = 2.0
    require_quote_after_latency: bool = True
    reject_if_spread_above_cents: float = 4.0
    reject_if_btc_age_above_sec: float = 5.0
    reject_if_feature_age_above_sec: float = 5.0
    apply_extra_slippage_cents: float = 1.0
    partial_fill_enabled: bool = False


@dataclass(slots=True)
class MultiStrategyTradeSelectionSettings:
    selection_enabled: bool = False
    mode: str = "best_per_market_direction"
    score_field: str = "estimated_edge"
    max_new_trades_per_market: int = 1
    max_new_trades_per_market_direction: int = 1
    allow_multiple_strategy_variants_same_market: bool = False


@dataclass(slots=True)
class MultiStrategyLiveTradingSettings:
    live_trading_enabled: bool = False
    dry_run_orders: bool = True
    max_order_usd: float = 1.0
    max_daily_loss_usd: float = 5.0
    max_daily_orders: int = 20
    max_open_exposure_usd: float = 5.0
    max_open_trades: int = 3
    max_trades_per_market: int = 1
    max_trades_per_market_direction: int = 1
    require_realistic_execution_passed: bool = True
    require_liquidity_check_passed: bool = True
    reject_if_btc_age_above_sec: float = 5.0
    reject_if_feature_age_above_sec: float = 5.0
    reject_if_spread_above_cents: float = 3.0
    kill_switch_file: str = "data/KILL_LIVE_TRADING"
    allow_market_order: bool = False
    use_limit_orders_only: bool = True


LiveOrderSubmitter = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class _RealisticEntryDecision:
    allowed: bool
    rejection_reason: str | None
    row: dict[str, Any]
    edge_inputs: dict[str, float | None]
    signal_entry_price: float | None
    delayed_entry_price: float | None
    entry_price_drift: float | None
    spread_cents_at_entry: float | None
    extra_slippage_cents_applied: float


@dataclass(slots=True)
class _PendingRealisticEntry:
    strategy: MultiStrategyDefinition
    config: BaselinePaperTraderConfig
    row: dict[str, Any]
    probability: float
    direction: str
    edge_inputs: dict[str, float | None]
    bankroll_decision: _BankrollSizingDecision | None
    model_name: str
    target_quote_timestamp: datetime
    created_at: datetime


@dataclass(slots=True)
class _TradeSelectionCandidate:
    strategy: MultiStrategyDefinition
    config: BaselinePaperTraderConfig
    row: dict[str, Any]
    probability: float
    direction: str
    edge_inputs: dict[str, float | None]
    model_name: str


def load_multi_strategy_config(config_path: str | Path) -> list[MultiStrategyDefinition]:
    payload = _load_config_payload(config_path)
    (
        strategies,
        _bankroll_settings,
        _realistic_execution_settings,
        _trade_selection_settings,
        _live_trading_settings,
    ) = _parse_multi_strategy_payload(payload)
    return strategies


def _load_multi_strategy_config_and_runtime_settings(
    config_path: str | Path,
) -> tuple[
    list[MultiStrategyDefinition],
    MultiStrategyBankrollSettings,
    MultiStrategyRealisticExecutionSettings,
    MultiStrategyTradeSelectionSettings,
    MultiStrategyLiveTradingSettings,
]:
    payload = _load_config_payload(config_path)
    return _parse_multi_strategy_payload(payload)


def _load_config_payload(config_path: str | Path) -> Any:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_multi_strategy_payload(
    payload: Any,
) -> tuple[
    list[MultiStrategyDefinition],
    MultiStrategyBankrollSettings,
    MultiStrategyRealisticExecutionSettings,
    MultiStrategyTradeSelectionSettings,
    MultiStrategyLiveTradingSettings,
]:
    raw_strategies = payload.get("strategies") if isinstance(payload, dict) else payload
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise MultiStrategyPaperTraderError("config must contain a non-empty strategies list")
    global_liquidity = _global_liquidity_settings(payload)
    strategies = [
        _parse_strategy_definition(
            _merge_global_liquidity_settings(item, global_liquidity),
            index=index,
        )
        for index, item in enumerate(raw_strategies)
    ]
    return (
        [strategy for strategy in strategies if strategy.enabled],
        _parse_bankroll_settings(payload),
        _parse_realistic_execution_settings(payload),
        _parse_trade_selection_settings(payload),
        _parse_live_trading_settings(payload),
    )


def run_multi_strategy_paper_trader(
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
    require_fresh_comparison_btc: bool = False,
    max_iterations: int | None = None,
    heartbeat_detail: str = "full",
    emit_logs: bool = True,
    live_order_submitter: LiveOrderSubmitter | None = None,
    min_feature_timestamp: str | datetime | None = None,
    live_start_now: bool = False,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    effective_min_feature_timestamp = _effective_min_feature_timestamp(
        min_feature_timestamp,
        live_start_now=live_start_now,
        now=started_at,
    )
    (
        strategies,
        bankroll_settings,
        realistic_execution_settings,
        trade_selection_settings,
        live_trading_settings,
    ) = _load_multi_strategy_config_and_runtime_settings(config_path)
    if not strategies:
        raise MultiStrategyPaperTraderError("no enabled strategies found")
    model = _load_model(model_path)
    feature_columns = _load_feature_columns(feature_columns_path)
    recorder = connect_recorder_read_only(recorder_db_path)
    output = connect_paper_output_db(output_db_path)
    try:
        ensure_paper_schema(output)
        ensure_multi_strategy_schema(output)
        _emit(
            emit_logs,
            "multi_strategy_paper_trader_started",
            run_id=run_id,
            recorder_db_path=recorder_db_path,
            output_db_path=output_db_path,
            model_path=model_path,
            active_strategies=len(strategies),
            strategy_ids=[strategy.strategy_id for strategy in strategies],
            min_feature_timestamp=effective_min_feature_timestamp,
            live_start_now=bool(live_start_now),
        )
        startup_report = _write_multi_strategy_startup_heartbeat(
            output,
            run_id=run_id,
            active_strategies=len(strategies),
            min_feature_timestamp=effective_min_feature_timestamp,
            heartbeat_detail=heartbeat_detail,
            now=started_at,
        )
        _emit(
            emit_logs,
            "multi_strategy_paper_trader_heartbeat",
            **_heartbeat_log_payload(startup_report, detail=heartbeat_detail),
        )
        if max_iterations is not None and int(max_iterations) <= 0:
            return {
                "status": "ok",
                "iterations": 0,
                "output_db_path": output_db_path,
                "last_poll": startup_report,
            }
        if live_trading_settings.live_trading_enabled:
            _emit(
                emit_logs,
                "multi_strategy_live_trading_warning",
                live_trading_enabled=True,
                dry_run_orders=live_trading_settings.dry_run_orders,
                live_submitter_available=live_order_submitter is not None,
                message=(
                    "live_trading_enabled=true; safety wrapper is active. "
                    "No real orders are submitted when dry_run_orders=true or no submitter is configured."
                ),
            )
        iterations = 0
        last_report: dict[str, Any] | None = None
        pending_realistic_entries: dict[
            tuple[str, str, str],
            _PendingRealisticEntry,
        ] = {}
        while True:
            iterations += 1
            last_report = run_multi_strategy_paper_trader_once(
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
                require_fresh_comparison_btc=require_fresh_comparison_btc,
                bankroll_settings=bankroll_settings,
                realistic_execution_settings=realistic_execution_settings,
                trade_selection_settings=trade_selection_settings,
                live_trading_settings=live_trading_settings,
                live_order_submitter=live_order_submitter,
                pending_realistic_entries=pending_realistic_entries,
                heartbeat_detail=heartbeat_detail,
                emit_logs=emit_logs,
                min_feature_timestamp=effective_min_feature_timestamp,
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


def _effective_min_feature_timestamp(
    min_feature_timestamp: str | datetime | None,
    *,
    live_start_now: bool,
    now: datetime,
) -> str | None:
    parsed = parse_timestamp(min_feature_timestamp)
    if min_feature_timestamp is not None and parsed is None:
        raise MultiStrategyPaperTraderError(
            f"invalid min_feature_timestamp {min_feature_timestamp!r}; expected ISO timestamp"
        )
    if parsed is None and live_start_now:
        parsed = now
    return to_iso(parsed) if parsed is not None else None


def _write_multi_strategy_startup_heartbeat(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    active_strategies: int,
    min_feature_timestamp: str | None,
    heartbeat_detail: str,
    now: datetime,
) -> dict[str, Any]:
    report = {
        "status": "starting",
        "run_id": run_id,
        "timestamp": to_iso(now),
        "active_strategies": int(active_strategies),
        "min_feature_timestamp": min_feature_timestamp,
        "latest_market_id": None,
        "latest_feature_timestamp": min_feature_timestamp,
        "probability_yes": None,
        "candidate_rows_seen": 0,
        "prediction_rows": 0,
        "candidates_logged": 0,
        "candidate_rows_total": _candidate_count(conn),
        "candidate_rows_written_this_poll": 0,
        "candidate_decisions_evaluated_this_poll": 0,
        "trades_opened": 0,
        "trades_closed": 0,
        "exit_counts": {},
        "skips_logged": 0,
        "settled_trades": _paper_status_count(conn, "settled"),
        "missing_feature_columns": [],
        "strategy_errors": {},
        "liquidity_fill_check_enabled": False,
        "liquidity_checked_count": 0,
        "liquidity_passed_count": 0,
        "liquidity_blocked_count": 0,
        "liquidity_missing_count": 0,
        "trade_selection_enabled": False,
        "trade_selection_candidates_before": 0,
        "trade_selection_candidates_after": 0,
        "trade_selection_suppressed_count": 0,
        "live_trading": {},
        "pending_realistic_entries": 0,
        "pending_realistic_entries_ready": 0,
        "pending_realistic_entries_expired": 0,
        "realistic_entries_opened_after_latency": 0,
        "realistic_entries_skipped_after_latency": 0,
        "latest_pending_target_quote_timestamp": None,
        "bankroll": {},
        "heartbeat_detail": heartbeat_detail,
        "strategies": [],
        "strategy_stats": [],
        "strategy_stats_summary": _summarize_strategy_stats([]),
    }
    _write_multi_strategy_heartbeat(conn, report)
    return report


def run_multi_strategy_paper_trader_once(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    model: Any,
    feature_columns: Sequence[str],
    strategies: Sequence[MultiStrategyDefinition],
    recorder_db_path: str,
    output_db_path: str,
    model_path: str,
    feature_columns_path: str,
    run_id: str | None,
    max_btc_age_sec: float,
    max_feature_age_sec: float,
    require_fresh_comparison_btc: bool = False,
    bankroll_settings: MultiStrategyBankrollSettings | None = None,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings | None = None,
    trade_selection_settings: MultiStrategyTradeSelectionSettings | None = None,
    live_trading_settings: MultiStrategyLiveTradingSettings | None = None,
    live_order_submitter: LiveOrderSubmitter | None = None,
    pending_realistic_entries: dict[tuple[str, str, str], _PendingRealisticEntry] | None = None,
    heartbeat_detail: str = "full",
    emit_logs: bool = True,
    now: datetime | None = None,
    min_feature_timestamp: str | datetime | None = None,
) -> dict[str, Any]:
    ensure_paper_schema(output)
    ensure_multi_strategy_schema(output)
    now = now or datetime.now(timezone.utc)
    bankroll_settings = bankroll_settings or MultiStrategyBankrollSettings()
    realistic_execution_settings = (
        realistic_execution_settings or MultiStrategyRealisticExecutionSettings()
    )
    trade_selection_settings = trade_selection_settings or MultiStrategyTradeSelectionSettings()
    live_trading_settings = live_trading_settings or MultiStrategyLiveTradingSettings()
    if pending_realistic_entries is None:
        pending_realistic_entries = {}
    processed_pending_keys: set[tuple[str, str, str]] = set()
    exit_counts = update_multi_strategy_trade_exits(
        recorder,
        output,
        strategies=strategies,
        realistic_execution_settings=realistic_execution_settings,
        now=now,
        emit_logs=emit_logs,
    )
    pending_stats = _process_pending_realistic_entries(
        recorder,
        output,
        pending_realistic_entries,
        realistic_execution_settings=realistic_execution_settings,
        now=now,
        emit_logs=emit_logs,
        processed_keys=processed_pending_keys,
        live_trading_settings=live_trading_settings,
        live_order_submitter=live_order_submitter,
    )
    settled = settle_open_paper_trades(
        recorder,
        output,
        model_path=model_path,
        emit_logs=emit_logs,
        now=now,
    )
    candidates = fetch_latest_candidate_rows(
        recorder,
        max_btc_age_sec=max_btc_age_sec,
        min_time_until_resolution_sec=0.0,
        max_time_until_resolution_sec=max(
            [strategy.max_time_until_resolution_sec for strategy in strategies] + [300.0]
        ),
        now=now,
        max_feature_age_sec=max_feature_age_sec,
        min_feature_timestamp=min_feature_timestamp,
    )
    base_config = BaselinePaperTraderConfig(
        recorder_db_path=recorder_db_path,
        output_db_path=output_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        max_btc_age_sec=max_btc_age_sec,
        max_feature_age_sec=max_feature_age_sec,
        min_time_until_resolution_sec=0.0,
        max_time_until_resolution_sec=1_000_000.0,
        require_fresh_comparison_btc=require_fresh_comparison_btc,
    )
    live_rows, live_skip_counts, _latest_rejection = _filter_candidate_eligibility(
        candidates,
        config=base_config,
        now=now,
        emit_logs=emit_logs,
    )
    prediction_rows, prerequisite_counts, _prereq_rejection = _filter_prediction_prerequisites(
        live_rows,
        feature_columns,
        emit_logs=emit_logs,
    )
    if _liquidity_checks_enabled(strategies):
        prediction_rows = _enrich_top_of_book_liquidity(recorder, prediction_rows)
    missing_columns = _missing_feature_columns(prediction_rows, feature_columns)
    model_name = _model_name(model)
    probability_cache = _predict_once_per_feature_row(
        model,
        prediction_rows if not missing_columns else [],
        feature_columns,
    )
    candidates_before = _candidate_count(output)
    trades_before = _trade_count(output)
    total_skips = sum(live_skip_counts.values()) + sum(prerequisite_counts.values())
    loop_result = _run_strategy_evaluation_loop(
        recorder,
        output,
        prediction_rows=prediction_rows,
        probability_cache=probability_cache,
        strategies=strategies,
        recorder_db_path=recorder_db_path,
        output_db_path=output_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        max_btc_age_sec=max_btc_age_sec,
        max_feature_age_sec=max_feature_age_sec,
        require_fresh_comparison_btc=require_fresh_comparison_btc,
        missing_columns=missing_columns,
        model_name=model_name,
        bankroll_settings=bankroll_settings,
        realistic_execution_settings=realistic_execution_settings,
        trade_selection_settings=trade_selection_settings,
        live_trading_settings=live_trading_settings,
        live_order_submitter=live_order_submitter,
        pending_realistic_entries=pending_realistic_entries,
        processed_pending_keys=processed_pending_keys,
        run_id=run_id,
        now=now,
        emit_logs=emit_logs,
    )
    per_strategy = loop_result["per_strategy"]
    total_skips += int(loop_result["skipped"])
    strategy_errors = loop_result["strategy_errors"]
    liquidity_aggregate = loop_result["liquidity_aggregate"]
    trade_selection_report = loop_result["trade_selection"]

    followup_pending_stats = _process_pending_realistic_entries(
        recorder,
        output,
        pending_realistic_entries,
        realistic_execution_settings=realistic_execution_settings,
        now=now,
        emit_logs=emit_logs,
        processed_keys=processed_pending_keys,
        live_trading_settings=live_trading_settings,
        live_order_submitter=live_order_submitter,
    )
    pending_stats = _merge_pending_stats(pending_stats, followup_pending_stats)
    candidates_after = _candidate_count(output)
    trades_after = _trade_count(output)
    candidate_rows_written = max(0, candidates_after - candidates_before)
    strategy_candidates_evaluated = sum(
        int(item.get("candidate_decisions_evaluated_this_poll") or item.get("candidates_seen") or 0)
        for item in per_strategy
    )
    latest_row = prediction_rows[0] if prediction_rows else live_rows[0] if live_rows else candidates[0] if candidates else {}
    report = {
        "status": "ok",
        "run_id": run_id,
        "timestamp": to_iso(datetime.now(timezone.utc)),
        "active_strategies": len(strategies),
        "min_feature_timestamp": (
            to_iso(parse_timestamp(min_feature_timestamp))
            if min_feature_timestamp is not None
            else None
        ),
        "latest_market_id": latest_row.get("market_id"),
        "latest_feature_timestamp": latest_row.get("timestamp"),
        "probability_yes": (
            probability_cache.get(_feature_key(latest_row)) if latest_row else None
        ),
        "candidate_rows_seen": len(candidates),
        "prediction_rows": len(prediction_rows),
        "candidates_logged": candidate_rows_written,
        "candidate_rows_total": candidates_after,
        "candidate_rows_written_this_poll": candidate_rows_written,
        "candidate_decisions_evaluated_this_poll": strategy_candidates_evaluated,
        "trades_opened": max(0, trades_after - trades_before),
        "trades_closed": int(exit_counts.get("closed", 0)),
        "exit_counts": dict(exit_counts),
        "skips_logged": total_skips,
        "settled_trades": settled,
        "missing_feature_columns": missing_columns,
        "strategy_errors": strategy_errors,
        "liquidity_fill_check_enabled": bool(liquidity_aggregate["enabled_strategies"]),
        "liquidity_checked_count": int(liquidity_aggregate["checked"]),
        "liquidity_passed_count": int(liquidity_aggregate["passed"]),
        "liquidity_blocked_count": int(liquidity_aggregate["blocked"]),
        "liquidity_missing_count": int(liquidity_aggregate["missing"]),
        "trade_selection_enabled": bool(trade_selection_report["enabled"]),
        "trade_selection_candidates_before": int(trade_selection_report["candidates_before"]),
        "trade_selection_candidates_after": int(trade_selection_report["candidates_after"]),
        "trade_selection_suppressed_count": int(trade_selection_report["suppressed_count"]),
        "live_trading": _live_trading_report(output, live_trading_settings),
        "pending_realistic_entries": int(pending_stats["pending_realistic_entries"]),
        "pending_realistic_entries_ready": int(
            pending_stats["pending_realistic_entries_ready"]
        ),
        "pending_realistic_entries_expired": int(
            pending_stats["pending_realistic_entries_expired"]
        ),
        "realistic_entries_opened_after_latency": int(
            pending_stats["realistic_entries_opened_after_latency"]
        ),
        "realistic_entries_skipped_after_latency": int(
            pending_stats["realistic_entries_skipped_after_latency"]
        ),
        "latest_pending_target_quote_timestamp": pending_stats[
            "latest_pending_target_quote_timestamp"
        ],
        "bankroll": _bankroll_report(output, bankroll_settings=bankroll_settings),
        "heartbeat_detail": heartbeat_detail,
        "strategies": per_strategy,
        "strategy_stats": per_strategy,
        "strategy_stats_summary": _summarize_strategy_stats(per_strategy),
    }
    _write_multi_strategy_heartbeat(output, report)
    _emit(
        emit_logs,
        "multi_strategy_paper_trader_heartbeat",
        **_heartbeat_log_payload(report, detail=heartbeat_detail),
    )
    return report


def ensure_multi_strategy_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS multi_strategy_paper_trader_heartbeats (
                timestamp TEXT PRIMARY KEY,
                status TEXT,
                run_id TEXT,
                active_strategies INTEGER,
                latest_market_id TEXT,
                latest_feature_timestamp TEXT,
                probability_yes REAL,
                candidates_logged INTEGER,
                trades_opened INTEGER,
                skips_logged INTEGER,
                liquidity_fill_check_enabled INTEGER,
                liquidity_checked_count INTEGER,
                liquidity_passed_count INTEGER,
                liquidity_blocked_count INTEGER,
                liquidity_missing_count INTEGER,
                pending_realistic_entries INTEGER,
                pending_realistic_entries_ready INTEGER,
                pending_realistic_entries_expired INTEGER,
                realistic_entries_opened_after_latency INTEGER,
                realistic_entries_skipped_after_latency INTEGER,
                latest_pending_target_quote_timestamp TEXT,
                trade_selection_enabled INTEGER,
                trade_selection_candidates_before INTEGER,
                trade_selection_candidates_after INTEGER,
                trade_selection_suppressed_count INTEGER,
                per_strategy_json TEXT,
                strategy_errors_json TEXT
            )
            """
        )
        for column, definition in {
            "status": "TEXT",
            "liquidity_fill_check_enabled": "INTEGER",
            "liquidity_checked_count": "INTEGER",
            "liquidity_passed_count": "INTEGER",
            "liquidity_blocked_count": "INTEGER",
            "liquidity_missing_count": "INTEGER",
            "candidate_rows_total": "INTEGER",
            "candidate_rows_written_this_poll": "INTEGER",
            "candidate_decisions_evaluated_this_poll": "INTEGER",
            "heartbeat_detail": "TEXT",
            "pending_realistic_entries": "INTEGER",
            "pending_realistic_entries_ready": "INTEGER",
            "pending_realistic_entries_expired": "INTEGER",
            "realistic_entries_opened_after_latency": "INTEGER",
            "realistic_entries_skipped_after_latency": "INTEGER",
            "latest_pending_target_quote_timestamp": "TEXT",
            "trade_selection_enabled": "INTEGER",
            "trade_selection_candidates_before": "INTEGER",
            "trade_selection_candidates_after": "INTEGER",
            "trade_selection_suppressed_count": "INTEGER",
        }.items():
            _ensure_column(conn, "multi_strategy_paper_trader_heartbeats", column, definition)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_id TEXT,
                market_id TEXT,
                direction TEXT,
                strategy_id TEXT,
                intended_price REAL,
                order_price REAL,
                size_usd REAL,
                size_shares REAL,
                reason TEXT,
                status TEXT NOT NULL,
                exchange_order_id TEXT,
                error TEXT,
                dry_run INTEGER,
                order_json TEXT,
                paper_trade_id INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_orders_market_direction
            ON live_orders (market_id, direction, strategy_id, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_orders_timestamp
            ON live_orders (timestamp)
            """
        )


def update_multi_strategy_trade_exits(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    strategies: Sequence[MultiStrategyDefinition],
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings | None = None,
    now: datetime | None = None,
    emit_logs: bool = True,
) -> dict[str, int]:
    ensure_paper_schema(output)
    now_dt = now or datetime.now(timezone.utc)
    realistic_execution_settings = (
        realistic_execution_settings or MultiStrategyRealisticExecutionSettings()
    )
    strategies_by_id = {strategy.strategy_id: strategy for strategy in strategies}
    rows = output.execute(
        """
        SELECT *
        FROM paper_trades
        WHERE status = 'open'
          AND COALESCE(exit_type, 'HOLD_TO_RESOLUTION') != 'HOLD_TO_RESOLUTION'
        ORDER BY id ASC
        """
    ).fetchall()
    counts: Counter[str] = Counter()
    for row in rows:
        strategy = strategies_by_id.get(str(row["strategy_id"] or ""))
        decision = _trade_exit_decision(
            recorder,
            output,
            row,
            strategy=strategy,
            realistic_execution_settings=realistic_execution_settings,
            now=now_dt,
        )
        if decision is None:
            continue
        _close_trade_for_cashout(output, row, decision=decision, now=now_dt)
        counts["closed"] += 1
        counts[str(decision["exit_reason"])] += 1
        _emit(
            emit_logs,
            "multi_strategy_trade_closed",
            trade_id=int(row["id"]),
            strategy_id=row["strategy_id"],
            run_id=row["run_id"],
            market_id=row["market_id"],
            signal_direction=row["signal_direction"],
            exit_reason=decision["exit_reason"],
            exit_price=decision["exit_price"],
            adjusted_exit_price=decision["adjusted_exit_price"],
            realized_pnl_usd=decision["realized_pnl_usd"],
            realized_roi=decision["realized_roi"],
        )
    return dict(counts)


def _run_strategy_evaluation_loop(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    prediction_rows: Sequence[dict[str, Any]],
    probability_cache: dict[tuple[str, str, str], float],
    strategies: Sequence[MultiStrategyDefinition],
    recorder_db_path: str,
    output_db_path: str,
    model_path: str,
    feature_columns_path: str,
    max_btc_age_sec: float,
    max_feature_age_sec: float,
    require_fresh_comparison_btc: bool,
    missing_columns: Sequence[str],
    model_name: str,
    bankroll_settings: MultiStrategyBankrollSettings,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    trade_selection_settings: MultiStrategyTradeSelectionSettings,
    live_trading_settings: MultiStrategyLiveTradingSettings,
    live_order_submitter: LiveOrderSubmitter | None,
    pending_realistic_entries: dict[tuple[str, str, str], _PendingRealisticEntry],
    processed_pending_keys: set[tuple[str, str, str]],
    run_id: str | None,
    now: datetime,
    emit_logs: bool,
) -> dict[str, Any]:
    states = {strategy.strategy_id: _empty_strategy_poll_state(strategy) for strategy in strategies}
    liquidity_aggregate: Counter[str] = Counter()
    strategy_errors: dict[str, str] = {}
    skipped = 0
    trade_selection_report = {
        "enabled": bool(trade_selection_settings.selection_enabled),
        "candidates_before": 0,
        "candidates_after": 0,
        "suppressed_count": 0,
    }
    if trade_selection_settings.selection_enabled and not missing_columns:
        eligible: list[_TradeSelectionCandidate] = []
        for strategy in strategies:
            config = _baseline_config_for_strategy(
                strategy,
                recorder_db_path=recorder_db_path,
                output_db_path=output_db_path,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                max_btc_age_sec=max_btc_age_sec,
                max_feature_age_sec=max_feature_age_sec,
                require_fresh_comparison_btc=require_fresh_comparison_btc,
            )
            state = states[strategy.strategy_id]
            try:
                for row in prediction_rows:
                    key = _feature_key(row)
                    if key not in probability_cache:
                        continue
                    pending_key = _pending_realistic_entry_key(row, strategy.strategy_id)
                    if (
                        _should_defer_realistic_entry(realistic_execution_settings)
                        and (
                            pending_key in pending_realistic_entries
                            or pending_key in processed_pending_keys
                        )
                    ):
                        state["latest_decision"] = "PENDING"
                        state["latest_feature_timestamp"] = str(row.get("timestamp") or "")
                        continue
                    state["candidates_seen"] += 1
                    state["latest_feature_timestamp"] = str(row.get("timestamp") or "")
                    candidate_or_reason = _preselect_strategy_row_candidate(
                        output,
                        row,
                        probability=probability_cache[key],
                        config=config,
                        strategy=strategy,
                        model_name=model_name,
                    )
                    if isinstance(candidate_or_reason, _TradeSelectionCandidate):
                        eligible.append(candidate_or_reason)
                        state["latest_decision"] = "ELIGIBLE"
                    elif candidate_or_reason != "none":
                        _record_strategy_skip(state, str(candidate_or_reason))
            except Exception as exc:
                state["error"] = str(exc)
                strategy_errors[strategy.strategy_id] = str(exc)
                _emit(
                    emit_logs,
                    "multi_strategy_paper_trader_strategy_error",
                    run_id=run_id,
                    strategy_id=strategy.strategy_id,
                    error=str(exc),
                )
        selected, suppressed = _select_trade_candidates(eligible, trade_selection_settings)
        trade_selection_report = {
            "enabled": True,
            "candidates_before": len(eligible),
            "candidates_after": len(selected),
            "suppressed_count": len(suppressed),
        }
        for candidate in suppressed:
            state = states[candidate.strategy.strategy_id]
            _record_trade_selection_suppressed_candidate(output, candidate)
            _record_strategy_skip(state, "trade_selection_deduped")
        for candidate in selected:
            state = states[candidate.strategy.strategy_id]
            result = _evaluate_strategy_row(
                output,
                candidate.row,
                probability=candidate.probability,
                config=candidate.config,
                strategy=candidate.strategy,
                model_name=model_name,
                bankroll_settings=bankroll_settings,
                realistic_execution_settings=realistic_execution_settings,
                live_trading_settings=live_trading_settings,
                live_order_submitter=live_order_submitter,
                pending_realistic_entries=pending_realistic_entries,
                recorder=recorder,
                now=now,
                emit_logs=emit_logs,
            )
            _record_strategy_result(state, result)
    else:
        for strategy in strategies:
            config = _baseline_config_for_strategy(
                strategy,
                recorder_db_path=recorder_db_path,
                output_db_path=output_db_path,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                max_btc_age_sec=max_btc_age_sec,
                max_feature_age_sec=max_feature_age_sec,
                require_fresh_comparison_btc=require_fresh_comparison_btc,
            )
            state = states[strategy.strategy_id]
            try:
                if missing_columns:
                    state["error"] = "missing_feature_columns: " + ", ".join(missing_columns)
                else:
                    for row in prediction_rows:
                        key = _feature_key(row)
                        if key not in probability_cache:
                            continue
                        pending_key = _pending_realistic_entry_key(row, strategy.strategy_id)
                        if (
                            _should_defer_realistic_entry(realistic_execution_settings)
                            and (
                                pending_key in pending_realistic_entries
                                or pending_key in processed_pending_keys
                            )
                        ):
                            state["latest_decision"] = "PENDING"
                            state["latest_feature_timestamp"] = str(row.get("timestamp") or "")
                            continue
                        state["candidates_seen"] += 1
                        state["latest_feature_timestamp"] = str(row.get("timestamp") or "")
                        result = _evaluate_strategy_row(
                            output,
                            row,
                            probability=probability_cache[key],
                            config=config,
                            strategy=strategy,
                            model_name=model_name,
                            bankroll_settings=bankroll_settings,
                            realistic_execution_settings=realistic_execution_settings,
                            live_trading_settings=live_trading_settings,
                            live_order_submitter=live_order_submitter,
                            pending_realistic_entries=pending_realistic_entries,
                            recorder=recorder,
                            now=now,
                            emit_logs=emit_logs,
                        )
                        _record_strategy_result(state, result)
            except Exception as exc:
                state["error"] = str(exc)
                strategy_errors[strategy.strategy_id] = str(exc)
                _emit(
                    emit_logs,
                    "multi_strategy_paper_trader_strategy_error",
                    run_id=run_id,
                    strategy_id=strategy.strategy_id,
                    error=str(exc),
                )
    per_strategy = []
    for strategy in strategies:
        state = states[strategy.strategy_id]
        skipped += int(state["skipped"])
        _add_liquidity_aggregate(liquidity_aggregate, strategy, state)
        per_strategy.append(_strategy_poll_report(output, strategy, state))
    return {
        "per_strategy": per_strategy,
        "skipped": skipped,
        "strategy_errors": strategy_errors,
        "liquidity_aggregate": liquidity_aggregate,
        "trade_selection": trade_selection_report,
    }


def _empty_strategy_poll_state(strategy: MultiStrategyDefinition) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "opened": 0,
        "skipped": 0,
        "candidates_seen": 0,
        "skip_reasons": Counter(),
        "latest_decision": None,
        "latest_rejection_reason": None,
        "latest_feature_timestamp": None,
        "error": None,
    }


def _record_strategy_result(state: dict[str, Any], result: str) -> None:
    if result == "opened":
        state["opened"] += 1
        state["latest_decision"] = "TRADE"
    elif result == "pending":
        state["latest_decision"] = "PENDING"
    elif result != "none":
        _record_strategy_skip(state, result)
    else:
        state["latest_decision"] = "NONE"


def _record_strategy_skip(state: dict[str, Any], reason: str) -> None:
    state["skipped"] += 1
    state["skip_reasons"][reason] += 1
    state["latest_decision"] = "SKIP"
    state["latest_rejection_reason"] = reason


def _add_liquidity_aggregate(
    aggregate: Counter[str],
    strategy: MultiStrategyDefinition,
    state: dict[str, Any],
) -> None:
    skip_reasons: Counter[str] = state["skip_reasons"]
    liquidity_blocked = (
        int(skip_reasons.get("liquidity_missing", 0))
        + int(skip_reasons.get("liquidity_too_low", 0))
        + int(skip_reasons.get("invalid_adjusted_entry_price", 0))
    )
    liquidity_passed = int(state["opened"]) if strategy.liquidity_fill_check_enabled else 0
    aggregate["enabled_strategies"] += int(strategy.liquidity_fill_check_enabled)
    aggregate["checked"] += liquidity_passed + liquidity_blocked
    aggregate["passed"] += liquidity_passed
    aggregate["blocked"] += liquidity_blocked
    aggregate["missing"] += int(skip_reasons.get("liquidity_missing", 0))


def _strategy_poll_report(
    output: sqlite3.Connection,
    strategy: MultiStrategyDefinition,
    state: dict[str, Any],
) -> dict[str, Any]:
    skip_reasons: Counter[str] = state["skip_reasons"]
    liquidity_blocked = (
        int(skip_reasons.get("liquidity_missing", 0))
        + int(skip_reasons.get("liquidity_too_low", 0))
        + int(skip_reasons.get("invalid_adjusted_entry_price", 0))
    )
    liquidity_passed = int(state["opened"]) if strategy.liquidity_fill_check_enabled else 0
    liquidity_checked = liquidity_passed + liquidity_blocked
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_name": strategy.strategy_name,
        "liquidity_fill_check_enabled": bool(strategy.liquidity_fill_check_enabled),
        "liquidity_checked_count": liquidity_checked,
        "liquidity_passed_count": liquidity_passed,
        "liquidity_blocked_count": liquidity_blocked,
        "liquidity_missing_count": int(skip_reasons.get("liquidity_missing", 0)),
        "candidates_seen": int(state["candidates_seen"]),
        "candidate_decisions_evaluated_this_poll": int(state["candidates_seen"]),
        "trades_opened": int(state["opened"]),
        "opened_trades": int(state["opened"]),
        "skipped_candidates": int(state["skipped"]),
        "skipped_threshold": int(skip_reasons.get("threshold", 0)),
        "skipped_non_positive_edge": int(skip_reasons.get("non_positive_edge", 0)),
        "skipped_min_estimated_edge": int(skip_reasons.get("min_estimated_edge", 0)),
        "skipped_time_window": int(skip_reasons.get("time_window", 0)),
        "skipped_max_open_trades": int(skip_reasons.get("max_open_trades", 0)),
        "skipped_one_trade_per_market": int(skip_reasons.get("one_trade_per_market", 0)),
        "skipped_trade_selection_deduped": int(
            skip_reasons.get("trade_selection_deduped", 0)
        ),
        "skipped_bankroll_insufficient_cash": int(
            skip_reasons.get("bankroll_insufficient_cash", 0)
        ),
        "skipped_bankroll_exposure_limit": int(
            skip_reasons.get("bankroll_exposure_limit", 0)
        ),
        "skipped_bankroll_depleted": int(skip_reasons.get("bankroll_depleted", 0)),
        "skipped_liquidity_missing": int(skip_reasons.get("liquidity_missing", 0)),
        "skipped_liquidity_too_low": int(skip_reasons.get("liquidity_too_low", 0)),
        "skipped_liquidity_regime_blocked": int(
            skip_reasons.get("liquidity_regime_blocked", 0)
        ),
        "skipped_invalid_adjusted_entry_price": int(
            skip_reasons.get("invalid_adjusted_entry_price", 0)
        ),
        "skipped_realistic_execution": sum(
            int(skip_reasons.get(reason, 0)) for reason in _REALISTIC_EXECUTION_SKIP_REASONS
        ),
        "skipped_quote_after_latency_missing": int(
            skip_reasons.get("quote_after_latency_missing", 0)
        ),
        "skipped_entry_price_drift_too_high": int(
            skip_reasons.get("entry_price_drift_too_high", 0)
        ),
        "skipped_spread_too_wide": int(skip_reasons.get("spread_too_wide", 0)),
        "skipped_btc_stale": int(skip_reasons.get("btc_stale", 0)),
        "skipped_feature_stale": int(skip_reasons.get("feature_stale", 0)),
        "skipped_probability_block": int(skip_reasons.get("probability_block", 0)),
        "skipped_probability_below_min": int(skip_reasons.get("probability_below_min", 0)),
        "skipped_probability_above_max": int(skip_reasons.get("probability_above_max", 0)),
        "latest_decision": state["latest_decision"],
        "latest_feature_timestamp": state["latest_feature_timestamp"],
        "open_trades": _strategy_trade_count(output, strategy.strategy_id, "open"),
        "closed_trades": _strategy_trade_count(output, strategy.strategy_id, "closed"),
        "settled_trades": _strategy_trade_count(output, strategy.strategy_id, "settled"),
        "awaiting_resolution_trades": _strategy_trade_count(
            output,
            strategy.strategy_id,
            "awaiting_resolution",
        ),
        "total_candidates": _strategy_candidate_count(output, strategy.strategy_id),
        "candidate_rows_total": _strategy_candidate_count(output, strategy.strategy_id),
        "latest_rejection_reason": (
            state["latest_rejection_reason"]
            or _latest_strategy_rejection(output, strategy.strategy_id)
        ),
        "error": state["error"],
    }


def _preselect_strategy_row_candidate(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    config: BaselinePaperTraderConfig,
    strategy: MultiStrategyDefinition,
    model_name: str,
) -> _TradeSelectionCandidate | str:
    enabled_directions = _candidate_directions(strategy)
    if not enabled_directions:
        return "none"
    selected_direction = _direction_for_probability(
        float(probability),
        long_threshold=strategy.long_threshold,
        short_threshold=strategy.short_threshold,
    )
    if selected_direction not in enabled_directions:
        selected_direction = None
    if _strategy_time_window_skip(row, strategy=strategy):
        for direction in enabled_directions:
            _record_side_candidate_skip(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason="time_window",
                config=config,
                model_name=model_name,
                strategy=strategy,
            )
        return "time_window"
    if _strategy_liquidity_regime_skip(row, strategy=strategy):
        for direction in enabled_directions:
            _record_side_candidate_skip(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason="liquidity_regime_blocked",
                config=config,
                model_name=model_name,
                strategy=strategy,
            )
        return "liquidity_regime_blocked"
    side_rejections: dict[str, str] = {}
    side_inputs: dict[str, dict[str, float | None]] = {}
    for direction in enabled_directions:
        edge_inputs = _directional_edge_inputs(
            row,
            probability_yes=float(probability),
            direction=direction,
            strategy=strategy,
        )
        side_inputs[direction] = edge_inputs
        rejection = _side_rejection_reason(
            probability=float(probability),
            direction=direction,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        if rejection is not None:
            side_rejections[direction] = rejection
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason=rejection,
                config=config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
    if selected_direction is None:
        return _primary_rejection(side_rejections) or "threshold"
    if selected_direction in side_rejections:
        return side_rejections[selected_direction]
    edge_inputs = side_inputs[selected_direction]
    if (
        edge_inputs["estimated_edge"] is not None
        and strategy.require_positive_edge
        and float(edge_inputs["estimated_edge"]) <= 0.0
    ):
        _record_edge_skip_candidate(
            output,
            row,
            probability=float(probability),
            direction=selected_direction,
            rejection_reason="non_positive_edge",
            config=config,
            model_name=model_name,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        return "non_positive_edge"
    if (
        edge_inputs["estimated_edge"] is not None
        and float(edge_inputs["estimated_edge"]) < float(strategy.min_estimated_edge)
    ):
        _record_edge_skip_candidate(
            output,
            row,
            probability=float(probability),
            direction=selected_direction,
            rejection_reason="min_estimated_edge",
            config=config,
            model_name=model_name,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        return "min_estimated_edge"
    return _TradeSelectionCandidate(
        strategy=strategy,
        config=config,
        row=row,
        probability=float(probability),
        direction=selected_direction,
        edge_inputs=edge_inputs,
        model_name=model_name,
    )


def _select_trade_candidates(
    candidates: Sequence[_TradeSelectionCandidate],
    settings: MultiStrategyTradeSelectionSettings,
) -> tuple[list[_TradeSelectionCandidate], list[_TradeSelectionCandidate]]:
    if not settings.selection_enabled or not candidates:
        return list(candidates), []
    if settings.mode != "best_per_market_direction":
        return list(candidates), []
    per_direction_limit = max(1, int(settings.max_new_trades_per_market_direction))
    market_limit = max(1, int(settings.max_new_trades_per_market))
    selected_by_direction: list[_TradeSelectionCandidate] = []
    for group in _group_selection_candidates(candidates, by_direction=True).values():
        selected_by_direction.extend(
            sorted(
                group,
                key=lambda candidate: _trade_selection_sort_key(candidate, settings),
                reverse=True,
            )[:per_direction_limit]
        )
    if not settings.allow_multiple_strategy_variants_same_market:
        market_limit = min(market_limit, 1)
    selected: list[_TradeSelectionCandidate] = []
    for group in _group_selection_candidates(selected_by_direction, by_direction=False).values():
        selected.extend(
            sorted(
                group,
                key=lambda candidate: _trade_selection_sort_key(candidate, settings),
                reverse=True,
            )[:market_limit]
        )
    selected_ids = {_trade_selection_identity(candidate) for candidate in selected}
    suppressed = [
        candidate
        for candidate in candidates
        if _trade_selection_identity(candidate) not in selected_ids
    ]
    selected.sort(key=lambda candidate: (str(candidate.row.get("market_id") or ""), candidate.strategy.strategy_id, candidate.direction))
    return selected, suppressed


def _group_selection_candidates(
    candidates: Sequence[_TradeSelectionCandidate],
    *,
    by_direction: bool,
) -> dict[tuple[str, str, str], list[_TradeSelectionCandidate]]:
    grouped: dict[tuple[str, str, str], list[_TradeSelectionCandidate]] = {}
    for candidate in candidates:
        key = (
            str(candidate.row.get("run_id") or ""),
            str(candidate.row.get("market_id") or ""),
            candidate.direction if by_direction else "",
        )
        grouped.setdefault(key, []).append(candidate)
    return grouped


def _trade_selection_identity(candidate: _TradeSelectionCandidate) -> tuple[str, str, str, str]:
    return (
        str(candidate.row.get("run_id") or ""),
        str(candidate.row.get("market_id") or ""),
        candidate.direction,
        candidate.strategy.strategy_id,
    )


def _trade_selection_sort_key(
    candidate: _TradeSelectionCandidate,
    settings: MultiStrategyTradeSelectionSettings,
) -> tuple[float, float, float, str]:
    primary = _selection_score_value(candidate, settings.score_field)
    probability = _float_or_none(candidate.edge_inputs.get("probability_for_direction"))
    adjusted = _float_or_none(candidate.edge_inputs.get("adjusted_entry_price"))
    return (
        -1.0e9 if primary is None else float(primary),
        -1.0e9 if probability is None else float(probability),
        1.0e9 if adjusted is None else -float(adjusted),
        _reverse_lexicographic_key(candidate.strategy.strategy_id),
    )


def _selection_score_value(
    candidate: _TradeSelectionCandidate,
    score_field: str,
) -> float | None:
    value = candidate.edge_inputs.get(score_field)
    if value is None:
        value = candidate.row.get(score_field)
    number = _float_or_none(value)
    if score_field in {"entry_price", "adjusted_entry_price"} and number is not None:
        return -float(number)
    return number


def _reverse_lexicographic_key(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in str(value))


def _record_trade_selection_suppressed_candidate(
    output: sqlite3.Connection,
    candidate: _TradeSelectionCandidate,
) -> None:
    _record_edge_skip_candidate(
        output,
        candidate.row,
        probability=candidate.probability,
        direction=candidate.direction,
        rejection_reason="trade_selection_deduped",
        config=candidate.config,
        model_name=candidate.model_name,
        edge_inputs=candidate.edge_inputs,
        strategy=candidate.strategy,
    )


def _evaluate_strategy_row(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    config: BaselinePaperTraderConfig,
    strategy: MultiStrategyDefinition,
    model_name: str,
    bankroll_settings: MultiStrategyBankrollSettings,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    live_trading_settings: MultiStrategyLiveTradingSettings,
    live_order_submitter: LiveOrderSubmitter | None,
    pending_realistic_entries: dict[tuple[str, str, str], _PendingRealisticEntry],
    recorder: sqlite3.Connection,
    now: datetime,
    emit_logs: bool,
) -> str:
    enabled_directions = _candidate_directions(strategy)
    if not enabled_directions:
        return "none"

    selected_direction = _direction_for_probability(
        float(probability),
        long_threshold=strategy.long_threshold,
        short_threshold=strategy.short_threshold,
    )
    if selected_direction not in enabled_directions:
        selected_direction = None

    if _strategy_time_window_skip(row, strategy=strategy):
        for direction in enabled_directions:
            _record_side_candidate_skip(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason="time_window",
                config=config,
                model_name=model_name,
                strategy=strategy,
            )
        return "time_window"
    if _strategy_liquidity_regime_skip(row, strategy=strategy):
        for direction in enabled_directions:
            _record_side_candidate_skip(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason="liquidity_regime_blocked",
                config=config,
                model_name=model_name,
                strategy=strategy,
            )
        return "liquidity_regime_blocked"

    side_rejections: dict[str, str] = {}
    side_inputs: dict[str, dict[str, float | None]] = {}
    side_realistic: dict[str, _RealisticEntryDecision] = {}
    defer_realistic_entry = _should_defer_realistic_entry(realistic_execution_settings)
    for direction in enabled_directions:
        edge_inputs = _directional_edge_inputs(
            row,
            probability_yes=float(probability),
            direction=direction,
            strategy=strategy,
        )
        if (
            realistic_execution_settings.realistic_execution_enabled
            and not defer_realistic_entry
        ):
            realistic_decision = _realistic_entry_decision(
                recorder,
                row,
                direction=direction,
                edge_inputs=edge_inputs,
                strategy=strategy,
                settings=realistic_execution_settings,
                now=now,
            )
            side_realistic[direction] = realistic_decision
            edge_inputs = realistic_decision.edge_inputs
        side_inputs[direction] = edge_inputs
        if (
            realistic_execution_settings.realistic_execution_enabled
            and direction in side_realistic
            and not side_realistic[direction].allowed
        ):
            rejection = (
                side_realistic[direction].rejection_reason
                or "quote_after_latency_missing"
            )
            side_rejections[direction] = rejection
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason=rejection,
                config=config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
            continue
        rejection = _side_rejection_reason(
            probability=float(probability),
            direction=direction,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        if rejection is not None:
            side_rejections[direction] = rejection
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=direction,
                rejection_reason=rejection,
                config=config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )

    if selected_direction is None:
        return _primary_rejection(side_rejections) or "threshold"
    if selected_direction in side_rejections:
        rejection = side_rejections[selected_direction]
        if rejection in _REALISTIC_EXECUTION_SKIP_REASONS:
            _record_realistic_execution_skip(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason=rejection,
                config=config,
                model_name=model_name,
                edge_inputs=side_inputs[selected_direction],
                realistic_decision=side_realistic[selected_direction],
                realistic_execution_settings=realistic_execution_settings,
                emit_logs=emit_logs,
            )
        return rejection

    edge_inputs = side_inputs[selected_direction]
    if (
        edge_inputs["estimated_edge"] is not None
        and strategy.require_positive_edge
        and float(edge_inputs["estimated_edge"]) <= 0.0
    ):
        _record_edge_skip_candidate(
            output,
            row,
            probability=float(probability),
            direction=selected_direction,
            rejection_reason="non_positive_edge",
            config=config,
            model_name=model_name,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        return "non_positive_edge"
    if (
        edge_inputs["estimated_edge"] is not None
        and float(edge_inputs["estimated_edge"]) < float(strategy.min_estimated_edge)
    ):
        _record_edge_skip_candidate(
            output,
            row,
            probability=float(probability),
            direction=selected_direction,
            rejection_reason="min_estimated_edge",
            config=config,
            model_name=model_name,
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        return "min_estimated_edge"
    active_config = config
    execution_row = row
    realistic_decision = side_realistic.get(selected_direction)
    if (
        realistic_execution_settings.realistic_execution_enabled
        and realistic_decision is not None
    ):
        active_config = replace(
            active_config,
            entry_slippage_cents=(
                float(active_config.entry_slippage_cents)
                + float(realistic_execution_settings.apply_extra_slippage_cents)
            ),
        )
        execution_row = realistic_decision.row
    bankroll_decision: _BankrollSizingDecision | None = None
    if bankroll_settings.bankroll_enabled:
        bankroll_decision = _bankroll_sizing_decision(
            output,
            row,
            direction=selected_direction,
            strategy=strategy,
            edge_inputs=edge_inputs,
            settings=bankroll_settings,
            now=datetime.now(timezone.utc),
        )
        if not bankroll_decision.allowed:
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason=bankroll_decision.rejection_reason or "bankroll_insufficient_cash",
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
            return bankroll_decision.rejection_reason or "bankroll_insufficient_cash"
        active_config = replace(
            active_config,
            stake_usd=float(bankroll_decision.stake_usd or active_config.stake_usd),
        )
    if defer_realistic_entry:
        strategy_id = strategy.strategy_id
        if _paper_status_count(output, "open", strategy_id=strategy_id) >= int(
            active_config.max_open_trades
        ):
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason="max_open_trades",
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
            return "max_open_trades"
        if active_config.one_trade_per_market and _has_existing_strategy_trade(
            output,
            row,
            strategy_id=strategy_id,
        ):
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason="one_trade_per_market",
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
            return "one_trade_per_market"
        precheck = _realistic_entry_precheck_decision(
            row,
            direction=selected_direction,
            edge_inputs=edge_inputs,
            settings=realistic_execution_settings,
            now=now,
        )
        if precheck is not None:
            rejection = precheck.rejection_reason or "feature_stale"
            _record_edge_skip_candidate(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason=rejection,
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
            )
            _record_realistic_execution_skip(
                output,
                row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason=rejection,
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                realistic_decision=precheck,
                realistic_execution_settings=realistic_execution_settings,
                emit_logs=emit_logs,
            )
            return rejection
        pending_key = _store_pending_realistic_entry(
            pending_realistic_entries,
            row,
            probability=float(probability),
            direction=selected_direction,
            config=active_config,
            strategy=strategy,
            model_name=model_name,
            edge_inputs=edge_inputs,
            bankroll_decision=bankroll_decision,
            settings=realistic_execution_settings,
            now=now,
        )
        pending_entry = pending_realistic_entries[pending_key]
        if now >= pending_entry.target_quote_timestamp:
            expired = now >= pending_entry.target_quote_timestamp + timedelta(
                seconds=float(realistic_execution_settings.max_quote_wait_sec)
            )
            pending_decision = _pending_realistic_entry_decision(
                recorder,
                pending_entry,
                settings=realistic_execution_settings,
                now=now,
                expired=expired,
            )
            if pending_decision is not None:
                result = _finish_pending_realistic_entry(
                    recorder,
                    output,
                    pending_entry,
                    realistic_decision=pending_decision,
                    realistic_execution_settings=realistic_execution_settings,
                    live_trading_settings=live_trading_settings,
                    live_order_submitter=live_order_submitter,
                    emit_logs=emit_logs,
                )
                pending_realistic_entries.pop(pending_key, None)
                return result
        return "pending"
    liquidity_decision: _LiquidityFillDecision | None = None
    if strategy.liquidity_fill_check_enabled:
        liquidity_decision = _liquidity_fill_decision(
            execution_row,
            direction=selected_direction,
            stake_usd=float(active_config.stake_usd),
            edge_inputs=edge_inputs,
            strategy=strategy,
        )
        if not liquidity_decision.allowed:
            rejection_reason = liquidity_decision.rejection_reason or "liquidity_too_low"
            _record_liquidity_skip(
                output,
                execution_row,
                probability=float(probability),
                direction=selected_direction,
                rejection_reason=rejection_reason,
                config=active_config,
                model_name=model_name,
                edge_inputs=edge_inputs,
                strategy=strategy,
                liquidity_decision=liquidity_decision,
                emit_logs=emit_logs,
            )
            if realistic_decision is not None:
                _apply_realistic_fields_to_latest_trade(
                    output,
                    execution_row,
                    strategy_id=strategy.strategy_id,
                    direction=selected_direction,
                    status="skipped",
                    skip_reason=rejection_reason,
                    realistic_decision=realistic_decision,
                    settings_enabled=True,
                    entry_latency_sec=realistic_execution_settings.entry_latency_sec,
                    exit_latency_sec=realistic_execution_settings.exit_latency_sec,
                )
            return rejection_reason
    result = _process_signal_row(
        output,
        execution_row,
        probability=float(probability),
        config=active_config,
        model_name=model_name,
        emit_logs=emit_logs,
    )
    if result == "opened":
        trade_id = _apply_trade_config_to_latest_trade(
            output,
            execution_row,
            strategy,
            selected_direction,
            bankroll_decision=bankroll_decision,
            liquidity_decision=liquidity_decision,
            realistic_decision=realistic_decision,
            realistic_execution_settings=realistic_execution_settings,
        )
        _maybe_process_live_order(
            output,
            execution_row,
            trade_id=trade_id,
            direction=selected_direction,
            strategy=strategy,
            live_trading_settings=live_trading_settings,
            live_order_submitter=live_order_submitter,
            now=now,
            emit_logs=emit_logs,
        )
    return result


def _should_defer_realistic_entry(
    settings: MultiStrategyRealisticExecutionSettings,
) -> bool:
    return (
        bool(settings.realistic_execution_enabled)
        and float(settings.entry_latency_sec) > 0.0
        and bool(settings.require_quote_after_latency)
    )


def _store_pending_realistic_entry(
    pending: dict[tuple[str, str, str], _PendingRealisticEntry],
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    config: BaselinePaperTraderConfig,
    strategy: MultiStrategyDefinition,
    model_name: str,
    edge_inputs: dict[str, float | None],
    bankroll_decision: _BankrollSizingDecision | None,
    settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
) -> tuple[str, str, str]:
    feature_time = parse_timestamp(row.get("timestamp")) or now
    key = _pending_realistic_entry_key(row, strategy.strategy_id)
    pending.setdefault(
        key,
        _PendingRealisticEntry(
            strategy=strategy,
            config=config,
            row=dict(row),
            probability=float(probability),
            direction=direction,
            edge_inputs=dict(edge_inputs),
            bankroll_decision=bankroll_decision,
            model_name=model_name,
            target_quote_timestamp=feature_time
            + timedelta(seconds=float(settings.entry_latency_sec)),
            created_at=now,
        ),
    )
    return key


def _pending_realistic_entry_key(
    row: dict[str, Any],
    strategy_id: str,
) -> tuple[str, str, str]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(strategy_id or ""),
    )


def _has_existing_strategy_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    strategy_id: str,
) -> bool:
    existing = output.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND COALESCE(strategy_id, 'baseline_default') = ?
          AND status IN ('open', 'awaiting_resolution', 'settled', 'closed')
        LIMIT 1
        """,
        (
            str(row.get("run_id") or ""),
            str(row.get("market_id") or ""),
            str(strategy_id or ""),
        ),
    ).fetchone()
    return existing is not None


def _empty_pending_stats() -> dict[str, Any]:
    return {
        "pending_realistic_entries": 0,
        "pending_realistic_entries_ready": 0,
        "pending_realistic_entries_expired": 0,
        "realistic_entries_opened_after_latency": 0,
        "realistic_entries_skipped_after_latency": 0,
        "latest_pending_target_quote_timestamp": None,
    }


def _merge_pending_stats(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    merged = _empty_pending_stats()
    for key in (
        "pending_realistic_entries_ready",
        "pending_realistic_entries_expired",
        "realistic_entries_opened_after_latency",
        "realistic_entries_skipped_after_latency",
    ):
        merged[key] = int(first.get(key) or 0) + int(second.get(key) or 0)
    merged["pending_realistic_entries"] = int(second.get("pending_realistic_entries") or 0)
    latest_values = [
        value
        for value in (
            first.get("latest_pending_target_quote_timestamp"),
            second.get("latest_pending_target_quote_timestamp"),
        )
        if value
    ]
    merged["latest_pending_target_quote_timestamp"] = max(latest_values) if latest_values else None
    return merged


def _process_pending_realistic_entries(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    pending: dict[tuple[str, str, str], _PendingRealisticEntry],
    *,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
    emit_logs: bool,
    processed_keys: set[tuple[str, str, str]] | None = None,
    live_trading_settings: MultiStrategyLiveTradingSettings | None = None,
    live_order_submitter: LiveOrderSubmitter | None = None,
) -> dict[str, Any]:
    stats = _empty_pending_stats()
    if not pending or not _should_defer_realistic_entry(realistic_execution_settings):
        stats["pending_realistic_entries"] = len(pending)
        stats["latest_pending_target_quote_timestamp"] = _latest_pending_target(pending)
        return stats

    for key, pending_entry in list(pending.items()):
        if now < pending_entry.target_quote_timestamp:
            continue
        stats["pending_realistic_entries_ready"] += 1
        expired = now >= pending_entry.target_quote_timestamp + timedelta(
            seconds=float(realistic_execution_settings.max_quote_wait_sec)
        )
        decision = _pending_realistic_entry_decision(
            recorder,
            pending_entry,
            settings=realistic_execution_settings,
            now=now,
            expired=expired,
        )
        if decision is None:
            continue
        if expired and decision.rejection_reason == "quote_after_latency_missing":
            stats["pending_realistic_entries_expired"] += 1
        result = _finish_pending_realistic_entry(
            recorder,
            output,
            pending_entry,
            realistic_decision=decision,
            realistic_execution_settings=realistic_execution_settings,
            live_trading_settings=live_trading_settings or MultiStrategyLiveTradingSettings(),
            live_order_submitter=live_order_submitter,
            emit_logs=emit_logs,
        )
        if result == "opened":
            stats["realistic_entries_opened_after_latency"] += 1
        else:
            stats["realistic_entries_skipped_after_latency"] += 1
        pending.pop(key, None)
        if processed_keys is not None:
            processed_keys.add(key)

    stats["pending_realistic_entries"] = len(pending)
    stats["latest_pending_target_quote_timestamp"] = _latest_pending_target(pending)
    return stats


def _latest_pending_target(
    pending: dict[tuple[str, str, str], _PendingRealisticEntry],
) -> str | None:
    if not pending:
        return None
    return to_iso(max(item.target_quote_timestamp for item in pending.values()))


def _pending_realistic_entry_decision(
    recorder: sqlite3.Connection,
    pending_entry: _PendingRealisticEntry,
    *,
    settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
    expired: bool,
) -> _RealisticEntryDecision | None:
    snapshot = _first_snapshot_at_or_after(
        recorder,
        pending_entry.row,
        target_time=pending_entry.target_quote_timestamp,
        now=now,
    )
    signal_entry = _float_or_none(pending_entry.edge_inputs.get("entry_price"))
    if snapshot is None:
        if not expired:
            return None
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="quote_after_latency_missing",
            row=dict(pending_entry.row),
            edge_inputs=dict(pending_entry.edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=None,
            settings=settings,
        )
    delayed_row = _row_with_delayed_snapshot(pending_entry.row, snapshot)
    delayed_entry = _live_entry_price(delayed_row, direction=pending_entry.direction)
    spread_cents = _spread_cents(delayed_row, direction=pending_entry.direction)
    if delayed_entry is None:
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="quote_after_latency_missing",
            row=delayed_row,
            edge_inputs=dict(pending_entry.edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    drift_cents = (
        abs(float(delayed_entry) - float(signal_entry)) * 100.0
        if signal_entry is not None
        else None
    )
    updated_inputs = _edge_inputs_with_extra_slippage(
        delayed_row,
        probability_yes=float(pending_entry.probability),
        direction=pending_entry.direction,
        strategy=pending_entry.strategy,
        settings=settings,
    )
    if (
        spread_cents is not None
        and spread_cents > float(settings.reject_if_spread_above_cents)
    ):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="spread_too_wide",
            row=delayed_row,
            edge_inputs=updated_inputs,
            signal_entry_price=signal_entry,
            delayed_entry_price=delayed_entry,
            entry_price_drift=drift_cents,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    if (
        drift_cents is not None
        and drift_cents > float(settings.max_entry_price_drift_cents)
    ):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="entry_price_drift_too_high",
            row=delayed_row,
            edge_inputs=updated_inputs,
            signal_entry_price=signal_entry,
            delayed_entry_price=delayed_entry,
            entry_price_drift=drift_cents,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    return _realistic_entry_result(
        allowed=True,
        rejection_reason=None,
        row=delayed_row,
        edge_inputs=updated_inputs,
        signal_entry_price=signal_entry,
        delayed_entry_price=delayed_entry,
        entry_price_drift=drift_cents,
        spread_cents_at_entry=spread_cents,
        settings=settings,
    )


def _finish_pending_realistic_entry(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    pending_entry: _PendingRealisticEntry,
    *,
    realistic_decision: _RealisticEntryDecision,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    live_trading_settings: MultiStrategyLiveTradingSettings,
    live_order_submitter: LiveOrderSubmitter | None,
    emit_logs: bool,
) -> str:
    row = realistic_decision.row
    edge_inputs = realistic_decision.edge_inputs
    if not realistic_decision.allowed:
        rejection = realistic_decision.rejection_reason or "quote_after_latency_missing"
        _record_edge_skip_candidate(
            output,
            row,
            probability=pending_entry.probability,
            direction=pending_entry.direction,
            rejection_reason=rejection,
            config=pending_entry.config,
            model_name=pending_entry.model_name,
            edge_inputs=edge_inputs,
            strategy=pending_entry.strategy,
        )
        _record_realistic_execution_skip(
            output,
            row,
            probability=pending_entry.probability,
            direction=pending_entry.direction,
            rejection_reason=rejection,
            config=pending_entry.config,
            model_name=pending_entry.model_name,
            edge_inputs=edge_inputs,
            realistic_decision=realistic_decision,
            realistic_execution_settings=realistic_execution_settings,
            emit_logs=emit_logs,
        )
        return rejection

    active_config = replace(
        pending_entry.config,
        entry_slippage_cents=(
            float(pending_entry.config.entry_slippage_cents)
            + float(realistic_execution_settings.apply_extra_slippage_cents)
        ),
    )
    if pending_entry.bankroll_decision is not None:
        active_config = replace(
            active_config,
            stake_usd=float(
                pending_entry.bankroll_decision.stake_usd or active_config.stake_usd
            ),
        )

    execution_row = row
    if pending_entry.strategy.liquidity_fill_check_enabled:
        execution_row = _enrich_top_of_book_liquidity(recorder, [execution_row])[0]
        realistic_decision.row = execution_row
    liquidity_decision: _LiquidityFillDecision | None = None
    if pending_entry.strategy.liquidity_fill_check_enabled:
        liquidity_decision = _liquidity_fill_decision(
            execution_row,
            direction=pending_entry.direction,
            stake_usd=float(active_config.stake_usd),
            edge_inputs=edge_inputs,
            strategy=pending_entry.strategy,
        )
        if not liquidity_decision.allowed:
            rejection = liquidity_decision.rejection_reason or "liquidity_too_low"
            _record_liquidity_skip(
                output,
                execution_row,
                probability=pending_entry.probability,
                direction=pending_entry.direction,
                rejection_reason=rejection,
                config=active_config,
                model_name=pending_entry.model_name,
                edge_inputs=edge_inputs,
                strategy=pending_entry.strategy,
                liquidity_decision=liquidity_decision,
                emit_logs=emit_logs,
            )
            _apply_realistic_fields_to_latest_trade(
                output,
                execution_row,
                strategy_id=pending_entry.strategy.strategy_id,
                direction=pending_entry.direction,
                status="skipped",
                skip_reason=rejection,
                realistic_decision=realistic_decision,
                settings_enabled=True,
                entry_latency_sec=realistic_execution_settings.entry_latency_sec,
                exit_latency_sec=realistic_execution_settings.exit_latency_sec,
            )
            return rejection

    result = _process_signal_row(
        output,
        execution_row,
        probability=pending_entry.probability,
        config=active_config,
        model_name=pending_entry.model_name,
        emit_logs=emit_logs,
    )
    if result == "opened":
        trade_id = _apply_trade_config_to_latest_trade(
            output,
            execution_row,
            pending_entry.strategy,
            pending_entry.direction,
            bankroll_decision=pending_entry.bankroll_decision,
            liquidity_decision=liquidity_decision,
            realistic_decision=realistic_decision,
            realistic_execution_settings=realistic_execution_settings,
        )
        _maybe_process_live_order(
            output,
            execution_row,
            trade_id=trade_id,
            direction=pending_entry.direction,
            strategy=pending_entry.strategy,
            live_trading_settings=live_trading_settings,
            live_order_submitter=live_order_submitter,
            now=datetime.now(timezone.utc),
            emit_logs=emit_logs,
        )
        return "opened"
    return result


def _apply_trade_config_to_latest_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    strategy: MultiStrategyDefinition,
    direction: str,
    *,
    bankroll_decision: _BankrollSizingDecision | None = None,
    liquidity_decision: _LiquidityFillDecision | None = None,
    realistic_decision: _RealisticEntryDecision | None = None,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings | None = None,
) -> int | None:
    trade = output.execute(
        """
        SELECT id, adjusted_entry_price
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND COALESCE(strategy_id, '') = ?
          AND signal_direction = ?
          AND status = 'open'
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            str(row.get("run_id") or ""),
            str(row.get("market_id") or ""),
            str(row.get("timestamp") or ""),
            strategy.strategy_id,
            direction,
        ),
    ).fetchone()
    if trade is None:
        return None
    adjusted_entry = _float_or_none(trade["adjusted_entry_price"])
    regimes = _regime_tags_for_row(row)
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET exit_type = ?,
                exit_slippage_cents = ?,
                take_profit_pct = ?,
                stop_loss_pct = ?,
                fixed_horizon_exit_sec = ?,
                trailing_stop_pct = ?,
                starting_bankroll_usd = ?,
                bankroll_before_trade = ?,
                available_cash_before_trade = ?,
                open_exposure_before_trade = ?,
                stake_fraction_of_bankroll = ?,
                max_open_exposure_usd = ?,
                bankroll_after_trade = ?,
                bankroll_status = ?,
                blown_up_at = ?,
                risk_sizing_reason = ?,
                btc_trend_regime = ?,
                volatility_regime = ?,
                spread_regime = ?,
                liquidity_regime = ?,
                time_regime = ?,
                requested_shares = ?,
                max_fillable_shares = ?,
                liquidity_fill_fraction_used = ?,
                liquidity_check_passed = ?,
                liquidity_skip_reason = ?,
                realistic_execution_enabled = ?,
                entry_latency_sec = ?,
                exit_latency_sec = ?,
                signal_entry_price = ?,
                delayed_entry_price = ?,
                entry_price_drift = ?,
                spread_cents_at_entry = ?,
                realistic_execution_skip_reason = ?,
                extra_slippage_cents_applied = ?,
                max_favorable_price = COALESCE(max_favorable_price, ?),
                max_adverse_price = COALESCE(max_adverse_price, ?)
            WHERE id = ?
            """,
            (
                strategy.exit_type,
                _round_or_none(strategy.exit_slippage_cents),
                _round_or_none(strategy.take_profit_pct),
                _round_or_none(strategy.stop_loss_pct),
                _round_or_none(strategy.fixed_horizon_exit_sec),
                _round_or_none(strategy.trailing_stop_pct),
                _round_or_none(bankroll_decision.starting_bankroll_usd) if bankroll_decision else None,
                _round_or_none(bankroll_decision.bankroll_before_trade) if bankroll_decision else None,
                _round_or_none(bankroll_decision.available_cash_before_trade) if bankroll_decision else None,
                _round_or_none(bankroll_decision.open_exposure_before_trade) if bankroll_decision else None,
                _round_or_none(bankroll_decision.stake_fraction_of_bankroll) if bankroll_decision else None,
                _round_or_none(bankroll_decision.max_open_exposure_usd) if bankroll_decision else None,
                _round_or_none(bankroll_decision.bankroll_after_trade) if bankroll_decision else None,
                bankroll_decision.bankroll_status if bankroll_decision else None,
                bankroll_decision.blown_up_at if bankroll_decision else None,
                bankroll_decision.risk_sizing_reason if bankroll_decision else None,
                regimes["btc_trend_regime"],
                regimes["volatility_regime"],
                regimes["spread_regime"],
                regimes["liquidity_regime"],
                regimes["time_regime"],
                _round_or_none(liquidity_decision.requested_shares) if liquidity_decision else None,
                _round_or_none(liquidity_decision.max_fillable_shares) if liquidity_decision else None,
                (
                    _round_or_none(liquidity_decision.liquidity_fill_fraction_used)
                    if liquidity_decision
                    else None
                ),
                liquidity_decision.liquidity_check_passed if liquidity_decision else None,
                liquidity_decision.rejection_reason if liquidity_decision else None,
                (
                    1
                    if realistic_execution_settings
                    and realistic_execution_settings.realistic_execution_enabled
                    else None
                ),
                (
                    _round_or_none(realistic_execution_settings.entry_latency_sec)
                    if realistic_execution_settings
                    and realistic_execution_settings.realistic_execution_enabled
                    else None
                ),
                (
                    _round_or_none(realistic_execution_settings.exit_latency_sec)
                    if realistic_execution_settings
                    and realistic_execution_settings.realistic_execution_enabled
                    else None
                ),
                _round_or_none(realistic_decision.signal_entry_price) if realistic_decision else None,
                _round_or_none(realistic_decision.delayed_entry_price) if realistic_decision else None,
                _round_or_none(realistic_decision.entry_price_drift) if realistic_decision else None,
                _round_or_none(realistic_decision.spread_cents_at_entry) if realistic_decision else None,
                realistic_decision.rejection_reason if realistic_decision else None,
                (
                    _round_or_none(realistic_execution_settings.apply_extra_slippage_cents)
                    if realistic_execution_settings
                    and realistic_execution_settings.realistic_execution_enabled
                    else None
                ),
                _round_or_none(adjusted_entry),
                _round_or_none(adjusted_entry),
                int(trade["id"]),
            ),
        )
    return int(trade["id"])


def _maybe_process_live_order(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    trade_id: int | None,
    direction: str,
    strategy: MultiStrategyDefinition,
    live_trading_settings: MultiStrategyLiveTradingSettings,
    live_order_submitter: LiveOrderSubmitter | None,
    now: datetime,
    emit_logs: bool,
) -> None:
    if not live_trading_settings.live_trading_enabled:
        return
    trade = _live_trade_row(output, trade_id)
    if trade is None:
        _record_live_order(
            output,
            row,
            direction=direction,
            strategy_id=strategy.strategy_id,
            intended_price=None,
            order_price=None,
            size_usd=None,
            size_shares=None,
            reason="paper_trade_missing",
            status="blocked",
            dry_run=1 if live_trading_settings.dry_run_orders else 0,
            order_json=None,
            paper_trade_id=trade_id,
            now=now,
        )
        return
    decision = _live_order_safety_decision(
        output,
        row,
        trade,
        direction=direction,
        strategy_id=strategy.strategy_id,
        settings=live_trading_settings,
        now=now,
    )
    order = decision["order"]
    if decision["allowed"] and live_trading_settings.dry_run_orders:
        _record_live_order(
            output,
            row,
            direction=direction,
            strategy_id=strategy.strategy_id,
            intended_price=order.get("intended_price"),
            order_price=order.get("order_price"),
            size_usd=order.get("size_usd"),
            size_shares=order.get("size_shares"),
            reason="dry_run_orders",
            status="dry_run",
            dry_run=1,
            order_json=order,
            paper_trade_id=trade_id,
            now=now,
        )
        _emit(emit_logs, "live_order_dry_run", **order)
        return
    if not decision["allowed"]:
        _record_live_order(
            output,
            row,
            direction=direction,
            strategy_id=strategy.strategy_id,
            intended_price=order.get("intended_price"),
            order_price=order.get("order_price"),
            size_usd=order.get("size_usd"),
            size_shares=order.get("size_shares"),
            reason=decision["reason"],
            status="blocked",
            error=decision.get("error"),
            dry_run=1 if live_trading_settings.dry_run_orders else 0,
            order_json=order,
            paper_trade_id=trade_id,
            now=now,
        )
        _emit(emit_logs, "live_order_blocked", reason=decision["reason"], **order)
        return
    if live_order_submitter is None:
        _record_live_order(
            output,
            row,
            direction=direction,
            strategy_id=strategy.strategy_id,
            intended_price=order.get("intended_price"),
            order_price=order.get("order_price"),
            size_usd=order.get("size_usd"),
            size_shares=order.get("size_shares"),
            reason="live_submitter_unavailable",
            status="blocked",
            error="live order submitter is not configured",
            dry_run=0,
            order_json=order,
            paper_trade_id=trade_id,
            now=now,
        )
        _emit(emit_logs, "live_order_blocked", reason="live_submitter_unavailable", **order)
        return
    try:
        response = live_order_submitter(order)
    except Exception as exc:
        _record_live_order(
            output,
            row,
            direction=direction,
            strategy_id=strategy.strategy_id,
            intended_price=order.get("intended_price"),
            order_price=order.get("order_price"),
            size_usd=order.get("size_usd"),
            size_shares=order.get("size_shares"),
            reason="submit_error",
            status="error",
            error=str(exc),
            dry_run=0,
            order_json=order,
            paper_trade_id=trade_id,
            now=now,
        )
        _emit(emit_logs, "live_order_error", error=str(exc), **order)
        return
    exchange_order_id = str(response.get("exchange_order_id") or response.get("id") or "")
    _record_live_order(
        output,
        row,
        direction=direction,
        strategy_id=strategy.strategy_id,
        intended_price=order.get("intended_price"),
        order_price=order.get("order_price"),
        size_usd=order.get("size_usd"),
        size_shares=order.get("size_shares"),
        reason="submitted",
        status="submitted",
        exchange_order_id=exchange_order_id or None,
        dry_run=0,
        order_json={**order, "response": response},
        paper_trade_id=trade_id,
        now=now,
    )
    _emit(emit_logs, "live_order_submitted", exchange_order_id=exchange_order_id, **order)


def _live_trade_row(output: sqlite3.Connection, trade_id: int | None) -> sqlite3.Row | None:
    if trade_id is None:
        return None
    return output.execute("SELECT * FROM paper_trades WHERE id = ?", (int(trade_id),)).fetchone()


def _live_order_safety_decision(
    output: sqlite3.Connection,
    row: dict[str, Any],
    trade: sqlite3.Row,
    *,
    direction: str,
    strategy_id: str,
    settings: MultiStrategyLiveTradingSettings,
    now: datetime,
) -> dict[str, Any]:
    intended_price = _float_or_none(trade["entry_price"])
    order_price = _float_or_none(trade["adjusted_entry_price"])
    size_usd = _float_or_none(trade["stake_usd"])
    size_shares = (
        size_usd / order_price
        if size_usd is not None and order_price is not None and order_price > 0
        else None
    )
    order = {
        "run_id": trade["run_id"],
        "market_id": trade["market_id"],
        "direction": direction,
        "strategy_id": strategy_id,
        "intended_price": _round_or_none(intended_price),
        "order_price": _round_or_none(order_price),
        "size_usd": _round_or_none(size_usd),
        "size_shares": _round_or_none(size_shares),
        "order_type": "limit" if settings.use_limit_orders_only or not settings.allow_market_order else "market",
        "dry_run": bool(settings.dry_run_orders),
    }
    reason = _live_order_block_reason(output, row, trade, order, settings=settings, now=now)
    return {"allowed": reason is None, "reason": reason, "order": order}


def _live_order_block_reason(
    output: sqlite3.Connection,
    row: dict[str, Any],
    trade: sqlite3.Row,
    order: dict[str, Any],
    *,
    settings: MultiStrategyLiveTradingSettings,
    now: datetime,
) -> str | None:
    if Path(settings.kill_switch_file).exists():
        return "kill_switch"
    size_usd = _float_or_none(order.get("size_usd"))
    order_price = _float_or_none(order.get("order_price"))
    if size_usd is None or size_usd <= 0:
        return "invalid_order_size"
    if order_price is None or order_price <= 0 or order_price >= 1:
        return "invalid_order_price"
    if size_usd > float(settings.max_order_usd):
        return "max_order_usd"
    if _live_daily_loss_usd(output, now=now) >= float(settings.max_daily_loss_usd):
        return "max_daily_loss_usd"
    if _live_daily_order_count(output, now=now) >= int(settings.max_daily_orders):
        return "max_daily_orders"
    if _live_open_exposure_usd(output) > float(settings.max_open_exposure_usd):
        return "max_open_exposure_usd"
    if _live_open_trade_count(output) > int(settings.max_open_trades):
        return "max_open_trades"
    market_id = str(trade["market_id"] or "")
    direction = str(trade["signal_direction"] or "")
    if _live_market_order_count(output, market_id=market_id) >= int(settings.max_trades_per_market):
        return "max_trades_per_market"
    if _live_market_order_count(output, market_id=market_id, direction=direction) >= int(
        settings.max_trades_per_market_direction
    ):
        return "max_trades_per_market_direction"
    if settings.require_realistic_execution_passed:
        if int(trade["realistic_execution_enabled"] or 0) != 1:
            return "realistic_execution_required"
        if trade["realistic_execution_skip_reason"] not in (None, ""):
            return "realistic_execution_failed"
    if settings.require_liquidity_check_passed and int(trade["liquidity_check_passed"] or 0) != 1:
        return "liquidity_check_required"
    btc_age = _float_or_none(trade["btc_chainlink_age_sec_at_feature"])
    if btc_age is None:
        btc_age = _float_or_none(row.get("btc_chainlink_age_sec_at_feature"))
    if btc_age is None or btc_age > float(settings.reject_if_btc_age_above_sec):
        return "btc_stale"
    feature_time = parse_timestamp(trade["signal_timestamp"])
    feature_age = (now - feature_time).total_seconds() if feature_time is not None else None
    if feature_age is None or feature_age > float(settings.reject_if_feature_age_above_sec):
        return "feature_stale"
    spread = _spread_cents(row, direction=direction)
    if spread is None:
        return "spread_unavailable"
    if spread > float(settings.reject_if_spread_above_cents):
        return "spread_too_wide"
    if not settings.allow_market_order and not settings.use_limit_orders_only:
        return "market_orders_disabled"
    return None


def _record_live_order(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    direction: str,
    strategy_id: str,
    intended_price: Any,
    order_price: Any,
    size_usd: Any,
    size_shares: Any,
    reason: str,
    status: str,
    dry_run: int,
    order_json: dict[str, Any] | None,
    paper_trade_id: int | None,
    now: datetime,
    exchange_order_id: str | None = None,
    error: str | None = None,
) -> None:
    with output:
        output.execute(
            """
            INSERT INTO live_orders (
                timestamp, run_id, market_id, direction, strategy_id,
                intended_price, order_price, size_usd, size_shares,
                reason, status, exchange_order_id, error, dry_run,
                order_json, paper_trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                to_iso(now),
                row.get("run_id"),
                row.get("market_id"),
                direction,
                strategy_id,
                _round_or_none(intended_price),
                _round_or_none(order_price),
                _round_or_none(size_usd),
                _round_or_none(size_shares),
                reason,
                status,
                exchange_order_id,
                error,
                int(dry_run),
                json.dumps(order_json or {}, sort_keys=True),
                paper_trade_id,
            ),
        )


def _live_daily_loss_usd(output: sqlite3.Connection, *, now: datetime) -> float:
    day_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    row = output.execute(
        """
        SELECT COALESCE(SUM(
            CASE
              WHEN COALESCE(pnl_usd, realized_pnl_usd, 0) < 0
              THEN -COALESCE(pnl_usd, realized_pnl_usd, 0)
              ELSE 0
            END
        ), 0) AS loss
        FROM paper_trades
        WHERE status IN ('closed', 'settled')
          AND created_at >= ?
        """,
        (to_iso(day_start),),
    ).fetchone()
    return float(row["loss"] or 0.0)


def _live_daily_order_count(output: sqlite3.Connection, *, now: datetime) -> int:
    day_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    placeholders = ", ".join("?" for _ in _LIVE_ORDER_ACTIVE_STATUSES)
    row = output.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM live_orders
        WHERE status IN ({placeholders})
          AND timestamp >= ?
        """,
        (*sorted(_LIVE_ORDER_ACTIVE_STATUSES), to_iso(day_start)),
    ).fetchone()
    return int(row["count"] or 0)


def _live_open_exposure_usd(output: sqlite3.Connection) -> float:
    row = output.execute(
        """
        SELECT COALESCE(SUM(stake_usd), 0) AS exposure
        FROM paper_trades
        WHERE status IN ('open', 'awaiting_resolution')
        """
    ).fetchone()
    return float(row["exposure"] or 0.0)


def _live_open_trade_count(output: sqlite3.Connection) -> int:
    row = output.execute(
        """
        SELECT COUNT(*) AS count
        FROM paper_trades
        WHERE status IN ('open', 'awaiting_resolution')
        """
    ).fetchone()
    return int(row["count"] or 0)


def _live_market_order_count(
    output: sqlite3.Connection,
    *,
    market_id: str,
    direction: str | None = None,
) -> int:
    params: list[Any] = [market_id, *sorted(_LIVE_ORDER_ACTIVE_STATUSES)]
    direction_sql = ""
    if direction is not None:
        direction_sql = " AND direction = ?"
        params.append(direction)
    placeholders = ", ".join("?" for _ in _LIVE_ORDER_ACTIVE_STATUSES)
    row = output.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM live_orders
        WHERE market_id = ?
          AND status IN ({placeholders})
          {direction_sql}
        """,
        tuple(params),
    ).fetchone()
    return int(row["count"] or 0)


def _live_trading_report(
    output: sqlite3.Connection,
    settings: MultiStrategyLiveTradingSettings,
) -> dict[str, Any]:
    if not settings.live_trading_enabled:
        return {"live_trading_enabled": False}
    rows = output.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM live_orders
        GROUP BY status
        """
    ).fetchall()
    return {
        "live_trading_enabled": True,
        "dry_run_orders": bool(settings.dry_run_orders),
        "status_counts": {str(row["status"]): int(row["count"] or 0) for row in rows},
    }


def _trade_exit_decision(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    trade: sqlite3.Row,
    *,
    strategy: MultiStrategyDefinition | None,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
) -> dict[str, Any] | None:
    exit_type = _trade_exit_type(trade, strategy)
    if exit_type == "HOLD_TO_RESOLUTION":
        return None
    close_time = parse_timestamp(trade["market_close_time"])
    if close_time is not None and now >= close_time:
        return None
    snapshot = _realistic_exit_snapshot(
        recorder,
        trade,
        settings=realistic_execution_settings,
        now=now,
    )
    if snapshot is None:
        return None
    direction = str(trade["signal_direction"] or "").upper()
    exit_price = _trade_exit_price(snapshot, direction=direction)
    if exit_price is None:
        return None
    adjusted_exit = _adjusted_exit_price(
        exit_price,
        exit_slippage_cents=(
            (_trade_setting(trade, strategy, "exit_slippage_cents", default=0.0) or 0.0)
            + (
                realistic_execution_settings.apply_extra_slippage_cents
                if realistic_execution_settings.realistic_execution_enabled
                else 0.0
            )
        ),
    )
    pnl, roi = _cashout_pnl(trade, adjusted_exit_price=adjusted_exit)
    favorable, adverse = _updated_price_extremes(trade, adjusted_exit)
    reason = _cashout_exit_reason(
        trade,
        strategy=strategy,
        exit_type=exit_type,
        now=now,
        realized_roi=roi,
        adjusted_exit_price=adjusted_exit,
        max_favorable_price=favorable,
    )
    if reason is None:
        _update_trade_price_extremes(output, trade, favorable=favorable, adverse=adverse)
        return None
    return {
        "exit_reason": reason,
        "exit_price": round(float(exit_price), 10),
        "adjusted_exit_price": round(float(adjusted_exit), 10),
        "realized_pnl_usd": round(float(pnl), 10),
        "realized_roi": round(float(roi), 10),
        "max_favorable_price": _round_or_none(favorable),
        "max_adverse_price": _round_or_none(adverse),
    }


def _regime_tags_for_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "btc_trend_regime": _btc_trend_regime(row),
        "volatility_regime": _volatility_regime(row),
        "spread_regime": _spread_regime(row),
        "liquidity_regime": _liquidity_regime(row),
        "time_regime": _time_regime(row),
    }


def _btc_trend_regime(row: dict[str, Any]) -> str:
    trend = _first_numeric_field(
        row,
        (
            "btc_price_change_60s",
            "btc_return_60s",
            "price_change_60s",
            "velocity_30s",
            "price_change_10s",
        ),
    )
    if trend is None:
        return "unknown"
    if trend >= 0.03:
        return "strong_up"
    if trend >= 0.005:
        return "weak_up"
    if trend <= -0.03:
        return "strong_down"
    if trend <= -0.005:
        return "weak_down"
    return "flat"


def _volatility_regime(row: dict[str, Any]) -> str:
    volatility = _first_numeric_field(
        row,
        (
            "rolling_volatility_60s",
            "volatility_60s",
            "price_volatility_60s",
            "rolling_volatility_30s",
            "volatility_30s",
        ),
    )
    if volatility is None:
        fallback = _first_numeric_field(row, ("price_change_60s", "price_change_10s"))
        volatility = abs(fallback) if fallback is not None else None
    if volatility is None:
        return "unknown"
    if volatility < 0.01:
        return "low"
    if volatility < 0.03:
        return "medium"
    return "high"


def _spread_regime(row: dict[str, Any]) -> str:
    spreads = [
        value
        for value in (
            _float_or_none(row.get("spread_yes")),
            _float_or_none(row.get("spread_no")),
        )
        if value is not None
    ]
    if not spreads:
        return "unknown"
    spread = sum(spreads) / len(spreads)
    if spread <= 0.02:
        return "tight"
    if spread <= 0.05:
        return "normal"
    return "wide"


def _liquidity_regime(row: dict[str, Any]) -> str:
    liquidity = _first_numeric_field(
        row,
        (
            "total_liquidity",
            "liquidity",
            "orderbook_liquidity",
            "book_liquidity",
            "depth_liquidity",
        ),
    )
    if liquidity is None:
        pieces = [
            _float_or_none(row.get(field))
            for field in (
                "liquidity_yes",
                "liquidity_no",
                "bid_size_yes",
                "ask_size_yes",
                "bid_size_no",
                "ask_size_no",
                "best_bid_size_yes",
                "best_ask_size_yes",
                "best_bid_size_no",
                "best_ask_size_no",
            )
        ]
        numeric = [value for value in pieces if value is not None]
        liquidity = sum(numeric) if numeric else None
    if liquidity is None:
        return "unknown"
    if liquidity < 100:
        return "thin"
    if liquidity < 1000:
        return "normal"
    return "deep"


def _time_regime(row: dict[str, Any]) -> str:
    seconds = _float_or_none(row.get("time_until_resolution"))
    if seconds is None:
        seconds = _float_or_none(row.get("seconds_before_close"))
    if seconds is None:
        return "unknown"
    if seconds < 30:
        return "0-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    return "120s+"


def _first_numeric_field(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _liquidity_checks_enabled(strategies: Sequence[MultiStrategyDefinition]) -> bool:
    return any(strategy.liquidity_fill_check_enabled for strategy in strategies)


def _enrich_top_of_book_liquidity(
    recorder: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows or not _table_exists(recorder, "order_book_levels"):
        return [dict(row) for row in rows]
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for outcome, target_field in (
            ("YES", "best_ask_size_yes"),
            ("NO", "best_ask_size_no"),
        ):
            if _top_of_book_shares(item, outcome) is not None:
                continue
            size = _order_book_level_size(recorder, item, outcome=outcome, book_side="ask")
            if size is not None:
                item[target_field] = size
        enriched.append(item)
    return enriched


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _order_book_level_size(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    outcome: str,
    book_side: str,
) -> float | None:
    result = conn.execute(
        """
        SELECT size
        FROM order_book_levels
        WHERE run_id = ?
          AND market_id = ?
          AND timestamp = ?
          AND UPPER(outcome_side) = ?
          AND LOWER(book_side) = ?
          AND level = 1
        LIMIT 1
        """,
        (
            str(row.get("run_id") or ""),
            str(row.get("market_id") or ""),
            str(row.get("_liquidity_lookup_timestamp") or row.get("timestamp") or ""),
            outcome.upper(),
            book_side.lower(),
        ),
    ).fetchone()
    if result is None:
        return None
    return _float_or_none(result["size"] if isinstance(result, sqlite3.Row) else result[0])


def _liquidity_fill_decision(
    row: dict[str, Any],
    *,
    direction: str,
    stake_usd: float,
    edge_inputs: dict[str, float | None],
    strategy: MultiStrategyDefinition,
) -> _LiquidityFillDecision:
    fraction = float(strategy.max_top_of_book_fill_fraction)
    adjusted_entry = _float_or_none(edge_inputs.get("adjusted_entry_price"))
    if adjusted_entry is None or adjusted_entry <= 0.0 or adjusted_entry >= 1.0:
        return _liquidity_decision(
            allowed=False,
            rejection_reason="invalid_adjusted_entry_price",
            requested_shares=None,
            top_of_book_shares=None,
            max_fillable_shares=None,
            fraction=fraction,
        )
    requested_shares = float(stake_usd) / float(adjusted_entry)
    top_of_book_shares = _top_of_book_shares(row, direction)
    if top_of_book_shares is None:
        if strategy.skip_if_liquidity_missing:
            return _liquidity_decision(
                allowed=False,
                rejection_reason="liquidity_missing",
                requested_shares=requested_shares,
                top_of_book_shares=None,
                max_fillable_shares=None,
                fraction=fraction,
            )
        return _liquidity_decision(
            allowed=True,
            rejection_reason=None,
            requested_shares=requested_shares,
            top_of_book_shares=None,
            max_fillable_shares=None,
            fraction=fraction,
        )
    max_fillable_shares = float(top_of_book_shares) * fraction
    if float(top_of_book_shares) < float(strategy.min_top_of_book_shares):
        return _liquidity_decision(
            allowed=False,
            rejection_reason="liquidity_too_low",
            requested_shares=requested_shares,
            top_of_book_shares=top_of_book_shares,
            max_fillable_shares=max_fillable_shares,
            fraction=fraction,
        )
    if requested_shares > max_fillable_shares:
        return _liquidity_decision(
            allowed=False,
            rejection_reason="liquidity_too_low",
            requested_shares=requested_shares,
            top_of_book_shares=top_of_book_shares,
            max_fillable_shares=max_fillable_shares,
            fraction=fraction,
        )
    return _liquidity_decision(
        allowed=True,
        rejection_reason=None,
        requested_shares=requested_shares,
        top_of_book_shares=top_of_book_shares,
        max_fillable_shares=max_fillable_shares,
        fraction=fraction,
    )


def _liquidity_decision(
    *,
    allowed: bool,
    rejection_reason: str | None,
    requested_shares: float | None,
    top_of_book_shares: float | None,
    max_fillable_shares: float | None,
    fraction: float,
) -> _LiquidityFillDecision:
    return _LiquidityFillDecision(
        allowed=allowed,
        rejection_reason=rejection_reason,
        requested_shares=_round_or_none(requested_shares),
        top_of_book_shares=_round_or_none(top_of_book_shares),
        max_fillable_shares=_round_or_none(max_fillable_shares),
        liquidity_fill_fraction_used=_round_or_none(fraction) or 0.0,
        liquidity_check_passed=1 if allowed else 0,
    )


def _top_of_book_shares(row: dict[str, Any], direction: str) -> float | None:
    fields = _YES_TOP_ASK_SIZE_FIELDS if direction == "YES" else _NO_TOP_ASK_SIZE_FIELDS
    return _first_numeric_field(row, fields)


def _realistic_entry_precheck_decision(
    row: dict[str, Any],
    *,
    direction: str,
    edge_inputs: dict[str, float | None],
    settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
) -> _RealisticEntryDecision | None:
    signal_entry = _float_or_none(edge_inputs.get("entry_price"))
    feature_time = parse_timestamp(row.get("timestamp"))
    if feature_time is None:
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="feature_stale",
            row=dict(row),
            edge_inputs=dict(edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=None,
            settings=settings,
        )
    if (now - feature_time).total_seconds() > float(settings.reject_if_feature_age_above_sec):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="feature_stale",
            row=dict(row),
            edge_inputs=dict(edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=None,
            settings=settings,
        )
    btc_age = _float_or_none(row.get("btc_chainlink_age_sec_at_feature"))
    if btc_age is None or btc_age > float(settings.reject_if_btc_age_above_sec):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="btc_stale",
            row=dict(row),
            edge_inputs=dict(edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=None,
            settings=settings,
        )
    return None


def _edge_inputs_with_extra_slippage(
    row: dict[str, Any],
    *,
    probability_yes: float,
    direction: str,
    strategy: MultiStrategyDefinition,
    settings: MultiStrategyRealisticExecutionSettings,
) -> dict[str, float | None]:
    return _directional_edge_inputs(
        row,
        probability_yes=probability_yes,
        direction=direction,
        strategy=replace(
            strategy,
            entry_slippage_cents=(
                float(strategy.entry_slippage_cents)
                + float(settings.apply_extra_slippage_cents)
            ),
        ),
    )


def _realistic_entry_decision(
    recorder: sqlite3.Connection,
    row: dict[str, Any],
    *,
    direction: str,
    edge_inputs: dict[str, float | None],
    strategy: MultiStrategyDefinition,
    settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
) -> _RealisticEntryDecision:
    signal_entry = _float_or_none(edge_inputs.get("entry_price"))
    precheck = _realistic_entry_precheck_decision(
        row,
        direction=direction,
        edge_inputs=edge_inputs,
        settings=settings,
        now=now,
    )
    if precheck is not None:
        return precheck
    if float(settings.entry_latency_sec) <= 0.0 or not settings.require_quote_after_latency:
        updated_inputs = _edge_inputs_with_extra_slippage(
            row,
            probability_yes=float(edge_inputs["probability_yes"] or 0.0),
            direction=direction,
            strategy=strategy,
            settings=settings,
        )
        return _realistic_entry_result(
            allowed=True,
            rejection_reason=None,
            row=dict(row),
            edge_inputs=updated_inputs,
            signal_entry_price=signal_entry,
            delayed_entry_price=signal_entry,
            entry_price_drift=None,
            spread_cents_at_entry=_spread_cents(row, direction=direction),
            settings=settings,
        )
    base_result = _realistic_entry_result(
        allowed=True,
        rejection_reason=None,
        row=dict(row),
        edge_inputs=dict(edge_inputs),
        signal_entry_price=signal_entry,
        delayed_entry_price=signal_entry,
        entry_price_drift=None,
        spread_cents_at_entry=_spread_cents(row, direction=direction),
        settings=settings,
    )
    feature_time = parse_timestamp(row.get("timestamp"))
    if feature_time is None:
        return base_result
    target_time = feature_time + timedelta(seconds=float(settings.entry_latency_sec))
    delayed_snapshot = _first_snapshot_at_or_after(
        recorder,
        row,
        target_time=target_time,
        now=now,
    )
    if delayed_snapshot is None:
        if settings.require_quote_after_latency:
            return _realistic_entry_result(
                allowed=False,
                rejection_reason="quote_after_latency_missing",
                row=dict(row),
                edge_inputs=dict(edge_inputs),
                signal_entry_price=signal_entry,
                delayed_entry_price=None,
                entry_price_drift=None,
                spread_cents_at_entry=None,
                settings=settings,
            )
        return base_result
    delayed_row = _row_with_delayed_snapshot(row, delayed_snapshot)
    delayed_entry = _live_entry_price(delayed_row, direction=direction)
    spread_cents = _spread_cents(delayed_row, direction=direction)
    if delayed_entry is None:
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="quote_after_latency_missing",
            row=delayed_row,
            edge_inputs=dict(edge_inputs),
            signal_entry_price=signal_entry,
            delayed_entry_price=None,
            entry_price_drift=None,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    drift_cents = (
        abs(float(delayed_entry) - float(signal_entry)) * 100.0
        if signal_entry is not None
        else None
    )
    updated_inputs = _edge_inputs_with_extra_slippage(
        delayed_row,
        probability_yes=float(edge_inputs["probability_yes"] or 0.0),
        direction=direction,
        strategy=strategy,
        settings=settings,
    )
    if (
        spread_cents is not None
        and spread_cents > float(settings.reject_if_spread_above_cents)
    ):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="spread_too_wide",
            row=delayed_row,
            edge_inputs=updated_inputs,
            signal_entry_price=signal_entry,
            delayed_entry_price=delayed_entry,
            entry_price_drift=drift_cents,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    if (
        drift_cents is not None
        and drift_cents > float(settings.max_entry_price_drift_cents)
    ):
        return _realistic_entry_result(
            allowed=False,
            rejection_reason="entry_price_drift_too_high",
            row=delayed_row,
            edge_inputs=updated_inputs,
            signal_entry_price=signal_entry,
            delayed_entry_price=delayed_entry,
            entry_price_drift=drift_cents,
            spread_cents_at_entry=spread_cents,
            settings=settings,
        )
    return _realistic_entry_result(
        allowed=True,
        rejection_reason=None,
        row=delayed_row,
        edge_inputs=updated_inputs,
        signal_entry_price=signal_entry,
        delayed_entry_price=delayed_entry,
        entry_price_drift=drift_cents,
        spread_cents_at_entry=spread_cents,
        settings=settings,
    )


def _realistic_entry_result(
    *,
    allowed: bool,
    rejection_reason: str | None,
    row: dict[str, Any],
    edge_inputs: dict[str, float | None],
    signal_entry_price: float | None,
    delayed_entry_price: float | None,
    entry_price_drift: float | None,
    spread_cents_at_entry: float | None,
    settings: MultiStrategyRealisticExecutionSettings,
) -> _RealisticEntryDecision:
    return _RealisticEntryDecision(
        allowed=allowed,
        rejection_reason=rejection_reason,
        row=row,
        edge_inputs=edge_inputs,
        signal_entry_price=_round_or_none(signal_entry_price),
        delayed_entry_price=_round_or_none(delayed_entry_price),
        entry_price_drift=_round_or_none(entry_price_drift),
        spread_cents_at_entry=_round_or_none(spread_cents_at_entry),
        extra_slippage_cents_applied=_round_or_none(settings.apply_extra_slippage_cents)
        or 0.0,
    )


def _first_snapshot_at_or_after(
    recorder: sqlite3.Connection,
    row: dict[str, Any],
    *,
    target_time: datetime,
    now: datetime,
) -> sqlite3.Row | None:
    return recorder.execute(
        """
        SELECT *
        FROM market_snapshots
        WHERE run_id = ?
          AND market_id = ?
          AND datetime(timestamp) >= datetime(?)
          AND datetime(timestamp) <= datetime(?)
        ORDER BY datetime(timestamp) ASC, timestamp ASC
        LIMIT 1
        """,
        (
            str(row.get("run_id") or ""),
            str(row.get("market_id") or ""),
            to_iso(target_time),
            to_iso(now),
        ),
    ).fetchone()


def _row_with_delayed_snapshot(
    row: dict[str, Any],
    snapshot: sqlite3.Row,
) -> dict[str, Any]:
    updated = dict(row)
    if "timestamp" in snapshot.keys():
        updated["_delayed_snapshot_timestamp"] = snapshot["timestamp"]
        updated["_liquidity_lookup_timestamp"] = snapshot["timestamp"]
    for field in (
        "best_bid_yes",
        "best_ask_yes",
        "best_bid_no",
        "best_ask_no",
        "spread_yes",
        "spread_no",
        "mid_price_yes",
        "mid_price_no",
        "last_trade_price",
        "last_trade_size",
        "last_trade_time",
        "has_orderbook",
        "has_trade_data",
    ):
        if field in snapshot.keys():
            updated[field] = snapshot[field]
    return updated


def _spread_cents(row: dict[str, Any], *, direction: str) -> float | None:
    spread_field = "spread_yes" if direction == "YES" else "spread_no"
    spread = _float_or_none(row.get(spread_field))
    if spread is None:
        bid_field = "best_bid_yes" if direction == "YES" else "best_bid_no"
        ask_field = "best_ask_yes" if direction == "YES" else "best_ask_no"
        bid = _float_or_none(row.get(bid_field))
        ask = _float_or_none(row.get(ask_field))
        if bid is not None and ask is not None:
            spread = ask - bid
    return _round_or_none(spread * 100.0) if spread is not None else None


def _trade_exit_type(
    trade: sqlite3.Row,
    strategy: MultiStrategyDefinition | None,
) -> str:
    value = getattr(strategy, "exit_type", None) if strategy is not None else None
    if value is None:
        value = trade["exit_type"] if "exit_type" in trade.keys() else None
    normalized = str(value or "HOLD_TO_RESOLUTION").upper()
    return normalized if normalized in _EXIT_TYPES else "HOLD_TO_RESOLUTION"


def _trade_setting(
    trade: sqlite3.Row,
    strategy: MultiStrategyDefinition | None,
    field: str,
    *,
    default: float | None = None,
) -> float | None:
    value = getattr(strategy, field, None) if strategy is not None else None
    if value is None and field in trade.keys():
        value = trade[field]
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _latest_snapshot_for_trade(
    recorder: sqlite3.Connection,
    trade: sqlite3.Row,
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
        (
            str(trade["run_id"] or ""),
            str(trade["market_id"] or ""),
            to_iso(now),
        ),
    ).fetchone()


def _realistic_exit_snapshot(
    recorder: sqlite3.Connection,
    trade: sqlite3.Row,
    *,
    settings: MultiStrategyRealisticExecutionSettings,
    now: datetime,
) -> sqlite3.Row | None:
    if not settings.realistic_execution_enabled:
        return _latest_snapshot_for_trade(recorder, trade, now=now)
    signal_time = now - timedelta(seconds=float(settings.exit_latency_sec))
    signal_snapshot = _latest_snapshot_for_trade(recorder, trade, now=signal_time)
    delayed_snapshot = _latest_snapshot_for_trade(recorder, trade, now=now)
    if signal_snapshot is None or delayed_snapshot is None:
        return None if settings.require_quote_after_latency else delayed_snapshot
    direction = str(trade["signal_direction"] or "").upper()
    signal_exit = _trade_exit_price(signal_snapshot, direction=direction)
    delayed_exit = _trade_exit_price(delayed_snapshot, direction=direction)
    if signal_exit is None or delayed_exit is None:
        return None
    drift_cents = abs(float(delayed_exit) - float(signal_exit)) * 100.0
    if drift_cents > float(settings.max_exit_price_drift_cents):
        return None
    return delayed_snapshot


def _trade_exit_price(snapshot: sqlite3.Row, *, direction: str) -> float | None:
    field = "best_bid_yes" if direction == "YES" else "best_bid_no"
    price = _float_or_none(snapshot[field])
    if price is None or price <= 0 or price >= 1:
        return None
    return price


def _adjusted_exit_price(
    exit_price: float,
    *,
    exit_slippage_cents: float | None,
) -> float:
    return max(0.0, float(exit_price) - float(exit_slippage_cents or 0.0) / 100.0)


def _cashout_pnl(
    trade: sqlite3.Row,
    *,
    adjusted_exit_price: float,
) -> tuple[float, float]:
    stake = float(_float_or_none(trade["stake_usd"]) or 0.0)
    adjusted_entry = float(_float_or_none(trade["adjusted_entry_price"]) or 0.0)
    shares = stake / adjusted_entry if adjusted_entry > 0 else 0.0
    pnl = shares * float(adjusted_exit_price) - stake
    roi = pnl / stake if stake else 0.0
    return pnl, roi


def _updated_price_extremes(
    trade: sqlite3.Row,
    adjusted_exit_price: float,
) -> tuple[float | None, float | None]:
    entry = _float_or_none(trade["adjusted_entry_price"])
    current = float(adjusted_exit_price)
    old_favorable = _float_or_none(trade["max_favorable_price"]) if "max_favorable_price" in trade.keys() else None
    old_adverse = _float_or_none(trade["max_adverse_price"]) if "max_adverse_price" in trade.keys() else None
    candidates = [value for value in (entry, current, old_favorable) if value is not None]
    favorable = max(candidates) if candidates else None
    adverse_candidates = [value for value in (entry, current, old_adverse) if value is not None]
    adverse = min(adverse_candidates) if adverse_candidates else None
    return favorable, adverse


def _update_trade_price_extremes(
    output: sqlite3.Connection,
    trade: sqlite3.Row,
    *,
    favorable: float | None,
    adverse: float | None,
) -> None:
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET max_favorable_price = ?,
                max_adverse_price = ?
            WHERE id = ?
              AND status = 'open'
            """,
            (_round_or_none(favorable), _round_or_none(adverse), int(trade["id"])),
        )


def _cashout_exit_reason(
    trade: sqlite3.Row,
    *,
    strategy: MultiStrategyDefinition | None,
    exit_type: str,
    now: datetime,
    realized_roi: float,
    adjusted_exit_price: float,
    max_favorable_price: float | None,
) -> str | None:
    entry_time = parse_timestamp(trade["created_at"]) or parse_timestamp(trade["signal_timestamp"])
    held_seconds = max(0.0, (now - entry_time).total_seconds()) if entry_time is not None else 0.0
    if exit_type == "FIXED_HORIZON_EXIT":
        horizon = _trade_setting(
            trade,
            strategy,
            "fixed_horizon_exit_sec",
            default=None,
        )
        if horizon is not None and held_seconds >= float(horizon):
            return "fixed_horizon"
    if exit_type in {"TAKE_PROFIT_STOP_LOSS", "TRAILING_STOP"}:
        take_profit = _trade_setting(trade, strategy, "take_profit_pct", default=None)
        stop_loss = _trade_setting(trade, strategy, "stop_loss_pct", default=None)
        if take_profit is not None and realized_roi >= float(take_profit):
            return "take_profit"
        if stop_loss is not None and realized_roi <= -abs(float(stop_loss)):
            return "stop_loss"
    if exit_type == "TRAILING_STOP":
        trailing = _trade_setting(trade, strategy, "trailing_stop_pct", default=None)
        if trailing is not None and max_favorable_price is not None:
            entry = _float_or_none(trade["adjusted_entry_price"])
            if entry is not None and max_favorable_price > entry:
                trigger_price = max_favorable_price * (1.0 - abs(float(trailing)))
                if adjusted_exit_price <= trigger_price:
                    return "trailing_stop"
    return None


def _close_trade_for_cashout(
    output: sqlite3.Connection,
    trade: sqlite3.Row,
    *,
    decision: dict[str, Any],
    now: datetime,
) -> None:
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET status = 'closed',
                exit_time = ?,
                exit_reason = ?,
                exit_price = ?,
                adjusted_exit_price = ?,
                realized_pnl_usd = ?,
                realized_roi = ?,
                max_favorable_price = ?,
                max_adverse_price = ?
            WHERE id = ?
              AND status = 'open'
            """,
            (
                to_iso(now),
                decision["exit_reason"],
                decision["exit_price"],
                decision["adjusted_exit_price"],
                decision["realized_pnl_usd"],
                decision["realized_roi"],
                decision["max_favorable_price"],
                decision["max_adverse_price"],
                int(trade["id"]),
            ),
        )


def _bankroll_sizing_decision(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    direction: str,
    strategy: MultiStrategyDefinition,
    edge_inputs: dict[str, float | None],
    settings: MultiStrategyBankrollSettings,
    now: datetime,
) -> _BankrollSizingDecision:
    state = _bankroll_state(output, row, settings=settings)
    current_bankroll = state["current_bankroll"]
    open_exposure = state["open_exposure"]
    available_cash = current_bankroll - open_exposure
    max_open_exposure = current_bankroll * float(settings.max_total_exposure_fraction)
    exposure_remaining = max_open_exposure - open_exposure
    status = "BLOWN_UP" if current_bankroll <= 0 else "ACTIVE"
    blown_up_at = to_iso(now) if status == "BLOWN_UP" else None
    common = {
        "starting_bankroll_usd": float(settings.starting_bankroll_usd),
        "bankroll_before_trade": current_bankroll,
        "available_cash_before_trade": available_cash,
        "open_exposure_before_trade": open_exposure,
        "max_open_exposure_usd": max_open_exposure,
        "bankroll_after_trade": current_bankroll,
        "bankroll_status": status,
        "blown_up_at": blown_up_at,
    }
    if status == "BLOWN_UP" and settings.stop_trading_on_bankroll_depleted:
        return _bankroll_decision(
            allowed=False,
            stake_usd=None,
            rejection_reason="bankroll_depleted",
            stake_fraction=None,
            risk_sizing_reason="bankroll depleted; new trades disabled",
            **common,
        )
    if available_cash < float(settings.min_stake_usd):
        return _bankroll_decision(
            allowed=False,
            stake_usd=None,
            rejection_reason="bankroll_insufficient_cash",
            stake_fraction=None,
            risk_sizing_reason="available cash below minimum stake",
            **common,
        )
    if exposure_remaining < float(settings.min_stake_usd):
        return _bankroll_decision(
            allowed=False,
            stake_usd=None,
            rejection_reason="bankroll_exposure_limit",
            stake_fraction=None,
            risk_sizing_reason="open exposure limit reached",
            **common,
        )
    probability_for_direction = float(edge_inputs.get("probability_for_direction") or 0.0)
    estimated_edge = float(edge_inputs.get("estimated_edge") or 0.0)
    max_fraction = max(0.0, float(settings.max_risk_fraction))
    base_fraction = min(max(0.0, float(settings.base_risk_fraction)), max_fraction)
    edge_bonus = max(0.0, estimated_edge - float(strategy.min_estimated_edge)) * 0.002
    confidence_bonus = max(0.0, probability_for_direction - 0.50) * 0.001
    risk_fraction = min(max_fraction, base_fraction + edge_bonus + confidence_bonus)
    raw_stake = current_bankroll * risk_fraction
    stake = max(float(settings.min_stake_usd), raw_stake)
    stake = min(stake, float(settings.max_stake_usd), available_cash, exposure_remaining)
    if stake < float(settings.min_stake_usd):
        reason = (
            "bankroll_exposure_limit"
            if exposure_remaining < available_cash
            else "bankroll_insufficient_cash"
        )
        return _bankroll_decision(
            allowed=False,
            stake_usd=None,
            rejection_reason=reason,
            stake_fraction=None,
            risk_sizing_reason="sized stake fell below minimum after bankroll constraints",
            **common,
        )
    sizing_reason = (
        f"{direction} stake sized from bankroll: "
        f"base_fraction={base_fraction:.4f}, risk_fraction={risk_fraction:.4f}, "
        f"probability_for_direction={probability_for_direction:.4f}, "
        f"estimated_edge={estimated_edge:.4f}"
    )
    return _bankroll_decision(
        allowed=True,
        stake_usd=stake,
        rejection_reason=None,
        stake_fraction=stake / current_bankroll if current_bankroll > 0 else None,
        risk_sizing_reason=sizing_reason,
        **common,
    )


def _bankroll_decision(
    *,
    allowed: bool,
    stake_usd: float | None,
    rejection_reason: str | None,
    stake_fraction: float | None,
    risk_sizing_reason: str,
    starting_bankroll_usd: float,
    bankroll_before_trade: float,
    available_cash_before_trade: float,
    open_exposure_before_trade: float,
    max_open_exposure_usd: float,
    bankroll_after_trade: float,
    bankroll_status: str,
    blown_up_at: str | None,
) -> _BankrollSizingDecision:
    return _BankrollSizingDecision(
        allowed=allowed,
        stake_usd=_round_or_none(stake_usd),
        rejection_reason=rejection_reason,
        starting_bankroll_usd=_round_or_none(starting_bankroll_usd) or 0.0,
        bankroll_before_trade=_round_or_none(bankroll_before_trade) or 0.0,
        available_cash_before_trade=_round_or_none(available_cash_before_trade) or 0.0,
        open_exposure_before_trade=_round_or_none(open_exposure_before_trade) or 0.0,
        stake_fraction_of_bankroll=_round_or_none(stake_fraction),
        max_open_exposure_usd=_round_or_none(max_open_exposure_usd) or 0.0,
        bankroll_after_trade=_round_or_none(bankroll_after_trade) or 0.0,
        bankroll_status=bankroll_status,
        blown_up_at=blown_up_at,
        risk_sizing_reason=risk_sizing_reason,
    )


def _bankroll_state(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    settings: MultiStrategyBankrollSettings,
) -> dict[str, float]:
    run_id = str(row.get("run_id") or "")
    realized_where, realized_params = _bankroll_scope_clause(run_id, settings)
    realized = output.execute(
        f"""
        SELECT COALESCE(SUM(
            CASE
              WHEN status = 'settled' THEN COALESCE(pnl_usd, 0)
              WHEN status = 'closed' THEN COALESCE(realized_pnl_usd, 0)
              ELSE 0
            END
        ), 0) AS realized_pnl
        FROM paper_trades
        WHERE status IN ('settled', 'closed')
          {realized_where}
        """,
        realized_params,
    ).fetchone()
    exposure_where, exposure_params = _bankroll_scope_clause(run_id, settings)
    exposure = output.execute(
        f"""
        SELECT COALESCE(SUM(stake_usd), 0) AS open_exposure
        FROM paper_trades
        WHERE status IN ('open', 'awaiting_resolution')
          {exposure_where}
        """,
        exposure_params,
    ).fetchone()
    realized_pnl = float(realized["realized_pnl"] or 0.0)
    open_exposure = float(exposure["open_exposure"] or 0.0)
    return {
        "current_bankroll": float(settings.starting_bankroll_usd) + realized_pnl,
        "open_exposure": open_exposure,
    }


def _bankroll_scope_clause(
    run_id: str,
    settings: MultiStrategyBankrollSettings,
) -> tuple[str, tuple[Any, ...]]:
    if settings.reset_bankroll_on_new_run_id:
        return "AND run_id = ?", (run_id,)
    return "", ()


def _bankroll_report(
    output: sqlite3.Connection,
    *,
    bankroll_settings: MultiStrategyBankrollSettings,
) -> dict[str, Any]:
    if not bankroll_settings.bankroll_enabled:
        return {"bankroll_enabled": False}
    row = output.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN status = 'settled' THEN COALESCE(pnl_usd, 0)
                            WHEN status = 'closed' THEN COALESCE(realized_pnl_usd, 0)
                            ELSE 0 END), 0) AS realized_pnl,
          COALESCE(SUM(CASE WHEN status IN ('open', 'awaiting_resolution')
                            THEN COALESCE(stake_usd, 0)
                            ELSE 0 END), 0) AS open_exposure
        FROM paper_trades
        """
    ).fetchone()
    current = float(bankroll_settings.starting_bankroll_usd) + float(row["realized_pnl"] or 0.0)
    status = "BLOWN_UP" if current <= 0 else "ACTIVE"
    return {
        "bankroll_enabled": True,
        "starting_bankroll_usd": _round_or_none(bankroll_settings.starting_bankroll_usd),
        "current_bankroll_usd": _round_or_none(current),
        "open_exposure_usd": _round_or_none(row["open_exposure"]),
        "available_cash_usd": _round_or_none(current - float(row["open_exposure"] or 0.0)),
        "bankroll_status": status,
    }


def _strategy_time_window_skip(
    row: dict[str, Any],
    *,
    strategy: MultiStrategyDefinition,
) -> bool:
    seconds = _float_or_none(row.get("time_until_resolution"))
    if seconds is None:
        return True
    if seconds < float(strategy.min_time_until_resolution_sec):
        return True
    return seconds > float(strategy.max_time_until_resolution_sec)


def _strategy_liquidity_regime_skip(
    row: dict[str, Any],
    *,
    strategy: MultiStrategyDefinition,
) -> bool:
    if not strategy.blocked_liquidity_regimes:
        return False
    liquidity_regime = _regime_tags_for_row(row)["liquidity_regime"]
    blocked = {str(value).strip().lower() for value in strategy.blocked_liquidity_regimes}
    return liquidity_regime.lower() in blocked


def _candidate_directions(strategy: MultiStrategyDefinition) -> tuple[str, ...]:
    allow_yes, allow_no = _direction_mode_flags(strategy.direction_mode)
    directions: list[str] = []
    if allow_yes:
        directions.append("YES")
    if allow_no:
        directions.append("NO")
    return tuple(directions)


def _directional_edge_inputs(
    row: dict[str, Any],
    *,
    probability_yes: float,
    direction: str,
    strategy: MultiStrategyDefinition,
) -> dict[str, float | None]:
    entry_price = _live_entry_price(row, direction=direction)
    adjusted_entry_price = (
        min(
            1.0,
            float(entry_price)
            + float(strategy.entry_slippage_cents) / 100.0
            + float(strategy.fee_cents) / 100.0,
        )
        if entry_price is not None
        else None
    )
    probability_no = 1.0 - float(probability_yes)
    probability_for_direction = (
        float(probability_yes) if direction == "YES" else probability_no
    )
    estimated_edge = (
        probability_for_direction - adjusted_entry_price
        if adjusted_entry_price is not None
        else None
    )
    return {
        "probability_yes": float(probability_yes),
        "probability_no": probability_no,
        "probability_for_direction": probability_for_direction,
        "entry_price": entry_price,
        "adjusted_entry_price": adjusted_entry_price,
        "estimated_edge": estimated_edge,
    }


def _record_side_candidate_skip(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    rejection_reason: str,
    config: BaselinePaperTraderConfig,
    model_name: str,
    strategy: MultiStrategyDefinition,
) -> None:
    edge_inputs = _directional_edge_inputs(
        row,
        probability_yes=probability,
        direction=direction,
        strategy=strategy,
    )
    _record_edge_skip_candidate(
        output,
        row,
        probability=probability,
        direction=direction,
        rejection_reason=rejection_reason,
        config=config,
        model_name=model_name,
        edge_inputs=edge_inputs,
        strategy=strategy,
    )


def _record_edge_skip_candidate(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    rejection_reason: str,
    config: BaselinePaperTraderConfig,
    model_name: str,
    edge_inputs: dict[str, float | None],
    strategy: MultiStrategyDefinition,
) -> None:
    threshold = strategy.long_threshold if direction == "YES" else strategy.short_threshold
    _record_trade_candidate(
        output,
        row,
        probability=probability,
        direction=direction,
        threshold=threshold,
        decision="SKIP",
        rejection_reason=rejection_reason,
        config=config,
        model_name=model_name,
        entry_price=edge_inputs["entry_price"],
        adjusted_entry_price=edge_inputs["adjusted_entry_price"],
        probability_for_direction=edge_inputs["probability_for_direction"],
        estimated_edge=edge_inputs["estimated_edge"],
    )


def _record_liquidity_skip(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    rejection_reason: str,
    config: BaselinePaperTraderConfig,
    model_name: str,
    edge_inputs: dict[str, float | None],
    strategy: MultiStrategyDefinition,
    liquidity_decision: _LiquidityFillDecision,
    emit_logs: bool,
) -> None:
    _record_edge_skip_candidate(
        output,
        row,
        probability=probability,
        direction=direction,
        rejection_reason=rejection_reason,
        config=config,
        model_name=model_name,
        edge_inputs=edge_inputs,
        strategy=strategy,
    )
    threshold = strategy.long_threshold if direction == "YES" else strategy.short_threshold
    _record_skipped_trade(
        output,
        row,
        probability=probability,
        direction=direction,
        threshold=threshold,
        skip_reason=rejection_reason,
        config=config,
        model_name=model_name,
        emit_logs=emit_logs,
        entry_price=edge_inputs.get("entry_price"),
        adjusted_entry_price=edge_inputs.get("adjusted_entry_price"),
        probability_for_direction=edge_inputs.get("probability_for_direction"),
        estimated_edge=edge_inputs.get("estimated_edge"),
    )
    _apply_liquidity_fields_to_latest_trade(
        output,
        row,
        strategy_id=strategy.strategy_id,
        direction=direction,
        status="skipped",
        skip_reason=rejection_reason,
        liquidity_decision=liquidity_decision,
    )


def _record_realistic_execution_skip(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    rejection_reason: str,
    config: BaselinePaperTraderConfig,
    model_name: str,
    edge_inputs: dict[str, float | None],
    realistic_decision: _RealisticEntryDecision,
    realistic_execution_settings: MultiStrategyRealisticExecutionSettings,
    emit_logs: bool,
) -> None:
    threshold = config.long_threshold if direction == "YES" else config.short_threshold
    _record_skipped_trade(
        output,
        realistic_decision.row,
        probability=probability,
        direction=direction,
        threshold=threshold,
        skip_reason=rejection_reason,
        config=config,
        model_name=model_name,
        emit_logs=emit_logs,
        entry_price=edge_inputs.get("entry_price"),
        adjusted_entry_price=edge_inputs.get("adjusted_entry_price"),
        probability_for_direction=edge_inputs.get("probability_for_direction"),
        estimated_edge=edge_inputs.get("estimated_edge"),
    )
    _apply_realistic_fields_to_latest_trade(
        output,
        row,
        strategy_id=_strategy_id_for_config(config),
        direction=direction,
        status="skipped",
        skip_reason=rejection_reason,
        realistic_decision=realistic_decision,
        settings_enabled=True,
        entry_latency_sec=realistic_execution_settings.entry_latency_sec,
        exit_latency_sec=realistic_execution_settings.exit_latency_sec,
    )


def _strategy_id_for_config(config: BaselinePaperTraderConfig) -> str:
    return str(config.strategy_id or "baseline_default")


def _apply_realistic_fields_to_latest_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    strategy_id: str,
    direction: str,
    status: str,
    skip_reason: str | None,
    realistic_decision: _RealisticEntryDecision,
    settings_enabled: bool,
    entry_latency_sec: float | None,
    exit_latency_sec: float | None,
) -> None:
    query = """
        SELECT id
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND COALESCE(strategy_id, '') = ?
          AND signal_direction = ?
          AND status = ?
    """
    params: list[Any] = [
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(row.get("timestamp") or ""),
        strategy_id,
        direction,
        status,
    ]
    if skip_reason is not None:
        query += " AND skip_reason = ?"
        params.append(skip_reason)
    query += " ORDER BY id DESC LIMIT 1"
    trade = output.execute(query, params).fetchone()
    if trade is None:
        return
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET realistic_execution_enabled = ?,
                entry_latency_sec = ?,
                exit_latency_sec = ?,
                signal_entry_price = ?,
                delayed_entry_price = ?,
                entry_price_drift = ?,
                spread_cents_at_entry = ?,
                realistic_execution_skip_reason = ?,
                extra_slippage_cents_applied = ?
            WHERE id = ?
            """,
            (
                1 if settings_enabled else 0,
                _round_or_none(entry_latency_sec),
                _round_or_none(exit_latency_sec),
                _round_or_none(realistic_decision.signal_entry_price),
                _round_or_none(realistic_decision.delayed_entry_price),
                _round_or_none(realistic_decision.entry_price_drift),
                _round_or_none(realistic_decision.spread_cents_at_entry),
                realistic_decision.rejection_reason,
                _round_or_none(realistic_decision.extra_slippage_cents_applied),
                int(trade["id"]),
            ),
        )


def _apply_liquidity_fields_to_latest_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    strategy_id: str,
    direction: str,
    status: str,
    skip_reason: str | None,
    liquidity_decision: _LiquidityFillDecision,
) -> None:
    query = """
        SELECT id
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND COALESCE(strategy_id, '') = ?
          AND signal_direction = ?
          AND status = ?
    """
    params: list[Any] = [
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(row.get("timestamp") or ""),
        strategy_id,
        direction,
        status,
    ]
    if skip_reason is not None:
        query += " AND skip_reason = ?"
        params.append(skip_reason)
    query += " ORDER BY id DESC LIMIT 1"
    trade = output.execute(query, params).fetchone()
    if trade is None:
        return
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET requested_shares = ?,
                max_fillable_shares = ?,
                liquidity_fill_fraction_used = ?,
                liquidity_check_passed = ?,
                liquidity_skip_reason = ?
            WHERE id = ?
            """,
            (
                _round_or_none(liquidity_decision.requested_shares),
                _round_or_none(liquidity_decision.max_fillable_shares),
                _round_or_none(liquidity_decision.liquidity_fill_fraction_used),
                liquidity_decision.liquidity_check_passed,
                liquidity_decision.rejection_reason,
                int(trade["id"]),
            ),
        )


def _side_rejection_reason(
    *,
    probability: float,
    direction: str,
    edge_inputs: dict[str, float | None],
    strategy: MultiStrategyDefinition,
) -> str | None:
    probability_for_direction = edge_inputs.get("probability_for_direction")
    if probability_for_direction is None:
        return "threshold"
    if (
        strategy.block_probability_above is not None
        and float(probability_for_direction) >= float(strategy.block_probability_above)
    ):
        return "probability_block"
    if (
        strategy.block_probability_below is not None
        and float(probability_for_direction) <= float(strategy.block_probability_below)
    ):
        return "probability_block"
    if (
        strategy.min_probability_for_direction is not None
        and float(probability_for_direction) < float(strategy.min_probability_for_direction)
    ):
        return "probability_below_min"
    if (
        strategy.max_probability_for_direction is not None
        and float(probability_for_direction) > float(strategy.max_probability_for_direction)
    ):
        return "probability_above_max"
    if not _side_threshold_passed(
        probability=probability,
        direction=direction,
        strategy=strategy,
    ):
        return "threshold"
    return None


def _side_threshold_passed(
    *,
    probability: float,
    direction: str,
    strategy: MultiStrategyDefinition,
) -> bool:
    if direction == "YES":
        return float(probability) >= float(strategy.long_threshold)
    return float(probability) <= float(strategy.short_threshold)


def _primary_rejection(rejections: dict[str, str]) -> str | None:
    priority = (
        "time_window",
        "feature_stale",
        "btc_stale",
        "quote_after_latency_missing",
        "spread_too_wide",
        "entry_price_drift_too_high",
        "probability_block",
        "probability_below_min",
        "probability_above_max",
        "threshold",
        "non_positive_edge",
        "min_estimated_edge",
        "liquidity_regime_blocked",
        "max_open_trades",
        "one_trade_per_market",
    )
    for reason in priority:
        if reason in rejections.values():
            return reason
    return next(iter(rejections.values()), None)


def _predict_once_per_feature_row(
    model: Any,
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[tuple[str, str, str], float]:
    if not rows:
        return {}
    probabilities = _predict_probabilities(model, rows, feature_columns)
    return {
        _feature_key(row): float(probability)
        for row, probability in zip(rows, probabilities)
    }


def _baseline_config_for_strategy(
    strategy: MultiStrategyDefinition,
    *,
    recorder_db_path: str,
    output_db_path: str,
    model_path: str,
    feature_columns_path: str,
    max_btc_age_sec: float,
    max_feature_age_sec: float,
    require_fresh_comparison_btc: bool,
) -> BaselinePaperTraderConfig:
    allow_yes, allow_no = _direction_mode_flags(strategy.direction_mode)
    return BaselinePaperTraderConfig(
        recorder_db_path=recorder_db_path,
        output_db_path=output_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        strategy_id=strategy.strategy_id,
        strategy_name=strategy.strategy_name,
        long_threshold=strategy.long_threshold,
        short_threshold=strategy.short_threshold,
        max_btc_age_sec=max_btc_age_sec,
        max_feature_age_sec=max_feature_age_sec,
        min_time_until_resolution_sec=0.0,
        max_time_until_resolution_sec=1_000_000.0,
        one_trade_per_market=strategy.one_trade_per_market,
        min_estimated_edge=strategy.min_estimated_edge,
        max_open_trades=strategy.max_open_trades,
        block_probability_above=strategy.block_probability_above,
        block_probability_below=strategy.block_probability_below,
        min_probability_for_direction=strategy.min_probability_for_direction,
        max_probability_for_direction=strategy.max_probability_for_direction,
        allow_yes=allow_yes,
        allow_no=allow_no,
        require_fresh_comparison_btc=require_fresh_comparison_btc,
        stake_usd=strategy.stake_usd,
        entry_slippage_cents=strategy.entry_slippage_cents,
        fee_cents=strategy.fee_cents,
        dry_run=True,
    )


def _parse_strategy_definition(
    item: Any,
    *,
    index: int,
) -> MultiStrategyDefinition:
    if not isinstance(item, dict):
        raise MultiStrategyPaperTraderError(f"strategy at index {index} must be an object")
    strategy_id = _required_string(item, "strategy_id", index=index)
    if not _STRATEGY_ID_RE.match(strategy_id):
        raise MultiStrategyPaperTraderError(
            f"strategy_id {strategy_id!r} must contain only letters, numbers, '.', '_' or '-'"
        )
    direction_mode = _required_string(item, "direction_mode", index=index).upper()
    if direction_mode not in _DIRECTION_MODES:
        raise MultiStrategyPaperTraderError(
            f"strategy at index {index} has invalid direction_mode {direction_mode!r}"
        )
    exit_type = str(item.get("exit_type") or "HOLD_TO_RESOLUTION").upper()
    if exit_type not in _EXIT_TYPES:
        raise MultiStrategyPaperTraderError(
            f"strategy at index {index} has invalid exit_type {exit_type!r}"
        )
    return MultiStrategyDefinition(
        strategy_id=strategy_id,
        strategy_name=str(item.get("strategy_name") or strategy_id),
        enabled=_optional_bool(item.get("enabled"), default=True),
        direction_mode=direction_mode,
        long_threshold=_required_float(item, "long_threshold", index=index),
        short_threshold=_required_float(item, "short_threshold", index=index),
        min_estimated_edge=_required_float(item, "min_estimated_edge", index=index),
        entry_slippage_cents=_required_float(item, "entry_slippage_cents", index=index),
        fee_cents=_required_float(item, "fee_cents", index=index),
        stake_usd=_optional_float(item.get("stake_usd"), default=1.0),
        max_open_trades=_required_int(item, "max_open_trades", index=index),
        one_trade_per_market=_optional_bool(
            item.get("one_trade_per_market"),
            default=True,
        ),
        min_time_until_resolution_sec=_required_float(
            item,
            "min_time_until_resolution_sec",
            index=index,
        ),
        max_time_until_resolution_sec=_required_float(
            item,
            "max_time_until_resolution_sec",
            index=index,
        ),
        block_probability_above=_optional_float(item.get("block_probability_above")),
        block_probability_below=_optional_float(item.get("block_probability_below")),
        require_positive_edge=_optional_bool(
            item.get("require_positive_edge"),
            default=True,
        ),
        min_probability_for_direction=_optional_float(
            item.get("min_probability_for_direction")
        ),
        max_probability_for_direction=_optional_float(
            item.get("max_probability_for_direction")
        ),
        exit_type=exit_type,
        exit_slippage_cents=_optional_float(
            item.get("exit_slippage_cents"),
            default=0.0,
            field="exit_slippage_cents",
            index=index,
        )
        or 0.0,
        take_profit_pct=_optional_float(
            item.get("take_profit_pct"),
            field="take_profit_pct",
            index=index,
        ),
        stop_loss_pct=_optional_float(
            item.get("stop_loss_pct"),
            field="stop_loss_pct",
            index=index,
        ),
        fixed_horizon_exit_sec=_optional_float(
            item.get("fixed_horizon_exit_sec"),
            field="fixed_horizon_exit_sec",
            index=index,
        ),
        trailing_stop_pct=_optional_float(
            item.get("trailing_stop_pct"),
            field="trailing_stop_pct",
            index=index,
        ),
        blocked_liquidity_regimes=_blocked_liquidity_regimes(item, index=index),
        liquidity_fill_check_enabled=_optional_bool(
            item.get("liquidity_fill_check_enabled"),
            default=False,
        ),
        max_top_of_book_fill_fraction=_bounded_optional_float(
            item,
            "max_top_of_book_fill_fraction",
            default=1.0,
            minimum=0.0,
            inclusive_minimum=False,
            index=index,
        ),
        min_top_of_book_shares=_bounded_optional_float(
            item,
            "min_top_of_book_shares",
            default=0.0,
            minimum=0.0,
            inclusive_minimum=True,
            index=index,
        ),
        skip_if_liquidity_missing=_optional_bool(
            item.get("skip_if_liquidity_missing"),
            default=True,
        ),
    )


def _blocked_liquidity_regimes(item: dict[str, Any], *, index: int) -> tuple[str, ...]:
    raw = item.get("blocked_liquidity_regimes")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise MultiStrategyPaperTraderError(
            f"strategy at index {index} blocked_liquidity_regimes must be a list"
        )
    values: list[str] = []
    for value in raw:
        text = str(value).strip().lower()
        if not text:
            continue
        if text not in _LIQUIDITY_REGIMES:
            raise MultiStrategyPaperTraderError(
                f"strategy at index {index} blocked_liquidity_regimes contains invalid regime {value!r}"
            )
        values.append(text)
    return tuple(dict.fromkeys(values))


def _global_liquidity_settings(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    fields = {
        "liquidity_fill_check_enabled",
        "max_top_of_book_fill_fraction",
        "min_top_of_book_shares",
        "skip_if_liquidity_missing",
    }
    raw: dict[str, Any] = {field: payload[field] for field in fields if field in payload}
    nested = payload.get("liquidity")
    if isinstance(nested, dict):
        mapped = dict(nested)
        if "enabled" in mapped and "liquidity_fill_check_enabled" not in mapped:
            mapped["liquidity_fill_check_enabled"] = mapped["enabled"]
        if "fill_check_enabled" in mapped and "liquidity_fill_check_enabled" not in mapped:
            mapped["liquidity_fill_check_enabled"] = mapped["fill_check_enabled"]
        raw.update({field: mapped[field] for field in fields if field in mapped})
    return raw


def _merge_global_liquidity_settings(
    item: Any,
    global_liquidity: dict[str, Any],
) -> Any:
    if not isinstance(item, dict) or not global_liquidity:
        return item
    merged = dict(item)
    for field, value in global_liquidity.items():
        merged.setdefault(field, value)
    return merged


def _parse_bankroll_settings(payload: Any) -> MultiStrategyBankrollSettings:
    if not isinstance(payload, dict):
        return MultiStrategyBankrollSettings()
    fields = {
        "bankroll_enabled",
        "starting_bankroll_usd",
        "base_risk_fraction",
        "max_risk_fraction",
        "max_total_exposure_fraction",
        "min_stake_usd",
        "max_stake_usd",
        "stop_trading_on_bankroll_depleted",
        "reset_bankroll_on_new_run_id",
    }
    raw: dict[str, Any] = {field: payload[field] for field in fields if field in payload}
    nested = payload.get("bankroll")
    if isinstance(nested, dict):
        raw.update({field: nested[field] for field in fields if field in nested})
    return MultiStrategyBankrollSettings(
        bankroll_enabled=_optional_bool(raw.get("bankroll_enabled"), default=False),
        starting_bankroll_usd=_bankroll_float(
            raw,
            "starting_bankroll_usd",
            default=100.0,
            minimum=0.0,
        ),
        base_risk_fraction=_bankroll_float(
            raw,
            "base_risk_fraction",
            default=0.01,
            minimum=0.0,
        ),
        max_risk_fraction=_bankroll_float(
            raw,
            "max_risk_fraction",
            default=0.05,
            minimum=0.0,
        ),
        max_total_exposure_fraction=_bankroll_float(
            raw,
            "max_total_exposure_fraction",
            default=0.20,
            minimum=0.0,
        ),
        min_stake_usd=_bankroll_float(raw, "min_stake_usd", default=0.25, minimum=0.0),
        max_stake_usd=_bankroll_float(raw, "max_stake_usd", default=5.0, minimum=0.0),
        stop_trading_on_bankroll_depleted=_optional_bool(
            raw.get("stop_trading_on_bankroll_depleted"),
            default=True,
        ),
        reset_bankroll_on_new_run_id=_optional_bool(
            raw.get("reset_bankroll_on_new_run_id"),
            default=True,
        ),
    )


def _parse_realistic_execution_settings(
    payload: Any,
) -> MultiStrategyRealisticExecutionSettings:
    if not isinstance(payload, dict):
        return MultiStrategyRealisticExecutionSettings()
    fields = {
        "realistic_execution_enabled",
        "entry_latency_sec",
        "exit_latency_sec",
        "max_quote_wait_sec",
        "max_entry_price_drift_cents",
        "max_exit_price_drift_cents",
        "require_quote_after_latency",
        "reject_if_spread_above_cents",
        "reject_if_btc_age_above_sec",
        "reject_if_feature_age_above_sec",
        "apply_extra_slippage_cents",
        "partial_fill_enabled",
    }
    raw: dict[str, Any] = {field: payload[field] for field in fields if field in payload}
    nested = payload.get("realistic_execution")
    if isinstance(nested, dict):
        raw.update({field: nested[field] for field in fields if field in nested})
    return MultiStrategyRealisticExecutionSettings(
        realistic_execution_enabled=_optional_bool(
            raw.get("realistic_execution_enabled"),
            default=False,
        ),
        entry_latency_sec=_settings_float(raw, "entry_latency_sec", default=1.0, minimum=0.0),
        exit_latency_sec=_settings_float(raw, "exit_latency_sec", default=1.0, minimum=0.0),
        max_quote_wait_sec=_settings_float(
            raw,
            "max_quote_wait_sec",
            default=3.0,
            minimum=0.0,
        ),
        max_entry_price_drift_cents=_settings_float(
            raw,
            "max_entry_price_drift_cents",
            default=2.0,
            minimum=0.0,
        ),
        max_exit_price_drift_cents=_settings_float(
            raw,
            "max_exit_price_drift_cents",
            default=2.0,
            minimum=0.0,
        ),
        require_quote_after_latency=_optional_bool(
            raw.get("require_quote_after_latency"),
            default=True,
        ),
        reject_if_spread_above_cents=_settings_float(
            raw,
            "reject_if_spread_above_cents",
            default=4.0,
            minimum=0.0,
        ),
        reject_if_btc_age_above_sec=_settings_float(
            raw,
            "reject_if_btc_age_above_sec",
            default=5.0,
            minimum=0.0,
        ),
        reject_if_feature_age_above_sec=_settings_float(
            raw,
            "reject_if_feature_age_above_sec",
            default=5.0,
            minimum=0.0,
        ),
        apply_extra_slippage_cents=_settings_float(
            raw,
            "apply_extra_slippage_cents",
            default=1.0,
            minimum=0.0,
        ),
        partial_fill_enabled=_optional_bool(raw.get("partial_fill_enabled"), default=False),
    )


def _parse_trade_selection_settings(payload: Any) -> MultiStrategyTradeSelectionSettings:
    if not isinstance(payload, dict):
        return MultiStrategyTradeSelectionSettings()
    raw = payload.get("trade_selection")
    if raw is None:
        return MultiStrategyTradeSelectionSettings()
    if not isinstance(raw, dict):
        raise MultiStrategyPaperTraderError("trade_selection must be an object when provided")
    mode = str(raw.get("mode") or "best_per_market_direction")
    if mode != "best_per_market_direction":
        raise MultiStrategyPaperTraderError(
            "trade_selection.mode must be 'best_per_market_direction'"
        )
    return MultiStrategyTradeSelectionSettings(
        selection_enabled=_optional_bool(raw.get("selection_enabled"), default=False),
        mode=mode,
        score_field=str(raw.get("score_field") or "estimated_edge"),
        max_new_trades_per_market=_settings_int(
            raw,
            "max_new_trades_per_market",
            default=1,
            minimum=1,
        ),
        max_new_trades_per_market_direction=_settings_int(
            raw,
            "max_new_trades_per_market_direction",
            default=1,
            minimum=1,
        ),
        allow_multiple_strategy_variants_same_market=_optional_bool(
            raw.get("allow_multiple_strategy_variants_same_market"),
            default=False,
        ),
    )


def _parse_live_trading_settings(payload: Any) -> MultiStrategyLiveTradingSettings:
    if not isinstance(payload, dict):
        return MultiStrategyLiveTradingSettings()
    fields = {
        "live_trading_enabled",
        "dry_run_orders",
        "max_order_usd",
        "max_daily_loss_usd",
        "max_daily_orders",
        "max_open_exposure_usd",
        "max_open_trades",
        "max_trades_per_market",
        "max_trades_per_market_direction",
        "require_realistic_execution_passed",
        "require_liquidity_check_passed",
        "reject_if_btc_age_above_sec",
        "reject_if_feature_age_above_sec",
        "reject_if_spread_above_cents",
        "kill_switch_file",
        "allow_market_order",
        "use_limit_orders_only",
    }
    raw: dict[str, Any] = {field: payload[field] for field in fields if field in payload}
    nested = payload.get("live_trading")
    if isinstance(nested, dict):
        raw.update({field: nested[field] for field in fields if field in nested})
    elif nested is not None:
        raise MultiStrategyPaperTraderError("live_trading must be an object when provided")
    kill_switch_file = str(raw.get("kill_switch_file") or "data/KILL_LIVE_TRADING").strip()
    if not kill_switch_file:
        raise MultiStrategyPaperTraderError("kill_switch_file must be non-empty")
    return MultiStrategyLiveTradingSettings(
        live_trading_enabled=_optional_bool(raw.get("live_trading_enabled"), default=False),
        dry_run_orders=_optional_bool(raw.get("dry_run_orders"), default=True),
        max_order_usd=_settings_float(raw, "max_order_usd", default=1.0, minimum=0.0),
        max_daily_loss_usd=_settings_float(
            raw,
            "max_daily_loss_usd",
            default=5.0,
            minimum=0.0,
        ),
        max_daily_orders=_settings_int(raw, "max_daily_orders", default=20, minimum=1),
        max_open_exposure_usd=_settings_float(
            raw,
            "max_open_exposure_usd",
            default=5.0,
            minimum=0.0,
        ),
        max_open_trades=_settings_int(raw, "max_open_trades", default=3, minimum=1),
        max_trades_per_market=_settings_int(
            raw,
            "max_trades_per_market",
            default=1,
            minimum=1,
        ),
        max_trades_per_market_direction=_settings_int(
            raw,
            "max_trades_per_market_direction",
            default=1,
            minimum=1,
        ),
        require_realistic_execution_passed=_optional_bool(
            raw.get("require_realistic_execution_passed"),
            default=True,
        ),
        require_liquidity_check_passed=_optional_bool(
            raw.get("require_liquidity_check_passed"),
            default=True,
        ),
        reject_if_btc_age_above_sec=_settings_float(
            raw,
            "reject_if_btc_age_above_sec",
            default=5.0,
            minimum=0.0,
        ),
        reject_if_feature_age_above_sec=_settings_float(
            raw,
            "reject_if_feature_age_above_sec",
            default=5.0,
            minimum=0.0,
        ),
        reject_if_spread_above_cents=_settings_float(
            raw,
            "reject_if_spread_above_cents",
            default=3.0,
            minimum=0.0,
        ),
        kill_switch_file=kill_switch_file,
        allow_market_order=_optional_bool(raw.get("allow_market_order"), default=False),
        use_limit_orders_only=_optional_bool(raw.get("use_limit_orders_only"), default=True),
    )


def _settings_float(
    item: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
) -> float:
    value = _optional_float(item.get(key), default=default, field=key)
    if value is None:
        return default
    if value < minimum:
        raise MultiStrategyPaperTraderError(f"{key} must be >= {minimum}")
    return value


def _settings_int(
    item: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = item.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise MultiStrategyPaperTraderError(f"{key} must be an integer") from None
    if parsed < minimum:
        raise MultiStrategyPaperTraderError(f"{key} must be >= {minimum}")
    return parsed


def _bankroll_float(
    item: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
) -> float:
    value = _optional_float(item.get(key), default=default, field=key)
    if value is None:
        return default
    if value < minimum:
        raise MultiStrategyPaperTraderError(f"{key} must be >= {minimum}")
    return value


def _bounded_optional_float(
    item: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    inclusive_minimum: bool,
    index: int,
) -> float:
    value = _optional_float(item.get(key), default=default, field=key, index=index)
    if value is None:
        return default
    if inclusive_minimum and value < minimum:
        raise MultiStrategyPaperTraderError(f"{key} must be >= {minimum}")
    if not inclusive_minimum and value <= minimum:
        raise MultiStrategyPaperTraderError(f"{key} must be > {minimum}")
    return value


def _direction_mode_flags(direction_mode: str) -> tuple[bool, bool]:
    if direction_mode == "YES_ONLY":
        return True, False
    if direction_mode == "NO_ONLY":
        return False, True
    return True, True


def _required_string(item: dict[str, Any], key: str, *, index: int) -> str:
    value = item.get(key)
    if value is None or str(value).strip() == "":
        raise MultiStrategyPaperTraderError(f"strategy at index {index} missing required {key}")
    return str(value).strip()


def _required_float(item: dict[str, Any], key: str, *, index: int) -> float:
    if key not in item:
        raise MultiStrategyPaperTraderError(f"strategy at index {index} missing required {key}")
    value = _optional_float(item.get(key), field=key, index=index)
    if value is None:
        raise MultiStrategyPaperTraderError(f"strategy at index {index} missing required {key}")
    return value


def _required_int(item: dict[str, Any], key: str, *, index: int) -> int:
    if key not in item:
        raise MultiStrategyPaperTraderError(f"strategy at index {index} missing required {key}")
    try:
        return int(item[key])
    except (TypeError, ValueError) as exc:
        raise MultiStrategyPaperTraderError(
            f"strategy at index {index} has invalid {key}"
        ) from exc


def _optional_float(
    value: Any,
    *,
    default: float | None = None,
    field: str = "value",
    index: int = 0,
) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MultiStrategyPaperTraderError(
            f"strategy at index {index} has invalid {field}"
        ) from exc


def _optional_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise MultiStrategyPaperTraderError(f"invalid boolean value {value!r}")


def _missing_feature_columns(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [column for column in feature_columns if column not in keys]


def _feature_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
        str(row.get("timestamp") or ""),
    )


def _candidate_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM paper_trade_candidates").fetchone()[0] or 0)


def _trade_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] or 0)


def _strategy_trade_count(conn: sqlite3.Connection, strategy_id: str, status: str) -> int:
    return _paper_status_count(conn, status, strategy_id=strategy_id)


def _strategy_candidate_count(conn: sqlite3.Connection, strategy_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM paper_trade_candidates
            WHERE strategy_id = ?
            """,
            (strategy_id,),
        ).fetchone()[0]
        or 0
    )


def _latest_strategy_rejection(conn: sqlite3.Connection, strategy_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT rejection_reason
        FROM paper_trade_candidates
        WHERE strategy_id = ?
          AND rejection_reason IS NOT NULL
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    return str(row["rejection_reason"]) if row is not None else None


def _summarize_strategy_stats(per_strategy: Sequence[dict[str, Any]]) -> dict[str, Any]:
    skip_reasons = Counter()
    for item in per_strategy:
        for key, value in item.items():
            parsed = _float_or_none(value)
            if key.startswith("skipped_") and parsed:
                skip_reasons[key.removeprefix("skipped_")] += int(parsed)
    active = [
        item
        for item in per_strategy
        if int(item.get("trades_opened") or 0) > 0
        or int(item.get("skipped_candidates") or 0) > 0
        or item.get("latest_rejection_reason")
        or item.get("error")
    ]
    active.sort(
        key=lambda item: (
            int(item.get("trades_opened") or 0),
            int(item.get("skipped_candidates") or 0),
            str(item.get("strategy_id") or ""),
        ),
        reverse=True,
    )
    return {
        "strategy_count": len(per_strategy),
        "active_strategy_count": len(active),
        "candidate_decisions_evaluated_this_poll": sum(
            int(item.get("candidate_decisions_evaluated_this_poll") or 0)
            for item in per_strategy
        ),
        "trades_opened": sum(int(item.get("trades_opened") or 0) for item in per_strategy),
        "skipped_candidates": sum(int(item.get("skipped_candidates") or 0) for item in per_strategy),
        "top_skip_reasons": dict(skip_reasons.most_common(10)),
        "top_active_strategies": [
            {
                "strategy_id": item.get("strategy_id"),
                "trades_opened": item.get("trades_opened"),
                "skipped_candidates": item.get("skipped_candidates"),
                "latest_decision": item.get("latest_decision"),
                "latest_rejection_reason": item.get("latest_rejection_reason"),
                "latest_feature_timestamp": item.get("latest_feature_timestamp"),
            }
            for item in active[:10]
        ],
    }


def _heartbeat_log_payload(report: dict[str, Any], *, detail: str) -> dict[str, Any]:
    if str(detail).lower() == "full":
        return report
    payload = dict(report)
    summary = report.get("strategy_stats_summary") or _summarize_strategy_stats(
        list(report.get("strategy_stats") or [])
    )
    payload["strategy_stats"] = summary
    payload["strategies"] = summary
    return payload


def _write_multi_strategy_heartbeat(
    conn: sqlite3.Connection,
    report: dict[str, Any],
) -> None:
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO multi_strategy_paper_trader_heartbeats (
                timestamp, status, run_id, active_strategies, latest_market_id,
                latest_feature_timestamp, probability_yes, candidates_logged,
                candidate_rows_total, candidate_rows_written_this_poll,
                candidate_decisions_evaluated_this_poll, heartbeat_detail,
                trades_opened, skips_logged, liquidity_fill_check_enabled,
                liquidity_checked_count, liquidity_passed_count,
                liquidity_blocked_count, liquidity_missing_count,
                pending_realistic_entries, pending_realistic_entries_ready,
                pending_realistic_entries_expired,
                realistic_entries_opened_after_latency,
                realistic_entries_skipped_after_latency,
                latest_pending_target_quote_timestamp,
                trade_selection_enabled,
                trade_selection_candidates_before,
                trade_selection_candidates_after,
                trade_selection_suppressed_count,
                per_strategy_json, strategy_errors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["timestamp"],
                str(report.get("status") or "ok"),
                report.get("run_id"),
                int(report.get("active_strategies") or 0),
                report.get("latest_market_id"),
                report.get("latest_feature_timestamp"),
                _float_or_none(report.get("probability_yes")),
                int(report.get("candidates_logged") or 0),
                int(report.get("candidate_rows_total") or 0),
                int(report.get("candidate_rows_written_this_poll") or 0),
                int(report.get("candidate_decisions_evaluated_this_poll") or 0),
                report.get("heartbeat_detail"),
                int(report.get("trades_opened") or 0),
                int(report.get("skips_logged") or 0),
                1 if report.get("liquidity_fill_check_enabled") else 0,
                int(report.get("liquidity_checked_count") or 0),
                int(report.get("liquidity_passed_count") or 0),
                int(report.get("liquidity_blocked_count") or 0),
                int(report.get("liquidity_missing_count") or 0),
                int(report.get("pending_realistic_entries") or 0),
                int(report.get("pending_realistic_entries_ready") or 0),
                int(report.get("pending_realistic_entries_expired") or 0),
                int(report.get("realistic_entries_opened_after_latency") or 0),
                int(report.get("realistic_entries_skipped_after_latency") or 0),
                report.get("latest_pending_target_quote_timestamp"),
                1 if report.get("trade_selection_enabled") else 0,
                int(report.get("trade_selection_candidates_before") or 0),
                int(report.get("trade_selection_candidates_after") or 0),
                int(report.get("trade_selection_suppressed_count") or 0),
                json.dumps(report.get("strategies") or [], sort_keys=True),
                json.dumps(report.get("strategy_errors") or {}, sort_keys=True),
            ),
        )


def _emit(enabled: bool, event: str, **fields: Any) -> None:
    if not enabled:
        return
    payload = {
        "event": event,
        "timestamp": to_iso(datetime.now(timezone.utc)),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


__all__ = [
    "MultiStrategyBankrollSettings",
    "MultiStrategyDefinition",
    "MultiStrategyLiveTradingSettings",
    "MultiStrategyPaperTraderError",
    "MultiStrategyRealisticExecutionSettings",
    "MultiStrategyTradeSelectionSettings",
    "ensure_multi_strategy_schema",
    "load_multi_strategy_config",
    "run_multi_strategy_paper_trader",
    "run_multi_strategy_paper_trader_once",
    "_regime_tags_for_row",
    "update_multi_strategy_trade_exits",
]
