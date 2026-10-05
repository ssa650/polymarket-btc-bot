from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .backtest_baseline_strategy import (
    _direction_for_probability,
    _load_feature_columns,
    _load_model,
    _predict_probabilities,
)
from .btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none


PAPER_TRADE_STATUSES = {"open", "awaiting_resolution", "settled", "skipped"}
REAL_PAPER_TRADE_STATUSES = {"open", "awaiting_resolution", "settled"}
_LOGGED_MISSING_FEATURE_SETS: set[tuple[str, ...]] = set()

HEARTBEAT_SKIP_REASONS: tuple[str, ...] = (
    "not_feature_ready",
    "gap_affected",
    "strict_validation_failed",
    "snapshot_quality_not_ok",
    "missing_orderbook",
    "missing_trade_data",
    "stale_canonical_btc",
    "stale_comparison_btc",
    "comparison_btc_stale_warning",
    "missing_price",
    "no_signal",
    "threshold",
    "max_open_trades",
    "one_trade_per_market",
    "stale_feature",
    "market_closed",
    "expired_feature",
    "time_window",
    "missing_btc_price_at_market_start",
    "invalid_seconds_after_start",
    "invalid_entry_price",
    "min_estimated_edge",
    "probability_block",
    "yes_disabled",
    "no_disabled",
)


class BaselinePaperTraderError(RuntimeError):
    pass


@dataclass(slots=True)
class BaselinePaperTraderConfig:
    recorder_db_path: str
    output_db_path: str
    model_path: str
    feature_columns_path: str
    strategy_id: str | None = None
    strategy_name: str | None = None
    poll_sec: float = 1.0
    long_threshold: float = 0.65
    short_threshold: float = 0.35
    max_btc_age_sec: float = 3.0
    max_feature_age_sec: float = 5.0
    min_time_until_resolution_sec: float = 0.0
    max_time_until_resolution_sec: float = 300.0
    one_trade_per_market: bool = True
    min_estimated_edge: float = 0.0
    max_open_trades: int = 1
    block_probability_above: float | None = None
    block_probability_below: float | None = None
    min_probability_for_direction: float | None = None
    max_probability_for_direction: float | None = None
    allow_yes: bool = True
    allow_no: bool = True
    require_fresh_comparison_btc: bool = False
    stake_usd: float = 1.0
    entry_slippage_cents: float = 0.01
    fee_cents: float = 0.0
    dry_run: bool = True


def run_baseline_paper_trader(
    config: BaselinePaperTraderConfig,
    *,
    max_iterations: int | None = None,
    emit_logs: bool = True,
) -> dict[str, Any]:
    model = _load_model(config.model_path)
    feature_columns = _load_feature_columns(config.feature_columns_path)
    recorder = connect_recorder_read_only(config.recorder_db_path)
    output = connect_paper_output_db(config.output_db_path)
    try:
        ensure_paper_schema(output)
        _emit(
            emit_logs,
            "paper_trader_started",
            recorder_db_path=config.recorder_db_path,
            output_db_path=config.output_db_path,
            model_path=config.model_path,
            dry_run=config.dry_run,
        )
        iterations = 0
        while True:
            iterations += 1
            try:
                poll_report = run_baseline_paper_trader_once(
                    recorder,
                    output,
                    model=model,
                    feature_columns=feature_columns,
                    config=config,
                    emit_logs=emit_logs,
                )
            except Exception as exc:
                _write_heartbeat(
                    output,
                    recorder_db_path=config.recorder_db_path,
                    output_db_path=config.output_db_path,
                latest_feature_timestamp=None,
                latest_seen_feature_timestamp=None,
                latest_eligible_feature_timestamp=None,
                latest_btc_chainlink_age_sec=None,
                latest_btc_binance_age_sec=None,
                latest_canonical_btc_age_sec=None,
                latest_comparison_btc_age_sec=None,
                skip_counts=Counter(),
                latest_rejection=None,
                loop_error=str(exc),
                )
                _emit(emit_logs, "paper_trader_error", error=str(exc))
                poll_report = {"status": "error", "error": str(exc)}
            if max_iterations is not None and iterations >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iterations,
                    "last_poll": poll_report,
                    "output_db_path": config.output_db_path,
                }
            time.sleep(max(0.0, float(config.poll_sec)))
    finally:
        recorder.close()
        output.close()


def run_baseline_paper_trader_once(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    model: Any,
    feature_columns: Sequence[str],
    config: BaselinePaperTraderConfig,
    emit_logs: bool = True,
) -> dict[str, Any]:
    ensure_paper_schema(output)
    now_dt = datetime.now(timezone.utc)
    settled = settle_open_paper_trades(
        recorder,
        output,
        model_path=config.model_path,
        emit_logs=emit_logs,
        now=now_dt,
    )
    candidates = fetch_latest_candidate_rows(
        recorder,
        max_btc_age_sec=config.max_btc_age_sec,
        min_time_until_resolution_sec=config.min_time_until_resolution_sec,
        max_time_until_resolution_sec=config.max_time_until_resolution_sec,
        now=now_dt,
        max_feature_age_sec=config.max_feature_age_sec,
    )
    latest_seen_feature_timestamp = _latest_feature_timestamp(recorder)
    live_rows, live_skip_counts, latest_rejection = _filter_candidate_eligibility(
        candidates,
        config=config,
        now=now_dt,
        emit_logs=emit_logs,
    )
    latest_eligible_feature_timestamp = (
        live_rows[0].get("timestamp") if live_rows else None
    )
    latest_feature_timestamp = latest_eligible_feature_timestamp or latest_seen_feature_timestamp
    latest_chain_age = _latest_btc_age(candidates, "btc_chainlink_age_sec_at_feature")
    latest_binance_age = _latest_btc_age(candidates, "btc_binance_age_sec_at_feature")

    missing_columns = _missing_feature_columns(live_rows, feature_columns)
    if missing_columns:
        error = "missing_feature_columns: " + ", ".join(missing_columns)
        _write_heartbeat(
            output,
            recorder_db_path=config.recorder_db_path,
            output_db_path=config.output_db_path,
            latest_feature_timestamp=latest_feature_timestamp,
            latest_seen_feature_timestamp=latest_seen_feature_timestamp,
            latest_eligible_feature_timestamp=latest_eligible_feature_timestamp,
            latest_btc_chainlink_age_sec=latest_chain_age,
            latest_btc_binance_age_sec=latest_binance_age,
            latest_canonical_btc_age_sec=latest_chain_age,
            latest_comparison_btc_age_sec=latest_binance_age,
            skip_counts=live_skip_counts,
            latest_rejection=latest_rejection,
            loop_error=error,
        )
        _emit_missing_feature_error(emit_logs, missing_columns)
        return {
            "status": "skipped",
            "reason": "missing_feature_columns",
            "missing_feature_columns": missing_columns,
            "settled_trades": settled,
            **_skip_count_report(live_skip_counts),
        }

    opened = 0
    skipped = 0
    model_name = _model_name(model)
    prediction_rows, prerequisite_counts, prereq_rejection = _filter_prediction_prerequisites(
        live_rows,
        feature_columns,
        emit_logs=emit_logs,
    )
    _merge_counts(live_skip_counts, prerequisite_counts)
    latest_rejection = _latest_rejection(latest_rejection, prereq_rejection)
    skipped += sum(live_skip_counts.values())
    probabilities = _predict_probabilities(model, prediction_rows, feature_columns)
    for row, probability in zip(prediction_rows, probabilities):
        result = _process_signal_row(
            output,
            row,
            probability=float(probability),
            config=config,
            model_name=model_name,
            emit_logs=emit_logs,
        )
        if result == "opened":
            opened += 1
        elif result != "none":
            skipped += 1
            live_skip_counts[result] += 1
            latest_rejection = _latest_rejection(
                latest_rejection,
                _rejection_payload(result, row),
            )

    _write_heartbeat(
        output,
        recorder_db_path=config.recorder_db_path,
        output_db_path=config.output_db_path,
        latest_feature_timestamp=latest_feature_timestamp,
        latest_seen_feature_timestamp=latest_seen_feature_timestamp,
        latest_eligible_feature_timestamp=latest_eligible_feature_timestamp,
        latest_btc_chainlink_age_sec=latest_chain_age,
        latest_btc_binance_age_sec=latest_binance_age,
        latest_canonical_btc_age_sec=latest_chain_age,
        latest_comparison_btc_age_sec=latest_binance_age,
        skip_counts=live_skip_counts,
        latest_rejection=latest_rejection,
        loop_error=None,
    )
    _emit(
        emit_logs,
        "paper_trader_heartbeat",
        candidates=len(candidates),
        opened_trades=opened,
        skipped_trades=skipped,
        settled_trades=settled,
        awaiting_resolution_trades=_paper_status_count(output, "awaiting_resolution"),
        latest_feature_timestamp=latest_feature_timestamp,
        latest_seen_feature_timestamp=latest_seen_feature_timestamp,
        latest_eligible_feature_timestamp=latest_eligible_feature_timestamp,
        latest_rejection_reason=(latest_rejection or {}).get("reason"),
        latest_rejection_market_id=(latest_rejection or {}).get("market_id"),
        latest_rejection_feature_timestamp=(latest_rejection or {}).get("feature_timestamp"),
        **_skip_count_report(live_skip_counts),
    )
    return {
        "status": "ok",
        "candidate_rows": len(candidates),
        "live_candidate_rows": len(live_rows),
        "prediction_rows": len(prediction_rows),
        "opened_trades": opened,
        "skipped_trades": skipped,
        "settled_trades": settled,
        **_skip_count_report(live_skip_counts),
    }


def connect_recorder_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def connect_paper_output_db(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_paper_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                signal_timestamp TEXT NOT NULL,
                question TEXT,
                market_start_time TEXT,
                market_close_time TEXT,
                time_until_resolution REAL,
                model_path TEXT,
                model_name TEXT,
                strategy_id TEXT,
                strategy_name TEXT,
                predicted_probability_yes REAL,
                probability_for_direction REAL,
                estimated_edge REAL,
                signal_direction TEXT NOT NULL,
                threshold_used REAL,
                stake_usd REAL NOT NULL,
                entry_price REAL,
                adjusted_entry_price REAL,
                best_bid_yes REAL,
                best_ask_yes REAL,
                best_bid_no REAL,
                best_ask_no REAL,
                btc_chainlink_price REAL,
                btc_binance_price REAL,
                btc_chainlink_age_sec_at_feature REAL,
                btc_binance_age_sec_at_feature REAL,
                feature_ready INTEGER,
                strict_validation_passed INTEGER,
                snapshot_quality_status TEXT,
                status TEXT NOT NULL,
                skip_reason TEXT,
                exit_type TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                exit_price REAL,
                adjusted_exit_price REAL,
                realized_pnl_usd REAL,
                realized_roi REAL,
                max_favorable_price REAL,
                max_adverse_price REAL,
                exit_slippage_cents REAL,
                take_profit_pct REAL,
                stop_loss_pct REAL,
                fixed_horizon_exit_sec REAL,
                trailing_stop_pct REAL,
                starting_bankroll_usd REAL,
                bankroll_before_trade REAL,
                available_cash_before_trade REAL,
                open_exposure_before_trade REAL,
                stake_fraction_of_bankroll REAL,
                max_open_exposure_usd REAL,
                bankroll_after_trade REAL,
                bankroll_status TEXT,
                blown_up_at TEXT,
                risk_sizing_reason TEXT,
                btc_trend_regime TEXT,
                volatility_regime TEXT,
                spread_regime TEXT,
                liquidity_regime TEXT,
                time_regime TEXT,
                requested_shares REAL,
                max_fillable_shares REAL,
                liquidity_fill_fraction_used REAL,
                liquidity_check_passed INTEGER,
                liquidity_skip_reason TEXT,
                realistic_execution_enabled INTEGER,
                entry_latency_sec REAL,
                exit_latency_sec REAL,
                signal_entry_price REAL,
                delayed_entry_price REAL,
                entry_price_drift REAL,
                spread_cents_at_entry REAL,
                realistic_execution_skip_reason TEXT,
                extra_slippage_cents_applied REAL,
                resolved_label TEXT,
                payout_usd REAL,
                pnl_usd REAL,
                roi REAL,
                settled_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trader_heartbeats (
                timestamp TEXT PRIMARY KEY,
                recorder_db_path TEXT,
                output_db_path TEXT,
                latest_feature_timestamp TEXT,
                latest_seen_feature_timestamp TEXT,
                latest_eligible_feature_timestamp TEXT,
                latest_btc_chainlink_age_sec REAL,
                latest_btc_binance_age_sec REAL,
                latest_canonical_btc_age_sec REAL,
                latest_comparison_btc_age_sec REAL,
                open_trades INTEGER,
                awaiting_resolution_trades INTEGER,
                settled_trades INTEGER,
                total_trades INTEGER,
                stale_feature_skips INTEGER,
                market_closed_skips INTEGER,
                expired_feature_skips INTEGER,
                not_feature_ready_skips INTEGER,
                gap_affected_skips INTEGER,
                strict_validation_failed_skips INTEGER,
                snapshot_quality_not_ok_skips INTEGER,
                missing_orderbook_skips INTEGER,
                missing_trade_data_skips INTEGER,
                canonical_btc_stale_skips INTEGER,
                comparison_btc_stale_skips INTEGER,
                comparison_btc_stale_warnings INTEGER,
                missing_price_skips INTEGER,
                no_signal_skips INTEGER,
                threshold_skips INTEGER,
                max_open_trades_skips INTEGER,
                one_trade_per_market_skips INTEGER,
                time_window_skips INTEGER,
                missing_btc_price_at_market_start_skips INTEGER,
                invalid_seconds_after_start_skips INTEGER,
                invalid_entry_price_skips INTEGER,
                min_estimated_edge_skips INTEGER,
                probability_block_skips INTEGER,
                latest_rejection_reason TEXT,
                latest_rejection_market_id TEXT,
                latest_rejection_feature_timestamp TEXT,
                loop_error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trade_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_id TEXT,
                market_id TEXT NOT NULL,
                question TEXT,
                feature_timestamp TEXT NOT NULL,
                market_start_time TEXT,
                market_close_time TEXT,
                strategy_id TEXT,
                strategy_name TEXT,
                candidate_direction TEXT,
                candidate_entry_price REAL,
                candidate_adjusted_entry_price REAL,
                stake_usd REAL,
                base_model_name TEXT,
                base_model_path TEXT,
                predicted_probability_yes REAL,
                probability_for_direction REAL,
                estimated_edge REAL,
                threshold_used REAL,
                min_estimated_edge_used REAL,
                time_until_resolution REAL,
                min_time_until_resolution_sec REAL,
                max_time_until_resolution_sec REAL,
                block_probability_above REAL,
                block_probability_below REAL,
                min_probability_for_direction REAL,
                max_probability_for_direction REAL,
                max_open_trades INTEGER,
                best_bid_yes REAL,
                best_ask_yes REAL,
                best_bid_no REAL,
                best_ask_no REAL,
                btc_chainlink_price REAL,
                btc_binance_price REAL,
                btc_chainlink_age_sec_at_feature REAL,
                btc_binance_age_sec_at_feature REAL,
                feature_ready INTEGER,
                strict_validation_passed INTEGER,
                snapshot_quality_status TEXT,
                decision TEXT NOT NULL,
                rejection_reason TEXT,
                linked_paper_trade_id INTEGER,
                actual_resolved_label TEXT,
                would_have_payout_usd REAL,
                would_have_pnl_usd REAL,
                would_have_roi REAL,
                settled_at TEXT
            )
            """
        )
        _ensure_column(conn, "paper_trades", "probability_for_direction", "REAL")
        _ensure_column(conn, "paper_trades", "estimated_edge", "REAL")
        _ensure_column(conn, "paper_trades", "strategy_id", "TEXT")
        _ensure_column(conn, "paper_trades", "strategy_name", "TEXT")
        for column, definition in {
            "exit_type": "TEXT",
            "exit_time": "TEXT",
            "exit_reason": "TEXT",
            "exit_price": "REAL",
            "adjusted_exit_price": "REAL",
            "realized_pnl_usd": "REAL",
            "realized_roi": "REAL",
            "max_favorable_price": "REAL",
            "max_adverse_price": "REAL",
            "exit_slippage_cents": "REAL",
            "take_profit_pct": "REAL",
            "stop_loss_pct": "REAL",
            "fixed_horizon_exit_sec": "REAL",
            "trailing_stop_pct": "REAL",
            "starting_bankroll_usd": "REAL",
            "bankroll_before_trade": "REAL",
            "available_cash_before_trade": "REAL",
            "open_exposure_before_trade": "REAL",
            "stake_fraction_of_bankroll": "REAL",
            "max_open_exposure_usd": "REAL",
            "bankroll_after_trade": "REAL",
            "bankroll_status": "TEXT",
            "blown_up_at": "TEXT",
            "risk_sizing_reason": "TEXT",
            "btc_trend_regime": "TEXT",
            "volatility_regime": "TEXT",
            "spread_regime": "TEXT",
            "liquidity_regime": "TEXT",
            "time_regime": "TEXT",
            "requested_shares": "REAL",
            "max_fillable_shares": "REAL",
            "liquidity_fill_fraction_used": "REAL",
            "liquidity_check_passed": "INTEGER",
            "liquidity_skip_reason": "TEXT",
            "realistic_execution_enabled": "INTEGER",
            "entry_latency_sec": "REAL",
            "exit_latency_sec": "REAL",
            "signal_entry_price": "REAL",
            "delayed_entry_price": "REAL",
            "entry_price_drift": "REAL",
            "spread_cents_at_entry": "REAL",
            "realistic_execution_skip_reason": "TEXT",
            "extra_slippage_cents_applied": "REAL",
        }.items():
            _ensure_column(conn, "paper_trades", column, definition)
        _ensure_column(conn, "paper_trader_heartbeats", "latest_seen_feature_timestamp", "TEXT")
        _ensure_column(conn, "paper_trader_heartbeats", "latest_eligible_feature_timestamp", "TEXT")
        _ensure_column(conn, "paper_trader_heartbeats", "latest_canonical_btc_age_sec", "REAL")
        _ensure_column(conn, "paper_trader_heartbeats", "latest_comparison_btc_age_sec", "REAL")
        _ensure_column(conn, "paper_trader_heartbeats", "awaiting_resolution_trades", "INTEGER DEFAULT 0")
        _ensure_column(conn, "paper_trader_heartbeats", "settled_trades", "INTEGER DEFAULT 0")
        for reason in HEARTBEAT_SKIP_REASONS:
            _ensure_column(
                conn,
                "paper_trader_heartbeats",
                _heartbeat_counter_column(reason),
                "INTEGER DEFAULT 0",
            )
        _ensure_column(conn, "paper_trader_heartbeats", "latest_rejection_reason", "TEXT")
        _ensure_column(conn, "paper_trader_heartbeats", "latest_rejection_market_id", "TEXT")
        _ensure_column(conn, "paper_trader_heartbeats", "latest_rejection_feature_timestamp", "TEXT")
        for column, definition in _paper_trade_candidate_columns().items():
            _ensure_column(conn, "paper_trade_candidates", column, definition)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trades_market_status
            ON paper_trades (run_id, market_id, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trades_strategy_status
            ON paper_trades (strategy_id, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trades_signal_time
            ON paper_trades (signal_timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trade_candidates_market
            ON paper_trade_candidates (market_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trade_candidates_run_feature
            ON paper_trade_candidates (run_id, feature_timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trade_candidates_decision
            ON paper_trade_candidates (decision)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trade_candidates_linked_trade
            ON paper_trade_candidates (linked_paper_trade_id)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_trade_candidates_dedupe
            ON paper_trade_candidates (
                COALESCE(run_id, ''),
                market_id,
                feature_timestamp,
                COALESCE(candidate_direction, ''),
                COALESCE(strategy_id, '')
            )
            """
        )


def _paper_trade_candidate_columns() -> dict[str, str]:
    return {
        "created_at": "TEXT",
        "run_id": "TEXT",
        "market_id": "TEXT",
        "question": "TEXT",
        "feature_timestamp": "TEXT",
        "market_start_time": "TEXT",
        "market_close_time": "TEXT",
        "strategy_id": "TEXT",
        "strategy_name": "TEXT",
        "candidate_direction": "TEXT",
        "candidate_entry_price": "REAL",
        "candidate_adjusted_entry_price": "REAL",
        "stake_usd": "REAL",
        "base_model_name": "TEXT",
        "base_model_path": "TEXT",
        "predicted_probability_yes": "REAL",
        "probability_for_direction": "REAL",
        "estimated_edge": "REAL",
        "threshold_used": "REAL",
        "min_estimated_edge_used": "REAL",
        "time_until_resolution": "REAL",
        "min_time_until_resolution_sec": "REAL",
        "max_time_until_resolution_sec": "REAL",
        "block_probability_above": "REAL",
        "block_probability_below": "REAL",
        "min_probability_for_direction": "REAL",
        "max_probability_for_direction": "REAL",
        "max_open_trades": "INTEGER",
        "best_bid_yes": "REAL",
        "best_ask_yes": "REAL",
        "best_bid_no": "REAL",
        "best_ask_no": "REAL",
        "btc_chainlink_price": "REAL",
        "btc_binance_price": "REAL",
        "btc_chainlink_age_sec_at_feature": "REAL",
        "btc_binance_age_sec_at_feature": "REAL",
        "feature_ready": "INTEGER",
        "strict_validation_passed": "INTEGER",
        "snapshot_quality_status": "TEXT",
        "decision": "TEXT",
        "rejection_reason": "TEXT",
        "linked_paper_trade_id": "INTEGER",
        "actual_resolved_label": "TEXT",
        "would_have_payout_usd": "REAL",
        "would_have_pnl_usd": "REAL",
        "would_have_roi": "REAL",
        "settled_at": "TEXT",
    }


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


def fetch_latest_candidate_rows(
    conn: sqlite3.Connection,
    *,
    max_btc_age_sec: float,
    min_time_until_resolution_sec: float,
    max_time_until_resolution_sec: float,
    now: datetime | None = None,
    max_feature_age_sec: float = 5.0,
    limit: int = 100,
    min_feature_timestamp: str | datetime | None = None,
) -> list[dict[str, Any]]:
    now_dt = now or datetime.now(timezone.utc)
    feature_cutoff = to_iso(
        datetime.fromtimestamp(
            now_dt.timestamp() - max(0.0, float(max_feature_age_sec)),
            tz=timezone.utc,
        )
    )
    min_feature_dt = parse_timestamp(min_feature_timestamp)
    min_feature_iso = to_iso(min_feature_dt) if min_feature_dt is not None else None
    recent_market_limit = max(int(limit), 500)
    sql_prefix = ""
    recent_snapshot_join = ""
    latest_feature_filter = """
        WHERE f.timestamp = (
            SELECT f2.timestamp
            FROM features f2
            WHERE f2.run_id = f.run_id
              AND f2.market_id = f.market_id
            ORDER BY datetime(f2.timestamp) DESC, f2.timestamp DESC
            LIMIT 1
          )
    """
    prefix_params: list[Any] = []
    latest_filter_params: list[Any] = []
    if min_feature_iso is not None:
        # Live-start mode should not evaluate the latest feature for every
        # historical market. Seed by recent snapshot market keys, then use the
        # run/market/timestamp feature index for the bounded live window.
        sql_prefix = """
        WITH recent_snapshot_markets AS (
            SELECT s.run_id, s.market_id, MAX(s.timestamp) AS latest_snapshot_timestamp
            FROM market_snapshots s
            WHERE datetime(s.timestamp) >= datetime(?)
            GROUP BY s.run_id, s.market_id
            ORDER BY datetime(latest_snapshot_timestamp) DESC, latest_snapshot_timestamp DESC
            LIMIT ?
        )
        """
        recent_snapshot_join = """
        JOIN recent_snapshot_markets rsm
          ON rsm.run_id = f.run_id
         AND rsm.market_id = f.market_id
        """
        latest_feature_filter = """
        WHERE datetime(f.timestamp) >= datetime(?)
          AND f.timestamp = (
            SELECT f2.timestamp
            FROM features f2
            WHERE f2.run_id = f.run_id
              AND f2.market_id = f.market_id
              AND datetime(f2.timestamp) >= datetime(?)
            ORDER BY datetime(f2.timestamp) DESC, f2.timestamp DESC
            LIMIT 1
          )
        """
        prefix_params = [min_feature_iso, recent_market_limit]
        latest_filter_params = [min_feature_iso, min_feature_iso]
    rows = conn.execute(
        f"""
        {sql_prefix}
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
        {recent_snapshot_join}
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
        {latest_feature_filter}
        ORDER BY
          CASE
            WHEN datetime(f.timestamp) >= datetime(?)
             AND m.close_time IS NOT NULL
             AND datetime(m.close_time) > datetime(?)
            THEN 0 ELSE 1
          END ASC,
          datetime(f.timestamp) DESC,
          f.timestamp DESC
        LIMIT ?
        """,
        (
            *prefix_params,
            to_iso(now_dt),
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            *latest_filter_params,
            feature_cutoff,
            to_iso(now_dt),
            int(limit),
        ),
    ).fetchall()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("time_until_resolution") is None and item.get("seconds_before_close") is not None:
            item["time_until_resolution"] = item.get("seconds_before_close")
        item["seconds_after_start"] = _seconds_after_start(item)
        normalized.append(item)
    return normalized


def settle_open_paper_trades(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    model_path: str | None = None,
    emit_logs: bool = True,
    now: datetime | None = None,
) -> int:
    now_dt = now or datetime.now(timezone.utc)
    open_rows = output.execute(
        """
        SELECT *
        FROM paper_trades
        WHERE status IN ('open', 'awaiting_resolution')
        ORDER BY id ASC
        """
    ).fetchall()
    settled = 0
    for trade in open_rows:
        close_time = parse_timestamp(trade["market_close_time"])
        if close_time is None or close_time > now_dt:
            continue
        resolution = _market_resolution_for_trade(
            recorder,
            run_id=str(trade["run_id"]),
            market_id=str(trade["market_id"]),
        )
        if not resolution["resolved"]:
            if str(trade["status"]) == "open":
                _mark_trade_awaiting_resolution(output, trade, now=now_dt)
                _emit(
                    emit_logs,
                    "paper_trade_awaiting_resolution",
                    id=int(trade["id"]),
                    run_id=trade["run_id"],
                    market_id=trade["market_id"],
                    signal_direction=trade["signal_direction"],
                    market_close_time=trade["market_close_time"],
                )
            continue
        label = resolution["label"]
        if label is None:
            if str(trade["status"]) == "open":
                _mark_trade_awaiting_resolution(output, trade, now=now_dt)
            _emit(
                emit_logs,
                "paper_trade_resolution_unavailable",
                id=int(trade["id"]),
                run_id=trade["run_id"],
                market_id=trade["market_id"],
                winning_outcome=resolution.get("winning_outcome"),
                winning_asset_id=resolution.get("winning_asset_id"),
                yes_token_id=resolution.get("yes_token_id"),
                no_token_id=resolution.get("no_token_id"),
            )
            continue

        direction = str(trade["signal_direction"])
        payout_per_share = 1.0 if (direction == "YES" and label == 1) or (direction == "NO" and label == 0) else 0.0
        stake = float(trade["stake_usd"] or 0.0)
        adjusted_entry = float(trade["adjusted_entry_price"] or 0.0)
        shares = stake / adjusted_entry if adjusted_entry > 0 else 0.0
        payout = shares * payout_per_share
        pnl = payout - stake
        roi = pnl / stake if stake else 0.0
        settled_at = to_iso(now_dt)
        with output:
            output.execute(
                """
                UPDATE paper_trades
                SET status = 'settled',
                    resolved_label = ?,
                    payout_usd = ?,
                    pnl_usd = ?,
                    roi = ?,
                    settled_at = ?
                WHERE id = ?
                """,
                (
                    "YES" if label == 1 else "NO",
                    round(payout, 10),
                    round(pnl, 10),
                    round(roi, 10),
                    settled_at,
                    int(trade["id"]),
                ),
            )
        settled += 1
        _emit(
            emit_logs,
            "paper_trade_settled",
            trade_id=int(trade["id"]),
            run_id=trade["run_id"],
            market_id=trade["market_id"],
            model_path=model_path,
            signal_direction=direction,
            resolved_label="YES" if label == 1 else "NO",
            payout_usd=round(payout, 10),
            pnl_usd=round(pnl, 10),
            roi=round(roi, 10),
        )
    _settle_resolved_trade_candidates(recorder, output, now=now_dt)
    return settled


def _settle_resolved_trade_candidates(
    recorder: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    now: datetime,
    limit: int = 1000,
) -> int:
    if not _paper_table_exists(output, "paper_trade_candidates"):
        return 0
    rows = output.execute(
        """
        SELECT *
        FROM paper_trade_candidates
        WHERE actual_resolved_label IS NULL
          AND market_close_time IS NOT NULL
          AND datetime(market_close_time) <= datetime(?)
        ORDER BY id ASC
        LIMIT ?
        """,
        (to_iso(now), int(limit)),
    ).fetchall()
    updated = 0
    for candidate in rows:
        resolution = _market_resolution_for_trade(
            recorder,
            run_id=str(candidate["run_id"] or ""),
            market_id=str(candidate["market_id"] or ""),
        )
        if not resolution["resolved"] or resolution["label"] is None:
            continue
        label = int(resolution["label"])
        payout, pnl, roi = _candidate_would_have_result(candidate, label=label)
        with output:
            output.execute(
                """
                UPDATE paper_trade_candidates
                SET actual_resolved_label = ?,
                    would_have_payout_usd = ?,
                    would_have_pnl_usd = ?,
                    would_have_roi = ?,
                    settled_at = ?
                WHERE id = ?
                """,
                (
                    "YES" if label == 1 else "NO",
                    payout,
                    pnl,
                    roi,
                    to_iso(now),
                    int(candidate["id"]),
                ),
            )
        updated += 1
    return updated


def _candidate_would_have_result(
    candidate: sqlite3.Row | dict[str, Any],
    *,
    label: int,
) -> tuple[float | None, float | None, float | None]:
    direction = str(candidate["candidate_direction"] or "").upper()
    if direction not in {"YES", "NO"}:
        return None, None, None
    adjusted_entry = _float_or_none(candidate["candidate_adjusted_entry_price"])
    stake = _float_or_none(candidate["stake_usd"])
    if adjusted_entry is None or adjusted_entry <= 0 or stake is None:
        return None, None, None
    payout_per_share = (
        1.0 if (direction == "YES" and label == 1) or (direction == "NO" and label == 0)
        else 0.0
    )
    shares = stake / adjusted_entry
    payout = shares * payout_per_share
    pnl = payout - stake
    roi = pnl / stake if stake else 0.0
    return round(payout, 10), round(pnl, 10), round(roi, 10)


def build_paper_trader_summary(
    paper_db_path: str,
    *,
    recent_limit: int = 10,
) -> dict[str, Any]:
    conn = connect_paper_output_db(paper_db_path)
    try:
        ensure_paper_schema(conn)
        rows = conn.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
        recent = conn.execute(
            """
            SELECT *
            FROM paper_trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(recent_limit),),
        ).fetchall()
        heartbeat = conn.execute(
            """
            SELECT *
            FROM paper_trader_heartbeats
            ORDER BY datetime(timestamp) DESC, timestamp DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    trade_dicts = [dict(row) for row in rows]
    settled = [row for row in trade_dicts if row.get("status") == "settled"]
    open_trades = [row for row in trade_dicts if row.get("status") == "open"]
    awaiting_resolution = [
        row for row in trade_dicts if row.get("status") == "awaiting_resolution"
    ]
    now = datetime.now(timezone.utc)
    wins = [row for row in settled if float(row.get("pnl_usd") or 0.0) > 0]
    pnl_by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("YES", "NO"):
        direction_rows = [row for row in settled if row.get("signal_direction") == direction]
        pnl_by_direction[direction] = {
            "trades": len(direction_rows),
            "total_pnl": round(sum(float(row.get("pnl_usd") or 0.0) for row in direction_rows), 10),
        }
    return {
        "paper_db_path": paper_db_path,
        "open_trades": len(open_trades),
        "awaiting_resolution_trades": len(awaiting_resolution),
        "settled_trades": len(settled),
        "win_rate": _rate(len(wins), len(settled)),
        "total_staked": round(
            sum(float(row.get("stake_usd") or 0.0) for row in trade_dicts if row.get("status") in {"open", "awaiting_resolution", "settled"}),
            10,
        ),
        "total_pnl": round(sum(float(row.get("pnl_usd") or 0.0) for row in settled), 10),
        "average_roi": _mean([float(row.get("roi") or 0.0) for row in settled]),
        "pnl_by_direction": pnl_by_direction,
        "status_counts": dict(Counter(str(row.get("status")) for row in trade_dicts)),
        "awaiting_resolution_age_buckets": _awaiting_resolution_age_buckets(
            awaiting_resolution,
            now=now,
        ),
        "oldest_awaiting_resolution_trades": _oldest_awaiting_resolution_trades(
            awaiting_resolution,
            now=now,
        ),
        "latest_heartbeat": dict(heartbeat) if heartbeat is not None else None,
        "recent_trades": [dict(row) for row in recent],
    }


def _awaiting_resolution_age_buckets(
    rows: Sequence[dict[str, Any]],
    *,
    now: datetime,
) -> dict[str, int]:
    ages = [_awaiting_resolution_age_sec(row, now=now) for row in rows]
    return {
        "older_than_10_minutes": sum(1 for age in ages if age is not None and age > 600),
        "older_than_30_minutes": sum(1 for age in ages if age is not None and age > 1800),
        "older_than_60_minutes": sum(1 for age in ages if age is not None and age > 3600),
    }


def _oldest_awaiting_resolution_trades(
    rows: Sequence[dict[str, Any]],
    *,
    now: datetime,
    limit: int = 10,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        age_sec = _awaiting_resolution_age_sec(row, now=now)
        enriched.append(
            {
                "id": row.get("id"),
                "run_id": row.get("run_id"),
                "market_id": row.get("market_id"),
                "close_time": row.get("market_close_time"),
                "age_sec": _round_or_none(age_sec),
                "signal_direction": row.get("signal_direction"),
                "resolved_label": row.get("resolved_label"),
                "payout_usd": row.get("payout_usd"),
                "pnl_usd": row.get("pnl_usd"),
                "recorder_resolved": None,
                "recorder_winning_outcome": None,
                "recorder_winning_asset_id": None,
            }
        )
    enriched.sort(key=lambda item: float(item.get("age_sec") or -1), reverse=True)
    return enriched[: int(limit)]


def _awaiting_resolution_age_sec(
    row: dict[str, Any],
    *,
    now: datetime,
) -> float | None:
    close_time = parse_timestamp(row.get("market_close_time"))
    if close_time is None:
        return None
    return max(0.0, (now - close_time).total_seconds())


def _filter_candidate_eligibility(
    rows: Sequence[dict[str, Any]],
    *,
    config: BaselinePaperTraderConfig,
    now: datetime,
    emit_logs: bool,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any] | None]:
    ready: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    latest_rejection: dict[str, Any] | None = None
    for row in rows:
        if _comparison_btc_stale(row, max_btc_age_sec=config.max_btc_age_sec):
            counts["comparison_btc_stale_warning"] += 1
        reason = _candidate_skip_reason(
            row,
            config=config,
            now=now,
        )
        if reason is not None:
            counts[reason] += 1
            rejection = _rejection_payload(reason, row)
            latest_rejection = _latest_rejection(latest_rejection, rejection)
            _emit(
                emit_logs,
                "paper_signal_skipped",
                run_id=row.get("run_id"),
                market_id=row.get("market_id"),
                signal_timestamp=row.get("timestamp"),
                skip_reason=reason,
            )
            continue
        ready.append(row)
    return ready, counts, latest_rejection


def _candidate_skip_reason(
    row: dict[str, Any],
    *,
    config: BaselinePaperTraderConfig,
    now: datetime,
) -> str | None:
    if _int_or_none(row.get("feature_ready")) != 1:
        return "not_feature_ready"
    if _int_or_none(row.get("is_gap_affected")) == 1:
        return "gap_affected"
    if _int_or_none(row.get("strict_validation_passed")) != 1:
        return "strict_validation_failed"
    if str(row.get("snapshot_quality_status") or "") != "ok":
        return "snapshot_quality_not_ok"
    if _int_or_none(row.get("has_orderbook")) != 1:
        return "missing_orderbook"
    if _int_or_none(row.get("has_trade_data")) != 1:
        return "missing_trade_data"

    close_time = parse_timestamp(row.get("close_time"))
    if close_time is None or now >= close_time:
        return "market_closed"

    time_until_resolution = _float_or_none(row.get("time_until_resolution"))
    if time_until_resolution is None or time_until_resolution <= 0:
        return "expired_feature"
    if time_until_resolution < float(config.min_time_until_resolution_sec):
        return "time_window"
    if time_until_resolution > float(config.max_time_until_resolution_sec):
        return "time_window"

    timestamp = parse_timestamp(row.get("timestamp"))
    if timestamp is None:
        return "stale_feature"
    age_sec = (now - timestamp).total_seconds()
    if age_sec > float(config.max_feature_age_sec):
        return "stale_feature"
    if _float_or_none(row.get("best_ask_yes")) is None and _float_or_none(row.get("best_ask_no")) is None:
        return "missing_price"
    if row.get("btc_chainlink_local_arrival_iso") is None:
        return "stale_canonical_btc"
    if row.get("btc_binance_local_arrival_iso") is None and config.require_fresh_comparison_btc:
        return "stale_comparison_btc"
    chain_age = _float_or_none(row.get("btc_chainlink_age_sec_at_feature"))
    binance_age = _float_or_none(row.get("btc_binance_age_sec_at_feature"))
    if chain_age is None or chain_age > float(config.max_btc_age_sec):
        return "stale_canonical_btc"
    if config.require_fresh_comparison_btc and (
        binance_age is None or binance_age > float(config.max_btc_age_sec)
    ):
        return "stale_comparison_btc"
    return None


def _comparison_btc_stale(row: dict[str, Any], *, max_btc_age_sec: float) -> bool:
    if row.get("btc_binance_local_arrival_iso") is None:
        return True
    age = _float_or_none(row.get("btc_binance_age_sec_at_feature"))
    return age is None or age > float(max_btc_age_sec)


def _skip_count_report(counts: Counter[str]) -> dict[str, int]:
    return {
        _heartbeat_counter_column(reason): int(counts.get(reason, 0))
        for reason in HEARTBEAT_SKIP_REASONS
    }


def _heartbeat_counter_column(reason: str) -> str:
    if reason == "stale_canonical_btc":
        return "canonical_btc_stale_skips"
    if reason == "stale_comparison_btc":
        return "comparison_btc_stale_skips"
    if reason == "comparison_btc_stale_warning":
        return "comparison_btc_stale_warnings"
    return f"{reason}_skips"


def _filter_prediction_prerequisites(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    *,
    emit_logs: bool,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any] | None]:
    ready: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    latest_rejection: dict[str, Any] | None = None
    needs_start_btc = "btc_price_at_market_start" in set(feature_columns)
    needs_seconds_after_start = "seconds_after_start" in set(feature_columns)
    for row in rows:
        reason: str | None = None
        if needs_start_btc and _float_or_none(row.get("btc_price_at_market_start")) is None:
            reason = "missing_btc_price_at_market_start"
        elif needs_seconds_after_start and _float_or_none(row.get("seconds_after_start")) is None:
            reason = "invalid_seconds_after_start"
        if reason is not None:
            counts[reason] += 1
            latest_rejection = _latest_rejection(
                latest_rejection,
                _rejection_payload(reason, row),
            )
            _emit(
                emit_logs,
                "paper_signal_skipped",
                run_id=row.get("run_id"),
                market_id=row.get("market_id"),
                signal_timestamp=row.get("timestamp"),
                skip_reason=reason,
            )
            continue
        ready.append(row)
    return ready, counts, latest_rejection


def _process_signal_row(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    config: BaselinePaperTraderConfig,
    model_name: str,
    emit_logs: bool,
) -> str:
    probability_value = _float_or_none(probability)
    if probability_value is None:
        _record_trade_candidate(
            output,
            row,
            probability=None,
            direction=None,
            threshold=None,
            decision="SKIP",
            rejection_reason="no_signal",
            config=config,
            model_name=model_name,
        )
        return "no_signal"

    direction = _direction_for_probability(
        probability_value,
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
    )
    if direction is None:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=None,
            threshold=None,
            decision="SKIP",
            rejection_reason="threshold",
            config=config,
            model_name=model_name,
        )
        return "threshold"

    entry = _live_entry_price(row, direction=direction)
    threshold = config.long_threshold if direction == "YES" else config.short_threshold
    if direction == "YES" and not config.allow_yes:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="yes_disabled",
            config=config,
            entry_price=entry,
            model_name=model_name,
        )
        return "yes_disabled"
    if direction == "NO" and not config.allow_no:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="no_disabled",
            config=config,
            entry_price=entry,
            model_name=model_name,
        )
        return "no_disabled"
    if entry is None:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="missing_entry_price",
            config=config,
            model_name=model_name,
        )
        _record_skipped_trade(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            skip_reason="missing_entry_price",
            config=config,
            model_name=model_name,
            emit_logs=emit_logs,
        )
        return "missing_price"

    adjusted = min(1.0, float(entry) + float(config.entry_slippage_cents) / 100.0 + float(config.fee_cents) / 100.0)
    probability_for_direction = _probability_for_direction(
        probability_value,
        direction=direction,
    )
    estimated_edge = probability_for_direction - adjusted
    if (
        config.block_probability_above is not None
        and probability_for_direction >= float(config.block_probability_above)
    ):
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="probability_block",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        return "probability_block"
    if (
        config.block_probability_below is not None
        and probability_for_direction <= float(config.block_probability_below)
    ):
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="probability_block",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        return "probability_block"
    if adjusted <= 0 or adjusted >= 1.0:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="invalid_entry_price",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        _record_skipped_trade(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            skip_reason="invalid_entry_price",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            model_name=model_name,
            emit_logs=emit_logs,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
        )
        return "invalid_entry_price"
    if estimated_edge < float(config.min_estimated_edge):
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="min_estimated_edge",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        _record_skipped_trade(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            skip_reason="min_estimated_edge",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            model_name=model_name,
            emit_logs=emit_logs,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
        )
        return "min_estimated_edge"

    strategy_id = _strategy_id(config)
    strategy_name = _strategy_name(config)
    if _paper_status_count(output, "open", strategy_id=strategy_id) >= int(config.max_open_trades):
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="max_open_trades",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        return "max_open_trades"

    if config.one_trade_per_market and _has_existing_trade(
        output,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        strategy_id=strategy_id,
    ):
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="one_trade_per_market",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        return "one_trade_per_market"

    now_iso = to_iso(datetime.now(timezone.utc))
    inserted = _insert_open_trade_if_allowed(
        output,
        row,
        created_at=now_iso,
        model_path=config.model_path,
        model_name=model_name,
        probability=probability_value,
        direction=direction,
        threshold=threshold,
        stake_usd=config.stake_usd,
        entry_price=entry,
        adjusted_entry_price=adjusted,
        probability_for_direction=probability_for_direction,
        estimated_edge=estimated_edge,
        one_trade_per_market=config.one_trade_per_market,
        strategy_id=strategy_id,
        strategy_name=strategy_name,
    )
    if not inserted:
        _record_trade_candidate(
            output,
            row,
            probability=probability_value,
            direction=direction,
            threshold=threshold,
            decision="SKIP",
            rejection_reason="one_trade_per_market",
            config=config,
            entry_price=entry,
            adjusted_entry_price=adjusted,
            probability_for_direction=probability_for_direction,
            estimated_edge=estimated_edge,
            model_name=model_name,
        )
        return "one_trade_per_market"
    _record_trade_candidate(
        output,
        row,
        probability=probability_value,
        direction=direction,
        threshold=threshold,
        decision="TRADE",
        rejection_reason=None,
        config=config,
        entry_price=entry,
        adjusted_entry_price=adjusted,
        probability_for_direction=probability_for_direction,
        estimated_edge=estimated_edge,
        model_name=model_name,
        linked_paper_trade_id=inserted,
    )
    _emit(
        emit_logs,
        "paper_trade_opened",
        run_id=row.get("run_id"),
        market_id=row.get("market_id"),
        signal_timestamp=row.get("timestamp"),
        signal_direction=direction,
        predicted_probability_yes=round(float(probability_value), 10),
        probability_for_direction=round(float(probability_for_direction), 10),
        estimated_edge=round(float(estimated_edge), 10),
        entry_price=round(float(entry), 10),
        adjusted_entry_price=round(float(adjusted), 10),
    )
    return "opened"


def _insert_open_trade_if_allowed(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    created_at: str,
    model_path: str,
    model_name: str,
    probability: float,
    direction: str,
    threshold: float,
    stake_usd: float,
    entry_price: float,
    adjusted_entry_price: float,
    probability_for_direction: float,
    estimated_edge: float,
    one_trade_per_market: bool,
    strategy_id: str,
    strategy_name: str,
) -> int | None:
    values = _trade_insert_values(
        row,
        created_at=created_at,
        model_path=model_path,
        model_name=model_name,
        probability=probability,
        direction=direction,
        threshold=threshold,
        stake_usd=stake_usd,
        entry_price=entry_price,
        adjusted_entry_price=adjusted_entry_price,
        probability_for_direction=probability_for_direction,
        estimated_edge=estimated_edge,
        strategy_id=strategy_id,
        strategy_name=strategy_name,
    )
    if not one_trade_per_market:
        with output:
            cursor = output.execute(
                """
                INSERT INTO paper_trades (
                    created_at, run_id, market_id, signal_timestamp, question,
                    market_start_time, market_close_time, time_until_resolution,
                    model_path, model_name, strategy_id, strategy_name,
                    predicted_probability_yes,
                    probability_for_direction, estimated_edge,
                    signal_direction, threshold_used, stake_usd,
                    entry_price, adjusted_entry_price,
                    best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                    btc_chainlink_price, btc_binance_price,
                    btc_chainlink_age_sec_at_feature, btc_binance_age_sec_at_feature,
                    feature_ready, strict_validation_passed, snapshot_quality_status,
                    status, skip_reason
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, 'open', NULL
                )
                """,
                values,
            )
            return int(cursor.lastrowid)

    with output:
        cursor = output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes,
                probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd,
                entry_price, adjusted_entry_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                btc_chainlink_price, btc_binance_price,
                btc_chainlink_age_sec_at_feature, btc_binance_age_sec_at_feature,
                feature_ready, strict_validation_passed, snapshot_quality_status,
                status, skip_reason
            )
            SELECT
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, 'open', NULL
            WHERE NOT EXISTS (
                SELECT 1
                FROM paper_trades
                WHERE run_id = ?
                  AND market_id = ?
                  AND COALESCE(strategy_id, 'baseline_default') = ?
                  AND status IN ('open', 'awaiting_resolution', 'settled', 'closed')
                LIMIT 1
            )
            """,
            (
                *values,
                str(row.get("run_id") or ""),
                str(row.get("market_id") or ""),
                strategy_id,
            ),
        )
        if int(cursor.rowcount or 0) <= 0:
            return None
        return int(cursor.lastrowid)


def _record_trade_candidate(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float | None,
    direction: str | None,
    threshold: float | None,
    decision: str,
    rejection_reason: str | None,
    config: BaselinePaperTraderConfig,
    model_name: str,
    entry_price: float | None = None,
    adjusted_entry_price: float | None = None,
    probability_for_direction: float | None = None,
    estimated_edge: float | None = None,
    linked_paper_trade_id: int | None = None,
) -> int | None:
    run_id = str(row.get("run_id") or "")
    market_id = str(row.get("market_id") or "")
    feature_timestamp = str(row.get("timestamp") or "")
    strategy_id = _strategy_id(config)
    if not market_id or not feature_timestamp:
        return None
    existing = output.execute(
        """
        SELECT id
        FROM paper_trade_candidates
        WHERE COALESCE(run_id, '') = ?
          AND market_id = ?
          AND feature_timestamp = ?
          AND COALESCE(candidate_direction, '') = ?
          AND COALESCE(strategy_id, '') = ?
        LIMIT 1
        """,
        (
            run_id,
            market_id,
            feature_timestamp,
            str(direction or ""),
            strategy_id,
        ),
    ).fetchone()
    if existing is not None:
        if linked_paper_trade_id is not None:
            with output:
                output.execute(
                    """
                    UPDATE paper_trade_candidates
                    SET linked_paper_trade_id = COALESCE(linked_paper_trade_id, ?),
                        decision = CASE
                          WHEN linked_paper_trade_id IS NULL THEN ?
                          ELSE decision
                        END,
                        rejection_reason = CASE
                          WHEN linked_paper_trade_id IS NULL THEN ?
                          ELSE rejection_reason
                        END
                    WHERE id = ?
                    """,
                    (int(linked_paper_trade_id), decision, rejection_reason, int(existing["id"])),
                )
        return int(existing["id"])

    if probability_for_direction is None and probability is not None and direction is not None:
        probability_for_direction = _probability_for_direction(probability, direction=direction)
    if estimated_edge is None and probability_for_direction is not None and adjusted_entry_price is not None:
        estimated_edge = probability_for_direction - float(adjusted_entry_price)
    created_at = to_iso(datetime.now(timezone.utc))
    values = (
        created_at,
        row.get("run_id"),
        row.get("market_id"),
        row.get("question"),
        row.get("timestamp"),
        row.get("start_time"),
        row.get("close_time"),
        strategy_id,
        _strategy_name(config),
        direction,
        _round_or_none(entry_price),
        _round_or_none(adjusted_entry_price),
        float(config.stake_usd),
        model_name,
        config.model_path,
        _round_or_none(probability),
        _round_or_none(probability_for_direction),
        _round_or_none(estimated_edge),
        _round_or_none(threshold),
        float(config.min_estimated_edge),
        _float_or_none(row.get("time_until_resolution")),
        float(config.min_time_until_resolution_sec),
        float(config.max_time_until_resolution_sec),
        _round_or_none(config.block_probability_above),
        _round_or_none(config.block_probability_below),
        _round_or_none(config.min_probability_for_direction),
        _round_or_none(config.max_probability_for_direction),
        int(config.max_open_trades),
        _float_or_none(row.get("best_bid_yes")),
        _float_or_none(row.get("best_ask_yes")),
        _float_or_none(row.get("best_bid_no")),
        _float_or_none(row.get("best_ask_no")),
        _float_or_none(row.get("btc_chainlink_price")),
        _float_or_none(row.get("btc_binance_price")),
        _float_or_none(row.get("btc_chainlink_age_sec_at_feature")),
        _float_or_none(row.get("btc_binance_age_sec_at_feature")),
        _int_or_none(row.get("feature_ready")),
        _int_or_none(row.get("strict_validation_passed")),
        row.get("snapshot_quality_status"),
        str(decision),
        rejection_reason,
        linked_paper_trade_id,
    )
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
                block_probability_above, block_probability_below,
                min_probability_for_direction, max_probability_for_direction,
                max_open_trades, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, btc_chainlink_price, btc_binance_price,
                btc_chainlink_age_sec_at_feature,
                btc_binance_age_sec_at_feature, feature_ready,
                strict_validation_passed, snapshot_quality_status, decision,
                rejection_reason, linked_paper_trade_id
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?
            )
            """,
            values,
        )
    if int(cursor.rowcount or 0) <= 0:
        return None
    return int(cursor.lastrowid)


def _strategy_id(config: BaselinePaperTraderConfig) -> str:
    return str(config.strategy_id or "baseline_default")


def _strategy_name(config: BaselinePaperTraderConfig) -> str:
    return str(config.strategy_name or "Baseline Default")


def _record_skipped_trade(
    output: sqlite3.Connection,
    row: dict[str, Any],
    *,
    probability: float,
    direction: str,
    threshold: float,
    skip_reason: str,
    config: BaselinePaperTraderConfig,
    model_name: str,
    emit_logs: bool,
    entry_price: float | None = None,
    adjusted_entry_price: float | None = None,
    probability_for_direction: float | None = None,
    estimated_edge: float | None = None,
) -> None:
    if _skip_already_recorded(
        output,
        run_id=str(row.get("run_id") or ""),
        market_id=str(row.get("market_id") or ""),
        signal_timestamp=str(row.get("timestamp") or ""),
        skip_reason=skip_reason,
        strategy_id=_strategy_id(config),
    ):
        return
    now_iso = to_iso(datetime.now(timezone.utc))
    with output:
        output.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes,
                probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd,
                entry_price, adjusted_entry_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                btc_chainlink_price, btc_binance_price,
                btc_chainlink_age_sec_at_feature, btc_binance_age_sec_at_feature,
                feature_ready, strict_validation_passed, snapshot_quality_status,
                status, skip_reason
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, 'skipped', ?
            )
            """,
            (
                *_trade_insert_values(
                    row,
                    created_at=now_iso,
                    model_path=config.model_path,
                    model_name=model_name,
                    probability=probability,
                    direction=direction,
                    threshold=threshold,
                    stake_usd=config.stake_usd,
                    entry_price=entry_price,
                    adjusted_entry_price=adjusted_entry_price,
                    probability_for_direction=probability_for_direction,
                    estimated_edge=estimated_edge,
                    strategy_id=_strategy_id(config),
                    strategy_name=_strategy_name(config),
                ),
                skip_reason,
            ),
        )
    _emit(
        emit_logs,
        "paper_signal_skipped",
        run_id=row.get("run_id"),
        market_id=row.get("market_id"),
        signal_timestamp=row.get("timestamp"),
        signal_direction=direction,
        skip_reason=skip_reason,
    )


def _trade_insert_values(
    row: dict[str, Any],
    *,
    created_at: str | None,
    model_path: str,
    model_name: str | None,
    probability: float,
    direction: str,
    threshold: float,
    stake_usd: float,
    entry_price: float | None,
    adjusted_entry_price: float | None,
    probability_for_direction: float | None = None,
    estimated_edge: float | None = None,
    strategy_id: str | None = None,
    strategy_name: str | None = None,
) -> tuple[Any, ...]:
    if probability_for_direction is None:
        probability_for_direction = _probability_for_direction(probability, direction=direction)
    if estimated_edge is None and adjusted_entry_price is not None:
        estimated_edge = probability_for_direction - float(adjusted_entry_price)
    return (
        created_at,
        row.get("run_id"),
        row.get("market_id"),
        row.get("timestamp"),
        row.get("question"),
        row.get("start_time"),
        row.get("close_time"),
        _float_or_none(row.get("time_until_resolution")),
        model_path,
        model_name,
        strategy_id,
        strategy_name,
        round(float(probability), 10),
        _round_or_none(probability_for_direction),
        _round_or_none(estimated_edge),
        direction,
        float(threshold),
        float(stake_usd),
        _round_or_none(entry_price),
        _round_or_none(adjusted_entry_price),
        _float_or_none(row.get("best_bid_yes")),
        _float_or_none(row.get("best_ask_yes")),
        _float_or_none(row.get("best_bid_no")),
        _float_or_none(row.get("best_ask_no")),
        _float_or_none(row.get("btc_chainlink_price")),
        _float_or_none(row.get("btc_binance_price")),
        _float_or_none(row.get("btc_chainlink_age_sec_at_feature")),
        _float_or_none(row.get("btc_binance_age_sec_at_feature")),
        _int_or_none(row.get("feature_ready")),
        _int_or_none(row.get("strict_validation_passed")),
        row.get("snapshot_quality_status"),
    )


def _live_entry_price(row: dict[str, Any], *, direction: str) -> float | None:
    if direction == "YES":
        return _valid_price(row.get("best_ask_yes"))
    return _valid_price(row.get("best_ask_no"))


def _probability_for_direction(probability_yes: float, *, direction: str) -> float:
    if direction == "NO":
        return 1.0 - float(probability_yes)
    return float(probability_yes)


def _valid_price(value: Any) -> float | None:
    price = _float_or_none(value)
    if price is None or price <= 0 or price >= 1:
        return None
    return price


def _model_name(model: Any) -> str:
    steps = getattr(model, "steps", None)
    if steps:
        try:
            return type(steps[-1][1]).__name__
        except (IndexError, TypeError):
            pass
    return type(model).__name__


def _seconds_after_start(row: dict[str, Any]) -> float | None:
    start = parse_timestamp(row.get("start_time"))
    timestamp = parse_timestamp(row.get("timestamp"))
    if start is None or timestamp is None:
        return None
    return max(0.0, (timestamp - start).total_seconds())


def _mark_trade_awaiting_resolution(
    output: sqlite3.Connection,
    trade: sqlite3.Row,
    *,
    now: datetime,
) -> None:
    with output:
        output.execute(
            """
            UPDATE paper_trades
            SET status = 'awaiting_resolution'
            WHERE id = ?
              AND status = 'open'
            """,
            (int(trade["id"]),),
        )


def _market_resolution_for_trade(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT resolved, winning_asset_id, winning_outcome, yes_token_id, no_token_id
        FROM markets
        WHERE run_id = ? AND market_id = ?
        LIMIT 1
        """,
        (run_id, market_id),
    ).fetchone()
    if row is None:
        return {
            "resolved": False,
            "label": None,
            "winning_asset_id": None,
            "winning_outcome": None,
            "yes_token_id": None,
            "no_token_id": None,
        }
    winning_asset_id = row["winning_asset_id"]
    winning_outcome = row["winning_outcome"]
    yes_token_id = row["yes_token_id"]
    no_token_id = row["no_token_id"]
    resolved = int(row["resolved"] or 0) == 1
    return {
        "resolved": resolved,
        "label": _normalise_winning_label(
            winning_outcome=winning_outcome,
            winning_asset_id=winning_asset_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        ) if resolved else None,
        "winning_asset_id": winning_asset_id,
        "winning_outcome": winning_outcome,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
    }


def _normalise_winning_label(
    *,
    winning_outcome: Any,
    winning_asset_id: Any,
    yes_token_id: Any,
    no_token_id: Any,
) -> int | None:
    normalized_outcome = str(winning_outcome or "").strip().upper()
    if normalized_outcome in {"YES", "Y", "UP", "LONG", "TRUE"}:
        return 1
    if normalized_outcome in {"NO", "N", "DOWN", "SHORT", "FALSE"}:
        return 0
    normalized_asset = str(winning_asset_id or "")
    if normalized_asset and normalized_asset == str(yes_token_id or ""):
        return 1
    if normalized_asset and normalized_asset == str(no_token_id or ""):
        return 0
    return None


def _has_existing_trade(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    strategy_id: str | None = None,
) -> bool:
    if strategy_id is not None:
        row = conn.execute(
            """
            SELECT 1
            FROM paper_trades
            WHERE run_id = ?
              AND market_id = ?
              AND COALESCE(strategy_id, 'baseline_default') = ?
              AND status IN ('open', 'awaiting_resolution', 'settled', 'closed')
            LIMIT 1
            """,
            (run_id, market_id, strategy_id),
        ).fetchone()
        return row is not None
    row = conn.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND status IN ('open', 'awaiting_resolution', 'settled', 'closed')
        LIMIT 1
        """,
        (run_id, market_id),
    ).fetchone()
    return row is not None


def _paper_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _paper_status_count(
    conn: sqlite3.Connection,
    status: str,
    *,
    strategy_id: str | None = None,
) -> int:
    if strategy_id is not None:
        return int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM paper_trades
                WHERE status = ?
                  AND COALESCE(strategy_id, 'baseline_default') = ?
                """,
                (status, strategy_id),
            ).fetchone()[0]
            or 0
        )
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status = ?",
            (status,),
        ).fetchone()[0]
        or 0
    )


def _merge_counts(target: Counter[str], source: Counter[str]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _rejection_payload(reason: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": reason,
        "market_id": row.get("market_id"),
        "feature_timestamp": row.get("timestamp"),
    }


def _latest_rejection(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    current_ts = parse_timestamp(current.get("feature_timestamp"))
    candidate_ts = parse_timestamp(candidate.get("feature_timestamp"))
    if current_ts is None:
        return candidate
    if candidate_ts is None:
        return current
    return candidate if candidate_ts >= current_ts else current


def _skip_already_recorded(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    signal_timestamp: str,
    skip_reason: str,
    strategy_id: str | None = None,
) -> bool:
    if strategy_id is not None:
        row = conn.execute(
            """
            SELECT 1
            FROM paper_trades
            WHERE run_id = ?
              AND market_id = ?
              AND signal_timestamp = ?
              AND status = 'skipped'
              AND skip_reason = ?
              AND COALESCE(strategy_id, 'baseline_default') = ?
            LIMIT 1
            """,
            (run_id, market_id, signal_timestamp, skip_reason, strategy_id),
        ).fetchone()
        return row is not None
    row = conn.execute(
        """
        SELECT 1
        FROM paper_trades
        WHERE run_id = ?
          AND market_id = ?
          AND signal_timestamp = ?
          AND status = 'skipped'
          AND skip_reason = ?
        LIMIT 1
        """,
        (run_id, market_id, signal_timestamp, skip_reason),
    ).fetchone()
    return row is not None


def _write_heartbeat(
    conn: sqlite3.Connection,
    *,
    recorder_db_path: str,
    output_db_path: str,
    latest_feature_timestamp: str | None,
    latest_seen_feature_timestamp: str | None,
    latest_eligible_feature_timestamp: str | None,
    latest_btc_chainlink_age_sec: float | None,
    latest_btc_binance_age_sec: float | None,
    latest_canonical_btc_age_sec: float | None,
    latest_comparison_btc_age_sec: float | None,
    skip_counts: Counter[str],
    latest_rejection: dict[str, Any] | None,
    loop_error: str | None,
) -> None:
    timestamp = to_iso(datetime.now(timezone.utc))
    open_trades = int(
        conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status = 'open'").fetchone()[0]
        or 0
    )
    awaiting_resolution_trades = int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status = 'awaiting_resolution'"
        ).fetchone()[0]
        or 0
    )
    settled_trades = int(
        conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status = 'settled'").fetchone()[0]
        or 0
    )
    total_trades = int(conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] or 0)
    skip_columns = [_heartbeat_counter_column(reason) for reason in HEARTBEAT_SKIP_REASONS]
    columns = [
        "timestamp",
        "recorder_db_path",
        "output_db_path",
        "latest_feature_timestamp",
        "latest_seen_feature_timestamp",
        "latest_eligible_feature_timestamp",
        "latest_btc_chainlink_age_sec",
        "latest_btc_binance_age_sec",
        "latest_canonical_btc_age_sec",
        "latest_comparison_btc_age_sec",
        "open_trades",
        "awaiting_resolution_trades",
        "settled_trades",
        "total_trades",
        *skip_columns,
        "latest_rejection_reason",
        "latest_rejection_market_id",
        "latest_rejection_feature_timestamp",
        "loop_error",
    ]
    values = [
        timestamp,
        recorder_db_path,
        output_db_path,
        latest_feature_timestamp,
        latest_seen_feature_timestamp,
        latest_eligible_feature_timestamp,
        latest_btc_chainlink_age_sec,
        latest_btc_binance_age_sec,
        latest_canonical_btc_age_sec,
        latest_comparison_btc_age_sec,
        open_trades,
        awaiting_resolution_trades,
        settled_trades,
        total_trades,
        *(int(skip_counts.get(reason, 0)) for reason in HEARTBEAT_SKIP_REASONS),
        (latest_rejection or {}).get("reason"),
        (latest_rejection or {}).get("market_id"),
        (latest_rejection or {}).get("feature_timestamp"),
        loop_error,
    ]
    placeholders = ", ".join("?" for _ in columns)
    with conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO paper_trader_heartbeats (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            values,
        )


def _latest_feature_timestamp(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT timestamp
        FROM features
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row["timestamp"]) if row is not None and row["timestamp"] is not None else None


def _latest_btc_age(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return _float_or_none(rows[0].get(key))


def _missing_feature_columns(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [column for column in feature_columns if column not in keys]


def _emit_missing_feature_error(enabled: bool, missing_columns: Sequence[str]) -> None:
    if not enabled:
        return
    key = tuple(sorted(str(column) for column in missing_columns))
    if key in _LOGGED_MISSING_FEATURE_SETS:
        return
    _LOGGED_MISSING_FEATURE_SETS.add(key)
    _emit(
        enabled,
        "paper_trader_error",
        error="missing_feature_columns: " + ", ".join(key),
    )


def _round_or_none(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return round(numeric, 10) if numeric is not None else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 10)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 10)


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
    "BaselinePaperTraderConfig",
    "BaselinePaperTraderError",
    "build_paper_trader_summary",
    "connect_recorder_read_only",
    "ensure_paper_schema",
    "fetch_latest_candidate_rows",
    "run_baseline_paper_trader",
    "run_baseline_paper_trader_once",
    "settle_open_paper_trades",
]
