from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.backtest_baseline_strategy import _load_feature_columns, _load_model
from src.export_meta_strategy_dataset import export_meta_strategy_dataset
from src.multi_strategy_paper_trader import (
    MultiStrategyRealisticExecutionSettings,
    MultiStrategyTradeSelectionSettings,
    _regime_tags_for_row,
    load_multi_strategy_config,
    run_multi_strategy_paper_trader,
    run_multi_strategy_paper_trader_once,
    update_multi_strategy_trade_exits,
)
from src.paper_trader_analytics import build_paper_trader_analytics_report
from src.baseline_paper_trader import (
    connect_paper_output_db,
    connect_recorder_read_only,
    ensure_paper_schema,
    settle_open_paper_trades,
)
from tests.test_baseline_paper_trader import (
    _fixture,
    _insert_candidate,
    _update_market_resolution,
)


BALANCED_LATE_CONFIG = "data/strategy_configs/live_strategy_grid_yes_no_late_prob_band.json"


def _write_config(path, strategies: list[dict], **extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"strategies": strategies, **extra}), encoding="utf-8")


def _strategy(
    strategy_id: str,
    *,
    direction_mode: str = "BOTH",
    long_threshold: float = 0.65,
    short_threshold: float = 0.35,
    min_estimated_edge: float = 0.0,
    max_open_trades: int = 1,
    one_trade_per_market: bool = True,
    min_time_until_resolution_sec: float = 0.0,
    max_time_until_resolution_sec: float = 300.0,
    require_positive_edge: bool = True,
    min_probability_for_direction: float | None = None,
    max_probability_for_direction: float | None = None,
    exit_type: str | None = None,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    fixed_horizon_exit_sec: float | None = None,
    exit_slippage_cents: float | None = None,
    entry_slippage_cents: float = 0.0,
    liquidity_fill_check_enabled: bool = False,
    max_top_of_book_fill_fraction: float = 1.0,
    min_top_of_book_shares: float = 0.0,
    skip_if_liquidity_missing: bool = True,
    blocked_liquidity_regimes: list[str] | None = None,
) -> dict:
    item = {
        "strategy_id": strategy_id,
        "strategy_name": f"Strategy {strategy_id}",
        "enabled": True,
        "direction_mode": direction_mode,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "min_estimated_edge": min_estimated_edge,
        "entry_slippage_cents": entry_slippage_cents,
        "fee_cents": 0.0,
        "stake_usd": 1.0,
        "max_open_trades": max_open_trades,
        "one_trade_per_market": one_trade_per_market,
        "require_positive_edge": require_positive_edge,
        "min_time_until_resolution_sec": min_time_until_resolution_sec,
        "max_time_until_resolution_sec": max_time_until_resolution_sec,
    }
    if liquidity_fill_check_enabled:
        item["liquidity_fill_check_enabled"] = True
        item["max_top_of_book_fill_fraction"] = max_top_of_book_fill_fraction
        item["min_top_of_book_shares"] = min_top_of_book_shares
        item["skip_if_liquidity_missing"] = skip_if_liquidity_missing
    if blocked_liquidity_regimes is not None:
        item["blocked_liquidity_regimes"] = blocked_liquidity_regimes
    if min_probability_for_direction is not None:
        item["min_probability_for_direction"] = min_probability_for_direction
    if max_probability_for_direction is not None:
        item["max_probability_for_direction"] = max_probability_for_direction
    if exit_type is not None:
        item["exit_type"] = exit_type
    if take_profit_pct is not None:
        item["take_profit_pct"] = take_profit_pct
    if stop_loss_pct is not None:
        item["stop_loss_pct"] = stop_loss_pct
    if fixed_horizon_exit_sec is not None:
        item["fixed_horizon_exit_sec"] = fixed_horizon_exit_sec
    if exit_slippage_cents is not None:
        item["exit_slippage_cents"] = exit_slippage_cents
    return item


def _run_once(tmp_path, strategies: list[dict], *, insert_rows: bool = True):
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    if insert_rows:
        _insert_candidate(
            recorder_db,
            market_id="yes_signal",
            signal=0.8,
            best_ask_yes=0.5,
        )
        _insert_candidate(
            recorder_db,
            market_id="no_signal",
            signal=0.2,
            best_ask_no=0.4,
        )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(config_path, strategies)
    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    return recorder_db, output_db, model_path, features_path, config_path, report


def _run_once_with_config(
    tmp_path,
    strategies: list[dict],
    *,
    config_extra: dict,
    insert_rows: bool = True,
    live_order_submitter=None,
):
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    if insert_rows:
        _insert_candidate(
            recorder_db,
            market_id="yes_signal",
            signal=0.8,
            best_ask_yes=0.5,
        )
        _insert_candidate(
            recorder_db,
            market_id="no_signal",
            signal=0.2,
            best_ask_no=0.4,
        )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(config_path, strategies, **config_extra)
    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
        live_order_submitter=live_order_submitter,
    )
    return recorder_db, output_db, model_path, features_path, config_path, report


def _realistic_execution_config(**overrides: float | bool) -> dict:
    config = {
        "realistic_execution_enabled": True,
        "entry_latency_sec": 1.0,
        "exit_latency_sec": 1.0,
        "max_quote_wait_sec": 3.0,
        "max_entry_price_drift_cents": 2.0,
        "max_exit_price_drift_cents": 2.0,
        "require_quote_after_latency": True,
        "reject_if_spread_above_cents": 4.0,
        "reject_if_btc_age_above_sec": 5.0,
        "reject_if_feature_age_above_sec": 10.0,
        "apply_extra_slippage_cents": 1.0,
        "partial_fill_enabled": False,
    }
    config.update(overrides)
    return config


def _live_trading_config(tmp_path, **overrides: float | bool | str) -> dict:
    config = {
        "live_trading_enabled": True,
        "dry_run_orders": True,
        "max_order_usd": 1.0,
        "max_daily_loss_usd": 5.0,
        "max_daily_orders": 20,
        "max_open_exposure_usd": 5.0,
        "max_open_trades": 3,
        "max_trades_per_market": 1,
        "max_trades_per_market_direction": 1,
        "require_realistic_execution_passed": False,
        "require_liquidity_check_passed": False,
        "reject_if_btc_age_above_sec": 100000.0,
        "reject_if_feature_age_above_sec": 100000.0,
        "reject_if_spread_above_cents": 10.0,
        "kill_switch_file": str(tmp_path / "KILL_LIVE_TRADING"),
        "allow_market_order": False,
        "use_limit_orders_only": True,
    }
    config.update(overrides)
    return config


def _rows(db_path, table: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        conn.close()


def _insert_exit_snapshot(
    db_path,
    *,
    market_id: str,
    best_bid_yes: float = 0.49,
    best_ask_yes: float = 0.50,
    best_bid_no: float = 0.39,
    best_ask_no: float = 0.40,
    timestamp: datetime | None = None,
) -> None:
    ts = timestamp or datetime.now(timezone.utc)
    spread_yes = (
        best_ask_yes - best_bid_yes
        if best_ask_yes is not None and best_bid_yes is not None
        else None
    )
    spread_no = (
        best_ask_no - best_bid_no
        if best_ask_no is not None and best_bid_no is not None
        else None
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO market_snapshots (
                run_id, timestamp, market_id, best_bid_yes, best_ask_yes,
                best_bid_no, best_ask_no, spread_yes, spread_no,
                mid_price_yes, mid_price_no, last_trade_price, last_trade_size,
                last_trade_time, has_orderbook, has_trade_data, strict_validation_passed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 1)
            """,
            (
                "run_live",
                ts.isoformat(),
                market_id,
                best_bid_yes,
                best_ask_yes,
                best_bid_no,
                best_ask_no,
                spread_yes,
                spread_no,
                0.50,
                0.50,
                0.51,
                5.0,
                ts.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _set_trade_created_at(db_path, created_at: datetime) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE paper_trades SET created_at = ?", (created_at.isoformat(),))
        conn.commit()
    finally:
        conn.close()


def _add_feature_top_ask_sizes(
    db_path,
    *,
    market_id: str,
    yes_size: float | None = None,
    no_size: float | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("ALTER TABLE features ADD COLUMN best_ask_size_yes REAL")
        conn.execute("ALTER TABLE features ADD COLUMN best_ask_size_no REAL")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc):
            conn.close()
            raise
    try:
        conn.execute(
            """
            UPDATE features
            SET best_ask_size_yes = ?,
                best_ask_size_no = ?
            WHERE market_id = ?
            """,
            (yes_size, no_size, market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_manual_paper_trade(
    db_path,
    *,
    status: str,
    stake_usd: float = 1.0,
    pnl_usd: float | None = None,
    realized_pnl_usd: float | None = None,
    market_id: str = "manual",
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        ensure_paper_schema(conn)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, strategy_id, strategy_name,
                predicted_probability_yes, probability_for_direction, estimated_edge,
                signal_direction, threshold_used, stake_usd, entry_price,
                adjusted_entry_price, status, pnl_usd, realized_pnl_usd
            ) VALUES (?, 'run_live', ?, ?, 'manual', ?, ?, 60,
                'model', 'model', 'manual_s', 'manual_s',
                0.8, 0.8, 0.3, 'YES', 0.7, ?, 0.5, 0.5,
                ?, ?, ?)
            """,
            (
                now,
                market_id,
                now,
                now,
                now,
                stake_usd,
                status,
                pnl_usd,
                realized_pnl_usd,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _run_exit_update(recorder_db, output_db, config_path, *, now: datetime | None = None) -> dict[str, int]:
    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(output_db))
    try:
        return update_multi_strategy_trade_exits(
            recorder,
            output,
            strategies=load_multi_strategy_config(config_path),
            now=now or datetime.now(timezone.utc),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()


def _run_direct_multi_once(
    recorder_db,
    output_db,
    model_path,
    features_path,
    config_path,
    *,
    realistic_settings: MultiStrategyRealisticExecutionSettings,
    trade_selection_settings: MultiStrategyTradeSelectionSettings | None = None,
    pending_realistic_entries: dict | None = None,
    now: datetime | None = None,
) -> dict:
    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(output_db))
    try:
        return run_multi_strategy_paper_trader_once(
            recorder,
            output,
            model=_load_model(str(model_path)),
            feature_columns=_load_feature_columns(str(features_path)),
            strategies=load_multi_strategy_config(config_path),
            recorder_db_path=str(recorder_db),
            output_db_path=str(output_db),
            model_path=str(model_path),
            feature_columns_path=str(features_path),
            run_id="test_multi",
            max_btc_age_sec=3.0,
            max_feature_age_sec=100000.0,
            realistic_execution_settings=realistic_settings,
            trade_selection_settings=trade_selection_settings,
            pending_realistic_entries=pending_realistic_entries,
            emit_logs=False,
            now=now or datetime.now(timezone.utc),
        )
    finally:
        recorder.close()
        output.close()


def test_load_multi_strategy_config_filters_disabled(tmp_path) -> None:
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [
            _strategy("enabled_yes", direction_mode="YES_ONLY"),
            {**_strategy("disabled_no", direction_mode="NO_ONLY"), "enabled": False},
        ],
    )

    strategies = load_multi_strategy_config(config_path)

    assert [strategy.strategy_id for strategy in strategies] == ["enabled_yes"]
    assert strategies[0].direction_mode == "YES_ONLY"


def test_load_multi_strategy_config_inherits_global_liquidity_block(tmp_path) -> None:
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [_strategy("global_liq", direction_mode="YES_ONLY")],
        liquidity={
            "enabled": True,
            "max_top_of_book_fill_fraction": 0.25,
            "min_top_of_book_shares": 3.0,
            "skip_if_liquidity_missing": True,
        },
    )

    strategies = load_multi_strategy_config(config_path)

    assert len(strategies) == 1
    assert strategies[0].liquidity_fill_check_enabled is True
    assert strategies[0].max_top_of_book_fill_fraction == 0.25
    assert strategies[0].min_top_of_book_shares == 3.0
    assert strategies[0].skip_if_liquidity_missing is True


def test_load_multi_strategy_config_reads_blocked_liquidity_regimes(tmp_path) -> None:
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [
            _strategy(
                "block_thin",
                direction_mode="NO_ONLY",
                blocked_liquidity_regimes=["thin"],
            )
        ],
    )

    strategies = load_multi_strategy_config(config_path)

    assert strategies[0].blocked_liquidity_regimes == ("thin",)


def test_blocked_liquidity_regime_skips_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute("ALTER TABLE features ADD COLUMN total_liquidity REAL")
        conn.commit()
    finally:
        conn.close()
    _insert_candidate(
        recorder_db,
        market_id="thin_no_signal",
        signal=0.1,
        best_ask_no=0.4,
        time_until_resolution=45.0,
    )
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute(
            "UPDATE features SET total_liquidity = 50 WHERE market_id = 'thin_no_signal'"
        )
        conn.commit()
    finally:
        conn.close()
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "no_thin_block",
                direction_mode="NO_ONLY",
                long_threshold=2.0,
                short_threshold=0.2,
                min_probability_for_direction=0.8,
                max_probability_for_direction=0.9,
                blocked_liquidity_regimes=["thin"],
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert trades == []
    assert any(
        row["strategy_id"] == "no_thin_block"
        and row["candidate_direction"] == "NO"
        and row["rejection_reason"] == "liquidity_regime_blocked"
        for row in candidates
    )
    stats = report["last_poll"]["strategy_stats"][0]
    assert stats["skipped_liquidity_regime_blocked"] == 1


def test_multiple_strategies_evaluated_from_same_feature_row(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once(
        tmp_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("yes_b", direction_mode="YES_ONLY", long_threshold=0.75, short_threshold=-1.0),
        ],
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    trade_keys = {(row["strategy_id"], row["market_id"], row["signal_direction"]) for row in trades}

    assert ("yes_a", "yes_signal", "YES") in trade_keys
    assert ("yes_b", "yes_signal", "YES") in trade_keys
    assert report["last_poll"]["probability_yes"] is not None
    assert report["last_poll"]["active_strategies"] == 2


def test_trade_selection_default_behavior_unchanged(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once(
        tmp_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("yes_b", direction_mode="YES_ONLY", long_threshold=0.75, short_threshold=-1.0),
        ],
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]

    assert len([row for row in trades if row["status"] == "open"]) == 2
    assert report["last_poll"]["trade_selection_enabled"] is False
    assert report["last_poll"]["trade_selection_suppressed_count"] == 0


def test_trade_selection_opens_only_one_trade_per_market_direction(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once_with_config(
        tmp_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("yes_b", direction_mode="YES_ONLY", long_threshold=0.75, short_threshold=-1.0),
        ],
        config_extra={
            "trade_selection": {
                "selection_enabled": True,
                "mode": "best_per_market_direction",
                "score_field": "estimated_edge",
                "max_new_trades_per_market": 1,
                "max_new_trades_per_market_direction": 1,
                "allow_multiple_strategy_variants_same_market": False,
            }
        },
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert len([row for row in trades if row["status"] == "open"]) == 1
    assert report["last_poll"]["trade_selection_enabled"] is True
    assert report["last_poll"]["trade_selection_candidates_before"] == 2
    assert report["last_poll"]["trade_selection_candidates_after"] == 1
    assert report["last_poll"]["trade_selection_suppressed_count"] == 1
    assert any(row["rejection_reason"] == "trade_selection_deduped" for row in candidates)


def test_trade_selection_selects_best_estimated_edge(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [
            _strategy("low_edge", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0, entry_slippage_cents=5.0),
            _strategy("high_edge", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0, entry_slippage_cents=0.0),
        ],
        config_extra={"trade_selection": {"selection_enabled": True}},
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert trade["strategy_id"] == "high_edge"
    assert any(
        row["strategy_id"] == "low_edge"
        and row["rejection_reason"] == "trade_selection_deduped"
        for row in candidates
    )


def test_trade_selection_tie_breakers_are_deterministic(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [
            _strategy("z_same", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("a_same", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
        ],
        config_extra={"trade_selection": {"selection_enabled": True}},
    )

    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["strategy_id"] == "a_same"


def test_live_trading_disabled_by_default_does_not_log_orders(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once(
        tmp_path,
        [_strategy("paper_only", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    assert _rows(output_db, "paper_trades")
    assert _rows(output_db, "live_orders") == []
    assert report["last_poll"]["live_trading"]["live_trading_enabled"] is False


def test_live_trading_dry_run_logs_order_without_submitter_call(tmp_path) -> None:
    submit_calls = []

    def submitter(order):
        submit_calls.append(order)
        return {"exchange_order_id": "should_not_submit"}

    _recorder_db, output_db, *_rest, report = _run_once_with_config(
        tmp_path,
        [_strategy("dry_run", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={"live_trading": _live_trading_config(tmp_path)},
        live_order_submitter=submitter,
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert submit_calls == []
    assert live_order["status"] == "dry_run"
    assert live_order["reason"] == "dry_run_orders"
    assert live_order["direction"] == "YES"
    assert live_order["size_usd"] == 1.0
    assert live_order["size_shares"] == 2.0
    assert report["last_poll"]["live_trading"]["status_counts"]["dry_run"] == 1


def test_live_trading_non_dry_run_fails_closed_without_submitter(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [_strategy("no_submitter", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={
            "live_trading": _live_trading_config(tmp_path, dry_run_orders=False)
        },
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert live_order["status"] == "blocked"
    assert live_order["reason"] == "live_submitter_unavailable"
    assert "submitter" in live_order["error"]


def test_live_trading_kill_switch_blocks_order(tmp_path) -> None:
    kill_switch = tmp_path / "KILL_LIVE_TRADING"
    kill_switch.write_text("stop", encoding="utf-8")

    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [_strategy("kill_switch", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={
            "live_trading": _live_trading_config(
                tmp_path,
                kill_switch_file=str(kill_switch),
            )
        },
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert live_order["status"] == "blocked"
    assert live_order["reason"] == "kill_switch"


def test_live_trading_max_order_size_blocks_order(tmp_path) -> None:
    strategy = _strategy("too_large", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)
    strategy["stake_usd"] = 2.0

    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [strategy],
        config_extra={"live_trading": _live_trading_config(tmp_path, max_order_usd=1.0)},
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert live_order["status"] == "blocked"
    assert live_order["reason"] == "max_order_usd"
    assert live_order["size_usd"] == 2.0


def test_live_trading_stale_btc_blocks_order(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [_strategy("btc_stale", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={
            "live_trading": _live_trading_config(
                tmp_path,
                reject_if_btc_age_above_sec=0.5,
            )
        },
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert live_order["status"] == "blocked"
    assert live_order["reason"] == "btc_stale"


def test_live_trading_max_daily_loss_blocks_order(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="loss_block", signal=0.8, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("loss_block", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        live_trading=_live_trading_config(tmp_path, max_daily_loss_usd=5.0),
    )
    conn = connect_paper_output_db(output_db)
    try:
        ensure_paper_schema(conn)
        from src.multi_strategy_paper_trader import ensure_multi_strategy_schema

        ensure_multi_strategy_schema(conn)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp,
                signal_direction, stake_usd, status, realized_pnl_usd
            ) VALUES (?, 'run_live', 'prior_loss', ?, 'YES', 1.0, 'closed', -6.0)
            """,
            (now_iso, now_iso),
        )
        conn.commit()
    finally:
        conn.close()

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    live_order = dict(_rows(output_db, "live_orders")[0])

    assert live_order["status"] == "blocked"
    assert live_order["reason"] == "max_daily_loss_usd"


def test_live_trading_duplicate_market_exposure_blocks_order(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="dup_market", signal=0.8, best_ask_yes=0.5)
    strategy = _strategy(
        "dup_market",
        direction_mode="YES_ONLY",
        long_threshold=0.7,
        short_threshold=-1.0,
        max_open_trades=10,
        one_trade_per_market=False,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [strategy],
        live_trading=_live_trading_config(tmp_path, max_trades_per_market=1),
    )
    kwargs = {
        "recorder_db_path": str(recorder_db),
        "model_path": str(model_path),
        "feature_columns_path": str(features_path),
        "config_path": str(config_path),
        "output_db_path": str(output_db),
        "run_id": "test_multi",
        "poll_sec": 0.0,
        "max_feature_age_sec": 100000.0,
        "max_iterations": 1,
        "emit_logs": False,
    }

    run_multi_strategy_paper_trader(**kwargs)
    run_multi_strategy_paper_trader(**kwargs)
    live_orders = [dict(row) for row in _rows(output_db, "live_orders")]

    assert live_orders[0]["status"] == "dry_run"
    assert live_orders[1]["status"] == "blocked"
    assert live_orders[1]["reason"] == "max_trades_per_market"


def test_yes_and_no_strategies_both_log_candidates(tmp_path) -> None:
    _recorder_db, output_db, *_rest = _run_once(
        tmp_path,
        [
            _strategy("yes_only", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=0.3),
            _strategy("no_only", direction_mode="NO_ONLY", long_threshold=0.7, short_threshold=0.3),
        ],
    )

    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]
    trades = [dict(row) for row in _rows(output_db, "paper_trades")]

    assert {row["strategy_id"] for row in candidates} == {"yes_only", "no_only"}
    assert ("yes_only", "YES") in {
        (row["strategy_id"], row["candidate_direction"]) for row in candidates
    }
    assert ("no_only", "NO") in {
        (row["strategy_id"], row["candidate_direction"]) for row in candidates
    }
    assert ("yes_only", "YES") in {
        (row["strategy_id"], row["signal_direction"]) for row in trades
    }
    assert ("no_only", "NO") in {
        (row["strategy_id"], row["signal_direction"]) for row in trades
    }


def test_duplicate_candidate_prevention(tmp_path) -> None:
    recorder_db, output_db, model_path, features_path, config_path, _report = _run_once(
        tmp_path,
        [_strategy("dedupe", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )
    initial_candidates = len(_rows(output_db, "paper_trade_candidates"))

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidates = _rows(output_db, "paper_trade_candidates")
    trades = _rows(output_db, "paper_trades")
    assert len(candidates) == initial_candidates
    assert len([row for row in trades if row["status"] == "open"]) == 1


def test_regime_classification_uses_available_feature_and_snapshot_fields() -> None:
    tags = _regime_tags_for_row(
        {
            "price_change_60s": 0.04,
            "rolling_volatility_60s": 0.04,
            "spread_yes": 0.01,
            "spread_no": 0.02,
            "total_liquidity": 1500,
            "time_until_resolution": 45,
        }
    )

    assert tags == {
        "btc_trend_regime": "strong_up",
        "volatility_regime": "high",
        "spread_regime": "tight",
        "liquidity_regime": "deep",
        "time_regime": "30-60s",
    }


def test_regime_classification_missing_fields_is_unknown() -> None:
    tags = _regime_tags_for_row({})

    assert tags == {
        "btc_trend_regime": "unknown",
        "volatility_regime": "unknown",
        "spread_regime": "unknown",
        "liquidity_regime": "unknown",
        "time_regime": "unknown",
    }


def test_open_trade_records_regime_tags_at_entry(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.executescript(
            """
            ALTER TABLE features ADD COLUMN price_change_60s REAL;
            ALTER TABLE features ADD COLUMN rolling_volatility_60s REAL;
            ALTER TABLE features ADD COLUMN total_liquidity REAL;
            """
        )
        conn.commit()
    finally:
        conn.close()
    _insert_candidate(
        recorder_db,
        market_id="regime_signal",
        signal=0.8,
        best_ask_yes=0.5,
        time_until_resolution=75.0,
    )
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute(
            """
            UPDATE features
            SET price_change_60s = -0.02,
                rolling_volatility_60s = 0.015,
                total_liquidity = 250
            WHERE market_id = 'regime_signal'
            """
        )
        conn.execute(
            """
            UPDATE market_snapshots
            SET spread_yes = 0.07,
                spread_no = 0.06
            WHERE market_id = 'regime_signal'
            """
        )
        conn.commit()
    finally:
        conn.close()

    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("regime_yes", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )
    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    assert trade["btc_trend_regime"] == "weak_down"
    assert trade["volatility_regime"] == "medium"
    assert trade["spread_regime"] == "wide"
    assert trade["liquidity_regime"] == "normal"
    assert trade["time_regime"] == "60-120s"


def test_yes_threshold_skip_logs_full_candidate_context(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="yes_threshold", signal=0.8, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_threshold", direction_mode="YES_ONLY", long_threshold=0.9, short_threshold=-1.0)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert candidate["decision"] == "SKIP"
    assert candidate["rejection_reason"] == "threshold"
    assert candidate["candidate_direction"] == "YES"
    assert candidate["candidate_entry_price"] == 0.5
    assert candidate["candidate_adjusted_entry_price"] == 0.5
    assert candidate["probability_for_direction"] == 0.8
    assert candidate["estimated_edge"] == 0.3
    assert candidate["threshold_used"] == 0.9


def test_no_threshold_skip_logs_full_candidate_context(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="no_threshold", signal=0.2, best_ask_no=0.4)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("no_threshold", direction_mode="NO_ONLY", long_threshold=2.0, short_threshold=0.1)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert candidate["decision"] == "SKIP"
    assert candidate["rejection_reason"] == "threshold"
    assert candidate["candidate_direction"] == "NO"
    assert candidate["candidate_entry_price"] == 0.4
    assert candidate["candidate_adjusted_entry_price"] == 0.4
    assert candidate["probability_for_direction"] == 0.8
    assert candidate["estimated_edge"] == 0.4
    assert candidate["threshold_used"] == 0.1


def test_time_window_skip_logs_full_candidate_context(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="time_skip", signal=0.8, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "yes_time_skip",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                min_time_until_resolution_sec=250.0,
                max_time_until_resolution_sec=300.0,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert candidate["rejection_reason"] == "time_window"
    assert candidate["candidate_direction"] == "YES"
    assert candidate["candidate_entry_price"] == 0.5
    assert candidate["candidate_adjusted_entry_price"] == 0.5
    assert candidate["probability_for_direction"] == 0.8
    assert candidate["estimated_edge"] == 0.3


def test_both_strategy_logs_yes_and_no_candidates_when_enabled(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="both_threshold",
        signal=0.5,
        best_ask_yes=0.55,
        best_ask_no=0.45,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("both_threshold", direction_mode="BOTH", long_threshold=0.9, short_threshold=0.1)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]
    by_direction = {row["candidate_direction"]: row for row in candidates}
    assert set(by_direction) == {"YES", "NO"}
    assert by_direction["YES"]["rejection_reason"] == "threshold"
    assert by_direction["YES"]["candidate_entry_price"] == 0.55
    assert by_direction["YES"]["probability_for_direction"] == 0.5
    assert by_direction["NO"]["rejection_reason"] == "threshold"
    assert by_direction["NO"]["candidate_entry_price"] == 0.45
    assert by_direction["NO"]["probability_for_direction"] == 0.5


def test_candidate_dedupe_keeps_both_direction_rows(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="both_dedupe",
        signal=0.5,
        best_ask_yes=0.55,
        best_ask_no=0.45,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("both_dedupe", direction_mode="BOTH", long_threshold=0.9, short_threshold=0.1)],
    )
    kwargs = dict(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    run_multi_strategy_paper_trader(**kwargs)
    run_multi_strategy_paper_trader(**kwargs)

    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]
    assert len(candidates) == 2
    assert {
        (row["strategy_id"], row["market_id"], row["candidate_direction"])
        for row in candidates
    } == {
        ("both_dedupe", "both_dedupe", "YES"),
        ("both_dedupe", "both_dedupe", "NO"),
    }


def test_candidate_below_min_probability_is_skipped(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="below_min", signal=0.65, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "yes_prob_min",
                direction_mode="YES_ONLY",
                long_threshold=0.60,
                short_threshold=-1.0,
                min_probability_for_direction=0.70,
                max_probability_for_direction=0.90,
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    stats = report["last_poll"]["strategy_stats"][0]
    assert candidate["rejection_reason"] == "probability_below_min"
    assert candidate["candidate_direction"] == "YES"
    assert candidate["probability_for_direction"] == 0.65
    assert candidate["min_probability_for_direction"] == 0.70
    assert candidate["max_probability_for_direction"] == 0.90
    assert stats["skipped_probability_below_min"] == 1
    assert _rows(output_db, "paper_trades") == []


def test_candidate_above_max_probability_is_skipped(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="above_max", signal=0.95, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "yes_prob_max",
                direction_mode="YES_ONLY",
                long_threshold=0.70,
                short_threshold=-1.0,
                min_probability_for_direction=0.70,
                max_probability_for_direction=0.90,
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    stats = report["last_poll"]["strategy_stats"][0]
    assert candidate["rejection_reason"] == "probability_above_max"
    assert candidate["candidate_direction"] == "YES"
    assert candidate["probability_for_direction"] == 0.95
    assert stats["skipped_probability_above_max"] == 1
    assert _rows(output_db, "paper_trades") == []


def test_candidate_inside_probability_band_can_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="inside_band", signal=0.80, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "yes_inside_band",
                direction_mode="YES_ONLY",
                long_threshold=0.70,
                short_threshold=-1.0,
                min_probability_for_direction=0.70,
                max_probability_for_direction=0.90,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert trade["signal_direction"] == "YES"
    assert trade["probability_for_direction"] == 0.80
    assert candidate["decision"] == "TRADE"
    assert candidate["min_probability_for_direction"] == 0.70
    assert candidate["max_probability_for_direction"] == 0.90


def test_yes_and_no_probability_bands_use_probability_for_direction(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="no_inside_band", signal=0.20, best_ask_no=0.4)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "no_inside_band",
                direction_mode="NO_ONLY",
                long_threshold=2.0,
                short_threshold=0.30,
                min_probability_for_direction=0.70,
                max_probability_for_direction=0.90,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert trade["signal_direction"] == "NO"
    assert trade["probability_for_direction"] == 0.8
    assert candidate["candidate_direction"] == "NO"
    assert candidate["probability_for_direction"] == 0.8


def _freeze_fixture_clock(monkeypatch) -> None:
    import src.multi_strategy_paper_trader as trader_module
    from tests.test_baseline_paper_trader import NOW as fixture_now

    class FixtureDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixture_now.astimezone(tz) if tz is not None else fixture_now.replace(tzinfo=None)

    monkeypatch.setattr(trader_module, 'datetime', FixtureDatetime)


def test_balanced_late_config_yes_strategy_uses_direct_yes_probability(tmp_path, monkeypatch) -> None:
    _freeze_fixture_clock(monkeypatch)
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="balanced_yes",
        signal=0.80,
        best_ask_yes=0.5,
        close_offset_sec=165,
    )
    # Synthetic directional gates; no private strategy grid is required.
    strategy = _strategy(
        "synthetic_yes", direction_mode="YES_ONLY",
        long_threshold=0.7, short_threshold=-1.0,
        min_estimated_edge=0.03, min_probability_for_direction=0.70,
        max_probability_for_direction=0.90, max_time_until_resolution_sec=60,
        entry_slippage_cents=0.01,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(config_path, [strategy])

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert trade["signal_direction"] == "YES"
    assert trade["probability_for_direction"] == 0.8
    assert candidate["candidate_direction"] == "YES"
    assert candidate["probability_for_direction"] == 0.8
    assert candidate["threshold_used"] == 0.7
    assert candidate["estimated_edge"] == 0.2999


def test_balanced_late_config_no_strategy_uses_inverse_probability(tmp_path, monkeypatch) -> None:
    _freeze_fixture_clock(monkeypatch)
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="balanced_no",
        signal=0.20,
        best_ask_no=0.4,
        close_offset_sec=165,
    )
    # Synthetic directional gates; no private strategy grid is required.
    strategy = _strategy(
        "synthetic_no", direction_mode="NO_ONLY",
        long_threshold=2.0, short_threshold=0.3,
        min_estimated_edge=0.03, min_probability_for_direction=0.70,
        max_probability_for_direction=0.90, max_time_until_resolution_sec=60,
        entry_slippage_cents=0.01,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(config_path, [strategy])

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    candidate = dict(_rows(output_db, "paper_trade_candidates")[0])
    assert trade["signal_direction"] == "NO"
    assert trade["probability_for_direction"] == 0.8
    assert candidate["candidate_direction"] == "NO"
    assert candidate["probability_for_direction"] == 0.8
    assert candidate["threshold_used"] == 0.3
    assert candidate["estimated_edge"] == 0.3999


def test_yes_trade_with_negative_edge_is_skipped(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="bad_yes", signal=0.70, best_ask_yes=0.80)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_guard", direction_mode="YES_ONLY", long_threshold=0.65, short_threshold=-1.0)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]
    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert candidates[0]["candidate_direction"] == "YES"
    assert candidates[0]["rejection_reason"] == "non_positive_edge"
    assert candidates[0]["estimated_edge"] < 0
    assert trades == []


def test_no_trade_with_negative_edge_is_skipped(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="bad_no", signal=0.25, best_ask_no=0.90)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("no_guard", direction_mode="NO_ONLY", long_threshold=2.0, short_threshold=0.35)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]
    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert candidates[0]["candidate_direction"] == "NO"
    assert candidates[0]["rejection_reason"] == "non_positive_edge"
    assert candidates[0]["estimated_edge"] < 0
    assert trades == []


def test_yes_trade_with_positive_edge_opens(tmp_path) -> None:
    case_dir = tmp_path / "positive_yes"
    case_dir.mkdir()
    recorder_db, _paper_db, model_path, features_path = _fixture(case_dir)
    _insert_candidate(recorder_db, market_id="good_yes", signal=0.80, best_ask_yes=0.50)
    config_path = case_dir / "strategy_grid.json"
    output_db = case_dir / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_positive", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )
    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert trades[0]["signal_direction"] == "YES"
    assert trades[0]["estimated_edge"] > 0


def test_no_trade_with_positive_edge_opens(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="good_no", signal=0.20, best_ask_no=0.40)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("no_positive", direction_mode="NO_ONLY", long_threshold=2.0, short_threshold=0.35)],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert trades[0]["signal_direction"] == "NO"
    assert trades[0]["probability_for_direction"] == 0.8
    assert trades[0]["estimated_edge"] > 0


def test_require_positive_edge_false_allows_zero_edge_legacy_behavior(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="zero_yes", signal=0.50, best_ask_yes=0.50)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "legacy_zero",
                direction_mode="YES_ONLY",
                long_threshold=0.50,
                short_threshold=-1.0,
                require_positive_edge=False,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert trades[0]["estimated_edge"] == 0.0


def test_yes_take_profit_exit_closes_trade(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "yes_tp",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
            )
        ],
    )

    _insert_exit_snapshot(recorder_db, market_id="yes_signal", best_bid_yes=0.53)
    counts = _run_exit_update(recorder_db, output_db, config_path)
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert counts["closed"] == 1
    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == 0.53
    assert trade["realized_pnl_usd"] == 0.06
    assert trade["realized_roi"] == 0.06


def test_yes_stop_loss_exit_closes_trade(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "yes_sl",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
            )
        ],
    )

    _insert_exit_snapshot(recorder_db, market_id="yes_signal", best_bid_yes=0.47)
    _run_exit_update(recorder_db, output_db, config_path)
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "stop_loss"
    assert trade["realized_pnl_usd"] == -0.06
    assert trade["realized_roi"] == -0.06


def test_no_take_profit_exit_closes_trade(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "no_tp",
                direction_mode="NO_ONLY",
                long_threshold=2.0,
                short_threshold=0.3,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
            )
        ],
    )

    _insert_exit_snapshot(recorder_db, market_id="no_signal", best_bid_no=0.43)
    _run_exit_update(recorder_db, output_db, config_path)
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["signal_direction"] == "NO"
    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "take_profit"
    assert trade["realized_pnl_usd"] == 0.075
    assert trade["realized_roi"] == 0.075


def test_no_stop_loss_exit_closes_trade(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "no_sl",
                direction_mode="NO_ONLY",
                long_threshold=2.0,
                short_threshold=0.3,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
            )
        ],
    )

    _insert_exit_snapshot(recorder_db, market_id="no_signal", best_bid_no=0.37)
    _run_exit_update(recorder_db, output_db, config_path)
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["signal_direction"] == "NO"
    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "stop_loss"
    assert trade["realized_pnl_usd"] == -0.075
    assert trade["realized_roi"] == -0.075


def test_fixed_horizon_exit_closes_trade(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "yes_fixed",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                exit_type="FIXED_HORIZON_EXIT",
                fixed_horizon_exit_sec=15,
            )
        ],
    )
    now = datetime.now(timezone.utc)
    _set_trade_created_at(output_db, now - timedelta(seconds=20))
    _insert_exit_snapshot(recorder_db, market_id="yes_signal", best_bid_yes=0.51, timestamp=now)

    _run_exit_update(recorder_db, output_db, config_path, now=now)
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "fixed_horizon"
    assert trade["adjusted_exit_price"] == 0.51


def test_closed_trade_is_not_settled_again(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "yes_closed",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
            )
        ],
    )
    _insert_exit_snapshot(recorder_db, market_id="yes_signal", best_bid_yes=0.53)
    _run_exit_update(recorder_db, output_db, config_path)
    close_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute("UPDATE paper_trades SET market_close_time = ?", (close_time.isoformat(),))
        conn.commit()
    finally:
        conn.close()
    _update_market_resolution(recorder_db, market_id="yes_signal", resolved=1, winning_outcome="YES")
    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(output_db))
    try:
        settled = settle_open_paper_trades(recorder, output, now=datetime.now(timezone.utc), emit_logs=False)
    finally:
        recorder.close()
        output.close()

    trade = dict(_rows(output_db, "paper_trades")[0])
    assert settled == 0
    assert trade["status"] == "closed"
    assert trade["settled_at"] is None
    assert trade["pnl_usd"] is None
    assert trade["realized_pnl_usd"] == 0.06


def test_hold_to_resolution_default_still_settles_normally(tmp_path) -> None:
    recorder_db, output_db, *_rest, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy(
                "yes_hold",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
            )
        ],
    )
    _insert_exit_snapshot(recorder_db, market_id="yes_signal", best_bid_yes=0.80)
    counts = _run_exit_update(recorder_db, output_db, config_path)
    assert counts == {}
    trade = dict(_rows(output_db, "paper_trades")[0])
    assert trade["status"] == "open"
    assert trade["exit_type"] == "HOLD_TO_RESOLUTION"

    close_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute("UPDATE paper_trades SET market_close_time = ?", (close_time.isoformat(),))
        conn.commit()
    finally:
        conn.close()
    _update_market_resolution(recorder_db, market_id="yes_signal", resolved=1, winning_outcome="YES")
    recorder = connect_recorder_read_only(str(recorder_db))
    output = connect_paper_output_db(str(output_db))
    try:
        settled = settle_open_paper_trades(recorder, output, now=datetime.now(timezone.utc), emit_logs=False)
    finally:
        recorder.close()
        output.close()

    trade = dict(_rows(output_db, "paper_trades")[0])
    assert settled == 1
    assert trade["status"] == "settled"
    assert trade["pnl_usd"] == 1.0
    assert trade["realized_pnl_usd"] is None


def test_fixed_stake_behavior_unchanged_when_bankroll_disabled(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once(
        tmp_path,
        [_strategy("fixed_yes", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["stake_usd"] == 1.0
    assert trade["starting_bankroll_usd"] is None
    assert trade["risk_sizing_reason"] is None


def test_liquidity_check_disabled_preserves_old_behavior(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once(
        tmp_path,
        [_strategy("no_liq_check", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["requested_shares"] is None
    assert trade["liquidity_check_passed"] is None


def test_liquidity_missing_skips_when_enabled(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once(
        tmp_path,
        [
            _strategy(
                "liq_missing",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            )
        ],
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert report["last_poll"]["strategies"][0]["skipped_liquidity_missing"] == 1
    assert trades[0]["status"] == "skipped"
    assert trades[0]["skip_reason"] == "liquidity_missing"
    assert trades[0]["liquidity_skip_reason"] == "liquidity_missing"
    assert trades[0]["requested_shares"] == 2.0
    assert trades[0]["liquidity_check_passed"] == 0
    assert candidates[0]["rejection_reason"] == "liquidity_missing"


def test_global_liquidity_block_enables_missing_liquidity_skip(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once_with_config(
        tmp_path,
        [
            _strategy(
                "global_liq_missing",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
            )
        ],
        config_extra={
            "liquidity": {
                "liquidity_fill_check_enabled": True,
                "skip_if_liquidity_missing": True,
            }
        },
    )

    trade = dict(_rows(output_db, "paper_trades")[0])
    stats = report["last_poll"]["strategies"][0]

    assert stats["liquidity_fill_check_enabled"] is True
    assert stats["liquidity_checked_count"] == 1
    assert stats["liquidity_blocked_count"] == 1
    assert stats["liquidity_missing_count"] == 1
    assert trade["status"] == "skipped"
    assert trade["skip_reason"] == "liquidity_missing"
    assert trade["liquidity_check_passed"] == 0


def test_low_liquidity_skips_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="low_liq", signal=0.8, best_ask_yes=0.5)
    _add_feature_top_ask_sizes(recorder_db, market_id="low_liq", yes_size=1.0)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "low_liq",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_liquidity_too_low"] == 1
    assert trade["status"] == "skipped"
    assert trade["skip_reason"] == "liquidity_too_low"
    assert trade["requested_shares"] == 2.0
    assert trade["max_fillable_shares"] == 1.0
    assert trade["liquidity_check_passed"] == 0


def test_sufficient_liquidity_allows_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="enough_liq", signal=0.8, best_ask_yes=0.5)
    _add_feature_top_ask_sizes(recorder_db, market_id="enough_liq", yes_size=5.0)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "enough_liq",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
                max_top_of_book_fill_fraction=0.5,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["requested_shares"] == 2.0
    assert trade["max_fillable_shares"] == 2.5
    assert trade["liquidity_fill_fraction_used"] == 0.5
    assert trade["liquidity_check_passed"] == 1
    assert trade["liquidity_skip_reason"] is None


def test_cashout_strategy_uses_liquidity_checks(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="cashout_liq", signal=0.8, best_ask_yes=0.5)
    _add_feature_top_ask_sizes(recorder_db, market_id="cashout_liq", yes_size=5.0)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "cashout_liq",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                exit_type="TAKE_PROFIT_STOP_LOSS",
                take_profit_pct=0.05,
                stop_loss_pct=0.05,
                liquidity_fill_check_enabled=True,
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])
    stats = report["last_poll"]["strategies"][0]

    assert trade["status"] == "open"
    assert trade["exit_type"] == "TAKE_PROFIT_STOP_LOSS"
    assert trade["liquidity_check_passed"] == 1
    assert trade["requested_shares"] == 2.0
    assert stats["liquidity_fill_check_enabled"] is True
    assert stats["liquidity_checked_count"] == 1
    assert stats["liquidity_passed_count"] == 1


def test_liquidity_check_reads_top_ask_size_from_order_book_levels(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="book_liq", signal=0.8, best_ask_yes=0.5)
    conn = sqlite3.connect(str(recorder_db))
    try:
        ts = conn.execute(
            "SELECT timestamp FROM features WHERE market_id = 'book_liq'"
        ).fetchone()[0]
        conn.execute(
            """
            CREATE TABLE order_book_levels (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                outcome_side TEXT,
                book_side TEXT,
                level INTEGER,
                price REAL,
                size REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO order_book_levels (
                run_id, timestamp, market_id, outcome_side, book_side, level, price, size
            ) VALUES ('run_live', ?, 'book_liq', 'YES', 'ask', 1, 0.5, 3.0)
            """,
            (ts,),
        )
        conn.commit()
    finally:
        conn.close()
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "book_liq",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            )
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["requested_shares"] == 2.0
    assert trade["max_fillable_shares"] == 3.0
    assert trade["liquidity_check_passed"] == 1


def test_invalid_adjusted_entry_price_skips_when_liquidity_check_enabled(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="invalid_entry", signal=0.8, best_ask_yes=None)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "invalid_entry",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            )
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_invalid_adjusted_entry_price"] == 1
    assert trade["status"] == "skipped"
    assert trade["skip_reason"] == "invalid_adjusted_entry_price"
    assert trade["liquidity_skip_reason"] == "invalid_adjusted_entry_price"
    assert trade["liquidity_check_passed"] == 0


def test_realistic_execution_disabled_preserves_instant_entry_behavior(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="realistic_disabled",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="realistic_disabled",
        best_ask_yes=0.60,
        timestamp=feature_time + timedelta(seconds=1),
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("realistic_disabled", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution={"realistic_execution_enabled": False, "apply_extra_slippage_cents": 1.0},
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["entry_price"] == 0.5
    assert trade["adjusted_entry_price"] == 0.5
    assert trade["realistic_execution_enabled"] is None
    assert trade["delayed_entry_price"] is None


def test_realistic_execution_zero_latency_can_use_signal_quote_with_extra_slippage(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    _insert_candidate(
        recorder_db,
        market_id="realistic_signal_quote",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("realistic_signal_quote", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(
            entry_latency_sec=0.0,
            require_quote_after_latency=False,
        ),
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["entry_price"] == 0.5
    assert trade["adjusted_entry_price"] == 0.51
    assert trade["realistic_execution_enabled"] == 1
    assert trade["delayed_entry_price"] == 0.5


def test_realistic_execution_uses_delayed_quote_and_extra_slippage(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="realistic_entry",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="realistic_entry",
        best_bid_yes=0.49,
        best_ask_yes=0.51,
        timestamp=feature_time + timedelta(seconds=1),
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("realistic_entry", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(),
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["entry_price"] == 0.51
    assert trade["adjusted_entry_price"] == 0.52
    assert trade["realistic_execution_enabled"] == 1
    assert trade["entry_latency_sec"] == 1.0
    assert trade["signal_entry_price"] == 0.5
    assert trade["delayed_entry_price"] == 0.51
    assert trade["entry_price_drift"] == 1.0
    assert trade["spread_cents_at_entry"] == 2.0
    assert trade["extra_slippage_cents_applied"] == 1.0


def test_realistic_execution_missing_delayed_quote_does_not_skip_immediately(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="missing_delayed",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("missing_delayed", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(),
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    assert report["last_poll"]["pending_realistic_entries"] == 1
    assert report["last_poll"]["strategies"][0]["skipped_quote_after_latency_missing"] == 0
    assert _rows(output_db, "paper_trades") == []


def test_realistic_execution_missing_delayed_quote_skips_after_max_wait(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=6)
    _insert_candidate(
        recorder_db,
        market_id="missing_delayed_expired",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("missing_delayed_expired", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(max_quote_wait_sec=3.0),
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_quote_after_latency_missing"] == 1
    assert trade["status"] == "skipped"
    assert trade["skip_reason"] == "quote_after_latency_missing"
    assert trade["realistic_execution_skip_reason"] == "quote_after_latency_missing"
    assert trade["realistic_execution_enabled"] == 1


def test_realistic_execution_delayed_quote_opens_trade_after_latency(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_candidate(
        recorder_db,
        market_id="pending_delayed_open",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("pending_delayed_open", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(),
    )
    pending: dict = {}
    settings = MultiStrategyRealisticExecutionSettings(**_realistic_execution_config())

    first = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=0.5),
    )
    assert first["pending_realistic_entries"] == 1
    assert _rows(output_db, "paper_trades") == []

    _insert_exit_snapshot(
        recorder_db,
        market_id="pending_delayed_open",
        best_bid_yes=0.50,
        best_ask_yes=0.51,
        timestamp=feature_time + timedelta(seconds=1.0),
    )
    second = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=1.2),
    )
    trades = _rows(output_db, "paper_trades")
    trade = dict(trades[0])

    assert second["pending_realistic_entries"] == 0
    assert second["realistic_entries_opened_after_latency"] == 1
    assert len(trades) == 1
    assert trade["status"] == "open"
    assert trade["entry_price"] == 0.51
    assert trade["adjusted_entry_price"] == 0.52


def test_realistic_execution_liquidity_check_runs_after_delayed_quote(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    delayed_time = feature_time + timedelta(seconds=1)
    _insert_candidate(
        recorder_db,
        market_id="pending_liquidity",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="pending_liquidity",
        best_bid_yes=0.50,
        best_ask_yes=0.51,
        timestamp=delayed_time,
    )
    conn = sqlite3.connect(str(recorder_db))
    try:
        conn.execute(
            """
            CREATE TABLE order_book_levels (
                run_id TEXT,
                timestamp TEXT,
                market_id TEXT,
                outcome_side TEXT,
                book_side TEXT,
                level INTEGER,
                price REAL,
                size REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO order_book_levels (
                run_id, timestamp, market_id, outcome_side, book_side, level, price, size
            ) VALUES ('run_live', ?, 'pending_liquidity', 'YES', 'ask', 1, 0.51, 3.0)
            """,
            (delayed_time.isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy(
                "pending_liquidity",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            )
        ],
        realistic_execution=_realistic_execution_config(),
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["status"] == "open"
    assert trade["entry_price"] == 0.51
    assert trade["adjusted_entry_price"] == 0.52
    assert trade["requested_shares"] == 1.9230769231
    assert trade["max_fillable_shares"] == 3.0
    assert trade["liquidity_check_passed"] == 1


def test_realistic_execution_missing_delayed_quote_waits_until_max_wait(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_candidate(
        recorder_db,
        market_id="pending_missing_wait",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("pending_missing_wait", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(max_quote_wait_sec=3.0),
    )
    pending: dict = {}
    settings = MultiStrategyRealisticExecutionSettings(
        **_realistic_execution_config(max_quote_wait_sec=3.0)
    )

    first = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=1.5),
    )
    second = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=3.5),
    )
    assert _rows(output_db, "paper_trades") == []
    third = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=4.1),
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert first["pending_realistic_entries"] == 1
    assert second["pending_realistic_entries"] == 1
    assert third["pending_realistic_entries_expired"] == 1
    assert trade["skip_reason"] == "quote_after_latency_missing"


def test_realistic_execution_duplicate_pending_candidate_is_not_created(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_candidate(
        recorder_db,
        market_id="pending_dedupe",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("pending_dedupe", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(),
    )
    pending: dict = {}
    settings = MultiStrategyRealisticExecutionSettings(**_realistic_execution_config())

    _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=0.2),
    )
    _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=settings,
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=0.4),
    )

    assert len(pending) == 1
    assert _rows(output_db, "paper_trades") == []


def test_trade_selection_creates_pending_entries_only_for_selected_candidates(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_candidate(
        recorder_db,
        market_id="selection_pending",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy("pending_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("pending_b", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
        ],
        realistic_execution=_realistic_execution_config(),
        trade_selection={"selection_enabled": True},
    )
    pending: dict = {}

    report = _run_direct_multi_once(
        recorder_db,
        output_db,
        model_path,
        features_path,
        config_path,
        realistic_settings=MultiStrategyRealisticExecutionSettings(**_realistic_execution_config()),
        trade_selection_settings=MultiStrategyTradeSelectionSettings(selection_enabled=True),
        pending_realistic_entries=pending,
        now=feature_time + timedelta(seconds=0.2),
    )

    assert report["trade_selection_candidates_before"] == 2
    assert report["trade_selection_candidates_after"] == 1
    assert report["pending_realistic_entries"] == 1
    assert len(pending) == 1
    assert _rows(output_db, "paper_trades") == []


def test_trade_selection_bankroll_exposure_uses_selected_candidates_only(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once_with_config(
        tmp_path,
        [
            _strategy("bankroll_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("bankroll_b", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
        ],
        config_extra={
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
            "base_risk_fraction": 0.01,
            "max_total_exposure_fraction": 0.01,
            "trade_selection": {"selection_enabled": True},
        },
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert len([row for row in trades if row["status"] == "open"]) == 1
    assert report["last_poll"]["trade_selection_suppressed_count"] == 1
    assert not any(row["rejection_reason"] == "bankroll_exposure_limit" for row in candidates)
    assert any(row["rejection_reason"] == "trade_selection_deduped" for row in candidates)


def test_realistic_execution_price_drift_skips(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="drift_skip",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="drift_skip",
        best_bid_yes=0.53,
        best_ask_yes=0.55,
        timestamp=feature_time + timedelta(seconds=1),
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("drift_skip", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(max_entry_price_drift_cents=2.0),
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_entry_price_drift_too_high"] == 1
    assert trade["skip_reason"] == "entry_price_drift_too_high"
    assert trade["entry_price_drift"] == 5.0


def test_realistic_execution_wide_spread_skips(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="spread_skip",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="spread_skip",
        best_bid_yes=0.45,
        best_ask_yes=0.50,
        timestamp=feature_time + timedelta(seconds=1),
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("spread_skip", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(reject_if_spread_above_cents=4.0),
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_spread_too_wide"] == 1
    assert trade["skip_reason"] == "spread_too_wide"
    assert trade["spread_cents_at_entry"] == 5.0


def test_realistic_execution_stale_btc_skips(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    feature_time = datetime.now(timezone.utc) - timedelta(seconds=3)
    _insert_candidate(
        recorder_db,
        market_id="btc_stale",
        signal=0.8,
        best_ask_yes=0.5,
        feature_time=feature_time,
    )
    _insert_exit_snapshot(
        recorder_db,
        market_id="btc_stale",
        best_ask_yes=0.50,
        timestamp=feature_time + timedelta(seconds=1),
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("btc_stale", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        realistic_execution=_realistic_execution_config(reject_if_btc_age_above_sec=0.5),
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["strategies"][0]["skipped_btc_stale"] == 1
    assert trade["skip_reason"] == "btc_stale"
    assert trade["realistic_execution_skip_reason"] == "btc_stale"


def test_bankroll_initializes_and_sizes_about_one_percent(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once_with_config(
        tmp_path,
        [_strategy("bankroll_yes", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={"bankroll_enabled": True, "starting_bankroll_usd": 100.0},
    )

    trade = dict(_rows(output_db, "paper_trades")[0])

    assert report["last_poll"]["bankroll"]["bankroll_enabled"] is True
    assert trade["starting_bankroll_usd"] == 100.0
    assert trade["bankroll_before_trade"] == 100.0
    assert trade["available_cash_before_trade"] == 100.0
    assert trade["open_exposure_before_trade"] == 0.0
    assert 1.0 <= trade["stake_usd"] <= 1.5
    assert trade["stake_fraction_of_bankroll"] == trade["stake_usd"] / 100.0
    assert trade["bankroll_status"] == "ACTIVE"
    assert "base_fraction" in trade["risk_sizing_reason"]


def test_bankroll_stake_never_exceeds_max_risk_fraction(tmp_path) -> None:
    _recorder_db, output_db, *_rest, _report = _run_once_with_config(
        tmp_path,
        [_strategy("bankroll_max", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        config_extra={
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
            "base_risk_fraction": 0.10,
            "max_risk_fraction": 0.05,
            "max_stake_usd": 100.0,
        },
    )

    trade = dict(_rows(output_db, "paper_trades")[0])

    assert trade["stake_usd"] == 5.0
    assert trade["stake_fraction_of_bankroll"] == 0.05


def test_bankroll_insufficient_available_cash_blocks_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="cash_block", signal=0.8, best_ask_yes=0.5)
    output_db = tmp_path / "paper_multi.db"
    _insert_manual_paper_trade(output_db, status="open", stake_usd=99.8, market_id="existing")
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [_strategy("cash_block_s", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        bankroll_enabled=True,
        starting_bankroll_usd=100.0,
        max_total_exposure_fraction=1.0,
        min_stake_usd=0.25,
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert report["last_poll"]["strategies"][0]["skipped_bankroll_insufficient_cash"] == 1
    assert candidates[0]["rejection_reason"] == "bankroll_insufficient_cash"


def test_bankroll_max_total_exposure_blocks_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="exposure_block", signal=0.8, best_ask_yes=0.5)
    output_db = tmp_path / "paper_multi.db"
    _insert_manual_paper_trade(output_db, status="open", stake_usd=20.0, market_id="existing")
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [_strategy("exposure_block_s", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        bankroll_enabled=True,
        starting_bankroll_usd=100.0,
        max_total_exposure_fraction=0.20,
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert report["last_poll"]["strategies"][0]["skipped_bankroll_exposure_limit"] == 1
    assert candidates[0]["rejection_reason"] == "bankroll_exposure_limit"


def test_bankroll_depletion_blocks_new_trade(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="depleted_block", signal=0.8, best_ask_yes=0.5)
    output_db = tmp_path / "paper_multi.db"
    _insert_manual_paper_trade(output_db, status="settled", stake_usd=100.0, pnl_usd=-100.0)
    config_path = tmp_path / "strategy_grid.json"
    _write_config(
        config_path,
        [_strategy("depleted_s", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
        bankroll_enabled=True,
        starting_bankroll_usd=100.0,
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert report["last_poll"]["bankroll"]["bankroll_status"] == "BLOWN_UP"
    assert report["last_poll"]["strategies"][0]["skipped_bankroll_depleted"] == 1
    assert candidates[0]["rejection_reason"] == "bankroll_depleted"


def test_max_open_trades_is_enforced_per_strategy(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="yes_1", signal=0.8, best_ask_yes=0.5)
    _insert_candidate(recorder_db, market_id="yes_2", signal=0.82, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0, max_open_trades=1),
            _strategy("yes_b", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0, max_open_trades=1),
        ],
    )
    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    candidates = [dict(row) for row in _rows(output_db, "paper_trade_candidates")]

    assert sum(1 for row in trades if row["strategy_id"] == "yes_a" and row["status"] == "open") == 1
    assert sum(1 for row in trades if row["strategy_id"] == "yes_b" and row["status"] == "open") == 1
    assert any(
        row["strategy_id"] == "yes_a" and row["rejection_reason"] == "max_open_trades"
        for row in candidates
    )
    assert any(
        row["strategy_id"] == "yes_b" and row["rejection_reason"] == "max_open_trades"
        for row in candidates
    )


def test_one_trade_per_market_is_enforced_per_strategy(tmp_path) -> None:
    recorder_db, output_db, model_path, features_path, config_path, _report = _run_once(
        tmp_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("yes_b", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
        ],
    )

    run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    trades = [dict(row) for row in _rows(output_db, "paper_trades")]
    assert sum(1 for row in trades if row["strategy_id"] == "yes_a" and row["status"] == "open") == 1
    assert sum(1 for row in trades if row["strategy_id"] == "yes_b" and row["status"] == "open") == 1


def test_analytics_groups_by_strategy_id(tmp_path) -> None:
    _recorder_db, output_db, *_rest = _run_once(
        tmp_path,
        [
            _strategy("yes_a", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("no_a", direction_mode="NO_ONLY", long_threshold=2.0, short_threshold=0.3),
        ],
    )
    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET status = 'settled',
                pnl_usd = CASE WHEN strategy_id = 'yes_a' THEN 0.5 ELSE -1.0 END,
                roi = CASE WHEN strategy_id = 'yes_a' THEN 0.5 ELSE -1.0 END,
                resolved_label = CASE WHEN strategy_id = 'yes_a' THEN 'YES' ELSE 'NO' END
            WHERE status = 'open'
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = build_paper_trader_analytics_report(
        paper_db_path=str(output_db),
        output_dir=str(tmp_path / "analytics"),
    )

    by_strategy = report["summary"]["strategy_breakdown"]
    assert by_strategy["yes_a"]["settled_pnl"] == 0.5
    assert by_strategy["no_a"]["settled_pnl"] == -1.0
    assert by_strategy["yes_a"]["candidate_trade_rate"] is not None


def test_heartbeat_includes_strategy_stats(tmp_path) -> None:
    _recorder_db, output_db, *_rest, report = _run_once(
        tmp_path,
        [
            _strategy("yes_trade", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy(
                "yes_time_skip",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                min_time_until_resolution_sec=250.0,
                max_time_until_resolution_sec=300.0,
            ),
            _strategy(
                "yes_liq_skip",
                direction_mode="YES_ONLY",
                long_threshold=0.7,
                short_threshold=-1.0,
                liquidity_fill_check_enabled=True,
            ),
        ],
    )

    stats = {
        item["strategy_id"]: item
        for item in report["last_poll"]["strategy_stats"]
    }

    assert stats["yes_trade"]["candidates_seen"] >= 1
    assert stats["yes_trade"]["candidate_decisions_evaluated_this_poll"] >= 1
    assert stats["yes_trade"]["trades_opened"] == 1
    assert stats["yes_trade"]["latest_decision"] in {"TRADE", "SKIP", "NONE"}
    assert stats["yes_time_skip"]["skipped_time_window"] >= 1
    assert stats["yes_time_skip"]["latest_rejection_reason"] in {"time_window", "threshold"}
    assert stats["yes_liq_skip"]["liquidity_fill_check_enabled"] is True
    assert stats["yes_liq_skip"]["liquidity_checked_count"] >= 1
    assert stats["yes_liq_skip"]["liquidity_blocked_count"] >= 1

    conn = sqlite3.connect(str(output_db))
    conn.row_factory = sqlite3.Row
    try:
        heartbeat = conn.execute(
            """
            SELECT *
            FROM multi_strategy_paper_trader_heartbeats
            ORDER BY datetime(timestamp) DESC, timestamp DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    stored = json.loads(heartbeat["per_strategy_json"])
    assert any(item["strategy_id"] == "yes_trade" for item in stored)
    assert heartbeat["candidate_rows_total"] >= 1
    assert heartbeat["candidate_rows_written_this_poll"] >= 1
    assert heartbeat["candidate_decisions_evaluated_this_poll"] >= 1
    assert heartbeat["liquidity_fill_check_enabled"] == 1
    assert heartbeat["liquidity_checked_count"] >= 1
    assert heartbeat["liquidity_blocked_count"] >= 1
    assert heartbeat["liquidity_missing_count"] >= 1


def test_compact_heartbeat_detail_summarizes_strategy_stats(tmp_path, capsys) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="yes_signal",
        signal=0.8,
        best_ask_yes=0.5,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [
            _strategy("yes_trade", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0),
            _strategy("yes_skip", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0, min_estimated_edge=0.99),
        ],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        heartbeat_detail="compact",
        emit_logs=True,
    )

    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("{")
    ]
    heartbeat = [
        event
        for event in events
        if event.get("event") == "multi_strategy_paper_trader_heartbeat"
    ][-1]
    assert isinstance(heartbeat["strategy_stats"], dict)
    assert heartbeat["strategy_stats"]["strategy_count"] == 2
    assert "top_active_strategies" in heartbeat["strategy_stats"]
    assert isinstance(report["last_poll"]["strategy_stats"], list)
    assert report["last_poll"]["heartbeat_detail"] == "compact"


def test_min_feature_timestamp_ignores_historical_features(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    old_ts = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=90)
    new_ts = old_ts + timedelta(seconds=60)
    min_ts = old_ts + timedelta(seconds=30)
    _insert_candidate(
        recorder_db,
        market_id="old_signal",
        signal=0.9,
        best_ask_yes=0.5,
        feature_time=old_ts,
        close_offset_sec=300,
    )
    _insert_candidate(
        recorder_db,
        market_id="new_signal",
        signal=0.9,
        best_ask_yes=0.5,
        feature_time=new_ts,
        close_offset_sec=300,
    )
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_live", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        min_feature_timestamp=min_ts.isoformat(),
        emit_logs=False,
    )

    rows = _rows(output_db, "paper_trades")
    assert report["last_poll"]["candidate_rows_seen"] == 1
    assert {row["market_id"] for row in rows} == {"new_signal"}


def test_multi_strategy_startup_heartbeat_written_without_poll(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_live", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_iterations=0,
        live_start_now=True,
        emit_logs=False,
    )

    conn = sqlite3.connect(str(output_db))
    conn.row_factory = sqlite3.Row
    try:
        heartbeat = conn.execute(
            """
            SELECT *
            FROM multi_strategy_paper_trader_heartbeats
            ORDER BY datetime(timestamp) DESC, timestamp DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    assert report["iterations"] == 0
    assert heartbeat["status"] == "starting"
    assert heartbeat["timestamp"] is not None
    assert heartbeat["latest_feature_timestamp"] is not None


def test_multi_strategy_writes_poll_heartbeat_when_no_candidates_after_min_timestamp(tmp_path) -> None:
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="old_signal", signal=0.9, best_ask_yes=0.5)
    config_path = tmp_path / "strategy_grid.json"
    output_db = tmp_path / "paper_multi.db"
    _write_config(
        config_path,
        [_strategy("yes_live", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    report = run_multi_strategy_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(output_db),
        run_id="test_multi",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        min_feature_timestamp=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
        emit_logs=False,
    )

    conn = sqlite3.connect(str(output_db))
    try:
        heartbeat_count = conn.execute(
            "SELECT COUNT(*) FROM multi_strategy_paper_trader_heartbeats"
        ).fetchone()[0]
    finally:
        conn.close()

    assert report["last_poll"]["candidate_rows_seen"] == 0
    assert heartbeat_count >= 2


def test_export_meta_strategy_dataset_reads_multi_strategy_db(tmp_path) -> None:
    _recorder_db, output_db, *_rest = _run_once(
        tmp_path,
        [_strategy("yes_export", direction_mode="YES_ONLY", long_threshold=0.7, short_threshold=-1.0)],
    )

    report = export_meta_strategy_dataset(
        paper_db_path=str(output_db),
        output_path=str(tmp_path / "meta.parquet"),
        output_csv_path=str(tmp_path / "meta.csv"),
        include_unresolved=True,
    )

    assert report["status"] == "ok"
    assert report["rows_exported"] >= 1


def test_run_multi_strategy_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-multi-strategy-paper-trader",
            "--db",
            "data/recorder.db",
            "--model-path",
            "data/models/baseline_latest/model_logistic_regression.joblib",
            "--feature-columns",
            "data/models/baseline_latest/feature_columns.json",
            "--config",
            "data/strategy_configs/live_strategy_grid.json",
            "--output-db",
            "data/paper_trades_multi_strategy.db",
            "--run-id",
            "multi_strategy_live_test",
        ],
    )

    args = main.parse_args()

    assert args.command == "run-multi-strategy-paper-trader"
    assert args.config == "data/strategy_configs/live_strategy_grid.json"
    assert args.output_db == "data/paper_trades_multi_strategy.db"
