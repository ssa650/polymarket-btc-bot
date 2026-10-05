from __future__ import annotations

import json
import sqlite3
import sys
from datetime import timedelta

from src.intramarket_paper_trader import (
    ensure_intramarket_schema,
    export_intramarket_strategy_dataset,
    load_intramarket_strategy_config,
    run_intramarket_paper_trader,
    run_intramarket_paper_trader_once,
    update_intramarket_positions,
)
from tests.test_baseline_paper_trader import (
    BASE,
    ProbabilityBySignalModel,
    _fixture,
    _insert_candidate,
    _iso,
)


INTRAMARKET_CASHOUT_CONFIG = "data/strategy_configs/intramarket_cashout_strategy_grid.json"


def _write_config(path, strategies: list[dict]) -> None:
    path.write_text(json.dumps({"strategies": strategies}), encoding="utf-8")


def _strategy(
    strategy_id: str,
    *,
    direction_mode: str = "NO_ONLY",
    exit_type: str = "FIXED_HORIZON_EXIT",
    exit_after_sec: float | None = 30.0,
    take_profit_cents: float | None = None,
    stop_loss_cents: float | None = None,
    max_hold_sec: float | None = None,
    take_profit_roi: float | None = None,
    stop_loss_roi: float | None = None,
    max_hold_seconds: float | None = None,
    min_hold_seconds: float | None = None,
    exit_probability_below: float | None = None,
    exit_probability_drop_from_entry: float | None = None,
    max_open_positions_per_strategy: int = 3,
    max_open_positions_per_market: int = 1,
    cooldown_after_exit_sec: float = 0.0,
    allow_reentry_per_market: bool = False,
) -> dict:
    item = {
        "strategy_id": strategy_id,
        "strategy_name": f"Strategy {strategy_id}",
        "enabled": True,
        "direction_mode": direction_mode,
        "entry_type": "MODEL_EDGE",
        "exit_type": exit_type,
        "min_probability_for_direction": 0.70,
        "max_probability_for_direction": 0.90,
        "min_estimated_edge": 0.05,
        "require_positive_edge": True,
        "min_time_until_resolution_sec": 30,
        "max_time_until_resolution_sec": 240,
        "stake_usd": 1.0,
        "entry_slippage_cents": 0.0,
        "exit_slippage_cents": 0.0,
        "fee_cents": 0.0,
        "max_open_positions_per_strategy": max_open_positions_per_strategy,
        "max_open_positions_per_market": max_open_positions_per_market,
        "cooldown_after_exit_sec": cooldown_after_exit_sec,
        "allow_reentry_per_market": allow_reentry_per_market,
    }
    if exit_after_sec is not None:
        item["exit_after_sec"] = exit_after_sec
    if take_profit_cents is not None:
        item["take_profit_cents"] = take_profit_cents
    if stop_loss_cents is not None:
        item["stop_loss_cents"] = stop_loss_cents
    if max_hold_sec is not None:
        item["max_hold_sec"] = max_hold_sec
    if take_profit_roi is not None:
        item["take_profit_roi"] = take_profit_roi
    if stop_loss_roi is not None:
        item["stop_loss_roi"] = stop_loss_roi
    if max_hold_seconds is not None:
        item["max_hold_seconds"] = max_hold_seconds
    if min_hold_seconds is not None:
        item["min_hold_seconds"] = min_hold_seconds
    if exit_probability_below is not None:
        item["exit_probability_below"] = exit_probability_below
    if exit_probability_drop_from_entry is not None:
        item["exit_probability_drop_from_entry"] = exit_probability_drop_from_entry
    return item


def _open_once(tmp_path, strategies: list[dict], *, market_id: str = "m1", signal: float = 0.2):
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id=market_id,
        signal=signal,
        best_ask_yes=0.50,
        best_ask_no=0.40,
    )
    config_path = tmp_path / "intramarket.json"
    _write_config(config_path, strategies)
    report = run_intramarket_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(paper_db),
        run_id="intra_test",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )
    return recorder_db, paper_db, config_path, report


def _rows(db_path, table: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        conn.close()


def _insert_snapshot(
    db_path,
    *,
    market_id: str,
    timestamp,
    best_bid_yes: float = 0.49,
    best_ask_yes: float = 0.50,
    best_bid_no: float = 0.50,
    best_ask_no: float = 0.40,
) -> None:
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
                _iso(timestamp),
                market_id,
                best_bid_yes,
                best_ask_yes,
                best_bid_no,
                best_ask_no,
                0.02,
                0.02,
                0.50,
                0.50,
                0.51,
                5.0,
                _iso(timestamp),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _update_market_resolution(db_path, *, market_id: str, winning_outcome: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            UPDATE markets
            SET resolved = 1, winning_outcome = ?
            WHERE market_id = ?
            """,
            (winning_outcome, market_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_load_intramarket_strategy_config(tmp_path) -> None:
    config_path = tmp_path / "intramarket.json"
    _write_config(
        config_path,
        [
            _strategy("enabled"),
            {**_strategy("disabled"), "enabled": False},
        ],
    )

    strategies = load_intramarket_strategy_config(config_path)

    assert [strategy.strategy_id for strategy in strategies] == ["enabled"]
    assert strategies[0].exit_type == "FIXED_HORIZON_EXIT"


def test_cashout_intramarket_strategy_config_loads_without_hold_to_resolution(tmp_path) -> None:
    # Synthetic settings from the test contract, independent of private grids.
    config_path = tmp_path / "cashout.json"
    _write_config(config_path, [
        _strategy(f"synthetic_{side}_{index}", direction_mode=side,
                  exit_type="TAKE_PROFIT_STOP_LOSS", exit_after_sec=None,
                  take_profit_roi=0.05, stop_loss_roi=0.05, max_hold_seconds=30,
                  exit_probability_below=0.60, exit_probability_drop_from_entry=0.12)
        for side in ("YES_ONLY", "NO_ONLY") for index in range(4)
    ])
    strategies = load_intramarket_strategy_config(config_path)

    assert len(strategies) == 8
    assert {strategy.direction_mode for strategy in strategies} == {"YES_ONLY", "NO_ONLY"}
    assert {strategy.exit_type for strategy in strategies} == {"TAKE_PROFIT_STOP_LOSS"}
    assert all(strategy.take_profit_roi is not None for strategy in strategies)
    assert all(strategy.stop_loss_roi is not None for strategy in strategies)
    assert all(strategy.max_hold_seconds is not None for strategy in strategies)
    assert all(strategy.exit_probability_below == 0.60 for strategy in strategies)
    assert all(strategy.exit_probability_drop_from_entry == 0.12 for strategy in strategies)


def test_fixed_30s_exit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [_strategy("fixed_30", exit_type="FIXED_HORIZON_EXIT", exit_after_sec=30)],
    )
    _insert_snapshot(
        recorder_db,
        market_id="m1",
        timestamp=BASE + timedelta(seconds=151),
        best_bid_no=0.50,
    )
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        ensure_intramarket_schema(output)
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=151),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "fixed_horizon"
    assert position["adjusted_exit_price"] == 0.50
    assert position["realized_pnl_usd"] == 0.25


def test_take_profit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [_strategy("tp", exit_type="TAKE_PROFIT_STOP_LOSS", exit_after_sec=None, take_profit_cents=0.08)],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_no=0.49)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=130),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "take_profit"


def test_stop_loss_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "sl",
                exit_type="TAKE_PROFIT_STOP_LOSS",
                exit_after_sec=None,
                stop_loss_cents=0.04,
            )
        ],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_no=0.35)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=130),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "stop_loss"
    assert position["realized_pnl_usd"] < 0


def test_take_profit_roi_exit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "tp_roi",
                exit_type="TAKE_PROFIT_STOP_LOSS",
                exit_after_sec=None,
                take_profit_roi=0.05,
            )
        ],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_no=0.43)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=130),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "take_profit"
    assert position["adjusted_exit_price"] == 0.43
    assert position["realized_roi"] == 0.075


def test_stop_loss_roi_exit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "sl_roi",
                exit_type="TAKE_PROFIT_STOP_LOSS",
                exit_after_sec=None,
                stop_loss_roi=-0.05,
            )
        ],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_no=0.37)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=130),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "stop_loss"
    assert position["adjusted_exit_price"] == 0.37
    assert position["realized_roi"] == -0.075


def test_max_hold_seconds_exit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "max_hold_seconds",
                exit_type="TIME_OR_SIGNAL_EXIT",
                exit_after_sec=None,
                max_hold_seconds=20,
            )
        ],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=141), best_bid_no=0.41)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=141),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "max_hold"
    assert position["adjusted_exit_price"] == 0.41


def test_probability_invalidation_exit_closes_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "prob_exit",
                exit_type="TIME_OR_SIGNAL_EXIT",
                exit_after_sec=None,
                exit_probability_below=0.60,
                exit_probability_drop_from_entry=0.15,
            )
        ],
    )
    _insert_candidate(
        recorder_db,
        market_id="m1",
        signal=0.50,
        best_ask_no=0.40,
        feature_time=BASE + timedelta(seconds=130),
        time_until_resolution=170,
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=131), best_bid_no=0.39)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            model=ProbabilityBySignalModel(),
            feature_columns=["signal"],
            now=BASE + timedelta(seconds=131),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "closed"
    assert position["exit_reason"] == "probability_below"
    assert position["adjusted_exit_price"] == 0.39


def test_yes_and_no_exit_prices_use_current_bid_sides(tmp_path) -> None:
    yes_tmp = tmp_path / "yes"
    no_tmp = tmp_path / "no"
    yes_tmp.mkdir()
    no_tmp.mkdir()
    yes_recorder, yes_paper, yes_config, _report = _open_once(
        yes_tmp,
        [
            _strategy(
                "yes_tp",
                direction_mode="YES_ONLY",
                exit_type="TAKE_PROFIT_STOP_LOSS",
                exit_after_sec=None,
                take_profit_roi=0.05,
            )
        ],
        signal=0.8,
    )
    _insert_snapshot(yes_recorder, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_yes=0.60)
    no_recorder, no_paper, no_config, _report = _open_once(
        no_tmp,
        [
            _strategy(
                "no_tp",
                direction_mode="NO_ONLY",
                exit_type="TAKE_PROFIT_STOP_LOSS",
                exit_after_sec=None,
                take_profit_roi=0.05,
            )
        ],
        signal=0.2,
    )
    _insert_snapshot(no_recorder, market_id="m1", timestamp=BASE + timedelta(seconds=130), best_bid_no=0.50)

    for recorder_db, paper_db, config_path in (
        (yes_recorder, yes_paper, yes_config),
        (no_recorder, no_paper, no_config),
    ):
        strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
        recorder = sqlite3.connect(str(recorder_db))
        recorder.row_factory = sqlite3.Row
        output = sqlite3.connect(str(paper_db))
        output.row_factory = sqlite3.Row
        try:
            update_intramarket_positions(
                recorder,
                output,
                strategies_by_id=strategies,
                now=BASE + timedelta(seconds=130),
                emit_logs=False,
            )
        finally:
            recorder.close()
            output.close()

    yes_position = dict(_rows(yes_paper, "paper_positions")[0])
    no_position = dict(_rows(no_paper, "paper_positions")[0])
    assert yes_position["side"] == "YES"
    assert yes_position["adjusted_exit_price"] == 0.60
    assert no_position["side"] == "NO"
    assert no_position["adjusted_exit_price"] == 0.50


def test_hold_to_resolution_settles_correctly(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "hold_yes",
                direction_mode="YES_ONLY",
                exit_type="HOLD_TO_RESOLUTION",
                exit_after_sec=None,
            )
        ],
        signal=0.8,
    )
    _update_market_resolution(recorder_db, market_id="m1", winning_outcome="YES")
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=301),
            emit_logs=False,
        )
    finally:
        recorder.close()
        output.close()

    position = dict(_rows(paper_db, "paper_positions")[0])
    assert position["status"] == "settled"
    assert position["resolved_label"] == "YES"
    assert position["settlement_pnl_usd"] == 1.0


def test_max_open_positions_enforced(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(recorder_db, market_id="m1", signal=0.2, best_ask_no=0.4)
    _insert_candidate(recorder_db, market_id="m2", signal=0.2, best_ask_no=0.4)
    config_path = tmp_path / "intramarket.json"
    _write_config(
        config_path,
        [_strategy("max_one", max_open_positions_per_strategy=1)],
    )

    run_intramarket_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(paper_db),
        run_id="intra_test",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    positions = [dict(row) for row in _rows(paper_db, "paper_positions")]
    candidates = [dict(row) for row in _rows(paper_db, "paper_intramarket_candidates")]
    assert len(positions) == 1
    assert any(row["rejection_reason"] == "max_open_positions" for row in candidates)


def test_cooldown_enforced(tmp_path) -> None:
    recorder_db, paper_db, config_path, _report = _open_once(
        tmp_path,
        [
            _strategy(
                "cooldown",
                cooldown_after_exit_sec=15,
                allow_reentry_per_market=True,
            )
        ],
    )
    _insert_snapshot(recorder_db, market_id="m1", timestamp=BASE + timedelta(seconds=151), best_bid_no=0.50)
    strategies = {strategy.strategy_id: strategy for strategy in load_intramarket_strategy_config(config_path)}
    recorder = sqlite3.connect(str(recorder_db))
    recorder.row_factory = sqlite3.Row
    output = sqlite3.connect(str(paper_db))
    output.row_factory = sqlite3.Row
    try:
        update_intramarket_positions(
            recorder,
            output,
            strategies_by_id=strategies,
            now=BASE + timedelta(seconds=151),
            emit_logs=False,
        )
        _insert_candidate(
            recorder_db,
            market_id="m1",
            signal=0.2,
            best_ask_no=0.4,
            feature_time=BASE + timedelta(seconds=152),
            time_until_resolution=148,
        )
        report = run_intramarket_paper_trader_once(
            recorder,
            output,
            model=ProbabilityBySignalModel(),
            feature_columns=["signal"],
            strategies=list(strategies.values()),
            recorder_db_path=str(recorder_db),
            output_db_path=str(paper_db),
            model_path="model.joblib",
            feature_columns_path="features.json",
            run_id="intra_test",
            max_btc_age_sec=3.0,
            max_feature_age_sec=100000.0,
            emit_logs=False,
            now=BASE + timedelta(seconds=152),
        )
    finally:
        recorder.close()
        output.close()

    candidates = [dict(row) for row in _rows(paper_db, "paper_intramarket_candidates")]
    assert any(row["rejection_reason"] == "cooldown" for row in candidates)
    assert report["strategy_stats"][0]["skipped_cooldown"] == 1


def test_yes_and_no_use_correct_bid_ask_sides_and_candidates_are_complete(tmp_path) -> None:
    recorder_db, paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="both",
        signal=0.5,
        best_ask_yes=0.3,
        best_ask_no=0.4,
    )
    config_path = tmp_path / "intramarket.json"
    _write_config(
        config_path,
        [
            {
                **_strategy("both", direction_mode="BOTH"),
                "min_probability_for_direction": 0.4,
                "max_probability_for_direction": 0.9,
                "min_estimated_edge": 0.0,
            }
        ],
    )

    run_intramarket_paper_trader(
        recorder_db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        config_path=str(config_path),
        output_db_path=str(paper_db),
        run_id="intra_test",
        poll_sec=0.0,
        max_feature_age_sec=100000.0,
        max_iterations=1,
        emit_logs=False,
    )

    candidates = [dict(row) for row in _rows(paper_db, "paper_intramarket_candidates")]
    by_side = {row["candidate_direction"]: row for row in candidates}
    assert by_side["YES"]["candidate_entry_price"] == 0.3
    assert by_side["NO"]["candidate_entry_price"] == 0.4
    assert by_side["YES"]["probability_for_direction"] == 0.5
    assert by_side["NO"]["estimated_edge"] == 0.1


def test_export_intramarket_strategy_dataset(tmp_path) -> None:
    _recorder_db, paper_db, _config_path, _report = _open_once(
        tmp_path,
        [_strategy("export_me")],
    )

    report = export_intramarket_strategy_dataset(
        paper_db_path=str(paper_db),
        output_path=str(tmp_path / "intramarket.parquet"),
        output_csv_path=str(tmp_path / "intramarket.csv"),
        include_unresolved=True,
    )

    assert report["status"] == "ok"
    assert report["rows_exported"] >= 1
    assert (tmp_path / "intramarket.parquet").exists()
    assert (tmp_path / "intramarket.csv").exists()


def test_run_intramarket_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-intramarket-paper-trader",
            "--db",
            "data/recorder.db",
            "--model-path",
            "data/models/baseline_latest/model_logistic_regression.joblib",
            "--feature-columns",
            "data/models/baseline_latest/feature_columns.json",
            "--config",
            "data/strategy_configs/intramarket_strategy_grid.json",
            "--output-db",
            "data/paper_trades_intramarket.db",
        ],
    )

    args = main.parse_args()

    assert args.command == "run-intramarket-paper-trader"
    assert args.config == "data/strategy_configs/intramarket_strategy_grid.json"
    assert args.output_db == "data/paper_trades_intramarket.db"
