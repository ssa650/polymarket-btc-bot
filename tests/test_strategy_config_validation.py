from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

from src.strategy_config import (
    filter_strategy_config,
    generate_paper_strategy_config_sweep,
    validate_strategy_config,
)


EXPANDED_CONFIG = "data/strategy_configs/live_strategy_grid_expanded.json"
GUARDED_CONFIG = "data/strategy_configs/live_strategy_grid_guarded.json"
GUARDED_PROB_BAND_CONFIG = "data/strategy_configs/live_strategy_grid_guarded_prob_band.json"
YES_LATE_PROB_BAND_CONFIG = "data/strategy_configs/live_strategy_grid_yes_late_prob_band.json"
YES_NO_LATE_PROB_BAND_CONFIG = "data/strategy_configs/live_strategy_grid_yes_no_late_prob_band.json"
YES_NO_LATE_CASHOUT_CONFIG = "data/strategy_configs/live_strategy_grid_yes_no_late_cashout.json"
FOCUSED_NO_CASH_CONFIGS = [
    "data/strategy_configs/no_cash_fh15_t085_30_60_no_thin_bankroll_100.json",
    "data/strategy_configs/no_cash_fh15_t085_30_60_no_thin_bankroll_1000.json",
    "data/strategy_configs/no_cash_fh15_t087_30_60_no_thin_bankroll_100.json",
    "data/strategy_configs/no_cash_fh15_t087_30_60_no_thin_bankroll_1000.json",
    "data/strategy_configs/no_cash_fh15_t085_00_90_no_thin_bankroll_100.json",
    "data/strategy_configs/no_cash_fh15_t085_00_90_no_thin_bankroll_1000.json",
]


def _write_config(path, strategies: list[dict], **extra) -> None:
    path.write_text(json.dumps({"strategies": strategies, **extra}), encoding="utf-8")


def _valid_strategy(strategy_id: str = "s1") -> dict:
    return {
        "strategy_id": strategy_id,
        "enabled": True,
        "direction_mode": "BOTH",
        "long_threshold": 0.65,
        "short_threshold": 0.35,
        "min_estimated_edge": 0.02,
        "entry_slippage_cents": 0.01,
        "fee_cents": 0.0,
        "stake_usd": 1.0,
        "max_open_trades": 10,
        "one_trade_per_market": True,
        "min_time_until_resolution_sec": 30,
        "max_time_until_resolution_sec": 240,
    }


@pytest.mark.private_config
def test_expanded_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(EXPANDED_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 36
    assert report["enabled_strategy_count"] == 36
    assert report["errors"] == []
    assert "yes_t060_edge002_30_180" in report["strategy_ids"]
    assert "no_t040_edge002_30_180" in report["strategy_ids"]
    assert "both_t065_035_edge002_30_240" in report["strategy_ids"]


@pytest.mark.private_config
def test_guarded_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(GUARDED_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 27
    assert report["enabled_strategy_count"] == 27
    assert report["errors"] == []
    assert "yes_t070_edge030_30_240" in report["strategy_ids"]
    assert "no_t030_edge030_30_240" in report["strategy_ids"]
    assert "both_t070_030_edge030_30_240" in report["strategy_ids"]


@pytest.mark.private_config
def test_guarded_probability_band_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(GUARDED_PROB_BAND_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 27
    assert report["enabled_strategy_count"] == 27
    assert report["errors"] == []
    assert "yes_t070_edge030_30_240" in report["strategy_ids"]
    assert "no_t030_edge030_30_240" in report["strategy_ids"]
    assert "both_t070_030_edge030_30_240" in report["strategy_ids"]


@pytest.mark.private_config
def test_yes_late_probability_band_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(YES_LATE_PROB_BAND_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 54
    assert report["enabled_strategy_count"] == 54
    assert report["errors"] == []
    assert "yes_late_t070_edge030_00_060" in report["strategy_ids"]
    assert "yes_late_t080_edge080_30_090" in report["strategy_ids"]


@pytest.mark.private_config
def test_yes_late_probability_band_config_is_yes_only_guarded() -> None:
    payload = json.loads(open(YES_LATE_PROB_BAND_CONFIG, encoding="utf-8").read())
    strategies = payload["strategies"]

    assert strategies
    assert {strategy["direction_mode"] for strategy in strategies} == {"YES_ONLY"}
    assert all(strategy["short_threshold"] < 0 for strategy in strategies)
    assert all(strategy["require_positive_edge"] is True for strategy in strategies)
    assert all(strategy["min_probability_for_direction"] == 0.70 for strategy in strategies)
    assert all(strategy["max_probability_for_direction"] == 0.90 for strategy in strategies)
    assert all(strategy["max_probability_for_direction"] <= 0.90 for strategy in strategies)
    assert {strategy["long_threshold"] for strategy in strategies} == {0.70, 0.75, 0.80}
    assert {strategy["min_estimated_edge"] for strategy in strategies} == {0.03, 0.05, 0.08}
    assert {strategy["min_time_until_resolution_sec"] for strategy in strategies} == {0.0, 15.0, 30.0}
    assert {strategy["max_time_until_resolution_sec"] for strategy in strategies} == {60.0, 90.0}


@pytest.mark.private_config
def test_yes_no_late_probability_band_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(YES_NO_LATE_PROB_BAND_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 108
    assert report["enabled_strategy_count"] == 108
    assert report["errors"] == []
    assert "yes_late_t070_edge030_00_060" in report["strategy_ids"]
    assert "no_late_t070_edge030_00_060" in report["strategy_ids"]
    assert "yes_late_t080_edge080_30_090" in report["strategy_ids"]
    assert "no_late_t080_edge080_30_090" in report["strategy_ids"]


@pytest.mark.private_config
def test_yes_no_late_probability_band_config_keeps_yes_strategies_unchanged() -> None:
    yes_payload = json.loads(open(YES_LATE_PROB_BAND_CONFIG, encoding="utf-8").read())
    balanced_payload = json.loads(open(YES_NO_LATE_PROB_BAND_CONFIG, encoding="utf-8").read())

    yes_strategies = yes_payload["strategies"]
    balanced_strategies = balanced_payload["strategies"]

    assert balanced_strategies[: len(yes_strategies)] == yes_strategies


@pytest.mark.private_config
def test_yes_no_late_probability_band_config_contains_mirrored_no_strategies() -> None:
    payload = json.loads(open(YES_NO_LATE_PROB_BAND_CONFIG, encoding="utf-8").read())
    strategies = payload["strategies"]
    yes = [strategy for strategy in strategies if strategy["direction_mode"] == "YES_ONLY"]
    no = [strategy for strategy in strategies if strategy["direction_mode"] == "NO_ONLY"]

    assert len(yes) == 54
    assert len(no) == 54
    assert {strategy["direction_mode"] for strategy in strategies} == {"YES_ONLY", "NO_ONLY"}
    assert all(strategy["long_threshold"] > 1 for strategy in no)
    assert all(strategy["short_threshold"] in {0.30, 0.25, 0.20} for strategy in no)
    assert all(strategy["require_positive_edge"] is True for strategy in no)
    assert all(strategy["min_probability_for_direction"] == 0.70 for strategy in no)
    assert all(strategy["max_probability_for_direction"] == 0.90 for strategy in no)
    assert {strategy["min_estimated_edge"] for strategy in no} == {0.03, 0.05, 0.08}
    assert {strategy["min_time_until_resolution_sec"] for strategy in no} == {0.0, 15.0, 30.0}
    assert {strategy["max_time_until_resolution_sec"] for strategy in no} == {60.0, 90.0}


@pytest.mark.private_config
def test_yes_no_late_cashout_strategy_config_validates_successfully() -> None:
    report = validate_strategy_config(YES_NO_LATE_CASHOUT_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 36
    assert report["enabled_strategy_count"] == 36
    assert report["errors"] == []
    assert "yes_cash_tp05_sl05_t070_edge050_00_090" in report["strategy_ids"]
    assert "no_cash_fh45_t080_edge050_00_090" in report["strategy_ids"]


@pytest.mark.private_config
def test_yes_no_late_cashout_config_is_balanced_and_uses_exit_rules() -> None:
    payload = json.loads(open(YES_NO_LATE_CASHOUT_CONFIG, encoding="utf-8").read())
    strategies = payload["strategies"]

    assert len([strategy for strategy in strategies if strategy["direction_mode"] == "YES_ONLY"]) == 18
    assert len([strategy for strategy in strategies if strategy["direction_mode"] == "NO_ONLY"]) == 18
    assert {strategy["exit_type"] for strategy in strategies} == {
        "FIXED_HORIZON_EXIT",
        "TAKE_PROFIT_STOP_LOSS",
    }
    assert all(strategy["require_positive_edge"] is True for strategy in strategies)
    assert all(strategy["min_probability_for_direction"] == 0.70 for strategy in strategies)
    assert all(strategy["max_probability_for_direction"] == 0.90 for strategy in strategies)
    assert all(strategy["one_trade_per_market"] is True for strategy in strategies)


@pytest.mark.private_config
def test_focused_no_cash_configs_validate_successfully() -> None:
    for config_path in FOCUSED_NO_CASH_CONFIGS:
        report = validate_strategy_config(config_path)

        assert report["status"] == "ok", (config_path, report["errors"])
        assert report["strategy_count"] == 1
        assert report["enabled_strategy_count"] == 1
        assert report["errors"] == []


@pytest.mark.private_config
def test_focused_no_cash_configs_are_no_only_fixed_horizon_and_no_thin() -> None:
    for config_path in FOCUSED_NO_CASH_CONFIGS:
        payload = json.loads(open(config_path, encoding="utf-8").read())
        strategy = payload["strategies"][0]

        assert payload["bankroll"]["bankroll_enabled"] is True
        assert payload["realistic_execution"]["realistic_execution_enabled"] is True
        assert payload["liquidity"]["enabled"] is True
        assert payload["trade_selection"]["selection_enabled"] is True
        assert payload["live_trading"]["live_trading_enabled"] is False
        assert strategy["direction_mode"] == "NO_ONLY"
        assert strategy["long_threshold"] > 1
        assert strategy["require_positive_edge"] is True
        assert strategy["exit_type"] == "FIXED_HORIZON_EXIT"
        assert strategy["fixed_horizon_exit_sec"] == 15
        assert strategy["blocked_liquidity_regimes"] == ["thin"]
        assert strategy["min_estimated_edge"] == 0.05
        assert strategy["max_probability_for_direction"] == 0.90


def test_blocked_liquidity_regimes_validation_rejects_invalid_values(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_liquidity_regime")
    strategy["blocked_liquidity_regimes"] = ["thin", "watery"]
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("blocked_liquidity_regimes" in error for error in report["errors"])


def test_require_positive_edge_must_be_boolean(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_edge_flag")
    strategy["require_positive_edge"] = "yes"
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("require_positive_edge" in error for error in report["errors"])


def test_probability_band_min_must_be_between_zero_and_one(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_min_prob")
    strategy["min_probability_for_direction"] = -0.1
    strategy["max_probability_for_direction"] = 0.9
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("min_probability_for_direction" in error for error in report["errors"])


def test_probability_band_max_must_be_between_zero_and_one(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_max_prob")
    strategy["min_probability_for_direction"] = 0.7
    strategy["max_probability_for_direction"] = 1.1
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("max_probability_for_direction" in error for error in report["errors"])


def test_probability_band_min_must_not_exceed_max(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_prob_range")
    strategy["min_probability_for_direction"] = 0.9
    strategy["max_probability_for_direction"] = 0.7
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("min_probability_for_direction must be <=" in error for error in report["errors"])


def test_bankroll_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("bankroll_ok")],
        bankroll={
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
            "base_risk_fraction": 0.01,
            "max_risk_fraction": 0.05,
            "max_total_exposure_fraction": 0.20,
            "min_stake_usd": 0.25,
            "max_stake_usd": 5.0,
            "stop_trading_on_bankroll_depleted": True,
            "reset_bankroll_on_new_run_id": True,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_bankroll_invalid_range_fails_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("bankroll_bad")],
        bankroll={
            "bankroll_enabled": True,
            "base_risk_fraction": 0.10,
            "max_risk_fraction": 0.05,
            "min_stake_usd": 5.0,
            "max_stake_usd": 1.0,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "max_stake_usd must be >= min_stake_usd" in report["errors"]
    assert "max_risk_fraction must be >= base_risk_fraction" in report["errors"]


def test_liquidity_fill_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("liquidity_ok")
    strategy.update(
        {
            "liquidity_fill_check_enabled": True,
            "max_top_of_book_fill_fraction": 0.5,
            "min_top_of_book_shares": 2.0,
            "skip_if_liquidity_missing": True,
        }
    )
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_global_liquidity_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("global_liquidity_ok")],
        liquidity={
            "enabled": True,
            "max_top_of_book_fill_fraction": 0.5,
            "min_top_of_book_shares": 2.0,
            "skip_if_liquidity_missing": True,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_liquidity_fill_invalid_values_fail_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("liquidity_bad")
    strategy.update(
        {
            "liquidity_fill_check_enabled": "yes",
            "max_top_of_book_fill_fraction": 0.0,
            "min_top_of_book_shares": -1.0,
            "skip_if_liquidity_missing": "no",
        }
    )
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "liquidity_bad.liquidity_fill_check_enabled must be boolean" in report["errors"]
    assert "liquidity_bad.max_top_of_book_fill_fraction must be > 0" in report["errors"]
    assert "liquidity_bad.min_top_of_book_shares must be >= 0" in report["errors"]
    assert "liquidity_bad.skip_if_liquidity_missing must be boolean" in report["errors"]


def test_global_liquidity_invalid_values_fail_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("global_liquidity_bad")],
        liquidity={
            "enabled": "yes",
            "max_top_of_book_fill_fraction": 0.0,
            "min_top_of_book_shares": -1.0,
            "skip_if_liquidity_missing": "no",
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "liquidity_fill_check_enabled must be boolean" in report["errors"]
    assert "max_top_of_book_fill_fraction must be > 0" in report["errors"]
    assert "min_top_of_book_shares must be >= 0" in report["errors"]
    assert "skip_if_liquidity_missing must be boolean" in report["errors"]


def test_realistic_execution_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("realistic_ok")],
        realistic_execution={
            "realistic_execution_enabled": True,
            "entry_latency_sec": 1.0,
            "exit_latency_sec": 1.0,
            "max_quote_wait_sec": 3.0,
            "max_entry_price_drift_cents": 2.0,
            "max_exit_price_drift_cents": 2.0,
            "require_quote_after_latency": True,
            "reject_if_spread_above_cents": 4.0,
            "reject_if_btc_age_above_sec": 5.0,
            "reject_if_feature_age_above_sec": 5.0,
            "apply_extra_slippage_cents": 1.0,
            "partial_fill_enabled": False,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_realistic_execution_invalid_values_fail_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("realistic_bad")],
        realistic_execution={
            "realistic_execution_enabled": "yes",
            "entry_latency_sec": -1.0,
            "exit_latency_sec": "slow",
            "max_quote_wait_sec": -0.1,
            "require_quote_after_latency": "true",
            "partial_fill_enabled": "false",
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "realistic_execution_enabled must be boolean" in report["errors"]
    assert "entry_latency_sec must be >= 0" in report["errors"]
    assert "exit_latency_sec must be numeric" in report["errors"]
    assert "max_quote_wait_sec must be >= 0" in report["errors"]
    assert "require_quote_after_latency must be boolean" in report["errors"]
    assert "partial_fill_enabled must be boolean" in report["errors"]


def test_trade_selection_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("select_ok")],
        trade_selection={
            "selection_enabled": True,
            "mode": "best_per_market_direction",
            "score_field": "estimated_edge",
            "max_new_trades_per_market": 1,
            "max_new_trades_per_market_direction": 1,
            "allow_multiple_strategy_variants_same_market": False,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_trade_selection_invalid_values_fail_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("select_bad")],
        trade_selection={
            "selection_enabled": "yes",
            "mode": "bad",
            "score_field": "",
            "max_new_trades_per_market": 0,
            "max_new_trades_per_market_direction": 1.5,
            "allow_multiple_strategy_variants_same_market": "no",
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "trade_selection.selection_enabled must be boolean" in report["errors"]
    assert "trade_selection.mode must be best_per_market_direction" in report["errors"]
    assert "trade_selection.score_field must be non-empty" in report["errors"]
    assert "trade_selection.max_new_trades_per_market must be an integer >= 1" in report["errors"]
    assert (
        "trade_selection.max_new_trades_per_market_direction must be an integer >= 1"
        in report["errors"]
    )
    assert (
        "trade_selection.allow_multiple_strategy_variants_same_market must be boolean"
        in report["errors"]
    )


def test_live_trading_config_validates_successfully(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("live_ok")],
        live_trading={
            "live_trading_enabled": False,
            "dry_run_orders": True,
            "max_order_usd": 1.0,
            "max_daily_loss_usd": 5.0,
            "max_daily_orders": 20,
            "max_open_exposure_usd": 5.0,
            "max_open_trades": 3,
            "max_trades_per_market": 1,
            "max_trades_per_market_direction": 1,
            "require_realistic_execution_passed": True,
            "require_liquidity_check_passed": True,
            "reject_if_btc_age_above_sec": 5.0,
            "reject_if_feature_age_above_sec": 5.0,
            "reject_if_spread_above_cents": 3.0,
            "kill_switch_file": "data/KILL_LIVE_TRADING",
            "allow_market_order": False,
            "use_limit_orders_only": True,
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "ok"
    assert report["errors"] == []


def test_live_trading_invalid_values_fail_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        [_valid_strategy("live_bad")],
        live_trading={
            "live_trading_enabled": "yes",
            "dry_run_orders": "true",
            "max_order_usd": -1.0,
            "max_daily_loss_usd": "bad",
            "max_daily_orders": 0,
            "max_open_exposure_usd": -0.1,
            "max_open_trades": 1.5,
            "max_trades_per_market": 0,
            "max_trades_per_market_direction": "one",
            "require_realistic_execution_passed": "yes",
            "require_liquidity_check_passed": "yes",
            "reject_if_btc_age_above_sec": -1.0,
            "reject_if_feature_age_above_sec": "stale",
            "reject_if_spread_above_cents": -0.5,
            "kill_switch_file": "",
            "allow_market_order": "no",
            "use_limit_orders_only": "yes",
        },
    )

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "live_trading.live_trading_enabled must be boolean" in report["errors"]
    assert "live_trading.dry_run_orders must be boolean" in report["errors"]
    assert "live_trading.max_order_usd must be >= 0" in report["errors"]
    assert "live_trading.max_daily_loss_usd must be numeric" in report["errors"]
    assert "live_trading.max_daily_orders must be an integer >= 1" in report["errors"]
    assert "live_trading.max_open_exposure_usd must be >= 0" in report["errors"]
    assert "live_trading.max_open_trades must be an integer >= 1" in report["errors"]
    assert "live_trading.max_trades_per_market must be an integer >= 1" in report["errors"]
    assert (
        "live_trading.max_trades_per_market_direction must be numeric"
        in report["errors"]
    )
    assert (
        "live_trading.require_realistic_execution_passed must be boolean"
        in report["errors"]
    )
    assert "live_trading.require_liquidity_check_passed must be boolean" in report["errors"]
    assert "live_trading.reject_if_btc_age_above_sec must be >= 0" in report["errors"]
    assert (
        "live_trading.reject_if_feature_age_above_sec must be numeric"
        in report["errors"]
    )
    assert "live_trading.reject_if_spread_above_cents must be >= 0" in report["errors"]
    assert "live_trading.kill_switch_file must be non-empty" in report["errors"]
    assert "live_trading.allow_market_order must be boolean" in report["errors"]
    assert "live_trading.use_limit_orders_only must be boolean" in report["errors"]


def test_filter_strategy_config_keeps_only_matching_strategies(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "filtered.json"
    keep = _valid_strategy("no_cash_fh15_t070_edge050_00_090")
    drop = _valid_strategy("no_cash_fh15_t075_edge050_00_090")
    _write_config(
        source,
        [keep, drop],
        trade_selection={
            "selection_enabled": True,
            "mode": "best_per_market_direction",
        },
        bankroll={
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
        },
    )

    report = filter_strategy_config(
        source,
        output,
        strategy_id_substrings=["no_cash_fh15_t070"],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert report["output_strategy_count"] == 1
    assert [strategy["strategy_id"] for strategy in payload["strategies"]] == [
        "no_cash_fh15_t070_edge050_00_090"
    ]


def test_filter_strategy_config_preserves_trade_selection_and_bankroll(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "filtered.json"
    _write_config(
        source,
        [
            _valid_strategy("no_cash_fh15_t070_edge050_00_090"),
            _valid_strategy("yes_cash_fh15_t070_edge050_00_090"),
        ],
        realistic_execution={
            "realistic_execution_enabled": True,
            "entry_latency_sec": 1.0,
        },
        liquidity={
            "liquidity_fill_check_enabled": True,
            "skip_if_liquidity_missing": True,
        },
        trade_selection={
            "selection_enabled": True,
            "mode": "best_per_market_direction",
            "max_new_trades_per_market": 1,
        },
        bankroll={
            "bankroll_enabled": True,
            "starting_bankroll_usd": 1000.0,
            "base_risk_fraction": 0.01,
        },
    )

    report = filter_strategy_config(
        source,
        output,
        strategy_id_substrings=["no_cash_fh15_t070"],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert payload["realistic_execution"]["realistic_execution_enabled"] is True
    assert payload["liquidity"]["liquidity_fill_check_enabled"] is True
    assert payload["trade_selection"]["selection_enabled"] is True
    assert payload["bankroll"]["starting_bankroll_usd"] == 1000.0


def test_filter_strategy_config_output_validates(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "filtered.json"
    _write_config(
        source,
        [
            _valid_strategy("no_cash_fh15_t070_edge050_00_090"),
            _valid_strategy("no_cash_fh15_t075_edge050_00_090"),
        ],
    )

    filter_strategy_config(source, output, strategy_id_substrings=["no_cash_fh15_t070"])
    report = validate_strategy_config(output)

    assert report["status"] == "ok"
    assert report["strategy_ids"] == ["no_cash_fh15_t070_edge050_00_090"]


def _write_sweep_base_config(path: Path) -> None:
    payload = {
        "bankroll": {
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
            "base_risk_fraction": 0.01,
            "max_risk_fraction": 0.05,
            "max_total_exposure_fraction": 0.2,
            "min_stake_usd": 0.25,
            "max_stake_usd": 5.0,
            "stop_trading_on_bankroll_depleted": True,
            "reset_bankroll_on_new_run_id": True,
        },
        "realistic_execution": {
            "realistic_execution_enabled": True,
            "entry_latency_sec": 1.0,
            "exit_latency_sec": 1.0,
            "max_quote_wait_sec": 3.0,
            "max_entry_price_drift_cents": 2.0,
            "max_exit_price_drift_cents": 2.0,
            "require_quote_after_latency": True,
            "reject_if_spread_above_cents": 4.0,
            "reject_if_btc_age_above_sec": 5.0,
            "reject_if_feature_age_above_sec": 5.0,
            "apply_extra_slippage_cents": 1.0,
            "partial_fill_enabled": False,
        },
        "liquidity": {
            "enabled": True,
            "max_top_of_book_fill_fraction": 1.0,
            "min_top_of_book_shares": 0.0,
            "skip_if_liquidity_missing": True,
        },
        "trade_selection": {
            "selection_enabled": True,
            "mode": "best_per_market_direction",
            "score_field": "estimated_edge",
            "max_new_trades_per_market": 1,
            "max_new_trades_per_market_direction": 1,
            "allow_multiple_strategy_variants_same_market": False,
        },
        "live_trading": {
            "live_trading_enabled": False,
            "dry_run_orders": True,
        },
        "strategies": [
            {
                "strategy_id": "no_cash_fh15_t085_00_90_edge050_no_thin",
                "strategy_name": "NO cashout fh15 p0.85 edge 0.05",
                "enabled": True,
                "direction_mode": "NO_ONLY",
                "long_threshold": 2.0,
                "short_threshold": 0.15,
                "min_estimated_edge": 0.05,
                "entry_slippage_cents": 0.01,
                "exit_slippage_cents": 0.01,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 3,
                "one_trade_per_market": True,
                "require_positive_edge": True,
                "min_probability_for_direction": 0.85,
                "max_probability_for_direction": 0.9,
                "min_time_until_resolution_sec": 0.0,
                "max_time_until_resolution_sec": 90.0,
                "exit_type": "FIXED_HORIZON_EXIT",
                "fixed_horizon_exit_sec": 15,
                "blocked_liquidity_regimes": ["thin"],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_paper_strategy_config_sweep_count_and_validation(tmp_path) -> None:
    base = tmp_path / "base.json"
    output_dir = tmp_path / "generated"
    _write_sweep_base_config(base)

    report = generate_paper_strategy_config_sweep(
        base_config_path=base,
        output_dir=output_dir,
        name_prefix="no_cash_sweep",
        thresholds="0.80,0.85",
        max_probabilities="none,0.90",
        time_windows="0:90,120:240",
        bankrolls="100,1000",
        blocked_liquidity_regimes="thin",
        fixed_horizon_sec=15,
    )

    assert report["status"] == "ok"
    assert report["generated_config_count"] == 2
    assert report["generated_strategy_count"] == 16
    for item in report["generated_configs"]:
        config_path = item["path"]
        validation = validate_strategy_config(config_path)
        assert validation["status"] == "ok"
        assert validation["strategy_count"] == 8
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert payload["live_trading"]["live_trading_enabled"] is False
        assert payload["trade_selection"]["selection_enabled"] is True
        assert payload["realistic_execution"]["realistic_execution_enabled"] is True
        assert payload["liquidity"]["enabled"] is True


def test_generate_paper_strategy_config_sweep_none_max_probability_and_time_window_encoding(
    tmp_path,
) -> None:
    base = tmp_path / "base.json"
    output_dir = tmp_path / "generated"
    _write_sweep_base_config(base)

    report = generate_paper_strategy_config_sweep(
        base_config_path=base,
        output_dir=output_dir,
        name_prefix="no_cash_sweep",
        thresholds="0.80",
        max_probabilities="none,0.95",
        time_windows="120:240",
        bankrolls="100",
        blocked_liquidity_regimes="thin",
        fixed_horizon_sec=15,
    )

    payload = json.loads(
        Path(report["generated_configs"][0]["path"]).read_text(encoding="utf-8")
    )
    strategies = payload["strategies"]
    assert len(strategies) == 2
    no_max = next(strategy for strategy in strategies if "pmaxnone" in strategy["strategy_id"])
    capped = next(strategy for strategy in strategies if "pmax095" in strategy["strategy_id"])
    assert "max_probability_for_direction" not in no_max
    assert capped["max_probability_for_direction"] == 0.95
    assert all("_120_240_" in strategy["strategy_id"] for strategy in strategies)
    assert all("_br100_" in strategy["strategy_id"] for strategy in strategies)
    assert all(strategy["short_threshold"] == 0.2 for strategy in strategies)
    assert all(strategy["min_probability_for_direction"] == 0.8 for strategy in strategies)


def test_generate_paper_strategy_config_sweep_refuses_live_trading(tmp_path) -> None:
    base = tmp_path / "base.json"
    _write_sweep_base_config(base)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["live_trading"]["live_trading_enabled"] = True
    base.write_text(json.dumps(payload), encoding="utf-8")

    try:
        generate_paper_strategy_config_sweep(
            base_config_path=base,
            output_dir=tmp_path / "generated",
            name_prefix="no_cash_sweep",
            thresholds="0.80",
            max_probabilities="none",
            time_windows="0:90",
            bankrolls="100",
            blocked_liquidity_regimes="thin",
            fixed_horizon_sec=15,
        )
    except ValueError as exc:
        assert "live_trading_enabled=true" in str(exc)
    else:
        raise AssertionError("expected live-trading config to be refused")


def test_duplicate_strategy_id_fails_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, [_valid_strategy("dup"), _valid_strategy("dup")])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert "duplicate strategy_id: dup" in report["errors"]


def test_invalid_direction_mode_fails_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_mode")
    strategy["direction_mode"] = "MAYBE"
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("direction_mode" in error for error in report["errors"])


def test_invalid_time_window_fails_validation(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    strategy = _valid_strategy("bad_window")
    strategy["min_time_until_resolution_sec"] = 240
    strategy["max_time_until_resolution_sec"] = 30
    _write_config(config_path, [strategy])

    report = validate_strategy_config(config_path)

    assert report["status"] == "error"
    assert any("max_time_until_resolution_sec" in error for error in report["errors"])


def test_validate_strategy_config_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "validate-strategy-config",
            "--config",
            EXPANDED_CONFIG,
        ],
    )

    args = main.parse_args()

    assert args.command == "validate-strategy-config"
    assert args.config == EXPANDED_CONFIG


def test_filter_strategy_config_cli_is_registered(monkeypatch, tmp_path) -> None:
    from src import main

    output = tmp_path / "filtered.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "filter-strategy-config",
            "--config",
            EXPANDED_CONFIG,
            "--output",
            str(output),
            "--strategy-id-contains",
            "no_cash_fh15_t070",
        ],
    )

    args = main.parse_args()

    assert args.command == "filter-strategy-config"
    assert args.config == EXPANDED_CONFIG
    assert args.output == str(output)
    assert args.strategy_id_contains == ["no_cash_fh15_t070"]


def test_generate_paper_strategy_config_sweep_cli_is_registered(monkeypatch, tmp_path) -> None:
    from src import main

    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "generate-paper-strategy-config-sweep",
            "--base-config",
            FOCUSED_NO_CASH_CONFIGS[0],
            "--output-dir",
            str(output),
            "--name-prefix",
            "no_cash_sweep",
            "--thresholds",
            "0.80,0.85",
            "--max-probabilities",
            "none,0.90",
            "--time-windows",
            "0:90,120:240",
            "--bankrolls",
            "100,1000",
            "--blocked-liquidity-regimes",
            "thin",
            "--fixed-horizon-sec",
            "15",
        ],
    )

    args = main.parse_args()

    assert args.command == "generate-paper-strategy-config-sweep"
    assert args.base_config == FOCUSED_NO_CASH_CONFIGS[0]
    assert args.output_dir == str(output)
    assert args.name_prefix == "no_cash_sweep"
    assert args.thresholds == "0.80,0.85"
    assert args.max_probabilities == "none,0.90"
    assert args.time_windows == "0:90,120:240"
    assert args.bankrolls == "100,1000"
    assert args.blocked_liquidity_regimes == "thin"
    assert args.fixed_horizon_sec == 15.0
