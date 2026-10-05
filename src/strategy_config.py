from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DIRECTION_MODES = {"YES_ONLY", "NO_ONLY", "BOTH", "FADE_YES_WITH_NO"}
EXIT_TYPES = {
    "HOLD_TO_RESOLUTION",
    "FIXED_HORIZON_EXIT",
    "TAKE_PROFIT_STOP_LOSS",
    "TRAILING_STOP",
}
LIQUIDITY_REGIMES = {"thin", "normal", "deep", "unknown"}


def validate_strategy_config(config_path: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "error",
            "strategy_count": 0,
            "enabled_strategy_count": 0,
            "errors": [f"json_parse_error: {exc}"],
            "warnings": [],
            "strategy_ids": [],
        }
    strategies = _raw_strategies(payload, errors)
    _validate_bankroll_config(payload, errors)
    _validate_realistic_execution_config(payload, errors)
    _validate_trade_selection_config(payload, errors)
    _validate_live_trading_config(payload, errors)
    _validate_global_liquidity_config(payload, errors)
    strategy_ids: list[str] = []
    seen: set[str] = set()
    enabled_count = 0
    for index, strategy in enumerate(strategies):
        if not isinstance(strategy, dict):
            errors.append(f"strategies[{index}] must be an object")
            continue
        strategy_id = str(strategy.get("strategy_id") or "").strip()
        if not strategy_id:
            errors.append(f"strategies[{index}].strategy_id is required")
        elif strategy_id in seen:
            errors.append(f"duplicate strategy_id: {strategy_id}")
        else:
            seen.add(strategy_id)
            strategy_ids.append(strategy_id)
        enabled = strategy.get("enabled", True)
        if not isinstance(enabled, bool):
            errors.append(f"{strategy_id or f'strategies[{index}]'}.enabled must be boolean")
        elif enabled:
            enabled_count += 1
        mode = strategy.get("direction_mode")
        if mode not in DIRECTION_MODES:
            errors.append(
                f"{strategy_id or f'strategies[{index}]'}.direction_mode must be one of {sorted(DIRECTION_MODES)}"
            )
        if mode == "FADE_YES_WITH_NO":
            _numeric_field(strategy, "long_threshold", strategy_id, index, errors)
            _numeric_field(strategy, "short_threshold", strategy_id, index, errors)
        else:
            _validate_threshold(
                strategy,
                "long_threshold",
                strategy_id=strategy_id,
                index=index,
                errors=errors,
                warnings=warnings,
                disabled_side="high",
            )
            _validate_threshold(
                strategy,
                "short_threshold",
                strategy_id=strategy_id,
                index=index,
                errors=errors,
                warnings=warnings,
                disabled_side="low",
            )
        edge = _numeric_field(strategy, "min_estimated_edge", strategy_id, index, errors)
        if edge is not None and edge < 0:
            errors.append(f"{_label(strategy_id, index)}.min_estimated_edge must be >= 0")
        min_t = _numeric_field(
            strategy,
            "min_time_until_resolution_sec",
            strategy_id,
            index,
            errors,
        )
        max_t = _numeric_field(
            strategy,
            "max_time_until_resolution_sec",
            strategy_id,
            index,
            errors,
        )
        if min_t is not None and min_t < 0:
            errors.append(
                f"{_label(strategy_id, index)}.min_time_until_resolution_sec must be >= 0"
            )
        if min_t is not None and max_t is not None and max_t <= min_t:
            errors.append(
                f"{_label(strategy_id, index)}.max_time_until_resolution_sec must be > min_time_until_resolution_sec"
            )
        max_open = _numeric_field(strategy, "max_open_trades", strategy_id, index, errors)
        if max_open is not None and max_open < 1:
            errors.append(f"{_label(strategy_id, index)}.max_open_trades must be >= 1")
        stake = _numeric_field(strategy, "stake_usd", strategy_id, index, errors)
        if stake is not None and stake <= 0:
            errors.append(f"{_label(strategy_id, index)}.stake_usd must be > 0")
        if "one_trade_per_market" in strategy and not isinstance(strategy["one_trade_per_market"], bool):
            errors.append(
                f"{_label(strategy_id, index)}.one_trade_per_market must be boolean"
            )
        if "require_positive_edge" in strategy and not isinstance(strategy["require_positive_edge"], bool):
            errors.append(
                f"{_label(strategy_id, index)}.require_positive_edge must be boolean"
            )
        if (
            "liquidity_fill_check_enabled" in strategy
            and not isinstance(strategy["liquidity_fill_check_enabled"], bool)
        ):
            errors.append(
                f"{_label(strategy_id, index)}.liquidity_fill_check_enabled must be boolean"
            )
        if (
            "skip_if_liquidity_missing" in strategy
            and not isinstance(strategy["skip_if_liquidity_missing"], bool)
        ):
            errors.append(
                f"{_label(strategy_id, index)}.skip_if_liquidity_missing must be boolean"
            )
        blocked_liquidity = strategy.get("blocked_liquidity_regimes")
        if blocked_liquidity is not None:
            if not isinstance(blocked_liquidity, list):
                errors.append(
                    f"{_label(strategy_id, index)}.blocked_liquidity_regimes must be a list"
                )
            else:
                for regime in blocked_liquidity:
                    value = str(regime).strip().lower()
                    if value not in LIQUIDITY_REGIMES:
                        errors.append(
                            f"{_label(strategy_id, index)}.blocked_liquidity_regimes contains invalid regime {regime!r}"
                        )
        max_fill_fraction = _optional_numeric_field(
            strategy,
            "max_top_of_book_fill_fraction",
            strategy_id,
            index,
            errors,
        )
        if max_fill_fraction is not None and max_fill_fraction <= 0:
            errors.append(
                f"{_label(strategy_id, index)}.max_top_of_book_fill_fraction must be > 0"
            )
        min_top_shares = _optional_numeric_field(
            strategy,
            "min_top_of_book_shares",
            strategy_id,
            index,
            errors,
        )
        if min_top_shares is not None and min_top_shares < 0:
            errors.append(
                f"{_label(strategy_id, index)}.min_top_of_book_shares must be >= 0"
            )
        exit_type = str(strategy.get("exit_type") or "HOLD_TO_RESOLUTION").upper()
        if exit_type not in EXIT_TYPES:
            errors.append(
                f"{_label(strategy_id, index)}.exit_type must be one of {sorted(EXIT_TYPES)}"
            )
        for optional_exit_field in (
            "exit_slippage_cents",
            "take_profit_pct",
            "stop_loss_pct",
            "fixed_horizon_exit_sec",
            "trailing_stop_pct",
        ):
            value = strategy.get(optional_exit_field)
            if value is not None:
                numeric = _numeric_field(
                    strategy,
                    optional_exit_field,
                    strategy_id,
                    index,
                    errors,
                )
                if optional_exit_field in {"exit_slippage_cents", "fixed_horizon_exit_sec"}:
                    if numeric is not None and numeric < 0:
                        errors.append(
                            f"{_label(strategy_id, index)}.{optional_exit_field} must be >= 0"
                        )
                elif optional_exit_field in {"take_profit_pct", "trailing_stop_pct"}:
                    if numeric is not None and numeric <= 0:
                        errors.append(
                            f"{_label(strategy_id, index)}.{optional_exit_field} must be > 0"
                        )
                elif optional_exit_field == "stop_loss_pct" and numeric is not None and numeric <= 0:
                    errors.append(
                        f"{_label(strategy_id, index)}.stop_loss_pct must be > 0"
                    )
        for optional_probability in ("block_probability_above", "block_probability_below"):
            value = strategy.get(optional_probability)
            if value is None:
                continue
            numeric = _numeric_field(strategy, optional_probability, strategy_id, index, errors)
            if numeric is not None and not 0 <= numeric <= 1:
                warnings.append(
                    f"{_label(strategy_id, index)}.{optional_probability} outside [0, 1]"
                )
        min_probability = _optional_probability_field(
            strategy,
            "min_probability_for_direction",
            strategy_id,
            index,
            errors,
        )
        max_probability = _optional_probability_field(
            strategy,
            "max_probability_for_direction",
            strategy_id,
            index,
            errors,
        )
        if (
            min_probability is not None
            and max_probability is not None
            and min_probability > max_probability
        ):
            errors.append(
                f"{_label(strategy_id, index)}.min_probability_for_direction must be <= max_probability_for_direction"
            )
        fade_yes_threshold = _optional_probability_field(
            strategy,
            "fade_yes_threshold",
            strategy_id,
            index,
            errors,
        )
        if mode == "FADE_YES_WITH_NO" and fade_yes_threshold is None:
            errors.append(
                f"{_label(strategy_id, index)}.fade_yes_threshold is required for FADE_YES_WITH_NO"
            )
    return {
        "status": "ok" if not errors else "error",
        "strategy_count": len(strategies),
        "enabled_strategy_count": enabled_count,
        "errors": errors,
        "warnings": warnings,
        "strategy_ids": strategy_ids,
    }


def summarize_strategy_config(config_path: str | Path) -> dict[str, Any]:
    validation = validate_strategy_config(config_path)
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {
            **validation,
            "count_by_direction_mode": {},
            "threshold_ranges": {},
            "edge_ranges": {},
            "time_window_ranges": {},
            "max_open_trades_range": {},
            "sample_strategy_ids": [],
        }
    strategies = [
        strategy
        for strategy in _raw_strategies(payload, [])
        if isinstance(strategy, dict)
    ]
    return {
        **validation,
        "count_by_direction_mode": dict(
            Counter(str(strategy.get("direction_mode") or "") for strategy in strategies)
        ),
        "threshold_ranges": {
            "long_threshold": _range(_numbers(strategies, "long_threshold")),
            "short_threshold": _range(_numbers(strategies, "short_threshold")),
        },
        "edge_ranges": {
            "min_estimated_edge": _range(_numbers(strategies, "min_estimated_edge")),
        },
        "probability_band_ranges": {
            "min_probability_for_direction": _range(
                _numbers(strategies, "min_probability_for_direction")
            ),
            "max_probability_for_direction": _range(
                _numbers(strategies, "max_probability_for_direction")
            ),
        },
        "exit_type_counts": dict(
            Counter(str(strategy.get("exit_type") or "HOLD_TO_RESOLUTION").upper() for strategy in strategies)
        ),
        "exit_ranges": {
            "exit_slippage_cents": _range(_numbers(strategies, "exit_slippage_cents")),
            "take_profit_pct": _range(_numbers(strategies, "take_profit_pct")),
            "stop_loss_pct": _range(_numbers(strategies, "stop_loss_pct")),
            "fixed_horizon_exit_sec": _range(_numbers(strategies, "fixed_horizon_exit_sec")),
            "trailing_stop_pct": _range(_numbers(strategies, "trailing_stop_pct")),
        },
        "liquidity_fill_ranges": {
            "max_top_of_book_fill_fraction": _range(
                _numbers(strategies, "max_top_of_book_fill_fraction")
            ),
            "min_top_of_book_shares": _range(
                _numbers(strategies, "min_top_of_book_shares")
            ),
        },
        "liquidity": _global_liquidity_summary(payload),
        "realistic_execution": _realistic_execution_summary(payload),
        "trade_selection": _trade_selection_summary(payload),
        "live_trading": _live_trading_summary(payload),
        "time_window_ranges": {
            "min_time_until_resolution_sec": _range(
                _numbers(strategies, "min_time_until_resolution_sec")
            ),
            "max_time_until_resolution_sec": _range(
                _numbers(strategies, "max_time_until_resolution_sec")
            ),
        },
        "max_open_trades_range": _range(_numbers(strategies, "max_open_trades")),
        "sample_strategy_ids": validation["strategy_ids"][:10],
    }


def filter_strategy_config(
    source_config_path: str | Path,
    output_config_path: str | Path,
    *,
    strategy_id_substrings: Sequence[str],
) -> dict[str, Any]:
    source_path = Path(source_config_path)
    output_path = Path(output_config_path)
    if not strategy_id_substrings:
        raise ValueError("at least one strategy_id substring is required")
    matchers = [str(value) for value in strategy_id_substrings if str(value)]
    if not matchers:
        raise ValueError("at least one non-empty strategy_id substring is required")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"source config not found: {source_path}") from exc
    except Exception as exc:
        raise ValueError(f"source config could not be read: {exc}") from exc
    errors: list[str] = []
    strategies = _raw_strategies(payload, errors)
    if errors:
        raise ValueError("; ".join(errors))
    kept = [
        strategy
        for strategy in strategies
        if isinstance(strategy, dict)
        and any(matcher in str(strategy.get("strategy_id") or "") for matcher in matchers)
    ]
    if not kept:
        raise ValueError(
            "no strategies matched strategy_id substrings: " + ", ".join(matchers)
        )
    filtered_payload = dict(payload)
    filtered_payload["strategies"] = kept
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(filtered_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_strategy_config(output_path)
    return {
        "status": "ok" if validation["status"] == "ok" else "error",
        "source_config": str(source_path),
        "output_config": str(output_path),
        "strategy_id_substrings": matchers,
        "source_strategy_count": len(strategies),
        "output_strategy_count": len(kept),
        "strategy_ids": [
            str(strategy.get("strategy_id") or "")
            for strategy in kept
            if isinstance(strategy, dict)
        ],
        "validation": validation,
    }


def generate_paper_strategy_config_sweep(
    *,
    base_config_path: str | Path,
    output_dir: str | Path,
    name_prefix: str,
    thresholds: str | Sequence[float],
    max_probabilities: str | Sequence[float | None],
    time_windows: str | Sequence[tuple[float, float]],
    bankrolls: str | Sequence[float],
    blocked_liquidity_regimes: str | Sequence[str] | None = None,
    fixed_horizon_sec: int | float = 15,
) -> dict[str, Any]:
    source_path = Path(base_config_path)
    destination = Path(output_dir)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"base config not found: {source_path}") from exc
    except Exception as exc:
        raise ValueError(f"base config could not be read: {exc}") from exc
    if _contains_live_trading_enabled(payload):
        raise ValueError("base config has live_trading_enabled=true; refusing to generate paper sweep")
    errors: list[str] = []
    strategies = _raw_strategies(payload, errors)
    if errors:
        raise ValueError("; ".join(errors))
    template = next((strategy for strategy in strategies if isinstance(strategy, dict)), None)
    if template is None:
        raise ValueError("base config does not contain a strategy template")

    threshold_values = _parse_float_list(thresholds, label="thresholds")
    max_probability_values = _parse_optional_probability_list(max_probabilities)
    window_values = _parse_time_windows(time_windows)
    bankroll_values = _parse_float_list(bankrolls, label="bankrolls")
    blocked_regimes = _parse_string_list(blocked_liquidity_regimes)
    fixed_horizon = float(fixed_horizon_sec)
    if fixed_horizon < 0:
        raise ValueError("fixed_horizon_sec must be >= 0")
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    if not max_probability_values:
        raise ValueError("at least one max probability value is required")
    if not window_values:
        raise ValueError("at least one time window is required")
    if not bankroll_values:
        raise ValueError("at least one bankroll is required")

    destination.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    for bankroll in bankroll_values:
        generated_payload = copy.deepcopy(payload)
        _force_paper_only(generated_payload)
        _set_bankroll(generated_payload, bankroll)
        generated_payload["generated_sweep"] = {
            "base_config": str(source_path),
            "name_prefix": str(name_prefix),
            "thresholds": threshold_values,
            "max_probabilities": [
                "none" if value is None else value for value in max_probability_values
            ],
            "time_windows": [
                {"min_sec": min_sec, "max_sec": max_sec}
                for min_sec, max_sec in window_values
            ],
            "bankroll": bankroll,
            "blocked_liquidity_regimes": blocked_regimes,
            "fixed_horizon_sec": fixed_horizon,
        }
        generated_strategies: list[dict[str, Any]] = []
        for threshold in threshold_values:
            for max_probability in max_probability_values:
                for min_time, max_time in window_values:
                    generated_strategies.append(
                        _strategy_from_sweep_template(
                            template,
                            name_prefix=str(name_prefix),
                            threshold=threshold,
                            max_probability=max_probability,
                            min_time=min_time,
                            max_time=max_time,
                            bankroll=bankroll,
                            blocked_liquidity_regimes=blocked_regimes,
                            fixed_horizon_sec=fixed_horizon,
                        )
                    )
        generated_payload["strategies"] = generated_strategies
        output_path = destination / f"{_safe_slug(str(name_prefix))}_bankroll_{_money_slug(bankroll)}.json"
        output_path.write_text(
            json.dumps(generated_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation = validate_strategy_config(output_path)
        generated.append(
            {
                "path": str(output_path),
                "bankroll": bankroll,
                "strategy_count": len(generated_strategies),
                "validation": validation,
            }
        )
        if validation.get("status") != "ok":
            validation_errors.append(
                {
                    "path": str(output_path),
                    "errors": validation.get("errors", []),
                }
            )

    status = "ok" if not validation_errors else "error"
    config_glob = str(destination / "*.json")
    return {
        "status": status,
        "base_config": str(source_path),
        "output_dir": str(destination),
        "name_prefix": str(name_prefix),
        "generated_config_count": len(generated),
        "generated_strategy_count": sum(int(item["strategy_count"]) for item in generated),
        "thresholds": threshold_values,
        "max_probabilities": [
            "none" if value is None else value for value in max_probability_values
        ],
        "time_windows": [
            {"min_sec": min_sec, "max_sec": max_sec}
            for min_sec, max_sec in window_values
        ],
        "bankrolls": bankroll_values,
        "blocked_liquidity_regimes": blocked_regimes,
        "fixed_horizon_sec": fixed_horizon,
        "generated_configs": generated,
        "validation_errors": validation_errors,
        "config_glob": config_glob,
        "experiment_launcher_command": (
            ".venv/bin/python -m src.main run-paper-strategy-experiment "
            "--recorder-db data/recorder.db "
            "--model-path data/models/candidate_gradient_boosting_latest/model.joblib "
            "--feature-columns data/models/candidate_gradient_boosting_latest/feature_columns.json "
            f"--config-glob {json.dumps(config_glob)} "
            "--output-dir data/experiments/<EXP_ID> "
            "--poll-sec 1.0 --max-feature-age-sec 8.0 --tmux"
        ),
    }


def _strategy_from_sweep_template(
    template: dict[str, Any],
    *,
    name_prefix: str,
    threshold: float,
    max_probability: float | None,
    min_time: float,
    max_time: float,
    bankroll: float,
    blocked_liquidity_regimes: Sequence[str],
    fixed_horizon_sec: float,
) -> dict[str, Any]:
    strategy = copy.deepcopy(template)
    mode = str(strategy.get("direction_mode") or "NO_ONLY").upper()
    if mode == "NO_ONLY":
        strategy["long_threshold"] = float(strategy.get("long_threshold", 2.0))
        if strategy["long_threshold"] <= 1.0:
            strategy["long_threshold"] = 2.0
        strategy["short_threshold"] = round(max(0.0, min(1.0, 1.0 - float(threshold))), 10)
    elif mode == "YES_ONLY":
        strategy["long_threshold"] = round(float(threshold), 10)
        strategy["short_threshold"] = float(strategy.get("short_threshold", -1.0))
        if strategy["short_threshold"] >= 0.0:
            strategy["short_threshold"] = -1.0
    elif mode == "BOTH":
        strategy["long_threshold"] = round(float(threshold), 10)
        strategy["short_threshold"] = round(max(0.0, min(1.0, 1.0 - float(threshold))), 10)
    strategy["min_probability_for_direction"] = round(float(threshold), 10)
    if max_probability is None:
        strategy.pop("max_probability_for_direction", None)
    else:
        strategy["max_probability_for_direction"] = round(float(max_probability), 10)
    strategy["min_time_until_resolution_sec"] = float(min_time)
    strategy["max_time_until_resolution_sec"] = float(max_time)
    strategy["exit_type"] = "FIXED_HORIZON_EXIT"
    strategy["fixed_horizon_exit_sec"] = fixed_horizon_sec
    strategy["require_positive_edge"] = bool(strategy.get("require_positive_edge", True))
    if blocked_liquidity_regimes:
        strategy["blocked_liquidity_regimes"] = list(blocked_liquidity_regimes)
    threshold_slug = _probability_slug(threshold)
    max_probability_slug = "none" if max_probability is None else _probability_slug(max_probability)
    time_slug = f"{_time_slug(min_time)}_{_time_slug(max_time)}"
    liquidity_slug = "_".join(str(value).strip().lower() for value in blocked_liquidity_regimes if str(value).strip())
    no_thin_slug = "_no_thin" if "thin" in {str(value).strip().lower() for value in blocked_liquidity_regimes} else ""
    strategy_id = (
        f"{_safe_slug(name_prefix)}_fh{_time_slug(fixed_horizon_sec)}_t{threshold_slug}"
        f"_{time_slug}_pmax{max_probability_slug}_br{_money_slug(bankroll)}{no_thin_slug}"
    )
    strategy["strategy_id"] = strategy_id
    strategy["strategy_name"] = (
        f"{mode} {name_prefix} fh{fixed_horizon_sec:g} "
        f"p>={threshold:.2f} {min_time:g}-{max_time:g}s "
        f"pmax={'none' if max_probability is None else f'{max_probability:.2f}'} "
        f"bankroll={bankroll:g}"
        + (f" block {liquidity_slug}" if liquidity_slug else "")
    )
    return strategy


def _force_paper_only(payload: dict[str, Any]) -> None:
    live = payload.get("live_trading")
    if isinstance(live, dict):
        live["live_trading_enabled"] = False
        live.setdefault("dry_run_orders", True)
    else:
        payload["live_trading"] = {
            "live_trading_enabled": False,
            "dry_run_orders": True,
        }
    if "live_trading_enabled" in payload:
        payload["live_trading_enabled"] = False


def _set_bankroll(payload: dict[str, Any], bankroll: float) -> None:
    nested = payload.get("bankroll")
    if isinstance(nested, dict):
        nested["starting_bankroll_usd"] = float(bankroll)
        nested.setdefault("bankroll_enabled", True)
        return
    payload["bankroll"] = {
        "bankroll_enabled": True,
        "starting_bankroll_usd": float(bankroll),
        "base_risk_fraction": float(payload.get("base_risk_fraction", 0.01)),
        "max_risk_fraction": float(payload.get("max_risk_fraction", 0.05)),
        "max_total_exposure_fraction": float(
            payload.get("max_total_exposure_fraction", 0.2)
        ),
        "min_stake_usd": float(payload.get("min_stake_usd", 0.25)),
        "max_stake_usd": float(payload.get("max_stake_usd", 5.0)),
        "stop_trading_on_bankroll_depleted": bool(
            payload.get("stop_trading_on_bankroll_depleted", True)
        ),
        "reset_bankroll_on_new_run_id": bool(
            payload.get("reset_bankroll_on_new_run_id", True)
        ),
    }


def _contains_live_trading_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("live_trading_enabled") is True:
            return True
        return any(_contains_live_trading_enabled(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_live_trading_enabled(item) for item in value)
    return False


def _parse_float_list(value: str | Sequence[float], *, label: str) -> list[float]:
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    values: list[float] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        try:
            parsed = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains non-numeric value: {item!r}") from exc
        values.append(parsed)
    return values


def _parse_optional_probability_list(
    value: str | Sequence[float | None],
) -> list[float | None]:
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    values: list[float | None] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if not text:
            continue
        if text in {"none", "null", "off", "disable", "disabled"}:
            values.append(None)
            continue
        try:
            parsed = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_probabilities contains invalid value: {item!r}") from exc
        if not 0.0 <= parsed <= 1.0:
            raise ValueError("max_probabilities values must be between 0 and 1 or none")
        values.append(parsed)
    return values


def _parse_time_windows(
    value: str | Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if isinstance(value, str):
        raw_items: Sequence[Any] = [item for item in value.split(",") if item.strip()]
    else:
        raw_items = list(value)
    windows: list[tuple[float, float]] = []
    for item in raw_items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            min_value = float(item[0])
            max_value = float(item[1])
        else:
            text = str(item).strip()
            if ":" not in text:
                raise ValueError(f"time window must use min:max format: {item!r}")
            left, right = text.split(":", 1)
            min_value = float(left)
            max_value = float(right)
        if min_value < 0:
            raise ValueError("time window minimum must be >= 0")
        if max_value <= min_value:
            raise ValueError("time window maximum must be > minimum")
        windows.append((min_value, max_value))
    return windows


def _parse_string_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip().lower() for item in raw_items if str(item).strip()]


def _probability_slug(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def _time_slug(value: float) -> str:
    return f"{int(round(float(value))):02d}" if float(value) < 100 else f"{int(round(float(value))):03d}"


def _money_slug(value: float) -> str:
    return str(int(round(float(value)))) if float(value).is_integer() else str(value).replace(".", "p")


def _safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "sweep"


def _raw_strategies(payload: Any, errors: list[str]) -> list[Any]:
    if not isinstance(payload, dict):
        errors.append("config root must be an object")
        return []
    strategies = payload.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        errors.append("strategies must be a non-empty list")
        return []
    return strategies


def _realistic_execution_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("realistic_execution")
    if isinstance(nested, dict):
        return dict(nested)
    return {
        key: payload[key]
        for key in (
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
        )
        if key in payload
    }


def _trade_selection_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("trade_selection")
    return dict(nested) if isinstance(nested, dict) else {}


def _live_trading_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("live_trading")
    if isinstance(nested, dict):
        return dict(nested)
    return {
        field: payload[field]
        for field in (
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
        )
        if field in payload
    }


def _global_liquidity_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    fields = (
        "liquidity_fill_check_enabled",
        "max_top_of_book_fill_fraction",
        "min_top_of_book_shares",
        "skip_if_liquidity_missing",
    )
    nested = payload.get("liquidity")
    if isinstance(nested, dict):
        result = dict(nested)
        if "enabled" in result and "liquidity_fill_check_enabled" not in result:
            result["liquidity_fill_check_enabled"] = result["enabled"]
        if "fill_check_enabled" in result and "liquidity_fill_check_enabled" not in result:
            result["liquidity_fill_check_enabled"] = result["fill_check_enabled"]
        return {field: result[field] for field in fields if field in result}
    return {
        field: payload[field]
        for field in fields
        if field in payload
    }


def _validate_bankroll_config(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
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
    if nested is not None and not isinstance(nested, dict):
        errors.append("bankroll must be an object when provided")
        return
    if isinstance(nested, dict):
        raw.update({field: nested[field] for field in fields if field in nested})
    for field in (
        "bankroll_enabled",
        "stop_trading_on_bankroll_depleted",
        "reset_bankroll_on_new_run_id",
    ):
        if field in raw and not isinstance(raw[field], bool):
            errors.append(f"{field} must be boolean")
    for field in (
        "starting_bankroll_usd",
        "base_risk_fraction",
        "max_risk_fraction",
        "max_total_exposure_fraction",
        "min_stake_usd",
        "max_stake_usd",
    ):
        if field not in raw:
            continue
        value = _numeric_value(raw[field])
        if value is None:
            errors.append(f"{field} must be numeric")
            continue
        if value < 0:
            errors.append(f"{field} must be >= 0")
    min_stake = _numeric_value(raw.get("min_stake_usd"))
    max_stake = _numeric_value(raw.get("max_stake_usd"))
    if min_stake is not None and max_stake is not None and max_stake < min_stake:
        errors.append("max_stake_usd must be >= min_stake_usd")
    base_risk = _numeric_value(raw.get("base_risk_fraction"))
    max_risk = _numeric_value(raw.get("max_risk_fraction"))
    if base_risk is not None and max_risk is not None and max_risk < base_risk:
        errors.append("max_risk_fraction must be >= base_risk_fraction")


def _validate_global_liquidity_config(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
    fields = {
        "liquidity_fill_check_enabled",
        "max_top_of_book_fill_fraction",
        "min_top_of_book_shares",
        "skip_if_liquidity_missing",
    }
    raw: dict[str, Any] = {field: payload[field] for field in fields if field in payload}
    nested = payload.get("liquidity")
    if nested is not None and not isinstance(nested, dict):
        errors.append("liquidity must be an object when provided")
        return
    if isinstance(nested, dict):
        mapped = dict(nested)
        if "enabled" in mapped and "liquidity_fill_check_enabled" not in mapped:
            mapped["liquidity_fill_check_enabled"] = mapped["enabled"]
        if "fill_check_enabled" in mapped and "liquidity_fill_check_enabled" not in mapped:
            mapped["liquidity_fill_check_enabled"] = mapped["fill_check_enabled"]
        raw.update({field: mapped[field] for field in fields if field in mapped})
    for field in ("liquidity_fill_check_enabled", "skip_if_liquidity_missing"):
        if field in raw and not isinstance(raw[field], bool):
            errors.append(f"{field} must be boolean")
    max_fraction = _numeric_value(raw.get("max_top_of_book_fill_fraction"))
    if "max_top_of_book_fill_fraction" in raw:
        if max_fraction is None:
            errors.append("max_top_of_book_fill_fraction must be numeric")
        elif max_fraction <= 0:
            errors.append("max_top_of_book_fill_fraction must be > 0")
    min_shares = _numeric_value(raw.get("min_top_of_book_shares"))
    if "min_top_of_book_shares" in raw:
        if min_shares is None:
            errors.append("min_top_of_book_shares must be numeric")
        elif min_shares < 0:
            errors.append("min_top_of_book_shares must be >= 0")


def _validate_realistic_execution_config(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
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
    if nested is not None and not isinstance(nested, dict):
        errors.append("realistic_execution must be an object when provided")
        return
    if isinstance(nested, dict):
        raw.update({field: nested[field] for field in fields if field in nested})
    for field in (
        "realistic_execution_enabled",
        "require_quote_after_latency",
        "partial_fill_enabled",
    ):
        if field in raw and not isinstance(raw[field], bool):
            errors.append(f"{field} must be boolean")
    for field in (
        "entry_latency_sec",
        "exit_latency_sec",
        "max_quote_wait_sec",
        "max_entry_price_drift_cents",
        "max_exit_price_drift_cents",
        "reject_if_spread_above_cents",
        "reject_if_btc_age_above_sec",
        "reject_if_feature_age_above_sec",
        "apply_extra_slippage_cents",
    ):
        if field not in raw:
            continue
        value = _numeric_value(raw[field])
        if value is None:
            errors.append(f"{field} must be numeric")
            continue
        if value < 0:
            errors.append(f"{field} must be >= 0")


def _validate_trade_selection_config(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
    raw = payload.get("trade_selection")
    if raw is None:
        return
    if not isinstance(raw, dict):
        errors.append("trade_selection must be an object when provided")
        return
    if "selection_enabled" in raw and not isinstance(raw["selection_enabled"], bool):
        errors.append("trade_selection.selection_enabled must be boolean")
    if (
        "allow_multiple_strategy_variants_same_market" in raw
        and not isinstance(raw["allow_multiple_strategy_variants_same_market"], bool)
    ):
        errors.append(
            "trade_selection.allow_multiple_strategy_variants_same_market must be boolean"
        )
    if raw.get("mode", "best_per_market_direction") != "best_per_market_direction":
        errors.append("trade_selection.mode must be best_per_market_direction")
    if "score_field" in raw and not str(raw["score_field"]).strip():
        errors.append("trade_selection.score_field must be non-empty")
    for field in ("max_new_trades_per_market", "max_new_trades_per_market_direction"):
        if field not in raw:
            continue
        value = _numeric_value(raw[field])
        if value is None:
            errors.append(f"trade_selection.{field} must be numeric")
        elif value < 1 or int(value) != value:
            errors.append(f"trade_selection.{field} must be an integer >= 1")


def _validate_live_trading_config(payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
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
    if nested is None:
        pass
    elif not isinstance(nested, dict):
        errors.append("live_trading must be an object when provided")
        return
    else:
        raw.update({field: nested[field] for field in fields if field in nested})
    for field in (
        "live_trading_enabled",
        "dry_run_orders",
        "require_realistic_execution_passed",
        "require_liquidity_check_passed",
        "allow_market_order",
        "use_limit_orders_only",
    ):
        if field in raw and not isinstance(raw[field], bool):
            errors.append(f"live_trading.{field} must be boolean")
    for field in (
        "max_order_usd",
        "max_daily_loss_usd",
        "max_open_exposure_usd",
        "reject_if_btc_age_above_sec",
        "reject_if_feature_age_above_sec",
        "reject_if_spread_above_cents",
    ):
        if field not in raw:
            continue
        value = _numeric_value(raw[field])
        if value is None:
            errors.append(f"live_trading.{field} must be numeric")
        elif value < 0:
            errors.append(f"live_trading.{field} must be >= 0")
    for field in (
        "max_daily_orders",
        "max_open_trades",
        "max_trades_per_market",
        "max_trades_per_market_direction",
    ):
        if field not in raw:
            continue
        value = _numeric_value(raw[field])
        if value is None:
            errors.append(f"live_trading.{field} must be numeric")
        elif value < 1 or int(value) != value:
            errors.append(f"live_trading.{field} must be an integer >= 1")
    if "kill_switch_file" in raw and not str(raw["kill_switch_file"] or "").strip():
        errors.append("live_trading.kill_switch_file must be non-empty")


def _validate_threshold(
    strategy: dict[str, Any],
    field: str,
    *,
    strategy_id: str,
    index: int,
    errors: list[str],
    warnings: list[str],
    disabled_side: str,
) -> None:
    value = _numeric_field(strategy, field, strategy_id, index, errors)
    if value is None:
        return
    label = _label(strategy_id, index)
    if field == "long_threshold" and value > 1:
        warnings.append(f"{label}.long_threshold disables YES side with value > 1")
        return
    if field == "short_threshold" and value < 0:
        warnings.append(f"{label}.short_threshold disables NO side with value < 0")
        return
    if not 0 <= value <= 1:
        direction = "> 1" if disabled_side == "high" else "< 0"
        errors.append(
            f"{label}.{field} must be between 0 and 1 unless intentionally disabled with {direction}"
        )


def _numeric_field(
    strategy: dict[str, Any],
    field: str,
    strategy_id: str,
    index: int,
    errors: list[str],
) -> float | None:
    if field not in strategy:
        errors.append(f"{_label(strategy_id, index)}.{field} is required")
        return None
    try:
        return float(strategy[field])
    except (TypeError, ValueError):
        errors.append(f"{_label(strategy_id, index)}.{field} must be numeric")
        return None


def _optional_numeric_field(
    strategy: dict[str, Any],
    field: str,
    strategy_id: str,
    index: int,
    errors: list[str],
) -> float | None:
    if field not in strategy or strategy.get(field) is None:
        return None
    return _numeric_field(strategy, field, strategy_id, index, errors)


def _numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_probability_field(
    strategy: dict[str, Any],
    field: str,
    strategy_id: str,
    index: int,
    errors: list[str],
) -> float | None:
    if field not in strategy or strategy.get(field) is None:
        return None
    value = _numeric_field(strategy, field, strategy_id, index, errors)
    if value is not None and not 0 <= value <= 1:
        errors.append(f"{_label(strategy_id, index)}.{field} must be between 0 and 1")
    return value


def _numbers(strategies: Sequence[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for strategy in strategies:
        try:
            values.append(float(strategy[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _range(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def _label(strategy_id: str, index: int) -> str:
    return strategy_id or f"strategies[{index}]"


__all__ = [
    "filter_strategy_config",
    "generate_paper_strategy_config_sweep",
    "summarize_strategy_config",
    "validate_strategy_config",
]
