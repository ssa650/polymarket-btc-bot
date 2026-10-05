from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .train_baseline_model import (
    _chronological_split,
    _filter_rows,
    _float_or_none,
    _load_parquet_rows,
    _matrix,
    _time_range,
)


TRADE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "market_id",
    "question",
    "start_time",
    "close_time",
    "signal_timestamp",
    "timestamp",
    "market_start_time",
    "market_close_time",
    "time_until_resolution",
    "direction",
    "predicted_probability",
    "predicted_prob_up",
    "entry_price",
    "adjusted_entry_price",
    "effective_entry_price",
    "stake_usd",
    "shares",
    "resolved_label",
    "label_yes_win",
    "payout_usd",
    "pnl",
    "pnl_usd",
    "roi",
    "btc_chainlink_price",
    "btc_binance_price",
    "btc_price_diff_binance_minus_chainlink",
)

SKIPPED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "market_id",
    "timestamp",
    "reason",
    "predicted_prob_up",
)

PROBABILITY_BUCKETS: tuple[str, ...] = tuple(
    f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}"
    for idx in range(10)
)

TIME_BUCKETS: tuple[str, ...] = ("0-30s", "30-60s", "60-120s", "120-180s", "180s+")
BACKTEST_SPLITS: tuple[str, ...] = ("test_only", "full_dataset", "market_holdout")

_KNOWN_AT_INFERENCE_SUSPICIOUS = {"time_until_resolution"}
_SUSPICIOUS_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "label",
    "future",
    "resolution",
    "resolved",
    "outcome",
    "win",
    "close_time",
    "market_end",
    "btc_price_at_resolution",
    "return_to_resolution",
    "yes_win",
    "up_at_resolution",
)


class BaselineBacktestError(RuntimeError):
    pass


def backtest_baseline_strategy(
    *,
    input_path: str,
    model_path: str,
    feature_columns_path: str,
    output_dir: str,
    max_btc_age_sec: float = 3.0,
    long_threshold: float = 0.65,
    short_threshold: float = 0.35,
    allow_multiple_per_market: bool = False,
    stake_usd: float = 1.0,
    entry_slippage_cents: float = 0.0,
    fee_cents: float = 0.0,
    backtest_split: str = "test_only",
    allow_suspicious_features: bool = False,
) -> dict[str, Any]:
    model = _load_model(model_path)
    feature_columns = _load_feature_columns(feature_columns_path)
    suspicious_features = _suspicious_feature_report(feature_columns)
    blocking_suspicious = [
        row for row in suspicious_features if not bool(row.get("known_at_inference"))
    ]
    if blocking_suspicious and not allow_suspicious_features:
        raise BaselineBacktestError(
            "Suspicious feature columns found. Re-run with "
            "--allow-suspicious-features to override: "
            + ", ".join(str(row["column"]) for row in blocking_suspicious)
        )
    if not hasattr(model, "predict_proba"):
        raise BaselineBacktestError("Model must support predict_proba for strategy backtest")

    rows_loaded = _load_parquet_rows(input_path)
    filtered_rows = sorted(
        _filter_rows(rows_loaded, max_btc_age_sec=max_btc_age_sec),
        key=lambda row: str(row.get("timestamp") or ""),
    )
    split = _split_backtest_rows(filtered_rows, backtest_split=backtest_split)
    train_rows = split["train_rows"]
    test_rows = split["test_rows"]
    rows = split["backtest_rows"]
    probabilities = _predict_probabilities(model, rows, feature_columns)

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    handled_markets: set[tuple[str, str]] = set()
    slippage = float(entry_slippage_cents) / 100.0
    fee = float(fee_cents) / 100.0

    for row, probability in zip(rows, probabilities):
        market_key = _market_key(row)
        direction = _direction_for_probability(
            probability,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )
        if direction is None:
            skipped.append(_skipped_row(row, reason="no_trade_signal", probability=probability))
            continue
        if not allow_multiple_per_market and market_key in handled_markets:
            skipped.append(
                _skipped_row(row, reason="already_traded_market", probability=probability)
            )
            continue
        if not allow_multiple_per_market:
            handled_markets.add(market_key)

        label_yes_win = _float_or_none(row.get("label_yes_win"))
        if label_yes_win is None:
            skipped.append(_skipped_row(row, reason="unresolved_market", probability=probability))
            continue
        entry_price = _entry_price(row, direction=direction)
        if entry_price is None:
            skipped.append(_skipped_row(row, reason="missing_entry_price", probability=probability))
            continue
        adjusted_entry = float(entry_price) + slippage + fee
        if adjusted_entry <= 0 or adjusted_entry >= 1:
            skipped.append(_skipped_row(row, reason="invalid_entry_price", probability=probability))
            continue

        payout_per_share = _payout_per_share(direction, int(label_yes_win))
        shares = float(stake_usd) / adjusted_entry
        payout_usd = shares * payout_per_share
        pnl_usd = payout_usd - float(stake_usd)
        roi = pnl_usd / float(stake_usd) if float(stake_usd) else 0.0
        trades.append(
            {
                "run_id": row.get("run_id"),
                "market_id": row.get("market_id"),
                "question": row.get("question"),
                "start_time": row.get("start_time"),
                "close_time": row.get("close_time"),
                "signal_timestamp": row.get("timestamp"),
                "timestamp": row.get("timestamp"),
                "market_start_time": row.get("start_time"),
                "market_close_time": row.get("close_time"),
                "time_until_resolution": _float_or_none(row.get("time_until_resolution")),
                "direction": direction,
                "predicted_probability": round(float(probability), 10),
                "predicted_prob_up": round(float(probability), 10),
                "entry_price": round(float(entry_price), 10),
                "adjusted_entry_price": round(float(adjusted_entry), 10),
                "effective_entry_price": round(float(adjusted_entry), 10),
                "stake_usd": float(stake_usd),
                "shares": round(shares, 10),
                "resolved_label": int(label_yes_win),
                "label_yes_win": int(label_yes_win),
                "payout_usd": round(payout_usd, 10),
                "pnl": round(pnl_usd, 10),
                "pnl_usd": round(pnl_usd, 10),
                "roi": round(roi, 10),
                "btc_chainlink_price": _float_or_none(row.get("btc_chainlink_price")),
                "btc_binance_price": _float_or_none(row.get("btc_binance_price")),
                "btc_price_diff_binance_minus_chainlink": _float_or_none(
                    row.get("btc_price_diff_binance_minus_chainlink")
                ),
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trades_path = output / "trades.csv"
    skipped_path = output / "skipped_rows.csv"
    summary_path = output / "backtest_summary.json"
    text_path = output / "backtest_summary.txt"
    _write_csv(trades_path, fieldnames=TRADE_COLUMNS, rows=trades)
    _write_csv(skipped_path, fieldnames=SKIPPED_COLUMNS, rows=skipped)
    summary = _build_summary(
        input_path=input_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        output_dir=str(output),
        rows_loaded=len(rows_loaded),
        rows_after_filters=len(filtered_rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        backtest_rows=len(rows),
        train_test_market_overlap_count=len(_market_keys(train_rows) & _market_keys(test_rows)),
        unique_markets_total=len(_market_keys(filtered_rows)),
        unique_markets_backtested=len(_market_keys(rows)),
        test_time_range=_time_range(test_rows),
        trades=trades,
        skipped=skipped,
        max_btc_age_sec=max_btc_age_sec,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        allow_multiple_per_market=allow_multiple_per_market,
        stake_usd=stake_usd,
        entry_slippage_cents=entry_slippage_cents,
        fee_cents=fee_cents,
        backtest_split=backtest_split,
        suspicious_feature_columns=suspicious_features,
        allow_suspicious_features=allow_suspicious_features,
    )
    summary["artifacts_written"] = [
        str(summary_path),
        str(trades_path),
        str(skipped_path),
        str(text_path),
    ]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(_render_summary_text(summary), encoding="utf-8")
    return summary


def _load_model(model_path: str) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise BaselineBacktestError(
            "Baseline strategy backtest requires joblib/scikit-learn. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(model_path)
    return joblib.load(path)


def _load_feature_columns(path: str) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise BaselineBacktestError("feature_columns must be a JSON list")
    return [str(column) for column in payload]


def _predict_probabilities(
    model: Any,
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> list[float]:
    if not rows:
        return []
    matrix = _matrix(rows, feature_columns)
    probabilities = model.predict_proba(matrix)
    if getattr(probabilities, "shape", None) is not None and probabilities.shape[1] >= 2:
        return [float(row[1]) for row in probabilities]
    try:
        return [float(row[1]) for row in probabilities]
    except (TypeError, IndexError):
        pass
    raise BaselineBacktestError("Model predict_proba output must include class 1")


def _split_backtest_rows(
    rows: Sequence[dict[str, Any]],
    *,
    backtest_split: str,
) -> dict[str, list[dict[str, Any]]]:
    if backtest_split not in BACKTEST_SPLITS:
        raise BaselineBacktestError(
            f"Unsupported backtest split {backtest_split!r}; expected one of {BACKTEST_SPLITS}"
        )
    ordered = list(rows)
    if backtest_split == "full_dataset":
        return {
            "train_rows": [],
            "test_rows": ordered,
            "backtest_rows": ordered,
        }
    if backtest_split == "test_only":
        split_index = _chronological_split(ordered)
        return {
            "train_rows": ordered[:split_index],
            "test_rows": ordered[split_index:],
            "backtest_rows": ordered[split_index:],
        }

    market_keys = sorted(
        {_market_key(row): _market_sort_key(row) for row in ordered}.items(),
        key=lambda item: item[1],
    )
    markets = [market_key for market_key, _sort_key in market_keys]
    if not markets:
        return {"train_rows": [], "test_rows": [], "backtest_rows": []}
    split_index = int(len(markets) * 0.70)
    if split_index <= 0:
        split_index = 1
    if split_index >= len(markets):
        split_index = len(markets) - 1
    train_markets = set(markets[:split_index])
    test_markets = set(markets[split_index:])
    test_rows = [row for row in ordered if _market_key(row) in test_markets]
    return {
        "train_rows": [row for row in ordered if _market_key(row) in train_markets],
        "test_rows": test_rows,
        "backtest_rows": test_rows,
    }


def _suspicious_feature_report(feature_columns: Sequence[str]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for column in feature_columns:
        lowered = column.lower()
        reasons = [
            f"substring:{needle}"
            for needle in _SUSPICIOUS_FEATURE_SUBSTRINGS
            if needle in lowered
        ]
        known = column in _KNOWN_AT_INFERENCE_SUSPICIOUS
        if reasons or known:
            report.append(
                {
                    "column": column,
                    "reasons": sorted(set(reasons)),
                    "known_at_inference": known,
                    "blocking": bool(reasons and not known),
                }
            )
    return report


def _direction_for_probability(
    probability: float,
    *,
    long_threshold: float,
    short_threshold: float,
) -> str | None:
    if probability >= float(long_threshold):
        return "YES"
    if probability <= float(short_threshold):
        return "NO"
    return None


def _entry_price(row: dict[str, Any], *, direction: str) -> float | None:
    if direction == "YES":
        return _valid_price(row.get("best_ask_yes"))
    no_ask = _valid_price(row.get("best_ask_no"))
    if no_ask is not None:
        return no_ask
    yes_bid = _valid_price(row.get("best_bid_yes"))
    if yes_bid is not None:
        return _valid_price(1.0 - yes_bid)
    return None


def _valid_price(value: Any) -> float | None:
    price = _float_or_none(value)
    if price is None or price <= 0 or price >= 1:
        return None
    return price


def _payout_per_share(direction: str, label_yes_win: int) -> float:
    if direction == "YES":
        return 1.0 if int(label_yes_win) == 1 else 0.0
    return 1.0 if int(label_yes_win) == 0 else 0.0


def _skipped_row(
    row: dict[str, Any],
    *,
    reason: str,
    probability: float,
) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "market_id": row.get("market_id"),
        "timestamp": row.get("timestamp"),
        "reason": reason,
        "predicted_prob_up": round(float(probability), 10),
    }


def _build_summary(
    *,
    input_path: str,
    model_path: str,
    feature_columns_path: str,
    output_dir: str,
    rows_loaded: int,
    rows_after_filters: int,
    train_rows: int,
    test_rows: int,
    backtest_rows: int,
    train_test_market_overlap_count: int,
    unique_markets_total: int,
    unique_markets_backtested: int,
    test_time_range: dict[str, str | None],
    trades: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    max_btc_age_sec: float,
    long_threshold: float,
    short_threshold: float,
    allow_multiple_per_market: bool,
    stake_usd: float,
    entry_slippage_cents: float,
    fee_cents: float,
    backtest_split: str,
    suspicious_feature_columns: Sequence[dict[str, Any]],
    allow_suspicious_features: bool,
) -> dict[str, Any]:
    pnl_values = [float(row["pnl_usd"]) for row in trades]
    roi_values = [float(row["roi"]) for row in trades]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    long_trades = [row for row in trades if row.get("direction") == "YES"]
    short_trades = [row for row in trades if row.get("direction") == "NO"]
    warnings: list[str] = []
    if backtest_split == "full_dataset":
        warnings.append("full_dataset_backtest_includes_training_rows")
    if allow_multiple_per_market:
        warnings.append("multiple_per_market_backtest_can_overstate_live_performance")
    if allow_suspicious_features and any(
        bool(row.get("blocking")) for row in suspicious_feature_columns
    ):
        warnings.append("suspicious_features_allowed")
    return {
        "status": "ok",
        "input_path": input_path,
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "output_dir": output_dir,
        "backtest_split": backtest_split,
        "max_btc_age_sec": max_btc_age_sec,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "allow_multiple_per_market": allow_multiple_per_market,
        "stake_usd": stake_usd,
        "entry_slippage_cents": entry_slippage_cents,
        "fee_cents": fee_cents,
        "rows_loaded": rows_loaded,
        "rows_after_filters": rows_after_filters,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "backtest_rows": backtest_rows,
        "unique_markets_total": unique_markets_total,
        "unique_markets_backtested": unique_markets_backtested,
        "train_test_market_overlap_count": train_test_market_overlap_count,
        "test_time_range": test_time_range,
        "warnings": warnings,
        "suspicious_feature_columns": list(suspicious_feature_columns),
        "allow_suspicious_features": allow_suspicious_features,
        "trades_taken": len(trades),
        "markets_traded": len(_market_keys(trades)),
        "skipped_rows_by_reason": dict(Counter(str(row.get("reason")) for row in skipped)),
        "win_rate": _rate(len(wins), len(trades)),
        "total_staked": round(sum(float(row["stake_usd"]) for row in trades), 10),
        "total_payout": round(sum(float(row["payout_usd"]) for row in trades), 10),
        "total_pnl": round(sum(pnl_values), 10),
        "average_roi_per_trade": _mean(roi_values),
        "median_roi_per_trade": round(float(median(roi_values)), 10) if roi_values else None,
        "max_drawdown_usd": _max_drawdown(pnl_values),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": _win_rate(long_trades),
        "short_win_rate": _win_rate(short_trades),
        "average_entry_price": _mean([float(row["entry_price"]) for row in trades]),
        "average_adjusted_entry_price": _mean(
            [float(row["adjusted_entry_price"]) for row in trades]
        ),
        "average_predicted_probability": _mean(
            [float(row["predicted_prob_up"]) for row in trades]
        ),
        "pnl_by_direction": _pnl_by_direction(trades),
        "pnl_by_probability_bucket": _pnl_by_probability_bucket(trades),
        "pnl_by_time_until_resolution_bucket": _pnl_by_time_bucket(trades),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _market_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("run_id") or ""), str(row.get("market_id") or ""))
        for row in rows
    }


def _pnl_by_direction(trades: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        direction: _aggregate_trades([row for row in trades if row.get("direction") == direction])
        for direction in ("YES", "NO")
    }


def _pnl_by_probability_bucket(
    trades: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in PROBABILITY_BUCKETS}
    for trade in trades:
        probability = float(trade.get("predicted_prob_up") or 0.0)
        grouped[_probability_bucket(probability)].append(trade)
    return {bucket: _aggregate_trades(rows) for bucket, rows in grouped.items()}


def _pnl_by_time_bucket(trades: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in TIME_BUCKETS}
    for trade in trades:
        grouped[_time_bucket(_float_or_none(trade.get("time_until_resolution")))].append(trade)
    return {bucket: _aggregate_trades(rows) for bucket, rows in grouped.items()}


def _aggregate_trades(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trades": len(trades),
        "total_pnl": round(sum(float(row["pnl_usd"]) for row in trades), 10),
        "average_roi": _mean([float(row["roi"]) for row in trades]),
        "win_rate": _win_rate(trades),
    }


def _probability_bucket(probability: float) -> str:
    idx = min(9, max(0, int(float(probability) * 10.0)))
    return PROBABILITY_BUCKETS[idx]


def _time_bucket(value: float | None) -> str:
    if value is None:
        return "180s+"
    if value < 30:
        return "0-30s"
    if value < 60:
        return "30-60s"
    if value < 120:
        return "60-120s"
    if value < 180:
        return "120-180s"
    return "180s+"


def _max_drawdown(pnl_values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        cumulative += float(pnl)
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return round(max_drawdown, 10)


def _win_rate(trades: Sequence[dict[str, Any]]) -> float | None:
    wins = sum(1 for row in trades if float(row.get("pnl_usd") or 0.0) > 0)
    return _rate(wins, len(trades))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 10)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 10)


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in fieldnames})


def _render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Baseline Strategy Backtest Summary",
        f"status={summary.get('status')}",
        f"input_path={summary.get('input_path')}",
        f"model_path={summary.get('model_path')}",
        f"backtest_split={summary.get('backtest_split')}",
        f"rows_loaded={summary.get('rows_loaded')}",
        f"rows_after_filters={summary.get('rows_after_filters')}",
        f"backtest_rows={summary.get('backtest_rows')}",
        f"trades_taken={summary.get('trades_taken')}",
        f"markets_traded={summary.get('markets_traded')}",
        f"win_rate={summary.get('win_rate')}",
        f"total_pnl={summary.get('total_pnl')}",
        f"max_drawdown_usd={summary.get('max_drawdown_usd')}",
        f"warnings={summary.get('warnings')}",
    ]
    return "\n".join(lines) + "\n"


def _market_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("run_id") or ""), str(row.get("market_id") or ""))


def _market_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("close_time") or ""),
        str(row.get("start_time") or ""),
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
    )


__all__ = [
    "BaselineBacktestError",
    "backtest_baseline_strategy",
]
