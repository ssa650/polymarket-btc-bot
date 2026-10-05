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
from .train_baseline_model import _float_or_none


DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
DEFAULT_HORIZONS_SEC: tuple[int, ...] = (5, 15, 30, 60)
SIDE_POLICIES = {"follow", "fade", "both"}


class TransformerPredictionAutopsyError(RuntimeError):
    pass


def build_transformer_prediction_autopsy_report(
    *,
    recorder_db_path: str,
    prediction_db_paths: Sequence[str],
    output_path: str | None,
    output_csv_path: str | None = None,
    horizons_sec: Sequence[int] = DEFAULT_HORIZONS_SEC,
    near_close_sec: float = 3.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    side_policy: str = "follow",
    min_summary_n: int = 10,
    now: datetime | None = None,
) -> dict[str, Any]:
    recorder = _connect_read_only(recorder_db_path)
    try:
        predictions = load_transformer_prediction_rows(prediction_db_paths)
        rows = replay_transformer_predictions(
            recorder,
            predictions,
            horizons_sec=horizons_sec,
            near_close_sec=near_close_sec,
            thresholds=thresholds,
            side_policy=side_policy,
        )
    finally:
        recorder.close()
    if output_path:
        _write_parquet(rows, Path(output_path))
    if output_csv_path:
        _write_csv(rows, Path(output_csv_path))
    return {
        "status": "ok",
        "generated_at": _iso(now or datetime.now(timezone.utc)),
        "recorder_db_path": recorder_db_path,
        "prediction_db_paths": list(prediction_db_paths),
        "prediction_rows_loaded": len(predictions),
        "rows_evaluated": len(rows),
        "horizons_sec": [int(value) for value in horizons_sec],
        "thresholds": [float(value) for value in thresholds],
        "side_policy": side_policy,
        "min_summary_n": int(min_summary_n),
        "near_close_sec": float(near_close_sec),
        "output_path": output_path,
        "output_csv_path": output_csv_path,
        "output_file_size_bytes": _file_size(Path(output_path)) if output_path else None,
        "output_csv_file_size_bytes": _file_size(Path(output_csv_path)) if output_csv_path else None,
        "label_semantics": audit_transformer_label_semantics(predictions),
        "summary": summarize_transformer_prediction_autopsy(
            rows,
            min_summary_n=int(min_summary_n),
        ),
        "calibration_bins": transformer_calibration_bins(predictions, rows),
    }


def load_transformer_prediction_rows(prediction_db_paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in prediction_db_paths:
        conn = _connect_read_only(path)
        try:
            if not _table_exists(conn, "transformer_predictions"):
                continue
            for row in conn.execute(
                "SELECT * FROM transformer_predictions ORDER BY id ASC"
            ).fetchall():
                item = dict(row)
                item["source_prediction_db"] = str(path)
                rows.append(item)
        finally:
            conn.close()
    return rows


def replay_transformer_predictions(
    recorder: sqlite3.Connection,
    predictions: Sequence[dict[str, Any]],
    *,
    horizons_sec: Sequence[int],
    near_close_sec: float,
    thresholds: Sequence[float],
    side_policy: str = "follow",
) -> list[dict[str, Any]]:
    policy = str(side_policy or "follow").lower()
    if policy not in SIDE_POLICIES:
        raise TransformerPredictionAutopsyError(
            f"side_policy must be one of {sorted(SIDE_POLICIES)}"
        )
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        path = load_market_share_price_path(recorder, prediction)
        for signal_side, executed_side, emitted_policy in _side_policy_candidates(policy):
            signal_probability = _side_probability(prediction, signal_side)
            if signal_probability is None:
                continue
            executed_probability = _side_probability(prediction, executed_side)
            entry_price = _entry_price_for_side(prediction, executed_side, path)
            for threshold in thresholds:
                if float(signal_probability) < float(threshold):
                    continue
                for horizon in horizons_sec:
                    rows.append(
                        simulate_prediction_horizon(
                            prediction,
                            path,
                            side_policy=emitted_policy,
                            signal_side=signal_side,
                            executed_side=executed_side,
                            threshold=float(threshold),
                            probability_for_signal_side=float(signal_probability),
                            probability_for_executed_side=executed_probability,
                            entry_price=entry_price,
                            horizon_sec=int(horizon),
                            near_close_sec=float(near_close_sec),
                        )
                    )
    return rows


def load_market_share_price_path(
    recorder: sqlite3.Connection,
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _table_exists(recorder, "market_snapshots"):
        return []
    market_id = str(prediction.get("market_id") or "")
    run_id = str(prediction.get("run_id") or "")
    signal_ts = _signal_time(prediction)
    close_ts = parse_timestamp(prediction.get("market_close_time"))
    if not market_id or signal_ts is None:
        return []
    where = ["market_id = ?", "datetime(timestamp) >= datetime(?)"]
    params: list[Any] = [market_id, signal_ts.isoformat()]
    if run_id:
        where.insert(0, "run_id = ?")
        params.insert(0, run_id)
    if close_ts is not None:
        where.append("datetime(timestamp) <= datetime(?)")
        params.append(close_ts.isoformat())
    records = recorder.execute(
        f"""
        SELECT *
        FROM market_snapshots
        WHERE {' AND '.join(where)}
        ORDER BY datetime(timestamp) ASC, timestamp ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in records]


def simulate_prediction_horizon(
    prediction: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    side_policy: str,
    signal_side: str,
    executed_side: str,
    threshold: float,
    probability_for_signal_side: float,
    probability_for_executed_side: float | None,
    entry_price: float | None,
    horizon_sec: int,
    near_close_sec: float,
) -> dict[str, Any]:
    selected, hit_reason = _select_horizon_exit(
        prediction,
        path,
        horizon_sec=int(horizon_sec),
        near_close_sec=float(near_close_sec),
    )
    exit_price = _exit_price_for_side(selected or {}, executed_side)
    pnl, roi = _pnl_and_roi(entry_price=entry_price, exit_price=exit_price)
    time_until = _float_or_none(prediction.get("time_until_resolution"))
    spread_regime = _spread_regime(prediction, selected or {})
    liquidity_regime = _liquidity_regime(selected or {})
    probability_yes = _float_or_none(prediction.get("probability_yes"))
    return {
        "source_prediction_db": prediction.get("source_prediction_db"),
        "prediction_id": prediction.get("id"),
        "market_id": prediction.get("market_id"),
        "run_id": prediction.get("run_id"),
        "signal_timestamp": prediction.get("signal_timestamp")
        or prediction.get("latest_feature_timestamp")
        or prediction.get("timestamp"),
        "side_policy": str(side_policy),
        "signal_side": signal_side,
        "executed_side": executed_side,
        "side": executed_side,
        "threshold": round(float(threshold), 4),
        "probability_for_signal_side": _round(probability_for_signal_side),
        "probability_for_executed_side": _round(probability_for_executed_side),
        "probability_for_side": _round(probability_for_executed_side),
        "probability_yes": _round(probability_yes),
        "probability_no": _round(prediction.get("probability_no")),
        "probability_yes_bin": _probability_bin(probability_yes),
        "entry_price_used": _round(entry_price),
        "exit_price_used": _round(exit_price),
        "horizon_sec": int(horizon_sec),
        "simulated_pnl_usd": _round(pnl),
        "simulated_roi": _round(roi),
        "hit_reason": hit_reason,
        "price_path_rows": len(path),
        "time_until_resolution": _round(time_until),
        "time_regime": _time_regime(time_until),
        "liquidity_regime": liquidity_regime,
        "spread_regime": spread_regime,
        "snapshot_quality_status": prediction.get("snapshot_quality_status"),
        "latest_feature_timestamp": prediction.get("latest_feature_timestamp"),
        "market_start_time": prediction.get("market_start_time"),
        "market_close_time": prediction.get("market_close_time"),
        "model_path": prediction.get("model_path"),
        "sequence_length": prediction.get("sequence_length"),
    }


def summarize_transformer_prediction_autopsy(
    rows: Sequence[dict[str, Any]],
    *,
    min_summary_n: int = 10,
) -> dict[str, Any]:
    group_fields = (
        ("side_policy", "signal_side", "executed_side", "threshold", "horizon_sec"),
        ("side_policy", "signal_side", "executed_side", "threshold", "horizon_sec", "time_regime"),
        ("side_policy", "signal_side", "executed_side", "threshold", "horizon_sec", "spread_regime"),
        ("side_policy", "signal_side", "executed_side", "threshold", "horizon_sec", "liquidity_regime"),
    )
    return {
        "rows": len(rows),
        "min_summary_n": int(min_summary_n),
        "by_policy_signal_executed_threshold_horizon": _group_summary(
            rows,
            group_fields[0],
            min_summary_n=min_summary_n,
        ),
        "by_policy_signal_executed_threshold_horizon_time_regime": _group_summary(
            rows,
            group_fields[1],
            min_summary_n=min_summary_n,
        ),
        "by_policy_signal_executed_threshold_horizon_spread_regime": _group_summary(
            rows,
            group_fields[2],
            min_summary_n=min_summary_n,
        ),
        "by_policy_signal_executed_threshold_horizon_liquidity_regime": _group_summary(
            rows,
            group_fields[3],
            min_summary_n=min_summary_n,
        ),
        "by_threshold_side_horizon": _group_summary(
            rows,
            ("threshold", "side", "horizon_sec"),
            min_summary_n=min_summary_n,
        ),
        "by_threshold_side_horizon_time_regime": _group_summary(
            rows,
            ("threshold", "side", "horizon_sec", "time_regime"),
            min_summary_n=min_summary_n,
        ),
        "by_threshold_side_horizon_spread_regime": _group_summary(
            rows,
            ("threshold", "side", "horizon_sec", "spread_regime"),
            min_summary_n=min_summary_n,
        ),
        "by_threshold_side_horizon_liquidity_regime": _group_summary(
            rows,
            ("threshold", "side", "horizon_sec", "liquidity_regime"),
            min_summary_n=min_summary_n,
        ),
    }


def resolve_prediction_db_paths(values: Sequence[str] | str | None) -> list[str]:
    items = _as_list(values)
    resolved: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        resolved.extend(matches if matches else [item])
    deduped: list[str] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(Path(path))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def parse_horizons(value: str | Sequence[int] | None) -> list[int]:
    if value is None:
        return list(DEFAULT_HORIZONS_SEC)
    if isinstance(value, str):
        return [int(float(part.strip())) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def audit_transformer_label_semantics(
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    model_paths = sorted(
        {
            str(row.get("model_path"))
            for row in predictions
            if row.get("model_path") not in (None, "")
        }
    )
    checked_configs: list[dict[str, Any]] = []
    warnings: list[str] = []
    proven = True
    for model_path in model_paths:
        config_path = Path(model_path).parent / "training_config.json"
        if not config_path.exists():
            checked_configs.append(
                {
                    "model_path": model_path,
                    "training_config_path": str(config_path),
                    "status": "missing",
                    "label_column": None,
                }
            )
            warnings.append(f"training_config_missing_for_model_path:{model_path}")
            proven = False
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            checked_configs.append(
                {
                    "model_path": model_path,
                    "training_config_path": str(config_path),
                    "status": "error",
                    "error": str(exc),
                    "label_column": None,
                }
            )
            warnings.append(f"training_config_unreadable_for_model_path:{model_path}")
            proven = False
            continue
        label_column = str(payload.get("label_column") or "")
        status = "ok" if label_column == "label_yes_win" else "warning"
        if status != "ok":
            proven = False
            warnings.append(
                f"unexpected_transformer_label_column:{model_path}:{label_column or 'missing'}"
            )
        checked_configs.append(
            {
                "model_path": model_path,
                "training_config_path": str(config_path),
                "status": status,
                "label_column": label_column,
            }
        )
    if not predictions:
        proven = False
        warnings.append("no_prediction_rows_loaded")
    return {
        "status": "ok" if proven else "warning",
        "probability_yes_meaning": (
            "sigmoid model probability for positive class label_yes_win=1; "
            "sequence export tests map label_yes_win=1 to label_resolved_up_down=UP"
        ),
        "label_yes_win_positive_class": "YES/UP outcome",
        "code_contract": {
            "transformer_sequence_dataset": "copies label column from final row; tests assert label_yes_win=1 maps to UP",
            "train_transformer_sequence_model": "loads labels from label_column and trains binary sigmoid positive class",
            "run_transformer_paper_trader": "emits sigmoid output as probability_yes",
        },
        "checked_training_configs": checked_configs,
        "warnings": warnings,
    }


def transformer_calibration_bins(
    predictions: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    horizons = sorted(
        {
            int(value)
            for value in (
                _int_or_none(row.get("horizon_sec"))
                for row in rows
            )
            if value is not None
        }
    )
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        label = f"{lower:.1f}-{upper:.1f}"
        prediction_group = [
            row for row in predictions
            if _in_probability_bin(_float_or_none(row.get("probability_yes")), lower, upper)
        ]
        row_group = [
            row for row in rows
            if _in_probability_bin(_float_or_none(row.get("probability_yes")), lower, upper)
        ]
        horizon_metrics: dict[str, Any] = {}
        for horizon in horizons:
            follow_yes = [
                row for row in row_group
                if row.get("side_policy") == "follow"
                and row.get("signal_side") == "YES"
                and row.get("executed_side") == "YES"
                and _int_or_none(row.get("horizon_sec")) == horizon
            ]
            fade_yes = [
                row for row in row_group
                if row.get("side_policy") == "fade"
                and row.get("signal_side") == "YES"
                and row.get("executed_side") == "NO"
                and _int_or_none(row.get("horizon_sec")) == horizon
            ]
            horizon_metrics[str(horizon)] = {
                "follow_yes_count": len(follow_yes),
                "avg_follow_yes_roi": _round(_mean(row.get("simulated_roi") for row in follow_yes)),
                "follow_yes_win_rate": _win_rate(follow_yes),
                "follow_yes_total_pnl": _round(_sum(row.get("simulated_pnl_usd") for row in follow_yes)),
                "fade_yes_count": len(fade_yes),
                "avg_fade_yes_roi": _round(_mean(row.get("simulated_roi") for row in fade_yes)),
                "fade_yes_win_rate": _win_rate(fade_yes),
                "fade_yes_total_pnl": _round(_sum(row.get("simulated_pnl_usd") for row in fade_yes)),
            }
        result.append(
            {
                "probability_yes_bin": label,
                "count": len(prediction_group),
                "distinct_markets": _distinct_markets(prediction_group),
                "avg_probability_yes": _round(
                    _mean(row.get("probability_yes") for row in prediction_group)
                ),
                "horizons": horizon_metrics,
            }
        )
    return result


def _select_horizon_exit(
    prediction: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    horizon_sec: int,
    near_close_sec: float,
) -> tuple[dict[str, Any] | None, str]:
    if not path:
        return None, "no_price_path"
    signal_ts = _signal_time(prediction)
    if signal_ts is None:
        return None, "invalid_signal_timestamp"
    target = signal_ts + timedelta(seconds=int(horizon_sec))
    close_ts = parse_timestamp(prediction.get("market_close_time"))
    if close_ts is not None:
        near_close_target = close_ts - timedelta(seconds=max(0.0, float(near_close_sec)))
        if target >= near_close_target:
            selected = _point_at_or_before(path, near_close_target)
            return selected or path[-1], "near_close"
    for point in path:
        timestamp = parse_timestamp(point.get("timestamp"))
        if timestamp is not None and timestamp >= target:
            return point, "fixed_horizon"
    return path[-1], "fixed_horizon_last_available"


def _point_at_or_before(
    path: Sequence[dict[str, Any]],
    target: datetime,
) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    for point in path:
        timestamp = parse_timestamp(point.get("timestamp"))
        if timestamp is not None and timestamp <= target:
            selected = point
        elif timestamp is not None and timestamp > target:
            break
    return selected


def _side_probability(prediction: dict[str, Any], side: str) -> float | None:
    if side == "YES":
        return _float_or_none(prediction.get("probability_yes"))
    probability_no = _float_or_none(prediction.get("probability_no"))
    if probability_no is not None:
        return probability_no
    probability_yes = _float_or_none(prediction.get("probability_yes"))
    return None if probability_yes is None else 1.0 - probability_yes


def _side_policy_candidates(policy: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    if policy in {"follow", "both"}:
        candidates.extend(
            [
                ("YES", "YES", "follow"),
                ("NO", "NO", "follow"),
            ]
        )
    if policy in {"fade", "both"}:
        candidates.extend(
            [
                ("YES", "NO", "fade"),
                ("NO", "YES", "fade"),
            ]
        )
    return candidates


def _entry_price_for_side(
    prediction: dict[str, Any],
    side: str,
    path: Sequence[dict[str, Any]],
) -> float | None:
    if side == "YES":
        fields = ("best_ask_yes", "yes_price", "mid_price_yes", "last_trade_price")
    else:
        fields = ("best_ask_no", "no_price", "mid_price_no", "last_trade_price")
    price = _first_float(prediction, fields)
    if price is not None:
        return price
    return _first_float(path[0], fields) if path else None


def _exit_price_for_side(row: dict[str, Any], side: str) -> float | None:
    if side == "YES":
        return _first_float(row, ("best_bid_yes", "mid_price_yes", "last_trade_price"))
    return _first_float(row, ("best_bid_no", "mid_price_no", "last_trade_price"))


def _pnl_and_roi(
    *,
    entry_price: float | None,
    exit_price: float | None,
) -> tuple[float | None, float | None]:
    if entry_price is None or exit_price is None or entry_price <= 0:
        return None, None
    stake = 1.0
    pnl = stake / entry_price * exit_price - stake
    return pnl, pnl / stake


def _group_summary(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    min_summary_n: int = 10,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    summary: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        if len(group_rows) < int(min_summary_n):
            continue
        wins = [
            row
            for row in group_rows
            if (_float_or_none(row.get("simulated_pnl_usd")) or 0.0) > 0
        ]
        concentration = _market_concentration(group_rows)
        item = {field: value for field, value in zip(fields, key)}
        item.update(
            {
                "n": len(group_rows),
                "distinct_markets": concentration["distinct_markets"],
                "total_simulated_pnl": _round(
                    sum(
                        value
                        for value in (
                            _float_or_none(row.get("simulated_pnl_usd"))
                            for row in group_rows
                        )
                        if value is not None
                    )
                ),
                "avg_simulated_roi": _round(
                    _mean(row.get("simulated_roi") for row in group_rows)
                ),
                "win_rate": len(wins) / len(group_rows) if group_rows else None,
                "top_market_pnl": concentration["top_market_pnl"],
                "top_market_pnl_share_abs": concentration["top_market_pnl_share_abs"],
                "top_3_market_pnl_share_abs": concentration["top_3_market_pnl_share_abs"],
            }
        )
        summary.append(item)
    summary.sort(
        key=lambda item: (
            str(item.get("side") or ""),
            float(item.get("threshold") or 0.0),
            int(item.get("horizon_sec") or 0),
            str(item),
        )
    )
    return summary


def _market_concentration(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_market: dict[str, float] = defaultdict(float)
    for row in rows:
        market_id = str(row.get("market_id") or "unknown")
        pnl = _float_or_none(row.get("simulated_pnl_usd"))
        if pnl is not None:
            by_market[market_id] += pnl
    if not by_market:
        return {
            "distinct_markets": 0,
            "top_market_pnl": None,
            "top_market_pnl_share_abs": None,
            "top_3_market_pnl_share_abs": None,
        }
    ordered = sorted(by_market.values(), key=lambda value: abs(value), reverse=True)
    total_abs = sum(abs(value) for value in by_market.values())
    top = ordered[0]
    return {
        "distinct_markets": len(by_market),
        "top_market_pnl": _round(top),
        "top_market_pnl_share_abs": _round(abs(top) / total_abs) if total_abs else None,
        "top_3_market_pnl_share_abs": _round(sum(abs(value) for value in ordered[:3]) / total_abs) if total_abs else None,
    }


def _signal_time(prediction: dict[str, Any]) -> datetime | None:
    return parse_timestamp(
        prediction.get("signal_timestamp")
        or prediction.get("latest_feature_timestamp")
        or prediction.get("timestamp")
    )


def _time_regime(seconds: Any) -> str:
    value = _float_or_none(seconds)
    if value is None:
        return "unknown"
    if value < 30:
        return "0-30s"
    if value < 60:
        return "30-60s"
    if value < 120:
        return "60-120s"
    return "120s+"


def _probability_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    bounded = max(0.0, min(1.0, float(value)))
    index = min(9, int(bounded * 10))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def _in_probability_bin(value: float | None, lower: float, upper: float) -> bool:
    if value is None:
        return False
    bounded = max(0.0, min(1.0, float(value)))
    if upper >= 1.0:
        return lower <= bounded <= upper
    return lower <= bounded < upper


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None


def _win_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    numeric = [
        value for value in (_float_or_none(row.get("simulated_pnl_usd")) for row in rows)
        if value is not None
    ]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def _sum(values: Sequence[Any]) -> float:
    return sum(value for value in (_float_or_none(item) for item in values) if value is not None)


def _distinct_markets(rows: Sequence[dict[str, Any]]) -> int:
    return len({str(row.get("market_id")) for row in rows if row.get("market_id") not in (None, "")})


def _spread_regime(prediction: dict[str, Any], snapshot: dict[str, Any]) -> str:
    values = [
        value
        for value in (
            _float_or_none(prediction.get("spread_yes")),
            _float_or_none(prediction.get("spread_no")),
            _float_or_none(snapshot.get("spread_yes")),
            _float_or_none(snapshot.get("spread_no")),
        )
        if value is not None
    ]
    if not values:
        return "unknown"
    spread = sum(values) / len(values)
    if spread <= 0.02:
        return "tight"
    if spread <= 0.05:
        return "normal"
    return "wide"


def _liquidity_regime(snapshot: dict[str, Any]) -> str:
    values = [
        value
        for value in (
            _float_or_none(snapshot.get("total_liquidity")),
            _float_or_none(snapshot.get("liquidity")),
            _float_or_none(snapshot.get("orderbook_liquidity")),
            _float_or_none(snapshot.get("best_bid_size_yes")),
            _float_or_none(snapshot.get("best_ask_size_yes")),
            _float_or_none(snapshot.get("best_bid_size_no")),
            _float_or_none(snapshot.get("best_ask_size_no")),
        )
        if value is not None
    ]
    if not values:
        return "unknown"
    liquidity = sum(values)
    if liquidity < 100:
        return "thin"
    if liquidity < 1000:
        return "normal"
    return "deep"


def _first_float(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _round(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return round(numeric, 10) if numeric is not None else None


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


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(rows)
    fieldnames = list(normalized[0].keys()) if normalized else [
        "prediction_id",
        "market_id",
        "signal_timestamp",
        "side",
        "threshold",
        "probability_for_side",
        "entry_price_used",
        "exit_price_used",
        "horizon_sec",
        "simulated_pnl_usd",
        "simulated_roi",
        "hit_reason",
        "price_path_rows",
        "time_until_resolution",
        "snapshot_quality_status",
    ]
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
        raise TransformerPredictionAutopsyError(
            "Transformer prediction autopsy exports require pyarrow."
        ) from exc
    return pa, pq


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _file_size(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
