from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .backtest_baseline_strategy import (
    _load_feature_columns,
    _load_model,
    _predict_probabilities,
)
from .baseline_paper_trader import connect_recorder_read_only
from .btc_price_feed import POLYMARKET_RTDS_BINANCE_SOURCE, POLYMARKET_RTDS_CHAINLINK_SOURCE
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none


PROBABILITY_BUCKETS: tuple[str, ...] = tuple(
    f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}" for idx in range(10)
)
TIME_BUCKETS: tuple[str, ...] = ("0-30s", "30-60s", "60-120s", "120-180s", "180s+")


class LiveModelAuditError(RuntimeError):
    pass


def audit_live_model_predictions(
    *,
    db_path: str,
    model_path: str,
    feature_columns_path: str,
    output_path: str,
    output_csv_path: str | None = None,
) -> dict[str, Any]:
    model = _load_model(model_path)
    feature_columns = _load_feature_columns(feature_columns_path)
    conn = connect_recorder_read_only(db_path)
    try:
        rows = _fetch_resolved_feature_rows(conn)
    finally:
        conn.close()
    missing_columns = _missing_feature_columns(rows, feature_columns)
    if missing_columns:
        raise LiveModelAuditError(
            "resolved feature rows are missing model feature columns: "
            + ", ".join(missing_columns)
        )
    probabilities = _predict_probabilities(model, rows, feature_columns)
    predictions = [
        _prediction_row(row, probability=probability)
        for row, probability in zip(rows, probabilities)
    ]
    report = _build_audit_report(
        predictions,
        db_path=db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        output_path=output_path,
        output_csv_path=output_csv_path,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if output_csv_path:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_predictions_csv(csv_path, predictions)
    return report


def _fetch_resolved_feature_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    now_iso = to_iso(datetime.now(timezone.utc))
    rows = conn.execute(
        """
        SELECT
          f.*,
          m.question,
          m.start_time,
          m.close_time,
          m.phase AS market_phase,
          m.yes_token_id,
          m.no_token_id,
          m.resolved,
          m.winning_outcome,
          m.winning_asset_id,
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
        JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        LEFT JOIN market_snapshots s
          ON s.run_id = f.run_id
         AND s.market_id = f.market_id
         AND s.timestamp = f.timestamp
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
        WHERE m.resolved = 1
          AND LOWER(m.winning_outcome) IN ('up', 'down')
          AND f.feature_ready = 1
          AND f.snapshot_quality_status = 'ok'
        ORDER BY datetime(f.timestamp) ASC, f.run_id ASC, f.market_id ASC
        """,
        (
            now_iso,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
        ),
    ).fetchall()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["actual_yes"] = 1 if str(item.get("winning_outcome")).lower() == "up" else 0
        if item.get("time_until_resolution") is None and item.get("seconds_before_close") is not None:
            item["time_until_resolution"] = item.get("seconds_before_close")
        item["seconds_after_start"] = _seconds_after_start(item)
        normalized.append(item)
    return normalized


def _prediction_row(row: dict[str, Any], *, probability: float) -> dict[str, Any]:
    actual = int(row["actual_yes"])
    predicted = 1 if float(probability) >= 0.5 else 0
    result = dict(row)
    result["predicted_probability_yes"] = round(float(probability), 10)
    result["predicted_label_yes"] = predicted
    result["prediction_correct"] = int(predicted == actual)
    return result


def _build_audit_report(
    rows: Sequence[dict[str, Any]],
    *,
    db_path: str,
    model_path: str,
    feature_columns_path: str,
    output_path: str,
    output_csv_path: str | None,
) -> dict[str, Any]:
    labels = [int(row["actual_yes"]) for row in rows]
    probabilities = [float(row["predicted_probability_yes"]) for row in rows]
    predictions = [int(row["predicted_label_yes"]) for row in rows]
    markets = {(str(row.get("run_id") or ""), str(row.get("market_id") or "")) for row in rows}
    summary = {
        "status": "ok",
        "db_path": db_path,
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "output_path": output_path,
        "output_csv_path": output_csv_path,
        "generated_at": to_iso(datetime.now(timezone.utc)),
        "overall": {
            "row_count": len(rows),
            "market_count": len(markets),
            "up_row_count": sum(labels),
            "down_row_count": len(labels) - sum(labels),
            "accuracy_at_threshold_0_50": _accuracy(labels, predictions),
            "precision_up": _precision(labels, predictions, positive_label=1),
            "recall_up": _recall(labels, predictions, positive_label=1),
            "precision_down": _precision(labels, predictions, positive_label=0),
            "recall_down": _recall(labels, predictions, positive_label=0),
            "log_loss": _log_loss(labels, probabilities),
            "brier_score": _brier_score(labels, probabilities),
            "roc_auc": _roc_auc(labels, probabilities),
        },
        "calibration_buckets": _calibration_buckets(rows),
        "time_until_resolution_buckets": _time_buckets(rows),
        "markets": _market_reports(rows),
        "warnings": _warnings(rows),
    }
    return summary


def _calibration_buckets(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in PROBABILITY_BUCKETS}
    for row in rows:
        groups[_probability_bucket(float(row["predicted_probability_yes"]))].append(row)
    return {bucket: _bucket_metrics(group) for bucket, group in groups.items()}


def _time_buckets(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in TIME_BUCKETS}
    for row in rows:
        groups[_time_bucket(row)].append(row)
    return {bucket: _bucket_metrics(group) for bucket, group in groups.items()}


def _bucket_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["actual_yes"]) for row in rows]
    probabilities = [float(row["predicted_probability_yes"]) for row in rows]
    predictions = [int(row["predicted_label_yes"]) for row in rows]
    markets = {(str(row.get("run_id") or ""), str(row.get("market_id") or "")) for row in rows}
    return {
        "rows": len(rows),
        "row_count": len(rows),
        "markets": len(markets),
        "market_count": len(markets),
        "avg_predicted_probability_yes": _mean(probabilities),
        "actual_up_rate": _mean([float(label) for label in labels]),
        "accuracy": _accuracy(labels, predictions),
        "log_loss": _log_loss(labels, probabilities),
        "brier_score": _brier_score(labels, probabilities),
    }


def _market_reports(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("run_id") or ""), str(row.get("market_id") or ""))].append(row)
    reports: list[dict[str, Any]] = []
    for (run_id, market_id), group in groups.items():
        probabilities = [float(row["predicted_probability_yes"]) for row in group]
        avg_probability = _mean(probabilities)
        actual = int(group[0]["actual_yes"])
        predicted = 1 if (avg_probability is not None and avg_probability >= 0.5) else 0
        reports.append(
            {
                "run_id": run_id,
                "market_id": market_id,
                "question": group[0].get("question"),
                "winning_outcome": group[0].get("winning_outcome"),
                "row_count": len(group),
                "avg_predicted_probability_yes": avg_probability,
                "final_prediction_by_avg_prob": "Up" if predicted == 1 else "Down",
                "market_level_correct": int(predicted == actual),
            }
        )
    reports.sort(key=lambda row: (str(row["run_id"]), str(row["market_id"])))
    return reports


def _warnings(rows: Sequence[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    market_count = len({(str(row.get("run_id") or ""), str(row.get("market_id") or "")) for row in rows})
    row_count = len(rows)
    if market_count < 30:
        warnings.append("market_count_below_30")
    if market_count and row_count / market_count > 10:
        warnings.append("row_count_much_larger_than_market_count_correlated_samples")
    buckets = _calibration_buckets(rows)
    for bucket, metrics in buckets.items():
        if int(metrics["rows"]) < 100:
            warnings.append(f"probability_bucket_{bucket}_below_100_rows")
    if _high_confidence_wrong_more_than_low_confidence(rows):
        warnings.append("high_confidence_buckets_wrong_more_than_low_confidence_buckets")
    return warnings


def _high_confidence_wrong_more_than_low_confidence(rows: Sequence[dict[str, Any]]) -> bool:
    high = [
        row for row in rows
        if float(row["predicted_probability_yes"]) >= 0.8
        or float(row["predicted_probability_yes"]) <= 0.2
    ]
    low = [row for row in rows if 0.4 <= float(row["predicted_probability_yes"]) < 0.6]
    if len(high) < 10 or len(low) < 10:
        return False
    high_accuracy = _accuracy(
        [int(row["actual_yes"]) for row in high],
        [int(row["predicted_label_yes"]) for row in high],
    )
    low_accuracy = _accuracy(
        [int(row["actual_yes"]) for row in low],
        [int(row["predicted_label_yes"]) for row in low],
    )
    return high_accuracy is not None and low_accuracy is not None and high_accuracy < low_accuracy


def _probability_bucket(probability: float) -> str:
    clamped = min(0.999999, max(0.0, float(probability)))
    idx = int(clamped * 10)
    return PROBABILITY_BUCKETS[idx]


def _time_bucket(row: dict[str, Any]) -> str:
    value = _float_or_none(row.get("time_until_resolution"))
    if value is None:
        value = _float_or_none(row.get("seconds_before_close"))
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


def _accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float | None:
    if not labels:
        return None
    return round(sum(1 for actual, pred in zip(labels, predictions) if actual == pred) / len(labels), 10)


def _precision(labels: Sequence[int], predictions: Sequence[int], *, positive_label: int) -> float | None:
    predicted_positive = [idx for idx, pred in enumerate(predictions) if pred == positive_label]
    if not predicted_positive:
        return None
    correct = sum(1 for idx in predicted_positive if labels[idx] == positive_label)
    return round(correct / len(predicted_positive), 10)


def _recall(labels: Sequence[int], predictions: Sequence[int], *, positive_label: int) -> float | None:
    actual_positive = [idx for idx, actual in enumerate(labels) if actual == positive_label]
    if not actual_positive:
        return None
    correct = sum(1 for idx in actual_positive if predictions[idx] == positive_label)
    return round(correct / len(actual_positive), 10)


def _log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if not labels:
        return None
    eps = 1e-15
    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = min(1.0 - eps, max(eps, float(probability)))
        total += int(label) * math.log(p) + (1 - int(label)) * math.log(1.0 - p)
    return round(-total / len(labels), 10)


def _brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if not labels:
        return None
    return round(
        sum((float(probability) - int(label)) ** 2 for label, probability in zip(labels, probabilities)) / len(labels),
        10,
    )


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(enumerate(probabilities), key=lambda item: float(item[1]))
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        tie_end = idx
        while tie_end + 1 < len(ranked) and float(ranked[tie_end + 1][1]) == float(ranked[idx][1]):
            tie_end += 1
        avg_rank = (idx + 1 + tie_end + 1) / 2.0
        for tied_idx in range(idx, tie_end + 1):
            original_idx = ranked[tied_idx][0]
            if int(labels[original_idx]) == 1:
                rank_sum += avg_rank
        idx = tie_end + 1
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(auc, 10)


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 10)


def _seconds_after_start(row: dict[str, Any]) -> float | None:
    start = parse_timestamp(row.get("start_time"))
    timestamp = parse_timestamp(row.get("timestamp"))
    if start is None or timestamp is None:
        return None
    return max(0.0, (timestamp - start).total_seconds())


def _missing_feature_columns(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [column for column in feature_columns if column not in keys]


def _write_predictions_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "market_id",
        "timestamp",
        "question",
        "winning_outcome",
        "actual_yes",
        "predicted_probability_yes",
        "predicted_label_yes",
        "prediction_correct",
        "time_until_resolution",
        "seconds_before_close",
        "btc_chainlink_price",
        "btc_binance_price",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _scalar(row.get(field)) for field in fieldnames})


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


__all__ = ["LiveModelAuditError", "audit_live_model_predictions"]
