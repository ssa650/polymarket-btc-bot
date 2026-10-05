from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .train_baseline_model import (
    PRIMARY_TARGET,
    RETURN_TARGET,
    _chronological_split,
    _filter_rows,
    _float_or_none,
    _future_return_by_class,
    _load_parquet_rows,
    _matrix,
    _positive_rate,
    _target,
    _time_range,
)


SUSPICIOUS_SUBSTRINGS: tuple[str, ...] = (
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


class BaselineAuditError(RuntimeError):
    pass


def audit_baseline_model(
    *,
    input_path: str,
    model_dir: str,
    output_dir: str,
    max_btc_age_sec: float = 3.0,
    split_by_market: bool = False,
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    deps = _load_audit_dependencies()
    rows_loaded = _load_parquet_rows(input_path)
    filtered_rows = sorted(
        _filter_rows(rows_loaded, max_btc_age_sec=max_btc_age_sec),
        key=lambda row: str(row.get("timestamp") or ""),
    )
    model_path = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _read_json(model_path / "baseline_metrics.json")
    feature_columns = _read_json(model_path / "feature_columns.json")
    if not isinstance(feature_columns, list):
        raise BaselineAuditError("feature_columns.json must contain a JSON list")
    feature_columns = [str(column) for column in feature_columns]

    split = _split_rows(
        filtered_rows,
        split_by_market=split_by_market,
        train_fraction=train_fraction,
    )
    train_rows = split["train_rows"]
    test_rows = split["test_rows"]
    models = _load_models(model_path, deps["joblib"])
    model_metrics = _evaluate_models(
        models=models,
        test_rows=test_rows,
        feature_columns=feature_columns,
        deps=deps,
    )
    per_market = _per_market_metrics(
        models=models,
        test_rows=test_rows,
        feature_columns=feature_columns,
        deps=deps,
    )
    suspicious = suspicious_feature_scan(
        feature_columns=feature_columns,
        rows=rows_loaded,
    )
    rf_importance_rows = _random_forest_importance_rows(
        models.get("random_forest"),
        feature_columns,
    )
    logistic_rows = _logistic_coefficient_rows(
        models.get("logistic_regression"),
        feature_columns,
    )
    rf_path = output / "random_forest_feature_importance.csv"
    logistic_path = output / "logistic_regression_coefficients.csv"
    _write_csv(
        rf_path,
        fieldnames=("rank", "feature", "importance"),
        rows=rf_importance_rows,
    )
    _write_csv(
        logistic_path,
        fieldnames=("rank", "feature", "coefficient", "abs_coefficient"),
        rows=logistic_rows,
    )

    train_markets = _market_keys(train_rows)
    test_markets = _market_keys(test_rows)
    overlap = sorted(train_markets & test_markets)
    report = {
        "status": "ok",
        "input_path": input_path,
        "model_dir": str(model_path),
        "output_dir": str(output),
        "max_btc_age_sec": max_btc_age_sec,
        "split_mode": "market_holdout" if split_by_market else "row_chronological",
        "train_fraction": train_fraction,
        "baseline_best_model": baseline_metrics.get("best_model_name"),
        "baseline_best_model_accuracy": baseline_metrics.get("best_model_accuracy"),
        "baseline_best_model_balanced_accuracy": baseline_metrics.get(
            "best_model_balanced_accuracy"
        ),
        "dataset_shape": {
            "rows_loaded": len(rows_loaded),
            "rows_after_filters": len(filtered_rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "unique_markets_total": len(_market_keys(filtered_rows)),
            "unique_markets_train": len(train_markets),
            "unique_markets_test": len(test_markets),
            "train_time_range": _time_range(train_rows),
            "test_time_range": _time_range(test_rows),
            "target_positive_rate": _positive_rate(_target(filtered_rows))
            if filtered_rows
            else None,
            "train_positive_rate": _positive_rate(_target(train_rows))
            if train_rows
            else None,
            "test_positive_rate": _positive_rate(_target(test_rows))
            if test_rows
            else None,
            "train_test_market_overlap_count": len(overlap),
            "train_test_market_overlap": [
                {"run_id": run_id, "market_id": market_id}
                for run_id, market_id in overlap[:50]
            ],
        },
        "split_metadata": split["metadata"],
        "models_loaded": sorted(models.keys()),
        "model_metrics": model_metrics,
        "per_market_test_performance": per_market,
        "suspicious_feature_columns": suspicious,
        "confidence_buckets": {
            name: metrics.get("confidence_buckets", [])
            for name, metrics in model_metrics.items()
            if metrics.get("confidence_buckets") is not None
        },
        "feature_importance_files": {
            "random_forest": str(rf_path),
            "logistic_regression": str(logistic_path),
        },
        "market_holdout_degradation": _market_holdout_degradation(
            baseline_metrics,
            model_metrics,
        )
        if split_by_market
        else {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = output / "audit_metrics.json"
    summary_path = output / "audit_summary.txt"
    report["artifacts_written"] = [
        str(metrics_path),
        str(rf_path),
        str(logistic_path),
        str(summary_path),
    ]
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(_render_audit_summary(report), encoding="utf-8")
    return report


def suspicious_feature_scan(
    *,
    feature_columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lower_substrings = tuple(value.lower() for value in SUSPICIOUS_SUBSTRINGS)
    for column in feature_columns:
        reasons: list[str] = []
        lowered = column.lower()
        for needle in lower_substrings:
            if needle in lowered:
                reasons.append(f"substring:{needle}")
        if "timestamp" in lowered or lowered.endswith("_iso"):
            reasons.append("timestamp_or_time_column")
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and not all(_is_numeric(value) for value in values):
            reasons.append("non_numeric_or_object_column")
        if reasons:
            result.append({"column": column, "reasons": sorted(set(reasons))})
    return result


def _load_audit_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise BaselineAuditError(
            "Baseline model audit requires scikit-learn and joblib. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return {
        "joblib": joblib,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise BaselineAuditError(f"Missing required model artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_models(model_dir: Path, joblib_module: Any) -> dict[str, Any]:
    specs = {
        "logistic_regression": model_dir / "model_logistic_regression.joblib",
        "random_forest": model_dir / "model_random_forest.joblib",
    }
    models: dict[str, Any] = {}
    for name, path in specs.items():
        if path.exists():
            models[name] = joblib_module.load(path)
    return models


def _split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split_by_market: bool,
    train_fraction: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "train_rows": [],
            "test_rows": [],
            "metadata": {"split_by_market": split_by_market},
        }
    if not split_by_market:
        split_idx = _chronological_split(rows)
        return {
            "train_rows": list(rows[:split_idx]),
            "test_rows": list(rows[split_idx:]),
            "metadata": {
                "split_by_market": False,
                "split_index": split_idx,
            },
        }

    market_order = sorted(
        {
            _market_key(row): _market_sort_key(row)
            for row in rows
        }.items(),
        key=lambda item: item[1],
    )
    markets = [market_key for market_key, _sort_key in market_order]
    split_idx = int(len(markets) * train_fraction)
    if split_idx <= 0:
        split_idx = 1
    if split_idx >= len(markets):
        split_idx = len(markets) - 1
    train_markets = set(markets[:split_idx])
    test_markets = set(markets[split_idx:])
    return {
        "train_rows": [row for row in rows if _market_key(row) in train_markets],
        "test_rows": [row for row in rows if _market_key(row) in test_markets],
        "metadata": {
            "split_by_market": True,
            "market_count": len(markets),
            "train_market_count": len(train_markets),
            "test_market_count": len(test_markets),
            "train_markets": [
                {"run_id": run_id, "market_id": market_id}
                for run_id, market_id in markets[:split_idx]
            ],
            "test_markets": [
                {"run_id": run_id, "market_id": market_id}
                for run_id, market_id in markets[split_idx:]
            ],
        },
    }


def _evaluate_models(
    *,
    models: dict[str, Any],
    test_rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    deps: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        metrics[name] = _evaluate_model(
            name=name,
            model=model,
            rows=test_rows,
            feature_columns=feature_columns,
            deps=deps,
        )
    return metrics


def _evaluate_model(
    *,
    name: str,
    model: Any,
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    deps: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {
            "model_name": name,
            "status": "no_test_rows",
        }
    x_test = _matrix(rows, feature_columns)
    y_test = _target(rows)
    predictions = [int(value) for value in model.predict(x_test)]
    report = {
        "model_name": name,
        "status": "ok",
        "row_count": len(rows),
        "positive_rate": _positive_rate(y_test),
        "prediction_positive_rate": _positive_rate(predictions),
        "accuracy": round(float(deps["accuracy_score"](y_test, predictions)), 6),
        "balanced_accuracy": round(
            float(deps["balanced_accuracy_score"](y_test, predictions)),
            6,
        ),
        "precision": round(
            float(deps["precision_score"](y_test, predictions, zero_division=0)),
            6,
        ),
        "recall": round(
            float(deps["recall_score"](y_test, predictions, zero_division=0)),
            6,
        ),
        "f1": round(float(deps["f1_score"](y_test, predictions, zero_division=0)), 6),
        "confusion_matrix": deps["confusion_matrix"](
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "roc_auc": None,
        "future_return_by_predicted_class": _future_return_by_class(
            predictions,
            [_float_or_none(row.get(RETURN_TARGET)) for row in rows],
        ),
        "confidence_buckets": None,
    }
    probabilities = _predict_positive_probability(model, x_test)
    if probabilities is not None:
        if len(set(y_test)) == 2:
            report["roc_auc"] = round(float(deps["roc_auc_score"](y_test, probabilities)), 6)
        report["confidence_buckets"] = _confidence_buckets(
            probabilities,
            y_test,
            [_float_or_none(row.get(RETURN_TARGET)) for row in rows],
        )
    return report


def _per_market_metrics(
    *,
    models: dict[str, Any],
    test_rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    deps: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        grouped[_market_key(row)].append(row)

    result: list[dict[str, Any]] = []
    for market_key, rows in sorted(grouped.items(), key=lambda item: _market_sort_key(item[1][0])):
        y_test = _target(rows)
        model_rows: dict[str, dict[str, Any]] = {}
        for name, model in models.items():
            x_test = _matrix(rows, feature_columns)
            predictions = [int(value) for value in model.predict(x_test)]
            model_rows[name] = {
                "accuracy": round(float(deps["accuracy_score"](y_test, predictions)), 6),
                "prediction_positive_rate": _positive_rate(predictions),
                "future_return_by_predicted_class": _future_return_by_class(
                    predictions,
                    [_float_or_none(row.get(RETURN_TARGET)) for row in rows],
                ),
            }
        sample = rows[0]
        result.append(
            {
                "run_id": market_key[0],
                "market_id": market_key[1],
                "question": sample.get("question"),
                "start_time": sample.get("start_time"),
                "close_time": sample.get("close_time"),
                "row_count": len(rows),
                "positive_rate": _positive_rate(y_test),
                "models": model_rows,
            }
        )
    return result


def _confidence_buckets(
    probabilities: Sequence[float],
    y_true: Sequence[int],
    future_returns: Sequence[float | None],
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for idx in range(10):
        low = idx / 10.0
        high = (idx + 1) / 10.0
        bucket_values: list[tuple[int, float | None, float]] = []
        for actual, future_return, probability in zip(y_true, future_returns, probabilities):
            bucket_idx = min(9, int(float(probability) * 10.0))
            if bucket_idx == idx:
                bucket_values.append((int(actual), future_return, float(probability)))
        actuals = [actual for actual, _ret, _prob in bucket_values]
        returns = [
            float(ret)
            for _actual, ret, _prob in bucket_values
            if ret is not None
        ]
        buckets.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "row_count": len(bucket_values),
                "actual_positive_rate": _positive_rate(actuals),
                "mean_future_return": _mean(returns),
                "prediction_positive_rate": 1.0 if low >= 0.5 and bucket_values else (0.0 if bucket_values else None),
            }
        )
    return buckets


def _predict_positive_probability(
    model: Any,
    x_test: Sequence[Sequence[float]],
) -> list[float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(x_test)
    except Exception:
        return None
    if getattr(probabilities, "shape", None) is not None and probabilities.shape[1] >= 2:
        return [float(row[1]) for row in probabilities]
    return None


def _random_forest_importance_rows(
    model: Any | None,
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    if model is None:
        return []
    estimator = _final_estimator(model, "randomforestclassifier")
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(
        zip(feature_columns, [float(value) for value in importances]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:30]
    return [
        {"rank": idx, "feature": feature, "importance": importance}
        for idx, (feature, importance) in enumerate(pairs, start=1)
    ]


def _logistic_coefficient_rows(
    model: Any | None,
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    if model is None:
        return []
    estimator = _final_estimator(model, "logisticregression")
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return []
    coefficients = [float(value) for value in coef[0]]
    pairs = sorted(
        zip(feature_columns, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:30]
    return [
        {
            "rank": idx,
            "feature": feature,
            "coefficient": coefficient,
            "abs_coefficient": abs(coefficient),
        }
        for idx, (feature, coefficient) in enumerate(pairs, start=1)
    ]


def _final_estimator(model: Any, expected_step: str) -> Any:
    named_steps = getattr(model, "named_steps", None)
    if named_steps and expected_step in named_steps:
        return named_steps[expected_step]
    return model


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
            writer.writerow(row)


def _render_audit_summary(report: dict[str, Any]) -> str:
    shape = report.get("dataset_shape", {})
    lines = [
        "Baseline Model Audit Summary",
        f"status={report.get('status')}",
        f"split_mode={report.get('split_mode')}",
        f"baseline_best_model={report.get('baseline_best_model')}",
        f"rows_after_filters={shape.get('rows_after_filters')}",
        f"train_rows={shape.get('train_rows')}",
        f"test_rows={shape.get('test_rows')}",
        f"train_test_market_overlap_count={shape.get('train_test_market_overlap_count')}",
        f"suspicious_feature_columns_found={len(report.get('suspicious_feature_columns', []))}",
    ]
    if report.get("suspicious_feature_columns"):
        lines.append("suspicious_feature_columns:")
        for item in report["suspicious_feature_columns"][:20]:
            lines.append(f"  {item.get('column')}: {','.join(item.get('reasons', []))}")
    lines.append("per_model_test_metrics:")
    for name, metrics in report.get("model_metrics", {}).items():
        lines.append(
            "  "
            + " ".join(
                [
                    f"model={name}",
                    f"accuracy={metrics.get('accuracy')}",
                    f"balanced_accuracy={metrics.get('balanced_accuracy')}",
                    f"roc_auc={metrics.get('roc_auc')}",
                ]
            )
        )
    if report.get("market_holdout_degradation"):
        lines.append("market_holdout_degradation:")
        for name, value in report["market_holdout_degradation"].items():
            lines.append(f"  model={name} balanced_accuracy_delta={value.get('balanced_accuracy_delta')}")
    return "\n".join(lines) + "\n"


def _market_holdout_degradation(
    baseline_metrics: dict[str, Any],
    model_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline_model_metrics = baseline_metrics.get("model_metrics") or {}
    result: dict[str, dict[str, Any]] = {}
    for name, current in model_metrics.items():
        baseline = baseline_model_metrics.get(name) or {}
        baseline_balanced = _float_or_none(baseline.get("balanced_accuracy"))
        current_balanced = _float_or_none(current.get("balanced_accuracy"))
        if baseline_balanced is None or current_balanced is None:
            continue
        delta = round(current_balanced - baseline_balanced, 6)
        result[name] = {
            "baseline_balanced_accuracy": baseline_balanced,
            "audit_balanced_accuracy": current_balanced,
            "balanced_accuracy_delta": delta,
            "much_worse_than_row_split": delta <= -0.10,
        }
    return result


def _market_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    return {_market_key(row) for row in rows}


def _market_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("run_id") or ""), str(row.get("market_id") or ""))


def _market_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("start_time") or ""),
        str(row.get("close_time") or ""),
        str(row.get("run_id") or ""),
        str(row.get("market_id") or ""),
    )


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    return False


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 10)


__all__ = [
    "BaselineAuditError",
    "SUSPICIOUS_SUBSTRINGS",
    "audit_baseline_model",
    "suspicious_feature_scan",
]
