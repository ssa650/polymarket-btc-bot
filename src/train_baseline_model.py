from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PRIMARY_TARGET = "label_btc_up_at_resolution"
RETURN_TARGET = "future_btc_return_to_resolution_from_feature"

IDENTIFIER_COLUMNS = {
    "run_id",
    "market_id",
    "question",
    "start_time",
    "close_time",
    "timestamp",
    "market_phase",
    "yes_token_id",
    "no_token_id",
    "label_resolved_up_down",
    "label_yes_win",
}

LEAKAGE_COLUMNS = {
    "export_row_usable",
    "btc_price_at_resolution",
    RETURN_TARGET,
}

ALLOWED_RESOLUTION_COLUMNS = {"time_until_resolution"}


class BaselineTrainingError(RuntimeError):
    pass


def train_baseline_model(
    *,
    input_path: str,
    output_dir: str,
    max_btc_age_sec: float = 3.0,
    min_rows: int = 200,
) -> dict[str, Any]:
    deps = _load_training_dependencies()
    rows = _load_parquet_rows(input_path)
    filtered_rows = _filter_rows(rows, max_btc_age_sec=max_btc_age_sec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if len(filtered_rows) < int(min_rows):
        summary = _insufficient_rows_summary(
            input_path=input_path,
            output_dir=str(output),
            rows_loaded=len(rows),
            rows_after_filters=len(filtered_rows),
            min_rows=int(min_rows),
        )
        artifacts = _write_summary_artifacts(
            summary,
            output,
            feature_columns=[],
        )
        summary["artifacts_written"] = artifacts
        return summary

    filtered_rows = sorted(filtered_rows, key=lambda row: str(row.get("timestamp") or ""))
    feature_columns = select_feature_columns(filtered_rows)
    if not feature_columns:
        summary = _insufficient_rows_summary(
            input_path=input_path,
            output_dir=str(output),
            rows_loaded=len(rows),
            rows_after_filters=len(filtered_rows),
            min_rows=int(min_rows),
            reason="no_numeric_features",
        )
        artifacts = _write_summary_artifacts(
            summary,
            output,
            feature_columns=[],
        )
        summary["artifacts_written"] = artifacts
        return summary

    split = _chronological_split(filtered_rows)
    train_rows = filtered_rows[:split]
    test_rows = filtered_rows[split:]
    x_train = _matrix(train_rows, feature_columns)
    y_train = _target(train_rows)
    x_test = _matrix(test_rows, feature_columns)
    y_test = _target(test_rows)
    future_returns_test = [
        _float_or_none(row.get(RETURN_TARGET))
        for row in test_rows
    ]

    model_reports: dict[str, dict[str, Any]] = {}
    artifacts_written: list[str] = []

    dummy = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["DummyClassifier"](strategy="most_frequent"),
    )
    dummy.fit(x_train, y_train)
    model_reports["dummy_most_frequent"] = _evaluate_classifier(
        name="dummy_most_frequent",
        model=dummy,
        x_test=x_test,
        y_test=y_test,
        future_returns=future_returns_test,
        metrics=deps,
    )

    trained_models: dict[str, Any] = {}
    if len(set(y_train)) >= 2:
        logistic = deps["make_pipeline"](
            deps["SimpleImputer"](strategy="median"),
            deps["StandardScaler"](),
            deps["LogisticRegression"](max_iter=1000, random_state=42),
        )
        logistic.fit(x_train, y_train)
        model_reports["logistic_regression"] = _evaluate_classifier(
            name="logistic_regression",
            model=logistic,
            x_test=x_test,
            y_test=y_test,
            future_returns=future_returns_test,
            metrics=deps,
        )
        trained_models["model_logistic_regression.joblib"] = logistic

        random_forest = deps["make_pipeline"](
            deps["SimpleImputer"](strategy="median"),
            deps["RandomForestClassifier"](
                n_estimators=50,
                random_state=42,
                min_samples_leaf=2,
            ),
        )
        random_forest.fit(x_train, y_train)
        model_reports["random_forest"] = _evaluate_classifier(
            name="random_forest",
            model=random_forest,
            x_test=x_test,
            y_test=y_test,
            future_returns=future_returns_test,
            metrics=deps,
        )
        trained_models["model_random_forest.joblib"] = random_forest

        gradient_boosting = deps["make_pipeline"](
            deps["SimpleImputer"](strategy="median"),
            deps["GradientBoostingClassifier"](random_state=42),
        )
        gradient_boosting.fit(x_train, y_train)
        model_reports["gradient_boosting"] = _evaluate_classifier(
            name="gradient_boosting",
            model=gradient_boosting,
            x_test=x_test,
            y_test=y_test,
            future_returns=future_returns_test,
            metrics=deps,
        )
        trained_models["model_gradient_boosting.joblib"] = gradient_boosting
    else:
        model_reports["logistic_regression"] = {
            "model_name": "logistic_regression",
            "status": "skipped",
            "reason": "train_split_has_single_class",
        }
        model_reports["random_forest"] = {
            "model_name": "random_forest",
            "status": "skipped",
            "reason": "train_split_has_single_class",
        }
        model_reports["gradient_boosting"] = {
            "model_name": "gradient_boosting",
            "status": "skipped",
            "reason": "train_split_has_single_class",
        }

    for filename, model in trained_models.items():
        path = output / filename
        deps["joblib"].dump(model, path)
        artifacts_written.append(str(path))

    best = _best_model(model_reports)
    summary = {
        "status": "ok",
        "input_path": input_path,
        "output_dir": str(output),
        "rows_loaded": len(rows),
        "rows_after_filters": len(filtered_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_count": len(feature_columns),
        "target_positive_rate": _positive_rate(_target(filtered_rows)),
        "train_positive_rate": _positive_rate(y_train),
        "test_positive_rate": _positive_rate(y_test),
        "train_time_range": _time_range(train_rows),
        "test_time_range": _time_range(test_rows),
        "best_model_name": best.get("model_name"),
        "best_model_accuracy": best.get("accuracy"),
        "best_model_balanced_accuracy": best.get("balanced_accuracy"),
        "model_metrics": model_reports,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    artifacts_written.extend(
        _write_summary_artifacts(
            summary,
            output,
            feature_columns=feature_columns,
            existing_artifacts=artifacts_written,
        )
    )
    summary["artifacts_written"] = artifacts_written
    return summary


def select_feature_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    columns = sorted({column for row in rows for column in row.keys()})
    selected: list[str] = []
    for column in columns:
        if _exclude_column(column):
            continue
        values = [row.get(column) for row in rows]
        non_null = [value for value in values if value is not None]
        if not non_null:
            continue
        if all(_is_numeric(value) for value in non_null):
            selected.append(column)
    return selected


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise BaselineTrainingError(
            "Baseline model training requires scikit-learn. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return {
        "joblib": joblib,
        "DummyClassifier": DummyClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def _load_parquet_rows(input_path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BaselineTrainingError(
            "Baseline model training requires pyarrow to read Parquet. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(input_path)
    return pq.read_table(path).to_pylist()


def _filter_rows(rows: Sequence[dict[str, Any]], *, max_btc_age_sec: float) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("export_row_usable") or 0) != 1:
            continue
        target = row.get(PRIMARY_TARGET)
        if target is None:
            continue
        btc_age = _float_or_none(row.get("btc_chainlink_age_sec_at_feature"))
        if btc_age is None or btc_age > float(max_btc_age_sec):
            continue
        filtered.append(row)
    return filtered


def _exclude_column(column: str) -> bool:
    if column in IDENTIFIER_COLUMNS or column in LEAKAGE_COLUMNS:
        return True
    if column.startswith("label_"):
        return True
    if "resolution" in column and column not in ALLOWED_RESOLUTION_COLUMNS:
        return True
    if column.startswith("future_"):
        return True
    if "timestamp" in column or column.endswith("_iso"):
        return True
    return False


def _matrix(rows: Sequence[dict[str, Any]], feature_columns: Sequence[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([
            _numeric_or_nan(row.get(column))
            for column in feature_columns
        ])
    return matrix


def _target(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [int(row[PRIMARY_TARGET]) for row in rows]


def _chronological_split(rows: Sequence[dict[str, Any]]) -> int:
    split = int(len(rows) * 0.7)
    if split <= 0:
        return 1
    if split >= len(rows):
        return len(rows) - 1
    return split


def _evaluate_classifier(
    *,
    name: str,
    model: Any,
    x_test: Sequence[Sequence[float]],
    y_test: Sequence[int],
    future_returns: Sequence[float | None],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    predictions = [int(value) for value in model.predict(x_test)]
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = [float(row[1]) for row in model.predict_proba(x_test)]
        except Exception:
            probabilities = None
    report = {
        "model_name": name,
        "status": "ok",
        "row_count": len(y_test),
        "positive_rate": _positive_rate(y_test),
        "prediction_positive_rate": _positive_rate(predictions),
        "accuracy": round(float(metrics["accuracy_score"](y_test, predictions)), 6),
        "balanced_accuracy": round(
            float(metrics["balanced_accuracy_score"](y_test, predictions)),
            6,
        ),
        "precision": round(
            float(metrics["precision_score"](y_test, predictions, zero_division=0)),
            6,
        ),
        "recall": round(
            float(metrics["recall_score"](y_test, predictions, zero_division=0)),
            6,
        ),
        "f1": round(
            float(metrics["f1_score"](y_test, predictions, zero_division=0)),
            6,
        ),
        "confusion_matrix": metrics["confusion_matrix"](
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "roc_auc": None,
        "future_return_by_predicted_class": _future_return_by_class(
            predictions,
            future_returns,
        ),
    }
    if probabilities is not None and len(set(y_test)) == 2:
        report["roc_auc"] = round(
            float(metrics["roc_auc_score"](y_test, probabilities)),
            6,
        )
    return report


def _future_return_by_class(
    predictions: Sequence[int],
    future_returns: Sequence[float | None],
) -> dict[str, float | None]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for predicted, future_return in zip(predictions, future_returns):
        if future_return is None:
            continue
        grouped[int(predicted)].append(float(future_return))
    return {
        str(label): _mean(grouped.get(label, []))
        for label in (0, 1)
    }


def _best_model(model_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        report
        for report in model_reports.values()
        if report.get("status") == "ok"
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda report: (
            float(report.get("balanced_accuracy") or 0.0),
            float(report.get("accuracy") or 0.0),
        ),
    )


def _write_summary_artifacts(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    feature_columns: Sequence[str],
    existing_artifacts: Sequence[str] = (),
) -> list[str]:
    metrics_path = output_dir / "baseline_metrics.json"
    feature_path = output_dir / "feature_columns.json"
    summary_path = output_dir / "training_summary.txt"
    artifacts = [
        *[str(path) for path in existing_artifacts],
        str(metrics_path),
        str(feature_path),
        str(summary_path),
    ]
    summary_to_write = {**summary, "artifacts_written": artifacts}
    metrics_path.write_text(
        json.dumps(summary_to_write, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feature_path.write_text(
        json.dumps(list(feature_columns), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(_render_summary_text(summary_to_write), encoding="utf-8")
    return [str(metrics_path), str(feature_path), str(summary_path)]


def _render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Baseline Training Summary",
        f"status={summary.get('status')}",
        f"input_path={summary.get('input_path')}",
        f"rows_loaded={summary.get('rows_loaded')}",
        f"rows_after_filters={summary.get('rows_after_filters')}",
        f"train_rows={summary.get('train_rows')}",
        f"test_rows={summary.get('test_rows')}",
        f"feature_count={summary.get('feature_count')}",
        f"best_model_name={summary.get('best_model_name')}",
        f"best_model_accuracy={summary.get('best_model_accuracy')}",
        f"best_model_balanced_accuracy={summary.get('best_model_balanced_accuracy')}",
    ]
    return "\n".join(lines) + "\n"


def _insufficient_rows_summary(
    *,
    input_path: str,
    output_dir: str,
    rows_loaded: int,
    rows_after_filters: int,
    min_rows: int,
    reason: str = "rows_after_filters_below_min_rows",
) -> dict[str, Any]:
    return {
        "status": "insufficient_rows",
        "reason": reason,
        "input_path": input_path,
        "output_dir": output_dir,
        "rows_loaded": rows_loaded,
        "rows_after_filters": rows_after_filters,
        "min_rows": min_rows,
        "train_rows": 0,
        "test_rows": 0,
        "feature_count": 0,
        "target_positive_rate": None,
        "best_model_name": None,
        "best_model_accuracy": None,
        "best_model_balanced_accuracy": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _time_range(rows: Sequence[dict[str, Any]]) -> dict[str, str | None]:
    timestamps = [str(row.get("timestamp")) for row in rows if row.get("timestamp")]
    return {
        "start": min(timestamps) if timestamps else None,
        "end": max(timestamps) if timestamps else None,
    }


def _positive_rate(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return round(sum(int(value) for value in values) / len(values), 6)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 10)


def _numeric_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    return False


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BaselineTrainingError",
    "PRIMARY_TARGET",
    "RETURN_TARGET",
    "select_feature_columns",
    "train_baseline_model",
]
