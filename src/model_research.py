from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .baseline_paper_trader import connect_recorder_read_only
from .live_model_audit import (
    _accuracy,
    _brier_score,
    _build_audit_report,
    _calibration_buckets,
    _fetch_resolved_feature_rows,
    _log_loss,
    _market_reports,
    _missing_feature_columns,
    _precision,
    _prediction_row,
    _recall,
    _roc_auc,
    _time_buckets,
)
from .models import parse_timestamp, to_iso
from .backtest_baseline_strategy import (
    _load_feature_columns,
    _load_model,
    _predict_probabilities,
)
from .train_baseline_model import _float_or_none, _is_numeric, _matrix


RESEARCH_MODEL_IDS: tuple[str, ...] = (
    "logistic_regression_default",
    "logistic_regression_balanced_class_weight",
    "random_forest_small",
    "gradient_boosting_classifier",
    "histogram_gradient_boosting_classifier",
    "calibrated_logistic_sigmoid",
    "calibrated_logistic_isotonic",
)

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
    "winning_asset_id",
    "winning_outcome",
}
LEAKAGE_COLUMNS = {
    "actual_yes",
    "resolved",
    "resolved_at",
    "feature_age_sec",
    "predicted_probability_yes",
    "predicted_label_yes",
    "prediction_correct",
}
ALLOWED_RESOLUTION_COLUMNS = {"time_until_resolution"}


class ModelResearchError(RuntimeError):
    pass


def research_model_candidates(
    *,
    db_path: str,
    output_dir: str,
    min_markets: int = 20,
    test_market_fraction: float = 0.25,
    random_seed: int = 42,
    baseline_audit_path: str = "data/analytics/model_audit/latest_model_audit.json",
) -> dict[str, Any]:
    deps = _load_research_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conn = connect_recorder_read_only(db_path)
    try:
        rows = _fetch_resolved_feature_rows(conn)
    finally:
        conn.close()
    split = _market_level_split(rows, test_market_fraction=test_market_fraction)
    market_count = len(split["all_markets"])
    if market_count < int(min_markets):
        report = {
            "status": "insufficient_markets",
            "reason": "resolved_market_count_below_min_markets",
            "db_path": db_path,
            "output_dir": str(output),
            "row_count": len(rows),
            "market_count": market_count,
            "min_markets": int(min_markets),
            "test_market_fraction": float(test_market_fraction),
            "random_seed": int(random_seed),
            "warnings": ["too_few_resolved_markets"],
            "generated_at": to_iso(datetime.now(timezone.utc)),
        }
        _write_research_report(report, output)
        return report
    feature_columns = _select_research_feature_columns(rows)
    if not feature_columns:
        report = {
            "status": "insufficient_features",
            "reason": "no_numeric_non_leakage_features",
            "db_path": db_path,
            "output_dir": str(output),
            "row_count": len(rows),
            "market_count": market_count,
            "warnings": ["no_feature_columns"],
            "generated_at": to_iso(datetime.now(timezone.utc)),
        }
        _write_research_report(report, output)
        return report
    train_rows = split["train_rows"]
    test_rows = split["test_rows"]
    x_train = _matrix(train_rows, feature_columns)
    y_train = _labels(train_rows)
    x_test = _matrix(test_rows, feature_columns)
    y_test = _labels(test_rows)
    candidates = _candidate_models(
        deps=deps,
        random_seed=int(random_seed),
        train_rows=len(train_rows),
        y_train=y_train,
    )
    model_reports: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    trained_model_ids: list[str] = []
    for model_id, model, skip_reason in candidates:
        if skip_reason:
            model_reports[model_id] = {
                "model_id": model_id,
                "status": "skipped",
                "reason": skip_reason,
            }
            warnings.append(f"{model_id}_skipped_{skip_reason}")
            continue
        try:
            model.fit(x_train, y_train)
            metrics = _evaluate_model(
                model_id=model_id,
                model=model,
                rows=test_rows,
                x_test=x_test,
                y_test=y_test,
            )
            model_reports[model_id] = metrics
            _write_model_artifacts(
                deps=deps,
                output_dir=output / model_id,
                model=model,
                feature_columns=feature_columns,
                metrics=metrics,
            )
            trained_model_ids.append(model_id)
        except Exception as exc:
            model_reports[model_id] = {
                "model_id": model_id,
                "status": "error",
                "error": str(exc),
            }
            warnings.append(f"{model_id}_error")
    baseline = _load_baseline_audit(baseline_audit_path)
    recommendation = _recommend_model(model_reports, baseline)
    if recommendation.get("recommended_model_id"):
        _copy_best_candidate(output, recommendation["recommended_model_id"])
    report = {
        "status": "ok",
        "db_path": db_path,
        "output_dir": str(output),
        "row_count": len(rows),
        "market_count": market_count,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_market_count": len(split["train_markets"]),
        "test_market_count": len(split["test_markets"]),
        "train_test_market_overlap_count": len(set(split["train_markets"]) & set(split["test_markets"])),
        "test_market_fraction": float(test_market_fraction),
        "random_seed": int(random_seed),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "baseline_audit_path": baseline_audit_path,
        "baseline_metrics": baseline,
        "model_metrics": model_reports,
        "trained_model_ids": trained_model_ids,
        "recommended_model_id": recommendation.get("recommended_model_id"),
        "recommendation": recommendation,
        "warnings": warnings + recommendation.get("warnings", []),
        "generated_at": to_iso(datetime.now(timezone.utc)),
    }
    _write_research_report(report, output)
    _write_comparison_csv(output / "model_comparison.csv", model_reports)
    return report


def promote_research_model_candidate(
    *,
    research_dir: str,
    model_id: str,
    output_dir: str,
    force: bool = False,
) -> dict[str, Any]:
    research_path = Path(research_dir)
    report_path = research_path / "research_report.json"
    if not report_path.exists():
        raise ModelResearchError(f"research_report.json not found in {research_dir}")
    try:
        research_report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelResearchError(f"research_report.json is not valid JSON: {exc}") from exc
    model_metrics = research_report.get("model_metrics")
    if not isinstance(model_metrics, dict):
        raise ModelResearchError("research_report.json does not contain model_metrics")
    metrics = model_metrics.get(model_id)
    if not isinstance(metrics, dict):
        raise ModelResearchError(f"model_id {model_id!r} was not found in research_report.json")
    if metrics.get("status") != "ok":
        raise ModelResearchError(
            f"model_id {model_id!r} is not promotable because status={metrics.get('status')!r}"
        )
    source_dir = research_path / model_id
    if not source_dir.is_dir():
        raise ModelResearchError(f"model artifact directory not found: {source_dir}")
    required_files = ("model.joblib", "feature_columns.json")
    missing = [name for name in required_files if not (source_dir / name).exists()]
    if missing:
        raise ModelResearchError(
            f"model artifact directory {source_dir} is missing required files: {', '.join(missing)}"
        )
    destination = Path(output_dir)
    if destination.exists():
        if not force:
            raise ModelResearchError(
                f"output directory already exists: {destination}. Pass --force to replace it."
            )
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copytree(source_dir, destination)
    promoted_at = to_iso(datetime.now(timezone.utc))
    metadata = {
        "status": "ok",
        "source_research_dir": str(research_path),
        "source_research_report": str(report_path),
        "model_id": model_id,
        "output_dir": str(destination),
        "promoted_at": promoted_at,
        "metrics": metrics,
        "baseline_metrics": research_report.get("baseline_metrics"),
        "research_recommendation": research_report.get("recommendation"),
        "warning": (
            "This directory contains a promoted candidate model for paper/live "
            "evaluation. It is not the production baseline and baseline_latest was not modified."
        ),
    }
    (destination / "promotion_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return metadata


def compare_model_artifacts(
    *,
    model_a_dir: str,
    model_b_dir: str,
    db_path: str,
    output_path: str,
) -> dict[str, Any]:
    output = Path(output_path)
    model_a = _audit_model_artifact_dir(
        model_dir=Path(model_a_dir),
        db_path=db_path,
        label="model_a",
    )
    model_b = _audit_model_artifact_dir(
        model_dir=Path(model_b_dir),
        db_path=db_path,
        label="model_b",
    )
    metrics_a = _extract_live_audit_metrics(model_a["audit_report"])
    metrics_b = _extract_live_audit_metrics(model_b["audit_report"])
    report = {
        "status": "ok",
        "db_path": db_path,
        "output_path": str(output),
        "generated_at": to_iso(datetime.now(timezone.utc)),
        "model_a": {
            "model_dir": str(Path(model_a_dir)),
            "model_path": str(model_a["model_path"]),
            "feature_columns_path": str(model_a["feature_columns_path"]),
            "metrics": metrics_a,
            "warnings": model_a["audit_report"].get("warnings", []),
        },
        "model_b": {
            "model_dir": str(Path(model_b_dir)),
            "model_path": str(model_b["model_path"]),
            "feature_columns_path": str(model_b["feature_columns_path"]),
            "metrics": metrics_b,
            "warnings": model_b["audit_report"].get("warnings", []),
        },
        "comparison": _compare_metric_sets(metrics_a, metrics_b),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


def render_model_artifact_comparison(report: dict[str, Any]) -> str:
    model_a = report.get("model_a") or {}
    model_b = report.get("model_b") or {}
    metrics_a = model_a.get("metrics") or {}
    metrics_b = model_b.get("metrics") or {}
    comparison = report.get("comparison") or {}
    lines = [
        f"status={report.get('status')}",
        f"output_path={report.get('output_path')}",
        "",
        "model_a:",
        f"  dir={model_a.get('model_dir')}",
        f"  accuracy={metrics_a.get('accuracy_at_0_5')} "
        f"market_accuracy={metrics_a.get('market_level_accuracy')} "
        f"log_loss={metrics_a.get('log_loss')} "
        f"brier={metrics_a.get('brier_score')} "
        f"roc_auc={metrics_a.get('roc_auc')}",
        "model_b:",
        f"  dir={model_b.get('model_dir')}",
        f"  accuracy={metrics_b.get('accuracy_at_0_5')} "
        f"market_accuracy={metrics_b.get('market_level_accuracy')} "
        f"log_loss={metrics_b.get('log_loss')} "
        f"brier={metrics_b.get('brier_score')} "
        f"roc_auc={metrics_b.get('roc_auc')}",
        "",
        "delta_b_minus_a:",
    ]
    for key in (
        "accuracy_at_0_5",
        "market_level_accuracy",
        "log_loss",
        "brier_score",
        "roc_auc",
    ):
        lines.append(f"  {key}={comparison.get(key, {}).get('delta_b_minus_a')}")
    return "\n".join(lines) + "\n"


def _load_research_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ModelResearchError(
            "Model research requires scikit-learn and joblib. Install project dependencies first."
        ) from exc
    return {
        "joblib": joblib,
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def _candidate_models(
    *,
    deps: dict[str, Any],
    random_seed: int,
    train_rows: int,
    y_train: Sequence[int],
) -> list[tuple[str, Any | None, str | None]]:
    min_class_count = min(Counter(int(label) for label in y_train).values() or [0])
    if len(set(y_train)) < 2:
        return [(model_id, None, "train_split_has_single_class") for model_id in RESEARCH_MODEL_IDS]
    logistic_default = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["StandardScaler"](),
        deps["LogisticRegression"](max_iter=1000, random_state=random_seed),
    )
    logistic_balanced = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["StandardScaler"](),
        deps["LogisticRegression"](
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",
        ),
    )
    random_forest = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["RandomForestClassifier"](
            n_estimators=80,
            random_state=random_seed,
            min_samples_leaf=2,
            max_depth=5,
        ),
    )
    gradient_boosting = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["GradientBoostingClassifier"](
            random_state=random_seed,
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
        ),
    )
    hist_gradient = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["HistGradientBoostingClassifier"](
            random_state=random_seed,
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
        ),
    )
    result: list[tuple[str, Any | None, str | None]] = [
        ("logistic_regression_default", logistic_default, None),
        ("logistic_regression_balanced_class_weight", logistic_balanced, None),
        ("random_forest_small", random_forest, None),
        ("gradient_boosting_classifier", gradient_boosting, None),
        ("histogram_gradient_boosting_classifier", hist_gradient, None),
    ]
    cv = min(3, int(min_class_count))
    if cv >= 2:
        result.append(
            (
                "calibrated_logistic_sigmoid",
                _calibrated_model(
                    deps,
                    method="sigmoid",
                    cv=cv,
                    random_seed=random_seed,
                ),
                None,
            )
        )
    else:
        result.append(("calibrated_logistic_sigmoid", None, "insufficient_class_counts_for_cv"))
    if train_rows >= 200 and cv >= 3:
        result.append(
            (
                "calibrated_logistic_isotonic",
                _calibrated_model(
                    deps,
                    method="isotonic",
                    cv=cv,
                    random_seed=random_seed,
                ),
                None,
            )
        )
    else:
        result.append(("calibrated_logistic_isotonic", None, "insufficient_rows_for_isotonic"))
    return result


def _calibrated_model(
    deps: dict[str, Any],
    *,
    method: str,
    cv: int,
    random_seed: int,
) -> Any:
    estimator = deps["make_pipeline"](
        deps["SimpleImputer"](strategy="median"),
        deps["StandardScaler"](),
        deps["LogisticRegression"](max_iter=1000, random_state=random_seed),
    )
    cls = deps["CalibratedClassifierCV"]
    try:
        return cls(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return cls(base_estimator=estimator, method=method, cv=cv)


def _market_level_split(
    rows: Sequence[dict[str, Any]],
    *,
    test_market_fraction: float,
) -> dict[str, Any]:
    market_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        market_rows[(str(row.get("run_id") or ""), str(row.get("market_id") or ""))].append(row)
    ordered_markets = sorted(
        market_rows,
        key=lambda key: _market_sort_key(market_rows[key]),
    )
    if not ordered_markets:
        return {
            "all_markets": [],
            "train_markets": [],
            "test_markets": [],
            "train_rows": [],
            "test_rows": [],
        }
    test_count = max(1, int(math.ceil(len(ordered_markets) * float(test_market_fraction))))
    if test_count >= len(ordered_markets):
        test_count = max(1, len(ordered_markets) - 1)
    train_markets = ordered_markets[:-test_count]
    test_markets = ordered_markets[-test_count:]
    train_set = set(train_markets)
    test_set = set(test_markets)
    ordered_rows = sorted(rows, key=lambda row: str(row.get("timestamp") or ""))
    return {
        "all_markets": ordered_markets,
        "train_markets": train_markets,
        "test_markets": test_markets,
        "train_rows": [row for row in ordered_rows if (str(row.get("run_id") or ""), str(row.get("market_id") or "")) in train_set],
        "test_rows": [row for row in ordered_rows if (str(row.get("run_id") or ""), str(row.get("market_id") or "")) in test_set],
    }


def _market_sort_key(rows: Sequence[dict[str, Any]]) -> tuple[str, str, str]:
    first = rows[0]
    for column in ("close_time", "end_time", "start_time", "timestamp"):
        value = first.get(column)
        parsed = parse_timestamp(value)
        if parsed is not None:
            return (to_iso(parsed) or "", str(first.get("run_id") or ""), str(first.get("market_id") or ""))
    return ("", str(first.get("run_id") or ""), str(first.get("market_id") or ""))


def _select_research_feature_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    columns = sorted({column for row in rows for column in row})
    selected: list[str] = []
    for column in columns:
        if _exclude_research_column(column):
            continue
        non_null = [row.get(column) for row in rows if row.get(column) is not None]
        if non_null and all(_is_numeric(value) for value in non_null):
            selected.append(column)
    return selected


def _exclude_research_column(column: str) -> bool:
    lowered = column.lower()
    if column in IDENTIFIER_COLUMNS or column in LEAKAGE_COLUMNS:
        return True
    if column.startswith("label_") or column.startswith("future_"):
        return True
    if "winning" in lowered or "outcome" in lowered:
        return True
    if "resolution" in lowered and column not in ALLOWED_RESOLUTION_COLUMNS:
        return True
    if "timestamp" in lowered or column.endswith("_iso"):
        return True
    return False


def _evaluate_model(
    *,
    model_id: str,
    model: Any,
    rows: Sequence[dict[str, Any]],
    x_test: Sequence[Sequence[float]],
    y_test: Sequence[int],
) -> dict[str, Any]:
    probabilities = _probabilities_for_model(model, x_test)
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    prediction_rows = [
        {
            **row,
            "actual_yes": int(label),
            "predicted_probability_yes": round(float(probability), 10),
            "predicted_label_yes": int(prediction),
            "prediction_correct": int(int(label) == int(prediction)),
        }
        for row, label, probability, prediction in zip(rows, y_test, probabilities, predictions)
    ]
    market_reports = _market_reports(prediction_rows)
    market_accuracy = _market_level_accuracy(market_reports)
    wrong_markets = [row for row in market_reports if int(row.get("market_level_correct") or 0) == 0]
    return {
        "model_id": model_id,
        "status": "ok",
        "row_count": len(prediction_rows),
        "market_count": len({(str(row.get("run_id") or ""), str(row.get("market_id") or "")) for row in prediction_rows}),
        "accuracy_at_0_5": _accuracy(y_test, predictions),
        "roc_auc": _roc_auc(y_test, probabilities),
        "brier_score": _brier_score(y_test, probabilities),
        "log_loss": _log_loss(y_test, probabilities),
        "precision_up": _precision(y_test, predictions, positive_label=1),
        "recall_up": _recall(y_test, predictions, positive_label=1),
        "precision_down": _precision(y_test, predictions, positive_label=0),
        "recall_down": _recall(y_test, predictions, positive_label=0),
        "calibration_buckets": _calibration_buckets(prediction_rows),
        "time_until_resolution_buckets": _time_buckets(prediction_rows),
        "market_level_accuracy": market_accuracy,
        "market_metrics": market_reports,
        "wrong_markets": wrong_markets,
        "high_confidence_bucket_failure": _has_high_confidence_bucket_failure(prediction_rows),
    }


def _probabilities_for_model(model: Any, matrix: Sequence[Sequence[float]]) -> list[float]:
    probabilities = model.predict_proba(matrix)
    if getattr(probabilities, "shape", None) is not None and probabilities.shape[1] >= 2:
        return [float(row[1]) for row in probabilities]
    return [float(row[1]) for row in probabilities]


def _market_level_accuracy(market_reports: Sequence[dict[str, Any]]) -> float | None:
    if not market_reports:
        return None
    return round(
        sum(int(row.get("market_level_correct") or 0) for row in market_reports) / len(market_reports),
        10,
    )


def _has_high_confidence_bucket_failure(rows: Sequence[dict[str, Any]]) -> bool:
    buckets = _calibration_buckets(rows)
    for bucket in ("0.0-0.1", "0.1-0.2", "0.8-0.9", "0.9-1.0"):
        metrics = buckets.get(bucket) or {}
        if int(metrics.get("row_count") or 0) < 10:
            continue
        accuracy = metrics.get("accuracy")
        if accuracy is not None and float(accuracy) < 0.55:
            return True
    return False


def _labels(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [int(row["actual_yes"]) for row in rows]


def _load_baseline_audit(path: str) -> dict[str, Any] | None:
    audit_path = Path(path)
    if not audit_path.exists():
        return None
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    overall = payload.get("overall") if isinstance(payload, dict) else None
    markets = payload.get("markets") if isinstance(payload, dict) else None
    if not isinstance(overall, dict):
        return None
    market_accuracy = None
    if isinstance(markets, list) and markets:
        market_accuracy = round(
            sum(int(row.get("market_level_correct") or 0) for row in markets) / len(markets),
            10,
        )
    return {
        "accuracy_at_0_5": overall.get("accuracy_at_threshold_0_50"),
        "roc_auc": overall.get("roc_auc"),
        "brier_score": overall.get("brier_score"),
        "log_loss": overall.get("log_loss"),
        "market_level_accuracy": market_accuracy,
        "row_count": overall.get("row_count"),
        "market_count": overall.get("market_count"),
    }


def _recommend_model(
    model_reports: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = [
        report
        for report in model_reports.values()
        if report.get("status") == "ok"
    ]
    warnings: list[str] = []
    if not candidates:
        return {"recommended_model_id": None, "reason": "no_successful_models", "warnings": warnings}
    if baseline is None:
        best = min(candidates, key=lambda row: float(row.get("log_loss") or 999.0))
        warnings.append("baseline_audit_missing")
        return {
            "recommended_model_id": best["model_id"],
            "reason": "lowest_log_loss_without_baseline_comparison",
            "warnings": warnings,
        }
    baseline_log_loss = _float_or_none(baseline.get("log_loss"))
    baseline_brier = _float_or_none(baseline.get("brier_score"))
    baseline_market_accuracy = _float_or_none(baseline.get("market_level_accuracy"))
    baseline_row_accuracy = _float_or_none(baseline.get("accuracy_at_0_5"))
    eligible: list[dict[str, Any]] = []
    for report in candidates:
        log_loss = _float_or_none(report.get("log_loss"))
        brier = _float_or_none(report.get("brier_score"))
        market_accuracy = _float_or_none(report.get("market_level_accuracy"))
        row_accuracy = _float_or_none(report.get("accuracy_at_0_5"))
        if baseline_log_loss is not None and (log_loss is None or log_loss >= baseline_log_loss):
            continue
        if baseline_brier is not None and (brier is None or brier >= baseline_brier):
            continue
        if (
            baseline_market_accuracy is not None
            and (market_accuracy is None or market_accuracy < baseline_market_accuracy)
        ):
            continue
        if (
            baseline_row_accuracy is not None
            and (row_accuracy is None or row_accuracy < baseline_row_accuracy - 0.02)
        ):
            continue
        if report.get("high_confidence_bucket_failure"):
            continue
        eligible.append(report)
    if not eligible:
        return {
            "recommended_model_id": None,
            "reason": "no_candidate_cleared_baseline_guardrails",
            "warnings": warnings,
        }
    best = min(eligible, key=lambda row: (float(row.get("log_loss") or 999.0), float(row.get("brier_score") or 999.0)))
    return {
        "recommended_model_id": best["model_id"],
        "reason": "passed_baseline_guardrails",
        "warnings": warnings,
    }


def _write_model_artifacts(
    *,
    deps: dict[str, Any],
    output_dir: Path,
    model: Any,
    feature_columns: Sequence[str],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    deps["joblib"].dump(model, output_dir / "model.joblib")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(feature_columns), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _copy_best_candidate(output_dir: Path, model_id: str) -> None:
    source = output_dir / model_id
    target = output_dir / "best_candidate"
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(source.name, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def _write_research_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (output_dir / "research_report.txt").write_text(
        _render_report_text(report),
        encoding="utf-8",
    )


def _render_report_text(report: dict[str, Any]) -> str:
    lines = [
        f"status={report.get('status')}",
        f"row_count={report.get('row_count')}",
        f"market_count={report.get('market_count')}",
        f"train_markets={report.get('train_market_count')}",
        f"test_markets={report.get('test_market_count')}",
        f"recommended_model_id={report.get('recommended_model_id')}",
        "",
        "models:",
    ]
    for model_id, metrics in sorted((report.get("model_metrics") or {}).items()):
        lines.append(
            f"- {model_id}: status={metrics.get('status')} "
            f"acc={metrics.get('accuracy_at_0_5')} "
            f"market_acc={metrics.get('market_level_accuracy')} "
            f"log_loss={metrics.get('log_loss')} "
            f"brier={metrics.get('brier_score')}"
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _write_comparison_csv(path: Path, model_reports: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "model_id",
        "status",
        "row_count",
        "market_count",
        "accuracy_at_0_5",
        "market_level_accuracy",
        "roc_auc",
        "brier_score",
        "log_loss",
        "precision_up",
        "recall_up",
        "precision_down",
        "recall_down",
        "high_confidence_bucket_failure",
        "reason",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_id, metrics in sorted(model_reports.items()):
            writer.writerow({field: metrics.get(field) for field in fieldnames} | {"model_id": model_id})


def _audit_model_artifact_dir(
    *,
    model_dir: Path,
    db_path: str,
    label: str,
) -> dict[str, Any]:
    model_path, feature_columns_path = _resolve_model_artifact_paths(model_dir)
    try:
        model = _load_model(str(model_path))
        feature_columns = _load_feature_columns(str(feature_columns_path))
    except Exception as exc:
        raise ModelResearchError(f"{label} artifact load failed: {exc}") from exc
    conn = connect_recorder_read_only(db_path)
    try:
        rows = _fetch_resolved_feature_rows(conn)
    finally:
        conn.close()
    missing_columns = _missing_feature_columns(rows, feature_columns)
    if missing_columns:
        raise ModelResearchError(
            f"{label} is missing resolved feature columns required by {feature_columns_path}: "
            + ", ".join(missing_columns)
        )
    probabilities = _predict_probabilities(model, rows, feature_columns)
    predictions = [
        _prediction_row(row, probability=probability)
        for row, probability in zip(rows, probabilities)
    ]
    audit_report = _build_audit_report(
        predictions,
        db_path=db_path,
        model_path=str(model_path),
        feature_columns_path=str(feature_columns_path),
        output_path="",
        output_csv_path=None,
    )
    return {
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "audit_report": audit_report,
    }


def _resolve_model_artifact_paths(model_dir: Path) -> tuple[Path, Path]:
    if not model_dir.is_dir():
        raise ModelResearchError(f"model directory not found: {model_dir}")
    feature_columns_path = model_dir / "feature_columns.json"
    if not feature_columns_path.exists():
        raise ModelResearchError(f"feature_columns.json not found in {model_dir}")
    model_candidates = [
        model_dir / "model.joblib",
        model_dir / "model_logistic_regression.joblib",
        model_dir / "model_random_forest.joblib",
    ]
    for candidate in model_candidates:
        if candidate.exists():
            return candidate, feature_columns_path
    joblib_files = sorted(model_dir.glob("*.joblib"))
    if len(joblib_files) == 1:
        return joblib_files[0], feature_columns_path
    if not joblib_files:
        raise ModelResearchError(f"no joblib model artifact found in {model_dir}")
    raise ModelResearchError(
        f"multiple joblib model artifacts found in {model_dir}; expected model.joblib "
        "or a known baseline model filename"
    )


def _extract_live_audit_metrics(report: dict[str, Any]) -> dict[str, Any]:
    overall = report.get("overall") or {}
    markets = report.get("markets") or []
    market_level_accuracy = None
    if isinstance(markets, list) and markets:
        market_level_accuracy = round(
            sum(int(row.get("market_level_correct") or 0) for row in markets) / len(markets),
            10,
        )
    return {
        "row_count": overall.get("row_count"),
        "market_count": overall.get("market_count"),
        "accuracy_at_0_5": overall.get("accuracy_at_threshold_0_50"),
        "market_level_accuracy": market_level_accuracy,
        "log_loss": overall.get("log_loss"),
        "brier_score": overall.get("brier_score"),
        "roc_auc": overall.get("roc_auc"),
        "precision_up": overall.get("precision_up"),
        "recall_up": overall.get("recall_up"),
        "precision_down": overall.get("precision_down"),
        "recall_down": overall.get("recall_down"),
    }


def _compare_metric_sets(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    higher_is_better = {
        "accuracy_at_0_5",
        "market_level_accuracy",
        "roc_auc",
        "precision_up",
        "recall_up",
        "precision_down",
        "recall_down",
    }
    lower_is_better = {"log_loss", "brier_score"}
    keys = [
        "accuracy_at_0_5",
        "market_level_accuracy",
        "log_loss",
        "brier_score",
        "roc_auc",
        "precision_up",
        "recall_up",
        "precision_down",
        "recall_down",
    ]
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        value_a = _float_or_none(metrics_a.get(key))
        value_b = _float_or_none(metrics_b.get(key))
        delta = None if value_a is None or value_b is None else round(value_b - value_a, 10)
        better = None
        if delta is not None:
            if key in higher_is_better:
                better = "model_b" if delta > 0 else "model_a" if delta < 0 else "tie"
            elif key in lower_is_better:
                better = "model_b" if delta < 0 else "model_a" if delta > 0 else "tie"
        result[key] = {
            "model_a": value_a,
            "model_b": value_b,
            "delta_b_minus_a": delta,
            "better": better,
        }
    return result


__all__ = [
    "ModelResearchError",
    "compare_model_artifacts",
    "promote_research_model_candidate",
    "render_model_artifact_comparison",
    "research_model_candidates",
]
