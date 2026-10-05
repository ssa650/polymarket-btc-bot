from __future__ import annotations

import json
import sys

import pytest

from src.model_research import (
    ModelResearchError,
    compare_model_artifacts,
    promote_research_model_candidate,
    research_model_candidates,
)
from tests.test_baseline_paper_trader import _fixture, _insert_candidate


def _research_fixture(tmp_path, *, market_count: int = 8):
    recorder_db, _paper_db, _model_path, _features_path = _fixture(tmp_path)
    for idx in range(market_count):
        label = idx % 2
        _insert_candidate(
            recorder_db,
            market_id=f"market_{idx:02d}",
            signal=0.85 if label == 1 else 0.15,
            resolved=1,
            winning_outcome="Up" if label == 1 else "Down",
            close_offset_sec=120 + idx,
            time_until_resolution=60 + idx,
        )
    return recorder_db


def test_model_research_market_level_split_prevents_leakage(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)

    report = research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(tmp_path / "research"),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )

    assert report["status"] == "ok"
    assert report["market_count"] == 8
    assert report["train_market_count"] == 6
    assert report["test_market_count"] == 2
    assert report["train_test_market_overlap_count"] == 0


def test_model_research_metrics_include_calibration_buckets_and_market_accuracy(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)

    report = research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(tmp_path / "research"),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )

    model = report["model_metrics"]["logistic_regression_default"]
    assert model["status"] == "ok"
    assert "0.7-0.8" in model["calibration_buckets"]
    assert "0-30s" in model["time_until_resolution_buckets"]
    assert model["market_level_accuracy"] is not None
    assert isinstance(model["wrong_markets"], list)


def test_model_research_handles_too_few_markets_gracefully(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=3)
    output_dir = tmp_path / "research"

    report = research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(output_dir),
        min_markets=20,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )

    assert report["status"] == "insufficient_markets"
    assert report["market_count"] == 3
    assert (output_dir / "research_report.json").exists()
    assert (output_dir / "research_report.txt").exists()


def test_model_research_writes_expected_files(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)
    output_dir = tmp_path / "research"

    report = research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(output_dir),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )

    assert report["status"] == "ok"
    assert (output_dir / "research_report.json").exists()
    assert (output_dir / "research_report.txt").exists()
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "logistic_regression_default" / "model.joblib").exists()
    assert (output_dir / "logistic_regression_default" / "feature_columns.json").exists()
    assert (output_dir / "logistic_regression_default" / "metrics.json").exists()
    assert (output_dir / "best_candidate").exists()


def test_research_model_candidates_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "research-model-candidates",
            "--db",
            "data/recorder.db",
            "--output-dir",
            "data/models/research_latest",
            "--min-markets",
            "20",
            "--test-market-fraction",
            "0.25",
            "--random-seed",
            "42",
        ],
    )

    args = main.parse_args()

    assert args.command == "research-model-candidates"
    assert args.output_dir == "data/models/research_latest"
    assert args.min_markets == 20
    assert args.test_market_fraction == 0.25


def test_promote_research_model_candidate_refuses_missing_report(tmp_path) -> None:
    with pytest.raises(ModelResearchError, match="research_report.json not found"):
        promote_research_model_candidate(
            research_dir=str(tmp_path / "missing_research"),
            model_id="logistic_regression_default",
            output_dir=str(tmp_path / "candidate"),
        )


def test_promote_research_model_candidate_refuses_unknown_model_id(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)
    research_dir = tmp_path / "research"
    research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(research_dir),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )

    with pytest.raises(ModelResearchError, match="was not found"):
        promote_research_model_candidate(
            research_dir=str(research_dir),
            model_id="not_a_model",
            output_dir=str(tmp_path / "candidate"),
        )


def test_promote_research_model_candidate_copies_expected_files(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)
    research_dir = tmp_path / "research"
    research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(research_dir),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )
    output_dir = tmp_path / "candidate"

    metadata = promote_research_model_candidate(
        research_dir=str(research_dir),
        model_id="logistic_regression_default",
        output_dir=str(output_dir),
    )

    assert metadata["status"] == "ok"
    assert metadata["model_id"] == "logistic_regression_default"
    assert "not the production baseline" in metadata["warning"]
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "feature_columns.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "promotion_metadata.json").exists()


def test_promote_research_model_candidate_refuses_overwrite_without_force(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)
    research_dir = tmp_path / "research"
    research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(research_dir),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )
    output_dir = tmp_path / "candidate"
    promote_research_model_candidate(
        research_dir=str(research_dir),
        model_id="logistic_regression_default",
        output_dir=str(output_dir),
    )

    with pytest.raises(ModelResearchError, match="already exists"):
        promote_research_model_candidate(
            research_dir=str(research_dir),
            model_id="logistic_regression_default",
            output_dir=str(output_dir),
        )

    metadata = promote_research_model_candidate(
        research_dir=str(research_dir),
        model_id="logistic_regression_default",
        output_dir=str(output_dir),
        force=True,
    )
    assert metadata["status"] == "ok"


def test_compare_model_artifacts_returns_both_model_metrics(tmp_path) -> None:
    recorder_db = _research_fixture(tmp_path, market_count=8)
    research_dir = tmp_path / "research"
    research_model_candidates(
        db_path=str(recorder_db),
        output_dir=str(research_dir),
        min_markets=4,
        test_market_fraction=0.25,
        baseline_audit_path=str(tmp_path / "missing_baseline.json"),
    )
    model_a_dir = tmp_path / "candidate_a"
    model_b_dir = tmp_path / "candidate_b"
    promote_research_model_candidate(
        research_dir=str(research_dir),
        model_id="logistic_regression_default",
        output_dir=str(model_a_dir),
    )
    promote_research_model_candidate(
        research_dir=str(research_dir),
        model_id="gradient_boosting_classifier",
        output_dir=str(model_b_dir),
    )

    report = compare_model_artifacts(
        model_a_dir=str(model_a_dir),
        model_b_dir=str(model_b_dir),
        db_path=str(recorder_db),
        output_path=str(tmp_path / "compare.json"),
    )

    assert report["status"] == "ok"
    assert report["model_a"]["metrics"]["row_count"] > 0
    assert report["model_b"]["metrics"]["row_count"] > 0
    assert "accuracy_at_0_5" in report["comparison"]
    assert "log_loss" in report["comparison"]
    assert (tmp_path / "compare.json").exists()


def test_model_promotion_and_comparison_cli_are_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "promote-research-model-candidate",
            "--research-dir",
            "data/models/research_latest",
            "--model-id",
            "gradient_boosting_classifier",
            "--output-dir",
            "data/models/candidate_gradient_boosting_latest",
        ],
    )
    args = main.parse_args()
    assert args.command == "promote-research-model-candidate"
    assert args.research_dir == "data/models/research_latest"
    assert args.model_id == "gradient_boosting_classifier"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "compare-model-artifacts",
            "--model-a-dir",
            "data/models/baseline_latest",
            "--model-b-dir",
            "data/models/candidate_gradient_boosting_latest",
            "--output",
            "data/analytics/model_compare/latest_compare.json",
        ],
    )
    args = main.parse_args()
    assert args.command == "compare-model-artifacts"
    assert args.model_a_dir == "data/models/baseline_latest"
    assert args.model_b_dir == "data/models/candidate_gradient_boosting_latest"
