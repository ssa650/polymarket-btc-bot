from __future__ import annotations

import json
import sqlite3
import sys

from src.live_model_audit import audit_live_model_predictions
from tests.test_baseline_paper_trader import _fixture, _insert_candidate


def _build_audit_fixture(tmp_path):
    recorder_db, _paper_db, model_path, features_path = _fixture(tmp_path)
    _insert_candidate(
        recorder_db,
        market_id="up_good",
        signal=0.8,
        resolved=1,
        winning_outcome="Up",
        time_until_resolution=75,
    )
    _insert_candidate(
        recorder_db,
        market_id="down_good",
        signal=0.2,
        resolved=1,
        winning_outcome="Down",
        time_until_resolution=45,
    )
    _insert_candidate(
        recorder_db,
        market_id="up_bad",
        signal=0.3,
        resolved=1,
        winning_outcome="Up",
        time_until_resolution=15,
    )
    _insert_candidate(
        recorder_db,
        market_id="unresolved_skip",
        signal=0.9,
        resolved=0,
        winning_outcome=None,
        time_until_resolution=90,
    )
    return recorder_db, model_path, features_path


def test_live_model_audit_maps_up_down_labels_and_skips_unresolved(tmp_path) -> None:
    recorder_db, model_path, features_path = _build_audit_fixture(tmp_path)

    report = audit_live_model_predictions(
        db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        output_path=str(tmp_path / "audit.json"),
        output_csv_path=str(tmp_path / "predictions.csv"),
    )

    overall = report["overall"]
    assert overall["row_count"] == 3
    assert overall["market_count"] == 3
    assert overall["up_row_count"] == 2
    assert overall["down_row_count"] == 1
    assert overall["accuracy_at_threshold_0_50"] == round(2 / 3, 10)
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "predictions.csv").exists()


def test_live_model_audit_bucket_metrics(tmp_path) -> None:
    recorder_db, model_path, features_path = _build_audit_fixture(tmp_path)

    report = audit_live_model_predictions(
        db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        output_path=str(tmp_path / "audit.json"),
    )

    buckets = report["calibration_buckets"]
    assert buckets["0.8-0.9"]["rows"] == 1
    assert buckets["0.8-0.9"]["actual_up_rate"] == 1.0
    assert buckets["0.2-0.3"]["rows"] == 1
    assert buckets["0.3-0.4"]["rows"] == 1
    time_buckets = report["time_until_resolution_buckets"]
    assert time_buckets["0-30s"]["row_count"] == 1
    assert time_buckets["30-60s"]["row_count"] == 1
    assert time_buckets["60-120s"]["row_count"] == 1


def test_live_model_audit_market_level_aggregation(tmp_path) -> None:
    recorder_db, model_path, features_path = _build_audit_fixture(tmp_path)

    report = audit_live_model_predictions(
        db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        output_path=str(tmp_path / "audit.json"),
    )

    markets = {row["market_id"]: row for row in report["markets"]}
    assert markets["up_good"]["winning_outcome"] == "Up"
    assert markets["up_good"]["final_prediction_by_avg_prob"] == "Up"
    assert markets["up_good"]["market_level_correct"] == 1
    assert markets["down_good"]["final_prediction_by_avg_prob"] == "Down"
    assert markets["up_bad"]["market_level_correct"] == 0


def test_live_model_audit_query_is_read_only(tmp_path) -> None:
    recorder_db, model_path, features_path = _build_audit_fixture(tmp_path)
    before = _counts(recorder_db)

    audit_live_model_predictions(
        db_path=str(recorder_db),
        model_path=str(model_path),
        feature_columns_path=str(features_path),
        output_path=str(tmp_path / "audit.json"),
    )

    assert _counts(recorder_db) == before


def test_audit_live_model_predictions_cli_smoke(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "audit-live-model-predictions",
            "--db",
            "data/recorder.db",
            "--model-path",
            "data/models/baseline_latest/model_logistic_regression.joblib",
            "--feature-columns",
            "data/models/baseline_latest/feature_columns.json",
            "--output",
            "data/analytics/model_audit/latest_model_audit.json",
            "--output-csv",
            "data/analytics/model_audit/latest_model_predictions.csv",
        ],
    )

    args = main.parse_args()

    assert args.command == "audit-live-model-predictions"
    assert args.output == "data/analytics/model_audit/latest_model_audit.json"
    assert args.output_csv == "data/analytics/model_audit/latest_model_predictions.csv"


def _counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("markets", "market_snapshots", "features", "btc_prices")
        }
    finally:
        conn.close()
