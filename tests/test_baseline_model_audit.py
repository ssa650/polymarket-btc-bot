from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from src.baseline_model_audit import audit_baseline_model, suspicious_feature_scan
from src.train_baseline_model import train_baseline_model


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _write_parquet(path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(idx: int, *, market_size: int = 20) -> dict:
    market_idx = idx // market_size
    start = BASE + timedelta(minutes=5 * market_idx)
    ts = start + timedelta(seconds=idx % market_size)
    close = start + timedelta(minutes=5)
    label = idx % 2
    return {
        "run_id": "run_audit",
        "market_id": f"market_{market_idx}",
        "question": f"Bitcoin Up or Down - market {market_idx}",
        "start_time": start.isoformat(),
        "close_time": close.isoformat(),
        "timestamp": ts.isoformat(),
        "market_phase": "active",
        "yes_token_id": "yes",
        "no_token_id": "no",
        "export_row_usable": 1,
        "label_btc_up_at_resolution": label,
        "label_resolved_up_down": "UP" if label else "DOWN",
        "label_yes_win": label,
        "btc_price_at_resolution": 101.0 + label,
        "future_btc_return_to_resolution_from_feature": 0.01 if label else -0.01,
        "time_until_resolution": (close - ts).total_seconds(),
        "time_since_market_created": (ts - start).total_seconds(),
        "seconds_after_start": (ts - start).total_seconds(),
        "seconds_before_close": (close - ts).total_seconds(),
        "best_bid_yes": 0.45 + 0.01 * label,
        "best_ask_yes": 0.47 + 0.01 * label,
        "best_bid_no": 0.53 - 0.01 * label,
        "best_ask_no": 0.55 - 0.01 * label,
        "spread_yes": 0.02,
        "spread_no": 0.02,
        "mid_price_yes": 0.46 + 0.01 * label,
        "mid_price_no": 0.54 - 0.01 * label,
        "last_trade_price": 0.46 + 0.01 * label,
        "last_trade_size": 3.0 + label,
        "last_trade_side": "BUY",
        "has_orderbook": 1,
        "has_trade_data": 1,
        "price_change_1s": 0.001 * label,
        "price_change_10s": 0.002 * label,
        "price_change_60s": 0.003 * label,
        "velocity_5s": 0.004 * label,
        "velocity_30s": 0.005 * label,
        "acceleration_5s": 0.006 * label,
        "rolling_mean_60s": 0.46,
        "distance_from_rolling_mean_60s": 0.01 * label,
        "total_bid_liquidity_yes": 10.0 + idx,
        "total_ask_liquidity_yes": 12.0 + idx,
        "liquidity_change_bid_5s": 0.5 * label,
        "orderbook_imbalance_yes": 0.1 * label,
        "buy_volume_5s": 2.0 + label,
        "sell_volume_5s": 1.0,
        "net_trade_flow_5s": 1.0 * label,
        "trade_flow_ratio_5s": 0.6,
        "rolling_volatility_10s": 0.001,
        "rolling_volatility_60s": 0.002,
        "volume_delta_1s": 0.5,
        "avg_volume_60s": 5.0,
        "volume_spike_ratio": 1.1,
        "largest_bid_wall_size_yes": 20.0,
        "largest_ask_wall_size_yes": 21.0,
        "distance_to_bid_wall": 0.01,
        "is_price_jump": label,
        "feature_ready": 1,
        "is_gap_affected": 0,
        "snapshot_quality_status": "ok",
        "strict_validation_passed": 1,
        "btc_chainlink_price": 100.0 + idx * 0.01,
        "btc_chainlink_age_sec_at_feature": 1.0,
        "btc_binance_price": 100.1 + idx * 0.01,
        "btc_binance_age_sec_at_feature": 1.0,
        "btc_price_diff_binance_minus_chainlink": 0.1,
        "btc_price_diff_pct_binance_minus_chainlink": 0.001,
        "btc_chainlink_exchange_timestamp": ts.isoformat(),
        "btc_chainlink_local_arrival_iso": ts.isoformat(),
        "btc_binance_exchange_timestamp": ts.isoformat(),
        "btc_binance_local_arrival_iso": ts.isoformat(),
    }


def _trained_fixture(tmp_path):
    input_path = tmp_path / "training.parquet"
    model_dir = tmp_path / "baseline"
    _write_parquet(input_path, [_row(idx) for idx in range(80)])
    train_baseline_model(
        input_path=str(input_path),
        output_dir=str(model_dir),
        min_rows=20,
    )
    return input_path, model_dir


def test_suspicious_scan_passes_clean_numeric_features() -> None:
    rows = [_row(0), _row(1)]

    suspicious = suspicious_feature_scan(
        feature_columns=["best_bid_yes", "btc_chainlink_price", "spread_yes"],
        rows=rows,
    )

    assert suspicious == []


def test_suspicious_scan_flags_leakage_and_object_columns() -> None:
    rows = [_row(0), _row(1)]

    suspicious = suspicious_feature_scan(
        feature_columns=[
            "future_btc_return_to_resolution_from_feature",
            "label_btc_up_at_resolution",
            "time_until_resolution",
            "market_phase",
        ],
        rows=rows,
    )
    by_column = {item["column"]: item["reasons"] for item in suspicious}

    assert "substring:future" in by_column["future_btc_return_to_resolution_from_feature"]
    assert "substring:label" in by_column["label_btc_up_at_resolution"]
    assert "substring:resolution" in by_column["time_until_resolution"]
    assert "non_numeric_or_object_column" in by_column["market_phase"]


def test_audit_baseline_model_writes_artifacts_and_metrics(tmp_path) -> None:
    input_path, model_dir = _trained_fixture(tmp_path)
    output_dir = tmp_path / "audit"

    report = audit_baseline_model(
        input_path=str(input_path),
        model_dir=str(model_dir),
        output_dir=str(output_dir),
    )

    assert report["status"] == "ok"
    assert report["dataset_shape"]["rows_loaded"] == 80
    assert report["dataset_shape"]["rows_after_filters"] == 80
    assert report["dataset_shape"]["train_rows"] == 56
    assert report["dataset_shape"]["test_rows"] == 24
    assert report["models_loaded"] == ["logistic_regression", "random_forest"]
    assert "logistic_regression" in report["model_metrics"]
    assert "random_forest" in report["model_metrics"]
    assert report["per_market_test_performance"]
    assert report["confidence_buckets"]["logistic_regression"]
    assert any(
        item["column"] == "time_until_resolution"
        for item in report["suspicious_feature_columns"]
    )
    assert (output_dir / "audit_metrics.json").exists()
    assert (output_dir / "audit_summary.txt").exists()
    assert (output_dir / "random_forest_feature_importance.csv").exists()
    assert (output_dir / "logistic_regression_coefficients.csv").exists()
    persisted = json.loads((output_dir / "audit_metrics.json").read_text())
    assert persisted["status"] == "ok"
    assert persisted["artifacts_written"]


def test_audit_baseline_model_split_by_market_metadata(tmp_path) -> None:
    input_path, model_dir = _trained_fixture(tmp_path)
    output_dir = tmp_path / "audit_market"

    report = audit_baseline_model(
        input_path=str(input_path),
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        split_by_market=True,
    )

    assert report["split_mode"] == "market_holdout"
    assert report["split_metadata"]["split_by_market"] is True
    assert report["split_metadata"]["market_count"] == 4
    assert report["split_metadata"]["train_market_count"] == 2
    assert report["split_metadata"]["test_market_count"] == 2
    assert report["dataset_shape"]["train_test_market_overlap_count"] == 0
    assert report["market_holdout_degradation"]


def test_audit_baseline_model_cli_is_offline(monkeypatch, tmp_path, capsys) -> None:
    input_path, model_dir = _trained_fixture(tmp_path)
    output_dir = tmp_path / "audit_cli"

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("audit-baseline-model must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "audit-baseline-model",
            "--input",
            str(input_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--split-by-market",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "SQLiteStore", forbidden)

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["split_mode"] == "market_holdout"
    assert payload["output_dir"] == str(output_dir)
    assert (output_dir / "audit_metrics.json").exists()
