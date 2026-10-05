from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from src.train_baseline_model import select_feature_columns, train_baseline_model


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
DEFAULT_LABEL = object()


def _write_parquet(path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(
    idx: int,
    *,
    usable: int = 1,
    label: int | None | object = DEFAULT_LABEL,
    btc_age: float = 1.0,
) -> dict:
    ts = BASE + timedelta(seconds=idx)
    resolved_label = idx % 2 if label is DEFAULT_LABEL else label
    signal = int(resolved_label or 0)
    return {
        "run_id": "run_train",
        "market_id": f"market_{idx // 20}",
        "question": "Bitcoin Up or Down",
        "start_time": BASE.isoformat(),
        "close_time": (BASE + timedelta(minutes=5)).isoformat(),
        "timestamp": ts.isoformat(),
        "market_phase": "active",
        "yes_token_id": "yes_token",
        "no_token_id": "no_token",
        "export_row_usable": usable,
        "label_btc_up_at_resolution": resolved_label,
        "label_resolved_up_down": (
            "UP" if resolved_label == 1 else ("DOWN" if resolved_label == 0 else None)
        ),
        "label_yes_win": resolved_label,
        "btc_price_at_resolution": 101.0 + signal,
        "future_btc_return_to_resolution_from_feature": 0.01 if signal == 1 else -0.01,
        "time_until_resolution": 300.0 - idx,
        "time_since_market_created": float(idx),
        "seconds_after_start": float(idx),
        "seconds_before_close": 300.0 - idx,
        "best_bid_yes": 0.45 + 0.01 * signal,
        "best_ask_yes": 0.47 + 0.01 * signal,
        "best_bid_no": 0.53 - 0.01 * signal,
        "best_ask_no": 0.55 - 0.01 * signal,
        "spread_yes": 0.02,
        "spread_no": 0.02,
        "mid_price_yes": 0.46 + 0.01 * signal,
        "mid_price_no": 0.54 - 0.01 * signal,
        "last_trade_price": 0.46 + 0.01 * signal,
        "last_trade_size": 3.0 + signal,
        "last_trade_side": "BUY",
        "has_orderbook": 1,
        "has_trade_data": 1,
        "price_change_1s": 0.001 * signal,
        "price_change_10s": 0.002 * signal,
        "price_change_60s": 0.003 * signal,
        "velocity_5s": 0.004 * signal,
        "velocity_30s": 0.005 * signal,
        "acceleration_5s": 0.006 * signal,
        "rolling_mean_60s": 0.46,
        "distance_from_rolling_mean_60s": 0.01 * signal,
        "total_bid_liquidity_yes": 10.0 + idx,
        "total_ask_liquidity_yes": 12.0 + idx,
        "liquidity_change_bid_5s": 0.5 * signal,
        "orderbook_imbalance_yes": 0.1 * signal,
        "buy_volume_5s": 2.0 + signal,
        "sell_volume_5s": 1.0,
        "net_trade_flow_5s": 1.0 * signal,
        "trade_flow_ratio_5s": 0.6,
        "rolling_volatility_10s": 0.001,
        "rolling_volatility_60s": 0.002,
        "volume_delta_1s": 0.5,
        "avg_volume_60s": 5.0,
        "volume_spike_ratio": 1.1,
        "largest_bid_wall_size_yes": 20.0,
        "largest_ask_wall_size_yes": 21.0,
        "distance_to_bid_wall": 0.01,
        "is_price_jump": signal,
        "feature_ready": 1,
        "is_gap_affected": 0,
        "snapshot_quality_status": "ok",
        "strict_validation_passed": 1,
        "btc_chainlink_price": 100.0 + idx * 0.01,
        "btc_chainlink_age_sec_at_feature": btc_age,
        "btc_binance_price": 100.1 + idx * 0.01,
        "btc_binance_age_sec_at_feature": btc_age,
        "btc_price_diff_binance_minus_chainlink": 0.1,
        "btc_price_diff_pct_binance_minus_chainlink": 0.001,
        "btc_chainlink_exchange_timestamp": ts.isoformat(),
        "btc_chainlink_local_arrival_iso": ts.isoformat(),
        "btc_binance_exchange_timestamp": ts.isoformat(),
        "btc_binance_local_arrival_iso": ts.isoformat(),
    }


def test_train_baseline_model_trains_and_writes_artifacts(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_dir = tmp_path / "models"
    _write_parquet(input_path, [_row(idx) for idx in range(60)])

    summary = train_baseline_model(
        input_path=str(input_path),
        output_dir=str(output_dir),
        min_rows=20,
    )

    assert summary["status"] == "ok"
    assert summary["rows_loaded"] == 60
    assert summary["rows_after_filters"] == 60
    assert summary["train_rows"] == 42
    assert summary["test_rows"] == 18
    assert summary["feature_count"] > 10
    assert summary["best_model_name"] in {
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "feature_columns.json").exists()
    assert (output_dir / "model_logistic_regression.joblib").exists()
    assert (output_dir / "model_random_forest.joblib").exists()
    assert (output_dir / "model_gradient_boosting.joblib").exists()
    assert (output_dir / "training_summary.txt").exists()
    metrics = json.loads((output_dir / "baseline_metrics.json").read_text())
    assert metrics["status"] == "ok"
    assert "dummy_most_frequent" in metrics["model_metrics"]
    assert "logistic_regression" in metrics["model_metrics"]
    assert "random_forest" in metrics["model_metrics"]
    assert "gradient_boosting" in metrics["model_metrics"]
    assert metrics["train_time_range"]["end"] < metrics["test_time_range"]["start"]


def test_train_baseline_model_filters_unusable_labels_and_stale_btc(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_dir = tmp_path / "models"
    rows = [_row(idx) for idx in range(20)]
    rows.extend(
        [
            _row(20, usable=0),
            _row(21, label=None),
            _row(22, btc_age=5.0),
        ]
    )
    _write_parquet(input_path, rows)

    summary = train_baseline_model(
        input_path=str(input_path),
        output_dir=str(output_dir),
        min_rows=10,
        max_btc_age_sec=3.0,
    )

    assert summary["status"] == "ok"
    assert summary["rows_loaded"] == 23
    assert summary["rows_after_filters"] == 20


def test_train_baseline_model_insufficient_rows_is_graceful(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_dir = tmp_path / "models"
    _write_parquet(input_path, [_row(idx) for idx in range(5)])

    summary = train_baseline_model(
        input_path=str(input_path),
        output_dir=str(output_dir),
        min_rows=200,
    )

    assert summary["status"] == "insufficient_rows"
    assert summary["rows_loaded"] == 5
    assert summary["rows_after_filters"] == 5
    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "feature_columns.json").exists()
    assert not (output_dir / "model_logistic_regression.joblib").exists()


def test_train_baseline_model_feature_selection_excludes_leakage() -> None:
    columns = select_feature_columns([_row(0), _row(1)])

    assert "time_until_resolution" in columns
    assert "btc_chainlink_price" in columns
    assert "btc_binance_price" in columns
    assert "btc_price_diff_binance_minus_chainlink" in columns
    assert "run_id" not in columns
    assert "market_id" not in columns
    assert "timestamp" not in columns
    assert "question" not in columns
    assert "export_row_usable" not in columns
    assert "label_btc_up_at_resolution" not in columns
    assert "btc_price_at_resolution" not in columns
    assert "future_btc_return_to_resolution_from_feature" not in columns
    assert "btc_chainlink_local_arrival_iso" not in columns


def test_train_baseline_model_cli_is_offline(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "training.parquet"
    output_dir = tmp_path / "models"
    _write_parquet(input_path, [_row(idx) for idx in range(40)])

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("train-baseline-model must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "train-baseline-model",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--min-rows",
            "20",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["rows_loaded"] == 40
    assert payload["rows_after_filters"] == 40
    assert payload["output_dir"] == str(output_dir)
