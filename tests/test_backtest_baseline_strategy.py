from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from src.backtest_baseline_strategy import BaselineBacktestError, backtest_baseline_strategy


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
FEATURE_COLUMNS = ["signal"]


class ProbabilityBySignalModel:
    def predict_proba(self, matrix):
        return [[1.0 - float(row[0]), float(row[0])] for row in matrix]


def _write_parquet(path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_model(path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ProbabilityBySignalModel(), path)


def _write_features(path, feature_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_columns or FEATURE_COLUMNS), encoding="utf-8")


def _row(
    idx: int,
    *,
    market_id: str,
    signal: float,
    label_yes_win: int | None = 1,
    best_ask_yes: float | None = 0.50,
    best_ask_no: float | None = 0.50,
    best_bid_yes: float | None = 0.49,
    usable: int = 1,
) -> dict:
    ts = BASE + timedelta(seconds=idx)
    close = BASE + timedelta(minutes=5)
    return {
        "run_id": "run_backtest",
        "market_id": market_id,
        "question": f"Bitcoin Up or Down - {market_id}",
        "start_time": BASE.isoformat(),
        "close_time": close.isoformat(),
        "timestamp": ts.isoformat(),
        "export_row_usable": usable,
        "label_btc_up_at_resolution": label_yes_win,
        "label_yes_win": label_yes_win,
        "time_until_resolution": (close - ts).total_seconds(),
        "signal": signal,
        "best_ask_yes": best_ask_yes,
        "best_ask_no": best_ask_no,
        "best_bid_yes": best_bid_yes,
        "btc_chainlink_price": 100.0 + idx,
        "btc_binance_price": 100.5 + idx,
        "btc_price_diff_binance_minus_chainlink": 0.5,
        "btc_chainlink_age_sec_at_feature": 1.0,
    }


def _fixture_paths(
    tmp_path,
    rows: list[dict],
    *,
    feature_columns: list[str] | None = None,
):
    input_path = tmp_path / "training.parquet"
    model_path = tmp_path / "model.joblib"
    feature_path = tmp_path / "feature_columns.json"
    output_dir = tmp_path / "backtest"
    _write_parquet(input_path, rows)
    _write_model(model_path)
    _write_features(feature_path, feature_columns=feature_columns)
    return input_path, model_path, feature_path, output_dir


def _read_csv(path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_backtest_baseline_strategy_known_yes_no_pnl(tmp_path) -> None:
    rows = [
        _row(0, market_id="m_yes", signal=0.8, label_yes_win=1, best_ask_yes=0.50),
        _row(1, market_id="m_no", signal=0.2, label_yes_win=0, best_ask_no=0.25),
        _row(2, market_id="m_none", signal=0.5, label_yes_win=1),
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="full_dataset",
    )
    trades = _read_csv(output_dir / "trades.csv")
    skipped = _read_csv(output_dir / "skipped_rows.csv")

    assert summary["trades_taken"] == 2
    assert summary["markets_traded"] == 2
    assert summary["long_trades"] == 1
    assert summary["short_trades"] == 1
    assert summary["win_rate"] == 1.0
    assert summary["total_staked"] == 2.0
    assert summary["total_payout"] == 6.0
    assert summary["total_pnl"] == 4.0
    assert summary["backtest_split"] == "full_dataset"
    assert "full_dataset_backtest_includes_training_rows" in summary["warnings"]
    assert summary["pnl_by_direction"]["YES"]["total_pnl"] == 1.0
    assert summary["pnl_by_direction"]["NO"]["total_pnl"] == 3.0
    assert trades[0]["direction"] == "YES"
    assert trades[1]["direction"] == "NO"
    assert skipped[0]["reason"] == "no_trade_signal"
    assert (output_dir / "backtest_summary.json").exists()
    assert (output_dir / "backtest_summary.txt").exists()


def test_backtest_baseline_strategy_one_trade_per_market_default(tmp_path) -> None:
    rows = [
        _row(0, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50),
        _row(1, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50),
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="full_dataset",
    )

    assert summary["trades_taken"] == 1
    assert summary["skipped_rows_by_reason"]["already_traded_market"] == 1


def test_backtest_baseline_strategy_allows_multiple_per_market(tmp_path) -> None:
    rows = [
        _row(0, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50),
        _row(1, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50),
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        allow_multiple_per_market=True,
        backtest_split="full_dataset",
    )

    assert summary["trades_taken"] == 2
    assert "already_traded_market" not in summary["skipped_rows_by_reason"]
    assert (
        "multiple_per_market_backtest_can_overstate_live_performance"
        in summary["warnings"]
    )


def test_backtest_baseline_strategy_skips_unresolved_and_missing_prices(tmp_path) -> None:
    unresolved = _row(0, market_id="unresolved", signal=0.8, label_yes_win=1)
    unresolved["label_yes_win"] = None
    rows = [
        unresolved,
        _row(1, market_id="missing_yes", signal=0.8, label_yes_win=1, best_ask_yes=None),
        _row(
            2,
            market_id="missing_no",
            signal=0.2,
            label_yes_win=0,
            best_ask_no=None,
            best_bid_yes=None,
        ),
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="full_dataset",
    )

    assert summary["trades_taken"] == 0
    assert summary["skipped_rows_by_reason"]["unresolved_market"] == 1
    assert summary["skipped_rows_by_reason"]["missing_entry_price"] == 2


def test_backtest_baseline_strategy_estimates_no_ask_and_applies_slippage(tmp_path) -> None:
    rows = [
        _row(
            0,
            market_id="m_no",
            signal=0.2,
            label_yes_win=0,
            best_ask_no=None,
            best_bid_yes=0.70,
        )
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        entry_slippage_cents=1.0,
        fee_cents=2.0,
        backtest_split="full_dataset",
    )
    trade = _read_csv(output_dir / "trades.csv")[0]

    assert summary["trades_taken"] == 1
    assert float(trade["entry_price"]) == 0.3
    assert float(trade["adjusted_entry_price"]) == 0.33
    assert float(trade["effective_entry_price"]) == 0.33
    assert round(float(trade["pnl_usd"]), 6) == round((1.0 / 0.33) - 1.0, 6)


def test_backtest_baseline_strategy_default_test_only_excludes_training_rows(tmp_path) -> None:
    rows = [
        _row(idx, market_id=f"m_{idx}", signal=0.8, label_yes_win=1, best_ask_yes=0.50)
        for idx in range(10)
    ]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
    )
    trades = _read_csv(output_dir / "trades.csv")

    assert summary["backtest_split"] == "test_only"
    assert summary["rows_after_filters"] == 10
    assert summary["train_rows"] == 7
    assert summary["test_rows"] == 3
    assert summary["backtest_rows"] == 3
    assert summary["trades_taken"] == 3
    assert trades[0]["market_id"] == "m_7"


def test_backtest_baseline_strategy_market_holdout_zero_overlap(tmp_path) -> None:
    rows = []
    for idx in range(8):
        market_idx = idx // 2
        row = _row(
            idx,
            market_id=f"m_{market_idx}",
            signal=0.8,
            label_yes_win=1,
            best_ask_yes=0.50,
        )
        row["start_time"] = (BASE + timedelta(minutes=market_idx * 5)).isoformat()
        row["close_time"] = (BASE + timedelta(minutes=(market_idx + 1) * 5)).isoformat()
        rows.append(row)
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="market_holdout",
    )
    trades = _read_csv(output_dir / "trades.csv")

    assert summary["backtest_split"] == "market_holdout"
    assert summary["train_test_market_overlap_count"] == 0
    assert summary["unique_markets_total"] == 4
    assert summary["unique_markets_backtested"] == 2
    assert summary["backtest_rows"] == 4
    assert {row["market_id"] for row in trades} == {"m_2", "m_3"}


def test_backtest_baseline_strategy_suspicious_feature_fails_unless_allowed(tmp_path) -> None:
    rows = [_row(0, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50)]
    input_path, model_path, feature_path, output_dir = _fixture_paths(
        tmp_path,
        rows,
        feature_columns=["signal", "future_btc_return_to_resolution_from_feature"],
    )

    with pytest.raises(BaselineBacktestError, match="Suspicious feature columns"):
        backtest_baseline_strategy(
            input_path=str(input_path),
            model_path=str(model_path),
            feature_columns_path=str(feature_path),
            output_dir=str(output_dir),
            backtest_split="full_dataset",
        )

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="full_dataset",
        allow_suspicious_features=True,
    )

    assert summary["trades_taken"] == 1
    assert "suspicious_features_allowed" in summary["warnings"]


def test_backtest_baseline_strategy_allows_time_until_resolution_feature(tmp_path) -> None:
    rows = [_row(0, market_id="m", signal=0.8, label_yes_win=1, best_ask_yes=0.50)]
    input_path, model_path, feature_path, output_dir = _fixture_paths(
        tmp_path,
        rows,
        feature_columns=["signal", "time_until_resolution"],
    )

    summary = backtest_baseline_strategy(
        input_path=str(input_path),
        model_path=str(model_path),
        feature_columns_path=str(feature_path),
        output_dir=str(output_dir),
        backtest_split="full_dataset",
    )

    assert summary["trades_taken"] == 1
    assert summary["suspicious_feature_columns"][0]["column"] == "time_until_resolution"
    assert summary["suspicious_feature_columns"][0]["known_at_inference"] is True


def test_backtest_baseline_strategy_cli_is_offline(monkeypatch, tmp_path, capsys) -> None:
    rows = [_row(0, market_id="m_yes", signal=0.8, label_yes_win=1, best_ask_yes=0.50)]
    input_path, model_path, feature_path, output_dir = _fixture_paths(tmp_path, rows)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backtest-baseline-strategy must not start recorder or network code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "backtest-baseline-strategy",
            "--input",
            str(input_path),
            "--model-path",
            str(model_path),
            "--feature-columns",
            str(feature_path),
            "--output-dir",
            str(output_dir),
            "--backtest-split",
            "full_dataset",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "SQLiteStore", forbidden)

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["trades_taken"] == 1
    assert payload["output_dir"] == str(output_dir)
