from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.backtest_dataset import (
    FEATURE_COLUMNS,
    BacktestDatasetError,
    inspect_backtest_dataset,
    load_backtest_dataset,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(
    *,
    market_id: str = "m1",
    timestamp: str = "2026-01-01T00:04:00+00:00",
    seconds_until_close: float = 60.0,
    has_btc_price: int = 1,
    has_full_orderbook: int = 1,
    has_trade_data: int = 1,
    label_available: int = 1,
    has_depth: int = 0,
    yes_won: int | None = 1,
    no_won: int | None = 0,
    winning_asset_id: str | None = "m1_yes",
    winning_outcome: str | None = "YES",
) -> dict:
    row = {
        "sequence_id": f"run_a:{market_id}",
        "run_id": "run_a",
        "market_id": market_id,
        "timestamp": timestamp,
        "market_start_time": "2026-01-01T00:00:00+00:00",
        "market_close_time": "2026-01-01T00:05:00+00:00",
        "seconds_since_market_start": 240.0,
        "seconds_until_close": seconds_until_close,
        "best_bid_yes": 0.49,
        "best_ask_yes": 0.51,
        "best_bid_no": 0.49,
        "best_ask_no": 0.51,
        "spread_yes": 0.02,
        "spread_no": 0.02,
        "mid_price_yes": 0.5,
        "mid_price_no": 0.5,
        "last_trade_price": 0.5 if has_trade_data else None,
        "last_trade_size": 1.0 if has_trade_data else None,
        "last_trade_time": timestamp if has_trade_data else None,
        "has_btc_price": has_btc_price,
        "btc_price": 50000.0 if has_btc_price else None,
        "btc_exchange_timestamp": "2026-01-01T00:03:58+00:00"
        if has_btc_price
        else None,
        "btc_local_arrival_ns": 1_767_225_660_000_000_000
        if has_btc_price
        else None,
        "btc_sample_age_sec": 2.0 if has_btc_price else None,
        "btc_source": "polymarket_rtds_chainlink" if has_btc_price else None,
        "latest_tick_size_yes": 0.001,
        "latest_tick_size_no": 0.001,
        "tick_size_age_sec": 5.0,
        "resolved": 1 if label_available else 0,
        "resolved_at": "2026-01-01T00:05:01+00:00" if label_available else None,
        "winning_asset_id": winning_asset_id if label_available else None,
        "winning_outcome": winning_outcome if label_available else None,
        "yes_won": yes_won if label_available else None,
        "no_won": no_won if label_available else None,
        "label_available": label_available,
        "has_full_orderbook": has_full_orderbook,
        "has_depth": has_depth,
        "yes_bids_json": (
            '[{"price":0.49,"size":10.0}]' if has_depth else "[]"
        ),
        "yes_asks_json": (
            '[{"price":0.51,"size":11.0}]' if has_depth else "[]"
        ),
        "no_bids_json": (
            '[{"price":0.48,"size":12.0}]' if has_depth else "[]"
        ),
        "no_asks_json": (
            '[{"price":0.52,"size":13.0}]' if has_depth else "[]"
        ),
        "has_trade_data": has_trade_data,
        "gap_affected": 0,
        "snapshot_quality_status": "ok",
    }
    for column in FEATURE_COLUMNS:
        row[column] = 0.1
    return row


def test_backtest_dataset_reader_groups_and_sorts_timelines(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _row(market_id="m2", timestamp="2026-01-01T00:02:00+00:00"),
            _row(market_id="m1", timestamp="2026-01-01T00:03:00+00:00"),
            _row(market_id="m1", timestamp="2026-01-01T00:01:00+00:00"),
        ],
    )

    dataset = load_backtest_dataset(path)

    assert list(dataset.timelines) == ["m1", "m2"]
    assert [r.timestamp.isoformat() for r in dataset.timelines["m1"].records] == [
        "2026-01-01T00:01:00+00:00",
        "2026-01-01T00:03:00+00:00",
    ]
    assert dataset.timelines["m1"].records[0].best_bid_yes == 0.49
    assert dataset.timelines["m1"].records[0].features["price_change_1s"] == 0.1


def test_backtest_dataset_reader_detects_missing_required_columns(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    bad_row = _row()
    bad_row.pop("best_bid_yes")
    _write_parquet(path, [bad_row])

    with pytest.raises(BacktestDatasetError, match="best_bid_yes"):
        load_backtest_dataset(path)


def test_backtest_dataset_filters_work(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _row(market_id="m1", seconds_until_close=30.0),
            _row(market_id="m1", has_btc_price=0, seconds_until_close=20.0),
            _row(market_id="m2", has_full_orderbook=0, seconds_until_close=10.0),
            _row(market_id="m2", has_trade_data=0, seconds_until_close=5.0),
            _row(market_id="m3", label_available=0, seconds_until_close=90.0),
            _row(market_id="m4", has_depth=1, seconds_until_close=25.0),
        ],
    )
    dataset = load_backtest_dataset(path)

    assert len(dataset.only_with_btc_price().records) == 5
    assert len(dataset.only_with_full_orderbook().records) == 5
    assert len(dataset.only_with_depth().records) == 1
    assert len(dataset.only_with_trade_data().records) == 5
    assert len(dataset.only_with_labels().records) == 5
    assert len(dataset.within_seconds_before_close(30).records) == 5
    assert set(dataset.for_market_ids(["m2"]).timelines) == {"m2"}
    filtered = load_backtest_dataset(
        path,
        market_ids=["m4"],
        require_btc_price=True,
        require_full_orderbook=True,
        require_depth=True,
        require_label_available=True,
        require_trade_data=True,
        time_window_before_close_sec=60,
    )
    assert len(filtered.records) == 1
    assert filtered.records[0].raw["yes_asks_json"] == '[{"price":0.51,"size":11.0}]'
    assert filtered.records[0].has_depth == 1


def test_backtest_dataset_preserves_resolved_and_unresolved_labels(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _row(market_id="resolved", yes_won=0, no_won=1, winning_outcome="NO"),
            _row(market_id="unresolved", label_available=0),
        ],
    )

    dataset = load_backtest_dataset(path)

    resolved = dataset.timelines["resolved"].records[0]
    unresolved = dataset.timelines["unresolved"].records[0]
    assert resolved.label_available == 1
    assert resolved.yes_won == 0
    assert resolved.no_won == 1
    assert resolved.winning_outcome == "NO"
    assert unresolved.label_available == 0
    assert unresolved.yes_won is None
    assert unresolved.no_won is None


def test_backtest_dataset_summary_and_inspection(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _row(market_id="m1", label_available=0),
            _row(market_id="m2", label_available=0),
        ],
    )

    summary_text = inspect_backtest_dataset(path)

    assert "total_rows=2" in summary_text
    assert "total_markets=2" in summary_text
    assert "btc_coverage_pct=100.00" in summary_text
    assert "label_coverage_pct=0.00" in summary_text
    assert "depth_coverage_pct=0.00" in summary_text
    assert "warning=no_labels_available" in summary_text
    assert "market_id=m1" in summary_text
    assert "market_id=m2" in summary_text


def test_inspect_backtest_dataset_cli(monkeypatch, tmp_path, capsys) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(path, [_row(market_id="m1")])

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inspect-backtest-dataset must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "inspect-backtest-dataset", "--export-path", str(path)],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    out = capsys.readouterr().out

    assert "=== Backtest Dataset ===" in out
    assert "total_rows=1" in out
    assert "total_markets=1" in out
