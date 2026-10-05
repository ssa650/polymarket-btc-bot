from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.backtest_dataset import BacktestDataset, FEATURE_COLUMNS
from src.backtest_dataset import REQUIRED_COLUMNS
from src.backtest_dataset import BacktestRecord
from src.paper_trader import (
    BUY,
    FILL_MODE_DEPTH_AWARE,
    FILL_MODE_TOP_OF_BOOK,
    NO,
    SELL,
    YES,
    NoOpStrategy,
    PaperDecision,
    PaperTraderConfig,
    RuleStrategy,
    build_strategy,
    build_paper_runs_report,
    export_paper_backtest_result,
    load_strategy_config,
    render_paper_runs_report,
    result_to_dict,
    result_summary_to_dict,
    run_paper_backtest_from_export,
    run_paper_trader,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _record(
    *,
    market_id: str = "m1",
    timestamp: datetime = BASE,
    best_bid_yes: float = 0.39,
    best_ask_yes: float = 0.40,
    best_bid_no: float = 0.59,
    best_ask_no: float = 0.60,
    tick_yes: float = 0.001,
    tick_no: float = 0.001,
    label_available: int = 0,
    yes_won: int | None = None,
    no_won: int | None = None,
    winning_outcome: str | None = None,
    raw: dict | None = None,
) -> BacktestRecord:
    return BacktestRecord(
        market_id=market_id,
        timestamp=timestamp,
        run_id="run_paper",
        seconds_since_market_start=(timestamp - BASE).total_seconds(),
        seconds_until_close=max(0.0, 300.0 - (timestamp - BASE).total_seconds()),
        best_bid_yes=best_bid_yes,
        best_ask_yes=best_ask_yes,
        best_bid_no=best_bid_no,
        best_ask_no=best_ask_no,
        spread_yes=best_ask_yes - best_bid_yes,
        spread_no=best_ask_no - best_bid_no,
        mid_price_yes=(best_bid_yes + best_ask_yes) / 2,
        mid_price_no=(best_bid_no + best_ask_no) / 2,
        last_trade_price=best_ask_yes,
        last_trade_size=1.0,
        last_trade_time=timestamp,
        btc_price=50_000.0,
        btc_exchange_timestamp=timestamp,
        btc_local_arrival_ns=1_767_225_660_000_000_000,
        btc_sample_age_sec=1.0,
        btc_source="mock",
        latest_tick_size_yes=tick_yes,
        latest_tick_size_no=tick_no,
        tick_size_age_sec=1.0,
        market_start_time=BASE,
        market_close_time=BASE + timedelta(minutes=5),
        resolved=1 if label_available else 0,
        resolved_at=BASE + timedelta(minutes=5, seconds=1) if label_available else None,
        winning_asset_id=f"{market_id}_{'yes' if yes_won else 'no'}"
        if label_available
        else None,
        winning_outcome=winning_outcome,
        yes_won=yes_won,
        no_won=no_won,
        label_available=label_available,
        has_btc_price=1,
        has_full_orderbook=1,
        has_trade_data=1,
        gap_affected=0,
        snapshot_quality_status="ok",
        features={column: 0.0 for column in FEATURE_COLUMNS},
        raw=dict(raw or {}),
    )


def _dataset(*records: BacktestRecord) -> BacktestDataset:
    return BacktestDataset.from_records(records)


def _decision(
    decision_id: str,
    side: str,
    outcome: str,
    quantity: float,
    *,
    market_id: str = "m1",
    timestamp: datetime = BASE,
    limit_price: float | None = None,
) -> PaperDecision:
    return PaperDecision(
        decision_id=decision_id,
        market_id=market_id,
        timestamp=timestamp,
        side=side,
        outcome=outcome,
        quantity=quantity,
        limit_price=limit_price,
    )


def test_noop_strategy_produces_zero_orders_and_unchanged_cash() -> None:
    result = run_paper_trader(
        _dataset(_record()),
        NoOpStrategy(),
        PaperTraderConfig(starting_cash=100.0),
    )

    assert result.total_orders == 0
    assert result.total_fills == 0
    assert result.ending_cash == 100.0
    assert result.realized_pnl == 0.0


def test_buy_yes_fills_at_best_ask_when_cash_is_sufficient() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.41)),
        RuleStrategy([_decision("buy_yes", BUY, YES, 10.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_fills == 1
    assert result.fills[0].price == 0.41
    assert result.fills[0].cash_delta == -4.1
    assert result.ending_cash == 5.9
    assert len(result.positions) == 1
    assert result.positions[0].market_id == "m1"
    assert result.positions[0].outcome == YES
    assert result.positions[0].quantity == 10.0


def test_buy_no_fills_at_best_ask_when_cash_is_sufficient() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_no=0.62)),
        RuleStrategy([_decision("buy_no", BUY, NO, 5.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_fills == 1
    assert result.fills[0].outcome == NO
    assert result.fills[0].price == 0.62
    assert result.ending_cash == 6.9


def test_sell_yes_fills_at_best_bid_when_position_exists() -> None:
    t1 = BASE + timedelta(seconds=1)
    result = run_paper_trader(
        _dataset(_record(timestamp=BASE, best_ask_yes=0.40), _record(timestamp=t1, best_bid_yes=0.55)),
        RuleStrategy(
            [
                _decision("buy_yes", BUY, YES, 10.0, timestamp=BASE),
                _decision("sell_yes", SELL, YES, 10.0, timestamp=t1),
            ]
        ),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert [fill.price for fill in result.fills] == [0.4, 0.55]
    assert result.ending_cash == 11.5
    assert result.positions == ()


def test_sell_no_fills_at_best_bid_when_position_exists() -> None:
    t1 = BASE + timedelta(seconds=1)
    result = run_paper_trader(
        _dataset(_record(timestamp=BASE, best_ask_no=0.60), _record(timestamp=t1, best_bid_no=0.72)),
        RuleStrategy(
            [
                _decision("buy_no", BUY, NO, 5.0, timestamp=BASE),
                _decision("sell_no", SELL, NO, 5.0, timestamp=t1),
            ]
        ),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert [fill.price for fill in result.fills] == [0.6, 0.72]
    assert result.ending_cash == 10.6
    assert result.positions == ()


def test_rejects_buy_when_cash_is_insufficient() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.50)),
        RuleStrategy([_decision("buy_too_big", BUY, YES, 100.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_orders == 1
    assert result.total_fills == 0
    assert result.rejected_orders == 1
    assert result.orders[0].reason == "insufficient_cash"
    assert result.ending_cash == 10.0


def test_rejects_sell_when_position_is_insufficient() -> None:
    result = run_paper_trader(
        _dataset(_record(best_bid_yes=0.50)),
        RuleStrategy([_decision("sell_without_position", SELL, YES, 1.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.rejected_orders == 1
    assert result.orders[0].reason == "insufficient_position"
    assert result.total_fills == 0


def test_limit_no_fill_does_not_change_cash_or_position() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.51)),
        RuleStrategy([_decision("low_limit", BUY, YES, 1.0, limit_price=0.50)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.no_fill_orders == 1
    assert result.orders[0].reason == "buy_limit_below_best_ask"
    assert result.total_fills == 0
    assert result.ending_cash == 10.0
    assert result.positions == ()


def test_tick_size_enforcement_rounds_buy_price_up() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.505, tick_yes=0.01)),
        RuleStrategy([_decision("tick_buy", BUY, YES, 1.0, limit_price=0.505)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.orders[0].normalized_limit_price == 0.51
    assert result.fills[0].price == 0.51
    assert result.ending_cash == 9.49


def test_top_of_book_fill_mode_remains_default_even_when_depth_is_present() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                best_ask_yes=0.41,
                raw={"yes_asks": [{"price": 0.50, "size": 10.0}]},
            )
        ),
        RuleStrategy([_decision("buy_yes", BUY, YES, 2.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.fills[0].price == 0.41
    assert result.fills[0].fill_mode == FILL_MODE_TOP_OF_BOOK
    assert result.orders[0].fill_mode == FILL_MODE_TOP_OF_BOOK
    assert result.orders[0].levels_consumed == 1


def test_depth_aware_buy_fills_fully_across_multiple_ask_levels() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 2.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("buy_yes_depth", BUY, YES, 3.0, limit_price=0.42)]),
        PaperTraderConfig(starting_cash=10.0, fill_mode=FILL_MODE_DEPTH_AWARE),
    )

    assert result.total_fills == 1
    assert result.fills[0].quantity == 3.0
    assert result.fills[0].price == 0.4133333333
    assert result.fills[0].average_fill_price == 0.4133333333
    assert result.fills[0].levels_consumed == 2
    assert result.fills[0].liquidity_available == 3.0
    assert result.fills[0].partial_fill is False
    assert result.orders[0].filled_size == 3.0
    assert result.orders[0].unfilled_size == 0.0
    assert result.ending_cash == 8.76


def test_depth_aware_buy_partially_fills_when_liquidity_is_insufficient() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 2.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("buy_yes_partial", BUY, YES, 5.0, limit_price=0.42)]),
        PaperTraderConfig(starting_cash=10.0, fill_mode=FILL_MODE_DEPTH_AWARE),
    )

    assert result.total_fills == 1
    assert result.orders[0].status == "partial_fill"
    assert result.orders[0].filled_size == 3.0
    assert result.orders[0].unfilled_size == 2.0
    assert result.orders[0].partial_fill is True
    assert result.orders[0].no_fill_reason == "insufficient_depth"
    assert result.fills[0].quantity == 3.0
    assert result.positions[0].quantity == 3.0


def test_depth_aware_buy_no_fills_when_partial_fills_are_disabled() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 2.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("buy_yes_no_partial", BUY, YES, 5.0, limit_price=0.42)]),
        PaperTraderConfig(
            starting_cash=10.0,
            fill_mode=FILL_MODE_DEPTH_AWARE,
            allow_partial_fills=False,
        ),
    )

    assert result.total_fills == 0
    assert result.no_fill_orders == 1
    assert result.orders[0].reason == "insufficient_depth"
    assert result.orders[0].liquidity_available == 3.0
    assert result.ending_cash == 10.0


def test_depth_aware_sell_fills_across_multiple_bid_levels() -> None:
    t1 = BASE + timedelta(seconds=1)
    result = run_paper_trader(
        _dataset(
            _record(
                timestamp=BASE,
                raw={"yes_asks": [{"price": 0.40, "size": 3.0}]},
            ),
            _record(
                timestamp=t1,
                raw={
                    "yes_bids": [
                        {"price": 0.60, "size": 1.0},
                        {"price": 0.58, "size": 2.0},
                    ]
                },
            ),
        ),
        RuleStrategy(
            [
                _decision("buy_yes", BUY, YES, 3.0, timestamp=BASE, limit_price=0.40),
                _decision("sell_yes", SELL, YES, 3.0, timestamp=t1, limit_price=0.58),
            ]
        ),
        PaperTraderConfig(starting_cash=10.0, fill_mode=FILL_MODE_DEPTH_AWARE),
    )

    assert [fill.quantity for fill in result.fills] == [3.0, 3.0]
    assert result.fills[1].price == 0.5866666667
    assert result.fills[1].levels_consumed == 2
    assert result.ending_cash == 10.56
    assert result.positions == ()


def test_depth_aware_limit_price_prevents_consuming_worse_levels() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 2.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("buy_yes_limit", BUY, YES, 3.0, limit_price=0.41)]),
        PaperTraderConfig(starting_cash=10.0, fill_mode=FILL_MODE_DEPTH_AWARE),
    )

    assert result.orders[0].status == "partial_fill"
    assert result.fills[0].quantity == 1.0
    assert result.fills[0].price == 0.4
    assert result.fills[0].levels_consumed == 1
    assert result.fills[0].liquidity_available == 1.0


def test_depth_aware_risk_controls_use_average_fill_notional() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 2.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("buy_yes_risk", BUY, YES, 3.0, limit_price=0.42)]),
        PaperTraderConfig(
            starting_cash=10.0,
            fill_mode=FILL_MODE_DEPTH_AWARE,
            max_notional_per_order=1.0,
        ),
    )

    assert result.rejected_orders == 1
    assert result.total_fills == 0
    assert result.orders[0].reason == "max_notional_per_order_exceeded"
    assert result.orders[0].average_fill_price == 0.4133333333
    assert result.orders[0].filled_size == 0.0


def test_depth_aware_settlement_works_after_partial_fill() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                label_available=1,
                yes_won=1,
                no_won=0,
                winning_outcome=YES,
                raw={"yes_asks": [{"price": 0.40, "size": 2.0}]},
            )
        ),
        RuleStrategy([_decision("buy_yes_partial_winner", BUY, YES, 5.0, limit_price=0.40)]),
        PaperTraderConfig(starting_cash=10.0, fill_mode=FILL_MODE_DEPTH_AWARE),
    )

    assert result.orders[0].status == "partial_fill"
    assert result.fills[0].quantity == 2.0
    assert result.unsettled_positions == 0
    assert result.settlements[0].quantity == 2.0
    assert result.settlements[0].settlement_pnl == 1.2
    assert result.ending_cash == 11.2


def test_settlement_pnl_works_for_yes_winner() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                best_ask_yes=0.40,
                label_available=1,
                yes_won=1,
                no_won=0,
                winning_outcome=YES,
            )
        ),
        RuleStrategy([_decision("buy_yes", BUY, YES, 10.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.ending_cash == 16.0
    assert result.realized_pnl == 6.0
    assert result.unsettled_positions == 0
    assert result.positions == ()


def test_settlement_pnl_works_for_no_winner() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                best_ask_no=0.30,
                label_available=1,
                yes_won=0,
                no_won=1,
                winning_outcome=NO,
            )
        ),
        RuleStrategy([_decision("buy_no", BUY, NO, 10.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.ending_cash == 17.0
    assert result.realized_pnl == 7.0
    assert result.unsettled_positions == 0


def test_unresolved_market_remains_unsettled_without_fake_settlement_pnl() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40, label_available=0)),
        RuleStrategy([_decision("buy_yes", BUY, YES, 10.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.ending_cash == 6.0
    assert result.realized_pnl == -4.0
    assert result.unsettled_positions == 1
    assert result.positions[0].quantity == 10.0


def test_simulator_works_over_multiple_markets() -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                market_id="m1",
                best_ask_yes=0.40,
                label_available=1,
                yes_won=1,
                no_won=0,
                winning_outcome=YES,
            ),
            _record(
                market_id="m2",
                timestamp=BASE,
                best_ask_no=0.30,
                label_available=1,
                yes_won=0,
                no_won=1,
                winning_outcome=NO,
            ),
        ),
        RuleStrategy(
            [
                _decision("buy_m1_yes", BUY, YES, 10.0, market_id="m1"),
                _decision("buy_m2_no", BUY, NO, 10.0, market_id="m2"),
            ]
        ),
        PaperTraderConfig(starting_cash=20.0),
    )

    assert result.markets_traded == 2
    assert result.total_fills == 2
    assert result.ending_cash == 33.0
    assert result.realized_pnl == 13.0


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def _parquet_row(
    *,
    market_id: str = "m_cli",
    timestamp: str = "2026-01-01T00:00:00+00:00",
    seconds_until_close: float = 300.0,
    best_ask_yes: float = 0.40,
    has_btc_price: int = 1,
    label_available: int = 0,
    yes_won: int | None = None,
    no_won: int | None = None,
    winning_outcome: str | None = None,
) -> dict:
    row = {
        "run_id": "run_paper_cli",
        "market_id": market_id,
        "timestamp": timestamp,
        "market_start_time": "2026-01-01T00:00:00+00:00",
        "market_close_time": "2026-01-01T00:05:00+00:00",
        "seconds_since_market_start": 300.0 - seconds_until_close,
        "seconds_until_close": seconds_until_close,
        "best_bid_yes": 0.39,
        "best_ask_yes": best_ask_yes,
        "best_bid_no": 0.59,
        "best_ask_no": 0.60,
        "spread_yes": best_ask_yes - 0.39,
        "spread_no": 0.01,
        "mid_price_yes": (0.39 + best_ask_yes) / 2,
        "mid_price_no": 0.595,
        "last_trade_price": best_ask_yes,
        "last_trade_size": 1.0,
        "last_trade_time": timestamp,
        "has_btc_price": has_btc_price,
        "btc_price": 50000.0 if has_btc_price else None,
        "btc_exchange_timestamp": timestamp if has_btc_price else None,
        "btc_local_arrival_ns": 1 if has_btc_price else None,
        "btc_sample_age_sec": 1.0 if has_btc_price else None,
        "btc_source": "mock" if has_btc_price else None,
        "latest_tick_size_yes": 0.001,
        "latest_tick_size_no": 0.001,
        "tick_size_age_sec": 1.0,
        "resolved": 1 if label_available else 0,
        "resolved_at": "2026-01-01T00:05:01+00:00" if label_available else None,
        "winning_asset_id": f"{market_id}_{'yes' if yes_won else 'no'}"
        if label_available
        else None,
        "winning_outcome": winning_outcome if label_available else None,
        "yes_won": yes_won if label_available else None,
        "no_won": no_won if label_available else None,
        "label_available": label_available,
        "has_full_orderbook": 1,
        "has_trade_data": 1,
        "gap_affected": 0,
        "snapshot_quality_status": "ok",
    }
    for column in FEATURE_COLUMNS:
        row[column] = 0.0
    assert set(REQUIRED_COLUMNS).issubset(row)
    return row


def test_run_paper_backtest_cli_noop_is_offline(monkeypatch, tmp_path, capsys) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(path, [_parquet_row()])

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run-paper-backtest must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-paper-backtest",
            "--export-path",
            str(path),
            "--strategy",
            "noop",
            "--paper-starting-cash",
            "123.0",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["starting_cash"] == 123.0
    assert payload["ending_cash"] == 123.0
    assert payload["total_orders"] == 0
    assert payload["total_fills"] == 0


def test_result_to_dict_serializes_datetimes() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        RuleStrategy([_decision("buy_yes", BUY, YES, 1.0)]),
        PaperTraderConfig(starting_cash=10.0),
    )

    payload = result_to_dict(result)

    assert payload["orders"][0]["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert payload["fills"][0]["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_noop_strategy_export_writes_summary_json(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(_record()),
        NoOpStrategy(),
        PaperTraderConfig(run_id="export_noop", starting_cash=42.0),
        strategy_name="noop",
        dataset_path="fixture.parquet",
    )

    paths = export_paper_backtest_result(result, tmp_path)

    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary["run_id"] == "export_noop"
    assert summary["strategy_name"] == "noop"
    assert summary["dataset_path"] == "fixture.parquet"
    assert summary["starting_cash"] == 42.0
    assert summary["ending_cash"] == 42.0
    assert summary["total_orders"] == 0
    assert Path(paths["orders_csv"]).exists()
    assert Path(paths["fills_csv"]).exists()
    assert Path(paths["settlements_csv"]).exists()


def test_csv_fills_and_orders_export(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        RuleStrategy([_decision("buy_yes", BUY, YES, 2.0)]),
        PaperTraderConfig(run_id="export_fills", starting_cash=10.0),
        strategy_name="rule",
    )

    paths = export_paper_backtest_result(result, tmp_path)

    with Path(paths["orders_csv"]).open("r", encoding="utf-8", newline="") as fh:
        orders = list(csv.DictReader(fh))
    with Path(paths["fills_csv"]).open("r", encoding="utf-8", newline="") as fh:
        fills = list(csv.DictReader(fh))

    assert orders[0]["status"] == "filled"
    assert orders[0]["decision_id"] == "buy_yes"
    assert fills[0]["price"] == "0.4"
    assert fills[0]["quantity"] == "2.0"


def test_risk_controls_reject_too_large_order() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        RuleStrategy([_decision("too_large", BUY, YES, 10.0)]),
        PaperTraderConfig(
            starting_cash=20.0,
            max_notional_per_order=3.0,
        ),
    )

    assert result.rejected_orders == 1
    assert result.orders[0].reason == "max_notional_per_order_exceeded"
    assert result.total_fills == 0
    assert result.ending_cash == 20.0


def test_risk_controls_reject_over_max_position() -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        RuleStrategy(
            [
                _decision("first", BUY, YES, 6.0),
                _decision("second", BUY, YES, 5.0),
            ]
        ),
        PaperTraderConfig(
            starting_cash=20.0,
            max_position_size_per_market=10.0,
        ),
    )

    assert result.total_fills == 1
    assert result.rejected_orders == 1
    assert result.orders[1].reason == "max_position_size_per_market_exceeded"
    assert result.positions[0].quantity == 6.0


def test_rule_strategy_config_loads_from_json_and_places_order(tmp_path) -> None:
    config_path = tmp_path / "rule.json"
    config_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "late_yes",
                        "outcome": "YES",
                        "order_size": 3.0,
                        "max_seconds_until_close": 60.0,
                        "buy_yes_ask_lte": 0.41,
                        "limit_price": 0.41,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = load_strategy_config(config_path)
    strategy = build_strategy("rule", strategy_config=config)

    result = run_paper_trader(
        _dataset(_record(timestamp=BASE + timedelta(seconds=250), best_ask_yes=0.40)),
        strategy,
        PaperTraderConfig(starting_cash=10.0),
        strategy_name="rule",
    )

    assert result.total_fills == 1
    assert result.orders[0].decision_id == "late_yes:m1"
    assert result.orders[0].normalized_limit_price == 0.41
    assert result.fills[0].quantity == 3.0


def test_rule_strategy_config_does_not_trade_when_conditions_fail() -> None:
    strategy = build_strategy(
        "rule",
        strategy_config={
            "rules": [
                {
                    "id": "cheap_yes",
                    "outcome": "YES",
                    "order_size": 1.0,
                    "buy_yes_ask_lte": 0.39,
                }
            ]
        },
    )

    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        strategy,
        PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_orders == 0
    assert result.total_fills == 0


def test_paper_require_labels_filters_to_labelled_rows(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _parquet_row(market_id="unlabelled", label_available=0, best_ask_yes=0.40),
            _parquet_row(
                market_id="labelled",
                label_available=1,
                yes_won=1,
                no_won=0,
                winning_outcome=YES,
                best_ask_yes=0.40,
            ),
        ],
    )

    result = run_paper_backtest_from_export(
        str(path),
        strategy_name="rule",
        strategy_config={
            "rules": [
                {
                    "id": "buy_labelled_yes",
                    "outcome": "YES",
                    "order_size": 1.0,
                    "buy_yes_ask_lte": 0.41,
                }
            ]
        },
        require_labels=True,
        config=PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_fills == 1
    assert result.fills[0].market_id == "labelled"
    assert result.unsettled_positions == 0


def test_paper_require_btc_price_filters_to_btc_covered_rows(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _parquet_row(market_id="no_btc", has_btc_price=0, best_ask_yes=0.40),
            _parquet_row(market_id="has_btc", has_btc_price=1, best_ask_yes=0.40),
        ],
    )

    result = run_paper_backtest_from_export(
        str(path),
        strategy_name="rule",
        strategy_config={
            "rules": [
                {
                    "id": "buy_btc_yes",
                    "outcome": "YES",
                    "order_size": 1.0,
                    "buy_yes_ask_lte": 0.41,
                }
            ]
        },
        require_btc_price=True,
        config=PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_fills == 1
    assert result.fills[0].market_id == "has_btc"


def test_paper_time_window_before_close_filters_rows(tmp_path) -> None:
    path = tmp_path / "training.parquet"
    _write_parquet(
        path,
        [
            _parquet_row(market_id="early", seconds_until_close=120.0, best_ask_yes=0.40),
            _parquet_row(market_id="late", seconds_until_close=30.0, best_ask_yes=0.40),
        ],
    )

    result = run_paper_backtest_from_export(
        str(path),
        strategy_name="rule",
        strategy_config={
            "rules": [
                {
                    "id": "buy_late_yes",
                    "outcome": "YES",
                    "order_size": 1.0,
                    "buy_yes_ask_lte": 0.41,
                }
            ]
        },
        time_window_before_close_sec=60.0,
        config=PaperTraderConfig(starting_cash=10.0),
    )

    assert result.total_fills == 1
    assert result.fills[0].market_id == "late"


def test_run_paper_backtest_cli_noop_writes_output_files(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "training.parquet"
    output_dir = tmp_path / "paper_runs"
    _write_parquet(path, [_parquet_row()])

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run-paper-backtest must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-paper-backtest",
            "--export-path",
            str(path),
            "--strategy",
            "noop",
            "--paper-run-id",
            "cli_noop",
            "--paper-output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_id"] == "cli_noop"
    assert payload["total_orders"] == 0
    assert Path(payload["output_files"]["summary_json"]).exists()
    assert Path(payload["output_files"]["orders_csv"]).exists()
    assert Path(payload["output_files"]["fills_csv"]).exists()
    assert Path(payload["output_files"]["settlements_csv"]).exists()
    summary = json.loads(Path(payload["output_files"]["summary_json"]).read_text())
    assert summary["run_id"] == "cli_noop"


def test_run_paper_backtest_cli_accepts_depth_aware_fill_flags(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "training.parquet"
    row = _parquet_row(best_ask_yes=0.40)
    row["yes_asks_json"] = json.dumps([{"price": 0.40, "size": 1.0}])
    _write_parquet(path, [row])
    config_path = tmp_path / "rule.json"
    config_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "depth_yes",
                        "outcome": YES,
                        "order_size": 2.0,
                        "limit_price": 0.40,
                        "buy_yes_ask_lte": 0.40,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run-paper-backtest must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run-paper-backtest",
            "--export-path",
            str(path),
            "--strategy",
            "rule",
            "--paper-strategy-config",
            str(config_path),
            "--paper-fill-mode",
            "depth_aware",
            "--paper-max-depth-levels",
            "1",
            "--paper-no-partial-fills",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["config_used"]["fill_mode"] == FILL_MODE_DEPTH_AWARE
    assert payload["config_used"]["max_depth_levels"] == 1
    assert payload["config_used"]["allow_partial_fills"] is False
    assert payload["total_orders"] == 1
    assert payload["total_fills"] == 0
    assert payload["no_fill_orders"] == 1


def _write_summary(output_dir: Path, **overrides: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": "run_a",
        "strategy_name": "noop",
        "dataset_path": "training.parquet",
        "starting_cash": 100.0,
        "ending_cash": 100.0,
        "realized_pnl": 0.0,
        "total_orders": 0,
        "total_fills": 0,
        "rejected_orders": 0,
        "no_fill_orders": 0,
        "unsettled_positions": 0,
        "markets_traded": 0,
        "config_used": {},
    }
    payload.update(overrides)
    path = output_dir / f"{payload['run_id']}_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv_rows(
    path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def test_paper_runs_report_reads_multiple_summaries_and_sorts_by_pnl(tmp_path) -> None:
    _write_summary(
        tmp_path,
        run_id="low",
        strategy_name="rule",
        ending_cash=95.0,
        realized_pnl=-5.0,
        total_orders=3,
        total_fills=2,
    )
    _write_summary(
        tmp_path,
        run_id="high",
        strategy_name="rule",
        ending_cash=112.0,
        realized_pnl=12.0,
        total_orders=4,
        total_fills=4,
    )
    _write_summary(
        tmp_path,
        run_id="mid",
        strategy_name="noop",
        ending_cash=101.0,
        realized_pnl=1.0,
        total_orders=0,
        total_fills=0,
    )

    report = build_paper_runs_report(tmp_path)
    text = render_paper_runs_report(report)

    assert [row["run_id"] for row in report["runs"]] == ["high", "mid", "low"]
    assert "run_id | strategy_name" in text
    assert "warning=mid:no_fills" in text


def test_paper_runs_report_top_limits_output(tmp_path) -> None:
    _write_summary(tmp_path, run_id="a", realized_pnl=1.0, total_fills=1)
    _write_summary(tmp_path, run_id="b", realized_pnl=3.0, total_fills=1)
    _write_summary(tmp_path, run_id="c", realized_pnl=2.0, total_fills=1)

    report = build_paper_runs_report(tmp_path, top=2)

    assert [row["run_id"] for row in report["runs"]] == ["b", "c"]
    assert report["top"] == 2


def test_paper_runs_report_sort_by_run_id_ascending(tmp_path) -> None:
    _write_summary(tmp_path, run_id="run_c", realized_pnl=100.0, total_fills=1)
    _write_summary(tmp_path, run_id="run_a", realized_pnl=-100.0, total_fills=1)
    _write_summary(tmp_path, run_id="run_b", realized_pnl=0.0, total_fills=1)

    report = build_paper_runs_report(tmp_path, sort_by="run_id")

    assert [row["run_id"] for row in report["runs"]] == ["run_a", "run_b", "run_c"]


def test_paper_runs_report_missing_directory_has_warning(tmp_path) -> None:
    missing = tmp_path / "missing"

    report = build_paper_runs_report(missing)
    text = render_paper_runs_report(report)

    assert report["summary_files"] == 0
    assert report["runs"] == []
    assert report["warnings"] == ["no_summary_files_found"]
    assert "no paper run summaries" in text
    assert "warning=no_summary_files_found" in text


def test_paper_runs_report_warns_for_unsettled_and_high_rejections(tmp_path) -> None:
    _write_summary(
        tmp_path,
        run_id="risky",
        total_orders=10,
        total_fills=1,
        rejected_orders=6,
        unsettled_positions=2,
    )

    report = build_paper_runs_report(tmp_path)

    assert "risky:unsettled_positions=2" in report["warnings"]
    assert "risky:high_rejected_orders=6" in report["warnings"]


def test_paper_runs_detail_report_loads_orders_fills_and_per_market_counts(tmp_path) -> None:
    _write_summary(
        tmp_path,
        run_id="detail",
        total_orders=4,
        total_fills=3,
        rejected_orders=1,
        no_fill_orders=1,
    )
    _write_csv_rows(
        tmp_path / "detail_orders.csv",
        fieldnames=(
            "order_id",
            "decision_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "limit_price",
            "normalized_limit_price",
            "status",
            "reason",
        ),
        rows=[
            {
                "order_id": "o1",
                "decision_id": "d1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 2.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "filled",
                "reason": "",
            },
            {
                "order_id": "o2",
                "decision_id": "d2",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": SELL,
                "outcome": YES,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "filled",
                "reason": "",
            },
            {
                "order_id": "o3",
                "decision_id": "d3",
                "market_id": "m2",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": NO,
                "quantity": 3.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "rejected",
                "reason": "max_notional_per_order_exceeded",
            },
            {
                "order_id": "o4",
                "decision_id": "d4",
                "market_id": "m2",
                "timestamp": BASE.isoformat(),
                "side": SELL,
                "outcome": NO,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "no_fill",
                "reason": "sell_limit_above_best_bid",
            },
        ],
    )
    _write_csv_rows(
        tmp_path / "detail_fills.csv",
        fieldnames=(
            "fill_id",
            "order_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "price",
            "fee",
            "cash_delta",
        ),
        rows=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 2.0,
                "price": 0.4,
                "fee": 0.0,
                "cash_delta": -0.8,
            },
            {
                "fill_id": "f2",
                "order_id": "o2",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": SELL,
                "outcome": YES,
                "quantity": 1.0,
                "price": 0.6,
                "fee": 0.0,
                "cash_delta": 0.6,
            },
            {
                "fill_id": "f3",
                "order_id": "o5",
                "market_id": "m2",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": NO,
                "quantity": 3.0,
                "price": 0.3,
                "fee": 0.0,
                "cash_delta": -0.9,
            },
        ],
    )

    report = build_paper_runs_report(tmp_path, include_detail=True)
    detail = report["details"]["detail"]
    by_market = {row["market_id"]: row for row in detail["per_market"]}
    text = render_paper_runs_report(report)

    assert by_market["m1"]["total_orders"] == 2
    assert by_market["m1"]["total_fills"] == 2
    assert by_market["m1"]["buy_fills"] == 1
    assert by_market["m1"]["sell_fills"] == 1
    assert by_market["m1"]["gross_notional"] == 1.4
    assert by_market["m1"]["realized_pnl"] == 0.2
    assert by_market["m1"]["open_position_shares_before_settlement"] == 1.0
    assert by_market["m1"]["settled_shares"] is None
    assert by_market["m1"]["unsettled_positions_after_settlement"] is None
    assert by_market["m1"]["settlement_pnl"] is None
    assert "unsettled_positions" not in by_market["m1"]
    assert by_market["m2"]["total_orders"] == 2
    assert by_market["m2"]["total_fills"] == 1
    assert by_market["m2"]["open_position_shares_before_settlement"] == 3.0
    assert by_market["m2"]["unsettled_positions_after_settlement"] is None
    assert detail["depth_detail_available"] is False
    assert detail["depth_summary"]["depth_detail_available"] is False
    assert by_market["m1"]["partial_fill_count"] is None
    assert by_market["m1"]["avg_levels_consumed"] is None
    assert "=== Detail ===" in text
    assert "settlement_detail_available=False" in text
    assert "warning=detail:settlement_detail_unavailable" in text
    assert "rejected_order_reason.max_notional_per_order_exceeded=1" in text


def test_paper_runs_detail_report_settled_losing_yes_is_not_unsettled(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                best_ask_yes=0.40,
                label_available=1,
                yes_won=0,
                no_won=1,
                winning_outcome=NO,
            )
        ),
        RuleStrategy([_decision("buy_losing_yes", BUY, YES, 1.0)]),
        PaperTraderConfig(run_id="settled_loser", starting_cash=10.0),
    )
    paths = export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    row = report["details"]["settled_loser"]["per_market"][0]

    assert result.unsettled_positions == 0
    assert Path(paths["settlements_csv"]).exists()
    assert report["details"]["settled_loser"]["settlement_detail_available"] is True
    assert row["open_position_shares_before_settlement"] == 1.0
    assert row["settled_shares"] == 1.0
    assert row["unsettled_positions_after_settlement"] == 0
    assert row["settlement_pnl"] == -0.4
    assert row["realized_pnl"] == -0.4


def test_paper_runs_detail_report_settled_winning_yes_shows_settlement(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                best_ask_yes=0.40,
                label_available=1,
                yes_won=1,
                no_won=0,
                winning_outcome=YES,
            )
        ),
        RuleStrategy([_decision("buy_winning_yes", BUY, YES, 2.0)]),
        PaperTraderConfig(run_id="settled_winner", starting_cash=10.0),
    )
    export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    row = report["details"]["settled_winner"]["per_market"][0]

    assert result.unsettled_positions == 0
    assert result.realized_pnl == 1.2
    assert row["open_position_shares_before_settlement"] == 2.0
    assert row["settled_shares"] == 2.0
    assert row["unsettled_positions_after_settlement"] == 0
    assert row["settlement_pnl"] == 1.2
    assert row["realized_pnl"] == 1.2


def test_paper_runs_detail_report_unresolved_market_remains_unsettled(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40, label_available=0)),
        RuleStrategy([_decision("buy_unresolved_yes", BUY, YES, 1.0)]),
        PaperTraderConfig(run_id="unresolved", starting_cash=10.0),
    )
    export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    row = report["details"]["unresolved"]["per_market"][0]

    assert result.unsettled_positions == 1
    assert report["details"]["unresolved"]["settlement_detail_available"] is True
    assert row["open_position_shares_before_settlement"] == 1.0
    assert row["settled_shares"] == 0.0
    assert row["unsettled_positions_after_settlement"] == 1
    assert row["settlement_pnl"] == 0.0


def test_paper_runs_detail_report_cohort_summaries(tmp_path) -> None:
    _write_summary(tmp_path, run_id="cohorts", total_orders=3, total_fills=2)
    _write_csv_rows(
        tmp_path / "cohorts_orders.csv",
        fieldnames=(
            "order_id",
            "decision_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "limit_price",
            "normalized_limit_price",
            "status",
            "reason",
        ),
        rows=[
            {
                "order_id": "o1",
                "decision_id": "d1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "filled",
                "reason": "",
            },
            {
                "order_id": "o2",
                "decision_id": "d2",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": SELL,
                "outcome": NO,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "filled",
                "reason": "",
            },
            {
                "order_id": "o3",
                "decision_id": "d3",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "no_fill",
                "reason": "buy_limit_below_best_ask",
            },
            {
                "order_id": "o4",
                "decision_id": "d4",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 10.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "rejected",
                "reason": "insufficient_cash",
            },
        ],
    )
    _write_csv_rows(
        tmp_path / "cohorts_fills.csv",
        fieldnames=(
            "fill_id",
            "order_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "price",
            "fee",
            "cash_delta",
        ),
        rows=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 1.0,
                "price": 0.4,
                "fee": 0.0,
                "cash_delta": -0.4,
            },
            {
                "fill_id": "f2",
                "order_id": "o2",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": SELL,
                "outcome": NO,
                "quantity": 1.0,
                "price": 0.5,
                "fee": 0.0,
                "cash_delta": 0.5,
            },
        ],
    )

    report = build_paper_runs_report(tmp_path, include_detail=True)
    cohorts = report["details"]["cohorts"]["cohorts"]

    assert cohorts["yes_fills"] == 1
    assert cohorts["no_fills"] == 1
    assert cohorts["buy_fills"] == 1
    assert cohorts["sell_fills"] == 1
    assert cohorts["no_fill_orders"] == 1
    assert cohorts["rejected_order_reasons"] == {"insufficient_cash": 1}
    assert cohorts["no_fill_reasons"] == {"buy_limit_below_best_ask": 1}


def test_paper_runs_detail_report_depth_summary_for_partial_fill(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(
            _record(
                raw={
                    "yes_asks": [
                        {"price": 0.40, "size": 1.0},
                        {"price": 0.42, "size": 1.0},
                    ]
                }
            )
        ),
        RuleStrategy([_decision("depth_partial", BUY, YES, 3.0, limit_price=0.42)]),
        PaperTraderConfig(
            run_id="depth_partial",
            starting_cash=10.0,
            fill_mode=FILL_MODE_DEPTH_AWARE,
        ),
    )
    export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    detail = report["details"]["depth_partial"]
    depth = detail["depth_summary"]
    per_market = detail["per_market"][0]
    text = render_paper_runs_report(report)

    assert detail["depth_detail_available"] is True
    assert depth["fill_mode_counts"] == {FILL_MODE_DEPTH_AWARE: 1}
    assert depth["partial_fill_count"] == 1
    assert depth["total_requested_size"] == 3.0
    assert depth["total_filled_size"] == 2.0
    assert depth["total_unfilled_size"] == 1.0
    assert depth["avg_levels_consumed"] == 2.0
    assert depth["avg_liquidity_available"] == 2.0
    assert depth["avg_fill_price"] == 0.41
    assert depth["no_fill_reason_counts"] == {"insufficient_depth": 1}
    assert per_market["partial_fill_count"] == 1
    assert per_market["avg_levels_consumed"] == 2.0
    assert per_market["total_unfilled_size"] == 1.0
    assert per_market["avg_fill_price"] == 0.41
    assert "depth:" in text
    assert "fill_mode.depth_aware=1" in text
    assert "depth_no_fill_reason.insufficient_depth=1" in text
    json.dumps(report)


def test_paper_runs_detail_report_top_of_book_depth_fields_are_safe(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(_record(best_ask_yes=0.40)),
        RuleStrategy([_decision("top_buy", BUY, YES, 2.0)]),
        PaperTraderConfig(run_id="top_run", starting_cash=10.0),
    )
    export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    depth = report["details"]["top_run"]["depth_summary"]

    assert report["details"]["top_run"]["depth_detail_available"] is True
    assert depth["fill_mode_counts"] == {FILL_MODE_TOP_OF_BOOK: 1}
    assert depth["partial_fill_count"] == 0
    assert depth["total_requested_size"] == 2.0
    assert depth["total_filled_size"] == 2.0
    assert depth["total_unfilled_size"] == 0.0
    assert depth["avg_levels_consumed"] == 1.0
    assert depth["avg_fill_price"] == 0.4


def test_paper_runs_detail_report_depth_no_fill_reasons_are_counted(tmp_path) -> None:
    result = run_paper_trader(
        _dataset(
            _record(raw={"yes_asks": [{"price": 0.40, "size": 1.0}]})
        ),
        RuleStrategy([_decision("depth_no_fill", BUY, YES, 2.0, limit_price=0.40)]),
        PaperTraderConfig(
            run_id="depth_no_fill",
            starting_cash=10.0,
            fill_mode=FILL_MODE_DEPTH_AWARE,
            allow_partial_fills=False,
        ),
    )
    export_paper_backtest_result(result, tmp_path)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    depth = report["details"]["depth_no_fill"]["depth_summary"]
    cohorts = report["details"]["depth_no_fill"]["cohorts"]

    assert depth["no_fill_reason_counts"] == {"insufficient_depth": 1}
    assert depth["total_requested_size"] == 2.0
    assert depth["total_filled_size"] == 0.0
    assert depth["total_unfilled_size"] == 2.0
    assert depth["avg_liquidity_available"] == 1.0
    assert cohorts["no_fill_reasons"] == {"insufficient_depth": 1}


def test_paper_runs_report_run_id_filter_and_warnings_only(tmp_path) -> None:
    _write_summary(tmp_path, run_id="quiet", total_orders=1, total_fills=1)
    _write_summary(
        tmp_path,
        run_id="warned",
        total_orders=2,
        total_fills=0,
        rejected_orders=2,
    )

    filtered = build_paper_runs_report(tmp_path, run_id="quiet")
    warnings_only = build_paper_runs_report(tmp_path, warnings_only=True)

    assert [row["run_id"] for row in filtered["runs"]] == ["quiet"]
    assert filtered["run_id_filter"] == "quiet"
    assert [row["run_id"] for row in warnings_only["runs"]] == ["warned"]


def test_report_paper_runs_cli_detail_json_output_is_offline(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _write_summary(tmp_path, run_id="detail_cli", total_orders=1, total_fills=1)
    _write_csv_rows(
        tmp_path / "detail_cli_orders.csv",
        fieldnames=(
            "order_id",
            "decision_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "limit_price",
            "normalized_limit_price",
            "status",
            "reason",
        ),
        rows=[
            {
                "order_id": "o1",
                "decision_id": "d1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 1.0,
                "limit_price": "",
                "normalized_limit_price": "",
                "status": "filled",
                "reason": "",
            }
        ],
    )
    _write_csv_rows(
        tmp_path / "detail_cli_fills.csv",
        fieldnames=(
            "fill_id",
            "order_id",
            "market_id",
            "timestamp",
            "side",
            "outcome",
            "quantity",
            "price",
            "fee",
            "cash_delta",
        ),
        rows=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "market_id": "m1",
                "timestamp": BASE.isoformat(),
                "side": BUY,
                "outcome": YES,
                "quantity": 1.0,
                "price": 0.4,
                "fee": 0.0,
                "cash_delta": -0.4,
            }
        ],
    )
    _write_csv_rows(
        tmp_path / "detail_cli_settlements.csv",
        fieldnames=(
            "settlement_id",
            "market_id",
            "timestamp",
            "outcome",
            "quantity",
            "average_price",
            "winning_outcome",
            "settlement_price",
            "cash_delta",
            "settlement_pnl",
        ),
        rows=[
            {
                "settlement_id": "s1",
                "market_id": "m1",
                "timestamp": (BASE + timedelta(minutes=5, seconds=1)).isoformat(),
                "outcome": YES,
                "quantity": 1.0,
                "average_price": 0.4,
                "winning_outcome": YES,
                "settlement_price": 1.0,
                "cash_delta": 1.0,
                "settlement_pnl": 0.6,
            }
        ],
    )

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("report-paper-runs must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "report-paper-runs",
            "--paper-output-dir",
            str(tmp_path),
            "--paper-run-id",
            "detail_cli",
            "--paper-report-detail",
            "--paper-report-json",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_id_filter"] == "detail_cli"
    assert payload["include_detail"] is True
    assert [row["run_id"] for row in payload["runs"]] == ["detail_cli"]
    assert payload["details"]["detail_cli"]["cohorts"]["yes_fills"] == 1
    row = payload["details"]["detail_cli"]["per_market"][0]
    assert row["settled_shares"] == 1.0
    assert row["unsettled_positions_after_settlement"] == 0
    assert row["settlement_pnl"] == 0.6


def test_paper_runs_detail_report_warns_when_csv_files_are_missing(tmp_path) -> None:
    _write_summary(tmp_path, run_id="missing_csv", total_orders=1, total_fills=1)

    report = build_paper_runs_report(tmp_path, include_detail=True)
    detail = report["details"]["missing_csv"]

    assert detail["orders_csv"] is None
    assert detail["fills_csv"] is None
    assert detail["settlements_csv"] is None
    assert detail["settlement_detail_available"] is False
    assert "missing_csv:missing_orders_csv" in report["warnings"]
    assert "missing_csv:missing_fills_csv" in report["warnings"]
    assert "missing_csv:settlement_detail_unavailable" in report["warnings"]


def test_report_paper_runs_cli_json_output_is_offline(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _write_summary(tmp_path, run_id="winner", realized_pnl=5.0, total_fills=2)
    _write_summary(tmp_path, run_id="loser", realized_pnl=-1.0, total_fills=1)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("report-paper-runs must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "report-paper-runs",
            "--paper-output-dir",
            str(tmp_path),
            "--paper-top",
            "1",
            "--paper-report-json",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary_files"] == 2
    assert payload["runs"][0]["run_id"] == "winner"
    assert len(payload["runs"]) == 1


def test_report_paper_runs_cli_text_output_sorts_by_total_fills(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _write_summary(tmp_path, run_id="few", total_fills=1, realized_pnl=100.0)
    _write_summary(tmp_path, run_id="many", total_fills=5, realized_pnl=0.0)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("report-paper-runs must not load settings or network clients")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "report-paper-runs",
            "--paper-output-dir",
            str(tmp_path),
            "--paper-sort-by",
            "total_fills",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    out = capsys.readouterr().out

    assert "=== Paper Run Report ===" in out
    assert out.index("many") < out.index("few")
