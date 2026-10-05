from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.features import MarketFeatureFrame, compute_feature_record, mid_and_spread
from src.models import MarketSnapshotRecord, PriceLevel, TradeRecord


def test_mid_and_spread() -> None:
    mid, spread = mid_and_spread(0.48, 0.52)
    assert mid == 0.5
    assert round(spread or 0.0, 8) == 0.04

    mid_none, spread_none = mid_and_spread(0.5, None)
    assert mid_none is None
    assert spread_none is None



def test_compute_feature_record_basic() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    frames = []
    for i in range(60):
        ts = now - timedelta(seconds=60 - i)
        frames.append(
            MarketFeatureFrame(
                timestamp=ts,
                mid_price_yes=0.01 * i,
                volume=100.0 + i,
                liquidity=1000.0 + i,
                total_bid_liquidity_yes=200.0 + i,
                total_ask_liquidity_yes=150.0 + i,
            )
        )

    snapshot = MarketSnapshotRecord(
        timestamp=now,
        market_id="m1",
        yes_price=0.6,
        no_price=0.4,
        best_bid_yes=0.59,
        best_ask_yes=0.61,
        best_bid_no=0.39,
        best_ask_no=0.41,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.60,
        mid_price_no=0.40,
        volume=160.0,
        liquidity=1000.0,
        last_trade_price=0.60,
        last_trade_size=10.0,
        last_trade_time=now,
        has_orderbook=1,
        has_trade_data=1,
    )

    trades = [
        TradeRecord(
            timestamp=now - timedelta(seconds=2),
            market_id="m1",
            trade_id="t1",
            price=0.60,
            size=12.0,
            side="buy",
            maker=None,
            taker=None,
        ),
        TradeRecord(
            timestamp=now - timedelta(seconds=1),
            market_id="m1",
            trade_id="t2",
            price=0.59,
            size=4.0,
            side="sell",
            maker=None,
            taker=None,
        ),
        TradeRecord(
            timestamp=now - timedelta(seconds=20),
            market_id="m1",
            trade_id="t3",
            price=0.58,
            size=99.0,
            side="buy",
            maker=None,
            taker=None,
        ),
    ]

    yes_bids = [PriceLevel(0.59, 100.0), PriceLevel(0.58, 50.0)]
    yes_asks = [PriceLevel(0.61, 80.0), PriceLevel(0.62, 20.0)]

    feature = compute_feature_record(
        snapshot=snapshot,
        frame_history=frames,
        recent_trades=trades,
        yes_bids=yes_bids,
        yes_asks=yes_asks,
        market_start_time=now - timedelta(hours=1),
        market_close_time=now + timedelta(hours=2),
        orderbook_depth=2,
        jump_threshold=0.02,
    )

    assert feature is not None
    assert round(feature.price_change_1s or 0.0, 5) == 0.01
    assert round(feature.price_change_10s or 0.0, 5) == 0.10
    assert round(feature.price_change_60s or 0.0, 5) == 0.60
    assert round(feature.velocity_5s or 0.0, 5) == 0.05
    assert round(feature.acceleration_5s or 0.0, 5) == 0.0

    assert round(feature.total_bid_liquidity_yes or 0.0, 5) == 150.0
    assert round(feature.total_ask_liquidity_yes or 0.0, 5) == 100.0
    assert round(feature.orderbook_imbalance_yes or 0.0, 5) == 0.2

    assert round(feature.buy_volume_5s or 0.0, 5) == 12.0
    assert round(feature.sell_volume_5s or 0.0, 5) == 4.0
    assert round(feature.net_trade_flow_5s or 0.0, 5) == 8.0

    assert round(feature.volume_delta_1s or 0.0, 5) == 1.0
    assert round(feature.avg_volume_60s or 0.0, 5) == 1.0
    assert round(feature.volume_spike_ratio or 0.0, 5) == 1.0

    assert feature.is_price_jump == 0
    assert (feature.time_since_market_created or 0.0) > 3500
    assert (feature.time_until_resolution or 0.0) > 7000


def test_compute_feature_record_requires_full_window_for_60s_features() -> None:
    now = datetime(2026, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    frames = []
    for i in range(10):
        ts = now - timedelta(seconds=9 - i)
        frames.append(
            MarketFeatureFrame(
                timestamp=ts,
                mid_price_yes=0.4 + (0.001 * i),
                volume=100.0 + i,
                liquidity=1000.0 + i,
                total_bid_liquidity_yes=200.0 + i,
                total_ask_liquidity_yes=180.0 + i,
            )
        )

    snapshot = MarketSnapshotRecord(
        timestamp=now,
        market_id="m2",
        yes_price=0.5,
        no_price=0.5,
        best_bid_yes=0.49,
        best_ask_yes=0.51,
        best_bid_no=0.49,
        best_ask_no=0.51,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.5,
        mid_price_no=0.5,
        volume=110.0,
        liquidity=1000.0,
        last_trade_price=None,
        last_trade_size=None,
        last_trade_time=None,
        has_orderbook=1,
        has_trade_data=0,
    )

    feature = compute_feature_record(
        snapshot=snapshot,
        frame_history=frames,
        recent_trades=[],
        yes_bids=[PriceLevel(0.49, 10.0)],
        yes_asks=[PriceLevel(0.51, 12.0)],
        market_start_time=now - timedelta(minutes=1),
        market_close_time=now + timedelta(minutes=4),
        orderbook_depth=1,
        jump_threshold=0.05,
    )

    assert feature is not None
    assert feature.price_change_60s is None
    assert feature.velocity_30s is None
    assert feature.rolling_mean_60s is None


def test_compute_feature_record_trade_features_are_null_without_recognized_trades() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    frames = []
    for i in range(60):
        ts = now - timedelta(seconds=60 - i)
        frames.append(
            MarketFeatureFrame(
                timestamp=ts,
                mid_price_yes=0.45 + (0.001 * i),
                volume=1000.0 + i,
                liquidity=1500.0 + i,
                total_bid_liquidity_yes=500.0 + i,
                total_ask_liquidity_yes=400.0 + i,
            )
        )

    snapshot = MarketSnapshotRecord(
        timestamp=now,
        market_id="m3",
        yes_price=0.5,
        no_price=0.5,
        best_bid_yes=0.49,
        best_ask_yes=0.51,
        best_bid_no=0.49,
        best_ask_no=0.51,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.5,
        mid_price_no=0.5,
        volume=1061.0,
        liquidity=1500.0,
        last_trade_price=None,
        last_trade_size=None,
        last_trade_time=None,
        has_orderbook=1,
        has_trade_data=0,
    )

    feature = compute_feature_record(
        snapshot=snapshot,
        frame_history=frames,
        recent_trades=[],
        yes_bids=[PriceLevel(0.49, 40.0)],
        yes_asks=[PriceLevel(0.51, 35.0)],
        market_start_time=now - timedelta(minutes=10),
        market_close_time=now + timedelta(minutes=1),
        orderbook_depth=1,
        jump_threshold=0.05,
    )

    assert feature is not None
    assert feature.buy_volume_5s is None
    assert feature.sell_volume_5s is None
    assert feature.net_trade_flow_5s is None
    assert feature.trade_flow_ratio_5s is None


def test_compute_feature_record_ignores_stale_trades_for_alignment() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    frames = []
    for i in range(60):
        ts = now - timedelta(seconds=60 - i)
        frames.append(
            MarketFeatureFrame(
                timestamp=ts,
                mid_price_yes=0.30 + (0.001 * i),
                volume=500.0 + i,
                liquidity=1200.0 + i,
                total_bid_liquidity_yes=300.0 + i,
                total_ask_liquidity_yes=280.0 + i,
            )
        )

    snapshot = MarketSnapshotRecord(
        timestamp=now,
        market_id="m4",
        yes_price=0.4,
        no_price=0.6,
        best_bid_yes=0.39,
        best_ask_yes=0.41,
        best_bid_no=0.59,
        best_ask_no=0.61,
        spread_yes=0.02,
        spread_no=0.02,
        mid_price_yes=0.4,
        mid_price_no=0.6,
        volume=560.0,
        liquidity=1800.0,
        last_trade_price=None,
        last_trade_size=None,
        last_trade_time=None,
        has_orderbook=1,
        has_trade_data=0,
    )

    trades = [
        TradeRecord(
            timestamp=now - timedelta(seconds=6),
            market_id="m4",
            trade_id="stale-buy",
            price=0.4,
            size=10.0,
            side="buy",
            maker=None,
            taker=None,
        ),
        TradeRecord(
            timestamp=now - timedelta(seconds=1),
            market_id="m4",
            trade_id="fresh-sell",
            price=0.4,
            size=2.0,
            side="sell",
            maker=None,
            taker=None,
        ),
    ]

    feature = compute_feature_record(
        snapshot=snapshot,
        frame_history=frames,
        recent_trades=trades,
        yes_bids=[PriceLevel(0.39, 30.0)],
        yes_asks=[PriceLevel(0.41, 35.0)],
        market_start_time=now - timedelta(minutes=10),
        market_close_time=now + timedelta(minutes=10),
        orderbook_depth=1,
        jump_threshold=0.05,
    )

    assert feature is not None
    assert round(feature.buy_volume_5s or 0.0, 5) == 0.0
    assert round(feature.sell_volume_5s or 0.0, 5) == 2.0
    assert round(feature.net_trade_flow_5s or 0.0, 5) == -2.0
