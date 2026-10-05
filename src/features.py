from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from .models import FeatureRecord, MarketSnapshotRecord, PriceLevel, TradeRecord


ROLLING_WINDOW_SECONDS = 60
TRADE_FLOW_WINDOW_SECONDS = 5


@dataclass(slots=True, frozen=True)
class MarketFeatureFrame:
    timestamp: datetime
    mid_price_yes: Optional[float]
    # Snapshot volume is treated as cumulative and converted to deltas at feature time.
    volume: Optional[float]
    liquidity: Optional[float]
    total_bid_liquidity_yes: Optional[float]
    total_ask_liquidity_yes: Optional[float]

    @property
    def price(self) -> Optional[float]:
        return self.mid_price_yes

def mid_and_spread(
    best_bid: Optional[float], best_ask: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if best_bid is None or best_ask is None:
        return None, None
    return (best_bid + best_ask) / 2.0, best_ask - best_bid



def liquidity_sum(levels: Sequence[PriceLevel], top_n: int) -> float:
    return float(sum(level.size for level in levels[: max(0, top_n)]))


def _frame_n_seconds_ago(
    history: Sequence[MarketFeatureFrame],
    now: datetime,
    seconds: int,
) -> Optional[MarketFeatureFrame]:
    cutoff = now - timedelta(seconds=seconds)
    for frame in reversed(history):
        if frame.timestamp <= cutoff:
            return frame
    return None


def _rolling_prices_with_now(
    history: Sequence[MarketFeatureFrame],
    now: datetime,
    seconds: int,
) -> list[float]:
    cutoff = now - timedelta(seconds=seconds)
    values: list[float] = []
    for frame in history:
        if frame.timestamp < cutoff:
            continue
        value = frame.mid_price_yes
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)



def _std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(var)



def _side_to_flow(side: Optional[str]) -> Optional[str]:
    if side is None:
        return None
    normalized = side.strip().lower()
    if normalized in {"buy", "bid", "b", "taker_buy"}:
        return "buy"
    if normalized in {"sell", "ask", "s", "taker_sell"}:
        return "sell"
    if "buy" in normalized or "bid" in normalized:
        return "buy"
    if "sell" in normalized or "ask" in normalized:
        return "sell"
    return None



def compute_feature_record(
    snapshot: MarketSnapshotRecord,
    frame_history: Sequence[MarketFeatureFrame],
    recent_trades: Sequence[TradeRecord],
    yes_bids: Sequence[PriceLevel],
    yes_asks: Sequence[PriceLevel],
    market_start_time: Optional[datetime],
    market_close_time: Optional[datetime],
    orderbook_depth: int,
    jump_threshold: float,
) -> Optional[FeatureRecord]:
    now = snapshot.timestamp
    mid_now = snapshot.mid_price_yes
    vol_now = snapshot.volume

    frame_1s = _frame_n_seconds_ago(frame_history, now, 1)
    frame_5s = _frame_n_seconds_ago(frame_history, now, 5)
    frame_6s = _frame_n_seconds_ago(frame_history, now, 6)
    frame_10s = _frame_n_seconds_ago(frame_history, now, 10)
    frame_30s = _frame_n_seconds_ago(frame_history, now, 30)
    frame_60s = _frame_n_seconds_ago(frame_history, now, 60)

    if mid_now is None:
        return None

    mid_1 = frame_1s.mid_price_yes if frame_1s is not None else None
    mid_5 = frame_5s.mid_price_yes if frame_5s is not None else None
    mid_10 = frame_10s.mid_price_yes if frame_10s is not None else None
    mid_30 = frame_30s.mid_price_yes if frame_30s is not None else None
    mid_60 = frame_60s.mid_price_yes if frame_60s is not None else None
    vol_1s = frame_1s.volume if frame_1s is not None else None

    price_change_1s = mid_now - mid_1 if mid_1 is not None else None
    price_change_10s = mid_now - mid_10 if mid_10 is not None else None
    price_change_60s = mid_now - mid_60 if mid_60 is not None else None

    velocity_5s = mid_now - mid_5 if mid_5 is not None else None
    velocity_30s = mid_now - mid_30 if mid_30 is not None else None

    velocity_5s_prev = None
    if frame_6s is not None and frame_6s.mid_price_yes is not None:
        velocity_5s_prev = mid_1 - frame_6s.mid_price_yes
    acceleration_5s = (
        velocity_5s - velocity_5s_prev
        if velocity_5s is not None and velocity_5s_prev is not None
        else None
    )

    rolling_mid_60: list[float] = []
    if mid_60 is not None:
        rolling_mid_60 = _rolling_prices_with_now(frame_history, now, ROLLING_WINDOW_SECONDS)
        rolling_mid_60.append(mid_now)
    rolling_mean_60s = _mean(rolling_mid_60)
    distance_from_rolling_mean_60s = (
        mid_now - rolling_mean_60s
        if mid_now is not None and rolling_mean_60s is not None
        else None
    )

    total_bid_liquidity_yes = liquidity_sum(yes_bids, orderbook_depth)
    total_ask_liquidity_yes = liquidity_sum(yes_asks, orderbook_depth)

    bid_liquidity_5s_ago = (
        frame_5s.total_bid_liquidity_yes if frame_5s is not None else None
    )
    liquidity_change_bid_5s = (
        total_bid_liquidity_yes - bid_liquidity_5s_ago
        if bid_liquidity_5s_ago is not None
        else None
    )

    imbalance_den = total_bid_liquidity_yes + total_ask_liquidity_yes
    orderbook_imbalance_yes = (
        (total_bid_liquidity_yes - total_ask_liquidity_yes) / imbalance_den
        if imbalance_den > 0
        else None
    )

    # 5s window is intentional: these fields are explicitly `*_5s`.
    trade_flow_cutoff = now - timedelta(seconds=TRADE_FLOW_WINDOW_SECONDS)
    trades_5s = [
        trade
        for trade in recent_trades
        if trade_flow_cutoff <= trade.timestamp <= now
    ]
    buy_volume_5s: Optional[float] = None
    sell_volume_5s: Optional[float] = None
    net_trade_flow_5s: Optional[float] = None
    trade_flow_ratio_5s: Optional[float] = None
    if trades_5s:
        buy_acc = 0.0
        sell_acc = 0.0
        recognized_trades = 0
        for trade in trades_5s:
            flow_side = _side_to_flow(trade.side)
            if flow_side == "buy":
                buy_acc += trade.size
                recognized_trades += 1
            elif flow_side == "sell":
                sell_acc += trade.size
                recognized_trades += 1

        if recognized_trades > 0:
            buy_volume_5s = buy_acc
            sell_volume_5s = sell_acc
            net_trade_flow_5s = buy_acc - sell_acc
            flow_denom = buy_acc + sell_acc
            if flow_denom > 0:
                trade_flow_ratio_5s = (buy_acc - sell_acc) / flow_denom

    rolling_mid_10: list[float] = []
    if mid_10 is not None:
        rolling_mid_10 = _rolling_prices_with_now(frame_history, now, 10)
        rolling_mid_10.append(mid_now)
    rolling_mid_60_for_vol: list[float] = []
    if mid_60 is not None:
        rolling_mid_60_for_vol = _rolling_prices_with_now(frame_history, now, ROLLING_WINDOW_SECONDS)
        rolling_mid_60_for_vol.append(mid_now)
    rolling_volatility_10s = _std(rolling_mid_10)
    rolling_volatility_60s = _std(rolling_mid_60_for_vol)

    volume_delta_1s = (
        vol_now - vol_1s
        if vol_now is not None and vol_1s is not None
        else None
    )

    avg_volume_60s = None
    volume_spike_ratio = None
    volume_frames = [
        frame
        for frame in frame_history
        if frame.timestamp >= now - timedelta(seconds=ROLLING_WINDOW_SECONDS)
        and frame.volume is not None
    ]
    if vol_now is not None:
        volume_frames.append(
            MarketFeatureFrame(
                timestamp=now,
                mid_price_yes=mid_now,
                volume=vol_now,
                liquidity=snapshot.liquidity,
                total_bid_liquidity_yes=total_bid_liquidity_yes,
                total_ask_liquidity_yes=total_ask_liquidity_yes,
            )
        )
    if len(volume_frames) >= 2 and mid_60 is not None:
        deltas = []
        for prev, curr in zip(volume_frames, volume_frames[1:]):
            if prev.volume is None or curr.volume is None:
                continue
            # Volume is cumulative; per-step activity is the delta.
            deltas.append(curr.volume - prev.volume)
        avg_volume_60s = _mean(deltas)

    if volume_delta_1s is not None and avg_volume_60s not in (None, 0.0):
        volume_spike_ratio = volume_delta_1s / avg_volume_60s

    largest_bid_wall_size_yes = max((level.size for level in yes_bids), default=None)
    largest_ask_wall_size_yes = max((level.size for level in yes_asks), default=None)

    largest_bid_wall_price = None
    if yes_bids:
        largest_bid = max(yes_bids, key=lambda level: level.size)
        largest_bid_wall_price = largest_bid.price

    distance_to_bid_wall = (
        mid_now - largest_bid_wall_price
        if mid_now is not None and largest_bid_wall_price is not None
        else None
    )

    time_since_market_created = None
    if market_start_time is not None:
        time_since_market_created = (now - market_start_time).total_seconds()

    time_until_resolution = None
    if market_close_time is not None:
        time_until_resolution = (market_close_time - now).total_seconds()

    is_price_jump = None
    if price_change_1s is not None:
        is_price_jump = 1 if abs(price_change_1s) > jump_threshold else 0

    return FeatureRecord(
        timestamp=now,
        market_id=snapshot.market_id,
        price_change_1s=price_change_1s,
        price_change_10s=price_change_10s,
        price_change_60s=price_change_60s,
        velocity_5s=velocity_5s,
        velocity_30s=velocity_30s,
        acceleration_5s=acceleration_5s,
        rolling_mean_60s=rolling_mean_60s,
        distance_from_rolling_mean_60s=distance_from_rolling_mean_60s,
        total_bid_liquidity_yes=total_bid_liquidity_yes,
        total_ask_liquidity_yes=total_ask_liquidity_yes,
        liquidity_change_bid_5s=liquidity_change_bid_5s,
        orderbook_imbalance_yes=orderbook_imbalance_yes,
        buy_volume_5s=buy_volume_5s,
        sell_volume_5s=sell_volume_5s,
        net_trade_flow_5s=net_trade_flow_5s,
        trade_flow_ratio_5s=trade_flow_ratio_5s,
        rolling_volatility_10s=rolling_volatility_10s,
        rolling_volatility_60s=rolling_volatility_60s,
        volume_delta_1s=volume_delta_1s,
        avg_volume_60s=avg_volume_60s,
        volume_spike_ratio=volume_spike_ratio,
        largest_bid_wall_size_yes=largest_bid_wall_size_yes,
        largest_ask_wall_size_yes=largest_ask_wall_size_yes,
        distance_to_bid_wall=distance_to_bid_wall,
        time_since_market_created=time_since_market_created,
        time_until_resolution=time_until_resolution,
        is_price_jump=is_price_jump,
        run_id=snapshot.run_id,
        strict_validation_passed=snapshot.strict_validation_passed,
        feature_ready=0,
        is_gap_affected=snapshot.is_gap_affected,
        snapshot_quality_status=snapshot.snapshot_quality_status,
        recorder_version=snapshot.recorder_version,
        schema_version=snapshot.schema_version,
    )
