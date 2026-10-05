"""Generate fictional recorder-shaped rows for exercising baseline training."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random


def synthetic_rows(count: int = 320, seed: int = 42) -> list[dict]:
    if count < 1:
        raise ValueError('Row count must be positive')
    rng = random.Random(seed)
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        market, second = divmod(index, 20)
        start = origin + timedelta(minutes=market * 5)
        timestamp = start + timedelta(seconds=second)
        close = start + timedelta(minutes=5)
        # Fictional alternating market outcomes. Prices are generated independently
        # of the labels; this is a pipeline exercise, not predictive evidence.
        label = market % 2
        mid = rng.uniform(0.35, 0.65)
        spread = 0.02
        rows.append({
            'run_id': 'synthetic_demo', 'market_id': f'synthetic_market_{market}',
            'question': 'Fictional BTC five-minute outcome',
            'timestamp': timestamp.isoformat(), 'start_time': start.isoformat(),
            'close_time': close.isoformat(), 'market_phase': 'active',
            'yes_token_id': 'synthetic_yes', 'no_token_id': 'synthetic_no',
            'export_row_usable': 1, 'feature_ready': 1, 'is_gap_affected': 0,
            'snapshot_quality_status': 'ok', 'strict_validation_passed': 1,
            'label_btc_up_at_resolution': label, 'label_yes_win': label,
            'label_resolved_up_down': 'UP' if label else 'DOWN',
            'future_btc_return_to_resolution_from_feature': 0.001 if label else -0.001,
            'time_until_resolution': float(300 - second),
            'seconds_after_start': float(second), 'seconds_before_close': float(300 - second),
            'best_bid_yes': mid - spread / 2, 'best_ask_yes': mid + spread / 2,
            'best_bid_no': 1 - mid - spread / 2, 'best_ask_no': 1 - mid + spread / 2,
            'spread_yes': spread, 'spread_no': spread,
            'mid_price_yes': mid, 'mid_price_no': 1 - mid,
            'total_bid_liquidity_yes': rng.uniform(10, 30),
            'total_ask_liquidity_yes': rng.uniform(10, 30),
            'orderbook_imbalance_yes': rng.uniform(-0.2, 0.2),
            'btc_chainlink_price': 100.0 + rng.uniform(-0.5, 0.5),
            'btc_chainlink_age_sec_at_feature': 1.0,
            'btc_binance_age_sec_at_feature': 1.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/synthetic/training.parquet')
    parser.add_argument('--rows', type=int, default=320)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; select a new path to preserve it')
    rows = synthetic_rows(args.rows, args.seed)
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    print(json.dumps({'status': 'ok', 'synthetic': True, 'rows': len(rows),
                      'output': str(output), 'purpose': 'Pipeline demonstration only'}))


if __name__ == '__main__':
    main()
