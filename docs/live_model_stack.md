# Live Model Stack

This stack is for shadow/paper monitoring of the BTC 5-minute models. It does
not promote models and does not place real orders by default.

## Start Recorder

```bash
RAW_WS_EVENTS_ENABLED=false \
BTC_PRICE_FEED_ENABLED=true \
BTC_PRICE_FEED_SOURCES=polymarket_rtds_chainlink,polymarket_rtds_binance \
BTC_PRICE_DROP_STALE=true \
BTC_PRICE_MAX_EXCHANGE_AGE_SEC=15 \
BTC_PRICE_STALE_RECONNECT_SEC=30 \
STARTUP_INTEGRITY_CHECK_MODE=quick \
.venv/bin/python -m src.main --db data/recorder.db
```

Long-lived readers can prevent WAL truncation. Use `wal-checkpoint` for safe
visibility into `.db`, `.db-wal`, and `.db-shm` sizes.

## Start Live Predictions

```bash
.venv/bin/python -m src.main run-live-model-predictions \
  --recorder-db data/recorder.db \
  --output-db data/live_predictions.db \
  --rf-model-path data/models/baseline_retrain_20260509T095118Z/model_random_forest.joblib \
  --rf-feature-columns data/models/baseline_retrain_20260509T095118Z/feature_columns.json \
  --transformer-model-dir data/models/transformer_sequence_retrain_20260509T095118Z \
  --poll-sec 1 \
  --max-feature-age-sec 5 \
  --max-btc-age-sec 15 \
  --min-confidence 0.55 \
  --run-id live_models_$(date -u +%Y%m%dT%H%M%SZ) \
  --shadow-only
```

## Score Predictions

```bash
.venv/bin/python -m src.main score-live-model-predictions \
  --recorder-db data/recorder.db \
  --prediction-db data/live_predictions.db \
  --lookback-hours 48
```

## Terminal Dashboard

```bash
.venv/bin/python -m src.main live-model-dashboard \
  --recorder-db data/recorder.db \
  --prediction-db data/live_predictions.db \
  --refresh-sec 2 \
  --lookback-markets 100
```

## Combined Launcher

```bash
.venv/bin/python -m src.main start-live-model-stack \
  --recorder-db data/recorder.db \
  --output-db data/live_predictions.db \
  --rf-model-path data/models/baseline_retrain_20260509T095118Z/model_random_forest.joblib \
  --rf-feature-columns data/models/baseline_retrain_20260509T095118Z/feature_columns.json \
  --transformer-model-dir data/models/transformer_sequence_retrain_20260509T095118Z \
  --shadow-only \
  --dashboard
```

The dashboard displays `SHADOW`, `PAPER`, or `LIVE`. `LIVE` requires explicitly
passing `--enable-live-trading`, and this stack still performs no real order
placement.
