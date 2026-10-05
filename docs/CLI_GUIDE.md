# Command and workflow guide

Commands below include database-writing maintenance, online recorder operations,
model training, and paper trading. Use disposable data when learning. Confirm
flags with `python -m src.main --help`. See the main README for current scope
and release limitations.

## Database Schema

Tables implemented:
- `markets`
- `market_snapshots`
- `order_book_levels`
- `trades`
- `features`
- `market_events`
- `recorder_metrics`

Schema is idempotent and includes indexes on `market_id`, `timestamp`, and composite `(market_id, timestamp)` where useful.

`markets` also stores:
- `yes_token_id`, `no_token_id`, `condition_id`
- `platform_status` (raw upstream market status from Polymarket)
- `phase` (derived app phase)
- legacy mirrors: `status`, `market_phase` (kept for backward compatibility)
- `tracking_state` (`discovered`, `selected_active`, `inactive`)

## Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create env config:
```bash
cp .env.example .env
```

3. (Optional) adjust values in `.env`.

Key discovery settings:
- `DISCOVERY_INTERVAL_SEC=120` (2 minutes)
- `DISCOVERY_PAGE_SIZE=500`
- `DISCOVERY_MAX_PAGES=60` (raise only if target markets are missing in deeper pages)
- `DISCOVERY_LOOKAHEAD_SEC=7200` (track only near-term BTC 5m markets; reduces WS load)

## Run

Run as module:
```bash
python -m src.main
```

Or use the root launcher:
```bash
python recorder.py
```

For a 24/7 run with the Polymarket RTDS BTC feed enabled:
```bash
scripts/run_recorder.sh
```

The script uses `.venv/bin/python`, sets BTC feed defaults, and then runs
`python -m src.main`. It does not contain API keys, wallet data, or secrets.

Local read-only dashboards:
```bash
python -m src.main dashboard
python -m src.main web-dashboard
```

The web dashboard serves `http://127.0.0.1:8765` by default, opens SQLite in
read-only mode, and never connects to Polymarket directly.

## Tests

```bash
pytest -q
```

## Diagnostics

```bash
python -m src.main --diagnostics
```

This prints the currently selected primary market (expected: exactly one `selected_active` market when an active BTC 5-minute market exists), standby state, and snapshot/feature completeness percentages.
It also reports trade ingestion totals/latest trade per tracked market and duplicate market-event counts.
It also prints strict-target validation fields:
- selected market effective duration in seconds
- strict 5-minute pass/fail for selected market
- strict 5-minute candidate count vs broad BTC candidate count
- whether discovery fallback was used

Optional historical repair command:
```bash
python -m src.main --repair-invalid-snapshot-trades
```
This nulls snapshot trade fields for rows where:
- `last_trade_time < market.start_time`
- `last_trade_time > snapshot.timestamp`

Runtime maintenance defaults:
- `AUTO_NORMALIZE_RESOLUTIONS_ENABLED=true`
- `AUTO_NORMALIZE_RESOLUTIONS_INTERVAL_SEC=60`
- `AUTO_REPAIR_STALE_ACTIVE_ENABLED=true`
- `AUTO_REPAIR_STALE_ACTIVE_INTERVAL_SEC=300`
- `AUTO_REPAIR_STALE_ACTIVE_GRACE_SEC=60`
- `RECORDER_HEARTBEAT_INTERVAL_SEC=30`

Automatic maintenance reuses the same safe logic as:
```bash
python -m src.main normalize-resolutions
python -m src.main repair-stale-active-markets
```

It updates market metadata only: resolution fields and stale
`selected_active` labels. It does not delete snapshots, trades, order books,
features, raw websocket events, BTC price rows, or market rows.

Discovery-debug command:
```bash
python -m src.main debug-discovery
```
This runs one discovery cycle and prints:
- per-page HTTP status, URL, response bytes, top-level payload keys, extraction path
- stage-by-stage filter counts
- strict vs broad BTC candidate counts and fallback mode
- first raw items and first candidate rejection decisions
- selected market preview (or `None`)

## Market State Model

The recorder keeps three distinct market-state concepts:
- `platform_status`: raw status from Polymarket payloads (`open`, `resolved`, `closed`, etc.)
- `tracking_state`: recorder-level handling state
  - `selected_active`: single market currently tracked in runtime state
  - `discovered`: discovered candidate metadata, not actively tracked
  - `inactive`: no longer tracked by this recorder
- `phase`: recorder-derived time lifecycle
  - `future`: `now < start_time`
  - `active`: `start_time <= now < close_time` and not terminal by status
  - `expired_waiting_resolution`: `now >= close_time` while upstream status is still non-terminal
  - `resolved`: terminal status (`resolved`, `settled`, `closed`, `finalized`)

## Primary Selection

Discovery may return many BTC Up/Down 5-minute candidates, but runtime tracking is single-market:
- choose market where `start_time <= now < close_time`
- if multiple match, choose nearest `close_time`
- if none are active, choose nearest upcoming `start_time` as standby (kept as `discovered`, not `selected_active`)
- hard invariant: selected market duration must be in `[240, 360]` seconds
- if no strict 5-minute candidate is valid, select nothing and keep scanning
- snapshots/features are written only for the single `selected_active` market while phase is `active`
- when `now >= close_time`, recorder triggers rotation discovery immediately

This ensures exactly one active market is tracked at a time and prevents snapshot/feature writes for future, resolved, or non-5-minute BTC products.

`market_opened` emission is transition-based and deduplicated:
- emitted once when a market becomes `selected_active`
- not re-emitted on rediscovery, resubscription, reconnects, or refreshes

## Startup and Recovery Flow

On startup:
1. initialize SQLite schema
2. discover active markets from Gamma (targeted BTC 5m slug scan first, broad scan fallback only if targeted returns zero) and filter to BTC Up/Down 5-minute markets
3. build token->market mapping
4. fetch initial orderbook snapshots via CLOB REST
5. connect CLOB websocket
6. maintain in-memory state continuously
7. persist snapshots/features/levels/trades/metrics every second

On websocket disconnect:
- reconnect with exponential backoff
- refresh orderbooks from CLOB REST before reprocessing stream data

During runtime discovery (default every 120s):
- detect newly created BTC 5-minute markets and dynamically subscribe new token ids
- prune markets no longer active (resolved/replaced) from in-memory tracking
- keep historical rows in SQLite for downstream training/export

Discovery endpoint and shape assumptions:
- primary endpoint mode: `GET https://gamma-api.polymarket.com/markets?active=true&closed=false&slug=btc-updown-5m-<epoch>`
- fallback endpoint mode (diagnostic/resilience): `GET https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=...&offset=...`
- common response shape: top-level list of market objects
- also supported: `{\"markets\": [...]}`, `{\"data\": [...]}`, `{\"data\": {\"markets\": [...]}}`, `{\"items\": [...]}`, `{\"results\": [...]}`, and `{\"events\": [{\"markets\": [...]}]}`
- normalization logs payload shape and extraction path so schema drift is visible

Discovery timestamp/phase logic:
- targeted discovery scans a rolling slug window (`btc-updown-5m-<epoch>`) around current UTC time to avoid selecting stale/higher-timeframe BTC products
- duration calculation (strict 5m) uses effective `eventStartTime/startTime/startDate` and `closeTime/closedTime/endDate`; falls back to question time-range parsing when explicit bounds are missing
- `start_time` precedence is `eventStartTime -> startTime -> startDate`
- `close_time` precedence is `closeTime -> closedTime -> endDate`
- phase classification for discovery uses effective window bounds (`eventStartTime`/`endDate` when present), plus platform flags (`active`, `closed`, `acceptingOrders`)
- non-5-minute BTC Up/Down candidates (for example 1H/4H/daily) are explicitly rejected for selection with `non_5m_market_rejected`
- strict selection never falls back to broader BTC Up/Down products

## Recorder Metrics Semantics

The recorder now stores explicit per-cycle metrics:
- `active_markets_snapshot_attempted`: active tracked markets considered for snapshotting this cycle
- `snapshots_written`: snapshot rows inserted this cycle
- `snapshot_markets_skipped`: tracked markets skipped due to phase/state/orderbook gating
- `discovery_candidates_seen`: candidate count returned from discovery before final BTC-5m selection
- `discovery_candidates_matched`: count matched by BTC 5-minute filter
- `discovery_strict_5m_candidates`: strict BTC Up/Down 5-minute candidates discovered this cycle
- `discovery_broad_btc_candidates`: BTC Up/Down candidates rejected as non-5m this cycle
- `discovery_fallback_used`: 1 if non-strict fallback was used (expected 0 in strict mode)
- `tracked_markets_count`: markets currently retained in runtime app state
- `api_call_count`: upstream API calls attempted this cycle
- `api_success_count`: successful upstream API calls this cycle
- `api_failure_count`: failed upstream API calls this cycle

Legacy columns are still populated for backward compatibility:
- `markets_polled` is a legacy alias of `active_markets_snapshot_attempted`
- `successful_market_fetches` is a legacy alias of `discovery_candidates_matched`
- `failed_markets` is a legacy alias of `api_failure_count`

## Feature Null Handling

Feature computation is strict about missing data:
- if source state is unavailable, feature values are stored as `NULL`
- missing values are never coerced to `0.0`
- windowed features require real lookback coverage for that full window
  - e.g. `price_change_10s` requires 10-second lookback
  - e.g. `rolling_mean_60s` and `rolling_volatility_60s` require 60-second coverage
- trade-flow features are `NULL` if no recognized buy/sell trades exist in window

## Trade-State Validation Rules

Snapshot trade fields are guarded so stale trade cache cannot leak between markets:
- `last_trade_*` values are only emitted when the trade is valid for the selected market window
- `last_trade_*` is taken from the most recent accepted trade timestamp (not payload iteration order)
- if `last_trade_time < market.start_time`, snapshot trade fields are set to `NULL` and `has_trade_data=0`
- if `last_trade_time > snapshot.timestamp`, snapshot trade fields are set to `NULL` and `has_trade_data=0`
- trades with token mismatches or outside market window are rejected from runtime trade state
- recorder logs warning events when stale or inconsistent trade state is detected

## Volume And Trade Feature Logic

The recorder separates three concepts:
- cumulative market volume in snapshots:
  - `snapshot.volume = initial_volume + accepted_trade_sizes_since_runtime_start`
- per-second volume delta features:
  - `volume_delta_1s = volume_now - volume_1s_ago`
  - `avg_volume_60s` is computed from rolling per-frame volume deltas only when a full 60s window exists
- trade-window flow features:
  - `buy_volume_5s`, `sell_volume_5s`, `net_trade_flow_5s`, `trade_flow_ratio_5s`
  - computed only from recognized BUY/SELL trades in the last 5 seconds for the active market

## Validation Command Coverage

`python -m src.main --diagnostics` reports:
- current selected active market and its `platform_status` / `tracking_state` / `phase`
- snapshot completeness:
  - `% non-null best_bid_yes`
  - `% non-null mid_price_yes`
  - `% non-null last_trade_price`
  - `% has_orderbook = 1`
  - `% has_trade_data = 1`
  - invalid snapshot trade row counts:
  - `last_trade_time < market.start_time`
  - `last_trade_time > snapshot.timestamp`
- feature completeness for core price/liquidity/trade/volume fields
- event integrity:
  - duplicate `(market_id, event_type)` groups
  - active market `market_opened` count
  - terminal event count for resolved markets
- market integrity:
  - counts by `tracking_state`
  - counts by `phase`
  - selected-active integrity alert
- trade ingestion summary:
  - total trade rows
  - latest trade per active market
  - recent 5-minute buy/sell side counts

## Current Limitations

- Websocket subscribe payload shape can change over time; this implementation uses a conservative adapter and logs unknown payloads.
- Some Polymarket API payload fields are not fully documented in this repo, so normalization includes explicit defensive fallbacks.
- Trade backfill parameter naming may require adjustment against current data-api docs.
- Retention cleanup is not yet implemented (schema and code paths are ready for it).
- If no active BTC Up/Down 5-minute markets exist, the recorder will run with zero tracked markets until new ones appear.
- Gamma market pagination can be large; discovery is bounded by `DISCOVERY_MAX_PAGES` and logs `market_discovery_truncated` if that bound is hit.
- Discovery can legitimately return zero selected markets when the platform has no live BTC Up/Down ~5-minute windows; use `debug-discovery` to confirm whether this is filtering logic vs. market availability.
- If response shape drifts and extraction fails, discovery logs `discovery_zero_markets_after_filtering` with stage counts and payload-shape diagnostics.
- Some BTC Up/Down markets expose hourly or longer windows; these are intentionally excluded from selection even if active.

## Assumptions

- Binary markets (`YES` / `NO`) are the primary target.
- `clobTokenIds` ordering is treated as outcome order when explicit token metadata is missing.
- Volume in snapshots is maintained as `initial_volume + observed_trade_sizes_since_start` when available.
- `yes_price` / `no_price` are derived from mid-price when orderbook sides are available.
- Discovery is restricted to live markets (`active=true`, not closed) and then filtered to questions containing both `BTC` and `Up or Down`.
- A BTC market is considered a 5-minute candidate when `close_time - start_time` is approximately 4-7 minutes.
- The recorder must only select strict BTC 5-minute markets; 1H BTC Up/Down markets must never be selected.

## Future Parquet Export Fit

This MVP keeps recent operational data in SQLite. A future export job can read old rows from:
- `market_snapshots`
- `order_book_levels`
- `trades`
- `features`

and write partitioned Parquet datasets for model training without changing recorder internals.

## How Pattern Bot / Paper Trader Can Read Later

Future consumers can read SQLite directly for near-realtime state:
- pattern bot: poll latest `market_snapshots` + `features`
- paper trader: read latest top-of-book from `order_book_levels` and recent `trades`

This decouples ingestion reliability from strategy experimentation.
