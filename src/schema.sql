PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS markets (
    run_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_id TEXT,
    question TEXT,
    description TEXT,
    category TEXT,
    outcomes TEXT,
    resolution_source TEXT,
    start_time TEXT,
    end_time TEXT,
    close_time TEXT,
    platform_status TEXT,
    phase TEXT,
    status TEXT,
    market_phase TEXT,
    tracking_state TEXT,
    yes_token_id TEXT,
    no_token_id TEXT,
    condition_id TEXT,
    resolved INTEGER DEFAULT 0,
    resolved_at TEXT,
    winning_asset_id TEXT,
    winning_outcome TEXT,
    strict_validation_passed INTEGER DEFAULT 0,
    strict_rejection_reason TEXT,
    recorder_version TEXT,
    schema_version TEXT,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    yes_price REAL,
    no_price REAL,
    best_bid_yes REAL,
    best_ask_yes REAL,
    best_bid_no REAL,
    best_ask_no REAL,
    spread_yes REAL,
    spread_no REAL,
    mid_price_yes REAL,
    mid_price_no REAL,
    volume REAL,
    liquidity REAL,
    last_trade_price REAL,
    last_trade_size REAL,
    last_trade_time TEXT,
    has_orderbook INTEGER,
    has_trade_data INTEGER,
    snapshot_quality_status TEXT,
    is_partial_orderbook INTEGER DEFAULT 0,
    missing_level_count INTEGER DEFAULT 0,
    time_gap_from_prev_snapshot_sec REAL,
    is_gap_affected INTEGER DEFAULT 0,
    feature_ready INTEGER DEFAULT 0,
    book_checksum TEXT,
    strict_validation_passed INTEGER DEFAULT 0,
    recorder_version TEXT,
    schema_version TEXT,
    PRIMARY KEY (run_id, timestamp, market_id),
    FOREIGN KEY (run_id, market_id) REFERENCES markets (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS order_book_levels (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    outcome_side TEXT NOT NULL,
    book_side TEXT NOT NULL,
    level INTEGER NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    recorder_version TEXT,
    schema_version TEXT,
    PRIMARY KEY (run_id, timestamp, market_id, outcome_side, book_side, level),
    FOREIGN KEY (run_id, market_id) REFERENCES markets (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    side TEXT,
    maker TEXT,
    taker TEXT,
    recorder_version TEXT,
    schema_version TEXT,
    UNIQUE (run_id, market_id, trade_id),
    FOREIGN KEY (run_id, market_id) REFERENCES markets (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS features (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,

    price_change_1s REAL,
    price_change_10s REAL,
    price_change_60s REAL,
    velocity_5s REAL,
    velocity_30s REAL,
    acceleration_5s REAL,
    rolling_mean_60s REAL,
    distance_from_rolling_mean_60s REAL,

    total_bid_liquidity_yes REAL,
    total_ask_liquidity_yes REAL,
    liquidity_change_bid_5s REAL,
    orderbook_imbalance_yes REAL,

    buy_volume_5s REAL,
    sell_volume_5s REAL,
    net_trade_flow_5s REAL,
    trade_flow_ratio_5s REAL,

    rolling_volatility_10s REAL,
    rolling_volatility_60s REAL,

    volume_delta_1s REAL,
    avg_volume_60s REAL,
    volume_spike_ratio REAL,

    largest_bid_wall_size_yes REAL,
    largest_ask_wall_size_yes REAL,
    distance_to_bid_wall REAL,

    time_since_market_created REAL,
    time_until_resolution REAL,

    is_price_jump INTEGER,
    feature_ready INTEGER DEFAULT 0,
    is_gap_affected INTEGER DEFAULT 0,
    snapshot_quality_status TEXT,
    strict_validation_passed INTEGER DEFAULT 0,
    recorder_version TEXT,
    schema_version TEXT,

    PRIMARY KEY (run_id, timestamp, market_id),
    FOREIGN KEY (run_id, market_id) REFERENCES markets (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS market_events (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details TEXT,
    recorder_version TEXT,
    schema_version TEXT,
    UNIQUE (run_id, timestamp, market_id, event_type, details),
    FOREIGN KEY (run_id, market_id) REFERENCES markets (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS raw_polymarket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    local_arrival_ns INTEGER NOT NULL,
    local_arrival_iso TEXT NOT NULL,
    exchange_timestamp TEXT,
    event_type TEXT,
    market_id TEXT,
    condition_id TEXT,
    asset_id TEXT,
    slug TEXT,
    parse_status TEXT NOT NULL DEFAULT 'ok',
    parse_error TEXT,
    raw_json TEXT NOT NULL,
    recorder_version TEXT,
    schema_version TEXT,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tick_size_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT,
    condition_id TEXT,
    asset_id TEXT,
    old_tick_size REAL,
    new_tick_size REAL,
    raw_json TEXT NOT NULL,
    recorder_version TEXT,
    schema_version TEXT,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS best_bid_ask_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    market_id TEXT,
    condition_id TEXT,
    asset_id TEXT,
    best_bid REAL,
    best_ask REAL,
    spread REAL,
    raw_json TEXT NOT NULL,
    recorder_version TEXT,
    schema_version TEXT,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS btc_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    price REAL NOT NULL,
    exchange_timestamp TEXT,
    local_arrival_ns INTEGER NOT NULL,
    local_arrival_iso TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    recorder_version TEXT,
    schema_version TEXT,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS recorder_metrics (
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    markets_polled INTEGER NOT NULL,
    successful_market_fetches INTEGER NOT NULL,
    failed_markets INTEGER NOT NULL,
    api_latency_ms REAL,
    db_write_time_ms REAL,
    cycle_duration_ms REAL,
    rows_inserted INTEGER NOT NULL,
    duplicate_rows_skipped INTEGER NOT NULL,
    active_markets_snapshot_attempted INTEGER,
    snapshots_written INTEGER,
    snapshot_markets_skipped INTEGER,
    discovery_candidates_seen INTEGER,
    discovery_candidates_matched INTEGER,
    discovery_strict_5m_candidates INTEGER,
    discovery_broad_btc_candidates INTEGER,
    discovery_fallback_used INTEGER,
    tracked_markets_count INTEGER,
    api_call_count INTEGER,
    api_success_count INTEGER,
    api_failure_count INTEGER,
    fetched_trades INTEGER DEFAULT 0,
    accepted_trades INTEGER DEFAULT 0,
    rejected_before_start INTEGER DEFAULT 0,
    rejected_after_close INTEGER DEFAULT 0,
    skipped_already_seen INTEGER DEFAULT 0,
    last_accepted_trade_ts TEXT,
    ws_reconnect_count INTEGER DEFAULT 0,
    raw_ws_events_seen INTEGER DEFAULT 0,
    raw_ws_events_written INTEGER DEFAULT 0,
    malformed_ws_events INTEGER DEFAULT 0,
    raw_ws_write_failures INTEGER DEFAULT 0,
    last_ws_event_age_sec REAL,
    subscribed_asset_count INTEGER DEFAULT 0,
    subscribed_asset_ids_json TEXT,
    recorder_version TEXT,
    schema_version TEXT,
    PRIMARY KEY (run_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_run_market_time
    ON market_snapshots (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_run_time
    ON market_snapshots (run_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_order_book_levels_run_market_time
    ON order_book_levels (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_run_time
    ON order_book_levels (run_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_trades_run_market_time
    ON trades (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_run_time
    ON trades (run_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_features_run_market_time
    ON features (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_events_run_market_time
    ON market_events (run_id, market_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_arrival
    ON raw_polymarket_events (run_id, local_arrival_ns);
CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_event_type
    ON raw_polymarket_events (run_id, event_type);
CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_market
    ON raw_polymarket_events (run_id, market_id);
CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_condition
    ON raw_polymarket_events (run_id, condition_id);
CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_asset
    ON raw_polymarket_events (run_id, asset_id);

CREATE INDEX IF NOT EXISTS idx_tick_size_changes_run_market_time
    ON tick_size_changes (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_tick_size_changes_run_asset_time
    ON tick_size_changes (run_id, asset_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_best_bid_ask_updates_run_market_time
    ON best_bid_ask_updates (run_id, market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_best_bid_ask_updates_run_asset_time
    ON best_bid_ask_updates (run_id, asset_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_btc_prices_run_arrival
    ON btc_prices (run_id, local_arrival_ns);
CREATE INDEX IF NOT EXISTS idx_btc_prices_source_arrival
    ON btc_prices (source, local_arrival_ns);

CREATE INDEX IF NOT EXISTS idx_markets_run_status
    ON markets (run_id, status);
CREATE INDEX IF NOT EXISTS idx_markets_run_platform_status
    ON markets (run_id, platform_status);
CREATE INDEX IF NOT EXISTS idx_markets_run_phase_v2
    ON markets (run_id, phase);
CREATE INDEX IF NOT EXISTS idx_markets_run_phase
    ON markets (run_id, market_phase);
CREATE INDEX IF NOT EXISTS idx_markets_run_tracking_state
    ON markets (run_id, tracking_state);
CREATE INDEX IF NOT EXISTS idx_markets_run_yes_token_id
    ON markets (run_id, yes_token_id);
CREATE INDEX IF NOT EXISTS idx_markets_run_no_token_id
    ON markets (run_id, no_token_id);
CREATE INDEX IF NOT EXISTS idx_markets_run_condition_id
    ON markets (run_id, condition_id);
CREATE INDEX IF NOT EXISTS idx_markets_run_resolved
    ON markets (run_id, resolved);
CREATE INDEX IF NOT EXISTS idx_markets_run_winning_asset_id
    ON markets (run_id, winning_asset_id);
CREATE INDEX IF NOT EXISTS idx_markets_run_strict_validation
    ON markets (run_id, strict_validation_passed);
