from __future__ import annotations

import json
import sqlite3
import sys
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.dashboard import connect_read_only
from src.db import SQLiteStore
from src.models import to_iso
from src.web_dashboard import (
    create_web_dashboard_server,
    fetch_paper_activity_payload,
    fetch_paper_closed_payload,
    fetch_paper_open_payload,
    fetch_paper_skips_payload,
    fetch_paper_traders_summary,
    fetch_series_payload,
    fetch_web_dashboard_payload,
    resolve_paper_db_paths,
)


RUN_ID = "run_web_dashboard"
MARKET_ID = "market_web"
YES = "yes_web_asset"
NO = "no_web_asset"


def _init_db(db_path) -> SQLiteStore:
    store = SQLiteStore(str(db_path), run_id=RUN_ID, quarantine_on_startup=False)
    store.init_schema()
    return store


def _insert_active_market(conn: sqlite3.Connection, *, now: datetime) -> None:
    start = now - timedelta(minutes=1)
    close = now + timedelta(minutes=4)
    conn.execute(
        """
        INSERT INTO markets (
            run_id, market_id, question, start_time, end_time, close_time,
            platform_status, status, phase, market_phase, tracking_state,
            yes_token_id, no_token_id, created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            MARKET_ID,
            "Bitcoin Up or Down - web dashboard",
            to_iso(start),
            to_iso(close),
            to_iso(close),
            "open",
            "open",
            "active",
            "active",
            "selected_active",
            YES,
            NO,
            to_iso(start),
            to_iso(now),
        ),
    )


def _insert_snapshot(
    conn: sqlite3.Connection,
    *,
    timestamp: datetime,
    yes_bid: float,
    no_bid: float,
) -> None:
    conn.execute(
        """
        INSERT INTO market_snapshots (
            run_id, timestamp, market_id,
            best_bid_yes, best_ask_yes, spread_yes,
            best_bid_no, best_ask_no, spread_no,
            mid_price_yes, mid_price_no,
            last_trade_price, last_trade_size, last_trade_time,
            has_orderbook, has_trade_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            MARKET_ID,
            yes_bid,
            yes_bid + 0.02,
            0.02,
            no_bid,
            no_bid + 0.02,
            0.02,
            yes_bid + 0.01,
            no_bid + 0.01,
            yes_bid,
            3.0,
            to_iso(timestamp),
            1,
            1,
        ),
    )


def _insert_btc(
    conn: sqlite3.Connection,
    *,
    timestamp: datetime,
    price: float,
    idx: int,
    source: str = "polymarket_rtds_chainlink",
) -> None:
    conn.execute(
        """
        INSERT INTO btc_prices (
            run_id, source, price, exchange_timestamp,
            local_arrival_ns, local_arrival_iso, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            source,
            price,
            to_iso(timestamp - timedelta(milliseconds=50)),
            1_767_225_660_000_000_000 + idx,
            to_iso(timestamp),
            json.dumps({"price": price}),
        ),
    )


def _insert_best_bid_ask(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    for asset_id, bid, ask in ((YES, 0.51, 0.53), (NO, 0.47, 0.49)):
        conn.execute(
            """
            INSERT INTO best_bid_ask_updates (
                run_id, timestamp, market_id, condition_id, asset_id,
                best_bid, best_ask, spread, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                RUN_ID,
                to_iso(timestamp),
                MARKET_ID,
                "condition_web",
                asset_id,
                bid,
                ask,
                ask - bid,
                "{}",
            ),
        )


def _insert_trade(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    conn.execute(
        """
        INSERT INTO trades (
            run_id, timestamp, market_id, trade_id, price, size, side
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            MARKET_ID,
            f"trade_web:{YES}",
            0.52,
            2.0,
            "BUY",
        ),
    )


def _insert_metric(conn: sqlite3.Connection, *, timestamp: datetime) -> None:
    conn.execute(
        """
        INSERT INTO recorder_metrics (
            run_id, timestamp, markets_polled, successful_market_fetches,
            failed_markets, rows_inserted, duplicate_rows_skipped,
            raw_ws_events_seen, raw_ws_events_written, malformed_ws_events,
            raw_ws_write_failures, last_ws_event_age_sec,
            subscribed_asset_count, ws_reconnect_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            RUN_ID,
            to_iso(timestamp),
            1,
            1,
            0,
            10,
            0,
            200,
            200,
            0,
            0,
            0.4,
            2,
            1,
        ),
    )


def _seed_live_db(db_path, *, now: datetime) -> None:
    store = _init_db(db_path)
    try:
        with store.conn:
            _insert_active_market(store.conn, now=now)
            for idx in range(3):
                ts = now - timedelta(seconds=2 - idx)
                _insert_snapshot(
                    store.conn,
                    timestamp=ts,
                    yes_bid=0.49 + idx * 0.01,
                    no_bid=0.48 - idx * 0.01,
                )
                _insert_btc(
                    store.conn,
                    timestamp=ts,
                    price=50_000.0 + idx * 10.0,
                    idx=idx,
                )
            _insert_best_bid_ask(store.conn, timestamp=now)
            _insert_trade(store.conn, timestamp=now)
            _insert_metric(store.conn, timestamp=now)
    finally:
        store.close()


def _table_counts(db_path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "markets",
                "market_snapshots",
                "btc_prices",
                "best_bid_ask_updates",
                "trades",
                "recorder_metrics",
            )
        }
    finally:
        conn.close()


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


def _create_paper_db(db_path, *, label: str = "paper", full_schema: bool = True) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        if full_schema:
            conn.execute(
                """
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    run_id TEXT,
                    market_id TEXT,
                    question TEXT,
                    signal_timestamp TEXT,
                    market_start_time TEXT,
                    market_close_time TEXT,
                    strategy_id TEXT,
                    strategy_name TEXT,
                    signal_direction TEXT,
                    status TEXT,
                    skip_reason TEXT,
                    stake_usd REAL,
                    entry_price REAL,
                    adjusted_entry_price REAL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    exit_price REAL,
                    adjusted_exit_price REAL,
                    payout_usd REAL,
                    pnl_usd REAL,
                    roi REAL,
                    realized_pnl_usd REAL,
                    realized_roi REAL,
                    settled_at TEXT,
                    predicted_probability_yes REAL,
                    probability_for_direction REAL,
                    estimated_edge REAL,
                    time_until_resolution REAL,
                    starting_bankroll_usd REAL,
                    bankroll_before_trade REAL,
                    available_cash_before_trade REAL,
                    open_exposure_before_trade REAL,
                    max_open_exposure_usd REAL,
                    bankroll_after_trade REAL,
                    liquidity_skip_reason TEXT,
                    realistic_execution_skip_reason TEXT,
                    btc_trend_regime TEXT,
                    volatility_regime TEXT,
                    spread_regime TEXT,
                    liquidity_regime TEXT,
                    time_regime TEXT
                )
                """
            )
            rows = [
                ("2026-01-01T00:00:01+00:00", "m_open", "s_yes", "YES", "open", None, 1.0, 0.50, 0.51, None, None, None, None, None, None, 0.82, 0.31, 120, 100, 100, 99, 0, 20, 100, None, None, "weak_up"),
                ("2026-01-01T00:00:02+00:00", "m_await", "s_yes", "YES", "awaiting_resolution", None, 1.0, 0.52, 0.53, None, None, None, None, None, None, 0.80, 0.27, 80, 100, 100, 98, 1, 20, 100, None, None, "weak_up"),
                ("2026-01-01T00:00:03+00:00", "m_closed", "s_cash", "NO", "closed", None, 1.0, 0.40, 0.41, "2026-01-01T00:00:12+00:00", 0.47, 0.46, None, None, 0.12, 0.30, 0.20, 60, 100, 100, 97, 2, 20, 100.12, None, None, "flat"),
                ("2026-01-01T00:00:04+00:00", "m_settled", "s_yes", "YES", "settled", None, 1.0, 0.60, 0.61, None, None, None, 1.64, 0.64, None, 0.84, 0.23, 30, 100, 100, 96, 3, 20, 100.76, None, None, "strong_up"),
                ("2026-01-01T00:00:05+00:00", "m_skip", "s_skip", "YES", "skipped", "quote_after_latency_missing", 1.0, 0.55, 0.56, None, None, None, None, None, None, 0.78, 0.22, 75, 100, 100, 96, 3, 20, 100.76, "liquidity_missing", "quote_after_latency_missing", "weak_down"),
            ]
            conn.executemany(
                """
                INSERT INTO paper_trades (
                    created_at, run_id, market_id, strategy_id, signal_direction,
                    status, skip_reason, stake_usd, entry_price, adjusted_entry_price,
                    exit_time, exit_price, adjusted_exit_price, payout_usd, pnl_usd,
                    realized_pnl_usd, probability_for_direction, estimated_edge,
                    time_until_resolution, starting_bankroll_usd, bankroll_before_trade,
                    available_cash_before_trade, open_exposure_before_trade,
                    max_open_exposure_usd, bankroll_after_trade, liquidity_skip_reason,
                    realistic_execution_skip_reason, btc_trend_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(row[0], RUN_ID, *row[1:]) for row in rows],
            )
        else:
            conn.execute(
                """
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    market_id TEXT,
                    status TEXT,
                    stake_usd REAL,
                    pnl_usd REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO paper_trades (created_at, market_id, status, stake_usd, pnl_usd)
                VALUES ('2026-01-01T00:00:01+00:00', ?, 'settled', 1.0, 0.25)
                """,
                (label,),
            )
        conn.commit()
    finally:
        conn.close()


def test_web_dashboard_api_returns_active_market_and_latest_data(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    before = _table_counts(db_path)

    payload = fetch_web_dashboard_payload(str(db_path), now=now)

    assert _table_counts(db_path) == before
    assert payload["active_market"]["market_id"] == MARKET_ID
    assert payload["btc_price"]["price"] == 50_020.0
    assert payload["btc_price"]["source"] == "polymarket_rtds_chainlink"
    assert payload["btc_price"]["sample_age_sec"] == 0.0
    assert payload["btc_prices_by_source"] == [
        {
            "exchange_timestamp": "2026-01-01T00:01:59.950000+00:00",
            "local_arrival_iso": "2026-01-01T00:02:00+00:00",
            "local_arrival_ns": 1_767_225_660_000_000_002,
            "price": 50_020.0,
            "sample_age_sec": 0.0,
            "source": "polymarket_rtds_chainlink",
        }
    ]
    assert payload["latest_snapshot"]["best_bid_yes"] == 0.51
    assert payload["best_bid_ask"]["YES"]["best_bid"] == 0.51
    assert payload["best_bid_ask"]["NO"]["best_ask"] == 0.49
    assert payload["health"]["raw_ws_events_seen"] == 200
    assert payload["recent_trades"][0]["outcome"] == "YES"
    assert payload["warnings"] == []


def test_web_dashboard_api_warns_when_btc_sample_is_stale(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("DELETE FROM btc_prices")
            _insert_btc(
                conn,
                timestamp=now - timedelta(seconds=4),
                price=50_100.0,
                idx=100,
            )
    finally:
        conn.close()

    payload = fetch_web_dashboard_payload(str(db_path), now=now)

    assert payload["btc_price"]["sample_age_sec"] == 4.0
    assert "stale_btc_price" in payload["warnings"]
    assert "stale_btc_price_source_polymarket_rtds_chainlink" in payload["warnings"]


def test_web_dashboard_api_warns_when_both_btc_sources_are_stale(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("DELETE FROM btc_prices")
            _insert_btc(
                conn,
                timestamp=now - timedelta(seconds=4),
                price=50_100.0,
                idx=100,
                source="polymarket_rtds_chainlink",
            )
            _insert_btc(
                conn,
                timestamp=now - timedelta(seconds=5),
                price=50_090.0,
                idx=101,
                source="polymarket_rtds_binance",
            )
    finally:
        conn.close()

    payload = fetch_web_dashboard_payload(str(db_path), now=now)

    ages = {
        row["source"]: row["sample_age_sec"]
        for row in payload["btc_prices_by_source"]
    }
    assert ages == {
        "polymarket_rtds_binance": 5.0,
        "polymarket_rtds_chainlink": 4.0,
    }
    assert "stale_btc_price_source_polymarket_rtds_chainlink" in payload["warnings"]
    assert "stale_btc_price_source_polymarket_rtds_binance" in payload["warnings"]
    assert "both_btc_sources_stale" in payload["warnings"]


def test_web_dashboard_preferred_btc_source_selection(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            _insert_btc(
                conn,
                timestamp=now,
                price=60_000.0,
                idx=100,
                source="polymarket_rtds_binance",
            )
    finally:
        conn.close()

    latest_payload = fetch_web_dashboard_payload(str(db_path), now=now)
    preferred_payload = fetch_web_dashboard_payload(
        str(db_path),
        now=now,
        preferred_btc_source="polymarket_rtds_chainlink",
    )
    series = fetch_series_payload(
        str(db_path),
        window_sec=300,
        now=now,
        preferred_btc_source="polymarket_rtds_chainlink",
    )

    assert latest_payload["btc_price"]["source"] == "polymarket_rtds_binance"
    assert latest_payload["btc_price"]["price"] == 60_000.0
    assert preferred_payload["btc_price"]["source"] == "polymarket_rtds_chainlink"
    assert preferred_payload["btc_price"]["price"] == 50_020.0
    assert series["btc_source"] == "polymarket_rtds_chainlink"
    assert {row["source"] for row in series["btc_price"]} == {
        "polymarket_rtds_chainlink"
    }


def test_web_dashboard_latest_btc_row_uses_local_arrival_order(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            _insert_btc(
                conn,
                timestamp=now - timedelta(seconds=1),
                price=50_999.0,
                idx=500,
            )
    finally:
        conn.close()

    dashboard = fetch_web_dashboard_payload(str(db_path), now=now)
    series = fetch_series_payload(str(db_path), window_sec=300, now=now)

    assert dashboard["btc_price"]["price"] == 50_999.0
    assert dashboard["btc_price"]["sample_age_sec"] == 1.0
    assert series["btc_price"][-1]["value"] == 50_999.0


def test_web_dashboard_series_returns_btc_yes_and_no_arrays(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)

    payload = fetch_series_payload(str(db_path), window_sec=300, now=now)

    assert payload["market_id"] == MARKET_ID
    assert [row["value"] for row in payload["btc_price"]] == [
        50_000.0,
        50_010.0,
        50_020.0,
    ]
    assert [row["value"] for row in payload["yes_best_bid"]] == [0.49, 0.5, 0.51]
    assert [row["value"] for row in payload["no_best_bid"]] == pytest.approx(
        [0.48, 0.47, 0.46]
    )
    assert payload["snapshot_count"] == 3


def test_web_dashboard_missing_active_market_returns_safe_warning(tmp_path) -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    store.close()

    dashboard = fetch_web_dashboard_payload(str(db_path), now=now)
    series = fetch_series_payload(str(db_path), window_sec=300, now=now)

    assert dashboard["active_market"] is None
    assert "no_selected_active_market" in dashboard["warnings"]
    assert "missing_btc_price" in dashboard["warnings"]
    assert series["market_id"] is None
    assert series["yes_best_bid"] == []
    assert series["no_best_bid"] == []


def test_web_dashboard_paper_summary_handles_full_schema(tmp_path) -> None:
    paper_db = tmp_path / "paper_full.db"
    _create_paper_db(paper_db, full_schema=True)

    payload = fetch_paper_traders_summary([str(paper_db)], now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))
    trader = payload["traders"][0]

    assert trader["label"] == "paper_full.db"
    assert trader["total_trades"] == 5
    assert trader["open_awaiting_trades"] == 2
    assert trader["closed_trades"] == 1
    assert trader["settled_trades"] == 1
    assert trader["skipped_trades"] == 1
    assert trader["realized_pnl"] == pytest.approx(0.76)
    assert trader["current_bankroll"] == pytest.approx(100.76)
    assert trader["skip_reason_counts"]["quote_after_latency_missing"] == 1
    assert trader["liquidity_block_counts"]["liquidity_missing"] == 1


def test_web_dashboard_paper_summary_works_without_paper_dbs() -> None:
    payload = fetch_paper_traders_summary([], now=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))

    assert payload["traders"] == []
    assert payload["totals"]["paper_db_count"] == 0


def test_web_dashboard_paper_summary_handles_old_missing_columns(tmp_path) -> None:
    paper_db = tmp_path / "paper_old.db"
    _create_paper_db(paper_db, full_schema=False)

    payload = fetch_paper_traders_summary([str(paper_db)], now=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))
    trader = payload["traders"][0]

    assert trader["status"] == "STALE"
    assert trader["total_trades"] == 1
    assert trader["settled_trades"] == 1
    assert trader["realized_pnl"] == 0.25
    assert trader["current_bankroll"] is None


def test_web_dashboard_activity_endpoint_merges_multiple_paper_dbs(tmp_path) -> None:
    paper_a = tmp_path / "paper_a.db"
    paper_b = tmp_path / "paper_b.db"
    _create_paper_db(paper_a, full_schema=True)
    _create_paper_db(paper_b, full_schema=False)

    payload = fetch_paper_activity_payload([str(paper_a), str(paper_b)], limit=10)

    labels = {row["trader_label"] for row in payload["rows"]}
    assert labels == {"paper_a.db", "paper_b.db"}
    assert any(row["market_id"] == "m_skip" for row in payload["rows"])


def test_web_dashboard_skips_endpoint_groups_skip_reasons(tmp_path) -> None:
    paper_db = tmp_path / "paper_skips.db"
    _create_paper_db(paper_db, full_schema=True)

    payload = fetch_paper_skips_payload([str(paper_db)], limit=20)

    assert payload["groups"]["skip_reason"]["quote_after_latency_missing"] == 1
    assert payload["groups"]["realistic_execution_skip_reason"]["quote_after_latency_missing"] == 1
    assert payload["groups"]["liquidity_skip_reason"]["liquidity_missing"] == 1
    assert payload["groups"]["strategy_id"]["s_skip"] == 1
    assert payload["rows"][0]["status"] == "skipped"


def test_web_dashboard_open_and_closed_paper_endpoints(tmp_path) -> None:
    paper_db = tmp_path / "paper_open_closed.db"
    _create_paper_db(paper_db, full_schema=True)

    open_payload = fetch_paper_open_payload([str(paper_db)])
    closed_payload = fetch_paper_closed_payload([str(paper_db)])

    assert {row["status"] for row in open_payload["rows"]} == {"open", "awaiting_resolution"}
    assert {row["status"] for row in closed_payload["rows"]} == {"closed", "settled"}
    assert closed_payload["rows"][0]["cashout_exit_price"] is not None


def test_web_dashboard_resolves_repeated_paper_db_and_glob(tmp_path) -> None:
    paper_a = tmp_path / "paper_a.db"
    paper_b = tmp_path / "paper_b.db"
    _create_paper_db(paper_a, full_schema=False)
    _create_paper_db(paper_b, full_schema=False)

    resolved = resolve_paper_db_paths(
        [str(paper_a), str(paper_a)],
        [str(tmp_path / "paper_*.db")],
    )

    assert resolved == [str(paper_a), str(paper_b)]


@pytest.mark.loopback
def test_web_dashboard_server_serves_html_and_json_read_only(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    db_path = tmp_path / "recorder.db"
    _seed_live_db(db_path, now=now)
    before = _table_counts(db_path)

    paper_db = tmp_path / "paper_server.db"
    _create_paper_db(paper_db, full_schema=True)

    server = create_web_dashboard_server(
        str(db_path),
        host="127.0.0.1",
        port=0,
        paper_db_paths=[str(paper_db)],
    )
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=3) as response:
            html = response.read().decode("utf-8")
        dashboard = _get_json(f"http://{host}:{port}/api/dashboard")
        recorder_summary = _get_json(f"http://{host}:{port}/api/recorder/summary")
        paper_summary = _get_json(f"http://{host}:{port}/api/paper-traders/summary")
        activity = _get_json(f"http://{host}:{port}/api/paper-traders/activity")
        skips = _get_json(f"http://{host}:{port}/api/paper-traders/skips")
        open_rows = _get_json(f"http://{host}:{port}/api/paper-traders/open")
        closed_rows = _get_json(f"http://{host}:{port}/api/paper-traders/closed")
        series = _get_json(f"http://{host}:{port}/api/series?window_sec=300")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert "Polymarket BTC 5m Recorder Dashboard" in html
    assert "sample age sec" in html
    assert "btc-stale-warning" in html
    assert "Paper Traders" in html
    assert dashboard["active_market"]["market_id"] == MARKET_ID
    assert recorder_summary["active_market"]["market_id"] == MARKET_ID
    assert paper_summary["traders"][0]["total_trades"] == 5
    assert activity["rows"]
    assert skips["groups"]["skip_reason"]["quote_after_latency_missing"] == 1
    assert open_rows["rows"]
    assert closed_rows["rows"]
    assert "sample_age_sec" in dashboard["btc_price"]
    assert series["btc_price"]
    assert _table_counts(db_path) == before


def test_web_dashboard_uses_read_only_sqlite_connections(tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    store.close()

    conn = connect_read_only(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                """
                INSERT INTO markets (run_id, market_id, created_at, last_updated)
                VALUES ('x', 'y', 'now', 'now')
                """
            )
    finally:
        conn.close()


def test_web_dashboard_cli_invokes_server_without_recorder(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "recorder.db"
    store = _init_db(db_path)
    store.close()

    import src.main as main_module
    import src.web_dashboard as web_dashboard_module

    calls: list[dict[str, object]] = []
    settings = SimpleNamespace(
        db_path=str(db_path),
        log_level="CRITICAL",
        log_json=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("web-dashboard must not start network or recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "web-dashboard",
            "--web-dashboard-host",
            "127.0.0.1",
            "--web-dashboard-port",
            "8765",
            "--web-dashboard-refresh-sec",
            "0.5",
            "--web-dashboard-btc-source",
            "polymarket_rtds_chainlink",
            "--paper-db",
            str(tmp_path / "paper_a.db"),
            "--paper-db",
            str(tmp_path / "paper_b.db"),
            "--paper-db-glob",
            str(tmp_path / "paper_*.db"),
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", lambda env_file=None: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "GammaClient", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(
        web_dashboard_module,
        "run_web_dashboard",
        lambda db_path, **kwargs: calls.append({"db_path": db_path, **kwargs}),
    )

    main_module.main()

    assert calls == [
        {
            "db_path": str(db_path),
            "host": "127.0.0.1",
            "port": 8765,
            "refresh_sec": 0.5,
            "preferred_btc_source": "polymarket_rtds_chainlink",
            "paper_db_paths": [str(tmp_path / "paper_a.db"), str(tmp_path / "paper_b.db")],
            "paper_db_globs": [str(tmp_path / "paper_*.db")],
        }
    ]
