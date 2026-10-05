from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from src.paper_trader_analytics import build_paper_trader_analytics_report


NOW = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _create_db(path, *, trades: bool = True, heartbeat: bool = True) -> None:
    conn = sqlite3.connect(str(path))
    try:
        if trades:
            conn.execute(
                """
                CREATE TABLE paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    run_id TEXT,
                    market_id TEXT,
                    signal_timestamp TEXT,
                    question TEXT,
                    market_start_time TEXT,
                    market_close_time TEXT,
                    time_until_resolution REAL,
                    model_path TEXT,
                    model_name TEXT,
                    predicted_probability_yes REAL,
                    signal_direction TEXT,
                    threshold_used REAL,
                    stake_usd REAL,
                    entry_price REAL,
                    adjusted_entry_price REAL,
                    best_bid_yes REAL,
                    best_ask_yes REAL,
                    best_bid_no REAL,
                    best_ask_no REAL,
                    btc_chainlink_price REAL,
                    btc_binance_price REAL,
                    btc_chainlink_age_sec_at_feature REAL,
                    btc_binance_age_sec_at_feature REAL,
                    feature_ready INTEGER,
                    strict_validation_passed INTEGER,
                    snapshot_quality_status TEXT,
                    status TEXT,
                    skip_reason TEXT,
                    exit_time TEXT,
                    exit_reason TEXT,
                    exit_price REAL,
                    adjusted_exit_price REAL,
                    realized_pnl_usd REAL,
                    realized_roi REAL,
                    max_favorable_price REAL,
                    max_adverse_price REAL,
                    starting_bankroll_usd REAL,
                    bankroll_before_trade REAL,
                    available_cash_before_trade REAL,
                    open_exposure_before_trade REAL,
                    stake_fraction_of_bankroll REAL,
                    max_open_exposure_usd REAL,
                    bankroll_after_trade REAL,
                    bankroll_status TEXT,
                    blown_up_at TEXT,
                    risk_sizing_reason TEXT,
                    btc_trend_regime TEXT,
                    volatility_regime TEXT,
                    spread_regime TEXT,
                    liquidity_regime TEXT,
                    time_regime TEXT,
                    requested_shares REAL,
                    max_fillable_shares REAL,
                    liquidity_fill_fraction_used REAL,
                    liquidity_check_passed INTEGER,
                    liquidity_skip_reason TEXT,
                    realistic_execution_enabled INTEGER,
                    entry_latency_sec REAL,
                    exit_latency_sec REAL,
                    signal_entry_price REAL,
                    delayed_entry_price REAL,
                    entry_price_drift REAL,
                    spread_cents_at_entry REAL,
                    realistic_execution_skip_reason TEXT,
                    extra_slippage_cents_applied REAL,
                    resolved_label TEXT,
                    payout_usd REAL,
                    pnl_usd REAL,
                    roi REAL,
                    settled_at TEXT
                )
                """
            )
        if heartbeat:
            conn.execute(
                """
                CREATE TABLE paper_trader_heartbeats (
                    timestamp TEXT PRIMARY KEY,
                    recorder_db_path TEXT,
                    output_db_path TEXT,
                    latest_feature_timestamp TEXT,
                    latest_seen_feature_timestamp TEXT,
                    latest_eligible_feature_timestamp TEXT,
                    latest_btc_chainlink_age_sec REAL,
                    latest_btc_binance_age_sec REAL,
                    open_trades INTEGER,
                    awaiting_resolution_trades INTEGER,
                    settled_trades INTEGER,
                    total_trades INTEGER,
                    stale_feature_skips INTEGER,
                    market_closed_skips INTEGER,
                    expired_feature_skips INTEGER,
                    loop_error TEXT
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def _insert_trade(
    path,
    *,
    run_id: str = "run1",
    market_id: str = "m1",
    direction: str = "YES",
    probability_yes: float = 0.7,
    adjusted_entry_price: float = 0.55,
    stake_usd: float = 1.0,
    status: str = "settled",
    pnl_usd: float | None = 0.8,
    roi: float | None = 0.8,
    time_until_resolution: float = 45.0,
    created_at: datetime | None = None,
    close_time: datetime | None = None,
    skip_reason: str | None = None,
) -> None:
    created = created_at or (NOW - timedelta(minutes=5))
    close = close_time or (NOW - timedelta(minutes=1))
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_timestamp, question,
                market_start_time, market_close_time, time_until_resolution,
                model_path, model_name, predicted_probability_yes,
                signal_direction, threshold_used, stake_usd,
                entry_price, adjusted_entry_price,
                best_bid_yes, best_ask_yes, best_bid_no, best_ask_no,
                btc_chainlink_price, btc_binance_price,
                btc_chainlink_age_sec_at_feature, btc_binance_age_sec_at_feature,
                feature_ready, strict_validation_passed, snapshot_quality_status,
                status, skip_reason, resolved_label, payout_usd, pnl_usd, roi, settled_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                _iso(created),
                run_id,
                market_id,
                _iso(created + timedelta(seconds=1)),
                "BTC Up or Down",
                _iso(created - timedelta(minutes=1)),
                _iso(close),
                time_until_resolution,
                "model.joblib",
                "RandomForestClassifier",
                probability_yes,
                direction,
                0.65 if direction == "YES" else 0.35,
                stake_usd,
                adjusted_entry_price,
                adjusted_entry_price,
                0.50,
                0.55,
                0.44,
                0.45,
                100.0,
                100.2,
                1.0,
                1.0,
                1,
                1,
                "ok",
                status,
                skip_reason,
                direction if status == "settled" and pnl_usd and pnl_usd > 0 else None,
                (stake_usd + (pnl_usd or 0.0)) if status == "settled" and pnl_usd is not None else None,
                pnl_usd,
                roi,
                _iso(NOW - timedelta(seconds=10)) if status == "settled" else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_heartbeat(path, *, eligible: str | None = None) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            INSERT INTO paper_trader_heartbeats (
                timestamp, recorder_db_path, output_db_path,
                latest_feature_timestamp, latest_seen_feature_timestamp,
                latest_eligible_feature_timestamp, latest_btc_chainlink_age_sec,
                latest_btc_binance_age_sec, open_trades,
                awaiting_resolution_trades, settled_trades, total_trades,
                stale_feature_skips, market_closed_skips, expired_feature_skips,
                loop_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(NOW),
                "recorder.db",
                str(path),
                _iso(NOW - timedelta(seconds=2)),
                _iso(NOW - timedelta(seconds=2)),
                eligible,
                1.0,
                1.0,
                0,
                0,
                2,
                2,
                0,
                0,
                0,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_multi_strategy_heartbeat(path, *, liquidity_enabled: bool = True) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE multi_strategy_paper_trader_heartbeats (
                timestamp TEXT PRIMARY KEY,
                run_id TEXT,
                active_strategies INTEGER,
                latest_market_id TEXT,
                latest_feature_timestamp TEXT,
                probability_yes REAL,
                candidates_logged INTEGER,
                trades_opened INTEGER,
                skips_logged INTEGER,
                liquidity_fill_check_enabled INTEGER,
                liquidity_checked_count INTEGER,
                liquidity_passed_count INTEGER,
                liquidity_blocked_count INTEGER,
                liquidity_missing_count INTEGER,
                per_strategy_json TEXT,
                strategy_errors_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO multi_strategy_paper_trader_heartbeats (
                timestamp, run_id, active_strategies, latest_market_id,
                latest_feature_timestamp, probability_yes, candidates_logged,
                trades_opened, skips_logged, liquidity_fill_check_enabled,
                liquidity_checked_count, liquidity_passed_count,
                liquidity_blocked_count, liquidity_missing_count,
                per_strategy_json, strategy_errors_json
            ) VALUES (?, 'run1', 1, 'm1', ?, 0.7, 1, 0, 1, ?, 0, 0, 0, 0, ?, '{}')
            """,
            (
                _iso(NOW),
                _iso(NOW - timedelta(seconds=2)),
                1 if liquidity_enabled else 0,
                json.dumps([
                    {
                        "strategy_id": "s1",
                        "liquidity_fill_check_enabled": liquidity_enabled,
                    }
                ]),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _add_strategy_id_column(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN strategy_id TEXT")
        conn.commit()
    finally:
        conn.close()


def _mark_closed_trade(
    path,
    *,
    market_id: str,
    strategy_id: str = "cashout_s",
    pnl: float,
    roi: float,
    exit_time: datetime | None = None,
) -> None:
    exit_at = exit_time or (NOW - timedelta(seconds=20))
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET status = 'closed',
                strategy_id = ?,
                exit_time = ?,
                exit_reason = ?,
                exit_price = ?,
                adjusted_exit_price = ?,
                realized_pnl_usd = ?,
                realized_roi = ?
            WHERE market_id = ?
            """,
            (strategy_id, _iso(exit_at), "take_profit" if pnl > 0 else "stop_loss", 0.60, 0.60, pnl, roi, market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _set_bankroll_fields(
    path,
    *,
    starting: float = 100.0,
    status: str = "ACTIVE",
    blown_up_at: str | None = None,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET starting_bankroll_usd = ?,
                bankroll_before_trade = ?,
                available_cash_before_trade = ?,
                open_exposure_before_trade = ?,
                stake_fraction_of_bankroll = stake_usd / ?,
                max_open_exposure_usd = ?,
                bankroll_after_trade = ?,
                bankroll_status = ?,
                blown_up_at = ?,
                risk_sizing_reason = 'test sizing'
            """,
            (starting, starting, starting, starting, starting, starting * 0.2, starting, status, blown_up_at),
        )
        conn.commit()
    finally:
        conn.close()


def _set_regime_fields(
    path,
    *,
    market_id: str,
    btc_trend_regime: str = "strong_up",
    volatility_regime: str = "medium",
    spread_regime: str = "normal",
    liquidity_regime: str = "normal",
    time_regime: str = "30-60s",
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET btc_trend_regime = ?,
                volatility_regime = ?,
                spread_regime = ?,
                liquidity_regime = ?,
                time_regime = ?
            WHERE market_id = ?
            """,
            (
                btc_trend_regime,
                volatility_regime,
                spread_regime,
                liquidity_regime,
                time_regime,
                market_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _set_liquidity_fields(
    path,
    *,
    market_id: str,
    requested_shares: float | None,
    max_fillable_shares: float | None,
    fill_fraction: float = 1.0,
    passed: int,
    skip_reason: str | None,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET requested_shares = ?,
                max_fillable_shares = ?,
                liquidity_fill_fraction_used = ?,
                liquidity_check_passed = ?,
                liquidity_skip_reason = ?
            WHERE market_id = ?
            """,
            (
                requested_shares,
                max_fillable_shares,
                fill_fraction,
                passed,
                skip_reason,
                market_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _set_realistic_execution_fields(
    path,
    *,
    market_id: str,
    enabled: int = 1,
    skip_reason: str | None = None,
    drift: float | None = 1.0,
    spread: float | None = 2.0,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            UPDATE paper_trades
            SET realistic_execution_enabled = ?,
                entry_latency_sec = 1.0,
                exit_latency_sec = 1.0,
                signal_entry_price = 0.50,
                delayed_entry_price = 0.51,
                entry_price_drift = ?,
                spread_cents_at_entry = ?,
                realistic_execution_skip_reason = ?,
                extra_slippage_cents_applied = 1.0
            WHERE market_id = ?
            """,
            (enabled, drift, spread, skip_reason, market_id),
        )
        conn.commit()
    finally:
        conn.close()


def _create_candidate_table(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE paper_trade_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                run_id TEXT,
                market_id TEXT,
                feature_timestamp TEXT,
                strategy_id TEXT,
                candidate_direction TEXT,
                candidate_adjusted_entry_price REAL,
                stake_usd REAL,
                predicted_probability_yes REAL,
                probability_for_direction REAL,
                estimated_edge REAL,
                decision TEXT,
                rejection_reason TEXT,
                actual_resolved_label TEXT,
                would_have_payout_usd REAL,
                would_have_pnl_usd REAL,
                would_have_roi REAL,
                settled_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_candidate_row(
    path,
    *,
    decision: str,
    rejection_reason: str | None = None,
    roi: float | None = 0.5,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            INSERT INTO paper_trade_candidates (
                created_at, run_id, market_id, feature_timestamp, strategy_id,
                candidate_direction, candidate_adjusted_entry_price, stake_usd,
                predicted_probability_yes, probability_for_direction,
                estimated_edge, decision, rejection_reason,
                actual_resolved_label, would_have_payout_usd,
                would_have_pnl_usd, would_have_roi, settled_at
            ) VALUES (?, 'run1', ?, ?, 's1', 'YES', 0.5, 1.0, 0.8, 0.8,
                0.3, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(NOW),
                f"candidate_{decision}_{rejection_reason or 'trade'}",
                _iso(NOW - timedelta(seconds=1)),
                decision,
                rejection_reason,
                "YES" if roi is not None else None,
                (1.0 + roi) if roi is not None else None,
                roi if roi is not None else None,
                roi,
                _iso(NOW) if roi is not None else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_missing_paper_trades_table_is_reported(tmp_path) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "report"
    _create_db(db, trades=False, heartbeat=False)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(output),
        now=NOW,
    )

    assert report["summary"]["paper_trades_table_missing"] is True
    assert "paper_trades_table_missing" in report["summary"]["warnings"]
    assert (output / "paper_trader_report.json").exists()
    assert (output / "paper_trader_report.txt").exists()
    assert (output / "trades_enriched.csv").exists()


def test_empty_table_reports_zero_trades(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["summary"]["total_trades"] == 0
    assert report["summary"]["settled_trades"] == 0
    assert "fewer_than_30_settled_trades" in report["summary"]["warnings"]


def test_probability_for_direction_and_estimated_edge_for_yes_and_no(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="yes", direction="YES", probability_yes=0.72, adjusted_entry_price=0.55)
    _insert_trade(db, market_id="no", direction="NO", probability_yes=0.20, adjusted_entry_price=0.65)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    by_market = {row["market_id"]: row for row in report["trades"]}

    assert by_market["yes"]["probability_for_direction"] == 0.72
    assert by_market["yes"]["estimated_edge"] == 0.17
    assert by_market["yes"]["estimated_edge_cents"] == 17.0
    assert by_market["no"]["probability_for_direction"] == 0.8
    assert by_market["no"]["estimated_edge"] == 0.15


def test_settled_win_loss_metrics_and_direction_breakdown(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="win_yes", direction="YES", pnl_usd=0.8, roi=0.8)
    _insert_trade(db, market_id="lose_no", direction="NO", probability_yes=0.8, pnl_usd=-1.0, roi=-1.0)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    trades = {row["market_id"]: row for row in report["trades"]}
    summary = report["summary"]

    assert trades["win_yes"]["settled_win"] == 1
    assert trades["lose_no"]["settled_win"] == 0
    assert summary["settled_pnl"] == -0.2
    assert summary["win_rate_settled"] == 0.5
    assert summary["direction_breakdown"]["YES"]["pnl"] == 0.8
    assert summary["direction_breakdown"]["NO"]["pnl"] == -1.0
    assert summary["win_rate_by_direction"]["YES"] == 1.0
    assert summary["win_rate_by_direction"]["NO"] == 0.0


def test_edge_and_time_buckets(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="negative", probability_yes=0.40, adjusted_entry_price=0.55, time_until_resolution=10)
    _insert_trade(db, market_id="small", probability_yes=0.56, adjusted_entry_price=0.55, time_until_resolution=45)
    _insert_trade(db, market_id="large", probability_yes=0.90, adjusted_entry_price=0.55, time_until_resolution=200)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["summary"]["edge_buckets"]["edge < 0"]["trade_count"] == 1
    assert report["summary"]["edge_buckets"]["0 to 0.02"]["trade_count"] == 1
    assert report["summary"]["edge_buckets"]["edge >= 0.20"]["trade_count"] == 1
    assert report["summary"]["time_until_resolution_buckets"]["0-30s"]["trade_count"] == 1
    assert report["summary"]["time_until_resolution_buckets"]["30-60s"]["trade_count"] == 1
    assert report["summary"]["time_until_resolution_buckets"]["180s+"]["trade_count"] == 1


def test_unresolved_exposure_worst_and_best_case(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="settled", status="settled", pnl_usd=2.0, adjusted_entry_price=0.50)
    _insert_trade(db, market_id="open", status="open", pnl_usd=None, roi=None, stake_usd=2.0, adjusted_entry_price=0.50)
    _insert_trade(db, market_id="awaiting", status="awaiting_resolution", pnl_usd=None, roi=None, stake_usd=3.0, adjusted_entry_price=0.25)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    exposure = report["summary"]["unresolved_exposure"]

    assert exposure["open_stake"] == 2.0
    assert exposure["awaiting_resolution_stake"] == 3.0
    assert exposure["total_unresolved_stake"] == 5.0
    assert exposure["worst_case_total_pnl_if_all_unresolved_lose"] == -3.0
    assert exposure["best_case_total_pnl_if_all_unresolved_win"] == 13.0


def test_duplicate_market_warning(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, run_id="run", market_id="m1")
    _insert_trade(db, run_id="run", market_id="m1")

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert "run:m1" in report["summary"]["duplicate_market_keys"]
    assert "duplicate_trades_in_same_run_id_market_id" in report["summary"]["warnings"]


def test_skipped_diagnostics_do_not_trigger_duplicate_market_warning(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, run_id="run", market_id="m1", status="settled")
    _insert_trade(
        db,
        run_id="run",
        market_id="m1",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="min_estimated_edge",
    )
    _insert_trade(
        db,
        run_id="run",
        market_id="m1",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="missing_entry_price",
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["summary"]["status_counts"]["skipped"] == 2
    assert report["summary"]["duplicate_market_keys"] == []
    assert "duplicate_trades_in_same_run_id_market_id" not in report["summary"]["warnings"]


def test_multiple_real_trades_still_trigger_duplicate_market_warning(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, run_id="run", market_id="m1", status="open", pnl_usd=None, roi=None)
    _insert_trade(
        db,
        run_id="run",
        market_id="m1",
        status="awaiting_resolution",
        pnl_usd=None,
        roi=None,
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert "run:m1" in report["summary"]["duplicate_market_keys"]
    assert "duplicate_trades_in_same_run_id_market_id" in report["summary"]["warnings"]


def test_stale_awaiting_resolution_warning(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(
        db,
        status="awaiting_resolution",
        pnl_usd=None,
        roi=None,
        created_at=NOW - timedelta(minutes=31),
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["trades"][0]["unresolved_age_sec"] == 1860.0
    assert "awaiting_resolution_trade_older_than_30_minutes" in report["summary"]["warnings"]


def test_latest_heartbeat_included_and_warning_for_seen_without_eligible(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_heartbeat(db, eligible=None)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["summary"]["latest_heartbeat"]["latest_seen_feature_timestamp"] == _iso(NOW - timedelta(seconds=2))
    assert "latest_eligible_feature_timestamp_null_while_seen_feature_recent" in report["summary"]["warnings"]


def test_analytics_includes_candidate_level_stats_when_table_exists(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _create_candidate_table(db)
    _insert_candidate_row(db, decision="TRADE", roi=0.4)
    _insert_candidate_row(db, decision="SKIP", rejection_reason="min_estimated_edge", roi=0.2)
    _insert_candidate_row(db, decision="SKIP", rejection_reason="threshold", roi=-1.0)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["total_candidates"] == 3
    assert summary["settled_candidates"] == 3
    assert summary["candidate_trade_rate"] == 0.3333333333
    assert summary["skipped_candidate_count"] == 2
    assert summary["skipped_candidate_would_have_positive_roi_count"] == 1
    assert summary["average_would_have_roi_by_decision"]["TRADE"] == 0.4
    assert summary["average_would_have_roi_by_rejection_reason"]["min_estimated_edge"] == 0.2


def test_strategy_breakdown_reports_edge_and_warns_on_non_positive_average(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN strategy_id TEXT")
        conn.commit()
    finally:
        conn.close()
    _insert_trade(
        db,
        market_id="bad_edge",
        probability_yes=0.45,
        adjusted_entry_price=0.55,
        pnl_usd=-1.0,
        roi=-1.0,
    )
    _insert_trade(
        db,
        market_id="good_edge",
        probability_yes=0.80,
        adjusted_entry_price=0.55,
        pnl_usd=0.8,
        roi=0.8,
    )
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("UPDATE paper_trades SET strategy_id = 'bad_strategy' WHERE market_id = 'bad_edge'")
        conn.execute("UPDATE paper_trades SET strategy_id = 'good_strategy' WHERE market_id = 'good_edge'")
        conn.commit()
    finally:
        conn.close()

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    by_strategy = report["summary"]["strategy_breakdown"]

    assert list(by_strategy) == ["good_strategy", "bad_strategy"]
    assert by_strategy["bad_strategy"]["trade_count"] == 1
    assert by_strategy["bad_strategy"]["settled_count"] == 1
    assert by_strategy["bad_strategy"]["settled_pnl"] == -1.0
    assert by_strategy["bad_strategy"]["win_rate"] == 0.0
    assert by_strategy["bad_strategy"]["avg_roi"] == -1.0
    assert by_strategy["bad_strategy"]["average_estimated_edge"] == -0.1
    assert by_strategy["bad_strategy"]["candidate_trade_rate"] is None
    assert "strategy_with_non_positive_average_estimated_edge" in report["summary"]["warnings"]
    assert (
        "strategy_non_positive_average_estimated_edge:bad_strategy"
        in report["summary"]["warnings"]
    )


def test_analytics_reports_cashout_pnl_and_strategy_drawdown(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _add_strategy_id_column(db)
    _insert_trade(db, market_id="closed_win", status="closed", pnl_usd=None, roi=None)
    _insert_trade(db, market_id="closed_loss", status="closed", pnl_usd=None, roi=None)
    _insert_trade(db, market_id="settled_win", pnl_usd=0.8, roi=0.8)
    _mark_closed_trade(db, market_id="closed_win", pnl=0.2, roi=0.2)
    _mark_closed_trade(db, market_id="closed_loss", pnl=-0.1, roi=-0.1)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("UPDATE paper_trades SET strategy_id = 'cashout_s' WHERE market_id = 'settled_win'")
        conn.commit()
    finally:
        conn.close()

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]
    strategy = summary["strategy_breakdown"]["cashout_s"]

    assert summary["closed_trades"] == 2
    assert summary["hold_to_resolution_pnl"] == 0.8
    assert summary["realized_cashout_pnl"] == 0.1
    assert summary["combined_realized_pnl"] == 0.9
    assert summary["cashout_win_rate"] == 0.5
    assert summary["average_cashout_win"] == 0.2
    assert summary["average_cashout_loss"] == -0.1
    assert strategy["closed_trades"] == 2
    assert strategy["hold_to_resolution_pnl"] == 0.8
    assert strategy["realized_cashout_pnl"] == 0.1
    assert strategy["combined_realized_pnl"] == 0.9
    assert strategy["max_drawdown"] == 0.1


def test_analytics_groups_performance_by_market_regime(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _add_strategy_id_column(db)
    _insert_trade(db, market_id="regime_settled_win", stake_usd=2.0, pnl_usd=0.8, roi=0.4)
    _insert_trade(db, market_id="regime_closed_loss", status="closed", pnl_usd=None, roi=None)
    _mark_closed_trade(db, market_id="regime_closed_loss", pnl=-0.2, roi=-0.2)
    _set_regime_fields(
        db,
        market_id="regime_settled_win",
        btc_trend_regime="strong_up",
        volatility_regime="high",
        spread_regime="tight",
        liquidity_regime="deep",
        time_regime="30-60s",
    )
    _set_regime_fields(
        db,
        market_id="regime_closed_loss",
        btc_trend_regime="strong_up",
        volatility_regime="high",
        spread_regime="tight",
        liquidity_regime="deep",
        time_regime="30-60s",
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    regimes = report["summary"]["regime_performance"]
    strong_up = regimes["btc_trend_regime"]["strong_up"]
    high_vol = regimes["volatility_regime"]["high"]
    tight_spread = regimes["spread_regime"]["tight"]
    deep_liquidity = regimes["liquidity_regime"]["deep"]
    time_bucket = regimes["time_regime"]["30-60s"]

    assert strong_up["trades"] == 2
    assert strong_up["settled_trades"] == 1
    assert strong_up["closed_trades"] == 1
    assert strong_up["pnl"] == 0.6
    assert strong_up["roi"] == 0.1
    assert strong_up["win_rate"] == 0.5
    assert strong_up["average_stake"] == 1.5
    assert strong_up["max_drawdown"] == 0.2
    assert high_vol["pnl"] == 0.6
    assert tight_spread["trades"] == 2
    assert deep_liquidity["win_rate"] == 0.5
    assert time_bucket["closed_trades"] == 1
    assert regimes["btc_trend_regime"]["unknown"]["trades"] == 0


def test_analytics_reports_bankroll_metrics(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="win", stake_usd=2.0, pnl_usd=1.0, roi=0.5)
    _insert_trade(db, market_id="loss", stake_usd=3.0, pnl_usd=-0.5, roi=-0.1666666667)
    _set_bankroll_fields(db, starting=100.0)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["starting_bankroll_usd"] == 100.0
    assert summary["ending_bankroll_usd"] == 100.5
    assert summary["bankroll_roi"] == 0.005
    assert summary["survived"] is True
    assert summary["blown_up"] is False
    assert summary["average_stake_usd"] == 2.5
    assert summary["largest_stake_usd"] == 3.0
    assert summary["profit_per_dollar_staked"] == 0.1


def test_analytics_reports_liquidity_fill_metrics(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="liq_open", status="open", pnl_usd=None, roi=None)
    _set_liquidity_fields(
        db,
        market_id="liq_open",
        requested_shares=2.0,
        max_fillable_shares=5.0,
        fill_fraction=1.0,
        passed=1,
        skip_reason=None,
    )
    _insert_trade(
        db,
        market_id="liq_missing",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="liquidity_missing",
    )
    _set_liquidity_fields(
        db,
        market_id="liq_missing",
        requested_shares=3.0,
        max_fillable_shares=None,
        fill_fraction=1.0,
        passed=0,
        skip_reason="liquidity_missing",
    )
    _insert_trade(
        db,
        market_id="liq_low",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="liquidity_too_low",
    )
    _set_liquidity_fields(
        db,
        market_id="liq_low",
        requested_shares=4.0,
        max_fillable_shares=1.0,
        fill_fraction=0.5,
        passed=0,
        skip_reason="liquidity_too_low",
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["liquidity_checked_trades"] == 3
    assert summary["liquidity_blocked_trades"] == 2
    assert summary["liquidity_missing_skips"] == 1
    assert summary["liquidity_too_low_skips"] == 1
    assert summary["liquidity_columns_populated_pct"] == 100.0
    assert summary["liquidity_check_enabled_detected"] is True
    assert summary["average_requested_shares"] == 3.0
    assert summary["average_max_fillable_shares"] == 3.0


def test_analytics_detects_liquidity_enabled_from_multi_strategy_heartbeat(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="old_null_liq", status="open", pnl_usd=None, roi=None)
    _insert_multi_strategy_heartbeat(db, liquidity_enabled=True)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["liquidity_checked_trades"] == 0
    assert summary["liquidity_columns_populated_pct"] == 0.0
    assert summary["liquidity_check_enabled_detected"] is True


def test_analytics_reports_realistic_execution_metrics(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, market_id="realistic_open", status="open", pnl_usd=None, roi=None)
    _set_realistic_execution_fields(
        db,
        market_id="realistic_open",
        drift=1.0,
        spread=2.0,
    )
    _insert_trade(
        db,
        market_id="realistic_skip",
        status="skipped",
        pnl_usd=None,
        roi=None,
        skip_reason="spread_too_wide",
    )
    _set_realistic_execution_fields(
        db,
        market_id="realistic_skip",
        skip_reason="spread_too_wide",
        drift=2.0,
        spread=5.0,
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["realistic_execution_checked_trades"] == 2
    assert summary["realistic_execution_blocked_trades"] == 1
    assert summary["skips_by_realistic_execution_reason"] == {"spread_too_wide": 1}
    assert summary["average_entry_price_drift"] == 1.5
    assert summary["average_spread_cents_at_entry"] == 3.5


def test_analytics_handles_old_db_without_realistic_execution_columns(tmp_path) -> None:
    db = tmp_path / "old_paper.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                run_id TEXT,
                market_id TEXT,
                signal_direction TEXT,
                predicted_probability_yes REAL,
                adjusted_entry_price REAL,
                stake_usd REAL,
                status TEXT,
                pnl_usd REAL,
                roi REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, run_id, market_id, signal_direction,
                predicted_probability_yes, adjusted_entry_price,
                stake_usd, status, pnl_usd, roi
            ) VALUES (?, 'run1', 'old1', 'YES', 0.7, 0.55, 1.0, 'settled', 0.8, 0.8)
            """,
            (_iso(NOW - timedelta(minutes=5)),),
        )
        conn.commit()
    finally:
        conn.close()

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["realistic_execution_checked_trades"] == 0
    assert summary["realistic_execution_blocked_trades"] == 0
    assert summary["skips_by_realistic_execution_reason"] == {}


def test_analytics_handles_old_rows_with_null_edge_fields(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(db, direction="NO", probability_yes=0.20, adjusted_entry_price=0.65)

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )

    assert report["trades"][0]["probability_for_direction"] == 0.8
    assert report["trades"][0]["estimated_edge"] == 0.15
    assert report["summary"]["probability_buckets"]["0.8-0.9"]["trade_count"] == 1


def test_analytics_unresolved_exposure_by_direction_and_edge_bucket(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    _insert_trade(
        db,
        direction="YES",
        status="open",
        probability_yes=0.80,
        adjusted_entry_price=0.55,
        pnl_usd=None,
        roi=None,
        stake_usd=2.0,
    )
    _insert_trade(
        db,
        direction="NO",
        status="awaiting_resolution",
        probability_yes=0.20,
        adjusted_entry_price=0.65,
        pnl_usd=None,
        roi=None,
        stake_usd=3.0,
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    summary = report["summary"]

    assert summary["unresolved_exposure_by_direction"]["YES"]["stake"] == 2.0
    assert summary["unresolved_exposure_by_direction"]["NO"]["stake"] == 3.0
    assert summary["unresolved_exposure_by_estimated_edge_bucket"]["edge >= 0.20"]["stake"] == 2.0
    assert summary["unresolved_exposure_by_estimated_edge_bucket"]["0.10 to 0.20"]["stake"] == 3.0


def test_analytics_probability_and_edge_underperformance_warnings(tmp_path) -> None:
    db = tmp_path / "paper.db"
    _create_db(db)
    for idx in range(4):
        _insert_trade(
            db,
            market_id=f"high_win_{idx}",
            probability_yes=0.95,
            adjusted_entry_price=0.55,
            pnl_usd=0.8,
            roi=0.8,
        )
    for idx in range(6):
        _insert_trade(
            db,
            market_id=f"high_loss_{idx}",
            probability_yes=0.95,
            adjusted_entry_price=0.55,
            pnl_usd=-1.0,
            roi=-1.0,
        )
    _insert_trade(
        db,
        market_id="lower_win",
        probability_yes=0.85,
        adjusted_entry_price=0.70,
        pnl_usd=0.4,
        roi=0.4,
    )

    report = build_paper_trader_analytics_report(
        paper_db_path=str(db),
        output_dir=str(tmp_path / "report"),
        now=NOW,
    )
    warnings = report["summary"]["warnings"]

    assert report["summary"]["probability_buckets"]["0.9-1.0"]["win_rate"] == 0.4
    assert report["summary"]["edge_buckets"]["edge >= 0.20"]["settled_trades"] == 10
    assert "high_probability_bucket_underperforms_lower_probability_bucket" in warnings
    assert "edge_ge_0_20_win_rate_below_55_percent" in warnings
    assert "probability block guardrails" in report["summary"]["recommendation"]


def test_cli_writes_all_artifacts_without_loading_recorder(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "paper.db"
    output = tmp_path / "report"
    _create_db(db)
    _insert_trade(db)

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("paper-trader-analytics must not load settings or recorder code")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "paper-trader-analytics",
            "--paper-db",
            str(db),
            "--output-dir",
            str(output),
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_trades"] == 1
    assert (output / "paper_trader_report.json").exists()
    assert (output / "paper_trader_report.txt").exists()
    assert (output / "trades_enriched.csv").exists()
