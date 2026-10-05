from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .backtest_baseline_strategy import _load_feature_columns, _load_model, _predict_probabilities
from .baseline_paper_trader import (
    BaselinePaperTraderConfig,
    _candidate_skip_reason,
    _comparison_btc_stale,
    _has_existing_trade,
    _live_entry_price,
    _paper_status_count,
    _probability_for_direction,
    _valid_price,
    connect_recorder_read_only,
)
from .btc_price_feed import POLYMARKET_RTDS_BINANCE_SOURCE, POLYMARKET_RTDS_CHAINLINK_SOURCE
from .models import parse_timestamp, to_iso
from .paper_trader_analytics import connect_paper_db_read_only
from .train_baseline_model import _float_or_none


REPORT_GATES: tuple[str, ...] = (
    "not_feature_ready",
    "strict_validation_failed",
    "gap_affected",
    "snapshot_quality_not_ok",
    "missing_orderbook",
    "missing_trade_data",
    "stale_canonical_btc",
    "stale_comparison_btc",
    "missing_btc_price_at_market_start",
    "missing_required_model_feature_column",
    "expired_feature",
    "time_window",
    "threshold",
    "min_estimated_edge",
    "probability_block",
    "max_open_trades",
    "one_trade_per_market",
    "missing_price",
)


def build_paper_trader_eligibility_report(
    *,
    recorder_db_path: str,
    paper_db_path: str,
    feature_columns_path: str,
    model_path: str,
    recent_minutes: float = 30.0,
    max_btc_age_sec: float = 3.0,
    max_feature_age_sec: float = 5.0,
    min_time_until_resolution_sec: float = 0.0,
    max_time_until_resolution_sec: float = 300.0,
    long_threshold: float = 0.65,
    short_threshold: float = 0.35,
    min_estimated_edge: float = 0.0,
    max_open_trades: int = 1,
    block_probability_above: float | None = None,
    block_probability_below: float | None = None,
    require_fresh_comparison_btc: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    feature_columns = _load_feature_columns(feature_columns_path)
    model = _load_model(model_path)
    config = BaselinePaperTraderConfig(
        recorder_db_path=recorder_db_path,
        output_db_path=paper_db_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        max_btc_age_sec=max_btc_age_sec,
        max_feature_age_sec=max_feature_age_sec,
        min_time_until_resolution_sec=min_time_until_resolution_sec,
        max_time_until_resolution_sec=max_time_until_resolution_sec,
        min_estimated_edge=min_estimated_edge,
        max_open_trades=max_open_trades,
        block_probability_above=block_probability_above,
        block_probability_below=block_probability_below,
        require_fresh_comparison_btc=require_fresh_comparison_btc,
    )

    recorder = connect_recorder_read_only(recorder_db_path)
    paper = _connect_optional_paper(paper_db_path)
    try:
        rows = _fetch_recent_feature_rows(
            recorder,
            since=now_dt - timedelta(minutes=float(recent_minutes)),
            now=now_dt,
        )
        paper_counts = _paper_counts(paper)
        latest_seen = _latest_seen_feature_timestamp(recorder)
        predictions = _safe_probabilities(model, rows, feature_columns)
        gate_counts: Counter[str] = Counter()
        top_rejections: Counter[str] = Counter()
        eligible_rows: list[dict[str, Any]] = []
        latest_rejection: dict[str, Any] | None = None

        open_trade_count = paper_counts["open_trades"]
        for idx, row in enumerate(rows):
            failures = _gate_failures(
                row,
                config=config,
                feature_columns=feature_columns,
                probability=predictions.get(idx),
                open_trade_count=open_trade_count,
                paper=paper,
                now=now_dt,
            )
            if failures:
                for reason in failures:
                    gate_counts[reason] += 1
                primary = failures[0]
                top_rejections[primary] += 1
                latest_rejection = _latest_rejection(
                    latest_rejection,
                    {
                        "reason": primary,
                        "market_id": row.get("market_id"),
                        "feature_timestamp": row.get("timestamp"),
                    },
                )
            else:
                eligible_rows.append(row)

        active_markets = {
            str(row.get("market_id"))
            for row in rows
            if _is_live_active(row, now=now_dt)
        }
        report = {
            "status": "ok",
            "recorder_db_path": recorder_db_path,
            "paper_db_path": paper_db_path,
            "feature_columns_path": feature_columns_path,
            "model_path": model_path,
            "recent_minutes": float(recent_minutes),
            "rows_considered": len(rows),
            "markets_considered": len({str(row.get("market_id")) for row in rows}),
            "live_active_markets_considered": len(active_markets),
            "latest_seen_feature_timestamp": latest_seen,
            "latest_eligible_feature_timestamp": (
                max((str(row.get("timestamp")) for row in eligible_rows), default=None)
            ),
            "latest_rejection_reason": (latest_rejection or {}).get("reason"),
            "latest_rejection_market_id": (latest_rejection or {}).get("market_id"),
            "latest_rejection_feature_timestamp": (latest_rejection or {}).get("feature_timestamp"),
            "gate_failures": {gate: int(gate_counts.get(gate, 0)) for gate in REPORT_GATES},
            "top_10_rejection_reasons": top_rejections.most_common(10),
            **paper_counts,
        }
        return report
    finally:
        recorder.close()
        if paper is not None:
            paper.close()


def render_paper_trader_eligibility_report(report: dict[str, Any]) -> str:
    lines = [
        "Paper Trader Eligibility Report",
        f"recorder_db={report.get('recorder_db_path')}",
        f"paper_db={report.get('paper_db_path')}",
        f"recent_minutes={report.get('recent_minutes')}",
        f"rows_considered={report.get('rows_considered')}",
        f"markets_considered={report.get('markets_considered')}",
        f"live_active_markets_considered={report.get('live_active_markets_considered')}",
        f"latest_seen_feature_timestamp={report.get('latest_seen_feature_timestamp')}",
        f"latest_eligible_feature_timestamp={report.get('latest_eligible_feature_timestamp')}",
        f"latest_rejection_reason={report.get('latest_rejection_reason')}",
        f"latest_rejection_market_id={report.get('latest_rejection_market_id')}",
        f"latest_rejection_feature_timestamp={report.get('latest_rejection_feature_timestamp')}",
        f"open_trades={report.get('open_trades')}",
        f"awaiting_resolution_trades={report.get('awaiting_resolution_trades')}",
        f"settled_trades={report.get('settled_trades')}",
        "",
        "Gate failures:",
    ]
    for gate, count in (report.get("gate_failures") or {}).items():
        lines.append(f"- {gate}: {count}")
    lines.append("")
    lines.append("Top rejection reasons:")
    for reason, count in report.get("top_10_rejection_reasons") or []:
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def _fetch_recent_feature_rows(
    conn: sqlite3.Connection,
    *,
    since: datetime,
    now: datetime,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          f.*,
          m.question,
          m.start_time,
          m.close_time,
          m.phase AS market_phase,
          m.tracking_state,
          s.best_bid_yes,
          s.best_ask_yes,
          s.best_bid_no,
          s.best_ask_no,
          s.has_orderbook,
          s.has_trade_data,
          s.strict_validation_passed AS snapshot_strict_validation_passed,
          CASE
            WHEN m.close_time IS NULL THEN f.time_until_resolution
            ELSE (julianday(m.close_time) - julianday(f.timestamp)) * 86400.0
          END AS seconds_before_close,
          btc_c.price AS btc_chainlink_price,
          btc_c.local_arrival_iso AS btc_chainlink_local_arrival_iso,
          CASE
            WHEN btc_c.local_arrival_iso IS NULL THEN NULL
            ELSE MAX(0.0, (julianday(f.timestamp) - julianday(btc_c.local_arrival_iso)) * 86400.0)
          END AS btc_chainlink_age_sec_at_feature,
          btc_cmp.price AS btc_binance_price,
          btc_cmp.local_arrival_iso AS btc_binance_local_arrival_iso,
          CASE
            WHEN btc_cmp.local_arrival_iso IS NULL THEN NULL
            ELSE MAX(0.0, (julianday(f.timestamp) - julianday(btc_cmp.local_arrival_iso)) * 86400.0)
          END AS btc_binance_age_sec_at_feature,
          COALESCE(btc_start_prior.price, btc_start_after.price)
            AS btc_price_at_market_start,
          (julianday(?) - julianday(f.timestamp)) * 86400.0 AS feature_age_sec
        FROM features f
        LEFT JOIN market_snapshots s
          ON s.run_id = f.run_id
         AND s.market_id = f.market_id
         AND s.timestamp = f.timestamp
        LEFT JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        LEFT JOIN btc_prices btc_c
          ON btc_c.id = (
            SELECT b.id FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND datetime(b.local_arrival_iso) <= datetime(f.timestamp)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_cmp
          ON btc_cmp.id = (
            SELECT b.id FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND datetime(b.local_arrival_iso) <= datetime(f.timestamp)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_start_prior
          ON btc_start_prior.id = (
            SELECT b.id FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND m.start_time IS NOT NULL
              AND datetime(b.local_arrival_iso) <= datetime(m.start_time)
            ORDER BY b.local_arrival_ns DESC, b.id DESC
            LIMIT 1
          )
        LEFT JOIN btc_prices btc_start_after
          ON btc_start_after.id = (
            SELECT b.id FROM btc_prices b
            WHERE b.run_id = f.run_id
              AND b.source = ?
              AND m.start_time IS NOT NULL
              AND datetime(b.local_arrival_iso) > datetime(m.start_time)
            ORDER BY b.local_arrival_ns ASC, b.id ASC
            LIMIT 1
          )
        WHERE datetime(f.timestamp) >= datetime(?)
        ORDER BY datetime(f.timestamp) DESC, f.timestamp DESC
        LIMIT ?
        """,
        (
            to_iso(now),
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_BINANCE_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            POLYMARKET_RTDS_CHAINLINK_SOURCE,
            to_iso(since),
            int(limit),
        ),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("time_until_resolution") is None and item.get("seconds_before_close") is not None:
            item["time_until_resolution"] = item.get("seconds_before_close")
        out.append(item)
    return out


def _gate_failures(
    row: dict[str, Any],
    *,
    config: BaselinePaperTraderConfig,
    feature_columns: Sequence[str],
    probability: float | None,
    open_trade_count: int,
    paper: sqlite3.Connection | None,
    now: datetime,
) -> list[str]:
    failures: list[str] = []
    base_reason = _candidate_skip_reason(
        row,
        config=config,
        now=now,
    )
    if base_reason is not None:
        failures.append(base_reason)
    if _comparison_btc_stale(row, max_btc_age_sec=config.max_btc_age_sec):
        failures.append("stale_comparison_btc")
    if "btc_price_at_market_start" in set(feature_columns) and _float_or_none(row.get("btc_price_at_market_start")) is None:
        failures.append("missing_btc_price_at_market_start")
    missing_columns = [column for column in feature_columns if column not in row]
    if missing_columns:
        failures.append("missing_required_model_feature_column")
    if probability is not None:
        direction = None
        if probability >= config.long_threshold:
            direction = "YES"
        elif probability <= config.short_threshold:
            direction = "NO"
        else:
            failures.append("threshold")
        if direction is not None:
            entry = _live_entry_price(row, direction=direction)
            if not _valid_price(entry):
                failures.append("missing_price")
            else:
                adjusted = min(
                    1.0,
                    float(entry)
                    + float(config.entry_slippage_cents) / 100.0
                    + float(config.fee_cents) / 100.0,
                )
                pdir = _probability_for_direction(probability, direction=direction)
                if config.block_probability_above is not None and pdir >= float(config.block_probability_above):
                    failures.append("probability_block")
                if config.block_probability_below is not None and pdir <= float(config.block_probability_below):
                    failures.append("probability_block")
                if pdir - adjusted < float(config.min_estimated_edge):
                    failures.append("min_estimated_edge")
                if open_trade_count >= int(config.max_open_trades):
                    failures.append("max_open_trades")
                if paper is not None and config.one_trade_per_market and _has_existing_trade(
                    paper,
                    run_id=str(row.get("run_id") or ""),
                    market_id=str(row.get("market_id") or ""),
                ):
                    failures.append("one_trade_per_market")
    else:
        failures.append("no_signal")
    return list(dict.fromkeys(failures))


def _safe_probabilities(
    model: Any,
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[int, float]:
    predictions: dict[int, float] = {}
    usable: list[dict[str, Any]] = []
    indexes: list[int] = []
    for idx, row in enumerate(rows):
        if all(column in row for column in feature_columns):
            usable.append(row)
            indexes.append(idx)
    if not usable:
        return predictions
    try:
        probabilities = _predict_probabilities(model, usable, feature_columns)
    except Exception:
        return predictions
    for idx, probability in zip(indexes, probabilities):
        predictions[idx] = float(probability)
    return predictions


def _paper_counts(conn: sqlite3.Connection | None) -> dict[str, int]:
    if conn is None:
        return {
            "open_trades": 0,
            "awaiting_resolution_trades": 0,
            "settled_trades": 0,
        }
    return {
        "open_trades": _paper_status_count(conn, "open"),
        "awaiting_resolution_trades": _paper_status_count(conn, "awaiting_resolution"),
        "settled_trades": _paper_status_count(conn, "settled"),
    }


def _connect_optional_paper(path: str) -> sqlite3.Connection | None:
    try:
        return connect_paper_db_read_only(path)
    except FileNotFoundError:
        return None


def _latest_seen_feature_timestamp(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT timestamp FROM features ORDER BY datetime(timestamp) DESC, timestamp DESC LIMIT 1"
    ).fetchone()
    return str(row["timestamp"]) if row is not None else None


def _latest_rejection(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    current_ts = parse_timestamp(current.get("feature_timestamp"))
    candidate_ts = parse_timestamp(candidate.get("feature_timestamp"))
    if current_ts is None:
        return candidate
    if candidate_ts is None:
        return current
    return candidate if candidate_ts >= current_ts else current


def _is_live_active(row: dict[str, Any], *, now: datetime) -> bool:
    close = parse_timestamp(row.get("close_time"))
    start = parse_timestamp(row.get("start_time"))
    if close is None:
        return False
    if start is not None and start > now:
        return False
    return close > now


__all__ = [
    "build_paper_trader_eligibility_report",
    "render_paper_trader_eligibility_report",
]
