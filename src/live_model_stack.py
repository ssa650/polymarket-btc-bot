from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote

from .backtest_baseline_strategy import (
    _load_feature_columns,
    _load_model,
    _predict_probabilities,
)
from .btc_price_feed import POLYMARKET_RTDS_CHAINLINK_SOURCE
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none
from .transformer_live_inference import (
    InferenceFn,
    TransformerLiveArtifacts,
    TransformerLiveInferenceError,
    TransformerLiveModel,
    _effective_sequence_length,
    _open_readonly_sqlite,
    load_transformer_live_artifacts,
    load_transformer_live_model,
    transformer_live_inference_heartbeat,
)


LIVE_MODEL_PREDICTIONS_TABLE = "live_model_predictions"
LIVE_MODEL_OUTCOMES_TABLE = "live_model_outcomes"
LIVE_MODEL_ACCURACY_TABLE = "live_model_accuracy"
LIVE_MODEL_HEARTBEATS_TABLE = "live_model_heartbeats"


class LiveModelStackError(RuntimeError):
    pass


@dataclass(slots=True)
class LoadedModel:
    model_name: str
    model_type: str
    model_path: str
    feature_columns_path: str
    feature_columns: list[str]
    model: Any


@dataclass(slots=True)
class LoadedTransformer:
    artifacts: TransformerLiveArtifacts
    sequence_length: int
    model_bundle: TransformerLiveModel | None = None
    inference_fn: InferenceFn | None = None


def connect_prediction_db(path: str) -> sqlite3.Connection:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(output))
    conn.row_factory = sqlite3.Row
    ensure_live_model_schema(conn)
    return conn


def ensure_live_model_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LIVE_MODEL_PREDICTIONS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            run_id TEXT,
            market_id TEXT NOT NULL,
            question TEXT,
            market_start_time TEXT,
            market_end_time TEXT,
            feature_timestamp TEXT,
            prediction_timestamp TEXT NOT NULL,
            probability_yes REAL,
            probability_no REAL,
            predicted_side TEXT,
            confidence REAL,
            entry_price_yes REAL,
            entry_price_no REAL,
            best_bid_yes REAL,
            best_ask_yes REAL,
            best_bid_no REAL,
            best_ask_no REAL,
            btc_price REAL,
            btc_source TEXT,
            btc_age_sec REAL,
            feature_ready INTEGER,
            snapshot_quality_status TEXT,
            stale_flags_json TEXT,
            raw_prediction_json TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LIVE_MODEL_OUTCOMES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resolved_at TEXT NOT NULL,
            market_id TEXT NOT NULL,
            winning_side TEXT,
            resolution_source TEXT,
            market_end_time TEXT,
            final_btc_price_if_available REAL,
            raw_resolution_json TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LIVE_MODEL_ACCURACY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            market_id TEXT NOT NULL,
            prediction_id INTEGER NOT NULL,
            predicted_side TEXT,
            winning_side TEXT,
            was_correct INTEGER,
            probability_yes REAL,
            confidence REAL,
            price_at_signal REAL,
            hypothetical_pnl_yes_no REAL,
            latency_to_resolution_sec REAL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LIVE_MODEL_HEARTBEATS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            run_id TEXT,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            latest_market_id TEXT,
            latest_feature_timestamp TEXT,
            rf_probability_yes REAL,
            transformer_probability_yes REAL,
            model_agreement TEXT,
            stale_flags_json TEXT,
            raw_heartbeat_json TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_live_model_predictions_unique_feature
        ON {LIVE_MODEL_PREDICTIONS_TABLE} (
            model_name, model_type, market_id, feature_timestamp
        )
        """
    )
    for table, column in (
        (LIVE_MODEL_PREDICTIONS_TABLE, "created_at"),
        (LIVE_MODEL_PREDICTIONS_TABLE, "market_id"),
        (LIVE_MODEL_PREDICTIONS_TABLE, "model_name"),
        (LIVE_MODEL_PREDICTIONS_TABLE, "feature_timestamp"),
        (LIVE_MODEL_OUTCOMES_TABLE, "market_id"),
        (LIVE_MODEL_ACCURACY_TABLE, "model_name"),
        (LIVE_MODEL_ACCURACY_TABLE, "market_id"),
        (LIVE_MODEL_HEARTBEATS_TABLE, "created_at"),
    ):
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table} ({column})"
        )
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_live_model_outcomes_market
        ON {LIVE_MODEL_OUTCOMES_TABLE} (market_id)
        """
    )
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_live_model_accuracy_prediction
        ON {LIVE_MODEL_ACCURACY_TABLE} (prediction_id)
        """
    )
    conn.commit()


def load_rf_live_model(
    *,
    model_path: str,
    feature_columns_path: str,
    model_name: str = "random_forest",
) -> LoadedModel:
    return LoadedModel(
        model_name=model_name,
        model_type="random_forest",
        model_path=str(model_path),
        feature_columns_path=str(feature_columns_path),
        feature_columns=list(_load_feature_columns(feature_columns_path)),
        model=_load_model(model_path),
    )


def load_transformer_live_stack(
    *,
    transformer_model_dir: str,
    requested_sequence_length: int = 120,
    inference_fn: InferenceFn | None = None,
    load_model: bool = True,
) -> LoadedTransformer:
    model_dir = Path(transformer_model_dir)
    artifacts = load_transformer_live_artifacts(
        model_path=str(model_dir / "model.pt"),
        feature_columns_path=str(model_dir / "feature_columns.json"),
        scaler_stats_path=str(model_dir / "scaler_stats.json"),
        training_config_path=str(model_dir / "training_config.json"),
    )
    sequence_length = _effective_sequence_length(
        requested_sequence_length=requested_sequence_length,
        training_config=artifacts.training_config,
    )
    model_bundle = None
    if load_model and inference_fn is None:
        model_bundle = load_transformer_live_model(artifacts)
    return LoadedTransformer(
        artifacts=artifacts,
        sequence_length=sequence_length,
        model_bundle=model_bundle,
        inference_fn=inference_fn,
    )


def run_live_model_predictions(
    *,
    recorder_db_path: str,
    output_db_path: str,
    rf_model_path: str | None,
    rf_feature_columns_path: str | None,
    transformer_model_dir: str | None,
    poll_sec: float = 1.0,
    max_feature_age_sec: float = 5.0,
    max_btc_age_sec: float = 15.0,
    min_confidence: float = 0.55,
    run_id: str | None = None,
    shadow_only: bool = True,
    enable_live_trading: bool = False,
    sequence_length: int = 120,
    max_iterations: int | None = None,
    emit_logs: bool = True,
    now: datetime | None = None,
    transformer_inference_fn: InferenceFn | None = None,
) -> dict[str, Any]:
    if enable_live_trading:
        print(
            json.dumps(
                {
                    "event": "live_trading_requested_but_no_order_placement_is_enabled",
                    "status": "blocked_by_default",
                },
                sort_keys=True,
            )
        )
    recorder_conn = _open_recorder_readonly(recorder_db_path)
    output_conn = connect_prediction_db(output_db_path)
    rf_model = (
        load_rf_live_model(
            model_path=str(rf_model_path),
            feature_columns_path=str(rf_feature_columns_path),
        )
        if rf_model_path and rf_feature_columns_path
        else None
    )
    transformer = None
    if transformer_model_dir:
        transformer = load_transformer_live_stack(
            transformer_model_dir=transformer_model_dir,
            requested_sequence_length=sequence_length,
            inference_fn=transformer_inference_fn,
            load_model=transformer_inference_fn is None,
        )
    resolved_run_id = run_id or _generated_run_id("live_models")
    mode = _mode(shadow_only=shadow_only, enable_live_trading=enable_live_trading)
    iterations = 0
    last_report: dict[str, Any] = {}
    try:
        while True:
            last_report = run_live_model_predictions_once(
                recorder_conn=recorder_conn,
                output_conn=output_conn,
                rf_model=rf_model,
                transformer=transformer,
                poll_sec=poll_sec,
                max_feature_age_sec=max_feature_age_sec,
                max_btc_age_sec=max_btc_age_sec,
                min_confidence=min_confidence,
                run_id=resolved_run_id,
                mode=mode,
                enable_live_trading=enable_live_trading,
                now=now,
            )
            if emit_logs:
                print(json.dumps({"event": "live_model_prediction_heartbeat", **last_report}, sort_keys=True))
            iterations += 1
            if max_iterations is not None and iterations >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iterations,
                    "run_id": resolved_run_id,
                    "mode": mode,
                    "output_db_path": output_db_path,
                    "last_heartbeat": last_report,
                }
            time.sleep(max(0.0, float(poll_sec)))
    finally:
        recorder_conn.close()
        output_conn.close()


def run_live_model_predictions_once(
    *,
    recorder_conn: sqlite3.Connection,
    output_conn: sqlite3.Connection,
    rf_model: LoadedModel | None,
    transformer: LoadedTransformer | None,
    poll_sec: float,
    max_feature_age_sec: float,
    max_btc_age_sec: float,
    min_confidence: float,
    run_id: str,
    mode: str,
    enable_live_trading: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    context = fetch_latest_live_context(
        recorder_conn,
        now=now_dt,
        max_feature_age_sec=max_feature_age_sec,
        max_btc_age_sec=max_btc_age_sec,
    )
    stale_flags = dict(context.get("stale_flags") or {})
    inserted: list[dict[str, Any]] = []
    model_errors: list[dict[str, Any]] = []
    rf_probability: float | None = None
    transformer_probability: float | None = None
    if context.get("healthy"):
        if rf_model is not None:
            try:
                rf_probability = _predict_live_probability(rf_model, context["feature_row"])
                if rf_probability is not None:
                    inserted_row = insert_live_model_prediction(
                        output_conn,
                        model_name=rf_model.model_name,
                        model_type=rf_model.model_type,
                        model_path=rf_model.model_path,
                        run_id=run_id,
                        context=context,
                        probability_yes=rf_probability,
                        min_confidence=min_confidence,
                        raw_prediction={
                            "feature_columns_path": rf_model.feature_columns_path,
                            "feature_columns_count": len(rf_model.feature_columns),
                            "mode": mode,
                        },
                        now=now_dt,
                    )
                    inserted.append(inserted_row)
            except Exception as exc:  # pragma: no cover - kept defensive for live loop
                model_errors.append({"model": "random_forest", "error": str(exc)})
        if transformer is not None:
            try:
                heartbeat = transformer_live_inference_heartbeat(
                    recorder_conn,
                    artifacts=transformer.artifacts,
                    model_bundle=transformer.model_bundle,
                    inference_fn=transformer.inference_fn,
                    sequence_length=transformer.sequence_length,
                    max_feature_age_sec=max_feature_age_sec,
                    allow_gap_affected=False,
                    sequence_row_policy="latest_clean_rows",
                    now=now_dt,
                )
                transformer_probability = _float_or_none(heartbeat.get("probability_yes"))
                if transformer_probability is not None:
                    inserted_row = insert_live_model_prediction(
                        output_conn,
                        model_name="transformer",
                        model_type="transformer",
                        model_path=transformer.artifacts.model_path,
                        run_id=run_id,
                        context=_context_from_transformer_heartbeat(context, heartbeat),
                        probability_yes=transformer_probability,
                        min_confidence=min_confidence,
                        raw_prediction={
                            "feature_columns_path": transformer.artifacts.feature_columns_path,
                            "scaler_stats_path": transformer.artifacts.scaler_stats_path,
                            "training_config_path": transformer.artifacts.training_config_path,
                            "sequence_length": transformer.sequence_length,
                            "sequence_ready": heartbeat.get("transformer_sequence_ready"),
                            "sequence_reason": heartbeat.get("reason"),
                            "mode": mode,
                        },
                        now=now_dt,
                    )
                    inserted.append(inserted_row)
                else:
                    model_errors.append(
                        {
                            "model": "transformer",
                            "reason": heartbeat.get("reason") or "sequence_unavailable",
                            "sequence_rows_available": heartbeat.get("sequence_rows_available"),
                        }
                    )
            except TransformerLiveInferenceError as exc:
                model_errors.append({"model": "transformer", "error": str(exc)})
    agreement = _model_agreement(rf_probability, transformer_probability, min_confidence)
    status = "healthy" if context.get("healthy") and inserted else "blocked"
    if not context.get("healthy"):
        status = "stale_or_not_ready"
    elif model_errors and not inserted:
        status = "model_unavailable"
    heartbeat = {
        "timestamp": to_iso(now_dt),
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "latest_market_id": context.get("market_id"),
        "latest_feature_timestamp": context.get("feature_timestamp"),
        "rf_probability_yes": rf_probability,
        "transformer_probability_yes": transformer_probability,
        "model_agreement": agreement,
        "predictions_inserted": sum(1 for row in inserted if row.get("inserted")),
        "predictions_deduped": sum(1 for row in inserted if not row.get("inserted")),
        "stale_flags": stale_flags,
        "model_errors": model_errors,
        "live_order_placement_blocked": bool(
            enable_live_trading and (stale_flags or status != "healthy")
        ),
        "poll_sec": float(poll_sec),
    }
    insert_live_model_heartbeat(output_conn, heartbeat)
    return heartbeat


def fetch_latest_live_context(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    max_feature_age_sec: float,
    max_btc_age_sec: float,
) -> dict[str, Any]:
    feature = _latest_active_feature(conn, now=now)
    btc_by_source = latest_btc_by_source(conn, now=now)
    canonical_btc = _canonical_btc_sample(btc_by_source)
    stale_flags: dict[str, Any] = {}
    if feature is None:
        stale_flags["no_active_feature"] = True
        return {
            "healthy": False,
            "stale_flags": stale_flags,
            "btc_by_source": btc_by_source,
            "feature_row": {},
        }
    feature_row = dict(feature)
    market_id = str(feature_row.get("market_id") or "")
    run_id = str(feature_row.get("run_id") or "")
    feature_timestamp = str(feature_row.get("timestamp") or "")
    feature_ts = parse_timestamp(feature_timestamp)
    feature_age = None
    if feature_ts is None:
        stale_flags["invalid_feature_timestamp"] = feature_timestamp
    else:
        feature_age = (now - feature_ts).total_seconds()
        if max_feature_age_sec >= 0 and feature_age > float(max_feature_age_sec):
            stale_flags["feature_stale"] = round(feature_age, 3)
    if int(_float_or_none(feature_row.get("feature_ready")) or 0) != 1:
        stale_flags["feature_not_ready"] = feature_row.get("feature_ready")
    if str(feature_row.get("snapshot_quality_status") or "").lower() not in {"ok", ""}:
        stale_flags["snapshot_quality_status"] = feature_row.get("snapshot_quality_status")
    if canonical_btc is None:
        stale_flags["btc_missing"] = True
    else:
        btc_age = _float_or_none(canonical_btc.get("age_sec"))
        if btc_age is None:
            stale_flags["btc_age_missing"] = True
        elif max_btc_age_sec >= 0 and btc_age > float(max_btc_age_sec):
            stale_flags["btc_stale"] = round(btc_age, 3)
    snapshot = _snapshot_for_feature(
        conn,
        run_id=run_id,
        market_id=market_id,
        feature_timestamp=feature_timestamp,
    )
    context = {
        "healthy": not stale_flags,
        "stale_flags": stale_flags,
        "feature_row": feature_row,
        "snapshot_row": snapshot,
        "btc_by_source": btc_by_source,
        "canonical_btc": canonical_btc,
        "run_id": run_id,
        "market_id": market_id,
        "question": feature_row.get("question"),
        "market_start_time": feature_row.get("market_start_time"),
        "market_end_time": feature_row.get("market_end_time"),
        "feature_timestamp": feature_timestamp,
        "feature_age_sec": feature_age,
        "feature_ready": feature_row.get("feature_ready"),
        "snapshot_quality_status": feature_row.get("snapshot_quality_status"),
        "time_until_resolution": _float_or_none(feature_row.get("time_until_resolution")),
    }
    context.update(_price_context(feature_row, snapshot))
    if canonical_btc:
        context.update(
            {
                "btc_price": _float_or_none(canonical_btc.get("price")),
                "btc_source": canonical_btc.get("source"),
                "btc_age_sec": _float_or_none(canonical_btc.get("age_sec")),
            }
        )
    return context


def latest_btc_by_source(conn: sqlite3.Connection, *, now: datetime) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "btc_prices"):
        return {}
    columns = _table_columns(conn, "btc_prices")
    if "source" not in columns:
        return {}
    time_columns = [
        column
        for column in ("exchange_timestamp", "local_arrival_iso", "inserted_at")
        if column in columns
    ]
    if not time_columns:
        time_columns = ["id"] if "id" in columns else ["rowid"]
    time_expr = (
        time_columns[0]
        if len(time_columns) == 1
        else f"COALESCE({', '.join(time_columns)})"
    )
    rows = conn.execute(
        f"""
        SELECT b.*
        FROM btc_prices b
        JOIN (
            SELECT source, MAX({time_expr}) AS max_ts
            FROM btc_prices
            GROUP BY source
        ) latest
          ON latest.source = b.source
         AND {time_expr.replace('COALESCE(', 'COALESCE(b.').replace(', ', ', b.').replace(')', ')') if time_expr.startswith('COALESCE(') else 'b.' + time_expr} = latest.max_ts
        """
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        source = str(payload.get("source") or "unknown")
        age_ts = (
            parse_timestamp(str(payload.get("exchange_timestamp") or ""))
            or parse_timestamp(str(payload.get("local_arrival_iso") or ""))
            or parse_timestamp(str(payload.get("inserted_at") or ""))
        )
        payload["age_sec"] = (now - age_ts).total_seconds() if age_ts else None
        result[source] = payload
    return result


def insert_live_model_prediction(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    model_type: str,
    model_path: str,
    run_id: str,
    context: dict[str, Any],
    probability_yes: float,
    min_confidence: float,
    raw_prediction: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    probability_yes = max(0.0, min(1.0, float(probability_yes)))
    probability_no = 1.0 - probability_yes
    predicted_side = _predicted_side(probability_yes, min_confidence)
    confidence = max(probability_yes, probability_no)
    values = {
        "created_at": to_iso(now),
        "model_name": model_name,
        "model_type": model_type,
        "run_id": run_id,
        "market_id": str(context.get("market_id") or ""),
        "question": context.get("question"),
        "market_start_time": context.get("market_start_time"),
        "market_end_time": context.get("market_end_time"),
        "feature_timestamp": context.get("feature_timestamp"),
        "prediction_timestamp": to_iso(now),
        "probability_yes": probability_yes,
        "probability_no": probability_no,
        "predicted_side": predicted_side,
        "confidence": confidence,
        "entry_price_yes": context.get("entry_price_yes"),
        "entry_price_no": context.get("entry_price_no"),
        "best_bid_yes": context.get("best_bid_yes"),
        "best_ask_yes": context.get("best_ask_yes"),
        "best_bid_no": context.get("best_bid_no"),
        "best_ask_no": context.get("best_ask_no"),
        "btc_price": context.get("btc_price"),
        "btc_source": context.get("btc_source"),
        "btc_age_sec": context.get("btc_age_sec"),
        "feature_ready": _int_or_none(context.get("feature_ready")),
        "snapshot_quality_status": context.get("snapshot_quality_status"),
        "stale_flags_json": json.dumps(context.get("stale_flags") or {}, sort_keys=True),
        "raw_prediction_json": json.dumps(raw_prediction, sort_keys=True),
    }
    columns = list(values.keys())
    cursor = conn.execute(
        f"""
        INSERT OR IGNORE INTO {LIVE_MODEL_PREDICTIONS_TABLE} (
            {", ".join(columns)}
        ) VALUES ({", ".join("?" for _ in columns)})
        """,
        [values[column] for column in columns],
    )
    conn.commit()
    return {
        "inserted": cursor.rowcount > 0,
        "model_name": model_name,
        "model_type": model_type,
        "market_id": values["market_id"],
        "feature_timestamp": values["feature_timestamp"],
        "probability_yes": probability_yes,
        "predicted_side": predicted_side,
    }


def insert_live_model_heartbeat(conn: sqlite3.Connection, heartbeat: dict[str, Any]) -> None:
    now = str(heartbeat.get("timestamp") or to_iso(datetime.now(timezone.utc)))
    conn.execute(
        f"""
        INSERT INTO {LIVE_MODEL_HEARTBEATS_TABLE} (
            created_at, run_id, mode, status, latest_market_id,
            latest_feature_timestamp, rf_probability_yes,
            transformer_probability_yes, model_agreement,
            stale_flags_json, raw_heartbeat_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            heartbeat.get("run_id"),
            heartbeat.get("mode"),
            heartbeat.get("status"),
            heartbeat.get("latest_market_id"),
            heartbeat.get("latest_feature_timestamp"),
            heartbeat.get("rf_probability_yes"),
            heartbeat.get("transformer_probability_yes"),
            heartbeat.get("model_agreement"),
            json.dumps(heartbeat.get("stale_flags") or {}, sort_keys=True),
            json.dumps(heartbeat, sort_keys=True),
        ),
    )
    conn.commit()


def score_live_model_predictions(
    *,
    recorder_db_path: str,
    prediction_db_path: str,
    lookback_hours: float = 48.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    cutoff = now_dt - timedelta(hours=float(lookback_hours))
    recorder_conn = _open_recorder_readonly(recorder_db_path)
    prediction_conn = connect_prediction_db(prediction_db_path)
    try:
        outcomes = _resolved_outcomes(recorder_conn, cutoff=cutoff)
        inserted_outcomes = 0
        for outcome in outcomes.values():
            inserted_outcomes += _insert_outcome(prediction_conn, outcome, now_dt)
        predictions = prediction_conn.execute(
            f"""
            SELECT p.*
            FROM {LIVE_MODEL_PREDICTIONS_TABLE} p
            LEFT JOIN {LIVE_MODEL_ACCURACY_TABLE} a ON a.prediction_id = p.id
            WHERE a.prediction_id IS NULL
              AND datetime(COALESCE(p.feature_timestamp, p.created_at)) >= datetime(?)
            ORDER BY datetime(COALESCE(p.feature_timestamp, p.created_at)) ASC
            """,
            (to_iso(cutoff),),
        ).fetchall()
        scored = 0
        pending = 0
        for prediction in predictions:
            outcome = outcomes.get(str(prediction["market_id"]))
            if outcome is None:
                pending += 1
                continue
            _insert_accuracy(prediction_conn, dict(prediction), outcome, now_dt)
            scored += 1
        return {
            "status": "ok",
            "prediction_db_path": prediction_db_path,
            "lookback_hours": float(lookback_hours),
            "outcomes_available": len(outcomes),
            "outcomes_inserted": inserted_outcomes,
            "predictions_considered": len(predictions),
            "predictions_scored": scored,
            "predictions_pending": pending,
        }
    finally:
        recorder_conn.close()
        prediction_conn.close()


def build_live_model_dashboard_report(
    *,
    recorder_db_path: str,
    prediction_db_path: str,
    lookback_markets: int = 100,
    max_btc_age_sec: float = 15.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    recorder_conn = _open_recorder_readonly(recorder_db_path)
    prediction_conn = connect_prediction_db(prediction_db_path)
    try:
        recorder = _recorder_health(recorder_conn, now=now_dt, max_btc_age_sec=max_btc_age_sec)
        current_market = fetch_latest_live_context(
            recorder_conn,
            now=now_dt,
            max_feature_age_sec=10**9,
            max_btc_age_sec=max_btc_age_sec,
        )
        latest_predictions = _latest_predictions(prediction_conn)
        latest_heartbeat = _latest_heartbeat_payload(prediction_conn)
        accuracy = _accuracy_summary(prediction_conn, lookback_markets=lookback_markets)
        calibration = _calibration_summary(prediction_conn)
        red_flags = _dashboard_red_flags(
            recorder=recorder,
            current_market=current_market,
            latest_predictions=latest_predictions,
            accuracy=accuracy,
            latest_heartbeat=latest_heartbeat,
        )
        mode = _latest_mode(prediction_conn)
        return {
            "timestamp": to_iso(now_dt),
            "mode": mode,
            "recorder": recorder,
            "current_market": _dashboard_market_payload(current_market),
            "latest_predictions": latest_predictions,
            "latest_heartbeat": latest_heartbeat,
            "accuracy": accuracy,
            "calibration_buckets": calibration,
            "red_flags": red_flags,
        }
    finally:
        recorder_conn.close()
        prediction_conn.close()


def render_live_model_dashboard(report: dict[str, Any]) -> str:
    lines: list[str] = []
    mode = str(report.get("mode") or "SHADOW").upper()
    lines.append(f"Live Model Dashboard [{mode}] {report.get('timestamp')}")
    lines.append("=" * 72)
    recorder = report.get("recorder") or {}
    lines.append(
        "Recorder: "
        f"feature_age={_fmt(recorder.get('latest_feature_age_sec'))}s "
        f"ready_5m={recorder.get('feature_ready_last_5m', 0)}/{recorder.get('feature_rows_last_5m', 0)} "
        f"btc={recorder.get('canonical_btc_status', 'unknown')} "
        f"wal={recorder.get('wal_size_mb', 0):.1f}MB"
    )
    market = report.get("current_market") or {}
    lines.append(
        "Market: "
        f"{market.get('market_id') or '-'} "
        f"t_close={_fmt(market.get('seconds_to_close'))}s "
        f"YES {market.get('best_bid_yes')}/{market.get('best_ask_yes')} "
        f"NO {market.get('best_bid_no')}/{market.get('best_ask_no')}"
    )
    lines.append("")
    lines.append("Latest Predictions")
    predictions = report.get("latest_predictions") or []
    if not predictions:
        lines.append("  none")
    for row in predictions:
        flags = row.get("stale_flags") or {}
        flag_text = ",".join(flags.keys()) if isinstance(flags, dict) and flags else "ok"
        lines.append(
            "  "
            f"{row.get('model_name')}({row.get('model_type')}): "
            f"YES={_fmt_prob(row.get('probability_yes'))} "
            f"NO={_fmt_prob(row.get('probability_no'))} "
            f"side={row.get('predicted_side')} "
            f"conf={_fmt_prob(row.get('confidence'))} "
            f"flags={flag_text}"
        )
    lines.append("")
    lines.append("Accuracy")
    accuracy = report.get("accuracy") or {}
    if not accuracy:
        lines.append("  no resolved predictions yet")
    for key in sorted(accuracy):
        row = accuracy[key]
        lines.append(
            "  "
            f"{key}: n={row.get('count', 0)} "
            f"acc10={_fmt_prob(row.get('last_10_accuracy'))} "
            f"acc25={_fmt_prob(row.get('last_25_accuracy'))} "
            f"acc50={_fmt_prob(row.get('last_50_accuracy'))} "
            f"acc100={_fmt_prob(row.get('last_100_accuracy'))} "
            f"pnl={_fmt(row.get('hypothetical_pnl'))}"
        )
    red_flags = report.get("red_flags") or []
    lines.append("")
    lines.append("Red Flags: " + (", ".join(red_flags) if red_flags else "none"))
    lines.append("Live trading remains blocked unless --enable-live-trading is explicitly used.")
    return "\n".join(lines)


def live_model_dashboard_loop(
    *,
    recorder_db_path: str,
    prediction_db_path: str,
    refresh_sec: float = 2.0,
    lookback_markets: int = 100,
    once: bool = False,
) -> None:
    while True:
        report = build_live_model_dashboard_report(
            recorder_db_path=recorder_db_path,
            prediction_db_path=prediction_db_path,
            lookback_markets=lookback_markets,
        )
        if not once:
            print("\033[2J\033[H", end="")
        print(render_live_model_dashboard(report), flush=True)
        if once:
            return
        time.sleep(max(0.1, float(refresh_sec)))


def start_live_model_stack(
    *,
    recorder_db_path: str,
    prediction_db_path: str,
    rf_model_path: str | None,
    rf_feature_columns_path: str | None,
    transformer_model_dir: str | None,
    poll_sec: float = 1.0,
    max_feature_age_sec: float = 5.0,
    max_btc_age_sec: float = 15.0,
    min_confidence: float = 0.55,
    run_id: str | None = None,
    shadow_only: bool = True,
    enable_live_trading: bool = False,
    dashboard: bool = True,
    dashboard_refresh_sec: float = 2.0,
) -> dict[str, Any]:
    recorder_ok = Path(recorder_db_path).exists()
    if not recorder_ok:
        raise LiveModelStackError(f"recorder DB not found: {recorder_db_path}")
    if enable_live_trading:
        print(
            json.dumps(
                {
                    "event": "live_trading_requested",
                    "warning": "no real order placement is implemented by this launcher",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not dashboard:
        return run_live_model_predictions(
            recorder_db_path=recorder_db_path,
            output_db_path=prediction_db_path,
            rf_model_path=rf_model_path,
            rf_feature_columns_path=rf_feature_columns_path,
            transformer_model_dir=transformer_model_dir,
            poll_sec=poll_sec,
            max_feature_age_sec=max_feature_age_sec,
            max_btc_age_sec=max_btc_age_sec,
            min_confidence=min_confidence,
            run_id=run_id,
            shadow_only=shadow_only,
            enable_live_trading=enable_live_trading,
        )
    prediction_thread = threading.Thread(
        target=run_live_model_predictions,
        kwargs={
            "recorder_db_path": recorder_db_path,
            "output_db_path": prediction_db_path,
            "rf_model_path": rf_model_path,
            "rf_feature_columns_path": rf_feature_columns_path,
            "transformer_model_dir": transformer_model_dir,
            "poll_sec": poll_sec,
            "max_feature_age_sec": max_feature_age_sec,
            "max_btc_age_sec": max_btc_age_sec,
            "min_confidence": min_confidence,
            "run_id": run_id,
            "shadow_only": shadow_only,
            "enable_live_trading": enable_live_trading,
            "emit_logs": False,
        },
        daemon=True,
    )
    scoring_thread = threading.Thread(
        target=_scoring_loop,
        kwargs={
            "recorder_db_path": recorder_db_path,
            "prediction_db_path": prediction_db_path,
        },
        daemon=True,
    )
    prediction_thread.start()
    scoring_thread.start()
    live_model_dashboard_loop(
        recorder_db_path=recorder_db_path,
        prediction_db_path=prediction_db_path,
        refresh_sec=dashboard_refresh_sec,
        lookback_markets=100,
        once=False,
    )
    return {"status": "stopped"}


def _scoring_loop(*, recorder_db_path: str, prediction_db_path: str) -> None:
    while True:
        try:
            score_live_model_predictions(
                recorder_db_path=recorder_db_path,
                prediction_db_path=prediction_db_path,
                lookback_hours=48,
            )
        except Exception as exc:  # pragma: no cover - operational guard
            print(json.dumps({"event": "live_model_scoring_error", "error": str(exc)}), flush=True)
        time.sleep(30.0)


def _latest_active_feature(conn: sqlite3.Connection, *, now: datetime) -> sqlite3.Row | None:
    if not _table_exists(conn, "features") or not _table_exists(conn, "markets"):
        return None
    market_cols = set(_table_columns(conn, "markets"))
    close_expr = "m.close_time" if "close_time" in market_cols else "m.end_time"
    start_expr = "m.start_time" if "start_time" in market_cols else "NULL"
    question_expr = "m.question" if "question" in market_cols else "NULL"
    query = f"""
        SELECT
            f.*,
            {question_expr} AS question,
            {start_expr} AS market_start_time,
            {close_expr} AS market_end_time
        FROM features f
        JOIN markets m
          ON m.run_id = f.run_id AND m.market_id = f.market_id
        WHERE {close_expr} IS NOT NULL
          AND datetime({close_expr}) >= datetime(?)
        ORDER BY datetime(f.timestamp) DESC, f.timestamp DESC
        LIMIT 1
    """
    return conn.execute(query, (to_iso(now - timedelta(seconds=5)),)).fetchone()


def _snapshot_for_feature(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    feature_timestamp: str,
) -> dict[str, Any]:
    if not _table_exists(conn, "market_snapshots"):
        return {}
    columns = set(_table_columns(conn, "market_snapshots"))
    timestamp_col = "timestamp" if "timestamp" in columns else None
    if timestamp_col is None:
        return {}
    row = conn.execute(
        """
        SELECT *
        FROM market_snapshots
        WHERE run_id = ? AND market_id = ? AND timestamp <= ?
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        LIMIT 1
        """,
        (run_id, market_id, feature_timestamp),
    ).fetchone()
    return dict(row) if row else {}


def _price_context(feature_row: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    best_bid_yes = _first_float((snapshot, feature_row), ("best_bid_yes", "yes_best_bid"))
    best_ask_yes = _first_float((snapshot, feature_row), ("best_ask_yes", "yes_best_ask"))
    best_bid_no = _first_float((snapshot, feature_row), ("best_bid_no", "no_best_bid"))
    best_ask_no = _first_float((snapshot, feature_row), ("best_ask_no", "no_best_ask"))
    mid_yes = _first_float((snapshot, feature_row), ("mid_price_yes", "yes_price"))
    mid_no = _first_float((snapshot, feature_row), ("mid_price_no", "no_price"))
    if mid_yes is None and best_bid_yes is not None and best_ask_yes is not None:
        mid_yes = (best_bid_yes + best_ask_yes) / 2.0
    if mid_no is None and best_bid_no is not None and best_ask_no is not None:
        mid_no = (best_bid_no + best_ask_no) / 2.0
    return {
        "best_bid_yes": best_bid_yes,
        "best_ask_yes": best_ask_yes,
        "best_bid_no": best_bid_no,
        "best_ask_no": best_ask_no,
        "entry_price_yes": best_ask_yes if best_ask_yes is not None else mid_yes,
        "entry_price_no": best_ask_no if best_ask_no is not None else mid_no,
    }


def _context_from_transformer_heartbeat(
    base_context: dict[str, Any],
    heartbeat: dict[str, Any],
) -> dict[str, Any]:
    context = dict(base_context)
    context.update(
        {
            "run_id": heartbeat.get("run_id") or base_context.get("run_id"),
            "market_id": heartbeat.get("latest_market_id") or base_context.get("market_id"),
            "feature_timestamp": heartbeat.get("latest_feature_timestamp")
            or base_context.get("feature_timestamp"),
            "market_start_time": heartbeat.get("market_start_time")
            or base_context.get("market_start_time"),
            "market_end_time": heartbeat.get("market_close_time")
            or base_context.get("market_end_time"),
            "time_until_resolution": heartbeat.get("time_until_resolution")
            or base_context.get("time_until_resolution"),
            "feature_ready": heartbeat.get("feature_ready") or base_context.get("feature_ready"),
            "snapshot_quality_status": heartbeat.get("snapshot_quality_status")
            or base_context.get("snapshot_quality_status"),
        }
    )
    for key in ("best_bid_yes", "best_ask_yes", "best_bid_no", "best_ask_no"):
        if heartbeat.get(key) is not None:
            context[key] = _float_or_none(heartbeat.get(key))
    if heartbeat.get("yes_price") is not None and context.get("entry_price_yes") is None:
        context["entry_price_yes"] = _float_or_none(heartbeat.get("yes_price"))
    if heartbeat.get("no_price") is not None and context.get("entry_price_no") is None:
        context["entry_price_no"] = _float_or_none(heartbeat.get("no_price"))
    return context


def _predict_live_probability(model: LoadedModel, row: dict[str, Any]) -> float | None:
    missing = [column for column in model.feature_columns if column not in row]
    if missing:
        raise LiveModelStackError(f"missing RF feature columns: {missing[:10]}")
    probability = _predict_probabilities(model.model, [row], model.feature_columns)[0]
    return max(0.0, min(1.0, float(probability)))


def _resolved_outcomes(conn: sqlite3.Connection, *, cutoff: datetime) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "markets"):
        return {}
    columns = set(_table_columns(conn, "markets"))
    if "resolved" not in columns or "winning_outcome" not in columns:
        return {}
    close_expr = "close_time" if "close_time" in columns else "end_time"
    rows = conn.execute(
        f"""
        SELECT *
        FROM markets
        WHERE resolved = 1
          AND winning_outcome IN ('Up', 'Down')
          AND datetime({close_expr}) >= datetime(?)
        """,
        (to_iso(cutoff),),
    ).fetchall()
    outcomes: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        winning = "YES" if payload.get("winning_outcome") == "Up" else "NO"
        outcomes[str(payload.get("market_id"))] = {
            "market_id": str(payload.get("market_id")),
            "winning_side": winning,
            "resolution_source": "markets.winning_outcome",
            "market_end_time": payload.get(close_expr),
            "resolved_at": payload.get("resolved_at") or payload.get(close_expr),
            "raw_resolution": payload,
        }
    return outcomes


def _insert_outcome(conn: sqlite3.Connection, outcome: dict[str, Any], now: datetime) -> int:
    cursor = conn.execute(
        f"""
        INSERT OR IGNORE INTO {LIVE_MODEL_OUTCOMES_TABLE} (
            resolved_at, market_id, winning_side, resolution_source,
            market_end_time, final_btc_price_if_available, raw_resolution_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            outcome.get("resolved_at") or to_iso(now),
            outcome.get("market_id"),
            outcome.get("winning_side"),
            outcome.get("resolution_source"),
            outcome.get("market_end_time"),
            None,
            json.dumps(outcome.get("raw_resolution") or {}, sort_keys=True, default=str),
        ),
    )
    conn.commit()
    return int(cursor.rowcount or 0)


def _insert_accuracy(
    conn: sqlite3.Connection,
    prediction: dict[str, Any],
    outcome: dict[str, Any],
    now: datetime,
) -> None:
    predicted_side = prediction.get("predicted_side")
    winning_side = outcome.get("winning_side")
    was_correct = None
    if predicted_side in {"YES", "NO"} and winning_side in {"YES", "NO"}:
        was_correct = 1 if predicted_side == winning_side else 0
    price = None
    if predicted_side == "YES":
        price = _float_or_none(prediction.get("entry_price_yes"))
    elif predicted_side == "NO":
        price = _float_or_none(prediction.get("entry_price_no"))
    pnl = _hypothetical_pnl(price=price, predicted_side=predicted_side, winning_side=winning_side)
    feature_ts = parse_timestamp(str(prediction.get("feature_timestamp") or ""))
    resolved_ts = parse_timestamp(str(outcome.get("resolved_at") or outcome.get("market_end_time") or ""))
    latency = (resolved_ts - feature_ts).total_seconds() if feature_ts and resolved_ts else None
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {LIVE_MODEL_ACCURACY_TABLE} (
            computed_at, model_name, model_type, market_id, prediction_id,
            predicted_side, winning_side, was_correct, probability_yes,
            confidence, price_at_signal, hypothetical_pnl_yes_no,
            latency_to_resolution_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_iso(now),
            prediction.get("model_name"),
            prediction.get("model_type"),
            prediction.get("market_id"),
            prediction.get("id"),
            predicted_side,
            winning_side,
            was_correct,
            prediction.get("probability_yes"),
            prediction.get("confidence"),
            price,
            pnl,
            latency,
        ),
    )
    conn.commit()


def _hypothetical_pnl(
    *,
    price: float | None,
    predicted_side: Any,
    winning_side: Any,
) -> float | None:
    if predicted_side not in {"YES", "NO"} or winning_side not in {"YES", "NO"}:
        return None
    if price is None or price <= 0 or price >= 1:
        return None
    if predicted_side == winning_side:
        return (1.0 / price) - 1.0
    return -1.0


def _recorder_health(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    max_btc_age_sec: float,
) -> dict[str, Any]:
    feature_rows_last_5m = 0
    feature_ready_last_5m = 0
    latest_feature_timestamp = None
    latest_feature_age_sec = None
    if _table_exists(conn, "features"):
        latest_feature_timestamp = conn.execute(
            "SELECT MAX(timestamp) FROM features"
        ).fetchone()[0]
        cutoff = to_iso(now - timedelta(minutes=5))
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows, SUM(CASE WHEN feature_ready = 1 THEN 1 ELSE 0 END) AS ready
            FROM features
            WHERE datetime(timestamp) >= datetime(?)
            """,
            (cutoff,),
        ).fetchone()
        feature_rows_last_5m = int(row["rows"] or 0)
        feature_ready_last_5m = int(row["ready"] or 0)
    latest_ts = parse_timestamp(str(latest_feature_timestamp or ""))
    if latest_ts:
        latest_feature_age_sec = (now - latest_ts).total_seconds()
    unresolved_backlog = _unresolved_market_backlog(conn, now=now)
    btc_by_source = latest_btc_by_source(conn, now=now)
    canonical = _canonical_btc_sample(btc_by_source)
    canonical_status = "missing"
    if canonical:
        age = _float_or_none(canonical.get("age_sec"))
        canonical_status = "ok" if age is not None and age <= max_btc_age_sec else "stale"
    db_path = _db_file_path(conn)
    sizes = _sqlite_sizes(db_path) if db_path else {}
    return {
        "latest_feature_timestamp": latest_feature_timestamp,
        "latest_feature_age_sec": latest_feature_age_sec,
        "feature_rows_last_5m": feature_rows_last_5m,
        "feature_ready_last_5m": feature_ready_last_5m,
        "unresolved_market_backlog": unresolved_backlog,
        "btc_by_source": btc_by_source,
        "canonical_btc_status": canonical_status,
        "db_size_mb": sizes.get("db_mb", 0.0),
        "wal_size_mb": sizes.get("wal_mb", 0.0),
        "disk_free_mb": sizes.get("disk_free_mb", 0.0),
    }


def _latest_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT p.*
        FROM {LIVE_MODEL_PREDICTIONS_TABLE} p
        JOIN (
            SELECT model_name, model_type, MAX(datetime(created_at)) AS max_created_at
            FROM {LIVE_MODEL_PREDICTIONS_TABLE}
            GROUP BY model_name, model_type
        ) latest
          ON latest.model_name = p.model_name
         AND latest.model_type = p.model_type
         AND latest.max_created_at = datetime(p.created_at)
        ORDER BY p.model_name
        """
    ).fetchall()
    result = []
    for row in rows:
        payload = dict(row)
        payload["stale_flags"] = _loads_json(payload.get("stale_flags_json"), {})
        result.append(payload)
    return result


def _latest_heartbeat_payload(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT raw_heartbeat_json
        FROM {LIVE_MODEL_HEARTBEATS_TABLE}
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return {}
    return _loads_json(row["raw_heartbeat_json"], {})


def _accuracy_summary(conn: sqlite3.Connection, *, lookback_markets: int) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM {LIVE_MODEL_ACCURACY_TABLE}
        ORDER BY datetime(computed_at) DESC, id DESC
        LIMIT ?
        """,
        (max(1000, int(lookback_markets) * 4),),
    ).fetchall()
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = dict(row)
        key = f"{payload.get('model_name')}:{payload.get('model_type')}"
        groups.setdefault(key, []).append(payload)
    summary: dict[str, dict[str, Any]] = {}
    for key, items in groups.items():
        by_market: dict[str, dict[str, Any]] = {}
        for item in items:
            market_id = str(item.get("market_id") or "")
            if market_id and market_id not in by_market:
                by_market[market_id] = item
        item_list = list(reversed(list(by_market.values())))
        summary[key] = {
            "count": len(item_list),
            "last_10_accuracy": _accuracy(item_list[-10:]),
            "last_25_accuracy": _accuracy(item_list[-25:]),
            "last_50_accuracy": _accuracy(item_list[-50:]),
            "last_100_accuracy": _accuracy(item_list[-100:]),
            "average_confidence": _mean(_float_or_none(i.get("confidence")) for i in item_list),
            "wins": sum(1 for i in item_list if i.get("was_correct") == 1),
            "losses": sum(1 for i in item_list if i.get("was_correct") == 0),
            "hypothetical_pnl": _sum(_float_or_none(i.get("hypothetical_pnl_yes_no")) for i in item_list),
        }
    return summary


def _calibration_summary(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT a.*, p.confidence
        FROM {LIVE_MODEL_ACCURACY_TABLE} a
        JOIN {LIVE_MODEL_PREDICTIONS_TABLE} p ON p.id = a.prediction_id
        """
    ).fetchall()
    buckets = {label: [] for label in ("0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00")}
    for row in rows:
        payload = dict(row)
        confidence = _float_or_none(payload.get("confidence"))
        if confidence is None:
            continue
        label = _confidence_bucket(confidence)
        if label in buckets:
            buckets[label].append(payload)
    return {
        label: {
            "count": len(items),
            "accuracy": _accuracy(items),
            "average_confidence": _mean(_float_or_none(i.get("confidence")) for i in items),
        }
        for label, items in buckets.items()
    }


def _dashboard_red_flags(
    *,
    recorder: dict[str, Any],
    current_market: dict[str, Any],
    latest_predictions: list[dict[str, Any]],
    accuracy: dict[str, Any],
    latest_heartbeat: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    feature_age = _float_or_none(recorder.get("latest_feature_age_sec"))
    if feature_age is None or feature_age > 30:
        flags.append("recorder_stale")
    if recorder.get("canonical_btc_status") != "ok":
        flags.append("btc_stale")
    if not latest_predictions:
        flags.append("no_model_predictions")
    if len(latest_predictions) >= 2:
        sides = {p.get("predicted_side") for p in latest_predictions if p.get("predicted_side") != "HOLD"}
        if len(sides) > 1:
            flags.append("model_disagreement")
    if not accuracy:
        flags.append("no_resolved_accuracy_yet")
    if int(recorder.get("unresolved_market_backlog") or 0) > 0:
        flags.append("unresolved_markets_backing_up")
    model_errors = latest_heartbeat.get("model_errors") or []
    if any(str(error.get("model")) == "transformer" for error in model_errors if isinstance(error, dict)):
        flags.append("transformer_sequence_unavailable")
    if current_market.get("stale_flags"):
        flags.extend(str(key) for key in (current_market.get("stale_flags") or {}).keys())
    if _float_or_none(recorder.get("wal_size_mb")) and recorder["wal_size_mb"] > 10240:
        flags.append("wal_over_10gb")
    return sorted(set(flags))


def _unresolved_market_backlog(conn: sqlite3.Connection, *, now: datetime) -> int:
    if not _table_exists(conn, "markets"):
        return 0
    columns = set(_table_columns(conn, "markets"))
    if "resolved" not in columns:
        return 0
    close_col = "close_time" if "close_time" in columns else "end_time" if "end_time" in columns else None
    if close_col is None:
        return 0
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM markets
        WHERE COALESCE(resolved, 0) = 0
          AND datetime({close_col}) < datetime(?)
        """,
        (to_iso(now - timedelta(minutes=10)),),
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _dashboard_market_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": context.get("market_id"),
        "question": context.get("question"),
        "market_start_time": context.get("market_start_time"),
        "market_end_time": context.get("market_end_time"),
        "seconds_to_close": context.get("time_until_resolution"),
        "best_bid_yes": context.get("best_bid_yes"),
        "best_ask_yes": context.get("best_ask_yes"),
        "best_bid_no": context.get("best_bid_no"),
        "best_ask_no": context.get("best_ask_no"),
        "stale_flags": context.get("stale_flags") or {},
    }


def _latest_mode(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        f"""
        SELECT mode
        FROM {LIVE_MODEL_HEARTBEATS_TABLE}
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row["mode"]) if row else "SHADOW"


def _open_recorder_readonly(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise LiveModelStackError(f"recorder DB not found: {path}")
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _canonical_btc_sample(samples: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if POLYMARKET_RTDS_CHAINLINK_SOURCE in samples:
        return samples[POLYMARKET_RTDS_CHAINLINK_SOURCE]
    if not samples:
        return None
    return min(
        samples.values(),
        key=lambda item: math.inf if _float_or_none(item.get("age_sec")) is None else float(item["age_sec"]),
    )


def _predicted_side(probability_yes: float, min_confidence: float) -> str:
    threshold = float(min_confidence)
    if probability_yes >= threshold:
        return "YES"
    if (1.0 - probability_yes) >= threshold:
        return "NO"
    return "HOLD"


def _model_agreement(
    rf_probability: float | None,
    transformer_probability: float | None,
    min_confidence: float,
) -> str:
    if rf_probability is None or transformer_probability is None:
        return "unavailable"
    rf_side = _predicted_side(rf_probability, min_confidence)
    transformer_side = _predicted_side(transformer_probability, min_confidence)
    if rf_side == "HOLD" or transformer_side == "HOLD":
        return "hold_or_uncertain"
    return "agree" if rf_side == transformer_side else "disagree"


def _mode(*, shadow_only: bool, enable_live_trading: bool) -> str:
    if enable_live_trading:
        return "LIVE"
    return "SHADOW" if shadow_only else "PAPER"


def _generated_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _first_float(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> float | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _db_file_path(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    path = row["file"] if isinstance(row, sqlite3.Row) else row[2]
    return Path(path) if path else None


def _sqlite_sizes(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    db_size = path.stat().st_size if path.exists() else 0
    wal_size = Path(str(path) + "-wal").stat().st_size if Path(str(path) + "-wal").exists() else 0
    shm_size = Path(str(path) + "-shm").stat().st_size if Path(str(path) + "-shm").exists() else 0
    usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("."))
    return {
        "db_mb": db_size / (1024 * 1024),
        "wal_mb": wal_size / (1024 * 1024),
        "shm_mb": shm_size / (1024 * 1024),
        "disk_free_mb": usage.free / (1024 * 1024),
    }


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.6:
        return "0.50-0.60"
    if confidence < 0.7:
        return "0.60-0.70"
    if confidence < 0.8:
        return "0.70-0.80"
    if confidence < 0.9:
        return "0.80-0.90"
    return "0.90-1.00"


def _accuracy(items: Sequence[dict[str, Any]]) -> float | None:
    values = [int(item["was_correct"]) for item in items if item.get("was_correct") in (0, 1)]
    if not values:
        return None
    return sum(values) / len(values)


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _sum(values: Iterable[float | None]) -> float:
    return float(sum(float(value) for value in values if value is not None))


def _loads_json(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def _fmt_prob(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "-"
    return f"{number:.3f}"
