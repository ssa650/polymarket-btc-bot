from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote

from .btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from .models import parse_timestamp, to_iso
from .train_baseline_model import _float_or_none
from .train_transformer_sequence_model import SequenceTransformerClassifier


class TransformerLiveInferenceError(RuntimeError):
    pass


@dataclass(slots=True)
class TransformerLiveArtifacts:
    model_path: str
    feature_columns_path: str
    scaler_stats_path: str
    training_config_path: str
    feature_columns: list[str]
    scaler_stats: dict[str, list[float]]
    training_config: dict[str, Any]


@dataclass(slots=True)
class TransformerLiveModel:
    torch: Any
    model: Any
    device: Any


InferenceFn = Callable[[list[list[float]], TransformerLiveArtifacts], float]


def run_transformer_paper_trader(
    *,
    recorder_db_path: str,
    model_path: str,
    feature_columns_path: str,
    scaler_stats_path: str | None = None,
    training_config_path: str | None = None,
    sequence_length: int = 120,
    poll_sec: float = 1.0,
    max_feature_age_sec: float = 5.0,
    allow_gap_affected: bool = False,
    allow_not_ready_sequence_rows: bool = False,
    min_ready_ratio: float = 1.0,
    sequence_row_policy: str = "latest_rows",
    diagnostics: bool = False,
    max_iterations: int | None = None,
    output_db_path: str | None = None,
    log_file_path: str | None = None,
    jsonl_log_file_path: str | None = None,
    heartbeat_log_file_path: str | None = None,
    emit_logs: bool = True,
    inference_fn: InferenceFn | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    artifacts = load_transformer_live_artifacts(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        scaler_stats_path=scaler_stats_path,
        training_config_path=training_config_path,
    )
    effective_sequence_length = _effective_sequence_length(
        requested_sequence_length=sequence_length,
        training_config=artifacts.training_config,
    )
    model_holder: dict[str, TransformerLiveModel] = {}

    def lazy_inference_fn(
        scaled_sequence: list[list[float]],
        loaded_artifacts: TransformerLiveArtifacts,
    ) -> float:
        if inference_fn is not None:
            return float(inference_fn(scaled_sequence, loaded_artifacts))
        if "model" not in model_holder:
            model_holder["model"] = load_transformer_live_model(loaded_artifacts)
        return predict_transformer_probability(model_holder["model"], scaled_sequence)

    conn = _open_readonly_sqlite(recorder_db_path)
    output_conn = _connect_prediction_output_db(output_db_path) if output_db_path else None
    jsonl_logs = _open_jsonl_logs(
        log_file_path,
        jsonl_log_file_path,
        heartbeat_log_file_path,
    )
    try:
        iteration = 0
        last_heartbeat: dict[str, Any] = {}
        while True:
            heartbeat = transformer_live_inference_heartbeat(
                conn,
                artifacts=artifacts,
                inference_fn=lazy_inference_fn,
                sequence_length=effective_sequence_length,
                max_feature_age_sec=max_feature_age_sec,
                allow_gap_affected=allow_gap_affected,
                allow_not_ready_sequence_rows=allow_not_ready_sequence_rows,
                min_ready_ratio=min_ready_ratio,
                sequence_row_policy=sequence_row_policy,
                diagnostics=diagnostics,
                now=now,
            )
            last_heartbeat = heartbeat
            if output_conn is not None:
                record_transformer_prediction(output_conn, heartbeat)
            _write_transformer_heartbeat_logs(jsonl_logs, heartbeat)
            _emit(emit_logs, "transformer_paper_trader_heartbeat", **heartbeat)
            iteration += 1
            if max_iterations is not None and iteration >= int(max_iterations):
                return {
                    "status": "ok",
                    "iterations": iteration,
                    "last_heartbeat": last_heartbeat,
                    "model_path": artifacts.model_path,
                    "feature_columns_path": artifacts.feature_columns_path,
                    "scaler_stats_path": artifacts.scaler_stats_path,
                    "training_config_path": artifacts.training_config_path,
                    "sequence_length": effective_sequence_length,
                    "sequence_row_policy": sequence_row_policy,
                    "diagnostics": diagnostics,
                    "output_db_path": output_db_path,
                    "log_file_path": log_file_path,
                    "jsonl_log_file_path": jsonl_log_file_path,
                    "heartbeat_log_file_path": heartbeat_log_file_path,
                }
            time.sleep(max(0.0, float(poll_sec)))
    finally:
        for handle in jsonl_logs:
            handle.close()
        if output_conn is not None:
            output_conn.close()
        conn.close()


def transformer_live_inference_heartbeat(
    conn: sqlite3.Connection,
    *,
    artifacts: TransformerLiveArtifacts,
    model_bundle: TransformerLiveModel | None = None,
    inference_fn: InferenceFn | None = None,
    sequence_length: int = 120,
    max_feature_age_sec: float = 5.0,
    allow_gap_affected: bool = False,
    allow_not_ready_sequence_rows: bool = False,
    min_ready_ratio: float = 1.0,
    sequence_row_policy: str = "latest_rows",
    diagnostics: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    sequence = build_latest_live_sequence(
        conn,
        feature_columns=artifacts.feature_columns,
        sequence_length=sequence_length,
        max_feature_age_sec=max_feature_age_sec,
        allow_gap_affected=allow_gap_affected,
        allow_not_ready_sequence_rows=allow_not_ready_sequence_rows,
        min_ready_ratio=min_ready_ratio,
        sequence_row_policy=sequence_row_policy,
        now=now_dt,
    )
    heartbeat = {
        "timestamp": to_iso(now_dt),
        "transformer_sequence_ready": bool(sequence.get("ready")),
        "sequence_rows_available": int(sequence.get("sequence_rows_available") or 0),
        "probability_yes": None,
        "model_path": artifacts.model_path,
        "latest_market_id": sequence.get("latest_market_id"),
        "latest_feature_timestamp": sequence.get("latest_feature_timestamp"),
        "sequence_length": int(sequence_length),
        "sequence_row_policy": sequence.get("sequence_row_policy", sequence_row_policy),
        "diagnostics": bool(diagnostics),
        "reason": sequence.get("reason"),
        "missing_feature_columns": sequence.get("missing_feature_columns", []),
    }
    for key in (
        "run_id",
        "market_start_time",
        "market_close_time",
        "time_until_resolution",
        "yes_price",
        "no_price",
        "best_bid_yes",
        "best_ask_yes",
        "best_bid_no",
        "best_ask_no",
        "mid_price_yes",
        "mid_price_no",
        "spread_yes",
        "spread_no",
        "feature_ready",
        "snapshot_quality_status",
        "strict_validation_passed",
        "total_sequence_rows_checked",
        "ready_ratio",
        "not_ready_rows_allowed",
        "feature_ready_false_count",
        "gap_affected_true_count",
        "earliest_sequence_timestamp",
        "latest_sequence_timestamp",
        "latest_row_feature_ready",
        "latest_row_snapshot_quality_status",
        "missing_or_null_counts_top20",
        "bad_row_samples",
        "available_feature_table_columns",
        "available_sequence_columns",
        "model_feature_columns_count",
        "clean_sequence_rows_available",
        "latest_clean_sequence_timestamp",
        "rejected_rows_by_reason",
    ):
        if key in sequence:
            heartbeat[key] = sequence[key]
    if not sequence.get("ready"):
        return heartbeat
    scaled_sequence = apply_scaler_to_sequence(
        sequence["sequence_values"],
        artifacts.scaler_stats,
    )
    if inference_fn is not None:
        probability = float(inference_fn(scaled_sequence, artifacts))
    elif model_bundle is not None:
        probability = predict_transformer_probability(model_bundle, scaled_sequence)
    else:
        raise TransformerLiveInferenceError("no transformer inference backend configured")
    heartbeat["probability_yes"] = max(0.0, min(1.0, float(probability)))
    return heartbeat


def load_transformer_live_artifacts(
    *,
    model_path: str,
    feature_columns_path: str,
    scaler_stats_path: str | None = None,
    training_config_path: str | None = None,
) -> TransformerLiveArtifacts:
    model = Path(model_path)
    feature_path = Path(feature_columns_path)
    scaler_path = Path(scaler_stats_path) if scaler_stats_path else model.parent / "scaler_stats.json"
    config_path = (
        Path(training_config_path)
        if training_config_path
        else model.parent / "training_config.json"
    )
    feature_columns = _read_json_list(feature_path, label="feature_columns")
    scaler = _read_json_dict(scaler_path, label="scaler_stats")
    training_config = _read_json_dict(config_path, label="training_config")
    _validate_scaler(scaler, feature_count=len(feature_columns), path=str(scaler_path))
    return TransformerLiveArtifacts(
        model_path=str(model),
        feature_columns_path=str(feature_path),
        scaler_stats_path=str(scaler_path),
        training_config_path=str(config_path),
        feature_columns=[str(column) for column in feature_columns],
        scaler_stats=scaler,
        training_config=training_config,
    )


def load_transformer_live_model(artifacts: TransformerLiveArtifacts) -> TransformerLiveModel:
    try:
        import torch
    except ImportError as exc:
        raise TransformerLiveInferenceError(
            "run-transformer-paper-trader requires PyTorch to load model.pt. "
            "Install PyTorch in the project venv before running live transformer inference."
        ) from exc
    checkpoint = torch.load(artifacts.model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TransformerLiveInferenceError("model.pt must contain a transformer checkpoint dict")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        model_config = _model_config_from_training_config(
            artifacts.training_config,
            feature_count=len(artifacts.feature_columns),
        )
    model = SequenceTransformerClassifier(
        torch,
        feature_count=int(model_config.get("feature_count") or len(artifacts.feature_columns)),
        d_model=int(model_config.get("d_model") or 32),
        nhead=int(model_config.get("nhead") or 4),
        num_layers=int(model_config.get("num_layers") or 1),
        dim_feedforward=int(model_config.get("dim_feedforward") or 64),
        dropout=float(model_config.get("dropout") or 0.0),
    )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise TransformerLiveInferenceError("model.pt is missing model_state_dict")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return TransformerLiveModel(torch=torch, model=model, device=device)


def predict_transformer_probability(
    model_bundle: TransformerLiveModel,
    scaled_sequence: Sequence[Sequence[float]],
) -> float:
    torch = model_bundle.torch
    tensor = torch.tensor([scaled_sequence], dtype=torch.float32, device=model_bundle.device)
    with torch.no_grad():
        logits = model_bundle.model(tensor)
        probability = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()[0]
    return float(probability)


def build_latest_live_sequence(
    conn: sqlite3.Connection,
    *,
    feature_columns: Sequence[str],
    sequence_length: int,
    max_feature_age_sec: float,
    allow_gap_affected: bool,
    allow_not_ready_sequence_rows: bool = False,
    min_ready_ratio: float = 1.0,
    sequence_row_policy: str = "latest_rows",
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    policy = str(sequence_row_policy or "latest_rows")
    if policy not in {"latest_rows", "latest_clean_rows"}:
        raise TransformerLiveInferenceError(
            "sequence_row_policy must be latest_rows or latest_clean_rows"
        )
    latest = _latest_active_feature(conn, now=now_dt)
    if latest is None:
        return {
            "ready": False,
            "reason": "no_active_feature",
            "sequence_rows_available": 0,
            "sequence_row_policy": policy,
            "latest_market_id": None,
            "latest_feature_timestamp": None,
            "sequence_values": [],
        }
    if policy == "latest_clean_rows":
        return _build_latest_clean_live_sequence(
            conn,
            latest=latest,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            max_feature_age_sec=max_feature_age_sec,
            now=now_dt,
            sequence_row_policy=policy,
        )
    rows = _fetch_sequence_rows(
        conn,
        run_id=str(latest["run_id"]),
        market_id=str(latest["market_id"]),
        latest_timestamp=str(latest["timestamp"]),
        limit=int(sequence_length),
    )
    rows = list(reversed(rows))
    base = {
        "run_id": str(latest["run_id"]),
        "latest_market_id": str(latest["market_id"]),
        "latest_feature_timestamp": str(latest["timestamp"]),
        "sequence_rows_available": len(rows),
        "sequence_row_policy": policy,
        "available_feature_table_columns": _table_columns(conn, "features"),
    }
    if rows:
        base.update(_prediction_context_from_row(rows[-1]))
    if len(rows) < int(sequence_length):
        return {
            **base,
            "ready": False,
            "reason": "insufficient_sequence_rows",
            "sequence_values": [],
        }
    latest_ts = parse_timestamp(str(latest["timestamp"]))
    if latest_ts is None:
        return {**base, "ready": False, "reason": "invalid_latest_feature_timestamp", "sequence_values": []}
    if max_feature_age_sec >= 0:
        age = (now_dt - latest_ts).total_seconds()
        if age > float(max_feature_age_sec):
            return {
                **base,
                "ready": False,
                "reason": "feature_stale",
                "feature_age_sec": age,
                "sequence_values": [],
            }
    readiness = _sequence_readiness_diagnostics(rows, feature_columns)
    base.update(readiness)
    base["not_ready_rows_allowed"] = 0
    latest_ready = _flag_enabled(rows[-1].get("feature_ready"))
    effective_min_ready_ratio = min(max(float(min_ready_ratio), 0.0), 1.0)
    not_ready_relaxed = bool(allow_not_ready_sequence_rows) or effective_min_ready_ratio < 1.0
    if readiness["feature_ready_false_count"] > 0:
        if not not_ready_relaxed:
            return {
                **base,
                "ready": False,
                "reason": "not_feature_ready",
                "sequence_values": [],
            }
        if not latest_ready:
            return {
                **base,
                "ready": False,
                "reason": "latest_row_not_feature_ready",
                "sequence_values": [],
            }
        if (
            effective_min_ready_ratio < 1.0
            and readiness["ready_ratio"] < effective_min_ready_ratio
        ):
            return {
                **base,
                "ready": False,
                "reason": "min_ready_ratio_not_met",
                "sequence_values": [],
            }
        base["not_ready_rows_allowed"] = int(readiness["feature_ready_false_count"])
    else:
        base["not_ready_rows_allowed"] = 0
    if not allow_gap_affected and readiness["gap_affected_true_count"] > 0:
        return {
            **base,
            "ready": False,
            "reason": "gap_affected",
            "sequence_values": [],
        }
    missing_columns = [
        str(column)
        for column in feature_columns
        if any(str(column) not in row for row in rows)
    ]
    if missing_columns:
        return {
            **base,
            "ready": False,
            "reason": "missing_feature_columns",
            "missing_feature_columns": missing_columns,
            "available_sequence_columns": sorted(
                {str(key) for row in rows for key in row.keys()}
            ),
            "model_feature_columns_count": len(list(feature_columns)),
            "sequence_values": [],
        }
    sequence_values = [
        [_float_or_none(row.get(str(column))) for column in feature_columns]
        for row in rows
    ]
    return {
        **base,
        "ready": True,
        "reason": None,
        "missing_feature_columns": [],
        "sequence_values": sequence_values,
    }


def _build_latest_clean_live_sequence(
    conn: sqlite3.Connection,
    *,
    latest: dict[str, Any],
    feature_columns: Sequence[str],
    sequence_length: int,
    max_feature_age_sec: float,
    now: datetime,
    sequence_row_policy: str,
) -> dict[str, Any]:
    checked_rows = _fetch_all_market_rows_through_timestamp(
        conn,
        run_id=str(latest["run_id"]),
        market_id=str(latest["market_id"]),
        latest_timestamp=str(latest["timestamp"]),
    )
    checked_rows = list(reversed(checked_rows))
    rejected_counts: dict[str, int] = {}
    clean_rows: list[dict[str, Any]] = []
    missing_model_columns: set[str] = set()
    for row in checked_rows:
        reasons, missing_columns = _clean_row_rejection_reasons(row, feature_columns)
        if reasons:
            for reason in reasons:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            missing_model_columns.update(missing_columns)
            continue
        clean_rows.append(row)

    selected_rows = clean_rows[-int(sequence_length):]
    latest_clean_timestamp = (
        str(selected_rows[-1].get("timestamp"))
        if selected_rows and selected_rows[-1].get("timestamp") is not None
        else None
    )
    base = {
        "run_id": str(latest["run_id"]),
        "latest_market_id": str(latest["market_id"]),
        "latest_feature_timestamp": latest_clean_timestamp or str(latest["timestamp"]),
        "sequence_rows_available": len(selected_rows),
        "clean_sequence_rows_available": len(clean_rows),
        "latest_clean_sequence_timestamp": latest_clean_timestamp,
        "rejected_rows_by_reason": dict(sorted(rejected_counts.items())),
        "sequence_row_policy": sequence_row_policy,
        "available_feature_table_columns": _table_columns(conn, "features"),
    }
    if selected_rows:
        base.update(_prediction_context_from_row(selected_rows[-1]))
    if len(selected_rows) < int(sequence_length):
        return {
            **base,
            "ready": False,
            "reason": "insufficient_clean_sequence_rows",
            "missing_feature_columns": sorted(missing_model_columns),
            "sequence_values": [],
        }
    if latest_clean_timestamp is None:
        return {
            **base,
            "ready": False,
            "reason": "invalid_latest_clean_sequence_timestamp",
            "sequence_values": [],
        }
    latest_ts = parse_timestamp(latest_clean_timestamp)
    if latest_ts is None:
        return {
            **base,
            "ready": False,
            "reason": "invalid_latest_clean_sequence_timestamp",
            "sequence_values": [],
        }
    if max_feature_age_sec >= 0:
        age = (now - latest_ts).total_seconds()
        if age > float(max_feature_age_sec):
            return {
                **base,
                "ready": False,
                "reason": "feature_stale",
                "feature_age_sec": age,
                "sequence_values": [],
            }
    readiness = _sequence_readiness_diagnostics(selected_rows, feature_columns)
    base.update(readiness)
    base["not_ready_rows_allowed"] = 0
    sequence_values = [
        [_float_or_none(row.get(str(column))) for column in feature_columns]
        for row in selected_rows
    ]
    return {
        **base,
        "ready": True,
        "reason": None,
        "missing_feature_columns": [],
        "sequence_values": sequence_values,
    }


def record_transformer_prediction(
    conn: sqlite3.Connection,
    heartbeat: dict[str, Any],
) -> bool:
    probability_yes = _float_or_none(heartbeat.get("probability_yes"))
    ensure_transformer_prediction_schema(conn)
    created_at = str(heartbeat.get("timestamp") or to_iso(datetime.now(timezone.utc)))
    market_id = str(heartbeat.get("latest_market_id") or "")
    latest_feature_timestamp = str(heartbeat.get("latest_feature_timestamp") or "")
    if probability_yes is not None:
        probability_yes = max(0.0, min(1.0, probability_yes))
    latest_sequence_timestamp = (
        heartbeat.get("latest_sequence_timestamp")
        or heartbeat.get("latest_clean_sequence_timestamp")
        or latest_feature_timestamp
    )
    values = {
        "created_at": created_at,
        "run_id": str(heartbeat.get("run_id") or ""),
        "market_id": market_id,
        "timestamp": created_at,
        "signal_timestamp": latest_feature_timestamp,
        "model_path": str(heartbeat.get("model_path") or ""),
        "sequence_length": int(heartbeat.get("sequence_length") or 0),
        "sequence_row_policy": str(heartbeat.get("sequence_row_policy") or ""),
        "transformer_sequence_ready": 1 if heartbeat.get("transformer_sequence_ready") else 0,
        "probability_yes": probability_yes,
        "probability_no": 1.0 - probability_yes if probability_yes is not None else None,
        "feature_timestamp": latest_feature_timestamp,
        "latest_feature_timestamp": latest_feature_timestamp,
        "latest_sequence_timestamp": latest_sequence_timestamp,
        "market_start_time": heartbeat.get("market_start_time"),
        "market_close_time": heartbeat.get("market_close_time"),
        "time_until_resolution": _float_or_none(heartbeat.get("time_until_resolution")),
        "yes_price": _float_or_none(heartbeat.get("yes_price")),
        "no_price": _float_or_none(heartbeat.get("no_price")),
        "best_bid_yes": _float_or_none(heartbeat.get("best_bid_yes")),
        "best_ask_yes": _float_or_none(heartbeat.get("best_ask_yes")),
        "best_bid_no": _float_or_none(heartbeat.get("best_bid_no")),
        "best_ask_no": _float_or_none(heartbeat.get("best_ask_no")),
        "mid_price_yes": _float_or_none(heartbeat.get("mid_price_yes")),
        "mid_price_no": _float_or_none(heartbeat.get("mid_price_no")),
        "spread_yes": _float_or_none(heartbeat.get("spread_yes")),
        "spread_no": _float_or_none(heartbeat.get("spread_no")),
        "feature_ready": _int_or_none(heartbeat.get("feature_ready")),
        "snapshot_quality_status": heartbeat.get("snapshot_quality_status"),
        "strict_validation_passed": _int_or_none(heartbeat.get("strict_validation_passed")),
        "ready_ratio": _float_or_none(heartbeat.get("ready_ratio")),
        "sequence_rows_available": int(heartbeat.get("sequence_rows_available") or 0),
        "diagnostics_reason": heartbeat.get("reason"),
        "reason": heartbeat.get("reason"),
        "missing_feature_columns_json": _json_dumps(heartbeat.get("missing_feature_columns")),
        "rejected_rows_by_reason_json": _json_dumps(heartbeat.get("rejected_rows_by_reason")),
    }
    columns = list(values.keys())
    placeholders = ", ".join("?" for _ in columns)
    with conn:
        cursor = conn.execute(
            f"""
            INSERT INTO transformer_predictions (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [values[column] for column in columns],
        )
    return cursor.rowcount > 0


def ensure_transformer_prediction_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transformer_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            run_id TEXT NOT NULL DEFAULT '',
            market_id TEXT NOT NULL,
            timestamp TEXT,
            signal_timestamp TEXT,
            model_path TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            sequence_row_policy TEXT,
            transformer_sequence_ready INTEGER,
            probability_yes REAL,
            probability_no REAL,
            feature_timestamp TEXT,
            latest_feature_timestamp TEXT,
            latest_sequence_timestamp TEXT,
            market_start_time TEXT,
            market_close_time TEXT,
            time_until_resolution REAL,
            yes_price REAL,
            no_price REAL,
            best_bid_yes REAL,
            best_ask_yes REAL,
            best_bid_no REAL,
            best_ask_no REAL,
            mid_price_yes REAL,
            mid_price_no REAL,
            spread_yes REAL,
            spread_no REAL,
            feature_ready INTEGER,
            snapshot_quality_status TEXT,
            strict_validation_passed INTEGER,
            ready_ratio REAL,
            sequence_rows_available INTEGER,
            diagnostics_reason TEXT,
            reason TEXT,
            missing_feature_columns_json TEXT,
            rejected_rows_by_reason_json TEXT
        )
        """
    )
    _ensure_transformer_prediction_columns(conn)
    conn.execute("DROP INDEX IF EXISTS idx_transformer_predictions_unique_feature")
    for column in ("timestamp", "market_id", "run_id", "sequence_length", "probability_yes"):
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_transformer_predictions_{column}
            ON transformer_predictions ({column})
            """
        )
    conn.commit()


def _ensure_transformer_prediction_columns(conn: sqlite3.Connection) -> None:
    existing = set(_table_columns(conn, "transformer_predictions"))
    additions = {
        "sequence_row_policy": "TEXT",
        "transformer_sequence_ready": "INTEGER",
        "feature_timestamp": "TEXT",
        "latest_sequence_timestamp": "TEXT",
        "missing_feature_columns_json": "TEXT",
        "rejected_rows_by_reason_json": "TEXT",
    }
    for column, column_type in additions.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE transformer_predictions ADD COLUMN {column} {column_type}"
            )


def _connect_prediction_output_db(path: str | None) -> sqlite3.Connection:
    if not path:
        raise TransformerLiveInferenceError("output prediction DB path is required")
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_transformer_prediction_schema(conn)
    return conn


def _open_jsonl_logs(*paths: str | None) -> list[Any]:
    handles: list[Any] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        log_path = Path(path)
        key = str(log_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handles.append(log_path.open("a", encoding="utf-8", buffering=1))
    return handles


def _write_transformer_heartbeat_logs(
    handles: Sequence[Any],
    heartbeat: dict[str, Any],
) -> None:
    if not handles:
        return
    payload = _compact_transformer_heartbeat_payload(heartbeat)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    for handle in handles:
        handle.write(line)
        handle.flush()


def _compact_transformer_heartbeat_payload(heartbeat: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "transformer_paper_trader_heartbeat",
        "timestamp": heartbeat.get("timestamp"),
        "run_id": heartbeat.get("run_id"),
        "market_id": heartbeat.get("latest_market_id"),
        "latest_market_id": heartbeat.get("latest_market_id"),
        "model_path": heartbeat.get("model_path"),
        "sequence_length": heartbeat.get("sequence_length"),
        "sequence_row_policy": heartbeat.get("sequence_row_policy"),
        "transformer_sequence_ready": bool(heartbeat.get("transformer_sequence_ready")),
        "readiness_status": "ready" if heartbeat.get("transformer_sequence_ready") else "not_ready",
        "probability_yes": heartbeat.get("probability_yes"),
        "reason": heartbeat.get("reason"),
        "feature_timestamp": heartbeat.get("latest_feature_timestamp"),
        "latest_feature_timestamp": heartbeat.get("latest_feature_timestamp"),
        "latest_sequence_timestamp": (
            heartbeat.get("latest_sequence_timestamp")
            or heartbeat.get("latest_clean_sequence_timestamp")
        ),
        "sequence_rows_available": heartbeat.get("sequence_rows_available"),
        "clean_sequence_rows_available": heartbeat.get("clean_sequence_rows_available"),
        "ready_ratio": heartbeat.get("ready_ratio"),
        "snapshot_quality_status": heartbeat.get("snapshot_quality_status"),
        "time_until_resolution": heartbeat.get("time_until_resolution"),
        "market_start_time": heartbeat.get("market_start_time"),
        "market_close_time": heartbeat.get("market_close_time"),
        "yes_price": heartbeat.get("yes_price"),
        "no_price": heartbeat.get("no_price"),
        "mid_price_yes": heartbeat.get("mid_price_yes"),
        "mid_price_no": heartbeat.get("mid_price_no"),
        "spread_yes": heartbeat.get("spread_yes"),
        "spread_no": heartbeat.get("spread_no"),
        "missing_feature_columns": heartbeat.get("missing_feature_columns", []),
        "rejected_rows_by_reason": heartbeat.get("rejected_rows_by_reason", {}),
    }


def report_transformer_predictions(
    prediction_db_paths: Sequence[str],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = to_iso(now or datetime.now(timezone.utc))
    rows, input_dbs = _load_prediction_db_rows(prediction_db_paths)

    probabilities = [
        probability
        for probability in (_float_or_none(row.get("probability_yes")) for row in rows)
        if probability is not None
    ]
    probability_rows = [
        row for row in rows if _float_or_none(row.get("probability_yes")) is not None
    ]
    ready_rows = [
        row for row in rows if _flag_enabled(row.get("transformer_sequence_ready"))
    ]
    latencies = [
        latency
        for latency in (_prediction_latency_sec(row) for row in rows)
        if latency is not None
    ]
    return {
        "status": "ok",
        "generated_at": generated_at,
        "prediction_db_paths": [str(path) for path in prediction_db_paths],
        "input_dbs": input_dbs,
        "total_rows": len(rows),
        "ready_rows": len(ready_rows),
        "non_null_probability_rows": len(probabilities),
        "market_count": _market_count(rows),
        "markets_with_probability_count": _market_count(probability_rows),
        "min_timestamp": _min_timestamp(rows),
        "max_timestamp": _max_timestamp(rows),
        "min_probability_yes": round(min(probabilities), 10) if probabilities else None,
        "average_probability_yes": (
            round(sum(probabilities) / len(probabilities), 10) if probabilities else None
        ),
        "max_probability_yes": round(max(probabilities), 10) if probabilities else None,
        "probability_buckets": _probability_buckets(probabilities),
        "threshold_counts": {
            "gte_0_75": sum(1 for value in probabilities if value >= 0.75),
            "gte_0_77": sum(1 for value in probabilities if value >= 0.77),
            "gte_0_79": sum(1 for value in probabilities if value >= 0.79),
            "gte_0_80": sum(1 for value in probabilities if value >= 0.80),
            "gte_0_85": sum(1 for value in probabilities if value >= 0.85),
            "gte_0_90": sum(1 for value in probabilities if value >= 0.90),
        },
        "average_prediction_feature_latency_sec": (
            round(sum(latencies) / len(latencies), 6) if latencies else None
        ),
        "max_prediction_feature_latency_sec": (
            round(max(latencies), 6) if latencies else None
        ),
        "readiness_failure_reasons": _reason_counts(rows),
        "top_rejection_reasons": _top_rejection_reasons(rows),
        "top_markets_by_max_probability": _top_markets_by_max_probability(rows),
    }


def analyze_transformer_calibration(
    *,
    prediction_db_paths: Sequence[str] | None = None,
    jsonl_log_paths: Sequence[str] | None = None,
    recorder_db_path: str | None = None,
    output_json_path: str | None = None,
    output_txt_path: str | None = None,
    thresholds: Sequence[float] = (0.75, 0.77, 0.79, 0.80),
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = to_iso(now or datetime.now(timezone.utc))
    db_rows, input_dbs = _load_prediction_db_rows(prediction_db_paths or [])
    jsonl_rows, input_jsonl_logs = _load_jsonl_prediction_rows(jsonl_log_paths or [])
    rows = [*db_rows, *jsonl_rows]
    probabilities = [
        probability
        for probability in (_float_or_none(row.get("probability_yes")) for row in rows)
        if probability is not None
    ]
    ready_rows = [
        row for row in rows if _flag_enabled(row.get("transformer_sequence_ready"))
    ]
    outcomes = _load_recorder_market_outcomes(recorder_db_path) if recorder_db_path else {}
    outcome_join = _join_transformer_prediction_outcomes(rows, outcomes)
    report = {
        "status": "ok",
        "generated_at": generated_at,
        "prediction_db_paths": [str(path) for path in prediction_db_paths or []],
        "jsonl_log_paths": [str(path) for path in jsonl_log_paths or []],
        "recorder_db_path": recorder_db_path,
        "input_dbs": input_dbs,
        "input_jsonl_logs": input_jsonl_logs,
        "rows": len(rows),
        "markets": _market_count(rows),
        "probability_rows": len(probabilities),
        "ready_rows": len(ready_rows),
        "threshold_hit_counts": {
            _threshold_key(threshold): sum(1 for value in probabilities if value >= float(threshold))
            for threshold in thresholds
        },
        "probability_summary": {
            "min": round(min(probabilities), 10) if probabilities else None,
            "average": round(sum(probabilities) / len(probabilities), 10) if probabilities else None,
            "max": round(max(probabilities), 10) if probabilities else None,
        },
        "per_market_probability": _per_market_probability(rows, outcomes),
        "outcome_join": {
            "available": bool(outcomes),
            "joined_rows": int(outcome_join.get("joined_rows") or 0),
            "joined_markets": int(outcome_join.get("joined_markets") or 0),
            "unjoined_rows": int(outcome_join.get("unjoined_rows") or 0),
        },
        "threshold_outcome_calibration": _threshold_outcome_calibration(
            rows,
            outcomes,
            thresholds,
        ),
        "warnings": _transformer_calibration_warnings(
            rows=rows,
            probabilities=probabilities,
            outcomes=outcomes,
            joined_rows=int(outcome_join.get("joined_rows") or 0),
        ),
        "recommendation": "paper/shadow only; no live-trading recommendation",
    }
    if output_json_path:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["output_json_path"] = str(path)
    if output_txt_path:
        path = Path(output_txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_transformer_calibration_report(report), encoding="utf-8")
        report["output_txt_path"] = str(path)
    return report


def backtest_transformer_threshold_strategy(
    *,
    prediction_db_paths: Sequence[str],
    recorder_db_path: str,
    output_json_path: str | None = None,
    output_txt_path: str | None = None,
    thresholds: Sequence[float] = (0.75, 0.77, 0.775, 0.78, 0.79, 0.80),
    min_time_until_resolution_options: Sequence[float] = (30.0, 60.0, 90.0, 120.0),
    max_time_until_resolution_options: Sequence[float] = (90.0, 120.0, 180.0, 240.0),
    exit_horizon_sec_options: Sequence[int] = (15, 30, 60),
    side: str = "BOTH",
    one_trade_per_market: bool = True,
    slippage_cents: float = 1.0,
    fee_cents: float = 0.0,
    min_done_trades: int = 5,
    split_by_time: bool = False,
    train_ratio: float = 0.5,
    min_test_done_trades: int = 20,
    strategy_filter: str | Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = to_iso(now or datetime.now(timezone.utc))
    prediction_rows, input_dbs = _load_prediction_db_rows(prediction_db_paths)
    predictions = [
        row for row in prediction_rows
        if _float_or_none(row.get("probability_yes")) is not None
    ]
    sides = _backtest_sides(side)
    strategy_filters = _strategy_filter_values(strategy_filter)
    recorder = _open_readonly_prediction_db(recorder_db_path)
    try:
        simulate_kwargs = {
            "thresholds": [float(value) for value in thresholds],
            "min_time_options": [float(value) for value in min_time_until_resolution_options],
            "max_time_options": [float(value) for value in max_time_until_resolution_options],
            "horizon_options": [int(value) for value in exit_horizon_sec_options],
            "sides": sides,
            "one_trade_per_market": bool(one_trade_per_market),
            "slippage_cents": float(slippage_cents),
            "fee_cents": float(fee_cents),
        }
        simulated = _filter_threshold_strategy_rows(
            _simulate_transformer_threshold_trades(
                recorder,
                predictions,
                **simulate_kwargs,
            ),
            strategy_filters,
        )
        validation = None
        if split_by_time:
            split = _split_predictions_by_market_time(
                predictions,
                train_ratio=float(train_ratio),
            )
            train_simulated = _filter_threshold_strategy_rows(
                _simulate_transformer_threshold_trades(
                    recorder,
                    split["train_predictions"],
                    **simulate_kwargs,
                ),
                strategy_filters,
            )
            test_simulated = _filter_threshold_strategy_rows(
                _simulate_transformer_threshold_trades(
                    recorder,
                    split["test_predictions"],
                    **simulate_kwargs,
                ),
                strategy_filters,
            )
            validation = _threshold_backtest_validation_report(
                train_rows=train_simulated,
                test_rows=test_simulated,
                split=split,
                min_done_trades=int(min_done_trades),
                min_test_done_trades=int(min_test_done_trades),
            )
    finally:
        recorder.close()
    warnings = [
        "paper/shadow only; do not go live",
        *(
            ["no_completed_backtest_trades"]
            if not any(row.get("done") for row in simulated)
            else []
        ),
    ]
    if validation:
        warnings.extend(validation.get("warnings") or [])
    report = {
        "status": "ok",
        "generated_at": generated_at,
        "prediction_db_paths": list(prediction_db_paths),
        "recorder_db_path": recorder_db_path,
        "input_dbs": input_dbs,
        "prediction_rows_loaded": len(prediction_rows),
        "probability_rows_loaded": len(predictions),
        "thresholds": [float(value) for value in thresholds],
        "min_time_until_resolution_sec_options": [
            float(value) for value in min_time_until_resolution_options
        ],
        "max_time_until_resolution_sec_options": [
            float(value) for value in max_time_until_resolution_options
        ],
        "exit_horizon_sec_options": [int(value) for value in exit_horizon_sec_options],
        "side": str(side).upper(),
        "one_trade_per_market": bool(one_trade_per_market),
        "slippage_cents": float(slippage_cents),
        "fee_cents": float(fee_cents),
        "min_done_trades": int(min_done_trades),
        "split_by_time": bool(split_by_time),
        "train_ratio": float(train_ratio),
        "min_test_done_trades": int(min_test_done_trades),
        "strategy_filter": sorted(strategy_filters) if strategy_filters else None,
        "trades": len(simulated),
        "done_trades": sum(1 for row in simulated if row.get("done")),
        "markets_covered": _market_count(simulated),
        "best_by_pnl": _strategy_metric_rows(simulated, min_done_trades=0)[:20],
        "best_by_avg_roi_min_done": _strategy_metric_rows(
            simulated,
            min_done_trades=int(min_done_trades),
            sort_by="average_roi",
        )[:20],
        "threshold_comparison": _comparison_metrics(
            simulated,
            fields=("threshold",),
            min_done_trades=0,
        ),
        "time_window_comparison": _comparison_metrics(
            simulated,
            fields=("min_time_until_resolution_sec", "max_time_until_resolution_sec"),
            min_done_trades=0,
        ),
        "side_comparison": _comparison_metrics(
            simulated,
            fields=("side",),
            min_done_trades=0,
        ),
        "horizon_comparison": _comparison_metrics(
            simulated,
            fields=("exit_horizon_sec",),
            min_done_trades=0,
        ),
        "validation": validation,
        "warnings": warnings,
        "recommendation": "paper/shadow only; do not go live",
    }
    if output_json_path:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        report["output_json_path"] = str(path)
    if output_txt_path:
        path = Path(output_txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_transformer_threshold_backtest(report), encoding="utf-8")
        report["output_txt_path"] = str(path)
    return report


def render_transformer_threshold_backtest(report: dict[str, Any]) -> str:
    lines = [
        "Transformer Threshold Strategy Backtest",
        f"generated_at: {report.get('generated_at')}",
        (
            f"prediction_rows={report.get('prediction_rows_loaded')} "
            f"probability_rows={report.get('probability_rows_loaded')} "
            f"trades={report.get('trades')} done_trades={report.get('done_trades')} "
            f"markets={report.get('markets_covered')}"
        ),
        "Warning: paper/shadow only; do not go live",
        "",
        "Best Strategies By PnL",
    ]
    for row in list(report.get("best_by_pnl") or [])[:10]:
        lines.append(_format_threshold_backtest_metric(row))
    lines.extend(["", f"Best Strategies By Avg ROI (min_done_trades={report.get('min_done_trades')})"])
    for row in list(report.get("best_by_avg_roi_min_done") or [])[:10]:
        lines.append(_format_threshold_backtest_metric(row))
    lines.extend(["", "Threshold Comparison"])
    for row in list(report.get("threshold_comparison") or [])[:10]:
        lines.append(_format_threshold_backtest_metric(row, include_strategy=False))
    lines.extend(["", "Time Window Comparison"])
    for row in list(report.get("time_window_comparison") or [])[:10]:
        lines.append(_format_threshold_backtest_metric(row, include_strategy=False))
    lines.extend(["", "YES vs NO Comparison"])
    for row in list(report.get("side_comparison") or []):
        lines.append(_format_threshold_backtest_metric(row, include_strategy=False))
    validation = dict(report.get("validation") or {})
    if validation:
        lines.extend(["", "Out-of-Sample Time Split Validation"])
        split = dict(validation.get("split") or {})
        lines.append(
            f"  train_markets={split.get('train_market_count')} "
            f"test_markets={split.get('test_market_count')} "
            f"train_rows={split.get('train_prediction_rows')} "
            f"test_rows={split.get('test_prediction_rows')}"
        )
        lines.append("  Train aggregate:")
        lines.append(_format_threshold_backtest_metric(dict(validation.get("train") or {})))
        lines.append("  Test aggregate:")
        lines.append(_format_threshold_backtest_metric(dict(validation.get("test") or {})))
        lines.extend(["", "Best Train Strategies With Test Performance"])
        for row in list(validation.get("strategy_comparison") or [])[:10]:
            lines.append(_format_threshold_validation_metric(row))
        validation_warnings = list(validation.get("warnings") or [])
        if validation_warnings:
            lines.extend(["", "Validation Warnings"])
            for warning in validation_warnings[:20]:
                lines.append(f"  {warning}")
    lines.extend(["", "Recommendation: paper/shadow only; do not go live"])
    return "\n".join(lines) + "\n"


def render_transformer_calibration_report(report: dict[str, Any]) -> str:
    probability = dict(report.get("probability_summary") or {})
    lines = [
        "Transformer Calibration Analysis",
        f"generated_at: {report.get('generated_at')}",
        f"rows: {report.get('rows')} markets: {report.get('markets')}",
        f"probability_rows: {report.get('probability_rows')} ready_rows: {report.get('ready_rows')}",
        (
            f"probability_yes min={probability.get('min')} "
            f"avg={probability.get('average')} max={probability.get('max')}"
        ),
        "",
        "Threshold Hit Counts",
    ]
    for threshold, count in dict(report.get("threshold_hit_counts") or {}).items():
        lines.append(f"  {threshold}: {count}")
    lines.extend(["", "Outcome Join"])
    outcome = dict(report.get("outcome_join") or {})
    lines.append(
        f"  available={outcome.get('available')} joined_rows={outcome.get('joined_rows')} "
        f"joined_markets={outcome.get('joined_markets')} unjoined_rows={outcome.get('unjoined_rows')}"
    )
    lines.extend(["", "Threshold Outcome Calibration"])
    for row in list(report.get("threshold_outcome_calibration") or []):
        lines.append(
            f"  {row.get('threshold')}: rows={row.get('rows')} markets={row.get('markets')} "
            f"resolved_rows={row.get('resolved_rows')} actual_yes_rate={row.get('actual_yes_rate')} "
            f"win_rate={row.get('win_rate')}"
        )
    lines.extend(["", "Top Markets By Max Probability"])
    for row in list(report.get("per_market_probability") or [])[:10]:
        lines.append(
            f"  {row.get('market_id')}: rows={row.get('row_count')} "
            f"max={row.get('max_probability_yes')} avg={row.get('average_probability_yes')} "
            f"actual_yes={row.get('actual_yes')}"
        )
    lines.extend(["", "Warnings"])
    for warning in list(report.get("warnings") or []):
        lines.append(f"  {warning}")
    lines.extend(["", "Recommendation: paper/shadow only; no live-trading recommendation"])
    return "\n".join(lines) + "\n"


def apply_scaler_to_sequence(
    sequence_values: Sequence[Sequence[Any]],
    scaler_stats: dict[str, list[float]],
) -> list[list[float]]:
    mean = [float(value) for value in scaler_stats["mean"]]
    std = [
        float(value) if float(value) > 1.0e-8 else 1.0
        for value in scaler_stats["std"]
    ]
    scaled: list[list[float]] = []
    for row in sequence_values:
        scaled_row: list[float] = []
        for index, raw_value in enumerate(row):
            value = _float_or_none(raw_value)
            if value is None:
                value = mean[index]
            scaled_row.append((float(value) - mean[index]) / std[index])
        scaled.append(scaled_row)
    return scaled


def _sequence_readiness_diagnostics(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    total = len(rows)
    ready_false = 0
    gap_true = 0
    bad_samples: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if not _flag_enabled(row.get("feature_ready")):
            ready_false += 1
            reasons.append("feature_ready_0")
        if _flag_enabled(row.get("is_gap_affected")):
            gap_true += 1
            reasons.append("gap_affected_1")
        if reasons and len(bad_samples) < 8:
            bad_samples.append(
                {
                    "timestamp": row.get("timestamp"),
                    "reasons": reasons,
                    "feature_ready": row.get("feature_ready"),
                    "is_gap_affected": row.get("is_gap_affected"),
                    "snapshot_quality_status": row.get("snapshot_quality_status"),
                }
            )
    timestamps = [str(row.get("timestamp")) for row in rows if row.get("timestamp") is not None]
    latest_row = rows[-1] if rows else {}
    return {
        "total_sequence_rows_checked": total,
        "feature_ready_false_count": ready_false,
        "gap_affected_true_count": gap_true,
        "ready_ratio": (total - ready_false) / total if total else 0.0,
        "earliest_sequence_timestamp": timestamps[0] if timestamps else None,
        "latest_sequence_timestamp": timestamps[-1] if timestamps else None,
        "latest_row_feature_ready": latest_row.get("feature_ready"),
        "latest_row_snapshot_quality_status": latest_row.get("snapshot_quality_status"),
        "missing_or_null_counts_top20": _missing_or_null_counts(rows, feature_columns),
        "bad_row_samples": bad_samples,
    }


def _missing_or_null_counts(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    for column in feature_columns:
        name = str(column)
        count = 0
        for row in rows:
            if name not in row or _float_or_none(row.get(name)) is None:
                count += 1
        if count:
            counts.append({"column": name, "missing_or_null_count": count})
    counts.sort(key=lambda item: (-int(item["missing_or_null_count"]), str(item["column"])))
    return counts[:20]


def _latest_active_feature(conn: sqlite3.Connection, *, now: datetime) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT f.run_id, f.market_id, f.timestamp
        FROM features f
        JOIN markets m
          ON m.run_id = f.run_id
         AND m.market_id = f.market_id
        WHERE m.close_time IS NOT NULL
          AND datetime(m.close_time) > datetime(?)
        ORDER BY datetime(f.timestamp) DESC, f.timestamp DESC
        LIMIT 1
        """,
        (to_iso(now),),
    ).fetchone()
    return dict(row) if row is not None else None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    return [str(row[1]) for row in rows]


def _fetch_sequence_rows(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    latest_timestamp: str,
    limit: int,
) -> list[dict[str, Any]]:
    feature_rows = conn.execute(
        """
        SELECT *
        FROM features
        WHERE run_id = ?
          AND market_id = ?
          AND datetime(timestamp) <= datetime(?)
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        LIMIT ?
        """,
        (run_id, market_id, latest_timestamp, int(limit)),
    ).fetchall()
    market = _market_row(conn, run_id=run_id, market_id=market_id)
    enriched: list[dict[str, Any]] = []
    for row in feature_rows:
        item = dict(row)
        _merge_market_fields(item, market)
        snapshot = _snapshot_row(
            conn,
            run_id=run_id,
            market_id=market_id,
            timestamp=str(item.get("timestamp")),
        )
        _merge_snapshot_fields(item, snapshot)
        _add_time_fields(item)
        _add_btc_fields(conn, item, run_id=run_id)
        enriched.append(item)
    return enriched


def _fetch_all_market_rows_through_timestamp(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    latest_timestamp: str,
) -> list[dict[str, Any]]:
    feature_rows = conn.execute(
        """
        SELECT *
        FROM features
        WHERE run_id = ?
          AND market_id = ?
          AND datetime(timestamp) <= datetime(?)
        ORDER BY datetime(timestamp) DESC, timestamp DESC
        """,
        (run_id, market_id, latest_timestamp),
    ).fetchall()
    market = _market_row(conn, run_id=run_id, market_id=market_id)
    enriched: list[dict[str, Any]] = []
    for row in feature_rows:
        item = dict(row)
        _merge_market_fields(item, market)
        snapshot = _snapshot_row(
            conn,
            run_id=run_id,
            market_id=market_id,
            timestamp=str(item.get("timestamp")),
        )
        _merge_snapshot_fields(item, snapshot)
        _add_time_fields(item)
        _add_btc_fields(conn, item, run_id=run_id)
        enriched.append(item)
    return enriched


def _clean_row_rejection_reasons(
    row: dict[str, Any],
    feature_columns: Sequence[str],
) -> tuple[list[str], set[str]]:
    reasons: list[str] = []
    missing_columns: set[str] = set()
    if not _flag_enabled(row.get("feature_ready")):
        reasons.append("not_feature_ready")
    strict_value = None
    if "strict_validation_passed" in row:
        strict_value = row.get("strict_validation_passed")
    elif "snapshot_strict_validation_passed" in row:
        strict_value = row.get("snapshot_strict_validation_passed")
    if strict_value is not None and not _flag_enabled(strict_value):
        reasons.append("strict_validation_failed")
    if "is_gap_affected" in row and _flag_enabled(row.get("is_gap_affected")):
        reasons.append("gap_affected")
    for column in feature_columns:
        name = str(column)
        if name not in row or _float_or_none(row.get(name)) is None:
            missing_columns.add(name)
    if missing_columns:
        reasons.append("missing_or_null_model_feature")
    return reasons, missing_columns


def _market_row(conn: sqlite3.Connection, *, run_id: str, market_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM markets
        WHERE run_id = ?
          AND market_id = ?
        LIMIT 1
        """,
        (run_id, market_id),
    ).fetchone()
    return dict(row) if row is not None else {}


def _snapshot_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    market_id: str,
    timestamp: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM market_snapshots
        WHERE run_id = ?
          AND market_id = ?
          AND timestamp = ?
        LIMIT 1
        """,
        (run_id, market_id, timestamp),
    ).fetchone()
    return dict(row) if row is not None else {}


def _merge_market_fields(item: dict[str, Any], market: dict[str, Any]) -> None:
    aliases = {
        "phase": "market_phase",
    }
    for key, value in market.items():
        if key in {"run_id", "market_id"}:
            continue
        target = aliases.get(str(key), str(key))
        if target not in item:
            item[target] = value


def _merge_snapshot_fields(item: dict[str, Any], snapshot: dict[str, Any]) -> None:
    for key, value in snapshot.items():
        if key in {"run_id", "market_id", "timestamp"}:
            continue
        if key == "strict_validation_passed":
            item.setdefault("snapshot_strict_validation_passed", value)
            continue
        item.setdefault(str(key), value)


def _prediction_context_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_start_time": row.get("start_time"),
        "market_close_time": row.get("close_time"),
        "time_until_resolution": _float_or_none(
            row.get("time_until_resolution")
            if row.get("time_until_resolution") is not None
            else row.get("seconds_before_close")
        ),
        "yes_price": _first_numeric(
            row,
            ("yes_price", "mid_price_yes", "best_bid_yes", "best_ask_yes"),
        ),
        "no_price": _first_numeric(
            row,
            ("no_price", "mid_price_no", "best_bid_no", "best_ask_no"),
        ),
        "best_bid_yes": _float_or_none(row.get("best_bid_yes")),
        "best_ask_yes": _float_or_none(row.get("best_ask_yes")),
        "best_bid_no": _float_or_none(row.get("best_bid_no")),
        "best_ask_no": _float_or_none(row.get("best_ask_no")),
        "mid_price_yes": _float_or_none(row.get("mid_price_yes")),
        "mid_price_no": _float_or_none(row.get("mid_price_no")),
        "spread_yes": _float_or_none(row.get("spread_yes")),
        "spread_no": _float_or_none(row.get("spread_no")),
        "feature_ready": row.get("feature_ready"),
        "snapshot_quality_status": row.get("snapshot_quality_status"),
        "strict_validation_passed": row.get("strict_validation_passed"),
    }


def _add_time_fields(item: dict[str, Any]) -> None:
    timestamp = parse_timestamp(item.get("timestamp"))
    start = parse_timestamp(item.get("start_time"))
    close = parse_timestamp(item.get("close_time"))
    if timestamp is not None and start is not None:
        item["seconds_after_start"] = max(0.0, (timestamp - start).total_seconds())
    if timestamp is not None and close is not None:
        seconds_before_close = (close - timestamp).total_seconds()
        item["seconds_before_close"] = seconds_before_close
        item.setdefault("time_until_resolution", seconds_before_close)


def _add_btc_fields(conn: sqlite3.Connection, item: dict[str, Any], *, run_id: str) -> None:
    timestamp = str(item.get("timestamp") or "")
    chainlink = _latest_btc_at_or_before(
        conn,
        run_id=run_id,
        source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
        timestamp=timestamp,
    )
    binance = _latest_btc_at_or_before(
        conn,
        run_id=run_id,
        source=POLYMARKET_RTDS_BINANCE_SOURCE,
        timestamp=timestamp,
    )
    _assign_btc(item, "btc_chainlink", chainlink, feature_timestamp=timestamp)
    _assign_btc(item, "btc_binance", binance, feature_timestamp=timestamp)
    chain_price = _float_or_none(item.get("btc_chainlink_price"))
    binance_price = _float_or_none(item.get("btc_binance_price"))
    if chain_price is not None and binance_price is not None:
        item["btc_price_diff_binance_minus_chainlink"] = binance_price - chain_price
        item["btc_price_diff_pct_binance_minus_chainlink"] = (
            (binance_price - chain_price) / chain_price if chain_price else None
        )
    start_time = item.get("start_time")
    start_price = None
    start_arrival = None
    if start_time:
        start_sample = _latest_btc_at_or_before(
            conn,
            run_id=run_id,
            source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
            timestamp=str(start_time),
        )
        if start_sample is None:
            start_sample = _earliest_btc_after(
                conn,
                run_id=run_id,
                source=POLYMARKET_RTDS_CHAINLINK_SOURCE,
                timestamp=str(start_time),
            )
        if start_sample is not None:
            start_price = start_sample.get("price")
            start_arrival = start_sample.get("local_arrival_iso")
    item["btc_price_at_market_start"] = start_price
    item["btc_price_at_market_start_local_arrival_iso"] = start_arrival


def _assign_btc(
    item: dict[str, Any],
    prefix: str,
    sample: dict[str, Any] | None,
    *,
    feature_timestamp: str,
) -> None:
    item[f"{prefix}_price"] = sample.get("price") if sample else None
    item[f"{prefix}_exchange_timestamp"] = sample.get("exchange_timestamp") if sample else None
    arrival = sample.get("local_arrival_iso") if sample else None
    item[f"{prefix}_local_arrival_iso"] = arrival
    item[f"{prefix}_age_sec_at_feature"] = _seconds_between(feature_timestamp, arrival)


def _latest_btc_at_or_before(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    source: str,
    timestamp: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM btc_prices
        WHERE run_id = ?
          AND source = ?
          AND datetime(local_arrival_iso) <= datetime(?)
        ORDER BY local_arrival_ns DESC, id DESC
        LIMIT 1
        """,
        (run_id, source, timestamp),
    ).fetchone()
    return dict(row) if row is not None else None


def _earliest_btc_after(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    source: str,
    timestamp: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM btc_prices
        WHERE run_id = ?
          AND source = ?
          AND datetime(local_arrival_iso) > datetime(?)
        ORDER BY local_arrival_ns ASC, id ASC
        LIMIT 1
        """,
        (run_id, source, timestamp),
    ).fetchone()
    return dict(row) if row is not None else None


def _seconds_between(end_value: Any, start_value: Any) -> float | None:
    end = parse_timestamp(end_value)
    start = parse_timestamp(start_value)
    if end is None or start is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return int(value) == 1
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_numeric(row: dict[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _round(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, 10) if parsed is not None else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _json_dumps(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return json.dumps(value, sort_keys=True)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _open_readonly_prediction_db(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _prediction_latency_sec(row: dict[str, Any]) -> float | None:
    prediction_time = parse_timestamp(row.get("timestamp") or row.get("created_at"))
    feature_time = parse_timestamp(
        row.get("feature_timestamp")
        or row.get("latest_feature_timestamp")
        or row.get("signal_timestamp")
    )
    if prediction_time is None or feature_time is None:
        return None
    return max(0.0, (prediction_time - feature_time).total_seconds())


def _timestamp_value(row: dict[str, Any]) -> str | None:
    value = row.get("timestamp") or row.get("created_at")
    return str(value) if value not in (None, "") else None


def _min_timestamp(rows: Sequence[dict[str, Any]]) -> str | None:
    values = [value for value in (_timestamp_value(row) for row in rows) if value is not None]
    return min(values) if values else None


def _max_timestamp(rows: Sequence[dict[str, Any]]) -> str | None:
    values = [value for value in (_timestamp_value(row) for row in rows) if value is not None]
    return max(values) if values else None


def _probability_buckets(probabilities: Sequence[float]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        if index == 9:
            count = sum(1 for value in probabilities if low <= value <= high)
        else:
            count = sum(1 for value in probabilities if low <= value < high)
        buckets.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "count": count,
            }
        )
    return buckets


def _reason_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if _flag_enabled(row.get("transformer_sequence_ready")) and row.get("probability_yes") is not None:
            continue
        reason = str(row.get("reason") or row.get("diagnostics_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_rejection_reasons(rows: Sequence[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    counts = _reason_counts(rows)
    return [
        {"reason": reason, "count": count}
        for reason, count in list(counts.items())[: int(limit)]
    ]


def _market_count(rows: Sequence[dict[str, Any]]) -> int:
    return len(
        {
            str(row.get("market_id") or row.get("latest_market_id") or "")
            for row in rows
            if row.get("market_id") not in (None, "")
            or row.get("latest_market_id") not in (None, "")
        }
    )


def _top_markets_by_max_probability(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        probability = _float_or_none(row.get("probability_yes"))
        market_id = str(row.get("market_id") or row.get("latest_market_id") or "")
        if probability is None or not market_id:
            continue
        grouped.setdefault(market_id, []).append(row)
    result: list[dict[str, Any]] = []
    for market_id, market_rows in grouped.items():
        probabilities = [
            value
            for value in (_float_or_none(row.get("probability_yes")) for row in market_rows)
            if value is not None
        ]
        timestamps = [
            value
            for value in (_timestamp_value(row) for row in market_rows)
            if value is not None
        ]
        result.append(
            {
                "market_id": market_id,
                "row_count": len(market_rows),
                "max_probability_yes": round(max(probabilities), 10),
                "average_probability_yes": round(sum(probabilities) / len(probabilities), 10),
                "min_probability_yes": round(min(probabilities), 10),
                "latest_timestamp": max(timestamps) if timestamps else None,
            }
        )
    result.sort(
        key=lambda item: (
            float(item.get("max_probability_yes") or -1.0),
            int(item.get("row_count") or 0),
            str(item.get("market_id") or ""),
        ),
        reverse=True,
    )
    return result[: int(limit)]


def _load_prediction_db_rows(
    prediction_db_paths: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    input_dbs: list[dict[str, Any]] = []
    for path in prediction_db_paths:
        db_path = Path(path)
        health: dict[str, Any] = {"path": str(path)}
        if not db_path.exists():
            health.update({"status": "missing", "rows": 0})
            input_dbs.append(health)
            continue
        conn = _open_readonly_prediction_db(str(db_path))
        try:
            if not _table_exists(conn, "transformer_predictions"):
                health.update({"status": "missing_table", "rows": 0})
                input_dbs.append(health)
                continue
            db_rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM transformer_predictions ORDER BY id ASC"
                ).fetchall()
            ]
            for row in db_rows:
                row["source_prediction_db"] = str(path)
            rows.extend(db_rows)
            health.update(
                {
                    "status": "ok",
                    "rows": len(db_rows),
                    "columns": _table_columns(conn, "transformer_predictions"),
                }
            )
        finally:
            conn.close()
        input_dbs.append(health)
    return rows, input_dbs


def _load_jsonl_prediction_rows(
    jsonl_log_paths: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    for path in jsonl_log_paths:
        log_path = Path(path)
        health: dict[str, Any] = {"path": str(path)}
        if not log_path.exists():
            health.update({"status": "missing", "rows": 0, "parse_errors": 0})
            health_rows.append(health)
            continue
        parse_errors = 0
        loaded = 0
        with log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                row = _prediction_row_from_jsonl(payload)
                row["source_jsonl_log"] = str(path)
                row["source_jsonl_line"] = line_number
                rows.append(row)
                loaded += 1
        health.update({"status": "ok", "rows": loaded, "parse_errors": parse_errors})
        health_rows.append(health)
    return rows, health_rows


def _prediction_row_from_jsonl(payload: dict[str, Any]) -> dict[str, Any]:
    probability_yes = _float_or_none(payload.get("probability_yes"))
    return {
        "created_at": payload.get("timestamp"),
        "timestamp": payload.get("timestamp"),
        "run_id": str(payload.get("run_id") or ""),
        "market_id": str(payload.get("market_id") or payload.get("latest_market_id") or ""),
        "model_path": payload.get("model_path"),
        "sequence_length": _int_or_none(payload.get("sequence_length")),
        "sequence_row_policy": payload.get("sequence_row_policy"),
        "transformer_sequence_ready": 1 if _flag_enabled(payload.get("transformer_sequence_ready")) else 0,
        "probability_yes": probability_yes,
        "probability_no": 1.0 - probability_yes if probability_yes is not None else None,
        "feature_timestamp": payload.get("feature_timestamp") or payload.get("latest_feature_timestamp"),
        "latest_feature_timestamp": payload.get("latest_feature_timestamp") or payload.get("feature_timestamp"),
        "latest_sequence_timestamp": payload.get("latest_sequence_timestamp"),
        "market_start_time": payload.get("market_start_time"),
        "market_close_time": payload.get("market_close_time"),
        "time_until_resolution": _float_or_none(payload.get("time_until_resolution")),
        "yes_price": _float_or_none(payload.get("yes_price")),
        "no_price": _float_or_none(payload.get("no_price")),
        "mid_price_yes": _float_or_none(payload.get("mid_price_yes")),
        "mid_price_no": _float_or_none(payload.get("mid_price_no")),
        "spread_yes": _float_or_none(payload.get("spread_yes")),
        "spread_no": _float_or_none(payload.get("spread_no")),
        "snapshot_quality_status": payload.get("snapshot_quality_status"),
        "ready_ratio": _float_or_none(payload.get("ready_ratio")),
        "sequence_rows_available": _int_or_none(payload.get("sequence_rows_available")),
        "reason": payload.get("reason"),
        "diagnostics_reason": payload.get("reason"),
        "missing_feature_columns_json": _json_dumps(payload.get("missing_feature_columns")),
        "rejected_rows_by_reason_json": _json_dumps(payload.get("rejected_rows_by_reason")),
    }


def _simulate_transformer_threshold_trades(
    recorder: sqlite3.Connection,
    predictions: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    min_time_options: Sequence[float],
    max_time_options: Sequence[float],
    horizon_options: Sequence[int],
    sides: Sequence[str],
    one_trade_per_market: bool,
    slippage_cents: float,
    fee_cents: float,
) -> list[dict[str, Any]]:
    outcomes = _load_recorder_market_outcomes_from_conn(recorder)
    simulated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    sorted_predictions = sorted(
        predictions,
        key=lambda row: (
            str(row.get("market_id") or ""),
            _prediction_signal_sort_key(row),
            int(row.get("id") or 0),
        ),
    )
    for prediction in sorted_predictions:
        probability_yes = _float_or_none(prediction.get("probability_yes"))
        if probability_yes is None:
            continue
        time_until = _float_or_none(prediction.get("time_until_resolution"))
        if time_until is None:
            continue
        market_id = str(prediction.get("market_id") or "")
        run_id = str(prediction.get("run_id") or "")
        path = _load_threshold_price_path(recorder, prediction)
        for threshold in thresholds:
            for min_time in min_time_options:
                for max_time in max_time_options:
                    if float(max_time) < float(min_time):
                        continue
                    if time_until < float(min_time) or time_until > float(max_time):
                        continue
                    for horizon in horizon_options:
                        for side in sides:
                            probability_for_side = (
                                probability_yes if side == "YES" else 1.0 - probability_yes
                            )
                            if probability_for_side < float(threshold):
                                continue
                            strategy_id = _threshold_strategy_id(
                                side=side,
                                threshold=float(threshold),
                                min_time=float(min_time),
                                max_time=float(max_time),
                                horizon=int(horizon),
                            )
                            dedupe_key = (
                                strategy_id,
                                run_id,
                                market_id,
                                side,
                            )
                            if one_trade_per_market and dedupe_key in seen:
                                continue
                            row = _simulate_threshold_trade(
                                prediction,
                                path,
                                outcomes,
                                strategy_id=strategy_id,
                                side=side,
                                threshold=float(threshold),
                                probability_for_side=float(probability_for_side),
                                min_time=float(min_time),
                                max_time=float(max_time),
                                horizon=int(horizon),
                                slippage_cents=float(slippage_cents),
                                fee_cents=float(fee_cents),
                            )
                            simulated.append(row)
                            if one_trade_per_market:
                                seen.add(dedupe_key)
    return simulated


def _strategy_filter_values(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _filter_threshold_strategy_rows(
    rows: Sequence[dict[str, Any]],
    strategy_filters: set[str],
) -> list[dict[str, Any]]:
    if not strategy_filters:
        return list(rows)
    return [
        row for row in rows
        if str(row.get("strategy_id") or "") in strategy_filters
    ]


def _split_predictions_by_market_time(
    predictions: Sequence[dict[str, Any]],
    *,
    train_ratio: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        key = _prediction_market_split_key(row)
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    ordered_markets: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        timestamps = [
            timestamp
            for timestamp in (_prediction_signal_time(row) for row in rows)
            if timestamp is not None
        ]
        first_timestamp = min(timestamps) if timestamps else None
        ordered_markets.append(
            {
                "market_key": key,
                "market_id": str(rows[0].get("market_id") or ""),
                "first_prediction_timestamp": (
                    first_timestamp.isoformat() if first_timestamp is not None else None
                ),
                "rows": rows,
            }
        )
    ordered_markets.sort(
        key=lambda item: (
            item.get("first_prediction_timestamp") or "",
            str(item.get("market_key") or ""),
        )
    )
    market_count = len(ordered_markets)
    if market_count <= 1:
        train_count = market_count
    else:
        ratio = min(max(float(train_ratio), 0.0), 1.0)
        train_count = int(market_count * ratio)
        train_count = min(max(train_count, 1), market_count - 1)
    train_markets = ordered_markets[:train_count]
    test_markets = ordered_markets[train_count:]
    return {
        "train_predictions": [
            row for market in train_markets for row in list(market.get("rows") or [])
        ],
        "test_predictions": [
            row for market in test_markets for row in list(market.get("rows") or [])
        ],
        "split": {
            "train_ratio": float(train_ratio),
            "market_count": market_count,
            "train_market_count": len(train_markets),
            "test_market_count": len(test_markets),
            "train_prediction_rows": sum(len(market.get("rows") or []) for market in train_markets),
            "test_prediction_rows": sum(len(market.get("rows") or []) for market in test_markets),
            "train_market_ids": [market.get("market_id") for market in train_markets],
            "test_market_ids": [market.get("market_id") for market in test_markets],
            "train_market_keys": [market.get("market_key") for market in train_markets],
            "test_market_keys": [market.get("market_key") for market in test_markets],
        },
    }


def _prediction_market_split_key(row: dict[str, Any]) -> str:
    market_id = str(row.get("market_id") or row.get("latest_market_id") or "")
    if not market_id:
        return ""
    run_id = str(row.get("run_id") or "")
    return f"{run_id}|{market_id}" if run_id else market_id


def _threshold_backtest_validation_report(
    *,
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    split: dict[str, Any],
    min_done_trades: int,
    min_test_done_trades: int,
) -> dict[str, Any]:
    train_by_strategy = {
        str(row.get("strategy_id") or ""): row
        for row in _comparison_metrics(train_rows, fields=("strategy_id",), min_done_trades=0)
    }
    test_by_strategy = {
        str(row.get("strategy_id") or ""): row
        for row in _comparison_metrics(test_rows, fields=("strategy_id",), min_done_trades=0)
    }
    comparison: list[dict[str, Any]] = []
    warnings: list[str] = []
    for strategy_id, train_metrics in train_by_strategy.items():
        test_metrics = test_by_strategy.get(strategy_id) or _empty_threshold_backtest_metrics()
        delta = _threshold_metric_delta(train_metrics, test_metrics)
        row_warnings: list[str] = []
        if (
            _float_or_none(train_metrics.get("pnl")) is not None
            and _float_or_none(test_metrics.get("pnl")) is not None
            and float(train_metrics.get("pnl") or 0.0) > 0
            and float(test_metrics.get("pnl") or 0.0) < 0
        ):
            warning = f"train_positive_test_negative:{strategy_id}"
            row_warnings.append(warning)
            warnings.append(warning)
        if int(test_metrics.get("done_trades") or 0) < int(min_test_done_trades):
            warning = f"test_sample_too_small:{strategy_id}"
            row_warnings.append(warning)
            warnings.append(warning)
        if int(train_metrics.get("done_trades") or 0) < int(min_done_trades):
            row_warnings.append(f"train_sample_below_min_done:{strategy_id}")
        comparison.append(
            {
                "strategy_id": strategy_id,
                "train": train_metrics,
                "test": test_metrics,
                "delta": delta,
                "warnings": row_warnings,
            }
        )
    comparison.sort(
        key=lambda item: (
            float(dict(item.get("train") or {}).get("pnl") or 0.0),
            float(dict(item.get("train") or {}).get("average_roi") or -999.0),
            int(dict(item.get("train") or {}).get("done_trades") or 0),
            str(item.get("strategy_id") or ""),
        ),
        reverse=True,
    )
    split_info = dict(split.get("split") or {})
    overlap = sorted(
        set(split_info.get("train_market_keys") or [])
        & set(split_info.get("test_market_keys") or [])
    )
    if overlap:
        warnings.append("market_split_overlap_detected")
    if not test_rows:
        warnings.append("empty_test_split")
    return {
        "split_by_time": True,
        "split": split_info,
        "train": _threshold_backtest_metrics(train_rows),
        "test": _threshold_backtest_metrics(test_rows),
        "train_strategy_count": len(train_by_strategy),
        "test_strategy_count": len(test_by_strategy),
        "strategy_comparison": comparison,
        "warnings": sorted(set(warnings)),
    }


def _empty_threshold_backtest_metrics() -> dict[str, Any]:
    return {
        "trades": 0,
        "done_trades": 0,
        "pnl": 0.0,
        "roi": None,
        "average_roi": None,
        "win_rate": None,
        "average_win": None,
        "average_loss": None,
        "payoff_ratio": None,
        "breakeven_win_rate": None,
        "best_trade": None,
        "worst_trade": None,
        "markets_covered": 0,
    }


def _threshold_metric_delta(
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("pnl", "average_roi", "win_rate"):
        train_value = _float_or_none(train_metrics.get(field))
        test_value = _float_or_none(test_metrics.get(field))
        result[field] = (
            _round(test_value - train_value)
            if train_value is not None and test_value is not None
            else None
        )
    result["done_trades"] = int(test_metrics.get("done_trades") or 0) - int(
        train_metrics.get("done_trades") or 0
    )
    return result


def _simulate_threshold_trade(
    prediction: dict[str, Any],
    path: Sequence[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    strategy_id: str,
    side: str,
    threshold: float,
    probability_for_side: float,
    min_time: float,
    max_time: float,
    horizon: int,
    slippage_cents: float,
    fee_cents: float,
) -> dict[str, Any]:
    price_penalty = (float(slippage_cents) + float(fee_cents)) / 100.0
    entry_price = _threshold_entry_price(prediction, path, side)
    adjusted_entry = None if entry_price is None else min(1.0, float(entry_price) + price_penalty)
    exit_point, hit_reason = _threshold_exit_point(prediction, path, horizon_sec=horizon)
    exit_price = _threshold_exit_price(exit_point or {}, side)
    adjusted_exit = None if exit_price is None else max(0.0, float(exit_price) - price_penalty)
    pnl, roi = _threshold_pnl(adjusted_entry, adjusted_exit)
    resolved_label = None
    if pnl is None:
        outcome = _prediction_outcome(prediction, outcomes)
        if outcome is not None and adjusted_entry is not None and 0 < adjusted_entry < 1:
            resolved_label = "YES" if outcome.get("actual_yes") == 1 else "NO"
            if resolved_label == side:
                pnl = 1.0 / adjusted_entry - 1.0
            else:
                pnl = -1.0
            roi = pnl
            hit_reason = "settled_resolution"
    signal_timestamp = (
        prediction.get("signal_timestamp")
        or prediction.get("latest_feature_timestamp")
        or prediction.get("feature_timestamp")
        or prediction.get("timestamp")
    )
    market_id = str(prediction.get("market_id") or "")
    run_id = str(prediction.get("run_id") or "")
    probability_yes = _float_or_none(prediction.get("probability_yes"))
    return {
        "strategy_id": strategy_id,
        "prediction_id": prediction.get("id"),
        "run_id": run_id,
        "market_id": market_id,
        "signal_timestamp": signal_timestamp,
        "side": side,
        "threshold": round(float(threshold), 6),
        "probability_yes": _round(probability_yes),
        "probability_for_side": _round(probability_for_side),
        "min_time_until_resolution_sec": _round(min_time),
        "max_time_until_resolution_sec": _round(max_time),
        "exit_horizon_sec": int(horizon),
        "time_until_resolution": _round(prediction.get("time_until_resolution")),
        "entry_price": _round(entry_price),
        "adjusted_entry_price": _round(adjusted_entry),
        "exit_price": _round(exit_price),
        "adjusted_exit_price": _round(adjusted_exit),
        "pnl": _round(pnl),
        "roi": _round(roi),
        "done": pnl is not None,
        "win": bool(pnl is not None and pnl > 0),
        "hit_reason": hit_reason,
        "resolved_label": resolved_label,
        "path_rows": len(path),
        "snapshot_quality_status": prediction.get("snapshot_quality_status"),
    }


def _load_threshold_price_path(
    recorder: sqlite3.Connection,
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _table_exists(recorder, "market_snapshots"):
        return []
    market_id = str(prediction.get("market_id") or "")
    run_id = str(prediction.get("run_id") or "")
    signal_ts = _prediction_signal_time(prediction)
    close_ts = parse_timestamp(prediction.get("market_close_time"))
    if not market_id or signal_ts is None:
        return []
    where = ["market_id = ?", "datetime(timestamp) >= datetime(?)"]
    params: list[Any] = [market_id, signal_ts.isoformat()]
    if run_id:
        where.insert(0, "run_id = ?")
        params.insert(0, run_id)
    if close_ts is not None:
        where.append("datetime(timestamp) <= datetime(?)")
        params.append(close_ts.isoformat())
    rows = recorder.execute(
        f"""
        SELECT *
        FROM market_snapshots
        WHERE {' AND '.join(where)}
        ORDER BY datetime(timestamp) ASC, timestamp ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _threshold_exit_point(
    prediction: dict[str, Any],
    path: Sequence[dict[str, Any]],
    *,
    horizon_sec: int,
) -> tuple[dict[str, Any] | None, str]:
    if not path:
        return None, "no_price_path"
    signal_ts = _prediction_signal_time(prediction)
    if signal_ts is None:
        return None, "invalid_signal_timestamp"
    target = signal_ts + timedelta(seconds=int(horizon_sec))
    for point in path:
        timestamp = parse_timestamp(point.get("timestamp"))
        if timestamp is not None and timestamp >= target:
            return point, "fixed_horizon"
    return None, "no_fixed_horizon_snapshot"


def _threshold_entry_price(
    prediction: dict[str, Any],
    path: Sequence[dict[str, Any]],
    side: str,
) -> float | None:
    fields = (
        ("best_ask_yes", "yes_price", "mid_price_yes", "last_trade_price")
        if side == "YES"
        else ("best_ask_no", "no_price", "mid_price_no", "last_trade_price")
    )
    price = _first_numeric(prediction, fields)
    if price is not None:
        return price
    return _first_numeric(path[0], fields) if path else None


def _threshold_exit_price(row: dict[str, Any], side: str) -> float | None:
    fields = (
        ("best_bid_yes", "mid_price_yes", "last_trade_price")
        if side == "YES"
        else ("best_bid_no", "mid_price_no", "last_trade_price")
    )
    return _first_numeric(row, fields)


def _threshold_pnl(
    adjusted_entry: float | None,
    adjusted_exit: float | None,
) -> tuple[float | None, float | None]:
    if adjusted_entry is None or adjusted_exit is None:
        return None, None
    if adjusted_entry <= 0 or adjusted_entry >= 1:
        return None, None
    pnl = 1.0 / adjusted_entry * adjusted_exit - 1.0
    return pnl, pnl


def _load_recorder_market_outcomes_from_conn(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "markets"):
        return {}
    columns = set(_table_columns(conn, "markets"))
    selected = [
        column
        for column in (
            "run_id",
            "market_id",
            "resolved",
            "winning_outcome",
            "winning_asset_id",
            "yes_token_id",
            "no_token_id",
            "close_time",
        )
        if column in columns
    ]
    if "market_id" not in selected:
        return {}
    rows = conn.execute(f"SELECT {', '.join(selected)} FROM markets").fetchall()
    outcomes: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        actual_yes = _actual_yes_from_market(item)
        if actual_yes is None:
            continue
        for key in _market_keys(item):
            outcomes[key] = {**item, "actual_yes": actual_yes}
    return outcomes


def _prediction_outcome(
    prediction: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in _market_keys(prediction):
        if key in outcomes:
            return outcomes[key]
    return None


def _backtest_sides(value: str) -> list[str]:
    side = str(value or "BOTH").upper()
    if side == "YES":
        return ["YES"]
    if side == "NO":
        return ["NO"]
    if side == "BOTH":
        return ["YES", "NO"]
    raise TransformerLiveInferenceError("side must be YES, NO, or BOTH")


def _threshold_strategy_id(
    *,
    side: str,
    threshold: float,
    min_time: float,
    max_time: float,
    horizon: int,
) -> str:
    threshold_label = str(round(float(threshold), 6)).replace(".", "")
    min_label = str(int(float(min_time)))
    max_label = str(int(float(max_time)))
    return f"transformer_{side.lower()}_t{threshold_label}_{min_label}_{max_label}_fh{int(horizon)}"


def _prediction_signal_time(prediction: dict[str, Any]) -> datetime | None:
    return parse_timestamp(
        prediction.get("signal_timestamp")
        or prediction.get("latest_feature_timestamp")
        or prediction.get("feature_timestamp")
        or prediction.get("timestamp")
        or prediction.get("created_at")
    )


def _prediction_signal_sort_key(prediction: dict[str, Any]) -> str:
    timestamp = _prediction_signal_time(prediction)
    return timestamp.isoformat() if timestamp is not None else ""


def _strategy_metric_rows(
    rows: Sequence[dict[str, Any]],
    *,
    min_done_trades: int,
    sort_by: str = "pnl",
) -> list[dict[str, Any]]:
    metrics = _comparison_metrics(rows, fields=("strategy_id",), min_done_trades=min_done_trades)
    if sort_by == "average_roi":
        metrics.sort(
            key=lambda row: (
                float(row.get("average_roi") or -999.0),
                float(row.get("pnl") or 0.0),
                int(row.get("done_trades") or 0),
                str(row.get("strategy_id") or ""),
            ),
            reverse=True,
        )
    return metrics


def _comparison_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    fields: Sequence[str],
    min_done_trades: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        metrics = _threshold_backtest_metrics(group_rows)
        if int(metrics.get("done_trades") or 0) < int(min_done_trades):
            continue
        item = {field: value for field, value in zip(fields, key)}
        item.update(metrics)
        result.append(item)
    result.sort(
        key=lambda row: (
            float(row.get("pnl") or 0.0),
            float(row.get("average_roi") or -999.0),
            int(row.get("done_trades") or 0),
            json.dumps({field: row.get(field) for field in fields}, sort_keys=True),
        ),
        reverse=True,
    )
    return result


def _threshold_backtest_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if row.get("done")]
    pnls = [
        value for value in (_float_or_none(row.get("pnl")) for row in done)
        if value is not None
    ]
    rois = [
        value for value in (_float_or_none(row.get("roi")) for row in done)
        if value is not None
    ]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    payoff = (
        _round(float(avg_win) / abs(float(avg_loss)))
        if avg_win is not None and avg_loss not in (None, 0)
        else None
    )
    return {
        "trades": len(rows),
        "done_trades": len(done),
        "pnl": _round(sum(pnls)) if pnls else 0.0,
        "roi": _round(sum(pnls) / len(done)) if done else None,
        "average_roi": _round(_mean(rois)),
        "win_rate": _round(len(wins) / len(pnls)) if pnls else None,
        "average_win": _round(avg_win),
        "average_loss": _round(avg_loss),
        "payoff_ratio": payoff,
        "breakeven_win_rate": (
            _round(1.0 / (1.0 + float(payoff))) if payoff not in (None, 0) else None
        ),
        "best_trade": _round(max(pnls)) if pnls else None,
        "worst_trade": _round(min(pnls)) if pnls else None,
        "markets_covered": _market_count(done),
    }


def _format_threshold_backtest_metric(
    row: dict[str, Any],
    *,
    include_strategy: bool = True,
) -> str:
    label_parts: list[str] = []
    if include_strategy and row.get("strategy_id") is not None:
        label_parts.append(str(row.get("strategy_id")))
    for field in (
        "side",
        "threshold",
        "min_time_until_resolution_sec",
        "max_time_until_resolution_sec",
        "exit_horizon_sec",
    ):
        if row.get(field) is not None:
            label_parts.append(f"{field}={row.get(field)}")
    label = " ".join(label_parts) if label_parts else "group"
    return (
        f"  {label}: trades={row.get('trades')} done={row.get('done_trades')} "
        f"markets={row.get('markets_covered')} pnl={row.get('pnl')} "
        f"avg_roi={row.get('average_roi')} win_rate={row.get('win_rate')} "
        f"avg_win={row.get('average_win')} avg_loss={row.get('average_loss')} "
        f"payoff={row.get('payoff_ratio')} worst={row.get('worst_trade')}"
    )


def _format_threshold_validation_metric(row: dict[str, Any]) -> str:
    train = dict(row.get("train") or {})
    test = dict(row.get("test") or {})
    warnings = list(row.get("warnings") or [])
    warning_text = f" warnings={','.join(warnings[:3])}" if warnings else ""
    return (
        f"  {row.get('strategy_id')}: "
        f"train_done={train.get('done_trades')} train_pnl={train.get('pnl')} "
        f"train_avg_roi={train.get('average_roi')} train_win_rate={train.get('win_rate')} | "
        f"test_done={test.get('done_trades')} test_pnl={test.get('pnl')} "
        f"test_avg_roi={test.get('average_roi')} test_win_rate={test.get('win_rate')}"
        f"{warning_text}"
    )


def _load_recorder_market_outcomes(recorder_db_path: str | None) -> dict[str, dict[str, Any]]:
    if not recorder_db_path:
        return {}
    db_path = Path(recorder_db_path)
    if not db_path.exists():
        return {}
    conn = _open_readonly_prediction_db(str(db_path))
    try:
        if not _table_exists(conn, "markets"):
            return {}
        columns = set(_table_columns(conn, "markets"))
        selected = [
            column
            for column in (
                "run_id",
                "market_id",
                "resolved",
                "winning_outcome",
                "winning_asset_id",
                "yes_token_id",
                "no_token_id",
                "close_time",
            )
            if column in columns
        ]
        if "market_id" not in selected:
            return {}
        rows = conn.execute(f"SELECT {', '.join(selected)} FROM markets").fetchall()
    finally:
        conn.close()
    outcomes: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        actual_yes = _actual_yes_from_market(item)
        if actual_yes is None:
            continue
        for key in _market_keys(item):
            outcomes[key] = {**item, "actual_yes": actual_yes}
    return outcomes


def _actual_yes_from_market(row: dict[str, Any]) -> int | None:
    resolved = row.get("resolved")
    if resolved not in (None, "") and not _flag_enabled(resolved):
        return None
    outcome = str(row.get("winning_outcome") or "").strip().lower()
    if outcome in {"yes", "up", "long", "true"}:
        return 1
    if outcome in {"no", "down", "short", "false"}:
        return 0
    winning_asset_id = str(row.get("winning_asset_id") or "")
    if winning_asset_id and winning_asset_id == str(row.get("yes_token_id") or ""):
        return 1
    if winning_asset_id and winning_asset_id == str(row.get("no_token_id") or ""):
        return 0
    return None


def _market_keys(row: dict[str, Any]) -> list[str]:
    run_id = str(row.get("run_id") or "")
    market_id = str(row.get("market_id") or "")
    keys = []
    if market_id:
        keys.append(market_id)
    if run_id or market_id:
        keys.append(f"{run_id}|{market_id}")
    return keys


def _outcome_for_prediction(
    row: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    run_id = str(row.get("run_id") or "")
    market_id = str(row.get("market_id") or row.get("latest_market_id") or "")
    for key in (f"{run_id}|{market_id}", market_id):
        if key in outcomes:
            return outcomes[key]
    return None


def _join_transformer_prediction_outcomes(
    rows: Sequence[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    joined_rows = [
        row for row in rows if _outcome_for_prediction(row, outcomes) is not None
    ]
    joined_markets = {
        str(row.get("market_id") or row.get("latest_market_id") or "")
        for row in joined_rows
        if row.get("market_id") not in (None, "")
        or row.get("latest_market_id") not in (None, "")
    }
    return {
        "joined_rows": len(joined_rows),
        "joined_markets": len(joined_markets),
        "unjoined_rows": len(rows) - len(joined_rows),
    }


def _threshold_outcome_calibration(
    rows: Sequence[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in thresholds:
        hits = [
            row
            for row in rows
            if (_float_or_none(row.get("probability_yes")) or -1.0) >= float(threshold)
        ]
        resolved = [
            (row, outcome)
            for row in hits
            if (outcome := _outcome_for_prediction(row, outcomes)) is not None
        ]
        wins = sum(1 for _row, outcome in resolved if int(outcome["actual_yes"]) == 1)
        losses = sum(1 for _row, outcome in resolved if int(outcome["actual_yes"]) == 0)
        markets = {
            str(row.get("market_id") or row.get("latest_market_id") or "")
            for row in hits
            if row.get("market_id") not in (None, "")
            or row.get("latest_market_id") not in (None, "")
        }
        result.append(
            {
                "threshold": _threshold_key(threshold),
                "rows": len(hits),
                "markets": len(markets),
                "resolved_rows": len(resolved),
                "wins_if_yes": wins,
                "losses_if_yes": losses,
                "actual_yes_rate": round(wins / len(resolved), 10) if resolved else None,
                "win_rate": round(wins / len(resolved), 10) if resolved else None,
            }
        )
    return result


def _per_market_probability(
    rows: Sequence[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base = _top_markets_by_max_probability(rows, limit=10_000)
    return [_attach_market_outcome_stub(row, outcomes) for row in base]


def _attach_market_outcome_stub(row: dict[str, Any], outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    outcome = outcomes.get(str(row.get("market_id") or ""))
    if outcome is None:
        return {**row, "actual_yes": None}
    return {**row, "actual_yes": outcome.get("actual_yes")}


def _transformer_calibration_warnings(
    *,
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    outcomes: dict[str, dict[str, Any]],
    joined_rows: int,
) -> list[str]:
    warnings = ["paper/shadow only; no live-trading recommendation"]
    if not outcomes:
        warnings.append("outcome join unavailable")
    elif joined_rows == 0:
        warnings.append("no prediction rows joined to recorder outcomes")
    if len(probabilities) < 100:
        warnings.append("sample is too small")
    if _market_count(rows) < 30:
        warnings.append("market sample is too small")
    return warnings


def _threshold_key(value: float) -> str:
    return f"gte_{float(value):.2f}".replace(".", "_")


def _open_readonly_sqlite(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise TransformerLiveInferenceError(f"recorder DB not found: {path}")
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _read_json_list(path: Path, *, label: str) -> list[Any]:
    if not path.exists():
        raise TransformerLiveInferenceError(f"{label} file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TransformerLiveInferenceError(f"{label} must be a JSON list: {path}")
    return payload


def _read_json_dict(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise TransformerLiveInferenceError(f"{label} file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TransformerLiveInferenceError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_scaler(
    scaler: dict[str, Any],
    *,
    feature_count: int,
    path: str,
) -> None:
    mean = scaler.get("mean")
    std = scaler.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise TransformerLiveInferenceError(f"scaler_stats must contain mean/std arrays: {path}")
    if len(mean) != feature_count or len(std) != feature_count:
        raise TransformerLiveInferenceError(
            f"scaler_stats length does not match feature_columns: {path}"
        )


def _effective_sequence_length(
    *,
    requested_sequence_length: int,
    training_config: dict[str, Any],
) -> int:
    value = training_config.get("sequence_length") or requested_sequence_length
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise TransformerLiveInferenceError("sequence_length must be an integer") from exc
    if length <= 0:
        raise TransformerLiveInferenceError("sequence_length must be positive")
    return length


def _model_config_from_training_config(
    training_config: dict[str, Any],
    *,
    feature_count: int,
) -> dict[str, Any]:
    return {
        "feature_count": feature_count,
        "d_model": int(training_config.get("d_model") or 32),
        "nhead": int(training_config.get("nhead") or 4),
        "num_layers": int(training_config.get("num_layers") or 1),
        "dim_feedforward": int(training_config.get("dim_feedforward") or 64),
        "dropout": float(training_config.get("dropout") or 0.0),
    }


def _emit(enabled: bool, event: str, **payload: Any) -> None:
    if not enabled:
        return
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)
