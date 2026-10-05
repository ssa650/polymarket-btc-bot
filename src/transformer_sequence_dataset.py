from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .train_baseline_model import select_feature_columns


class TransformerSequenceExportError(RuntimeError):
    pass


def export_transformer_sequence_dataset(
    *,
    input_path: str,
    output_path: str,
    sequence_length: int = 120,
    stride_sec: float = 1.0,
    min_time_until_resolution_sec: float | None = 0.0,
    max_time_until_resolution_sec: float | None = 240.0,
    label_column: str = "label_yes_win",
    market_id_column: str = "market_id",
    timestamp_column: str = "timestamp",
    include_not_ready: bool = False,
    keep_gap_affected: bool = False,
) -> dict[str, Any]:
    if int(sequence_length) <= 0:
        raise TransformerSequenceExportError("sequence_length must be > 0")
    if float(stride_sec) <= 0:
        raise TransformerSequenceExportError("stride_sec must be > 0")
    rows = _load_parquet_rows(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    filtered_rows, skipped = _filter_usable_rows(
        rows,
        include_not_ready=include_not_ready,
        keep_gap_affected=keep_gap_affected,
        min_time_until_resolution_sec=min_time_until_resolution_sec,
        max_time_until_resolution_sec=max_time_until_resolution_sec,
        label_column=label_column,
        market_id_column=market_id_column,
        timestamp_column=timestamp_column,
    )
    feature_columns = select_feature_columns(filtered_rows)
    grouped = _group_rows(filtered_rows, market_id_column=market_id_column)
    market_ranks = _market_chronological_ranks(grouped, timestamp_column=timestamp_column)

    sequence_rows: list[dict[str, Any]] = []
    sequences_per_market: dict[str, int] = {}
    sequence_index = 0
    for market_id in sorted(grouped):
        market_rows = sorted(
            grouped[market_id],
            key=lambda row: (
                _timestamp_sort_key(row.get(timestamp_column)),
                str(row.get(timestamp_column) or ""),
            ),
        )
        count_before = len(sequence_rows)
        for start_index in _sequence_start_indices(
            market_rows,
            sequence_length=int(sequence_length),
            stride_sec=float(stride_sec),
            timestamp_column=timestamp_column,
        ):
            window = market_rows[start_index : start_index + int(sequence_length)]
            if len(window) < int(sequence_length):
                continue
            sequence_rows.append(
                _build_sequence_row(
                    window,
                    sequence_index=sequence_index,
                    market_sequence_index=len(sequence_rows) - count_before,
                    market_id=str(market_id),
                    feature_columns=feature_columns,
                    label_column=label_column,
                    timestamp_column=timestamp_column,
                    market_rank=market_ranks.get(str(market_id)),
                )
            )
            sequence_index += 1
        sequences_per_market[str(market_id)] = len(sequence_rows) - count_before

    _write_parquet_rows(sequence_rows, output)
    positive_rate = _positive_rate([row.get(label_column) for row in sequence_rows])
    time_range = _sequence_time_range(sequence_rows)
    summary = {
        "status": "ok",
        "input_path": input_path,
        "output_path": str(output),
        "output_file_size_bytes": output.stat().st_size if output.exists() else 0,
        "sequence_length": int(sequence_length),
        "stride_sec": float(stride_sec),
        "label_column": label_column,
        "market_id_column": market_id_column,
        "timestamp_column": timestamp_column,
        "include_not_ready": bool(include_not_ready),
        "keep_gap_affected": bool(keep_gap_affected),
        "total_input_rows": len(rows),
        "usable_rows": len(filtered_rows),
        "markets": len(grouped),
        "sequences_exported": len(sequence_rows),
        "positive_label_rate": positive_rate,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "sequences_per_market": sequences_per_market,
        "sequences_per_market_summary": _count_summary(sequences_per_market.values()),
        "time_range": time_range,
        "skipped_rows_by_reason": dict(sorted(skipped.items())),
        "market_split_metadata": {
            "split_unit": "market",
            "market_sort_key": f"latest_{timestamp_column}",
            "market_chronological_rank_column": "market_chronological_rank",
            "market_split_sort_timestamp_column": "market_split_sort_timestamp",
        },
    }
    return summary


def _load_parquet_rows(path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TransformerSequenceExportError(
            "Transformer sequence export requires pyarrow to read Parquet files."
        ) from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_parquet_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TransformerSequenceExportError(
            "Transformer sequence export requires pyarrow to write Parquet files."
        ) from exc
    if rows:
        table = pa.Table.from_pylist(list(rows))
    else:
        table = pa.Table.from_pylist(
            [],
            schema=pa.schema(
                [
                    ("run_id", pa.string()),
                    ("market_id", pa.string()),
                    ("sequence_index", pa.int64()),
                    ("market_sequence_index", pa.int64()),
                    ("market_chronological_rank", pa.int64()),
                    ("market_split_sort_timestamp", pa.string()),
                    ("sequence_start_timestamp", pa.string()),
                    ("sequence_end_timestamp", pa.string()),
                    ("sequence_length", pa.int64()),
                    ("seconds_before_close_start", pa.float64()),
                    ("seconds_before_close_end", pa.float64()),
                    ("label_yes_win", pa.float64()),
                    ("label_resolved_up_down", pa.string()),
                    ("feature_columns", pa.string()),
                    ("sequence_values", pa.string()),
                ]
            ),
        )
    pq.write_table(table, path)


def _filter_usable_rows(
    rows: Sequence[dict[str, Any]],
    *,
    include_not_ready: bool,
    keep_gap_affected: bool,
    min_time_until_resolution_sec: float | None,
    max_time_until_resolution_sec: float | None,
    label_column: str,
    market_id_column: str,
    timestamp_column: str,
) -> tuple[list[dict[str, Any]], defaultdict[str, int]]:
    skipped: defaultdict[str, int] = defaultdict(int)
    usable: list[dict[str, Any]] = []
    for row in rows:
        if _missing(row.get(market_id_column)):
            skipped["missing_market_id"] += 1
            continue
        if _missing(row.get(timestamp_column)):
            skipped["missing_timestamp"] += 1
            continue
        if _missing(row.get(label_column)):
            skipped["missing_label"] += 1
            continue
        if not include_not_ready and not _is_truthy(row.get("feature_ready"), default=True):
            skipped["not_feature_ready"] += 1
            continue
        if not keep_gap_affected and _is_truthy(row.get("is_gap_affected"), default=False):
            skipped["gap_affected"] += 1
            continue
        seconds_before_close = _seconds_before_close(row)
        if (
            min_time_until_resolution_sec is not None
            and seconds_before_close is not None
            and seconds_before_close < float(min_time_until_resolution_sec)
        ):
            skipped["below_min_time_until_resolution"] += 1
            continue
        if (
            max_time_until_resolution_sec is not None
            and seconds_before_close is not None
            and seconds_before_close > float(max_time_until_resolution_sec)
        ):
            skipped["above_max_time_until_resolution"] += 1
            continue
        usable.append(dict(row))
    return usable, skipped


def _group_rows(
    rows: Sequence[dict[str, Any]],
    *,
    market_id_column: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(market_id_column))].append(dict(row))
    return dict(grouped)


def _market_chronological_ranks(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    timestamp_column: str,
) -> dict[str, int]:
    sortable = []
    for market_id, rows in grouped.items():
        latest = max(
            (_timestamp_sort_key(row.get(timestamp_column)) for row in rows),
            default=datetime.min.replace(tzinfo=timezone.utc),
        )
        sortable.append((latest, str(market_id)))
    return {
        market_id: index
        for index, (_timestamp, market_id) in enumerate(sorted(sortable))
    }


def _sequence_start_indices(
    rows: Sequence[dict[str, Any]],
    *,
    sequence_length: int,
    stride_sec: float,
    timestamp_column: str,
) -> list[int]:
    if len(rows) < sequence_length:
        return []
    starts: list[int] = []
    last_start_time: datetime | None = None
    fallback_stride = max(1, int(round(stride_sec)))
    for index in range(0, len(rows) - sequence_length + 1):
        start_time = parse_timestamp(rows[index].get(timestamp_column))
        if start_time is None:
            if index % fallback_stride == 0:
                starts.append(index)
            continue
        if last_start_time is None or start_time >= last_start_time + timedelta(seconds=stride_sec):
            starts.append(index)
            last_start_time = start_time
    return starts


def _build_sequence_row(
    window: Sequence[dict[str, Any]],
    *,
    sequence_index: int,
    market_sequence_index: int,
    market_id: str,
    feature_columns: Sequence[str],
    label_column: str,
    timestamp_column: str,
    market_rank: int | None,
) -> dict[str, Any]:
    first = window[0]
    last = window[-1]
    feature_columns_list = list(feature_columns)
    sequence_values = [
        [_float_or_none(row.get(column)) for column in feature_columns_list]
        for row in window
    ]
    row: dict[str, Any] = {
        "run_id": first.get("run_id"),
        "market_id": market_id,
        "sequence_index": int(sequence_index),
        "market_sequence_index": int(market_sequence_index),
        "market_chronological_rank": market_rank,
        "market_split_sort_timestamp": str(last.get(timestamp_column) or ""),
        "sequence_start_timestamp": str(first.get(timestamp_column) or ""),
        "sequence_end_timestamp": str(last.get(timestamp_column) or ""),
        "sequence_length": len(window),
        "seconds_before_close_start": _seconds_before_close(first),
        "seconds_before_close_end": _seconds_before_close(last),
        label_column: _label_value(last.get(label_column)),
        "feature_columns": json.dumps(feature_columns_list, separators=(",", ":")),
        "sequence_values": json.dumps(sequence_values, separators=(",", ":")),
    }
    if "label_yes_win" in first and label_column != "label_yes_win":
        row["label_yes_win"] = _label_value(last.get("label_yes_win"))
    if "label_resolved_up_down" in first:
        row["label_resolved_up_down"] = last.get("label_resolved_up_down")
    return row


def _seconds_before_close(row: dict[str, Any]) -> float | None:
    value = _float_or_none(row.get("seconds_before_close"))
    if value is not None:
        return value
    return _float_or_none(row.get("time_until_resolution"))


def _label_value(value: Any) -> Any:
    if _missing(value):
        return None
    parsed = _float_or_none(value)
    if parsed is not None and parsed.is_integer():
        return int(parsed)
    return parsed if parsed is not None else value


def _positive_rate(values: Sequence[Any]) -> float | None:
    parsed = [_float_or_none(value) for value in values if not _missing(value)]
    if not parsed:
        return None
    return sum(1 for value in parsed if value >= 0.5) / len(parsed)


def _count_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    counts = [int(value) for value in values]
    if not counts:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
    }


def _sequence_time_range(rows: Sequence[dict[str, Any]]) -> dict[str, str | None]:
    if not rows:
        return {"start": None, "end": None}
    starts = [str(row.get("sequence_start_timestamp") or "") for row in rows]
    ends = [str(row.get("sequence_end_timestamp") or "") for row in rows]
    return {"start": min(starts), "end": max(ends)}


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_sort_key(value: Any) -> datetime:
    return parse_timestamp(value) or datetime.min.replace(tzinfo=timezone.utc)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_truthy(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}
