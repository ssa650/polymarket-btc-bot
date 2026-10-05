from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from src.transformer_sequence_dataset import export_transformer_sequence_dataset


BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _write_parquet(path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _read_parquet_rows(path) -> list[dict]:
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _row(
    market_id: str,
    offset_sec: int,
    *,
    label: int | None = 1,
    feature_ready: int = 1,
    gap: int = 0,
    run_id: str = "run_seq",
) -> dict:
    ts = BASE + timedelta(seconds=offset_sec)
    return {
        "run_id": run_id,
        "market_id": market_id,
        "timestamp": ts.isoformat(),
        "question": "Bitcoin Up or Down",
        "start_time": BASE.isoformat(),
        "close_time": (BASE + timedelta(minutes=5)).isoformat(),
        "feature_ready": feature_ready,
        "is_gap_affected": gap,
        "strict_validation_passed": 1,
        "snapshot_quality_status": "ok",
        "export_row_usable": 1,
        "label_yes_win": label,
        "label_resolved_up_down": "UP" if label == 1 else ("DOWN" if label == 0 else None),
        "future_btc_return_to_resolution_from_feature": 0.01 if label == 1 else -0.01,
        "btc_price_at_resolution": 101.0,
        "seconds_before_close": 240.0 - offset_sec,
        "time_until_resolution": 240.0 - offset_sec,
        "seconds_after_start": float(offset_sec),
        "signal": float(label or 0) + offset_sec / 100.0,
        "best_bid_yes": 0.45,
        "best_ask_yes": 0.47,
        "spread_yes": 0.02,
        "btc_chainlink_price": 100.0 + offset_sec,
        "btc_chainlink_age_sec_at_feature": 1.0,
    }


def test_export_transformer_sequence_dataset_builds_fixed_length_sequences(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_path = tmp_path / "sequences.parquet"
    rows = [_row("m1", idx, label=1) for idx in range(5)]
    rows.extend(_row("m2", 20 + idx, label=0) for idx in range(4))
    _write_parquet(input_path, rows)

    summary = export_transformer_sequence_dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        sequence_length=3,
        stride_sec=1,
        min_time_until_resolution_sec=0,
        max_time_until_resolution_sec=240,
    )
    sequence_rows = _read_parquet_rows(output_path)

    assert summary["status"] == "ok"
    assert summary["total_input_rows"] == 9
    assert summary["usable_rows"] == 9
    assert summary["markets"] == 2
    assert summary["sequences_exported"] == 5
    assert summary["positive_label_rate"] == 3 / 5
    assert summary["sequences_per_market"] == {"m1": 3, "m2": 2}
    assert output_path.exists()
    first = sequence_rows[0]
    feature_columns = json.loads(first["feature_columns"])
    sequence_values = json.loads(first["sequence_values"])
    assert first["market_id"] == "m1"
    assert first["sequence_length"] == 3
    assert first["label_yes_win"] == 1
    assert first["label_resolved_up_down"] == "UP"
    assert "signal" in feature_columns
    assert "label_yes_win" not in feature_columns
    assert "future_btc_return_to_resolution_from_feature" not in feature_columns
    assert len(sequence_values) == 3
    assert all(len(values) == len(feature_columns) for values in sequence_values)
    assert "market_chronological_rank" in first
    assert "market_split_sort_timestamp" in first


def test_export_transformer_sequence_dataset_filters_quality_rows(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_path = tmp_path / "sequences.parquet"
    rows = [_row("m1", idx, label=1) for idx in range(3)]
    rows.append(_row("m2", 10, label=0, feature_ready=0))
    rows.append(_row("m2", 11, label=0, gap=1))
    rows.append(_row("m2", 12, label=None))
    _write_parquet(input_path, rows)

    summary = export_transformer_sequence_dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        sequence_length=3,
        stride_sec=1,
    )

    assert summary["usable_rows"] == 3
    assert summary["sequences_exported"] == 1
    assert summary["skipped_rows_by_reason"] == {
        "gap_affected": 1,
        "missing_label": 1,
        "not_feature_ready": 1,
    }


def test_export_transformer_sequence_dataset_can_include_not_ready_and_gap_rows(tmp_path) -> None:
    input_path = tmp_path / "training.parquet"
    output_path = tmp_path / "sequences.parquet"
    rows = [
        _row("m1", 0, label=1),
        _row("m1", 1, label=1, feature_ready=0),
        _row("m1", 2, label=1, gap=1),
    ]
    _write_parquet(input_path, rows)

    summary = export_transformer_sequence_dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        sequence_length=3,
        stride_sec=1,
        include_not_ready=True,
        keep_gap_affected=True,
    )

    assert summary["usable_rows"] == 3
    assert summary["sequences_exported"] == 1
    assert summary["skipped_rows_by_reason"] == {}


def test_export_transformer_sequence_dataset_cli_is_offline(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "training.parquet"
    output_path = tmp_path / "sequences.parquet"
    _write_parquet(input_path, [_row("m1", idx, label=1) for idx in range(3)])

    import src.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("export-transformer-sequence-dataset must not start recorder")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "export-transformer-sequence-dataset",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--sequence-length",
            "3",
            "--stride-sec",
            "1",
        ],
    )
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["sequences_exported"] == 1
    assert output_path.exists()

