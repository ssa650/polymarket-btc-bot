from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from src.train_transformer_sequence_model import (
    load_sequence_dataset,
    market_level_split,
    train_transformer_sequence_model,
)


BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _write_parquet(path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _sequence_row(
    market_index: int,
    sequence_index: int,
    *,
    label: int,
    feature_columns: list[str] | None = None,
    sequence_length: int = 4,
) -> dict:
    features = feature_columns or ["signal", "price", "seconds_before_close"]
    start = BASE + timedelta(minutes=market_index * 5, seconds=sequence_index)
    values = [
        [
            float(label) + step * 0.01,
            0.4 + 0.1 * label + step * 0.001,
            240.0 - step,
        ]
        for step in range(sequence_length)
    ]
    return {
        "run_id": "run_transformer",
        "market_id": f"market_{market_index}",
        "sequence_index": market_index * 10 + sequence_index,
        "market_sequence_index": sequence_index,
        "market_chronological_rank": market_index,
        "market_split_sort_timestamp": (start + timedelta(seconds=sequence_length)).isoformat(),
        "sequence_start_timestamp": start.isoformat(),
        "sequence_end_timestamp": (start + timedelta(seconds=sequence_length - 1)).isoformat(),
        "sequence_length": sequence_length,
        "seconds_before_close_start": 240.0,
        "seconds_before_close_end": 240.0 - sequence_length + 1,
        "label_yes_win": label,
        "label_resolved_up_down": "UP" if label == 1 else "DOWN",
        "feature_columns": json.dumps(features),
        "sequence_values": json.dumps(values),
    }


def _synthetic_rows(markets: int = 4, sequences_per_market: int = 3) -> list[dict]:
    rows: list[dict] = []
    for market_index in range(markets):
        label = market_index % 2
        for sequence_index in range(sequences_per_market):
            rows.append(_sequence_row(market_index, sequence_index, label=label))
    return rows


def test_load_sequence_dataset_parses_json_sequence_values(tmp_path) -> None:
    input_path = tmp_path / "sequences.parquet"
    _write_parquet(input_path, [_sequence_row(0, 0, label=1)])

    dataset = load_sequence_dataset(str(input_path), label_column="label_yes_win")

    assert dataset.feature_columns == ["signal", "price", "seconds_before_close"]
    assert dataset.labels == [1]
    assert dataset.market_ids == ["market_0"]
    assert len(dataset.sequences) == 1
    assert len(dataset.sequences[0]) == 4
    assert len(dataset.sequences[0][0]) == 3


def test_market_level_split_uses_newest_markets_for_test() -> None:
    rows = _synthetic_rows(markets=4, sequences_per_market=2)

    split = market_level_split(rows, test_market_fraction=0.25, validation_market_fraction=0.2)

    assert split["train_market_ids"] == ["market_0", "market_1"]
    assert split["validation_market_ids"] == ["market_2"]
    assert split["test_market_ids"] == ["market_3"]
    assert not set(split["train_market_ids"]) & set(split["test_market_ids"])
    assert not set(split["train_market_ids"]) & set(split["validation_market_ids"])
    assert not set(split["validation_market_ids"]) & set(split["test_market_ids"])


def test_train_transformer_sequence_model_writes_artifacts_when_torch_available(tmp_path) -> None:
    pytest.importorskip("torch")
    input_path = tmp_path / "sequences.parquet"
    output_dir = tmp_path / "model"
    _write_parquet(input_path, _synthetic_rows(markets=4, sequences_per_market=3))

    summary = train_transformer_sequence_model(
        input_path=str(input_path),
        output_dir=str(output_dir),
        label_column="label_yes_win",
        test_market_fraction=0.25,
        validation_market_fraction=0.2,
        max_epochs=1,
        batch_size=2,
        learning_rate=0.001,
        random_seed=7,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
    )

    assert summary["status"] == "ok"
    assert summary["train_test_market_overlap"] == []
    assert summary["train_validation_market_overlap"] == []
    assert summary["validation_test_market_overlap"] == []
    assert summary["train_rows"] == 6
    assert summary["validation_rows"] == 3
    assert summary["test_rows"] == 3
    assert summary["best_epoch"] == 1
    assert summary["epoch_history"][0]["validation_metrics"]["row_count"] == 3
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "training_summary.txt").exists()
    assert (output_dir / "feature_columns.json").exists()
    assert (output_dir / "scaler_stats.json").exists()
    assert (output_dir / "training_config.json").exists()


def test_train_transformer_sequence_model_cli_is_registered_and_offline(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    input_path = tmp_path / "sequences.parquet"
    output_dir = tmp_path / "model"
    _write_parquet(input_path, _synthetic_rows(markets=2, sequences_per_market=1))

    import src.main as main_module
    import src.train_transformer_sequence_model as trainer_module

    def fake_train(**kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "ok",
            "input_path": kwargs["input_path"],
            "output_dir": kwargs["output_dir"],
            "label_column": kwargs["label_column"],
            "epochs": kwargs["epochs"],
            "batch_size": kwargs["batch_size"],
            "learning_rate": kwargs["learning_rate"],
            "max_epochs": kwargs["max_epochs"],
            "validation_market_fraction": kwargs["validation_market_fraction"],
            "dropout": kwargs["dropout"],
            "weight_decay": kwargs["weight_decay"],
            "patience": kwargs["patience"],
            "d_model": kwargs["d_model"],
            "nhead": kwargs["nhead"],
            "num_layers": kwargs["num_layers"],
            "dim_feedforward": kwargs["dim_feedforward"],
            "positive_class_weight": kwargs["positive_class_weight"],
        }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("train-transformer-sequence-model must not start recorder")

    monkeypatch.setattr(trainer_module, "train_transformer_sequence_model", fake_train)
    monkeypatch.setattr(main_module, "load_settings", forbidden)
    monkeypatch.setattr(main_module, "RecorderApp", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "train-transformer-sequence-model",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--label-column",
            "label_yes_win",
            "--epochs",
            "1",
            "--max-epochs",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--validation-market-fraction",
            "0.2",
            "--dropout",
            "0.3",
            "--weight-decay",
            "0.002",
            "--patience",
            "4",
            "--d-model",
            "16",
            "--nhead",
            "4",
            "--num-layers",
            "1",
            "--dim-feedforward",
            "32",
            "--positive-class-weight",
            "auto",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["input_path"] == str(input_path)
    assert payload["output_dir"] == str(output_dir)
    assert payload["epochs"] == 1
    assert payload["max_epochs"] == 2
    assert payload["batch_size"] == 2
    assert payload["learning_rate"] == 0.001
    assert payload["validation_market_fraction"] == 0.2
    assert payload["dropout"] == 0.3
    assert payload["weight_decay"] == 0.002
    assert payload["patience"] == 4
    assert payload["d_model"] == 16
    assert payload["nhead"] == 4
    assert payload["num_layers"] == 1
    assert payload["dim_feedforward"] == 32
    assert payload["positive_class_weight"] == "auto"
