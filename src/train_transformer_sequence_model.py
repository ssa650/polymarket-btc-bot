from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class TransformerTrainingError(RuntimeError):
    pass


@dataclass(slots=True)
class SequenceDataset:
    sequences: list[list[list[float | None]]]
    labels: list[int]
    market_ids: list[str]
    feature_columns: list[str]
    rows: list[dict[str, Any]]


def train_transformer_sequence_model(
    *,
    input_path: str,
    output_dir: str,
    label_column: str = "label_yes_win",
    test_market_fraction: float = 0.25,
    validation_market_fraction: float = 0.2,
    epochs: int | None = None,
    max_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    random_seed: int = 42,
    dropout: float = 0.25,
    weight_decay: float = 0.001,
    patience: int = 5,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 64,
    positive_class_weight: str | float = "auto",
) -> dict[str, Any]:
    deps = _load_dependencies()
    torch = deps["torch"]
    np = deps["np"]
    dataset = load_sequence_dataset(input_path, label_column=label_column)
    split = market_level_split(
        dataset.rows,
        test_market_fraction=test_market_fraction,
        validation_market_fraction=validation_market_fraction,
    )
    train_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row.get("market_id") or "") in split["train_market_ids"]
    ]
    validation_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row.get("market_id") or "") in split["validation_market_ids"]
    ]
    test_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row.get("market_id") or "") in split["test_market_ids"]
    ]
    if not train_indices or not validation_indices or not test_indices:
        raise TransformerTrainingError(
            "market-level split produced empty train, validation, or test rows"
        )

    _set_random_seeds(torch, np, int(random_seed))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_raw = _array_for_indices(np, dataset.sequences, train_indices)
    validation_raw = _array_for_indices(np, dataset.sequences, validation_indices)
    test_raw = _array_for_indices(np, dataset.sequences, test_indices)
    scaler = _fit_scaler(np, train_raw)
    train_x = _apply_scaler(np, train_raw, scaler)
    validation_x = _apply_scaler(np, validation_raw, scaler)
    test_x = _apply_scaler(np, test_raw, scaler)
    train_y = np.asarray([dataset.labels[index] for index in train_indices], dtype=np.float32)
    validation_y = np.asarray(
        [dataset.labels[index] for index in validation_indices],
        dtype=np.float32,
    )
    test_y = np.asarray([dataset.labels[index] for index in test_indices], dtype=np.float32)

    effective_max_epochs = max(1, int(epochs if epochs is not None else max_epochs))
    _validate_model_config(
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=float(dropout),
    )
    model_config = {
        "feature_count": len(dataset.feature_columns),
        "sequence_length": int(train_x.shape[1]),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
        "pooling": "mean",
    }
    model = SequenceTransformerClassifier(
        torch,
        feature_count=model_config["feature_count"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    class_weight = _positive_class_weight_value(train_y.tolist(), positive_class_weight)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_weight], dtype=torch.float32, device=device)
    )
    validation_loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader = _make_loader(
        torch,
        train_x,
        train_y,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        seed=int(random_seed),
    )
    epoch_history = []
    best_state = None
    best_epoch = 0
    best_selection_value: float | None = None
    best_selection_metric = "validation_loss"
    bad_epochs = 0
    for epoch in range(effective_max_epochs):
        loss = _train_one_epoch(
            torch,
            model,
            train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        train_probabilities_epoch = _predict_probabilities(
            torch,
            model,
            train_x,
            batch_size,
            device,
        )
        validation_probabilities_epoch = _predict_probabilities(
            torch,
            model,
            validation_x,
            batch_size,
            device,
        )
        train_metrics_epoch = _classification_metrics(
            train_y.tolist(),
            train_probabilities_epoch,
            deps,
        )
        validation_metrics_epoch = _classification_metrics(
            validation_y.tolist(),
            validation_probabilities_epoch,
            deps,
        )
        validation_loss = _evaluation_loss(
            torch,
            model,
            validation_x,
            validation_y,
            batch_size=batch_size,
            loss_fn=validation_loss_fn,
            device=device,
        )
        selection_metric, selection_value, maximize = _validation_selection_metric(
            validation_metrics_epoch,
            validation_loss,
        )
        improved = _is_better(
            selection_value,
            best_selection_value,
            maximize=maximize,
        )
        if improved:
            best_state = _clone_state_dict(model)
            best_epoch = epoch + 1
            best_selection_metric = selection_metric
            best_selection_value = selection_value
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics_epoch,
            "validation_metrics": validation_metrics_epoch,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        epoch_history.append(epoch_record)
        _print_epoch_progress(epoch_record)
        if bad_epochs >= max(1, int(patience)):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_probabilities = _predict_probabilities(torch, model, test_x, batch_size, device)
    train_probabilities = _predict_probabilities(torch, model, train_x, batch_size, device)
    validation_probabilities = _predict_probabilities(
        torch,
        model,
        validation_x,
        batch_size,
        device,
    )
    metrics = {
        "status": "ok",
        "input_path": input_path,
        "output_dir": str(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_column": label_column,
        "rows_loaded": len(dataset.rows),
        "market_count": len(set(dataset.market_ids)),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "test_rows": len(test_indices),
        "train_markets": len(split["train_market_ids"]),
        "validation_markets": len(split["validation_market_ids"]),
        "test_markets": len(split["test_market_ids"]),
        "train_market_ids": split["train_market_ids"],
        "validation_market_ids": split["validation_market_ids"],
        "test_market_ids": split["test_market_ids"],
        "train_test_market_overlap": sorted(
            set(split["train_market_ids"]) & set(split["test_market_ids"])
        ),
        "train_validation_market_overlap": sorted(
            set(split["train_market_ids"]) & set(split["validation_market_ids"])
        ),
        "validation_test_market_overlap": sorted(
            set(split["validation_market_ids"]) & set(split["test_market_ids"])
        ),
        "feature_count": len(dataset.feature_columns),
        "sequence_length": model_config["sequence_length"],
        "epoch_history": epoch_history,
        "best_epoch": best_epoch,
        "best_selection_metric": best_selection_metric,
        "best_selection_value": best_selection_value,
        "early_stopped": len(epoch_history) < effective_max_epochs,
        "positive_class_weight": class_weight,
        "train_metrics": _classification_metrics(train_y.tolist(), train_probabilities, deps),
        "validation_metrics": _classification_metrics(
            validation_y.tolist(),
            validation_probabilities,
            deps,
        ),
        "test_metrics": _classification_metrics(test_y.tolist(), test_probabilities, deps),
        "market_split": {
            "test_market_fraction": float(test_market_fraction),
            "validation_market_fraction": float(validation_market_fraction),
            "sort": "oldest markets train, newest markets test",
        },
    }
    artifacts = _write_artifacts(
        torch,
        output,
        model=model,
        model_config=model_config,
        training_config={
            "input_path": input_path,
            "label_column": label_column,
            "test_market_fraction": float(test_market_fraction),
            "validation_market_fraction": float(validation_market_fraction),
            "epochs": None if epochs is None else int(epochs),
            "max_epochs": int(max_epochs),
            "effective_max_epochs": effective_max_epochs,
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "random_seed": int(random_seed),
            "dropout": float(dropout),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "positive_class_weight": positive_class_weight,
            "sequence_length": model_config["sequence_length"],
            "feature_count": model_config["feature_count"],
        },
        metrics=metrics,
        feature_columns=dataset.feature_columns,
        scaler=scaler,
    )
    metrics["artifacts_written"] = artifacts
    return metrics


def load_sequence_dataset(input_path: str, *, label_column: str = "label_yes_win") -> SequenceDataset:
    rows = _load_parquet_rows(input_path)
    sequences: list[list[list[float | None]]] = []
    labels: list[int] = []
    market_ids: list[str] = []
    feature_columns: list[str] | None = None
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        if _missing(row.get(label_column)):
            continue
        parsed_features = _parse_feature_columns(row.get("feature_columns"))
        parsed_sequence = _parse_sequence_values(row.get("sequence_values"))
        if not parsed_features or not parsed_sequence:
            continue
        if feature_columns is None:
            feature_columns = parsed_features
        elif parsed_features != feature_columns:
            raise TransformerTrainingError("inconsistent feature_columns across sequence rows")
        if any(len(step) != len(parsed_features) for step in parsed_sequence):
            raise TransformerTrainingError("sequence_values width does not match feature_columns")
        labels.append(int(float(row[label_column])))
        sequences.append(parsed_sequence)
        market_ids.append(str(row.get("market_id") or ""))
        kept_rows.append(dict(row))
    if not kept_rows:
        raise TransformerTrainingError("no usable sequence rows found")
    return SequenceDataset(
        sequences=sequences,
        labels=labels,
        market_ids=market_ids,
        feature_columns=feature_columns or [],
        rows=kept_rows,
    )


def market_level_split(
    rows: Sequence[dict[str, Any]],
    *,
    test_market_fraction: float = 0.25,
    validation_market_fraction: float = 0.2,
) -> dict[str, Any]:
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_market.setdefault(str(row.get("market_id") or ""), []).append(dict(row))
    markets = sorted(
        by_market,
        key=lambda market_id: (
            _market_sort_key(by_market[market_id]),
            market_id,
        ),
    )
    if len(markets) < 2:
        raise TransformerTrainingError("at least two markets are required for market-level split")
    fraction = min(max(float(test_market_fraction), 0.0), 1.0)
    test_count = max(1, int(math.ceil(len(markets) * fraction)))
    if test_count >= len(markets):
        test_count = len(markets) - 1
    train_validation_markets = markets[:-test_count]
    test_markets = markets[-test_count:]
    validation_count = 0
    validation_fraction = min(max(float(validation_market_fraction), 0.0), 1.0)
    if validation_fraction > 0.0 and len(train_validation_markets) > 1:
        validation_count = max(1, int(math.ceil(len(train_validation_markets) * validation_fraction)))
        if validation_count >= len(train_validation_markets):
            validation_count = len(train_validation_markets) - 1
    if validation_count:
        train_markets = train_validation_markets[:-validation_count]
        validation_markets = train_validation_markets[-validation_count:]
    else:
        train_markets = train_validation_markets
        validation_markets = []
    return {
        "train_market_ids": train_markets,
        "validation_market_ids": validation_markets,
        "test_market_ids": test_markets,
        "market_count": len(markets),
    }


class SequenceTransformerClassifier:
    def __init__(
        self,
        torch_module: Any,
        *,
        feature_count: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        torch = torch_module

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = torch.nn.Linear(feature_count, d_model)
                self.position = torch.nn.Parameter(torch.zeros(1, 512, d_model))
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = torch.nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                )
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(d_model),
                    torch.nn.Linear(d_model, 1),
                )

            def forward(self, x):
                seq_len = x.shape[1]
                projected = self.input_projection(x)
                if seq_len > self.position.shape[1]:
                    raise ValueError("sequence length exceeds positional encoding capacity")
                encoded = self.encoder(projected + self.position[:, :seq_len, :])
                pooled = encoded.mean(dim=1)
                return self.head(pooled).squeeze(-1)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def _load_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as pq
        import torch
    except ImportError as exc:
        raise TransformerTrainingError(
            "Transformer sequence training requires PyTorch, numpy, and pyarrow. "
            "Install PyTorch in the project venv before running train-transformer-sequence-model."
        ) from exc
    deps: dict[str, Any] = {"np": np, "pq": pq, "torch": torch}
    try:
        from sklearn.metrics import (
            balanced_accuracy_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        deps.update(
            {
                "balanced_accuracy_score": balanced_accuracy_score,
                "brier_score_loss": brier_score_loss,
                "confusion_matrix": confusion_matrix,
                "f1_score": f1_score,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "roc_auc_score": roc_auc_score,
            }
        )
    except ImportError:
        pass
    return deps


def _validate_model_config(
    *,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> None:
    if d_model <= 0:
        raise TransformerTrainingError("d_model must be > 0")
    if nhead <= 0:
        raise TransformerTrainingError("nhead must be > 0")
    if d_model % nhead != 0:
        raise TransformerTrainingError("d_model must be divisible by nhead")
    if num_layers <= 0:
        raise TransformerTrainingError("num_layers must be > 0")
    if dim_feedforward <= 0:
        raise TransformerTrainingError("dim_feedforward must be > 0")
    if not 0.0 <= dropout < 1.0:
        raise TransformerTrainingError("dropout must be >= 0 and < 1")


def _load_parquet_rows(path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TransformerTrainingError("pyarrow is required to read sequence parquet files") from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _parse_feature_columns(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise TransformerTrainingError("feature_columns must be a JSON list")
    return [str(item) for item in parsed]


def _parse_sequence_values(value: Any) -> list[list[float | None]]:
    if value is None:
        return []
    parsed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(parsed, list):
        raise TransformerTrainingError("sequence_values must be a JSON list")
    sequence: list[list[float | None]] = []
    for step in parsed:
        if not isinstance(step, list):
            raise TransformerTrainingError("sequence_values must be a list of lists")
        sequence.append([_float_or_none(item) for item in step])
    return sequence


def _array_for_indices(np: Any, sequences: Sequence[Any], indices: Sequence[int]) -> Any:
    selected = [sequences[index] for index in indices]
    return np.asarray(selected, dtype=np.float32)


def _fit_scaler(np: Any, train_x: Any) -> dict[str, list[float]]:
    mean = np.nanmean(train_x, axis=(0, 1))
    std = np.nanstd(train_x, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1.0e-8), std, 1.0)
    return {
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }


def _apply_scaler(np: Any, x: Any, scaler: dict[str, list[float]]) -> Any:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    filled = np.where(np.isfinite(x), x, mean.reshape(1, 1, -1))
    return ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def _make_loader(
    torch: Any,
    x: Any,
    y: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _train_one_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    *,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_y.shape[0])
        total_rows += int(batch_y.shape[0])
    return total_loss / max(1, total_rows)


def _evaluation_loss(
    torch: Any,
    model: Any,
    x: Any,
    y: Any,
    *,
    batch_size: int,
    loss_fn: Any,
    device: Any,
) -> float:
    loader = _make_loader(
        torch,
        x,
        y,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        seed=0,
    )
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += float(loss.item()) * int(batch_y.shape[0])
            total_rows += int(batch_y.shape[0])
    return total_loss / max(1, total_rows)


def _predict_probabilities(
    torch: Any,
    model: Any,
    x: Any,
    batch_size: int,
    device: Any,
) -> list[float]:
    loader = _make_loader(
        torch,
        x,
        [0.0] * len(x),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        seed=0,
    )
    model.eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            logits = model(batch_x.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
    return [float(value) for value in probabilities]


def _positive_class_weight_value(labels: Sequence[float], value: str | float) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            positives = sum(1 for label in labels if int(label) == 1)
            negatives = sum(1 for label in labels if int(label) == 0)
            if positives <= 0:
                return 1.0
            return max(0.01, float(negatives) / float(positives))
        if normalized in {"none", "off", ""}:
            return 1.0
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise TransformerTrainingError(
                "positive_class_weight must be 'auto', 'none', or a positive number"
            ) from exc
    else:
        parsed = float(value)
    if parsed <= 0:
        raise TransformerTrainingError("positive_class_weight must be > 0")
    return float(parsed)


def _validation_selection_metric(
    validation_metrics: dict[str, Any],
    validation_loss: float,
) -> tuple[str, float, bool]:
    roc_auc = validation_metrics.get("roc_auc")
    if roc_auc is not None:
        return "validation_roc_auc", float(roc_auc), True
    return "validation_loss", float(validation_loss), False


def _is_better(value: float, best: float | None, *, maximize: bool) -> bool:
    if best is None:
        return True
    if maximize:
        return value > best + 1.0e-9
    return value < best - 1.0e-9


def _clone_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _print_epoch_progress(epoch_record: dict[str, Any]) -> None:
    validation_metrics = epoch_record.get("validation_metrics") or {}
    payload = {
        "event": "transformer_sequence_epoch",
        "epoch": epoch_record.get("epoch"),
        "train_loss": epoch_record.get("train_loss"),
        "validation_loss": epoch_record.get("validation_loss"),
        "validation_accuracy": validation_metrics.get("accuracy"),
        "validation_roc_auc": validation_metrics.get("roc_auc"),
        "selection_metric": epoch_record.get("selection_metric"),
        "selection_value": epoch_record.get("selection_value"),
        "improved": epoch_record.get("improved"),
        "bad_epochs": epoch_record.get("bad_epochs"),
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _classification_metrics(
    labels: Sequence[float],
    probabilities: Sequence[float],
    deps: dict[str, Any],
) -> dict[str, Any]:
    y_true = [int(value) for value in labels]
    y_prob = [float(value) for value in probabilities]
    y_pred = [1 if value >= 0.5 else 0 for value in y_prob]
    total = len(y_true)
    positives = sum(y_true)
    true_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    true_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    false_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    false_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    accuracy = (true_positive + true_negative) / total if total else None
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    recall_neg = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    balanced_accuracy = (recall + recall_neg) / 2.0
    brier = sum((prob - truth) ** 2 for truth, prob in zip(y_true, y_prob)) / total if total else None
    roc_auc = None
    if len(set(y_true)) == 2 and "roc_auc_score" in deps:
        try:
            roc_auc = float(deps["roc_auc_score"](y_true, y_prob))
        except Exception:
            roc_auc = None
    return {
        "row_count": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
        "positive_prediction_rate": sum(y_pred) / total if total else None,
        "target_positive_rate": positives / total if total else None,
    }


def _write_artifacts(
    torch: Any,
    output: Path,
    *,
    model: Any,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    metrics: dict[str, Any],
    feature_columns: Sequence[str],
    scaler: dict[str, list[float]],
) -> list[str]:
    artifacts: list[str] = []
    model_path = output / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
        },
        model_path,
    )
    artifacts.append(str(model_path))
    for filename, payload in (
        ("metrics.json", metrics),
        ("feature_columns.json", list(feature_columns)),
        ("scaler_stats.json", scaler),
        ("training_config.json", training_config),
    ):
        path = output / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.append(str(path))
    summary_path = output / "training_summary.txt"
    summary_path.write_text(_render_training_summary(metrics), encoding="utf-8")
    artifacts.append(str(summary_path))
    return artifacts


def _render_training_summary(metrics: dict[str, Any]) -> str:
    validation = metrics.get("validation_metrics") or {}
    test = metrics.get("test_metrics") or {}
    lines = [
        "Transformer Sequence Model",
        f"status={metrics.get('status')}",
        f"rows_loaded={metrics.get('rows_loaded')}",
        f"train_rows={metrics.get('train_rows')}",
        f"validation_rows={metrics.get('validation_rows')}",
        f"test_rows={metrics.get('test_rows')}",
        f"feature_count={metrics.get('feature_count')}",
        f"sequence_length={metrics.get('sequence_length')}",
        f"best_epoch={metrics.get('best_epoch')}",
        f"best_selection_metric={metrics.get('best_selection_metric')}",
        f"best_selection_value={metrics.get('best_selection_value')}",
        f"validation_accuracy={validation.get('accuracy')}",
        f"validation_roc_auc={validation.get('roc_auc')}",
        f"test_accuracy={test.get('accuracy')}",
        f"test_balanced_accuracy={test.get('balanced_accuracy')}",
        f"test_roc_auc={test.get('roc_auc')}",
        f"test_brier_score={test.get('brier_score')}",
    ]
    return "\n".join(lines) + "\n"


def _set_random_seeds(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _market_sort_key(rows: Sequence[dict[str, Any]]) -> tuple[int, str]:
    ranks = [_int_or_none(row.get("market_chronological_rank")) for row in rows]
    valid_ranks = [rank for rank in ranks if rank is not None]
    rank = min(valid_ranks) if valid_ranks else 10**12
    timestamps = [
        str(row.get("market_split_sort_timestamp") or row.get("sequence_end_timestamp") or "")
        for row in rows
    ]
    return rank, max(timestamps) if timestamps else ""


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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
