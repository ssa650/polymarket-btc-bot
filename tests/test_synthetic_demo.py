from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from scripts.create_synthetic_dataset import synthetic_rows
from src.train_baseline_model import select_feature_columns, train_baseline_model


def test_synthetic_demo_trains_with_both_classes_and_excludes_labels(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = synthetic_rows()
    assert rows == synthetic_rows()
    assert {row['label_btc_up_at_resolution'] for row in rows[:224]} == {0, 1}
    assert {row['label_btc_up_at_resolution'] for row in rows[224:]} == {0, 1}
    assert all(datetime.fromisoformat(row['start_time']) <= datetime.fromisoformat(row['timestamp'])
               < datetime.fromisoformat(row['close_time']) for row in rows)
    features = select_feature_columns(rows)
    assert not any('label' in name or name.startswith('future_') for name in features)
    dataset = tmp_path / 'synthetic.parquet'
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    output = tmp_path / 'models'
    report = train_baseline_model(input_path=str(dataset), output_dir=str(output))
    assert report['status'] == 'ok'
    assert report['rows_after_filters'] == 320
    assert (output / 'model_logistic_regression.joblib').is_file()
    assert (output / 'model_random_forest.joblib').is_file()


def test_shipped_json_sample_is_generated_and_fictional():
    path = Path(__file__).resolve().parents[1] / 'examples/synthetic_training_sample.json'
    payload = json.loads(path.read_text())
    assert payload['synthetic'] is True
    assert payload['rows'] == synthetic_rows(24)
