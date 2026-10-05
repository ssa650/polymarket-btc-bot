# Architecture

The recorder and research modules are independent layers in the same source tree.
This preparation changes documentation, configuration and test support; it does
not alter runtime Python logic or the SQL schema.

| Layer | Main modules | Contract |
| --- | --- | --- |
| Configuration | `config.py`, `main.py` | Dotenv/environment settings and command dispatch |
| Transport | `polymarket/gamma_client.py`, `clob_client.py`, `ws_client.py`, `btc_price_feed.py` | Public feed ingestion, normalization, REST bootstrap and reconnects |
| Capture | `recorder.py`, `state.py`, `models.py`, `features.py` | Primary market, rolling book/trade state, timestamps, snapshots and quality flags |
| Storage | `db.py`, `schema.sql` | SQLite observations and recorder metadata; explicit upgrades and maintenance |
| Research datasets | `training_dataset_export.py`, `transformer_sequence_dataset.py`, `backtest_dataset.py` | Features, labels and chronological sequences; exclude future/resolution labels from live input |
| Offline models | `train_baseline_model.py`, `train_transformer_sequence_model.py` | Scikit-learn baselines and optional PyTorch transformer artifacts |
| Inference and simulation | `transformer_live_inference.py`, `live_model_stack.py`, paper trader modules | Freshness gates, shadow predictions, modeled fills, risk/exit settings and paper outputs |
| Observation | `dashboard.py`, `web_dashboard.py`, audit/report modules | Local read-only dashboard views and research/operational reports |

Recorder data, model outputs and paper databases can use separate paths. Model
and strategy commands may write outputs even without connecting to a feed. The
live-model launcher logs that real order placement is not enabled when a live
flag is requested. Preserve this boundary; a flag name does not establish an
implemented real execution path.

Tests exercise synthetic snapshots, normalization, reconnects, schema integrity,
dataset/model paths, simulated fills and freshness controls. They do not establish
continuous service, parity with Ubuntu, model validity or live performance.
