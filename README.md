# Polymarket BTC 5-Minute Recorder and ML Paper-Trading Toolkit

Capture Bitcoin Up/Down market data, build research datasets, and evaluate
model-driven strategies with simulated execution.

This Python project is for developers, market-data engineers, and ML researchers
who need a timestamped path from order books and BTC reference prices to offline
analysis. It combines a Polymarket recorder with baseline and transformer model
workflows, shadow predictions, paper trading, and local monitoring.

**Status:** local preparation for a possible public source release. Public
release, ownership review, and license selection are pending. No trained models,
private datasets, experiment reports, or credentials are included in this
candidate. The Mac checkout has not been verified against the former Ubuntu
deployment.

## Capabilities

- Discover BTC five-minute Up/Down markets with Gamma and select a primary market.
- Bootstrap CLOB order books, normalize websocket events, and persist snapshots,
  trades, features, lifecycle metadata, and health metrics in SQLite.
- Optionally collect BTC reference prices and archive raw websocket events.
- Export training/sequence datasets and fit scikit-learn baseline models.
- Support optional PyTorch transformer training/inference. PyTorch is a separate
  dependency and was absent from the verified Mac environment.
- Compare shadow predictions, simulate strategy execution, inspect paper results,
  and display local dashboards. The inspected live-model launcher has no real
  order-placement implementation.

## Quick start with synthetic data

Run from a clean source directory without a local `.env`. The tests create
temporary market data and model artifacts themselves; no account or private
dataset is needed.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --require-hashes \
  -r requirements-macos-arm64-py314.lock.txt
.venv/bin/python scripts/test_offline.py -q \
  tests/test_recorder_integration.py \
  tests/test_train_baseline_model.py \
  tests/test_release_configuration.py
.venv/bin/python -m src.main validate-strategy-config \
  --config configs/paper_strategy.example.json
```

The sample checks market selection through snapshot generation, fits baseline
models on synthetic rows, and validates configuration. Example strategy numbers
are arbitrary demonstration settings, not tuned private results. Installation
downloads dependencies; the test runner blocks TCP/UDP connections in its Python
process and removes inherited credential and recorder environment variables.

The hash lock covers verified macOS ARM64 / Python 3.14 wheel artifacts, including
test dependencies. A fresh installation was verified on macOS 26.3.1 with Python
3.14.6. Other platforms require their own resolved lock and validation; Linux
parity with the former deployment is not established. For development with bounded
ranges, use `requirements-dev.txt`. Optional transformer dependencies have a
separate `requirements-transformer-macos-arm64-py314.lock.txt`; install it into a
separate environment with `--require-hashes --only-binary=:all:`. No dependency
wheels are vendored in this source candidate.

## Train a baseline on fictional data

The [24-row JSON sample](examples/synthetic_training_sample.json) illustrates
the schema. The generator creates a training-sized 320-row Parquet dataset from
fixed-seed fictional markets, prices and outcomes:

```bash
.venv/bin/python scripts/create_synthetic_dataset.py
.venv/bin/python -m src.main train-baseline-model \
  --input data/synthetic/training.parquet \
  --output-dir data/synthetic/models
```

The generator refuses to overwrite an existing file. Use a new model output
directory for each run because the existing trainer writes named artifacts.
The verified command loads 320 rows, uses 224 for training and 96 for testing,
and writes baseline metrics, feature columns, a training summary and joblib model
files. Those generated artifacts remain ignored local files. Toy metrics exercise
the pipeline; they do not demonstrate market forecasting skill or profitability.
The baseline's chronological row split can put snapshots of the same market on
both sides; use separate held-out markets when assessing real predictive validity.

[License recommendation and compatibility tradeoffs](docs/LICENSE_OPTIONS.md)
remain subject to owner approval.

Current fitted weights and private observations are excluded by the owner's
code-first release decision. A future demo checkpoint would need a separate
provenance/data-rights, leakage and redistribution review; none is supplied here.

## Configuration and online recording

After offline checks, create configuration in a fresh source checkout:

```bash
cp .env.example .env
.venv/bin/python -m src.main --help
# Starts public feed connections and writes a new local recorder database:
.venv/bin/python -m src.main --env-file .env
```

Do not overwrite an existing `.env`. The example uses public feed URLs, local
`data/recorder.db`, a disabled optional BTC feed, and disabled raw event archival.
Environment variables override dotenv values. No wallet or API credential is
required by the inspected recorder. `scripts/run_recorder.sh` enables the
optional BTC feed by default; its behavior differs from the minimal example.
Live discovery, feed reachability, and sustained uptime were not tested here.

Key settings include `RECORDER_DB_PATH`, `DISCOVERY_INTERVAL_SEC`,
`SNAPSHOT_INTERVAL_SEC`, `BTC_PRICE_FEED_ENABLED`, `RAW_WS_EVENTS_ENABLED`, and
SQLite timeouts/checkpointing. Use separate paths for experiments and paper
outputs. Keep logs and artifacts private until their release is reviewed.

## Architecture and workflows

```text
Gamma discovery -> market selector -> CLOB REST / websocket -> normalized state
BTC reference feeds -------------------------------------> snapshots / features
                                                          |
                                                          v
                                                       SQLite
                                                          |
                               exports -> baseline / transformer training
                                                          |
                     fresh features -> shadow inference -> paper simulations
                                                          |
                                              reports / local dashboards
```

See [architecture](docs/ARCHITECTURE.md) and [the command guide](docs/CLI_GUIDE.md)
for module responsibilities, exports, models and paper workflows. The
[live-model stack guide](docs/live_model_stack.md) describes existing interfaces.
Command-guide maintenance operations can modify or prune datasets; they are
outside this preparation and the synthetic quick start.

There is no verified hosted demo or screenshot bundled with this preparation.
The synthetic quick start is the shortest reproducible demonstration. The
existing `web-dashboard` command can display your own local recorder/paper
databases; it is not a hosted public demo.

## Tests and contributions

```bash
.venv/bin/python scripts/test_offline.py -q
```

The ordinary suite uses synthetic inputs. Thirteen checks for the owner's exact
deployment grids are preserved behind `--private-configs`; their configs are
excluded. One localhost HTTP integration test is behind `--run-loopback` and must
run outside the socket-blocked runner. One transformer training test skips when
PyTorch is unavailable in the base environment; all 32 transformer training and
inference tests passed in the separately hash-locked PyTorch 2.14.1 environment. See [contributing](CONTRIBUTING.md) for commands and a
safe local contribution path.

## Limitations

- Upstream payload changes, resolutions, stale prices, missing books, gaps, and
  SQLite writer contention affect data quality and recorder availability.
- Historical evaluation can leak future information or overfit markets and
  thresholds. Model probabilities do not guarantee calibration.
- Paper P&L depends on modeled prices, fees, slippage, latency, liquidity and
  exits. It does not establish live profitability.
- No profitability claim, benchmark, paid API evaluation, or live order test was
  performed in this preparation.
- Create models/strategies from reviewed data. Load only trusted model artifacts:
  existing joblib/PyTorch loaders deserialize files.
- No authentication layer is claimed for local dashboards. Keep the default
  loopback binding and review exposure before changing it.
- Existing Git history contains private recorded/derived artifacts. Ignore rules
  do not remove tracked files or historical exposure. Do not publish that history
  until the owner approves a release method and data handling.

## License and project metadata

A project license has not been selected or applied. The owner must confirm rights
and select a license before public release. Third-party dependencies retain their
own licenses; see [dependency inventory](docs/DEPENDENCIES.md).

[Suggested GitHub About text and topics](docs/DISCOVERY.md) are provided for owner
review. No remote metadata has changed. If the synthetic workflow helps your
research, a star on a future public release is welcome.
