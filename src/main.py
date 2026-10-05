from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings
from .db import SQLiteStore, run_sqlite_wal_checkpoint
from .diagnostics import print_diagnostics
from .logging_config import configure_logging
from .polymarket.gamma_client import GammaClient
from .recorder import RecorderApp
from .training_export import export_training_parquet
from .validator import validate_db_quality


SCHEMA_UPGRADE_TARGET_TABLES = (
    "raw_polymarket_events",
    "tick_size_changes",
    "best_bid_ask_updates",
    "btc_prices",
)

SCHEMA_UPGRADE_PRESERVE_TABLES = (
    "markets",
    "market_snapshots",
    "order_book_levels",
    "trades",
    "features",
    "market_events",
    "recorder_metrics",
)


async def _run_with_signals(app: RecorderApp) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        logging.getLogger("main").info("stop_signal_received")
        asyncio.create_task(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # add_signal_handler may be unavailable on some platforms.
            signal.signal(sig, lambda *_: _request_stop())

    await app.run()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket recorder bot")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=(
            "run",
            "debug-discovery",
            "export-training",
            "export-training-dataset",
            "export-recorder-archive-to-parquet",
            "export-transformer-sequence-dataset",
            "validate-db",
            "fresh-db",
            "upgrade-db",
            "repair-stale-active-markets",
            "normalize-resolutions",
            "inspect-backtest-dataset",
            "run-paper-backtest",
            "report-paper-runs",
            "train-baseline-model",
            "train-transformer-sequence-model",
            "audit-baseline-model",
            "audit-live-model-predictions",
            "research-model-candidates",
            "promote-research-model-candidate",
            "compare-model-artifacts",
            "backtest-baseline-strategy",
            "run-baseline-paper-trader",
            "paper-trader-summary",
            "paper-trader-analytics",
            "paper-trader-live-status",
            "compact-paper-live-status",
            "overnight-health-report",
            "paper-strategy-leaderboard",
            "paper-trader-promotion-report",
            "paper-trade-resolution-autopsy",
            "trade-autopsy",
            "export-meta-trade-dataset",
            "transformer-prediction-autopsy",
            "report-transformer-predictions",
            "transformer-calibration-analysis",
            "backtest-transformer-threshold-strategy",
            "export-paper-trader-dataset",
            "export-meta-strategy-dataset",
            "run-multi-strategy-paper-trader",
            "run-transformer-paper-trader",
            "run-transformer-shadow-paper-strategy",
            "run-transformer-prediction-paper-strategy",
            "backfill-transformer-prediction-paper-pnl",
            "run-paper-strategy-experiment",
            "report-paper-strategy-experiment",
            "monitor-paper-strategy-experiment",
            "analyze-paper-strategy-experiment",
            "recommend-next-paper-sweep",
            "mine-paper-strategies",
            "compare-model-experiments",
            "run-live-model-predictions",
            "score-live-model-predictions",
            "live-model-dashboard",
            "start-live-model-stack",
            "run-intramarket-paper-trader",
            "intramarket-paper-analytics",
            "intramarket-strategy-leaderboard",
            "export-intramarket-strategy-dataset",
            "live-paper-maintenance-loop",
            "generate-paper-strategy-config-sweep",
            "filter-strategy-config",
            "validate-strategy-config",
            "strategy-config-summary",
            "paper-trader-eligibility-report",
            "refresh-expired-market-resolutions",
            "dashboard",
            "web-dashboard",
            "btc-feed-config",
            "probe-btc-rtds",
            "wal-checkpoint",
            "dataset-quality",
            "daily-recorder-maintenance",
        ),
        help="Optional command. Use 'debug-discovery' for one-shot discovery diagnostics.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: .env if present)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print database diagnostics and exit",
    )
    parser.add_argument(
        "--repair-invalid-snapshot-trades",
        action="store_true",
        help=(
            "Null historical snapshot trade fields where "
            "last_trade_time is before market start or after snapshot timestamp"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Preview a repair command without writing changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow commands that support replacement to overwrite their output path.",
    )
    parser.add_argument(
        "--db",
        default="data/recorder.db",
        help="Recorder SQLite DB path for run-baseline-paper-trader.",
    )
    parser.add_argument(
        "--recorder-db",
        default=None,
        help="Recorder SQLite DB path for live-paper-maintenance-loop.",
    )
    parser.add_argument(
        "--output-db",
        default="data/paper_trades.db",
        help=(
            "Output SQLite DB path for paper traders. For run-transformer-paper-trader, "
            "predictions are logged only when this option is explicitly supplied."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append compact JSONL operational logs for transformer paper/logging commands.",
    )
    parser.add_argument(
        "--jsonl-log-file",
        default=None,
        help="Append compact JSONL logs for transformer paper/logging commands.",
    )
    parser.add_argument(
        "--jsonl-log",
        action="append",
        default=None,
        help="Transformer heartbeat JSONL path or glob for transformer-calibration-analysis.",
    )
    parser.add_argument(
        "--heartbeat-log-file",
        default=None,
        help="Append compact JSONL heartbeat logs for transformer paper/logging commands.",
    )
    parser.add_argument(
        "--prediction-db",
        action="append",
        default=None,
        help=(
            "Transformer prediction SQLite DB path or glob for "
            "transformer-prediction-autopsy. Repeat to include multiple DBs."
        ),
    )
    parser.add_argument(
        "--paper-db",
        action="append",
        default=None,
        help=(
            "Paper trades SQLite DB path. Repeat for web-dashboard; "
            "single-DB commands use the last provided value. "
            "Defaults to data/paper_trades.db when omitted."
        ),
    )
    parser.add_argument(
        "--paper-db-path",
        action="append",
        default=None,
        help=(
            "Repeatable paper trades DB path for paper-trader-promotion-report. "
            "If omitted, --paper-db is used."
        ),
    )
    parser.add_argument(
        "--strategy-id",
        action="append",
        default=None,
        help=(
            "Strategy id filter. Repeat for paper-strategy-leaderboard; "
            "single-strategy commands use the last provided value."
        ),
    )
    parser.add_argument(
        "--strategy-name",
        default=None,
        help="Strategy display name for export-paper-trader-dataset.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append export-paper-trader-dataset rows to master comparison parquet files.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include skipped diagnostic rows in export-paper-trader-dataset trade files.",
    )
    parser.add_argument(
        "--include-awaiting",
        action="store_true",
        help="Include awaiting_resolution rows in export-paper-trader-dataset trade files.",
    )
    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Include unresolved candidates in export-meta-strategy-dataset.",
    )
    parser.add_argument(
        "--min-created-at",
        default=None,
        help="Minimum candidate created_at timestamp for export-meta-strategy-dataset.",
    )
    parser.add_argument(
        "--max-created-at",
        default=None,
        help="Maximum candidate created_at timestamp for export-meta-strategy-dataset.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes string for export-paper-trader-dataset metadata.",
    )
    parser.add_argument(
        "--config",
        default="data/strategy_configs/live_strategy_grid.json",
        help=(
            "JSON config path for strategy commands. For filter-strategy-config, "
            "this is the source config."
        ),
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Base strategy config path for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--config-glob",
        default=None,
        help="Strategy config glob for run-paper-strategy-experiment.",
    )
    parser.add_argument(
        "--name-prefix",
        default="paper_sweep",
        help="Strategy id/name prefix for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated probability thresholds for strategy sweep generation "
            "or transformer threshold backtests."
        ),
    )
    parser.add_argument(
        "--strategy-filter",
        default=None,
        help="Exact strategy_id filter for focused transformer threshold backtests.",
    )
    parser.add_argument(
        "--max-probabilities",
        default="none,0.90,0.95",
        help="Comma-separated max probability values or none for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--time-windows",
        default="0:90,30:90,60:120,90:180,120:240,150:240",
        help="Comma-separated min:max second windows for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--bankrolls",
        default="100,1000",
        help="Comma-separated bankroll values for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--blocked-liquidity-regimes",
        default=None,
        help="Comma-separated liquidity regimes to block in generated strategy configs.",
    )
    parser.add_argument(
        "--backtest-json-glob",
        action="append",
        default=[],
        help="Transformer backtest JSON glob for mine-paper-strategies.",
    )
    parser.add_argument(
        "--fixed-horizon-sec",
        type=float,
        default=15.0,
        help="Fixed horizon exit seconds for generate-paper-strategy-config-sweep.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment id for run-paper-strategy-experiment.",
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=None,
        help="Experiment directory for report-paper-strategy-experiment.",
    )
    parser.add_argument(
        "--paper-experiment-dir",
        default=None,
        help="Paper experiment directory for compare-model-experiments.",
    )
    parser.add_argument(
        "--transformer-prediction-db",
        action="append",
        default=None,
        help=(
            "Transformer prediction DB path or glob for compare-model-experiments. "
            "Repeat to include multiple DBs."
        ),
    )
    parser.add_argument(
        "--transformer-autopsy-parquet",
        default=None,
        help="Transformer prediction autopsy parquet/csv path for compare-model-experiments.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Launch paper strategy experiment workers in tmux sessions.",
    )
    parser.add_argument(
        "--heartbeat-detail",
        choices=("compact", "full"),
        default="compact",
        help=(
            "Heartbeat log detail for run-multi-strategy-paper-trader and "
            "paper strategy experiment workers."
        ),
    )
    parser.add_argument(
        "--verbose-txt",
        action="store_true",
        help="Write verbose TXT output for paper strategy experiment analysis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include verbose command-specific details when supported.",
    )
    parser.add_argument(
        "--stale-worker-sec",
        type=float,
        default=120.0,
        help="Stale worker threshold for monitor-paper-strategy-experiment.",
    )
    parser.add_argument(
        "--strategy-id-contains",
        action="append",
        default=None,
        help=(
            "Keep strategies whose strategy_id contains this substring. "
            "Repeat for filter-strategy-config."
        ),
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=1.0,
        help="Polling interval for run-baseline-paper-trader.",
    )
    parser.add_argument(
        "--max-feature-age-sec",
        type=float,
        default=5.0,
        help="Maximum live feature age for run-baseline-paper-trader.",
    )
    parser.add_argument(
        "--live-start-now",
        action="store_true",
        help=(
            "For run-multi-strategy-paper-trader, ignore feature rows older than "
            "the command start time so large recorder DBs do not scan historical features."
        ),
    )
    parser.add_argument(
        "--min-feature-timestamp",
        default=None,
        help=(
            "For run-multi-strategy-paper-trader, ignore feature rows older than this "
            "ISO timestamp."
        ),
    )
    parser.add_argument(
        "--paper-max-iterations",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stale-active-grace-sec",
        type=float,
        default=60.0,
        help="Grace period before stale selected_active markets are demoted.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for recorder writes. Defaults to generated UTC id.",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Start a fresh run by resetting current DB file before recorder starts.",
    )
    parser.add_argument(
        "--archive-old-runs",
        action="store_true",
        help="When used with --fresh-run, archive old DB file instead of deleting it.",
    )
    parser.add_argument(
        "--export-path",
        default="data/training_export.parquet",
        help="Output path for training export parquet.",
    )
    parser.add_argument(
        "--output",
        default="data/exports/training_dataset.parquet",
        help="Output parquet path for export-training-dataset.",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Optional TXT output path for paper-trader-promotion-report.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for analysis/reporting commands.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path for export-training-dataset.",
    )
    parser.add_argument(
        "--export-dir",
        default="data/exports/daily",
        help="Directory for daily-recorder-maintenance parquet exports.",
    )
    parser.add_argument(
        "--archive-dir",
        default="data/archive/sqlite",
        help="Directory for daily-recorder-maintenance SQLite archives.",
    )
    parser.add_argument(
        "--archive-db",
        action="append",
        default=None,
        help=(
            "Recorder archive SQLite DB path or glob for export-recorder-archive-to-parquet. "
            "Repeat to export multiple archives."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/exports/training_dataset.parquet",
        help="Input parquet path for train-baseline-model.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/baseline",
        help="Output directory for train-baseline-model or audit-baseline-model artifacts.",
    )
    parser.add_argument(
        "--output-config-dir",
        default=None,
        help="Output config directory for recommend-next-paper-sweep.",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models/baseline",
        help="Model artifact directory for audit-baseline-model.",
    )
    parser.add_argument(
        "--model-path",
        default="data/models/baseline/model_random_forest.joblib",
        help="Model joblib path for backtest-baseline-strategy.",
    )
    parser.add_argument(
        "--feature-columns",
        default="data/models/baseline/feature_columns.json",
        help="Feature columns JSON path for backtest-baseline-strategy.",
    )
    parser.add_argument(
        "--rf-model-path",
        default=None,
        help="Random Forest model joblib path for run-live-model-predictions.",
    )
    parser.add_argument(
        "--rf-feature-columns",
        default=None,
        help="Random Forest feature_columns.json path for run-live-model-predictions.",
    )
    parser.add_argument(
        "--transformer-model-dir",
        default=None,
        help="Transformer model artifact directory for live model stack commands.",
    )
    parser.add_argument(
        "--scaler-stats",
        default=None,
        help=(
            "Scaler stats JSON path for run-transformer-paper-trader. "
            "Defaults to scaler_stats.json next to --model-path."
        ),
    )
    parser.add_argument(
        "--training-config",
        default=None,
        help=(
            "Training config JSON path for run-transformer-paper-trader. "
            "Defaults to training_config.json next to --model-path."
        ),
    )
    parser.add_argument(
        "--max-btc-age-sec",
        type=float,
        default=3.0,
        help="Maximum canonical BTC sample age for train-baseline-model rows.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum model confidence required before live model display marks YES/NO instead of HOLD.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=48.0,
        help="Resolved-market lookback for score-live-model-predictions.",
    )
    parser.add_argument(
        "--lookback-markets",
        type=int,
        default=100,
        help="Resolved-market lookback count for live-model-dashboard accuracy sections.",
    )
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=2.0,
        help="Terminal refresh interval for live-model-dashboard.",
    )
    parser.add_argument(
        "--shadow-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run live model stack in SHADOW mode by default. Use --no-shadow-only for PAPER mode.",
    )
    parser.add_argument(
        "--enable-live-trading",
        action="store_true",
        help="Explicitly request LIVE mode. No order placement is enabled by default.",
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For start-live-model-stack, run the terminal dashboard alongside prediction/scoring loops.",
    )
    parser.add_argument(
        "--min-settled",
        type=int,
        default=30,
        help="Minimum settled trades for paper-strategy-leaderboard rows.",
    )
    parser.add_argument(
        "--direction",
        choices=("YES", "NO", "yes", "no"),
        default=None,
        help="Optional YES/NO direction filter for paper-strategy-leaderboard.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum filtered rows required before training baseline models.",
    )
    parser.add_argument(
        "--min-markets",
        type=int,
        default=20,
        help="Minimum resolved markets required for research-model-candidates.",
    )
    parser.add_argument(
        "--min-done-trades",
        type=int,
        default=30,
        help="Minimum closed/settled trades required for mine-paper-strategies.",
    )
    parser.add_argument(
        "--min-closed-trades",
        type=int,
        default=None,
        help="Minimum closed/settled trades required for mined strategy candidates.",
    )
    parser.add_argument(
        "--min-roi",
        type=float,
        default=0.0,
        help="Minimum average ROI for mined strategy candidates.",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate for mined strategy candidates.",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Maximum allowed worst-trade drawdown for mined strategy candidates.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum mined strategy candidate configs to generate.",
    )
    parser.add_argument(
        "--allow-duplicate-market-trades",
        action="store_true",
        help="Do not penalize duplicate strategy/market trades in mine-paper-strategies.",
    )
    parser.add_argument(
        "--test-market-fraction",
        type=float,
        default=0.25,
        help="Newest-market test fraction for research-model-candidates.",
    )
    parser.add_argument(
        "--split-by-time",
        action="store_true",
        help="Split transformer threshold backtests by market first prediction timestamp.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Chronological train market ratio for split transformer threshold backtests.",
    )
    parser.add_argument(
        "--min-test-done-trades",
        type=int,
        default=20,
        help="Minimum completed test trades for split transformer threshold validation.",
    )
    parser.add_argument(
        "--validation-market-fraction",
        type=float,
        default=0.2,
        help="Newest training-market validation fraction for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for research and transformer training commands.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Legacy epoch override for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum epochs for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Transformer dropout for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="AdamW weight decay for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=32,
        help="Transformer model width for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Transformer attention heads for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Transformer encoder layer count for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=64,
        help="Transformer feedforward width for train-transformer-sequence-model.",
    )
    parser.add_argument(
        "--positive-class-weight",
        default="auto",
        help="'auto', 'none', or numeric positive class weight for transformer training.",
    )
    parser.add_argument(
        "--research-dir",
        default="data/models/research_latest",
        help="Research output directory for model promotion.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Research model id to promote.",
    )
    parser.add_argument(
        "--model-a-dir",
        default=None,
        help="First model artifact directory for compare-model-artifacts.",
    )
    parser.add_argument(
        "--model-b-dir",
        default=None,
        help="Second model artifact directory for compare-model-artifacts.",
    )
    parser.add_argument(
        "--split-by-market",
        action="store_true",
        help="Use market-level chronological holdout for audit-baseline-model.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological train fraction for audit-baseline-model.",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.65,
        help="Predicted UP probability threshold for buying YES in backtest-baseline-strategy.",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.35,
        help="Predicted UP probability threshold for buying NO in backtest-baseline-strategy.",
    )
    parser.add_argument(
        "--allow-multiple-per-market",
        action="store_true",
        help="Allow more than one baseline backtest trade per run_id/market_id.",
    )
    parser.add_argument(
        "--one-trade-per-market",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open at most one live baseline paper trade per run_id/market_id.",
    )
    parser.add_argument(
        "--min-estimated-edge",
        type=float,
        default=0.0,
        help="Minimum live paper-trader estimated edge required before opening a trade.",
    )
    parser.add_argument(
        "--max-open-trades",
        type=int,
        default=1,
        help="Maximum concurrently open live baseline paper trades.",
    )
    parser.add_argument(
        "--block-probability-above",
        type=float,
        default=None,
        help="Block live paper trades when probability_for_direction is at or above this value.",
    )
    parser.add_argument(
        "--block-probability-below",
        type=float,
        default=None,
        help="Block live paper trades when probability_for_direction is at or below this value.",
    )
    parser.add_argument(
        "--require-fresh-comparison-btc",
        action="store_true",
        help="Require fresh comparison BTC source for live baseline paper trader.",
    )
    parser.add_argument(
        "--stake-usd",
        type=float,
        default=1.0,
        help="Stake per simulated baseline backtest trade.",
    )
    parser.add_argument(
        "--entry-slippage-cents",
        type=float,
        default=None,
        help="Entry slippage in cents for baseline backtest or live baseline paper trader.",
    )
    parser.add_argument(
        "--fee-cents",
        type=float,
        default=0.0,
        help="Per-trade fee in cents applied to baseline backtest entry price.",
    )
    parser.add_argument(
        "--backtest-split",
        choices=("test_only", "full_dataset", "market_holdout"),
        default="test_only",
        help="Dataset slice for backtest-baseline-strategy.",
    )
    parser.add_argument(
        "--allow-suspicious-features",
        action="store_true",
        help="Allow baseline backtest with suspicious/leaky feature columns.",
    )
    parser.add_argument(
        "--min-time-until-resolution-sec",
        default=None,
        help="Minimum seconds before close. Some commands accept comma-separated options.",
    )
    parser.add_argument(
        "--max-time-until-resolution-sec",
        default=None,
        help="Maximum seconds before close. Some commands accept comma-separated options.",
    )
    parser.add_argument(
        "--include-not-ready",
        action="store_true",
        help="Include rows that fail default dataset usability filters.",
    )
    parser.add_argument(
        "--keep-gap-affected",
        action="store_true",
        help="Keep gap-affected rows in export-transformer-sequence-dataset.",
    )
    parser.add_argument(
        "--allow-gap-affected",
        action="store_true",
        help="Allow gap-affected live rows for run-transformer-paper-trader.",
    )
    parser.add_argument(
        "--allow-not-ready-sequence-rows",
        action="store_true",
        help=(
            "Allow older not-ready rows in live transformer sequences when the "
            "latest row is feature_ready=1."
        ),
    )
    parser.add_argument(
        "--min-ready-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum feature_ready ratio for live transformer sequences. "
            "Set below 1.0 to permit partially ready historical sequence rows."
        ),
    )
    parser.add_argument(
        "--sequence-row-policy",
        choices=("latest_rows", "latest_clean_rows"),
        default="latest_rows",
        help=(
            "Live transformer sequence row policy. latest_rows preserves the "
            "current behavior; latest_clean_rows builds from latest clean model-ready rows."
        ),
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=120,
        help="Fixed row count per transformer sequence.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=1.0,
        help="Minimum timestamp stride between transformer sequence starts.",
    )
    parser.add_argument(
        "--label-column",
        default="label_yes_win",
        help="Label column for export-transformer-sequence-dataset.",
    )
    parser.add_argument(
        "--market-id-column",
        default="market_id",
        help="Market id column for export-transformer-sequence-dataset.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        default=None,
        help="Optional market_id filter. Repeat for trade-autopsy.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Timestamp column for export-transformer-sequence-dataset.",
    )
    parser.add_argument(
        "--canonical-btc-source",
        default="polymarket_rtds_chainlink",
        help="Canonical BTC source for export-training-dataset labels and features.",
    )
    parser.add_argument(
        "--comparison-btc-source",
        default="polymarket_rtds_binance",
        help="Comparison BTC source for export-training-dataset features.",
    )
    parser.add_argument(
        "--export-run-id",
        action="append",
        default=None,
        help="Run id to include in export. Repeat to include multiple.",
    )
    parser.add_argument(
        "--merge-runs",
        action="store_true",
        help="Merge multiple run_ids in export (off by default).",
    )
    parser.add_argument(
        "--export-require-trades",
        action="store_true",
        help="Export only rows with aligned trade data available.",
    )
    parser.add_argument(
        "--export-require-full-orderbook",
        action="store_true",
        help="Export only snapshots with complete orderbook depth.",
    )
    parser.add_argument(
        "--export-keep-gap-affected",
        action="store_true",
        help="Keep gap-affected rows in export (default excludes them).",
    )
    parser.add_argument(
        "--label-horizon-sec",
        type=int,
        default=60,
        help="Label horizon metadata for export rows.",
    )
    parser.add_argument(
        "--validate-run-id",
        default=None,
        help="Optional run_id target for validate-db command.",
    )
    parser.add_argument(
        "--dashboard-refresh-sec",
        type=float,
        default=1.0,
        help="Refresh interval for the read-only terminal dashboard.",
    )
    parser.add_argument(
        "--dashboard-once",
        action="store_true",
        help="Render the read-only terminal dashboard once and exit.",
    )
    parser.add_argument(
        "--recent-minutes",
        type=float,
        default=None,
        help="Limit dataset-quality checks to rows from the last N minutes where possible.",
    )
    parser.add_argument(
        "--older-than-minutes",
        type=float,
        default=2.0,
        help="Age threshold for refresh-expired-market-resolutions.",
    )
    parser.add_argument(
        "--resolution-older-than-minutes",
        type=float,
        default=2.0,
        help="Age threshold for live-paper-maintenance-loop resolution refresh.",
    )
    parser.add_argument(
        "--analytics-dir",
        default="data/analytics/live_paper",
        help="Output directory for live-paper-maintenance-loop analytics artifacts.",
    )
    parser.add_argument(
        "--analytics-every-minutes",
        type=float,
        default=5.0,
        help="Analytics interval for live-paper-maintenance-loop.",
    )
    parser.add_argument(
        "--export-every-minutes",
        type=float,
        default=15.0,
        help="Meta-strategy export interval for live-paper-maintenance-loop.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one live-paper-maintenance-loop pass and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for export/maintenance/report commands that support it.",
    )
    parser.add_argument(
        "--status-filter",
        default="closed,settled",
        help="Comma-separated paper trade statuses for trade-autopsy.",
    )
    parser.add_argument(
        "--fixed-horizons-sec",
        default="5,15,30,60",
        help="Comma-separated fixed horizon seconds for trade-autopsy.",
    )
    parser.add_argument(
        "--horizons-sec",
        default=None,
        help="Comma-separated horizon seconds for transformer-prediction-autopsy.",
    )
    parser.add_argument(
        "--exit-horizon-sec",
        default=None,
        help="Comma-separated fixed exit horizons for backtest-transformer-threshold-strategy.",
    )
    parser.add_argument(
        "--side",
        choices=("YES", "NO", "BOTH", "yes", "no", "both"),
        default="BOTH",
        help="Side filter for backtest-transformer-threshold-strategy.",
    )
    parser.add_argument(
        "--side-policy",
        choices=("follow", "fade", "both"),
        default="follow",
        help="Transformer prediction autopsy side policy: follow, fade, or both.",
    )
    parser.add_argument(
        "--min-summary-n",
        type=int,
        default=10,
        help="Minimum rows required for transformer autopsy summary groups.",
    )
    parser.add_argument(
        "--shadow-fixed-horizons-sec",
        default="15,30,60",
        help="Comma-separated fixed horizon seconds for run-transformer-shadow-paper-strategy.",
    )
    parser.add_argument(
        "--shadow-probability-threshold",
        type=float,
        default=0.90,
        help="YES probability threshold for run-transformer-shadow-paper-strategy.",
    )
    parser.add_argument(
        "--stop-loss-take-profit",
        default=None,
        help=(
            "Stop-loss/take-profit policies for trade-autopsy, either JSON or "
            "comma-separated stop:take pairs such as -0.2:0.3,-0.3:0.5."
        ),
    )
    parser.add_argument(
        "--near-close-sec",
        type=float,
        default=3.0,
        help="Seconds before market close for trade-autopsy near-close exits.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print trade-autopsy summary without writing output files.",
    )
    parser.add_argument(
        "--autopsy-input",
        default=None,
        help="Input trade-autopsy parquet for export-meta-trade-dataset.",
    )
    parser.add_argument(
        "--label-policy",
        default="stop_loss_20_take_profit_30",
        help="Autopsy policy to use as labels for export-meta-trade-dataset.",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        default=0.0,
        help="ROI threshold for export-meta-trade-dataset label_roi_above_threshold.",
    )
    parser.add_argument(
        "--progress-every-rows",
        type=int,
        default=50000,
        help="Progress log interval in exported rows for export-training-dataset.",
    )
    parser.add_argument(
        "--progress-every-sec",
        type=float,
        default=10.0,
        help="Progress log interval in seconds for export-training-dataset.",
    )
    parser.add_argument(
        "--export-chunk-size",
        type=int,
        default=10000,
        help="Rows per read/write chunk for export-training-dataset.",
    )
    parser.add_argument(
        "--keep-recent-hours",
        type=float,
        default=12.0,
        help="Retention window for daily-recorder-maintenance cleanup.",
    )
    parser.add_argument(
        "--maintenance-delete-batch-size",
        type=int,
        default=50000,
        help="Rows per delete batch for daily-recorder-maintenance.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply refresh-expired-market-resolutions updates. Default is dry-run.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Run WAL checkpoints around daily-recorder-maintenance cleanup.",
    )
    parser.add_argument(
        "--web-dashboard-host",
        default="127.0.0.1",
        help="Host for the read-only local web dashboard.",
    )
    parser.add_argument(
        "--web-dashboard-port",
        type=int,
        default=8765,
        help="Port for the read-only local web dashboard.",
    )
    parser.add_argument(
        "--web-dashboard-refresh-sec",
        type=float,
        default=1.0,
        help="Browser refresh interval for the read-only local web dashboard.",
    )
    parser.add_argument(
        "--web-dashboard-btc-source",
        default=None,
        help=(
            "Optional BTC source filter for the read-only web dashboard, "
            "for example polymarket_rtds_chainlink."
        ),
    )
    parser.add_argument(
        "--paper-db-glob",
        action="append",
        default=[],
        help=(
            "Glob of paper trader DBs to show in web-dashboard or paper-trader-live-status, for example "
            "--paper-db-glob 'data/paper_trades_*.db'. Recorder-only dashboard works when omitted."
        ),
    )
    parser.add_argument(
        "--limit-activity",
        type=int,
        default=20,
        help="Recent paper trade rows to show for paper-trader-live-status.",
    )
    parser.add_argument(
        "--show-skips",
        action="store_true",
        help="Show recent skipped paper trades in paper-trader-live-status.",
    )
    parser.add_argument(
        "--show-open",
        action="store_true",
        help="Show open and awaiting trades in paper-trader-live-status.",
    )
    parser.add_argument(
        "--show-closed",
        action="store_true",
        help="Show recent closed and settled trades in paper-trader-live-status.",
    )
    parser.add_argument(
        "--watch-sec",
        type=float,
        default=None,
        help="Refresh paper-trader-live-status repeatedly at this interval.",
    )
    parser.add_argument(
        "--btc-source",
        default="polymarket_rtds_chainlink",
        help="BTC RTDS source for probe-btc-rtds.",
    )
    parser.add_argument(
        "--btc-symbol",
        default=None,
        help="BTC RTDS symbol for probe-btc-rtds.",
    )
    parser.add_argument(
        "--btc-rtds-url",
        default="wss://ws-live-data.polymarket.com",
        help="Polymarket RTDS websocket URL for probe-btc-rtds.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Maximum RTDS messages to print for probe-btc-rtds.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Maximum probe runtime in seconds for probe-btc-rtds.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Print only raw RTDS messages for probe-btc-rtds.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("PASSIVE", "FULL", "RESTART", "TRUNCATE", "passive", "full", "restart", "truncate"),
        default=None,
        help=(
            "SQLite WAL checkpoint mode for wal-checkpoint. The command always "
            "runs PASSIVE first; TRUNCATE is optional and can be blocked by long-lived readers."
        ),
    )
    parser.add_argument(
        "--truncate-wal",
        action="store_true",
        help=(
            "After PASSIVE checkpoint, request PRAGMA wal_checkpoint(TRUNCATE). "
            "Never deletes .db-wal manually; long-lived readers can prevent truncation."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="noop",
        help="Paper-backtest strategy name: noop or rule.",
    )
    parser.add_argument(
        "--paper-run-id",
        default=None,
        help=(
            "Run id used in paper-backtest result metadata and output filenames, "
            "or run id filter for report-paper-runs."
        ),
    )
    parser.add_argument(
        "--paper-output-dir",
        default=None,
        help="Optional directory for paper-backtest summary JSON and CSV exports.",
    )
    parser.add_argument(
        "--paper-sort-by",
        choices=("realized_pnl", "ending_cash", "total_fills", "run_id"),
        default="realized_pnl",
        help="Sort field for report-paper-runs.",
    )
    parser.add_argument(
        "--paper-top",
        type=int,
        default=None,
        help="Limit report-paper-runs output to the top N rows.",
    )
    parser.add_argument(
        "--paper-report-json",
        action="store_true",
        help="Print report-paper-runs output as JSON.",
    )
    parser.add_argument(
        "--paper-report-detail",
        action="store_true",
        help="Include per-market and cohort details from saved paper CSV outputs.",
    )
    parser.add_argument(
        "--paper-warnings-only",
        action="store_true",
        help="Show only paper runs with report warnings.",
    )
    parser.add_argument(
        "--paper-strategy-config",
        default=None,
        help="Optional JSON strategy config for --strategy rule.",
    )
    parser.add_argument(
        "--paper-require-labels",
        action="store_true",
        help="Run paper backtest only on rows with settlement labels.",
    )
    parser.add_argument(
        "--paper-require-btc-price",
        action="store_true",
        help="Run paper backtest only on rows with BTC price context.",
    )
    parser.add_argument(
        "--paper-time-window-before-close-sec",
        type=float,
        default=None,
        help="Run paper backtest only on rows within this many seconds before close.",
    )
    parser.add_argument(
        "--paper-starting-cash",
        type=float,
        default=1000.0,
        help="Starting cash for run-paper-backtest.",
    )
    parser.add_argument(
        "--paper-fee-bps",
        type=float,
        default=0.0,
        help="Fee in basis points for run-paper-backtest.",
    )
    parser.add_argument(
        "--paper-fixed-slippage",
        type=float,
        default=0.0,
        help="Fixed per-share price slippage for run-paper-backtest.",
    )
    parser.add_argument(
        "--paper-slippage-bps",
        type=float,
        default=0.0,
        help="Slippage in basis points for run-paper-backtest.",
    )
    parser.add_argument(
        "--paper-fill-mode",
        choices=("top_of_book", "depth_aware"),
        default="top_of_book",
        help="Fill model for run-paper-backtest.",
    )
    parser.add_argument(
        "--paper-max-depth-levels",
        type=int,
        default=None,
        help="Maximum book depth levels to consume in depth-aware paper fills.",
    )
    parser.add_argument(
        "--paper-no-partial-fills",
        action="store_true",
        help="Disable partial fills for depth-aware paper fills.",
    )
    parser.add_argument(
        "--paper-max-position-size-per-market",
        type=float,
        default=None,
        help="Reject paper orders that would exceed this open quantity per market.",
    )
    parser.add_argument(
        "--paper-max-notional-per-order",
        type=float,
        default=None,
        help="Reject paper orders above this notional.",
    )
    parser.add_argument(
        "--paper-max-total-open-notional",
        type=float,
        default=None,
        help="Reject paper buys that would exceed this total open notional.",
    )
    args = parser.parse_args()
    multi_paper_db_commands = {
        "web-dashboard",
        "paper-trader-live-status",
        "trade-autopsy",
        "export-meta-trade-dataset",
    }
    if args.command not in multi_paper_db_commands and isinstance(args.paper_db, list):
        args.paper_db = args.paper_db[-1] if args.paper_db else "data/paper_trades.db"
    return args


def _generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def _single_strategy_id(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[-1]) if value else None
    return str(value)


def _strategy_id_filters(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _paper_db_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _experiment_dir_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _single_experiment_dir(value) -> str | None:
    values = _experiment_dir_values(value)
    return values[-1] if values else None


def _single_float_option(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def _float_csv_option(value, default: str | list[float] | tuple[float, ...]) -> list[float]:
    raw = default if value in (None, "") else value
    if isinstance(raw, str):
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    return [float(item) for item in raw]


def _int_csv_option(value, default: str | list[int] | tuple[int, ...]) -> list[int]:
    raw = default if value in (None, "") else value
    if isinstance(raw, str):
        return [int(float(part.strip())) for part in raw.split(",") if part.strip()]
    return [int(item) for item in raw]


def _single_prediction_db(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[-1]) if value else None
    return str(value)


def _cli_option_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def _single_paper_db(value, default: str = "data/paper_trades.db") -> str:
    values = _paper_db_values(value)
    return values[-1] if values else default


def _paper_db_paths_for_promotion(args: argparse.Namespace) -> list[str]:
    paths = [str(path) for path in (args.paper_db_path or []) if str(path).strip()]
    if not paths:
        paths = _paper_db_values(args.paper_db)
    if not paths:
        paths = ["data/paper_trades.db"]
    return paths


def _prepare_fresh_run_db(db_path: str, archive_old_runs: bool) -> None:
    base = Path(db_path)
    if not base.exists():
        return
    if archive_old_runs:
        archive_name = (
            f"{base.stem}.archive.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{base.suffix}"
        )
        archive_path = base.with_name(archive_name)
        shutil.move(str(base), str(archive_path))
    else:
        base.unlink(missing_ok=True)

    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{suffix}")
        sidecar.unlink(missing_ok=True)


def _user_table_names(db_path: str) -> set[str]:
    path = Path(db_path)
    if not path.exists():
        return set()
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def _table_row_counts(db_path: str, tables: tuple[str, ...]) -> dict[str, int | None]:
    path = Path(db_path)
    if not path.exists():
        return {table: None for table in tables}
    conn = sqlite3.connect(str(path))
    try:
        existing = _user_table_names(db_path)
        counts: dict[str, int | None] = {}
        for table in tables:
            if table not in existing:
                counts[table] = None
                continue
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
        return counts
    finally:
        conn.close()


def _upgrade_db_schema(settings, run_id: str | None = None) -> dict[str, object]:
    before_tables = _user_table_names(settings.db_path)
    before_counts = _table_row_counts(settings.db_path, SCHEMA_UPGRADE_PRESERVE_TABLES)
    db = SQLiteStore(
        db_path=settings.db_path,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        run_id=run_id or "legacy",
        quarantine_on_startup=False,
    )
    try:
        db.init_schema(
            backfill_existing_rows=False,
            dedupe_lifecycle_events=False,
            recover_malformed=False,
        )
    finally:
        db.close()

    after_tables = _user_table_names(settings.db_path)
    after_counts = _table_row_counts(settings.db_path, SCHEMA_UPGRADE_PRESERVE_TABLES)
    preserved_counts = {
        table: {"before": before_counts.get(table), "after": after_counts.get(table)}
        for table in SCHEMA_UPGRADE_PRESERVE_TABLES
        if before_counts.get(table) is not None
    }
    return {
        "status": "ok",
        "action": "schema_upgraded",
        "db_path": settings.db_path,
        "created_tables": sorted(after_tables - before_tables),
        "target_tables_present": {
            table: table in after_tables for table in SCHEMA_UPGRADE_TARGET_TABLES
        },
        "preserved_row_counts": preserved_counts,
        "schema_only": True,
    }


def _print_debug_discovery_report(report: dict) -> None:
    stage_counts = report.get("stage_counts", {})
    print("=== Discovery Debug ===")
    print(f"now_utc={report.get('now_utc')}")
    print("stage_counts:")
    for key in (
        "raw_markets_returned",
        "matching_btc_bitcoin",
        "matching_up_or_down",
        "matching_approx_5m_duration",
        "with_token_ids_present",
        "after_phase_classification",
        "final_selected",
    ):
        print(f"  {key}={stage_counts.get(key, 0)}")

    print(f"phase_counts={report.get('phase_counts', {})}")
    print(f"rejection_counts={report.get('rejection_counts', {})}")
    print(f"discovery_source={report.get('discovery_source')}")
    print(f"strict_candidate_count={report.get('strict_candidate_count')}")
    print(f"strict_phase_eligible_count={report.get('strict_phase_eligible_count')}")
    print(f"broad_candidate_count={report.get('broad_candidate_count')}")
    print(f"broad_phase_eligible_count={report.get('broad_phase_eligible_count')}")
    print(f"candidate_pool_count={report.get('candidate_pool_count')}")
    print(f"fallback_mode={report.get('fallback_mode')}")
    print(f"fallback_reason={report.get('fallback_reason')}")
    print(f"selected_market_preview={report.get('selected_market_preview')}")

    print("\nhttp_pages:")
    pages = report.get("pages", [])
    if not pages:
        print("  none")
    for page in pages:
        print(
            "  "
            + " | ".join(
                [
                    f"page_index={page.get('page_index')}",
                    f"status_code={page.get('status_code')}",
                    f"request_url={page.get('request_url')}",
                    f"response_size_bytes={page.get('response_size_bytes')}",
                    f"top_level_type={page.get('top_level_type')}",
                    f"top_level_keys={page.get('top_level_keys')}",
                    f"extraction_path={page.get('extraction_path')}",
                    f"raw_item_count={page.get('raw_item_count')}",
                    f"normalized_market_count={page.get('normalized_market_count')}",
                ]
            )
        )

    print("\nfirst_raw_items:")
    raw_samples = report.get("raw_samples", [])
    if not raw_samples:
        print("  none")
    for sample in raw_samples[:5]:
        print(f"  {sample}")

    print("\nfirst_candidate_decisions:")
    candidate_details = report.get("candidate_details", [])
    if not candidate_details:
        print("  none")
    for row in candidate_details:
        print(f"  {row}")

    print("\nstrict_matched_btc_candidates:")
    strict_candidates = report.get("strict_matched_candidates", [])
    if not strict_candidates:
        print("  none")
    for row in strict_candidates:
        print(f"  {row}")


async def _run_debug_discovery(settings) -> None:
    client = GammaClient(
        base_url=settings.gamma_api_url,
        timeout_sec=settings.request_timeout_sec,
        page_size=settings.discovery_page_size,
        max_pages=settings.discovery_max_pages,
        lookahead_sec=settings.discovery_lookahead_sec,
    )
    try:
        report = await client.debug_discovery_once()
    finally:
        await client.close()
    _print_debug_discovery_report(report)


def main() -> None:
    args = parse_args()
    if args.command == "probe-btc-rtds":
        from .btc_price_feed import btc_price_feed_symbol_for_source, probe_polymarket_rtds_btc_feed

        symbol = args.btc_symbol or btc_price_feed_symbol_for_source(
            str(args.btc_source).strip().lower(),
            "btc/usd",
        )
        asyncio.run(
            probe_polymarket_rtds_btc_feed(
                source=args.btc_source,
                symbol=symbol,
                ws_url=args.btc_rtds_url,
                max_messages=args.max_messages,
                timeout_sec=args.timeout_sec,
                raw_only=args.raw_only,
            )
        )
        return
    if args.command == "inspect-backtest-dataset":
        from .backtest_dataset import inspect_backtest_dataset

        print(inspect_backtest_dataset(args.export_path))
        return
    if args.command == "run-paper-backtest":
        from .paper_trader import (
            PaperTraderConfig,
            export_paper_backtest_result,
            result_summary_to_dict,
            run_paper_backtest_from_export,
        )

        result = run_paper_backtest_from_export(
            args.export_path,
            strategy_name=args.strategy,
            strategy_config_path=args.paper_strategy_config,
            require_labels=args.paper_require_labels,
            require_btc_price=args.paper_require_btc_price,
            time_window_before_close_sec=args.paper_time_window_before_close_sec,
            config=PaperTraderConfig(
                run_id=args.paper_run_id or "paper_run",
                starting_cash=args.paper_starting_cash,
                fee_bps=args.paper_fee_bps,
                fixed_slippage=args.paper_fixed_slippage,
                slippage_bps=args.paper_slippage_bps,
                fill_mode=args.paper_fill_mode,
                max_depth_levels=args.paper_max_depth_levels,
                allow_partial_fills=not args.paper_no_partial_fills,
                max_position_size_per_market=args.paper_max_position_size_per_market,
                max_notional_per_order=args.paper_max_notional_per_order,
                max_total_open_notional=args.paper_max_total_open_notional,
            ),
        )
        summary = result_summary_to_dict(result)
        if args.paper_output_dir:
            summary["output_files"] = export_paper_backtest_result(
                result,
                args.paper_output_dir,
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "report-paper-runs":
        from .paper_trader import build_paper_runs_report, render_paper_runs_report

        output_dir = args.paper_output_dir or "data/paper_runs"
        report = build_paper_runs_report(
            output_dir,
            sort_by=args.paper_sort_by,
            top=args.paper_top,
            run_id=args.paper_run_id,
            include_detail=args.paper_report_detail,
            warnings_only=args.paper_warnings_only,
        )
        if args.paper_report_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_paper_runs_report(report))
        return
    if args.command == "train-baseline-model":
        from .train_baseline_model import BaselineTrainingError, train_baseline_model

        try:
            report = train_baseline_model(
                input_path=args.input,
                output_dir=args.output_dir,
                max_btc_age_sec=args.max_btc_age_sec,
                min_rows=args.min_rows,
            )
        except BaselineTrainingError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "input_path": args.input,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "train-transformer-sequence-model":
        from .train_transformer_sequence_model import (
            TransformerTrainingError,
            train_transformer_sequence_model,
        )

        try:
            report = train_transformer_sequence_model(
                input_path=args.input,
                output_dir=args.output_dir,
                label_column=args.label_column,
                test_market_fraction=args.test_market_fraction,
                validation_market_fraction=args.validation_market_fraction,
                epochs=args.epochs,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_seed=args.random_seed,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                patience=args.patience,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                positive_class_weight=args.positive_class_weight,
            )
        except TransformerTrainingError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "input_path": args.input,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "audit-baseline-model":
        from .baseline_model_audit import BaselineAuditError, audit_baseline_model

        try:
            report = audit_baseline_model(
                input_path=args.input,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                max_btc_age_sec=args.max_btc_age_sec,
                split_by_market=args.split_by_market,
                train_fraction=args.train_fraction,
            )
        except BaselineAuditError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "input_path": args.input,
                "model_dir": args.model_dir,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "audit-live-model-predictions":
        from .live_model_audit import LiveModelAuditError, audit_live_model_predictions

        output_path = (
            "data/analytics/model_audit/latest_model_audit.json"
            if args.output == "data/exports/training_dataset.parquet"
            else args.output
        )
        try:
            report = audit_live_model_predictions(
                db_path=args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                output_path=output_path,
                output_csv_path=args.output_csv,
            )
        except LiveModelAuditError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "db_path": args.db,
                "model_path": args.model_path,
                "feature_columns_path": args.feature_columns,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "research-model-candidates":
        from .model_research import ModelResearchError, research_model_candidates

        try:
            report = research_model_candidates(
                db_path=args.db,
                output_dir=args.output_dir,
                min_markets=args.min_markets,
                test_market_fraction=args.test_market_fraction,
                random_seed=args.random_seed,
            )
        except ModelResearchError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "db_path": args.db,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "promote-research-model-candidate":
        from .model_research import ModelResearchError, promote_research_model_candidate

        try:
            if not args.model_id:
                raise ModelResearchError("promote-research-model-candidate requires --model-id")
            if args.output_dir == "data/models/baseline":
                raise ModelResearchError(
                    "promote-research-model-candidate requires an explicit --output-dir "
                    "for the candidate artifact directory"
                )
            report = promote_research_model_candidate(
                research_dir=args.research_dir,
                model_id=args.model_id,
                output_dir=args.output_dir,
                force=args.force,
            )
        except ModelResearchError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "research_dir": args.research_dir,
                "model_id": args.model_id,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "compare-model-artifacts":
        from .model_research import (
            ModelResearchError,
            compare_model_artifacts,
            render_model_artifact_comparison,
        )

        output_path = (
            "data/analytics/model_compare/latest_compare.json"
            if args.output == "data/exports/training_dataset.parquet"
            else args.output
        )
        try:
            if not args.model_a_dir:
                raise ModelResearchError("compare-model-artifacts requires --model-a-dir")
            if not args.model_b_dir:
                raise ModelResearchError("compare-model-artifacts requires --model-b-dir")
            report = compare_model_artifacts(
                model_a_dir=args.model_a_dir,
                model_b_dir=args.model_b_dir,
                db_path=args.db,
                output_path=output_path,
            )
        except ModelResearchError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "db_path": args.db,
                "model_a_dir": args.model_a_dir,
                "model_b_dir": args.model_b_dir,
                "output_path": output_path,
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        print(render_model_artifact_comparison(report), end="")
        return
    if args.command == "backtest-baseline-strategy":
        from .backtest_baseline_strategy import (
            BaselineBacktestError,
            backtest_baseline_strategy,
        )

        try:
            report = backtest_baseline_strategy(
                input_path=args.input,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                output_dir=args.output_dir,
                max_btc_age_sec=args.max_btc_age_sec,
                long_threshold=args.long_threshold,
                short_threshold=args.short_threshold,
                allow_multiple_per_market=args.allow_multiple_per_market,
                stake_usd=args.stake_usd,
                entry_slippage_cents=(
                    0.0 if args.entry_slippage_cents is None else args.entry_slippage_cents
                ),
                fee_cents=args.fee_cents,
                backtest_split=args.backtest_split,
                allow_suspicious_features=args.allow_suspicious_features,
            )
        except BaselineBacktestError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "input_path": args.input,
                "model_path": args.model_path,
                "feature_columns_path": args.feature_columns,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-baseline-paper-trader":
        from .baseline_paper_trader import (
            BaselinePaperTraderConfig,
            BaselinePaperTraderError,
            run_baseline_paper_trader,
        )

        try:
            report = run_baseline_paper_trader(
                BaselinePaperTraderConfig(
                    recorder_db_path=args.db,
                    output_db_path=args.output_db,
                    model_path=args.model_path,
                    feature_columns_path=args.feature_columns,
                    strategy_id=_single_strategy_id(args.strategy_id),
                    strategy_name=args.strategy_name,
                    poll_sec=args.poll_sec,
                    long_threshold=args.long_threshold,
                    short_threshold=args.short_threshold,
                    max_btc_age_sec=args.max_btc_age_sec,
                    max_feature_age_sec=args.max_feature_age_sec,
                    min_time_until_resolution_sec=(
                        0.0
                        if args.min_time_until_resolution_sec is None
                        else _single_float_option(args.min_time_until_resolution_sec)
                    ),
                    max_time_until_resolution_sec=(
                        300.0
                        if args.max_time_until_resolution_sec is None
                        else _single_float_option(args.max_time_until_resolution_sec)
                    ),
                    one_trade_per_market=args.one_trade_per_market,
                    min_estimated_edge=args.min_estimated_edge,
                    max_open_trades=args.max_open_trades,
                    block_probability_above=args.block_probability_above,
                    block_probability_below=args.block_probability_below,
                    require_fresh_comparison_btc=args.require_fresh_comparison_btc,
                    stake_usd=args.stake_usd,
                    entry_slippage_cents=(
                        0.01
                        if args.entry_slippage_cents is None
                        else args.entry_slippage_cents
                    ),
                    fee_cents=args.fee_cents,
                    dry_run=True if args.dry_run is None else bool(args.dry_run),
                ),
                max_iterations=args.paper_max_iterations,
            )
        except BaselinePaperTraderError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "recorder_db_path": args.db,
                "output_db_path": args.output_db,
            }
        if args.paper_max_iterations is not None:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "paper-trader-summary":
        from .baseline_paper_trader import build_paper_trader_summary

        print(
            json.dumps(
                build_paper_trader_summary(_single_paper_db(args.paper_db)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "paper-trader-analytics":
        from .paper_trader_analytics import build_paper_trader_analytics_report

        report = build_paper_trader_analytics_report(
            paper_db_path=_single_paper_db(args.paper_db),
            output_dir=args.output_dir,
            strategy_id=_single_strategy_id(args.strategy_id),
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return
    if args.command == "paper-trader-live-status":
        from .paper_trader_live_status import run_paper_trader_live_status

        run_paper_trader_live_status(
            recorder_db_path=args.recorder_db or args.db,
            paper_db_paths=args.paper_db,
            paper_db_globs=args.paper_db_glob,
            limit_activity=args.limit_activity,
            show_skips=args.show_skips,
            show_open=args.show_open,
            show_closed=args.show_closed,
            watch_sec=args.watch_sec,
        )
        return
    if args.command == "compact-paper-live-status":
        from .compact_paper_live_status import (
            build_compact_paper_live_status,
            render_compact_paper_live_status,
        )

        report = build_compact_paper_live_status(
            _single_paper_db(args.paper_db),
            limit=5 if args.limit is None else args.limit,
            verbose=bool(args.verbose),
        )
        print(render_compact_paper_live_status(report), end="")
        return
    if args.command == "overnight-health-report":
        from .overnight_health_report import (
            build_overnight_health_report,
            render_overnight_health_report,
            write_overnight_health_report_json,
        )

        report = build_overnight_health_report(
            recorder_db_path=args.recorder_db or args.db,
            prediction_db_inputs=args.prediction_db or [],
            paper_db_paths=_paper_db_values(args.paper_db),
            paper_db_globs=args.paper_db_glob or [],
        )
        if args.output_json:
            write_overnight_health_report_json(report, args.output_json)
        print(render_overnight_health_report(report), end="")
        return
    if args.command == "paper-strategy-leaderboard":
        from .paper_strategy_leaderboard import (
            build_paper_strategy_leaderboard,
            render_paper_strategy_leaderboard,
        )

        try:
            report = build_paper_strategy_leaderboard(
                paper_db_path=_single_paper_db(args.paper_db),
                min_settled=args.min_settled,
                direction=args.direction,
                strategy_ids=_strategy_id_filters(args.strategy_id),
                output_path=(
                    None
                    if args.output == "data/exports/training_dataset.parquet"
                    else args.output
                ),
                output_csv_path=args.output_csv,
            )
        except ValueError as exc:
            report = {"status": "error", "error": str(exc)}
        print(render_paper_strategy_leaderboard(report))
        return
    if args.command == "paper-trader-promotion-report":
        from .paper_trader_promotion_report import (
            build_paper_trader_promotion_report,
            render_paper_trader_promotion_report,
        )

        output_path = (
            "data/analytics/paper_promotion_report.json"
            if args.output == "data/exports/training_dataset.parquet"
            else args.output
        )
        output_txt_path = (
            args.output_txt
            if args.output_txt
            else str(Path(output_path).with_suffix(".txt"))
        )
        try:
            report = build_paper_trader_promotion_report(
                paper_db_paths=_paper_db_paths_for_promotion(args),
                recorder_db_path=args.recorder_db or args.db,
                output_path=output_path,
                output_txt_path=output_txt_path,
            )
        except ValueError as exc:
            report = {"status": "error", "error": str(exc)}
        print(render_paper_trader_promotion_report(report))
        return
    if args.command == "paper-trade-resolution-autopsy":
        from .paper_trade_resolution_autopsy import (
            build_paper_trade_resolution_autopsy,
            render_paper_trade_resolution_autopsy,
        )

        recorder_db_path = args.recorder_db or args.db
        try:
            report = build_paper_trade_resolution_autopsy(
                recorder_db_path=recorder_db_path,
                paper_db_path=_single_paper_db(args.paper_db),
            )
        except FileNotFoundError as exc:
            report = {"status": "error", "error": f"db_not_found: {exc}"}
        print(render_paper_trade_resolution_autopsy(report))
        return
    if args.command == "trade-autopsy":
        from .trade_autopsy import (
            TradeAutopsyError,
            build_trade_autopsy_report,
            parse_fixed_horizons,
            parse_stop_loss_take_profit,
            resolve_paper_db_paths,
        )

        paper_db_paths = resolve_paper_db_paths(
            paper_db_values=_paper_db_values(args.paper_db),
            paper_db_globs=args.paper_db_glob,
        )
        if not paper_db_paths:
            report = {"status": "error", "error": "paper_db_required"}
        else:
            output_path = args.output
            if output_path == "data/exports/training_dataset.parquet":
                output_path = "data/analytics/trade_autopsy.parquet"
            try:
                report = build_trade_autopsy_report(
                    recorder_db_path=args.recorder_db or args.db,
                    paper_db_paths=paper_db_paths,
                    output_path=None if args.summary_only or args.dry_run else output_path,
                    output_csv_path=None if args.summary_only or args.dry_run else args.output_csv,
                    status_filter=args.status_filter,
                    strategy_ids=_strategy_id_filters(args.strategy_id),
                    market_ids=_strategy_id_filters(args.market_id),
                    limit=args.limit,
                    fixed_horizons_sec=parse_fixed_horizons(args.fixed_horizons_sec),
                    stop_loss_take_profit=parse_stop_loss_take_profit(args.stop_loss_take_profit),
                    near_close_sec=args.near_close_sec,
                    include_skipped=args.include_skipped,
                    summary_only=bool(args.summary_only or args.dry_run),
                )
            except (TradeAutopsyError, FileNotFoundError, RuntimeError, ValueError) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "transformer-prediction-autopsy":
        from .transformer_prediction_autopsy import (
            TransformerPredictionAutopsyError,
            build_transformer_prediction_autopsy_report,
            parse_horizons,
            resolve_prediction_db_paths,
        )

        prediction_db_paths = resolve_prediction_db_paths(args.prediction_db)
        if not prediction_db_paths:
            report = {"status": "error", "error": "prediction_db_required"}
        else:
            output_path = args.output
            if output_path == "data/exports/training_dataset.parquet":
                output_path = "data/analytics/transformer_prediction_autopsy.parquet"
            horizons_value = args.horizons_sec or args.fixed_horizons_sec
            try:
                report = build_transformer_prediction_autopsy_report(
                    recorder_db_path=args.recorder_db or args.db,
                    prediction_db_paths=prediction_db_paths,
                    output_path=None if args.dry_run else output_path,
                    output_csv_path=None if args.dry_run else args.output_csv,
                    horizons_sec=parse_horizons(horizons_value),
                    near_close_sec=args.near_close_sec,
                    side_policy=args.side_policy,
                    min_summary_n=args.min_summary_n,
                )
            except (
                TransformerPredictionAutopsyError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "report-transformer-predictions":
        from .transformer_live_inference import report_transformer_predictions
        from .transformer_prediction_autopsy import resolve_prediction_db_paths

        prediction_inputs = args.prediction_db
        if not prediction_inputs:
            if _cli_option_provided("--output-db"):
                prediction_inputs = [args.output_db]
            else:
                prediction_inputs = [args.db]
        prediction_db_paths = resolve_prediction_db_paths(prediction_inputs)
        report = report_transformer_predictions(prediction_db_paths)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "transformer-calibration-analysis":
        from .transformer_live_inference import analyze_transformer_calibration
        from .transformer_prediction_autopsy import resolve_prediction_db_paths

        prediction_db_paths = resolve_prediction_db_paths(args.prediction_db)
        jsonl_log_paths = resolve_prediction_db_paths(args.jsonl_log)
        if not prediction_db_paths and not jsonl_log_paths:
            report = {
                "status": "error",
                "error": "prediction_db_or_jsonl_log_required",
            }
        else:
            output_json = args.output_json
            output_txt = args.output_txt
            try:
                report = analyze_transformer_calibration(
                    prediction_db_paths=prediction_db_paths,
                    jsonl_log_paths=jsonl_log_paths,
                    recorder_db_path=args.recorder_db or args.db,
                    output_json_path=output_json,
                    output_txt_path=output_txt,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "backtest-transformer-threshold-strategy":
        from .transformer_live_inference import backtest_transformer_threshold_strategy
        from .transformer_prediction_autopsy import resolve_prediction_db_paths

        prediction_db_paths = resolve_prediction_db_paths(args.prediction_db)
        if not prediction_db_paths:
            report = {"status": "error", "error": "prediction_db_required"}
        else:
            try:
                report = backtest_transformer_threshold_strategy(
                    prediction_db_paths=prediction_db_paths,
                    recorder_db_path=args.recorder_db or args.db,
                    output_json_path=args.output_json,
                    output_txt_path=args.output_txt,
                    thresholds=_float_csv_option(
                        args.thresholds,
                        "0.75,0.77,0.775,0.78,0.79,0.80",
                    ),
                    min_time_until_resolution_options=_float_csv_option(
                        args.min_time_until_resolution_sec,
                        "30,60,90,120",
                    ),
                    max_time_until_resolution_options=_float_csv_option(
                        args.max_time_until_resolution_sec,
                        "90,120,180,240",
                    ),
                    exit_horizon_sec_options=_int_csv_option(
                        args.exit_horizon_sec,
                        "15,30,60",
                    ),
                    side=args.side,
                    one_trade_per_market=bool(args.one_trade_per_market),
                    slippage_cents=(
                        1.0
                        if args.entry_slippage_cents is None
                        else float(args.entry_slippage_cents)
                    ),
                    fee_cents=float(args.fee_cents),
                    min_done_trades=int(args.min_done_trades),
                    split_by_time=bool(args.split_by_time),
                    train_ratio=float(args.train_ratio),
                    min_test_done_trades=int(args.min_test_done_trades),
                    strategy_filter=args.strategy_filter,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-transformer-shadow-paper-strategy":
        from .transformer_prediction_autopsy import parse_horizons
        from .transformer_shadow_paper_trader import (
            TransformerShadowPaperTraderError,
            run_transformer_shadow_paper_strategy,
        )

        try:
            report = run_transformer_shadow_paper_strategy(
                recorder_db_path=args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                scaler_stats_path=args.scaler_stats,
                training_config_path=args.training_config,
                output_db_path=args.output_db,
                run_id=args.run_id,
                sequence_length=args.sequence_length,
                sequence_row_policy=args.sequence_row_policy,
                poll_sec=args.poll_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                probability_threshold=args.shadow_probability_threshold,
                min_time_until_resolution_sec=(
                    30.0
                    if args.min_time_until_resolution_sec is None
                    else _single_float_option(args.min_time_until_resolution_sec)
                ),
                max_time_until_resolution_sec=(
                    120.0
                    if args.max_time_until_resolution_sec is None
                    else _single_float_option(args.max_time_until_resolution_sec)
                ),
                fixed_horizons_sec=parse_horizons(args.shadow_fixed_horizons_sec),
                max_iterations=args.paper_max_iterations,
            )
        except (TransformerShadowPaperTraderError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "model_path": args.model_path,
                "feature_columns_path": args.feature_columns,
            }
        if args.paper_max_iterations is not None or report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "export-meta-trade-dataset":
        from .trade_autopsy import (
            TradeAutopsyError,
            export_meta_trade_dataset,
            parse_fixed_horizons,
            parse_stop_loss_take_profit,
            resolve_paper_db_paths,
        )

        paper_db_paths = resolve_paper_db_paths(
            paper_db_values=_paper_db_values(args.paper_db),
            paper_db_globs=args.paper_db_glob,
        )
        label_column = (
            "label_positive_roi"
            if args.label_column == "label_yes_win"
            else args.label_column
        )
        output_path = args.output
        if output_path == "data/exports/training_dataset.parquet":
            output_path = "data/exports/meta_trade_dataset.parquet"
        try:
            report = export_meta_trade_dataset(
                autopsy_input_path=args.autopsy_input,
                output_path=output_path,
                output_csv_path=args.output_csv,
                label_policy=args.label_policy,
                label_column=label_column,
                roi_threshold=args.roi_threshold,
                recorder_db_path=args.recorder_db or args.db,
                paper_db_paths=paper_db_paths,
                status_filter=args.status_filter,
                fixed_horizons_sec=parse_fixed_horizons(args.fixed_horizons_sec),
                stop_loss_take_profit=parse_stop_loss_take_profit(args.stop_loss_take_profit),
                near_close_sec=args.near_close_sec,
            )
        except (TradeAutopsyError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "export-paper-trader-dataset":
        from .export_paper_trader_dataset import export_paper_trader_dataset

        try:
            report = export_paper_trader_dataset(
                paper_db_path=_single_paper_db(args.paper_db),
                strategy_id=_single_strategy_id(args.strategy_id),
                strategy_name=args.strategy_name,
                output_dir=args.output_dir,
                append=args.append,
                include_skipped=args.include_skipped,
                include_awaiting=args.include_awaiting,
                notes=args.notes,
            )
        except ValueError as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "export-meta-strategy-dataset":
        from .export_meta_strategy_dataset import export_meta_strategy_dataset

        try:
            report = export_meta_strategy_dataset(
                paper_db_path=_single_paper_db(args.paper_db),
                output_path=args.output,
                output_csv_path=args.output_csv,
                include_unresolved=args.include_unresolved,
                strategy_id=_single_strategy_id(args.strategy_id),
                min_created_at=args.min_created_at,
                max_created_at=args.max_created_at,
            )
        except ValueError as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-transformer-prediction-paper-strategy":
        from .transformer_shadow_paper_trader import (
            TransformerShadowPaperTraderError,
            run_transformer_prediction_paper_strategy,
        )

        prediction_db = _single_prediction_db(args.prediction_db)
        if not prediction_db:
            report = {"status": "error", "error": "prediction_db_required"}
        else:
            try:
                report = run_transformer_prediction_paper_strategy(
                    recorder_db_path=args.db,
                    prediction_db_path=prediction_db,
                    config_path=args.config,
                    output_db_path=args.output_db,
                    run_id=args.run_id,
                    poll_sec=args.poll_sec,
                    max_iterations=args.paper_max_iterations,
                )
            except (
                TransformerShadowPaperTraderError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "backfill-transformer-prediction-paper-pnl":
        from .transformer_shadow_paper_trader import (
            TransformerShadowPaperTraderError,
            backfill_transformer_prediction_paper_pnl,
        )

        try:
            report = backfill_transformer_prediction_paper_pnl(
                recorder_db_path=args.recorder_db or args.db,
                paper_db_path=_single_paper_db(args.paper_db),
                dry_run=bool(args.dry_run) and not bool(args.apply),
            )
        except (
            TransformerShadowPaperTraderError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-transformer-paper-trader":
        from .transformer_live_inference import (
            TransformerLiveInferenceError,
            run_transformer_paper_trader,
        )

        try:
            report = run_transformer_paper_trader(
                recorder_db_path=args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                scaler_stats_path=args.scaler_stats,
                training_config_path=args.training_config,
                sequence_length=args.sequence_length,
                poll_sec=args.poll_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                allow_gap_affected=args.allow_gap_affected,
                allow_not_ready_sequence_rows=args.allow_not_ready_sequence_rows,
                min_ready_ratio=args.min_ready_ratio,
                sequence_row_policy=args.sequence_row_policy,
                diagnostics=args.diagnostics,
                max_iterations=args.paper_max_iterations,
                output_db_path=args.output_db if _cli_option_provided("--output-db") else None,
                log_file_path=args.log_file,
                jsonl_log_file_path=args.jsonl_log_file,
                heartbeat_log_file_path=args.heartbeat_log_file,
            )
        except TransformerLiveInferenceError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "model_path": args.model_path,
                "feature_columns_path": args.feature_columns,
            }
        if args.paper_max_iterations is not None or report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-paper-strategy-experiment":
        from .paper_strategy_experiment import (
            PaperStrategyExperimentError,
            run_paper_strategy_experiment,
        )

        experiment_output_dir = (
            args.output_dir
            if _cli_option_provided("--output-dir")
            else None
        )
        try:
            report = run_paper_strategy_experiment(
                recorder_db_path=args.recorder_db or args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                config_glob=args.config_glob or args.config,
                experiment_id=args.experiment_id,
                output_dir=experiment_output_dir,
                poll_sec=args.poll_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                heartbeat_detail=args.heartbeat_detail,
                prediction_db_path=_single_prediction_db(args.prediction_db),
                dry_run=bool(args.dry_run),
                use_tmux=bool(args.tmux),
            )
        except (PaperStrategyExperimentError, OSError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "report-paper-strategy-experiment":
        from .paper_strategy_experiment import (
            PaperStrategyExperimentError,
            build_paper_strategy_experiment_report,
        )

        experiment_dir = _single_experiment_dir(args.experiment_dir) or (
            args.output_dir if _cli_option_provided("--output-dir") else None
        )
        if not experiment_dir:
            report = {"status": "error", "error": "experiment_dir_required"}
        else:
            output_path = args.output
            if output_path == "data/exports/training_dataset.parquet":
                output_path = str(Path(experiment_dir) / "report.json")
            output_txt = args.output_txt or str(Path(experiment_dir) / "report.txt")
            try:
                report = build_paper_strategy_experiment_report(
                    experiment_dir=experiment_dir,
                    recorder_db_path=args.recorder_db or args.db,
                    output_path=output_path,
                    output_txt_path=output_txt,
                )
            except (PaperStrategyExperimentError, FileNotFoundError, RuntimeError, ValueError) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "monitor-paper-strategy-experiment":
        from .paper_strategy_experiment import (
            PaperStrategyExperimentError,
            monitor_paper_strategy_experiment,
            render_paper_strategy_experiment_monitor,
        )

        experiment_dir = _single_experiment_dir(args.experiment_dir) or (
            args.output_dir if _cli_option_provided("--output-dir") else None
        )
        if not experiment_dir:
            report = {"status": "error", "error": "experiment_dir_required"}
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        try:
            report = monitor_paper_strategy_experiment(
                experiment_dir=experiment_dir,
                stale_after_sec=args.stale_worker_sec,
            )
        except (PaperStrategyExperimentError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        print(render_paper_strategy_experiment_monitor(report), end="")
        return
    if args.command == "analyze-paper-strategy-experiment":
        from .paper_strategy_experiment_analysis import (
            PaperStrategyExperimentAnalysisError,
            analyze_paper_strategy_experiment,
        )

        experiment_dir = _single_experiment_dir(args.experiment_dir) or (
            args.output_dir if _cli_option_provided("--output-dir") else None
        )
        if not experiment_dir:
            report = {"status": "error", "error": "experiment_dir_required"}
        else:
            output_json = args.output_json or str(Path(experiment_dir) / "analysis.json")
            output_txt = args.output_txt or str(Path(experiment_dir) / "analysis.txt")
            try:
                report = analyze_paper_strategy_experiment(
                    experiment_dir=experiment_dir,
                    output_json_path=output_json,
                    output_txt_path=output_txt,
                    verbose_txt=bool(args.verbose_txt),
                )
            except (
                PaperStrategyExperimentAnalysisError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "recommend-next-paper-sweep":
        from .paper_strategy_experiment_analysis import (
            PaperStrategyExperimentAnalysisError,
            recommend_next_paper_sweep,
        )

        experiment_dir = _single_experiment_dir(args.experiment_dir) or (
            args.output_dir if _cli_option_provided("--output-dir") else None
        )
        output_config_dir = args.output_config_dir
        if not experiment_dir:
            report = {"status": "error", "error": "experiment_dir_required"}
        else:
            if output_config_dir is None:
                output_config_dir = str(Path(experiment_dir) / "recommended_next_sweep")
            try:
                report = recommend_next_paper_sweep(
                    experiment_dir=experiment_dir,
                    output_config_dir=output_config_dir,
                )
            except (
                PaperStrategyExperimentAnalysisError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "mine-paper-strategies":
        from .paper_strategy_experiment_analysis import (
            PaperStrategyExperimentAnalysisError,
            mine_paper_strategies,
        )

        experiment_dirs = _experiment_dir_values(args.experiment_dir)
        paper_db_inputs = _paper_db_values(args.paper_db)
        if not experiment_dirs and not paper_db_inputs and not args.paper_db_glob and not args.backtest_json_glob:
            report = {
                "status": "error",
                "error": "experiment_dir_or_paper_db_or_backtest_json_required",
            }
        else:
            try:
                report = mine_paper_strategies(
                    experiment_dirs=experiment_dirs,
                    paper_db_paths=paper_db_inputs,
                    paper_db_globs=args.paper_db_glob,
                    backtest_json_globs=args.backtest_json_glob,
                    output_config_dir=args.output_config_dir,
                    min_done_trades=args.min_done_trades,
                    min_closed_trades=args.min_closed_trades,
                    min_markets=args.min_markets,
                    min_roi=args.min_roi,
                    min_win_rate=args.min_win_rate,
                    max_drawdown=args.max_drawdown,
                    top_n=args.top_n,
                    allow_duplicate_market_trades=bool(args.allow_duplicate_market_trades),
                    output_json_path=args.output_json,
                    output_txt_path=args.output_txt,
                )
            except (
                PaperStrategyExperimentAnalysisError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "compare-model-experiments":
        from .model_experiment_comparison import (
            ModelExperimentComparisonError,
            compare_model_experiments,
        )

        experiment_dir = args.paper_experiment_dir or _single_experiment_dir(args.experiment_dir) or (
            args.output_dir if _cli_option_provided("--output-dir") else None
        )
        output_path = args.output
        if output_path == "data/exports/training_dataset.parquet":
            output_path = "data/analytics/model_experiment_comparison_latest.json"
        output_txt = args.output_txt or "data/analytics/model_experiment_comparison_latest.txt"
        output_csv = args.output_csv or None
        if not experiment_dir:
            report = {"status": "error", "error": "paper_experiment_dir_required"}
        else:
            try:
                report = compare_model_experiments(
                    recorder_db_path=args.recorder_db or args.db,
                    paper_experiment_dir=experiment_dir,
                    transformer_prediction_db_paths=args.transformer_prediction_db
                    or args.prediction_db,
                    transformer_autopsy_path=args.transformer_autopsy_parquet,
                    output_path=output_path,
                    output_txt_path=output_txt,
                    output_csv_path=output_csv,
                )
            except (ModelExperimentComparisonError, FileNotFoundError, RuntimeError, ValueError) as exc:
                report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-live-model-predictions":
        from .live_model_stack import LiveModelStackError, run_live_model_predictions

        output_db = args.output_db if _cli_option_provided("--output-db") else "data/live_predictions.db"
        rf_model_path = args.rf_model_path or (
            args.model_path if _cli_option_provided("--model-path") else None
        )
        rf_feature_columns = args.rf_feature_columns or (
            args.feature_columns if _cli_option_provided("--feature-columns") else None
        )
        try:
            report = run_live_model_predictions(
                recorder_db_path=args.recorder_db or args.db,
                output_db_path=output_db,
                rf_model_path=rf_model_path,
                rf_feature_columns_path=rf_feature_columns,
                transformer_model_dir=args.transformer_model_dir,
                poll_sec=args.poll_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                max_btc_age_sec=args.max_btc_age_sec,
                min_confidence=args.min_confidence,
                run_id=args.run_id,
                shadow_only=bool(args.shadow_only),
                enable_live_trading=bool(args.enable_live_trading),
                sequence_length=args.sequence_length,
                max_iterations=args.paper_max_iterations,
            )
        except (LiveModelStackError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
        if args.paper_max_iterations is not None or report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "score-live-model-predictions":
        from .live_model_stack import LiveModelStackError, score_live_model_predictions

        prediction_db = _single_prediction_db(args.prediction_db) or (
            args.output_db if _cli_option_provided("--output-db") else "data/live_predictions.db"
        )
        try:
            report = score_live_model_predictions(
                recorder_db_path=args.recorder_db or args.db,
                prediction_db_path=prediction_db,
                lookback_hours=args.lookback_hours,
            )
        except (LiveModelStackError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "live-model-dashboard":
        from .live_model_stack import live_model_dashboard_loop

        prediction_db = _single_prediction_db(args.prediction_db) or (
            args.output_db if _cli_option_provided("--output-db") else "data/live_predictions.db"
        )
        live_model_dashboard_loop(
            recorder_db_path=args.recorder_db or args.db,
            prediction_db_path=prediction_db,
            refresh_sec=args.refresh_sec,
            lookback_markets=args.lookback_markets,
            once=bool(args.dashboard_once),
        )
        return
    if args.command == "start-live-model-stack":
        from .live_model_stack import LiveModelStackError, start_live_model_stack

        output_db = args.output_db if _cli_option_provided("--output-db") else "data/live_predictions.db"
        rf_model_path = args.rf_model_path or (
            args.model_path if _cli_option_provided("--model-path") else None
        )
        rf_feature_columns = args.rf_feature_columns or (
            args.feature_columns if _cli_option_provided("--feature-columns") else None
        )
        try:
            report = start_live_model_stack(
                recorder_db_path=args.recorder_db or args.db,
                prediction_db_path=output_db,
                rf_model_path=rf_model_path,
                rf_feature_columns_path=rf_feature_columns,
                transformer_model_dir=args.transformer_model_dir,
                poll_sec=args.poll_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                max_btc_age_sec=args.max_btc_age_sec,
                min_confidence=args.min_confidence,
                run_id=args.run_id,
                shadow_only=bool(args.shadow_only),
                enable_live_trading=bool(args.enable_live_trading),
                dashboard=bool(args.dashboard),
                dashboard_refresh_sec=args.refresh_sec,
            )
        except (LiveModelStackError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {"status": "error", "error": str(exc)}
        if report is not None and report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-multi-strategy-paper-trader":
        from .multi_strategy_paper_trader import (
            MultiStrategyPaperTraderError,
            run_multi_strategy_paper_trader,
        )

        try:
            report = run_multi_strategy_paper_trader(
                recorder_db_path=args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                config_path=args.config,
                output_db_path=args.output_db,
                run_id=args.run_id,
                poll_sec=args.poll_sec,
                max_btc_age_sec=args.max_btc_age_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                require_fresh_comparison_btc=args.require_fresh_comparison_btc,
                max_iterations=args.paper_max_iterations,
                heartbeat_detail=args.heartbeat_detail,
                min_feature_timestamp=args.min_feature_timestamp,
                live_start_now=bool(args.live_start_now),
            )
        except MultiStrategyPaperTraderError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "config_path": args.config,
                "output_db_path": args.output_db,
            }
        if args.paper_max_iterations is not None or report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "run-intramarket-paper-trader":
        from .intramarket_paper_trader import (
            IntramarketPaperTraderError,
            run_intramarket_paper_trader,
        )

        try:
            report = run_intramarket_paper_trader(
                recorder_db_path=args.db,
                model_path=args.model_path,
                feature_columns_path=args.feature_columns,
                config_path=args.config,
                output_db_path=args.output_db,
                run_id=args.run_id,
                poll_sec=args.poll_sec,
                max_btc_age_sec=args.max_btc_age_sec,
                max_feature_age_sec=args.max_feature_age_sec,
                max_iterations=args.paper_max_iterations,
            )
        except IntramarketPaperTraderError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "config_path": args.config,
                "output_db_path": args.output_db,
            }
        if args.paper_max_iterations is not None or report.get("status") == "error":
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "intramarket-paper-analytics":
        from .intramarket_paper_trader import build_intramarket_paper_analytics_report

        report = build_intramarket_paper_analytics_report(
            paper_db_path=_single_paper_db(args.paper_db),
            output_dir=args.output_dir,
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return
    if args.command == "intramarket-strategy-leaderboard":
        from .intramarket_paper_trader import (
            build_intramarket_strategy_leaderboard,
            render_intramarket_strategy_leaderboard,
        )

        report = build_intramarket_strategy_leaderboard(
            paper_db_path=_single_paper_db(args.paper_db),
            min_closed=args.min_settled,
        )
        print(render_intramarket_strategy_leaderboard(report))
        return
    if args.command == "export-intramarket-strategy-dataset":
        from .intramarket_paper_trader import export_intramarket_strategy_dataset

        report = export_intramarket_strategy_dataset(
            paper_db_path=_single_paper_db(args.paper_db),
            output_path=args.output,
            output_csv_path=args.output_csv,
            include_unresolved=args.include_unresolved,
            strategy_id=_single_strategy_id(args.strategy_id),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "generate-paper-strategy-config-sweep":
        from .strategy_config import generate_paper_strategy_config_sweep

        try:
            report = generate_paper_strategy_config_sweep(
                base_config_path=args.base_config or args.config,
                output_dir=args.output_dir,
                name_prefix=args.name_prefix,
                thresholds=args.thresholds or "0.80,0.85,0.87,0.90",
                max_probabilities=args.max_probabilities,
                time_windows=args.time_windows,
                bankrolls=args.bankrolls,
                blocked_liquidity_regimes=args.blocked_liquidity_regimes,
                fixed_horizon_sec=args.fixed_horizon_sec,
            )
        except ValueError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "base_config": args.base_config or args.config,
                "output_dir": args.output_dir,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "filter-strategy-config":
        from .strategy_config import filter_strategy_config

        try:
            report = filter_strategy_config(
                args.config,
                args.output,
                strategy_id_substrings=args.strategy_id_contains or [],
            )
        except ValueError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "source_config": args.config,
                "output_config": args.output,
                "strategy_id_substrings": args.strategy_id_contains or [],
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "validate-strategy-config":
        from .strategy_config import validate_strategy_config

        print(
            json.dumps(
                validate_strategy_config(args.config),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "strategy-config-summary":
        from .strategy_config import summarize_strategy_config

        print(
            json.dumps(
                summarize_strategy_config(args.config),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "paper-trader-eligibility-report":
        from .paper_trader_eligibility import (
            build_paper_trader_eligibility_report,
            render_paper_trader_eligibility_report,
        )

        report = build_paper_trader_eligibility_report(
            recorder_db_path=args.db,
            paper_db_path=_single_paper_db(args.paper_db),
            feature_columns_path=args.feature_columns,
            model_path=args.model_path,
            recent_minutes=30.0 if args.recent_minutes is None else args.recent_minutes,
            max_btc_age_sec=args.max_btc_age_sec,
            max_feature_age_sec=args.max_feature_age_sec,
            min_time_until_resolution_sec=(
                0.0
                if args.min_time_until_resolution_sec is None
                else _single_float_option(args.min_time_until_resolution_sec)
            ),
            max_time_until_resolution_sec=(
                300.0
                if args.max_time_until_resolution_sec is None
                else _single_float_option(args.max_time_until_resolution_sec)
            ),
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            min_estimated_edge=args.min_estimated_edge,
            max_open_trades=args.max_open_trades,
            block_probability_above=args.block_probability_above,
            block_probability_below=args.block_probability_below,
            require_fresh_comparison_btc=args.require_fresh_comparison_btc,
        )
        print(render_paper_trader_eligibility_report(report))
        return
    if args.command == "daily-recorder-maintenance":
        from .daily_recorder_maintenance import (
            render_daily_recorder_maintenance_report,
            run_daily_recorder_maintenance,
        )

        report = run_daily_recorder_maintenance(
            db_path=args.db,
            export_dir=args.export_dir,
            archive_dir=args.archive_dir,
            keep_recent_hours=args.keep_recent_hours,
            checkpoint=args.checkpoint,
            dry_run=bool(args.dry_run),
            chunk_size=args.export_chunk_size,
            delete_batch_size=args.maintenance_delete_batch_size,
        )
        print(render_daily_recorder_maintenance_report(report))
        return
    if args.command == "export-transformer-sequence-dataset":
        from .transformer_sequence_dataset import (
            TransformerSequenceExportError,
            export_transformer_sequence_dataset,
        )

        try:
            report = export_transformer_sequence_dataset(
                input_path=args.input,
                output_path=args.output,
                sequence_length=args.sequence_length,
                stride_sec=args.stride_sec,
                min_time_until_resolution_sec=_single_float_option(args.min_time_until_resolution_sec),
                max_time_until_resolution_sec=_single_float_option(args.max_time_until_resolution_sec),
                label_column=args.label_column,
                market_id_column=args.market_id_column,
                timestamp_column=args.timestamp_column,
                include_not_ready=args.include_not_ready,
                keep_gap_affected=args.keep_gap_affected,
            )
        except TransformerSequenceExportError as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "input_path": args.input,
                "output_path": args.output,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    settings = load_settings(env_file=args.env_file)
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    if args.command == "live-paper-maintenance-loop":
        from .expired_resolution_refresh import refresh_expired_market_resolutions_async
        from .live_paper_maintenance_loop import (
            LivePaperMaintenanceConfig,
            run_live_paper_maintenance_loop,
        )

        recorder_db_path = args.recorder_db or args.db

        def _refresh_resolution(**kwargs: object) -> dict:
            async def _run_refresh() -> dict:
                gamma = GammaClient(
                    base_url=settings.gamma_api_url,
                    timeout_sec=settings.request_timeout_sec,
                    page_size=settings.discovery_page_size,
                    max_pages=settings.discovery_max_pages,
                    lookahead_sec=settings.discovery_lookahead_sec,
                )
                try:
                    return await refresh_expired_market_resolutions_async(
                        db_path=recorder_db_path,
                        gamma_client=gamma,
                        older_than_minutes=float(
                            kwargs.get("older_than_minutes", args.resolution_older_than_minutes)
                        ),
                        limit=int(kwargs.get("limit", args.limit or 100)),
                        apply=bool(kwargs.get("apply", False)),
                        now=kwargs.get("now"),
                    )
                finally:
                    await gamma.close()

            return asyncio.run(_run_refresh())

        report = run_live_paper_maintenance_loop(
            LivePaperMaintenanceConfig(
                recorder_db_path=recorder_db_path,
                paper_db_path=_single_paper_db(args.paper_db),
                export_dir=args.export_dir,
                analytics_dir=args.analytics_dir,
                poll_sec=args.poll_sec,
                resolution_older_than_minutes=args.resolution_older_than_minutes,
                export_every_minutes=args.export_every_minutes,
                analytics_every_minutes=args.analytics_every_minutes,
                resolution_limit=100 if args.limit is None else args.limit,
                strategy_id=_single_strategy_id(args.strategy_id),
                dry_run=bool(args.dry_run),
                once=args.once,
            ),
            resolution_refresh_fn=_refresh_resolution,
            max_iterations=args.paper_max_iterations,
        )
        if args.once or args.paper_max_iterations is not None:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "refresh-expired-market-resolutions":
        from .expired_resolution_refresh import refresh_expired_market_resolutions_async

        async def _run_refresh() -> dict:
            gamma = GammaClient(
                base_url=settings.gamma_api_url,
                timeout_sec=settings.request_timeout_sec,
                page_size=settings.discovery_page_size,
                max_pages=settings.discovery_max_pages,
                lookahead_sec=settings.discovery_lookahead_sec,
            )
            try:
                return await refresh_expired_market_resolutions_async(
                    db_path=args.db,
                    gamma_client=gamma,
                    older_than_minutes=args.older_than_minutes,
                    limit=100 if args.limit is None else args.limit,
                    apply=args.apply,
                )
            finally:
                await gamma.close()

        report = asyncio.run(_run_refresh())
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "dataset-quality":
        from .dataset_quality import build_dataset_quality_report, render_dataset_quality_report

        report = build_dataset_quality_report(
            settings.db_path,
            recent_minutes=args.recent_minutes,
        )
        print(render_dataset_quality_report(report))
        return
    if args.command == "dashboard":
        from .dashboard import run_dashboard

        run_dashboard(
            settings.db_path,
            refresh_sec=args.dashboard_refresh_sec,
            once=args.dashboard_once,
        )
        return
    if args.command == "web-dashboard":
        from .web_dashboard import run_web_dashboard

        run_web_dashboard(
            settings.db_path,
            host=args.web_dashboard_host,
            port=args.web_dashboard_port,
            refresh_sec=args.web_dashboard_refresh_sec,
            preferred_btc_source=args.web_dashboard_btc_source,
            paper_db_paths=args.paper_db,
            paper_db_globs=args.paper_db_glob,
        )
        return
    if args.command == "btc-feed-config":
        from .btc_price_feed import describe_btc_price_feed_config

        print(json.dumps(describe_btc_price_feed_config(settings), indent=2, sort_keys=True))
        return
    if args.command == "wal-checkpoint":
        report = run_sqlite_wal_checkpoint(
            args.db,
            mode=args.checkpoint_mode or "PASSIVE",
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            truncate=bool(args.truncate_wal),
        )
        logging.getLogger("main").info(
            "sqlite_wal_checkpoint_completed",
            extra=report,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "debug-discovery":
        asyncio.run(_run_debug_discovery(settings))
        return
    if args.command == "export-training":
        report = export_training_parquet(
            db_path=settings.db_path,
            output_path=args.export_path,
            run_ids=args.export_run_id,
            merge_runs=args.merge_runs,
            require_trades=args.export_require_trades,
            require_full_orderbook_depth=args.export_require_full_orderbook,
            exclude_gap_affected=not args.export_keep_gap_affected,
            label_horizon_sec=args.label_horizon_sec,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "export-training-dataset":
        from .training_dataset_export import export_training_dataset

        report = export_training_dataset(
            db_path=args.db,
            output_path=args.output,
            output_csv_path=args.output_csv,
            recent_minutes=args.recent_minutes,
            min_time_until_resolution_sec=_single_float_option(args.min_time_until_resolution_sec),
            max_time_until_resolution_sec=_single_float_option(args.max_time_until_resolution_sec),
            include_not_ready=args.include_not_ready,
            canonical_btc_source=args.canonical_btc_source,
            comparison_btc_source=args.comparison_btc_source,
            limit=args.limit,
            chunk_size=args.export_chunk_size,
            progress_every_rows=args.progress_every_rows,
            progress_every_sec=args.progress_every_sec,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "export-recorder-archive-to-parquet":
        from .recorder_archive_export import (
            RecorderArchiveExportError,
            export_recorder_archive_to_parquet,
            render_recorder_archive_export_report,
        )

        archive_inputs = args.archive_db or []
        try:
            report = export_recorder_archive_to_parquet(
                archive_db_inputs=archive_inputs,
                output_dir=args.output_dir,
                run_id=args.run_id,
                append=bool(args.append),
                chunk_size=args.export_chunk_size,
            )
        except (RecorderArchiveExportError, FileNotFoundError, RuntimeError, ValueError) as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "archive_db_inputs": archive_inputs,
                "output_dir": args.output_dir,
            }
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(render_recorder_archive_export_report(report))
        return
    if args.command == "validate-db":
        report = validate_db_quality(settings.db_path, run_id=args.validate_run_id)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "upgrade-db":
        report = _upgrade_db_schema(settings, run_id=args.run_id or "legacy")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "repair-stale-active-markets":
        db = SQLiteStore(
            db_path=settings.db_path,
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            run_id=args.run_id or "legacy",
            quarantine_on_startup=False,
        )
        try:
            report = db.repair_stale_selected_active_markets(
                dry_run=bool(args.dry_run),
                grace_sec=args.stale_active_grace_sec,
                run_id=args.run_id,
            )
        finally:
            db.close()
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "normalize-resolutions":
        db = SQLiteStore(
            db_path=settings.db_path,
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            run_id=args.run_id or "legacy",
            quarantine_on_startup=False,
        )
        try:
            if not bool(args.dry_run):
                db.init_schema(
                    backfill_existing_rows=False,
                    dedupe_lifecycle_events=False,
                    recover_malformed=False,
                )
            report = db.normalize_market_resolutions(
                dry_run=bool(args.dry_run),
                run_id=args.run_id,
            )
        finally:
            db.close()
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "fresh-db":
        _prepare_fresh_run_db(
            settings.db_path,
            archive_old_runs=True,
        )
        db = SQLiteStore(
            db_path=settings.db_path,
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            run_id=args.run_id or "legacy",
        )
        try:
            db.init_schema()
        finally:
            db.close()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "db_path": settings.db_path,
                    "action": "fresh_db_initialized",
                },
                sort_keys=True,
            )
        )
        return

    if args.diagnostics and not args.repair_invalid_snapshot_trades:
        print_diagnostics(
            settings.db_path,
            lookahead_sec=settings.discovery_lookahead_sec,
            raw_ws_events_enabled=getattr(settings, "raw_ws_events_enabled", None),
        )
        return

    if args.repair_invalid_snapshot_trades:
        db = SQLiteStore(
            db_path=settings.db_path,
            busy_timeout_ms=settings.sqlite_busy_timeout_ms,
            run_id=args.run_id or "legacy",
        )
        try:
            db.init_schema()
            db.refresh_market_phases(run_id=args.run_id or "legacy")
            if args.repair_invalid_snapshot_trades:
                before = db.snapshot_trade_integrity_counts(run_id=args.run_id or "legacy")
                repaired = db.repair_invalid_snapshot_trade_fields(run_id=args.run_id or "legacy")
                after = db.snapshot_trade_integrity_counts(run_id=args.run_id or "legacy")
                print("=== Snapshot Trade Repair ===")
                print(f"rows_repaired={repaired}")
                print(
                    "before_invalid="
                    f"{before.get('total_invalid', 0)} "
                    f"(before_start={before.get('last_trade_before_market_start', 0)}, "
                    f"after_snapshot={before.get('last_trade_after_snapshot_time', 0)})"
                )
                print(
                    "after_invalid="
                    f"{after.get('total_invalid', 0)} "
                    f"(before_start={after.get('last_trade_before_market_start', 0)}, "
                    f"after_snapshot={after.get('last_trade_after_snapshot_time', 0)})"
                )
        finally:
            db.close()
        if args.diagnostics:
            print_diagnostics(
                settings.db_path,
                lookahead_sec=settings.discovery_lookahead_sec,
                raw_ws_events_enabled=getattr(settings, "raw_ws_events_enabled", None),
            )
        return

    run_id = args.run_id or _generate_run_id()
    os.environ["RECORDER_RUN_ID"] = run_id
    if args.fresh_run:
        _prepare_fresh_run_db(settings.db_path, archive_old_runs=args.archive_old_runs)
    app = RecorderApp(settings, run_id=run_id)
    try:
        asyncio.run(_run_with_signals(app))
    except KeyboardInterrupt:
        # asyncio signal handling already requests graceful shutdown.
        pass


if __name__ == "__main__":
    main()
