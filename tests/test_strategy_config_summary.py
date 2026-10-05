from __future__ import annotations

import sys

import pytest

from src.strategy_config import summarize_strategy_config


EXPANDED_CONFIG = "data/strategy_configs/live_strategy_grid_expanded.json"


@pytest.mark.private_config
def test_strategy_config_summary_reports_counts_and_ranges() -> None:
    report = summarize_strategy_config(EXPANDED_CONFIG)

    assert report["status"] == "ok"
    assert report["strategy_count"] == 36
    assert report["enabled_strategy_count"] == 36
    assert report["count_by_direction_mode"] == {
        "BOTH": 12,
        "NO_ONLY": 12,
        "YES_ONLY": 12,
    }
    assert report["threshold_ranges"]["long_threshold"]["min"] == 0.55
    assert report["threshold_ranges"]["long_threshold"]["max"] == 2.0
    assert report["threshold_ranges"]["short_threshold"]["min"] == -1.0
    assert report["threshold_ranges"]["short_threshold"]["max"] == 0.45
    assert report["edge_ranges"]["min_estimated_edge"] == {"min": 0.0, "max": 0.08}
    assert report["time_window_ranges"]["min_time_until_resolution_sec"] == {
        "min": 15.0,
        "max": 60.0,
    }
    assert report["time_window_ranges"]["max_time_until_resolution_sec"] == {
        "min": 120.0,
        "max": 290.0,
    }
    assert report["max_open_trades_range"] == {"min": 10.0, "max": 10.0}
    assert len(report["sample_strategy_ids"]) == 10


def test_strategy_config_summary_cli_is_registered(monkeypatch) -> None:
    from src import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "strategy-config-summary",
            "--config",
            EXPANDED_CONFIG,
        ],
    )

    args = main.parse_args()

    assert args.command == "strategy-config-summary"
    assert args.config == EXPANDED_CONFIG
