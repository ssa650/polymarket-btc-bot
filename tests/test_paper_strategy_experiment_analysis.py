from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from src.paper_strategy_experiment_analysis import (
    analyze_paper_strategy_experiment,
    mine_paper_strategies,
    recommend_next_paper_sweep,
    render_paper_strategy_experiment_analysis,
)
from src.strategy_config import validate_strategy_config


def _write_config(path: Path) -> None:
    payload = {
        "bankroll": {
            "bankroll_enabled": True,
            "starting_bankroll_usd": 100.0,
            "base_risk_fraction": 0.01,
            "max_risk_fraction": 0.05,
        },
        "realistic_execution": {
            "realistic_execution_enabled": True,
            "entry_latency_sec": 1.0,
        },
        "liquidity": {
            "liquidity_fill_check_enabled": True,
        },
        "trade_selection": {
            "selection_enabled": True,
            "mode": "best_per_market_direction",
        },
        "live_trading": {
            "live_trading_enabled": False,
            "dry_run_orders": True,
        },
        "strategies": [
            {
                "strategy_id": "s_good",
                "strategy_name": "Good strategy",
                "enabled": True,
                "direction_mode": "NO_ONLY",
                "long_threshold": 2.0,
                "short_threshold": 0.15,
                "min_estimated_edge": 0.05,
                "max_probability_for_direction": 0.90,
                "min_time_until_resolution_sec": 30,
                "max_time_until_resolution_sec": 60,
                "fixed_horizon_exit_sec": 15,
                "entry_slippage_cents": 0.01,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 3,
                "one_trade_per_market": True,
                "require_positive_edge": True,
            },
            {
                "strategy_id": "s_bad",
                "strategy_name": "Bad strategy",
                "enabled": True,
                "direction_mode": "NO_ONLY",
                "long_threshold": 2.0,
                "short_threshold": 0.10,
                "min_estimated_edge": 0.05,
                "max_probability_for_direction": 0.95,
                "min_time_until_resolution_sec": 60,
                "max_time_until_resolution_sec": 120,
                "fixed_horizon_exit_sec": 15,
                "entry_slippage_cents": 0.01,
                "fee_cents": 0.0,
                "stake_usd": 1.0,
                "max_open_trades": 3,
                "one_trade_per_market": True,
                "require_positive_edge": True,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_paper_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            market_id TEXT,
            strategy_id TEXT,
            signal_direction TEXT,
            status TEXT,
            stake_usd REAL,
            pnl_usd REAL,
            roi REAL,
            realized_pnl_usd REAL,
            realized_roi REAL,
            threshold_used REAL,
            time_until_resolution REAL,
            time_regime TEXT,
            btc_trend_regime TEXT,
            spread_regime TEXT,
            liquidity_regime TEXT,
            entry_price_drift REAL,
            spread_cents_at_entry REAL,
            exit_reason TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            strategy_id TEXT,
            decision TEXT,
            rejection_reason TEXT,
            threshold_used REAL,
            min_time_until_resolution_sec REAL,
            max_time_until_resolution_sec REAL,
            max_probability_for_direction REAL,
            starting_bankroll_usd REAL
        );
        """
    )
    good_pnls = [0.6, 0.6, 0.6, 0.6, -0.2, -0.2]
    bad_pnls = [0.5, -1.0, -1.0, -1.0, -1.0, -1.0]
    rows = []
    for idx, pnl in enumerate(good_pnls):
        rows.append(
            (
                f"2026-01-01T00:00:{idx:02d}+00:00",
                f"m_good_{idx}",
                "s_good",
                "NO",
                "closed",
                1.0,
                None,
                None,
                pnl,
                pnl,
                0.85,
                45.0,
                "30-60s",
                "weak_down",
                "tight",
                "normal",
                0.5 if pnl > 0 else 3.0,
                1.5 if pnl > 0 else 5.0,
                "fixed_horizon",
            )
        )
    for idx, pnl in enumerate(bad_pnls):
        rows.append(
            (
                f"2026-01-01T00:01:{idx:02d}+00:00",
                f"m_bad_{idx}",
                "s_bad",
                "NO",
                "closed",
                1.0,
                None,
                None,
                pnl,
                pnl,
                0.90,
                90.0,
                "60-120s",
                "strong_down",
                "wide",
                "thin",
                4.0,
                8.0,
                "fixed_horizon",
            )
        )
    rows.append(
        (
            "2026-01-01T00:02:00+00:00",
            "m_skip",
            "s_good",
            "NO",
            "skipped",
            1.0,
            None,
            None,
            None,
            None,
            0.85,
            45.0,
            "30-60s",
            "weak_down",
            "tight",
            "normal",
            0.0,
            1.0,
            None,
        )
    )
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, market_id, strategy_id, signal_direction, status,
            stake_usd, pnl_usd, roi, realized_pnl_usd, realized_roi,
            threshold_used, time_until_resolution, time_regime,
            btc_trend_regime, spread_regime, liquidity_regime,
            entry_price_drift, spread_cents_at_entry, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    candidate_rows = [
        ("2026-01-01T00:00:00+00:00", "s_good", "TRADE", None, 0.85, 30.0, 60.0, 0.90, 100.0),
        ("2026-01-01T00:00:01+00:00", "s_good", "TRADE", None, 0.85, 30.0, 60.0, 0.90, 100.0),
        ("2026-01-01T00:00:02+00:00", "s_good", "SKIP", "threshold", 0.85, 30.0, 60.0, 0.90, 100.0),
        ("2026-01-01T00:01:00+00:00", "s_bad", "TRADE", None, 0.90, 60.0, 120.0, 0.95, 100.0),
        ("2026-01-01T00:01:01+00:00", "s_bad", "SKIP", "min_estimated_edge", 0.90, 60.0, 120.0, 0.95, 100.0),
        ("2026-01-01T00:01:02+00:00", "s_bad", "SKIP", "time_window", 0.90, 60.0, 120.0, 0.95, 100.0),
    ]
    conn.executemany(
        """
        INSERT INTO paper_trade_candidates (
            created_at, strategy_id, decision, rejection_reason, threshold_used,
            min_time_until_resolution_sec, max_time_until_resolution_sec,
            max_probability_for_direction, starting_bankroll_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        candidate_rows,
    )
    conn.commit()
    conn.close()


def _setup_experiment(tmp_path: Path) -> tuple[Path, Path]:
    experiment_dir = tmp_path / "experiment"
    paper_dir = experiment_dir / "paper_dbs"
    config_dir = experiment_dir / "configs"
    paper_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    paper_db = paper_dir / "worker.db"
    config_path = config_dir / "config.json"
    _create_paper_db(paper_db)
    _write_config(config_path)
    manifest = {
        "experiment_id": "exp_analysis",
        "workers": [
            {
                "worker_id": "worker",
                "paper_db_path": str(paper_db),
                "config_path": str(config_path),
                "run_id": "run_worker",
            }
        ],
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return experiment_dir, config_path


def test_analyze_paper_strategy_experiment_outputs_rankings_and_funnel(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_json = tmp_path / "analysis.json"
    output_txt = tmp_path / "analysis.txt"

    report = analyze_paper_strategy_experiment(
        experiment_dir=str(experiment_dir),
        output_json_path=str(output_json),
        output_txt_path=str(output_txt),
    )

    assert report["status"] == "ok"
    assert report["summary"]["done_trades"] == 12
    assert report["summary"]["pnl"] == -2.5
    top = report["strategy_rankings"]["min_done_5"]["sorted_by_pnl"][0]
    assert top["strategy_id"] == "s_good"
    assert top["pnl"] == 2.0
    assert top["payoff_ratio"] == 3.0
    assert report["strategy_rankings"]["min_done_10"]["sorted_by_pnl"] == []
    assert report["regime_interactions"]["strategy_id_x_time_regime"]["groups"]
    assert report["loss_tail_analysis"]["largest_losing_trades"][0]["strategy_id"] == "s_bad"
    assert any(
        row["pmax"] == "0.95"
        for row in report["loss_tail_analysis"]["pmax_loss_tail_comparison"]
    )
    assert report["candidate_funnel"]["decision_counts"] == {"SKIP": 3, "TRADE": 3}
    assert report["candidate_funnel"]["rejection_reasons"]["time_window"] == 1
    assert output_json.exists()
    assert output_txt.exists()
    text = output_txt.read_text(encoding="utf-8")
    assert "Executive Summary" in text
    assert "Overall PnL=-2.5 ROI=-0.2083333333" in text
    assert "Live Trading Recommendation: DO NOT GO LIVE" in text
    assert "Warnings: do not go live, small sample, negative overall expectancy" in text
    assert "Top 5 Strategies by PnL (done_trades >= 5)" in text
    assert "s_good: done=6 pnl=2.0" in text
    assert "Top 5 Strategies by Avg ROI (done_trades >= 20)" in text
    assert "Worst 5 Loss Drivers" in text
    assert "liquidity_regime liquidity_regime=thin" in text
    assert "Suggested Next Sweep Configs" in text
    assert "warning=below_30_done_trades" in text
    assert "Largest Losing Trades" in text
    assert "Candidate Rejections By Reason" in text
    assert "  threshold: count=1" in text
    assert "  None: count=" not in text
    assert "DO NOT GO LIVE" in text


def test_render_paper_strategy_experiment_analysis_respects_verbose_txt_limit() -> None:
    strategy_rows = [
        {
            "strategy_id": f"s_{idx}",
            "done_trades": 30 + idx,
            "pnl": float(idx),
            "average_roi": idx / 100,
            "win_rate": 0.5,
            "payoff_ratio": 1.2,
        }
        for idx in range(1, 7)
    ]
    report = {
        "experiment_id": "exp",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {
            "pnl": 1.0,
            "roi": 0.1,
            "average_roi": 0.1,
            "win_rate": 0.6,
            "payoff_ratio": 1.5,
            "done_trades": 120,
        },
        "strategy_rankings": {
            "min_done_5": {
                "sorted_by_pnl": strategy_rows,
                "sorted_by_avg_roi": strategy_rows,
            },
            "min_done_20": {
                "sorted_by_avg_roi": strategy_rows,
            },
        },
        "loss_tail_analysis": {"largest_losing_trades": []},
        "candidate_funnel": {
            "decision_counts": {"SKIP": 6},
            "rejection_reasons": {
                f"reason_{idx}": idx
                for idx in range(1, 7)
            },
        },
        "next_sweep_recommendations": {
            "status": "ok",
            "suggested_strategies": strategy_rows,
        },
    }

    concise = render_paper_strategy_experiment_analysis(report)
    verbose = render_paper_strategy_experiment_analysis(report, verbose=True)

    assert "s_6" not in concise
    assert "reason_1: count=1" not in concise
    assert "s_6" in verbose
    assert "reason_1: count=1" in verbose


def test_recommend_next_paper_sweep_writes_paper_only_config(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_dir = tmp_path / "recommended"

    report = recommend_next_paper_sweep(
        experiment_dir=str(experiment_dir),
        output_config_dir=str(output_dir),
        min_done=5,
    )

    assert report["status"] == "ok"
    suggested = report["recommendations"]["suggested_strategies"]
    assert [row["strategy_id"] for row in suggested] == ["s_good"]
    assert report["generated_config_paths"]
    config_payload = json.loads(Path(report["generated_config_paths"][0]).read_text(encoding="utf-8"))
    assert config_payload["live_trading"]["live_trading_enabled"] is False
    strategy_ids = {row["strategy_id"] for row in config_payload["strategies"]}
    assert any(strategy_id.startswith("s_good") for strategy_id in strategy_ids)
    assert not any(strategy_id.startswith("s_bad") for strategy_id in strategy_ids)
    assert (output_dir / "next_sweep_recommendations.json").exists()


def test_mine_paper_strategies_outputs_leaderboard_and_suggestions(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_json = tmp_path / "mined.json"
    output_txt = tmp_path / "mined.txt"

    report = mine_paper_strategies(
        experiment_dirs=[str(experiment_dir)],
        min_done_trades=5,
        min_markets=5,
        output_json_path=str(output_json),
        output_txt_path=str(output_txt),
    )

    assert report["status"] == "ok"
    assert report["overall_experiment_health"]["experiments_loaded"] == 1
    assert report["overall_experiment_health"]["done_trades"] == 12
    assert report["overall_experiment_health"]["markets_covered"] == 12
    leaderboard = {row["strategy_id"]: row for row in report["strategy_leaderboard"]}
    assert leaderboard["s_good"]["eligible_for_validation"] is True
    assert leaderboard["s_good"]["pnl"] == 2.0
    assert leaderboard["s_bad"]["eligible_for_validation"] is False
    assert "non_positive_pnl" in leaderboard["s_bad"]["excluded_reasons"]
    assert report["promising_regimes"]["time_regime"][0]["time_regime"] == "30-60s"
    suggestions = report["suggested_next_paper_sweep_configs"]
    assert suggestions
    assert suggestions[0]["source_strategy_id"] == "s_good"
    suggestion_config = suggestions[0]["config"]
    assert suggestion_config["live_trading"]["live_trading_enabled"] is False
    assert suggestion_config["bankroll"]["bankroll_enabled"] is True
    assert suggestion_config["liquidity"]["liquidity_fill_check_enabled"] is True
    assert output_json.exists()
    text = output_txt.read_text(encoding="utf-8")
    assert "Paper Strategy Mining Report" in text
    assert "DO NOT GO LIVE" in text


def test_mine_paper_strategies_marks_small_samples_and_detects_loss_tail(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    paper_db = experiment_dir / "paper_dbs" / "worker.db"
    conn = sqlite3.connect(paper_db)
    tail_rows = []
    for idx, pnl in enumerate([0.1, 0.1, 0.1, 0.1, 0.1, -2.0]):
        tail_rows.append(
            (
                f"2026-01-01T00:03:{idx:02d}+00:00",
                f"m_tail_{idx}",
                "s_tail",
                "NO",
                "closed",
                1.0,
                None,
                None,
                pnl,
                pnl,
                0.85,
                45.0,
                "30-60s",
                "weak_down",
                "tight",
                "normal",
                0.0,
                1.0,
                "fixed_horizon",
            )
        )
    conn.executemany(
        """
        INSERT INTO paper_trades (
            created_at, market_id, strategy_id, signal_direction, status,
            stake_usd, pnl_usd, roi, realized_pnl_usd, realized_roi,
            threshold_used, time_until_resolution, time_regime,
            btc_trend_regime, spread_regime, liquidity_regime,
            entry_price_drift, spread_cents_at_entry, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tail_rows,
    )
    conn.commit()
    conn.close()

    report = mine_paper_strategies(
        experiment_dirs=[str(experiment_dir)],
        min_done_trades=7,
        min_markets=7,
    )
    leaderboard = {row["strategy_id"]: row for row in report["strategy_leaderboard"]}
    assert leaderboard["s_good"]["sample_status"] == "small_sample"
    assert "small_done_trade_sample" in leaderboard["s_good"]["excluded_reasons"]

    report = mine_paper_strategies(
        experiment_dirs=[str(experiment_dir)],
        min_done_trades=5,
        min_markets=5,
    )
    loss_tail_ids = {row["strategy_id"] for row in report["bad_loss_tail_strategies"]}
    assert "s_tail" in loss_tail_ids
    tail_row = next(row for row in report["bad_loss_tail_strategies"] if row["strategy_id"] == "s_tail")
    assert "high_win_rate_bad_payoff" in tail_row["loss_tail_reason"]
    assert "rare_large_loss_tail" in tail_row["loss_tail_reason"]


def test_mine_paper_strategies_generates_configs_from_paper_db_and_backtest_json(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    paper_db = experiment_dir / "paper_dbs" / "worker.db"
    backtest_json = tmp_path / "transformer_backtest.json"
    output_dir = tmp_path / "mined_configs"
    backtest_payload = {
        "status": "ok",
        "slippage_cents": 1.0,
        "fee_cents": 0.0,
        "best_by_pnl": [
            {
                "strategy_id": "transformer_yes_t075_60_90_fh60",
                "done_trades": 8,
                "trades": 8,
                "markets_covered": 8,
                "pnl": 3.2,
                "roi": 0.4,
                "average_roi": 0.4,
                "win_rate": 0.75,
                "average_win": 0.8,
                "average_loss": -0.4,
                "payoff_ratio": 2.0,
                "breakeven_win_rate": 0.3333333333,
                "best_trade": 1.2,
                "worst_trade": -0.4,
            }
        ],
        "best_by_avg_roi_min_done": [],
        "recommendation": "paper/shadow only; do not go live",
    }
    backtest_json.write_text(json.dumps(backtest_payload), encoding="utf-8")

    report = mine_paper_strategies(
        paper_db_globs=[str(paper_db)],
        backtest_json_globs=[str(backtest_json)],
        output_config_dir=str(output_dir),
        min_closed_trades=5,
        min_markets=5,
        min_roi=0.05,
        min_win_rate=0.5,
        top_n=2,
    )

    assert report["status"] == "ok"
    assert report["overall_experiment_health"]["paper_dbs_loaded"] == 1
    assert report["overall_experiment_health"]["backtest_jsons_loaded"] == 1
    selected_ids = {row["strategy_id"] for row in report["selected_strategy_candidates"]}
    assert "s_good" in selected_ids
    assert "transformer_yes_t075_60_90_fh60" in selected_ids
    assert report["generated_config_paths"]
    assert report["metadata_json_path"]
    assert "/metadata/" in report["metadata_json_path"]
    config_glob_paths = sorted((output_dir / "configs").glob("*.json"))
    assert len(config_glob_paths) == len(report["generated_config_paths"])
    assert not list((output_dir / "configs").glob("*metadata*.json"))
    for config_path in config_glob_paths:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert payload["live_trading"]["live_trading_enabled"] is False
        assert validate_strategy_config(config_path)["status"] == "ok"
    transformer_config = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in config_glob_paths
        if "transformer_yes_t075_60_90_fh60" in path.name
    )
    assert transformer_config["prediction_source"] == "transformer_predictions"
    strategy = transformer_config["strategies"][0]
    assert strategy["direction_mode"] == "YES_ONLY"
    assert strategy["long_threshold"] == 0.75
    assert strategy["min_time_until_resolution_sec"] == 60.0
    assert strategy["max_time_until_resolution_sec"] == 90.0
    assert strategy["fixed_horizon_exit_sec"] == 60


def test_mine_paper_strategies_penalizes_duplicate_market_trades(tmp_path) -> None:
    experiment_dir, _config_path = _setup_experiment(tmp_path)
    paper_db = experiment_dir / "paper_dbs" / "worker.db"
    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        INSERT INTO paper_trades (
            created_at, market_id, strategy_id, signal_direction, status,
            stake_usd, pnl_usd, roi, realized_pnl_usd, realized_roi,
            threshold_used, time_until_resolution, time_regime,
            btc_trend_regime, spread_regime, liquidity_regime,
            entry_price_drift, spread_cents_at_entry, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-01-01T00:04:00+00:00",
            "m_good_0",
            "s_good",
            "NO",
            "closed",
            1.0,
            0.2,
            0.2,
            None,
            None,
            0.85,
            45.0,
            "30-60s",
            "weak_down",
            "tight",
            "normal",
            0.0,
            1.0,
            "fixed_horizon",
        ),
    )
    conn.commit()
    conn.close()

    report = mine_paper_strategies(
        paper_db_globs=[str(paper_db)],
        min_closed_trades=5,
        min_markets=5,
    )

    row = next(item for item in report["strategy_leaderboard"] if item["strategy_id"] == "s_good")
    assert row["duplicate_market_groups"] == 1
    assert row["duplicate_trade_count"] == 1
    assert row["duplicate_market_penalty"] == 1
    assert "duplicate_strategy_market_trades" in row["warnings"]


def test_paper_strategy_experiment_analysis_cli_smoke(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_json = tmp_path / "analysis_cli.json"
    output_txt = tmp_path / "analysis_cli.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "analyze-paper-strategy-experiment",
            "--experiment-dir",
            str(experiment_dir),
            "--output-json",
            str(output_json),
            "--output-txt",
            str(output_txt),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert output_json.exists()
    assert output_txt.exists()


def test_recommend_next_paper_sweep_cli_smoke(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_dir = tmp_path / "recommended_cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "recommend-next-paper-sweep",
            "--experiment-dir",
            str(experiment_dir),
            "--output-config-dir",
            str(output_dir),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert (output_dir / "next_sweep_recommendations.json").exists()


def test_mine_paper_strategies_cli_smoke(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    experiment_dir, _config_path = _setup_experiment(tmp_path)
    output_json = tmp_path / "mine_cli.json"
    output_txt = tmp_path / "mine_cli.txt"
    output_config_dir = tmp_path / "mine_cli_configs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "mine-paper-strategies",
            "--experiment-dir",
            str(experiment_dir),
            "--experiment-dir",
            str(experiment_dir),
            "--min-done-trades",
            "5",
            "--min-markets",
            "5",
            "--min-closed-trades",
            "5",
            "--min-roi",
            "0.0",
            "--min-win-rate",
            "0.4",
            "--top-n",
            "1",
            "--output-config-dir",
            str(output_config_dir),
            "--output-json",
            str(output_json),
            "--output-txt",
            str(output_txt),
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["overall_experiment_health"]["experiments_requested"] == 2
    assert payload["overall_experiment_health"]["experiments_loaded"] == 2
    assert output_json.exists()
    assert output_txt.exists()
    assert len(list((output_config_dir / "configs").glob("*.json"))) == 1
    assert (output_config_dir / "metadata" / "selected_strategy_metadata.json").exists()
