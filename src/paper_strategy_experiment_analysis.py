from __future__ import annotations

import copy
import glob
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote


DONE_STATUSES = {"closed", "settled"}
DEFAULT_MIN_DONE_THRESHOLDS = (5, 10, 20, 30)
DEFAULT_MIN_GROUP_DONE = 5
TOP_N = 25


class PaperStrategyExperimentAnalysisError(RuntimeError):
    pass


def analyze_paper_strategy_experiment(
    *,
    experiment_dir: str,
    output_json_path: str | None = None,
    output_txt_path: str | None = None,
    verbose_txt: bool = False,
    min_done_thresholds: Sequence[int] = DEFAULT_MIN_DONE_THRESHOLDS,
    min_group_done: int = DEFAULT_MIN_GROUP_DONE,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    loaded = _load_experiment(experiment_dir)
    trades = [
        _enrich_trade(row, loaded["strategy_metadata"])
        for row in loaded["trade_rows"]
    ]
    candidates = [
        _enrich_candidate(row, loaded["strategy_metadata"])
        for row in loaded["candidate_rows"]
    ]
    done_trades = [row for row in trades if _status(row) in DONE_STATUSES]
    report = {
        "status": "ok",
        "generated_at": _iso(now_dt),
        "experiment_dir": str(Path(experiment_dir)),
        "experiment_id": loaded["manifest"].get("experiment_id"),
        "input_health": loaded["input_health"],
        "summary": _overall_summary(trades, done_trades, candidates),
        "strategy_rankings": _strategy_rankings(done_trades, min_done_thresholds),
        "regime_interactions": _regime_interactions(done_trades, min_group_done),
        "loss_tail_analysis": _loss_tail_analysis(done_trades),
        "candidate_funnel": _candidate_funnel(candidates, done_trades),
        "next_sweep_recommendations": _recommendation_payload(
            done_trades,
            loaded["strategy_metadata"],
            min_done=max(min(min_done_thresholds), 5),
        ),
        "live_trading_recommendation": "do_not_go_live_from_this_report",
    }
    if output_json_path:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        report["output_json_path"] = str(path)
    if output_txt_path:
        path = Path(output_txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_paper_strategy_experiment_analysis(
                report,
                verbose=bool(verbose_txt),
            ),
            encoding="utf-8",
        )
        report["output_txt_path"] = str(path)
    return report


def recommend_next_paper_sweep(
    *,
    experiment_dir: str,
    output_config_dir: str,
    min_done: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    analysis = analyze_paper_strategy_experiment(
        experiment_dir=experiment_dir,
        min_done_thresholds=DEFAULT_MIN_DONE_THRESHOLDS,
        now=now,
    )
    loaded = _load_experiment(experiment_dir)
    recommendations = _recommendation_payload(
        [
            _enrich_trade(row, loaded["strategy_metadata"])
            for row in loaded["trade_rows"]
            if _status(row) in DONE_STATUSES
        ],
        loaded["strategy_metadata"],
        min_done=min_done,
    )
    output_dir = Path(output_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_configs = _write_recommended_configs(
        recommendations,
        loaded["strategy_metadata"],
        output_dir,
    )
    payload = {
        "status": "ok" if recommendations["suggested_strategies"] else "no_promising_strategies",
        "generated_at": _iso(now or datetime.now(timezone.utc)),
        "experiment_dir": str(Path(experiment_dir)),
        "output_config_dir": str(output_dir),
        "recommendation_note": "paper-only recommendations; do not use as live trading approval",
        "analysis_summary": analysis["summary"],
        "recommendations": recommendations,
        "generated_config_paths": generated_configs,
    }
    recommendation_path = output_dir / "next_sweep_recommendations.json"
    recommendation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    payload["output_json_path"] = str(recommendation_path)
    return payload


def mine_paper_strategies(
    *,
    experiment_dirs: Sequence[str] = (),
    paper_db_paths: Sequence[str] = (),
    paper_db_globs: Sequence[str] = (),
    backtest_json_globs: Sequence[str] = (),
    output_config_dir: str | None = None,
    min_done_trades: int = 30,
    min_closed_trades: int | None = None,
    min_markets: int = 20,
    min_roi: float = 0.0,
    min_win_rate: float = 0.0,
    max_drawdown: float | None = None,
    top_n: int = 10,
    allow_duplicate_market_trades: bool = False,
    output_json_path: str | None = None,
    output_txt_path: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    effective_min_done = int(min_closed_trades if min_closed_trades is not None else min_done_trades)
    experiment_dir_list = [str(path) for path in experiment_dirs]
    resolved_paper_dbs, unmatched_paper_db_globs = _resolve_globbed_paths(
        [*paper_db_paths, *paper_db_globs]
    )
    resolved_backtest_jsons, unmatched_backtest_json_globs = _resolve_globbed_paths(
        backtest_json_globs
    )
    all_trades: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_metadata: dict[str, dict[str, Any]] = {}
    backtest_metric_rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for experiment_dir in experiment_dir_list:
        try:
            loaded = _load_experiment(experiment_dir)
        except (PaperStrategyExperimentAnalysisError, OSError, json.JSONDecodeError) as exc:
            errors.append({"experiment_dir": str(experiment_dir), "error": str(exc)})
            continue
        experiment_id = loaded["manifest"].get("experiment_id")
        metadata = loaded["strategy_metadata"]
        all_metadata.update(metadata)
        trades = [_enrich_trade(row, metadata) for row in loaded["trade_rows"]]
        candidates = [_enrich_candidate(row, metadata) for row in loaded["candidate_rows"]]
        for row in trades:
            row["source_experiment_dir"] = str(Path(experiment_dir))
            row["source_experiment_id"] = experiment_id
        for row in candidates:
            row["source_experiment_dir"] = str(Path(experiment_dir))
            row["source_experiment_id"] = experiment_id
        all_trades.extend(trades)
        all_candidates.extend(candidates)
        done = [row for row in trades if _status(row) in DONE_STATUSES]
        experiments.append(
            {
                "experiment_dir": str(Path(experiment_dir)),
                "experiment_id": experiment_id,
                "input_health": loaded["input_health"],
                "summary": _overall_summary(trades, done, candidates),
            }
        )
    for paper_db_path in resolved_paper_dbs:
        try:
            loaded = _load_paper_db_direct(paper_db_path)
        except (OSError, sqlite3.Error) as exc:
            errors.append({"paper_db_path": str(paper_db_path), "error": str(exc)})
            continue
        trades = [_enrich_trade(row, {}) for row in loaded["trade_rows"]]
        candidates = [_enrich_candidate(row, {}) for row in loaded["candidate_rows"]]
        for row in trades:
            row["source_paper_db"] = str(paper_db_path)
            row["source_type"] = "paper_db"
        for row in candidates:
            row["source_paper_db"] = str(paper_db_path)
            row["source_type"] = "paper_db"
        all_trades.extend(trades)
        all_candidates.extend(candidates)
        done = [row for row in trades if _status(row) in DONE_STATUSES]
        experiments.append(
            {
                "experiment_dir": None,
                "experiment_id": f"paper_db:{Path(paper_db_path).name}",
                "input_health": loaded["input_health"],
                "summary": _overall_summary(trades, done, candidates),
            }
        )
    for backtest_path in resolved_backtest_jsons:
        try:
            backtest_metric_rows.extend(_load_backtest_metric_rows(backtest_path))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"backtest_json_path": str(backtest_path), "error": str(exc)})
    done_trades = [row for row in all_trades if _status(row) in DONE_STATUSES]
    leaderboard = _mined_strategy_leaderboard(
        done_trades,
        backtest_metric_rows=backtest_metric_rows,
        min_done_trades=effective_min_done,
        min_markets=int(min_markets),
        min_roi=float(min_roi),
        min_win_rate=float(min_win_rate),
        max_drawdown=max_drawdown,
        allow_duplicate_market_trades=bool(allow_duplicate_market_trades),
    )
    promising_regimes = _mined_promising_regimes(
        done_trades,
        min_done_trades=effective_min_done,
        min_markets=int(min_markets),
    )
    selected_candidates = _strategy_candidate_rows(
        leaderboard,
        min_done_trades=effective_min_done,
        min_markets=int(min_markets),
        min_roi=float(min_roi),
        min_win_rate=float(min_win_rate),
        max_drawdown=max_drawdown,
        top_n=int(top_n),
    )
    generated_artifacts = (
        _write_mined_strategy_candidate_configs(
            selected_candidates,
            all_metadata,
            output_config_dir=output_config_dir,
            generated_at=now_dt,
        )
        if output_config_dir
        else {
            "generated_config_paths": [],
            "metadata_json_path": None,
            "summary_txt_path": None,
        }
    )
    report = {
        "status": "ok" if experiments or backtest_metric_rows else "no_inputs_loaded",
        "generated_at": _iso(now_dt),
        "experiment_dirs": [str(Path(path)) for path in experiment_dir_list],
        "paper_db_paths": resolved_paper_dbs,
        "paper_db_globs": list(paper_db_globs),
        "backtest_json_paths": resolved_backtest_jsons,
        "backtest_json_globs": list(backtest_json_globs),
        "unmatched_paper_db_globs": unmatched_paper_db_globs,
        "unmatched_backtest_json_globs": unmatched_backtest_json_globs,
        "min_done_trades": effective_min_done,
        "min_closed_trades": effective_min_done,
        "min_markets": int(min_markets),
        "min_roi": float(min_roi),
        "min_win_rate": float(min_win_rate),
        "max_drawdown": max_drawdown,
        "top_n": int(top_n),
        "allow_duplicate_market_trades": bool(allow_duplicate_market_trades),
        "overall_experiment_health": {
            "experiments_requested": len(experiment_dir_list),
            "experiments_loaded": len(experiments),
            "load_errors": errors,
            "total_trades": len(all_trades),
            "done_trades": len(done_trades),
            "total_candidates": len(all_candidates),
            "paper_dbs_loaded": len(resolved_paper_dbs),
            "backtest_jsons_loaded": len(resolved_backtest_jsons),
            "backtest_metric_rows_loaded": len(backtest_metric_rows),
            "markets_covered": _market_count_for_rows(done_trades),
            "summary": _overall_summary(all_trades, done_trades, all_candidates),
            "experiments": experiments,
        },
        "strategy_leaderboard": leaderboard,
        "promising_regimes": promising_regimes,
        "bad_loss_tail_strategies": _bad_loss_tail_strategies(
            leaderboard,
            min_done_trades=effective_min_done,
            min_markets=int(min_markets),
        ),
        "suggested_next_paper_sweep_configs": _mined_next_sweep_suggestions(
            leaderboard,
            all_metadata,
            min_done_trades=effective_min_done,
            min_markets=int(min_markets),
        ),
        "selected_strategy_candidates": selected_candidates,
        "generated_config_paths": generated_artifacts["generated_config_paths"],
        "metadata_json_path": generated_artifacts["metadata_json_path"],
        "summary_txt_path": generated_artifacts["summary_txt_path"],
        "recommendation": "paper/shadow validation only; never a live trading recommendation",
    }
    if output_config_dir:
        _write_mined_metadata(report, output_config_dir)
    if output_json_path:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        report["output_json_path"] = str(path)
    if output_txt_path:
        path = Path(output_txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_mined_paper_strategies(report), encoding="utf-8")
        report["output_txt_path"] = str(path)
    return report


def render_mined_paper_strategies(report: dict[str, Any]) -> str:
    health = dict(report.get("overall_experiment_health") or {})
    summary = dict(health.get("summary") or {})
    lines = [
        "Paper Strategy Mining Report",
        f"generated_at: {report.get('generated_at')}",
        f"experiments_loaded: {health.get('experiments_loaded')} / {health.get('experiments_requested')}",
        f"paper_dbs_loaded: {health.get('paper_dbs_loaded', 0)}",
        f"backtest_jsons_loaded: {health.get('backtest_jsons_loaded', 0)}",
        (
            f"done_trades={summary.get('done_trades')} markets={health.get('markets_covered')} "
            f"pnl={summary.get('pnl')} roi={summary.get('roi')} win_rate={summary.get('win_rate')}"
        ),
        (
            f"minimums: closed_trades={report.get('min_closed_trades')} "
            f"markets={report.get('min_markets')} roi={report.get('min_roi')} "
            f"win_rate={report.get('min_win_rate')}"
        ),
        "Recommendation: paper/shadow validation only; DO NOT GO LIVE",
        "",
        "Strategy Leaderboard",
    ]
    for row in list(report.get("strategy_leaderboard") or [])[:10]:
        lines.append(
            f"  {row.get('strategy_id')}: status={row.get('sample_status')} "
            f"done={row.get('done_trades')} markets={row.get('markets_covered')} "
            f"pnl={row.get('pnl')} avg_roi={row.get('average_roi')} "
            f"win_rate={row.get('win_rate')} payoff={row.get('payoff_ratio')} "
            f"worst={row.get('worst_trade')} score={row.get('candidate_score')} "
            f"duplicates={row.get('duplicate_market_groups', 0)}"
        )
    lines.extend(["", "Promising Regimes"])
    for field, rows in dict(report.get("promising_regimes") or {}).items():
        lines.append(f"  {field}")
        for row in list(rows or [])[:5]:
            label = row.get(field)
            lines.append(
                f"    {label}: done={row.get('done_trades')} markets={row.get('markets_covered')} "
                f"pnl={row.get('pnl')} avg_roi={row.get('average_roi')} win_rate={row.get('win_rate')}"
            )
    lines.extend(["", "Bad Loss-Tail Strategies"])
    for row in list(report.get("bad_loss_tail_strategies") or [])[:10]:
        lines.append(
            f"  {row.get('strategy_id')}: done={row.get('done_trades')} "
            f"win_rate={row.get('win_rate')} avg_win={row.get('average_win')} "
            f"avg_loss={row.get('average_loss')} worst={row.get('worst_trade')} "
            f"reason={row.get('loss_tail_reason')}"
        )
    lines.extend(["", "Suggested Next Paper-Only Sweep Configs"])
    for row in list(report.get("selected_strategy_candidates") or report.get("suggested_next_paper_sweep_configs") or [])[:10]:
        lines.append(
            f"  {row.get('strategy_id')}: source={row.get('source_strategy_id') or row.get('strategy_id')} "
            f"reason={row.get('selection_reason') or row.get('reason')}"
        )
    if not (report.get("selected_strategy_candidates") or report.get("suggested_next_paper_sweep_configs")):
        lines.append("  none")
    if report.get("generated_config_paths"):
        lines.extend(["", "Generated Configs"])
        for path in report.get("generated_config_paths") or []:
            lines.append(f"  {path}")
        lines.append(f"metadata: {report.get('metadata_json_path')}")
    lines.append("")
    lines.append("Never recommend live trading from this report.")
    return "\n".join(lines) + "\n"


def render_paper_strategy_experiment_analysis(
    report: dict[str, Any],
    *,
    verbose: bool = False,
) -> str:
    summary = dict(report.get("summary") or {})
    warnings = _executive_warnings(summary)
    top_limit = TOP_N if bool(verbose) else 5
    lines = [
        "Paper Strategy Experiment Analysis",
        f"experiment_id: {report.get('experiment_id')}",
        f"generated_at: {report.get('generated_at')}",
        "",
        "Executive Summary",
        (
            f"Overall PnL={summary.get('pnl')} ROI={summary.get('roi')} "
            f"win_rate={summary.get('win_rate')} payoff_ratio={summary.get('payoff_ratio')}"
        ),
        f"Live Trading Recommendation: {_live_trading_recommendation(summary)}",
        "Warnings: " + (", ".join(warnings) if warnings else "none"),
        "",
        f"Top {top_limit} Strategies by PnL (done_trades >= 5)",
    ]
    ranking = dict(report.get("strategy_rankings") or {})
    for row in list((ranking.get("min_done_5") or {}).get("sorted_by_pnl") or [])[:top_limit]:
        lines.append("  " + _format_strategy_line(row))
    if not list((ranking.get("min_done_5") or {}).get("sorted_by_pnl") or []):
        lines.append("  none")
    lines.extend(["", f"Top {top_limit} Strategies by Avg ROI (done_trades >= 20)"])
    avg_roi_rows = list((ranking.get("min_done_20") or {}).get("sorted_by_avg_roi") or [])
    for row in avg_roi_rows[:top_limit]:
        lines.append("  " + _format_strategy_line(row))
    if not avg_roi_rows:
        lines.append("  none")
    lines.extend(["", f"Worst {top_limit} Loss Drivers"])
    for row in _top_loss_drivers(report)[:top_limit]:
        lines.append(
            f"  {row.get('driver')}: loss_trades={row.get('losing_trades')} "
            f"total_loss={row.get('total_loss')} avg_loss={row.get('average_loss')} "
            f"worst={row.get('worst_trade')}"
        )
    if not _top_loss_drivers(report):
        lines.append("  none")
    lines.extend(["", "Suggested Next Sweep Configs"])
    for line in _suggested_next_sweep_lines(report)[:top_limit]:
        lines.append(f"  {line}")
    if not _suggested_next_sweep_lines(report):
        lines.append("  none")
    lines.extend(
        [
            "",
            "Details",
            (
                f"trades total={summary.get('total_trades')} done={summary.get('done_trades')} "
                f"closed={summary.get('closed_trades')} settled={summary.get('settled_trades')} "
                f"skipped={summary.get('skipped_trades')}"
            ),
            (
                f"pnl={summary.get('pnl')} roi={summary.get('roi')} avg_roi={summary.get('average_roi')} "
                f"win_rate={summary.get('win_rate')} avg_win={summary.get('average_win')} "
                f"avg_loss={summary.get('average_loss')} payoff_ratio={summary.get('payoff_ratio')}"
            ),
            "note: negative expectancy can come from loss size even when win rate is near breakeven.",
            "",
            "Top Strategies By PnL (min_done_5)",
        ]
    )
    min_done_5 = dict(ranking.get("min_done_5") or {})
    for row in list(min_done_5.get("sorted_by_pnl") or [])[:top_limit]:
        lines.append(
            f"  {row.get('strategy_id')}: done={row.get('done_trades')} pnl={row.get('pnl')} "
            f"avg_roi={row.get('average_roi')} win_rate={row.get('win_rate')} "
            f"payoff={row.get('payoff_ratio')} small_sample={row.get('small_sample')}"
        )
    lines.extend(["", "Top Strategies By Avg ROI (min_done_5)"])
    for row in list(min_done_5.get("sorted_by_avg_roi") or [])[:top_limit]:
        lines.append(
            f"  {row.get('strategy_id')}: done={row.get('done_trades')} pnl={row.get('pnl')} "
            f"avg_roi={row.get('average_roi')} win_rate={row.get('win_rate')}"
        )
    lines.extend(["", "Largest Losing Trades"])
    for row in list((report.get("loss_tail_analysis") or {}).get("largest_losing_trades") or [])[:top_limit]:
        lines.append(
            f"  id={row.get('paper_trade_id')} strategy={row.get('strategy_id')} "
            f"pnl={row.get('pnl')} roi={row.get('roi')} time={row.get('time_regime')} "
            f"liq={row.get('liquidity_regime')} spread={row.get('spread_regime')} btc={row.get('btc_trend_regime')}"
        )
    lines.extend(["", "Candidate Funnel"])
    funnel = dict(report.get("candidate_funnel") or {})
    lines.append(f"candidate_decision_counts: {json.dumps(funnel.get('decision_counts') or {}, sort_keys=True)}")
    lines.append("Candidate Rejections By Reason")
    for reason, count in _top_count_items(funnel.get("rejection_reasons") or {}, limit=top_limit):
        lines.append(f"  {reason}: count={count}")
    if not funnel.get("rejection_reasons"):
        lines.append("  none")
    lines.extend(["", "Next Sweep"])
    recs = dict(report.get("next_sweep_recommendations") or {})
    lines.append(f"status: {recs.get('status')}")
    for row in list(recs.get("suggested_strategies") or [])[:top_limit]:
        sample_warning = (
            " warning=below_30_done_trades"
            if int(row.get("done_trades") or 0) < 30
            else ""
        )
        lines.append(
            f"  {row.get('strategy_id')} reason={row.get('reason')} "
            f"done={row.get('done_trades')} pnl={row.get('pnl')} avg_roi={row.get('average_roi')}"
            f"{sample_warning}"
        )
    lines.append("")
    lines.append("Live trading recommendation: DO NOT GO LIVE from this analysis alone.")
    return "\n".join(lines) + "\n"


def _executive_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = ["do not go live"]
    done = int(summary.get("done_trades") or 0)
    if done < 100:
        warnings.append("small sample")
    pnl = _float_or_none(summary.get("pnl"))
    average_roi = _float_or_none(summary.get("average_roi"))
    roi = _float_or_none(summary.get("roi"))
    if (pnl is not None and pnl < 0) or (average_roi is not None and average_roi < 0) or (roi is not None and roi < 0):
        warnings.append("negative overall expectancy")
    return warnings


def _live_trading_recommendation(summary: dict[str, Any]) -> str:
    warnings = set(_executive_warnings(summary))
    if "small sample" in warnings or "negative overall expectancy" in warnings:
        return "DO NOT GO LIVE"
    return "DO NOT GO LIVE; paper-only evidence requires independent validation"


def _format_strategy_line(row: dict[str, Any]) -> str:
    return (
        f"{row.get('strategy_id')}: done={row.get('done_trades')} pnl={row.get('pnl')} "
        f"avg_roi={row.get('average_roi')} win_rate={row.get('win_rate')} "
        f"payoff={row.get('payoff_ratio')}"
    )


def _top_loss_drivers(report: dict[str, Any]) -> list[dict[str, Any]]:
    loss_tail = dict(report.get("loss_tail_analysis") or {})
    summary = dict(loss_tail.get("loss_driver_summary") or {})
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for driver_type, groups in summary.items():
        for item in list(groups or []):
            labels = [
                f"{key}={value}"
                for key, value in item.items()
                if key not in {"losing_trades", "average_loss", "total_loss", "worst_trade"}
            ]
            label = f"{driver_type} " + " ".join(labels)
            if label in seen:
                continue
            seen.add(label)
            rows.append({"driver": label.strip(), **item})
    rows.sort(
        key=lambda item: (
            float(item.get("total_loss") or 0.0),
            float(item.get("worst_trade") or 0.0),
            str(item.get("driver") or ""),
        )
    )
    return rows


def _suggested_next_sweep_lines(report: dict[str, Any]) -> list[str]:
    recs = dict(report.get("next_sweep_recommendations") or {})
    lines: list[str] = []
    for row in list(recs.get("suggested_strategies") or []):
        sample_warning = (
            " warning=below_30_done_trades"
            if int(row.get("done_trades") or 0) < 30
            else ""
        )
        lines.append(
            f"{row.get('strategy_id')} done={row.get('done_trades')} pnl={row.get('pnl')} "
            f"avg_roi={row.get('average_roi')} reason={row.get('reason')}{sample_warning}"
        )
    for row in list(recs.get("exploration_variants") or []):
        strategy = dict(row.get("strategy") or {})
        strategy_id = strategy.get("strategy_id")
        if not strategy_id:
            continue
        lines.append(
            f"{strategy_id} source={row.get('source_strategy_id')} "
            f"variant={row.get('variant_type')}"
        )
    return lines


def _top_count_items(counts: dict[str, int], *, limit: int = TOP_N) -> list[tuple[str, int]]:
    return sorted(
        ((str(key), int(value)) for key, value in counts.items() if key not in (None, "")),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[: int(limit)]


def _load_experiment(experiment_dir: str) -> dict[str, Any]:
    root = Path(experiment_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise PaperStrategyExperimentAnalysisError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategy_metadata = _load_strategy_metadata(manifest, root)
    trade_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    paper_db_health: list[dict[str, Any]] = []
    workers = list(manifest.get("workers") or [])
    for worker in workers:
        paper_db_path = str(worker.get("paper_db_path") or "")
        if not paper_db_path:
            paper_db_health.append({"status": "missing_path", "paper_db_path": paper_db_path})
            continue
        db_path = Path(paper_db_path)
        if not db_path.exists():
            paper_db_health.append({"status": "missing", "paper_db_path": paper_db_path})
            continue
        conn = _connect_read_only(db_path)
        try:
            trades = _read_table(conn, "paper_trades")
            candidates = _read_table(conn, "paper_trade_candidates")
        finally:
            conn.close()
        for row in trades:
            row["source_paper_db"] = paper_db_path
            row["source_config_path"] = worker.get("config_path")
            row["source_worker_id"] = worker.get("worker_id")
        for row in candidates:
            row["source_paper_db"] = paper_db_path
            row["source_config_path"] = worker.get("config_path")
            row["source_worker_id"] = worker.get("worker_id")
        trade_rows.extend(trades)
        candidate_rows.extend(candidates)
        paper_db_health.append(
            {
                "status": "ok",
                "paper_db_path": paper_db_path,
                "trade_rows": len(trades),
                "candidate_rows": len(candidates),
            }
        )
    return {
        "manifest": manifest,
        "strategy_metadata": strategy_metadata,
        "trade_rows": trade_rows,
        "candidate_rows": candidate_rows,
        "input_health": {
            "manifest_path": str(manifest_path),
            "paper_db_count": len(workers),
            "trade_rows_loaded": len(trade_rows),
            "candidate_rows_loaded": len(candidate_rows),
            "paper_dbs": paper_db_health,
        },
    }


def _load_strategy_metadata(manifest: dict[str, Any], experiment_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for config_path in _config_paths_from_manifest(manifest):
        resolved = _resolve_config_path(config_path, experiment_root)
        if resolved is None:
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        top_level = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key != "strategies"
        }
        bankroll = payload.get("bankroll") if isinstance(payload.get("bankroll"), dict) else {}
        if not bankroll and bool(payload.get("bankroll_enabled")):
            bankroll = {
                key: payload.get(key)
                for key in (
                    "bankroll_enabled",
                    "starting_bankroll_usd",
                    "base_risk_fraction",
                    "max_risk_fraction",
                    "max_total_exposure_fraction",
                    "min_stake_usd",
                    "max_stake_usd",
                )
                if key in payload
            }
        for strategy in list(payload.get("strategies") or []):
            if not isinstance(strategy, dict):
                continue
            strategy_id = str(strategy.get("strategy_id") or "")
            if not strategy_id:
                continue
            direction_mode = str(strategy.get("direction_mode") or "BOTH")
            threshold = _strategy_threshold(strategy, direction_mode)
            min_time = _float_or_none(strategy.get("min_time_until_resolution_sec"))
            max_time = _float_or_none(strategy.get("max_time_until_resolution_sec"))
            pmax = _pmax(strategy)
            metadata[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_name": strategy.get("strategy_name"),
                "direction_mode": direction_mode,
                "threshold": threshold,
                "long_threshold": _float_or_none(strategy.get("long_threshold")),
                "short_threshold": _float_or_none(strategy.get("short_threshold")),
                "min_time_until_resolution_sec": min_time,
                "max_time_until_resolution_sec": max_time,
                "time_window": _format_time_window(min_time, max_time),
                "pmax": _format_pmax(pmax),
                "max_probability_for_direction": pmax,
                "bankroll": _bankroll_label(bankroll),
                "config_path": str(resolved),
                "top_level_config": top_level,
                "strategy_config": copy.deepcopy(strategy),
            }
    return metadata


def _config_paths_from_manifest(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for worker in list(manifest.get("workers") or []):
        if worker.get("config_path"):
            paths.append(str(worker["config_path"]))
    for path in list(manifest.get("config_paths") or []):
        paths.append(str(path))
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _resolve_config_path(path: str, experiment_root: Path) -> Path | None:
    candidates = [Path(path), experiment_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _enrich_trade(row: dict[str, Any], strategy_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(row)
    meta = strategy_metadata.get(str(result.get("strategy_id") or "")) or {}
    _merge_defaults(result, meta)
    result["pnl"] = _result_pnl(result)
    result["roi"] = _result_roi(result)
    result["stake_usd"] = _float_or_none(result.get("stake_usd")) or 1.0
    result["analysis_threshold"] = _threshold_for_row(result, meta)
    result["threshold_group"] = _format_number(result["analysis_threshold"])
    min_time = _float_or_none(result.get("min_time_until_resolution_sec"))
    max_time = _float_or_none(result.get("max_time_until_resolution_sec"))
    if min_time is None:
        min_time = meta.get("min_time_until_resolution_sec")
    if max_time is None:
        max_time = meta.get("max_time_until_resolution_sec")
    result["time_window"] = result.get("time_window") or _format_time_window(
        min_time,
        max_time,
        result.get("time_regime"),
        result.get("time_until_resolution"),
    )
    result["pmax"] = _format_pmax(
        _float_or_none(result.get("max_probability_for_direction"))
        if result.get("max_probability_for_direction") not in (None, "")
        else meta.get("max_probability_for_direction")
    )
    result["bankroll"] = result.get("bankroll") or _format_bankroll_from_row(result) or meta.get("bankroll") or "unknown"
    return result


def _enrich_candidate(row: dict[str, Any], strategy_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(row)
    meta = strategy_metadata.get(str(result.get("strategy_id") or "")) or {}
    _merge_defaults(result, meta)
    result["analysis_threshold"] = _threshold_for_row(result, meta)
    result["threshold_group"] = _format_number(result["analysis_threshold"])
    min_time = _float_or_none(result.get("min_time_until_resolution_sec"))
    max_time = _float_or_none(result.get("max_time_until_resolution_sec"))
    if min_time is None:
        min_time = meta.get("min_time_until_resolution_sec")
    if max_time is None:
        max_time = meta.get("max_time_until_resolution_sec")
    result["time_window"] = _format_time_window(
        min_time,
        max_time,
        result.get("time_regime"),
        result.get("time_until_resolution"),
    )
    result["pmax"] = _format_pmax(
        _float_or_none(result.get("max_probability_for_direction"))
        if result.get("max_probability_for_direction") not in (None, "")
        else meta.get("max_probability_for_direction")
    )
    result["bankroll"] = _format_bankroll_from_row(result) or meta.get("bankroll") or "unknown"
    return result


def _merge_defaults(row: dict[str, Any], meta: dict[str, Any]) -> None:
    for key in (
        "strategy_name",
        "direction_mode",
        "config_path",
        "long_threshold",
        "short_threshold",
        "min_time_until_resolution_sec",
        "max_time_until_resolution_sec",
        "max_probability_for_direction",
    ):
        if row.get(key) in (None, "") and meta.get(key) not in (None, ""):
            row[key] = meta[key]


def _threshold_for_row(row: dict[str, Any], meta: dict[str, Any]) -> float | None:
    meta_threshold = _float_or_none(meta.get("threshold"))
    if meta_threshold is not None:
        return meta_threshold
    raw = _float_or_none(row.get("threshold_used"))
    if raw is None:
        return None
    direction = str(row.get("signal_direction") or row.get("candidate_direction") or "")
    direction_mode = str(row.get("direction_mode") or meta.get("direction_mode") or "")
    if direction == "NO" or direction_mode == "NO_ONLY":
        return 1.0 - raw if 0.0 <= raw <= 1.0 else raw
    return raw


def _overall_summary(
    trades: Sequence[dict[str, Any]],
    done_trades: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _performance_metrics(done_trades)
    metrics.update(
        {
            "total_trades": len(trades),
            "done_trades": len(done_trades),
            "closed_trades": sum(1 for row in trades if _status(row) == "closed"),
            "settled_trades": sum(1 for row in trades if _status(row) == "settled"),
            "open_trades": sum(1 for row in trades if _status(row) == "open"),
            "awaiting_resolution_trades": sum(1 for row in trades if _status(row) == "awaiting_resolution"),
            "skipped_trades": sum(1 for row in trades if _status(row) == "skipped"),
            "total_candidates": len(candidates),
            "candidate_decision_counts": _counts(row.get("decision") for row in candidates),
        }
    )
    return metrics


def _strategy_rankings(
    done_trades: Sequence[dict[str, Any]],
    min_done_thresholds: Sequence[int],
) -> dict[str, Any]:
    all_rows = _grouped_metrics(done_trades, ("strategy_id",), include_small=True)
    rankings: dict[str, Any] = {"all": all_rows}
    for threshold in min_done_thresholds:
        eligible = [row for row in all_rows if int(row.get("done_trades") or 0) >= int(threshold)]
        rankings[f"min_done_{threshold}"] = {
            "threshold": int(threshold),
            "sorted_by_pnl": sorted(
                eligible,
                key=lambda row: (float(row.get("pnl") or 0.0), float(row.get("average_roi") or -999.0)),
                reverse=True,
            ),
            "sorted_by_avg_roi": sorted(
                eligible,
                key=lambda row: (float(row.get("average_roi") or -999.0), float(row.get("pnl") or 0.0)),
                reverse=True,
            ),
            "small_sample_rows": [
                row for row in all_rows if int(row.get("done_trades") or 0) < int(threshold)
            ],
        }
    return rankings


def _regime_interactions(
    done_trades: Sequence[dict[str, Any]],
    min_group_done: int,
) -> dict[str, Any]:
    specs = {
        "strategy_id_x_time_regime": ("strategy_id", "time_regime"),
        "strategy_id_x_btc_trend_regime": ("strategy_id", "btc_trend_regime"),
        "strategy_id_x_spread_regime": ("strategy_id", "spread_regime"),
        "strategy_id_x_liquidity_regime": ("strategy_id", "liquidity_regime"),
        "time_regime_x_btc_trend_regime_x_spread_regime": (
            "time_regime",
            "btc_trend_regime",
            "spread_regime",
        ),
    }
    result: dict[str, Any] = {}
    for name, fields in specs.items():
        all_rows = _grouped_metrics(done_trades, fields, include_small=True)
        included = [row for row in all_rows if int(row.get("done_trades") or 0) >= int(min_group_done)]
        result[name] = {
            "min_done_trades": int(min_group_done),
            "groups": included,
            "excluded_small_sample_count": len(all_rows) - len(included),
            "small_sample_warning": bool(any(int(row.get("done_trades") or 0) < 30 for row in included)),
        }
    return result


def _loss_tail_analysis(done_trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    losing = [row for row in done_trades if (_float_or_none(row.get("pnl")) or 0.0) < 0]
    winning = [row for row in done_trades if (_float_or_none(row.get("pnl")) or 0.0) > 0]
    return {
        "largest_losing_trades": [
            _trade_snapshot(row)
            for row in sorted(losing, key=lambda row: float(row.get("pnl") or 0.0))[:TOP_N]
        ],
        "largest_winning_trades": [
            _trade_snapshot(row)
            for row in sorted(winning, key=lambda row: float(row.get("pnl") or 0.0), reverse=True)[:TOP_N]
        ],
        "average_loss_by_strategy": _loss_groups(losing, ("strategy_id",)),
        "average_loss_by_time_regime": _loss_groups(losing, ("time_regime",)),
        "average_loss_by_liquidity_regime": _loss_groups(losing, ("liquidity_regime",)),
        "average_loss_by_spread_regime": _loss_groups(losing, ("spread_regime",)),
        "average_loss_by_btc_trend_regime": _loss_groups(losing, ("btc_trend_regime",)),
        "pmax_loss_tail_comparison": _grouped_metrics(done_trades, ("pmax",), include_small=True),
        "loss_driver_summary": {
            "entry_price_drift": _loss_groups(losing, ("entry_price_drift_bucket",)),
            "spread_cents_at_entry": _loss_groups(losing, ("spread_cents_bucket",)),
            "liquidity_regime": _loss_groups(losing, ("liquidity_regime",)),
            "time_regime": _loss_groups(losing, ("time_regime",)),
            "btc_trend_regime": _loss_groups(losing, ("btc_trend_regime",)),
        },
    }


def _candidate_funnel(
    candidates: Sequence[dict[str, Any]],
    done_trades: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = _counts(row.get("decision") for row in candidates)
    return {
        "candidate_count": len(candidates),
        "decision_counts": decision_counts,
        "rejection_reasons": _counts(row.get("rejection_reason") for row in candidates),
        "rejections_by_strategy_id": _count_group(candidates, ("strategy_id", "rejection_reason")),
        "rejections_by_threshold": _count_group(candidates, ("threshold_group", "rejection_reason")),
        "rejections_by_time_window": _count_group(candidates, ("time_window", "rejection_reason")),
        "rejections_by_pmax": _count_group(candidates, ("pmax", "rejection_reason")),
        "rejections_by_bankroll": _count_group(candidates, ("bankroll", "rejection_reason")),
        "trade_conversion_by_config": _conversion_groups(candidates, ("config_path",)),
        "trade_conversion_by_strategy": _conversion_groups(candidates, ("strategy_id",)),
        "done_trade_count_by_strategy": _counts(row.get("strategy_id") for row in done_trades),
    }


def _recommendation_payload(
    done_trades: Sequence[dict[str, Any]],
    strategy_metadata: dict[str, dict[str, Any]],
    *,
    min_done: int,
) -> dict[str, Any]:
    strategy_rows = _grouped_metrics(done_trades, ("strategy_id",), include_small=True)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in strategy_rows:
        strategy_id = str(row.get("strategy_id") or "")
        reason = _recommendation_reject_reason(row, min_done=min_done)
        if reason:
            rejected.append({"strategy_id": strategy_id, "reason": reason, **row})
            continue
        selected.append(
            {
                "strategy_id": strategy_id,
                "reason": "positive expectancy with acceptable payoff ratio",
                "negative_regime_notes": _negative_regime_notes(done_trades, strategy_id),
                **row,
            }
        )
    selected.sort(
        key=lambda row: (
            float(row.get("pnl") or 0.0),
            float(row.get("average_roi") or 0.0),
            int(row.get("done_trades") or 0),
        ),
        reverse=True,
    )
    exploration = _exploration_variants(selected, strategy_metadata)
    return {
        "status": "ok" if selected else "no_promising_strategies",
        "min_done_trades": int(min_done),
        "suggested_strategies": selected[:TOP_N],
        "rejected_strategy_count": len(rejected),
        "rejected_strategies": rejected[:TOP_N],
        "exploration_variants": exploration,
        "rules": [
            "prefer groups with minimum sample size",
            "exclude groups with bad payoff ratio",
            "exclude negative-expectancy regimes unless sample is too small",
            "include exploration variants around promising areas",
            "do not recommend live trading",
        ],
    }


def _recommendation_reject_reason(row: dict[str, Any], *, min_done: int) -> str | None:
    done = int(row.get("done_trades") or 0)
    if done < int(min_done):
        return "too_few_done_trades"
    if _float_or_none(row.get("pnl")) is None or float(row.get("pnl") or 0.0) <= 0:
        return "negative_or_zero_pnl"
    if _float_or_none(row.get("average_roi")) is None or float(row.get("average_roi") or 0.0) <= 0:
        return "negative_or_zero_average_roi"
    payoff = _float_or_none(row.get("payoff_ratio"))
    if payoff is None or payoff < 1.0:
        return "bad_payoff_ratio"
    win_rate = _float_or_none(row.get("win_rate"))
    breakeven = _float_or_none(row.get("breakeven_win_rate"))
    if win_rate is not None and breakeven is not None and win_rate < breakeven:
        return "below_breakeven_win_rate"
    return None


def _negative_regime_notes(done_trades: Sequence[dict[str, Any]], strategy_id: str) -> list[str]:
    rows = [row for row in done_trades if str(row.get("strategy_id") or "") == strategy_id]
    notes: list[str] = []
    for field in ("time_regime", "liquidity_regime", "spread_regime", "btc_trend_regime"):
        for group in _grouped_metrics(rows, (field,), include_small=True):
            if int(group.get("done_trades") or 0) >= DEFAULT_MIN_GROUP_DONE and float(group.get("pnl") or 0.0) < 0:
                notes.append(f"{field}={group.get(field)} has negative pnl {group.get('pnl')}")
    return notes


def _exploration_variants(
    selected: Sequence[dict[str, Any]],
    strategy_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for row in selected[:10]:
        strategy_id = str(row.get("strategy_id") or "")
        meta = strategy_metadata.get(strategy_id) or {}
        strategy = copy.deepcopy(meta.get("strategy_config") or {})
        if not strategy:
            continue
        variants.append({"source_strategy_id": strategy_id, "variant_type": "base", "strategy": strategy})
        threshold = _float_or_none(row.get("threshold"))
        direction_mode = str(strategy.get("direction_mode") or "")
        for delta in (-0.02, 0.02):
            if threshold is None:
                continue
            variant = copy.deepcopy(strategy)
            new_threshold = max(0.01, min(0.99, threshold + delta))
            if direction_mode == "NO_ONLY":
                variant["short_threshold"] = round(1.0 - new_threshold, 4)
            elif direction_mode == "YES_ONLY":
                variant["long_threshold"] = round(new_threshold, 4)
            else:
                variant["long_threshold"] = round(new_threshold, 4)
                variant["short_threshold"] = round(1.0 - new_threshold, 4)
            variant["strategy_id"] = f"{strategy_id}_explore_t{int(round(new_threshold * 100)):03d}"
            variant["strategy_name"] = f"{strategy.get('strategy_name') or strategy_id} explore {new_threshold:.2f}"
            variants.append(
                {
                    "source_strategy_id": strategy_id,
                    "variant_type": f"threshold_delta_{delta:+.2f}",
                    "strategy": variant,
                }
            )
    return variants[:30]


def _write_recommended_configs(
    recommendations: dict[str, Any],
    strategy_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    variants = list(recommendations.get("exploration_variants") or [])
    if not variants:
        return []
    base_meta = strategy_metadata.get(str(variants[0].get("source_strategy_id") or "")) or {}
    top_level = copy.deepcopy(base_meta.get("top_level_config") or {})
    top_level["live_trading"] = {"live_trading_enabled": False, "dry_run_orders": True}
    top_level["recommendation_note"] = "paper-only generated config; not live trading approval"
    strategies = [copy.deepcopy(item["strategy"]) for item in variants if item.get("strategy")]
    payload = {**top_level, "strategies": strategies}
    payload = _force_paper_only_config(payload)
    config_path = output_dir / "recommended_sweep.json"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return [str(config_path)]


def _load_paper_db_direct(paper_db_path: str) -> dict[str, Any]:
    db_path = Path(paper_db_path)
    conn = _connect_read_only(db_path)
    try:
        trades = _read_table(conn, "paper_trades")
        candidates = _read_table(conn, "paper_trade_candidates")
    finally:
        conn.close()
    for row in trades:
        row["source_paper_db"] = str(db_path)
    for row in candidates:
        row["source_paper_db"] = str(db_path)
    return {
        "trade_rows": trades,
        "candidate_rows": candidates,
        "input_health": {
            "status": "ok",
            "paper_db_path": str(db_path),
            "trade_rows": len(trades),
            "candidate_rows": len(candidates),
        },
    }


def _load_backtest_metric_rows(backtest_json_path: str) -> list[dict[str, Any]]:
    path = Path(backtest_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    validation = payload.get("validation")
    if isinstance(validation, dict):
        for item in list(validation.get("strategy_comparison") or []):
            if not isinstance(item, dict):
                continue
            strategy_id = str(item.get("strategy_id") or "")
            metrics = dict(item.get("test") or {})
            if not strategy_id or not metrics:
                continue
            row = _normalize_backtest_metric_row(
                strategy_id=strategy_id,
                metrics=metrics,
                payload=payload,
                source_path=path,
                source_split="test",
            )
            row["train_metrics"] = item.get("train")
            row["test_metrics"] = item.get("test")
            row["validation_delta"] = item.get("delta")
            row["validation_warnings"] = item.get("warnings") or []
            rows[strategy_id] = row
    for collection_name in ("best_by_pnl", "best_by_avg_roi_min_done"):
        for item in list(payload.get(collection_name) or []):
            if not isinstance(item, dict):
                continue
            strategy_id = str(item.get("strategy_id") or "")
            if not strategy_id or strategy_id in rows:
                continue
            rows[strategy_id] = _normalize_backtest_metric_row(
                strategy_id=strategy_id,
                metrics=item,
                payload=payload,
                source_path=path,
                source_split="all",
            )
    return list(rows.values())


def _normalize_backtest_metric_row(
    *,
    strategy_id: str,
    metrics: dict[str, Any],
    payload: dict[str, Any],
    source_path: Path,
    source_split: str,
) -> dict[str, Any]:
    parsed = _parse_transformer_strategy_id(strategy_id)
    row = {
        "strategy_id": strategy_id,
        "source_type": "transformer_backtest",
        "source_backtest_json": str(source_path),
        "source_split": source_split,
        "done_trades": int(metrics.get("done_trades") or 0),
        "trades": int(metrics.get("trades") or metrics.get("done_trades") or 0),
        "pnl": _round(metrics.get("pnl")) or 0.0,
        "roi": _round(metrics.get("roi")),
        "average_roi": _round(metrics.get("average_roi")),
        "win_rate": _round(metrics.get("win_rate")),
        "average_win": _round(metrics.get("average_win")),
        "average_loss": _round(metrics.get("average_loss")),
        "payoff_ratio": _round(metrics.get("payoff_ratio")),
        "breakeven_win_rate": _round(metrics.get("breakeven_win_rate")),
        "worst_trade": _round(metrics.get("worst_trade")),
        "best_trade": _round(metrics.get("best_trade")),
        "markets_covered": int(metrics.get("markets_covered") or 0),
        "duplicate_market_groups": 0,
        "duplicate_trade_count": 0,
        "direction_mode": _direction_mode_from_side(parsed.get("side")),
        "signal_direction": parsed.get("side"),
        "threshold": parsed.get("threshold") or _float_or_none(metrics.get("threshold")),
        "min_time_until_resolution_sec": parsed.get("min_time"),
        "max_time_until_resolution_sec": parsed.get("max_time"),
        "fixed_horizon_exit_sec": parsed.get("horizon"),
        "exit_horizon_sec": parsed.get("horizon"),
        "slippage_cents": _float_or_none(payload.get("slippage_cents")) or 0.0,
        "fee_cents": _float_or_none(payload.get("fee_cents")) or 0.0,
        "recommendation": payload.get("recommendation"),
    }
    if row["breakeven_win_rate"] is None and row["payoff_ratio"] not in (None, 0):
        row["breakeven_win_rate"] = _round(1.0 / (1.0 + float(row["payoff_ratio"])))
    return row


def _parse_transformer_strategy_id(strategy_id: str) -> dict[str, Any]:
    parts = strategy_id.split("_")
    result: dict[str, Any] = {}
    if len(parts) >= 2 and parts[0] == "transformer":
        side = parts[1].upper()
        if side in {"YES", "NO"}:
            result["side"] = side
    for part in parts:
        if part.startswith("t") and len(part) > 1 and part[1:].isdigit():
            digits = part[1:]
            result["threshold"] = int(digits) / float(10 ** max(1, len(digits) - 1))
        elif part.startswith("fh") and part[2:].isdigit():
            result["horizon"] = int(part[2:])
    numeric_parts = [part for part in parts if part.replace(".", "", 1).isdigit()]
    if len(numeric_parts) >= 2:
        result["min_time"] = float(numeric_parts[-2])
        result["max_time"] = float(numeric_parts[-1])
    return result


def _resolve_globbed_paths(inputs: Sequence[str]) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for raw in inputs:
        text = str(raw).strip()
        if not text:
            continue
        matches = sorted(glob.glob(text)) if glob.has_magic(text) else [text]
        if glob.has_magic(text) and not matches:
            unmatched.append(text)
            continue
        for match in matches:
            if match not in seen:
                seen.add(match)
                paths.append(match)
    return paths, unmatched


def _mined_strategy_leaderboard(
    done_trades: Sequence[dict[str, Any]],
    *,
    backtest_metric_rows: Sequence[dict[str, Any]] = (),
    min_done_trades: int,
    min_markets: int,
    min_roi: float = 0.0,
    min_win_rate: float = 0.0,
    max_drawdown: float | None = None,
    allow_duplicate_market_trades: bool = False,
) -> list[dict[str, Any]]:
    rows = _grouped_metrics(done_trades, ("strategy_id",), include_small=True)
    duplicate_info = _duplicate_market_info(done_trades)
    for row in rows:
        strategy_id = str(row.get("strategy_id") or "")
        row.update(duplicate_info.get(strategy_id) or {
            "duplicate_market_groups": 0,
            "duplicate_trade_count": 0,
        })
        row["source_type"] = "paper_trades"
        _annotate_mined_row(
            row,
            min_done_trades=min_done_trades,
            min_markets=min_markets,
            min_roi=min_roi,
            min_win_rate=min_win_rate,
            max_drawdown=max_drawdown,
            allow_duplicate_market_trades=allow_duplicate_market_trades,
        )
    rows.extend(copy.deepcopy(row) for row in backtest_metric_rows)
    for row in rows:
        if "eligible_for_validation" in row:
            continue
        _annotate_mined_row(
            row,
            min_done_trades=min_done_trades,
            min_markets=min_markets,
            min_roi=min_roi,
            min_win_rate=min_win_rate,
            max_drawdown=max_drawdown,
            allow_duplicate_market_trades=allow_duplicate_market_trades,
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("eligible_for_validation")),
            float(row.get("candidate_score") or -999999.0),
            float(row.get("pnl") or 0.0),
            float(row.get("average_roi") or -999.0),
            int(row.get("done_trades") or 0),
            str(row.get("strategy_id") or ""),
        ),
        reverse=True,
    )
    return rows


def _annotate_mined_row(
    row: dict[str, Any],
    *,
    min_done_trades: int,
    min_markets: int,
    min_roi: float,
    min_win_rate: float,
    max_drawdown: float | None,
    allow_duplicate_market_trades: bool,
) -> None:
    done = int(row.get("done_trades") or 0)
    markets = int(row.get("markets_covered") or 0)
    excluded: list[str] = []
    if done < int(min_done_trades):
        excluded.append("small_done_trade_sample")
    if markets < int(min_markets):
        excluded.append("small_market_sample")
    if float(row.get("pnl") or 0.0) <= 0:
        excluded.append("non_positive_pnl")
    if (_float_or_none(row.get("average_roi")) or 0.0) < float(min_roi):
        excluded.append("below_min_roi")
    if (_float_or_none(row.get("win_rate")) or 0.0) < float(min_win_rate):
        excluded.append("below_min_win_rate")
    worst = _float_or_none(row.get("worst_trade"))
    if max_drawdown is not None and worst is not None and worst < -abs(float(max_drawdown)):
        excluded.append("max_drawdown_exceeded")
    if not _payoff_is_acceptable(row):
        excluded.append("payoff_below_breakeven")
    duplicate_groups = int(row.get("duplicate_market_groups") or 0)
    if duplicate_groups and not allow_duplicate_market_trades:
        row["duplicate_market_penalty"] = duplicate_groups
        row.setdefault("warnings", [])
        row["warnings"] = list(row.get("warnings") or []) + ["duplicate_strategy_market_trades"]
    else:
        row["duplicate_market_penalty"] = 0
    row["eligible_for_validation"] = not excluded
    row["sample_status"] = "eligible" if not excluded else "small_sample"
    row["excluded_reasons"] = excluded
    row["candidate_score"] = _candidate_score(row)


def _candidate_score(row: dict[str, Any]) -> float:
    pnl = _float_or_none(row.get("pnl")) or 0.0
    avg_roi = _float_or_none(row.get("average_roi")) or 0.0
    win_rate = _float_or_none(row.get("win_rate")) or 0.0
    payoff = _float_or_none(row.get("payoff_ratio")) or 0.0
    done = int(row.get("done_trades") or 0)
    duplicate_penalty = float(row.get("duplicate_market_penalty") or 0.0) * 10.0
    return _round(pnl + avg_roi * 100.0 + win_rate * 10.0 + payoff + done * 0.01 - duplicate_penalty) or 0.0


def _duplicate_market_info(done_trades: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in done_trades:
        strategy_id = str(row.get("strategy_id") or "")
        market_id = str(row.get("market_id") or "")
        if not strategy_id or not market_id:
            continue
        grouped[(strategy_id, market_id)] += 1
    result: dict[str, dict[str, int]] = defaultdict(
        lambda: {"duplicate_market_groups": 0, "duplicate_trade_count": 0}
    )
    for (strategy_id, _market_id), count in grouped.items():
        if count <= 1:
            continue
        result[strategy_id]["duplicate_market_groups"] += 1
        result[strategy_id]["duplicate_trade_count"] += count - 1
    return dict(result)


def _direction_mode_from_side(side: Any) -> str:
    normalized = str(side or "").upper()
    if normalized == "YES":
        return "YES_ONLY"
    if normalized == "NO":
        return "NO_ONLY"
    return "BOTH"


def _strategy_candidate_rows(
    leaderboard: Sequence[dict[str, Any]],
    *,
    min_done_trades: int,
    min_markets: int,
    min_roi: float,
    min_win_rate: float,
    max_drawdown: float | None,
    top_n: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in leaderboard:
        if not row.get("eligible_for_validation"):
            continue
        selected.append(
            {
                "strategy_id": row.get("strategy_id"),
                "source_strategy_id": row.get("strategy_id"),
                "source_type": row.get("source_type"),
                "selection_reason": (
                    "passed minimum evidence: positive pnl, roi, win-rate, payoff, and sample rules"
                ),
                "minimums": {
                    "min_closed_trades": int(min_done_trades),
                    "min_markets": int(min_markets),
                    "min_roi": float(min_roi),
                    "min_win_rate": float(min_win_rate),
                    "max_drawdown": max_drawdown,
                },
                "score": row.get("candidate_score"),
                "metrics": row,
            }
        )
    selected.sort(
        key=lambda item: (
            float(item.get("score") or -999999.0),
            float(dict(item.get("metrics") or {}).get("pnl") or 0.0),
            str(item.get("strategy_id") or ""),
        ),
        reverse=True,
    )
    return selected[: max(0, int(top_n))]


def _write_mined_strategy_candidate_configs(
    selected_candidates: Sequence[dict[str, Any]],
    strategy_metadata: dict[str, dict[str, Any]],
    *,
    output_config_dir: str | None,
    generated_at: datetime,
) -> dict[str, Any]:
    if not output_config_dir:
        return {
            "generated_config_paths": [],
            "metadata_json_path": None,
            "summary_txt_path": None,
        }
    output_root = Path(output_config_dir)
    configs_dir = output_root / "configs"
    metadata_dir = output_root / "metadata"
    configs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    generated_config_paths: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        metrics = dict(candidate.get("metrics") or {})
        strategy_id = str(candidate.get("strategy_id") or f"candidate_{index}")
        config = _candidate_config_payload(metrics, strategy_metadata)
        config_path = configs_dir / f"{index:02d}_{_safe_filename(strategy_id)}.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        generated_config_paths.append(str(config_path))
        metadata_rows.append(
            {
                **candidate,
                "generated_config_path": str(config_path),
                "generated_at": _iso(generated_at),
            }
        )
    metadata_path = metadata_dir / "selected_strategy_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at": _iso(generated_at),
                "recommendation": "paper/shadow only; never live trading approval",
                "generated_config_glob": str(configs_dir / "*.json"),
                "selected_strategies": metadata_rows,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = output_root / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Mined Paper Strategy Candidate Configs",
                f"generated_at: {_iso(generated_at)}",
                f"generated_config_glob: {configs_dir / '*.json'}",
                "recommendation: paper/shadow only; do not go live",
                "",
                *[
                    f"{row.get('strategy_id')}: score={row.get('score')} config={row.get('generated_config_path')}"
                    for row in metadata_rows
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "generated_config_paths": generated_config_paths,
        "metadata_json_path": str(metadata_path),
        "summary_txt_path": str(summary_path),
    }


def _write_mined_metadata(report: dict[str, Any], output_config_dir: str) -> None:
    metadata_path = report.get("metadata_json_path")
    if metadata_path:
        return
    output_root = Path(output_config_dir)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "selected_strategy_metadata.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": report.get("generated_at"),
                "recommendation": report.get("recommendation"),
                "selected_strategies": report.get("selected_strategy_candidates") or [],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    report["metadata_json_path"] = str(path)


def _candidate_config_payload(
    metrics: dict[str, Any],
    strategy_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strategy_id = str(metrics.get("strategy_id") or "candidate_strategy")
    meta = strategy_metadata.get(strategy_id) or {}
    if meta.get("strategy_config"):
        strategy = copy.deepcopy(meta["strategy_config"])
    else:
        strategy = _infer_strategy_config(metrics)
    strategy["strategy_id"] = f"{strategy_id}_candidate"
    strategy["strategy_name"] = (
        f"{strategy.get('strategy_name') or strategy_id} mined validation candidate"
    )
    strategy["enabled"] = True
    strategy["one_trade_per_market"] = True
    strategy.setdefault("stake_usd", 1.0)
    strategy.setdefault("max_open_trades", 1)
    top_level = copy.deepcopy(meta.get("top_level_config") or {})
    if metrics.get("source_type") == "transformer_backtest":
        top_level["prediction_source"] = "transformer_predictions"
    top_level["live_trading"] = {"live_trading_enabled": False, "dry_run_orders": True}
    top_level["recommendation_note"] = "paper-only mined config; not live trading approval"
    top_level["strategies"] = [strategy]
    return _force_paper_only_config(top_level)


def _infer_strategy_config(metrics: dict[str, Any]) -> dict[str, Any]:
    direction_mode = str(metrics.get("direction_mode") or "").upper()
    side = str(metrics.get("signal_direction") or "").upper()
    if direction_mode not in {"YES_ONLY", "NO_ONLY", "BOTH"}:
        direction_mode = _direction_mode_from_side(side)
    threshold = _float_or_none(metrics.get("threshold"))
    if threshold is None:
        threshold = _float_or_none(metrics.get("analysis_threshold"))
    if threshold is None:
        threshold = 0.5
    strategy = {
        "strategy_id": str(metrics.get("strategy_id") or "candidate_strategy"),
        "strategy_name": str(metrics.get("strategy_id") or "Candidate strategy"),
        "enabled": True,
        "direction_mode": direction_mode,
        "long_threshold": round(float(threshold), 6) if direction_mode != "NO_ONLY" else 2.0,
        "short_threshold": (
            round(1.0 - float(threshold), 6)
            if direction_mode == "NO_ONLY"
            else (-1.0 if direction_mode == "YES_ONLY" else round(1.0 - float(threshold), 6))
        ),
        "min_estimated_edge": 0.0,
        "entry_slippage_cents": _float_or_none(metrics.get("slippage_cents")) or 0.0,
        "exit_slippage_cents": _float_or_none(metrics.get("slippage_cents")) or 0.0,
        "fee_cents": _float_or_none(metrics.get("fee_cents")) or 0.0,
        "stake_usd": 1.0,
        "max_open_trades": 1,
        "one_trade_per_market": True,
        "require_positive_edge": False,
        "min_time_until_resolution_sec": _float_or_none(metrics.get("min_time_until_resolution_sec")) or 0.0,
        "max_time_until_resolution_sec": _float_or_none(metrics.get("max_time_until_resolution_sec")) or 300.0,
    }
    horizon = _float_or_none(metrics.get("fixed_horizon_exit_sec")) or _float_or_none(metrics.get("exit_horizon_sec"))
    if horizon is not None:
        strategy["exit_type"] = "FIXED_HORIZON_EXIT"
        strategy["fixed_horizon_exit_sec"] = horizon
    return strategy


def _force_paper_only_config(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    _disable_live_trading_flags(result)
    result["live_trading"] = {
        **(result.get("live_trading") if isinstance(result.get("live_trading"), dict) else {}),
        "live_trading_enabled": False,
        "dry_run_orders": True,
    }
    return result


def _disable_live_trading_flags(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "live_trading_enabled":
                value[key] = False
            else:
                _disable_live_trading_flags(item)
    elif isinstance(value, list):
        for item in value:
            _disable_live_trading_flags(item)


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe[:120] or "strategy"


def _mined_promising_regimes(
    done_trades: Sequence[dict[str, Any]],
    *,
    min_done_trades: int,
    min_markets: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "time_regime",
        "btc_trend_regime",
        "spread_regime",
        "liquidity_regime",
        "pmax",
        "threshold_group",
        "bankroll",
    ):
        rows = _grouped_metrics(done_trades, (field,), include_small=True)
        promising = [
            row
            for row in rows
            if int(row.get("done_trades") or 0) >= int(min_done_trades)
            and int(row.get("markets_covered") or 0) >= int(min_markets)
            and float(row.get("pnl") or 0.0) > 0
            and (_float_or_none(row.get("average_roi")) or 0.0) > 0
            and _payoff_is_acceptable(row)
        ]
        result[field] = sorted(
            promising,
            key=lambda row: (
                float(row.get("pnl") or 0.0),
                float(row.get("average_roi") or 0.0),
                int(row.get("done_trades") or 0),
            ),
            reverse=True,
        )[:TOP_N]
    return result


def _bad_loss_tail_strategies(
    leaderboard: Sequence[dict[str, Any]],
    *,
    min_done_trades: int,
    min_markets: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in leaderboard:
        if int(row.get("done_trades") or 0) < int(min_done_trades):
            continue
        if int(row.get("markets_covered") or 0) < int(min_markets):
            continue
        win_rate = _float_or_none(row.get("win_rate")) or 0.0
        avg_win = _float_or_none(row.get("average_win"))
        avg_loss = _float_or_none(row.get("average_loss"))
        worst = _float_or_none(row.get("worst_trade"))
        payoff = _float_or_none(row.get("payoff_ratio"))
        reasons: list[str] = []
        if win_rate >= 0.55 and (payoff is None or payoff < 1.0):
            reasons.append("high_win_rate_bad_payoff")
        if avg_win is not None and avg_loss is not None and abs(avg_loss) > avg_win:
            reasons.append("average_loss_larger_than_average_win")
        if avg_win is not None and worst is not None and worst < -3.0 * abs(avg_win):
            reasons.append("rare_large_loss_tail")
        if reasons:
            result.append({**row, "loss_tail_reason": ",".join(reasons)})
    result.sort(
        key=lambda row: (
            float(row.get("win_rate") or 0.0),
            float(abs(_float_or_none(row.get("worst_trade")) or 0.0)),
        ),
        reverse=True,
    )
    return result[:TOP_N]


def _mined_next_sweep_suggestions(
    leaderboard: Sequence[dict[str, Any]],
    strategy_metadata: dict[str, dict[str, Any]],
    *,
    min_done_trades: int,
    min_markets: int,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for row in leaderboard:
        if not row.get("eligible_for_validation"):
            continue
        if float(row.get("pnl") or 0.0) <= 0:
            continue
        if (_float_or_none(row.get("average_roi")) or 0.0) <= 0:
            continue
        if not _payoff_is_acceptable(row):
            continue
        win_rate = _float_or_none(row.get("win_rate"))
        breakeven = _float_or_none(row.get("breakeven_win_rate"))
        if win_rate is not None and breakeven is not None and win_rate < breakeven:
            continue
        strategy_id = str(row.get("strategy_id") or "")
        meta = strategy_metadata.get(strategy_id) or {}
        if not meta.get("strategy_config"):
            continue
        strategy = copy.deepcopy(meta["strategy_config"])
        strategy["strategy_id"] = f"{strategy_id}_validate_next"
        strategy["strategy_name"] = f"{strategy.get('strategy_name') or strategy_id} validation candidate"
        top_level = copy.deepcopy(meta.get("top_level_config") or {})
        top_level["live_trading"] = {
            "live_trading_enabled": False,
            "dry_run_orders": True,
        }
        top_level["recommendation_note"] = (
            "paper-only generated suggestion; not live trading approval"
        )
        top_level["strategies"] = [strategy]
        suggestions.append(
            {
                "strategy_id": strategy["strategy_id"],
                "source_strategy_id": strategy_id,
                "reason": "eligible positive expectancy paper-only validation candidate",
                "min_done_trades": int(min_done_trades),
                "min_markets": int(min_markets),
                "source_metrics": row,
                "config": top_level,
            }
        )
    return suggestions[:TOP_N]


def _payoff_is_acceptable(row: dict[str, Any]) -> bool:
    payoff = _float_or_none(row.get("payoff_ratio"))
    if payoff is not None and payoff >= 1.0:
        return True
    win_rate = _float_or_none(row.get("win_rate"))
    breakeven = _float_or_none(row.get("breakeven_win_rate"))
    return bool(win_rate is not None and breakeven is not None and win_rate >= breakeven)


def _market_count_for_rows(rows: Sequence[dict[str, Any]]) -> int:
    return len(
        {
            str(row.get("market_id"))
            for row in rows
            if row.get("market_id") not in (None, "")
        }
    )


def _grouped_metrics(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    include_small: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_group_value(row, field) for field in fields)
        grouped[key].append(row)
    results: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        item = {field: value for field, value in zip(fields, key)}
        item.update(_performance_metrics(group_rows))
        item["small_sample"] = int(item.get("done_trades") or 0) < 30
        item["sample_warnings"] = _sample_warnings(int(item.get("done_trades") or 0))
        results.append(item)
    results.sort(
        key=lambda item: (
            float(item.get("pnl") or 0.0),
            float(item.get("average_roi") or -999.0),
            int(item.get("done_trades") or 0),
            json.dumps({field: item.get(field) for field in fields}, sort_keys=True),
        ),
        reverse=True,
    )
    return results


def _performance_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_float_or_none(row.get("pnl")) for row in rows]
    pnls = [value for value in pnls if value is not None]
    rois = [_float_or_none(row.get("roi")) for row in rows]
    rois = [value for value in rois if value is not None]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    payoff_ratio = (
        _round(float(avg_win) / abs(float(avg_loss)))
        if avg_win is not None and avg_loss not in (None, 0)
        else None
    )
    return {
        "done_trades": len(rows),
        "pnl": _round(sum(pnls)),
        "roi": _round(sum(pnls) / _sum(row.get("stake_usd") for row in rows)) if rows else None,
        "average_roi": _mean(rois),
        "win_rate": _round(len(wins) / len(pnls)) if pnls else None,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "breakeven_win_rate": (
            _round(1.0 / (1.0 + float(payoff_ratio))) if payoff_ratio not in (None, 0) else None
        ),
        "worst_trade": _round(min(pnls)) if pnls else None,
        "best_trade": _round(max(pnls)) if pnls else None,
        "threshold": _mean(row.get("analysis_threshold") for row in rows),
        "markets_covered": len({str(row.get("market_id")) for row in rows if row.get("market_id") not in (None, "")}),
    }


def _loss_groups(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for item in _grouped_metrics(rows, fields, include_small=True):
        result.append(
            {
                **{field: item.get(field) for field in fields},
                "losing_trades": item.get("done_trades"),
                "average_loss": _average_loss_for_group(rows, fields, item),
                "total_loss": item.get("pnl"),
                "worst_trade": item.get("worst_trade"),
            }
        )
    return result


def _average_loss_for_group(rows: Sequence[dict[str, Any]], fields: Sequence[str], item: dict[str, Any]) -> float | None:
    group_rows = [
        row for row in rows if all(_group_value(row, field) == item.get(field) for field in fields)
    ]
    return _mean(row.get("pnl") for row in group_rows)


def _count_group(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        key = tuple(_group_value(row, field) for field in fields)
        if any(value == "unknown" for value in key):
            continue
        grouped[key] += 1
    result = []
    for key, count in grouped.items():
        item = {field: value for field, value in zip(fields, key)}
        item["count"] = count
        result.append(item)
    result.sort(key=lambda item: (int(item["count"]), json.dumps(item, sort_keys=True)), reverse=True)
    return result[:TOP_N]


def _conversion_groups(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_group_value(row, field) for field in fields)].append(row)
    result = []
    for key, group_rows in grouped.items():
        decision_counts = _counts(row.get("decision") for row in group_rows)
        trade_count = int(decision_counts.get("TRADE", 0))
        item = {field: value for field, value in zip(fields, key)}
        item.update(
            {
                "candidate_count": len(group_rows),
                "trade_count": trade_count,
                "skip_count": int(decision_counts.get("SKIP", 0)),
                "conversion_rate": _round(trade_count / len(group_rows)) if group_rows else None,
                "decision_counts": decision_counts,
            }
        )
        result.append(item)
    result.sort(key=lambda item: (float(item.get("conversion_rate") or 0.0), int(item.get("candidate_count") or 0)), reverse=True)
    return result[:TOP_N]


def _trade_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_trade_id": row.get("id"),
        "source_paper_db": row.get("source_paper_db"),
        "strategy_id": row.get("strategy_id"),
        "market_id": row.get("market_id"),
        "status": row.get("status"),
        "pnl": _round(row.get("pnl")),
        "roi": _round(row.get("roi")),
        "stake_usd": _round(row.get("stake_usd")),
        "time_until_resolution": _round(row.get("time_until_resolution")),
        "threshold_used": _round(row.get("threshold_used")),
        "pmax": row.get("pmax"),
        "time_regime": row.get("time_regime"),
        "btc_trend_regime": row.get("btc_trend_regime"),
        "liquidity_regime": row.get("liquidity_regime"),
        "spread_regime": row.get("spread_regime"),
        "entry_price_drift": _round(row.get("entry_price_drift")),
        "spread_cents_at_entry": _round(row.get("spread_cents_at_entry")),
        "created_at": row.get("created_at"),
        "exit_reason": row.get("exit_reason"),
    }


def _group_value(row: dict[str, Any], field: str) -> str:
    if field == "entry_price_drift_bucket":
        return _bucket(row.get("entry_price_drift"), [0, 1, 2, 5], suffix="c")
    if field == "spread_cents_bucket":
        return _bucket(row.get("spread_cents_at_entry"), [0, 2, 4, 8], suffix="c")
    value = row.get(field)
    if value in (None, ""):
        return "unknown"
    if field in {"threshold", "threshold_used"}:
        return _format_number(value)
    return str(value)


def _bucket(value: Any, cuts: Sequence[float], *, suffix: str = "") -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "unknown"
    previous = None
    for cut in cuts:
        if parsed <= cut:
            if previous is None:
                return f"<= {cut:g}{suffix}"
            return f"{previous:g}-{cut:g}{suffix}"
        previous = cut
    return f"> {cuts[-1]:g}{suffix}"


def _sample_warnings(done_trades: int) -> list[str]:
    warnings = []
    for threshold in DEFAULT_MIN_DONE_THRESHOLDS:
        if done_trades < threshold:
            warnings.append(f"below_{threshold}_done_trades")
    return warnings


def _strategy_threshold(strategy: dict[str, Any], direction_mode: str) -> float | None:
    if direction_mode == "NO_ONLY":
        short = _float_or_none(strategy.get("short_threshold"))
        return 1.0 - short if short is not None and short >= 0 else None
    if direction_mode == "YES_ONLY":
        return _float_or_none(strategy.get("long_threshold"))
    long_value = _float_or_none(strategy.get("long_threshold"))
    short = _float_or_none(strategy.get("short_threshold"))
    if long_value is not None and 0 <= long_value <= 1:
        return long_value
    if short is not None and 0 <= short <= 1:
        return 1.0 - short
    return None


def _pmax(strategy: dict[str, Any]) -> float | None:
    for key in ("max_probability_for_direction", "block_probability_above"):
        value = _float_or_none(strategy.get(key))
        if value is not None:
            return value
    return None


def _format_pmax(value: Any) -> str:
    parsed = _float_or_none(value)
    return "none" if parsed is None else _format_number(parsed)


def _format_time_window(
    min_value: Any,
    max_value: Any,
    fallback_regime: Any = None,
    time_until: Any = None,
) -> str:
    parsed_min = _float_or_none(min_value)
    parsed_max = _float_or_none(max_value)
    if parsed_min is not None or parsed_max is not None:
        return f"{_format_number(parsed_min)}-{_format_number(parsed_max)}s"
    if fallback_regime not in (None, ""):
        return str(fallback_regime)
    parsed_time = _float_or_none(time_until)
    if parsed_time is None:
        return "unknown"
    if parsed_time < 30:
        return "0-30s"
    if parsed_time < 60:
        return "30-60s"
    if parsed_time < 120:
        return "60-120s"
    return "120s+"


def _format_number(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "unknown"
    return str(int(parsed)) if float(parsed).is_integer() else str(_round(parsed))


def _bankroll_label(bankroll: dict[str, Any]) -> str:
    if not bankroll or not bool(bankroll.get("bankroll_enabled")):
        return "fixed"
    start = _float_or_none(bankroll.get("starting_bankroll_usd"))
    return f"bankroll_{_format_number(start)}" if start is not None else "bankroll_enabled"


def _format_bankroll_from_row(row: dict[str, Any]) -> str | None:
    start = _float_or_none(row.get("starting_bankroll_usd"))
    if start is not None:
        return f"bankroll_{_format_number(start)}"
    status = row.get("bankroll_status")
    return str(status) if status not in (None, "") else None


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "").lower()


def _result_pnl(row: dict[str, Any]) -> float | None:
    for field in ("realized_pnl_usd", "pnl_usd", "simulated_pnl_usd"):
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _result_roi(row: dict[str, Any]) -> float | None:
    for field in ("realized_roi", "roi", "simulated_roi"):
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _read_table(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()]


def _connect_read_only(path: Path) -> sqlite3.Connection:
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone() is not None


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return dict(sorted(counter.items()))


def _sum(values: Iterable[Any]) -> float:
    return sum(value for value in (_float_or_none(item) for item in values) if value is not None)


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (_float_or_none(item) for item in values) if value is not None]
    return _round(sum(numeric) / len(numeric)) if numeric else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _round(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, 10) if parsed is not None else None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "PaperStrategyExperimentAnalysisError",
    "analyze_paper_strategy_experiment",
    "mine_paper_strategies",
    "recommend_next_paper_sweep",
    "render_mined_paper_strategies",
    "render_paper_strategy_experiment_analysis",
]
