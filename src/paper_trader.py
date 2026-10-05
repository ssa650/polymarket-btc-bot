from __future__ import annotations

import csv
import json
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .backtest_dataset import BacktestDataset, BacktestRecord, load_backtest_dataset


BUY = "BUY"
SELL = "SELL"
YES = "YES"
NO = "NO"
FILL_MODE_TOP_OF_BOOK = "top_of_book"
FILL_MODE_DEPTH_AWARE = "depth_aware"
PAPER_FILL_MODES = {FILL_MODE_TOP_OF_BOOK, FILL_MODE_DEPTH_AWARE}
PAPER_REPORT_SORT_FIELDS = {"realized_pnl", "ending_cash", "total_fills", "run_id"}
PAPER_REPORT_FIELDS: tuple[str, ...] = (
    "run_id",
    "strategy_name",
    "starting_cash",
    "ending_cash",
    "realized_pnl",
    "total_orders",
    "total_fills",
    "rejected_orders",
    "no_fill_orders",
    "unsettled_positions",
    "markets_traded",
)
PAPER_REPORT_MARKET_FIELDS: tuple[str, ...] = (
    "market_id",
    "total_orders",
    "total_fills",
    "buy_fills",
    "sell_fills",
    "gross_notional",
    "realized_pnl",
    "open_position_shares_before_settlement",
    "settled_shares",
    "unsettled_positions_after_settlement",
    "settlement_pnl",
    "partial_fill_count",
    "avg_levels_consumed",
    "total_unfilled_size",
    "avg_fill_price",
)
PAPER_REPORT_DEPTH_FIELDS: tuple[str, ...] = (
    "fill_mode",
    "requested_size",
    "filled_size",
    "unfilled_size",
    "average_fill_price",
    "levels_consumed",
    "liquidity_available",
    "partial_fill",
    "no_fill_reason",
)


class PaperTraderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PaperTraderConfig:
    run_id: str = "paper_run"
    starting_cash: float = 1000.0
    fee_bps: float = 0.0
    fixed_slippage: float = 0.0
    slippage_bps: float = 0.0
    default_tick_size: float = 0.001
    fill_mode: str = FILL_MODE_TOP_OF_BOOK
    max_depth_levels: int | None = None
    allow_partial_fills: bool = True
    max_position_size_per_market: float | None = None
    max_notional_per_order: float | None = None
    max_total_open_notional: float | None = None


@dataclass(frozen=True, slots=True)
class PaperDecision:
    decision_id: str
    side: str
    outcome: str
    quantity: float
    market_id: str | None = None
    timestamp: datetime | None = None
    limit_price: float | None = None
    min_seconds_until_close: float | None = None
    max_seconds_until_close: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PaperRule:
    rule_id: str
    side: str
    outcome: str
    quantity: float
    market_id: str | None = None
    limit_price: float | None = None
    min_seconds_until_close: float | None = None
    max_seconds_until_close: float | None = None
    buy_yes_ask_lte: float | None = None
    buy_no_ask_lte: float | None = None
    emit_once_per_market: bool = True
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PaperOrder:
    order_id: str
    decision_id: str
    market_id: str
    timestamp: datetime
    side: str
    outcome: str
    quantity: float
    limit_price: float | None
    normalized_limit_price: float | None
    status: str
    reason: str
    fill_mode: str = FILL_MODE_TOP_OF_BOOK
    requested_size: float = 0.0
    filled_size: float = 0.0
    unfilled_size: float = 0.0
    average_fill_price: float | None = None
    levels_consumed: int = 0
    liquidity_available: float | None = None
    partial_fill: bool = False
    no_fill_reason: str = ""


@dataclass(frozen=True, slots=True)
class PaperFill:
    fill_id: str
    order_id: str
    market_id: str
    timestamp: datetime
    side: str
    outcome: str
    quantity: float
    price: float
    fee: float
    cash_delta: float
    fill_mode: str = FILL_MODE_TOP_OF_BOOK
    requested_size: float = 0.0
    filled_size: float = 0.0
    unfilled_size: float = 0.0
    average_fill_price: float | None = None
    levels_consumed: int = 0
    liquidity_available: float | None = None
    partial_fill: bool = False
    no_fill_reason: str = ""


@dataclass(frozen=True, slots=True)
class PaperPosition:
    market_id: str
    outcome: str
    quantity: float
    average_price: float


@dataclass(frozen=True, slots=True)
class PaperSettlement:
    settlement_id: str
    market_id: str
    timestamp: datetime | None
    outcome: str
    quantity: float
    average_price: float
    winning_outcome: str
    settlement_price: float
    cash_delta: float
    settlement_pnl: float


@dataclass(frozen=True, slots=True)
class PaperTraderRunResult:
    run_id: str
    strategy_name: str
    dataset_path: str | None
    config_used: Mapping[str, Any]
    starting_cash: float
    ending_cash: float
    realized_pnl: float
    unsettled_positions: int
    total_orders: int
    total_fills: int
    rejected_orders: int
    no_fill_orders: int
    markets_traded: int
    orders: tuple[PaperOrder, ...]
    fills: tuple[PaperFill, ...]
    positions: tuple[PaperPosition, ...]
    settlements: tuple[PaperSettlement, ...] = ()


@dataclass(frozen=True, slots=True)
class _FillExecution:
    fill_mode: str
    requested_size: float
    filled_size: float
    unfilled_size: float
    average_fill_price: float | None
    notional: float
    levels_consumed: int
    liquidity_available: float | None
    partial_fill: bool
    no_fill_reason: str


class PaperStrategy(Protocol):
    def generate_decisions(
        self,
        record: BacktestRecord,
        positions: Mapping[tuple[str, str], PaperPosition],
        cash: float,
    ) -> Sequence[PaperDecision]:
        ...


class NoOpStrategy:
    def generate_decisions(
        self,
        record: BacktestRecord,
        positions: Mapping[tuple[str, str], PaperPosition],
        cash: float,
    ) -> Sequence[PaperDecision]:
        _ = (record, positions, cash)
        return ()


class RuleStrategy:
    """Deterministic fixture/rule strategy for offline tests and simple smoke runs."""

    def __init__(
        self,
        decisions: Sequence[PaperDecision] = (),
        *,
        rules: Sequence[PaperRule] = (),
        emit_once: bool = True,
    ) -> None:
        self._decisions = tuple(decisions)
        self._rules = tuple(rules)
        self._emit_once = bool(emit_once)
        self._emitted: set[str] = set()
        self._emitted_rule_markets: set[tuple[str, str]] = set()

    def generate_decisions(
        self,
        record: BacktestRecord,
        positions: Mapping[tuple[str, str], PaperPosition],
        cash: float,
    ) -> Sequence[PaperDecision]:
        _ = (positions, cash)
        matched: list[PaperDecision] = []
        for decision in self._decisions:
            if self._emit_once and decision.decision_id in self._emitted:
                continue
            if not _decision_matches_record(decision, record):
                continue
            matched.append(decision)
            self._emitted.add(decision.decision_id)
        for rule in self._rules:
            if not _rule_matches_record(rule, record):
                continue
            rule_key = (rule.rule_id, record.market_id)
            if rule.emit_once_per_market and rule_key in self._emitted_rule_markets:
                continue
            matched.append(
                PaperDecision(
                    decision_id=(
                        f"{rule.rule_id}:{record.market_id}"
                        if rule.emit_once_per_market
                        else f"{rule.rule_id}:{record.market_id}:{record.timestamp.isoformat()}"
                    ),
                    market_id=record.market_id,
                    timestamp=record.timestamp,
                    side=rule.side,
                    outcome=rule.outcome,
                    quantity=rule.quantity,
                    limit_price=rule.limit_price,
                    min_seconds_until_close=rule.min_seconds_until_close,
                    max_seconds_until_close=rule.max_seconds_until_close,
                    reason=rule.reason or rule.rule_id,
                )
            )
            self._emitted_rule_markets.add(rule_key)
        return tuple(matched)


def run_paper_trader(
    dataset: BacktestDataset,
    strategy: PaperStrategy,
    config: PaperTraderConfig | None = None,
    *,
    strategy_name: str | None = None,
    dataset_path: str | None = None,
) -> PaperTraderRunResult:
    cfg = config or PaperTraderConfig()
    _validate_config(cfg)
    cash = float(cfg.starting_cash)
    positions: dict[tuple[str, str], PaperPosition] = {}
    orders: list[PaperOrder] = []
    fills: list[PaperFill] = []
    order_seq = 0

    for timeline in dataset.timelines.values():
        for record in timeline.records:
            decisions = strategy.generate_decisions(record, positions, cash)
            for decision in decisions:
                order_seq += 1
                order, fill, cash = _process_decision(
                    decision,
                    record,
                    config=cfg,
                    cash=cash,
                    positions=positions,
                    order_seq=order_seq,
                )
                orders.append(order)
                if fill is not None:
                    fills.append(fill)

    cash, settlements = _settle_positions(dataset, positions=positions, cash=cash)
    final_positions = tuple(
        position
        for position in sorted(
            positions.values(),
            key=lambda item: (item.market_id, item.outcome),
        )
        if position.quantity > 0
    )
    ending_cash = _round_money(cash)
    return PaperTraderRunResult(
        run_id=str(cfg.run_id),
        strategy_name=str(strategy_name or _strategy_name(strategy)),
        dataset_path=dataset_path,
        config_used=_config_to_dict(cfg),
        starting_cash=float(cfg.starting_cash),
        ending_cash=ending_cash,
        realized_pnl=_round_money(ending_cash - float(cfg.starting_cash)),
        unsettled_positions=len(final_positions),
        total_orders=len(orders),
        total_fills=len(fills),
        rejected_orders=sum(1 for order in orders if order.status == "rejected"),
        no_fill_orders=sum(1 for order in orders if order.status == "no_fill"),
        markets_traded=len({fill.market_id for fill in fills}),
        orders=tuple(orders),
        fills=tuple(fills),
        positions=final_positions,
        settlements=settlements,
    )


def run_paper_backtest_from_export(
    export_path: str,
    *,
    strategy_name: str = "noop",
    strategy_config: Mapping[str, Any] | None = None,
    strategy_config_path: str | None = None,
    require_labels: bool = False,
    require_btc_price: bool = False,
    time_window_before_close_sec: float | None = None,
    config: PaperTraderConfig | None = None,
) -> PaperTraderRunResult:
    dataset = load_backtest_dataset(
        export_path,
        require_label_available=require_labels,
        require_btc_price=require_btc_price,
        time_window_before_close_sec=time_window_before_close_sec,
    )
    loaded_strategy_config = (
        load_strategy_config(strategy_config_path)
        if strategy_config_path is not None
        else dict(strategy_config or {})
    )
    strategy = build_strategy(strategy_name, strategy_config=loaded_strategy_config)
    return run_paper_trader(
        dataset,
        strategy,
        config=config,
        strategy_name=strategy_name,
        dataset_path=str(export_path),
    )


def build_strategy(
    name: str,
    *,
    strategy_config: Mapping[str, Any] | None = None,
) -> PaperStrategy:
    normalized = name.strip().lower()
    if normalized == "noop":
        return NoOpStrategy()
    if normalized == "rule":
        return build_rule_strategy(strategy_config or {})
    raise PaperTraderError(f"Unsupported paper strategy: {name}")


def load_strategy_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise PaperTraderError("Paper strategy config must be a JSON object")
    return loaded


def build_rule_strategy(config: Mapping[str, Any]) -> RuleStrategy:
    defaults = dict(config)
    raw_rules = defaults.pop("rules", None)
    if raw_rules is None:
        raw_rules = [defaults] if defaults else []
    if not isinstance(raw_rules, list):
        raise PaperTraderError("Rule strategy config field 'rules' must be a list")

    emit_once = bool(config.get("emit_once", True))
    rules: list[PaperRule] = []
    for idx, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise PaperTraderError("Each rule strategy rule must be a JSON object")
        merged = {**defaults, **raw_rule}
        quantity = merged.get("quantity", merged.get("order_size", merged.get("size")))
        if quantity is None:
            raise PaperTraderError("Rule strategy requires order_size or quantity")
        rules.append(
            PaperRule(
                rule_id=str(merged.get("id") or merged.get("rule_id") or f"rule_{idx + 1}"),
                side=str(merged.get("side", BUY)),
                outcome=str(merged.get("outcome", YES)),
                quantity=float(quantity),
                market_id=_str_or_none(merged.get("market_id")),
                limit_price=_float_or_none(merged.get("limit_price")),
                min_seconds_until_close=_float_or_none(
                    merged.get("min_seconds_until_close")
                ),
                max_seconds_until_close=_float_or_none(
                    merged.get("max_seconds_until_close")
                ),
                buy_yes_ask_lte=_float_or_none(merged.get("buy_yes_ask_lte")),
                buy_no_ask_lte=_float_or_none(merged.get("buy_no_ask_lte")),
                emit_once_per_market=bool(merged.get("emit_once_per_market", True)),
                reason=str(merged.get("reason") or merged.get("id") or ""),
            )
        )
    return RuleStrategy(rules=rules, emit_once=emit_once)


def result_to_dict(result: PaperTraderRunResult) -> dict[str, Any]:
    payload = asdict(result)
    for key in ("orders", "fills", "positions", "settlements"):
        payload[key] = [_jsonable(row) for row in payload[key]]
    return _jsonable(payload)


def result_summary_to_dict(result: PaperTraderRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "strategy_name": result.strategy_name,
        "dataset_path": result.dataset_path,
        "starting_cash": result.starting_cash,
        "ending_cash": result.ending_cash,
        "realized_pnl": result.realized_pnl,
        "total_orders": result.total_orders,
        "total_fills": result.total_fills,
        "rejected_orders": result.rejected_orders,
        "no_fill_orders": result.no_fill_orders,
        "unsettled_positions": result.unsettled_positions,
        "markets_traded": result.markets_traded,
        "config_used": _jsonable(dict(result.config_used)),
    }


def export_paper_backtest_result(
    result: PaperTraderRunResult,
    output_dir: str | Path,
    *,
    write_csv: bool = True,
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename(result.run_id or "paper_run")
    summary_path = target / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(result_summary_to_dict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    paths = {"summary_json": str(summary_path)}
    if write_csv:
        orders_path = target / f"{prefix}_orders.csv"
        fills_path = target / f"{prefix}_fills.csv"
        settlements_path = target / f"{prefix}_settlements.csv"
        _write_csv(
            orders_path,
            [_jsonable(asdict(order)) for order in result.orders],
            fieldnames=(
                "order_id",
                "decision_id",
                "market_id",
                "timestamp",
                "side",
                "outcome",
                "quantity",
                "limit_price",
                "normalized_limit_price",
                "status",
                "reason",
                "fill_mode",
                "requested_size",
                "filled_size",
                "unfilled_size",
                "average_fill_price",
                "levels_consumed",
                "liquidity_available",
                "partial_fill",
                "no_fill_reason",
            ),
        )
        _write_csv(
            fills_path,
            [_jsonable(asdict(fill)) for fill in result.fills],
            fieldnames=(
                "fill_id",
                "order_id",
                "market_id",
                "timestamp",
                "side",
                "outcome",
                "quantity",
                "price",
                "fee",
                "cash_delta",
                "fill_mode",
                "requested_size",
                "filled_size",
                "unfilled_size",
                "average_fill_price",
                "levels_consumed",
                "liquidity_available",
                "partial_fill",
                "no_fill_reason",
            ),
        )
        _write_csv(
            settlements_path,
            [_jsonable(asdict(settlement)) for settlement in result.settlements],
            fieldnames=(
                "settlement_id",
                "market_id",
                "timestamp",
                "outcome",
                "quantity",
                "average_price",
                "winning_outcome",
                "settlement_price",
                "cash_delta",
                "settlement_pnl",
            ),
        )
        paths["orders_csv"] = str(orders_path)
        paths["fills_csv"] = str(fills_path)
        paths["settlements_csv"] = str(settlements_path)
    return paths


def load_paper_run_summaries(output_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(output_dir)
    if not base.exists() or not base.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(base.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, dict):
            raise PaperTraderError(f"Paper summary is not a JSON object: {path}")
        row = dict(loaded)
        row["summary_path"] = str(path)
        summaries.append(row)
    return summaries


def build_paper_runs_report(
    output_dir: str | Path,
    *,
    sort_by: str = "realized_pnl",
    top: int | None = None,
    run_id: str | None = None,
    include_detail: bool = False,
    warnings_only: bool = False,
) -> dict[str, Any]:
    if sort_by not in PAPER_REPORT_SORT_FIELDS:
        raise PaperTraderError(
            f"Unsupported paper report sort field: {sort_by}. "
            f"Expected one of: {', '.join(sorted(PAPER_REPORT_SORT_FIELDS))}"
        )
    summaries = load_paper_run_summaries(output_dir)
    if run_id is not None:
        summaries = [row for row in summaries if str(row.get("run_id") or "") == run_id]
    sorted_summaries = sorted(
        summaries,
        key=lambda row: _paper_report_sort_key(row, sort_by),
        reverse=(sort_by != "run_id"),
    )
    if top is not None and top >= 0:
        sorted_summaries = sorted_summaries[:top]
    details: dict[str, Any] = {}
    detail_warnings: list[str] = []
    if include_detail:
        for row in sorted_summaries:
            run_id_value = str(row.get("run_id") or "unknown_run")
            detail = _paper_run_detail(row)
            details[run_id_value] = detail
            detail_warnings.extend(detail.get("warnings", []))

    warnings = [*_paper_report_warnings(summaries), *detail_warnings]
    runs = [_paper_report_row(row) for row in sorted_summaries]
    if warnings_only:
        warning_run_ids = {
            warning.split(":", 1)[0]
            for warning in warnings
            if ":" in warning
        }
        runs = [row for row in runs if str(row.get("run_id") or "") in warning_run_ids]
        if include_detail:
            details = {
                key: value for key, value in details.items() if key in warning_run_ids
            }
    return {
        "paper_output_dir": str(output_dir),
        "summary_files": len(summaries),
        "sort_by": sort_by,
        "top": top,
        "run_id_filter": run_id,
        "include_detail": include_detail,
        "warnings_only": warnings_only,
        "runs": runs,
        "warnings": warnings,
        **({"details": details} if include_detail else {}),
    }


def render_paper_runs_report(report: Mapping[str, Any]) -> str:
    runs = list(report.get("runs") or [])
    lines = [
        "=== Paper Run Report ===",
        f"paper_output_dir={report.get('paper_output_dir')}",
        f"summary_files={report.get('summary_files', 0)}",
        f"sort_by={report.get('sort_by')}",
        f"top={report.get('top') if report.get('top') is not None else 'all'}",
        "",
    ]
    if not runs:
        lines.append("no paper run summaries")
    else:
        widths = {
            field: max(
                len(field),
                max(len(_paper_report_cell(row.get(field))) for row in runs),
            )
            for field in PAPER_REPORT_FIELDS
        }
        header = " | ".join(field.ljust(widths[field]) for field in PAPER_REPORT_FIELDS)
        divider = "-+-".join("-" * widths[field] for field in PAPER_REPORT_FIELDS)
        lines.extend([header, divider])
        for row in runs:
            lines.append(
                " | ".join(
                    _paper_report_cell(row.get(field)).ljust(widths[field])
                    for field in PAPER_REPORT_FIELDS
                )
            )

    details = report.get("details")
    if isinstance(details, dict) and details:
        lines.extend(["", "=== Detail ==="])
        for run_id, detail in details.items():
            lines.append(f"run_id={run_id}")
            lines.append(
                "settlement_detail_available="
                f"{bool(detail.get('settlement_detail_available'))}"
            )
            per_market = list(detail.get("per_market") or [])
            if per_market:
                widths = {
                    field: max(
                        len(field),
                        max(
                            len(_paper_report_cell(row.get(field)))
                            for row in per_market
                        ),
                    )
                    for field in PAPER_REPORT_MARKET_FIELDS
                }
                lines.append("per_market:")
                lines.append(
                    " | ".join(
                        field.ljust(widths[field])
                        for field in PAPER_REPORT_MARKET_FIELDS
                    )
                )
                lines.append(
                    "-+-".join("-" * widths[field] for field in PAPER_REPORT_MARKET_FIELDS)
                )
                for row in per_market:
                    lines.append(
                        " | ".join(
                            _paper_report_cell(row.get(field)).ljust(widths[field])
                            for field in PAPER_REPORT_MARKET_FIELDS
                        )
                    )
            else:
                lines.append("per_market: none")
            cohorts = detail.get("cohorts") or {}
            lines.append("cohorts:")
            for key in (
                "yes_fills",
                "no_fills",
                "buy_fills",
                "sell_fills",
                "no_fill_orders",
            ):
                lines.append(f"  {key}={cohorts.get(key, 0)}")
            rejected = cohorts.get("rejected_order_reasons") or {}
            if rejected:
                for reason, count in sorted(rejected.items()):
                    lines.append(f"  rejected_order_reason.{reason}={count}")
            else:
                lines.append("  rejected_order_reasons=none")
            no_fill_reasons = cohorts.get("no_fill_reasons") or {}
            if no_fill_reasons:
                for reason, count in sorted(no_fill_reasons.items()):
                    lines.append(f"  no_fill_reason.{reason}={count}")
            else:
                lines.append("  no_fill_reasons=none")

            depth = detail.get("depth_summary") or {}
            lines.append("depth:")
            for key in (
                "depth_detail_available",
                "partial_fill_count",
                "total_requested_size",
                "total_filled_size",
                "total_unfilled_size",
                "avg_levels_consumed",
                "avg_liquidity_available",
                "avg_fill_price",
            ):
                lines.append(f"  {key}={_paper_report_cell(depth.get(key))}")
            fill_modes = depth.get("fill_mode_counts") or {}
            if fill_modes:
                for mode, count in sorted(fill_modes.items()):
                    lines.append(f"  fill_mode.{mode}={count}")
            else:
                lines.append("  fill_modes=none")
            depth_no_fill_reasons = depth.get("no_fill_reason_counts") or {}
            if depth_no_fill_reasons:
                for reason, count in sorted(depth_no_fill_reasons.items()):
                    lines.append(f"  depth_no_fill_reason.{reason}={count}")
            else:
                lines.append("  depth_no_fill_reasons=none")

    warnings = list(report.get("warnings") or [])
    lines.extend(["", "=== Warnings ==="])
    if warnings:
        lines.extend(f"warning={warning}" for warning in warnings)
    else:
        lines.append("none")
    return "\n".join(lines)


def _process_decision(
    decision: PaperDecision,
    record: BacktestRecord,
    *,
    config: PaperTraderConfig,
    cash: float,
    positions: dict[tuple[str, str], PaperPosition],
    order_seq: int,
) -> tuple[PaperOrder, PaperFill | None, float]:
    side = _normalize_side(decision.side)
    outcome = _normalize_outcome(decision.outcome)
    order_id = f"paper_order_{order_seq}"
    requested_size = float(decision.quantity)
    base_order = {
        "order_id": order_id,
        "decision_id": decision.decision_id,
        "market_id": record.market_id,
        "timestamp": record.timestamp,
        "side": side,
        "outcome": outcome,
        "quantity": requested_size,
        "limit_price": decision.limit_price,
        "fill_mode": config.fill_mode,
        "requested_size": requested_size,
    }
    if decision.quantity <= 0:
        return (
            PaperOrder(
                **base_order,
                normalized_limit_price=None,
                status="rejected",
                reason="invalid_quantity",
                filled_size=0.0,
                unfilled_size=max(0.0, requested_size),
                no_fill_reason="invalid_quantity",
            ),
            None,
            cash,
        )

    tick_size = _tick_size_for(record, outcome=outcome, config=config)
    normalized_limit = (
        _normalize_price(decision.limit_price, tick_size=tick_size, side=side)
        if decision.limit_price is not None
        else None
    )
    execution = _fill_execution(
        record,
        side=side,
        outcome=outcome,
        requested_size=requested_size,
        normalized_limit=normalized_limit,
        config=config,
    )
    if execution.filled_size <= 0 or execution.average_fill_price is None:
        return (
            PaperOrder(
                **base_order,
                normalized_limit_price=normalized_limit,
                status="no_fill",
                reason=execution.no_fill_reason,
                filled_size=0.0,
                unfilled_size=execution.unfilled_size,
                average_fill_price=None,
                levels_consumed=execution.levels_consumed,
                liquidity_available=execution.liquidity_available,
                partial_fill=False,
                no_fill_reason=execution.no_fill_reason,
            ),
            None,
            cash,
        )

    fill_price = float(execution.average_fill_price)
    quantity = float(execution.filled_size)
    notional = _round_money(execution.notional)
    fee = _round_money(notional * max(0.0, float(config.fee_bps)) / 10_000.0)
    position_key = (record.market_id, outcome)
    position = positions.get(position_key)
    risk_rejection = _risk_rejection_reason(
        side=side,
        record=record,
        quantity=quantity,
        notional=notional,
        positions=positions,
        config=config,
    )
    if risk_rejection is not None:
        return (
            PaperOrder(
                **base_order,
                normalized_limit_price=normalized_limit,
                status="rejected",
                reason=risk_rejection,
                filled_size=0.0,
                unfilled_size=requested_size,
                average_fill_price=fill_price,
                levels_consumed=execution.levels_consumed,
                liquidity_available=execution.liquidity_available,
                partial_fill=False,
                no_fill_reason=risk_rejection,
            ),
            None,
            cash,
        )

    if side == BUY:
        total_cost = _round_money(notional + fee)
        if total_cost > cash + 1e-12:
            return (
                PaperOrder(
                    **base_order,
                    normalized_limit_price=normalized_limit,
                    status="rejected",
                    reason="insufficient_cash",
                    filled_size=0.0,
                    unfilled_size=requested_size,
                    average_fill_price=fill_price,
                    levels_consumed=execution.levels_consumed,
                    liquidity_available=execution.liquidity_available,
                    partial_fill=False,
                    no_fill_reason="insufficient_cash",
                ),
                None,
                cash,
            )
        cash = _round_money(cash - total_cost)
        positions[position_key] = _add_position(
            position,
            market_id=record.market_id,
            outcome=outcome,
            quantity=quantity,
            price=fill_price,
        )
        cash_delta = -total_cost
    else:
        if position is None or position.quantity + 1e-12 < quantity:
            return (
                PaperOrder(
                    **base_order,
                    normalized_limit_price=normalized_limit,
                    status="rejected",
                    reason="insufficient_position",
                    filled_size=0.0,
                    unfilled_size=requested_size,
                    average_fill_price=fill_price,
                    levels_consumed=execution.levels_consumed,
                    liquidity_available=execution.liquidity_available,
                    partial_fill=False,
                    no_fill_reason="insufficient_position",
                ),
                None,
                cash,
            )
        proceeds = _round_money(notional - fee)
        cash = _round_money(cash + proceeds)
        positions[position_key] = _reduce_position(position, quantity=quantity)
        cash_delta = proceeds

    order = PaperOrder(
        **base_order,
        normalized_limit_price=normalized_limit,
        status="partial_fill" if execution.partial_fill else "filled",
        reason=decision.reason,
        filled_size=quantity,
        unfilled_size=execution.unfilled_size,
        average_fill_price=fill_price,
        levels_consumed=execution.levels_consumed,
        liquidity_available=execution.liquidity_available,
        partial_fill=execution.partial_fill,
        no_fill_reason=execution.no_fill_reason,
    )
    fill = PaperFill(
        fill_id=f"paper_fill_{order_seq}",
        order_id=order_id,
        market_id=record.market_id,
        timestamp=record.timestamp,
        side=side,
        outcome=outcome,
        quantity=quantity,
        price=fill_price,
        fee=fee,
        cash_delta=_round_money(cash_delta),
        fill_mode=config.fill_mode,
        requested_size=requested_size,
        filled_size=quantity,
        unfilled_size=execution.unfilled_size,
        average_fill_price=fill_price,
        levels_consumed=execution.levels_consumed,
        liquidity_available=execution.liquidity_available,
        partial_fill=execution.partial_fill,
        no_fill_reason=execution.no_fill_reason,
    )
    return order, fill, cash


def _fill_execution(
    record: BacktestRecord,
    *,
    side: str,
    outcome: str,
    requested_size: float,
    normalized_limit: float | None,
    config: PaperTraderConfig,
) -> _FillExecution:
    if config.fill_mode == FILL_MODE_DEPTH_AWARE:
        return _depth_aware_fill_execution(
            record,
            side=side,
            outcome=outcome,
            requested_size=requested_size,
            normalized_limit=normalized_limit,
            config=config,
        )
    return _top_of_book_fill_execution(
        record,
        side=side,
        outcome=outcome,
        requested_size=requested_size,
        normalized_limit=normalized_limit,
        config=config,
    )


def _top_of_book_fill_execution(
    record: BacktestRecord,
    *,
    side: str,
    outcome: str,
    requested_size: float,
    normalized_limit: float | None,
    config: PaperTraderConfig,
) -> _FillExecution:
    top_price = _top_of_book_price(record, side=side, outcome=outcome)
    if top_price is None:
        reason = f"missing_top_of_book_{side.lower()}_{outcome.lower()}"
        return _no_fill_execution(
            fill_mode=FILL_MODE_TOP_OF_BOOK,
            requested_size=requested_size,
            liquidity_available=None,
            no_fill_reason=reason,
        )

    if normalized_limit is not None:
        if side == BUY and normalized_limit < top_price:
            return _no_fill_execution(
                fill_mode=FILL_MODE_TOP_OF_BOOK,
                requested_size=requested_size,
                liquidity_available=0.0,
                no_fill_reason="buy_limit_below_best_ask",
            )
        if side == SELL and normalized_limit > top_price:
            return _no_fill_execution(
                fill_mode=FILL_MODE_TOP_OF_BOOK,
                requested_size=requested_size,
                liquidity_available=0.0,
                no_fill_reason="sell_limit_above_best_bid",
            )

    tick_size = _tick_size_for(record, outcome=outcome, config=config)
    fill_price = _apply_slippage(top_price, side=side, config=config)
    fill_price = _normalize_price(fill_price, tick_size=tick_size, side=side)
    return _FillExecution(
        fill_mode=FILL_MODE_TOP_OF_BOOK,
        requested_size=requested_size,
        filled_size=requested_size,
        unfilled_size=0.0,
        average_fill_price=fill_price,
        notional=_round_money(fill_price * requested_size),
        levels_consumed=1,
        liquidity_available=requested_size,
        partial_fill=False,
        no_fill_reason="",
    )


def _depth_aware_fill_execution(
    record: BacktestRecord,
    *,
    side: str,
    outcome: str,
    requested_size: float,
    normalized_limit: float | None,
    config: PaperTraderConfig,
) -> _FillExecution:
    book_side = "asks" if side == BUY else "bids"
    levels = _depth_levels_for(
        record,
        outcome=outcome,
        book_side=book_side,
        max_depth_levels=config.max_depth_levels,
    )
    if not levels:
        return _no_fill_execution(
            fill_mode=FILL_MODE_DEPTH_AWARE,
            requested_size=requested_size,
            liquidity_available=0.0,
            no_fill_reason="no_liquidity",
        )

    acceptable: list[tuple[float, float]] = []
    for price, size in levels:
        if normalized_limit is None:
            acceptable.append((price, size))
        elif side == BUY and price <= normalized_limit:
            acceptable.append((price, size))
        elif side == SELL and price >= normalized_limit:
            acceptable.append((price, size))

    liquidity_available = _round_quantity(sum(size for _price, size in acceptable))
    if liquidity_available <= 1e-12:
        return _no_fill_execution(
            fill_mode=FILL_MODE_DEPTH_AWARE,
            requested_size=requested_size,
            liquidity_available=0.0,
            no_fill_reason="limit_not_crossed" if normalized_limit is not None else "no_liquidity",
        )

    filled_size = min(requested_size, liquidity_available)
    partial_fill = filled_size + 1e-12 < requested_size
    if partial_fill and not config.allow_partial_fills:
        return _no_fill_execution(
            fill_mode=FILL_MODE_DEPTH_AWARE,
            requested_size=requested_size,
            liquidity_available=liquidity_available,
            no_fill_reason="insufficient_depth",
        )

    remaining = filled_size
    notional = 0.0
    levels_consumed = 0
    for price, size in acceptable:
        if remaining <= 1e-12:
            break
        consumed = min(size, remaining)
        if consumed <= 0:
            continue
        notional += consumed * price
        remaining -= consumed
        levels_consumed += 1

    raw_average_fill_price = notional / filled_size
    average_fill_price = _apply_slippage(raw_average_fill_price, side=side, config=config)
    fill_notional = (
        _round_money(notional)
        if config.fixed_slippage == 0 and config.slippage_bps == 0
        else _round_money(average_fill_price * filled_size)
    )
    return _FillExecution(
        fill_mode=FILL_MODE_DEPTH_AWARE,
        requested_size=requested_size,
        filled_size=_round_quantity(filled_size),
        unfilled_size=_round_quantity(max(0.0, requested_size - filled_size)),
        average_fill_price=_round_money(average_fill_price),
        notional=fill_notional,
        levels_consumed=levels_consumed,
        liquidity_available=liquidity_available,
        partial_fill=partial_fill,
        no_fill_reason="insufficient_depth" if partial_fill else "",
    )


def _no_fill_execution(
    *,
    fill_mode: str,
    requested_size: float,
    liquidity_available: float | None,
    no_fill_reason: str,
) -> _FillExecution:
    return _FillExecution(
        fill_mode=fill_mode,
        requested_size=requested_size,
        filled_size=0.0,
        unfilled_size=max(0.0, requested_size),
        average_fill_price=None,
        notional=0.0,
        levels_consumed=0,
        liquidity_available=liquidity_available,
        partial_fill=False,
        no_fill_reason=no_fill_reason,
    )


def _settle_positions(
    dataset: BacktestDataset,
    *,
    positions: dict[tuple[str, str], PaperPosition],
    cash: float,
) -> tuple[float, tuple[PaperSettlement, ...]]:
    settlements: list[PaperSettlement] = []
    for market_id, timeline in dataset.timelines.items():
        winning_outcome, settlement_timestamp = _winning_outcome_and_timestamp(
            timeline.records
        )
        if winning_outcome is None:
            continue
        for outcome in (YES, NO):
            key = (market_id, outcome)
            position = positions.get(key)
            if position is None or position.quantity <= 0:
                continue
            settlement_price = 1.0 if outcome == winning_outcome else 0.0
            cash_delta = _round_money(position.quantity * settlement_price)
            settlement_pnl = _round_money(
                position.quantity * (settlement_price - position.average_price)
            )
            settlements.append(
                PaperSettlement(
                    settlement_id=f"paper_settlement_{len(settlements) + 1}",
                    market_id=market_id,
                    timestamp=settlement_timestamp,
                    outcome=outcome,
                    quantity=position.quantity,
                    average_price=position.average_price,
                    winning_outcome=winning_outcome,
                    settlement_price=settlement_price,
                    cash_delta=cash_delta,
                    settlement_pnl=settlement_pnl,
                )
            )
            cash = _round_money(cash + cash_delta)
            positions[key] = PaperPosition(
                market_id=market_id,
                outcome=outcome,
                quantity=0.0,
                average_price=position.average_price,
            )
    return cash, tuple(settlements)


def _winning_outcome(records: Sequence[BacktestRecord]) -> str | None:
    winning_outcome, _timestamp = _winning_outcome_and_timestamp(records)
    return winning_outcome


def _winning_outcome_and_timestamp(
    records: Sequence[BacktestRecord],
) -> tuple[str | None, datetime | None]:
    for record in reversed(records):
        if int(record.label_available or 0) != 1:
            continue
        if record.yes_won == 1:
            return YES, record.resolved_at or record.timestamp
        if record.no_won == 1:
            return NO, record.resolved_at or record.timestamp
        if record.winning_outcome:
            normalized = str(record.winning_outcome).strip().upper()
            if normalized in {YES, "UP"}:
                return YES, record.resolved_at or record.timestamp
            if normalized in {NO, "DOWN"}:
                return NO, record.resolved_at or record.timestamp
    return None, None


def _add_position(
    position: PaperPosition | None,
    *,
    market_id: str,
    outcome: str,
    quantity: float,
    price: float,
) -> PaperPosition:
    if position is None or position.quantity <= 0:
        return PaperPosition(
            market_id=market_id,
            outcome=outcome,
            quantity=quantity,
            average_price=price,
        )
    new_quantity = position.quantity + quantity
    average_price = (
        (position.quantity * position.average_price) + (quantity * price)
    ) / new_quantity
    return PaperPosition(
        market_id=market_id,
        outcome=outcome,
        quantity=_round_quantity(new_quantity),
        average_price=_round_money(average_price),
    )


def _reduce_position(position: PaperPosition, *, quantity: float) -> PaperPosition:
    return PaperPosition(
        market_id=position.market_id,
        outcome=position.outcome,
        quantity=_round_quantity(max(0.0, position.quantity - quantity)),
        average_price=position.average_price,
    )


def _decision_matches_record(decision: PaperDecision, record: BacktestRecord) -> bool:
    if decision.market_id is not None and decision.market_id != record.market_id:
        return False
    if decision.timestamp is not None and decision.timestamp != record.timestamp:
        return False
    if (
        decision.min_seconds_until_close is not None
        and record.seconds_until_close is not None
        and record.seconds_until_close < decision.min_seconds_until_close
    ):
        return False
    if (
        decision.max_seconds_until_close is not None
        and record.seconds_until_close is not None
        and record.seconds_until_close > decision.max_seconds_until_close
    ):
        return False
    return True


def _rule_matches_record(rule: PaperRule, record: BacktestRecord) -> bool:
    if rule.market_id is not None and rule.market_id != record.market_id:
        return False
    if (
        rule.min_seconds_until_close is not None
        and record.seconds_until_close is not None
        and record.seconds_until_close < rule.min_seconds_until_close
    ):
        return False
    if (
        rule.max_seconds_until_close is not None
        and record.seconds_until_close is not None
        and record.seconds_until_close > rule.max_seconds_until_close
    ):
        return False
    if rule.buy_yes_ask_lte is not None:
        if record.best_ask_yes is None or record.best_ask_yes > rule.buy_yes_ask_lte:
            return False
    if rule.buy_no_ask_lte is not None:
        if record.best_ask_no is None or record.best_ask_no > rule.buy_no_ask_lte:
            return False
    return True


def _top_of_book_price(record: BacktestRecord, *, side: str, outcome: str) -> float | None:
    if side == BUY:
        return record.best_ask_yes if outcome == YES else record.best_ask_no
    return record.best_bid_yes if outcome == YES else record.best_bid_no


def _depth_levels_for(
    record: BacktestRecord,
    *,
    outcome: str,
    book_side: str,
    max_depth_levels: int | None,
) -> list[tuple[float, float]]:
    raw_value = _raw_depth_value(record.raw, outcome=outcome, book_side=book_side)
    levels = _parse_depth_levels(raw_value)
    reverse = book_side == "bids"
    levels = sorted(levels, key=lambda item: item[0], reverse=reverse)
    if max_depth_levels is not None and max_depth_levels >= 0:
        levels = levels[:max_depth_levels]
    return levels


def _raw_depth_value(
    raw: Mapping[str, Any],
    *,
    outcome: str,
    book_side: str,
) -> Any:
    outcome_key = outcome.lower()
    side_key = book_side.lower()
    flat_keys = (
        f"{outcome_key}_{side_key}",
        f"{outcome_key}_{side_key}_json",
        f"{side_key}_{outcome_key}",
        f"{side_key}_{outcome_key}_json",
        f"{outcome_key}_book_{side_key}",
        f"{outcome_key}_book_{side_key}_json",
        f"orderbook_{outcome_key}_{side_key}",
        f"orderbook_{outcome_key}_{side_key}_json",
        f"order_book_{outcome_key}_{side_key}",
        f"order_book_{outcome_key}_{side_key}_json",
    )
    for key in flat_keys:
        if key in raw and raw.get(key) not in (None, ""):
            return raw.get(key)

    container_keys = (
        "order_book_depth",
        "order_book_depth_json",
        "orderbook_depth",
        "orderbook_depth_json",
        "order_book_levels",
        "order_book_levels_json",
        "orderbook",
        "orderbook_json",
        "depth",
        "depth_json",
    )
    for key in container_keys:
        parsed = _parse_json_if_string(raw.get(key))
        nested = _nested_depth_value(parsed, outcome_key=outcome_key, side_key=side_key)
        if nested not in (None, ""):
            return nested
    return None


def _nested_depth_value(
    value: Any,
    *,
    outcome_key: str,
    side_key: str,
) -> Any:
    if not isinstance(value, Mapping):
        return None
    for key in (
        f"{outcome_key}_{side_key}",
        f"{side_key}_{outcome_key}",
        f"{outcome_key}_{side_key}_json",
        f"{side_key}_{outcome_key}_json",
    ):
        if key in value and value.get(key) not in (None, ""):
            return value.get(key)
    for outcome_candidate in (outcome_key, outcome_key.upper()):
        nested = value.get(outcome_candidate)
        if isinstance(nested, Mapping):
            for side_candidate in (side_key, side_key.upper()):
                if side_candidate in nested and nested.get(side_candidate) not in (None, ""):
                    return nested.get(side_candidate)
    return None


def _parse_depth_levels(value: Any) -> list[tuple[float, float]]:
    parsed = _parse_json_if_string(value)
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, Mapping):
        if "levels" in parsed:
            parsed = parsed.get("levels")
        elif "price" in parsed or "size" in parsed:
            parsed = [parsed]
        else:
            return []
    if not isinstance(parsed, SequenceABC) or isinstance(parsed, (str, bytes, bytearray)):
        return []

    levels: list[tuple[float, float]] = []
    for item in parsed:
        price: object | None
        size: object | None
        if isinstance(item, Mapping):
            price = item.get("price", item.get("px"))
            size = item.get("size", item.get("quantity", item.get("shares")))
        elif isinstance(item, SequenceABC) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) < 2:
                continue
            price = item[0]
            size = item[1]
        else:
            continue
        try:
            price_float = float(price)  # type: ignore[arg-type]
            size_float = float(size)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if price_float < 0 or size_float <= 0:
            continue
        levels.append((price_float, size_float))
    return levels


def _parse_json_if_string(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _tick_size_for(
    record: BacktestRecord,
    *,
    outcome: str,
    config: PaperTraderConfig,
) -> float:
    tick = record.latest_tick_size_yes if outcome == YES else record.latest_tick_size_no
    if tick is None or tick <= 0:
        tick = config.default_tick_size
    return max(float(tick), 0.000001)


def _apply_slippage(price: float, *, side: str, config: PaperTraderConfig) -> float:
    adjusted = float(price)
    fixed = max(0.0, float(config.fixed_slippage))
    bps = max(0.0, float(config.slippage_bps))
    if side == BUY:
        adjusted += fixed
        adjusted *= 1.0 + (bps / 10_000.0)
    else:
        adjusted -= fixed
        adjusted *= 1.0 - (bps / 10_000.0)
    return min(1.0, max(0.0, adjusted))


def _risk_rejection_reason(
    *,
    side: str,
    record: BacktestRecord,
    quantity: float,
    notional: float,
    positions: Mapping[tuple[str, str], PaperPosition],
    config: PaperTraderConfig,
) -> str | None:
    if (
        config.max_notional_per_order is not None
        and notional > float(config.max_notional_per_order) + 1e-12
    ):
        return "max_notional_per_order_exceeded"
    if side != BUY:
        return None

    if config.max_position_size_per_market is not None:
        market_position = sum(
            position.quantity
            for (market_id, _outcome), position in positions.items()
            if market_id == record.market_id and position.quantity > 0
        )
        if (
            market_position + quantity
            > float(config.max_position_size_per_market) + 1e-12
        ):
            return "max_position_size_per_market_exceeded"

    if config.max_total_open_notional is not None:
        open_notional = sum(
            position.quantity * position.average_price
            for position in positions.values()
            if position.quantity > 0
        )
        if open_notional + notional > float(config.max_total_open_notional) + 1e-12:
            return "max_total_open_notional_exceeded"
    return None


def _normalize_price(price: float, *, tick_size: float, side: str) -> float:
    price_dec = Decimal(str(price))
    tick_dec = Decimal(str(tick_size))
    rounding = ROUND_CEILING if side == BUY else ROUND_FLOOR
    ticks = (price_dec / tick_dec).to_integral_value(rounding=rounding)
    normalized = ticks * tick_dec
    normalized = min(Decimal("1"), max(Decimal("0"), normalized))
    return float(normalized)


def _normalize_side(side: str) -> str:
    normalized = str(side).strip().upper()
    if normalized not in {BUY, SELL}:
        raise PaperTraderError(f"Unsupported order side: {side}")
    return normalized


def _normalize_outcome(outcome: str) -> str:
    normalized = str(outcome).strip().upper()
    if normalized not in {YES, NO}:
        raise PaperTraderError(f"Unsupported outcome: {outcome}")
    return normalized


def _round_money(value: float) -> float:
    return round(float(value), 10)


def _round_quantity(value: float) -> float:
    return round(float(value), 10)


def _strategy_name(strategy: PaperStrategy) -> str:
    if isinstance(strategy, NoOpStrategy):
        return "noop"
    if isinstance(strategy, RuleStrategy):
        return "rule"
    return strategy.__class__.__name__


def _config_to_dict(config: PaperTraderConfig) -> dict[str, Any]:
    return _jsonable(asdict(config))


def _validate_config(config: PaperTraderConfig) -> None:
    if config.fill_mode not in PAPER_FILL_MODES:
        raise PaperTraderError(
            f"Unsupported paper fill mode: {config.fill_mode}. "
            f"Expected one of: {', '.join(sorted(PAPER_FILL_MODES))}"
        )
    if config.max_depth_levels is not None and int(config.max_depth_levels) < 0:
        raise PaperTraderError("max_depth_levels must be non-negative when set")


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _paper_run_detail(summary: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(summary.get("run_id") or "unknown_run")
    summary_path = Path(str(summary.get("summary_path") or ""))
    prefix = (
        summary_path.name[: -len("_summary.json")]
        if summary_path.name.endswith("_summary.json")
        else _safe_filename(run_id)
    )
    base = summary_path.parent if summary_path.parent != Path("") else Path(".")
    orders_path = base / f"{prefix}_orders.csv"
    fills_path = base / f"{prefix}_fills.csv"
    settlements_path = base / f"{prefix}_settlements.csv"
    warnings: list[str] = []

    if orders_path.exists():
        orders, order_fields = _read_csv_rows_with_fieldnames(orders_path)
    else:
        orders = []
        order_fields = set()
        warnings.append(f"{run_id}:missing_orders_csv")

    if fills_path.exists():
        fills, fill_fields = _read_csv_rows_with_fieldnames(fills_path)
    else:
        fills = []
        fill_fields = set()
        warnings.append(f"{run_id}:missing_fills_csv")

    if settlements_path.exists():
        settlements, _settlement_fields = _read_csv_rows_with_fieldnames(settlements_path)
        settlement_detail_available = True
    else:
        settlements = []
        settlement_detail_available = False
        warnings.append(f"{run_id}:settlement_detail_unavailable")

    depth_detail_available = bool(
        (order_fields | fill_fields) & set(PAPER_REPORT_DEPTH_FIELDS)
    )
    return {
        "orders_csv": str(orders_path) if orders_path.exists() else None,
        "fills_csv": str(fills_path) if fills_path.exists() else None,
        "settlements_csv": str(settlements_path) if settlements_path.exists() else None,
        "settlement_detail_available": settlement_detail_available,
        "depth_detail_available": depth_detail_available,
        "per_market": _paper_per_market_detail(
            orders=orders,
            fills=fills,
            settlements=settlements,
            settlement_detail_available=settlement_detail_available,
            depth_detail_available=depth_detail_available,
        ),
        "cohorts": _paper_cohort_detail(orders=orders, fills=fills),
        "depth_summary": _paper_depth_summary(
            orders=orders,
            fills=fills,
            depth_detail_available=depth_detail_available,
        ),
        "warnings": warnings,
    }


def _paper_per_market_detail(
    *,
    orders: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    settlements: Sequence[Mapping[str, Any]],
    settlement_detail_available: bool,
    depth_detail_available: bool,
) -> list[dict[str, Any]]:
    by_market: dict[str, dict[str, Any]] = {}
    positions: dict[tuple[str, str], tuple[float, float]] = {}

    for order in orders:
        market_id = str(order.get("market_id") or "unknown_market")
        row = by_market.setdefault(market_id, _empty_market_detail(market_id))
        row["total_orders"] += 1
        if depth_detail_available:
            row["total_unfilled_size"] = _round_quantity(
                float(row["total_unfilled_size"] or 0.0)
                + _float_report_value(order.get("unfilled_size"))
            )
            if _bool_report_value(order.get("partial_fill")) or str(
                order.get("status") or ""
            ).lower() == "partial_fill":
                row["partial_fill_count"] += 1

    for fill in fills:
        market_id = str(fill.get("market_id") or "unknown_market")
        side = str(fill.get("side") or "").upper()
        outcome = str(fill.get("outcome") or "").upper()
        quantity = _float_report_value(fill.get("quantity"))
        price = _float_report_value(fill.get("price"))
        fee = _float_report_value(fill.get("fee"))
        row = by_market.setdefault(market_id, _empty_market_detail(market_id))
        row["total_fills"] += 1
        if side == BUY:
            row["buy_fills"] += 1
            current_quantity, current_average = positions.get(
                (market_id, outcome),
                (0.0, 0.0),
            )
            new_quantity = current_quantity + quantity
            new_average = (
                ((current_quantity * current_average) + (quantity * price) + fee)
                / new_quantity
                if new_quantity > 0
                else 0.0
            )
            positions[(market_id, outcome)] = (
                _round_quantity(new_quantity),
                _round_money(new_average),
            )
        elif side == SELL:
            row["sell_fills"] += 1
            current_quantity, current_average = positions.get(
                (market_id, outcome),
                (0.0, 0.0),
            )
            matched_quantity = min(quantity, current_quantity)
            proceeds = (quantity * price) - fee
            if matched_quantity > 0:
                basis = matched_quantity * current_average
                row["realized_pnl"] = _round_money(
                    float(row["realized_pnl"]) + proceeds - basis
                )
            else:
                row["realized_pnl"] = _round_money(
                    float(row["realized_pnl"]) + proceeds
                )
            positions[(market_id, outcome)] = (
                _round_quantity(max(0.0, current_quantity - quantity)),
                current_average,
            )
        row["gross_notional"] = _round_money(
            float(row["gross_notional"]) + abs(quantity * price)
        )
        if depth_detail_available:
            filled_size = _float_report_value(fill.get("filled_size")) or quantity
            average_fill_price = _float_report_value(
                fill.get("average_fill_price", fill.get("price"))
            )
            if "levels_consumed" in fill and fill.get("levels_consumed") not in (None, ""):
                row["_levels_consumed_sum"] += _float_report_value(
                    fill.get("levels_consumed")
                )
                row["_levels_consumed_count"] += 1
            if average_fill_price > 0 and filled_size > 0:
                row["_fill_price_notional"] += average_fill_price * filled_size
                row["_fill_price_quantity"] += filled_size

    positions_after_settlement = dict(positions)
    for (market_id, _outcome), (quantity, _average) in positions.items():
        if quantity <= 1e-12:
            continue
        row = by_market.setdefault(market_id, _empty_market_detail(market_id))
        row["open_position_shares_before_settlement"] = _round_quantity(
            float(row["open_position_shares_before_settlement"]) + quantity
        )

    if settlement_detail_available:
        for settlement in settlements:
            market_id = str(settlement.get("market_id") or "unknown_market")
            outcome = str(settlement.get("outcome") or "").upper()
            quantity = _float_report_value(settlement.get("quantity"))
            settlement_pnl = _float_report_value(settlement.get("settlement_pnl"))
            row = by_market.setdefault(market_id, _empty_market_detail(market_id))
            row["settled_shares"] = _round_quantity(
                float(row["settled_shares"] or 0.0) + quantity
            )
            row["settlement_pnl"] = _round_money(
                float(row["settlement_pnl"] or 0.0) + settlement_pnl
            )
            row["realized_pnl"] = _round_money(
                float(row["realized_pnl"]) + settlement_pnl
            )
            current_quantity, current_average = positions_after_settlement.get(
                (market_id, outcome),
                (0.0, 0.0),
            )
            positions_after_settlement[(market_id, outcome)] = (
                _round_quantity(max(0.0, current_quantity - quantity)),
                current_average,
            )

        for (
            market_id,
            _outcome,
        ), (quantity, _average) in positions_after_settlement.items():
            if quantity <= 1e-12:
                continue
            row = by_market.setdefault(market_id, _empty_market_detail(market_id))
            row["unsettled_positions_after_settlement"] += 1
    else:
        for row in by_market.values():
            if float(row["open_position_shares_before_settlement"] or 0.0) > 0:
                row["settled_shares"] = None
                row["unsettled_positions_after_settlement"] = None
                row["settlement_pnl"] = None

    for row in by_market.values():
        if depth_detail_available:
            if row["_levels_consumed_count"] > 0:
                row["avg_levels_consumed"] = _round_report_average(
                    row["_levels_consumed_sum"],
                    row["_levels_consumed_count"],
                )
            if row["_fill_price_quantity"] > 0:
                row["avg_fill_price"] = _round_money(
                    row["_fill_price_notional"] / row["_fill_price_quantity"]
                )
        else:
            row["partial_fill_count"] = None
            row["avg_levels_consumed"] = None
            row["total_unfilled_size"] = None
            row["avg_fill_price"] = None

    return [
        _strip_internal_report_fields(row)
        for _market_id, row in sorted(by_market.items(), key=lambda item: item[0])
    ]


def _paper_cohort_detail(
    *,
    orders: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rejected_reasons: dict[str, int] = {}
    no_fill_reasons: dict[str, int] = {}
    cohorts = {
        "yes_fills": 0,
        "no_fills": 0,
        "buy_fills": 0,
        "sell_fills": 0,
        "no_fill_orders": 0,
        "rejected_order_reasons": rejected_reasons,
        "no_fill_reasons": no_fill_reasons,
    }
    for fill in fills:
        outcome = str(fill.get("outcome") or "").upper()
        side = str(fill.get("side") or "").upper()
        if outcome == YES:
            cohorts["yes_fills"] += 1
        elif outcome == NO:
            cohorts["no_fills"] += 1
        if side == BUY:
            cohorts["buy_fills"] += 1
        elif side == SELL:
            cohorts["sell_fills"] += 1

    for order in orders:
        status = str(order.get("status") or "").lower()
        reason = str(order.get("reason") or "unknown")
        if status == "no_fill":
            cohorts["no_fill_orders"] += 1
            no_fill_reason = str(order.get("no_fill_reason") or reason or "unknown")
            no_fill_reasons[no_fill_reason] = no_fill_reasons.get(no_fill_reason, 0) + 1
        elif status == "rejected":
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    return cohorts


def _paper_depth_summary(
    *,
    orders: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    depth_detail_available: bool,
) -> dict[str, Any]:
    fill_mode_counts: dict[str, int] = {}
    no_fill_reason_counts: dict[str, int] = {}
    summary: dict[str, Any] = {
        "depth_detail_available": depth_detail_available,
        "fill_mode_counts": fill_mode_counts,
        "partial_fill_count": 0,
        "total_requested_size": None,
        "total_filled_size": None,
        "total_unfilled_size": None,
        "avg_levels_consumed": None,
        "avg_liquidity_available": None,
        "avg_fill_price": None,
        "no_fill_reason_counts": no_fill_reason_counts,
    }
    if not depth_detail_available:
        return summary

    mode_rows = orders if any("fill_mode" in row for row in orders) else fills
    for row in mode_rows:
        mode = str(row.get("fill_mode") or "").strip()
        if mode:
            fill_mode_counts[mode] = fill_mode_counts.get(mode, 0) + 1

    size_rows = orders if any("requested_size" in row for row in orders) else fills
    summary["total_requested_size"] = _round_quantity(
        sum(_float_report_value(row.get("requested_size")) for row in size_rows)
    )
    summary["total_filled_size"] = _round_quantity(
        sum(_float_report_value(row.get("filled_size")) for row in size_rows)
    )
    summary["total_unfilled_size"] = _round_quantity(
        sum(_float_report_value(row.get("unfilled_size")) for row in size_rows)
    )

    partial_rows = orders if orders else fills
    summary["partial_fill_count"] = sum(
        1
        for row in partial_rows
        if _bool_report_value(row.get("partial_fill"))
        or str(row.get("status") or "").lower() == "partial_fill"
    )

    level_values = [
        _float_report_value(row.get("levels_consumed"))
        for row in fills
        if row.get("levels_consumed") not in (None, "")
    ]
    if level_values:
        summary["avg_levels_consumed"] = _round_report_average(
            sum(level_values),
            len(level_values),
        )

    liquidity_rows = orders if any("liquidity_available" in row for row in orders) else fills
    liquidity_values = [
        _float_report_value(row.get("liquidity_available"))
        for row in liquidity_rows
        if row.get("liquidity_available") not in (None, "")
    ]
    if liquidity_values:
        summary["avg_liquidity_available"] = _round_report_average(
            sum(liquidity_values),
            len(liquidity_values),
        )

    price_notional = 0.0
    price_quantity = 0.0
    for fill in fills:
        filled_size = _float_report_value(fill.get("filled_size")) or _float_report_value(
            fill.get("quantity")
        )
        average_fill_price = _float_report_value(
            fill.get("average_fill_price", fill.get("price"))
        )
        if filled_size > 0 and average_fill_price > 0:
            price_notional += average_fill_price * filled_size
            price_quantity += filled_size
    if price_quantity > 0:
        summary["avg_fill_price"] = _round_money(price_notional / price_quantity)

    for order in orders:
        reason = str(order.get("no_fill_reason") or "").strip()
        if not reason and str(order.get("status") or "").lower() == "no_fill":
            reason = str(order.get("reason") or "unknown")
        if reason:
            no_fill_reason_counts[reason] = no_fill_reason_counts.get(reason, 0) + 1
    return summary


def _empty_market_detail(market_id: str) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "total_orders": 0,
        "total_fills": 0,
        "buy_fills": 0,
        "sell_fills": 0,
        "gross_notional": 0.0,
        "realized_pnl": 0.0,
        "open_position_shares_before_settlement": 0.0,
        "settled_shares": 0.0,
        "unsettled_positions_after_settlement": 0,
        "settlement_pnl": 0.0,
        "partial_fill_count": 0,
        "avg_levels_consumed": None,
        "total_unfilled_size": 0.0,
        "avg_fill_price": None,
        "_levels_consumed_sum": 0.0,
        "_levels_consumed_count": 0,
        "_fill_price_notional": 0.0,
        "_fill_price_quantity": 0.0,
    }


def _strip_internal_report_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows, _fieldnames = _read_csv_rows_with_fieldnames(path)
    return rows


def _read_csv_rows_with_fieldnames(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader], set(reader.fieldnames or ())


def _paper_report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in PAPER_REPORT_FIELDS}


def _paper_report_sort_key(row: Mapping[str, Any], sort_by: str) -> Any:
    if sort_by == "run_id":
        return str(row.get("run_id") or "")
    value = row.get(sort_by)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _paper_report_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["no_summary_files_found"]
    warnings: list[str] = []
    for row in rows:
        run_id = str(row.get("run_id") or "unknown_run")
        total_orders = _int_report_value(row.get("total_orders"))
        total_fills = _int_report_value(row.get("total_fills"))
        rejected_orders = _int_report_value(row.get("rejected_orders"))
        unsettled_positions = _int_report_value(row.get("unsettled_positions"))
        if total_fills == 0:
            warnings.append(f"{run_id}:no_fills")
        if unsettled_positions > 0:
            warnings.append(f"{run_id}:unsettled_positions={unsettled_positions}")
        if rejected_orders > 0 and (
            rejected_orders >= 10
            or (total_orders > 0 and rejected_orders / total_orders >= 0.5)
        ):
            warnings.append(f"{run_id}:high_rejected_orders={rejected_orders}")
    return warnings


def _paper_report_cell(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _int_report_value(value: object) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_report_value(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_report_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _round_report_average(total: object, count: object) -> float | None:
    count_float = _float_report_value(count)
    if count_float <= 0:
        return None
    return _round_money(_float_report_value(total) / count_float)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "BUY",
    "FILL_MODE_DEPTH_AWARE",
    "FILL_MODE_TOP_OF_BOOK",
    "NO",
    "PAPER_FILL_MODES",
    "SELL",
    "YES",
    "NoOpStrategy",
    "PaperDecision",
    "PaperFill",
    "PaperOrder",
    "PaperPosition",
    "PaperRule",
    "PaperSettlement",
    "PaperStrategy",
    "PaperTraderConfig",
    "PaperTraderError",
    "PaperTraderRunResult",
    "RuleStrategy",
    "PAPER_REPORT_FIELDS",
    "PAPER_REPORT_MARKET_FIELDS",
    "PAPER_REPORT_SORT_FIELDS",
    "build_paper_runs_report",
    "build_strategy",
    "build_rule_strategy",
    "export_paper_backtest_result",
    "load_strategy_config",
    "load_paper_run_summaries",
    "render_paper_runs_report",
    "result_to_dict",
    "result_summary_to_dict",
    "run_paper_backtest_from_export",
    "run_paper_trader",
]
