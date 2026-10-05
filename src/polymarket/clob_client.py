from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Sequence

import httpx

from ..models import OrderBookSnapshot, TradeRecord
from .normalization import normalize_orderbook_snapshot, normalize_trade_history


def _raw_trade_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        trades = payload.get("trades")
        if isinstance(trades, list):
            return len(trades)
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
    return 0


class ClobClient:
    """Client for CLOB orderbook snapshots and trade history backfills."""

    def __init__(
        self,
        clob_base_url: str,
        data_api_base_url: str,
        timeout_sec: float = 10.0,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._clob = httpx.AsyncClient(base_url=clob_base_url, timeout=timeout_sec)
        self._data = httpx.AsyncClient(base_url=data_api_base_url, timeout=timeout_sec)

    async def close(self) -> None:
        await self._clob.aclose()
        await self._data.aclose()

    async def fetch_orderbook_snapshot(self, token_id: str) -> Optional[OrderBookSnapshot]:
        resp = await self._clob.get("/book", params={"token_id": token_id})
        if resp.status_code == 404:
            # Token is valid in metadata but has no live orderbook yet.
            return None
        resp.raise_for_status()
        payload = resp.json()
        return normalize_orderbook_snapshot(payload, token_id=token_id)

    async def fetch_recent_trades(
        self,
        market_id: str,
        token_ids: Optional[Sequence[str]] = None,
        condition_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[TradeRecord]:
        # NOTE: Data API market filters can be inconsistent. We request recent trades
        # and enforce deterministic local filtering by condition/token/since.
        token_filter = {str(token) for token in (token_ids or []) if token}
        query_plan_base: list[tuple[str, dict[str, str]]] = []

        primary_market_value = str(condition_id) if condition_id else str(market_id)
        query_plan_base.append(
            (
                "primary_market_param",
                {"market": primary_market_value, "limit": str(limit)},
            )
        )
        if condition_id:
            query_plan_base.append(
                (
                    "fallback_condition_id_param",
                    {"condition_id": str(condition_id), "limit": str(limit)},
                )
            )
        if condition_id and str(market_id) != str(condition_id):
            query_plan_base.append(
                (
                    "fallback_market_id_param",
                    {"market": str(market_id), "limit": str(limit)},
                )
            )

        diagnostics: list[dict[str, Any]] = []
        query_plan: list[tuple[str, dict[str, str], bool]] = []
        for label, params in query_plan_base:
            query_plan.append((label, dict(params), True))
            if since is not None:
                # Fallback pass in case upstream ignores/handles `since` inconsistently.
                query_plan.append((f"{label}_without_since", dict(params), False))

        for label, params, include_since in query_plan:
            params_for_request = dict(params)
            if since is not None and include_since:
                params_for_request["since"] = since.isoformat()

            request_url = None
            try:
                build_request = getattr(self._data, "build_request", None)
                if callable(build_request):
                    request_url = str(
                        build_request("GET", "/trades", params=params_for_request).url
                    )
            except Exception:
                request_url = None

            self._log.debug(
                "trade_backfill_raw_request",
                extra={
                    "query_label": label,
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "include_since_param": include_since,
                    "local_since_comparison_operator": "trade.timestamp >= since",
                    "params": params_for_request,
                    "request_url": request_url,
                },
            )

            try:
                resp = await self._data.get("/trades", params=params_for_request)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                diagnostics.append(
                    {
                        "query_label": label,
                        "request_url": request_url,
                        "raw_count": 0,
                        "filtered_count": 0,
                        "filtered_before_since_count": 0,
                        "error": repr(exc),
                    }
                )
                self._log.warning(
                    "trade_backfill_query_failed",
                    extra={
                        "query_label": label,
                        "market_id": market_id,
                        "condition_id": condition_id,
                        "request_url": request_url,
                        "error": repr(exc),
                    },
                )
                continue

            raw_count = _raw_trade_count(payload)
            normalized_no_filter = normalize_trade_history(
                payload,
                market_id=market_id,
                token_ids=None,
                condition_id=None,
            )
            normalized_condition = normalize_trade_history(
                payload,
                market_id=market_id,
                token_ids=None,
                condition_id=condition_id,
            )
            normalized_filtered = normalize_trade_history(
                payload,
                market_id=market_id,
                token_ids=token_filter,
                condition_id=condition_id,
            )
            filtered_before_since_count = len(normalized_filtered)
            trades = normalized_filtered
            if since is not None:
                # Data API `since` can be ignored by upstream. Enforce local cutoff
                # so recorder watermark logic remains deterministic.
                trades = [trade for trade in trades if trade.timestamp >= since]

            diag = {
                "query_label": label,
                "request_url": request_url,
                "include_since_param": include_since,
                "raw_count": raw_count,
                "normalized_no_filter_count": len(normalized_no_filter),
                "normalized_condition_count": len(normalized_condition),
                "filtered_before_since_count": filtered_before_since_count,
                "filtered_count": len(trades),
                "token_filter_count": len(token_filter),
                "since": since.isoformat() if since is not None else None,
                "local_since_comparison_operator": "trade.timestamp >= since",
            }
            diagnostics.append(diag)
            self._log.debug("trade_backfill_raw_response", extra=diag)

            if trades:
                if label != "primary_market_param":
                    self._log.info(
                        "trade_backfill_fallback_query_used",
                        extra={
                            "query_label": label,
                            "market_id": market_id,
                            "count": len(trades),
                        },
                    )
                return trades

        # No trades survived filters. Emit a precise reason for observability.
        reason = "empty_market_window"
        if diagnostics and all(int(diag.get("raw_count", 0)) == 0 for diag in diagnostics):
            reason = "wrong_endpoint_or_no_raw_data"
        elif any(
            int(diag.get("filtered_before_since_count", 0)) > 0
            and int(diag.get("filtered_count", 0)) == 0
            for diag in diagnostics
        ):
            reason = "cursor_too_restrictive"
        elif token_filter and any(
            int(diag.get("normalized_condition_count", 0)) > 0
            and int(diag.get("filtered_before_since_count", 0)) == 0
            for diag in diagnostics
        ):
            reason = "wrong_token_filter"
        elif condition_id and any(
            int(diag.get("normalized_no_filter_count", 0)) > 0
            and int(diag.get("normalized_condition_count", 0)) == 0
            for diag in diagnostics
        ):
            reason = "wrong_market_filter"

        self._log.info(
            "trade_backfill_no_trades",
            extra={
                "market_id": market_id,
                "condition_id": condition_id,
                "reason": reason,
                "diagnostics": diagnostics[:4],
            },
        )
        return []
