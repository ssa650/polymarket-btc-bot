from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..models import (
    BestBidAskRecord,
    MarketEventRecord,
    MarketMetadata,
    OrderBookSnapshot,
    PriceLevel,
    RawPolymarketEventRecord,
    TickSizeChangeRecord,
    TradeRecord,
    WSOrderBookUpdate,
    WSPriceChange,
    WSStatusEvent,
    local_arrival_iso_from_ns,
    parse_exchange_timestamp,
    parse_timestamp,
    utc_now,
)

STATUS_EVENT_TYPES = {
    "market_opened",
    "market_closed",
    "market_resolved",
    "trading_halted",
    "status_changed",
}


@dataclass(slots=True)
class NormalizedWSPayload:
    orderbook_updates: List[WSOrderBookUpdate] = field(default_factory=list)
    price_changes: List[WSPriceChange] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    last_trade_prices: List[TradeRecord] = field(default_factory=list)
    status_events: List[WSStatusEvent] = field(default_factory=list)
    tick_size_changes: List[TickSizeChangeRecord] = field(default_factory=list)
    best_bid_ask_updates: List[BestBidAskRecord] = field(default_factory=list)
    new_markets: List[MarketMetadata] = field(default_factory=list)
    market_events: List[MarketEventRecord] = field(default_factory=list)


@dataclass(slots=True)
class GammaMarketExtraction:
    items: List[Mapping[str, Any]] = field(default_factory=list)
    top_level_type: str = "unknown"
    top_level_keys: List[str] = field(default_factory=list)
    extraction_path: str = "unrecognized"



def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _looks_like_market_object(obj: Mapping[str, Any]) -> bool:
    has_identity = any(
        key in obj for key in ("id", "market_id", "conditionId", "condition_id")
    )
    has_marketish_fields = any(
        key in obj
        for key in (
            "question",
            "title",
            "clobTokenIds",
            "outcomes",
            "startDate",
            "startTime",
            "closeTime",
            "endDate",
        )
    )
    return has_identity and has_marketish_fields


def _market_list_from_candidate(
    candidate: Any,
    path: str,
) -> tuple[List[Mapping[str, Any]], Optional[str]]:
    if isinstance(candidate, list):
        if all(isinstance(item, dict) for item in candidate):
            dict_items = [item for item in candidate if isinstance(item, dict)]
            if dict_items and _looks_like_market_object(dict_items[0]):
                return dict_items, path

            flattened: List[Mapping[str, Any]] = []
            for item in dict_items:
                markets = item.get("markets")
                if isinstance(markets, list):
                    flattened.extend(
                        [entry for entry in markets if isinstance(entry, dict)]
                    )
            if flattened:
                return flattened, f"{path}[].markets"

    if isinstance(candidate, dict):
        if _looks_like_market_object(candidate):
            return [candidate], path
        for nested_key in ("markets", "data", "items", "results"):
            nested_candidate = candidate.get(nested_key)
            nested_items, nested_path = _market_list_from_candidate(
                nested_candidate,
                f"{path}.{nested_key}",
            )
            if nested_items:
                return nested_items, nested_path

    return [], None


def extract_gamma_market_items(payload: Any) -> GammaMarketExtraction:
    """
    Extract a best-effort market list from Gamma payloads with variable shapes.

    Supported examples:
    - list[market]
    - {"markets": [...]}
    - {"data": [...]} or {"data": {"markets": [...]}}
    - {"items": [...]}, {"results": [...]}
    - {"events": [...]} with nested `markets`
    """
    if isinstance(payload, list):
        items, extraction_path = _market_list_from_candidate(payload, "root")
        return GammaMarketExtraction(
            items=items,
            top_level_type="list",
            top_level_keys=[],
            extraction_path=extraction_path or "unrecognized_list",
        )

    if isinstance(payload, dict):
        top_level_keys = sorted([str(key) for key in payload.keys()])
        for key in ("markets", "data", "items", "results", "events"):
            items, extraction_path = _market_list_from_candidate(
                payload.get(key),
                f"root.{key}",
            )
            if items:
                return GammaMarketExtraction(
                    items=items,
                    top_level_type="dict",
                    top_level_keys=top_level_keys,
                    extraction_path=extraction_path or f"root.{key}",
                )

        if _looks_like_market_object(payload):
            return GammaMarketExtraction(
                items=[payload],
                top_level_type="dict",
                top_level_keys=top_level_keys,
                extraction_path="root",
            )

        for key, value in payload.items():
            items, extraction_path = _market_list_from_candidate(value, f"root.{key}")
            if items:
                return GammaMarketExtraction(
                    items=items,
                    top_level_type="dict",
                    top_level_keys=top_level_keys,
                    extraction_path=extraction_path or f"root.{key}",
                )

        return GammaMarketExtraction(
            items=[],
            top_level_type="dict",
            top_level_keys=top_level_keys,
            extraction_path="unrecognized_dict_shape",
        )

    return GammaMarketExtraction(
        items=[],
        top_level_type=type(payload).__name__,
        top_level_keys=[],
        extraction_path="unsupported_payload_type",
    )



def _parse_outcomes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            return _parse_outcomes(parsed)
        return [stripped]
    if isinstance(value, list):
        outcomes: List[str] = []
        for item in value:
            if isinstance(item, str):
                outcomes.append(item)
            elif isinstance(item, dict):
                label = item.get("outcome") or item.get("name") or item.get("label")
                if label is not None:
                    outcomes.append(str(label))
        return outcomes
    return []



def _normalize_outcome_label(label: str) -> str:
    clean = label.strip().upper()
    if clean in {"Y", "YES", "TRUE"}:
        return "YES"
    if clean in {"N", "NO", "FALSE"}:
        return "NO"
    return clean


def _extract_clob_token_list(market: Mapping[str, Any]) -> List[str]:
    raw = (
        market.get("clobTokenIds")
        or market.get("clob_token_ids")
        or market.get("outcomeTokenIds")
        or market.get("outcome_token_ids")
        or market.get("asset_ids")
        or market.get("assets_ids")
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if value is not None]



def _extract_token_ids(market: Mapping[str, Any], outcomes: List[str]) -> Dict[str, str]:
    token_map: Dict[str, str] = {}

    tokens_raw = market.get("tokens")
    if isinstance(tokens_raw, list):
        for token in tokens_raw:
            if not isinstance(token, dict):
                continue
            token_id = token.get("token_id") or token.get("tokenId") or token.get("asset_id")
            outcome = token.get("outcome") or token.get("side") or token.get("name")
            if token_id is None or outcome is None:
                continue
            token_map[_normalize_outcome_label(str(outcome))] = str(token_id)

    yes_token = market.get("yesTokenId") or market.get("yes_token_id")
    no_token = market.get("noTokenId") or market.get("no_token_id")
    if yes_token is not None:
        token_map.setdefault("YES", str(yes_token))
    if no_token is not None:
        token_map.setdefault("NO", str(no_token))

    clob_ids = _extract_clob_token_list(market)
    if clob_ids:
        if outcomes and len(outcomes) == len(clob_ids):
            for label, token_id in zip(outcomes, clob_ids):
                token_map.setdefault(_normalize_outcome_label(label), token_id)
        elif len(clob_ids) == 2:
            # TODO: Verify position-to-outcome ordering against official docs.
            token_map.setdefault("YES", clob_ids[0])
            token_map.setdefault("NO", clob_ids[1])

    return token_map


def _extract_yes_no_token_ids(
    market: Mapping[str, Any],
    outcomes: List[str],
    token_map: Mapping[str, str],
) -> tuple[Optional[str], Optional[str]]:
    yes_token = market.get("yesTokenId") or market.get("yes_token_id")
    no_token = market.get("noTokenId") or market.get("no_token_id")

    yes = str(yes_token) if yes_token is not None else token_map.get("YES")
    no = str(no_token) if no_token is not None else token_map.get("NO")

    clob_ids = _extract_clob_token_list(market)
    if len(clob_ids) >= 2:
        yes = yes or clob_ids[0]
        no = no or clob_ids[1]

    if (yes is None or no is None) and len(outcomes) == 2:
        first_key = _normalize_outcome_label(outcomes[0])
        second_key = _normalize_outcome_label(outcomes[1])
        yes = yes or token_map.get(first_key)
        no = no or token_map.get(second_key)

    if yes == no:
        no = None
    return yes, no


def _extract_status(market: Mapping[str, Any]) -> Optional[str]:
    status = market.get("status")
    if status is not None:
        return str(status)
    if market.get("resolved") is True:
        return "resolved"
    if market.get("closed") is True:
        return "closed"
    if market.get("active") is True:
        return "open"
    return None


def _is_live_market(market: Mapping[str, Any]) -> bool:
    if market.get("active") is False:
        return False
    if market.get("closed") is True:
        return False

    status = market.get("status")
    if status is not None and str(status).strip().lower() in {
        "closed",
        "resolved",
        "finalized",
        "settled",
    }:
        return False
    return True


def _extract_event_id_from_market(market: Mapping[str, Any]) -> Optional[str]:
    direct = market.get("event_id") or market.get("eventId")
    if direct is not None:
        return str(direct)

    events = market.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id") or event.get("event_id")
            if event_id is not None:
                return str(event_id)
    return None


def _normalize_market_object(
    market_obj: Mapping[str, Any],
    *,
    event_id: Optional[str],
    category: Optional[str],
    now: Any,
    include_inactive: bool = True,
) -> Optional[MarketMetadata]:
    if not include_inactive and not _is_live_market(market_obj):
        return None

    market_id = (
        market_obj.get("id")
        or market_obj.get("market_id")
        or market_obj.get("conditionId")
        or market_obj.get("condition_id")
    )
    if market_id is None:
        return None

    outcomes = _parse_outcomes(
        market_obj.get("outcomes")
        or market_obj.get("outcomeNames")
        or market_obj.get("outcome_names")
    )
    outcome_tokens = _extract_token_ids(market_obj, outcomes)
    yes_token_id, no_token_id = _extract_yes_no_token_ids(
        market_obj,
        outcomes=outcomes,
        token_map=outcome_tokens,
    )

    question = market_obj.get("question") or market_obj.get("title")
    merged_event_id = event_id or _extract_event_id_from_market(market_obj)
    platform_status = _extract_status(market_obj)

    return MarketMetadata(
        market_id=str(market_id),
        event_id=merged_event_id,
        question=str(question) if question is not None else None,
        description=str(market_obj.get("description"))
        if market_obj.get("description") is not None
        else None,
        category=str(market_obj.get("category") or category)
        if market_obj.get("category") is not None or category is not None
        else None,
        outcomes=outcomes,
        resolution_source=str(
            market_obj.get("resolutionSource")
            or market_obj.get("resolution_source")
            or ""
        )
        or None,
        start_time=parse_timestamp(
            market_obj.get("eventStartTime")
            or market_obj.get("startTime")
            or market_obj.get("startDate")
            or market_obj.get("start_time")
            or market_obj.get("startDateIso")
        ),
        end_time=parse_timestamp(
            market_obj.get("endDate")
            or market_obj.get("end_time")
            or market_obj.get("endDateIso")
        ),
        close_time=parse_timestamp(
            market_obj.get("closeTime")
            or market_obj.get("close_time")
            or market_obj.get("closedTime")
            or market_obj.get("endDate")
        ),
        platform_status=platform_status,
        status=platform_status,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        condition_id=str(
            market_obj.get("conditionId") or market_obj.get("condition_id") or ""
        )
        or None,
        created_at=now,
        last_updated=now,
        token_ids=outcome_tokens,
        initial_volume=_safe_float(
            market_obj.get("volume")
            or market_obj.get("volumeNum")
            or market_obj.get("volume_num")
        ),
        initial_liquidity=_safe_float(
            market_obj.get("liquidity")
            or market_obj.get("liquidityNum")
            or market_obj.get("liquidity_num")
        ),
    )


def normalize_gamma_markets(payload: Any, include_inactive: bool = False) -> List[MarketMetadata]:
    """Normalize Gamma `/markets` payload into internal metadata objects."""
    now = utc_now()
    extracted = extract_gamma_market_items(payload)
    items = extracted.items

    normalized: List[MarketMetadata] = []
    for market in items:
        market_obj = _as_dict(market)
        if market_obj is None:
            continue
        metadata = _normalize_market_object(
            market_obj,
            event_id=_extract_event_id_from_market(market_obj),
            category=market_obj.get("category"),
            now=now,
            include_inactive=include_inactive,
        )
        if metadata is not None:
            normalized.append(metadata)
    return normalized


def normalize_gamma_events(payload: Any) -> List[MarketMetadata]:
    now = utc_now()

    if isinstance(payload, dict):
        events = payload.get("events")
    elif isinstance(payload, list):
        events = payload
    else:
        events = []

    normalized: List[MarketMetadata] = []

    for event in events or []:
        event_obj = _as_dict(event)
        if event_obj is None:
            continue

        event_id = event_obj.get("id") or event_obj.get("event_id")
        category = event_obj.get("category")

        markets = event_obj.get("markets")
        if not isinstance(markets, list):
            # Some payloads may return market-like objects at top level.
            markets = [event_obj]

        for market in markets:
            market_obj = _as_dict(market)
            if market_obj is None:
                continue
            metadata = _normalize_market_object(
                market_obj,
                event_id=str(event_id) if event_id is not None else None,
                category=str(category) if category is not None else None,
                now=now,
                include_inactive=False,
            )
            if metadata is not None:
                normalized.append(metadata)

    return normalized



def parse_price_levels(raw_levels: Any, side: str) -> List[PriceLevel]:
    levels: List[PriceLevel] = []

    if not isinstance(raw_levels, list):
        return levels

    for item in raw_levels:
        price = None
        size = None

        if isinstance(item, dict):
            price = _safe_float(item.get("price") or item.get("p"))
            size = _safe_float(
                item.get("size")
                or item.get("quantity")
                or item.get("amount")
                or item.get("s")
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price = _safe_float(item[0])
            size = _safe_float(item[1])

        if price is None or size is None:
            continue
        levels.append(PriceLevel(price=price, size=size))

    if side == "bid":
        levels.sort(key=lambda level: level.price, reverse=True)
    else:
        levels.sort(key=lambda level: level.price)

    return levels



def normalize_orderbook_snapshot(payload: Any, token_id: str) -> Optional[OrderBookSnapshot]:
    if not isinstance(payload, dict):
        return None

    ts = parse_timestamp(payload.get("timestamp") or payload.get("ts") or payload.get("time"))
    if ts is None:
        ts = utc_now()

    bids = parse_price_levels(payload.get("bids"), side="bid")
    asks = parse_price_levels(payload.get("asks"), side="ask")

    return OrderBookSnapshot(token_id=str(token_id), timestamp=ts, bids=bids, asks=asks)



def normalize_trade_item(payload: Mapping[str, Any], market_id: str) -> Optional[TradeRecord]:
    asset = (
        payload.get("asset")
        or payload.get("asset_id")
        or payload.get("token_id")
        or payload.get("assetId")
        or payload.get("tokenId")
    )
    base_trade_id = (
        payload.get("id")
        or payload.get("trade_id")
        or payload.get("txHash")
        or payload.get("tx_hash")
        or payload.get("transactionHash")
        or payload.get("transaction_hash")
        or payload.get("hash")
    )
    price = _safe_float(payload.get("price"))
    size = _safe_float(payload.get("size") or payload.get("amount") or payload.get("quantity"))

    if base_trade_id is None or price is None or size is None:
        return None

    if asset is not None:
        trade_id = f"{base_trade_id}:{asset}"
    else:
        trade_id = str(base_trade_id)

    ts = parse_exchange_timestamp(
        payload.get("timestamp") or payload.get("time") or payload.get("createdAt")
    )
    if ts is None:
        ts = utc_now()

    side = payload.get("side") or payload.get("taker_side") or payload.get("takerSide")

    return TradeRecord(
        timestamp=ts,
        market_id=market_id,
        trade_id=str(trade_id),
        price=price,
        size=size,
        side=str(side) if side is not None else None,
        maker=str(payload.get("maker")) if payload.get("maker") is not None else None,
        taker=str(payload.get("taker")) if payload.get("taker") is not None else None,
    )



def normalize_trade_history(
    payload: Any,
    market_id: str,
    token_ids: Optional[set[str]] = None,
    condition_id: Optional[str] = None,
) -> List[TradeRecord]:
    if isinstance(payload, dict):
        items = payload.get("trades") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    token_filter = {str(token) for token in (token_ids or set()) if token}
    condition_filter = condition_id.lower() if condition_id else None

    normalized: List[TradeRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if token_filter:
            asset = (
                item.get("asset")
                or item.get("asset_id")
                or item.get("token_id")
                or item.get("assetId")
                or item.get("tokenId")
            )
            if asset is not None and str(asset) not in token_filter:
                continue
            if asset is None and condition_filter is None:
                continue
        if condition_filter:
            cond = item.get("conditionId") or item.get("condition_id")
            if cond is not None and str(cond).lower() != condition_filter:
                continue
        trade = normalize_trade_item(item, market_id=market_id)
        if trade is not None:
            normalized.append(trade)
    return normalized



def _iter_ws_payload_objects(message_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    event_message = message_obj.get("event_message")
    if isinstance(event_message, dict):
        yield event_message
        return

    data = message_obj.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        yield data
        return

    events = message_obj.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                yield event
        return

    yield message_obj


def _iter_raw_ws_event_objects(message_obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(message_obj, list):
        for item in message_obj:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(message_obj, dict):
        return

    data = message_obj.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        yield data
        return

    events = message_obj.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                yield event
        return

    yield message_obj


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _extract_slug(obj: Mapping[str, Any]) -> Optional[str]:
    slug = obj.get("slug")
    if slug is not None:
        return str(slug)
    event_message = obj.get("event_message")
    if isinstance(event_message, Mapping):
        nested_slug = event_message.get("slug")
        if nested_slug is not None:
            return str(nested_slug)
    return None


def _extract_condition_id(obj: Mapping[str, Any]) -> Optional[str]:
    condition_id = obj.get("condition_id") or obj.get("conditionId")
    if condition_id is not None:
        return str(condition_id)
    market_value = obj.get("market")
    if market_value is not None:
        return str(market_value)
    return None


def _extract_market_id(obj: Mapping[str, Any]) -> Optional[str]:
    market_id = obj.get("market_id") or obj.get("market")
    if market_id is not None:
        return str(market_id)
    direct_id = obj.get("id")
    if direct_id is not None and obj.get("event_type") == "new_market":
        return str(direct_id)
    return None


def normalize_raw_polymarket_events(
    message: str | bytes | Dict[str, Any] | List[Dict[str, Any]],
    *,
    local_arrival_ns: int,
) -> List[RawPolymarketEventRecord]:
    """
    Convert one raw Polymarket message into append-only raw event records.

    This helper is intentionally not wired into live websocket handling in Stage 2.
    It preserves raw JSON and enough normalized identifiers/timestamps for later
    replay and parser repair.
    """
    local_arrival_iso = local_arrival_iso_from_ns(local_arrival_ns)
    try:
        if isinstance(message, bytes):
            raw_text = message.decode("utf-8")
            message_obj = json.loads(raw_text)
        elif isinstance(message, str):
            raw_text = message
            message_obj = json.loads(message)
        else:
            message_obj = message
            raw_text = _json_dumps_compact(message)
    except Exception as exc:
        if isinstance(message, bytes):
            raw_preview = message.decode("utf-8", errors="replace")
        else:
            raw_preview = str(message)
        return [
            RawPolymarketEventRecord(
                local_arrival_ns=local_arrival_ns,
                local_arrival_iso=local_arrival_iso,
                exchange_timestamp=None,
                event_type="unknown",
                market_id=None,
                condition_id=None,
                asset_id=None,
                slug=None,
                parse_status="error",
                parse_error=repr(exc),
                raw_json=_json_dumps_compact({"raw_message": raw_preview}),
            )
        ]

    records: List[RawPolymarketEventRecord] = []
    for obj in _iter_raw_ws_event_objects(message_obj):
        event_type = str(obj.get("event_type") or obj.get("type") or "unknown").lower()
        exchange_timestamp = parse_exchange_timestamp(
            obj.get("timestamp")
            or obj.get("ts")
            or obj.get("time")
            or obj.get("createdAt")
        )
        asset_id = (
            obj.get("asset_id")
            or obj.get("token_id")
            or obj.get("asset")
            or obj.get("tokenId")
            or obj.get("assetId")
        )
        records.append(
            RawPolymarketEventRecord(
                local_arrival_ns=local_arrival_ns,
                local_arrival_iso=local_arrival_iso,
                exchange_timestamp=exchange_timestamp,
                event_type=event_type,
                market_id=_extract_market_id(obj),
                condition_id=_extract_condition_id(obj),
                asset_id=str(asset_id) if asset_id is not None else None,
                slug=_extract_slug(obj),
                parse_status="ok",
                parse_error=None,
                raw_json=_json_dumps_compact(obj),
            )
        )

    if records:
        return records

    return [
        RawPolymarketEventRecord(
            local_arrival_ns=local_arrival_ns,
            local_arrival_iso=local_arrival_iso,
            exchange_timestamp=None,
            event_type="unknown",
            market_id=None,
            condition_id=None,
            asset_id=None,
            slug=None,
            parse_status="error",
            parse_error="no_event_objects",
            raw_json=raw_text,
        )
    ]


def _ws_condition_id(obj: Mapping[str, Any]) -> Optional[str]:
    return _extract_condition_id(obj)


def _ws_market_id(
    obj: Mapping[str, Any],
    mapped_market: Optional[Tuple[str, str]],
) -> Optional[str]:
    if mapped_market is not None:
        return mapped_market[0]
    return (
        str(
            obj.get("market_id")
            or obj.get("condition_id")
            or obj.get("conditionId")
            or obj.get("market")
            or ""
        )
        or None
    )


def _ws_asset_id(obj: Mapping[str, Any]) -> Optional[str]:
    asset_id = (
        obj.get("asset_id")
        or obj.get("token_id")
        or obj.get("asset")
        or obj.get("tokenId")
        or obj.get("assetId")
    )
    return str(asset_id) if asset_id is not None else None


def _normalize_ws_new_market(obj: Mapping[str, Any], timestamp: Any) -> Optional[MarketMetadata]:
    condition_id = _ws_condition_id(obj)
    market_id = (
        obj.get("market_id")
        or obj.get("condition_id")
        or obj.get("conditionId")
        or obj.get("market")
        or obj.get("id")
    )
    if market_id is None:
        return None

    market_obj = dict(obj)
    market_obj["market_id"] = str(market_id)
    market_obj.pop("id", None)
    if condition_id is not None:
        market_obj.setdefault("condition_id", condition_id)
    event_id = obj.get("event_id") or obj.get("eventId") or obj.get("id")

    metadata = _normalize_market_object(
        market_obj,
        event_id=str(event_id) if event_id is not None else None,
        category=market_obj.get("category"),
        now=timestamp,
        include_inactive=True,
    )
    if metadata is not None:
        metadata.tracking_state = "ws_discovered"
        metadata.phase = metadata.phase or "future"
        metadata.market_phase = metadata.phase
    return metadata


def _resolved_market_placeholder(
    market_id: str,
    *,
    condition_id: Optional[str],
    event_id: Optional[str],
    timestamp: Any,
) -> MarketMetadata:
    return MarketMetadata(
        market_id=market_id,
        event_id=event_id,
        question=None,
        description=None,
        category=None,
        outcomes=[],
        resolution_source=None,
        start_time=None,
        end_time=None,
        close_time=None,
        platform_status="resolved",
        phase="resolved",
        tracking_state="inactive",
        status="resolved",
        market_phase="resolved",
        condition_id=condition_id,
        created_at=timestamp,
        last_updated=timestamp,
    )



def normalize_ws_message(
    message: str | bytes | Dict[str, Any] | List[Dict[str, Any]],
    token_to_market: Mapping[str, Tuple[str, str]],
) -> NormalizedWSPayload:
    if isinstance(message, bytes):
        message_obj = json.loads(message.decode("utf-8"))
    elif isinstance(message, str):
        message_obj = json.loads(message)
    else:
        message_obj = message

    if isinstance(message_obj, list):
        candidate_objects = [item for item in message_obj if isinstance(item, dict)]
    elif isinstance(message_obj, dict):
        candidate_objects = list(_iter_ws_payload_objects(message_obj))
    else:
        return NormalizedWSPayload()

    normalized = NormalizedWSPayload()

    for obj in candidate_objects:
        event_type = str(obj.get("event_type") or obj.get("type") or "").lower()
        token_id = (
            obj.get("token_id")
            or obj.get("asset_id")
            or obj.get("asset")
            or obj.get("tokenId")
            or obj.get("assetId")
        )
        token_id_str = str(token_id) if token_id is not None else None
        timestamp = parse_exchange_timestamp(
            obj.get("timestamp") or obj.get("ts") or obj.get("time") or obj.get("createdAt")
        ) or utc_now()

        mapped_market = token_to_market.get(token_id_str or "") if token_id_str else None
        market_id = _ws_market_id(obj, mapped_market)
        condition_id = _ws_condition_id(obj)
        raw_json = _json_dumps_compact(obj)

        bids_raw = obj.get("bids")
        asks_raw = obj.get("asks")
        if bids_raw is None:
            bids_raw = obj.get("buys")
        if asks_raw is None:
            asks_raw = obj.get("sells")

        if bids_raw is not None or asks_raw is not None:
            if token_id_str is not None:
                normalized.orderbook_updates.append(
                    WSOrderBookUpdate(
                        timestamp=timestamp,
                        token_id=token_id_str,
                        bids=parse_price_levels(bids_raw, "bid"),
                        asks=parse_price_levels(asks_raw, "ask"),
                    )
                )

        if event_type in {"price_change", "price_changes"}:
            changes = obj.get("price_changes")
            if isinstance(changes, list):
                for change in changes:
                    if not isinstance(change, dict):
                        continue
                    change_token = (
                        change.get("asset_id")
                        or change.get("token_id")
                        or token_id_str
                    )
                    if change_token is None:
                        continue

                    price = _safe_float(change.get("price"))
                    size = _safe_float(change.get("size"))
                    if price is None or size is None:
                        continue

                    side_raw = str(change.get("side") or "").strip().lower()
                    if side_raw in {"buy", "bid"}:
                        book_side = "bid"
                    elif side_raw in {"sell", "ask"}:
                        book_side = "ask"
                    else:
                        continue

                    normalized.price_changes.append(
                        WSPriceChange(
                            timestamp=timestamp,
                            token_id=str(change_token),
                            book_side=book_side,
                            price=price,
                            size=size,
                        )
                    )

        if event_type == "last_trade_price":
            trade_market_id = market_id
            if token_id_str and token_id_str in token_to_market:
                trade_market_id = token_to_market[token_id_str][0]
            if trade_market_id is not None:
                trade = normalize_trade_item(obj, trade_market_id)
                if trade is not None:
                    normalized.trades.append(trade)
                    normalized.last_trade_prices.append(trade)

        if event_type == "tick_size_change":
            old_tick_size = _safe_float(
                obj.get("old_tick_size") or obj.get("oldTickSize")
            )
            new_tick_size = _safe_float(
                obj.get("new_tick_size")
                or obj.get("newTickSize")
                or obj.get("tick_size")
                or obj.get("tickSize")
            )
            if old_tick_size is not None or new_tick_size is not None:
                normalized.tick_size_changes.append(
                    TickSizeChangeRecord(
                        timestamp=timestamp,
                        market_id=market_id,
                        condition_id=condition_id,
                        asset_id=_ws_asset_id(obj),
                        old_tick_size=old_tick_size,
                        new_tick_size=new_tick_size,
                        raw_json=raw_json,
                    )
                )

        if event_type == "best_bid_ask":
            best_bid = _safe_float(obj.get("best_bid") or obj.get("bestBid"))
            best_ask = _safe_float(obj.get("best_ask") or obj.get("bestAsk"))
            spread = _safe_float(obj.get("spread"))
            if spread is None and best_bid is not None and best_ask is not None:
                spread = max(0.0, best_ask - best_bid)
            if best_bid is not None or best_ask is not None:
                normalized.best_bid_ask_updates.append(
                    BestBidAskRecord(
                        timestamp=timestamp,
                        market_id=market_id,
                        condition_id=condition_id,
                        asset_id=_ws_asset_id(obj),
                        best_bid=best_bid,
                        best_ask=best_ask,
                        spread=spread,
                        raw_json=raw_json,
                    )
                )

        if event_type == "new_market":
            metadata = _normalize_ws_new_market(obj, timestamp)
            if metadata is not None:
                normalized.new_markets.append(metadata)
                normalized.market_events.append(
                    MarketEventRecord(
                        timestamp=timestamp,
                        market_id=metadata.market_id,
                        event_type="new_market",
                        details=raw_json,
                    )
                )

        if event_type == "market_resolved" and market_id is not None:
            normalized.market_events.append(
                MarketEventRecord(
                    timestamp=timestamp,
                    market_id=market_id,
                    event_type="market_resolved",
                    details=raw_json,
                )
            )

        # Some websocket payloads emit trade arrays under `trades`.
        trade_items = obj.get("trades")
        if isinstance(trade_items, list):
            for trade_obj in trade_items:
                if not isinstance(trade_obj, dict):
                    continue
                trade_market_id = market_id
                trade_token = (
                    trade_obj.get("asset")
                    or trade_obj.get("asset_id")
                    or trade_obj.get("token_id")
                )
                if trade_token is not None:
                    mapped_trade = token_to_market.get(str(trade_token))
                    if mapped_trade is not None:
                        trade_market_id = mapped_trade[0]
                if trade_market_id is None and token_id_str and token_id_str in token_to_market:
                    trade_market_id = token_to_market[token_id_str][0]
                if trade_market_id is None:
                    continue
                trade = normalize_trade_item(trade_obj, trade_market_id)
                if trade is not None:
                    normalized.trades.append(trade)

        if event_type in {"trade", "last_trade", "fill"} or (
            event_type != "last_trade_price"
            and obj.get("price") is not None
            and obj.get("size") is not None
            and obj.get("trade_id") is not None
        ):
            trade_market_id = market_id
            if token_id_str and token_id_str in token_to_market:
                trade_market_id = token_to_market[token_id_str][0]
            if trade_market_id is not None:
                trade = normalize_trade_item(obj, trade_market_id)
                if trade is not None:
                    normalized.trades.append(trade)

        if event_type in STATUS_EVENT_TYPES:
            if market_id is not None:
                normalized.status_events.append(
                    WSStatusEvent(
                        timestamp=timestamp,
                        market_id=market_id,
                        event_type=event_type,
                        details=raw_json,
                    )
                )
        elif event_type in {"status", "market_status"} and market_id is not None:
            normalized.status_events.append(
                WSStatusEvent(
                    timestamp=timestamp,
                    market_id=market_id,
                    event_type="status_changed",
                    details=raw_json,
                )
            )

    return normalized
