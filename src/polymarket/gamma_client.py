from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from ..market_validation import (
    market_has_yes_no_token_ids,
    question_has_btc_keyword,
    question_has_up_or_down_phrase,
    validate_strict_btc_5m_market,
)
from ..models import MarketMetadata, parse_timestamp, to_iso
from .normalization import extract_gamma_market_items, normalize_gamma_markets

_QUESTION_TIME_RANGE_RE = re.compile(
    r"(?P<h1>\d{1,2}):(?P<m1>\d{2})\s*(?P<ap1>am|pm)?\s*[–-]\s*"
    r"(?P<h2>\d{1,2}):(?P<m2>\d{2})\s*(?P<ap2>am|pm)?",
    re.IGNORECASE,
)

_TERMINAL_STATUSES = {"closed", "resolved", "finalized", "settled"}
_OPEN_STATUSES = {"open", "active"}
_FIVE_MIN_EXACT_SEC = 300.0
_BTC_5M_SLUG_PREFIX = "btc-updown-5m-"
_TARGETED_SLUG_LOOKBACK_WINDOWS = 12
_TARGETED_SLUG_LOOKAHEAD_WINDOWS = 18


def _has_btc_keyword(question: Optional[str]) -> bool:
    return question_has_btc_keyword(question)


def _has_up_or_down_phrase(question: Optional[str]) -> bool:
    return question_has_up_or_down_phrase(question)


def _to_minutes(hour: int, minute: int, am_pm: Optional[str]) -> Optional[int]:
    if minute < 0 or minute > 59:
        return None
    if am_pm is None:
        if hour < 0 or hour > 23:
            return None
        return hour * 60 + minute

    if hour < 1 or hour > 12:
        return None
    normalized = am_pm.lower()
    hour_24 = hour % 12
    if normalized == "pm":
        hour_24 += 12
    return hour_24 * 60 + minute


def _duration_from_question_seconds(question: Optional[str]) -> Optional[float]:
    if not question:
        return None
    match = _QUESTION_TIME_RANGE_RE.search(question)
    if match is None:
        return None

    h1 = int(match.group("h1"))
    m1 = int(match.group("m1"))
    h2 = int(match.group("h2"))
    m2 = int(match.group("m2"))
    ap1 = match.group("ap1")
    ap2 = match.group("ap2")

    if ap1 is None and ap2 is not None:
        ap1 = ap2
    if ap2 is None and ap1 is not None:
        ap2 = ap1

    start_min = _to_minutes(h1, m1, ap1)
    end_min = _to_minutes(h2, m2, ap2)
    if start_min is None or end_min is None:
        return None
    if end_min < start_min:
        end_min += 24 * 60
    return float((end_min - start_min) * 60)



def _duration_seconds(market: MarketMetadata) -> Optional[float]:
    if market.start_time is not None and market.close_time is not None:
        return float((market.close_time - market.start_time).total_seconds())
    return _duration_from_question_seconds(market.question)


def _duration_is_strict_5m(duration_sec: Optional[float]) -> bool:
    return duration_sec is not None and abs(duration_sec - _FIVE_MIN_EXACT_SEC) <= 1e-6



def _has_token_ids(market: MarketMetadata) -> bool:
    return market_has_yes_no_token_ids(market)



def _extract_raw_market_id(raw_market: Mapping[str, Any]) -> Optional[str]:
    market_id = (
        raw_market.get("id")
        or raw_market.get("market_id")
        or raw_market.get("conditionId")
        or raw_market.get("condition_id")
    )
    if market_id is None:
        return None
    return str(market_id)



def _normalized_platform_status(market: MarketMetadata, raw_market: Optional[Mapping[str, Any]]) -> str:
    status = (market.platform_status or market.status or "").strip().lower()
    if status:
        return status
    if raw_market is None:
        return status
    raw_status = raw_market.get("status")
    if raw_status is not None:
        return str(raw_status).strip().lower()
    if raw_market.get("closed") is True:
        return "closed"
    if raw_market.get("active") is True:
        return "open"
    return status



def _is_platform_open(market: MarketMetadata, raw_market: Optional[Mapping[str, Any]]) -> bool:
    status = _normalized_platform_status(market, raw_market)
    if status in _TERMINAL_STATUSES:
        return False
    if raw_market is not None and raw_market.get("closed") is True:
        return False
    if raw_market is not None and raw_market.get("active") is True:
        return True
    return status in _OPEN_STATUSES



def _effective_start_time(market: MarketMetadata, raw_market: Optional[Mapping[str, Any]]) -> Optional[datetime]:
    if raw_market is not None:
        for key in ("eventStartTime", "startTime", "marketStartTime", "startDate"):
            parsed = parse_timestamp(raw_market.get(key))
            if parsed is not None:
                return parsed
    return market.start_time



def _effective_end_time(market: MarketMetadata, raw_market: Optional[Mapping[str, Any]]) -> Optional[datetime]:
    if raw_market is not None:
        for key in ("closeTime", "closedTime", "endDate", "marketEndTime"):
            parsed = parse_timestamp(raw_market.get(key))
            if parsed is not None:
                return parsed
    return market.close_time or market.end_time


def _effective_duration_seconds(
    market: MarketMetadata,
    raw_market: Optional[Mapping[str, Any]],
) -> Optional[float]:
    start = _effective_start_time(market, raw_market)
    end = _effective_end_time(market, raw_market)
    if start is not None and end is not None:
        return float((end - start).total_seconds())
    return _duration_from_question_seconds(market.question)



def _window_distance_seconds(
    market: MarketMetadata,
    now: datetime,
    raw_market: Optional[Mapping[str, Any]] = None,
) -> float:
    start = _effective_start_time(market, raw_market)
    end = _effective_end_time(market, raw_market)

    if start is not None and end is not None:
        if start <= now < end:
            return 0.0
        if now < start:
            return (start - now).total_seconds()
        return (now - end).total_seconds()
    if start is not None:
        return (start - now).total_seconds() if now < start else 0.0
    if end is not None:
        return (now - end).total_seconds() if now >= end else 0.0
    return 0.0



def _is_within_tracking_window(
    market: MarketMetadata,
    now: datetime,
    lookahead_sec: int,
    raw_market: Optional[Mapping[str, Any]] = None,
) -> bool:
    if lookahead_sec <= 0:
        return True
    start = _effective_start_time(market, raw_market)
    if start is None:
        return True
    return start <= (now + timedelta(seconds=lookahead_sec))



def _extract_raw_time_fields(raw_market: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if raw_market is None:
        return {}

    explicit = [
        "eventStartTime",
        "startTime",
        "startDate",
        "closeTime",
        "closedTime",
        "endDate",
        "acceptingOrdersTimestamp",
        "createdAt",
        "updatedAt",
        "endDateIso",
        "startDateIso",
        "deployingTimestamp",
    ]
    out: Dict[str, Any] = {}
    for key in explicit:
        if key in raw_market:
            out[key] = raw_market.get(key)

    # Include additional date/time-like fields for observability if present.
    extra_keys = [
        key
        for key in raw_market.keys()
        if key not in out and ("date" in key.lower() or "time" in key.lower())
    ]
    for key in sorted(extra_keys)[:12]:
        out[key] = raw_market.get(key)
    return out



def _derive_phase_for_discovery(
    market: MarketMetadata,
    now: datetime,
    raw_market: Optional[Mapping[str, Any]],
) -> tuple[str, str]:
    status = _normalized_platform_status(market, raw_market)
    if status in _TERMINAL_STATUSES:
        return "resolved", "terminal_platform_status"
    if raw_market is not None and raw_market.get("closed") is True:
        return "resolved", "raw_closed_flag"

    start = _effective_start_time(market, raw_market)
    end = _effective_end_time(market, raw_market)

    if start is not None and now < start:
        return "future", "before_effective_start_time"

    if end is not None and now >= end:
        return "expired_waiting_resolution", "after_effective_end_time"

    return "active", "within_effective_window_or_missing_bounds"



def _is_target_market(market: MarketMetadata) -> bool:
    return validate_strict_btc_5m_market(market).is_valid



def _dedupe_markets(markets: Iterable[MarketMetadata]) -> List[MarketMetadata]:
    out: List[MarketMetadata] = []
    seen: set[str] = set()
    for market in markets:
        if market.market_id in seen:
            continue
        seen.add(market.market_id)
        out.append(market)
    return out



def _select_nearest_candidate(
    markets: Sequence[MarketMetadata],
    now: datetime,
    raw_by_market_id: Mapping[str, Mapping[str, Any]],
) -> Optional[MarketMetadata]:
    if not markets:
        return None
    return min(
        markets,
        key=lambda market: (
            0 if _is_platform_open(market, raw_by_market_id.get(market.market_id)) else 1,
            _window_distance_seconds(
                market,
                now,
                raw_by_market_id.get(market.market_id),
            ),
            market.close_time or market.start_time or now,
        ),
    )



def _targeted_slug_epochs(now: datetime) -> list[int]:
    now_epoch = int(now.timestamp())
    aligned_epoch = (now_epoch // 300) * 300
    start_epoch = aligned_epoch - (_TARGETED_SLUG_LOOKBACK_WINDOWS * 300)
    end_epoch = aligned_epoch + (_TARGETED_SLUG_LOOKAHEAD_WINDOWS * 300)
    return list(range(start_epoch, end_epoch + 1, 300))


class GammaClient:
    """Client for Polymarket Gamma market discovery API."""

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 10.0,
        page_size: int = 500,
        max_pages: int = 60,
        lookahead_sec: int = 7200,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_sec)
        self._page_size = page_size
        self._max_pages = max_pages
        self._lookahead_sec = lookahead_sec
        self.last_discovery_stats: dict[str, int] = {
            "total_markets_from_api": 0,
            "selected_btc_5m_markets": 0,
            "pages_fetched": 0,
        }
        self.last_discovery_debug: dict[str, Any] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_active_markets(self) -> List[MarketMetadata]:
        """Discover markets and return candidate pool for tracking/metadata."""
        selected, report = await self._discover_with_report()
        self.last_discovery_debug = report
        return selected

    async def debug_discovery_once(self) -> dict[str, Any]:
        """Run one discovery cycle and return full debug diagnostics."""
        _, report = await self._discover_with_report()
        self.last_discovery_debug = report
        return report

    async def fetch_market_by_id(
        self,
        market_id: str,
        *,
        condition_id: str | None = None,
    ) -> Any:
        """Fetch one market payload for explicit resolution refresh."""
        identifiers = [str(value) for value in (market_id, condition_id) if value]
        last_error: Exception | None = None
        for identifier in identifiers:
            try:
                resp = await self._client.get(f"/markets/{identifier}")
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                last_error = exc
            try:
                resp = await self._client.get(
                    "/markets",
                    params={"id": identifier, "limit": "1"},
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return None

    async def _fetch_targeted_slug_markets(
        self,
        now: datetime,
    ) -> tuple[
        List[MarketMetadata],
        List[dict[str, Any]],
        List[dict[str, Any]],
        dict[str, Mapping[str, Any]],
    ]:
        discovered: List[MarketMetadata] = []
        raw_samples: List[dict[str, Any]] = []
        page_logs: List[dict[str, Any]] = []
        raw_by_market_id: dict[str, Mapping[str, Any]] = {}

        slug_epochs = _targeted_slug_epochs(now)
        for idx, epoch in enumerate(slug_epochs):
            slug = f"{_BTC_5M_SLUG_PREFIX}{epoch}"
            params = {
                "active": "true",
                "closed": "false",
                "limit": "5",
                "slug": slug,
            }
            request_url = str(
                self._client.build_request("GET", "/markets", params=params).url
            )
            try:
                resp = await self._client.get("/markets", params=params)
            except Exception as exc:
                self._log.warning(
                    "targeted_slug_discovery_http_error",
                    extra={
                        "request_url": request_url,
                        "slug": slug,
                        "epoch": epoch,
                        "error": repr(exc),
                    },
                )
                continue

            response_size_bytes = len(resp.content)
            if resp.status_code >= 400:
                self._log.warning(
                    "targeted_slug_discovery_http_non_200",
                    extra={
                        "request_url": request_url,
                        "slug": slug,
                        "epoch": epoch,
                        "status_code": resp.status_code,
                        "response_size_bytes": response_size_bytes,
                        "response_prefix": resp.text[:250],
                    },
                )
                continue

            try:
                raw_payload = resp.json()
            except Exception as exc:
                self._log.warning(
                    "targeted_slug_discovery_json_parse_failed",
                    extra={
                        "request_url": request_url,
                        "slug": slug,
                        "epoch": epoch,
                        "status_code": resp.status_code,
                        "response_size_bytes": response_size_bytes,
                        "error": repr(exc),
                        "response_prefix": resp.text[:250],
                    },
                )
                continue

            extraction = extract_gamma_market_items(raw_payload)
            page_markets = normalize_gamma_markets(raw_payload, include_inactive=True)
            discovered.extend(page_markets)

            for raw_item in extraction.items:
                if not isinstance(raw_item, Mapping):
                    continue
                market_id = _extract_raw_market_id(raw_item)
                if market_id is None:
                    continue
                raw_by_market_id[market_id] = raw_item
                if len(raw_samples) < 10:
                    raw_samples.append(
                        {
                            "market_id": market_id,
                            "question": raw_item.get("question") or raw_item.get("title"),
                            "active": raw_item.get("active"),
                            "closed": raw_item.get("closed"),
                            "status": raw_item.get("status"),
                            "raw_time_fields": _extract_raw_time_fields(raw_item),
                            "discovery_slug": slug,
                            "discovery_epoch": epoch,
                        }
                    )

            page_info = {
                "mode": "targeted_slug_scan",
                "query_index": idx,
                "slug": slug,
                "epoch": epoch,
                "request_url": request_url,
                "status_code": resp.status_code,
                "response_size_bytes": response_size_bytes,
                "top_level_type": extraction.top_level_type,
                "top_level_keys": extraction.top_level_keys[:20],
                "extraction_path": extraction.extraction_path,
                "raw_item_count": len(extraction.items),
                "normalized_market_count": len(page_markets),
            }
            page_logs.append(page_info)
            self._log.info("discovery_http_page", extra=page_info)

        deduped = _dedupe_markets(discovered)
        self._log.info(
            "targeted_slug_discovery_summary",
            extra={
                "mode": "targeted_slug_scan",
                "now_utc": to_iso(now),
                "query_count": len(slug_epochs),
                "matched_market_count": len(deduped),
                "earliest_epoch": min(slug_epochs) if slug_epochs else None,
                "latest_epoch": max(slug_epochs) if slug_epochs else None,
            },
        )
        return deduped, raw_samples, page_logs, raw_by_market_id

    async def _fetch_broad_markets(
        self,
    ) -> tuple[
        List[MarketMetadata],
        List[dict[str, Any]],
        List[dict[str, Any]],
        dict[str, Mapping[str, Any]],
    ]:
        discovered: List[MarketMetadata] = []
        raw_samples: List[dict[str, Any]] = []
        page_logs: List[dict[str, Any]] = []
        raw_by_market_id: dict[str, Mapping[str, Any]] = {}

        offset = 0
        for page_index in range(self._max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": str(self._page_size),
                "offset": str(offset),
            }
            request_url = str(
                self._client.build_request("GET", "/markets", params=params).url
            )

            try:
                resp = await self._client.get("/markets", params=params)
            except Exception as exc:
                self._log.warning(
                    "discovery_http_error",
                    extra={
                        "page_index": page_index,
                        "request_url": request_url,
                        "error": repr(exc),
                    },
                )
                if page_index == 0:
                    raise
                break

            response_size_bytes = len(resp.content)
            if resp.status_code >= 400:
                self._log.warning(
                    "discovery_http_non_200",
                    extra={
                        "page_index": page_index,
                        "request_url": request_url,
                        "status_code": resp.status_code,
                        "response_size_bytes": response_size_bytes,
                        "response_prefix": resp.text[:250],
                    },
                )
                if page_index == 0:
                    resp.raise_for_status()
                break

            try:
                raw_payload = resp.json()
            except Exception as exc:
                self._log.warning(
                    "discovery_json_parse_failed",
                    extra={
                        "page_index": page_index,
                        "request_url": request_url,
                        "status_code": resp.status_code,
                        "response_size_bytes": response_size_bytes,
                        "error": repr(exc),
                        "response_prefix": resp.text[:250],
                    },
                )
                if page_index == 0:
                    raise
                break

            extraction = extract_gamma_market_items(raw_payload)
            page_markets = normalize_gamma_markets(raw_payload, include_inactive=True)
            discovered.extend(page_markets)

            for raw_item in extraction.items:
                if not isinstance(raw_item, Mapping):
                    continue
                market_id = _extract_raw_market_id(raw_item)
                if market_id is None:
                    continue
                raw_by_market_id[market_id] = raw_item
                if len(raw_samples) < 10:
                    raw_samples.append(
                        {
                            "market_id": market_id,
                            "question": raw_item.get("question") or raw_item.get("title"),
                            "active": raw_item.get("active"),
                            "closed": raw_item.get("closed"),
                            "status": raw_item.get("status"),
                            "raw_time_fields": _extract_raw_time_fields(raw_item),
                        }
                    )

            page_info = {
                "mode": "broad_scan",
                "page_index": page_index,
                "request_url": request_url,
                "status_code": resp.status_code,
                "response_size_bytes": response_size_bytes,
                "top_level_type": extraction.top_level_type,
                "top_level_keys": extraction.top_level_keys[:20],
                "extraction_path": extraction.extraction_path,
                "raw_item_count": len(extraction.items),
                "normalized_market_count": len(page_markets),
            }
            page_logs.append(page_info)
            self._log.info("discovery_http_page", extra=page_info)

            if not isinstance(raw_payload, list):
                break
            if len(raw_payload) < self._page_size:
                break
            offset += self._page_size
        else:
            self._log.warning(
                "market_discovery_truncated",
                extra={"page_size": self._page_size, "max_pages": self._max_pages},
            )

        return discovered, raw_samples, page_logs, raw_by_market_id

    async def _discover_with_report(self) -> tuple[List[MarketMetadata], dict[str, Any]]:
        now = datetime.now(timezone.utc)
        discovered, raw_samples, page_logs, raw_by_market_id = (
            await self._fetch_targeted_slug_markets(now)
        )
        discovery_source = "targeted_slug_scan"
        if not discovered:
            self._log.warning(
                "targeted_slug_discovery_empty_fallback_to_broad_scan",
                extra={
                    "now_utc": to_iso(now),
                    "slug_prefix": _BTC_5M_SLUG_PREFIX,
                    "lookback_windows": _TARGETED_SLUG_LOOKBACK_WINDOWS,
                    "lookahead_windows": _TARGETED_SLUG_LOOKAHEAD_WINDOWS,
                },
            )
            discovered, raw_samples, page_logs, raw_by_market_id = (
                await self._fetch_broad_markets()
            )
            discovery_source = "broad_scan_fallback"
        fetched_pages = len(page_logs)

        stage_counts = {
            "raw_markets_returned": len(discovered),
            "matching_btc_bitcoin": 0,
            "matching_up_or_down": 0,
            "matching_approx_5m_duration": 0,
            "with_token_ids_present": 0,
            "after_phase_classification": 0,
            "final_selected": 0,
        }

        reasons = Counter()
        phase_counts = Counter()
        strict_candidates: List[MarketMetadata] = []
        strict_phase_eligible: List[MarketMetadata] = []
        broad_candidates: List[MarketMetadata] = []
        broad_phase_eligible: List[MarketMetadata] = []

        candidate_details: List[dict[str, Any]] = []
        strict_matched_candidates: List[dict[str, Any]] = []

        for market in discovered:
            raw_market = raw_by_market_id.get(market.market_id)
            strict_validation = validate_strict_btc_5m_market(market)
            normalized = strict_validation.normalized_fields
            has_btc = bool(normalized.get("has_btc_keyword", False))
            has_up_or_down = bool(normalized.get("has_up_or_down_phrase", False))
            effective_duration = _effective_duration_seconds(market, raw_market)
            canonical_duration = normalized.get("duration_sec")
            has_duration = _duration_is_strict_5m(
                float(canonical_duration) if canonical_duration is not None else None
            )
            has_tokens = bool(normalized.get("has_token_ids", False))
            computed_duration = _duration_seconds(market)
            phase, phase_reason = _derive_phase_for_discovery(
                market,
                now,
                raw_market,
            )
            within_tracking_window = _is_within_tracking_window(
                market,
                now,
                self._lookahead_sec,
                raw_market,
            )
            if not strict_validation.is_valid:
                rejection_reason = strict_validation.reason or "strict_unknown"
            elif phase not in {"active", "future"}:
                rejection_reason = f"phase_{phase}"
            elif not within_tracking_window:
                rejection_reason = "outside_tracking_window"
            else:
                rejection_reason = "selected"
            selected_strict = (
                strict_validation.is_valid
                and phase in {"active", "future"}
                and within_tracking_window
            )
            selected_broad = (
                has_btc
                and has_up_or_down
                and has_tokens
                and not has_duration
                and phase in {"active", "future"}
                and within_tracking_window
            )

            if has_btc:
                stage_counts["matching_btc_bitcoin"] += 1
            if has_btc and has_up_or_down:
                stage_counts["matching_up_or_down"] += 1
            if has_btc and has_up_or_down and has_duration:
                stage_counts["matching_approx_5m_duration"] += 1
            if has_btc and has_up_or_down and has_duration and has_tokens:
                stage_counts["with_token_ids_present"] += 1
            if strict_validation.is_valid:
                strict_candidates.append(market)

            if selected_strict:
                stage_counts["after_phase_classification"] += 1
                stage_counts["final_selected"] += 1
                strict_phase_eligible.append(market)

            if has_btc and has_up_or_down and has_tokens and not has_duration:
                broad_candidates.append(market)
            if selected_broad:
                broad_phase_eligible.append(market)

            phase_counts[phase] += 1
            reasons[rejection_reason] += 1

            detail = {
                "market_id": market.market_id,
                "question": market.question,
                "start_time": to_iso(market.start_time),
                "end_time": to_iso(market.end_time),
                "close_time": to_iso(market.close_time),
                "event_start_time_raw": raw_market.get("eventStartTime")
                if raw_market is not None
                else None,
                "start_date_raw": raw_market.get("startDate")
                if raw_market is not None
                else None,
                "end_date_raw": raw_market.get("endDate")
                if raw_market is not None
                else None,
                "close_time_raw": raw_market.get("closeTime")
                if raw_market is not None
                else None,
                "accepting_orders_timestamp_raw": raw_market.get("acceptingOrdersTimestamp")
                if raw_market is not None
                else None,
                "created_at_raw": raw_market.get("createdAt")
                if raw_market is not None
                else None,
                "platform_status": market.platform_status or market.status,
                "active_flag": raw_market.get("active") if raw_market is not None else None,
                "closed_flag": raw_market.get("closed") if raw_market is not None else None,
                "accepting_orders": raw_market.get("acceptingOrders")
                if raw_market is not None
                else None,
                "raw_time_fields": _extract_raw_time_fields(raw_market),
                "now_utc": to_iso(now),
                "computed_duration_sec": computed_duration,
                "effective_duration_sec": effective_duration,
                "canonical_duration_sec": canonical_duration,
                "computed_phase": phase,
                "phase_reason": phase_reason,
                "within_tracking_window": within_tracking_window,
                "strict_selection_eligible": selected_strict,
                "selected_broad": selected_broad,
                "strict_rejection_reason": None if selected_strict else rejection_reason,
                "strict_validator_reason": strict_validation.reason,
            }

            if len(candidate_details) < 10:
                candidate_details.append(detail)
                self._log.info("discovery_candidate_evaluation", extra=detail)

            if strict_validation.is_valid:
                strict_matched_candidates.append(detail)
                self._log.info("discovery_strict_btc_candidate", extra=detail)
            elif (
                has_btc
                and has_up_or_down
                and has_tokens
                and not has_duration
            ):
                self._log.info("discovery_non_5m_btc_candidate_rejected", extra=detail)

        fallback_mode = "none"
        fallback_reason = None
        candidate_pool = sorted(
            _dedupe_markets(strict_candidates),
            key=lambda market: (
                _window_distance_seconds(
                    market,
                    now,
                    raw_by_market_id.get(market.market_id),
                ),
                market.close_time or market.start_time or now,
            ),
        )

        selected_preview_market = _select_nearest_candidate(
            strict_phase_eligible,
            now,
            raw_by_market_id,
        )
        selected_preview = (
            {
                "market_id": selected_preview_market.market_id,
                "question": selected_preview_market.question,
                "start_time": to_iso(selected_preview_market.start_time),
                "close_time": to_iso(selected_preview_market.close_time),
                "effective_duration_sec": _effective_duration_seconds(
                    selected_preview_market,
                    raw_by_market_id.get(selected_preview_market.market_id),
                ),
                "window_distance_sec": _window_distance_seconds(
                    selected_preview_market,
                    now,
                    raw_by_market_id.get(selected_preview_market.market_id),
                ),
                "platform_open": _is_platform_open(
                    selected_preview_market,
                    raw_by_market_id.get(selected_preview_market.market_id),
                ),
            }
            if selected_preview_market is not None
            else None
        )

        summary_extra = {
            "now_utc": to_iso(now),
            "discovery_source": discovery_source,
            "pages_fetched": fetched_pages,
            "response_page_count": len(page_logs),
            "raw_markets_returned": stage_counts["raw_markets_returned"],
            "matching_btc_bitcoin": stage_counts["matching_btc_bitcoin"],
            "matching_up_or_down": stage_counts["matching_up_or_down"],
            "matching_approx_5m_duration": stage_counts["matching_approx_5m_duration"],
            "with_token_ids_present": stage_counts["with_token_ids_present"],
            "after_phase_classification": stage_counts["after_phase_classification"],
            "final_selected": stage_counts["final_selected"],
            "phase_counts": dict(phase_counts),
            "rejection_counts": dict(reasons),
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "strict_candidate_count": len(strict_candidates),
            "strict_phase_eligible_count": len(strict_phase_eligible),
            "broad_candidate_count": len(_dedupe_markets(broad_candidates)),
            "broad_phase_eligible_count": len(_dedupe_markets(broad_phase_eligible)),
            "candidate_pool_count": len(candidate_pool),
            "selected_market_preview": selected_preview,
        }
        self._log.info("discovered_markets", extra=summary_extra)

        if not candidate_pool:
            self._log.warning("discovery_zero_markets_after_filtering", extra=summary_extra)
        elif not strict_phase_eligible:
            self._log.warning(
                "discovery_no_phase_eligible_5m_candidate",
                extra=summary_extra,
            )

        self.last_discovery_stats = {
            "total_markets_from_api": stage_counts["raw_markets_returned"],
            "selected_btc_5m_markets": len(strict_candidates),
            "phase_eligible_markets": len(strict_phase_eligible),
            "broad_btc_candidates": len(_dedupe_markets(broad_candidates)),
            "fallback_used": 0,
            "candidate_pool_count": len(candidate_pool),
            "pages_fetched": fetched_pages,
            "discovery_source": 1 if discovery_source == "targeted_slug_scan" else 0,
        }

        report: dict[str, Any] = {
            "now_utc": to_iso(now),
            "discovery_source": discovery_source,
            "pages": page_logs,
            "raw_samples": raw_samples,
            "candidate_details": candidate_details,
            "strict_matched_candidates": strict_matched_candidates,
            "strict_candidate_count": len(strict_candidates),
            "strict_phase_eligible_count": len(strict_phase_eligible),
            "broad_candidate_count": len(_dedupe_markets(broad_candidates)),
            "broad_phase_eligible_count": len(_dedupe_markets(broad_phase_eligible)),
            "candidate_pool_count": len(candidate_pool),
            "stage_counts": stage_counts,
            "phase_counts": dict(phase_counts),
            "rejection_counts": dict(reasons),
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "selected_market_preview": selected_preview,
            "candidate_market_ids": [market.market_id for market in strict_candidates],
            "phase_eligible_market_ids": [
                market.market_id for market in strict_phase_eligible
            ],
            "candidate_pool_market_ids": [market.market_id for market in candidate_pool],
            "selected_market_ids": [
                market.market_id for market in strict_phase_eligible
            ],
        }
        return candidate_pool, report
