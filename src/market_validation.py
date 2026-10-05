from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import MarketMetadata


STRICT_BTC_5M_DURATION_SEC = 300.0
_BTC_KEYWORD_RE = re.compile(r"(?i)\b(?:btc|bitcoin)\b")
_UP_OR_DOWN_RE = re.compile(r"(?i)\bup\s+or\s+down\b")


@dataclass(slots=True, frozen=True)
class StrictMarketValidationResult:
    is_valid: bool
    reason: Optional[str]
    normalized_fields: dict[str, object]


def question_has_btc_keyword(question: Optional[str]) -> bool:
    if not question:
        return False
    return _BTC_KEYWORD_RE.search(question) is not None


def question_has_up_or_down_phrase(question: Optional[str]) -> bool:
    if not question:
        return False
    return _UP_OR_DOWN_RE.search(question) is not None


def question_matches_strict_btc_up_or_down(question: Optional[str]) -> bool:
    return question_has_btc_keyword(question) and question_has_up_or_down_phrase(question)


def market_has_yes_no_token_ids(market: MarketMetadata) -> bool:
    explicit = bool(market.yes_token_id and market.no_token_id)
    fallback = bool(market.token_ids.get("YES") and market.token_ids.get("NO"))
    return explicit or fallback


def _duration_rejection_reason(duration_sec: float) -> str:
    rounded = int(round(duration_sec))
    if abs(duration_sec - rounded) <= 1e-6:
        return f"invalid_duration_{rounded}s"
    return f"invalid_duration_{duration_sec:.3f}s"


def validate_strict_btc_5m_market(
    market: MarketMetadata,
) -> StrictMarketValidationResult:
    duration_sec: Optional[float] = None
    if market.start_time is not None and market.close_time is not None:
        duration_sec = float((market.close_time - market.start_time).total_seconds())

    has_btc_keyword = question_has_btc_keyword(market.question)
    has_up_or_down_phrase = question_has_up_or_down_phrase(market.question)
    matches_question_pattern = has_btc_keyword and has_up_or_down_phrase
    has_token_ids = market_has_yes_no_token_ids(market)

    reason: Optional[str] = None
    if market.start_time is None or market.close_time is None:
        reason = "missing_start_or_close_time"
    elif duration_sec is None or abs(duration_sec - STRICT_BTC_5M_DURATION_SEC) > 1e-6:
        reason = _duration_rejection_reason(duration_sec or 0.0)
    elif not has_token_ids:
        reason = "missing_token_ids"
    elif not matches_question_pattern:
        reason = "question_pattern_mismatch"

    normalized_fields: dict[str, object] = {
        "market_id": market.market_id,
        "start_time": market.start_time,
        "close_time": market.close_time,
        "duration_sec": duration_sec,
        "has_token_ids": has_token_ids,
        "has_btc_keyword": has_btc_keyword,
        "has_up_or_down_phrase": has_up_or_down_phrase,
        "matches_question_pattern": matches_question_pattern,
        "question": market.question,
    }
    return StrictMarketValidationResult(
        is_valid=reason is None,
        reason=reason,
        normalized_fields=normalized_fields,
    )
