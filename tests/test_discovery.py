from __future__ import annotations

from datetime import datetime, timezone

from src.market_validation import validate_strict_btc_5m_market
from src.models import MarketMetadata
from src.polymarket.gamma_client import _is_target_market
from src.recorder import classify_market_phase, select_primary_market



def _market(question: str, start: datetime, close: datetime) -> MarketMetadata:
    return MarketMetadata(
        market_id="m1",
        event_id="e1",
        question=question,
        description=None,
        category="crypto",
        outcomes=["Yes", "No"],
        resolution_source=None,
        start_time=start,
        end_time=close,
        close_time=close,
        status="open",
        token_ids={"YES": "y", "NO": "n"},
    )


def test_target_market_filter_matches_btc_up_or_down_and_5m() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close_5m = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    close_10m = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)

    assert _is_target_market(_market("BTC Up or Down - 05:00-05:05", start, close_5m))
    assert _is_target_market(
        _market("Bitcoin Up or Down - January 1, 05:00AM-05:05AM ET", start, close_5m)
    )
    assert not _is_target_market(_market("ETH Up or Down - 05:00-05:05", start, close_5m))
    assert not _is_target_market(_market("BTC price target", start, close_5m))
    assert not _is_target_market(_market("BTC Up or Down - long window", start, close_10m))


def test_target_market_filter_requires_exact_duration_from_time_bounds() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    close_long = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)

    assert not _is_target_market(
        _market("BTC Up or Down - January 1, 05:00AM-05:05AM ET", start, close_long)
    )


def test_shared_strict_validator_matches_bitcoin_questions() -> None:
    market = _market(
        "Bitcoin Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    result = validate_strict_btc_5m_market(market)
    assert result.is_valid is True
    assert result.reason is None


def test_discovery_selection_consistency_no_question_pattern_mismatch_for_bitcoin() -> None:
    now = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    market = _market(
        "Bitcoin Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    market.market_id = "btc_bitcoin"

    result = validate_strict_btc_5m_market(market)
    assert result.is_valid is True

    selected, state, _tracking, _phases, _skipped, selection_reasons = select_primary_market(
        [market],
        now=now,
    )
    assert selected is not None
    assert state == "selected_active"
    assert "question_pattern_mismatch" not in selection_reasons["btc_bitcoin"]


def test_select_primary_market_prefers_active_with_nearest_close() -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    active_1 = _market(
        "BTC Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    active_1.market_id = "active_1"
    active_2 = _market(
        "BTC Up or Down - 00:00-00:04",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
    )
    active_2.market_id = "active_2"
    upcoming = _market(
        "BTC Up or Down - 00:05-00:10",
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
    )
    upcoming.market_id = "upcoming"

    selected, state, tracking, phases, skipped, selection_reasons = select_primary_market(
        [active_1, active_2, upcoming], now=now
    )

    assert selected is not None
    assert selected.market_id == "active_1"
    assert state == "selected_active"
    assert tracking["active_1"] == "selected_active"
    assert tracking["active_2"] == "inactive"
    assert phases["active_2"] == "active"
    assert phases["upcoming"] == "future"
    assert skipped["strict_rejected_invalid_duration_240s"] == 1
    assert skipped["future_not_selected"] == 1
    assert selection_reasons["active_2"] == "strict_rejected_invalid_duration_240s"


def test_select_primary_market_uses_upcoming_when_none_active() -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    upcoming_1 = _market(
        "BTC Up or Down - 00:05-00:10",
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
    )
    upcoming_1.market_id = "upcoming_1"
    upcoming_2 = _market(
        "BTC Up or Down - 00:10-00:15",
        datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
    )
    upcoming_2.market_id = "upcoming_2"

    selected, state, tracking, phases, skipped, selection_reasons = select_primary_market(
        [upcoming_2, upcoming_1], now=now
    )

    assert selected is not None
    assert selected.market_id == "upcoming_1"
    assert state == "future_standby"
    assert tracking["upcoming_1"] == "discovered"
    assert tracking["upcoming_2"] == "discovered"
    assert phases["upcoming_1"] == "future"
    assert skipped["future_not_selected"] == 1
    assert selection_reasons["upcoming_1"] == "selected_future_standby"


def test_classify_market_phase_uses_expired_waiting_resolution_when_not_terminal() -> None:
    now = datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc)
    market = _market(
        "BTC Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    market.status = "open"

    assert classify_market_phase(market, now) == "expired_waiting_resolution"


def test_select_primary_market_does_not_select_when_all_candidates_expired() -> None:
    now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    older = _market(
        "BTC Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    older.market_id = "older"
    older.status = "open"
    older.platform_status = "open"
    newer = _market(
        "BTC Up or Down - 00:50-00:55",
        datetime(2026, 1, 1, 0, 50, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 55, tzinfo=timezone.utc),
    )
    newer.market_id = "newer"
    newer.status = "open"
    newer.platform_status = "open"

    selected, state, tracking, phases, skipped, selection_reasons = select_primary_market(
        [older, newer],
        now=now,
    )

    assert selected is None
    assert state is None
    assert tracking["newer"] == "inactive"
    assert tracking["older"] == "inactive"
    assert phases["newer"] == "expired_waiting_resolution"
    assert skipped["resolved_not_selected"] == 2
    assert selection_reasons["newer"] == "phase_expired_waiting_resolution"


def test_select_primary_market_rejects_non_5m_selected_invariant() -> None:
    now = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
    hourly = _market(
        "BTC Up or Down - 00:00-01:00",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
    )
    hourly.market_id = "hourly"

    selected, state, tracking, phases, skipped, selection_reasons = select_primary_market(
        [hourly],
        now=now,
    )

    assert selected is None
    assert state is None
    assert tracking["hourly"] == "inactive"
    assert phases["hourly"] == "active"
    assert skipped["strict_rejected_invalid_duration_3600s"] == 1
    assert selection_reasons["hourly"] == "strict_rejected_invalid_duration_3600s"


def test_select_primary_market_picks_exactly_one_active_strict_market() -> None:
    now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    active = _market(
        "Bitcoin Up or Down - 00:00-00:05",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    active.market_id = "active"
    future = _market(
        "Bitcoin Up or Down - 00:05-00:10",
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
    )
    future.market_id = "future"

    selected, state, tracking, _phases, _skipped, selection_reasons = select_primary_market(
        [active, future],
        now=now,
    )

    assert selected is not None
    assert selected.market_id == "active"
    assert state == "selected_active"
    assert tracking["active"] == "selected_active"
    assert selection_reasons["active"] == "selected_active"
