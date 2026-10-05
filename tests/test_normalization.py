from __future__ import annotations

import json

from src.polymarket.normalization import (
    extract_gamma_market_items,
    normalize_gamma_events,
    normalize_gamma_markets,
    normalize_orderbook_snapshot,
    normalize_trade_history,
    normalize_ws_message,
)


def test_normalize_gamma_events_extracts_market_and_tokens() -> None:
    payload = [
        {
            "id": "event-1",
            "category": "crypto",
            "markets": [
                {
                    "id": "market-1",
                    "question": "Will BTC be above $100k?",
                    "description": "Test",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": "[\"tok_yes\", \"tok_no\"]",
                    "status": "open",
                    "startDate": "2026-01-01T00:00:00Z",
                    "endDate": "2026-02-01T00:00:00Z",
                    "closeTime": "2026-01-31T23:00:00Z",
                }
            ],
        }
    ]

    markets = normalize_gamma_events(payload)
    assert len(markets) == 1

    market = markets[0]
    assert market.market_id == "market-1"
    assert market.event_id == "event-1"
    assert market.category == "crypto"
    assert market.outcomes == ["Yes", "No"]
    assert market.token_ids["YES"] == "tok_yes"
    assert market.token_ids["NO"] == "tok_no"


def test_normalize_gamma_events_skips_closed_or_inactive_markets() -> None:
    payload = [
        {
            "id": "event-1",
            "markets": [
                {
                    "id": "open-market",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": "[\"a\", \"b\"]",
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "closed-market",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": "[\"c\", \"d\"]",
                    "active": True,
                    "closed": True,
                },
                {
                    "id": "inactive-market",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": "[\"e\", \"f\"]",
                    "active": False,
                    "closed": False,
                },
            ],
        }
    ]

    markets = normalize_gamma_events(payload)
    assert [market.market_id for market in markets] == ["open-market"]


def test_normalize_gamma_markets_extracts_event_from_market_events_field() -> None:
    payload = [
        {
            "id": "market-1",
            "question": "BTC Up or Down - 05:00-05:05",
            "outcomes": "[\"Yes\", \"No\"]",
            "clobTokenIds": "[\"tok_yes\", \"tok_no\"]",
            "active": True,
            "closed": False,
            "events": [{"id": "event-99"}],
            "startDate": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-01T00:05:00Z",
        },
        {
            "id": "market-2",
            "question": "BTC Up or Down - 05:05-05:10",
            "outcomes": "[\"Yes\", \"No\"]",
            "clobTokenIds": "[\"tok_yes2\", \"tok_no2\"]",
            "active": True,
            "closed": True,
        },
    ]

    markets = normalize_gamma_markets(payload)
    assert len(markets) == 1
    assert markets[0].market_id == "market-1"
    assert markets[0].event_id == "event-99"
    assert markets[0].yes_token_id == "tok_yes"
    assert markets[0].no_token_id == "tok_no"


def test_extract_gamma_market_items_supports_nested_data_markets_shape() -> None:
    payload = {
        "data": {
            "markets": [
                {
                    "id": "market-1",
                    "question": "BTC Up or Down - 05:00-05:05",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"tok_yes\", \"tok_no\"]",
                    "active": True,
                    "closed": False,
                }
            ]
        }
    }
    extracted = extract_gamma_market_items(payload)
    assert extracted.extraction_path == "root.data.markets"
    assert extracted.top_level_type == "dict"
    assert len(extracted.items) == 1


def test_extract_gamma_market_items_supports_events_shape() -> None:
    payload = {
        "events": [
            {
                "id": "event-1",
                "markets": [
                    {
                        "id": "market-1",
                        "question": "BTC Up or Down - 05:00-05:05",
                        "outcomes": "[\"Yes\", \"No\"]",
                        "clobTokenIds": "[\"tok_yes\", \"tok_no\"]",
                        "active": True,
                        "closed": False,
                    }
                ],
            }
        ]
    }
    extracted = extract_gamma_market_items(payload)
    assert extracted.extraction_path == "root.events[].markets"
    assert len(extracted.items) == 1


def test_normalize_gamma_markets_maps_up_down_tokens_to_yes_no_slots() -> None:
    payload = [
        {
            "id": "market-updown",
            "question": "BTC Up or Down — 05:00-05:05",
            "outcomes": "[\"Up\", \"Down\"]",
            "clobTokenIds": "[\"up_tok\", \"down_tok\"]",
            "conditionId": "0xabc",
            "active": True,
            "closed": False,
            "startDate": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-01T00:05:00Z",
        }
    ]

    markets = normalize_gamma_markets(payload)
    assert len(markets) == 1
    market = markets[0]
    assert market.yes_token_id == "up_tok"
    assert market.no_token_id == "down_tok"
    assert market.condition_id == "0xabc"


def test_normalize_orderbook_snapshot_sorts_levels() -> None:
    payload = {
        "timestamp": "2026-01-01T00:00:00Z",
        "bids": [{"price": "0.40", "size": "10"}, {"price": "0.50", "size": "8"}],
        "asks": [{"price": "0.60", "size": "7"}, {"price": "0.55", "size": "5"}],
    }

    snapshot = normalize_orderbook_snapshot(payload, token_id="tok_yes")
    assert snapshot is not None
    assert [level.price for level in snapshot.bids] == [0.5, 0.4]
    assert [level.price for level in snapshot.asks] == [0.55, 0.6]


def test_normalize_ws_message_maps_token_to_market() -> None:
    token_map = {"tok_yes": ("market-1", "YES")}
    raw = json.dumps(
        {
            "data": [
                {
                    "event_type": "book",
                    "token_id": "tok_yes",
                    "bids": [[0.51, 10]],
                    "asks": [[0.53, 11]],
                    "timestamp": "2026-01-01T00:00:00Z",
                },
                {
                    "event_type": "trade",
                    "token_id": "tok_yes",
                    "trade_id": "trade-1",
                    "price": "0.52",
                    "size": "7",
                    "side": "buy",
                    "timestamp": "2026-01-01T00:00:01Z",
                },
            ]
        }
    )

    normalized = normalize_ws_message(raw, token_to_market=token_map)
    assert len(normalized.orderbook_updates) == 1
    assert len(normalized.trades) == 1
    assert normalized.trades[0].market_id == "market-1"
    assert normalized.trades[0].trade_id == "trade-1:tok_yes"


def test_normalize_ws_message_supports_buys_sells_shape() -> None:
    token_map = {"tok_yes": ("market-1", "YES")}
    raw = json.dumps(
        {
            "event_message": {
                "asset_id": "tok_yes",
                "timestamp": "2026-01-01T00:00:00Z",
                "buys": [{"price": "0.51", "size": "10"}],
                "sells": [{"price": "0.53", "size": "11"}],
            }
        }
    )
    normalized = normalize_ws_message(raw, token_to_market=token_map)
    assert len(normalized.orderbook_updates) == 1
    assert normalized.orderbook_updates[0].bids[0].price == 0.51
    assert normalized.orderbook_updates[0].asks[0].price == 0.53


def test_normalize_ws_message_supports_root_list_and_price_changes() -> None:
    token_map = {"tok_yes": ("market-1", "YES")}
    raw = json.dumps(
        [
            {
                "event_type": "book",
                "asset_id": "tok_yes",
                "timestamp": "2026-01-01T00:00:00Z",
                "bids": [{"price": "0.51", "size": "10"}],
                "asks": [{"price": "0.53", "size": "11"}],
            }
        ]
    )
    normalized = normalize_ws_message(raw, token_to_market=token_map)
    assert len(normalized.orderbook_updates) == 1

    raw_price_change = json.dumps(
        {
            "event_type": "price_change",
            "timestamp": "2026-01-01T00:00:01Z",
            "price_changes": [
                {
                    "asset_id": "tok_yes",
                    "price": "0.52",
                    "size": "9",
                    "side": "BUY",
                }
            ],
        }
    )
    normalized_change = normalize_ws_message(raw_price_change, token_to_market=token_map)
    assert len(normalized_change.price_changes) == 1
    assert normalized_change.price_changes[0].book_side == "bid"


def test_normalize_trade_history_supports_transaction_hash_and_filters() -> None:
    payload = [
        {
            "transactionHash": "0xaaa",
            "asset": "tok_yes",
            "conditionId": "0xcond",
            "price": "0.52",
            "size": "15",
            "side": "BUY",
            "timestamp": "2026-01-01T00:00:01Z",
        },
        {
            "transactionHash": "0xbbb",
            "asset": "tok_other",
            "conditionId": "0xother",
            "price": "0.48",
            "size": "7",
            "side": "SELL",
            "timestamp": "2026-01-01T00:00:02Z",
        },
    ]

    trades = normalize_trade_history(
        payload,
        market_id="m1",
        token_ids={"tok_yes"},
        condition_id="0xcond",
    )
    assert len(trades) == 1
    assert trades[0].trade_id == "0xaaa:tok_yes"
    assert trades[0].market_id == "m1"


def test_normalize_trade_history_allows_missing_asset_when_condition_matches() -> None:
    payload = [
        {
            "transactionHash": "0xabc",
            "conditionId": "0xcond",
            "price": "0.54",
            "size": "9",
            "side": "BUY",
            "timestamp": "2026-01-01T00:00:03Z",
        }
    ]

    trades = normalize_trade_history(
        payload,
        market_id="m1",
        token_ids={"tok_yes", "tok_no"},
        condition_id="0xcond",
    )
    assert len(trades) == 1
    assert trades[0].trade_id == "0xabc"
    assert trades[0].market_id == "m1"
