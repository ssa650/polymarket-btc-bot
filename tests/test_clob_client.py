from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.polymarket.clob_client import ClobClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return

    def json(self):
        return self._payload


class _FakeDataClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def get(self, path: str, params=None):
        self.calls.append({"path": path, "params": dict(params or {})})
        return _FakeResponse(self.payload)


def test_fetch_recent_trades_enforces_local_since_cutoff() -> None:
    payload = [
        {
            "transactionHash": "0x1",
            "asset": "tok_yes",
            "conditionId": "0xcond",
            "price": "0.51",
            "size": "2",
            "side": "BUY",
            "timestamp": 1775286600,
        },
        {
            "transactionHash": "0x2",
            "asset": "tok_yes",
            "conditionId": "0xcond",
            "price": "0.52",
            "size": "3",
            "side": "BUY",
            "timestamp": 1775286610,
        },
    ]

    fake_data = _FakeDataClient(payload)
    client = ClobClient.__new__(ClobClient)
    client._data = fake_data
    client._log = logging.getLogger("test_clob_client")

    since = datetime.fromtimestamp(1775286605, tz=timezone.utc)
    trades = asyncio.run(
        client.fetch_recent_trades(
            market_id="m1",
            token_ids=["tok_yes"],
            condition_id="0xcond",
            since=since,
            limit=50,
        )
    )

    assert len(trades) == 1
    assert trades[0].trade_id == "0x2:tok_yes"
    assert fake_data.calls[0]["params"]["since"] == since.isoformat()
