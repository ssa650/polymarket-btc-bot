from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Awaitable, Callable, List, Optional, Set

import websockets
from websockets.asyncio.client import ClientConnection


class WSClient:
    """Resilient websocket client with reconnect/backoff logic."""

    def __init__(
        self,
        url: str,
        ping_interval_sec: float,
        ping_timeout_sec: float,
        reconnect_min_sec: float,
        reconnect_max_sec: float,
    ) -> None:
        self.url = url
        self.ping_interval_sec = ping_interval_sec
        self.ping_timeout_sec = ping_timeout_sec
        self.reconnect_min_sec = reconnect_min_sec
        self.reconnect_max_sec = reconnect_max_sec
        self._log = logging.getLogger(self.__class__.__name__)
        self._ws: Optional[ClientConnection] = None
        self._subscribed_tokens: Set[str] = set()
        self._pending_tokens: Set[str] = set()
        self._sub_lock = asyncio.Lock()
        self.reconnect_count = 0

    async def subscribe_tokens(self, token_ids: List[str]) -> None:
        """Dynamically subscribe to new token ids when they are discovered."""
        clean = {token for token in token_ids if token}
        if not clean:
            return

        async with self._sub_lock:
            self._pending_tokens.update(clean)
            ws = self._ws
            already_subscribed = set(self._subscribed_tokens)

        to_subscribe = sorted(clean - already_subscribed)
        if ws is None or not to_subscribe:
            return

        try:
            await self._subscribe(ws, to_subscribe, operation="subscribe")
            async with self._sub_lock:
                self._subscribed_tokens.update(to_subscribe)
                self._pending_tokens.difference_update(to_subscribe)
            self._log.info("ws_dynamic_subscribe", extra={"token_count": len(to_subscribe)})
        except Exception:
            # Keep token ids pending so they are retried on reconnect.
            self._log.warning(
                "ws_dynamic_subscribe_failed",
                extra={"token_count": len(to_subscribe)},
                exc_info=True,
            )

    async def unsubscribe_tokens(self, token_ids: List[str]) -> None:
        """Unsubscribe token ids from the active websocket connection."""
        clean = {token for token in token_ids if token}
        if not clean:
            return

        async with self._sub_lock:
            self._pending_tokens.difference_update(clean)
            ws = self._ws
            subscribed = set(self._subscribed_tokens)

        to_unsubscribe = sorted(clean & subscribed)
        if ws is None or not to_unsubscribe:
            return

        try:
            await self._subscribe(ws, to_unsubscribe, operation="unsubscribe")
            async with self._sub_lock:
                self._subscribed_tokens.difference_update(to_unsubscribe)
            self._log.info(
                "ws_dynamic_unsubscribe",
                extra={"token_count": len(to_unsubscribe)},
            )
        except Exception:
            self._log.warning(
                "ws_dynamic_unsubscribe_failed",
                extra={"token_count": len(to_unsubscribe)},
                exc_info=True,
            )

    async def request_reconnect(self, reason: str) -> None:
        """Close active websocket so run loop reconnects with fresh subscriptions."""
        async with self._sub_lock:
            ws = self._ws
        if ws is None:
            return
        try:
            await ws.close(code=1000, reason=reason[:120])
            self._log.info("ws_reconnect_requested", extra={"reason": reason})
        except Exception:
            self._log.warning("ws_reconnect_request_failed", extra={"reason": reason}, exc_info=True)

    async def run_forever(
        self,
        get_token_ids: Callable[[], List[str]],
        on_message: Callable[[str | bytes], Awaitable[None]],
        stop_event: asyncio.Event,
        on_reconnect: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        backoff = self.reconnect_min_sec
        has_connected_once = False
        while not stop_event.is_set():
            token_ids = set(get_token_ids())
            async with self._sub_lock:
                token_ids.update(self._pending_tokens)
            is_reconnect = has_connected_once
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval_sec,
                    ping_timeout=self.ping_timeout_sec,
                    max_queue=None,
                    max_size=None,
                ) as ws:
                    if is_reconnect:
                        self.reconnect_count += 1
                    async with self._sub_lock:
                        self._ws = ws
                        self._subscribed_tokens.clear()
                    self._log.info(
                        "ws_connected",
                        extra={
                            "token_count": len(token_ids),
                            "url": self.url,
                            "reconnect": is_reconnect,
                            "reconnect_count": self.reconnect_count,
                        },
                    )
                    initial_tokens = sorted(token_ids)
                    await self._subscribe(ws, initial_tokens)
                    async with self._sub_lock:
                        self._subscribed_tokens.update(initial_tokens)
                        self._pending_tokens.difference_update(initial_tokens)
                    if is_reconnect and on_reconnect is not None:
                        try:
                            await on_reconnect()
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            self._log.warning(
                                "ws_reconnect_recovery_failed",
                                extra={"reconnect_count": self.reconnect_count},
                                exc_info=True,
                            )
                    has_connected_once = True
                    backoff = self.reconnect_min_sec

                    while not stop_event.is_set():
                        raw = await ws.recv()
                        await on_message(raw)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delay = min(backoff, self.reconnect_max_sec)
                jitter = random.uniform(0.0, 0.25 * delay)
                wait_for = delay + jitter
                self._log.warning(
                    "ws_disconnected",
                    extra={
                        "error": str(exc),
                        "reconnect_in_sec": round(wait_for, 3),
                        "reconnect_count": self.reconnect_count,
                    },
                    exc_info=True,
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=wait_for)
                except asyncio.TimeoutError:
                    pass
                backoff = min(self.reconnect_max_sec, max(self.reconnect_min_sec, backoff * 2))
            finally:
                async with self._sub_lock:
                    self._ws = None
                    self._subscribed_tokens.clear()

    async def _subscribe(
        self,
        ws: ClientConnection,
        token_ids: List[str],
        *,
        operation: str | None = None,
    ) -> None:
        if not token_ids:
            return

        chunk_size = 100
        for idx in range(0, len(token_ids), chunk_size):
            payload: dict[str, object] = {"assets_ids": token_ids[idx : idx + chunk_size]}
            if operation is None:
                payload["type"] = "market"
                payload["custom_feature_enabled"] = True
            else:
                payload["operation"] = operation
                if operation == "subscribe":
                    payload["custom_feature_enabled"] = True
            try:
                await ws.send(json.dumps(payload))
            except Exception:
                self._log.warning(
                    "ws_subscribe_failed",
                    extra={"payload": payload},
                    exc_info=True,
                )
