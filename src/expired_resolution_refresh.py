from __future__ import annotations

import json
import sqlite3
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .models import RECORDER_VERSION, SCHEMA_VERSION, parse_timestamp, to_iso
from .polymarket.normalization import extract_gamma_market_items


def refresh_expired_market_resolutions(
    *,
    db_path: str,
    gamma_client: Any,
    older_than_minutes: float = 2.0,
    limit: int = 100,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    conn = sqlite3.connect(str(Path(db_path)))
    conn.row_factory = sqlite3.Row
    try:
        candidates = _load_candidates(
            conn,
            older_than=now_dt - timedelta(minutes=float(older_than_minutes)),
            limit=int(limit),
        )
        checked: list[dict[str, Any]] = []
        resolved: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        updated = 0
        events_inserted = 0
        for row in candidates:
            payload = _fetch_market_payload(gamma_client, dict(row))
            resolution = _extract_resolution(payload, dict(row), now=now_dt)
            checked.append(
                {
                    "run_id": row["run_id"],
                    "market_id": row["market_id"],
                    "close_time": row["close_time"],
                    "resolved": bool(resolution.get("resolved")),
                    "winning_outcome": resolution.get("winning_outcome"),
                    "winning_asset_id": resolution.get("winning_asset_id"),
                }
            )
            if not resolution.get("resolved"):
                unresolved.append(checked[-1])
                continue
            resolved.append(checked[-1])
            if apply:
                changed = _apply_resolution(conn, dict(row), resolution, now=now_dt)
                updated += int(changed["updated"])
                events_inserted += int(changed["event_inserted"])
        return {
            "status": "ok",
            "dry_run": not bool(apply),
            "db_path": db_path,
            "older_than_minutes": float(older_than_minutes),
            "limit": int(limit),
            "candidates_found": len(candidates),
            "markets_checked": len(checked),
            "resolved_found": len(resolved),
            "still_unresolved": len(unresolved),
            "markets_updated": updated,
            "market_events_inserted": events_inserted,
            "resolved": resolved[:50],
            "unresolved": unresolved[:50],
        }
    finally:
        conn.close()


async def refresh_expired_market_resolutions_async(
    **kwargs: Any,
) -> dict[str, Any]:
    db_path = str(kwargs["db_path"])
    gamma_client = kwargs["gamma_client"]
    older_than_minutes = float(kwargs.get("older_than_minutes", 2.0))
    limit = int(kwargs.get("limit", 100))
    apply = bool(kwargs.get("apply", False))
    now_dt = kwargs.get("now") or datetime.now(timezone.utc)
    conn = sqlite3.connect(str(Path(db_path)))
    conn.row_factory = sqlite3.Row
    try:
        candidates = _load_candidates(
            conn,
            older_than=now_dt - timedelta(minutes=older_than_minutes),
            limit=limit,
        )
        checked: list[dict[str, Any]] = []
        resolved: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        updated = 0
        events_inserted = 0
        for row in candidates:
            payload = _fetch_market_payload(gamma_client, dict(row))
            if inspect.isawaitable(payload):
                payload = await payload
            resolution = _extract_resolution(payload, dict(row), now=now_dt)
            checked.append(
                {
                    "run_id": row["run_id"],
                    "market_id": row["market_id"],
                    "close_time": row["close_time"],
                    "resolved": bool(resolution.get("resolved")),
                    "winning_outcome": resolution.get("winning_outcome"),
                    "winning_asset_id": resolution.get("winning_asset_id"),
                }
            )
            if not resolution.get("resolved"):
                unresolved.append(checked[-1])
                continue
            resolved.append(checked[-1])
            if apply:
                changed = _apply_resolution(conn, dict(row), resolution, now=now_dt)
                updated += int(changed["updated"])
                events_inserted += int(changed["event_inserted"])
        return {
            "status": "ok",
            "dry_run": not bool(apply),
            "db_path": db_path,
            "older_than_minutes": older_than_minutes,
            "limit": limit,
            "candidates_found": len(candidates),
            "markets_checked": len(checked),
            "resolved_found": len(resolved),
            "still_unresolved": len(unresolved),
            "markets_updated": updated,
            "market_events_inserted": events_inserted,
            "resolved": resolved[:50],
            "unresolved": unresolved[:50],
        }
    finally:
        conn.close()


def _load_candidates(
    conn: sqlite3.Connection,
    *,
    older_than: datetime,
    limit: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM markets
        WHERE close_time IS NOT NULL
          AND datetime(close_time) < datetime(?)
          AND COALESCE(resolved, 0) <> 1
        ORDER BY
          CASE
            WHEN COALESCE(phase, market_phase, '') = 'expired_waiting_resolution' THEN 0
            WHEN lower(COALESCE(platform_status, status, '')) = 'open' THEN 1
            ELSE 2
          END ASC,
          datetime(close_time) ASC,
          market_id ASC
        LIMIT ?
        """,
        (to_iso(older_than), int(limit)),
    ).fetchall()


def _fetch_market_payload(gamma_client: Any, row: dict[str, Any]) -> Any:
    if hasattr(gamma_client, "fetch_market_by_id"):
        return gamma_client.fetch_market_by_id(
            row.get("market_id"),
            condition_id=row.get("condition_id"),
        )
    if hasattr(gamma_client, "fetch_market_resolution"):
        return gamma_client.fetch_market_resolution(row)
    raise RuntimeError("gamma_client must provide fetch_market_by_id or fetch_market_resolution")


def _extract_resolution(
    payload: Any,
    row: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    item = _first_market_payload(payload)
    if item is None:
        return {"resolved": False}
    winning_asset_id = _first_str(
        item,
        (
            "winning_asset_id",
            "winningAssetId",
            "winning_asset",
            "winningTokenId",
            "winning_token_id",
        ),
    )
    winning_outcome = _first_str(
        item,
        (
            "winning_outcome",
            "winningOutcome",
            "winner",
            "winningOutcomeName",
            "outcome",
            "result",
        ),
    )
    if winning_asset_id is None:
        winning_asset_id = _winning_asset_from_tokens(item)
    if winning_outcome is None and winning_asset_id is not None:
        if winning_asset_id == row.get("yes_token_id"):
            winning_outcome = "YES"
        elif winning_asset_id == row.get("no_token_id"):
            winning_outcome = "NO"
    resolved_flag = bool(item.get("resolved") is True)
    status = str(item.get("status") or "").strip().lower()
    closed = item.get("closed") is True
    has_winner = winning_asset_id is not None or winning_outcome is not None
    resolved = has_winner and (
        resolved_flag or closed or status in {"resolved", "closed", "finalized", "settled"}
    )
    resolved_at = to_iso(
        parse_timestamp(
            item.get("resolved_at")
            or item.get("resolvedAt")
            or item.get("resolutionTime")
            or item.get("closedTime")
            or item.get("endDate")
        )
        or now
    )
    return {
        "resolved": bool(resolved),
        "resolved_at": resolved_at,
        "winning_asset_id": winning_asset_id,
        "winning_outcome": winning_outcome,
        "raw": item,
    }


def _first_market_payload(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        extraction = extract_gamma_market_items(payload)
        if extraction.items:
            return extraction.items[0]
        return payload
    extraction = extract_gamma_market_items(payload)
    return extraction.items[0] if extraction.items else None


def _first_str(item: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _winning_asset_from_tokens(item: Mapping[str, Any]) -> str | None:
    tokens = item.get("tokens")
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        if token.get("winner") is True or token.get("winning") is True:
            value = token.get("token_id") or token.get("tokenId") or token.get("asset_id")
            return str(value) if value is not None else None
    return None


def _apply_resolution(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    resolution: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, int]:
    now_iso = to_iso(now)
    with conn:
        cursor = conn.execute(
            """
            UPDATE markets
            SET resolved = 1,
                resolved_at = COALESCE(?, resolved_at),
                winning_asset_id = COALESCE(?, winning_asset_id),
                winning_outcome = COALESCE(?, winning_outcome),
                platform_status = 'resolved',
                status = 'resolved',
                phase = 'resolved',
                market_phase = 'resolved',
                tracking_state = 'inactive',
                last_updated = ?
            WHERE run_id = ?
              AND market_id = ?
            """,
            (
                resolution.get("resolved_at"),
                resolution.get("winning_asset_id"),
                resolution.get("winning_outcome"),
                now_iso,
                row.get("run_id"),
                row.get("market_id"),
            ),
        )
        conn.execute(
            """
            INSERT INTO market_events (
                run_id, timestamp, market_id, event_type, details,
                recorder_version, schema_version
            ) VALUES (?, ?, ?, 'market_resolution_backfilled', ?, ?, ?)
            """,
            (
                row.get("run_id"),
                now_iso,
                row.get("market_id"),
                json.dumps(
                    {
                        "source": "refresh_expired_market_resolutions",
                        "resolved_at": resolution.get("resolved_at"),
                        "winning_asset_id": resolution.get("winning_asset_id"),
                        "winning_outcome": resolution.get("winning_outcome"),
                    },
                    sort_keys=True,
                ),
                RECORDER_VERSION,
                SCHEMA_VERSION,
            ),
        )
    return {"updated": int(cursor.rowcount or 0), "event_inserted": 1}


__all__ = [
    "refresh_expired_market_resolutions",
    "refresh_expired_market_resolutions_async",
]
