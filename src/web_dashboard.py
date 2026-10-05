from __future__ import annotations

import json
import logging
import sqlite3
from glob import glob
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

from .btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from .dashboard import connect_read_only, fetch_dashboard_data
from .models import parse_timestamp, to_iso


DEFAULT_WEB_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_WEB_DASHBOARD_PORT = 8765
DEFAULT_WEB_DASHBOARD_REFRESH_SEC = 1.0

_PAPER_TRADE_FIELDS = (
    "id",
    "created_at",
    "run_id",
    "market_id",
    "question",
    "signal_timestamp",
    "market_start_time",
    "market_close_time",
    "strategy_id",
    "strategy_name",
    "signal_direction",
    "status",
    "skip_reason",
    "stake_usd",
    "entry_price",
    "adjusted_entry_price",
    "exit_time",
    "exit_reason",
    "exit_price",
    "adjusted_exit_price",
    "payout_usd",
    "pnl_usd",
    "roi",
    "realized_pnl_usd",
    "realized_roi",
    "settled_at",
    "predicted_probability_yes",
    "probability_for_direction",
    "estimated_edge",
    "time_until_resolution",
    "starting_bankroll_usd",
    "bankroll_before_trade",
    "available_cash_before_trade",
    "open_exposure_before_trade",
    "stake_fraction_of_bankroll",
    "max_open_exposure_usd",
    "bankroll_after_trade",
    "bankroll_status",
    "requested_shares",
    "max_fillable_shares",
    "liquidity_fill_fraction_used",
    "liquidity_check_passed",
    "liquidity_skip_reason",
    "realistic_execution_enabled",
    "entry_latency_sec",
    "exit_latency_sec",
    "entry_price_drift",
    "spread_cents_at_entry",
    "realistic_execution_skip_reason",
    "extra_slippage_cents_applied",
    "btc_trend_regime",
    "volatility_regime",
    "spread_regime",
    "liquidity_regime",
    "time_regime",
)


def fetch_web_dashboard_payload(
    db_path: str,
    *,
    now: datetime | None = None,
    preferred_btc_source: str | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    data = fetch_dashboard_data(
        db_path,
        now=now_dt,
        preferred_btc_source=preferred_btc_source,
    )
    btc_price = _jsonable(data.btc_price)
    btc_age_sec = _btc_sample_age_sec(btc_price, now=now_dt)
    if isinstance(btc_price, dict):
        btc_price["sample_age_sec"] = btc_age_sec
    btc_prices_by_source = _fetch_latest_btc_prices_by_source(
        db_path,
        now=now_dt,
    )

    warnings = list(data.warnings)
    if btc_age_sec is not None and btc_age_sec > 3 and "stale_btc_price" not in warnings:
        warnings.append("stale_btc_price")
    stale_sources = [
        str(row.get("source"))
        for row in btc_prices_by_source
        if _float_or_none(row.get("sample_age_sec")) is not None
        and float(row.get("sample_age_sec")) > 3.0
    ]
    for source in stale_sources:
        warning = f"stale_btc_price_source_{source}"
        if warning not in warnings:
            warnings.append(warning)
    if {
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    }.issubset(set(stale_sources)):
        warnings.append("both_btc_sources_stale")
    return {
        "now": now_dt.isoformat(),
        "db_path": data.db_path,
        "active_market": _jsonable(data.active_market),
        "btc_price": btc_price,
        "btc_prices_by_source": btc_prices_by_source,
        "btc_preferred_source": preferred_btc_source,
        "latest_snapshot": _jsonable(data.latest_snapshot),
        "best_bid_ask": _jsonable(data.best_bid_ask),
        "recent_trades": _jsonable(data.recent_trades),
        "health": _jsonable(data.health),
        "event_counts": _jsonable(data.event_counts),
        "warnings": warnings,
    }


def fetch_series_payload(
    db_path: str,
    *,
    window_sec: int = 300,
    now: datetime | None = None,
    preferred_btc_source: str | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    safe_window = max(1, min(int(window_sec), 24 * 60 * 60))
    cutoff = now_dt - timedelta(seconds=safe_window)
    cutoff_iso = to_iso(cutoff) or cutoff.isoformat()

    conn = connect_read_only(db_path)
    try:
        active_market = _fetch_active_market(conn)
        run_id = _value(active_market, "run_id")
        market_id = _value(active_market, "market_id")
        btc_source = preferred_btc_source or _fetch_latest_btc_source(conn)
        btc = _fetch_btc_series(
            conn,
            cutoff_iso=cutoff_iso,
            source=btc_source,
        )
        snapshots = (
            _fetch_snapshot_series(
                conn,
                cutoff_iso=cutoff_iso,
                run_id=str(run_id) if run_id else None,
                market_id=str(market_id) if market_id else None,
            )
            if market_id is not None
            else []
        )
        return {
            "now": now_dt.isoformat(),
            "window_sec": safe_window,
            "market_id": market_id,
            "btc_source": btc_source,
            "btc_price": btc,
            "yes_best_bid": [
                {"timestamp": row["timestamp"], "value": row["best_bid_yes"]}
                for row in snapshots
                if row.get("best_bid_yes") is not None
            ],
            "no_best_bid": [
                {"timestamp": row["timestamp"], "value": row["best_bid_no"]}
                for row in snapshots
                if row.get("best_bid_no") is not None
            ],
            "snapshot_count": len(snapshots),
        }
    finally:
        conn.close()


def resolve_paper_db_paths(
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_path in paper_db_paths or ():
        if not raw_path:
            continue
        path = str(Path(raw_path))
        key = str(Path(path).resolve()) if Path(path).exists() else path
        if key not in seen:
            seen.add(key)
            resolved.append(path)
    for pattern in paper_db_globs or ():
        if not pattern:
            continue
        for match in sorted(glob(str(pattern))):
            key = str(Path(match).resolve())
            if key not in seen:
                seen.add(key)
                resolved.append(match)
    return resolved


def fetch_recorder_summary_payload(
    db_path: str,
    *,
    now: datetime | None = None,
    preferred_btc_source: str | None = None,
) -> dict[str, Any]:
    payload = fetch_web_dashboard_payload(
        db_path,
        now=now,
        preferred_btc_source=preferred_btc_source,
    )
    active = payload.get("active_market") or {}
    snapshot = payload.get("latest_snapshot") or {}
    best = payload.get("best_bid_ask") or {}
    btc = payload.get("btc_price") or {}
    health = payload.get("health") or {}
    now_dt = parse_timestamp(payload.get("now")) or _utc(now or datetime.now(timezone.utc))
    snapshot_age = _age_from_timestamp(snapshot.get("timestamp"), now=now_dt)
    feature_age = _age_from_timestamp(health.get("latest_feature_timestamp"), now=now_dt)
    ws_age = _float_or_none(health.get("last_ws_event_age_sec"))
    btc_age = _float_or_none(btc.get("sample_age_sec"))
    health_status = "OK"
    if payload.get("warnings"):
        health_status = "STALE" if any("stale" in str(w) for w in payload["warnings"]) else "WARN"
    if active == {}:
        health_status = "ERROR"
    return {
        "now": payload.get("now"),
        "db_path": payload.get("db_path"),
        "health_status": health_status,
        "active_market": {
            "market_id": active.get("market_id"),
            "question": active.get("question"),
            "short_question": _short_text(active.get("question"), 52),
            "time_left_sec": _time_left_sec(active.get("close_time"), now=now_dt),
            "phase": active.get("phase"),
            "tracking_state": active.get("tracking_state"),
            "start_time": active.get("start_time"),
            "close_time": active.get("close_time"),
            "yes_token_id": active.get("yes_token_id"),
            "no_token_id": active.get("no_token_id"),
        }
        if active
        else None,
        "btc": btc,
        "btc_prices_by_source": payload.get("btc_prices_by_source") or [],
        "yes": _price_summary(best.get("YES"), snapshot, side="yes", now=now_dt),
        "no": _price_summary(best.get("NO"), snapshot, side="no", now=now_dt),
        "recorder_health": {
            **health,
            "status": health_status,
            "snapshot_age_sec": snapshot_age,
            "feature_age_sec": feature_age,
            "btc_age_sec": btc_age,
            "ws_event_age_sec": ws_age,
        },
        "warnings": payload.get("warnings") or [],
    }


def fetch_paper_traders_summary(
    paper_db_paths: Sequence[str] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    traders = [_paper_trader_summary(path, now=now_dt) for path in paper_db_paths or ()]
    totals = {
        "paper_db_count": len(traders),
        "total_trades": sum(int(t.get("total_trades") or 0) for t in traders),
        "open_awaiting_trades": sum(int(t.get("open_awaiting_trades") or 0) for t in traders),
        "settled_trades": sum(int(t.get("settled_trades") or 0) for t in traders),
        "closed_trades": sum(int(t.get("closed_trades") or 0) for t in traders),
        "skipped_trades": sum(int(t.get("skipped_trades") or 0) for t in traders),
        "realized_pnl": _round_or_none(
            sum(float(t.get("realized_pnl") or 0.0) for t in traders)
        ),
    }
    return {"now": now_dt.isoformat(), "traders": traders, "totals": totals}


def fetch_paper_activity_payload(
    paper_db_paths: Sequence[str] | None,
    *,
    limit: int = 100,
    status_in: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    rows: list[dict[str, Any]] = []
    for path in paper_db_paths or ():
        rows.extend(
            _paper_rows_for_dashboard(
                path,
                limit=limit,
                status_in=status_in,
                now=now_dt,
            )
        )
    rows.sort(key=lambda row: _sort_timestamp(row.get("activity_time")), reverse=True)
    return {"now": now_dt.isoformat(), "rows": rows[: max(1, int(limit))]}


def fetch_paper_skips_payload(
    paper_db_paths: Sequence[str] | None,
    *,
    limit: int = 100,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = fetch_paper_activity_payload(
        paper_db_paths,
        limit=limit,
        status_in=("skipped",),
        now=now,
    )
    rows = payload["rows"]
    return {
        **payload,
        "groups": {
            "skip_reason": _count_field(rows, "skip_reason"),
            "realistic_execution_skip_reason": _count_field(
                rows,
                "realistic_execution_skip_reason",
            ),
            "liquidity_skip_reason": _count_field(rows, "liquidity_skip_reason"),
            "strategy_id": _count_field(rows, "strategy_id"),
            "trader_label": _count_field(rows, "trader_label"),
        },
    }


def fetch_paper_open_payload(
    paper_db_paths: Sequence[str] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    return fetch_paper_activity_payload(
        paper_db_paths,
        limit=1000,
        status_in=("open", "awaiting_resolution"),
        now=now,
    )


def fetch_paper_closed_payload(
    paper_db_paths: Sequence[str] | None,
    *,
    limit: int = 100,
    now: datetime | None = None,
) -> dict[str, Any]:
    return fetch_paper_activity_payload(
        paper_db_paths,
        limit=limit,
        status_in=("closed", "settled"),
        now=now,
    )


def create_web_dashboard_server(
    db_path: str,
    *,
    host: str = DEFAULT_WEB_DASHBOARD_HOST,
    port: int = DEFAULT_WEB_DASHBOARD_PORT,
    refresh_sec: float = DEFAULT_WEB_DASHBOARD_REFRESH_SEC,
    preferred_btc_source: str | None = None,
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
) -> ThreadingHTTPServer:
    refresh_ms = max(250, int(float(refresh_sec) * 1000))
    paper_paths = resolve_paper_db_paths(paper_db_paths, paper_db_globs)

    class Handler(WebDashboardRequestHandler):
        dashboard_db_path = db_path
        dashboard_refresh_ms = refresh_ms
        dashboard_btc_source = preferred_btc_source
        paper_db_paths = paper_paths

    return ThreadingHTTPServer((host, int(port)), Handler)


def run_web_dashboard(
    db_path: str,
    *,
    host: str = DEFAULT_WEB_DASHBOARD_HOST,
    port: int = DEFAULT_WEB_DASHBOARD_PORT,
    refresh_sec: float = DEFAULT_WEB_DASHBOARD_REFRESH_SEC,
    preferred_btc_source: str | None = None,
    paper_db_paths: Sequence[str] | None = None,
    paper_db_globs: Sequence[str] | None = None,
) -> None:
    log = logging.getLogger("web_dashboard")
    server = create_web_dashboard_server(
        db_path,
        host=host,
        port=port,
        refresh_sec=refresh_sec,
        preferred_btc_source=preferred_btc_source,
        paper_db_paths=paper_db_paths,
        paper_db_globs=paper_db_globs,
    )
    actual_host, actual_port = server.server_address[:2]
    log.info(
        "web_dashboard_started",
        extra={
            "db_path": db_path,
            "url": f"http://{actual_host}:{actual_port}",
            "refresh_sec": refresh_sec,
            "preferred_btc_source": preferred_btc_source,
            "paper_db_paths": resolve_paper_db_paths(paper_db_paths, paper_db_globs),
            "read_only": True,
        },
    )
    print(f"web_dashboard_url=http://{actual_host}:{actual_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log.info("web_dashboard_stopped")


class WebDashboardRequestHandler(BaseHTTPRequestHandler):
    dashboard_db_path = ""
    dashboard_refresh_ms = 1000
    dashboard_btc_source: str | None = None
    paper_db_paths: Sequence[str] = ()
    server_version = "PolymarketRecorderDashboard/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"", "/"}:
                self._send_html(_dashboard_html(self.dashboard_refresh_ms))
                return
            if parsed.path == "/api/dashboard":
                self._send_json(
                    fetch_web_dashboard_payload(
                        self.dashboard_db_path,
                        preferred_btc_source=self.dashboard_btc_source,
                    )
                )
                return
            if parsed.path == "/api/recorder/summary":
                self._send_json(
                    fetch_recorder_summary_payload(
                        self.dashboard_db_path,
                        preferred_btc_source=self.dashboard_btc_source,
                    )
                )
                return
            if parsed.path == "/api/paper-traders/summary":
                self._send_json(fetch_paper_traders_summary(self.paper_db_paths))
                return
            if parsed.path == "/api/paper-traders/activity":
                params = parse_qs(parsed.query)
                limit = _safe_int(params.get("limit", ["100"])[0], default=100)
                self._send_json(
                    fetch_paper_activity_payload(self.paper_db_paths, limit=limit)
                )
                return
            if parsed.path == "/api/paper-traders/skips":
                params = parse_qs(parsed.query)
                limit = _safe_int(params.get("limit", ["100"])[0], default=100)
                self._send_json(fetch_paper_skips_payload(self.paper_db_paths, limit=limit))
                return
            if parsed.path == "/api/paper-traders/open":
                self._send_json(fetch_paper_open_payload(self.paper_db_paths))
                return
            if parsed.path == "/api/paper-traders/closed":
                params = parse_qs(parsed.query)
                limit = _safe_int(params.get("limit", ["100"])[0], default=100)
                self._send_json(fetch_paper_closed_payload(self.paper_db_paths, limit=limit))
                return
            if parsed.path == "/api/series":
                params = parse_qs(parsed.query)
                window_sec = _safe_int(params.get("window_sec", ["300"])[0], default=300)
                self._send_json(
                    fetch_series_payload(
                        self.dashboard_db_path,
                        window_sec=window_sec,
                        preferred_btc_source=self.dashboard_btc_source,
                    )
                )
                return
            if parsed.path == "/healthz":
                self._send_json({"ok": True})
                return
            self.send_error(404, "not found")
        except FileNotFoundError:
            self._send_json({"error": "database_not_found"}, status=404)
        except sqlite3.Error as exc:
            self._send_json({"error": "sqlite_error", "detail": str(exc)}, status=500)
        except Exception as exc:
            logging.getLogger("web_dashboard").exception("web_dashboard_request_error")
            self._send_json({"error": "internal_error", "detail": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        logging.getLogger("web_dashboard.http").debug(format, *args)

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _dashboard_html(refresh_ms: int) -> str:
    return _compact_dashboard_html(refresh_ms)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polymarket BTC 5m Recorder Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --blue: #2563eb;
      --green: #059669;
      --red: #dc2626;
      --amber: #b45309;
      --ink: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    main {{
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 16px;
      display: grid;
      gap: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .wide {{ grid-column: 1 / -1; }}
    h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      color: var(--ink);
      overflow-wrap: anywhere;
    }}
    .kv {{
      display: grid;
      grid-template-columns: minmax(100px, 0.44fr) minmax(0, 1fr);
      gap: 6px 10px;
    }}
    .kv span:nth-child(odd), .muted {{ color: var(--muted); }}
    canvas {{
      width: 100%;
      height: 330px;
      display: block;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      padding: 8px 6px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .warnings {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .warning {{
      color: var(--amber);
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 6px;
      padding: 4px 8px;
      overflow-wrap: anywhere;
    }}
    .ok {{ color: var(--green); }}
    .stale {{
      margin-top: 8px;
      color: var(--red);
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      margin: 8px 0 0;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 700px) {{
      header {{ display: block; }}
      .grid {{ grid-template-columns: 1fr; }}
      .kv {{ grid-template-columns: 1fr; }}
      canvas {{ height: 280px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Polymarket BTC 5m Recorder Dashboard</h1>
    <div class="muted" id="status">loading</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel">
        <h2>Active Market</h2>
        <div class="value" id="market-id">none</div>
        <div class="kv" id="market-card"></div>
      </div>
      <div class="panel">
        <h2>BTC Price</h2>
        <div class="value" id="btc-price">none</div>
        <div class="kv" id="btc-card"></div>
        <div class="stale" id="btc-stale-warning" hidden>BTC sample stale</div>
      </div>
      <div class="panel">
        <h2>YES Top Of Book</h2>
        <div class="value" id="yes-bid">none</div>
        <div class="kv" id="yes-card"></div>
      </div>
      <div class="panel">
        <h2>NO Top Of Book</h2>
        <div class="value" id="no-bid">none</div>
        <div class="kv" id="no-card"></div>
      </div>
      <div class="panel">
        <h2>Recorder Health</h2>
        <div class="value" id="health-main">none</div>
        <div class="kv" id="health-card"></div>
      </div>
    </section>

    <section class="panel wide">
      <h2>Live Series</h2>
      <canvas id="chart" width="1400" height="330"></canvas>
      <div class="legend">
        <span><span class="dot" style="background: var(--blue)"></span>BTC price</span>
        <span><span class="dot" style="background: var(--green)"></span>YES best bid</span>
        <span><span class="dot" style="background: var(--red)"></span>NO best bid</span>
      </div>
    </section>

    <section class="panel wide">
      <h2>Warnings</h2>
      <div class="warnings" id="warnings"><span class="ok">none</span></div>
    </section>

    <section class="panel wide">
      <h2>Recent Trades</h2>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Side</th>
            <th>Price</th>
            <th>Size</th>
            <th>Outcome</th>
            <th>Asset</th>
          </tr>
        </thead>
        <tbody id="trades"><tr><td colspan="6">none</td></tr></tbody>
      </table>
    </section>
  </main>
  <script>
    const REFRESH_MS = {int(refresh_ms)};
    const WINDOW_SEC = 300;

    function text(value) {{
      if (value === null || value === undefined || value === '') return 'none';
      return String(value);
    }}
    function num(value, digits = 4) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'none';
      return Number(value).toFixed(digits);
    }}
    function kv(target, pairs) {{
      target.innerHTML = '';
      for (const [key, value] of pairs) {{
        const k = document.createElement('span');
        const v = document.createElement('span');
        k.textContent = key;
        v.textContent = text(value);
        target.append(k, v);
      }}
    }}
    async function loadJson(path) {{
      const response = await fetch(path, {{cache: 'no-store'}});
      if (!response.ok) throw new Error(`${{path}} ${{response.status}}`);
      return response.json();
    }}
    function updateCards(data) {{
      const market = data.active_market || {{}};
      const btc = data.btc_price || {{}};
      const snap = data.latest_snapshot || {{}};
      const yes = (data.best_bid_ask || {{}}).YES || {{}};
      const no = (data.best_bid_ask || {{}}).NO || {{}};
      const health = data.health || {{}};

      document.getElementById('market-id').textContent = text(market.market_id);
      kv(document.getElementById('market-card'), [
        ['question', market.question],
        ['start', market.start_time],
        ['close', market.close_time],
        ['phase', market.phase],
        ['tracking', market.tracking_state],
        ['YES token', market.yes_token_id],
        ['NO token', market.no_token_id],
      ]);

      document.getElementById('btc-price').textContent = num(btc.price, 2);
      kv(document.getElementById('btc-card'), [
        ['source', btc.source],
        ['exchange ts', btc.exchange_timestamp],
        ['local arrival', btc.local_arrival_iso],
        ['sample age sec', num(btc.sample_age_sec, 3)],
      ]);
      const stale = document.getElementById('btc-stale-warning');
      stale.hidden = !(Number(btc.sample_age_sec) > 3);

      document.getElementById('yes-bid').textContent = num(yes.best_bid ?? snap.best_bid_yes);
      kv(document.getElementById('yes-card'), [
        ['best bid', yes.best_bid ?? snap.best_bid_yes],
        ['best ask', yes.best_ask ?? snap.best_ask_yes],
        ['spread', yes.spread ?? snap.spread_yes],
        ['updated', yes.timestamp ?? snap.timestamp],
      ]);

      document.getElementById('no-bid').textContent = num(no.best_bid ?? snap.best_bid_no);
      kv(document.getElementById('no-card'), [
        ['best bid', no.best_bid ?? snap.best_bid_no],
        ['best ask', no.best_ask ?? snap.best_ask_no],
        ['spread', no.spread ?? snap.spread_no],
        ['updated', no.timestamp ?? snap.timestamp],
      ]);

      document.getElementById('health-main').textContent = text(health.timestamp);
      kv(document.getElementById('health-card'), [
        ['raw seen', health.raw_ws_events_seen],
        ['raw written', health.raw_ws_events_written],
        ['malformed', health.malformed_ws_events],
        ['write failures', health.raw_ws_write_failures],
        ['last ws age', health.last_ws_event_age_sec],
        ['subscribed', health.subscribed_asset_count],
        ['reconnects', health.ws_reconnect_count],
      ]);

      const warnings = document.getElementById('warnings');
      warnings.innerHTML = '';
      if (!data.warnings || data.warnings.length === 0) {{
        const ok = document.createElement('span');
        ok.className = 'ok';
        ok.textContent = 'none';
        warnings.append(ok);
      }} else {{
        for (const warning of data.warnings) {{
          const el = document.createElement('span');
          el.className = 'warning';
          el.textContent = warning;
          warnings.append(el);
        }}
      }}

      const trades = document.getElementById('trades');
      trades.innerHTML = '';
      if (!data.recent_trades || data.recent_trades.length === 0) {{
        trades.innerHTML = '<tr><td colspan="6">none</td></tr>';
      }} else {{
        for (const trade of data.recent_trades) {{
          const row = document.createElement('tr');
          for (const value of [
            trade.timestamp,
            trade.side,
            trade.price,
            trade.size,
            trade.outcome,
            trade.asset_id,
          ]) {{
            const cell = document.createElement('td');
            cell.textContent = text(value);
            row.append(cell);
          }}
          trades.append(row);
        }}
      }}
    }}
    function drawSeries(series) {{
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#d8dee8';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const y = 24 + i * ((height - 52) / 4);
        ctx.beginPath();
        ctx.moveTo(48, y);
        ctx.lineTo(width - 18, y);
        ctx.stroke();
      }}

      const btc = (series.btc_price || []).map(p => [Date.parse(p.timestamp), Number(p.value)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const yes = (series.yes_best_bid || []).map(p => [Date.parse(p.timestamp), Number(p.value)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const no = (series.no_best_bid || []).map(p => [Date.parse(p.timestamp), Number(p.value)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const allTimes = [...btc, ...yes, ...no].map(p => p[0]);
      if (allTimes.length === 0) {{
        ctx.fillStyle = '#667085';
        ctx.fillText('No series data', 56, 50);
        return;
      }}
      const minT = Math.min(...allTimes);
      const maxT = Math.max(...allTimes);
      const plot = {{x0: 48, y0: 22, x1: width - 18, y1: height - 30}};
      const x = t => plot.x0 + ((t - minT) / Math.max(1, maxT - minT)) * (plot.x1 - plot.x0);
      function drawLine(points, color, minY, maxY) {{
        if (points.length === 0) return;
        const y = value => plot.y1 - ((value - minY) / Math.max(0.000001, maxY - minY)) * (plot.y1 - plot.y0);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((point, idx) => {{
          const px = x(point[0]);
          const py = y(point[1]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        ctx.stroke();
      }}
      const btcVals = btc.map(p => p[1]);
      const probVals = [...yes, ...no].map(p => p[1]);
      drawLine(btc, '#2563eb', Math.min(...btcVals), Math.max(...btcVals));
      drawLine(yes, '#059669', Math.min(...probVals), Math.max(...probVals));
      drawLine(no, '#dc2626', Math.min(...probVals), Math.max(...probVals));
      ctx.fillStyle = '#667085';
      ctx.fillText(`BTC rows: ${{btc.length}} | YES rows: ${{yes.length}} | NO rows: ${{no.length}}`, 56, height - 10);
    }}
    async function refresh() {{
      try {{
        const [dashboard, series] = await Promise.all([
          loadJson('/api/dashboard'),
          loadJson(`/api/series?window_sec=${{WINDOW_SEC}}`),
        ]);
        updateCards(dashboard);
        drawSeries(series);
        document.getElementById('status').textContent = `refreshed ${{dashboard.now}}`;
      }} catch (err) {{
        document.getElementById('status').textContent = `error: ${{err.message}}`;
      }}
    }}
    refresh();
    setInterval(refresh, REFRESH_MS);
  </script>
</body>
</html>
"""


def _compact_dashboard_html(refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polymarket BTC 5m Recorder Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef2f6;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --text: #101828;
      --muted: #667085;
      --line: #d0d7e2;
      --blue: #1d4ed8;
      --green: #047857;
      --red: #b91c1c;
      --amber: #b45309;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0; font-size: 16px; line-height: 1.2; letter-spacing: 0; }}
    main {{ width: min(1680px, 100%); margin: 0 auto; padding: 12px; }}
    .tabs {{ display: flex; gap: 6px; overflow-x: auto; padding: 0 0 10px; }}
    .tab-btn {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 6px 9px;
      font: inherit;
      white-space: nowrap;
      cursor: pointer;
    }}
    .tab-btn.active {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
    .tab {{ display: none; }}
    .tab.active {{ display: block; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 9px;
      min-width: 0;
    }}
    .card h2 {{
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .metric {{ display: flex; align-items: baseline; gap: 7px; min-width: 0; }}
    .value {{ font-size: 19px; font-weight: 750; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .sub {{ color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 700;
      background: #e5e7eb;
      color: #374151;
    }}
    .status-OK, .status-ACTIVE {{ background: #dcfce7; color: var(--green); }}
    .status-STALE, .status-WARN {{ background: #fef3c7; color: var(--amber); }}
    .status-ERROR {{ background: #fee2e2; color: var(--red); }}
    .status-NO_TRADES {{ background: #e0e7ff; color: var(--blue); }}
    details {{ margin-top: 7px; }}
    summary {{ color: var(--blue); cursor: pointer; font-size: 12px; user-select: none; }}
    .kv {{ display: grid; grid-template-columns: minmax(96px, 0.42fr) minmax(0, 1fr); gap: 4px 8px; margin-top: 6px; }}
    .kv span:nth-child(odd) {{ color: var(--muted); }}
    .kv span:nth-child(even) {{ overflow-wrap: anywhere; }}
    .section {{ display: grid; gap: 10px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 10px;
      min-width: 0;
    }}
    .panel-title {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .panel-title h2 {{ margin: 0; font-size: 13px; text-transform: uppercase; color: var(--muted); letter-spacing: 0; }}
    .trader-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 6px 5px;
      text-align: left;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 12px;
    }}
    th {{ color: var(--muted); font-weight: 700; background: var(--panel-2); position: sticky; top: 45px; }}
    .table-wrap {{ overflow-x: auto; max-height: 560px; border: 1px solid var(--line); border-radius: 8px; }}
    .warnings {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .warning {{ color: var(--amber); background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 3px 7px; }}
    .ok {{ color: var(--green); }}
    .stale {{ margin-top: 4px; color: var(--red); font-weight: 700; }}
    canvas {{ width: 100%; height: 230px; display: block; background: #fff; border: 1px solid var(--line); border-radius: 8px; }}
    @media (max-width: 1280px) {{ .summary-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} .trader-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 760px) {{ header {{ display: block; }} .summary-grid, .trader-grid {{ grid-template-columns: 1fr; }} .kv {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Polymarket BTC 5m Live Cockpit</h1>
    <div class="sub" id="status">loading</div>
  </header>
  <main>
    <nav class="tabs" id="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="paper">Paper Traders</button>
      <button class="tab-btn" data-tab="activity">Live Activity</button>
      <button class="tab-btn" data-tab="skips">Skips</button>
      <button class="tab-btn" data-tab="open">Open Trades</button>
      <button class="tab-btn" data-tab="closed">Closed/Cashouts</button>
      <button class="tab-btn" data-tab="recorder">Recorder Details</button>
    </nav>

    <section id="tab-overview" class="tab active section">
      <div class="summary-grid">
        <article class="card" id="market-card"></article>
        <article class="card" id="btc-card"></article>
        <article class="card" id="yes-card"></article>
        <article class="card" id="no-card"></article>
        <article class="card" id="health-card"></article>
        <article class="card" id="paper-summary-card"></article>
      </div>
      <div class="panel">
        <div class="panel-title"><h2>Live Series</h2><span class="sub">BTC, YES bid, NO bid</span></div>
        <canvas id="chart" width="1400" height="230"></canvas>
      </div>
      <div class="panel">
        <div class="panel-title"><h2>Warnings</h2></div>
        <div class="warnings" id="warnings"><span class="ok">none</span></div>
      </div>
    </section>

    <section id="tab-paper" class="tab section">
      <div class="trader-grid" id="paper-trader-cards"></div>
    </section>
    <section id="tab-activity" class="tab section"><div class="panel"><div class="panel-title"><h2>Recent Activity</h2></div><div class="table-wrap"><table id="activity-table"></table></div></div></section>
    <section id="tab-skips" class="tab section"><div class="panel"><div class="panel-title"><h2>Skip Groups</h2></div><div id="skip-groups" class="trader-grid"></div></div><div class="panel"><div class="panel-title"><h2>Recent Skips</h2></div><div class="table-wrap"><table id="skips-table"></table></div></div></section>
    <section id="tab-open" class="tab section"><div class="panel"><div class="panel-title"><h2>Open / Awaiting</h2></div><div class="table-wrap"><table id="open-table"></table></div></div></section>
    <section id="tab-closed" class="tab section"><div class="panel"><div class="panel-title"><h2>Closed / Settled</h2></div><div class="table-wrap"><table id="closed-table"></table></div></div></section>
    <section id="tab-recorder" class="tab section"><div class="panel"><div class="panel-title"><h2>Recorder Raw Details</h2></div><div id="recorder-details" class="kv"></div></div></section>
  </main>
  <script>
    const REFRESH_MS = {int(refresh_ms)};
    const WINDOW_SEC = 300;
    const tableColumns = {{
      activity: [
        ['activity_time', 'time'], ['trader_label', 'trader'], ['market_id', 'market'],
        ['strategy_id', 'strategy'], ['signal_direction', 'dir'], ['status', 'status'],
        ['stake_usd', 'stake'], ['entry_price', 'entry'], ['adjusted_entry_price', 'adj entry'],
        ['cashout_exit_price', 'exit'], ['pnl', 'pnl'], ['roi_combined', 'roi'],
        ['probability_for_direction', 'prob'], ['estimated_edge', 'edge'],
        ['time_until_resolution', 't-res'], ['bankroll_after_trade', 'bankroll'],
        ['skip_reason', 'skip'], ['realistic_execution_skip_reason', 'realism'],
        ['liquidity_skip_reason', 'liquidity']
      ],
      open: [
        ['trader_label', 'trader'], ['created_at', 'created'], ['market_id', 'market'],
        ['strategy_id', 'strategy'], ['signal_direction', 'dir'], ['status', 'status'],
        ['stake_usd', 'stake'], ['entry_price', 'entry'], ['adjusted_entry_price', 'adj entry'],
        ['time_until_resolution', 't-res'], ['probability_for_direction', 'prob'],
        ['estimated_edge', 'edge'], ['bankroll_before_trade', 'bankroll before'],
        ['open_exposure_before_trade', 'open exposure']
      ],
      closed: [
        ['trader_label', 'trader'], ['created_at', 'entry time'], ['exit_time', 'exit time'],
        ['settled_at', 'settled'], ['strategy_id', 'strategy'], ['signal_direction', 'dir'],
        ['status', 'status'], ['stake_usd', 'stake'], ['entry_price', 'entry'],
        ['cashout_exit_price', 'exit/payout'], ['pnl', 'pnl'], ['roi_combined', 'roi'],
        ['win_loss', 'result']
      ]
    }};
    function text(value) {{ return value === null || value === undefined || value === '' ? '—' : String(value); }}
    function num(value, digits = 4) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed.toFixed(digits) : '—';
    }}
    function money(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? `${{parsed < 0 ? '-' : ''}}$${{Math.abs(parsed).toFixed(2)}}` : '—';
    }}
    function pct(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? `${{(parsed * 100).toFixed(1)}}%` : '—';
    }}
    function kvHtml(pairs) {{
      return `<div class="kv">${{pairs.map(([k, v]) => `<span>${{k}}</span><span title="${{text(v)}}">${{text(v)}}</span>`).join('')}}</div>`;
    }}
    function details(title, pairs) {{ return `<details><summary>${{title}}</summary>${{kvHtml(pairs)}}</details>`; }}
    async function loadJson(path) {{
      const response = await fetch(path, {{cache: 'no-store'}});
      if (!response.ok) throw new Error(`${{path}} ${{response.status}}`);
      return response.json();
    }}
    function renderCard(id, html) {{ document.getElementById(id).innerHTML = html; }}
    function renderRecorder(rec) {{
      const market = rec.active_market || {{}};
      const btc = rec.btc || {{}};
      const yes = rec.yes || {{}};
      const no = rec.no || {{}};
      const health = rec.recorder_health || {{}};
      renderCard('market-card', `
        <h2>Active Market</h2><div class="metric"><span class="value">${{text(market.market_id)}}</span></div>
        <div class="sub">${{text(market.short_question)}}</div>
        <div class="sub">${{num(market.time_left_sec, 0)}}s left · ${{text(market.phase)}} · ${{text(market.tracking_state)}}</div>
        ${{details('details', [['start', market.start_time], ['close', market.close_time], ['YES token', market.yes_token_id], ['NO token', market.no_token_id], ['question', market.question]])}}
      `);
      renderCard('btc-card', `
        <h2>BTC Price</h2><div class="metric"><span class="value">${{money(btc.price)}}</span><span class="status-pill ${{Number(btc.sample_age_sec) > 3 ? 'status-STALE' : 'status-OK'}}">${{num(btc.sample_age_sec, 1)}}s</span></div>
        <div class="sub">${{text(btc.source)}}</div><div class="stale" id="btc-stale-warning" ${{Number(btc.sample_age_sec) > 3 ? '' : 'hidden'}}>BTC sample stale</div>
        ${{details('source details', [['sample age sec', num(btc.sample_age_sec, 3)], ['exchange timestamp', btc.exchange_timestamp], ['local arrival', btc.local_arrival_iso], ['sources', (rec.btc_prices_by_source || []).map(x => `${{x.source}} ${{money(x.price)}} ${{num(x.sample_age_sec, 1)}}s`).join(' | ')]])}}
      `);
      renderCard('yes-card', `<h2>YES Price</h2><div class="metric"><span class="value">${{num(yes.mid_price, 3)}}</span></div><div class="sub">bid ${{num(yes.best_bid, 3)}} · ask ${{num(yes.best_ask, 3)}} · spread ${{num(yes.spread, 3)}}</div>${{details('details', [['updated', yes.timestamp], ['age sec', num(yes.age_sec, 2)]])}}`);
      renderCard('no-card', `<h2>NO Price</h2><div class="metric"><span class="value">${{num(no.mid_price, 3)}}</span></div><div class="sub">bid ${{num(no.best_bid, 3)}} · ask ${{num(no.best_ask, 3)}} · spread ${{num(no.spread, 3)}}</div>${{details('details', [['updated', no.timestamp], ['age sec', num(no.age_sec, 2)]])}}`);
      renderCard('health-card', `<h2>Recorder Health</h2><div class="metric"><span class="value">${{text(health.status)}}</span><span class="status-pill status-${{text(health.status)}}">${{text(health.status)}}</span></div><div class="sub">snap ${{num(health.snapshot_age_sec, 1)}}s · feature ${{num(health.feature_age_sec, 1)}}s · BTC ${{num(health.btc_age_sec, 1)}}s</div>${{details('raw metrics', [['WS age', health.ws_event_age_sec], ['raw seen', health.raw_ws_events_seen], ['raw written', health.raw_ws_events_written], ['malformed', health.malformed_ws_events], ['write failures', health.raw_ws_write_failures], ['subscribed assets', health.subscribed_asset_count], ['reconnect count', health.ws_reconnect_count]])}}`);
      const warnings = document.getElementById('warnings');
      warnings.innerHTML = (rec.warnings || []).length ? rec.warnings.map(w => `<span class="warning">${{w}}</span>`).join('') : '<span class="ok">none</span>';
      document.getElementById('recorder-details').innerHTML = kvHtml(Object.entries(health));
    }}
    function renderPaper(summary) {{
      const totals = summary.totals || {{}};
      renderCard('paper-summary-card', `<h2>Paper Trader Summary</h2><div class="metric"><span class="value">${{money(totals.realized_pnl)}}</span><span class="status-pill">${{text(totals.paper_db_count)}} DBs</span></div><div class="sub">${{text(totals.total_trades)}} rows · open/awaiting ${{text(totals.open_awaiting_trades)}} · skipped ${{text(totals.skipped_trades)}}</div>`);
      const target = document.getElementById('paper-trader-cards');
      if (!summary.traders || summary.traders.length === 0) {{ target.innerHTML = '<div class="panel">No paper DBs configured.</div>'; return; }}
      target.innerHTML = summary.traders.map(t => `
        <article class="card">
          <h2>${{text(t.label)}}</h2>
          <div class="metric"><span class="value">${{money(t.realized_pnl)}}</span><span class="status-pill status-${{text(t.status)}}">${{text(t.status)}}</span></div>
          <div class="sub">${{text(t.total_trades)}} rows · open/awaiting ${{text(t.open_awaiting_trades)}} · closed ${{text(t.closed_trades)}} · settled ${{text(t.settled_trades)}} · skipped ${{text(t.skipped_trades)}}</div>
          <div class="sub">win ${{pct(t.win_rate)}} · avg ROI ${{pct(t.avg_roi)}} · avg stake ${{money(t.avg_stake)}}</div>
          <div class="sub">bankroll ${{money(t.current_bankroll)}} / start ${{money(t.starting_bankroll)}} · ROI ${{pct(t.bankroll_roi)}} · exposure ${{money(t.open_exposure)}} / ${{money(t.max_exposure_allowed)}}</div>
          ${{details('strategy / skip details', [['latest', t.latest_activity_time], ['top strategies', (t.top_strategies_by_pnl || []).map(x => `${{x.strategy_id}} ${{money(x.pnl)}}`).join(' | ')], ['bottom strategies', (t.bottom_strategies_by_pnl || []).map(x => `${{x.strategy_id}} ${{money(x.pnl)}}`).join(' | ')], ['skip reasons', JSON.stringify(t.skip_reason_counts || {{}})], ['liquidity blocks', JSON.stringify(t.liquidity_block_counts || {{}})], ['realistic blocks', JSON.stringify(t.realistic_execution_block_counts || {{}})]])}}
        </article>`).join('');
    }}
    function renderTable(id, rows, cols) {{
      const table = document.getElementById(id);
      table.innerHTML = `<thead><tr>${{cols.map(([, label]) => `<th>${{label}}</th>`).join('')}}</tr></thead><tbody></tbody>`;
      const body = table.querySelector('tbody');
      if (!rows || rows.length === 0) {{ body.innerHTML = `<tr><td colspan="${{cols.length}}">none</td></tr>`; return; }}
      body.innerHTML = rows.map(row => `<tr>${{cols.map(([key]) => `<td title="${{text(row[key])}}">${{formatCell(key, row[key])}}</td>`).join('')}}</tr>`).join('');
    }}
    function formatCell(key, value) {{
      if (['pnl', 'stake_usd', 'entry_price', 'adjusted_entry_price', 'cashout_exit_price', 'bankroll_after_trade', 'bankroll_before_trade', 'open_exposure_before_trade'].includes(key)) return num(value, 4);
      if (['roi_combined', 'probability_for_direction', 'estimated_edge'].includes(key)) return num(value, 4);
      return text(value);
    }}
    function renderSkips(skips) {{
      const groups = skips.groups || {{}};
      const target = document.getElementById('skip-groups');
      target.innerHTML = Object.entries(groups).map(([name, values]) => `<div class="card"><h2>${{name}}</h2>${{kvHtml(Object.entries(values || {{}}).slice(0, 10))}}</div>`).join('') || '<div class="panel">No skips.</div>';
      renderTable('skips-table', skips.rows || [], tableColumns.activity);
    }}
    function drawSeries(series) {{
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
      const width = canvas.width, height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#d0d7e2'; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{ const y = 20 + i * ((height - 45) / 4); ctx.beginPath(); ctx.moveTo(42, y); ctx.lineTo(width - 14, y); ctx.stroke(); }}
      const map = rows => (rows || []).map(p => [Date.parse(p.timestamp), Number(p.value)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      const btc = map(series.btc_price), yes = map(series.yes_best_bid), no = map(series.no_best_bid);
      const allTimes = [...btc, ...yes, ...no].map(p => p[0]);
      if (!allTimes.length) {{ ctx.fillStyle = '#667085'; ctx.fillText('No series data', 50, 45); return; }}
      const minT = Math.min(...allTimes), maxT = Math.max(...allTimes);
      const plot = {{x0: 42, y0: 16, x1: width - 14, y1: height - 26}};
      const x = t => plot.x0 + ((t - minT) / Math.max(1, maxT - minT)) * (plot.x1 - plot.x0);
      function line(points, color, minY, maxY) {{ if (!points.length) return; const y = v => plot.y1 - ((v - minY) / Math.max(0.000001, maxY - minY)) * (plot.y1 - plot.y0); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(x(p[0]), y(p[1])) : ctx.moveTo(x(p[0]), y(p[1]))); ctx.stroke(); }}
      const btcVals = btc.map(p => p[1]), probVals = [...yes, ...no].map(p => p[1]);
      line(btc, '#1d4ed8', Math.min(...btcVals), Math.max(...btcVals));
      line(yes, '#047857', Math.min(...probVals), Math.max(...probVals));
      line(no, '#b91c1c', Math.min(...probVals), Math.max(...probVals));
    }}
    async function refresh() {{
      try {{
        const [rec, paper, activity, skips, open, closed, series] = await Promise.all([
          loadJson('/api/recorder/summary'),
          loadJson('/api/paper-traders/summary'),
          loadJson('/api/paper-traders/activity?limit=100'),
          loadJson('/api/paper-traders/skips?limit=100'),
          loadJson('/api/paper-traders/open'),
          loadJson('/api/paper-traders/closed?limit=100'),
          loadJson(`/api/series?window_sec=${{WINDOW_SEC}}`)
        ]);
        renderRecorder(rec); renderPaper(paper); renderSkips(skips); drawSeries(series);
        renderTable('activity-table', activity.rows || [], tableColumns.activity);
        renderTable('open-table', open.rows || [], tableColumns.open);
        renderTable('closed-table', closed.rows || [], tableColumns.closed);
        document.getElementById('status').textContent = `refreshed ${{rec.now}}`;
      }} catch (err) {{
        document.getElementById('status').textContent = `error: ${{err.message}}`;
      }}
    }}
    document.getElementById('tabs').addEventListener('click', event => {{
      const btn = event.target.closest('.tab-btn'); if (!btn) return;
      document.querySelectorAll('.tab-btn').forEach(x => x.classList.toggle('active', x === btn));
      document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.id === `tab-${{btn.dataset.tab}}`));
    }});
    refresh();
    setInterval(refresh, REFRESH_MS);
  </script>
</body>
</html>
"""


def _paper_trader_summary(path: str, *, now: datetime) -> dict[str, Any]:
    label = Path(path).name
    base = {
        "paper_db_path": path,
        "label": label,
        "status": "ERROR",
        "error": None,
    }
    try:
        conn = connect_read_only(path)
    except FileNotFoundError:
        return {**base, "error": "paper_db_missing"}
    except sqlite3.Error as exc:
        return {**base, "error": f"sqlite_error: {exc}"}
    try:
        if not _table_exists(conn, "paper_trades"):
            return {**base, "error": "paper_trades_table_missing"}
        columns = _table_columns(conn, "paper_trades")
        rows = _paper_trade_rows(conn, columns)
    except sqlite3.Error as exc:
        return {**base, "error": f"sqlite_error: {exc}"}
    finally:
        conn.close()

    if not rows:
        return {
            **base,
            "status": "NO_TRADES",
            "total_trades": 0,
            "open_awaiting_trades": 0,
            "closed_trades": 0,
            "settled_trades": 0,
            "skipped_trades": 0,
            "realized_pnl": 0.0,
            "avg_roi": None,
            "win_rate": None,
            "avg_stake": None,
            "starting_bankroll": None,
            "current_bankroll": None,
            "bankroll_roi": None,
            "open_exposure": 0.0,
            "max_exposure_allowed": None,
            "latest_activity_time": None,
        }

    status_counts = _count_field(rows, "status")
    done_rows = [row for row in rows if str(row.get("status") or "") in {"closed", "settled"}]
    non_skipped = [row for row in rows if str(row.get("status") or "") != "skipped"]
    pnl_total = sum(_combined_pnl(row) for row in rows)
    roi_values = [
        value
        for value in (_coalesce_number(row, "roi", "realized_roi") for row in done_rows)
        if value is not None
    ]
    wins = sum(1 for row in done_rows if _combined_pnl(row) > 0)
    stakes = [value for value in (_float_or_none(row.get("stake_usd")) for row in non_skipped) if value is not None]
    latest_time = max((_activity_time(row) for row in rows), default=None)
    latest_age = (now - latest_time).total_seconds() if latest_time is not None else None
    starting_bankroll = _first_non_null_number(rows, "starting_bankroll_usd")
    current_bankroll = _latest_non_null_number_by_created(rows, "bankroll_after_trade")
    if current_bankroll is None and starting_bankroll is not None:
        current_bankroll = starting_bankroll + pnl_total
    open_rows = [
        row
        for row in rows
        if str(row.get("status") or "") in {"open", "awaiting_resolution"}
    ]
    open_exposure = sum(_float_or_none(row.get("stake_usd")) or 0.0 for row in open_rows)
    max_exposure = _latest_non_null_number(rows, "max_open_exposure_usd")
    health_status = "ACTIVE"
    if latest_age is None:
        health_status = "NO_TRADES"
    elif latest_age > 60:
        health_status = "STALE"
    return {
        **base,
        "status": health_status,
        "error": None,
        "total_trades": len(rows),
        "open_trades": int(status_counts.get("open", 0)),
        "awaiting_resolution_trades": int(status_counts.get("awaiting_resolution", 0)),
        "open_awaiting_trades": int(status_counts.get("open", 0))
        + int(status_counts.get("awaiting_resolution", 0)),
        "closed_trades": int(status_counts.get("closed", 0)),
        "settled_trades": int(status_counts.get("settled", 0)),
        "skipped_trades": int(status_counts.get("skipped", 0)),
        "status_counts": status_counts,
        "realized_pnl": _round_or_none(pnl_total),
        "avg_roi": _mean(roi_values),
        "win_rate": _round_or_none(wins / len(done_rows)) if done_rows else None,
        "avg_stake": _mean(stakes),
        "starting_bankroll": _round_or_none(starting_bankroll),
        "current_bankroll": _round_or_none(current_bankroll),
        "bankroll_roi": (
            _round_or_none((current_bankroll - starting_bankroll) / starting_bankroll)
            if current_bankroll is not None and starting_bankroll not in (None, 0)
            else None
        ),
        "open_exposure": _round_or_none(open_exposure),
        "max_exposure_allowed": _round_or_none(max_exposure),
        "latest_activity_time": to_iso(latest_time) if latest_time else None,
        "latest_activity_age_sec": _round_or_none(latest_age),
        "top_strategies_by_pnl": _strategy_pnl(rows, reverse=True),
        "bottom_strategies_by_pnl": _strategy_pnl(rows, reverse=False),
        "skip_reason_counts": _count_field(
            [row for row in rows if row.get("status") == "skipped"],
            "skip_reason",
        ),
        "liquidity_block_counts": _count_field(rows, "liquidity_skip_reason"),
        "realistic_execution_block_counts": _count_field(
            rows,
            "realistic_execution_skip_reason",
        ),
        "regime_performance": _regime_performance(rows),
    }


def _paper_rows_for_dashboard(
    path: str,
    *,
    limit: int,
    status_in: Sequence[str] | None,
    now: datetime,
) -> list[dict[str, Any]]:
    try:
        conn = connect_read_only(path)
    except (FileNotFoundError, sqlite3.Error) as exc:
        return [
            {
                "paper_db_path": path,
                "trader_label": Path(path).name,
                "status": "ERROR",
                "error": str(exc),
                "activity_time": now.isoformat(),
            }
        ]
    try:
        if not _table_exists(conn, "paper_trades"):
            return []
        columns = _table_columns(conn, "paper_trades")
        rows = _paper_trade_rows(
            conn,
            columns,
            limit=max(1, int(limit)),
            status_in=status_in,
        )
    finally:
        conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        activity_time = _activity_time(item)
        item["paper_db_path"] = path
        item["trader_label"] = Path(path).name
        item["activity_time"] = to_iso(activity_time) if activity_time else None
        item["pnl"] = _round_or_none(_combined_pnl(item))
        item["roi_combined"] = _round_or_none(_coalesce_number(item, "roi", "realized_roi"))
        item["cashout_exit_price"] = _coalesce_number(
            item,
            "adjusted_exit_price",
            "exit_price",
            "payout_usd",
        )
        item["win_loss"] = (
            "win"
            if item["pnl"] is not None and float(item["pnl"]) > 0
            else "loss"
            if item["pnl"] is not None and str(item.get("status") or "") in {"closed", "settled"}
            else None
        )
        result.append(item)
    result.sort(key=lambda row: _sort_timestamp(row.get("activity_time")), reverse=True)
    return result


def _paper_trade_rows(
    conn: sqlite3.Connection,
    columns: set[str],
    *,
    limit: int | None = None,
    status_in: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    select_sql = ", ".join(
        f'"{field}"' if field in columns else f"NULL AS {field}"
        for field in _PAPER_TRADE_FIELDS
    )
    where_sql = ""
    params: list[Any] = []
    if status_in:
        placeholders = ", ".join("?" for _ in status_in)
        where_sql = f"WHERE status IN ({placeholders})" if "status" in columns else "WHERE 0"
        params.extend(status_in)
    order_sql = "ORDER BY id DESC" if "id" in columns else "ORDER BY rowid DESC"
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(max(1, int(limit)))
    rows = conn.execute(
        f"SELECT {select_sql} FROM paper_trades {where_sql} {order_sql} {limit_sql}",
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _combined_pnl(row: dict[str, Any]) -> float:
    return _coalesce_number(row, "pnl_usd", "realized_pnl_usd") or 0.0


def _coalesce_number(row: dict[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _first_non_null_number(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    for row in rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _latest_non_null_number(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    sorted_rows = sorted(rows, key=lambda row: _activity_time(row) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for row in sorted_rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _created_time(row: dict[str, Any]) -> datetime | None:
    created = parse_timestamp(row.get("created_at"))
    if created is not None:
        return _utc(created)
    return _activity_time(row)


def _latest_non_null_number_by_created(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _created_time(row) or datetime.min.replace(tzinfo=timezone.utc),
            int(row.get("id") or 0),
        ),
        reverse=True,
    )
    for row in sorted_rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _activity_time(row: dict[str, Any]) -> datetime | None:
    candidates = [
        parse_timestamp(row.get(field))
        for field in ("exit_time", "settled_at", "created_at", "signal_timestamp")
    ]
    candidates = [_utc(value) for value in candidates if value is not None]
    return max(candidates) if candidates else None


def _sort_timestamp(value: Any) -> float:
    parsed = parse_timestamp(value)
    return _utc(parsed).timestamp() if parsed is not None else 0.0


def _count_field(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _strategy_pnl(rows: Sequence[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status") or "") == "skipped":
            continue
        strategy_id = str(row.get("strategy_id") or "unknown")
        item = grouped.setdefault(
            strategy_id,
            {"strategy_id": strategy_id, "trades": 0, "pnl": 0.0},
        )
        item["trades"] += 1
        item["pnl"] += _combined_pnl(row)
    values = list(grouped.values())
    values.sort(key=lambda item: (float(item["pnl"]), int(item["trades"])), reverse=reverse)
    for item in values:
        item["pnl"] = _round_or_none(item["pnl"])
    return values[:5]


def _regime_performance(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "btc_trend_regime",
        "volatility_regime",
        "spread_regime",
        "liquidity_regime",
        "time_regime",
    ):
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            regime = str(row.get(field) or "unknown")
            item = grouped.setdefault(regime, {"regime": regime, "trades": 0, "pnl": 0.0})
            item["trades"] += 1
            item["pnl"] += _combined_pnl(row)
        values = list(grouped.values())
        values.sort(key=lambda item: (-abs(float(item["pnl"])), item["regime"]))
        for item in values:
            item["pnl"] = _round_or_none(item["pnl"])
        result[field] = values[:8]
    return result


def _mean(values: Sequence[float]) -> float | None:
    return _round_or_none(sum(values) / len(values)) if values else None


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    parsed = _float_or_none(value)
    return round(parsed, digits) if parsed is not None else None


def _short_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _time_left_sec(close_time: Any, *, now: datetime) -> float | None:
    close = parse_timestamp(close_time)
    if close is None:
        return None
    return _round_or_none(max(0.0, (_utc(close) - now).total_seconds()))


def _age_from_timestamp(timestamp: Any, *, now: datetime) -> float | None:
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return None
    return _round_or_none(max(0.0, (now - _utc(parsed)).total_seconds()))


def _price_summary(
    best_row: dict[str, Any] | None,
    snapshot: dict[str, Any],
    *,
    side: str,
    now: datetime,
) -> dict[str, Any]:
    prefix = "yes" if side == "yes" else "no"
    bid = _coalesce_number(best_row or {}, "best_bid")
    ask = _coalesce_number(best_row or {}, "best_ask")
    spread = _coalesce_number(best_row or {}, "spread")
    timestamp = (best_row or {}).get("timestamp")
    if bid is None:
        bid = _float_or_none(snapshot.get(f"best_bid_{prefix}"))
    if ask is None:
        ask = _float_or_none(snapshot.get(f"best_ask_{prefix}"))
    if spread is None:
        spread = _float_or_none(snapshot.get(f"spread_{prefix}"))
    if timestamp is None:
        timestamp = snapshot.get("timestamp")
    mid = (bid + ask) / 2 if bid is not None and ask is not None else bid or ask
    return {
        "mid_price": _round_or_none(mid),
        "best_bid": _round_or_none(bid),
        "best_ask": _round_or_none(ask),
        "spread": _round_or_none(spread),
        "timestamp": timestamp,
        "age_sec": _age_from_timestamp(timestamp, now=now),
    }


def _fetch_active_market(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not _table_exists(conn, "markets"):
        return None
    row = conn.execute(
        """
        SELECT run_id, market_id
        FROM markets
        WHERE tracking_state = 'selected_active'
        ORDER BY datetime(start_time) DESC, datetime(last_updated) DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row is not None else None


def _fetch_btc_series(
    conn: sqlite3.Connection,
    *,
    cutoff_iso: str,
    source: str | None = None,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "btc_prices"):
        return []
    params: list[object] = [cutoff_iso]
    source_sql = ""
    if source:
        source_sql = "AND source = ?"
        params.append(source)
    rows = conn.execute(
        f"""
        SELECT local_arrival_iso AS timestamp, price AS value, source, local_arrival_ns
        FROM btc_prices
        WHERE local_arrival_iso IS NOT NULL
          AND datetime(local_arrival_iso) >= datetime(?)
          {source_sql}
        ORDER BY local_arrival_ns ASC, datetime(local_arrival_iso) ASC, id ASC
        """,
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_latest_btc_source(conn: sqlite3.Connection) -> str | None:
    if not _table_exists(conn, "btc_prices"):
        return None
    row = conn.execute(
        """
        SELECT source
        FROM btc_prices
        WHERE source IS NOT NULL AND source <> ''
        ORDER BY local_arrival_ns DESC, datetime(local_arrival_iso) DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None or row["source"] is None:
        return None
    return str(row["source"])


def _fetch_latest_btc_prices_by_source(
    db_path: str,
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    conn = connect_read_only(db_path)
    try:
        if not _table_exists(conn, "btc_prices"):
            return []
        rows = conn.execute(
            """
            SELECT
              p.source,
              p.price,
              p.exchange_timestamp,
              p.local_arrival_ns,
              p.local_arrival_iso
            FROM btc_prices p
            JOIN (
              SELECT source, MAX(local_arrival_ns) AS max_arrival_ns
              FROM btc_prices
              GROUP BY source
            ) latest
              ON p.source = latest.source
             AND p.local_arrival_ns = latest.max_arrival_ns
            ORDER BY p.source ASC, p.id DESC
            """
        ).fetchall()
    finally:
        conn.close()

    latest: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in rows:
        item = dict(row)
        source = str(item.get("source") or "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        item["sample_age_sec"] = _btc_sample_age_sec(item, now=now)
        latest.append(item)
    return latest


def _fetch_snapshot_series(
    conn: sqlite3.Connection,
    *,
    cutoff_iso: str,
    run_id: str | None,
    market_id: str | None,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "market_snapshots") or market_id is None:
        return []
    params: list[object] = [market_id, cutoff_iso]
    run_sql = ""
    if run_id:
        run_sql = "AND run_id = ?"
        params.append(run_id)
    rows = conn.execute(
        f"""
        SELECT timestamp, best_bid_yes, best_bid_no
        FROM market_snapshots
        WHERE market_id = ?
          AND datetime(timestamp) >= datetime(?)
          {run_sql}
        ORDER BY datetime(timestamp) ASC
        """,
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def _value(row: dict[str, Any] | None, key: str) -> Any:
    return row.get(key) if row is not None else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _btc_sample_age_sec(
    btc_price: dict[str, Any] | None,
    *,
    now: datetime,
) -> float | None:
    if not isinstance(btc_price, dict):
        return None
    arrival = parse_timestamp(btc_price.get("local_arrival_iso"))
    if arrival is None:
        return None
    return max(0.0, (now - _utc(arrival)).total_seconds())


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (value != value):
        return None
    return value


__all__ = [
    "DEFAULT_WEB_DASHBOARD_HOST",
    "DEFAULT_WEB_DASHBOARD_PORT",
    "WebDashboardRequestHandler",
    "create_web_dashboard_server",
    "fetch_paper_activity_payload",
    "fetch_paper_closed_payload",
    "fetch_paper_open_payload",
    "fetch_paper_skips_payload",
    "fetch_paper_traders_summary",
    "fetch_recorder_summary_payload",
    "fetch_series_payload",
    "fetch_web_dashboard_payload",
    "resolve_paper_db_paths",
    "run_web_dashboard",
]
