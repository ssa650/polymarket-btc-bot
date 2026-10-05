from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from src.compact_paper_live_status import (
    build_compact_paper_live_status,
    render_compact_paper_live_status,
)


def _create_paper_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            status TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            rejection_reason TEXT
        );
        CREATE TABLE multi_strategy_paper_trader_heartbeats (
            timestamp TEXT PRIMARY KEY,
            latest_feature_timestamp TEXT,
            candidate_decisions_evaluated_this_poll INTEGER,
            candidate_rows_total INTEGER,
            candidate_rows_written_this_poll INTEGER,
            candidates_logged INTEGER,
            trades_opened INTEGER,
            skips_logged INTEGER,
            per_strategy_json TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO paper_trades (created_at, status) VALUES (?, ?)",
        [
            ("2026-01-01T00:00:00+00:00", "closed"),
            ("2026-01-01T00:00:01+00:00", "settled"),
            ("2026-01-01T00:00:02+00:00", "open"),
        ],
    )
    conn.executemany(
        "INSERT INTO paper_trade_candidates (created_at, rejection_reason) VALUES (?, ?)",
        [
            ("2026-01-01T00:00:00+00:00", "threshold"),
            ("2026-01-01T00:00:01+00:00", "threshold"),
            ("2026-01-01T00:00:02+00:00", "time_window"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO multi_strategy_paper_trader_heartbeats (
            timestamp, latest_feature_timestamp,
            candidate_decisions_evaluated_this_poll, candidate_rows_total,
            candidate_rows_written_this_poll, candidates_logged,
            trades_opened, skips_logged, per_strategy_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                2,
                10,
                1,
                1,
                0,
                1,
                json.dumps([{"strategy_id": "old", "latest_rejection_reason": "threshold"}]),
            ),
            (
                "2026-01-01T00:00:02+00:00",
                "2026-01-01T00:00:01+00:00",
                4,
                20,
                3,
                3,
                1,
                2,
                json.dumps(
                    [
                        {
                            "strategy_id": "s1",
                            "latest_rejection_reason": "threshold",
                            "skipped_min_estimated_edge": 2,
                        },
                        {
                            "strategy_id": "s2",
                            "skipped_probability_below_min": 1,
                        },
                    ]
                ),
            ),
        ],
    )
    conn.commit()
    conn.close()


def test_compact_paper_live_status_omits_strategy_stats_by_default(tmp_path) -> None:
    paper_db = tmp_path / "paper.db"
    _create_paper_db(paper_db)

    report = build_compact_paper_live_status(str(paper_db), limit=1)

    assert report["status"] == "ok"
    assert len(report["heartbeats"]) == 1
    heartbeat = report["heartbeats"][0]
    assert list(heartbeat.keys()) == [
        "timestamp",
        "latest_feature_timestamp",
        "candidate_rows_seen",
        "candidate_rows_total",
        "candidates_logged",
        "trades_opened",
        "trades_closed",
        "settled_trades",
        "skips_logged",
        "trades",
        "open_trades",
        "closed_trades",
        "pnl",
        "avg_roi",
        "wins",
        "losses",
        "latest_created_at",
        "latest_heartbeat_timestamp",
        "status",
        "top_rejection_reasons",
    ]
    assert report["heartbeat_source"] == "multi_strategy_paper_trader_heartbeats"
    assert heartbeat["candidate_rows_seen"] == 4
    assert heartbeat["candidate_rows_total"] == 20
    assert heartbeat["trades"] == 3
    assert heartbeat["open_trades"] == 1
    assert heartbeat["trades_closed"] == 1
    assert heartbeat["settled_trades"] == 1
    assert heartbeat["top_rejection_reasons"][0] == {
        "reason": "min_estimated_edge",
        "count": 2,
    }
    rendered = render_compact_paper_live_status(report)
    assert "strategy_stats" not in rendered
    assert "per_strategy_json" not in rendered


def test_compact_paper_live_status_verbose_includes_strategy_stats(tmp_path) -> None:
    paper_db = tmp_path / "paper.db"
    _create_paper_db(paper_db)

    report = build_compact_paper_live_status(str(paper_db), limit=1, verbose=True)

    heartbeat = report["heartbeats"][0]
    assert "strategy_stats" in heartbeat
    assert heartbeat["strategy_stats"][0]["strategy_id"] == "s1"


def test_compact_paper_live_status_uses_candidate_rejection_fallback(tmp_path) -> None:
    paper_db = tmp_path / "old_paper.db"
    conn = sqlite3.connect(paper_db)
    conn.executescript(
        """
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rejection_reason TEXT
        );
        CREATE TABLE multi_strategy_paper_trader_heartbeats (
            timestamp TEXT PRIMARY KEY,
            latest_feature_timestamp TEXT,
            candidates_logged INTEGER,
            trades_opened INTEGER,
            skips_logged INTEGER
        );
        INSERT INTO paper_trade_candidates (rejection_reason) VALUES ('threshold');
        INSERT INTO paper_trade_candidates (rejection_reason) VALUES ('threshold');
        INSERT INTO paper_trade_candidates (rejection_reason) VALUES ('time_window');
        INSERT INTO multi_strategy_paper_trader_heartbeats (
            timestamp, latest_feature_timestamp, candidates_logged,
            trades_opened, skips_logged
        ) VALUES ('2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 3, 0, 3);
        """
    )
    conn.commit()
    conn.close()

    report = build_compact_paper_live_status(str(paper_db), limit=5)

    heartbeat = report["heartbeats"][0]
    assert heartbeat["candidate_rows_seen"] == 3
    assert heartbeat["candidate_rows_total"] == 0
    assert heartbeat["top_rejection_reasons"] == [
        {"reason": "threshold", "count": 2},
        {"reason": "time_window", "count": 1},
    ]


def test_compact_paper_live_status_shows_startup_heartbeat_timestamp(tmp_path) -> None:
    paper_db = tmp_path / "startup_paper.db"
    conn = sqlite3.connect(paper_db)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            status TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            rejection_reason TEXT
        );
        CREATE TABLE multi_strategy_paper_trader_heartbeats (
            timestamp TEXT PRIMARY KEY,
            status TEXT,
            latest_feature_timestamp TEXT,
            candidates_logged INTEGER,
            trades_opened INTEGER,
            skips_logged INTEGER
        );
        INSERT INTO multi_strategy_paper_trader_heartbeats (
            timestamp, status, latest_feature_timestamp, candidates_logged,
            trades_opened, skips_logged
        ) VALUES (
            '2026-01-01T00:00:00+00:00',
            'starting',
            '2026-01-01T00:00:00+00:00',
            0,
            0,
            0
        );
        """
    )
    conn.commit()
    conn.close()

    report = build_compact_paper_live_status(str(paper_db), limit=1)

    heartbeat = report["heartbeats"][0]
    assert heartbeat["status"] == "starting"
    assert heartbeat["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert heartbeat["latest_heartbeat_timestamp"] == "2026-01-01T00:00:00+00:00"


def test_compact_paper_live_status_uses_paper_trader_heartbeats(tmp_path) -> None:
    paper_db = tmp_path / "baseline_paper.db"
    conn = sqlite3.connect(paper_db)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            status TEXT,
            pnl_usd REAL,
            roi REAL
        );
        CREATE TABLE paper_trader_heartbeats (
            timestamp TEXT PRIMARY KEY,
            latest_seen_feature_timestamp TEXT,
            open_trades INTEGER,
            settled_trades INTEGER,
            total_trades INTEGER,
            threshold_skips INTEGER,
            time_window_skips INTEGER
        );
        INSERT INTO paper_trades (created_at, status, pnl_usd, roi)
        VALUES ('2026-01-01T00:00:00+00:00', 'settled', 0.25, 0.25);
        INSERT INTO paper_trader_heartbeats (
            timestamp, latest_seen_feature_timestamp, open_trades,
            settled_trades, total_trades, threshold_skips, time_window_skips
        ) VALUES (
            '2026-01-01T00:00:10+00:00', '2026-01-01T00:00:09+00:00',
            0, 1, 1, 3, 1
        );
        """
    )
    conn.commit()
    conn.close()

    report = build_compact_paper_live_status(str(paper_db), limit=1)

    assert report["heartbeat_source"] == "paper_trader_heartbeats"
    heartbeat = report["heartbeats"][0]
    assert heartbeat["latest_feature_timestamp"] == "2026-01-01T00:00:09+00:00"
    assert heartbeat["pnl"] == 0.25
    assert heartbeat["avg_roi"] == 0.25
    assert heartbeat["top_rejection_reasons"] == [
        {"reason": "threshold", "count": 3},
        {"reason": "time_window", "count": 1},
    ]


def test_compact_paper_live_status_falls_back_to_paper_trades_with_realized_pnl(tmp_path) -> None:
    paper_db = tmp_path / "transformer_paper.db"
    conn = sqlite3.connect(paper_db)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            status TEXT,
            realized_pnl_usd REAL,
            realized_roi REAL,
            pnl_usd REAL,
            roi REAL,
            skip_reason TEXT
        );
        CREATE TABLE paper_trade_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            rejection_reason TEXT
        );
        INSERT INTO paper_trades (
            created_at, status, realized_pnl_usd, realized_roi, pnl_usd, roi, skip_reason
        ) VALUES
            ('2026-01-01T00:00:00+00:00', 'closed', 0.40, 0.40, NULL, NULL, NULL),
            ('2026-01-01T00:00:01+00:00', 'closed', -0.10, -0.10, NULL, NULL, NULL),
            ('2026-01-01T00:00:02+00:00', 'open', NULL, NULL, NULL, NULL, NULL),
            ('2026-01-01T00:00:03+00:00', 'skipped', NULL, NULL, NULL, NULL, 'threshold');
        INSERT INTO paper_trade_candidates (created_at, rejection_reason)
        VALUES ('2026-01-01T00:00:03+00:00', 'threshold');
        """
    )
    conn.commit()
    conn.close()

    report = build_compact_paper_live_status(str(paper_db), limit=1)

    assert report["heartbeat_source"] == "paper_trades_fallback"
    summary = report["summary"]
    assert summary["trades"] == 4
    assert summary["closed_trades"] == 2
    assert summary["open_trades"] == 1
    assert summary["pnl"] == 0.3
    assert summary["avg_roi"] == 0.15
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    heartbeat = report["heartbeats"][0]
    assert heartbeat["pnl"] == 0.3
    assert heartbeat["avg_roi"] == 0.15
    assert heartbeat["top_rejection_reasons"][0] == {"reason": "threshold", "count": 2}


def test_compact_paper_live_status_cli(tmp_path, monkeypatch, capsys) -> None:
    import src.main as main_module

    paper_db = tmp_path / "paper.db"
    _create_paper_db(paper_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "compact-paper-live-status",
            "--paper-db",
            str(paper_db),
            "--limit",
            "1",
        ],
    )

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["paper_db_path"] == str(paper_db)
    assert len(payload["heartbeats"]) == 1
    assert "strategy_stats" not in payload["heartbeats"][0]
