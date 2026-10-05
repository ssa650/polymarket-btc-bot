from __future__ import annotations

from pathlib import Path
import socket

import pytest

from src import config
from src.strategy_config import validate_strategy_config, summarize_strategy_config
from scripts.test_offline import block_network

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_strategy_example_validates_and_summarizes():
    path = ROOT / 'configs/paper_strategy.example.json'
    report = validate_strategy_config(path)
    assert report['status'] == 'ok', report['errors']
    assert report['strategy_count'] == 1
    summary = summarize_strategy_config(path)
    assert summary['count_by_direction_mode'] == {'YES_ONLY': 1}
    assert summary['max_open_trades_range'] == {'min': 1.0, 'max': 1.0}


def test_example_config_loads_with_public_endpoints_and_local_storage(monkeypatch):
    monkeypatch.setattr(config.os, 'environ', {})
    settings = config.load_settings(str(ROOT / '.env.example'))
    assert settings.db_path == 'data/recorder.db'
    assert settings.gamma_api_url == 'https://gamma-api.polymarket.com'
    assert settings.clob_api_url == 'https://clob.polymarket.com'
    assert settings.btc_price_feed_enabled is False
    assert settings.raw_ws_events_enabled is False
    assert settings.enable_trade_backfill is False


def test_explicit_environment_takes_precedence_over_example(monkeypatch):
    monkeypatch.setattr(config.os, 'environ', {'RECORDER_DB_PATH': 'temporary/fixture.db'})
    settings = config.load_settings(str(ROOT / '.env.example'))
    assert settings.db_path == 'temporary/fixture.db'


def test_offline_guard_blocks_tcp_and_udp_without_sending(monkeypatch):
    # Restore class methods even when run through the guarded suite.
    monkeypatch.setattr(socket.socket, 'connect', socket.socket.connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', socket.socket.connect_ex)
    monkeypatch.setattr(socket.socket, 'sendto', socket.socket.sendto)
    block_network()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match='Network access is blocked'):
            sock.connect(('127.0.0.1', 9))
        with pytest.raises(RuntimeError, match='Network access is blocked'):
            sock.connect_ex(('127.0.0.1', 9))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(RuntimeError, match='Network access is blocked'):
            sock.sendto(b'synthetic', ('127.0.0.1', 9))
