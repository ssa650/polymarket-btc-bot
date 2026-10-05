from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption('--private-configs', action='store_true',
                     help='Run owner deployment-grid tests; requires private local configs.')
    parser.addoption('--run-loopback', action='store_true',
                     help='Run the localhost HTTP integration test outside the socket-blocked runner.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'private_config: verifies owner deployment grids in data/')
    config.addinivalue_line('markers', 'loopback: starts and queries a local HTTP server')


def pytest_collection_modifyitems(config, items):
    import pytest
    for item in items:
        if 'private_config' in item.keywords and not config.getoption('--private-configs'):
            item.add_marker(pytest.mark.skip(reason='Owner deployment grid excluded; use --private-configs explicitly'))
        if 'loopback' in item.keywords and not config.getoption('--run-loopback'):
            item.add_marker(pytest.mark.skip(reason='Local socket integration; use --run-loopback outside offline runner'))
