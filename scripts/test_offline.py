"""Run the synthetic test suite without inherited credentials or network calls."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def block_network() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def deny_internet(original):
        def guarded(sock, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError('Network access is blocked by the offline test runner')
            return original(sock, *args, **kwargs)
        return guarded

    socket.socket.connect = deny_internet(original_connect)
    socket.socket.connect_ex = deny_internet(original_connect_ex)
    socket.socket.sendto = deny_internet(original_sendto)


def main() -> int:
    if (ROOT / '.env').exists():
        print('Use a clean source copy without a local .env for offline tests.', file=sys.stderr)
        return 2
    if '--child' not in sys.argv:
        # Do not propagate wallet keys, API tokens, or recorder paths into tests.
        environment = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'TMPDIR') if key in os.environ}
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        return subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()),
                                '--child', *sys.argv[1:]], cwd=ROOT, env=environment)
    sys.path.insert(0, str(ROOT))
    block_network()
    import pytest
    arguments = [arg for arg in sys.argv[1:] if arg != '--child']
    return pytest.main(['-p', 'no:cacheprovider', *arguments])


if __name__ == '__main__':
    raise SystemExit(main())
