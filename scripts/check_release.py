"""Read-only audit of a source candidate and its fresh local Git history."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_key_block': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'provider_token': rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})',
    'credential_url': rb'https?://[^\s/\x00]+:[^\s/\x00]+@',
    'credential_assignment': rb'(?im)^\s*(?:API_KEY|PRIVATE_KEY|SECRET_KEY|PASSWORD|MNEMONIC|SEED_PHRASE|ACCESS_TOKEN|AUTH_TOKEN)\s*=\s*[^\s#]{8,}',
}


def findings(data: bytes, path: str, scope: str) -> list[dict]:
    result = []
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, data):
            result.append({'scope': scope, 'path': path, 'line': data[:match.start()].count(b'\n') + 1,
                           'type': kind, 'value': '[REDACTED]'})
    return result


def forbidden(path: str) -> bool:
    return (path.startswith(('data/', 'logs/', 'models/', 'exports/', 'artifacts/', 'private/'))
            or path == '.env' or (path.startswith('.env.') and path != '.env.example')
            or any(part in ('.venv', '__pycache__', '_compiled') for part in Path(path).parts)
            or Path(path).suffix in ('.pyc', '.db', '.sqlite', '.sqlite3', '.csv', '.parquet',
                                     '.joblib', '.pt', '.pth', '.pem', '.key', '.log'))


def audit(root: Path = ROOT) -> dict:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], env=env)
    if not (root / '.git').is_dir():
        raise RuntimeError('Candidate must have its own fresh local Git repository')
    paths = git('ls-files', '-z').decode().split('\0')[:-1]
    result = {'tracked_files': len(paths), 'findings': [], 'excluded_artifact_violations': [],
              'commit_count': int(git('rev-list', '--count', '--all')),
              'refs': git('for-each-ref', '--format=%(refname)').decode().splitlines(),
              'has_remote': bool(git('remote').strip()),
              'has_object_alternates': (root / '.git/objects/info/alternates').exists(),
              'clean': not git('status', '--porcelain=v1', '--untracked-files=all').strip(),
              'fsck_ok': subprocess.run(['git', '-C', str(root), 'fsck', '--full', '--no-reflogs'],
                                        env=env, capture_output=True).returncode == 0,
              'limitations': 'Heuristic secret scan; review values privately if a location is flagged.'}
    for path in paths:
        if forbidden(path):
            result['excluded_artifact_violations'].append(path)
        result['findings'].extend(findings((root / path).read_bytes(), path, 'working_tree'))
    for record in git('rev-list', '--objects', '--all', '--reflog').decode().splitlines():
        oid, _, path = record.partition(' ')
        kind = git('cat-file', '-t', oid).decode().strip()
        if kind not in ('blob', 'commit', 'tag'):
            continue
        if kind == 'blob' and forbidden(path):
            result['excluded_artifact_violations'].append('history:' + path)
        result['findings'].extend(findings(git('cat-file', kind, oid),
                                          f'{oid[:12]}:{path or kind}', 'fresh_history'))
    result['passed'] = (not result['findings'] and not result['excluded_artifact_violations']
                        and result['commit_count'] == 2 and result['refs'] == ['refs/heads/main']
                        and not result['has_remote'] and not result['has_object_alternates']
                        and result['clean'] and result['fsck_ok'])
    return result


if __name__ == '__main__':
    report = audit()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 1)
