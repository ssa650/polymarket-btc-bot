# Contributing

This is the owner-approved Apache-2.0 open-source edition. Contributions should
use synthetic inputs and preserve the separation from private data/history.

1. Use a clean source copy without private `.env`, data, logs or model artifacts.
2. On the verified Mac platform, install `requirements-macos-arm64-py314.lock.txt`
   with `--require-hashes --only-binary=:all:`. Other platforms need validation;
   `requirements-dev.txt` contains development ranges.
3. Run `python scripts/test_offline.py -q`. Use synthetic regression cases and
   preserve timestamps, gap/staleness gates, chronological splits, recorder
   persistence and paper/live separation.
4. Describe the problem, change, checks and unverified integrations. Review the
   diff and new file list for credentials and private data before sharing.

The offline socket guard covers its Python process, not arbitrary subprocesses or
native libraries. Do not add network/tool integrations to the offline suite or
run it from a copy containing a private `.env`.

Deployment-grid tests verify exact owner settings, not generic parser behavior:

```bash
# Only in an owner-controlled checkout with reviewed private configs:
python -m pytest -q --private-configs -m private_config
# On a machine permitting local sockets, using synthetic databases:
python -m pytest -q --run-loopback -m loopback
```

Do not copy deployment grids into public fixtures to make those checks pass. The
generic suite constructs synthetic strategies, markets and models. A compatible
PyTorch environment is needed for the optional transformer training test; the
separate Mac hash lock passed all 32 transformer module tests. Report skips and
their reasons. `python scripts/check_release.py` audits the isolated preparation
checkout: it requires the verified source-only root commit, only the `main` ref,
no remote or object alternates, a clean tree, and no flagged secrets or private
artifacts in tracked files or reachable/reflog history. Run it before adding a
publication remote. Normal clones have a remote and may contain remote-tracking
refs, so they do not meet these preparation-specific conditions.
