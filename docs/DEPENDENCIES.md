# Dependency and license inventory

Observed from the existing Mac virtual environment during preparation. The
version-pinned file includes runtime and test dependencies; unrelated local
reverse-engineering tools and pip are excluded. Package metadata is evidence,
not clearance of bundled libraries, model/data rights, or the project itself.

| Distribution | Observed version | License metadata |
| --- | --- | --- |
| anyio | 4.12.1 | MIT |
| certifi | 2026.2.25 | MPL-2.0 |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.11 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| joblib | 1.5.3 | BSD-3-Clause |
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.0 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.6.0 | MIT |
| pyarrow | 24.0.0 | Apache-2.0 |
| Pygments | 2.19.2 | BSD-2-Clause |
| pytest | 8.4.2 | MIT |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-style license text; inspect bundled notices (metadata starts with copyright) |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| websockets | 15.0.1 | BSD-3-Clause |

Direct runtime imports: httpx, websockets, pyarrow, scikit-learn, numpy and
joblib. pytest is a development dependency. numpy/joblib are now explicit rather
than relying on transitive installation. PyTorch is an optional direct dependency
for transformer training/inference. The base environment intentionally excludes
it; the separate verified environment uses PyTorch 2.14.1 and its pinned closure.
See `TRANSFORMER_DEPENDENCY_LICENSES.json` for exact wheel metadata/notice hashes.

The 18 base and 10 additional transformer wheels were downloaded from public
PyPI, installed into fresh independent environments using `--require-hashes` and
`--no-index`, and checked against their declared versions. Both platform locks
cover macOS ARM64 / CPython 3.14. Wheels are not included in the source candidate.
`DEPENDENCY_LICENSES.json` records base artifact hashes, declared licenses,
requirements and bundled notice file hashes. License-content flags are review
signals, not proof that every mentioned component is linked on this platform.

NumPy/SciPy notices reference OpenBLAS, LAPACK, GCC runtime and libquadmath. The
inspected SciPy wheel bundles libgcc, libgfortran and libquadmath; GCC runtime
exceptions and LGPL terms require attention in a binary redistribution review.
certifi has MPL-2.0 obligations. PyTorch carries a compound permissive SPDX
expression and 100 notice files. This source-only candidate does not vendor these
artifacts or modify them. Keep notices intact if a future distribution packages
third-party binaries, and evaluate component/source obligations for that form.
Linux installation and source parity with Ubuntu remain unverified because no
usable registered Ubuntu environment or running local Docker engine was available.

No root LICENSE, COPYING or NOTICE was found among tracked files. A suitable
owner-review option is [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0),
which provides an explicit contributor patent grant and retention/redistribution
conditions. [MIT](https://opensource.org/license/mit) is a simpler permissive
alternative. No license text or grant has been applied. The owner must decide,
confirm authorship/rights and attribution needs, and separately review API data,
trained models, fixtures and any copied third-party code.

See [license recommendation and tradeoffs](LICENSE_OPTIONS.md) for owner approval.
