# Third-party dependency notices

The project's original source, documentation and fictional examples use
Apache-2.0 under the owner's approval. External dependencies retain their own
licenses; this project's LICENSE does not relicense them. Existing notices in
source files are preserved. No third-party wheel, library binary or vendored
dependency source is shipped in this source edition.

The inspected dependency versions, declared licenses, bundled notice file paths
and SHA-256 hashes are recorded in
[the base inventory](docs/DEPENDENCY_LICENSES.json) and
[the optional transformer inventory](docs/TRANSFORMER_DEPENDENCY_LICENSES.json).
[The dependency guide](docs/DEPENDENCIES.md) explains the distribution scope.

Most top-level dependencies use MIT, BSD or Apache terms. certifi uses MPL-2.0.
The inspected NumPy/SciPy notices mention OpenBLAS, LAPACK, GCC runtime and
libquadmath; SciPy bundles GCC/Fortran/quadmath libraries in its wheel. GPL-family
runtime exceptions and LGPL terms require component-specific treatment when
redistributing those binaries. PyTorch has a compound permissive SPDX declaration
and additional bundled notices. Top-level labels do not replace those terms.

Installing dependencies separately does not remove their license obligations.
If a future package/container redistributes dependency artifacts, retain the
actual upstream license/notice texts and satisfy applicable attribution,
source-availability and component/exception conditions. The current source-only
edition includes metadata and references, not a claim of binary-distribution
clearance. Private data and existing trained weights remain outside this edition
and outside its code-license grant.
