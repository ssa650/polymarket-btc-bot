# Applied project license and compatibility notes

**Owner-approved license: Apache-2.0**, applied in the root LICENSE and NOTICE. It suits a reusable ML/data-engineering toolkit because it
offers permissive reuse plus an explicit contributor patent grant and patent
termination rule. The tradeoff is more redistribution/notice/change requirements
than MIT. Dependency licenses remain separate.
[Apache-2.0 terms](https://www.apache.org/licenses/LICENSE-2.0).

**Alternative considered: MIT**, a shorter permissive license. Its
text requires retaining the copyright/permission notice and disclaims warranty;
it has no comparable express patent clause. [MIT terms](https://opensource.org/license/mit).

Compatibility considerations for this source-only candidate:

- MIT/BSD/Apache-licensed dependencies keep their own attribution and license
  conditions. Choosing a project license does not relicense those packages.
- certifi is MPL-2.0. Mozilla permits combination with BSD/Apache code, with
  obligations attaching to the covered files and distributions. A separate import
  does not make the bot's source MPL by itself. Vendoring or modifying certifi would
  require its notices/source obligations to be handled.
  [Mozilla MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).
- NumPy/SciPy wheel notices include OpenBLAS, LAPACK, GCC runtime and libquadmath
  texts. The inspected SciPy wheel actually bundles GCC/Fortran/quadmath libraries.
  GPL-family/runtime-exception and LGPL obligations need component-specific
  treatment when redistributing binaries. This candidate ships no dependency
  wheels; a future binary/container bundle needs a fresh notice/source review.
- PyTorch's wheel declares multiple permissive SPDX terms, including an LLVM
  exception, and contains many bundled notices. Its top-level metadata alone is
  not an inventory of every redistributed component.
- Apache-2.0 is compatible with GPLv3 but not GPLv2-only; this matters if future
  source is incorporated from GPLv2-only projects. Do not infer permission to
  relicense GPL code as Apache/MIT.
  [Apache compatibility FAQ](https://www.apache.org/foundation/license-faq.html).

This records the approved code-license decision, not universal rights clearance.
Private datasets and existing trained weights are excluded; their rights are not
resolved by choosing a code license. Public hosting is a separate operation.
