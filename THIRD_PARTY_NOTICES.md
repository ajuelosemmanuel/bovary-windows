# Third-party notices and inventory (2026-11-14)

Bovary's project-authored software and documentation are licensed under the [MIT License](LICENSE), adopted on 2026-11-14. Third-party components retain the notices and terms below; the project license does not replace them. See [licensing scope](docs/licensing.md).

| Component | Relationship / observed license | Release treatment |
|---|---|---|
| llama.cpp and ggml at `2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c` | MIT; extension patches contain upstream-derived context/code | Full upstream notice retained in `licenses/llama.cpp-MIT.txt`. Fetch source at the locked revision. Static CPU binary links this engine. |
| llamafile CPU matrix multiplication | MIT, Copyright 2024 Mozilla Foundation; compiled with `GGML_LLAMAFILE=ON` | Full source notice preserved in `licenses/llamafile-MIT.txt`. |
| YaRN rotary scaling in ggml CPU operations | MIT, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng | Attribution and MIT terms preserved in `licenses/YaRN-MIT.txt`. |
| xxHash in the compiled vendor-hash target | BSD-2-Clause, Yann Collet | Full upstream notice retained in `licenses/xxhash-BSD-2-Clause.txt`. |
| SHA-1 / SHA-256 in vendor-hash | Public-domain declarations by Steve Reid / Igor Pavlov, SHA-256 based on Wei Dai public-domain code | Declarations retained in `licenses/sha1-public-domain.txt` and `licenses/sha256-public-domain.txt`. |
| rotate-bits used by SHA-256 | MIT, Copyright (c) 2021 William Casarin | Full notice retained in `licenses/rotate-bits-MIT.txt`. |
| TurboFieldfare at `fc2210c7119d08cafd8e97a16f6f37b0cea8bd83` | Apache-2.0 reference research; bounded expert reads informed the design | No Swift/Metal source or package dependencies bundled. Windows C++ implementation extends llama.cpp rather than porting its transformer. Full reference license retained conservatively in `licenses/TurboFieldfare-Apache-2.0.txt`; reference source remains authoritative. |
| Gemma GGUF checkpoint | Separate publisher/model terms; Q4 source provenance lives in local model manifest | No weights bundled or relicensed. Existing verified model reused. Do not infer terms from the software MIT license. |
| NVIDIA CUDA 13.3.1 / cuBLAS / CUDA runtime | Proprietary NVIDIA redistribution terms retained in local toolkit | No GPU DLLs or toolkit bundled. A GPU binary release requires component-by-component redistribution review and notice collection. |
| Microsoft Visual C++ / Universal CRT / OpenMP runtime | Microsoft runtime terms; CPU imports include MSVCP140, VCRUNTIME140(_1), VCOMP140 and Windows system DLLs | Not bundled; install supported Microsoft x64 redistributable. Do not copy system DLLs into a release. |
| Python / Tk | PSF and Tcl/Tk licenses | Interpreter not bundled; user-installed Python with Tk is required. A future bundled interpreter needs its complete notices. |
| NumPy 2.5.4, Matplotlib 3.11.2, Zstandard 0.23.0 | Optional analysis only; separate package/upstream licenses and native subcomponents | Not runtime dependencies and not bundled. Inspect their installed license inventories if redistributed. |
| setuptools 80.9.0 | MIT build tooling | Build dependency only, not bundled in source/portable artifact. |
| wheel 0.45.1 | MIT build tooling | Build dependency only, not bundled in source/portable artifact. |
| GitHub checkout/setup-python actions | MIT automation tools pinned by commit | Referenced by hosted CI; not bundled. |

The bounded local inventory found no copied TurboFieldfare Swift/Metal source files in `src/` or `patches/`. The versioned patch modifies the pinned llama.cpp engine and includes upstream-derived context; its MIT notice is preserved. The upstream engine and TurboFieldfare reference license copies in `licenses/` were checked against their inspected local source copies. This inventory does not certify authorship of every pre-existing contribution. Later copied/derived Apache code must retain its applicable license, attribution, modification notices and any relevant upstream NOTICE material. The complete fetched llama.cpp tree has further optional-tool/vendor notices; those optional tools are disabled and excluded from the current CPU payload. Re-audit the actual binary and enabled dependencies when build options change.

Distribute `LICENSE`, this inventory and the applicable full notices in `licenses/` with the source and CPU payload. Preserve MIT copyright/permission notices and BSD source/binary attribution requirements. Official license texts and identifiers: [OSI MIT](https://opensource.org/license/mit), [SPDX MIT](https://spdx.org/licenses/MIT.html), [SPDX BSD-2-Clause](https://spdx.org/licenses/BSD-2-Clause.html), [SPDX Apache-2.0](https://spdx.org/licenses/Apache-2.0.html). The retained upstream files remain authoritative for their components.

Model weights and proprietary runtimes are separate works. No project software license grants API-account access or permission to upload private prompts.

