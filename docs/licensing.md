# Licensing

Bovary adopted the [MIT License](../LICENSE) on 14 November 2026 under the release owner's authorization. The project notice is `Copyright (c) 2026 Bovary contributors`. This collective attribution describes the project's contributors; it does not invent an individual identity or assign upstream ownership to the project.

MIT covers the project-authored software and associated documentation. It permits reuse, modification and distribution, including commercial use, with its copyright and permission notice retained. The full permission, condition and warranty disclaimer are in `LICENSE`. The license choice follows the official [OSI MIT text](https://opensource.org/license/mit) and uses the [SPDX identifier `MIT`](https://spdx.org/licenses/MIT.html).

MIT was selected as a short permissive license suited to this Python/C++ workbench and its MIT-licensed llama.cpp integration. That choice does not replace third-party licenses. The [third-party inventory](../THIRD_PARTY_NOTICES.md) identifies components and the full retained notices in `licenses/`:

- The pinned llama.cpp/ggml engine and upstream-derived patch context retain the ggml authors' MIT notice. The CPU payload also retains Mozilla llamafile, YaRN and rotate-bits MIT notices.
- xxHash retains its [BSD-2-Clause](https://spdx.org/licenses/BSD-2-Clause.html) notice and conditions, including reproduction of those notices with binary distributions. SHA-1 and SHA-256 retain their upstream public-domain declarations.
- TurboFieldfare was inspected as research reference material. The bounded local review found no copied Swift/Metal files in `src/` or `patches/`. Its full [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) reference license is retained separately. If later releases incorporate Apache-derived code, preserve the applicable license, attributions, modification notices and any relevant upstream NOTICE material.

The source/portable packages must retain `LICENSE`, `THIRD_PARTY_NOTICES.md` and the applicable `licenses/` files. Build configuration changes require checking the actual linked components; the complete engine source contains additional notices for optional tools excluded from the current CPU payload.

Model weights are separately obtained and remain governed by their checkpoint publisher's terms. The project's MIT license does not cover them. The prepared CPU payload contains no weights, NVIDIA toolkit/DLLs, Microsoft redistributable, or Python/Tk interpreter. Those runtimes and any future GPU redistribution retain their own terms. Optional analysis/build tools are recorded in the inventory and are not shipped as project runtime dependencies.

This records the adopted project license and the inspected release scope. It does not certify ownership of every pre-existing contribution or authorize publication, model redistribution, or paid API use.
