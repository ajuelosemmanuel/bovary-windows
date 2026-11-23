# Release review and publication boundary

`public-files.txt` is the exact proposed first-commit allowlist. It excludes the entire live `work/` and `outputs/` trees, including model weights, CUDA archives, private settings/prompts/responses, token/logit dumps, raw logs, credentials and local source snapshots. The retained research workspace is not moved or deleted. Historical results are exported selectively to `reports/` with relative paths and selected numeric fields only.

The [0.2.0a16 Windows release](https://github.com/ajuelosemmanuel/bovary-windows/releases/tag/v0.2.0a16) provides the model-choice launcher ZIP, Python wheel and CPU runtime archive. The [launcher source](../launcher/) handles installation, model selection and prompts. The commands below prepare and validate the source and CPU review artifacts. See the [candidate assessment](implementation-assessment.md) for the scope of verification, and follow [the security policy](../SECURITY.md) when sharing diagnostics.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_release.py --check
# In an intended public Git checkout, also reject unlisted/missing tracked files:
.\.venv\Scripts\python.exe scripts/prepare_release.py --check --check-tracked
$version = .\.venv\Scripts\python.exe -c "from scripts.prepare_release import project_metadata; print(project_metadata()['version'])"
if ($LASTEXITCODE -ne 0) { throw "Could not read candidate version." }
$buildDir = "work/build/bovary-$version-release"
$captureDir = "outputs/bovary-$version-release-build-001"
.\.venv\Scripts\python.exe scripts/prepare_release.py --output-dir "work/release-review/bovary-$version-source-001"
# After building and capturing a NEW current CPU build manifest:
.\.venv\Scripts\python.exe tools/inventory_build_manifest.py --backend cpu --build-dir $buildDir --build-command "cmake --build $buildDir --config Release --parallel 2" --output-dir $captureDir
if ($LASTEXITCODE -ne 0) { throw "Build inventory failed." }
.\.venv\Scripts\python.exe scripts/prepare_release.py --output-dir "work/release-review/bovary-$version-cpu-001" --cpu-build-manifest "$captureDir/build-manifest.json"
# Optionally add --wheel <path-to-fresh-wheel>; package bytes must match source.
```

Use new output directories. Preparation writes deterministic source archives, the exact proposed first-commit diff from an empty tree, per-file SHA256/size provenance and `SHA256SUMS.txt`. Optional CPU review archives preserve the working source layout and add only the CPU executable; hashes and CPU-only build flags must match a current manifest. No Python, Tk, VC runtime, GPU DLL or model is bundled. Portable means relocatable source/native payload with documented prerequisites, not a self-contained installer. Clean-machine interpreter/runtime and GUI installation remain a separate downloadable-runtime qualification step.

The proposed diff supports binary assets and can be applied in an empty directory with `git -c core.autocrlf=false apply <diff>`. Use that override when verifying exact file hashes on Windows; a global line-ending conversion setting can otherwise change text bytes during application. The source archive preserves the reviewed bytes directly.

The proposed repository's `.gitattributes` disables automatic text conversion, preserving the reviewed mixed source bytes and the engine patch digest regardless of a user's global `core.autocrlf` setting. Text files still produce normal readable Git diffs. Do not normalize hash-pinned patches without deliberately updating their lock and verification evidence.

The portable CPU artifact requires an isolated build with `BOVARY_SANITIZE_PATHS=ON`. Original local binaries embed absolute source paths in assertions and remain private. The compiler mapping changes diagnostic path strings; the release guard scans both ASCII and UTF-16 forms and never rewrites a compiled binary. Do not reuse the older local build manifest for the mapped artifact.

To reproduce that build in a Visual Studio x64 Native Tools terminal with CMake and Ninja on PATH, bootstrap the engine, then run:

```powershell
$version = .\.venv\Scripts\python.exe -c "from scripts.prepare_release import project_metadata; print(project_metadata()['version'])"
if ($LASTEXITCODE -ne 0) { throw "Could not read candidate version." }
$buildDir = "work/build/bovary-$version-release"
cmake -S . -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DBOVARY_CPU_ONLY=ON -DBOVARY_SANITIZE_PATHS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
if ($LASTEXITCODE -ne 0) { throw "CPU configuration failed." }
cmake --build $buildDir --config Release --parallel 2
```

To use a separate bootstrap checkout, pass `-DBOVARY_ENGINE_DIR=<directory>` at configuration time. It must match the pinned engine and patch. Use a new build directory when reproducing the native targets. The inventory tool verifies the patched checkout before recording the lock patch as engine provenance. When Git serializes the same changes differently, the tool keeps its original diff separately. Reuse a prior capture only after verifying its exact executable, current native/build source snapshots, engine identity and Ninja dependency freshness.

The bounded sanity scan rejects ASCII/UTF-16 private absolute user paths and common credential shapes, forbidden paths/types, large files and symlinks. It is an automated aid: inspect every proposed file and binary dependency before publication. An allowlist is intentionally stronger than ignore rules; ignore rules do not remove previously committed data. There is no root Git history in the original workspace, so the proposed diff is additions from an empty repository. No root Git repository, public remote, push, release upload or external message is created by packaging.

When changing the public list, run `.\.venv\Scripts\python.exe scripts/check_distribution.py --write-manifest` and review both files. The default source checks work on extracted archives without Git. The opt-in `--check-tracked` additionally requires a real project-root Git checkout and exact tracked-file equality; hosted CI uses it so a tracked file outside the reviewed list cannot pass unnoticed. The built Python sdist is checked against the same allowlist and the pinned backend's exact generated inventory. Both PKG-INFO copies must match project identity, README and licenses; commands, top-level packages, inert setup.cfg and SOURCES inventory are validated. Checked extraction into a new directory feeds a wheel rebuild, strict wheel/source/RECORD validation and isolated installed checks. CI sends that checked rebuilt wheel to its separate consumer job.

Bovary uses MIT for project-authored code, with the collective contributor notice in `LICENSE`. The selected repository slug is `bovary`; the owner still chooses the hosting account/visibility and authorizes publication. Third-party license obligations remain separate. Live API evaluation additionally requires provider/account, permitted data and explicit spending cap; default budget is zero. v0.3 cannot be called qualified until representative retained paired evidence passes the frozen quality/operating thresholds and demonstrates a measured baseline benefit.

The wheel, CPU archive, launcher ZIP, verification and provenance reports, and `SHA256SUMS.txt` are separate release assets. [Build the launcher ZIP](../launcher/README.md) from the approved package and native build inputs. Before publishing, verify the exact launcher ZIP and the hashes of its inner wheel and CPU archive against the approved release files. Exercise first-run setup, one answer from each supported model and `/models` switching from a freshly extracted copy on supported Windows hardware. Record the ZIP hash, hardware, and result with the release evidence. Attach the approved files and link them from the release notes.

Repository/issue URLs in package metadata must use the actual approved GitHub account and repository once chosen. No account is inferred from a local username, and no placeholder URL claims a live project. Source preparation and source-only review do not establish a hosted CI result or a tested clean-machine installer.

The optional benchmark configuration helpers require `--build-manifest <fresh-capture>/build-manifest.json`. They verify the executable/source/engine provenance from that capture; an old machine-specific driver/compiler record is not substituted for a current build.
