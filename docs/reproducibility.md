# Fresh Windows source checkout

Prerequisites: Windows x64, AVX2/FMA CPU, Python 3.11+ with Tk (tested 3.13), Git, CMake 3.22+, Ninja (Visual Studio CMake tools component or an existing PATH installation), Visual Studio C++ x64 tools/Windows SDK and the matching Microsoft x64 runtime. CPU mode needs neither CUDA nor an NVIDIA driver.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe scripts/bootstrap_engine.py
if ($LASTEXITCODE -ne 0) { throw "Engine bootstrap failed." }
.\.venv\Scripts\python.exe scripts/bootstrap_engine.py --check
if ($LASTEXITCODE -ne 0) { throw "Pinned engine check failed." }
.\scripts\build.cmd cpu 4 test-paths
if ($LASTEXITCODE -ne 0) { throw "CPU build failed." }
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Source checks failed." }
.\.venv\Scripts\bovary-settings.exe --smoke-test
.\.venv\Scripts\bovary.exe --help
```

The initial `python -m venv .venv` uses your installed interpreter; all subsequent Python commands use `.venv` directly. No activation or global PATH edit is required.

After the editable installation, `Bovary.cmd` opens the GUI using this checkout's `.venv/Scripts/pythonw.exe`. It requires that environment to exist.

Bovary builds `bovary-infer.exe` and `bovary-quantize.exe` under `work/build/bovary-cpu` or the explicitly selected `work/build/bovary-cuda`. Bootstrap fetches only the exact `engine.lock.json` revision and applies the hash-pinned patch. It verifies the full resulting working source through a temporary Git index. Repeated calls are idempotent; modified revisions, extra source files and conflicts fail without resets or cleanup. A failed first fetch can resume only when the origin still matches and the unborn repository contains no files outside `.git` and no index entries. Other partial states require review or a new directory. `--check` performs no fetch/patch. `--directory` can prepare a separate checkout; configure it using `cmake -DBOVARY_ENGINE_DIR=<directory>`. The default build script intentionally uses and verifies `work/engine/bovary-llama.cpp`.

Runtime remains standard-library-only. Extended source-distribution checks use the separate pinned build tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe scripts/check_distribution.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "from setuptools.build_meta import build_sdist; build_sdist('dist')"
if ($LASTEXITCODE -ne 0) { throw "Source distribution build failed." }
$sdist = .\.venv\Scripts\python.exe -c "from scripts.prepare_release import distribution_component, project_metadata; p = project_metadata(); print('dist/' + distribution_component(p['name']) + '-' + p['version'] + '.tar.gz')"
if ($LASTEXITCODE -ne 0) { throw "Could not read project metadata." }
.\.venv\Scripts\python.exe scripts/check_distribution.py --sdist $sdist
```

The actual-sdist regression requires those exact build pins; runtime-only test environments skip that one build test. Hosted CI installs the pins, builds the real sdist and checks every reviewed source byte. `MANIFEST.in` is generated from the public allowlist, so the sdist includes the native/bootstrap files and test fixtures required by the documented workbench commands.

Weights are deliberately absent. Follow the complete [model setup](configuration.md#verify-or-obtain-the-pinned-checkpoint): verify an existing file offline with `.\.venv\Scripts\python.exe tools/verify_model.py` or explicitly choose the pinned download after reviewing its source and publisher license. Verification never copies the model. No ordinary test downloads or loads weights. New verification tests use tiny mocked identities; they do not read the 14.6 GB checkpoint.

After configuring `BOVARY_MODEL` and `BOVARY_MODEL_MANIFEST` with the model setup guide, an explicit offline readiness check must succeed before inference. Keep this exit-status guard in PowerShell:

```powershell
.\.venv\Scripts\python.exe tools/verify_model.py --model $env:BOVARY_MODEL --manifest $env:BOVARY_MODEL_MANIFEST
if ($LASTEXITCODE -ne 0) { throw "Model verification failed; inference was not started." }
.\.venv\Scripts\python.exe -m bovary --prompt 'Reply with exactly: Hello' --until-finished --deadline-s 300
```

The optional downloader uses a stable per-directory `.bovary-download.lock` file. Concurrent writers are rejected before accessing a proof or partial checkpoint; process exit releases the OS lock without deleting that file. Never remove it to bypass another owner.


## Build and try the CUDA worker

Use a Windows x64 source checkout, the build prerequisites above, and an NVIDIA driver compatible with CUDA 13.3.1. The build script targets NVIDIA compute capability 8.6. This profile was exercised on an RTX 3060 Laptop GPU with 6 GiB of VRAM and driver 610.60. See the [NVIDIA CUDA 13.3.1 release notes](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/) for driver requirements.

From the checkout root, create the `.venv` and install the editable package using the steps above. Then obtain the pinned source and the local toolkit and build the worker. The toolkit helper downloads NVIDIA archives, verifies their SHA-256 hashes against NVIDIA's versioned manifest, and extracts them into `work/deps/cuda-13.3`. The driver is installed separately.

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap_engine.py
if ($LASTEXITCODE -ne 0) { throw "Engine bootstrap failed." }
.\.venv\Scripts\python.exe tools/inventory_cuda_setup.py
if ($LASTEXITCODE -ne 0) { throw "CUDA toolkit setup failed." }
.\scripts\build.cmd cuda 4
if ($LASTEXITCODE -ne 0) { throw "CUDA build failed." }
```

Check the actual worker before loading weights. The PATH change in this example applies to this PowerShell session and lets the direct probe locate the locally extracted DLLs. Bovary also looks in those toolkit directories under `BOVARY_HOME` when starting its own CUDA worker.

```powershell
$env:BOVARY_HOME = (Get-Location).Path
$env:BOVARY_CUDA_EXE = Join-Path $env:BOVARY_HOME 'work\build\bovary-cuda\bovary-infer.exe'
$cuda = Join-Path $env:BOVARY_HOME 'work\deps\cuda-13.3'
$env:Path = "$cuda\bin\x64;$cuda\bin;$env:Path"
$marker = & $env:BOVARY_CUDA_EXE --host-cache-protocol-version
if ($LASTEXITCODE -ne 0 -or $marker -cne 'bovary-worker 2') { throw "CUDA host-cache worker preflight failed. Check the local toolkit and rebuild the worker." }
```

After [configuring the checkpoint and manifest](configuration.md#verify-or-obtain-the-pinned-checkpoint), run a bounded request with the opt-in system RAM cache. Budget the extra RAM in addition to the GPU expert cache and the application's working memory. To compare the same request with the cache disabled, set `--host-cache-mib 0`.

```powershell
.\.venv\Scripts\python.exe tools/verify_model.py --model $env:BOVARY_MODEL --manifest $env:BOVARY_MODEL_MANIFEST
if ($LASTEXITCODE -ne 0) { throw "Model verification failed; inference was not started." }
.\.venv\Scripts\bovary.exe mini --preset gpu-prefill-2g --host-cache-mib 4096 --tokens 48 --prompt 'Return only a Python expression using min and max to clamp x to the inclusive range lo to hi.'
if ($LASTEXITCODE -ne 0) { throw "CUDA Mini smoke failed." }
```

An installed wheel can use the same source-built worker. Keep `BOVARY_HOME`, `BOVARY_CUDA_EXE`, and the model variables set as above and invoke the wheel's `bovary.exe` from its own environment. The one-click ZIP includes the Gemma CUDA worker. Setup downloads the required cuBLAS DLLs on compatible systems. See the [third-party terms](licensing.md) for details.

## CPU release builds and hosted checks

Public CPU binaries additionally require `-DBOVARY_SANITIZE_PATHS=ON` in an isolated build. Earlier release preparation on MSVC 19.51 identified `/experimental:deterministic` as required for `/pathmap` and verified that combination with a tiny C++ `__FILE__` fixture. The current mapped build and its exact identity are recorded separately in the release record; the tiny compiler fixture alone does not establish those results. The mapping is opt-in and leaves the working local CPU/GPU builds unchanged. Packaging independently scans the resulting executable for private ASCII/UTF-16 user-directory paths and fails closed if they remain. This experimental compiler option is verified on the exercised compiler; do not assume equivalent behavior on other versions. Keep engine source under the mapped project root for this release path.

Hosted CI uses disposable `windows-2022`, read-only contents permission, no provider secrets, no model downloads and CPU-only build/no-model smoke. Action revisions were checked against [checkout v4.2.2](https://github.com/actions/checkout/releases/tag/v4.2.2) and [setup-python v5.6.0](https://github.com/actions/setup-python/releases/tag/v5.6.0) on 2026-11-14. Full commit pins and avoidance of public-PR self-hosted runners follow [GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use). The hosted workflow is prepared; local checks do not establish a successful remote CI run. SDK/API tests use fake providers and prove control behavior, not commercial model quality.

## Check output-limit behavior before release

Worker protocol 2 is required for the current adapter. Verify the new binary's `--require-cpu-only --protocol-version` marker and use a fresh build manifest; the older protocol-1 binaries do not validate this feature.

Ordinary automated tests use fake workers/providers and should cover the unchanged 128-token default, the local-only/deadline/no-escalation contract, effective generation bounds, stop-reason consistency, truncation without validation, cancellation and recovery. They must not download or load weights or send paid requests.

An explicitly scheduled serial CPU readiness check must additionally exercise a meaningful longer answer with Until finished, natural model completion, numeric and context truncation, cancellation/deadline, recovery and unload. Record the exact binary/model/configuration and actual outcomes, preserving earlier results in separate directories. A Hello-only diagnostic cannot establish longer-answer behavior. The release record reports which of these checks actually passed; prepared commands are not evidence that they ran. Use bounded deadlines and the shared inference lock for every manual model check.

The 0.2.0a6 lead-run CPU check completed the exact 1-through-60 sequence at 229 tokens with EOS, checked explicit/context truncation and interrupted partial answers, recovered successfully, and reran the fixed numerical comparison. Its separate three-case diagnostic passed two validators and rejected fenced JSON in the third. See the [release record](../reports/bovary-release-preparation.json) for the scoped outcomes; this does not qualify representative quality or performance.

Build inventory captures refuse existing capture directories, stale targets and builds configured from another source tree. Ninja dry-run dependency checks run before and after capture without compiling; configuration generators require a current `--build-manifest` with successful native smoke and matching source/binary identities.

## Release verification environment and Windows fixtures

Release verification requires the exact pins in `requirements-build.txt` in the same `.venv` used for tests. A skipped actual-sdist test is incomplete release evidence. After installation, verify the environment before running the full suite:

```powershell
.\.venv\Scripts\python.exe -c "import sys, importlib.metadata as m; print(sys.version); assert {n:m.version(n) for n in ('setuptools','wheel')} == {'setuptools':'80.9.0','wheel':'0.45.1'}"
if ($LASTEXITCODE -ne 0) { throw "Pinned release build environment is incomplete." }
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Release suite failed." }
```

On Windows, the venv `python.exe` may redirect to a separate interpreter process and retain inherited pipe handles. Closing fd 0 in that interpreter does not close the redirector's copy; killing and waiting for the redirector does not synchronize with the actual lock owner's death. The closed-stdin and downloader-death fixtures use `sys._base_executable` for their standard-library-only children. The downloader flushes its PID to stdout before publishing readiness and asserts it equals the process being killed. Other worker tests still exercise venv launchers and owned process-tree cleanup. Production code needs no retry or error reclassification for these fixture errors. Never delete the stable lock file or retry a live owner's contention to obtain a passing result.
