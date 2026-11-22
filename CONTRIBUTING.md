# Contributing

Thanks for helping improve Bovary. Keep changes focused and describe the user-visible behavior they address. The current scope is a Windows alpha; CPU, privacy and budget defaults must remain explicit.

## Development setup

Use Python 3.11+ (3.13 is tested), Git, CMake 3.22+, Ninja (from the Visual Studio CMake tools component or PATH), and Visual Studio x64 C++ build tools. Create `.venv` with `python -m venv .venv` first. Install with `.\.venv\Scripts\python.exe -m pip install -e .`; run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. The implementation and public imports live in `python/bovary`; the only console commands are `bovary` and `bovary-settings`. The source launcher `Bovary.cmd` uses the checkout's `.venv` to open the settings window. Runtime has no third-party Python dependencies; optional analysis packages are pinned separately in `requirements-analysis.txt`.

Bootstrap with `.\.venv\Scripts\python.exe scripts/bootstrap_engine.py`, then run `scripts\build.cmd cpu 4`. Default tests and CI must not download models, contact paid APIs, query a GPU, or initialize GPU backends. Keep explicit model/hardware validation separate. Serialize real inference through the shared lock; only the lead schedules performance runs.

Install `requirements-build.txt` into the same `.venv` before release tests; the actual-sdist regression must run with zero skips. Use `scripts\build.cmd cpu 4 test-paths` when checking the native Unicode-path fixture as well as the main targets. See the [build guide](docs/reproducibility.md) for the complete setup recipe.

## Where changes belong

| Location | Purpose |
|---|---|
| `python/bovary/` | Shared API, worker adapter, CLI, GUI and routing |
| `src/` and `patches/` | Native entry points and locked engine changes |
| `tests/` | Synthetic, no-model regression tests |
| `docs/` | Configuration, build, evaluation and release guides |
| `reports/` | Curated evidence with an explicit validation scope |

Use the `BOVARY_` configuration and CMake names documented in `docs/configuration.md` and `docs/reproducibility.md`. Keep patches against the exact `engine.lock.json` revision. Bootstrap rejects unexpected revisions, extra source files and conflicting changes; it never resets a checkout. Preserve historical output and hardlinked raw artifacts. New results belong in new ignored directories.

## Proposing a change

Explain the problem, the resulting behavior and how you checked it. Add a regression when it exercises a meaningful failure or boundary; update user-facing examples when behavior changes. Keep performance claims separate from functional checks. Describe any untested hardware or provider behavior honestly.

For sensitive reports use [SECURITY.md](SECURITY.md). Public issues and pull requests must not contain keys, private prompts, model files or unsanitized logs. There is no need to run a live provider or download weights to contribute a routine fix.

## Public source boundary

Before proposing a public change, update `public-files.txt` intentionally and run `.\.venv\Scripts\python.exe scripts/prepare_release.py --check`. The allowlist, not broad `git add .`, defines the public boundary. Inspect the exact first-commit diff and artifacts. Never include settings, credentials, private prompts, model files or raw logs. Do not submit automated upstream contributions from this project; read upstream instructions before any separate contribution.

After changing the list, regenerate its literal source-distribution manifest with `.\.venv\Scripts\python.exe scripts/check_distribution.py --write-manifest`. In a Git checkout use `.\.venv\Scripts\python.exe scripts/prepare_release.py --check --check-tracked` to catch extra or missing tracked files. Install `requirements-build.txt` to run the actual-sdist regression; CI installs these build-only pins and checks the built archive against every allowlisted source byte.

Project-authored contributions use the MIT license in `LICENSE`; preserve third-party notices and provenance, as described in `docs/licensing.md`. No public remote, push or release upload is part of local preparation.

## Model setup regressions

Model setup safety tests use tiny mocked pins and reject network access. Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_setup.py -v` after changing verification/download code. Keep the known checkpoint pin shared by both tools, preserve existing models/proofs, and never use the real checkpoint or network in these tests.

Downloader regressions must use real child processes with tiny fixtures to check exclusive ownership and OS cleanup after process death, while preventing every network request. Keep its stable lock file in place after release. Verify ASCII-safe JSON on redirected non-UTF-8 stdout without changing UTF-8 manifest contents. Every runnable PowerShell verification/download recipe must check `$LASTEXITCODE` before proceeding to inference.
