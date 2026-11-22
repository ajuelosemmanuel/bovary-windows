# Master prompt for finishing Bovary

Run this prompt from the project root.

---

Act as the maintainer deciding whether to release Bovary as a Windows CPU source alpha. Implement and verify the work during this run. Give the owner a usable local GitHub candidate or an explicit hold or archive verdict.

**Establish what is worth shipping.** Identify a user, a task they can finish, and a simpler baseline. Start with a Windows user running the pinned GGUF locally, issuing a second request while weights are warm, cancelling, and recovering. Trace this through the CLI, desktop window and Python API. Read the current version in `pyproject.toml`, the README, `python/bovary/`, the engine pin, the tests, the release assessment and the dated reports. Inspect actual code and logs whenever they disagree. Attribute each measurement to its version, host and workload. Require paired measurements for claims about improved speed, memory, cost or quality.

**Use three subagents.** Give one the user workflow and evidence, one runtime correctness and reproduction, and one GitHub packaging and public-file review. Agree on file ownership before editing. The lead runs resource-intensive checks serially and integrates the work. Ask a reviewer to challenge the verdict against the final candidate.

**Fix the first failure a user would hit.** Exercise the documented entry points in a fresh directory. Verify offline Suggest yields zero attempts, doctor gives actionable diagnostics on an unconfigured install, and the GUI can construct. Trace failures at the request, worker, or installer boundary. For timing failures, distinguish process launch, readiness, the operation under test, and cleanup. Preserve production deadlines, cancellation, process ownership, privacy and budget defaults. Keep a failing run and a focused before/after regression. Reuse existing verified assets where available. Use synthetic fixtures for routine tests. Keep live commercial spend at zero, Auto disabled, and model downloads or GPU experiments outside the source-release checks.

**Verify the candidate people will receive.** Use the documented Windows Python environment and pinned build tools. Run these commands individually and inspect each exit status.

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap_engine.py --check
.\scripts\build.cmd cpu 2 test-paths
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/prepare_release.py --check
.\.venv\Scripts\python.exe scripts/check_distribution.py
```

On a fresh checkout, follow the README bootstrap step before the offline engine check. Repeat previously intermittent worker and downloader cases. Run full discovery again from a clean public export with its own source on `PYTHONPATH`. Build the native fixtures there, or point `BOVARY_TEST_NATIVE_DIR` at the verified CPU build, put Ninja on PATH and use an interpreter with the pinned build tools. Report any skips and what they leave untested. Follow the [release procedure](release-review.md) and [Windows CI workflow](../.github/workflows/ci.yml) to build a wheel and sdist, check their contents, rebuild from the checked sdist, and test installation outside the checkout. Probe the native CPU protocol and record which native executable was used.

Prepare a fresh local export using `public-files.txt`. Initialize Git in the export, stage the listed files, then run `scripts/prepare_release.py --check --check-tracked` with the test interpreter. Inspect licenses, notices, staged diff, source archive and checksums. Keep weights, credentials, private settings, raw evidence and generated binaries outside the source repository. Obtain owner authorization for the hosting destination and visibility before publishing externally.

**Make the decision.**

- **Ready for GitHub as a source alpha** requires a demonstrated user workflow, repeatable passing checks on the frozen candidate, a verified installed package and an exact public export. When weights are unavailable, prior real-model evidence can support a narrowly scoped claim if its model, binary and relevant execution path still match. Identify those dependencies and label the evidence as historical. Report hosted Actions and clean Windows installation results only when observed.
- **Hold** when a useful workflow exists but a required check fails or cannot be verified. Give the failing command, expected and observed behavior, and the smallest next experiment that could lift the hold.
- **Archive** when the available implementation offers no reproducible useful workflow and the simpler baseline covers the stated need. Name the baseline and evidence that would change that conclusion. A flaky test or missing GPU benchmark alone is insufficient evidence to archive.

Deliver the verdict first, followed by the runnable quick start, corrected defects, exact test and repeat counts, remaining limits, and links to the reviewed source and checksum record. Keep public claims consistent across the README, release notes and assessment. Finish when the evidence supports one of the three verdicts.
