# Changelog

## 0.2.0a17 (Windows CUDA-first alpha)

- Start Gemma 26B with CUDA automatically when the Windows launcher verifies a compatible NVIDIA driver and CUDA worker.
- Offer CPU execution when CUDA is unavailable and keep an explicit CPU choice in the launcher.
- Package the Gemma CUDA and Inkling workers in the Windows model-choice ZIP. Setup retrieves the verified runtime and model weights separately.

## 0.2.0a16 (Windows tester review)

- Package the Windows alpha for a private tester review in the Bovary Windows repository. Update the install commands and download link to match the current artifacts.
- Identify the version used for the CUDA timing comparison and the code repair replay.

## 0.2.0a15 (Windows alpha candidate)

- Add an optional CPU-resident expert cache for the 2 GiB CUDA preset. The selected admission policy prioritizes frequently reused experts and limits cache pollution.
- Expose the host-cache budget in Mini, the CLI and the GUI. Require a matching CUDA worker and keep the default CPU path available.

## 0.2.0a14 (Windows alpha candidate)

- Package the Mini command for independent tasks, with CPU execution by default and explicit CUDA presets.
- Support batched prompt processing with the 2 GiB GPU expert cache and opt-in performance-core placement on compatible Windows systems.
- Document wheel installation with the separate CPU runtime archive and a locally verified Gemma checkpoint.
- Include the optimized Inkling 2 Mini worker as an opt-in source preview with its own engine pin and build instructions.

## 0.2.0a13 (release candidate, unpublished)

- Emit lossless UTF-8 plain CLI output on redirected Windows streams; preserve JSONL and propagate output failures after cleanup.
- Retain only active workers for process-exit cleanup; unloaded adapters can be collected. Add reusable close/context cleanup.
- Add offline `bovary doctor` with redacted human/JSON diagnostics, bounded CPU protocol probes and optional explicit hashing.
- Add GUI model/proof/executable selection, local path persistence, verification guidance and model lifecycle state display.
- Decode redirected JSONL session input as strict UTF-8, including an optional initial byte-order mark.
- Verify the actual Windows parent and worker PIDs. Measure oversized-marker rejection after observing the excess byte and closed-pipe failure after readiness. Dedicated deadline and cancellation tests cover startup and teardown.
- Isolate fixture environment overrides so empty Windows environment values survive for later Git subprocesses.
- Preserve local/private CPU defaults, provider session budgets, model provenance and cancellation safeguards.

## 0.2.0a12 (release candidate, unpublished)

- Correct Windows closed-input and downloader owner-death fixtures to launch the actual Python interpreter, avoiding virtual-environment redirector handle retention and PID indirection. Preserve strict failure, cleanup, exclusion and recovery checks; runtime behavior is unchanged.
- Install exact build pins in the documented interpreter and repeat release verification with fresh artifacts. Treat a11's claimed passing suite as historical and superseded by the reproduced audit failures.

## 0.2.0a11 (finalized private candidate, unpublished)

- Redesign the README with a clearer product overview, capability/limit summaries and guided Windows onboarding.
- Add a security policy covering supported alpha versions, private vulnerability reporting and safe diagnostic sharing; improve contribution and issue guidance.
- Clarify source-upload readiness: local source/package checks precede publication, while hosted CI runs after an authorized push. Keep clean-desktop, GPU and live-provider qualification separate from source availability.
- Change documentation and release identity only. Preserve the a10 runtime fixes, all finalized artifacts and the unchanged a7 native executable without new model inference.

## 0.2.0a10 (finalized private candidate, unpublished)

- Preserve typed local failures, interrupted text and attempt accounting when a worker has already closed stdin; cleanup still reaps the owned process and releases the shared lock for recovery.
- Reject resumed HTTP responses with an invalid or mismatched Content-Range before appending bytes, preserving the prior partial checkpoint.
- Reject unencodable prompt strings and overflowing numeric budget values as invalid requests before a backend attempt.
- Derive documented source-distribution and review paths from project metadata so commands follow the current candidate version.
- Retain strict packaging/installed checks, the unchanged a7 native executable and historical CPU evidence. Preserve all earlier release artifacts, stages and upload repositories without new model inference.

## 0.2.0a9 (finalized private candidate, unpublished)

- Scan ZIP container metadata, gzip wrappers and expanded TAR/PAX/header bytes for ASCII/UTF-16 credential/private-path shapes, including individual field boundaries; add safe and rejected metadata fixtures.
- Bound native prerequisite probes by the request deadline and preserve permission checks; verify owned-child cleanup and the unchanged real CPU binary without loading a model.
- Retain strict sdist-to-wheel rebuilding and independent isolated installed checks for both wheel build paths. Record the final integrated suite in the current release report.
- Reuse the unchanged a7 native executable and explicitly inherited CPU evidence. Preserve all earlier artifacts, stages and upload repositories; perform no new native build or model inference.

## 0.2.0a8 (finalized private candidate, unpublished)

- Validate generated sdist metadata, commands, build settings and source inventory; safely extract checked bytes and test a rebuilt wheel through strict checks and isolated installation.
- Send the sdist-rebuilt wheel through the separate hosted consumer CI job; hosted execution and clean desktop installation remain external gates.
- Apply bounded ASCII/UTF-16 credential/private-path scanning consistently across source, wheel and CPU payloads.
- Serialize model downloads and preserve Unicode values through ASCII-escaped CLI/verifier JSON; expand process-lock, encoding and build-tool tests.
- Pass 207 source tests with zero skips. Reuse the unchanged a7 native executable and its explicitly inherited CPU evidence; perform no a8 native rebuild or model run. Preserve every earlier candidate and upload repository.

## 0.2.0a7 (finalized private candidate, unpublished)

- Fix Windows Unicode arguments/file/environment paths and add real native path regressions.
- Add full existing-model verification guidance/helper, PATH-aware build-tool discovery and explicit virtual-environment commands.
- Bind experimental provenance to current build/model/source identities; refuse capture overwrites and stale or foreign build dependencies.
- Strictly validate wheel structure, pure-Python tags, metadata, path inventory, licenses and RECORD hashes/sizes; add reusable isolated installed CLI/API/withdrawn-GUI checks.
- Prepare a separate hosted Windows artifact-consumer CI job without a source checkout. Hosted CI and clean desktop installation remain unexecuted external gates.
- Pass 193 source tests, real Unicode-model CPU numerical comparison and worker lifecycle checks. Preserve all earlier candidates and the scoped a6 diagnostic findings without broad quality claims.

## 0.2.0a6 (finalized private candidate, unpublished)

- Add local generation without an explicit output cap through `output_tokens=None`, CLI `--until-finished` / `--tokens unlimited`, and a GUI control; keep the default numeric cap at 128.
- Require a deadline and forbid commercial escalation for no-cap requests. Commercial requests retain an explicit positive cap.
- Upgrade the native worker to protocol 2 with finite effective generation limits and explicit `eos`, `token_limit` and `context_limit` stop reasons. Limit exhaustion stays `truncated` and is not graded by runtime validators.
- Pass 153 source tests with zero skips and fresh CPU numerical/functional checks: natural 229-token completion beyond the old default cap, explicit token/context truncation, retained interrupted text and recovery. Preserve the separate two-pass/one-needs-review authored task finding and every earlier finalized artifact.

## 0.2.0a5 (finalized private candidate, unpublished)

- Make Bovary the sole package implementation, Python namespace and pair of installed CLI/settings commands; remove compatibility aliases, fallback environment variables and obsolete product wrappers.
- Use canonical `BOVARY_*` settings/build options, `BOVARY_STREAM*` engine controls and `bovary-infer`/`bovary-quantize` native names.
- Prepare a separately pinned engine checkout and new CPU build identity while preserving every earlier finalized candidate and private measurement.
- Retain MIT licensing, third-party notices and reviewed source/distribution boundaries. The mapped CPU binary matched the retained authored short-prompt logits and token IDs exactly; no representative model-quality, performance, GPU or live-provider qualification is claimed. Auto remains disabled.

## 0.2.0a4 and 0.2.0a3 (superseded private naming candidates)

- Prepared unpublished branding/licensing candidates with compatibility layers. Those layers are removed in 0.2.0a5; finalized private artifacts remain unchanged.

## 0.2.0a2 (unreleased)

- Expose validator files and explicit one-step escalation in the GUI through the shared request contract; invalid validator configuration fails before execution.
- Bound worker protocol reads/queues, reload replaced executables, resolve relative native paths under the configured home, and retain failed actual attempts.
- Reject stale concurrent provider authorization after session disabling, bound streamed output, and enforce attempt-derived evidence cost/usage/timing checks.
- Preserve the complete reviewed native/Python workbench in source distributions.
- Check tracked files against the explicit public allowlist in hosted CI.
- Recover an interrupted first engine fetch only when the origin matches and the unborn checkout has no files or index entries.
- Keep v0.2.0a1 review artifacts unchanged; no live provider or production Auto qualification is claimed.

## 0.2.0a1 (unreleased)

- Shared request/result/event contract and CLI/GUI workbench integration.
- Persistent native local worker; explicit CPU/GPU permission and unload.
- One opt-in commercial adapter, fake-provider tests, validators and evaluation tooling.
- Pinned dependency bootstrap, source allowlist, Windows CPU CI and review-only packaging.
- Auto remains disabled until retained, representative paired evidence qualifies a named workload. No live commercial evaluation or publication is claimed.
