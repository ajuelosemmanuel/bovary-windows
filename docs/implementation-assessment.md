# Bovary implementation assessments

## Windows Mini alpha 0.2.0a14

This candidate packages the Gemma Mini command with CPU execution by default. The regular CUDA worker is an explicit source-build option. The source also includes an Inkling 2 Mini developer preview for packed SSD streaming and greedy source-token verification. The [release notes](../RELEASE_NOTES.md) give measured edit latency, worker RAM and GPU memory for that profile. The release assets carry the verification results for the packaged Gemma command.

## Bovary 0.2.0a13 assessment

## Windows CPU source alpha, 20 November 2026

The revised source passed all 258 tests on Windows x64 with Python 3.13.1, including native CPU permission checks, Unicode paths, process ownership and fixture-based provider checks. The CPU build command verified the pinned engine and Ninja found the CPU and path-test targets up to date. Git for Windows required ordinary Windows subprocess permissions to exercise two local-clone fixtures.

The verifier rehashed the existing 14,618,145,824-byte checkpoint. Two successive offline CLI requests returned the exact expected word `Hello` and reached natural completion. The second request reused the loaded model. Both ran with a 2,048-token context, two threads and a 16-token cap. The CPU executable SHA256 was `04e19b0edd4f25e2b548ced4aae80d626d62e23da1b5a51bdb3768b65ab73380`. This is functional evidence for one model and host. Comparative cost, speed and representative task quality remain open.

The review reproduced silent text corruption in redirected UTF-8 JSONL on a Windows legacy code page. The CLI now decodes the input explicitly. The revised worker tests time rejection after the invalid data or closed pipe becomes observable and verify the identities of the processes they terminate. The Git test failures had a separate cause. Restoring a mocked process environment on Windows removed empty Git configuration values needed by child processes. The fixtures now pass isolated environment mappings.

Four regression cases each passed 20 consecutive test cycles: oversized protocol output, closed worker input, forced parent exit, and downloader lock-owner death. An additional real CPU request through the Python API was cancelled during streamed output. It retained partial text and the interrupted worker exited. A new worker then returned the exact expected word and unloaded successfully.

## Earlier a13 runs

**Integrated verification history.** An early run executed 257 tests with 23 failed subcases, two errors and zero skips. It encountered request deadlines and parser-probe timeouts. A later full run passed all 257 tests. Another full run reported four failures and three errors in Git fixtures. Its bootstrap traceback shows Git rejected a missing command-line configuration value. The five affected fixture methods passed in a separate run, and 30 repetitions of three process-timing tests passed. The full-run logs are retained privately as `work/publish-reaudit-a13-final.log` and `work/publish-reaudit-a13-confirmation.log`. For release decisions, use the integrated result for the exact source candidate being prepared.

The 17 November 2026 0.2.0a13 candidate repairs CLI encoding and unloaded-adapter retention, adds offline diagnostics, and exposes model setup in the GUI. The independent code review identified and prompted fixes for GUI blank-path precedence, cancellation state display, unavailable source verifier guidance and smoke-output path redaction. Local verification records are kept in fresh a13 output/review directories. Hosted CI, clean-desktop installation and GPU/live-provider qualification remain unverified. Earlier evidence below is historical and applies to its recorded artifacts.

## Historical 0.2.0a12 assessment

The 17 November 2026 a12 candidate corrected the Windows worker and downloader fixtures to control the interpreter that owns the child handles. The audit reproduced the a11 closed-stdin failure and intermittent lock recovery error. The a12 release report records that candidate’s integrated suite, repetitions, package and installed checks. [a12 release report](../reports/bovary-release-preparation.json)

## Runtime behavior

`python/bovary` is the sole implementation and public namespace. CLI, GUI and API share the Workbench contract. Commands are `bovary` and `bovary-settings`; `Bovary.cmd` uses the configured source virtual environment. CPU/local/private/offline and zero commercial budget remain the defaults.

Local requests retain loaded weights between runs while independent requests use fresh context/KV state. A bounded pipe protocol and shared loaded-worker lock expose no network listener. Cancellation, deadline, unload, child failure and Windows Job Object ownership clean up owned processes. GPU profiles require explicit permission.

The default output cap remains 128. Local-only `output_tokens=None` requires a deadline and forbids escalation. CLI `--until-finished` / `--tokens unlimited` and the GUI control use worker protocol 2 with a finite context-dependent bound. Model end permits normal validation; token/context exhaustion yields ungraded `truncated` results. Interrupted sole local attempts retain observed partial text with the failed/cancelled status.

Commercial requests require an explicit positive output cap, provider/account/data authorization, checked prices/limits, credentials, positive request/session caps, permission and a deadline. Offline and Private dominate cloud flags. Unknown billing retains its reservation and retries are disabled. The session cap is not durable account-wide accounting. Auto remains disabled regardless of imported summaries.

## Historical a12 source readiness and inherited runtime evidence

The source-upload review covers the exact public allowlist, license/notices, local documentation links, release-provenance identities and source-only Git history. It found no confirmed source-upload blocker in the retained a10 upload tree; the a12 fixture/package checks are recorded with this candidate. Hosted CI is a post-push check, not a result that can be established before the first authorized upload. The new [security policy](../SECURITY.md) explains private reporting and diagnostic sharing for this alpha.

The retained a10 request-boundary regressions reject unencodable prompt strings and overflowing budget values with typed invalid-request results before any backend attempt. Worker regressions cover a pipe already closed during cleanup, preservation of the original typed error/observed attempt, release of the shared lock and recovery. Resumed-download fixtures verify Content-Range before appending and preserve existing partial bytes on malformed or mismatched responses. These use synthetic workers/responses and do not establish live CDN behavior or model quality.

Source regressions include ZIP/gzip/TAR/PAX container-privacy canaries, malformed sdist metadata/build-settings/source-inventory regressions, checked extraction and an actual pinned-tool sdist-to-wheel rebuild, strict wheel/RECORD regressions, ASCII/UTF-16 privacy checks, download locking, redirected Unicode JSON, isolated-check preflight, native Unicode paths, model-verification guards, source/build freshness, provenance and interruption behavior. Native fixture tests exercise Hebrew, café and emoji arguments/files/environment paths. Withdrawn GUI/controller checks use no desktop input.

The unchanged a7 mapped CPU executable has SHA256 `e0e157db53f9b529ad1de868bd184c1f0695e65c57fd1f35fd400db7e96c6a41`. In the retained a7 check, with Windows ANSI code page 1252, the full 14,618,145,824-byte model was hash-verified through a Unicode hardlink without copying the model payload. Unicode executable, model, prompt, metrics and logit paths worked. All 7,340,032 authored baseline float32 logit bytes and token IDs matched. This is a narrow numerical regression check for the recorded configuration.

Cold/warm Hello requests and post-cancellation recovery passed their exact validator, for three passes. The warm PID was reused, interruption retained partial text, recovery used a new PID and final Unload released the worker. The real CPU measurement-configuration helper accepted the fresh build manifest. The a12 release record identifies the binary, scope and private evidence. [a12 release record](../reports/bovary-release-preparation.json)

The separate a6 Until finished diagnostic is retained as historical evidence: exact sequence 1 through 60, natural EOS at 229 output tokens, explicit/context truncation, deadline/cancellation and recovery. Its three-case task diagnostic passed classification and formatting; fenced JSON was correctly rejected as `needs_review`. These are not newly measured a12 quality results or representative production evidence. Earlier [worker](../reports/worker-validation.json), [continuation](../reports/continuation-validation.json) and [release](../reports/release-verification.json) reports likewise apply to their recorded artifacts.

## Release validation and remaining gates

The sdist checker requires both package-metadata copies to match the reviewed identity, README and licenses; commands, top-level packages, inert setup.cfg and SOURCES inventory must also match. Bounded checked extraction feeds a fresh wheel build. Source, wheel and CPU payloads share ASCII/UTF-16 credential/private-path scanning. Bounded raw ZIP/gzip and expanded TAR scans also cover comments, extra fields, PAX/global headers and trailing TAR bytes; trailing ZIP bytes and mismatched local/central extra fields are rejected. Logical metadata values are scanned separately to preserve credential boundaries. Wheel preparation checks exact source bytes, safe paths/file inventory, bounded ZIP sizes, metadata/tag identity, license bytes and every RECORD hash/size. The standalone [installed check](../scripts/check_installed.py) uses a fresh venv and empty working directory, clears source/runtime/provider overrides, installs without indexes/dependencies and checks canonical interfaces, Until finished/default limits, a withdrawn GUI constructor and relocated native protocol without loading a model. Final results accompany the candidate in `verification.json` and `provenance.json`.

Native prerequisite probes share the request deadline and retain CPU/GPU permission checks. The one-second no-model deadline fixture returned with its owned child stopped, and the real unchanged CPU binary passed prerequisite probing. These are process-control checks without model inference. The CPU payload requires successful no-model smoke, matching source home and current native/build source snapshots, exact executable/engine identities, path sanitization and Ninja dependency freshness. Captures are unique or refuse existing output directories. Configuration helpers use verified identities and preserve unknown driver information as unknown.

CI prepares direct-source and sdist-rebuilt wheel checks plus a separate Windows hosted consumer job with no product checkout; that consumer receives the checked rebuilt wheel and matching native hash. Neither job has been remotely exercised; hosted CI follows the first authorized push. A hosted image includes runtimes and does not establish clean desktop installation. Clean-machine GUI/runtime validation, GPU inference, live-provider authorization/evaluation and reviewed representative paired evidence remain external qualification steps, separate from the alpha source upload. Population latency, cost and broad quality remain unqualified.

Project-authored code uses [MIT](../LICENSE); [third-party, model and runtime terms](licensing.md) remain separate. The [public allowlist](../public-files.txt) excludes live research, settings, credentials, models and raw logs. No weights, GPU DLLs or interpreter are bundled. Publication still requires approval of the exact artifacts and hosting account/visibility; local preparation publishes nothing.
