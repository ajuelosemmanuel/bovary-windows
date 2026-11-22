# Local configuration

`BOVARY_HOME` selects the project/runtime root. From an editable checkout it defaults to that checkout; an installed wheel defaults to the working directory. Runtime files remain under `work/` there. A fresh checkout must bootstrap/build and configure a verified checkpoint; no model is automatically downloaded.

## Verify or obtain the pinned checkpoint

The default supported text checkpoint is pinned to:

| Field | Value |
|---|---|
| GGUF repository | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| Revision | `1ca99aae748b4b01f609fa51eb23682ee5f00b85` |
| Filename | `gemma-4-26B-A4B-it-Q4_0.gguf` |
| Bytes | `14618145824` |
| SHA256 | `d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03` |

The [GGUF model card](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) links to the [Google source model card](https://huggingface.co/google/gemma-4-26B-A4B-it). The Google card identifies Apache 2.0 and links to the [publisher's license](https://ai.google.dev/gemma/docs/gemma_4_license), checked on 15 November 2026. Review those source materials before acquiring or redistributing weights. Bovary's MIT license covers its software, not the weights. The helper pins the artifact identity above; it does not establish model fitness or grant rights.

If you installed the wheel and Windows CPU ZIP, follow the two checkpoint options in the [packaged Windows quick start](../README.md#install-the-windows-alpha-from-release-files). Those commands run from the download directory and use the helpers under `.\runtime\tools\`. The commands below run from the root of a source checkout.

For an existing copy, create a new proof path beside it without copying the model:

```powershell
.\.venv\Scripts\python.exe tools/verify_model.py --model 'D:\Models\gemma-4-26B-A4B-it-Q4_0.gguf' --manifest 'D:\Models\bovary-model-provenance.json'
if ($LASTEXITCODE -ne 0) { throw "Model verification failed; do not start inference." }
$env:BOVARY_HOME = $PWD.Path
$env:BOVARY_MODEL = 'D:\Models\gemma-4-26B-A4B-it-Q4_0.gguf'
$env:BOVARY_MODEL_MANIFEST = 'D:\Models\bovary-model-provenance.json'
```

Verification reads the complete file offline, checks its pinned size and SHA256, and refuses a file that changes during hashing. It publishes complete manifest bytes atomically without replacing another proof. An existing valid same-identity manifest is retained byte for byte; a malformed or conflicting manifest is rejected, so choose a new proof path rather than deleting an existing record. Neither the model nor system configuration is changed. The default manifest location is `model-provenance.json` beside the supplied model. Verifier stdout JSON uses ASCII Unicode escapes so non-UTF-8 redirected Windows streams can carry it; JSON parsing restores the original filename. The proof itself remains UTF-8. A failed verification never renews a prior proof: keep the PowerShell exit-status guard and do not run the changed file.

If no copy exists, the explicitly requested download below creates `work/models/gemma-4-26B-A4B-it-Q4_0.gguf` and its verification manifest. It transfers about 14.6 GB, resumes a `.partial` file, requires remaining bytes plus 3 GiB reserve, and downloads no optional model components. Setup and ordinary tests never invoke it automatically. A per-directory interprocess lock is acquired before reading a proof or accessing any partial/model file, and remains held through verification and publication. Another downloader fails immediately. `.bovary-download.lock` stays in the directory for reuse; the OS releases its lock when the owner exits or dies. Do not delete the lock file to bypass an active owner.

```powershell
.\.venv\Scripts\python.exe tools/download_model.py --directory work/models
if ($LASTEXITCODE -ne 0) { throw "Model download or verification failed; do not start inference." }
```

`BOVARY_MODEL` selects an existing GGUF. `BOVARY_MODEL_MANIFEST` selects its completed proof. Without overrides, runtime reads `work/models/model-provenance.json` and the named file there. The verifier records the identity of the exact Gemma file it hashed. On Windows, a matching per-file change-journal marker allows runtime to check the file without rehashing. If the marker is unavailable on the filesystem or absent from the proof, runtime reads the weights to recheck SHA256 within the request deadline. The GUI reports that limitation without reading the weights itself. Regenerate an older bound proof with a new manifest path to record the journal marker where supported.

Old Gemma proofs without a file identity require a one-time full verification. Use the guarded verifier command above with a new `--manifest` path, then set `BOVARY_MODEL_MANIFEST` to that path or select the new proof in **Model setup…**. The downloader also accepts `--manifest` for a new proof path if the default is already occupied. Reverify after replacing, editing or copying the file to a new filesystem location. The new proof is local to the verified file. Mixed weights remain a separate quantization experiment rather than a default worker profile.

## Runtime paths and controls

`BOVARY_CPU_EXE` / `BOVARY_CUDA_EXE` relocate the native binaries. Default paths are `work/build/bovary-cpu/bovary-infer.exe` and `work/build/bovary-cuda/bovary-infer.exe`. A CPU request requires a CPU-only binary even when relocated. A GPU profile never becomes a CPU fallback and CPU never becomes a GPU fallback.

New CLI/GUI sessions default to local CPU. The GUI stores device/profile choices in ignored `work/bovary-settings.json`. Without that file, it starts with CPU defaults. Routing settings, API keys, prompts and answers are not saved there. Run `.\.venv\Scripts\bovary.exe --help` for bounded request controls and `.\.venv\Scripts\bovary-settings.exe` for the window. From a configured source checkout, `Bovary.cmd` opens that window with `.venv/Scripts/pythonw.exe`. `--session` reads independent JSON request objects and `{"unload":true}`; command-line cloud/GPU permissions remain outer limits for those requests. CLI deadlines are seconds from request construction, internally absolute monotonic deadlines.

OpenAI credentials belong only in `OPENAI_API_KEY`. An approved provider JSON additionally identifies account, permitted data classes, model/capabilities, checked pricing, positive session cap and enabled status. A positive request cap, deadline, allowed destination and explicit data class are also required. Offline and Private dominate everything. See [provider controls](provider.md). A provider session cap is process-instance-local; do not mistake restarting the app for an account-wide budget ledger. No live requests were authorized during implementation.

History is currently unsupported as a separate contract field. Put intentional instructions/history into the prompt yourself, accounting for its full context and privacy implications. Independent worker requests always get fresh contexts.

The factory and offline planner both prefer the existing `benchmarks/decision_policy.json` under the configured home. An installed wheel in a home without that editable file uses its bundled snapshot. This keeps checkout policy edits effective across CLI and GUI. Unverified summary entries remain excluded from runtime qualification and Auto stays disabled.

Task-specific validator JSON and explicit validation escalation are available in the GUI, CLI and Python API. The GUI's Validator JSON field accepts a file path; Allow one commercial escalation requires a validator and remains off by default. Offline and Private override that checkbox. GUI request controls are locked while a run is active, so changed values cannot misleadingly appear to apply to an in-flight request. A task/category hint alone does not validate an answer; a request without a validator displays `not_checked`.

CLI and GUI share a 64 KiB UTF-8 JSON validator loader. Invalid schemas, unknown fixtures and malformed or overly deep JSON fail before any backend attempt. Validator file paths are relative to the launching process working directory; an absolute path is clearest when using a separately configured home. `benchmarks/validators/hello.json` is a cleared synthetic formatting example. Validator contents and escalation/provider settings are never saved in resource settings.

Relative native executable, model and manifest paths resolve under the configured home. Replacing an executable or configured worker script at the same path invalidates the warm worker using file metadata; the next request reloads it. Worker protocol text is bounded by requested token count and native token-piece size, and completed usage must agree with emitted frames/context. Actual startup/request failures retain an attempt snapshot with partial text and unknown usage/cost where appropriate; rejected preflight/version/lock checks do not invent an attempt.

## Output limits and completion

Without an override, `Request.output_tokens`, CLI and new GUI settings use a 128-token output cap. CLI `--tokens N` keeps a positive explicit cap. CLI `--until-finished` and `--tokens unlimited` select `output_tokens=None`; the GUI's Until finished control selects the same local-only contract.

`None` requires `mode="local"`, a finite future monotonic deadline and `allow_escalation=False`. It is rejected for Commercial, Suggest or Auto. An API request can use:

```python
import time
from bovary import Request

request = Request(
    prompt="Explain how a heat pump moves heat in three short paragraphs.",
    output_tokens=None,
    deadline=time.monotonic() + 300,
)
```

This removes the explicit output cap, not the context bound. The native worker derives a finite `generation_limit` from actual tokenized prompt usage and remaining context. The returned `usage.output_limit` records that effective bound; detailed native metrics retain `generation_limit`. It stops at the model's end-of-generation token, that context limit, deadline or cancellation. It does not roll the context or restart generation to extend an answer.

Successful native responses retain `usage.finish_reason`: `eos` for the model end signal, `token_limit` for an explicit output cap, or `context_limit` for exhausted context. Only natural completion can become a `complete` result and proceed to validation; exhausted limits produce `truncated`, with validation `not_checked`. A natural end signal still needs a task-specific validator when correctness matters. Truncation does not trigger commercial escalation. Deadline and cancellation interrupt/unload instead of being reported as natural completion. An interrupted sole local attempt retains its observed partial text in the returned result and GUI while keeping the failed/cancelled status; validators do not treat it as complete.

Commercial requests must keep an explicit positive token cap so preflight can reserve a bounded request charge. Existing data/privacy/budget gates remain in force. Increasing context or deadline can increase time and memory use; Until finished is not infinite generation.


## Doctor and GUI model setup

`bovary doctor` works before native setup or model installation. It never loads a model, initializes a GPU backend, contacts a provider, downloads files, changes system settings or writes verification proofs. Use `bovary doctor --json` for schema version 1: each check has a stable `id`, `status`, `message` and optional `remedy`.

Statuses are `pass` (the described check passed), `fail` (a required setup check failed), `optional` (a source-build component is absent) and `unknown` (not established). Exit codes are **0** for no detected required failure, **1** for a required failure, **2** for invalid arguments/configuration. Exit 0 with unknown checks is not a readiness guarantee. CPU SIMD capabilities are not inferred from a CPU name; Microsoft runtime file presence does not prove versions or exact dependencies. Build tools outside PATH may still be discoverable by the build script. GPU/cloud tools are unnecessary for CPU setup.

`--timeout-s 10` sets the overall probe/hash deadline; OS process creation, filesystem operations and process-tree cleanup can add time. CPU protocol probes are parser-only, bounded and reaped. `--home`, `--model`, `--manifest` and `--executable` override the matching environment/default paths. Paths are redacted by default in human and JSON output; `--show-paths` explicitly includes private paths, so sanitize it before sharing. Neither output includes environment values or credentials.

Normal model checks inspect the stored proof, filename, current size and file identity for bound proofs. They **do not perform fresh full-file hashing**. If Windows has no journal marker to compare, the default model file check reports unknown. For a current proof, `bovary doctor --verify-hash --timeout-s 600` explicitly compares all bytes with the stored proof, subject to the deadline. To create a new pinned checkpoint proof, use the guarded `tools/verify_model.py` command above.

In the GUI, choose **Model setup…** for native file pickers and **Check paths / verification command** for consistency checks and a PowerShell verification command. Full hashing runs outside the GUI. The Windows CPU ZIP supplies `tools/verify_model.py` under the extracted runtime. If you installed the wheel alone, use the helper from a source checkout or the matching source distribution. Select the model and its completed proof together.

At launch, path precedence is environment variables, then locally saved GUI paths, then runtime defaults. An explicit selection in the current window overrides the launch values and is saved under `paths` in ignored `work/bovary-settings.json`. On the next launch environment variables override saved paths again. Relative paths resolve under `BOVARY_HOME`/effective home. Blank selections explicitly use runtime defaults without rereading path environment variables. CPU and optional GPU executable selections are separate; changing device never silently falls back. CLI/API do not read GUI settings.

Changing model/executable selections unloads cached workers before saving and reuse, while retaining provider objects and their session budget accounting. Setup controls are disabled during a request. The state label reports `unloaded`, `loading` while local execution is starting, `generating` when local text arrives, `ready` when an idle worker remains loaded, and `stopping` after cancellation. Result/error details remain separate. Settings save only resource choices and nonsecret paths, never prompts, answers, credentials or routing authorizations. Do not share that private settings file.

`LocalAdapter.unload()` remains reusable. `close()` and context-manager exit provide the same reusable cleanup, rather than permanently disabling the adapter. Only active workers are retained for reliable interpreter-exit cleanup; abandoned unloaded adapters can be garbage-collected. Active requests must be cancelled before unloading or closing.
