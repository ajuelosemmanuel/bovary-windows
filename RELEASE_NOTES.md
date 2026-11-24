# Bovary 0.2.0a18 Windows alpha

**[Install Bovary for Windows x64](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a18/bovary-0.2.0a18-windows-x64-setup.exe)** · [Installer checksum](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a18/bovary-0.2.0a18-windows-x64-setup.exe.sha256) · [Portable ZIP](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a18/bovary-0.2.0a18-windows-cuda-x64.zip)

Run setup and open **Bovary Mini** from the Start menu. The per-user installer is unsigned. At the model menu, choose **S** to select a folder on another drive, **T** to test your laptop with Gemma, or press **Enter** to ask a question. Gemma uses CUDA on compatible NVIDIA laptops and CPU otherwise. Type `/keep` in Gemma before asking related questions; `/unload` releases the model and clears the conversation.

The [README](README.md) includes the hardware requirements, free disk space for each model, and the benchmark procedure. Inside the kit, `setup/gemma-cuda/` contains the NVIDIA, CCCL and CUB notices. Inkling engine notices are under `setup/inkling-native/`.

---

# Bovary 0.2.0a17 Windows alpha

**[Windows x64 installer](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a17/bovary-0.2.0a17-windows-x64-setup.exe)** · [Installer SHA-256](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a17/bovary-0.2.0a17-windows-x64-setup.exe.sha256)

Run setup, choose **Install**, then **Finish**. Reopen **Bovary Mini** from the Start menu. The installer wraps the verified CUDA-first ZIP shown below.

**[Download Bovary for Windows x64](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a17/bovary-0.2.0a17-windows-cuda-x64.zip)** · [SHA-256 checksums](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a17/SHA256SUMS.txt)

Extract the ZIP and double-click **Start Bovary.cmd**. Press **Enter** to try Gemma 26B. On a compatible NVIDIA laptop the launcher fetches 426 MB of CUDA libraries on first use and displays **Gemma 26B (CUDA)**. Other laptops use the included CPU worker. Choose **C** to force CPU. Allow 20 GB of free space for Gemma. The 14.6 GB checkpoint downloads separately.

Choose **I** for Inkling 2 Mini on compatible NVIDIA hardware or **B** to install both models. Type `/models` at the prompt to switch. [Hardware, disk space, and setup](README.md#start-bovary-on-windows)

Mini now reports the selected preset in its JSON results. For manual CPU installation, download the matching a17 wheel and CPU runtime archive and follow the [PowerShell instructions](README.md#install-the-windows-alpha-from-release-files).

---

# Bovary 0.2.0a16 Windows alpha

**[Download the Gemma and Inkling Windows launcher](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a16/Bovary-Windows-a16-model-choice.zip).** Extract the ZIP, open **Start Bovary.cmd**, then choose Gemma, Inkling, or both. [Model requirements and setup](README.md#start-bovary-on-windows) · [Checksums](https://github.com/ajuelosemmanuel/bovary-windows/releases/download/v0.2.0a16/SHA256SUMS.txt)

The launcher checks for a working AVX2/FMA CPU runtime before the model menu. For Inkling or both models, use an NVIDIA GPU with a CUDA 13.5-compatible driver and a local NTFS SSD. The launcher downloads the selected checkpoints and stores setup files in **Bovary Data** alongside **Start Bovary.cmd**.

For manual Gemma CPU setup, download the wheel, Windows x64 CPU runtime ZIP, and SHA256SUMS.txt from the [Windows alpha release](https://github.com/ajuelosemmanuel/bovary-windows/releases/tag/v0.2.0a16). Follow the [quick start](README.md#install-the-windows-alpha-from-release-files) to check the downloads, verify the separately supplied Gemma checkpoint, and try an offline Mini request. The CPU worker requires AVX2/FMA.

The code repair replay shows the 0.2.0a14 pilot. The CUDA host-cache comparison reports 0.2.0a15 measurements on an RTX 3060 laptop. CUDA testers can [build the optional worker](docs/reproducibility.md#build-and-try-the-cuda-worker) and choose the gpu-prefill-2g preset.

---

# Bovary 0.2.0a15 Windows alpha candidate

Install the Python wheel and extract the matching Windows x64 CPU archive. The [quick start](README.md#install-the-windows-alpha-from-release-files) covers offline checkpoint verification and a first Mini request. The CPU worker requires AVX2/FMA.

The optional host expert cache pairs with the regular CUDA worker and the `gpu-prefill-2g` preset. It uses a separate RAM budget in addition to the GPU cache. See the [usage guide](docs/usage.md) for the worker and preset requirements.

---

# Bovary 0.2.0a14 Windows alpha candidate

Install the Python wheel and extract the matching Windows x64 CPU archive. The [packaged quick start](README.md#install-the-windows-alpha-from-release-files) walks through checkpoint verification and a first offline `bovary mini` request. The CPU build targets AVX2/FMA-capable machines.

CUDA is opt-in through a separately built regular worker and an explicit Mini preset. `gpu-prefill-2g` processes prompt tokens together with a 2 GiB GPU expert cache. `--performance-cores` restricts the inference worker to available performance cores on compatible hybrid Windows systems. Timings depend on the laptop, prompt, and system load.

Packed SSD remains a development preview. Its extra 12.85 GB layout and experimental executable are separate from the alpha package. The checkpoint is acquired separately under its model terms. Local workers run offline once configured.

### Inkling 2 Mini developer preview

The GitHub source includes a separate Windows CUDA build for Inkling 2 Mini IQ4_NL. It reads experts from a packed SSD sidecar. For source edits, the worker can propose an opening from the input code and verify those tokens in the same batch as the prompt. On a mismatch it restores the verified state and continues greedy generation. See the [preview guide](previews/inkling/README.md) for the pinned engine, build and run instructions.

On one 75-token Python edit that copied much of the supplied function, the previous Inkling profile took 50.48–55.76 seconds across two fresh launches. The selected profile passed all nine checks and matched all 75 reference token IDs on each of three fresh launches on an RTX 3060 laptop.

| Measurement on that edit | Previous Inkling profile | Preview profile |
|---|---:|---:|
| Launch to first visible text | 14.79–16.58 s | 15.75–16.93 s |
| Launch to complete visible answer | 50.48–55.76 s | 26.71–28.42 s |
| Packed expert bytes requested per run | 78.93 GB | 42.96 GB |
| Largest resident worker working set | 662.17 MiB | 685.54 MiB |

The preview also used about 4.24 GiB of dedicated GPU memory on that edit. Long answers with little source text to copy can take minutes. The packaged Gemma Mini command and the Inkling developer preview have separate setup paths.

---

# Bovary 0.2.0a13 release candidate

The revised candidate passed 258 tests on Windows. Earlier candidates had intermittent worker failures and Git fixture errors. The assessment records what changed and which claims the current evidence supports. [Release assessment](docs/implementation-assessment.md)

This unpublished Windows CPU alpha adds offline setup diagnostics and GUI model selection, repairs Unicode CLI output, and allows unloaded adapters to be collected. Strict UTF-8 plain output preserves answers on redirected Windows streams; JSONL remains lossless and separate from stderr events. Broken output pipes remain errors after cleanup.

`bovary doctor --json` reports required failures, optional missing build tools and unknown capabilities. It redacts paths by default, uses bounded owned CPU parser probes and never loads a model. Full hashing is explicitly optional and compares with the stored proof; it never creates provenance. **Model setup…** selects existing local model/proof/executable paths, explains offline verification and unloads previous workers without resetting provider session accounting.

See [configuration](docs/configuration.md#doctor-and-gui-model-setup) for path precedence and diagnostic exit codes. `LocalAdapter.close()` and context-manager cleanup remain reusable like `unload()`. Local CPU/private/offline/zero-budget defaults remain in force.

On 20 November 2026, two offline CLI requests using the hash-verified 14.6 GB checkpoint returned the exact expected word on CPU. The second request reused the loaded model. Redirected session input now accepts strict UTF-8 and an optional initial byte-order mark. The GUI checks exercise withdrawn construction and setup commands. Hosted CI, a pristine desktop installation, GPU inference and live commercial execution remain unverified for this candidate.

---

# Bovary 0.2.0a12 release candidate

Local inference, on your terms.

This unpublished alpha candidate repairs two Windows test fixtures and refreshes release evidence. The venv redirector retained the supposedly closed stdin handle and made kill/wait target a launcher rather than the downloader lock owner. Both fixtures now use the base interpreter; the lock fixture also verifies its owner PID before termination. Runtime behavior remains unchanged. Earlier finalized candidates and upload repositories remain unchanged.

## What changed

- Correct closed-stdin and downloader-death fixtures to launch the base interpreter and verify actual lock ownership. Keep strict failure, cleanup, mutual exclusion and recovery assertions.
- Require the exact build pins in the documented test environment and refresh evidence from repeated regressions, the complete suite and newly built packages. Earlier a11 artifacts remain historical.
- Distinguish local source-upload checks from hosted CI, which runs after an authorized push, and from later desktop/GPU/live-provider qualification.
- Preserve the a10 worker cleanup, download-resume and request-validation fixes, along with archive-container privacy, strict sdist/wheel and independent installed-package checks.

The canonical package/commands remain `bovary` and `bovary-settings`. Output controls retain the default 128 tokens and local-only `--until-finished` / `--tokens unlimited` with a deadline and no escalation. Commercial requests retain an explicit positive cap. Context/token exhaustion stays `truncated` and ungraded; interrupted sole local attempts retain partial text.

## Validation

The source review checks the public boundary, license/notices, installation guidance, local links and package identities. The retained runtime regressions cover closed-stdin cleanup/recovery, malformed resumed-response headers, rejected request values and archive/package boundaries. The final integrated suite result is recorded in [the release record](reports/bovary-release-preparation.json). Final direct-source and sdist-rebuilt wheel installation results accompany the candidate in `verification.json` and `provenance.json`.

No a12 native rebuild or model inference was performed. The unchanged a7 mapped CPU executable retains SHA256 `e0e157db53f9b529ad1de868bd184c1f0695e65c57fd1f35fd400db7e96c6a41`; packaging requires matching native/build source identities and Ninja freshness. Its retained a7 evidence covers Hebrew, café and emoji paths on Windows ANSI code page 1252, full verification of the 14,618,145,824-byte model through a Unicode hardlink, all 7,340,032 baseline logit bytes and token IDs matching, three Hello validator passes, warm PID reuse, interrupted partial text, recovery and final unload.

The earlier a6 Until finished diagnostic remains historical: exact sequence 1 through 60 at 229 output tokens; two task validators passed and fenced JSON was rejected as `needs_review`. These checks do not qualify representative model quality, performance or cost. [The release record](reports/bovary-release-preparation.json) separates current tests from inherited runtime evidence.

## Remaining limits

Python 3.11+ with Tk, the Microsoft x64/OpenMP runtime, AVX2/FMA and separately verified model weights remain prerequisites. No weights, interpreter, Microsoft redistributable or GPU DLL is bundled; this is not a standalone installer.

The alpha source is prepared for upload after its recorded local checks. Hosted CI is prepared and runs after an authorized push; no remote result is claimed before that push. A hosted image does not establish clean desktop installation. Clean-machine GUI/runtime installation, GPU inference and approved live-provider/representative paired evaluation remain separate qualification steps. Auto is disabled. Provider-session accounting is not a durable account-wide billing ledger. Publication and external announcements require explicit authorization of the exact artifacts and hosting account/visibility.
