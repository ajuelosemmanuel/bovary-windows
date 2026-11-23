# Running Bovary

For a packaged installation, start with the [Windows quick start](../README.md#install-the-windows-alpha-from-release-files). The commands below assume a source checkout unless indicated.

## Build from source

### 1. Install the Python workbench

Clone this repository or extract its source ZIP. Open **PowerShell in the repository root** and run:

```powershell
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "Build-tool installation failed." }
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Bovary installation failed." }
.\.venv\Scripts\python.exe -m bovary --version
```

Use Python **3.11 or newer with Tk**; Python 3.13 is the exercised interpreter. The runtime uses the Python standard library. The pinned packages above are build tools. No environment activation or system PATH edit is required.

Check your setup offline before building the engine or obtaining weights:

```powershell
.\.venv\Scripts\python.exe -m bovary doctor
.\.venv\Scripts\python.exe -m bovary doctor --json
```

Missing model/native files are reported as setup failures, with remedies. Unknown checks do not establish readiness. [Doctor and GUI setup](configuration.md#doctor-and-gui-model-setup) explains statuses, privacy and path precedence.

Try the offline planner before building the engine or obtaining weights:

```powershell
.\.venv\Scripts\python.exe -m bovary --mode suggest --task extract --prompt 'Return the name from: name: Alex' --json
```

`suggest` returns a recommendation with no model load, prompt upload or paid request. It does not execute that recommendation.

### 2. Build the CPU engine

| Requirement | Purpose |
|---|---|
| Windows x64 and an AVX2/FMA CPU | Native CPU runtime |
| Git, CMake 3.22+ and Ninja | Fetch and build the pinned engine |
| Visual Studio C++ x64 tools and Windows SDK | Compile the native executable |
| Matching Microsoft x64 C++/OpenMP runtime | Run the compiled executable |

Ninja can come from Visual Studio's CMake tools component or an existing PATH installation. CPU mode requires neither CUDA nor an NVIDIA driver.

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap_engine.py
if ($LASTEXITCODE -ne 0) { throw "Engine bootstrap failed." }
.\scripts\build.cmd cpu 4 test-paths
if ($LASTEXITCODE -ne 0) { throw "CPU build failed." }
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Source checks failed." }
```

Bootstrap fetches the locked engine source and verifies the pinned patch. Tests use synthetic fixtures, not model weights or paid APIs. The `test-paths` build option includes the native Unicode-path fixture. See the [complete build guide](reproducibility.md) for tool discovery and reproducible builds.

## Run a model

Choose **one** setup path below. Keep the PowerShell failure guards: inference must not follow a failed verification or download.

### Use an existing checkpoint

Replace the two model paths with your own. Verification reads the entire pinned checkpoint in place without copying it. It writes a completed proof only after the file's size and SHA256 match.

```powershell
$env:BOVARY_HOME = $PWD.Path
$env:BOVARY_MODEL = 'D:\Models\gemma-4-26B-A4B-it-Q4_0.gguf'
$env:BOVARY_MODEL_MANIFEST = 'D:\Models\bovary-model-provenance.json'
.\.venv\Scripts\python.exe tools/verify_model.py --model $env:BOVARY_MODEL --manifest $env:BOVARY_MODEL_MANIFEST
if ($LASTEXITCODE -ne 0) { throw "Model verification failed; inference was not started." }
```

Keep verified weights unchanged. A valid existing proof is preserved; conflicting or malformed proofs are refused. [Model configuration](configuration.md#verify-or-obtain-the-pinned-checkpoint) lists the exact artifact identity and proof behavior.

### Download the pinned checkpoint

Review the [GGUF source card](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF), [publisher card](https://huggingface.co/google/gemma-4-26B-A4B-it) and [model licensing notes](licensing.md) first. This optional command downloads approximately **14.6 GB** and requires the remaining download size plus a **3 GiB disk reserve**. Model file size is not a RAM requirement or a memory-use guarantee.

```powershell
.\.venv\Scripts\python.exe tools/download_model.py --directory work/models
if ($LASTEXITCODE -ne 0) { throw "Model download or verification failed; inference was not started." }
$env:BOVARY_HOME = $PWD.Path
$env:BOVARY_MODEL = Join-Path $PWD.Path 'work/models/gemma-4-26B-A4B-it-Q4_0.gguf'
$env:BOVARY_MODEL_MANIFEST = Join-Path $PWD.Path 'work/models/model-provenance.json'
```

The downloader resumes valid partial transfers, checks HTTP ranges and verifies the complete file before promotion. One process owns the model directory at a time. It downloads no optional vision, MTP or DFlash assets.

### Ask your first question

From the same configured shell:

```powershell
.\.venv\Scripts\python.exe -m bovary --prompt 'Explain heat pumps in two sentences.' --until-finished --deadline-s 300
```

For the desktop interface, run:

```powershell
.\.venv\Scripts\bovary-settings.exe
```

`Bovary.cmd` opens the same GUI using this checkout's virtual environment. Use **Model setup…** to select an existing GGUF, verification manifest and CPU executable, or launch from a configured shell. These environment assignments affect that shell and its children; they do not change system settings. Resource settings and selected nonsecret model paths are stored in ignored `work/bovary-settings.json`; API keys and prompts are not saved there.

Missing models or binaries produce explicit errors. Bovary does not silently switch device or destination.

## Output and sessions

### Mini mode

Use `bovary mini` for independent tasks that should release the 26B Gemma worker as soon as they finish.
Mini automatically selects a compatible Gemma CUDA worker when available and uses CPU otherwise. Choose `--preset cpu-only` to force CPU. To build a CUDA worker, use the [build guide](reproducibility.md). For a wheel installation, point `BOVARY_CUDA_EXE` at the resulting `bovary-infer.exe`, make its CUDA runtime dependencies available, and select `--preset gpu-prefill-2g`. On a compatible hybrid Windows CPU, add `--performance-cores`.

**Packed SSD preview.** The development experiment uses a separate reader and a 12.85 GB copy of the routed expert weights. The alpha CPU and regular CUDA paths read the original GGUF. Benchmark figures for the packed layout apply only to that experimental reader.
From the project root, start a low-RAM session on the laptop's CUDA worker:

```powershell
$env:PYTHONPATH = "$PWD/python"
.\.venv\Scripts\python.exe -m bovary mini --preset balanced
```

Enter a task on each line. Mini keeps up to 16 completed answers in session memory. Repeating an identical request with the same settings and supplied file contents skips inference. Use `/clear` to forget those answers and `/quit` to exit.

For fresh tasks on the 6 GiB RTX 3060 Laptop GPU, use `--preset gpu-cache-2g` to
give the GPU a 2 GiB expert cache. On a ten-token Python answer this
reduced process read I/O from 20.60 to 14.01 GB. Both presets used about
512 MiB peak worker RAM. Total device memory in use, including other apps,
peaked at 5,352 MiB with the larger cache on that task. On a 51-token Python
function, reads fell from 39.13 to 24.44 GB and generation rose from 1.42 to
2.37 tokens/s. Each configuration ran twice per prompt with a fresh worker
and identical output tokens. Choose `balanced` when other apps
need the extra GPU memory.

For batched prompt processing with the same 2 GiB GPU expert cache, select
`--preset gpu-prefill-2g` in the CLI or Mini, or **Batched prompts — 2 GiB cache**
in the desktop app. From the project root, rebuild the CUDA worker
with `.\scripts\build.cmd cuda 4` if you have a worker that reports
`streaming requires --prompt-batch 1`.

On the same laptop, batch 128 reduced median end-to-end time from 29.20 to
24.57 seconds for a 33-token prompt and from 77.98 to 25.87 seconds for a
107-token prompt, compared with `gpu-cache-2g` at batch 1. Prompt processing
for the 107-token case fell from 69.73 to 14.56 seconds, and peak worker RAM
rose from 512 to 638 MiB. Each setting ran twice per task with fresh workers
and identical output token IDs. Windows file cache state was uncontrolled.

For a single fresh request, use `gpu-prefill-2g`. Run this 33-token clamp
example from the project root.

```powershell
$env:PYTHONPATH = "$PWD/python"
python -m bovary mini --preset gpu-prefill-2g --tokens 128 --prompt "Return only a Python expression using min and max to clamp x to the inclusive range lo to hi."
```

In two direct CLI runs per task, the first stdout byte arrived at 5.74 and
6.49 seconds for the clamp and at 7.48 and 7.64 seconds for a 107-token
Python edit. All four runs returned the complete correct answer from a fresh
worker. Separate instrumented trials measured about 638 MiB peak worker
working set. Windows file cache state was uncontrolled.

On hybrid Windows systems, Mini's opt-in `--performance-cores` places the
native worker on the performance cores that its inherited CPU affinity permits.
Add it to the command above to try the measured GPU configuration. The flag
requires Windows to report distinct CPU efficiency classes in a single
processor group. Mini reports an error if the requested placement is unavailable.

With a current CUDA worker, `--host-cache-mib` gives a local streaming CUDA
preset an optional expert-weight cache in system RAM. Enter a budget from 0 to
4096 MiB. Zero disables the cache and is the default. A positive budget keeps
the preset's GPU expert cache and uses extra RAM to serve some expert weights
that would otherwise come from the SSD. On each request, the worker reserves
the requested cache budget and copies an expert weight into it after fetching
that weight twice. The worker fills the budget as it discovers reused weights
and releases the allocation with the request context. The worker uses
additional RAM for inference and cache bookkeeping. Rebuild the CUDA worker
if the native executable reports an incompatible version.

```powershell
bovary mini --preset gpu-prefill-2g --performance-cores --host-cache-mib 4096 --prompt "Your coding task"
```

On the RTX 3060 laptop, the 4 GiB option reduced median CLI time from
19.25 to 16.79 seconds in three paired runs of the 107-token Python edit.
First visible output took 6.10 and 6.40 seconds, respectively. Peak worker RAM
rose from 635 to 4,731 MiB, total GPU memory in use peaked at a median
5,040 MiB in both settings, and expert reads fell from 28.52 to 19.25 GB.
The edit returned the same 69 output tokens in all six runs. The shorter
clamp took 7.41 and 7.87 seconds in one pair. Each trial started a fresh CUDA
worker on performance cores. Windows file cache state was uncontrolled.

Use `--prompt` once for a single task, or repeat it up to eight times to share one model load across a batch. Each task has a fresh 2,048-token context, two CPU threads, and a 96-token default answer cap (adjustable with `--tokens` up to 256). A task or batch has a shared 180-second inference deadline. Use `--context-file` to supply one UTF-8 file of up to 8 KiB, and `--json` for cache, timing, and completion status. The default `auto` preset reports the selected profile in the Mini JSON result. The explicit `balanced` preset shifts work onto the NVIDIA GPU and streams weights from storage.

If a worker stops after generating part of an answer, Mini’s JSON failure row includes its `attempt`. Inspect `attempt.text` for the partial answer and `attempt.timing` for elapsed time. A plain-text batch prints the partial answer to stdout and the failure message to stderr.

For a targeted Python fix, select a unique expression with `--replace-expr`.
Gemma generates the replacement, then Mini reconstructs the file and prints it
to stdout. The input file stays untouched. Mini checks that the reconstructed
syntax tree changes only at the selection, adds parentheses when needed, and compiles the revised file without executing it.
The caller can test whether the new expression does what the task requires.
On a six-line Python edit with Gemma 26B and the 2 GiB expert cache, two fresh
runs per method produced the same file. Generating only the edit used four
output tokens and averaged 74.5 seconds. Rewriting the function used 69 output
tokens and averaged 109.1 seconds. Both methods processed the full input.

```powershell
.\.venv\Scripts\python.exe -m bovary mini --preset gpu-cache-2g --context-file .\example.py --replace-expr 'len(items) - 1' --prompt 'Fix the range stop so no final item is lost.'
```

### Interactive mode

Open an interactive local session from the source checkout:

```powershell
$env:PYTHONPATH = "$PWD/python"
.\.venv\Scripts\python.exe -m bovary --interactive
```

Enter one prompt per line. Answers stream as they are generated. Each prompt has its own context and reuses the loaded weights. If a prompt exceeds the context window, enter a shorter prompt in the same session. Type `/unload` to release memory between prompts or `/quit` to exit.

To try the CPU settings measured on the i7-12700H laptop, append
`--threads 2 --cpu-prompt-batch 32`. In four fresh-worker runs of the same
52-token prompt, two threads averaged 60.3 seconds to completion and eight
threads averaged 71.9 seconds. All four returned the expected one-word answer.
Prompt batch controls how many input tokens the engine processes together
(1–128, default 32). Use `--json` to compare time to first token
(`timing.ttft_s`) and prompt processing time (`timing.native.prompt_s`) on your own
prompts. Compare fresh and warm requests separately.

The default output cap is **128 tokens**. Choose a positive cap with `--tokens 64`, or use `--until-finished` / `--tokens unlimited` to let the local model continue until its end signal, context limit, deadline or cancellation. The GUI has the same **Until finished** option.

Until finished requires a finite deadline and supports local execution without commercial escalation. It does not extend or roll the context window, and a model end signal does not establish answer correctness.

| Stop condition | Result |
|---|---|
| Model end signal | `complete` before optional validation; a failing validator can yield `needs_review` |
| Token cap or remaining context exhausted | `truncated`, not graded as a complete answer |
| Deadline, cancellation or worker failure | `cancelled` or `failed`; worker unloaded |

Repeated prompts in one process reuse the loaded model. Redirected `--session` input accepts UTF-8 JSONL with an optional initial byte-order mark. `--json` emits one result per line and streams text events to stderr:

```powershell
.\.venv\Scripts\python.exe -m bovary --prompt 'Reply exactly: Alpha' --prompt 'Reply exactly: Beta' --tokens 16 --json
'{"prompt":"Reply exactly: Hello"}', '{"unload":true}' | .\.venv\Scripts\python.exe -m bovary --session --tokens 16
```

Plain output and streamed stderr text use strict UTF-8 on real console/file streams. Embedded streams retain their own text interface; output failures remain errors. JSON output escapes Unicode when needed for redirected Windows streams; parsing restores the original text. In the GUI, **Unload model** releases an idle worker, and closing the window reaps it. A cancellation or crash requires a fresh worker on the next request. Separate conversation history is unsupported; include intentional history in the prompt.

## Python API

Run this with `.\.venv\Scripts\python.exe` from the configured repository root:

```python
import time
from bovary import Request, create_workbench

workbench = create_workbench()
try:
    result = workbench.execute(Request(
        prompt="Explain heat pumps in two sentences.",
        output_tokens=None,  # Until finished; omit for the default 128-token cap
        deadline=time.monotonic() + 300,
    ))
    print(result.to_dict())
finally:
    workbench.unload()
```

Pass `on_event` to receive streamed events and a `threading.Event` as `cancel` to request cancellation. The API opens no desktop window or network listener.
