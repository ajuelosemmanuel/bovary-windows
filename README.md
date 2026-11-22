<p align="center">
  <img src="docs/images/bovary-banner.png" width="100%" alt="Bovary, with the portfolio pixel-art portrait on a dusty rose background">
</p>

# Bovary

**Run small, verifiable coding tasks with Gemma 4 26B on your Windows laptop.** Give Mini a source file and an expression to repair. It prints the revised code for review and releases the worker’s memory when the task ends. Once you’ve set up the model, your prompts and code stay on your machine. Local runs need no API key.

[Install the packaged CPU version](#install-the-windows-alpha-from-release-files) · [Build the NVIDIA CUDA version](docs/reproducibility.md#build-and-try-the-cuda-worker) · [Try a code edit](#try-a-code-edit)

## A local Python repair

![Captioned terminal replay showing Mini fix an inclusive endpoint and a separate check pass all 405 cases](docs/images/bovary-mini-repair-demo.gif)

In this replay of a recorded 0.2.0a14 run on an RTX 3060 Laptop GPU, Mini changed `start <= value < end` to `start <= value <= end` in **9.30 seconds** with a fresh CUDA worker and no answer-cache hit. A separate check passed all **405 input combinations**. [Replay details and pilot results](docs/use-cases.md#repair-demo)

## Measured speed and memory on a 6 GB GPU

The CUDA worker keeps frequently used expert weights on the GPU and streams others from SSD. An optional host cache uses additional system RAM to cut SSD reads. For 0.2.0a15, we measured three pairs on an RTX 3060 laptop of fresh runs with the same 107-token prompt for a full-function edit. The table shows medians.

| Extra host cache | First visible token | Time to finish | Peak worker RAM | Peak GPU memory across all apps |
|---|---:|---:|---:|---:|
| 0 MiB (default) | 6.10 s | 19.25 s | 635 MiB | 5,040 MiB |
| 4,096 MiB | 6.40 s | 16.79 s | 4,731 MiB | 5,040 MiB |

Setting `--host-cache-mib 4096` took **12.7% less time** for this edit, and all six runs returned the same 69 output tokens. In one pair, a shorter clamp task took 7.41 seconds at 0 MiB and 7.87 seconds at 4096 MiB. Each trial started a fresh worker on performance cores. Windows file cache state was uncontrolled. [Timings and settings](docs/usage.md#mini-mode)

## Choose your Windows setup

- **Packaged CPU.** Install the Python wheel and CPU runtime ZIP from the release files below. The CPU worker is the default and runs without an NVIDIA GPU.
- **Source-built NVIDIA CUDA.** Build the CUDA worker, make its runtime dependencies available, and select `--preset gpu-prefill-2g`. With the wheel, set `BOVARY_CUDA_EXE` to the built `bovary-infer.exe`. [CUDA build instructions](docs/reproducibility.md#build-and-try-the-cuda-worker)

Both paths use the same pinned **14.6 GB Gemma checkpoint**. [Verify or obtain the checkpoint](docs/configuration.md#verify-or-obtain-the-pinned-checkpoint)

## Install the Windows alpha from release files

You’ll need Windows x64, an AVX2/FMA-capable CPU, Python 3.11+ with Tk, and the Microsoft x64 C++/OpenMP runtime. Start with the wheel, CPU runtime ZIP, and `SHA256SUMS.txt` from the [0.2.0a16 Windows review release](https://github.com/ajuelosemmanuel/bovary-windows/releases/tag/v0.2.0a16). Open the release while signed in to a GitHub account with tester access. Compare the two package hashes with the manifest, then open PowerShell in the download directory.

```powershell
Expand-Archive .\bovary-0.2.0a16-cpu-review.zip -DestinationPath .\runtime
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps .\bovary-0.2.0a16-py3-none-any.whl
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed." }
$env:BOVARY_HOME = (Resolve-Path .\runtime).Path
```

Mini uses one pinned **14.6 GB** Gemma checkpoint. If you already have it, edit the model path below. If you need the weights, the [checkpoint guide](docs/configuration.md#verify-or-obtain-the-pinned-checkpoint) covers the model terms, download, and disk reserve.

```powershell
$env:BOVARY_MODEL = 'D:\Models\gemma-4-26B-A4B-it-Q4_0.gguf'
$env:BOVARY_MODEL_MANIFEST = Join-Path $env:BOVARY_HOME 'model-provenance.json'
.\.venv\Scripts\python.exe .\runtime\tools\verify_model.py --model $env:BOVARY_MODEL --manifest $env:BOVARY_MODEL_MANIFEST
if ($LASTEXITCODE -ne 0) { throw "Checkpoint verification failed." }
.\.venv\Scripts\bovary.exe mini --prompt 'What is 17 plus 28? Answer with just the number.'
```

The 0.2.0a15 smoke test returned `45` from a fresh offline install. If setup needs attention, run `.\.venv\Scripts\bovary.exe doctor` in the same shell.

## Try a code edit

Suppose a loop skips the last item in a list. Save this as `example.py`:

```python
def print_items(items):
    for i in range(len(items) - 1):
        print(items[i])
```

Ask Mini to replace just the loop bound:

```powershell
.\.venv\Scripts\bovary.exe mini --context-file .\example.py --replace-expr 'len(items) - 1' --prompt 'Include the final item.'
```

Gemma generates the replacement expression. Mini checks that the surrounding Python syntax tree is preserved and prints the revised function for you to review. Your file stays untouched.

Mini repaired both selected expressions in our eight-task pilot. It also missed one of two seeded bugs during review. Watch the [captioned repair demo](docs/use-cases.md#repair-demo) or see the [coding tasks, timings, and checks](docs/use-cases.md).

## Choose how to run

| If you want to… | Start here |
|---|---|
| Run several independent tasks with one model load | Repeat `--prompt` up to eight times in one [Mini batch](docs/usage.md#mini-mode). |
| Keep the model warm between prompts | Start an [interactive session](docs/usage.md#interactive-mode). Use `/unload` to release its memory. |
| Use a desktop window | Launch `.\.venv\Scripts\bovary-settings.exe` from the configured shell. [Desktop setup](docs/configuration.md#doctor-and-gui-model-setup) |
| Stream answers from Python | Use the [Python API](docs/usage.md#python-api) and its cancellation and unload controls. |
| Experiment with Inkling 2 Mini | The [developer preview](previews/inkling/README.md) has a separate CUDA worker and packed SSD layout. It can propose tokens from code in your prompt and verify them during generation. |

## Project

[Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) · [Release notes](RELEASE_NOTES.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

Bovary’s software uses the [MIT license](LICENSE). Model weights and runtime dependencies have their own terms. [Third-party notices](THIRD_PARTY_NOTICES.md)
