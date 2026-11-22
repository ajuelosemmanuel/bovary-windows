# Inkling 2 Mini for Windows

This source-only developer preview builds a separate, opt-in worker. It uses the IQ4_NL Inkling 2 Mini checkpoint with a 2 GiB GPU expert cache and reads routed expert weights from an aligned sidecar. It also proposes the opening of a code edit from source already in the prompt. The worker verifies proposed tokens against greedy decoding before emitting them and resumes ordinary decoding when a proposal diverges.

The original build completed one 119-prompt-token Python edit in 26.71–28.42 seconds across three fresh processes and matched the reference’s 75 generated tokens in each run. This single edit benefited from a matching source prefix. General prompts, longer generations and other hardware can behave differently. Rebuilding verifies source and runtime compatibility. It does not repeat the latency measurement.

## Build and verify

Use an AVX2-capable x64 Windows machine with an NVIDIA driver compatible with the CUDA 13.5 plugin, an x64 MSVC developer shell, Python 3.11+, Git, CMake 3.22+ and Ninja on PATH. Install the Microsoft x64 C++ runtime as needed. The build links the locally compiled llama.cpp engine and loads the pinned CUDA plugin dynamically.

Save the [b12600 source ZIP](https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/b12600) as `b12600.zip`. Get these two Windows x64 ZIPs from the [b12600 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12600) and keep them in the same downloads directory.

- `llama-b12600-bin-win-cuda-13.5-x64.zip`
- `cudart-llama-bin-win-cuda-13.5-x64.zip`

The script checks all archive SHA-256 identities against [upstream.json](upstream.json), including the archive pin for commit `29b52d398b294509746ec0a3f28948a3107c0213`. The output path must be new for each clean reconstruction. Run from the Bovary source root.

```powershell
python previews/inkling/preview.py check --upstream .\downloads\b12600.zip --output .\inkling-check
python previews/inkling/preview.py build --upstream .\downloads\b12600.zip --cuda-archives .\downloads --output .\inkling-build
python previews/inkling/preview.py smoke --bin .\inkling-build\native\bin
python -m unittest discover -s previews/inkling -p test_preview.py -v
```

`check` extracts the pinned archive and runs `git apply --check` followed by the actual application of each patch in order. It requires no CUDA archives, model or GPU. `build` reconstructs the same source in a new directory, builds with MSVC, stages only the selected DLLs from the verified archives locally, checks protocol version 2 and explicitly registers the CUDA plugin without loading model weights. It writes a SHA-256 manifest for the resulting binaries. A new build can have different executable hashes from the measured reference due to toolchain, path and linker metadata.

Clean reconstruction and build were verified with MSVC 19.51.36260 on an RTX 3060 Laptop GPU. The documented launcher returned `9` for a prompt containing `square(3)`, ended normally, and recorded 28 draft tokens, zero accepts and one restart. The smoke build used worker SHA-256 `a6e435e2f21069dd6bf1777fb143345d921c697b5e0f2d2285f1aac9729b999e` and engine SHA-256 `b5480fd36f02206de2f0c46a99f98dc6f4d9e552ce3e38d9e5de787f84c697ac`. That GPU smoke preceded the bounded prompt-read change.

## Prepare caller-owned inputs

Supply the exact checkpoint at the pinned revision in [the model repository](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF). Use a short, local NTFS path on an SSD for the unbuffered reads. The checkpoint is about 17.63 GB and the sidecar about 15.80 GB. The packer verifies the checkpoint, copies fused gate/up and down tiles in physical-layer and expert order, aligns each packed record to 4096 bytes, and checks the resulting file against the measured sidecar hash.

```powershell
python previews/inkling/pack.py --model .\models\Inkling-2-Mini-IQ4_NL.gguf --output .\models\Inkling-2-Mini-IQ4_NL.experts.pack
python previews/inkling/run.py register --model .\models\Inkling-2-Mini-IQ4_NL.gguf --pack .\models\Inkling-2-Mini-IQ4_NL.experts.pack --receipt .\inkling-inputs.json
python previews/inkling/run.py run --bin .\inkling-build\native\bin --receipt .\inkling-inputs.json --prompt .\prompt.txt --metrics .\inkling-metrics.json
```

Registration hashes the model and sidecar once. Each run checks their registered paths, file identities, sizes and timestamps before launch. Register to a new receipt after moving, modifying, or replacing either file. Keep the local receipt with your own inputs because it records their absolute paths.

The launcher streams native output and saves generation metrics to a new JSON file. The one-shot runner uses the measured inference flags below. The historical test used protocol worker mode and separately bound the process to the laptop's performance cores. The preview's one-shot launcher uses normal Windows scheduling.

| Setting | Value |
| --- | --- |
| Native arguments | `--context 2048 --threads 2 --gpu-layers 99 --cpu-experts --gpu-embedding --cache-mib 2048 --prompt-batch 176 --load-mode direct --direct-answer --no-op-offload` |
| Prompt and limit | `--prompt-file <file> --predict 160` by default, plus caller-selected `--model` and `--metrics` |
| Sidecar | `BOVARY_STREAM=1`, plus paths in `BOVARY_STREAM_MODEL` and `BOVARY_STREAM_PACKED` |
| Windows I/O | `BOVARY_STREAM_IO=unbuffered`, `BOVARY_STREAM_POOL=pinned2`, `BOVARY_STREAM_BUFFER_MIB=9`, `BOVARY_STREAM_LOAD_BUFFER_MIB=1` |
| Expert cache | `BOVARY_STREAM_PHYSICAL_CACHE=1`, `BOVARY_STREAM_RESERVE_UBATCH=1` |
| Decoder | `BOVARY_STREAM_RIGHTSIZE_CTX=1`, `BOVARY_STREAM_LOOKUP_DEPTH=16`, `BOVARY_STREAM_JOINT_DRAFT=48` |
| CUDA | `GGML_CUDA_DISABLE_GRAPHS=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `GGML_BACKEND_PATH=<staged ggml-cuda.dll>` |

See the [third-party notices](NOTICE.md) before redistributing any local build products or model files.
