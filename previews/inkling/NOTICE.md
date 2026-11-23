# Inkling Windows developer preview notices

The preview worker, scripts and project-authored modifications follow Bovary's [MIT license](../../LICENSE). The source patches modify llama.cpp and GGML at [commit 29b52d3](https://github.com/ggml-org/llama.cpp/tree/29b52d398b294509746ec0a3f28948a3107c0213). Retain the [upstream MIT text](llama.cpp-MIT.txt) and the corresponding notices for components in the fetched upstream tree. The root [notice inventory](../../THIRD_PARTY_NOTICES.md) lists further notices relevant to the shared engine dependencies.

The builder uses developer-provided upstream CUDA backend and NVIDIA runtime archives. Their terms remain separate, as do the Microsoft build tools and C++ runtime. The source preview ships no backend binaries, runtime DLLs or model files. The launcher and local builder use hash-locked CUDA 13.5 archives. The archive containing the CUDA backend also includes an LLVM OpenMP license. The runtime ZIP contains cuBLAS, cuBLASLt and CUDART DLLs. For the applicable NVIDIA terms and cuBLAS notices, see the [CUDA component inventory](../../docs/cuda-redistribution.md).

The Inkling checkpoint and its expert sidecar are caller-owned inputs. The sidecar contains model weights copied from the checkpoint and is governed by the applicable model terms. The hashes in [upstream.json](upstream.json) identify the exact measured inputs.
