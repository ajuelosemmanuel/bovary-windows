# Build the Windows model-choice ZIP

Start with the approved wheel and CPU review ZIP for the version in `pyproject.toml`. Verify them against the release's `SHA256SUMS.txt`. Build the [Inkling preview](../previews/inkling/README.md) in a neutral path on Windows, outside any user profile. Keep the native build's `build-identity.json` and the `LICENSE-jsonhpp` file in the pinned engine's `licenses` directory.

```powershell
$release = 'C:\release'
$preview = 'C:\bovary-preview'
$engine = Join-Path $preview 'llama.cpp-b12600'
$gemma = 'C:\bovary-gemma-cuda'
$toolkit = 'C:\cuda-toolkit'
$gemmaHash = Read-Host 'Approved SHA-256 for the neutral Gemma CUDA build'
python launcher/build_kit.py `
  --wheel (Join-Path $release 'bovary-0.2.0a17-py3-none-any.whl') `
  --cpu-archive (Join-Path $release 'bovary-0.2.0a17-cpu-review.zip') `
  --checksums (Join-Path $release 'SHA256SUMS.txt') `
  --native-bin (Join-Path $preview 'native/bin') `
  --native-checksums (Join-Path $preview 'build-identity.json') `
  --jsonhpp-license (Join-Path $engine 'licenses/LICENSE-jsonhpp') `
  --gemma-cuda-worker (Join-Path $gemma 'work/build/bovary-cuda/bovary-infer.exe') `
  --gemma-cuda-sha256 $gemmaHash `
  --gemma-cuda-license (Join-Path $toolkit 'LICENSE') `
  --output (Join-Path $release 'bovary-0.2.0a17-windows-cuda-x64.zip')
```

Set the release and build directories and use the asset names for the version you are packaging. The builder verifies the approved input hashes and embeds a checksum manifest for the two release files. CUDA dependencies and model weights download during setup.

For a CUDA-first Gemma kit, build the [Gemma CUDA worker](../docs/reproducibility.md#build-and-try-the-cuda-worker) from the pinned checkout in a neutral directory. Supply its executable using `--gemma-cuda-worker`, the approved native build digest using `--gemma-cuda-sha256`, and the toolkit's `LICENSE` file using `--gemma-cuda-license`. The launcher probes the NVIDIA driver and worker before choosing its backend. The automatic profile targets the tested 6 GB GPU class, compute capability 8.6 or newer, and a CUDA 13 driver. On first use it fetches the pinned 426 MB runtime archive. Choose **C** for CPU-only, including on NVIDIA laptops.

Run `python -m unittest discover -s launcher -p test_build_kit.py -v` to check ZIP assembly and the embedded checksum handshake without models. Follow the [release review](../docs/release-review.md) for the Windows hardware test.
