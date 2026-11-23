# Build the Windows model-choice ZIP

Start with the approved wheel and CPU review ZIP for the version in `pyproject.toml`. Verify them against the release's `SHA256SUMS.txt`. Build the [Inkling preview](../previews/inkling/README.md) in a neutral path on Windows, outside any user profile. Keep the native build's `build-identity.json` and the `LICENSE-jsonhpp` file in the pinned engine's `licenses` directory.

```powershell
$release = 'C:\release'
$preview = 'C:\bovary-preview'
$engine = Join-Path $preview 'llama.cpp-b12600'
python launcher/build_kit.py `
  --wheel (Join-Path $release 'bovary-0.2.0a16-py3-none-any.whl') `
  --cpu-archive (Join-Path $release 'bovary-0.2.0a16-cpu-review.zip') `
  --checksums (Join-Path $release 'SHA256SUMS.txt') `
  --native-bin (Join-Path $preview 'native/bin') `
  --native-checksums (Join-Path $preview 'build-identity.json') `
  --jsonhpp-license (Join-Path $engine 'licenses/LICENSE-jsonhpp') `
  --output (Join-Path $release 'Bovary-Windows-model-choice.zip')
```

Set the release and build directories and use the asset names for the version you are packaging. The builder verifies the approved input hashes and embeds a checksum manifest for the two release files. CUDA dependencies and model weights download during setup.

Run `python -m unittest discover -s launcher -p test_build_kit.py -v` to check ZIP assembly and the embedded checksum handshake without models. Follow the [release review](../docs/release-review.md) for the Windows hardware test.
