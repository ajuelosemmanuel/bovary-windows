"""Install the optional Inkling CUDA preview beside the friend launcher."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
from zipfile import ZipFile

from pack import create
from preview import LOCK, NATIVE, stage_cuda, verify
from run import register, registered_inputs


ARCHIVE_SIZES = {
    "llama-b12600-bin-win-cuda-13.5-x64.zip": 225604843,
    "cudart-llama-bin-win-cuda-13.5-x64.zip": 426314199,
}
MODEL = LOCK["model"]


def gemma_cuda_environment(state, payload, env, worker_digest):
    """Stage Gemma's three NVIDIA runtime DLLs without installing Inkling or its model."""
    from cuda_default import compatible_device, gemma_preset

    home = Path(env["BOVARY_HOME"])
    if gemma_preset(home, env) != "cpu-only":
        return env
    worker = payload / "gemma-cuda/bovary-infer.exe"
    if not worker_digest or not worker.is_file() or not compatible_device():
        return None
    verify(worker, worker_digest)
    archive_name = "cudart-llama-bin-win-cuda-13.5-x64.zip"
    record = LOCK["cuda_archives"][archive_name]
    folder = state / "inkling"  # Both installers can reuse the verified runtime archive.
    folder.mkdir(exist_ok=True)
    runtime = state / ("cuda-runtime-" + record["sha256"][:16])
    if not runtime.is_dir():
        print("Preparing the Gemma CUDA runtime (426 MB download on first use)...", flush=True)
        archive = folder / archive_name
        download(f"https://github.com/ggml-org/llama.cpp/releases/download/b12600/{archive_name}",
                 archive, ARCHIVE_SIZES[archive_name], record["sha256"])
        with tempfile.TemporaryDirectory(dir=state) as temp:
            stage = Path(temp) / "bin"
            stage.mkdir()
            with ZipFile(archive) as source:
                for name in record["files"]:
                    matches = [item for item in source.infolist() if Path(item.filename).name == name]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one {name} in the CUDA runtime")
                    with source.open(matches[0]) as src, (stage / name).open("xb") as dst:
                        shutil.copyfileobj(src, dst)
            stage.rename(runtime)
    configured = dict(env, BOVARY_CUDA_EXE=str(worker),
                      PATH=str(runtime) + os.pathsep + env.get("PATH", ""))
    return configured if gemma_preset(home, configured) != "cpu-only" else None


def download(url, destination, size, digest):
    if destination.exists():
        verify(destination, digest)
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if not partial.exists() or partial.stat().st_size < size:
        subprocess.run(["curl.exe", "--disable", "--fail", "--location", "--proto", "=https",
                        "--proto-redir", "=https", "--retry", "3", "--connect-timeout", "30",
                        "--continue-at", "-", "--output", str(partial), url], check=True)
    if partial.stat().st_size != size:
        raise ValueError(f"Unexpected size for {partial.name}. Remove that partial file and retry.")
    try:
        verify(partial, digest)
    except ValueError as error:
        raise ValueError(f"Checksum mismatch. Remove {partial} and retry.") from error
    partial.rename(destination)


def install(state, payload):
    try:
        driver = ctypes.WinDLL("nvcuda.dll", winmode=0x800)
        devices = ctypes.c_int()
        if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(devices)) or devices.value < 1:
            raise OSError("no CUDA device")
    except OSError as error:
        raise ValueError("Inkling requires an NVIDIA CUDA GPU and compatible driver. Choose Gemma for CPU.") from error

    folder = state / "inkling"
    folder.mkdir(exist_ok=True)
    binaries = folder / "bin"
    model = folder / MODEL["filename"]
    pack = model.with_suffix(".experts.pack")
    receipt = folder / "inputs.json"
    if not (binaries.is_dir() and receipt.is_file()):
        downloads = {model: MODEL["bytes"]} if not receipt.is_file() else {}
        if not binaries.is_dir():
            downloads.update({folder / name: size for name, size in ARCHIVE_SIZES.items()})
        needed = 3 * 1024**3 + (800_000_000 if not binaries.is_dir() else 0)
        # The sidecar is rebuilt atomically; only downloads can resume from a partial file.
        if not receipt.is_file() and not pack.is_file():
            needed += MODEL["packed_bytes"]
        for path, size in downloads.items():
            partial = path.with_suffix(path.suffix + ".partial")
            have = path.stat().st_size if path.is_file() else partial.stat().st_size if partial.is_file() else 0
            needed += max(0, size - have)
        if shutil.disk_usage(folder).free < needed:
            raise ValueError(f"Inkling needs {needed / 1e9:.1f} GB free to finish this setup.")
        if not binaries.is_dir():
            for name, data in LOCK["cuda_archives"].items():
                download(f"https://github.com/ggml-org/llama.cpp/releases/download/b12600/{name}",
                         folder / name, ARCHIVE_SIZES[name], data["sha256"])
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                stage = Path(temp) / "bin"
                stage.mkdir()
                pins = json.loads((payload / "inkling-native/sha256.json").read_bytes())
                for name in NATIVE:
                    shutil.copyfile(payload / "inkling-native" / name, stage / name)
                    verify(stage / name, pins[name])
                stage_cuda(folder, stage)
                # A separate process releases Windows DLL handles before publishing the directory.
                subprocess.run([sys.executable, str(state / "runtime/previews/inkling/preview.py"),
                                "smoke", "--bin", str(stage)], check=True)
                stage.rename(binaries)
        if not receipt.exists():
            print("Downloading and verifying Inkling (17.63 GB)...", flush=True)
            download(f"https://huggingface.co/{MODEL['repository']}/resolve/{MODEL['revision']}/{model.name}",
                     model, MODEL["bytes"], MODEL["sha256"])
            if not pack.exists():
                print("Preparing the verified SSD layout (15.80 GB)...", flush=True)
                create(model, pack)
            # Publish the registration only after both files have passed validation.
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                staged = Path(temp) / "inputs.json"
                register(model, pack, staged)
                staged.rename(receipt)
    registered_inputs(receipt)
    os.environ.update(BOVARY_INKLING_BIN=str(binaries), BOVARY_INKLING_RECEIPT=str(receipt))
