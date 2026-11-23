"""Assemble the Windows model-choice ZIP from approved release inputs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIEW = ROOT / "previews" / "inkling"
NATIVE = ("bovary-infer.exe", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "llama.dll")
JSONHPP_SHA256 = "c0d068392ea65358b798b8c165103560f06e9e3b38c4ab4e2d8810a7b931af86"
CUDA_TOOLKIT_LICENSE_SHA256 = "088381bc2d891e719a2a9398645b00bb45f3b24231473a8283ac7e3e66b8a028"
HASH = re.compile(r"[0-9a-fA-F]{64}")
PRIVATE_PATH = re.compile(rb"[a-z]:[/\\]users[/\\]", re.IGNORECASE)
CPU_REQUIRED = {
    "work/build/bovary-cpu/bovary-infer.exe",
    "tools/download_model.py",
    "tools/verify_model.py",
    *(f"previews/inkling/{name}" for name in ("pack.py", "preview.py", "run.py", "upstream.json")),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def release_checksums(path):
    checksums = {}
    for row in path.read_text(encoding="utf-8-sig").splitlines():
        fields = row.split()
        if len(fields) != 2 or not HASH.fullmatch(fields[0]) or fields[1] in checksums:
            raise ValueError(f"Malformed or duplicate checksum in {path}: {row!r}")
        digest, name = fields
        checksums[name] = digest
    return checksums


def checked_bytes(path, expected):
    if not isinstance(expected, str) or not HASH.fullmatch(expected):
        raise ValueError(f"Missing or invalid expected SHA-256 for {path.name}")
    data = path.read_bytes()
    if sha256(data) != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}")
    return data


def build_kit(*, wheel, cpu_archive, checksums, native_bin, native_checksums, jsonhpp_license,
              output, gemma_cuda_worker=None, gemma_cuda_sha256=None, gemma_cuda_license=None):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if (wheel.name, cpu_archive.name) != (f"bovary-{version}-py3-none-any.whl", f"bovary-{version}-cpu-review.zip"):
        raise ValueError(f"Expected the wheel and CPU review ZIP for Bovary {version}")
    approved = release_checksums(checksums)
    payloads = {path.name: checked_bytes(path, approved.get(path.name)) for path in (cpu_archive, wheel)}
    with ZipFile(io.BytesIO(payloads[cpu_archive.name])) as cpu:
        missing = CPU_REQUIRED.difference(cpu.namelist())
        if missing:
            raise ValueError(f"CPU archive lacks launcher inputs: {', '.join(sorted(missing))}")
        notices = {name: cpu.read(name) for name in ("LICENSE", "THIRD_PARTY_NOTICES.md")}

    native_approved = json.loads(native_checksums.read_bytes())
    if not isinstance(native_approved, dict):
        raise ValueError("Expected a JSON map of native filenames to build hashes")
    native = {name: checked_bytes(native_bin / name, native_approved.get(name)) for name in NATIVE}
    for name, data in native.items():
        if PRIVATE_PATH.search(data.replace(b"\x00", b"")):
            raise ValueError(f"Private Windows build path in {name}. Build from a neutral path.")

    manifest = "".join(f"{sha256(data)}  {name}\n" for name, data in payloads.items()).encode("ascii")
    setup = (HERE / "friend_setup.py").read_text(encoding="utf-8")
    for marker, value in {
        "@CPU_ARCHIVE@": cpu_archive.name,
        "@WHEEL@": wheel.name,
        "@MANIFEST_SHA256@": sha256(manifest),
        "@GEMMA_CUDA_SHA256@": (gemma_cuda_sha256 or "").lower(),
        "@INSTALL_MARKER@": f"installed-{version}-{sha256(manifest)[:16]}",
    }.items():
        if setup.count(marker) != 1:
            raise ValueError(f"Expected one {marker} substitution in friend_setup.py")
        setup = setup.replace(marker, value)
    members = {
        "Start Bovary.cmd": (HERE / "start_bovary.cmd").read_bytes(),
        "READ ME.txt": (HERE / "usage.txt").read_bytes(),
        "setup/friend_setup.py": setup.encode("utf-8"),
        "setup/inkling_setup.py": (HERE / "inkling_setup.py").read_bytes(),
        "setup/ask.py": (PREVIEW / "ask.py").read_bytes(),
        "setup/cuda_default.py": (ROOT / "python/bovary/cuda_default.py").read_bytes(),
        "setup/SHA256SUMS.txt": manifest,
        **{f"setup/{name}": data for name, data in (payloads | notices).items()},
        **{f"setup/inkling-native/{name}": data for name, data in native.items()},
        "setup/inkling-native/sha256.json": (json.dumps({name: sha256(data) for name, data in native.items()}, indent=2) + "\n").encode("ascii"),
        "setup/inkling-native/llama.cpp-MIT.txt": (PREVIEW / "llama.cpp-MIT.txt").read_bytes(),
        "setup/inkling-native/LICENSE-jsonhpp": checked_bytes(jsonhpp_license, JSONHPP_SHA256),
    }
    if any((gemma_cuda_worker, gemma_cuda_sha256, gemma_cuda_license)):
        if not all((gemma_cuda_worker, gemma_cuda_sha256, gemma_cuda_license)):
            raise ValueError("Supply the Gemma CUDA worker, its SHA-256, and the toolkit license together")
        worker = checked_bytes(gemma_cuda_worker, gemma_cuda_sha256)
        if PRIVATE_PATH.search(worker.replace(b"\x00", b"")):
            raise ValueError("Private Windows build path in Gemma CUDA worker")
        members["setup/gemma-cuda/bovary-infer.exe"] = worker
        members["setup/gemma-cuda/CUDA-LICENSE.txt"] = checked_bytes(gemma_cuda_license, CUDA_TOOLKIT_LICENSE_SHA256)
    # Fixed member metadata makes identical input bytes produce the same archive.
    with ZipFile(output, "x") as kit:
        for name, data in members.items():
            member = ZipInfo(name)
            member.create_system = 0
            kit.writestr(member, data, compress_type=ZIP_DEFLATED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--cpu-archive", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path, help="Approved release SHA256SUMS.txt")
    parser.add_argument("--native-bin", required=True, type=Path, help="Neutral-path Inkling build's native/bin")
    parser.add_argument("--native-checksums", required=True, type=Path, help="JSON manifest from the approved native build")
    parser.add_argument("--jsonhpp-license", required=True, type=Path, help="licenses/LICENSE-jsonhpp from the pinned engine")
    parser.add_argument("--output", required=True, type=Path, help="New launcher ZIP")
    parser.add_argument("--gemma-cuda-worker", type=Path, help="Neutral-path Windows Gemma CUDA build")
    parser.add_argument("--gemma-cuda-sha256", help="Approved digest of the Gemma CUDA executable")
    parser.add_argument("--gemma-cuda-license", type=Path, help="LICENSE from the pinned CUDA 13.3.1 toolkit")
    args = parser.parse_args()
    try:
        build_kit(**vars(args))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot build the launcher ZIP: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
