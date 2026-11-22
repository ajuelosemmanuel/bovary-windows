"""Reconstruct, build and smoke-test the opt-in Windows Inkling worker."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "upstream.json").read_bytes())
NATIVE = ("bovary-infer.exe", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "llama.dll")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def reconstruct(archive, output):
    verify(archive, LOCK["llama_cpp"]["sha256"])
    for name, expected in LOCK["preview_source"].items():
        verify(HERE / name, expected)
    for patch in LOCK["patches"]:
        verify(HERE / patch["name"], patch["sha256"])
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(output)
    source = output / "llama.cpp-b12600"
    env = os.environ.copy()
    env["GIT_CEILING_DIRECTORIES"] = str(source.parent)
    for patch in LOCK["patches"]:
        args = [f"-p{patch['strip']}", str(HERE / patch["name"])]
        for switches in (["--check"], []):
            subprocess.run(["git", "apply", *switches, *args], cwd=source, env=env, check=True)
    for name, expected in LOCK["patched_engine_source"].items():
        verify(source / name, expected)
    verify(source / "LICENSE", digest(HERE / "llama.cpp-MIT.txt"))
    return source


def stage_cuda(directory, destination):
    for name, record in LOCK["cuda_archives"].items():
        archive = directory / name
        verify(archive, record["sha256"])
        with ZipFile(archive) as zip_file:
            for dll in record["files"]:
                matches = [entry for entry in zip_file.infolist() if Path(entry.filename).name == dll]
                if len(matches) != 1:
                    raise ValueError(f"expected one {dll} in {archive}")
                with zip_file.open(matches[0]) as data, (destination / dll).open("xb") as output:
                    shutil.copyfileobj(data, output)


def smoke(directory):
    worker = directory / NATIVE[0]
    result = subprocess.check_output([str(worker), "--protocol-version"], cwd=directory, timeout=30)
    if result.strip() != b"bovary-worker 2":
        raise ValueError(f"Unexpected protocol version: {result!r}")
    plugin = directory / "ggml-cuda.dll"
    with os.add_dll_directory(str(directory)):
        # Resolve sibling CUDA runtime DLLs using the explicit DLL search path.
        _cuda = ctypes.CDLL(str(plugin))
        ggml = ctypes.CDLL(str(directory / "ggml.dll"))
        ggml.ggml_backend_load.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load.restype = ctypes.c_void_p
        base = ctypes.CDLL(str(directory / "ggml-base.dll"))
        base.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_name.restype = ctypes.c_char_p
        backend = ggml.ggml_backend_load(os.fsencode(plugin))
        if not backend or base.ggml_backend_reg_name(backend) != b"CUDA":
            raise ValueError(f"Cannot register the CUDA backend: {plugin}")
    print("Worker protocol 2 and CUDA plugin registration passed.")


def build(source, output, cuda_archives, jobs):
    native = output / "native"
    subprocess.run(["cmake", "-S", str(HERE), "-B", str(native), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release",
                    f"-DINKLING_ENGINE_SOURCE={source}"], check=True)
    subprocess.run(["cmake", "--build", str(native), "--parallel", str(jobs)], check=True)
    bin_dir = native / "bin"
    stage_cuda(cuda_archives, bin_dir)
    smoke(bin_dir)
    artifacts = [*NATIVE, *(dll for record in LOCK["cuda_archives"].values() for dll in record["files"])]
    manifest = {name: digest(bin_dir / name) for name in artifacts}
    (output / "build-identity.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built preview in {bin_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "build"):
        command = sub.add_parser(name)
        command.add_argument("--upstream", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "build":
            command.add_argument("--cuda-archives", type=Path, required=True)
            command.add_argument("--jobs", type=int, default=4)
    sub.add_parser("smoke").add_argument("--bin", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "smoke":
        smoke(args.bin.resolve())
        return
    output = args.output.resolve()
    source = reconstruct(args.upstream.resolve(), output)
    if args.action == "build":
        build(source, output, args.cuda_archives.resolve(), args.jobs)
    else:
        print(f"Verified and reconstructed {source}")


if __name__ == "__main__":
    main()
