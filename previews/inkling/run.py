"""Register local inputs or run one prompt using the opt-in Inkling setup."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from preview import LOCK

MANAGED_PREFIXES = ("BOVARY_STREAM", "BOVARY_PACKED", "LOWCOST_STREAM", "LLAMA_ARG_", "GGML_")
PROFILE = {
    "BOVARY_STREAM": "1",
    "BOVARY_STREAM_IO": "unbuffered",
    "BOVARY_STREAM_LOAD_BUFFER_MIB": "1",
    "BOVARY_STREAM_BUFFER_MIB": "9",
    "BOVARY_STREAM_POOL": "pinned2",
    "GGML_CUDA_DISABLE_GRAPHS": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "BOVARY_STREAM_PHYSICAL_CACHE": "1",
    "BOVARY_STREAM_RESERVE_UBATCH": "1",
    "BOVARY_STREAM_LOOKUP_DEPTH": "16",
    "BOVARY_STREAM_RIGHTSIZE_CTX": "1",
    "BOVARY_STREAM_JOINT_DRAFT": "48",
}

INPUT_FIELDS = {"model": ("bytes", "sha256"), "pack": ("packed_bytes", "packed_sha256")}
MAX_PROMPT_BYTES = 1024 * 1024


def validate_prompt(path):
    message = "Pass a nonempty UTF-8 prompt of at most 1 MiB"
    with path.open("rb") as source:
        prompt = source.read(MAX_PROMPT_BYTES + 1)
    if len(prompt) > MAX_PROMPT_BYTES:
        raise ValueError(message)
    try:
        valid = bool(prompt.decode("utf-8").strip())
    except UnicodeDecodeError as error:
        raise ValueError(message) from error
    if not valid:
        raise ValueError(message)


def identity(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("The checkpoint and sidecar must be regular files")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def register(model, pack, receipt):
    records = {}
    for kind, path in (("model", model), ("pack", pack)):
        path = path.resolve(strict=True)
        size, sha = INPUT_FIELDS[kind]
        with path.open("rb") as source:
            before = identity(os.fstat(source.fileno()))
            if before[2] != LOCK["model"][size]:
                raise ValueError(f"Unexpected input length: {path}")
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = identity(os.fstat(source.fileno()))
        if digest != LOCK["model"][sha] or before != after or before != identity(path.stat()):
            raise ValueError(f"Input changed or did not match the pinned SHA-256: {path}")
        records[kind] = dict(path=str(path), identity=before, sha256=digest)
    with receipt.open("x", encoding="utf-8") as output:
        json.dump({"version": 1, "inputs": records}, output, indent=2)
        output.write("\n")


def registered_inputs(receipt):
    data = json.loads(receipt.read_bytes())
    if data.get("version") != 1:
        raise ValueError(f"Unsupported input registration: {receipt}")
    result = {}
    for kind, (size, sha) in INPUT_FIELDS.items():
        record = data["inputs"][kind]
        path = Path(record["path"]).resolve(strict=True)
        current = identity(path.stat())
        if (str(path) != record["path"] or current != record["identity"]
                or current[2] != LOCK["model"][size] or record["sha256"] != LOCK["model"][sha]):
            raise ValueError(f"Input changed since registration: {path}")
        result[kind] = path
    return result


def configuration(args):
    worker = args.bin / "bovary-infer.exe"
    cmd = [str(worker), "--model", str(args.model), "--context", "2048",
           "--threads", "2", "--gpu-layers", "99", "--cpu-experts", "--gpu-embedding",
           "--cache-mib", "2048", "--prompt-batch", "176", "--load-mode", "direct",
           "--direct-answer", "--no-op-offload", "--prompt-file", str(args.prompt),
           "--predict", str(args.tokens), "--metrics", str(args.metrics)]
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(MANAGED_PREFIXES)}
    env.update(PROFILE, BOVARY_STREAM_MODEL=str(args.model), BOVARY_STREAM_PACKED=str(args.pack),
               GGML_BACKEND_PATH=str(args.bin / "ggml-cuda.dll"))
    return cmd, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("register")
    for name in ("model", "pack", "receipt"):
        add.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("bin", "receipt", "prompt", "metrics"):
        run.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    run.add_argument("--tokens", type=int, default=160)
    args = parser.parse_args()
    if args.action == "register":
        register(args.model, args.pack, args.receipt)
        print(f"Registered model and sidecar: {args.receipt.resolve()}")
        return 0
    if not 1 <= args.tokens < 2048:
        parser.error("Tokens must be between 1 and 2047, including room for the prompt")
    if args.metrics.exists():
        parser.error("Select a new metrics output file")
    for kind, path in registered_inputs(args.receipt).items():
        setattr(args, kind, path)
    try:
        validate_prompt(args.prompt)
    except ValueError as error:
        parser.error(str(error))
    cmd, env = configuration(args)
    return subprocess.run(cmd, cwd=args.bin, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
