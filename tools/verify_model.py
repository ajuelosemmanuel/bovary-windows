"""Verify existing pinned weights offline and publish a non-clobbering proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.modelproof import (
    GEMMA_REPOSITORY as REPO, GEMMA_REVISION as REV, GEMMA_FILENAME as NAME,
    GEMMA_BYTES as SIZE, GEMMA_SHA256 as SHA, check_binding, file_identity, path_identity,
)


def expected_manifest(filename):
    return {"repository": REPO, "revision": REV, "filename": filename,
            "bytes": SIZE, "sha256": SHA, "verified": True,
            "optional_components": "vision, MTP, DFlash omitted"}


def verify_file(model):
    """Read the entire file without modifying it; reject changes during hashing."""
    model = Path(model)
    with model.open("rb") as source:
        before = file_identity(source)
        if before["bytes"] != SIZE:
            raise ValueError(f"Checkpoint size mismatch: expected {SIZE} bytes")
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        if before != file_identity(source):
            raise ValueError("Checkpoint changed while it was being verified")
    if before != path_identity(model):
        raise ValueError("Checkpoint path changed while it was being verified")
    if digest != SHA:
        raise ValueError("Checkpoint SHA256 mismatch; no verification proof was written")
    return {**expected_manifest(model.name), "file_identity": before}


def existing_manifest(path, expected):
    """Preserve an existing valid proof, rejecting conflicting or malformed ones."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Manifest must be a regular file, not a symbolic link")
    try:
        with path.open("rb") as source:
            raw = source.read(65537)
    except FileNotFoundError:
        return None
    try:
        if len(raw) > 65536:
            raise ValueError("Manifest exceeds 64 KiB")
        value = json.loads(raw.decode("utf-8-sig"))
        if (not isinstance(value, dict) or value.get("verified") is not True
                or value.get("filename") != expected["filename"]
                or type(value.get("bytes")) is not int or value["bytes"] != expected["bytes"]
                or not isinstance(value.get("sha256"), str)
                or value["sha256"].lower() != expected["sha256"]):
            raise ValueError("Manifest identity differs")
    except (UnicodeError, ValueError) as error:
        raise ValueError("Existing manifest is invalid or describes another checkpoint; choose a new manifest path") from error
    return value


def publish_no_replace(source, destination):
    """Atomically publish on Windows or POSIX, refusing an existing target."""
    if os.name == "nt":
        os.rename(source, destination)  # Windows rename never replaces a target.
    else:
        os.link(source, destination)


def write_manifest(path, metadata):
    """Publish complete bytes atomically, without replacing an existing proof."""
    path = Path(path)
    retained = existing_manifest(path, metadata)
    if retained is not None:
        if retained.get("file_identity") != metadata.get("file_identity"):
            raise ValueError("Existing manifest is stale or legacy; choose a new manifest path")
        return retained
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                prefix=path.name+".", suffix=".tmp", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(metadata, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            publish_no_replace(temporary, path)
        except FileExistsError:
            retained = existing_manifest(path, metadata)
            if retained is None:
                raise ValueError("Manifest changed during verification; retry with a new manifest path")
            if retained.get("file_identity") != metadata.get("file_identity"):
                raise ValueError("Existing manifest is stale or legacy; choose a new manifest path")
            return retained
        return metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_model(model, manifest=None):
    model = Path(model)
    manifest = Path(manifest) if manifest is not None else model.parent/"model-provenance.json"
    metadata = expected_manifest(model.name)
    existing_manifest(manifest, metadata)
    return write_manifest(manifest, verify_file(model))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("work/models")/NAME)
    parser.add_argument("--manifest", type=Path, help="Default: model-provenance.json beside the model")
    args = parser.parse_args(argv)
    try:
        metadata = verify_model(args.model, args.manifest)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Verification failed: {error}\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
