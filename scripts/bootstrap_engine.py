"""Fetch the pinned engine and apply its patch. Never reset or clean user files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "engine.lock.json"


def git(directory, *args, env=None):
    result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode().strip()


def verify(directory, lock, patch):
    if git(directory, "rev-parse", "HEAD") != lock["revision"]:
        raise RuntimeError("Unexpected engine revision; preserve this checkout and use a new --directory")
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git(directory, "read-tree", "HEAD", env=env)
        git(directory, "apply", "--cached", "--whitespace=nowarn", str(patch), env=env)
        differences = git(directory, "diff", "--name-only", env=env)
        untracked = git(directory, "ls-files", "--others", "--exclude-standard", env=env)
        if differences or untracked:
            raise RuntimeError("Engine differs from pinned patched tree:\n" + differences + "\n" + untracked)


def bootstrap(directory, *, check=False, lock_path=LOCK):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    patch = (ROOT / lock["patch"]).resolve()
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise RuntimeError("Patch hash does not match engine.lock.json")
    directory = Path(directory).resolve()
    if not (directory / ".git").exists():
        if check:
            raise RuntimeError("Engine absent; run python scripts/bootstrap_engine.py first")
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("Engine directory is nonempty and is not a Git checkout")
        directory.mkdir(parents=True, exist_ok=True)
        git(directory, "init")
        git(directory, "remote", "add", "origin", lock["repository"])
    try:
        revision = git(directory, "rev-parse", "--verify", "HEAD")
    except RuntimeError:
        if check:
            raise RuntimeError("Engine checkout is incomplete; rerun bootstrap without --check") from None
        # A failed first fetch can leave only an initialized, unborn repository.
        # Never repair a checkout containing user files, index entries or a new origin.
        try:
            branch = git(directory, "symbolic-ref", "-q", "HEAD")
            origin = git(directory, "remote", "get-url", "origin")
        except RuntimeError:
            raise RuntimeError("Unexpected incomplete engine repository; use a new --directory") from None
        if (not branch.startswith("refs/heads/") or origin != lock["repository"]
                or any(path.name != ".git" for path in directory.iterdir())
                or git(directory, "ls-files")):
            raise RuntimeError("Incomplete engine contains user files, index entries or an unexpected origin; no changes were made")
        git(directory, "fetch", "--depth", "1", "origin", lock["revision"])
        git(directory, "checkout", "--detach", "FETCH_HEAD")
        revision = git(directory, "rev-parse", "--verify", "HEAD")
    if revision != lock["revision"]:
        raise RuntimeError("Unexpected engine revision; no changes were made")
    try:
        verify(directory, lock, patch)
    except RuntimeError:
        if check or git(directory, "status", "--porcelain"):
            raise
        git(directory, "apply", "--check", str(patch))
        git(directory, "apply", "--whitespace=nowarn", str(patch))
        verify(directory, lock, patch)
    return {"revision": lock["revision"], "patch_sha256": lock["patch_sha256"], "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "work/engine/bovary-llama.cpp")
    parser.add_argument("--check", action="store_true", help="Offline verification only; no fetching or patching")
    args = parser.parse_args()
    try:
        print(json.dumps(bootstrap(args.directory, check=args.check), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Engine bootstrap: {error}\n")


if __name__ == "__main__":
    main()
