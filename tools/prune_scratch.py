"""Plan by default; remove only verified task-created scratch after benchmarking."""
import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from urllib.parse import urlparse

try:
    from .benchmark import SerialLock
    from .footprint import ROOT, allocation_info, is_reparse
except ImportError:
    from benchmark import SerialLock
    from footprint import ROOT, allocation_info, is_reparse

ARCHIVES = {
    "cuda_nvcc": "cuda_nvcc-windows-x86_64-13.3.73-archive.zip",
    "cuda_crt": "cuda_crt-windows-x86_64-13.3.73-archive.zip",
    "cuda_cudart": "cuda_cudart-windows-x86_64-13.3.29-archive.zip",
    "libcublas": "libcublas-windows-x86_64-13.6.0.2-archive.zip",
    "cccl": "cccl-windows-x86_64-13.3.3.4.1-archive.zip",
    "libnvvm": "libnvvm-windows-x86_64-13.3.73-archive.zip",
}

def checked(path):
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Resolved scratch target escaped project")
    for part in (path, *path.parents):
        if is_reparse(part):
            raise ValueError("Refusing scratch reparse path")
        if part == ROOT:
            break
    return resolved

def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def tree_files(path):
    result = []
    def refuse_unreadable(error):
        raise error
    for parent, directories, files in os.walk(path, followlinks=False, onerror=refuse_unreadable):
        for child in directories:
            checked(Path(parent, child))
        for child in files:
            file = checked(Path(parent, child))
            result.append((file, file.stat().st_size))
    return result

def plan(targets):
    records = []
    if "cuda-archives" in targets:
        manifest = json.loads(checked(ROOT/"work/deps/cuda-13.3/portable-install.json").read_text(encoding="utf-8"))
        if manifest.get("cuda_release") != "13.3.1":
            raise ValueError("Unexpected portable toolkit provenance")
        components = {x["component"]: x for x in manifest["components"]}
        for component, filename in ARCHIVES.items():
            record = components[component]
            if Path(urlparse(record["source"]).path).name != filename:
                raise ValueError("Archive provenance filename changed")
            file = ROOT/"work/deps"/filename
            if not file.exists():
                records.append({"path": str(file), "kind": "archive", "status": "absent"})
                continue
            file = checked(file)
            if file.stat().st_size != record["download_bytes"] or hash_file(file) != record["sha256"]:
                raise ValueError(f"Archive differs from verified task download: {filename}")
            info = allocation_info(file)
            records.append({"path": str(file), "kind": "archive", "status": "verified",
                            "logical_bytes": file.stat().st_size, "sha256": record["sha256"], **info})
    if "duplicate-research-engine" in targets:
        target = ROOT/"work/research/llama.cpp"
        if not target.exists():
            records.append({"path": str(target), "kind": "duplicate_repository", "status": "absent"})
        else:
            target = checked(target)
            if target != (ROOT/"work/research/llama.cpp").absolute() or target == (ROOT/"work/engine/bovary-llama.cpp").resolve():
                raise ValueError("Duplicate-repository target changed")
            git_dir = checked(target/".git")
            if not git_dir.is_dir():
                raise ValueError("Refusing a worktree/shared gitdir")
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(checked(git_dir/"config"), encoding="utf-8")
            if parser.get('remote "origin"', "url") not in ("https://github.com/ggml-org/llama.cpp.git", "https://github.com/ggml-org/llama.cpp"):
                raise ValueError("Repository origin differs from the task clone")
            revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if revision != "2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c":
                raise ValueError("Repository revision changed; refusing automatic removal")
            status = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=all", "--ignored=matching"], text=True)
            if status.strip():
                raise ValueError("Repository has local changes; refusing removal")
            files = tree_files(target)
            records.append({"path": str(target), "kind": "duplicate_repository", "status": "verified",
                            "revision": revision, "files": len(files), "logical_bytes": sum(size for _,size in files)})
    return records

def clear_readonly(function, path, exception):
    path = checked(path)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    function(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=("cuda-archives", "duplicate-research-engine"),
                        default=["cuda-archives", "duplicate-research-engine"])
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/scratch-pruning.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != (ROOT/"outputs").resolve() or not output.name.startswith("scratch-prun") or output.suffix != ".json":
        raise ValueError("Report must be outputs/scratch-prun*.json")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "apply": args.apply,
              "preserved": ["work/engine/bovary-llama.cpp", "work/research/turbo-fieldfare", "work/models", "outputs raw measurements", "extracted CUDA toolkit"],
              "limitations": "Only the two explicitly named scratch categories are eligible. Model files, shared user caches and modified repositories are never pruned."}
    with SerialLock(ROOT/"work/benchmark.lock"):
        report["targets"] = plan(args.targets)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for item in report["targets"]:
            if item["status"] != "verified" or not args.apply:
                continue
            path = checked(item["path"])
            if item["kind"] == "archive":
                path.unlink()
            else:
                # Check every child before this recursive operation; do not follow links.
                tree_files(path)
                shutil.rmtree(path, onexc=clear_readonly)
            item["status"] = "removed"
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"apply": args.apply, "targets": report["targets"], "report": str(output)}, indent=2))

if __name__ == "__main__":
    main()
