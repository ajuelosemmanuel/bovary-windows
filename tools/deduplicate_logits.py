"""Dry-run by default: share only byte-identical, completed raw logit artifacts."""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

try:
    from .benchmark import SerialLock
    from .footprint import ROOT, allocation_info, is_reparse, kernel
except ImportError:
    from benchmark import SerialLock
    from footprint import ROOT, allocation_info, is_reparse, kernel

OUTPUTS = ROOT / "outputs"
CHUNK = 1024 * 1024

def safe_path(path, exists=True):
    path = Path(path).absolute()
    if not path.is_relative_to(OUTPUTS.absolute()):
        raise ValueError("Logit operations are restricted to project outputs")
    for part in (path, *path.parents):
        if is_reparse(part):
            raise ValueError(f"Reparse path refused: {part}")
        if part == OUTPUTS:
            break
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(OUTPUTS.resolve()) or not OUTPUTS.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Resolved path escaped project outputs")
    return resolved

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()

def equal_bytes(left, right):
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if x != y:
                return False
            if not x:
                return True

@contextmanager
def deny_writers(path):
    # Permit atomic renaming while denying new content writers during validation.
    handle = kernel.CreateFileW(str(path), 0x80, 1 | 4, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "Cannot hold completed logit file against writers", str(path))
    try:
        yield
    finally:
        kernel.CloseHandle(handle)

def completed_file(path):
    metrics_path, result_path = path.parent/"native-metrics.json", path.parent/"result.json"
    if not metrics_path.exists() or not result_path.exists():
        return False
    metrics = json.loads(safe_path(metrics_path).read_text(encoding="utf-8"))
    result = json.loads(safe_path(result_path).read_text(encoding="utf-8"))
    if result.get("return_code") != 0 or result.get("timed_out"):
        return False
    vocab = metrics.get("vocab_size")
    tokens = metrics.get("actual_generated_tokens")
    if not isinstance(vocab, int) or not 1 <= vocab <= 2**20 or not isinstance(tokens, int) or tokens < 0:
        return False
    rows = tokens + int(bool(metrics.get("early_stop")))
    return rows > 0 and path.stat().st_size == rows * vocab * 4

def collect():
    safe_path(OUTPUTS)
    files, skipped = [], []
    for directory, names, leaves in os.walk(OUTPUTS, followlinks=False,
                                          onerror=lambda error: skipped.append({"path": str(error.filename), "reason": str(error)})):
        for name in sorted(names):
            if is_reparse(Path(directory, name)):
                skipped.append({"path": str(Path(directory, name)), "reason": "Reparse directory excluded"})
        names[:] = sorted(name for name in names if not is_reparse(Path(directory, name)))
        if "logits.f32" not in leaves:
            continue
        path = Path(directory, "logits.f32")
        try:
            path = safe_path(path)
            if not completed_file(path):
                skipped.append({"path": str(path), "reason": "No complete successful run and consistent full logit dump"})
                continue
            info = allocation_info(path)
            if info.get("reparse_point") or not info.get("file_id") or info.get("allocated_bytes") is None:
                raise ValueError("Regular-file identity unavailable")
            stat = path.stat()
            files.append({"path": path, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "info": info})
        except (OSError, ValueError, KeyError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    return sorted(files, key=lambda row: str(row["path"]).lower()), skipped

def run(apply=False, checkpoint=None):
    if os.name != "nt" or kernel is None:
        raise RuntimeError("This audited hardlink/accounting implementation requires Windows")
    files, skipped = collect()
    sizes, hash_cache, groups = {}, {}, {}
    for item in files:
        sizes.setdefault(item["bytes"], []).append(item)
    for size, subset in sizes.items():
        if len(subset) < 2:
            continue
        for item in subset:
            key = (item["info"]["file_id"], size, item["mtime_ns"])
            value = hash_cache.setdefault(key, None)
            if value is None:
                value = hash_cache[key] = digest(item["path"])
            groups.setdefault((size, value), []).append(item)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "apply": apply,
              "scope": "Only completed outputs/**/logits.f32; model/source/binary paths excluded",
              "valid_files": len(files), "skipped": skipped, "operations": [], "inode_accounted_freed_bytes": 0,
              "limitations": "Hardlink aliases retain identical bytes but share future edits; treat raw artifacts as immutable. Inode accounting measures final named links removed from the project; other open readers can delay physical reclamation. It excludes directory/MFT overhead and other-ID extent sharing. Whole-volume free-space deltas include unrelated activity."}
    volume_before = shutil.disk_usage(ROOT).free
    planned_removed_links = {}
    if checkpoint:
        checkpoint(report)
    for (_, sha256), subset in groups.items():
        if len(subset) < 2:
            continue
        source = subset[0]
        for item in subset[1:]:
            canonical, candidate = safe_path(source["path"]), safe_path(item["path"])
            before = allocation_info(candidate)
            operation = {"source": str(canonical.relative_to(ROOT)), "candidate": str(candidate.relative_to(ROOT)),
                         "logical_bytes": item["bytes"], "sha256": sha256,
                         "old_file_id": before.get("file_id"), "source_file_id": allocation_info(canonical).get("file_id"),
                         "old_link_count": before.get("link_count"), "old_allocated_bytes": before.get("allocated_bytes")}
            if operation["old_file_id"] == operation["source_file_id"]:
                operation["status"] = "already_shared"
            elif not apply:
                operation["status"] = "would_share"
                # Two duplicate paths can already share a noncanonical inode.
                # Charge its allocation only when the plan removes its last link.
                identity = before["file_id"]
                remaining = before["link_count"] - planned_removed_links.get(identity, 0)
                operation["projected_remaining_links_before"] = remaining
                operation["inode_accounted_freed_bytes"] = before["allocated_bytes"] if remaining == 1 else 0
                planned_removed_links[identity] = planned_removed_links.get(identity, 0) + 1
            else:
                temporary = candidate.with_name(f".logits-dedup-{uuid.uuid4().hex}.tmp")
                try:
                    with deny_writers(canonical), deny_writers(candidate):
                        old, current_source = allocation_info(candidate), allocation_info(canonical)
                        if old.get("file_id") != operation["old_file_id"] or current_source.get("file_id") != operation["source_file_id"]:
                            raise ValueError("File identity changed since planning")
                        if digest(canonical) != sha256 or digest(candidate) != sha256 or not equal_bytes(canonical, candidate):
                            raise ValueError("Bytes changed or digest group was not byte-identical")
                        source_stat, candidate_stat = canonical.stat(), candidate.stat()
                        os.link(canonical, safe_path(temporary, exists=False), follow_symlinks=False)
                    # MoveFileEx used by os.replace can refuse a hardlinked file
                    # while the validation handle denies writes, despite SHARE_DELETE.
                    # Raw artifacts are immutable and the benchmark lock is held.
                    # Close handles, then recheck identities/sizes/mtimes immediately.
                    for path, expected_info, expected_stat in ((canonical, current_source, source_stat), (candidate, old, candidate_stat)):
                        info, stat = allocation_info(path), path.stat()
                        if info.get("file_id") != expected_info.get("file_id") or (stat.st_size, stat.st_mtime_ns) != (expected_stat.st_size, expected_stat.st_mtime_ns):
                            raise ValueError("File changed before atomic replacement")
                    os.replace(temporary, candidate)
                    operation["replacement_applied"] = True
                    after = allocation_info(candidate)
                    if after.get("file_id") != current_source.get("file_id") or candidate.stat().st_size != item["bytes"]:
                        raise RuntimeError("Hardlink replacement failed identity/size verification")
                    operation.update(status="shared", new_file_id=after["file_id"], new_link_count=after["link_count"],
                                     inode_accounted_freed_bytes=old["allocated_bytes"] if old["link_count"] == 1 else 0,
                                     candidate_previous_mtime_ns=item["mtime_ns"], canonical_mtime_ns=canonical.stat().st_mtime_ns)
                except (OSError, ValueError, RuntimeError) as error:
                    operation.update(status="failed", error=str(error), inode_accounted_freed_bytes=0)
                finally:
                    if temporary.exists():
                        try:
                            safe_path(temporary).unlink()
                        except (OSError, ValueError) as error:
                            operation.update(temporary_cleanup_error=str(error), temporary_path=str(temporary))
            if operation.get("status") == "shared":
                report["inode_accounted_freed_bytes"] += operation["inode_accounted_freed_bytes"]
            report["operations"].append(operation)
            if checkpoint:
                checkpoint(report)
    report["observed_volume_free_delta_bytes"] = shutil.disk_usage(ROOT).free-volume_before
    report["projected_inode_allocation_bytes"] = sum(op.get("inode_accounted_freed_bytes", 0) for op in report["operations"] if op["status"] == "would_share")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Replace verified duplicates with atomic hardlink aliases")
    parser.add_argument("--output", type=Path, default=OUTPUTS/"logit-deduplication.json")
    args = parser.parse_args()
    output = safe_path(args.output, exists=False)
    if output.parent != OUTPUTS.resolve() or output.suffix.lower() != ".json" or not output.name.startswith("logit-dedup"):
        raise ValueError("Manifest must be outputs/logit-dedup*.json, never a raw run artifact")
    with SerialLock(ROOT/"work/benchmark.lock"):
        report = run(args.apply, lambda value: output.write_text(json.dumps(value, indent=2), encoding="utf-8"))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"apply": report["apply"], "valid_files": report["valid_files"],
                      "operations": len(report["operations"]), "inode_accounted_freed_bytes": report["inode_accounted_freed_bytes"],
                      "manifest": str(output)}, indent=2))

if __name__ == "__main__":
    main()
