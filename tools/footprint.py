"""Point-in-time logical, per-path allocated, and unique-file allocated storage."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
kernel = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
if kernel:
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetCompressedFileSizeW.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

class StandardInfo(ctypes.Structure):
    _fields_ = [("allocation", ctypes.c_longlong), ("end", ctypes.c_longlong),
                ("links", wintypes.DWORD), ("delete_pending", wintypes.BOOLEAN), ("directory", wintypes.BOOLEAN)]

class FileIdInfo(ctypes.Structure):
    _fields_ = [("volume_serial", ctypes.c_ulonglong), ("file_id", ctypes.c_ubyte * 16)]

class AttributeTagInfo(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]

def is_reparse(path):
    path = Path(path)
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())

def allocation_info(path):
    """File ID is volume-local; allocated bytes exclude directory/MFT overhead."""
    path = Path(path)
    if not kernel:
        stat = path.stat()
        return {"allocated_bytes": getattr(stat, "st_blocks", 0) * 512 if hasattr(stat, "st_blocks") else None,
                "standard_allocation_bytes": None, "file_id": f"{stat.st_dev}:{stat.st_ino}",
                "link_count": stat.st_nlink, "compressed": None, "sparse": None,
                "reparse_point": is_reparse(path), "method": "stat.st_blocks"}
    # OPEN_REPARSE_POINT avoids silently querying a link target.
    handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        return {"allocated_bytes": None, "file_id": None, "error": ctypes.get_last_error()}
    try:
        standard, identity, attributes = StandardInfo(), FileIdInfo(), AttributeTagInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 1, ctypes.byref(standard), ctypes.sizeof(standard)):
            return {"allocated_bytes": None, "file_id": None, "error": ctypes.get_last_error()}
        got_id = kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(identity), ctypes.sizeof(identity))
        got_attributes = kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes))
        compressed = bool(attributes.attributes & 0x800) if got_attributes else None
        sparse = bool(attributes.attributes & 0x200) if got_attributes else None
        reparse = bool(attributes.attributes & 0x400) if got_attributes else is_reparse(path)
        allocated, method = standard.allocation, "FileStandardInfo.AllocationSize"
        if compressed or sparse:
            # For normal files this API returns logical size, so use it only here.
            high = wintypes.DWORD()
            ctypes.set_last_error(0)
            low = kernel.GetCompressedFileSizeW(str(path), ctypes.byref(high))
            if low == 0xffffffff and ctypes.get_last_error():
                allocated = None
            else:
                allocated = (high.value << 32) | low
            method = "GetCompressedFileSizeW for compressed/sparse stream"
        return {"allocated_bytes": allocated, "standard_allocation_bytes": standard.allocation,
                "file_id": f"{identity.volume_serial:016x}:{bytes(identity.file_id).hex()}" if got_id else None,
                "link_count": standard.links, "compressed": compressed, "sparse": sparse,
                "reparse_point": reparse, "method": method}
    finally:
        kernel.CloseHandle(handle)

def allocation(path):
    """Compatible per-file allocation API; it does not deduplicate hardlinks."""
    return allocation_info(path).get("allocated_bytes")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/footprint.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Footprint report must remain inside the project")
    groups, errors, seen, aliases, flagged = {}, [], {}, [], []
    for parent, dirs, files in os.walk(ROOT, followlinks=False,
                                       onerror=lambda error: errors.append({"path": str(error.filename), "error": str(error)})):
        for name in sorted(dirs):
            if is_reparse(Path(parent, name)):
                errors.append({"path": str(Path(parent, name).relative_to(ROOT)), "error": "reparse directory skipped"})
        dirs[:] = sorted(d for d in dirs if not is_reparse(Path(parent, d)))
        for name in sorted(files):
            path = Path(parent, name)
            relative = path.relative_to(ROOT)
            if is_reparse(path) or not path.resolve().is_relative_to(ROOT.resolve()):
                errors.append({"path": str(relative), "error": "reparse/outside path skipped"})
                continue
            key = "/".join(relative.parts[:2]) if relative.parts[0] == "work" else relative.parts[0]
            group = groups.setdefault(key, {"logical_bytes": 0, "allocated_bytes": 0, "unique_allocated_bytes": 0,
                                           "files": 0, "allocation_unavailable_files": 0, "identity_unavailable_files": 0})
            try:
                info = allocation_info(path)
                if info.get("reparse_point"):
                    raise ValueError("reparse point skipped")
                group["logical_bytes"] += path.stat().st_size
                group["files"] += 1
                allocated, identity = info.get("allocated_bytes"), info.get("file_id")
                if identity is None:
                    group["identity_unavailable_files"] += 1
                if allocated is None:
                    group["allocation_unavailable_files"] += 1
                    continue
                group["allocated_bytes"] += allocated
                if identity is None:
                    group["unique_allocated_bytes"] += allocated
                elif identity not in seen:
                    seen[identity] = str(relative)
                    group["unique_allocated_bytes"] += allocated
                else:
                    aliases.append({"path": str(relative), "first_path": seen[identity], "file_id": identity,
                                    "allocated_bytes_already_counted": allocated})
                if info.get("compressed") or info.get("sparse"):
                    flagged.append({"path": str(relative), **info})
            except (OSError, ValueError) as error:
                errors.append({"path": str(relative), "error": str(error)})
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "groups": groups,
              "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
              "logical_total_bytes": sum(g["logical_bytes"] for g in groups.values()),
              "allocated_total_bytes": sum(g["allocated_bytes"] for g in groups.values()),
              "unique_allocated_total_bytes": sum(g["unique_allocated_bytes"] for g in groups.values()),
              "hardlink_aliases": aliases, "compressed_or_sparse_files": flagged, "errors": errors,
              "allocation_complete": not errors and all(g["allocation_unavailable_files"] == 0 for g in groups.values()),
              "identity_complete": not errors and all(g["identity_unavailable_files"] == 0 for g in groups.values()),
              "accounting": "allocated_total_bytes preserves legacy per-path counting. unique_allocated_total_bytes charges each volume/file ID once; group charge goes to its first deterministic sorted path. Unknown identities conservatively count per path.",
              "limitations": "Point-in-time snapshot, not installation peak. Totals include only successfully queried default file streams; check completeness flags. Excludes alternate data streams, directory/MFT overhead and physical extent sharing between different file IDs. FileStandardInfo handles normal streams; compressed/sparse streams use GetCompressedFileSizeW. Hardlinks outside this project can retain data after project deletion. The report file changes after this snapshot."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
