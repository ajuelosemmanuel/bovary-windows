"""Bind a verified checkpoint digest to the filesystem object that was read."""
import hashlib
import os
from pathlib import Path
import struct


GEMMA_REPOSITORY = "ggml-org/gemma-4-26B-A4B-it-GGUF"
GEMMA_REVISION = "1ca99aae748b4b01f609fa51eb23682ee5f00b85"
GEMMA_FILENAME = "gemma-4-26B-A4B-it-Q4_0.gguf"
GEMMA_BYTES = 14618145824
GEMMA_SHA256 = "d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03"
REVERIFY = "Verify the current checkpoint with tools/verify_model.py and select the new manifest"


def _windows_change_time(fd):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in ("created", "accessed", "written", "changed")]
        _fields_.append(("attributes", wintypes.DWORD))

    query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    info = FileBasicInfo()
    if not query(msvcrt.get_osfhandle(fd), 0, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.changed


def _windows_file_usn(fd):
    """Read this file's latest journal record when the filesystem supplies it."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    ioctl = ctypes.WinDLL("kernel32", use_last_error=True).DeviceIoControl
    ioctl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                      ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    ioctl.restype = wintypes.BOOL
    versions = (ctypes.c_ushort * 2)(2, 3)
    output = ctypes.create_string_buffer(1024)
    count = wintypes.DWORD()
    # FSCTL_READ_FILE_USN_DATA using the file handle, never a privileged volume handle.
    if not ioctl(msvcrt.get_osfhandle(fd), 0x000900EB, versions, ctypes.sizeof(versions),
                 output, ctypes.sizeof(output), ctypes.byref(count), None) or count.value < 8:
        return None
    length, version, _ = struct.unpack_from("<IHH", output)
    offset = {2: 24, 3: 40}.get(version)
    if offset is None or not offset + 8 <= length <= count.value:
        return None
    usn, = struct.unpack_from("<q", output, offset)
    return usn if usn > 0 else None


def file_identity(stream):
    fd = stream.fileno()
    info = os.fstat(fd)
    changed = _windows_change_time(fd) if os.name == "nt" else info.st_ctime_ns
    if not info.st_ino or not changed:
        raise ValueError("Filesystem does not expose a stable checkpoint identity")
    identity = {"device": info.st_dev, "file": info.st_ino, "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns, "change_time": changed}
    if os.name == "nt":
        usn = _windows_file_usn(fd)
        if usn is not None:
            identity["usn"] = usn
    return identity


def path_identity(path):
    with Path(path).open("rb") as stream:
        return file_identity(stream)


def requires_digest_check(manifest, observed):
    return (os.name == "nt" and observed is not None
            and ("usn" not in manifest["file_identity"] or "usn" not in observed))


def check_binding(model, manifest, *, allow_full_hash=True, interrupt=None):
    """Reject stale bound proofs and unbound proofs for the pinned checkpoint."""
    expected = manifest.get("file_identity")
    if "file_identity" not in manifest:
        if (manifest.get("repository") == GEMMA_REPOSITORY
                or str(manifest.get("sha256", "")).lower() == GEMMA_SHA256
                or str(manifest.get("filename", "")).casefold() == GEMMA_FILENAME.casefold()
                or Path(model).name.casefold() == GEMMA_FILENAME.casefold()):
            raise ValueError(f"Legacy Gemma verification proof has no file binding. {REVERIFY}.")
        return None
    mismatch = f"Checkpoint file no longer matches its verification proof. {REVERIFY}."
    with Path(model).open("rb") as source:
        observed = file_identity(source)
        baseline = observed.keys() - {"usn"}
        if (type(expected) is not dict or expected.keys() - {"usn"} != baseline
                or any(type(value) is not int for value in expected.values())
                or any(expected[key] != observed[key] for key in baseline)
                or ("usn" in expected and "usn" in observed and expected["usn"] != observed["usn"])):
            raise ValueError(mismatch)
        # Windows change timestamps may be coalesced or restored. Without a
        # journal generation on both sides, verify bytes instead of trusting them.
        if allow_full_hash and requires_digest_check(manifest, observed):
            digest = hashlib.sha256()
            while True:
                if interrupt is not None:
                    interrupt()
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if (digest.hexdigest() != manifest["sha256"].lower()
                    or file_identity(source) != observed or path_identity(model) != observed):
                raise ValueError(mismatch)
    return observed
