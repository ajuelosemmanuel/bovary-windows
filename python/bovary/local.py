"""Bounded single-child local worker; importing this module touches no hardware."""
from __future__ import annotations

import atexit
import codecs
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import threading
import time

from .contract import Event, ExecutionError, Result
from .modelproof import check_binding


PROFILES = {
    "cpu-only": (0, 0, 1, 1, "serial", 32),
    "gpu-fast": (8, 0, 8, 4, "pinned2", 128),
    "balanced": (99, 1024, 8, 4, "pinned2", 1),
    "gpu-more-vram": (99, 1536, 8, 4, "pinned2", 1),
    "gpu-cache-2g": (99, 2048, 8, 4, "pinned2", 1),
    "gpu-prefill-2g": (99, 2048, 8, 4, "pinned2", 128),
    "minimum-ram": (99, 512, 1, 1, "serial", 1),
    "minimum-vram": (99, 32, 8, 64, "pinned2", 1),
}
MAX_PROMPT_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 32 * 1024 * 1024
FRAME_LIMITS = {b"R": 4096, b"T": 1024, b"E": 16384, b"D": MAX_FRAME_BYTES}
_OWNED_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW | 0x00000004 if os.name == "nt" else 0  # CREATE_SUSPENDED

# Retain only adapters that own a worker. Registering every instance's bound
# method with atexit keeps even never-used/unloaded adapters alive forever.
_ACTIVE_ADAPTERS = set()
_ACTIVE_ADAPTERS_LOCK = threading.Lock()


def _dispose_active_adapters():
    with _ACTIVE_ADAPTERS_LOCK:
        adapters = tuple(_ACTIVE_ADAPTERS)
    for adapter in adapters:
        try:
            adapter._dispose()
        except Exception:
            # One failed cleanup must not prevent cleanup of other workers.
            # Windows Job Objects still enforce process-tree ownership.
            continue


atexit.register(_dispose_active_adapters)


def _object(payload):
    try:
        def reject_constant(value):
            raise ValueError("Nonfinite JSON constant")
        value = json.loads(payload, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value
    except (ValueError, UnicodeError) as error:
        raise ExecutionError("protocol", "Worker sent malformed JSON metadata") from error


def windows_performance_core_mask():
    """Return the fastest CPU set class on a single-group hybrid Windows host."""
    if os.name != "nt":
        raise ValueError("--performance-cores requires Windows")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                      wintypes.HANDLE, wintypes.ULONG]
    query.restype = wintypes.BOOL
    needed = wintypes.ULONG()
    query(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    data = ctypes.create_string_buffer(needed.value)
    if not query(data, len(data), ctypes.byref(needed), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())

    masks = {}
    offset = 0
    while offset < needed.value:
        if needed.value - offset < 8:
            raise ValueError("Windows reported incomplete CPU set information")
        size, kind = struct.unpack_from("<II", data, offset)
        if size < 8 or size > needed.value - offset:
            raise ValueError("Windows reported an invalid CPU set record")
        if kind == 0:  # CpuSetInformation
            if size < 19:
                raise ValueError("Windows reported an incomplete CPU set record")
            group, logical, _, _, _, efficiency = struct.unpack_from("<H5B", data, offset + 12)
            if group != 0 or logical >= ctypes.sizeof(ctypes.c_size_t) * 8:
                raise ValueError("--performance-cores requires a single Windows processor group")
            masks[efficiency] = masks.get(efficiency, 0) | (1 << logical)
        offset += size
    if len(masks) < 2:
        raise ValueError("Windows did not report distinct performance and efficiency CPU classes")
    return masks[max(masks)]


class _WindowsJob:
    """Close-on-parent-exit ownership for this worker process only."""
    def __init__(self, process):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle)))):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(handle)
            raise error
        self.handle = handle

    def set_worker_affinity(self, process, mask):
        """Set and verify the owned worker's mask before the first thread resumes."""
        import ctypes
        from ctypes import wintypes

        mask_type = ctypes.c_size_t
        query = self.kernel.GetProcessAffinityMask
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(mask_type), ctypes.POINTER(mask_type)]
        query.restype = wintypes.BOOL
        apply = self.kernel.SetProcessAffinityMask
        apply.argtypes = [wintypes.HANDLE, mask_type]
        apply.restype = wintypes.BOOL
        handle = int(process._handle)
        active, system = mask_type(), mask_type()
        try:
            if not query(handle, ctypes.byref(active), ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            mask &= active.value
            if not mask:
                raise ExecutionError("worker_affinity", "The worker's inherited CPU mask excludes all performance cores")
            if mask == active.value:
                return
            if not apply(handle, mask) or not query(handle, ctypes.byref(active), ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            if active.value != mask:
                raise ExecutionError("worker_affinity", "Windows did not apply the requested performance-core mask")
        except OSError as error:
            raise ExecutionError("worker_affinity", f"Could not set performance-core placement: {error}") from error

    def resume(self, process):
        """Resume the initial thread only after its suspended process is owned."""
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                        ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

        kernel = self.kernel
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        kernel.Thread32First.restype = wintypes.BOOL
        kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        kernel.Thread32Next.restype = wintypes.BOOL
        kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenThread.restype = wintypes.HANDLE
        kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel.ResumeThread.restype = wintypes.DWORD
        snapshot = kernel.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            present = kernel.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32OwnerProcessID == process.pid:
                    thread = kernel.OpenThread(0x0002, False, entry.th32ThreadID)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        previous = kernel.ResumeThread(thread)
                        if previous == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                        if previous != 1:
                            raise OSError("Unexpected initial worker thread suspend count")
                        return
                    finally:
                        kernel.CloseHandle(thread)
                entry.dwSize = ctypes.sizeof(entry)
                present = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("Suspended worker initial thread was not found")
        finally:
            kernel.CloseHandle(snapshot)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _stop_owned_process(process, job):
    """Kill the owned tree before any caller closes pipes used by its readers."""
    if job is not None and os.name == "nt":
        job.close()
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class SerialLock:
    """Shared inference and benchmark lock, held until model unload."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            # Lock the same byte as the benchmark even when the file is empty.
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise ExecutionError("busy", "Another inference worker or benchmark holds the shared lock") from None
        self.file = stream

    def release(self):
        stream, self.file = self.file, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        finally:
            stream.close()


class LocalAdapter:
    """One persistent model, serial independent requests and explicit unload.

    Cancel/deadline disposes the child and waits for it. The next request loads
    a fresh child. Configuration changes also dispose the previous worker.
    No prompt or response files are written by this adapter.
    Relative model, manifest and executable paths use the configured home.
    """
    def __init__(self, root=None, *, model=None, manifest=None, executable=None,
                 command_prefix=None, startup_timeout_s=180, worker_affinity=None):
        if root is None:
            from .runtime import project_home
            root = project_home()
        self.root = Path(root).expanduser().resolve()
        model = model or os.environ.get("BOVARY_MODEL")
        manifest = manifest or os.environ.get("BOVARY_MODEL_MANIFEST")
        # Explicit relative paths are interpreted under the configured home.
        self.model_path = self._home_path(model) if model else None
        self.manifest_path = self._home_path(manifest) if manifest else None
        self.exe = self._home_path(executable) if executable else None
        self.command_prefix = list(command_prefix) if command_prefix else None
        self.startup_timeout_s = startup_timeout_s
        self._worker_affinity = worker_affinity
        self.process = None
        self._job = None
        self._key = None
        self._lock = SerialLock(self.root / "work/benchmark.lock")
        self._operation = threading.Lock()
        self._threads = []
        self._frames = None
        self._diagnostics = deque(maxlen=32)
        self._ready = {}
        self._model = {}
        self._attempt_started = False

    def _home_path(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None

    def _checkpoint(self, *, interrupt=None):
        path = self.manifest_path or self.root / "work/models/model-provenance.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError("Verification manifest must be an object")
            filename = manifest.get("filename")
            if not isinstance(filename, str) or not filename or filename != Path(filename).name:
                raise ValueError("Verification manifest must name a checkpoint file")
            model = (self.model_path or self.root / "work/models" / filename).resolve()
            if model.name != filename:
                raise ValueError("Verification manifest names another checkpoint file")
            if (manifest.get("verified") is not True or not model.is_file()
                    or type(manifest["bytes"]) is not int or manifest["bytes"] < 1
                    or model.stat().st_size != manifest["bytes"]
                    or not isinstance(manifest.get("sha256"), str)
                    or len(manifest["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in manifest["sha256"].lower())):
                raise ValueError("Checkpoint does not match a completed verification manifest")
            check_binding(model, manifest, interrupt=interrupt)
            return model, manifest
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise ExecutionError("model_unavailable", f"Verify an existing checkpoint and configure its manifest: {error}") from error

    def _configuration(self, request, *, cancel=None):
        if request.local_profile not in PROFILES:
            raise ExecutionError("invalid_profile", "Unsupported persistent profile; mixed weights require an explicit experiment")
        cpu = request.local_profile == "cpu-only"
        model, manifest = self._checkpoint(interrupt=lambda: self._check_interrupt(request, cancel))
        override = os.environ.get("BOVARY_CPU_EXE" if cpu else "BOVARY_CUDA_EXE")
        executable = self.exe or (self._home_path(override) if override else self.root / "work/build" / ("bovary-cpu" if cpu else "bovary-cuda") / "bovary-infer.exe")
        if self.command_prefix is None and not executable.is_file():
            raise ExecutionError("backend_unavailable", "Build first: scripts\\build.cmd " + ("cpu" if cpu else "cuda") + " 4")
        layers, cache, buffer, load_buffer, pool, batch = PROFILES[request.local_profile]
        if cpu and request.cpu_prompt_batch is not None:
            batch = request.cpu_prompt_batch
        streaming = not cpu and request.local_profile != "gpu-fast"
        prefix = list(self.command_prefix or [str(executable)])
        fingerprints = []
        for index, value in enumerate(prefix):
            path = self._home_path(value)
            if index == 0 and not path.is_file():
                located = shutil.which(value)
                if located:
                    path = Path(located).resolve()
            if path.is_file():
                prefix[index] = str(path)
                stat = path.stat()
                fingerprints.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        command = prefix + [
            "--worker", "--model", str(model), "--context", str(request.context_tokens),
            "--gpu-layers", str(layers), "--load-mode", "direct" if streaming else "mmap",
            "--cache-mib", str(cache), "--threads", str(request.threads), "--prompt-batch", str(batch)]
        if cpu:
            command.append("--require-cpu-only")
        if streaming:
            command += ["--cpu-experts", "--gpu-embedding"]
        if request.host_cache_mib:
            command += ["--host-cache-mib", str(request.host_cache_mib)]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("BOVARY_STREAM")}
        environment.pop("GGML_BACKEND_PATH", None)
        environment["BOVARY_STREAM"] = "1" if streaming else "0"
        if streaming:
            environment.update(BOVARY_STREAM_MODEL=str(model), BOVARY_STREAM_IO="unbuffered",
                               BOVARY_STREAM_BUFFER_MIB=str(buffer), BOVARY_STREAM_LOAD_BUFFER_MIB=str(load_buffer),
                               BOVARY_STREAM_POOL=pool)
        if not cpu:
            cuda = self.root / "work/deps/cuda-13.3"
            environment["PATH"] = str(cuda / "bin/x64") + os.pathsep + str(cuda / "bin") + os.pathsep + environment.get("PATH", "")
        # The verified digest is part of the identity, alongside runtime choices.
        identity = manifest.get("file_identity")
        model_identity = tuple(sorted(identity.items())) if identity is not None else model.stat().st_mtime_ns
        key = (tuple(command), tuple(fingerprints), manifest["sha256"], model_identity)
        return command, environment, key, manifest

    @staticmethod
    def _check_interrupt(request, cancel):
        if cancel is not None and cancel.is_set():
            raise ExecutionError("cancelled", "Local inference cancelled; worker unloaded")
        if request.deadline is not None and time.monotonic() >= request.deadline:
            raise ExecutionError("deadline", "Local inference deadline expired; worker unloaded")

    def _next_frame(self, request, cancel, deadline=None):
        while True:
            self._check_interrupt(request, cancel)
            if deadline is not None and time.monotonic() >= deadline:
                raise ExecutionError("startup_timeout", "Worker did not become ready before its startup deadline")
            try:
                kind, value = self._frames.get(timeout=.05)
            except queue.Empty:
                continue
            if kind == "!":
                raise ExecutionError("worker_crashed", f"Worker stopped or sent invalid protocol: {value}. " + "".join(self._diagnostics)[-4000:])
            if kind == "P":
                raise ExecutionError("protocol", f"Worker sent invalid protocol: {value}")
            return kind, value

    def _preflight(self, command, environment, request, cancel):
        """Read a parser-only version marker before starting any backend."""
        prefix = command[:command.index("--worker")]
        version_flag = "--host-cache-protocol-version" if request.host_cache_mib else "--protocol-version"
        probe = prefix + (["--require-cpu-only"] if request.local_profile == "cpu-only" else []) + [version_flag]
        hint = "Rebuild: scripts\\build.cmd " + ("cpu" if request.local_profile == "cpu-only" else "cuda") + " 4"
        process, job, reader = None, None, None
        marker = bytearray()
        capture_done = threading.Event()
        capture_error = []
        markers = (b"bovary-worker 2\n", b"bovary-worker 2\r\n")
        marker_limit = max(map(len, markers))
        try:
            self._check_interrupt(request, cancel)
            process = subprocess.Popen(probe, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, env=environment, cwd=self.root,
                                       creationflags=_OWNED_CREATION_FLAGS, start_new_session=os.name != "nt")
            job = _WindowsJob(process)
            self._check_interrupt(request, cancel)
            job.resume(process)

            def capture_marker():
                try:
                    while len(marker) <= marker_limit:
                        chunk = process.stdout.read1(marker_limit + 1 - len(marker))
                        if not chunk:
                            break
                        marker.extend(chunk)
                except (OSError, ValueError) as error:
                    capture_error.append(error)
                finally:
                    # This thread owns the buffered reader. The main thread
                    # kills its writers first and never blocks on its read lock.
                    process.stdout.close()
                    capture_done.set()

            reader = threading.Thread(target=capture_marker, daemon=True)
            reader.start()
            deadline = time.monotonic() + min(5, self.startup_timeout_s)
            while True:
                self._check_interrupt(request, cancel)
                if time.monotonic() >= deadline:
                    raise ExecutionError("backend_version", "Native protocol version check timed out. " + hint)
                if capture_done.is_set():
                    if capture_error or bytes(marker) not in markers:
                        raise ExecutionError("backend_version", "Native executable is stale or incompatible with this worker/CPU mode. " + hint)
                    if process.poll() is not None:
                        if process.returncode != 0:
                            raise ExecutionError("backend_version", "Native executable is stale or incompatible with this worker/CPU mode. " + hint)
                        break
                    try:
                        process.wait(timeout=.05)
                    except subprocess.TimeoutExpired:
                        pass
                else:
                    capture_done.wait(timeout=.05)
            self._check_interrupt(request, cancel)
        except OSError as error:
            raise ExecutionError("backend_version", "Cannot verify native executable protocol. " + hint) from error
        finally:
            if process is not None:
                _stop_owned_process(process, job)
                if reader is None or reader.ident is None:
                    process.stdout.close()
                else:
                    reader.join(timeout=1)

    def _start(self, command, environment, key, manifest, request, cancel):
        self._check_interrupt(request, cancel)
        self._lock.acquire()
        self._diagnostics.clear()
        self._frames = queue.Queue(maxsize=4)
        frames = self._frames
        try:
            self._check_interrupt(request, cancel)
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=environment, cwd=self.root,
                                       creationflags=_OWNED_CREATION_FLAGS, start_new_session=os.name != "nt")
            self.process = process
            with _ACTIVE_ADAPTERS_LOCK:
                _ACTIVE_ADAPTERS.add(self)
            self._attempt_started = True
            self._job = _WindowsJob(process)
            self._check_interrupt(request, cancel)
            if self._worker_affinity is not None:
                self._job.set_worker_affinity(process, self._worker_affinity)
            self._job.resume(process)

            def emit(item):
                while self.process is process:
                    try:
                        frames.put(item, timeout=.05)
                        return
                    except queue.Full:
                        pass

            def read_frames():
                try:
                    while True:
                        line = process.stdout.readline(100)
                        if not line:
                            emit(("!", "worker stdout closed"))
                            return
                        fields = line.rstrip(b"\n").split(b" ")
                        if len(fields) != 2 or fields[0] not in (b"R", b"T", b"D", b"E") or not fields[1].isdigit():
                            raise ValueError("invalid frame header or EOF")
                        size = int(fields[1])
                        if size > FRAME_LIMITS[fields[0]]:
                            raise ValueError("frame exceeds its kind-specific size limit")
                        chunks, remaining = [], size
                        while remaining:
                            chunk = process.stdout.read(min(remaining, 65536))
                            if not chunk:
                                raise ValueError("truncated frame")
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        emit((fields[0].decode("ascii"), b"".join(chunks)))
                except ValueError as error:
                    emit(("P", str(error)))
                except OSError as error:
                    emit(("!", str(error)))

            def read_diagnostics():
                try:
                    while chunk := process.stderr.read1(4096):
                        self._diagnostics.append(chunk.decode("utf-8", errors="replace"))
                except (OSError, ValueError):
                    pass

            self._threads = [threading.Thread(target=read_frames, daemon=True),
                             threading.Thread(target=read_diagnostics, daemon=True)]
            for thread in self._threads:
                thread.start()
            kind, payload = self._next_frame(request, cancel, time.monotonic() + self.startup_timeout_s)
            if kind != "R":
                raise ExecutionError("protocol", "Expected ready frame; rebuild the worker executable")
            self._ready = _object(payload)
            if type(self._ready.get("protocol")) is not int or self._ready["protocol"] != 2:
                raise ExecutionError("protocol", "Unsupported worker protocol")
            self._key, self._model = key, manifest
        except OSError as error:
            self._dispose()
            raise ExecutionError("backend_unavailable", f"Could not start and own the worker: {error}") from error
        except BaseException:
            self._dispose()
            raise

    def _dispose(self):
        process, self.process = self.process, None
        job, self._job = self._job, None
        try:
            if process is not None:
                _stop_owned_process(process, job)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except (OSError, ValueError):
                            # A failed buffered stdin flush may fail again on
                            # close. Preserve the request error and finish
                            # closing the other pipes and joining readers.
                            pass
                for thread in self._threads:
                    thread.join(timeout=1)
        finally:
            if job is not None:
                job.close()
            self._threads = []
            self._key = None
            self._ready = {}
            try:
                self._lock.release()
            finally:
                with _ACTIVE_ADAPTERS_LOCK:
                    _ACTIVE_ADAPTERS.discard(self)

    def unload(self):
        """Unload an idle worker. Active requests use their cancellation event."""
        if not self._operation.acquire(blocking=False):
            raise ExecutionError("busy", "Cancel the active request before unloading")
        try:
            self._dispose()
        finally:
            self._operation.release()

    def close(self):
        """Release an idle worker, like unload(); this adapter remains reusable.

        Cancel and finish an active request before closing. Context-manager exit
        uses the same busy check and never interrupts a request implicitly.
        """
        self.unload()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def execute(self, request, on_event=None, cancel=None):
        request.check()
        if "local" not in request.allowed_destinations:
            raise ExecutionError("destination_forbidden", "Local destination is not allowed")
        self._check_interrupt(request, cancel)
        try:
            prompt = request.prompt.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ExecutionError("invalid_request", "Prompt must be valid UTF-8 text") from error
        if len(prompt) > MAX_PROMPT_BYTES:
            raise ExecutionError("context_overflow", "Prompt exceeds the worker's 1 MiB byte bound")
        if not self._operation.acquire(blocking=False):
            raise ExecutionError("busy", "A local request is already active")
        start, first_text, parts = time.monotonic(), None, []
        self._attempt_started = False
        emitted_tokens, emitted_bytes = 0, 0
        requested_count = request.output_tokens if request.output_tokens is not None else 0
        output_bound = request.output_tokens if request.output_tokens is not None else request.context_tokens - 1
        manifest, warm, startup, configuration_time = {}, False, 0.0, None
        sender = None
        context_rejected = False
        try:
            command, environment, key, manifest = self._configuration(request, cancel=cancel)
            configuration_time = time.monotonic() - start
            warm = self.loaded and self._key == key
            startup = 0.0
            if not warm:
                self._dispose()
                self._preflight(command, environment, request, cancel)
                self._check_interrupt(request, cancel)
                self._start(command, environment, key, manifest, request, cancel)
                if "file_identity" in manifest:
                    _, current = self._checkpoint(interrupt=lambda: self._check_interrupt(request, cancel))
                    if (current.get("file_identity") != manifest["file_identity"]
                            or current["sha256"] != manifest["sha256"]):
                        raise ExecutionError("model_unavailable", "Checkpoint verification changed during worker startup")
                startup = time.monotonic() - start
            process = self.process
            self._attempt_started = True

            def send():
                try:
                    process.stdin.write(f"RUN {requested_count} {len(prompt)}\n".encode("ascii"))
                    process.stdin.write(prompt)
                    process.stdin.flush()
                except (OSError, ValueError) as error:
                    try:
                        self._frames.put_nowait(("!", str(error)))
                    except queue.Full:
                        pass

            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                kind, payload = self._next_frame(request, cancel)
                if kind == "T":
                    emitted_tokens += 1
                    emitted_bytes += len(payload)
                    if (len(payload) > 1024 or emitted_tokens > output_bound
                            or emitted_bytes > output_bound * 1024):
                        raise ExecutionError("protocol", "Worker text exceeds the requested output bound")
                    text = decoder.decode(payload)
                    if first_text is None:
                        first_text = time.monotonic() - start
                    parts.append(text)
                    if on_event is not None:
                        on_event(Event("text", request.request_id, text=text))
                elif kind == "E":
                    error = _object(payload)
                    if (not isinstance(error.get("code"), str) or not error["code"]
                            or not isinstance(error.get("message"), str)):
                        raise ExecutionError("protocol", "Worker sent malformed error metadata")
                    # Protocol 2 returns to its command loop after a context-size
                    # rejection, before allocating a request context or emitting text.
                    context_rejected = error["code"] == "context_overflow" and emitted_tokens == 0
                    raise ExecutionError(error["code"], error["message"])
                elif kind == "D":
                    metrics = _object(payload)
                    if (type(metrics.get("prompt_tokens")) is not int or metrics["prompt_tokens"] < 1
                            or type(metrics.get("actual_generated_tokens")) is not int
                            or not 0 <= metrics["actual_generated_tokens"] == emitted_tokens <= output_bound
                            or type(metrics.get("requested_generated_tokens")) is not int
                            or metrics["requested_generated_tokens"] != requested_count
                            or type(metrics.get("generation_limit")) is not int
                            or not 1 <= metrics["generation_limit"] <= output_bound
                            or metrics["prompt_tokens"] + metrics["generation_limit"] > request.context_tokens
                            or metrics["generation_limit"] != (requested_count or request.context_tokens - metrics["prompt_tokens"])
                            or type(metrics.get("early_stop")) is not bool):
                        raise ExecutionError("protocol", "Worker sent invalid usage metadata")
                    generation_limit = metrics["generation_limit"]
                    if metrics["early_stop"]:
                        valid_finish = metrics.get("finish_reason") == "eos" and emitted_tokens < generation_limit
                    else:
                        valid_finish = (metrics.get("finish_reason") == ("context_limit" if requested_count == 0 else "token_limit")
                                        and emitted_tokens == generation_limit)
                    if not valid_finish:
                        raise ExecutionError("protocol", "Worker sent inconsistent generation termination metadata")
                    parts.append(decoder.decode(b"", final=True))
                    timing = {"end_to_end_s": time.monotonic() - start, "ttft_s": first_text,
                              "startup_s": startup, "configuration_s": configuration_time,
                              "warm": warm, "native": metrics}
                    return Result(request.request_id, "local", text="".join(parts),
                                  status="complete" if metrics["finish_reason"] == "eos" else "truncated",
                                  model=manifest.get("filename", self.model_path.name if self.model_path else ""),
                                  configuration=request.local_configuration,
                                  reasons=["Explicit local destination", "Loaded model reused" if warm else "Cold worker startup", "Fresh context per independent request"],
                                  usage={"input_tokens": metrics["prompt_tokens"], "output_tokens": metrics["actual_generated_tokens"],
                                         "finish_reason": metrics["finish_reason"], "output_limit": generation_limit,
                                         "model_sha256": manifest["sha256"]}, timing=timing,
                                  cost_status="unmeasured_local_compute")
                else:
                    raise ExecutionError("protocol", "Unexpected worker response")
        except BaseException as error:
            if not context_rejected:
                self._dispose()
            if isinstance(error, Exception):
                if not isinstance(error, ExecutionError):
                    error = ExecutionError("local_execution", "Local execution failed unexpectedly")
                if self._attempt_started:
                    error.attempt = Result(request.request_id, "local", text="".join(parts),
                        status="cancelled" if error.code == "cancelled" else "failed",
                        model=manifest.get("filename", ""),
                        configuration=request.local_configuration,
                        reasons=["Actual native worker startup/request failed"],
                        usage={"input_tokens": None, "output_tokens": None, "emitted_tokens": min(emitted_tokens, output_bound),
                               "output_limit": request.output_tokens,
                               "model_sha256": manifest.get("sha256")},
                        timing={"end_to_end_s": time.monotonic()-start, "ttft_s": first_text,
                                 "startup_s": startup if warm or startup > 0 else None,
                                 "configuration_s": configuration_time, "warm": warm},
                        cost_status="unmeasured_local_compute",
                        error={"code": error.code, "message": str(error), "uncertain_billing": error.uncertain_billing}).to_dict()
                raise error
            raise
        finally:
            if sender is not None:
                sender.join(timeout=1)
            self._operation.release()
