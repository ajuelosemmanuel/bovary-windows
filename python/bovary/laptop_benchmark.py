"""Run a short local Gemma check with fixed questions and shareable measurements."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import platform
from statistics import median
from threading import Event, Thread
import time

from . import __version__
from .checkpoint import PinnedGemma
from .contract import ExecutionError, Request
from .mini import CONTEXT_TOKENS, PRESETS
from .modelproof import GEMMA_FILENAME, GEMMA_SHA256


CASES = (
    ("addition", "What is 17 plus 28? Answer with just the number.", "45"),
    ("planning-note", "Return only the replacement meeting day from this fictional note. "
     "The bridge review planned for Tuesday was canceled. The team moved it to Thursday. "
     "The Friday slot is reserved for the safety briefing. The location and guest list "
     "remain the same. Which day is the replacement bridge review?", "Thursday"),
    ("document-lookup", "Use only this fictional dispatch register. Return only the requested locker code. "
     "The south warehouse is closing for repairs and its stock will move by route Ash. "
     "Route Alder carries blue labels to the north warehouse and has locker code 6214. "
     "Its dispatch lead is Mira, and it leaves one hour before route Cedar. "
     "Route Cedar uses locker code 8537. It carries orange labels and serves the east depot. "
     "Route Ash was initially assigned code 1936. The warehouse supervisor canceled "
     "that assignment after the third inventory check and issued locker code 4728 instead. "
     "The Ash crew should still report to bay six at noon and sign the register. "
     "What is the current locker code for route Ash?", "4728"),
)
OUTPUT_TOKENS = 16
THREADS = 2
INTERVAL_S = 0.2
MIB = 1024 ** 2


class ProcessMemory(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
            "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_usage",
        )
    ]


class ProcessIo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "read_operations", "write_operations", "other_operations",
        "read_bytes", "write_bytes", "other_bytes",
    )]


class WindowsProcessCounters:
    def __init__(self):
        self.memory = ctypes.WinDLL("psapi.dll", winmode=0x800).GetProcessMemoryInfo
        self.memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        self.io = ctypes.WinDLL("kernel32.dll", winmode=0x800).GetProcessIoCounters
        self.io.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessIo)]

    def sample(self, process):
        memory, io = ProcessMemory(), ProcessIo()
        memory.size = ctypes.sizeof(memory)
        handle = int(process._handle)
        if self.memory(handle, ctypes.byref(memory), memory.size):
            yield "worker_peak_working_set_mib", memory.peak_working_set / MIB
            yield "worker_sampled_peak_private_commit_mib", memory.private_usage / MIB
        if self.io(handle, ctypes.byref(io)):
            yield "worker_sampled_read_transfer_mib", io.read_bytes / MIB


class GpuMemory(ctypes.Structure):
    _fields_ = [(field, ctypes.c_ulonglong) for field in ("total", "free", "used")]


class NvidiaMemory:
    """NVML device-wide FB memory for NVIDIA GPU index zero."""

    def __init__(self):
        self.library = None
        self.device = ctypes.c_void_p()
        self.info = None

    def __enter__(self):
        if os.name == "nt":
            try:
                library = ctypes.WinDLL("nvml.dll", winmode=0x800)
                if library.nvmlInit_v2() == 0:
                    self.library = library
                    library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.device))
                    if self.device:
                        name = ctypes.create_string_buffer(96)
                        status = library.nvmlDeviceGetName(self.device, name, ctypes.sizeof(name))
                        memory = self.memory_info()
                        if status == 0 and memory is not None:
                            self.info = {"name": name.value.decode("ascii", errors="replace"),
                                         "total_mib": memory.total // MIB}
            except (OSError, AttributeError):
                pass
        return self

    def memory_info(self):
        if self.device:
            used = GpuMemory()
            if self.library.nvmlDeviceGetMemoryInfo(self.device, ctypes.byref(used)) == 0:
                return used
        return None

    def sample(self):
        memory = self.memory_info()
        return memory.used / MIB if memory is not None else None

    def __exit__(self, *_):
        if self.library is not None:
            self.library.nvmlShutdown()


def maximum(previous, value):
    return previous if value is None else value if previous is None else max(previous, value)


class MemorySampler:
    def __init__(self, worker_process):
        self.worker_process = worker_process
        self.stop = Event()
        self.metrics = dict(sample_interval_s=INTERVAL_S, worker_peak_working_set_mib=None,
                            worker_sampled_peak_private_commit_mib=None, worker_sampled_read_transfer_mib=None,
                            gpu0_baseline_used_mib=None,
                            gpu0_sampled_peak_used_mib=None)
        self.gpu = NvidiaMemory()
        self.counters = WindowsProcessCounters() if os.name == "nt" else None
        self.hardware = {}
        self.thread = Thread(target=self.measure, daemon=True)

    def measure(self):
        while not self.stop.is_set():
            process = self.worker_process()
            if process is not None and self.counters is not None:
                for key, value in self.counters.sample(process):
                    self.metrics[key] = maximum(self.metrics[key], value)
            key = "gpu0_sampled_peak_used_mib"
            self.metrics[key] = maximum(self.metrics[key], self.gpu.sample())
            self.stop.wait(INTERVAL_S)

    def __enter__(self):
        self.gpu.__enter__()
        self.metrics["gpu0_baseline_used_mib"] = self.gpu.sample()
        self.hardware = {"cpu": windows_cpu_name(), "nvidia_gpu0": self.gpu.info}
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.gpu.__exit__()


def windows_cpu_name():
    if os.name != "nt":
        return platform.processor() or None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        return None


def row(case, timing, answer, *, state, input_tokens=None, output_tokens=None, failure=None):
    case_id, _, expected = case
    ttft = timing.get("ttft_s")
    elapsed = timing.get("end_to_end_s")
    return dict(case=case_id, state=state, correct=(failure is None and
                answer.strip().rstrip(".").casefold() == expected.casefold()),
                status=failure or "complete", ttft_s=ttft, elapsed_s=elapsed,
                input_tokens=input_tokens, output_tokens=output_tokens)


def summarize(backend, rows, sampler):
    warm = [value["ttft_s"] for value in rows if value["state"] == "loaded" and value["ttft_s"] is not None]
    return dict(schema_version=1, backend=backend, model=GEMMA_FILENAME, model_sha256=GEMMA_SHA256,
                hardware=sampler.hardware, cases=rows, correct=sum(value["correct"] for value in rows),
                total=len(rows), request_first_text_s=rows[0]["ttft_s"],
                loaded_median_first_text_s=median(warm) if warm else None,
                memory={key: round(value, 3) if value is not None else None
                        for key, value in sampler.metrics.items()})


def run_bovary(preset, worker_factory=PinnedGemma):
    rows = []
    with worker_factory() as worker, MemorySampler(lambda: worker.process) as memory:
        for case in CASES:
            state = "loaded" if worker.loaded else "startup"
            try:
                result = worker.execute(Request(
                    case[1], local_profile=preset, allow_gpu=preset != "cpu-only", threads=THREADS,
                    context_tokens=CONTEXT_TOKENS, output_tokens=OUTPUT_TOKENS, deadline=time.monotonic() + 180,
                ))
                rows.append(row(case, result.timing, result.text,
                                state="loaded" if result.timing.get("warm") else "startup",
                                input_tokens=result.usage.get("input_tokens"), output_tokens=result.usage.get("output_tokens"),
                                failure=None if result.status == "complete" else "incomplete"))
            except ExecutionError:
                rows.append(row(case, {}, "", state=state, failure="worker_failed"))
    return summarize(f"Bovary {__version__} / {preset}", rows, memory)


def display(report):
    def format_value(value, units):
        return "unavailable" if value is None else f"{value:.2f} {units}"

    print(f"{report['backend']}  |  {report['model']}")
    print(f"Correct answers: {report['correct']}/{report['total']}")
    print(f"First text including worker startup: {format_value(report['request_first_text_s'], 's')}")
    print(f"Median first text with model loaded: {format_value(report['loaded_median_first_text_s'], 's')}")
    memory = report["memory"]
    print(f"Peak worker RAM (working set): {format_value(memory['worker_peak_working_set_mib'], 'MiB')}")
    print(f"Sampled worker private commit: {format_value(memory['worker_sampled_peak_private_commit_mib'], 'MiB')}")
    print(f"Sampled worker read transfers: {format_value(memory['worker_sampled_read_transfer_mib'], 'MiB')}")
    print(f"Sampled device-wide NVIDIA VRAM peak: {format_value(memory['gpu0_sampled_peak_used_mib'], 'MiB')} "
          f"(baseline {format_value(memory['gpu0_baseline_used_mib'], 'MiB')})")



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--json-only", action="store_true", help="Write only the sanitized JSON")
    args = parser.parse_args(argv)
    report = run_bovary(args.preset)
    if not args.json_only:
        display(report)
        print("\nShareable report (built-in cases, no questions or answers):")
    print(json.dumps(report, indent=2))
    return 0 if report["correct"] == report["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
