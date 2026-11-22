"""Dependency-free Windows process/system/PDH measurements.

Unavailable metrics are None, never zero. Counters do not establish cold-cache
conditions or assign system file-cache growth to one process.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import os
import re


class MemoryStatus(C.Structure):
    _fields_ = [("length", W.DWORD), ("load", W.DWORD)] + [
        (n, C.c_ulonglong) for n in ("total_phys", "available_phys", "total_pagefile",
        "available_pagefile", "total_virtual", "available_virtual", "available_extended_virtual")]


class ProcessMemory(C.Structure):
    _fields_ = [("cb", W.DWORD), ("page_fault_count", W.DWORD)] + [
        (n, C.c_size_t) for n in ("peak_working_set", "working_set", "quota_peak_paged_pool",
        "quota_paged_pool", "quota_peak_nonpaged_pool", "quota_nonpaged_pool", "pagefile_usage",
        "peak_pagefile_usage", "private_usage")]


class PerformanceInfo(C.Structure):
    _fields_ = [("cb", W.DWORD)] + [(n, C.c_size_t) for n in (
        "commit_total", "commit_limit", "commit_peak", "physical_total", "physical_available",
        "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size")] + [
        (n, W.DWORD) for n in ("handle_count", "process_count", "thread_count")]


class IoCounters(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ("read_operations", "write_operations",
        "other_operations", "read_bytes", "write_bytes", "other_bytes")]


class SystemPowerStatus(C.Structure):
    _fields_ = [(name, W.BYTE) for name in ("ac_line_status", "battery_flag", "battery_life_percent", "system_status_flag")] + [
        (name, W.DWORD) for name in ("battery_life_time", "battery_full_life_time")]


def power_snapshot():
    if os.name != "nt": return {"available": False, "error": "Windows power API unavailable"}
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemPowerStatus.argtypes = [C.POINTER(SystemPowerStatus)]
    status = SystemPowerStatus()
    if not kernel.GetSystemPowerStatus(C.byref(status)):
        return {"available": False, "error": f"GetSystemPowerStatus: {C.get_last_error()}"}
    return {"available": True, "ac_line_status_raw": status.ac_line_status,
        "ac_state": {0: "offline", 1: "online", 255: "unknown"}.get(status.ac_line_status, "unknown"),
        "battery_flag_raw": status.battery_flag,
        "battery_present": False if status.battery_flag == 128 else None if status.battery_flag == 255 else True,
        "battery_percent": None if status.battery_life_percent == 255 else status.battery_life_percent,
        "battery_saver": bool(status.system_status_flag),
        "battery_remaining_seconds": None if status.battery_life_time == 0xffffffff else status.battery_life_time,
        "battery_full_life_seconds": None if status.battery_full_life_time == 0xffffffff else status.battery_full_life_time}


class PdhValueUnion(C.Union):
    _fields_ = [("long_value", W.LONG), ("double_value", C.c_double),
                ("large_value", C.c_longlong), ("string_value", C.c_void_p)]


class PdhValue(C.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", W.DWORD), ("value", PdhValueUnion)]


class PdhItem(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", PdhValue)]


def process_instance_value(pid, instance_pids, instance_values):
    """Join one collection by full PDH instance name and exact numeric PID."""
    matches = [name for name, value in instance_pids.items() if value == pid]
    return instance_values.get(matches[0]) if len(matches) == 1 else None


class PdhSampler:
    """English counter paths work independently of Windows display language."""
    PATHS = {
        "_process_instance_pid": r"\Process(*)\ID Process",
        "process_virtual_address_bytes": r"\Process(*)\Virtual Bytes",
        "system_page_reads_per_second": r"\Memory\Page Reads/sec",
        "system_pages_input_per_second": r"\Memory\Pages Input/sec",
        "system_cache_bytes_pdh": r"\Memory\Cache Bytes",
        "gpu_process_dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
        "gpu_process_shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
        "gpu_adapter_dedicated_bytes": r"\GPU Adapter Memory(*)\Dedicated Usage",
        "gpu_adapter_shared_bytes": r"\GPU Adapter Memory(*)\Shared Usage",
    }

    def __init__(self, pid: int):
        self.pid, self.query, self.counters, self.errors = pid, C.c_void_p(), {}, {}
        self.pdh = C.WinDLL("pdh", use_last_error=True)
        self.pdh.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
        self.pdh.PdhGetFormattedCounterValue.argtypes = [C.c_void_p, W.DWORD, C.c_void_p, C.POINTER(PdhValue)]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [C.c_void_p]
        status = self.pdh.PdhOpenQueryW(None, 0, C.byref(self.query))
        if status:
            raise OSError(f"PdhOpenQueryW: {status & 0xffffffff:#x}")
        for name, path in self.PATHS.items():
            handle = C.c_void_p()
            status = self.pdh.PdhAddEnglishCounterW(self.query, path, 0, C.byref(handle))
            if status:
                self.errors[name] = f"PdhAddEnglishCounterW: {status & 0xffffffff:#x}"
            else:
                self.counters[name] = handle
        self.pdh.PdhCollectQueryData(self.query)

    def array_values(self, name, handle):
        size, count = W.DWORD(), W.DWORD()
        # DOUBLE | NOSCALE: identifiers and byte counters retain their native units.
        self.pdh.PdhGetFormattedCounterArrayW(handle, 0x1200, C.byref(size), C.byref(count), None)
        if not size.value: return {}
        buf = C.create_string_buffer(size.value)
        status = self.pdh.PdhGetFormattedCounterArrayW(handle, 0x1200, C.byref(size), C.byref(count), buf)
        if status:
            self.errors[name] = f"PdhGetFormattedCounterArrayW: {status & 0xffffffff:#x}"
            return {}
        items = C.cast(buf, C.POINTER(PdhItem))
        return {items[i].name: items[i].value.double_value for i in range(count.value)
            if items[i].name and items[i].value.status in (0, 1)}

    def sample(self):
        result = {name: None for name in self.PATHS if not name.startswith("_")}
        status = self.pdh.PdhCollectQueryData(self.query)
        if status:
            self.errors["collect"] = f"{status & 0xffffffff:#x}"
            return result
        instance_pids, virtual_bytes = {}, {}
        for name, handle in self.counters.items():
            if name in ("_process_instance_pid", "process_virtual_address_bytes"):
                values = self.array_values(name, handle)
                if name == "_process_instance_pid": instance_pids = values
                else: virtual_bytes = values
                continue
            if name.startswith("gpu_"):
                values = [value for instance, value in self.array_values(name, handle).items()
                    if not name.startswith("gpu_process_") or re.search(rf"(?:^|_)pid_{self.pid}(?:_|$)", instance)]
                # No matching instance is unavailable; it may not have registered yet.
                if values:
                    result[name] = sum(values)
            else:
                value = PdhValue()
                status = self.pdh.PdhGetFormattedCounterValue(handle, 0x1200, None, C.byref(value))
                if not status and value.status in (0, 1):
                    result[name] = value.double_value
        result["process_virtual_address_bytes"] = process_instance_value(self.pid, instance_pids, virtual_bytes)
        return result

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


class WindowsSampler:
    def __init__(self, pid: int, enable_pdh: bool = True):
        self.pid, self.errors = pid, {}
        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        self.psapi = C.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.kernel.OpenProcess.restype = W.HANDLE
        self.kernel.CloseHandle.argtypes = [W.HANDLE]
        self.kernel.GlobalMemoryStatusEx.argtypes = [C.POINTER(MemoryStatus)]
        self.kernel.GetProcessIoCounters.argtypes = [W.HANDLE, C.POINTER(IoCounters)]
        self.psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.POINTER(ProcessMemory), W.DWORD]
        self.psapi.GetPerformanceInfo.argtypes = [C.POINTER(PerformanceInfo), W.DWORD]
        self.process = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not self.process:
            self.errors["process"] = f"OpenProcess: {C.get_last_error()}"
        self.pdh = None
        if enable_pdh:
            try:
                self.pdh = PdhSampler(pid)
            except Exception as exc:
                self.errors["pdh"] = repr(exc)

    def sample(self):
        result = {name: None for name in (
            "process_working_set_bytes", "process_peak_working_set_bytes", "process_private_commit_bytes", "process_virtual_address_bytes",
            "process_page_faults_total", "process_io_read_transfer_bytes_total", "process_io_read_operations_total",
            "system_total_physical_bytes", "system_available_physical_bytes", "system_used_physical_bytes",
            "system_commit_bytes", "system_commit_limit_bytes", "system_cache_including_standby_bytes")}
        memory = ProcessMemory(cb=C.sizeof(ProcessMemory))
        if self.process and self.psapi.GetProcessMemoryInfo(self.process, C.byref(memory), C.sizeof(memory)):
            result.update(process_working_set_bytes=memory.working_set,
                process_peak_working_set_bytes=memory.peak_working_set,
                process_private_commit_bytes=memory.private_usage,
                process_page_faults_total=memory.page_fault_count)
        io = IoCounters()
        if self.process and self.kernel.GetProcessIoCounters(self.process, C.byref(io)):
            result.update(process_io_read_transfer_bytes_total=io.read_bytes,
                          process_io_read_operations_total=io.read_operations)
        status = MemoryStatus(length=C.sizeof(MemoryStatus))
        if self.kernel.GlobalMemoryStatusEx(C.byref(status)):
            result.update(system_total_physical_bytes=status.total_phys,
                system_available_physical_bytes=status.available_phys,
                system_used_physical_bytes=status.total_phys-status.available_phys)
        perf = PerformanceInfo(cb=C.sizeof(PerformanceInfo))
        if self.psapi.GetPerformanceInfo(C.byref(perf), C.sizeof(perf)):
            result.update(system_commit_bytes=perf.commit_total*perf.page_size,
                system_commit_limit_bytes=perf.commit_limit*perf.page_size,
                system_cache_including_standby_bytes=perf.system_cache*perf.page_size)
        if self.pdh:
            result.update(self.pdh.sample())
        return result

    def close(self):
        if self.process:
            self.kernel.CloseHandle(self.process)
            self.process = None
        if self.pdh:
            self.errors.update(self.pdh.errors)
            self.pdh.close()


class UnavailableSampler:
    errors = {"platform": "Windows native metrics unavailable on this platform"}
    def __init__(self, *_args, **_kwargs): pass
    def sample(self): return {}
    def close(self): pass


Sampler = WindowsSampler if os.name == "nt" else UnavailableSampler
