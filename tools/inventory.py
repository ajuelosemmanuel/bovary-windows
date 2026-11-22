"""Capture Windows hardware/build inventory and a bounded buffered/unbuffered SSD probe.

Run from workspace: python tools/inventory.py [--disk-probe]
No system settings are changed. The disk probe uses only its own 128 MiB scratch file.
"""
from __future__ import annotations
import argparse
import ctypes as C
from ctypes import wintypes as W
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def command(args, timeout=30):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"command": args, "error": str(e)}

class MEMORYSTATUSEX(C.Structure):
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD)] + [(n, C.c_ulonglong) for n in
        ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

class SYSTEM_POWER_STATUS(C.Structure):
    _fields_ = [(n, W.BYTE) for n in ("ACLineStatus", "BatteryFlag", "BatteryLifePercent", "SystemStatusFlag")] + [(n, W.DWORD) for n in ("BatteryLifeTime", "BatteryFullLifeTime")]

def native_status():
    k = C.WinDLL("kernel32", use_last_error=True)
    m = MEMORYSTATUSEX(); m.dwLength=C.sizeof(m)
    if not k.GlobalMemoryStatusEx(C.byref(m)): raise C.WinError(C.get_last_error())
    p=SYSTEM_POWER_STATUS(); k.GetSystemPowerStatus(C.byref(p))
    return {"memory": {n: getattr(m,n) for n,_ in m._fields_},
            "memory_note": "ullTotalPageFile is system commit limit, not pagefile bytes; virtual address space is not physical residency.",
            "cpu_os_instruction_support": {name:bool(k.IsProcessorFeaturePresent(flag)) for name,flag in (("AVX",39),("AVX2",40),("AVX512F",41))},
            "power": {n:getattr(p,n) for n,_ in p._fields_}}

def disk_probe():
    """No cache eviction: buffered=warm after write; unbuffered bypasses Windows file cache."""
    scratch=ROOT/"work"/"environment"/"ssd-probe.bin"; scratch.parent.mkdir(parents=True, exist_ok=True)
    total=128*1024*1024; block=1024*1024
    k=C.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,C.c_void_p,W.DWORD,W.DWORD,W.HANDLE]; k.CreateFileW.restype=W.HANDLE
    k.VirtualAlloc.argtypes=[C.c_void_p,C.c_size_t,W.DWORD,W.DWORD]; k.VirtualAlloc.restype=C.c_void_p
    k.VirtualFree.argtypes=[C.c_void_p,C.c_size_t,W.DWORD]
    k.ReadFile.argtypes=[W.HANDLE,C.c_void_p,W.DWORD,C.POINTER(W.DWORD),C.c_void_p]
    k.SetFilePointerEx.argtypes=[W.HANDLE,C.c_longlong,C.c_void_p,W.DWORD]
    k.CloseHandle.argtypes=[W.HANDLE]
    # Nonzero incompressible-enough data; file isn't marked compressed/sparse.
    payload=os.urandom(block)
    with scratch.open("wb", buffering=0) as f:
        for _ in range(total//block): f.write(payload)
        os.fsync(f.fileno())
    buf=k.VirtualAlloc(None,block,0x3000,0x04)
    if not buf: raise C.WinError(C.get_last_error())
    results=[]
    try:
        for unbuffered in (False,True):
            flags=0x80 | (0x20000000 if unbuffered else 0)
            h=k.CreateFileW(str(scratch),0x80000000,1,None,3,flags,None)
            if h == C.c_void_p(-1).value:
                results.append({"unbuffered":unbuffered,"error":str(C.WinError(C.get_last_error()))}); continue
            try:
                for pattern,sz in (("sequential_1MiB",block),("random_64KiB",65536)):
                    lat=[]; count=total//sz if sz==block else 512
                    for i in range(count):
                        offset=i*sz if sz==block else ((i*104729)%(total//sz))*sz
                        n=W.DWORD(); t=time.perf_counter_ns()
                        if not k.SetFilePointerEx(h,offset,None,0): raise C.WinError(C.get_last_error())
                        if not k.ReadFile(h,buf,sz,C.byref(n),None): raise C.WinError(C.get_last_error())
                        if n.value != sz: raise RuntimeError("short read")
                        lat.append((time.perf_counter_ns()-t)/1e6)
                    seconds=sum(lat)/1000; ordered=sorted(lat)
                    results.append({"unbuffered":unbuffered,"pattern":pattern,"bytes":count*sz,
                        "elapsed_seconds":seconds,"MiB_s":count*sz/1048576/seconds,
                        "latency_ms_p50":ordered[len(lat)//2],"latency_ms_p95":ordered[int(.95*len(lat))],
                        "latency_ms_samples":lat})
            finally: k.CloseHandle(h)
    finally:
        k.VirtualFree(buf,0,0x8000); scratch.unlink(missing_ok=True)
    return {"file_bytes":total,"results":results,"limitations":[
        "Buffered reads warm after file creation; not cold-cache results.",
        "Unbuffered bypasses OS file cache, not SSD/controller cache; 128MiB file is not full-model inference.",
        "One synchronous outstanding read; no claim about async inference throughput."]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--disk-probe",action="store_true"); args=ap.parse_args()
    out={"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"platform":platform.platform(),"native":native_status(),
         "drive_usage":dict(zip(("total","used","free"),shutil.disk_usage(ROOT))),
         "tools":{n:shutil.which(n) for n in ("git","cmake","ninja","cl","gcc","clang","python","rustc","cargo","nvcc","nvidia-smi","llama-cli","ollama")}}
    for name,args2 in {"gpu":["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,pci.bus_id,compute_cap,power.draw,power.limit,temperature.gpu","--format=csv,noheader"],
        "power_plan":["powercfg","/getactivescheme"],"cmake_version":["cmake","--version"],"git_version":["git","--version"]}.items(): out[name]=command(args2)
    ps="""$ErrorActionPreference='Stop'; $cpu=Get-CimInstance Win32_Processor; $sys=Get-CimInstance Win32_ComputerSystem; $disk=Get-PhysicalDisk; [pscustomobject]@{cpu=$cpu|Select-Object Name,NumberOfCores,NumberOfLogicalProcessors;ram_bytes=$sys.TotalPhysicalMemory;disks=$disk|Select-Object FriendlyName,MediaType,BusType,Size,HealthStatus} | ConvertTo-Json -Depth 5"""
    out["cim"]=command(["powershell.exe","-NoProfile","-Command",ps])
    for key in ("cim",):
        if out[key].get("exit_code")==0:
            try: out[key]["data"]=json.loads(out[key]["stdout"])
            except ValueError: pass
    cachepaths=[Path.home()/".cache/huggingface/hub",Path.home()/".ollama/models",Path.home()/".lmstudio/models",Path.home()/"Models",Path("C:/Models")]
    out["model_cache_roots"]=[{"path":str(p),"exists":p.exists(),"top_level_entries":[x.name for x in p.iterdir()] if p.exists() else []} for p in cachepaths]
    if args.disk_probe: out["disk_probe"]=disk_probe()
    dst=ROOT/"outputs/environment.json"; dst.parent.mkdir(exist_ok=True); dst.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(str(dst)); print(json.dumps({"ram":out["native"]["memory"]["ullTotalPhys"],"free_disk":out["drive_usage"]["free"],"gpu":out["gpu"]},indent=2))
if __name__=="__main__": main()
