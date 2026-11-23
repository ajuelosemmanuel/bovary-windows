"""Select the tested Windows Gemma CUDA profile without loading model weights."""
import ctypes
import os
from pathlib import Path
import subprocess


def compatible_device():
    if os.name != "nt":
        return False
    try:
        driver = ctypes.WinDLL("nvcuda.dll", winmode=0x800)
        device, major, minor, version = (ctypes.c_int() for _ in range(4))
        memory = ctypes.c_size_t()
        return (
            driver.cuInit(0) == 0
            and driver.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value >= 13000
            and driver.cuDeviceGet(ctypes.byref(device), 0) == 0
            and driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device) == 0
            and (major.value, minor.value) >= (8, 6)
            and driver.cuDeviceTotalMem_v2(ctypes.byref(memory), device) == 0
            # The driver reports slightly less than the installed capacity on Windows.
            and memory.value >= 5632 * 1024**2
        )
    except (OSError, AttributeError):
        return False


def gemma_preset(home, env):
    path = Path(home) / env.get("BOVARY_CUDA_EXE", "work/build/bovary-cuda/bovary-infer.exe")
    if path.is_file() and compatible_device():
        toolkit = Path(home) / "work/deps/cuda-13.3/bin"
        search = os.pathsep.join((str(toolkit / "x64"), str(toolkit), env.get("PATH", "")))
        try:
            probe = subprocess.run([str(path), "--host-cache-protocol-version"],
                                   env=dict(env, PATH=search), capture_output=True, timeout=10)
            if probe.returncode == 0 and probe.stdout.strip() == b"bovary-worker 2":
                return "gpu-prefill-2g"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "cpu-only"
