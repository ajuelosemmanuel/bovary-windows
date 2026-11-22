"""Offline CPU setup diagnostics. No model loading, downloads or provider access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time

from .runtime import project_home
from .modelproof import check_binding, requires_digest_check

STATUSES = ('pass', 'fail', 'optional', 'unknown')
VERIFY_REMEDY = ('From the source checkout run: python tools/verify_model.py '
                 '--model <model.gguf> --manifest <new-proof.json>; check the exit code before inference. '
                 'Then select that model and proof, or set BOVARY_MODEL and BOVARY_MODEL_MANIFEST.')


def _check(name, status, message, remedy=None):
    value = {'id': name, 'status': status, 'message': message}
    if remedy:
        value['remedy'] = remedy
    return value


def resolve_paths(home=None, *, model_path=None, manifest_path=None, executable=None, use_environment=True):
    """Explicit paths override environment; relative paths use the effective home."""
    root = Path(home).expanduser().resolve() if home is not None else project_home()
    def path(value, fallback):
        result = Path(value or fallback).expanduser()
        return (result if result.is_absolute() else root / result).resolve()
    environment = os.environ if use_environment else {}
    manifest = path(manifest_path or environment.get('BOVARY_MODEL_MANIFEST'), 'work/models/model-provenance.json')
    model = model_path or environment.get('BOVARY_MODEL')
    if not model:
        try:
            metadata = _read_manifest(manifest)
            filename = metadata.get('filename')
            # Match the runtime's default directory and basename behavior.
            model = Path('work/models') / Path(filename).name if isinstance(filename, str) and filename else None
        except (OSError, ValueError, UnicodeError, RecursionError):
            pass
    return {'home': root, 'model': path(model, 'work/models/gemma-4-26B-A4B-it-Q4_0.gguf'),
            'manifest': manifest,
            'executable': path(executable or environment.get('BOVARY_CPU_EXE'), 'work/build/bovary-cpu/bovary-infer.exe')}


def _read_manifest(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('Manifest exceeds 64 KiB')
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Manifest must be an object')
    return value


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def check_model(model, manifest, *, verify_hash=False, deadline=None, cancel=None):
    """Check model/proof consistency without loading it or writing a proof.

    The default checks metadata only. Explicit hashing compares with the stored
    proof; it does not establish the publisher's artifact identity or provenance.
    """
    model, manifest = Path(model), Path(manifest)
    checks = []
    try:
        metadata = _read_manifest(manifest)
        if (metadata.get('verified') is not True
                or type(metadata.get('bytes')) is not int or metadata['bytes'] < 1
                or not isinstance(metadata.get('sha256'), str)
                or re.fullmatch(r'[0-9a-fA-F]{64}', metadata['sha256']) is None
                or not isinstance(metadata.get('filename'), str)
                or metadata['filename'] != model.name):
            raise ValueError('Incomplete or mismatched verification manifest')
    except (OSError, ValueError, UnicodeError, RecursionError):
        return [_check('model_manifest', 'fail', 'Verification manifest is missing, malformed, or names another model.', VERIFY_REMEDY)]
    checks.append(_check('model_manifest', 'pass', 'Stored completed proof has a matching filename and well-formed size/SHA256. This is not a fresh hash verification.'))
    try:
        before = model.stat()
        if not model.is_file() or before.st_size != metadata['bytes']:
            raise ValueError('Size differs')
    except (OSError, ValueError):
        checks.append(_check('model_file', 'fail', 'Model file is missing, unreadable, or its size differs from the proof.', VERIFY_REMEDY))
        return checks
    try:
        binding = check_binding(model, metadata, allow_full_hash=False)
    except (OSError, ValueError):
        checks.append(_check('model_file', 'fail', 'Model file identity differs from the proof or the proof lacks a required file binding.', VERIFY_REMEDY))
        return checks
    if requires_digest_check(metadata, binding) and not verify_hash:
        checks.append(_check('model_file', 'unknown',
                             'File metadata matches the proof, but a Windows change-journal marker is unavailable. '
                             'Use explicit hashing to check the bytes now; runtime will check them before use.'))
    else:
        message = ('Model filesystem identity matches the stored full-file proof.' if binding else
                   'Model exists and its size matches the stored proof; contents have not been freshly verified.')
        checks.append(_check('model_file', 'pass', message))
    if not verify_hash:
        checks.append(_check('model_hash', 'unknown', 'Full model hashing was not requested.', 'Use bovary doctor --verify-hash --timeout-s 600 to compare all bytes with the stored proof; use the pinned verifier to establish provenance.'))
        return checks
    try:
        digest = hashlib.sha256()
        with model.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if _identity(opened) != _identity(before):
                raise ValueError('Model changed')
            while True:
                if (cancel is not None and cancel.is_set()) or (deadline is not None and time.monotonic() >= deadline):
                    checks.append(_check('model_hash', 'unknown', 'Full hashing was cancelled or reached the diagnostic timeout; no proof was written.'))
                    return checks
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if _identity(opened) != _identity(os.fstat(stream.fileno())):
                raise ValueError('Model changed')
        if (_identity(before) != _identity(model.stat()) or digest.hexdigest() != metadata['sha256'].lower()
                or (binding is not None and check_binding(model, metadata, allow_full_hash=False) != binding)):
            raise ValueError('Hash mismatch or model changed')
    except (OSError, ValueError):
        checks.append(_check('model_hash', 'fail', 'Model changed during hashing or its SHA256 differs from the stored proof; no proof was written.', VERIFY_REMEDY))
        return checks
    checks.append(_check('model_hash', 'pass', 'Fresh full-file SHA256 matches the stored proof; publisher provenance is not independently established and no proof was written.'))
    return checks


def _probe(command, *, deadline, cwd=None):
    """Capture at most 4 KiB and own the entire probe tree, even inherited pipes."""
    from .local import _WindowsJob, _OWNED_CREATION_FLAGS, _stop_owned_process
    if time.monotonic() >= deadline:
        return 'timeout', b''
    process = job = reader = None
    output = bytearray()
    done = threading.Event()
    errors = []
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, cwd=cwd, creationflags=_OWNED_CREATION_FLAGS,
                                   start_new_session=os.name != 'nt')
        job = _WindowsJob(process)
        job.resume(process)
        def capture():
            try:
                while len(output) <= 4096:
                    chunk = process.stdout.read1(4097 - len(output))
                    if not chunk:
                        break
                    output.extend(chunk)
            except (OSError, ValueError):
                errors.append(True)
            finally:
                process.stdout.close()
                done.set()
        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        while time.monotonic() < deadline:
            if len(output) > 4096:
                return 'oversized', b''
            if done.wait(min(.02, max(0, deadline - time.monotonic()))) and process.poll() is not None:
                if len(output) > 4096:
                    return 'oversized', b''
                return ('ok' if process.returncode == 0 and not errors else 'failed'), bytes(output)
        return 'timeout', b''
    except OSError:
        return 'unavailable', b''
    finally:
        if process is not None:
            _stop_owned_process(process, job)
            if reader is None or reader.ident is None:
                process.stdout.close()
            else:
                reader.join(timeout=1)
        if job is not None:
            job.close()


def _windows_avx2():
    """The documented PF_AVX2 flag needs Windows 10 2004+ (SDK 19041).

    https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
    """
    try:
        import ctypes
        from ctypes import wintypes
        version = sys.getwindowsversion()
        if version.major < 10 or version.build < 19041:
            return None
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        query = kernel.IsProcessorFeaturePresent
        query.argtypes, query.restype = [wintypes.DWORD], wintypes.BOOL
        return bool(query(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
    except (OSError, AttributeError):
        return None


def _cpu_check():
    # Do not execute generated machine code or infer FMA from a CPU brand/AVX2.
    if os.name != 'nt':
        return _check('cpu_features', 'unknown', 'AVX2/FMA support is not inspected on this platform.', 'Use Windows x64 with an AVX2/FMA CPU; consult the CPU specifications.')
    avx2 = _windows_avx2()
    if avx2 is False:
        return _check('cpu_features', 'fail', 'Windows reports that AVX2 instructions are unavailable.', 'Use a Windows x64 host with AVX2/FMA support; do not start the current CPU engine on this host.')
    if avx2 is True:
        return _check('cpu_features', 'unknown', 'Windows reports AVX2 support. FMA support is not independently established; a protocol probe does not exercise model compute.', 'Confirm FMA support in your CPU specifications before inference.')
    return _check('cpu_features', 'unknown', 'AVX2/FMA support cannot be established reliably by these offline checks. A protocol probe does not exercise model compute.', 'Confirm AVX2 and FMA support in your CPU specifications before inference.')


def _runtime_check(executable):
    if os.name != 'nt':
        return _check('microsoft_runtime', 'unknown', 'Microsoft runtime checks apply to Windows x64.')
    # Search only the loader's application/system locations, never load arbitrary DLLs.
    system = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32'
    locations = (executable.parent, system)
    missing = [name for name in ('vcruntime140.dll', 'msvcp140.dll', 'vcomp140.dll')
               if not any((directory / name).is_file() for directory in locations)]
    if missing:
        return _check('microsoft_runtime', 'unknown', 'Common Microsoft x64 C++/OpenMP DLLs were not all found: ' + ', '.join(missing) + '. Exact imports depend on the build.', 'If the native probe fails with a loader error, install the matching Microsoft Visual C++ x64 Redistributable/OpenMP runtime for your build.')
    return _check('microsoft_runtime', 'pass', 'Common Microsoft C++/OpenMP DLL files exist in application/system locations; versions and exact binary dependencies are not proven. The native probe tests loader startup.')


def diagnose(home=None, *, model_path=None, manifest_path=None, executable=None,
             timeout_s=10, verify_hash=False, show_paths=False):
    """Return a stable, JSON-safe report. Probe time uses one overall deadline.

    Exit 0: no detected required failure (unknowns remain unknown); 1: one or
    more required failures. CLI usage/configuration errors use exit 2.
    Filesystem/OS process teardown time may extend the probe deadline.
    """
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
        raise ValueError('Diagnostic timeout must be positive and at most 86400 seconds')
    deadline = time.monotonic() + timeout_s
    paths = resolve_paths(home, model_path=model_path, manifest_path=manifest_path, executable=executable)
    checks = [_check('python', 'pass' if sys.version_info >= (3, 11) else 'fail',
                     'Python ' + platform.python_version() + '; Bovary requires Python 3.11+.', 'Install Python 3.11+ x64 with Tcl/Tk.'),
              _check('platform', 'pass' if sys.platform == 'win32' and platform.machine().lower() in ('amd64', 'x86_64') and struct.calcsize('P') == 8 else 'fail',
                     'Supported release target: Windows x64 with 64-bit Python. Current: ' + sys.platform + '/' + platform.machine() + '.', 'Use Windows x64 and 64-bit Python.'),
              _cpu_check()]
    checks.append(_check('home', 'pass' if paths['home'].is_dir() else 'fail', 'Effective home is ' + ('an existing directory.' if paths['home'].is_dir() else 'missing or not a directory.'), 'Set BOVARY_HOME to your checkout/runtime directory.'))
    state, output = _probe([sys.executable, '-I', '-c', 'import tkinter; t = tkinter.Tcl(); print("tk-available")'], deadline=min(deadline, time.monotonic() + 5))
    checks.append(_check('tk', 'pass' if state == 'ok' and output.strip() == b'tk-available' else ('unknown' if state == 'timeout' else 'fail'), 'Tkinter/Tcl import probe ' + ('passed without a visible window.' if state == 'ok' and output.strip() == b'tk-available' else 'did not pass (' + state + ').'), 'Install Python with the Tcl/Tk component; a headless import does not prove display availability.'))
    exe = paths['executable']
    checks.append(_runtime_check(exe))
    if not exe.is_file():
        checks.append(_check('native_executable', 'fail', 'CPU native executable is missing.', 'Run scripts\\build.cmd cpu 4, or select the existing CPU executable / set BOVARY_CPU_EXE.'))
    else:
        checks.append(_check('native_executable', 'pass', 'Configured native executable exists.'))
        state, output = _probe([str(exe), '--require-cpu-only', '--protocol-version'], deadline=min(deadline, time.monotonic() + 5), cwd=paths['home'])
        valid = state == 'ok' and output in (b'bovary-worker 2\n', b'bovary-worker 2\r\n')
        checks.append(_check('native_protocol', 'pass' if valid else ('unknown' if state == 'timeout' else 'fail'), 'CPU-only parser protocol probe ' + ('passed (protocol 2); no model/backend was initialized.' if valid else 'did not pass (' + state + '); binary may be stale, incompatible, or missing runtime dependencies.'), 'Rebuild with scripts\\build.cmd cpu 4; select that CPU binary. Check Microsoft x64 runtime dependencies if startup fails.'))
    checks.extend(check_model(paths['model'], paths['manifest'], verify_hash=verify_hash, deadline=deadline))
    for tool, command in (('git', 'git'), ('cmake', 'cmake'), ('ninja', 'ninja'), ('compiler', 'cl')):
        available = shutil.which(command) is not None
        checks.append(_check('build_' + tool, 'pass' if available else 'optional',
                             command + (' is available on PATH (version/toolchain not validated).' if available else ' was not found on PATH; needed only for source builds. Visual Studio tools may be available outside this shell.'),
                             'For source builds use Git, CMake 3.22+, Ninja and Visual Studio C++ x64 tools/Windows SDK; scripts\\build.cmd discovers Visual Studio tools.'))
        if tool == 'cmake' and available:
            state, output = _probe([shutil.which(command), '--version'], deadline=min(deadline, time.monotonic()+1))
            version = re.match(rb'cmake version (\d+)\.(\d+)(?:\.|\s)', output)
            supported = state == 'ok' and version is not None and tuple(map(int, version.groups())) >= (3, 22)
            checks.append(_check('build_cmake_version', 'pass' if supported else ('unknown' if state == 'timeout' else 'optional'),
                                 'CMake 3.22+ version check ' + ('passed.' if supported else 'did not establish a supported version; this matters only for source builds.'),
                                 'For source builds select CMake 3.22+ on PATH or the Visual Studio CMake tools component.'))
    return {'schema_version': 1, 'checks': checks,
            'paths': {key: str(value) if show_paths else '<redacted>' for key, value in paths.items()},
            'paths_redacted': not show_paths, 'full_hash_requested': bool(verify_hash),
            'exit_code': 1 if any(check['status'] == 'fail' for check in checks) else 0}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bovary doctor', description=__doc__, epilog='Exit codes: 0 no detected required failures (unknown checks remain unknown); 1 required failure; 2 invalid arguments. Paths are redacted unless --show-paths is explicit.')
    parser.add_argument('--json', action='store_true', help='Emit structured JSON')
    parser.add_argument('--home', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--executable', type=Path, help='CPU-only native executable')
    parser.add_argument('--timeout-s', type=float, default=10, help='Overall probe/hash deadline, seconds (default 10); OS cleanup may add time')
    parser.add_argument('--verify-hash', action='store_true', help='Explicitly read all model bytes and compare with stored proof; never write a proof')
    parser.add_argument('--show-paths', action='store_true', help='Include private local paths; sanitize before sharing')
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.home, model_path=args.model, manifest_path=args.manifest, executable=args.executable, timeout_s=args.timeout_s, verify_hash=args.verify_hash, show_paths=args.show_paths)
    except (OSError, ValueError) as error:
        # Exception strings can include private paths, so expose only the category.
        parser.error('Invalid diagnostic configuration (' + type(error).__name__ + '); check paths and timeout.')
    if args.json:
        print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    else:
        print('Bovary doctor - offline CPU setup (no model loaded)')
        for key, value in report['paths'].items():
            # Escape private Unicode paths losslessly even in legacy redirected streams.
            print(key + ': ' + json.dumps(value, ensure_ascii=True))
        for check in report['checks']:
            print('[' + check['status'].upper() + '] ' + check['id'] + ': ' + check['message'])
            if check.get('remedy') and check['status'] != 'pass':
                print('  Remedy: ' + check['remedy'])
        print('Unknown checks do not establish readiness. Exit code: ' + str(report['exit_code']))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
