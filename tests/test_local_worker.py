import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bovary.contract import ExecutionError, Request
from bovary.local import LocalAdapter, SerialLock
from bovary import local


class WindowsAffinityTests(unittest.TestCase):
    def test_worker_preserves_inherited_affinity_restrictions(self):
        job = object.__new__(local._WindowsJob)
        job.kernel = Mock()
        process = Mock(_handle=27)
        active = 0b1010

        def query(handle, process_mask, system_mask):
            self.assertEqual(handle, 27)
            process_mask._obj.value = active
            system_mask._obj.value = 0b1111
            return True

        def apply(handle, mask):
            nonlocal active
            self.assertEqual(handle, 27)
            active = mask
            return True

        job.kernel.GetProcessAffinityMask.side_effect = query
        job.kernel.SetProcessAffinityMask.side_effect = apply
        job.set_worker_affinity(process, 0b0011)
        job.kernel.SetProcessAffinityMask.assert_called_once_with(27, 0b0010)
        self.assertEqual(job.kernel.GetProcessAffinityMask.call_count, 2)
        job.kernel.SetProcessAffinityMask.reset_mock()
        with self.assertRaises(ExecutionError) as error:
            job.set_worker_affinity(process, 0b1101)
        self.assertEqual(error.exception.code, "worker_affinity")
        job.kernel.SetProcessAffinityMask.assert_not_called()


class LocalWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models = self.root / "work/models"
        self.models.mkdir(parents=True)
        (self.models / "fixture.gguf").write_bytes(b"fixture")
        (self.models / "model-provenance.json").write_text(json.dumps({
            "filename": "fixture.gguf", "bytes": 7, "sha256": "a" * 64, "verified": True}))
        self.adapter = self.make_adapter()

    def make_adapter(self, **kwargs):
        # Fixtures use only stdlib; own the actual process, not a venv redirector.
        return LocalAdapter(self.root, command_prefix=[sys._base_executable, "-S", str(ROOT / "tests/fixtures/fake_worker.py")], **kwargs)

    def tearDown(self):
        self.adapter.unload()
        self.temporary.cleanup()

    def request(self, prompt="hello", **kwargs):
        return Request(prompt, **{"context_tokens": 64, "output_tokens": 3, "threads": 1, **kwargs})

    def test_streaming_host_cache_changes_worker_identity_and_requires_capable_native(self):
        baseline = self.request(local_profile="gpu-prefill-2g", allow_gpu=True)
        cached = self.request(local_profile="gpu-prefill-2g", allow_gpu=True, host_cache_mib=512)
        with patch.dict(os.environ, {"BOVARY_STREAM_HOST_CACHE_MIB": "4096"}):
            baseline_command, baseline_env, baseline_key, _ = self.adapter._configuration(baseline)
            cached_command, cached_env, cached_key, _ = self.adapter._configuration(cached)
        self.assertNotIn("--host-cache-mib", baseline_command)
        self.assertEqual(cached_command, [*baseline_command, "--host-cache-mib", "512"])
        self.assertEqual(cached_env, baseline_env)
        self.assertNotIn("BOVARY_STREAM_HOST_CACHE_MIB", cached_env)
        self.assertEqual(cached_env["BOVARY_STREAM_IO"], "unbuffered")
        self.assertNotEqual(cached_key, baseline_key)
        probe = LocalAdapter(ROOT)
        with self.assertRaises(ExecutionError) as error:
            probe._preflight(cached_command, cached_env, cached, None)
        self.assertEqual(error.exception.code, "backend_version")
        self.assertFalse(probe.loaded)

    def test_until_finished_sends_zero_and_completes_at_eog_on_reused_worker(self):
        capped = self.adapter.execute(self.request())
        result = self.adapter.execute(self.request(output_tokens=None, deadline=time.monotonic() + 5))
        self.assertTrue(result.timing["warm"])
        self.assertEqual(result.timing["native"]["pid"], capped.timing["native"]["pid"])
        self.assertEqual(result.timing["native"]["requested_generated_tokens"], 0)
        self.assertEqual(result.usage["output_limit"], 64 - len("hello"))
        self.assertEqual(result.usage["finish_reason"], "eos")
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.text, capped.text)

    def test_until_finished_context_limit_and_explicit_token_limit_are_truncated(self):
        for prompt, output_tokens, reason, limit in (("CONTEXT_LIMIT", None, "context_limit", 4),
                                                    ("TOKEN_LIMIT", 3, "token_limit", 3)):
            with self.subTest(reason=reason):
                result = self.adapter.execute(self.request(prompt, output_tokens=output_tokens, deadline=time.monotonic() + 5))
                self.assertEqual(result.status, "truncated")
                self.assertEqual(result.usage["finish_reason"], reason)
                self.assertEqual(result.usage["output_limit"], limit)
                self.assertEqual(result.usage["output_tokens"], limit)
                self.assertEqual(result.text, "x" * limit)
                self.assertFalse(result.timing["native"]["early_stop"])

    def test_until_finished_empty_eog_and_no_remaining_context(self):
        result = self.adapter.execute(self.request("EOG_EMPTY", output_tokens=None, deadline=time.monotonic() + 5))
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.text, "")
        self.assertEqual(result.usage["finish_reason"], "eos")
        self.assertEqual(result.usage["output_tokens"], 0)
        self.assertIsNone(result.timing["ttft_s"])
        process = self.adapter.process
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("NO_CONTEXT", output_tokens=None, deadline=time.monotonic() + 5))
        self.assertEqual(caught.exception.code, "context_overflow")
        self.assertEqual(caught.exception.attempt["status"], "failed")
        recovered = self.adapter.execute(self.request())
        self.assertTrue(recovered.timing["warm"])
        self.assertEqual(process.pid, recovered.timing["native"]["pid"])

    def test_until_finished_cancel_and_deadline_reap_child_and_retain_attempt(self):
        self.adapter.execute(self.request())
        process = self.adapter.process
        cancel = threading.Event()
        timer = threading.Timer(.1, cancel.set)
        timer.start()
        try:
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request("SILENT", output_tokens=None, deadline=time.monotonic() + 5), cancel=cancel)
        finally:
            timer.cancel()
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertIsNotNone(caught.exception.attempt)
        self.assertIsNone(caught.exception.attempt["usage"]["output_limit"])
        self.assertIsNotNone(process.poll())
        self.adapter.execute(self.request())
        process = self.adapter.process
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("SILENT", output_tokens=None, deadline=time.monotonic() + .1))
        self.assertEqual(caught.exception.code, "deadline")
        self.assertIsNotNone(caught.exception.attempt)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.adapter.process)

    def test_fake_protocol_reuses_pid_and_resets_request_payload(self):
        events = []
        first = self.adapter.execute(self.request("marker"), on_event=events.append)
        second = self.adapter.execute(self.request("new"))
        self.assertFalse(first.timing["warm"])
        self.assertGreater(first.timing["startup_s"], 0)
        self.assertGreater(first.timing["configuration_s"], 0)
        self.assertGreaterEqual(first.timing["startup_s"], first.timing["configuration_s"])
        self.assertTrue(second.timing["warm"])
        self.assertEqual(second.timing["startup_s"], 0)
        self.assertGreater(second.timing["configuration_s"], 0)
        self.assertEqual(first.timing["native"]["pid"], second.timing["native"]["pid"])
        self.assertEqual(first.text, "".join(event.text for event in events))
        self.assertEqual(second.text, "reply:new 世界")
        self.assertNotIn("marker", second.text)
        self.assertIsNone(first.cost_usd)
        self.assertEqual(first.configuration, "cpu-only-ctx64-threads1-gemma4-text-v1")
        self.assertEqual(first.cost_status, "unmeasured_local_compute")
        self.assertFalse(list(self.root.rglob("prompt.txt")))

    def test_fake_protocol_configuration_change_reloads_and_unload_releases_lock(self):
        first = self.adapter.execute(self.request())
        self.adapter.execute(Request("hello", context_tokens=128, output_tokens=3, threads=1))
        self.assertNotEqual(first.timing["native"]["pid"], self.adapter.process.pid)
        lock = SerialLock(self.root / "work/benchmark.lock")
        with self.assertRaises(ExecutionError) as caught:
            lock.acquire()
        self.assertEqual(caught.exception.code, "busy")
        process = self.adapter.process
        self.adapter.unload()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.adapter.loaded)
        lock.acquire()
        lock.release()

    def test_cpu_prompt_batch_configures_worker_and_separates_measurements(self):
        first = self.adapter.execute(self.request())
        tuned = self.request(cpu_prompt_batch=64)
        command, *_ = self.adapter._configuration(tuned)
        self.assertEqual(command[command.index("--prompt-batch") + 1], "64")
        second = self.adapter.execute(tuned)
        self.assertNotEqual(first.configuration, second.configuration)
        self.assertFalse(second.timing["warm"])
        self.assertTrue(self.adapter.execute(tuned).timing["warm"])
        for batch, profile in ((0, "cpu-only"), (129, "cpu-only"), (True, "cpu-only"), (32, "gpu-fast")):
            with self.subTest(batch=batch, profile=profile), self.assertRaises(ExecutionError):
                self.request(cpu_prompt_batch=batch, local_profile=profile).check()

    def test_cpu_environment_and_forbidden_gpu_are_checked_before_start(self):
        with patch.object(os, 'environ', {**os.environ, "BOVARY_STREAM": "1", "BOVARY_STREAM_TRACE": "private.csv",
                                     "GGML_BACKEND_PATH": "unexpected.dll", "PATH": "unchanged"}):
            command, environment, *_ = self.adapter._configuration(self.request())
        self.assertEqual(environment["BOVARY_STREAM"], "0")
        self.assertEqual(environment["PATH"], "unchanged")
        self.assertNotIn("GGML_BACKEND_PATH", environment)
        self.assertFalse(any(key.startswith("BOVARY_STREAM_") for key in environment))
        self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--cache-mib") + 1], "0")
        self.assertNotIn("--gpu-embedding", command)
        self.assertIn("--require-cpu-only", command)
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(local_profile="balanced"))
        self.assertEqual(caught.exception.code, "gpu_forbidden")
        self.assertIsNone(self.adapter.process)

    def test_two_gib_expert_cache_uses_the_low_ram_streaming_path(self):
        args, env, *_ = self.adapter._configuration(self.request(local_profile="gpu-cache-2g", allow_gpu=True))
        self.assertEqual(args[args.index("--cache-mib") + 1], "2048")
        self.assertEqual(args[args.index("--load-mode") + 1], "direct")
        self.assertEqual(env["BOVARY_STREAM"], "1")
        self.assertEqual(env["BOVARY_STREAM_BUFFER_MIB"], "8")
        self.assertEqual(env["BOVARY_STREAM_LOAD_BUFFER_MIB"], "4")
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(local_profile="gpu-cache-2g"))
        self.assertEqual(caught.exception.code, "gpu_forbidden")

    def test_gpu_prefill_preset_uses_the_same_streaming_settings_with_larger_batches(self):
        cache = self.request(local_profile="gpu-cache-2g", allow_gpu=True)
        prefill = self.request(local_profile="gpu-prefill-2g", allow_gpu=True)
        cache_args, cache_env, *_ = self.adapter._configuration(cache)
        prefill_args, prefill_env, *_ = self.adapter._configuration(prefill)
        batch_index = cache_args.index("--prompt-batch") + 1
        self.assertEqual(cache_args[batch_index], "1")
        self.assertEqual(prefill_args[batch_index], "128")
        self.assertEqual(cache_args[:batch_index], prefill_args[:batch_index])
        self.assertEqual(cache_args[batch_index + 1:], prefill_args[batch_index + 1:])
        self.assertEqual(cache_env, prefill_env)

        baseline = self.adapter.execute(cache)
        first = self.adapter.execute(prefill)
        second = self.adapter.execute(prefill)
        self.assertNotEqual(baseline.configuration, first.configuration)
        self.assertFalse(first.timing["warm"])
        self.assertTrue(second.timing["warm"])
        self.assertEqual(first.timing["native"]["pid"], second.timing["native"]["pid"])

    def test_crash_and_bad_frame_unload_then_next_request_recovers(self):
        for prompt, code in (("CRASH", "worker_crashed"), ("BADFRAME", "protocol")):
            with self.subTest(prompt=prompt):
                self.adapter.execute(self.request())
                process = self.adapter.process
                with self.assertRaises(ExecutionError) as caught:
                    self.adapter.execute(self.request(prompt))
                self.assertEqual(caught.exception.code, code)
                self.assertFalse(self.adapter.loaded)
                self.assertIsNotNone(process.poll())
                result = self.adapter.execute(self.request("recovered"))
                self.assertFalse(result.timing["warm"])

    def test_cancel_and_deadline_reap_silent_child(self):
        self.adapter.execute(self.request())
        cancel = threading.Event()
        errors = []
        process = self.adapter.process

        def run():
            try:
                self.adapter.execute(self.request("SILENT"), cancel=cancel)
            except ExecutionError as error:
                errors.append(error.code)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(.1)
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "busy")
        cancel.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["cancelled"])
        self.assertIsNotNone(process.poll())
        self.adapter.execute(self.request())
        process = self.adapter.process
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("SILENT", deadline=time.monotonic() + .1))
        self.assertEqual(caught.exception.code, "deadline")
        self.assertIsNotNone(process.poll())

    def test_cancellation_during_startup_and_preflight_failures(self):
        self.adapter.command_prefix.append("--fake-delay-ready")
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(deadline=time.monotonic() + .1))
        self.assertEqual(caught.exception.code, "deadline")
        self.assertIsNone(self.adapter.process)
        self.adapter.command_prefix.pop()
        for request, code in ((self.request(allowed_destinations=("commercial",)), "destination_forbidden"),
                              (self.request("x" * (1024 * 1024 + 1)), "context_overflow")):
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(request)
            self.assertEqual(caught.exception.code, code)
        self.assertIsNone(self.adapter.process)

    def test_missing_model_and_backend_never_fall_back(self):
        (self.models / "fixture.gguf").unlink()
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "model_unavailable")
        self.assertIsNone(self.adapter.process)

    def test_default_adapter_root_respects_runtime_home(self):
        with patch.object(os, 'environ', {**os.environ, "BOVARY_HOME": str(self.root)}):
            adapter = LocalAdapter()
        try:
            self.assertEqual(adapter.root, self.root.resolve())
        finally:
            adapter.unload()

    def test_bovary_environment_selects_model_manifest_and_device_executable(self):
        cpu_executable = self.root / "native/cpu.exe"
        cuda_executable = self.root / "native/cuda.exe"
        cpu_executable.parent.mkdir()
        cpu_executable.touch()
        cuda_executable.touch()
        with patch.object(os, 'environ', {
                "BOVARY_MODEL": "work/models/fixture.gguf",
                "BOVARY_MODEL_MANIFEST": "work/models/model-provenance.json",
                "BOVARY_CPU_EXE": "native/cpu.exe", "BOVARY_CUDA_EXE": "native/cuda.exe"}):
            adapter = LocalAdapter(self.root)
            try:
                cpu_command, _, *_ = adapter._configuration(self.request())
                cuda_command, _, *_ = adapter._configuration(self.request(local_profile="gpu-fast", allow_gpu=True))
                self.assertEqual(adapter.model_path, self.models / "fixture.gguf")
                self.assertEqual(adapter.manifest_path, self.models / "model-provenance.json")
                self.assertEqual(Path(cpu_command[0]), cpu_executable)
                self.assertEqual(Path(cuda_command[0]), cuda_executable)
                self.assertIn("--require-cpu-only", cpu_command)
                self.assertNotIn("--require-cpu-only", cuda_command)
            finally:
                adapter.unload()

    def test_stale_or_wrong_executable_is_rejected_before_worker_or_lock(self):
        for flag in ("--fake-old-brand", "--fake-old-protocol", "--fake-wrong-version", "--fake-cuda-only"):
            with self.subTest(flag=flag):
                self.adapter.command_prefix.append(flag)
                try:
                    with self.assertRaises(ExecutionError) as caught:
                        self.adapter.execute(self.request())
                    self.assertEqual(caught.exception.code, "backend_version")
                    self.assertIn("scripts\\build.cmd cpu 4", str(caught.exception))
                    self.assertIsNone(self.adapter.process)
                    self.assertFalse((self.root / "work/benchmark.lock").exists())
                finally:
                    self.adapter.command_prefix.pop()

    def test_old_ready_protocol_is_rejected_and_worker_is_unloaded(self):
        self.adapter.command_prefix.append("--fake-old-ready")
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "protocol")
        self.assertIsNone(self.adapter.process)

    def test_version_preflight_is_bounded_and_cancelled_without_model_start(self):
        self.adapter.command_prefix.append("--fake-delay-version")
        self.adapter.startup_timeout_s = .1
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "backend_version")
        self.assertIsNone(self.adapter.process)
        self.assertFalse((self.root / "work/benchmark.lock").exists())
        self.adapter.startup_timeout_s = 5
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(deadline=time.monotonic() + .1))
        self.assertEqual(caught.exception.code, "deadline")
        self.assertIsNone(self.adapter.process)

    def assert_probe_children_reaped(self, record):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.assertTrue(record.is_file(), "Probe never spawned the stdout-holding child")
        for pid in json.loads(record.read_text()).values():
            handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0, f"Probe process {pid} survived cleanup")
                finally:
                    kernel.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Windows Job descendant ownership")
    def test_inherited_preflight_stdout_obeys_deadline_and_cancellation_and_reaps_tree(self):
        self.adapter.command_prefix.append("--fake-hold-version-pipe")
        for code in ("deadline", "cancelled"):
            with self.subTest(code=code):
                record = self.root / (code + "-probe.json")
                started = self.root / (code + "-worker-started")
                cancel = threading.Event()
                begin = time.monotonic()
                request = self.request(deadline=begin + 10)
                stop_watcher = threading.Event()
                trigger_times = []
                ready_before_interrupt = []
                def interrupt_after_probe_ready():
                    ready_limit = time.monotonic() + 5
                    while not record.is_file() and time.monotonic() < ready_limit:
                        if stop_watcher.wait(.01):
                            return
                    ready = record.is_file()
                    if ready and stop_watcher.wait(.05):
                        return
                    ready_before_interrupt.append(ready)
                    trigger_times.append(time.monotonic())
                    if ready and code == "deadline":
                        request.deadline = trigger_times[-1] + .2
                    else:
                        cancel.set()
                watcher = threading.Thread(target=interrupt_after_probe_ready, daemon=True)
                watcher.start()
                try:
                    with patch.object(os, 'environ', {**os.environ, "BOVARY_FAKE_PROBE_RECORD": str(record),
                                                 "BOVARY_FAKE_WORKER_STARTED": str(started)}):
                        with self.assertRaises(ExecutionError) as caught:
                            self.adapter.execute(request, cancel=cancel)
                finally:
                    stop_watcher.set()
                    watcher.join(timeout=1)
                self.assertEqual(ready_before_interrupt, [True], "Probe did not spawn its stdout-holding child before interruption")
                self.assertEqual(caught.exception.code, code)
                self.assertLess(time.monotonic() - trigger_times[0], 2)
                self.assertIsNone(caught.exception.attempt)
                self.assertIsNone(self.adapter.process)
                self.assertFalse(started.exists())
                self.assertFalse((self.root / "work/benchmark.lock").exists())
                self.assert_probe_children_reaped(record)

    @unittest.skipUnless(os.name == "nt", "Windows Job descendant ownership")
    def test_oversized_preflight_marker_is_captured_with_a_small_bound_and_tree_reaped(self):
        self.adapter.command_prefix += ["--fake-hold-version-pipe", "--fake-oversized-version"]
        record = self.root / "oversized-probe.json"
        started = self.root / "oversized-worker-started"
        popen = subprocess.Popen
        reads = []
        class Reader:
            def __init__(self, stream): self.stream = stream
            def read1(self, size):
                data = self.stream.read1(size)
                reads.append((size, len(data), time.monotonic()))
                return data
            def close(self): self.stream.close()
        def tracked_popen(*args, **kwargs):
            process = popen(*args, **kwargs)
            process.stdout = Reader(process.stdout)
            return process
        with patch.object(os, 'environ', {**os.environ, "BOVARY_FAKE_PROBE_RECORD": str(record),
                                     "BOVARY_FAKE_WORKER_STARTED": str(started)}), \
                patch("bovary.local.subprocess.Popen", side_effect=tracked_popen):
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request())
        returned_at = time.monotonic()
        self.assert_probe_children_reaped(record)
        self.assertEqual(caught.exception.code, "backend_version")
        self.assertIn("incompatible", str(caught.exception))
        self.assertTrue(reads)
        self.assertLessEqual(max(size for size, _, _ in reads), 128)
        captured = sum(count for _, count, _ in reads)
        self.assertGreater(captured, len(b"bovary-worker 2\r\n"))
        self.assertLessEqual(captured, 128)
        # The final read observes the excess byte. Cold interpreter startup
        # precedes that observation and has its own preflight deadline.
        self.assertLess(returned_at - reads[-1][2], 2)
        self.assertFalse(started.exists())

    @unittest.skipUnless(os.name == "nt", "Windows suspended launch ownership")
    def test_ownership_or_resume_failure_terminates_suspended_probe_without_executing_it(self):
        popen = subprocess.Popen
        for failure in ("assignment", "resume"):
            with self.subTest(failure=failure):
                children = []
                started = self.root / (failure + "-started")
                def tracked_popen(*args, **kwargs):
                    process = popen(*args, **kwargs)
                    children.append(process)
                    return process
                context = (patch("bovary.local._WindowsJob", side_effect=OSError("assignment fixture")) if failure == "assignment"
                           else patch.object(local._WindowsJob, "resume", side_effect=OSError("resume fixture")))
                with patch.object(os, 'environ', {**os.environ, "BOVARY_FAKE_WORKER_STARTED": str(started)}), \
                        patch("bovary.local.subprocess.Popen", side_effect=tracked_popen), context:
                    with self.assertRaises(ExecutionError) as caught:
                        self.adapter.execute(self.request())
                self.assertEqual(caught.exception.code, "backend_version")
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertFalse(started.exists())
                self.assertIsNone(self.adapter.process)
                lock = SerialLock(self.root / "work/benchmark.lock")
                lock.acquire()
                lock.release()

    def test_interrupt_after_preflight_cannot_start_a_worker(self):
        def expires(command, environment, request, cancel):
            request.deadline = time.monotonic() - 1
        with patch.object(self.adapter, "_preflight", side_effect=expires), \
                patch.object(self.adapter, "_start") as start:
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "deadline")
        start.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows suspended launch ownership")
    def test_worker_ownership_failure_reaps_child_preserves_attempt_and_releases_lock(self):
        popen = subprocess.Popen
        for failure in ("assignment", "resume"):
            with self.subTest(failure=failure):
                children = []
                started = self.root / (failure + "-worker-started")
                def tracked_popen(*args, **kwargs):
                    process = popen(*args, **kwargs)
                    children.append(process)
                    return process
                context = (patch("bovary.local._WindowsJob", side_effect=OSError("assignment fixture")) if failure == "assignment"
                           else patch.object(local._WindowsJob, "resume", side_effect=OSError("resume fixture")))
                with patch.object(os, 'environ', {**os.environ, "BOVARY_FAKE_WORKER_STARTED": str(started)}), \
                        patch.object(self.adapter, "_preflight"), \
                        patch("bovary.local.subprocess.Popen", side_effect=tracked_popen), context:
                    with self.assertRaises(ExecutionError) as caught:
                        self.adapter.execute(self.request())
                self.assertEqual(caught.exception.code, "backend_unavailable")
                self.assertIsNotNone(caught.exception.attempt)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertFalse(started.exists())
                self.assertIsNone(self.adapter.process)
                lock = SerialLock(self.root / "work/benchmark.lock")
                lock.acquire()
                lock.release()

    def test_interrupt_after_lock_acquisition_is_checked_before_worker_popen(self):
        request = self.request()
        def expires(): request.deadline = time.monotonic() - 1
        with patch.object(self.adapter._lock, "acquire", side_effect=expires), \
                patch("bovary.local.subprocess.Popen") as popen:
            with self.assertRaises(ExecutionError) as caught:
                self.adapter._start([], {}, (), {}, request, None)
        self.assertEqual(caught.exception.code, "deadline")
        popen.assert_not_called()

    def test_fake_cuda_version_probe_omits_cpu_guard(self):
        self.adapter.command_prefix.append("--fake-cuda-only")
        request = self.request(local_profile="balanced", allow_gpu=True)
        result = self.adapter.execute(request)
        self.assertEqual(result.configuration, "balanced-ctx64-threads1-gemma4-text-v1")

    @unittest.skipUnless(os.name == "nt", "Windows worker ownership")
    def test_forced_parent_exit_kills_owned_fake_worker(self):
        import ctypes
        from ctypes import wintypes
        ready = self.root / "parent-ready.json"
        script = (
            "import json,os,sys,time; from pathlib import Path; "
            "from bovary.local import LocalAdapter; from bovary.contract import Request; "
            "a=LocalAdapter(sys.argv[1],command_prefix=[sys._base_executable,'-S',sys.argv[2]]); "
            "result=a.execute(Request('hello',context_tokens=64,output_tokens=3,threads=1)); "
            "pids={'parent':os.getpid(),'owned':a.process.pid,'worker':result.timing['native']['pid']}; "
            "ready=Path(sys.argv[3]); pending=ready.with_suffix('.tmp'); "
            "pending.write_text(json.dumps(pids)); pending.replace(ready); time.sleep(30)"
        )
        process = subprocess.Popen([sys._base_executable, "-S", "-c", script, str(self.root),
                                    str(ROOT / "tests/fixtures/fake_worker.py"), str(ready)],
                                   env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        child = None
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.02)
            self.assertTrue(ready.exists(), "Fake parent did not become ready")
            pids = json.loads(ready.read_text())
            self.assertEqual(pids["parent"], process.pid)
            self.assertEqual(pids["worker"], pids["owned"])
            child = kernel.OpenProcess(0x100001, False, pids["worker"])
            self.assertTrue(child)
            process.terminate()  # Bypass Python cleanup to test OS ownership.
            process.wait(timeout=5)
            self.assertEqual(kernel.WaitForSingleObject(child, 3000), 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            process.stderr.close()
            if child:
                if kernel.WaitForSingleObject(child, 0) != 0:
                    kernel.TerminateProcess(child, 1)
                kernel.CloseHandle(child)


if __name__ == "__main__":
    unittest.main()
