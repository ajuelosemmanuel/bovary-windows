"""Worker boundary regressions using a tiny process fixture, never a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bovary.contract import ExecutionError, Request
from bovary.local import LocalAdapter, SerialLock


FIXTURE = r'''
import sys
VERSION = "version-one"
if "--protocol-version" in sys.argv:
    print("bovary-worker 2",flush=True)
    sys.exit(0)
import json,os,time
if os.name == "nt":
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)
if "--die-before-ready" in sys.argv:
    sys.exit(9)
def frame(kind,data):
    if isinstance(data,dict): data=json.dumps(data).encode()
    sys.stdout.buffer.write(f"{kind} {len(data)}\n".encode()+data)
    sys.stdout.buffer.flush()
if "--oversized-ready-header" in sys.argv:
    sys.stdout.buffer.write(b"R 4097\n")
    sys.stdout.buffer.flush()
    time.sleep(30)
frame("R",{"protocol":2})
context=int(sys.argv[sys.argv.index("--context")+1])
if "--stall-input" in sys.argv:
    time.sleep(30)
for header in sys.stdin.buffer:
    _,count,size=header.split()
    requested=int(count)
    limit=requested or context-4
    text=sys.stdin.buffer.read(int(size)).decode()
    if text in ("T_HEADER_ONLY","E_HEADER_ONLY","D_HEADER_ONLY"):
        declared={"T_HEADER_ONLY":b"T 1025\n","E_HEADER_ONLY":b"E 16385\n","D_HEADER_ONLY":b"D 33554433\n"}
        sys.stdout.buffer.write(declared[text])
        sys.stdout.buffer.flush()
        time.sleep(30)
    if text=="FLOOD":
        for _ in range((requested or context-1)+3): frame("T",b"x")
    elif text=="OVERLONG":
        frame("T",b"x"*2048)
    elif text=="EOG_AT_LIMIT":
        for _ in range(limit): frame("T",b"x")
    else:
        frame("T",VERSION.encode())
    if text=="PARTIALCRASH":
        sys.exit(9)
    metrics={"prompt_tokens":4,"actual_generated_tokens":1,"early_stop":True,
             "requested_generated_tokens":requested,"generation_limit":limit,
             "finish_reason":"eos","pid":os.getpid()}
    changes={"MISMATCH":{"actual_generated_tokens":0},
             "BADCONTEXT":{"prompt_tokens":99999},
             "ZERO_LIMIT":{"generation_limit":0},
             "NEGATIVE_LIMIT":{"generation_limit":-1},
             "BOOL_LIMIT":{"generation_limit":True},
             "WRONG_LIMIT":{"generation_limit":limit+1},
             "WRONG_REQUESTED":{"requested_generated_tokens":requested+1},
             "BOOL_REQUESTED":{"requested_generated_tokens":True},
             "WRONG_FINISH":{"finish_reason":"token_limit" if requested==0 else "context_limit"},
             "EOS_WITHOUT_EOG":{"early_stop":False},
             "LIMIT_BEFORE_BOUND":{"early_stop":False,"finish_reason":"context_limit" if requested==0 else "token_limit"},
             "EOG_AT_LIMIT":{"actual_generated_tokens":limit}}
    metrics.update(changes.get(text,{}))
    if text=="NONFINITE":
        metrics["total_s"]=float('nan')
    if text=="MISSING_LIMIT": del metrics["generation_limit"]
    if text=="MISSING_FINISH": del metrics["finish_reason"]
    frame("D",metrics)
'''


class WorkerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        models = self.root / "work/models"
        models.mkdir(parents=True)
        (models / "fixture.gguf").write_bytes(b"fixture")
        (models / "model-provenance.json").write_text(json.dumps({
            "filename": "fixture.gguf", "bytes": 7, "sha256": "b" * 64, "verified": True}))
        self.script = self.root / "edge_worker.py"
        self.script.write_text(FIXTURE)
        # These stdlib-only fixtures must own the actual interpreter and its pipes.
        self.adapter = LocalAdapter(self.root, command_prefix=[sys._base_executable, "-S", str(self.script)])

    def tearDown(self):
        self.adapter.unload()
        self.temporary.cleanup()

    def request(self, prompt="hello", **changes):
        return Request(prompt, **{"context_tokens": 64, "output_tokens": 3, "threads": 1, **changes})

    def test_closed_worker_stdin_preserves_failure_attempt_and_finishes_cleanup(self):
        # Finish one normal request only after closing the child's stdin. The
        # second request then measures rejection from a known closed pipe.
        self.script.write_text(FIXTURE.replace('    frame("D",metrics)',
            '    os.close(0)\n    frame("D",metrics)\n    time.sleep(5)'))
        self.assertEqual(self.adapter.execute(self.request()).text, "version-one")
        process = self.adapter.process
        readers = list(self.adapter._threads)
        began = time.monotonic()
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(deadline=began + 5))
        self.assertEqual(caught.exception.code, "worker_crashed")
        self.assertLess(time.monotonic() - began, 3, "Closed input must fail before the worker's five-second sleep ends")
        self.assertIsNotNone(caught.exception.attempt)
        self.assertEqual(caught.exception.attempt["destination"], "local")
        self.assertFalse(self.adapter.loaded)
        self.assertIsNotNone(process.poll())
        self.assertTrue(all(pipe.closed for pipe in (process.stdin, process.stdout, process.stderr)))
        self.assertTrue(all(not reader.is_alive() for reader in readers))
        lock = SerialLock(self.root / "work/benchmark.lock")
        lock.acquire()
        lock.release()
        self.script.write_text(FIXTURE)
        result = self.adapter.execute(self.request())
        self.assertEqual(result.text, "version-one")

    def test_generation_limits_and_termination_metadata_are_validated(self):
        prompts = ("ZERO_LIMIT", "NEGATIVE_LIMIT", "BOOL_LIMIT", "WRONG_LIMIT", "WRONG_REQUESTED",
                   "BOOL_REQUESTED", "MISSING_LIMIT", "MISSING_FINISH", "WRONG_FINISH",
                   "EOS_WITHOUT_EOG", "LIMIT_BEFORE_BOUND", "EOG_AT_LIMIT")
        for output_tokens in (3, None):
            for prompt in prompts:
                with self.subTest(prompt=prompt, output_tokens=output_tokens):
                    with self.assertRaises(ExecutionError) as caught:
                        self.adapter.execute(self.request(prompt, output_tokens=output_tokens, deadline=time.monotonic() + 5))
                    self.assertEqual(caught.exception.code, "protocol")
                    self.assertFalse(self.adapter.loaded)
                    self.assertIsNotNone(caught.exception.attempt)

    def test_until_finished_stream_remains_bounded_before_final_metadata(self):
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("FLOOD", output_tokens=None, deadline=time.monotonic() + 5))
        self.assertEqual(caught.exception.code, "protocol")
        self.assertFalse(self.adapter.loaded)
        self.assertEqual(len(caught.exception.attempt["text"]), 63)
        self.assertEqual(caught.exception.attempt["usage"]["emitted_tokens"], 63)

    def test_replacing_same_worker_path_invalidates_warm_process(self):
        first = self.adapter.execute(self.request())
        old_process = self.adapter.process
        stamp = self.script.stat().st_mtime_ns
        self.script.write_text(FIXTURE.replace("version-one", "version-two"))
        os.utime(self.script, ns=(stamp + 2000000000, stamp + 2000000000))
        second = self.adapter.execute(self.request())
        self.assertFalse(second.timing["warm"], "Changed worker file must not reuse the old executable")
        self.assertEqual(second.text, "version-two")
        self.assertNotEqual(first.timing["native"]["pid"], second.timing["native"]["pid"])
        self.assertIsNotNone(old_process.poll())

    def test_text_frames_cannot_exceed_requested_output_bound(self):
        for prompt in ("FLOOD", "OVERLONG"):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ExecutionError) as caught:
                    self.adapter.execute(self.request(prompt))
                self.assertEqual(caught.exception.code, "protocol")
                self.assertFalse(self.adapter.loaded)
                self.assertIsNotNone(caught.exception.attempt)
                self.assertLessEqual(len(caught.exception.attempt["text"]), 3)
                self.assertIsNone(caught.exception.attempt["usage"]["output_tokens"])

    def test_reported_prompt_usage_must_fit_context(self):
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("BADCONTEXT"))
        self.assertEqual(caught.exception.code, "protocol")
        self.assertFalse(self.adapter.loaded)

    def test_oversized_advertised_frames_fail_without_waiting_for_any_body(self):
        for prompt in ("T_HEADER_ONLY", "E_HEADER_ONLY", "D_HEADER_ONLY"):
            with self.subTest(prompt=prompt):
                self.adapter.execute(self.request())
                process = self.adapter.process
                self.assertEqual(self.adapter._frames.maxsize, 4)
                with self.assertRaises(ExecutionError) as caught:
                    self.adapter.execute(self.request(prompt, deadline=time.monotonic() + 3))
                self.assertEqual(caught.exception.code, "protocol")
                self.assertIsNotNone(process.poll())
                self.assertFalse(self.adapter.loaded)

    def test_oversized_ready_header_fails_before_reading_body(self):
        self.adapter.command_prefix.append("--oversized-ready-header")
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request(deadline=time.monotonic() + 3))
        self.assertEqual(caught.exception.code, "protocol")
        self.assertIsNotNone(caught.exception.attempt)
        self.assertFalse(self.adapter.loaded)

    def test_relative_runtime_root_is_resolved_before_child_cwd(self):
        relative_root = Path(os.path.relpath(self.root, Path.cwd()))
        adapter = LocalAdapter(relative_root, command_prefix=[sys.executable, str(self.script)])
        try:
            self.assertEqual(adapter.root, self.root.resolve())
        finally:
            adapter.unload()

    def test_relative_explicit_paths_use_configured_home(self):
        adapter = LocalAdapter(self.root, model="work/models/fixture.gguf",
                               manifest="work/models/model-provenance.json", executable="native/worker.exe")
        try:
            self.assertEqual(adapter.model_path, self.root / "work/models/fixture.gguf")
            self.assertEqual(adapter.manifest_path, self.root / "work/models/model-provenance.json")
            self.assertEqual(adapter.exe, self.root / "native/worker.exe")
        finally:
            adapter.unload()

    def test_malformed_usage_and_nonfinite_metadata_reap_and_recover(self):
        for prompt in ("MISMATCH", "NONFINITE"):
            with self.subTest(prompt=prompt):
                self.adapter.execute(self.request())
                process = self.adapter.process
                with self.assertRaises(ExecutionError) as caught:
                    self.adapter.execute(self.request(prompt))
                self.assertEqual(caught.exception.code, "protocol")
                self.assertIsNotNone(process.poll())
                self.assertFalse(self.adapter.loaded)
                self.assertEqual(self.adapter.execute(self.request()).text, "version-one")

    def test_failure_attempt_retains_partial_text_without_inventing_usage(self):
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request("PARTIALCRASH"))
        error = caught.exception
        self.assertEqual(error.code, "worker_crashed")
        self.assertIsNotNone(error.attempt)
        self.assertEqual(error.attempt["text"], "version-one")
        self.assertEqual(error.attempt["usage"]["emitted_tokens"], 1)
        self.assertIsNone(error.attempt["usage"]["input_tokens"])
        self.assertIsNone(error.attempt["usage"]["output_tokens"])
        self.assertIsNone(error.attempt["cost_usd"])
        self.assertEqual(error.attempt["error"]["code"], "worker_crashed")
        self.assertFalse(self.adapter.loaded)

    def test_worker_startup_failure_is_retained_as_an_actual_attempt(self):
        self.adapter.command_prefix.append("--die-before-ready")
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "worker_crashed")
        self.assertIsNotNone(caught.exception.attempt)
        self.assertEqual(caught.exception.attempt["text"], "")
        self.assertIsNone(caught.exception.attempt["usage"]["input_tokens"])
        self.assertIsNone(caught.exception.attempt["timing"]["startup_s"])
        self.assertFalse(self.adapter.loaded)

    def test_preflight_and_busy_lock_are_not_native_attempts(self):
        self.adapter.command_prefix = [sys.executable, "-c", "raise SystemExit(1)"]
        with self.assertRaises(ExecutionError) as caught:
            self.adapter.execute(self.request())
        self.assertEqual(caught.exception.code, "backend_version")
        self.assertIsNone(caught.exception.attempt)
        self.adapter.command_prefix = [sys.executable, str(self.script)]
        lock = SerialLock(self.root / "work/benchmark.lock")
        lock.acquire()
        try:
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request())
            self.assertEqual(caught.exception.code, "busy")
            self.assertIsNone(caught.exception.attempt)
        finally:
            lock.release()
        self.assertIsNone(self.adapter.process)

    def test_benchmark_lock_and_worker_lock_exclude_each_other(self):
        sys.path.insert(0, str(ROOT))
        from tools.benchmark import SerialLock as BenchmarkLock
        path = self.root / "work/benchmark.lock"
        self.adapter.execute(self.request())
        with self.assertRaises((OSError, RuntimeError)):
            with BenchmarkLock(path):
                self.fail("Benchmark acquired an active warm-worker lock")
        self.adapter.unload()
        with BenchmarkLock(path):
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request())
            self.assertEqual(caught.exception.code, "busy")
            self.assertIsNone(caught.exception.attempt)

    def test_separate_benchmark_process_holding_lock_reports_busy_without_attempt(self):
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from tools.benchmark import SerialLock\n"
            "with SerialLock(sys.argv[2]):\n"
            "    Path(sys.argv[3]).write_text('ready')\n"
            "    sys.stdin.buffer.read(1)\n"
        )
        ready = self.root / "benchmark-lock-ready"
        process = subprocess.Popen([sys._base_executable, "-S", "-c", script, str(ROOT),
                                    str(self.root / "work/benchmark.lock"), str(ready)],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists())
            with self.assertRaises(ExecutionError) as caught:
                self.adapter.execute(self.request())
            self.assertEqual(caught.exception.code, "busy")
            self.assertIsNone(caught.exception.attempt)
        finally:
            try:
                _, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
                raise
        self.assertEqual(process.returncode, 0, stderr.decode(errors="replace"))
        self.assertIsNone(self.adapter.process)

    def test_cancel_reaps_worker_blocked_on_large_stdin_write(self):
        self.adapter.command_prefix.append("--stall-input")
        cancel = threading.Event()
        errors = []

        def run():
            try:
                self.adapter.execute(self.request("x" * (1024 * 1024)), cancel=cancel)
            except ExecutionError as error:
                errors.append(error.code)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 10
        while self.adapter.process is None and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        process = self.adapter.process
        try:
            self.assertIsNotNone(process)
            time.sleep(.1)
        finally:
            cancel.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "Cancellation must interrupt a full stdin pipe")
        self.assertEqual(errors, ["cancelled"])
        self.assertIsNotNone(process.poll())
        lock = SerialLock(self.root / "work/benchmark.lock")
        lock.acquire()
        lock.release()


if __name__ == "__main__":
    unittest.main()
