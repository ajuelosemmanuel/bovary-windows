"""Synthetic lifecycle checks: no model, hardware backend or network."""
import gc
from contextlib import redirect_stdout
import io
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
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from bovary.contract import ExecutionError, Request
from bovary.launcher_chat import SWITCH_MODEL, chat
from bovary.local import LocalAdapter, _ACTIVE_ADAPTERS, _dispose_active_adapters


class LocalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        models = self.root / 'work/models'
        models.mkdir(parents=True)
        (models / 'fixture.gguf').write_bytes(b'fixture')
        (models / 'model-provenance.json').write_text(json.dumps({
            'filename': 'fixture.gguf', 'bytes': 7, 'sha256': 'a' * 64, 'verified': True}))
        self.adapters = []

    def tearDown(self):
        for adapter in self.adapters:
            adapter.unload()
        self.adapters.clear()
        self.temporary.cleanup()

    def adapter(self):
        adapter = LocalAdapter(self.root, command_prefix=[
            sys.executable, str(ROOT / 'tests/fixtures/fake_worker.py')])
        self.adapters.append(adapter)
        return adapter

    @staticmethod
    def request(prompt='hello'):
        return Request(prompt, context_tokens=64, output_tokens=3, threads=1,
                       deadline=time.monotonic() + 30)

    def test_unused_and_unloaded_adapters_collect_repeatedly(self):
        references = []
        for index in range(30):
            adapter = LocalAdapter(self.root)
            references.append(weakref.ref(adapter))
            if index % 2:
                adapter.unload()
            del adapter
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_loaded_unload_collects_and_releases_process_and_registry(self):
        for _ in range(5):
            adapter = self.adapter()
            adapter.execute(self.request())
            process = adapter.process
            self.assertIn(adapter, _ACTIVE_ADAPTERS)
            reference = weakref.ref(adapter)
            adapter.unload()
            self.assertIsNotNone(process.poll())
            self.assertNotIn(adapter, _ACTIVE_ADAPTERS)
            self.adapters.remove(adapter)
            del adapter
            gc.collect()
            self.assertIsNone(reference())

    def test_launcher_chat_follows_up_and_releases_worker(self):
        adapter = self.adapter()
        responses = []
        execute = adapter.execute

        def record(request, **kwargs):
            result = execute(request, **kwargs)
            responses.append((request.prompt, result.timing["warm"], adapter.process))
            return result

        with patch.object(adapter, "execute", side_effect=record), patch(
            "builtins.input", side_effect=[
                "fresh1", "fresh2", "/keep", "remember apricot", "what did I say?",
                "/new", "new chat", "/unload", "fresh3", "/models",
            ]
        ), redirect_stdout(io.StringIO()), adapter:
            self.assertEqual(chat(adapter, "cpu-only"), SWITCH_MODEL)

        self.assertEqual([warm for _, warm, _ in responses], [False, False, False, True, True, False])
        self.assertEqual([prompt for prompt, _, _ in responses if not prompt.startswith("Continue")],
                         ["fresh1", "fresh2", "remember apricot", "new chat", "fresh3"])
        history = json.loads(responses[3][0].split("\n", 1)[1])
        self.assertEqual(history["latest_user"], "what did I say?")
        self.assertEqual(history["previous_turns"][0]["user"], "remember apricot")
        self.assertIn("reply:remember apricot", history["previous_turns"][0]["assistant"])
        self.assertIs(responses[2][2], responses[4][2])
        self.assertIsNot(responses[4][2], responses[5][2])
        self.assertTrue(all(process.poll() is not None for _, _, process in responses))

    def test_abandoned_active_worker_retained_until_exit_cleanup(self):
        adapter = self.adapter()
        adapter.execute(self.request())
        process = adapter.process
        reference = weakref.ref(adapter)
        self.adapters.remove(adapter)
        del adapter
        gc.collect()
        self.assertIsNotNone(reference())
        _dispose_active_adapters()
        gc.collect()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(reference())

    def test_context_exit_and_close_allow_reuse(self):
        adapter = self.adapter()
        with adapter as entered:
            self.assertIs(entered, adapter)
            adapter.execute(self.request())
            first = adapter.process
        self.assertIsNotNone(first.poll())
        adapter.execute(self.request())
        second = adapter.process
        self.assertIsNot(first, second)
        adapter.close()
        adapter.close()
        self.assertIsNotNone(second.poll())
        self.assertNotIn(adapter, _ACTIVE_ADAPTERS)

    def test_context_exception_disposes_worker(self):
        adapter = self.adapter()
        with self.assertRaisesRegex(ValueError, 'caller failure'):
            with adapter:
                adapter.execute(self.request())
                process = adapter.process
                raise ValueError('caller failure')
        self.assertIsNotNone(process.poll())
        self.assertNotIn(adapter, _ACTIVE_ADAPTERS)

    def test_busy_close_cancellation_and_recovery(self):
        adapter = self.adapter()
        adapter.execute(self.request())
        process = adapter.process
        cancel = threading.Event()
        entered = threading.Event()
        errors = []
        original = adapter._next_frame

        def next_frame(*args, **kwargs):
            entered.set()
            return original(*args, **kwargs)

        adapter._next_frame = next_frame

        def run():
            try:
                adapter.execute(self.request('SILENT'), cancel=cancel)
            except ExecutionError as error:
                errors.append(error.code)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaises(ExecutionError) as caught:
                adapter.close()
            self.assertEqual(caught.exception.code, 'busy')
        finally:
            cancel.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['cancelled'])
        self.assertIsNotNone(process.poll())
        self.assertNotIn(adapter, _ACTIVE_ADAPTERS)
        adapter.execute(self.request())
        self.assertIn(adapter, _ACTIVE_ADAPTERS)

    def test_normal_interpreter_exit_reaps_registered_worker(self):
        script = '''
import sys
from bovary.local import LocalAdapter
from bovary.contract import Request
a = LocalAdapter(sys.argv[1], command_prefix=[sys.executable, sys.argv[2]])
a.execute(Request('hello', context_tokens=64, output_tokens=3, threads=1))
print(a.process.pid, flush=True)
sys.stdin.readline()
'''
        parent = subprocess.Popen([sys.executable, '-c', script, str(self.root),
                                   str(ROOT / 'tests/fixtures/fake_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')})
        handle = None
        try:
            pid = int(parent.stdout.readline())
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000, False, pid)
                self.assertTrue(handle)
            _, errors = parent.communicate('\n', timeout=10)
            self.assertEqual(parent.returncode, 0, errors)
            if handle:
                self.assertEqual(kernel.WaitForSingleObject(handle, 3000), 0)
            else:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.communicate(timeout=5)
            if handle:
                kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
