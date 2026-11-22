"""Lossless CLI input and output on Windows code pages and embedded streams."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from bovary import cli
from bovary.contract import Event, Result
from bovary.local import LocalAdapter
from bovary.routing import Workbench

TEXT = '\u05e9\u05dc\u05d5\u05dd caf\u00e9 \U0001f9ea'


class TextSink:
    """An embedded stream that deliberately has no encoding/buffer methods."""
    def __init__(self):
        self.parts = []

    def write(self, value):
        self.parts.append(value)
        return len(value)

    def flush(self):
        pass

    def getvalue(self):
        return ''.join(self.parts)


class Adapter:
    def __init__(self):
        self.unloaded = False
        self.cancelled = False

    def execute(self, request, on_event, cancel):
        on_event(Event('text', request.request_id, text=TEXT))
        self.cancelled = cancel.is_set()
        return Result(request.request_id, 'local', text=TEXT, model=TEXT)

    def unload(self):
        self.unloaded = True


class UnicodeOutputTests(unittest.TestCase):
    def test_interactive_plain_lines_unload_and_quit(self):
        from unittest.mock import Mock
        adapter = Mock(wraps=Adapter())
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, 'create_workbench', return_value=Workbench(local=adapter)), \
                patch.object(sys, 'stdin', io.StringIO(f'\nfirst\n/unload\n{TEXT}\n/quit\nignored\n')), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(cli.main(['--interactive']), 0)
        requests = [call.args[0] for call in adapter.execute.call_args_list]
        self.assertEqual([request.prompt for request in requests], ['first', TEXT])
        self.assertTrue(all(request.offline and request.private for request in requests))
        self.assertEqual(adapter.unload.call_count, 2)
        self.assertIn('Model unloaded.', stdout.getvalue())
        self.assertEqual(stdout.getvalue().count(TEXT), 2)

    def invoke(self, stdout, stderr, *, json_output=False, adapter=None):
        adapter = adapter or Adapter()
        with patch.object(cli, 'create_workbench', return_value=Workbench(local=adapter)), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(['--prompt', TEXT, *(['--json'] if json_output else [])])
        return code, adapter

    def test_redirected_jsonl_decodes_utf8_on_a_legacy_code_page(self):
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python'), 'PYTHONIOENCODING': 'cp1252'}
        payload = json.dumps({'prompt': 'classify', 'workload': TEXT}, ensure_ascii=False) + '\n'
        for encoding in ('utf-8', 'utf-8-sig'):
            with self.subTest(encoding=encoding):
                result = subprocess.run([sys.executable, '-m', 'bovary', '--session', '--mode', 'suggest'],
                    input=payload.encode(encoding), capture_output=True, env=env, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['validation']['decision']['workload'], TEXT)

        invalid = subprocess.run([sys.executable, '-m', 'bovary', '--session', '--mode', 'suggest'],
            input=b'{"prompt":"caf\xe9"}\n', capture_output=True, env=env, timeout=20)
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertEqual(invalid.stdout, b'')
        self.assertEqual(json.loads(invalid.stderr)['status'], 'failed')

    def test_stringio_and_minimal_custom_streams(self):
        for factory in (io.StringIO, TextSink):
            for json_output in (False, True):
                with self.subTest(factory=factory, json=json_output):
                    stdout, stderr = factory(), factory()
                    code, adapter = self.invoke(stdout, stderr, json_output=json_output)
                    self.assertEqual(code, 0)
                    self.assertTrue(adapter.unloaded)
                    if json_output:
                        self.assertEqual(json.loads(stdout.getvalue())['text'], TEXT)
                        self.assertEqual(stderr.getvalue(), TEXT)
                    else:
                        self.assertIn(TEXT, stdout.getvalue())
                        self.assertEqual(json.loads(stderr.getvalue())['model'], TEXT)

    def test_cp1252_redirected_files_are_utf8_and_jsonl_stays_ascii(self):
        with tempfile.TemporaryDirectory() as directory:
            for json_output in (False, True):
                with self.subTest(json=json_output):
                    out_path, err_path = Path(directory)/'out', Path(directory)/'err'
                    with out_path.open('w', encoding='cp1252') as stdout, err_path.open('w', encoding='cp1252') as stderr:
                        code, _ = self.invoke(stdout, stderr, json_output=json_output)
                    self.assertEqual(code, 0)
                    if json_output:
                        result = json.loads(out_path.read_bytes().decode('ascii'))
                        self.assertEqual(result['text'], TEXT)
                        self.assertEqual(err_path.read_text(encoding='utf-8'), TEXT)
                    else:
                        self.assertIn(TEXT, out_path.read_text(encoding='utf-8'))
                        self.assertEqual(json.loads(err_path.read_text(encoding='utf-8'))['model'], TEXT)

    def test_actual_local_worker_streams_unicode_without_callback_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root/'work/models'
            models.mkdir(parents=True)
            (models/'fixture.gguf').write_bytes(b'fixture')
            (models/'model-provenance.json').write_text(json.dumps({
                'filename': 'fixture.gguf', 'bytes': 7, 'sha256': 'a'*64, 'verified': True}))
            for json_output in (False, True):
                with self.subTest(json=json_output):
                    adapter = LocalAdapter(root, command_prefix=[sys.executable, str(ROOT/'tests/fixtures/fake_worker.py')])
                    out_data, err_data = io.BytesIO(), io.BytesIO()
                    stdout = io.TextIOWrapper(out_data, encoding='cp1252')
                    stderr = io.TextIOWrapper(err_data, encoding='cp1252')
                    try:
                        code, _ = self.invoke(stdout, stderr, json_output=json_output, adapter=adapter)
                        self.assertEqual(code, 0)
                        self.assertIsNone(adapter.process)
                        answer = 'reply:' + TEXT + ' \u4e16\u754c'
                        if json_output:
                            self.assertEqual(json.loads(out_data.getvalue().decode('ascii'))['text'], answer)
                            self.assertEqual(err_data.getvalue().decode('utf-8'), answer)
                        else:
                            self.assertIn(answer, out_data.getvalue().decode('utf-8'))
                        class BrokenSink(TextSink):
                            def write(self, value):
                                raise BrokenPipeError('native output reader closed')
                        with self.assertRaisesRegex(BrokenPipeError, 'native output reader closed'):
                            self.invoke(io.StringIO() if json_output else BrokenSink(),
                                BrokenSink() if json_output else io.StringIO(),
                                json_output=json_output, adapter=adapter)
                        self.assertIsNone(adapter.process)
                    finally:
                        adapter.unload()
                        stdout.close()
                        stderr.close()

    def test_streaming_broken_pipe_is_preserved_and_cancels_and_unloads(self):
        class BrokenSink(TextSink):
            def write(self, value):
                raise BrokenPipeError('reader closed')
        for json_output in (False, True):
            with self.subTest(json=json_output):
                adapter = Adapter()
                stdout = io.StringIO() if json_output else BrokenSink()
                stderr = BrokenSink() if json_output else io.StringIO()
                with self.assertRaisesRegex(BrokenPipeError, 'reader closed'):
                    self.invoke(stdout, stderr, json_output=json_output, adapter=adapter)
                self.assertTrue(adapter.cancelled)
                self.assertTrue(adapter.unloaded)


if __name__ == '__main__':
    unittest.main()
