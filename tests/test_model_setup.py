"""Model setup safety using tiny mocked pins; no checkpoint or network access."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import download_model, verify_model
from bovary.contract import ExecutionError
from bovary.doctor import check_model
from bovary.local import LocalAdapter
from bovary import modelproof


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b"tiny fixture weights"
        self.pins = dict(NAME="fixture.gguf", SIZE=len(self.payload), SHA=hashlib.sha256(self.payload).hexdigest())
        self.patch = mock.patch.multiple(verify_model, **self.pins)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.download_pins = mock.patch.multiple(download_model, NAME=self.pins['NAME'], SIZE=self.pins['SIZE'])
        self.download_pins.start()
        self.addCleanup(self.download_pins.stop)
        self.model = self.root/"modèle-模型.gguf"
        self.model.write_bytes(self.payload)
        self.manifest = self.root/"proof.json"

    def test_runtime_and_doctor_require_the_filename_in_the_proof(self):
        verify_model.verify_model(self.model, self.manifest)
        other = self.root / "same-bytes.gguf"
        other.write_bytes(self.payload)
        self.assertEqual(check_model(other, self.manifest)[0]['status'], 'fail')
        adapter = LocalAdapter(self.root, model=other, manifest=self.manifest)
        with self.assertRaisesRegex(ExecutionError, 'manifest names another checkpoint file'):
            adapter._checkpoint()

    def test_existing_unicode_model_verification_is_offline_and_idempotent(self):
        original = self.model.stat()
        with mock.patch('urllib.request.urlopen') as network:
            metadata = verify_model.verify_model(self.model, self.manifest)
            proof = self.manifest.read_bytes()
            timestamp = self.manifest.stat().st_mtime_ns
            self.assertEqual(verify_model.verify_model(self.model, self.manifest), metadata)
        network.assert_not_called()
        self.assertTrue(metadata['verified'])
        self.assertEqual(metadata['filename'], self.model.name)
        self.assertEqual(metadata['sha256'], self.pins['SHA'])
        self.assertEqual(metadata['bytes'], len(self.payload))
        self.assertEqual(self.manifest.read_bytes(), proof)
        self.assertEqual(self.manifest.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertEqual(self.model.stat().st_mtime_ns, original.st_mtime_ns)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_size_and_digest_failures_do_not_create_proof_or_modify_model(self):
        for payload in (b"short", b"x"*len(self.payload)):
            with self.subTest(payload=payload):
                self.model.write_bytes(payload)
                with self.assertRaises(ValueError):
                    verify_model.verify_model(self.model, self.manifest)
                self.assertFalse(self.manifest.exists())
                self.assertEqual(self.model.read_bytes(), payload)

    def test_existing_conflicting_or_invalid_proofs_are_preserved_before_hashing(self):
        good = verify_model.expected_manifest(self.model.name)
        for proof in ('{', '[]', json.dumps({**good, 'verified':False}),
                      json.dumps({**good, 'filename':'other.gguf'}),
                      json.dumps({**good, 'bytes':True}), json.dumps({**good, 'sha256':'f'*64})):
            with self.subTest(proof=proof):
                self.manifest.write_text(proof, encoding='utf-8')
                with mock.patch.object(verify_model, 'verify_file') as hashing, self.assertRaises(ValueError):
                    verify_model.verify_model(self.model, self.manifest)
                hashing.assert_not_called()
                self.assertEqual(self.manifest.read_text(encoding='utf-8'), proof)

    def test_existing_valid_proof_stays_unchanged_when_model_digest_fails(self):
        verify_model.verify_model(self.model, self.manifest)
        before = self.manifest.read_bytes()
        self.model.write_bytes(b"x"*len(self.payload))
        with self.assertRaises(ValueError):
            verify_model.verify_model(self.model, self.manifest)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_bound_proof_rejects_same_size_rewrite_even_with_restored_mtime(self):
        verify_model.verify_model(self.model, self.manifest)
        adapter = LocalAdapter(self.root, model=self.model, manifest=self.manifest)
        original = self.model.stat()
        self.assertEqual(adapter._checkpoint()[0], self.model.resolve())
        self.model.write_bytes(b'x' * len(self.payload))
        os.utime(self.model, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(self.model.stat().st_mtime_ns, original.st_mtime_ns)
        with self.assertRaisesRegex(ExecutionError, 'no longer matches'):
            adapter._checkpoint()
        self.assertEqual(check_model(self.model, self.manifest, verify_hash=True)[-1]['status'], 'fail')

    def test_bound_proof_with_change_marker_does_not_hash_at_runtime(self):
        with mock.patch.object(modelproof, '_windows_file_usn', return_value=42):
            verify_model.verify_model(self.model, self.manifest)
            adapter = LocalAdapter(self.root, model=self.model, manifest=self.manifest)
            with mock.patch.object(hashlib, 'sha256', side_effect=AssertionError('must not rehash')):
                self.assertEqual(adapter._checkpoint()[0], self.model.resolve())
                self.assertEqual(check_model(self.model, self.manifest)[1]['status'], 'pass')

    @unittest.skipUnless(os.name == 'nt', 'Windows change-journal fallback')
    def test_missing_usn_checks_bytes_when_all_timestamps_collide(self):
        with mock.patch.object(modelproof, '_windows_file_usn', return_value=None):
            proof = verify_model.verify_model(self.model, self.manifest)
        identity = proof['file_identity']
        adapter = LocalAdapter(self.root, model=self.model, manifest=self.manifest)
        with mock.patch.object(modelproof, 'file_identity', return_value=identity):
            self.assertEqual(adapter._checkpoint()[0], self.model.resolve())
            self.model.write_bytes(b'x' * len(self.payload))
            with self.assertRaisesRegex(ExecutionError, 'no longer matches'):
                adapter._checkpoint()
            with mock.patch.object(hashlib, 'sha256', side_effect=AssertionError('doctor implicitly hashed')):
                self.assertEqual(check_model(self.model, self.manifest)[1]['status'], 'unknown')

    def test_legacy_gemma_proof_requires_a_new_manifest_and_keeps_old_bytes(self):
        legacy = json.dumps(verify_model.expected_manifest(self.model.name))
        self.manifest.write_text(legacy, encoding='utf-8')
        adapter = LocalAdapter(self.root, model=self.model, manifest=self.manifest)
        with self.assertRaisesRegex(ExecutionError, 'no file binding'):
            adapter._checkpoint()
        self.assertEqual(check_model(self.model, self.manifest)[1]['status'], 'fail')
        with self.assertRaisesRegex(ValueError, 'choose a new manifest path'):
            verify_model.verify_model(self.model, self.manifest)
        self.assertEqual(self.manifest.read_text(encoding='utf-8'), legacy)
        new_manifest = self.root/'fresh-proof.json'
        verify_model.verify_model(self.model, new_manifest)
        self.assertEqual(LocalAdapter(self.root, model=self.model, manifest=new_manifest)._checkpoint()[0],
                         self.model.resolve())

    def test_download_uses_existing_local_bytes_to_publish_bound_proof_at_new_path(self):
        model = self.root/self.pins['NAME']
        model.write_bytes(self.payload)
        old = self.root/'model-provenance.json'
        old.write_text(json.dumps(verify_model.expected_manifest(model.name)), encoding='utf-8')
        output = self.root/'new-proof.json'
        with mock.patch('urllib.request.urlopen') as network, contextlib.redirect_stdout(io.StringIO()):
            download_model.main(['--directory', str(self.root), '--manifest', str(output)])
        network.assert_not_called()
        proof = json.loads(output.read_bytes())
        self.assertEqual(proof['file_identity'], verify_model.path_identity(model))
        self.assertNotIn('file_identity', json.loads(old.read_bytes()))

    def test_file_changed_during_hashing_never_gets_a_proof(self):
        digest_file = hashlib.file_digest
        def hash_then_change(source, algorithm):
            digest = digest_file(source, algorithm)
            self.model.write_bytes(self.payload+b'changed')
            return digest
        with mock.patch.object(hashlib, 'file_digest', side_effect=hash_then_change), self.assertRaisesRegex(ValueError, 'changed'):
            verify_model.verify_model(self.model, self.manifest)
        self.assertFalse(self.manifest.exists())

    def test_concurrent_conflicting_proof_is_not_overwritten_and_temp_is_removed(self):
        conflicting = b'{"owner":"concurrent proof"}'
        def publish(source, destination):
            destination.write_bytes(conflicting)
            raise FileExistsError('fixture concurrent creator')
        with mock.patch.object(verify_model, 'publish_no_replace', side_effect=publish), self.assertRaises(ValueError):
            verify_model.verify_model(self.model, self.manifest)
        self.assertEqual(self.manifest.read_bytes(), conflicting)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_publication_failure_leaves_no_partial_manifest(self):
        with mock.patch.object(verify_model, 'publish_no_replace', side_effect=OSError('fixture publication failure')), self.assertRaises(OSError):
            verify_model.verify_model(self.model, self.manifest)
        self.assertFalse(self.manifest.exists())
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_downloader_reuses_existing_or_complete_partial_without_network(self):
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                destination = directory/self.pins['NAME']
                candidate = destination.with_suffix('.gguf.partial') if partial else destination
                candidate.write_bytes(self.payload)
                with mock.patch('urllib.request.urlopen') as network, contextlib.redirect_stdout(io.StringIO()):
                    download_model.main(['--directory', str(directory)])
                network.assert_not_called()
                self.assertEqual(destination.read_bytes(), self.payload)
                self.assertFalse(destination.with_suffix('.gguf.partial').exists())
                proof = json.loads((directory/'model-provenance.json').read_text())
                self.assertEqual(proof['filename'], destination.name)
                self.assertEqual(proof['sha256'], self.pins['SHA'])
                self.assertEqual(proof['file_identity'], verify_model.path_identity(destination))

    def test_download_link_publication_binds_the_destination_after_unlink(self):
        destination = self.root/self.pins['NAME']
        partial = destination.with_suffix('.gguf.partial')
        partial.write_bytes(self.payload)
        def publish(source, target):
            if os.name == 'nt':
                target.write_bytes(source.read_bytes())
            else:
                os.link(source, target)

        def observe(path):
            self.assertFalse(partial.exists())
            return verify_model.path_identity(path)

        with (mock.patch.object(download_model, 'publish_no_replace', side_effect=publish),
              mock.patch.object(download_model, 'path_identity', side_effect=observe),
              contextlib.redirect_stdout(io.StringIO())):
            download_model.main(['--directory', str(self.root)])
        self.assertFalse(partial.exists())
        proof = json.loads((self.root/'model-provenance.json').read_bytes())
        self.assertEqual(proof['file_identity'], verify_model.path_identity(destination))

    def test_downloader_preserves_corrupt_existing_file_and_conflicting_proof(self):
        destination = self.root/self.pins['NAME']
        destination.write_bytes(b"x"*len(self.payload))
        with mock.patch('urllib.request.urlopen') as network, self.assertRaises(ValueError):
            download_model.main(['--directory', str(self.root)])
        network.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"x"*len(self.payload))
        proof = self.root/'model-provenance.json'
        proof.write_text('{"verified":false}', encoding='utf-8')
        with mock.patch('urllib.request.urlopen') as network, self.assertRaises(ValueError):
            download_model.main(['--directory', str(self.root)])
        network.assert_not_called()
        self.assertEqual(proof.read_text(), '{"verified":false}')

    @staticmethod
    def response(payload, status=206, headers=None):
        stream = io.BytesIO(payload)
        stream.status, stream.headers = status, headers or {}
        return stream

    def test_downloader_validates_full_and_resumed_response_before_publishing(self):
        for have, status, length in ((0, 200, True), (0, 206, True), (4, 206, True), (4, 206, False)):
            with self.subTest(have=have, status=status, length=length), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                partial = (directory/self.pins['NAME']).with_suffix('.gguf.partial')
                if have:
                    partial.write_bytes(self.payload[:have])
                headers = {'Content-Range':f'bytes {have}-{len(self.payload)-1}/{len(self.payload)}'} if status == 206 else {}
                if length:
                    headers['Content-Length'] = str(len(self.payload)-have)
                response = self.response(self.payload[have:], status, headers)
                with mock.patch.object(download_model.urllib.request, 'urlopen', return_value=response) as network, \
                        mock.patch.object(download_model.shutil, 'disk_usage', return_value=mock.Mock(free=10*1024**3)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    download_model.main(['--directory', str(directory)])
                self.assertEqual(network.call_count, 1)
                self.assertEqual(network.call_args.args[0].get_header('Range'), f'bytes={have}-')
                self.assertEqual((directory/self.pins['NAME']).read_bytes(), self.payload)
                self.assertTrue(json.loads((directory/'model-provenance.json').read_text())['verified'])
                self.assertFalse(partial.exists())

    def test_invalid_download_response_preserves_resumable_prefix_without_retry_or_publication(self):
        total = len(self.payload)
        good_range = f'bytes 4-{total-1}/{total}'
        cases = [
            (206, {}),
            (206, {'Content-Range':f'bytes 0-{total-1}/{total}'}),
            (206, {'Content-Range':f'bytes 4-{total-2}/{total}'}),
            (206, {'Content-Range':f'bytes 4-{total-1}/{total+1}'}),
            (206, {'Content-Range':'bytes */20'}),
            (206, {'Content-Range':good_range+', '+good_range}),
            (206, {'Content-Range':'bytes '+'9'*5000+'-19/20'}),
            (206, {'Content-Range':good_range, 'Content-Length':str(total)}),
            (206, {'Content-Range':good_range, 'Content-Length':'invalid'}),
            (206, {'Content-Range':good_range, 'Content-Length':'9'*5000}),
            (206, {'Content-Range':good_range, 'Content-Encoding':'gzip'}),
            (200, {'Content-Length':str(total)}),
            (201, {'Content-Length':str(total-4)}),
        ]
        for status, headers in cases:
            with self.subTest(status=status, headers=str(headers)[:100]), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                partial = (directory/self.pins['NAME']).with_suffix('.gguf.partial')
                original = self.payload[:4]
                partial.write_bytes(original)
                response = self.response(self.payload[4:], status, headers)
                with mock.patch.object(download_model.urllib.request, 'urlopen', return_value=response) as network, \
                        mock.patch.object(download_model.shutil, 'disk_usage', return_value=mock.Mock(free=10*1024**3)), \
                        mock.patch.object(download_model.time, 'sleep') as sleep, \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(download_model.DownloadProtocolError):
                    download_model.main(['--directory', str(directory)])
                self.assertEqual(network.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(partial.read_bytes(), original)
                self.assertFalse((directory/self.pins['NAME']).exists())
                self.assertFalse((directory/'model-provenance.json').exists())

    def test_initial_download_rejects_declared_wrong_length_without_creating_partial(self):
        response = self.response(self.payload, 200, {'Content-Length':str(len(self.payload)+1)})
        with mock.patch.object(download_model.urllib.request, 'urlopen', return_value=response), \
                mock.patch.object(download_model.shutil, 'disk_usage', return_value=mock.Mock(free=10*1024**3)), \
                self.assertRaises(download_model.DownloadProtocolError):
            download_model.main(['--directory', str(self.root)])
        self.assertFalse((self.root/self.pins['NAME']).with_suffix('.gguf.partial').exists())
        self.assertFalse((self.root/'model-provenance.json').exists())

    def test_download_body_overrun_is_rejected_before_excess_bytes_are_written(self):
        partial = (self.root/self.pins['NAME']).with_suffix('.gguf.partial')
        original = self.payload[:4]
        partial.write_bytes(original)
        headers = {'Content-Range':f'bytes 4-{len(self.payload)-1}/{len(self.payload)}'}
        response = self.response(self.payload[4:]+b'excess', headers=headers)
        with mock.patch.object(download_model.urllib.request, 'urlopen', return_value=response) as network, \
                mock.patch.object(download_model.shutil, 'disk_usage', return_value=mock.Mock(free=10*1024**3)), \
                self.assertRaises(download_model.DownloadProtocolError):
            download_model.main(['--directory', str(self.root)])
        self.assertEqual(network.call_count, 1)
        self.assertEqual(partial.read_bytes(), original)
        self.assertFalse((self.root/self.pins['NAME']).exists())
        self.assertFalse((self.root/'model-provenance.json').exists())

    def test_short_valid_response_resumes_its_written_prefix_on_retry(self):
        partial = (self.root/self.pins['NAME']).with_suffix('.gguf.partial')
        partial.write_bytes(self.payload[:4])
        total = len(self.payload)
        responses = [self.response(self.payload[4:8], headers={'Content-Range':f'bytes 4-{total-1}/{total}'}),
                     self.response(self.payload[8:], headers={'Content-Range':f'bytes 8-{total-1}/{total}'})]
        with mock.patch.object(download_model.urllib.request, 'urlopen', side_effect=responses) as network, \
                mock.patch.object(download_model.shutil, 'disk_usage', return_value=mock.Mock(free=10*1024**3)), \
                mock.patch.object(download_model.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            download_model.main(['--directory', str(self.root)])
        self.assertEqual([call.args[0].get_header('Range') for call in network.call_args_list], ['bytes=4-', 'bytes=8-'])
        sleep.assert_called_once()
        self.assertEqual((self.root/self.pins['NAME']).read_bytes(), self.payload)
        self.assertTrue((self.root/'model-provenance.json').exists())
        self.assertFalse(partial.exists())

    def test_downloader_refuses_shared_partial_before_writing_existing_model(self):
        partial = (self.root/self.pins['NAME']).with_suffix('.gguf.partial')
        partial.write_bytes(self.payload)
        is_symlink = Path.is_symlink
        def shared_partial(path):
            return path == partial or is_symlink(path)
        with mock.patch.object(Path, 'is_symlink', shared_partial), mock.patch('urllib.request.urlopen') as network, self.assertRaisesRegex(RuntimeError, 'link'):
            download_model.main(['--directory', str(self.root)])
        network.assert_not_called()
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertEqual(partial.read_bytes(), self.payload)

    def test_cli_reports_verified_metadata_and_default_manifest_location(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(verify_model.main(['--model', str(self.model)]), 0)
        self.assertEqual(json.loads(output.getvalue())['filename'], self.model.name)
        self.assertTrue((self.root/'model-provenance.json').exists())

    def test_cli_unicode_metadata_is_ascii_safe_on_cp1252_stdout(self):
        raw = io.BytesIO()
        output = io.TextIOWrapper(raw, encoding='cp1252')
        try:
            with contextlib.redirect_stdout(output):
                self.assertEqual(verify_model.main(['--model', str(self.model), '--manifest', str(self.manifest)]), 0)
            output.flush()
            serialized = raw.getvalue()
            self.assertTrue(serialized.isascii())
            self.assertEqual(json.loads(serialized)['filename'], self.model.name)
            self.assertEqual(json.loads(self.manifest.read_text(encoding='utf-8'))['filename'], self.model.name)
            self.assertIn(self.model.name.encode('utf-8'), self.manifest.read_bytes())
        finally:
            output.close()

    def test_second_process_is_rejected_before_proof_read_network_or_partial_write(self):
        partial = (self.root/self.pins['NAME']).with_suffix('.gguf.partial')
        partial.write_bytes(b'resumable fixture')
        proof = self.root/'model-provenance.json'
        proof.write_bytes(b'preserve existing proof bytes')
        code = r'''
from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1])
from tools import download_model as download
def forbidden(*args,**kwargs):
    raise AssertionError('Second process inspected model state or opened network before acquiring lock')
download.existing_manifest=forbidden
download.urllib.request.urlopen=forbidden
try:
    download.main(['--directory',sys.argv[2]])
except RuntimeError as error:
    if 'Another downloader is active' not in str(error): raise
    print(str(error))
    raise SystemExit(23)
raise AssertionError('Second downloader acquired an owned lock')
'''
        with download_model.DownloadLock(self.root):
            result = subprocess.run([sys.executable, '-c', code, str(ROOT), str(self.root)],
                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn('Another downloader is active', result.stdout)
        self.assertEqual(partial.read_bytes(), b'resumable fixture')
        self.assertEqual(proof.read_bytes(), b'preserve existing proof bytes')
        self.assertTrue((self.root/'.bovary-download.lock').exists())

    def test_process_death_releases_lock_without_deleting_stable_lock_file(self):
        ready = self.root/'lock-ready'
        code = r'''
from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1])
from tools.download_model import DownloadLock
with DownloadLock(sys.argv[2]):
    import os
    print(os.getpid(), flush=True)
    Path(sys.argv[3]).write_text('held')
    sys.stdin.buffer.read(1)
'''
        # On Windows sys.executable may be the venv redirector. Killing/waiting
        # for it does not wait for the interpreter that actually owns the lock.
        process = subprocess.Popen([sys._base_executable, '-c', code, str(ROOT), str(self.root), str(ready)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+5
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(), 'Child did not acquire the lock')
            # The PID was flushed before creating ready, so this pipe read
            # cannot race a Windows rename/file-sharing operation on ready.
            self.assertEqual(int(process.stdout.readline()), process.pid, 'Must kill the actual lock owner')
            lock_path = self.root/'.bovary-download.lock'
            identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
            with self.assertRaisesRegex(RuntimeError, 'Another downloader is active'):
                with download_model.DownloadLock(self.root):
                    self.fail('Parent acquired child-owned lock')
            process.kill()
            process.wait(timeout=5)
            with download_model.DownloadLock(self.root):
                self.assertEqual((lock_path.stat().st_dev, lock_path.stat().st_ino), identity)
            self.assertTrue(lock_path.exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()

    def test_main_holds_lock_through_verification_and_proof_publication(self):
        destination = self.root/self.pins['NAME']
        destination.write_bytes(self.payload)
        observed = []
        originals = {name:getattr(download_model,name) for name in ('existing_manifest','verify_file','write_manifest')}
        def guarded(name):
            def call(*args, **kwargs):
                with self.assertRaisesRegex(RuntimeError, 'Another downloader is active'):
                    with download_model.DownloadLock(self.root):
                        self.fail('Downloader released lock before '+name)
                observed.append(name)
                return originals[name](*args, **kwargs)
            return call
        with mock.patch.multiple(download_model, **{name:guarded(name) for name in originals}), \
                mock.patch('urllib.request.urlopen') as network, contextlib.redirect_stdout(io.StringIO()):
            download_model.main(['--directory',str(self.root)])
        self.assertEqual(observed, ['existing_manifest','verify_file','write_manifest'])
        network.assert_not_called()

    def test_powershell_model_recipes_stop_on_setup_failure_and_planner_uses_venv(self):
        checked = 0
        for name in ('README.md','docs/usage.md','docs/configuration.md','docs/reproducibility.md'):
            lines = (ROOT/name).read_text(encoding='utf-8').splitlines()
            for index, line in enumerate(lines):
                if line.startswith('.\\.venv\\Scripts\\python.exe ') and any(tool in line for tool in ('verify_model.py','download_model.py')):
                    with self.subTest(document=name, command=line):
                        self.assertRegex(lines[index+1], r'^if \(\$LASTEXITCODE -ne 0\) \{ throw .+ \}$')
                    checked += 1
        self.assertGreaterEqual(checked, 5)
        planner = (ROOT/'docs/model-decisions.md').read_text(encoding='utf-8')
        self.assertNotIn('python -m bovary.policy', planner)
        self.assertIn('.\\.venv\\Scripts\\python.exe -m bovary.policy', planner)


if __name__ == '__main__':
    unittest.main()
