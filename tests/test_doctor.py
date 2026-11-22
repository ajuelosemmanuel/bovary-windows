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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from bovary import doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'modèle-שלום-😀.gguf'
        self.model.write_bytes(b'tiny synthetic fixture')
        self.manifest = self.root / 'proof.json'
        self.metadata = {'filename': self.model.name, 'bytes': self.model.stat().st_size,
                         'sha256': hashlib.sha256(self.model.read_bytes()).hexdigest(), 'verified': True}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding='utf-8')

    @staticmethod
    def checks(report):
        return {item['id']: item for item in report['checks']}

    def test_default_checks_are_offline_redacted_and_do_not_hash_or_load_model(self):
        with patch.object(doctor, '_probe', return_value=('ok', b'tk-available\n')), \
                patch.object(doctor.hashlib, 'sha256', side_effect=AssertionError('No default hash')), \
                patch('socket.create_connection', side_effect=AssertionError('No network')):
            report = doctor.diagnose(self.root, model_path=self.model, manifest_path=self.manifest)
        checks = self.checks(report)
        self.assertEqual(checks['model_file']['status'], 'pass')
        self.assertEqual(checks['model_manifest']['status'], 'pass')
        self.assertEqual(checks['model_hash']['status'], 'unknown')
        self.assertEqual(checks['native_executable']['status'], 'fail')
        self.assertEqual(report['exit_code'], 1)
        self.assertTrue(report['paths_redacted'])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertNotIn(self.model.name, json.dumps(report, ensure_ascii=False))
        self.assertTrue(all(item['status'] in doctor.STATUSES for item in report['checks']))

    def test_explicit_paths_override_environment_and_relative_paths_use_home(self):
        with patch.object(os, 'environ', {**os.environ, 'BOVARY_HOME':str(self.root / 'ignored'),
                                    'BOVARY_MODEL':'environment.gguf', 'BOVARY_MODEL_MANIFEST':'environment.json',
                                    'BOVARY_CPU_EXE':'environment.exe'}):
            paths = doctor.resolve_paths(self.root, model_path=self.model.name,
                                         manifest_path='proof.json', executable='selected.exe')
            self.assertEqual(paths['model'], self.model)
            self.assertEqual(paths['manifest'], self.manifest)
            self.assertEqual(paths['executable'], self.root / 'selected.exe')
            self.assertEqual(doctor.resolve_paths()['home'], self.root / 'ignored')
            defaults = doctor.resolve_paths(self.root, use_environment=False)
            self.assertEqual(defaults['model'], self.root / 'work/models/gemma-4-26B-A4B-it-Q4_0.gguf')
            self.assertEqual(defaults['manifest'], self.root / 'work/models/model-provenance.json')
            self.assertEqual(defaults['executable'], self.root / 'work/build/bovary-cpu/bovary-infer.exe')

    def test_manifest_default_model_matches_runtime_default_directory(self):
        manifest = self.root / 'work/models/model-provenance.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(self.metadata))
        with patch.object(os, 'environ', {}):
            paths = doctor.resolve_paths(self.root)
        self.assertEqual(paths['model'], self.root / 'work/models' / self.model.name)

    def test_missing_and_malformed_manifest_size_and_filename_mismatches(self):
        for update in ({'filename':'other.gguf'}, {'bytes':True}, {'sha256':'x' * 64}, {'verified':False}):
            with self.subTest(update=update):
                self.manifest.write_text(json.dumps({**self.metadata, **update}))
                checks = doctor.check_model(self.model, self.manifest)
                self.assertEqual(checks[0]['status'], 'fail')
                self.assertIn('verify_model.py', checks[0]['remedy'])
        self.write_manifest()
        self.model.write_bytes(b'changed size')
        self.assertEqual(doctor.check_model(self.model, self.manifest)[1]['status'], 'fail')
        for raw in (b'[]', b'bad-json', b' ' * 65537, b'[' * 1100):
            self.manifest.write_bytes(raw)
            self.assertEqual(doctor.check_model(self.model, self.manifest)[0]['status'], 'fail')

    def test_full_hash_explicit_and_checks_changes_without_writing_proof(self):
        original = self.manifest.read_bytes()
        checks = doctor.check_model(self.model, self.manifest, verify_hash=True, deadline=time.monotonic()+2)
        self.assertEqual(checks[-1]['status'], 'pass')
        self.assertIn('publisher provenance is not independently established', checks[-1]['message'])
        self.model.write_bytes(b'x' * self.metadata['bytes'])
        checks = doctor.check_model(self.model, self.manifest, verify_hash=True, deadline=time.monotonic()+2)
        self.assertEqual(checks[-1]['status'], 'fail')
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_hash_timeout_does_not_claim_success(self):
        checks = doctor.check_model(self.model, self.manifest, verify_hash=True, deadline=time.monotonic()-1)
        self.assertEqual(checks[-1]['status'], 'unknown')
        self.assertIn('timeout', checks[-1]['message'])

    def test_real_protocol_probe_is_cpu_guarded_bounded_and_reaped(self):
        executable = self.root / 'native.exe'
        executable.write_bytes(b'fixture-not-executed')
        observed = []
        real_probe = doctor._probe
        def substituted(command, **kwargs):
            observed.append(command)
            if Path(command[0]) == executable:
                return real_probe([sys.executable, str(ROOT/'tests/fixtures/fake_worker.py'), *command[1:]], **kwargs)
            return real_probe(command, **kwargs)
        with patch.object(doctor, '_probe', side_effect=substituted):
            report = doctor.diagnose(self.root, model_path=self.model, manifest_path=self.manifest, executable=executable)
        self.assertEqual(self.checks(report)['native_protocol']['status'], 'pass')
        self.assertEqual(Path(observed[1][0]), executable)
        self.assertEqual(observed[1][1:], ['--require-cpu-only', '--protocol-version'])
        self.assertFalse(any('--model' in command or '--worker' in command for command in observed))

    def test_probe_timeout_and_inherited_pipe_cleanup(self):
        pid_file = self.root / 'pid'
        command = [sys._base_executable, '-c',
                   'import subprocess,sys; child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
                   'open(sys.argv[1],"w").write(str(child.pid))', str(pid_file)]
        started = time.monotonic()
        status, output = doctor._probe(command, deadline=started+5)
        self.assertEqual(status, 'timeout')
        self.assertEqual(output, b'')
        self.assertLess(time.monotonic()-started, 8)
        self.assertTrue(pid_file.exists(), 'Grandchild must have started for inherited-pipe regression')
        pid = int(pid_file.read_text())
        if os.name == 'nt':
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x00100000, False, pid)
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(ctypes.c_void_p(handle), 1000), 0)
                finally:
                    kernel.CloseHandle(ctypes.c_void_p(handle))
        else:
            # A killed child can remain a zombie briefly until reaped by init.
            stat = Path('/proc') / str(pid) / 'stat'
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], 'Z')

    def test_probe_rejects_oversized_output(self):
        started = time.monotonic()
        state, output = doctor._probe([sys._base_executable, '-c', 'import sys,time;sys.stdout.write("x"*10000);sys.stdout.flush();time.sleep(60)'], deadline=started+5)
        self.assertEqual(state, 'oversized')
        self.assertEqual(output, b'')
        self.assertLess(time.monotonic()-started, 8)

    def test_cli_json_redirection_and_usage_exit_codes(self):
        binary = io.BytesIO()
        stream = io.TextIOWrapper(binary, encoding='cp1252')
        with patch.object(doctor, '_probe', return_value=('ok', b'tk-available\n')), contextlib.redirect_stdout(stream):
            code = doctor.main(['--home', str(self.root), '--model', str(self.model), '--manifest', str(self.manifest), '--json', '--show-paths'])
        stream.flush()
        report = json.loads(binary.getvalue().decode('cp1252'))
        self.assertEqual(Path(report['paths']['model']), self.model)
        self.assertEqual(code, 1)
        for timeout in ('0', '-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                doctor.main(['--timeout-s', timeout])
            self.assertEqual(caught.exception.code, 2)

    def test_optional_missing_build_tools_do_not_create_required_failure(self):
        with patch.object(doctor, '_probe', return_value=('ok', b'tk-available\n')), patch.object(doctor.shutil, 'which', return_value=None):
            report = doctor.diagnose(self.root, model_path=self.model, manifest_path=self.manifest)
        checks = self.checks(report)
        for name in ('git', 'cmake', 'ninja', 'compiler'):
            self.assertEqual(checks['build_' + name]['status'], 'optional')
        self.assertNotIn('gpu', checks)

    def test_detected_avx2_absence_fails_but_fma_is_never_inferred(self):
        if os.name == 'nt':
            for detected, status in ((True, 'unknown'), (False, 'fail'), (None, 'unknown')):
                with self.subTest(detected=detected), patch.object(doctor, '_windows_avx2', return_value=detected):
                    self.assertEqual(doctor._cpu_check()['status'], status)
        else:
            self.assertEqual(doctor._cpu_check()['status'], 'unknown')

    def test_old_cmake_is_optional_for_cpu_user_and_version_is_bounded(self):
        def probe(command, **kwargs):
            return ('ok', b'cmake version 3.20.0\n') if command[-1] == '--version' else ('ok', b'tk-available\n')
        with patch.object(doctor, '_probe', side_effect=probe), patch.object(doctor.shutil, 'which', return_value='fixture-tool'):
            report = doctor.diagnose(self.root, model_path=self.model, manifest_path=self.manifest)
        self.assertEqual(self.checks(report)['build_cmake_version']['status'], 'optional')


if __name__ == '__main__':
    unittest.main()
