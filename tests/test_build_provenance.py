import contextlib
import io
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import build_provenance as provenance
import make_config
import make_cpu_config
import make_final_config
import make_pool_config


class BuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        names = ('src/infer.cpp','src/quantize.cpp','src/native_paths.h','CMakeLists.txt',
                 'scripts/build.cmd','scripts/find_build_tools.cmd')
        for name in names:
            p = self.root/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture source '+name)
        (self.root/'engine.lock.json').write_text(json.dumps({'revision':'pin', 'patch_sha256':'patch'}))
        self.binary = self.root/'bovary-infer.exe'
        self.binary.write_bytes(b'fixture executable')
        self.manifest = {'backend':'cpu', 'engine_commit':'pin', 'engine_diff_sha256':'patch',
            'smoke':{'passed':True},
            'build_freshness':{'up_to_date':True,'targets':['bovary-infer.exe']},
            'build_options':{'BOVARY_CPU_ONLY':'ON','GGML_CUDA':'OFF','CMAKE_CXX_COMPILER':'fixture compiler','CMAKE_HOME_DIRECTORY':str(self.root)},
            'application_source_sha256':provenance.digest(self.root/'src/infer.cpp'),
            'root_cmake_sha256':provenance.digest(self.root/'CMakeLists.txt'),
            'build_script_sha256':provenance.digest(self.root/'scripts/build.cmd'),
            'source_snapshots':[{'source':n,'sha256':provenance.digest(self.root/n)} for n in names],
            'executables':[{'path':str(self.binary),'bytes':self.binary.stat().st_size,'sha256':provenance.digest(self.binary)}]}
        self.path = self.root/'capture.json'
        models = self.root/'work/models'
        models.mkdir(parents=True)
        model = models/'fixture.gguf'
        model.write_bytes(b'fixture')
        (models/'model-provenance.json').write_text(json.dumps({'filename':model.name,'bytes':7,
            'sha256':provenance.digest(model),'verified':True,'revision':'model pin'}))
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_identity_comes_from_captured_build_and_unknown_driver_stays_unknown(self):
        binary, metadata = provenance.load_build(self.path, 'cpu', self.root)
        self.assertEqual(binary, self.binary)
        self.assertEqual(metadata['native_exe_sha256'], provenance.digest(self.binary))
        self.assertEqual(metadata['build_options']['CMAKE_CXX_COMPILER'], 'fixture compiler')
        self.assertIsNone(metadata['driver'])
        self.assertEqual(metadata['source_commit'], 'pin')

    def test_changed_executable_or_sources_cannot_reuse_old_proof(self):
        self.binary.write_bytes(b'replaced executable')
        with self.assertRaisesRegex(ValueError, 'executable'):
            provenance.load_build(self.path, 'cpu', self.root)
        self.binary.write_bytes(b'fixture executable')
        (self.root/'src/native_paths.h').write_text('changed Unicode implementation')
        with self.assertRaisesRegex(ValueError, 'source identity'):
            provenance.load_build(self.path, 'cpu', self.root)

    def test_wrong_backend_or_engine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'backend'):
            provenance.load_build(self.path, 'cuda', self.root)
        self.manifest['engine_diff_sha256'] = 'stale patch'
        self.save()
        with self.assertRaisesRegex(ValueError, 'engine lock'):
            provenance.load_build(self.path, 'cpu', self.root)

    def test_failed_smoke_or_foreign_source_cannot_be_called_verified(self):
        self.manifest['smoke']['passed'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'smoke'):
            provenance.load_build(self.path, 'cpu', self.root)
        self.manifest['smoke']['passed'] = True
        self.manifest['build_options']['CMAKE_HOME_DIRECTORY'] = str(self.root/'foreign')
        self.save()
        with self.assertRaisesRegex(ValueError, 'source tree'):
            provenance.load_build(self.path, 'cpu', self.root)
        self.manifest['build_options'].pop('CMAKE_HOME_DIRECTORY')
        self.save()
        with self.assertRaisesRegex(ValueError, 'source tree'):
            provenance.load_build(self.path, 'cpu', self.root)

    def test_quantizer_identity_uses_its_own_source_and_requires_build_freshness(self):
        quantizer = self.root/'bovary-quantize.exe'
        quantizer.write_bytes(b'quantizer fixture')
        self.manifest['executables'].append({'path':str(quantizer),'bytes':quantizer.stat().st_size,'sha256':provenance.digest(quantizer)})
        self.manifest['build_freshness']['targets'].append(quantizer.name)
        self.save()
        _, metadata = provenance.load_build(self.path, 'cpu', self.root, quantizer.name)
        self.assertEqual(metadata['native_source_sha256'], provenance.digest(self.root/'src/quantize.cpp'))
        self.assertNotEqual(metadata['native_source_sha256'], provenance.digest(self.root/'src/infer.cpp'))
        self.manifest.pop('build_freshness')
        self.save()
        with self.assertRaisesRegex(ValueError, 'freshness'):
            provenance.load_build(self.path, 'cpu', self.root, quantizer.name)

    def test_cpu_generator_uses_selected_manifest_and_never_invents_hardware(self):
        output = self.root/'config.json'
        with mock.patch.object(make_cpu_config, 'ROOT', self.root), \
                mock.patch.object(sys, 'argv', ['make_cpu_config','--build-manifest',str(self.path),'--output',str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            make_cpu_config.main()
        config = json.loads(output.read_text())['configurations'][0]
        self.assertEqual(Path(config['command'][0]), self.binary)
        self.assertIn('--require-cpu-only', config['command'])
        self.assertFalse(config['gpu_telemetry'])
        self.assertIsNone(config['metadata']['driver'])
        self.assertEqual(config['metadata']['model_bytes'], 7)

    def test_gpu_template_derives_build_identity_without_querying_a_gpu(self):
        self.manifest['backend'] = 'cuda'
        self.manifest['build_options'].update(BOVARY_CPU_ONLY='OFF', GGML_CUDA='ON')
        self.save()
        output = self.root/'gpu.json'
        with mock.patch.object(make_config, 'ROOT', self.root), \
                mock.patch.object(sys, 'argv', ['make_config','--build-manifest',str(self.path),'--output',str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            make_config.main()
        configs = json.loads(output.read_text())['configurations']
        self.assertTrue(configs)
        for config in configs:
            self.assertEqual(Path(config['command'][0]), self.binary)
            self.assertEqual(config['metadata']['build_options']['CMAKE_CXX_COMPILER'], 'fixture compiler')
            self.assertIsNone(config['metadata']['driver'])
            self.assertNotIn('610.60', json.dumps(config['metadata']))

    def test_dependent_generators_forward_proof_and_preserve_old_templates(self):
        self.manifest['backend'] = 'cuda'
        self.manifest['build_options'].update(BOVARY_CPU_ONLY='OFF', GGML_CUDA='ON')
        self.save()
        tools = self.root/'tools'
        tools.mkdir()
        for name in ('make_config.py', 'build_provenance.py'):
            shutil.copyfile(ROOT/'tools'/name, tools/name)
        for module, mode, basename in ((make_pool_config,'smoke','pool-template.json'),
                                       (make_final_config,'backend-check','final-template.json')):
            with self.subTest(module=module.__name__):
                previous = self.root/'work'/basename
                previous.write_bytes(b'preserve prior experiment template')
                output = self.root/(module.__name__+'.json')
                with mock.patch.object(module, 'ROOT', self.root), \
                        mock.patch.object(sys, 'argv', [module.__name__,'--mode',mode,'--build-manifest',str(self.path),'--output',str(output)]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                self.assertEqual(previous.read_bytes(), b'preserve prior experiment template')
                configurations = json.loads(output.read_text())['configurations']
                self.assertTrue(configurations)
                for config in configurations:
                    self.assertEqual(config['metadata']['source_commit'], 'pin')
                    self.assertIsNone(config['metadata']['driver'])
                    self.assertNotIn('paused/closed by user', json.dumps(config['metadata']))
