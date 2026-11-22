"""Model selection must preserve provider ownership and private settings."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from bovary import gui
from bovary.contract import ExecutionError


class GuiSetupTests(unittest.TestCase):
    def test_environment_over_saved_and_explicit_selection_over_environment(self):
        paths = gui.effective_paths({'model': 'saved.gguf'}, {'BOVARY_MODEL': 'env.gguf'})
        self.assertEqual(paths['model'], 'env.gguf')
        local = SimpleNamespace(model_path=None, manifest_path=None, exe=None)
        window = gui.SettingsWindow.__new__(gui.SettingsWindow)
        window.model_paths = {**paths, 'model': 'selected.gguf'}
        window.workbench = SimpleNamespace(local=local, unload=mock.Mock())
        window.device = SimpleNamespace(get=lambda: 'CPU')
        window.apply_model_paths()
        window.workbench.unload.assert_called_once()
        self.assertEqual(local.model_path, gui.ROOT / 'selected.gguf')
        window.apply_model_paths()
        window.workbench.unload.assert_called_once()

    def test_save_paths_unloads_all_cached_workers_without_replacing_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            window = gui.SettingsWindow.__new__(gui.SettingsWindow)
            window.process = None
            window.model_paths = {}
            window.settings_path = Path(directory) / 'settings.json'
            window.settings = lambda: {**gui.DEFAULTS, 'paths': window.model_paths,
                                       'prompt': 'private', 'api_key': 'private'}
            provider = object()
            workbench = SimpleNamespace(unload=mock.Mock(), commercial=provider)
            window.workbench_cache = {'approved': workbench}
            window.commit_model_paths({'cpu_executable': 'cpu.exe'})
            workbench.unload.assert_called_once()
            self.assertIs(workbench.commercial, provider)
            saved = json.loads(window.settings_path.read_text())
            self.assertEqual(saved['paths'], {'cpu_executable': 'cpu.exe'})
            self.assertNotIn('prompt', saved)
            self.assertNotIn('api_key', saved)
            window.process = True
            with self.assertRaises(ExecutionError):
                window.commit_model_paths({})
            self.assertEqual(window.model_paths, {'cpu_executable': 'cpu.exe'})
            window.process = None
            window.model_state = mock.Mock()
            with mock.patch.object(gui, 'save_settings', side_effect=OSError('read-only settings')):
                with self.assertRaises(OSError):
                    window.commit_model_paths({'cpu_executable': 'another.exe'})
            self.assertEqual(window.model_paths, {'cpu_executable': 'cpu.exe'})
            window.model_state.set.assert_called_with('unloaded')

    def test_mismatched_proof_rejected_before_unload_or_save(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.gguf'
            proof = Path(directory) / 'proof.json'
            model.write_bytes(b'tiny')
            proof.write_text(json.dumps({'filename': 'another.gguf', 'verified': True,
                                         'bytes': 4, 'sha256': 'a' * 64}))
            window = gui.SettingsWindow.__new__(gui.SettingsWindow)
            window.process = None
            worker = mock.Mock()
            window.workbench_cache = {'local': worker}
            with self.assertRaisesRegex(ValueError, 'manifest'):
                window.commit_model_paths({'model': str(model), 'manifest': str(proof)})
            worker.unload.assert_not_called()

    def test_blank_selections_ignore_environment_and_installed_verifier_is_explained(self):
        with mock.patch.object(gui.os, 'environ', {**gui.os.environ, 'BOVARY_MODEL': 'private.gguf', 'BOVARY_MODEL_MANIFEST': 'private.json', 'BOVARY_CPU_EXE': 'private.exe'}):
            paths = gui.resolved_gui_paths({})
        self.assertEqual(paths['manifest'], gui.ROOT / 'work/models/model-provenance.json')
        self.assertEqual(paths['executable'], gui.ROOT / 'work/build/bovary-cpu/bovary-infer.exe')
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(gui, 'ROOT', Path(directory)):
            self.assertIn('source distribution', gui.verification_command({}))

    def test_queued_output_does_not_overwrite_stopping_or_commercial_state(self):
        from test_gui_execution import GuiExecutionTests
        with tempfile.TemporaryDirectory() as directory:
            window = GuiExecutionTests().window(Path(directory) / 'settings.json')
            state = mock.Mock()
            window.model_state = state
            window.workbench = SimpleNamespace(local=SimpleNamespace(loaded=True))
            window._local_request = True
            window.cancel.set()
            window.events.put(('output', 'late local text'))
            window.pump()
            state.set.assert_not_called()
            window.cancel.clear()
            window._local_request = False
            window.events.put(('output', 'commercial text'))
            window.pump()
            state.set.assert_not_called()
            window._local_request = True
            window.events.put(('output', 'local text'))
            window.pump()
            state.set.assert_called_once_with('generating')

    def test_selected_model_checks_default_manifest_before_configuration(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(gui, 'ROOT', Path(directory)):
            root = Path(directory)
            model = root / 'different.gguf'
            model.write_bytes(b'tiny')
            manifest = root / 'work/models/model-provenance.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'filename': 'previous.gguf', 'bytes': 4, 'sha256': 'a' * 64, 'verified': True}))
            window = gui.SettingsWindow.__new__(gui.SettingsWindow)
            window.process = None
            with self.assertRaisesRegex(ValueError, 'manifest'):
                window.commit_model_paths({'model': str(model)})

    def test_path_schema_and_powershell_quoting(self):
        for paths in ([], {'api_key': 'secret'}, {'model': 12}, {'model': '\0'}):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                gui.validate_settings({**gui.DEFAULTS, 'paths': paths})
        command = gui.verification_command({'model': "D:/Models/café's שלום🧪.gguf"})
        self.assertIn("café''s שלום🧪.gguf", command)
        self.assertIn('--manifest', command)


if __name__ == '__main__':
    unittest.main()
