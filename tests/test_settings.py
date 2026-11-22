import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from bovary import gui
from bovary.local import LocalAdapter


class SettingsTests(unittest.TestCase):
    def test_settings_roundtrip_and_device_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            gui.save_settings(path, gui.DEFAULTS)
            self.assertEqual(json.loads(path.read_text()), gui.DEFAULTS)
            self.assertEqual(gui.validate_settings({**gui.DEFAULTS, 'tokens':3000})['tokens'], 3000)
            self.assertFalse(list(path.parent.glob('*.tmp')))
            for changes in ({'device':'GPU'}, {'threads':True}, {'context':8192}, {'tokens':4096},
                            {'tokens':True}, {'tokens':'None'}, {'tokens':0}, {'tokens':float('inf')}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    gui.validate_settings({**gui.DEFAULTS, **changes})

    def test_until_finished_settings_roundtrip_without_changing_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            settings = {**gui.DEFAULTS, 'tokens':None}
            gui.save_settings(path, settings)
            self.assertEqual(gui.validate_settings(json.loads(path.read_text())), settings)
            self.assertEqual(gui.DEFAULTS['tokens'], 128)
            self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_gpu_settings_build_detached_requests_without_a_backend(self):
        for profile, batch, cache in (('gpu-fast', '128', '0'), ('gpu-more-vram', '1', '1536'),
                                      ('balanced', '1', '1024'), ('gpu-cache-2g', '1', '2048'),
                                      ('gpu-prefill-2g', '128', '2048')):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                request = gui.build_request({**gui.DEFAULTS, 'device':'GPU', 'preset':profile}, 'Hello')
                self.assertEqual(request.local_profile, profile)
                self.assertTrue(request.offline)
                self.assertTrue(request.private)
                root = Path(directory)
                model, executable = root/'model.gguf', root/'bovary-infer.exe'
                model.write_bytes(b'fixture')
                executable.touch()
                adapter = LocalAdapter(root, executable=executable)
                try:
                    with mock.patch.object(adapter, '_checkpoint', return_value=(model, {'sha256':'a'*64})), mock.patch.object(os, 'environ', {}):
                        command, environment, *_ = adapter._configuration(request)
                    self.assertFalse(adapter.loaded)
                    self.assertEqual(Path(command[0]), executable)
                    self.assertEqual(command[command.index('--gpu-layers')+1], '8' if profile == 'gpu-fast' else '99')
                    self.assertEqual(command[command.index('--prompt-batch')+1], batch)
                    self.assertEqual(command[command.index('--cache-mib')+1], cache)
                    self.assertNotIn('--host-cache-mib', command)
                    self.assertEqual(environment['BOVARY_STREAM'], '0' if profile == 'gpu-fast' else '1')
                    self.assertEqual('--cpu-experts' in command, profile != 'gpu-fast')
                finally:
                    adapter.unload()

    def test_host_cache_settings_roundtrip_and_legacy_defaults(self):
        settings = {**gui.DEFAULTS, 'device':'GPU', 'preset':'balanced', 'host_cache_mib':768}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            gui.save_settings(path, settings)
            self.assertEqual(gui.validate_settings(json.loads(path.read_text())), settings)
        legacy = dict(settings)
        del legacy['host_cache_mib']
        self.assertEqual(gui.validate_settings(legacy)['host_cache_mib'], 0)
        for value in (-1, gui.MAX_HOST_CACHE_MIB+1, True, 1.5, '768'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Host expert cache'):
                gui.validate_settings({**settings, 'host_cache_mib':value})
        for device, preset in (('CPU', 'cpu-only'), ('GPU', 'gpu-fast')):
            with self.subTest(device=device, preset=preset):
                self.assertEqual(gui.validate_settings({**settings, 'device':device, 'preset':preset})['host_cache_mib'], 0)
        for mode in ('local', 'commercial', 'suggest', 'auto'):
            with self.subTest(mode=mode):
                request = gui.build_request(settings, 'Hello', mode=mode)
                self.assertEqual(request.host_cache_mib, 768 if mode == 'local' else 0)
                self.assertTrue(request.private and request.offline)

    def test_eligible_gui_request_passes_cache_budget_to_the_worker_arguments(self):
        settings = {**gui.DEFAULTS, 'device':'GPU', 'preset':'balanced', 'host_cache_mib':768}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, executable = root/'model.gguf', root/'bovary-infer.exe'
            model.write_bytes(b'fixture')
            executable.touch()
            adapter = LocalAdapter(root, executable=executable)
            with mock.patch.object(adapter, '_checkpoint', return_value=(model, {'sha256':'a'*64})), mock.patch.object(os, 'environ', {}):
                command, *_ = adapter._configuration(gui.build_request(settings, 'Hello'))
            self.assertEqual(command[command.index('--host-cache-mib')+1], '768')
            self.assertFalse(adapter.loaded)

    def test_bovary_settings_have_one_canonical_path(self):
        self.assertEqual(gui.SETTINGS, gui.ROOT/'work/bovary-settings.json')


if __name__ == '__main__':
    unittest.main()

