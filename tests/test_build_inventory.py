import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capture_inventory', ROOT/'tools/inventory_build_manifest.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryCaptureTests(unittest.TestCase):
    def test_pinned_patch_requires_independent_worktree_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = b'canonical patch bytes'
            revision = 'pinned revision'
            patch_sha256 = hashlib.sha256(patch).hexdigest()
            (root/'engine.patch').write_bytes(patch)
            (root/'engine.lock.json').write_text(json.dumps({
                'revision': revision, 'patch': 'engine.patch', 'patch_sha256': patch_sha256,
            }), encoding='utf-8')
            proof = json.dumps({'revision': revision, 'patch_sha256': patch_sha256, 'verified': True})
            with mock.patch.object(inventory, 'ROOT', root), mock.patch.object(inventory.subprocess, 'run') as runner:
                engine = root/'engine'
                self.assertIsNone(inventory.verified_pin_patch(engine, 'different revision'))
                runner.assert_not_called()
                runner.return_value = subprocess.CompletedProcess([], 2, stdout='')
                self.assertIsNone(inventory.verified_pin_patch(engine, revision))
                runner.return_value = subprocess.CompletedProcess([], 0, stdout=proof)
                self.assertEqual(inventory.verified_pin_patch(engine, revision), patch)
                self.assertEqual(runner.call_args.args[0][-3:], ['--check', '--directory', str(engine)])
                (root/'engine.patch').write_bytes(b'changed patch bytes')
                with self.assertRaisesRegex(ValueError, 'patch changed'):
                    inventory.verified_pin_patch(engine, revision)

    def test_existing_capture_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory)/'proof'
            capture.mkdir()
            evidence = capture/'build-manifest.json'
            evidence.write_bytes(b'previous immutable evidence')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                inventory.create_capture(capture, 'cpu')
            self.assertEqual(evidence.read_bytes(), b'previous immutable evidence')
            self.assertEqual(list(capture.iterdir()), [evidence])

    def test_default_allocates_unique_captures(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(inventory, 'ROOT', Path(directory)):
            first = inventory.create_capture(None, 'cpu')
            second = inventory.create_capture(None, 'cpu')
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())
            self.assertEqual(first.parent, Path(directory)/'outputs/build-inventory')

    def test_freshness_rejects_other_source_roots_before_running_a_tool(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(inventory.subprocess, 'run') as runner:
            with self.assertRaisesRegex(ValueError, 'another source tree'):
                inventory.check_build_freshness(Path(directory), {'CMAKE_HOME_DIRECTORY':directory}, ['fixture.out'])
            runner.assert_not_called()

    def test_missing_source_root_is_not_inferred_from_working_directory(self):
        with mock.patch.object(inventory.subprocess, 'run') as runner:
            for home in (None, '', ' '):
                with self.subTest(home=home), self.assertRaisesRegex(ValueError, 'another source tree'):
                    inventory.check_build_freshness(ROOT, {'CMAKE_HOME_DIRECTORY':home}, ['fixture.out'])
            runner.assert_not_called()

    def test_real_ninja_dry_run_rejects_stale_target_without_rebuilding(self):
        ninja = shutil.which('ninja')
        cache = ROOT/'work/build/bovary-cpu/CMakeCache.txt'
        if not ninja and cache.exists():
            ninja = inventory.read_options(cache.parent).get('CMAKE_MAKE_PROGRAM')
        if not ninja or not Path(ninja).is_file():
            self.skipTest('Ninja required for real dependency freshness check')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'copy_fixture.py').write_text('from pathlib import Path\nimport sys\nPath(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n')
            (root/'source.txt').write_bytes(b'first version')
            command = '"'+sys.executable.replace('$','$$')+'" copy_fixture.py $in $out'
            (root/'build.ninja').write_text('rule copy\n  command = '+command+'\nbuild fixture.out: copy source.txt\n')
            subprocess.run([ninja, '-C', str(root), 'fixture.out'], check=True, capture_output=True)
            options = {'CMAKE_HOME_DIRECTORY':str(root), 'CMAKE_MAKE_PROGRAM':ninja}
            with mock.patch.object(inventory, 'ROOT', root):
                self.assertTrue(inventory.check_build_freshness(root, options, ['fixture.out'])['up_to_date'])
                previous = (root/'fixture.out').read_bytes()
                source = root/'source.txt'
                source.write_bytes(b'changed source')
                stamp = (root/'fixture.out').stat().st_mtime_ns + 2_000_000_000
                os.utime(source, ns=(stamp,stamp))
                with self.assertRaisesRegex(ValueError, 'not up to date'):
                    inventory.check_build_freshness(root, options, ['fixture.out'])
                self.assertEqual((root/'fixture.out').read_bytes(), previous)
