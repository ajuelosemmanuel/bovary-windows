"""The Bovary alpha exposes only its canonical package and configuration."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import bovary
from bovary import cli
from bovary.contract import Request
from bovary.runtime import create_workbench, project_home


class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_canonical_implementation(self):
        self.assertIs(bovary.Request, Request)
        self.assertIs(bovary.create_workbench, create_workbench)
        self.assertEqual(Request.__module__, 'bovary.contract')
        self.assertFalse((ROOT/'python/lowcostgemma').exists())

    def test_suggest_does_not_execute_a_backend(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(['--mode', 'suggest', '--task', 'extract', '--prompt', 'name: Ada', '--json'])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'suggested')
        self.assertEqual(result['attempts'], [])

    def test_version_requires_no_prompt_or_backend(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            cli.main(['--version'])
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f'Bovary {bovary.__version__}')

    def test_only_bovary_home_is_supported(self):
        with mock.patch.object(os, 'environ', {'BOVARY_HOME': str(ROOT/'work/new-home'),
                'LOWCOSTGEMMA_HOME': str(ROOT/'work/ignored-home')}):
            self.assertEqual(project_home(), (ROOT/'work/new-home').resolve())
        with mock.patch.object(os, 'environ', {'LOWCOSTGEMMA_HOME': str(ROOT/'work/ignored-home')}):
            self.assertNotEqual(project_home(), (ROOT/'work/ignored-home').resolve())


if __name__ == '__main__':
    unittest.main()
