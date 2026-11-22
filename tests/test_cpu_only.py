import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = Path(os.environ.get("BOVARY_TEST_NATIVE_DIR", ROOT / "work/build/bovary-cpu")).resolve()
sys.path.insert(0, str(ROOT / "python"))
from bovary.contract import Request
from bovary.local import LocalAdapter


class CpuOnlyTests(unittest.TestCase):
    def test_adapter_defaults_to_cpu_without_gpu_binary_or_backend_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "work/build/bovary-cpu/bovary-infer.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            model = root / "fixture.gguf"
            model.write_bytes(b"fixture")
            adapter = LocalAdapter(root)
            try:
                with mock.patch.object(adapter, "_checkpoint", return_value=(model, {"sha256": "a" * 64})), \
                        mock.patch.object(os, 'environ', {"PATH": "unchanged", "GGML_BACKEND_PATH": "unwanted.dll",
                                                    "BOVARY_STREAM": "1", "BOVARY_STREAM_TRACE": "unwanted.csv"}):
                    command, environment, *_ = adapter._configuration(Request("Hello"))
                self.assertEqual(Path(command[0]), executable)
                self.assertEqual(command[command.index("--gpu-layers") + 1], "0")
                self.assertEqual(command[command.index("--cache-mib") + 1], "0")
                self.assertEqual(command[command.index("--prompt-batch") + 1], "32")
                self.assertIn("--require-cpu-only", command)
                self.assertNotIn("--gpu-embedding", command)
                self.assertNotIn("--cpu-experts", command)
                self.assertEqual(environment["BOVARY_STREAM"], "0")
                self.assertEqual(environment["PATH"], "unchanged")
                self.assertNotIn("GGML_BACKEND_PATH", environment)
                self.assertFalse(any(key.startswith("BOVARY_STREAM_") for key in environment))
            finally:
                adapter.unload()

    @unittest.skipUnless((NATIVE_DIR / "bovary-infer.exe").exists(), "Bovary CPU-only build required")
    def test_native_cpu_build_rejects_gpu_settings_before_loading(self):
        executable = NATIVE_DIR / "bovary-infer.exe"
        for option in (["--gpu-layers", "1"], ["--cache-mib", "32"], ["--gpu-embedding"]):
            with self.subTest(option=option):
                result = subprocess.run([str(executable), "--model", "unused", "--prompt-file", "unused", *option],
                    env={**os.environ, "BOVARY_STREAM": "0"}, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                self.assertIn("CPU-only build requires zero GPU layers/cache", result.stderr)
                self.assertNotIn("bovary_cpu: cpu_only=1", result.stderr)


if __name__ == "__main__":
    unittest.main()
