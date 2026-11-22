"""Windows Unicode native paths, using tiny synthetic files and no weights."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = Path(os.environ.get("BOVARY_TEST_NATIVE_DIR", ROOT / "work/build/bovary-cpu")).resolve()
INFER = NATIVE_DIR / "bovary-infer.exe"
QUANTIZE = NATIVE_DIR / "bovary-quantize.exe"
PATH_FIXTURE = NATIVE_DIR / "bovary-paths-test.exe"
UNICODE_NAME = "\u05e9\u05dc\u05d5\u05dd-caf\u00e9-\U0001f9ea"


def unsupported_gguf():
    """A valid GGUF metadata envelope with zero tensors, never a real model."""
    def string(value):
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data
    return (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("general.architecture")
            + struct.pack("<I", 8) + string("bovary-unicode-fixture"))


@unittest.skipUnless(os.name == "nt", "Windows wide argument handling")
class NativeUnicodePathsTests(unittest.TestCase):
    def native_run(self, command, overrides=None):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("BOVARY_STREAM")}
        environment["BOVARY_STREAM"] = "0"
        environment.pop("GGML_BACKEND_PATH", None)
        environment.update(overrides or {})
        return subprocess.run(list(map(str, command)), env=environment, capture_output=True, timeout=15)

    @unittest.skipUnless(PATH_FIXTURE.is_file(), "Build bovary-paths-test with scripts/build.cmd cpu 2 test-paths")
    def test_wide_arguments_and_unicode_file_read_write_and_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / UNICODE_NAME
            folder.mkdir()
            source = folder / (UNICODE_NAME + "-input.txt")
            destination = folder / (UNICODE_NAME + "-output.txt")
            content = (UNICODE_NAME + "\nfixture bytes\n").encode("utf-8")
            source.write_bytes(content)
            trace = Path(str(destination) + ".trace")
            result = self.native_run([PATH_FIXTURE, source, destination], {
                "BOVARY_STREAM_MODEL": str(source), "BOVARY_STREAM_TRACE": str(trace)})
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.replace(b"\r\n", b"\n"),
                             (str(source) + "\n" + str(destination) + "\n").encode("utf-8") + content)
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(Path(str(destination) + ".partial").exists())
            self.assertEqual(trace.read_bytes(), b"Unicode trace fixture\n")

    @unittest.skipUnless(INFER.is_file(), "Bovary CPU-only build required")
    def test_infer_opens_unicode_prompt_and_synthetic_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / UNICODE_NAME
            folder.mkdir()
            metadata = folder / (UNICODE_NAME + ".gguf")
            prompt = folder / (UNICODE_NAME + ".txt")
            metadata.write_bytes(unsupported_gguf())
            prompt.write_text("Synthetic path diagnostic", encoding="utf-8")
            result = self.native_run([INFER, "--require-cpu-only", "--model", metadata,
                                      "--prompt-file", prompt, "--predict", "1", "--context", "64"])
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"unknown model architecture: 'bovary-unicode-fixture'", result.stderr)
            self.assertNotIn(b"cannot open prompt", result.stderr)

    @unittest.skipUnless(QUANTIZE.is_file(), "Bovary CPU quantizer build required")
    def test_quantizer_opens_unicode_metadata_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / UNICODE_NAME
            folder.mkdir()
            metadata = folder / (UNICODE_NAME + ".gguf")
            destination = folder / (UNICODE_NAME + "-output.gguf")
            metadata.write_bytes(unsupported_gguf())
            command = [QUANTIZE, "--input", metadata, "--output", destination, "--dry-run"]
            result = self.native_run(command)
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"this experiment requires a Gemma4 GGUF", result.stderr)
            destination.write_bytes(b"retain this file")
            result = self.native_run(command)
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"output or partial output already exists", result.stderr)
            self.assertEqual(destination.read_bytes(), b"retain this file")


if __name__ == "__main__":
    unittest.main()
