import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import friend_setup
from build_kit import CPU_REQUIRED, NATIVE, ROOT, build_kit, sha256


class BuildKitTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = Path(scratch.name)
        version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        self.wheel = self.path / f"bovary-{version}-py3-none-any.whl"
        self.wheel.write_bytes(b"test wheel")
        self.cpu = self.path / f"bovary-{version}-cpu-review.zip"
        with ZipFile(self.cpu, "w") as cpu:
            for name in sorted(CPU_REQUIRED | {"LICENSE", "THIRD_PARTY_NOTICES.md"}):
                cpu.writestr(name, "test CPU archive")
        self.release = self.path / "SHA256SUMS.txt"
        self.release.write_text(
            "".join(f"{sha256(path.read_bytes())}  {path.name}\n" for path in (self.wheel, self.cpu))
            + f"{'0' * 64}  unrelated.zip\n",
            encoding="ascii",
        )
        self.native = self.path / "native"
        self.native.mkdir()
        for name in NATIVE:
            (self.native / name).write_bytes(name.encode("ascii"))
        self.receipt = self.path / "native.json"
        self.write_native_receipt()
        self.license = self.path / "LICENSE-jsonhpp"
        self.license.write_bytes(b"test jsonhpp license")
        self.output = self.path / "kit.zip"

    def write_native_receipt(self):
        receipt = {name: sha256((self.native / name).read_bytes()) for name in NATIVE}
        self.receipt.write_text(json.dumps(receipt), encoding="ascii")

    def build(self, output=None):
        with patch("build_kit.JSONHPP_SHA256", sha256(self.license.read_bytes())):
            build_kit(wheel=self.wheel, cpu_archive=self.cpu, checksums=self.release, native_bin=self.native,
                      native_checksums=self.receipt, jsonhpp_license=self.license, output=output or self.output)

    def test_embedded_manifest_is_accepted_on_install(self):
        self.build()
        second_zip = self.path / "kit-second.zip"
        self.build(second_zip)
        self.assertEqual(self.output.read_bytes(), second_zip.read_bytes())
        extracted = self.path / "consumer install"
        with ZipFile(self.output) as kit:
            self.assertIn("Start Bovary.cmd", kit.namelist())
            self.assertIn("READ ME.txt", kit.namelist())
            self.assertNotIn(b"unrelated.zip", kit.read("setup/SHA256SUMS.txt"))
            kit.extractall(extracted)
        spec = importlib.util.spec_from_file_location("packaged_setup", extracted / "setup/friend_setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        setup.STATE.mkdir()
        with patch.object(setup, "install_native_dependencies"), patch.object(
                setup, "run", return_value=SimpleNamespace(stdout="bovary-worker 2\n")) as run:
            setup.install()
        self.assertTrue(setup.WORKER.is_file())
        self.assertEqual(run.call_count, 3)

    def test_rejects_tampered_inputs(self):
        path = self.native / NATIVE[0]
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.build()
        private_path = "C:" + str(Path("/Users/developer/source"))
        path.write_bytes(private_path.encode("utf-16-le"))
        self.write_native_receipt()
        with self.assertRaisesRegex(ValueError, "Private Windows build path"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_gemma_replaces_saved_settings(self):
        settings = self.path / "model-path.json"
        old_model = self.path / "old.gguf"
        old_proof = self.path / "stale.json"
        new_model = self.path / "new.gguf"
        for path in (old_model, old_proof, new_model):
            path.touch()

        def verify(*args):
            self.assertEqual(args[-3], new_model)
            args[-1].write_text("verified", encoding="utf8")

        saved_paths = json.dumps([str(old_model), str(old_proof)])
        for saved in (saved_paths, "{invalid", "[null, {}]"):
            with self.subTest(saved=saved), patch.object(friend_setup, "STATE", self.path), patch.object(
                friend_setup, "run", side_effect=verify
            ) as run, patch("builtins.input", return_value=str(new_model)):
                settings.write_text(saved, encoding="utf8")
                if saved == saved_paths:
                    self.assertEqual(friend_setup.model_setup(), (old_model, old_proof))
                else:
                    with self.assertRaisesRegex(ValueError, "Choose E to register"):
                        friend_setup.model_setup()
                model, proof = friend_setup.model_setup(use_existing=True)
                self.assertEqual(model, new_model)
                self.assertEqual(proof.read_text(encoding="utf8"), "verified")
                self.assertEqual(json.loads(settings.read_text(encoding="utf8")), [str(model), str(proof)])
                run.assert_called_once()


class InklingSetupTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = Path(scratch.name)
        spec = importlib.util.spec_from_file_location("inkling_setup", ROOT / "launcher/inkling_setup.py")
        self.setup = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(ROOT / "previews/inkling"), *sys.path]):
            spec.loader.exec_module(self.setup)

    def test_retry_verifies_and_publishes_resumed_download(self):
        url = "https://example.test/model"
        model = self.path / "test.gguf"
        partial = model.with_suffix(".gguf.partial")
        data = b"verified model"

        def curl(args, *, check):
            self.assertTrue(check)
            self.assertEqual(args[args.index("--continue-at") + 1], "-")
            self.assertEqual(Path(args[args.index("--output") + 1]), partial)
            if not partial.exists():
                partial.write_bytes(data[:4])
                raise subprocess.CalledProcessError(18, args)
            self.assertEqual(partial.read_bytes(), data[:4])
            with partial.open("ab") as output:
                output.write(data[4:])

        with patch.object(self.setup.subprocess, "run", side_effect=curl) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.setup.download(url, model, len(data), sha256(data))
            self.assertFalse(model.exists())
            self.setup.download(url, model, len(data), sha256(data))
            self.setup.download(url, model, len(data), sha256(data))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(model.read_bytes(), data)
        self.assertFalse(partial.exists())

    def test_preflight_reserves_full_sidecar_space(self):
        folder = self.path / "inkling"
        (folder / "bin").mkdir(parents=True)
        model = {"filename": "test.gguf", "bytes": 10, "packed_bytes": 20}
        (folder / "test.gguf.partial").write_bytes(b"seen")
        (folder / "test.experts.pack.partial").write_bytes(b"x" * 19)

        def device_count(result):
            result._obj.value = 1
            return 0

        driver = SimpleNamespace(cuInit=lambda _: 0, cuDeviceGetCount=device_count)
        disk_usage = SimpleNamespace(free=3 * 1024**3 + 6 + 19)
        with patch.object(self.setup, "MODEL", model), patch.object(
            self.setup.ctypes, "WinDLL", return_value=driver, create=True
        ), patch.object(self.setup.shutil, "disk_usage", return_value=disk_usage), patch.object(
            self.setup, "download"
        ) as download:
            with self.assertRaisesRegex(ValueError, "free to finish"):
                self.setup.install(self.path, self.path)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
