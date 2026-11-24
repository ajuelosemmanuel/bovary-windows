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
from build_kit import CPU_REQUIRED, GEMMA_CUDA_NOTICES, NATIVE, ROOT, build_kit, sha256


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

    def build(self, output=None, **kwargs):
        with patch("build_kit.JSONHPP_SHA256", sha256(self.license.read_bytes())):
            build_kit(wheel=self.wheel, cpu_archive=self.cpu, checksums=self.release, native_bin=self.native,
                      native_checksums=self.receipt, jsonhpp_license=self.license, output=output or self.output,
                      **kwargs)

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

    def test_gemma_cuda_worker_and_notices_are_bundled(self):
        worker = self.path / "gemma-cuda.exe"
        worker.write_bytes(b"test Gemma CUDA worker")
        worker_hash = sha256(worker.read_bytes())
        with self.assertRaisesRegex(ValueError, "toolkit license"):
            self.build(gemma_cuda_worker=worker, gemma_cuda_sha256=worker_hash)

        license_path = self.path / "CUDA-LICENSE.txt"
        license_path.write_bytes(b"test toolkit license")
        with patch("build_kit.CUDA_TOOLKIT_LICENSE_SHA256", sha256(license_path.read_bytes())):
            self.build(gemma_cuda_worker=worker, gemma_cuda_sha256=worker_hash,
                       gemma_cuda_license=license_path)
        with ZipFile(self.output) as kit:
            self.assertEqual(kit.read("setup/gemma-cuda/bovary-infer.exe"), worker.read_bytes())
            self.assertEqual(kit.read("setup/gemma-cuda/CUDA-LICENSE.txt"), license_path.read_bytes())
            self.assertIn(worker_hash.encode(), kit.read("setup/friend_setup.py"))
            for name in GEMMA_CUDA_NOTICES:
                self.assertEqual(kit.read(f"setup/gemma-cuda/{Path(name).name}"), (ROOT / name).read_bytes())

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
            ) as run, patch.object(friend_setup, "verified", side_effect=lambda model, _: model == old_model), patch(
                "builtins.input", return_value=str(new_model)
            ):
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

    def test_choose_storage_and_reuse_bound_model_without_download(self):
        folder = self.path / "alternate drive" / "models"
        with patch.object(friend_setup, "STATE", self.path), patch("builtins.input", return_value=str(folder)):
            friend_setup.choose_storage()
            selected, explicit = friend_setup.model_storage()
            self.assertEqual(selected, folder.resolve())
            self.assertTrue(explicit)
            with patch.object(friend_setup, "verified", return_value=True), patch.object(friend_setup, "run") as run:
                model, receipt = friend_setup.model_setup()
                run.assert_not_called()
            self.assertEqual(model, selected / friend_setup.MODEL)
            self.assertEqual(receipt, selected / "model-provenance.json")

    def test_unavailable_storage_cannot_fall_back_to_download(self):
        with patch.object(friend_setup, "run") as run:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                friend_setup.model_setup(storage=(self.path / "disconnected", True))
            run.assert_not_called()

    def test_explicit_registration_is_remembered_for_selected_library(self):
        model = self.path / "Gemma elsewhere.gguf"
        model.touch()
        library = self.path / "models"
        library.mkdir()
        with patch.object(friend_setup, "STATE", self.path), patch("builtins.input", return_value=str(model)), patch.object(
            friend_setup, "run"
        ) as run, patch.object(friend_setup, "verified", side_effect=lambda _, proof: proof.parent == self.path
                               and proof.name.startswith("verified-")):
            registered = friend_setup.model_setup(use_existing=True, storage=(library, True))
            self.assertEqual(registered[0], model)
            self.assertEqual(friend_setup.model_setup(storage=(library, True)), registered)
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

    def test_gemma_stages_only_runtime_and_reuses_it(self):
        worker = self.path / "gemma-cuda/bovary-infer.exe"
        worker.parent.mkdir()
        worker.write_bytes(b"fixture worker")
        expected = sha256(worker.read_bytes())
        env = {"BOVARY_HOME": str(self.path)}
        archive_name = "cudart-llama-bin-win-cuda-13.5-x64.zip"
        folder = self.path / "inkling"
        folder.mkdir()
        archive = folder / archive_name
        with ZipFile(archive, "w") as output:
            output.writestr("bin/cudart64_13.dll", b"fixture DLL")
        record = {"sha256": sha256(archive.read_bytes()), "files": ["cudart64_13.dll"]}
        probe = SimpleNamespace(compatible_device=lambda: False, gemma_preset=lambda *_: "cpu-only")
        with patch.dict(sys.modules, {"cuda_default": probe}), patch.dict(
                self.setup.LOCK["cuda_archives"], {archive_name: record}), patch.object(
                self.setup, "download", wraps=self.setup.download) as download:
            self.assertIsNone(self.setup.gemma_cuda_environment(self.path, self.path, env, expected))
            download.assert_not_called()
            probe.compatible_device = lambda: True
            probe.gemma_preset = lambda _, env: "gpu-prefill-2g" if "BOVARY_CUDA_EXE" in env else "cpu-only"
            for _ in range(2):
                selected = self.setup.gemma_cuda_environment(self.path, self.path, env, expected)
                self.assertEqual(selected["BOVARY_CUDA_EXE"], str(worker))
                runtime = Path(selected["PATH"].split(self.setup.os.pathsep)[0])
                self.assertEqual((runtime / "cudart64_13.dll").read_bytes(), b"fixture DLL")
            download.assert_called_once()

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

    def test_linked_or_oversized_partial_is_preserved(self):
        model = self.path / "test.gguf"
        partial = model.with_suffix(".gguf.partial")
        partial.write_bytes(b"12345")
        with patch.object(self.setup.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Unsafe partial"):
                self.setup.download("https://example.test/model", model, 3, "unused")
            self.assertEqual(partial.read_bytes(), b"12345")
            original_stat = type(partial).stat

            def linked_stat(path, *args, **kwargs):
                stat = original_stat(path, *args, **kwargs)
                if path == partial:
                    return SimpleNamespace(st_mode=stat.st_mode, st_size=stat.st_size, st_nlink=2)
                return stat

            with patch.object(type(partial), "stat", linked_stat):
                with self.assertRaisesRegex(ValueError, "Unsafe partial"):
                    self.setup.download("https://example.test/model", model, 6, "unused")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
