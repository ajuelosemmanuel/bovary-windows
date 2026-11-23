"""Test executable discovery only; no toolchain or build is launched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[1]
FINDER = ROOT / "scripts/find_build_tools.cmd"
UNICODE_NAME = "\u05e9\u05dc\u05d5\u05dd-caf\u00e9-\U0001f9ea"


@unittest.skipUnless(os.name == "nt", "Windows build script discovery")
class BuildToolDiscoveryTests(unittest.TestCase):
    def test_source_build_uses_checkout_venv_without_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source checkout with spaces"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "build.cmd").write_bytes((ROOT / "scripts/build.cmd").read_bytes())
            (scripts / "bootstrap_engine.py").write_text(
                "import json, sys\nprint(json.dumps([sys.prefix, sys.argv[1:]]))\nsys.exit(1)\n",
                encoding="ascii",
            )
            venv.create(root / ".venv", with_pip=False)
            command_processor = os.environ["COMSPEC"]
            result = subprocess.run([command_processor, "/d", "/v:off", "/c", "scripts\\build.cmd"],
                                    cwd=root, env={**os.environ, "PATH": ""}, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            prefix, options = json.loads(result.stdout)
            self.assertEqual(Path(prefix), root / ".venv")
            self.assertEqual(options, ["--directory", str(root / "work/engine/bovary-llama.cpp"), "--check"])

    @staticmethod
    def tool(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"discovery fixture, never executed")
        return path

    def discover(self, root, directories, visual_studio, *, finder=FINDER, temp_dir=None):
        harness = root / "discover.cmd"
        # Keep the batch source ASCII. cmd receives path values from the wide
        # inherited environment, then Python emits ASCII-escaped JSON instead
        # of routing Unicode paths through cmd's console output code page.
        harness.write_text(
            '@echo off\ncall "%BOVARY_TEST_FINDER%" "%BOVARY_TEST_VSROOT%"\n'
            'if errorlevel 1 exit /b %errorlevel%\n'
            '"%BOVARY_TEST_PYTHON%" -I -c "import json,os; print(json.dumps('
            '{key:os.environ[key] for key in (\'CMAKE_EXE\',\'NINJA_EXE\')},ensure_ascii=True))"\n',
            encoding="ascii")
        environment = {**os.environ, "PATH": os.pathsep.join(map(str, directories)),
                       "BOVARY_TEST_FINDER": str(finder), "BOVARY_TEST_VSROOT": str(visual_studio),
                       "BOVARY_TEST_PYTHON": str(Path(sys.executable).resolve())}
        if temp_dir is not None:
            environment.update(TEMP=str(temp_dir), TMP=str(temp_dir))
        command_processor = os.environ.get("COMSPEC", str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe"))
        return subprocess.run([command_processor, "/d", "/v:off", "/c", str(harness)], cwd=root,
                              env=environment, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10)

    def assert_tools(self, result, cmake, ninja):
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(Path(values["CMAKE_EXE"]), cmake)
        self.assertEqual(Path(values["NINJA_EXE"]), ninja)

    def test_path_tools_take_precedence_over_visual_studio_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "tools with spaces"
            cmake = self.tool(tools / "cmake.exe")
            ninja = self.tool(tools / "ninja.exe")
            visual_studio = root / "Visual Studio"
            self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
            self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
            result = self.discover(root, [tools], visual_studio)
            self.assert_tools(result, cmake, ninja)

    def test_each_missing_path_tool_uses_its_visual_studio_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "path tools"
            cmake = self.tool(tools / "cmake.exe")
            visual_studio = root / "Visual Studio"
            ninja = self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
            result = self.discover(root, [tools], visual_studio)
            self.assert_tools(result, cmake, ninja)
            cmake.unlink()
            cmake = self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
            result = self.discover(root, [], visual_studio)
            self.assert_tools(result, cmake, ninja)

    def test_missing_cmake_and_ninja_have_clear_separate_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visual_studio = root / "no VS bundle"
            result = self.discover(root, [], visual_studio)
            self.assertEqual(result.returncode, 1)
            self.assertIn("CMake not found", result.stderr)
            tools = root / "path tools"
            self.tool(tools / "cmake.exe")
            result = self.discover(root, [tools], visual_studio)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Ninja not found", result.stderr)
            self.assertIn("PATH", result.stderr)

    def test_unicode_checkout_temp_and_path_tools_survive_console_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / (UNICODE_NAME + " checkout with spaces")
            finder = root / "scripts/find_build_tools.cmd"
            finder.parent.mkdir(parents=True)
            finder.write_bytes(FINDER.read_bytes())
            unicode_temp = root / (UNICODE_NAME + " TEMP with spaces")
            unicode_temp.mkdir()
            tools = root / (UNICODE_NAME + " PATH tools")
            cmake = self.tool(tools / "cmake.exe")
            ninja = self.tool(tools / "ninja.exe")
            visual_studio = root / (UNICODE_NAME + " Visual Studio")
            with tempfile.TemporaryDirectory(dir=unicode_temp) as harness_directory:
                result = self.discover(Path(harness_directory), [tools], visual_studio,
                                       finder=finder, temp_dir=unicode_temp)
                self.assert_tools(result, cmake, ninja)
                # The subprocess output is ASCII even though its values are not.
                result.stdout.encode("ascii")

    def test_unicode_visual_studio_fallback_with_empty_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / (UNICODE_NAME + " fallback with spaces")
            finder = root / "checkout/scripts/find_build_tools.cmd"
            finder.parent.mkdir(parents=True)
            finder.write_bytes(FINDER.read_bytes())
            visual_studio = root / (UNICODE_NAME + " Visual Studio")
            cmake = self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
            ninja = self.tool(visual_studio / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
            result = self.discover(root, [], visual_studio, finder=finder)
            self.assert_tools(result, cmake, ninja)


if __name__ == "__main__":
    unittest.main()
