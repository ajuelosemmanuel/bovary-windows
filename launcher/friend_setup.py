"""Set up and run the two-model Bovary Windows kit."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "Bovary Data"
RUNTIME = STATE / "runtime"
PYTHON = STATE / "env/Scripts/python.exe"
WORKER = RUNTIME / "work/build/bovary-cpu/bovary-infer.exe"
MODEL = "gemma-4-26B-A4B-it-Q4_0.gguf"
ARCHIVE = "@CPU_ARCHIVE@"
WHEEL = "@WHEEL@"
MANIFEST_HASH = "@MANIFEST_SHA256@"
BOOTSTRAP_MARKER = "@INSTALL_MARKER@"
GEMMA_CUDA_SHA256 = "@GEMMA_CUDA_SHA256@"
def run(*args, **options):
    return subprocess.run(args, check=True, **options)


def install_native_dependencies():
    from ctypes import WinDLL
    try:
        for name in ("VCOMP140", "MSVCP140", "VCRUNTIME140", "VCRUNTIME140_1"):
            WinDLL(name + ".dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        return
    except OSError:
        pass
    print("The CPU worker needs Microsoft's x64 Visual C++ runtime.")
    if input("Download and install the signed Microsoft runtime? Windows may ask for approval [y/N]: ").strip().lower() != "y":
        raise ValueError("The CPU runtime is required. Reopen Bovary when ready.")
    with tempfile.TemporaryDirectory() as directory:
        installer = Path(directory) / "vc_redist.x64.exe"
        # Keep the filename out of the PowerShell source and check the
        # Authenticode chain and publisher before Windows runs the installer.
        run("powershell.exe", "-NoProfile", "-Command",
            "$ErrorActionPreference='Stop'; (New-Object Net.WebClient).DownloadFile('https://aka.ms/vc14/vc_redist.x64.exe',$env:BOVARY_VC_INSTALLER); "
            "$s=Get-AuthenticodeSignature -LiteralPath $env:BOVARY_VC_INSTALLER; "
            "if ($s.Status -ne 'Valid' -or $s.SignerCertificate.GetNameInfo('SimpleName',$false) -ne 'Microsoft Corporation') "
            "{ throw 'Microsoft installer signature verification failed' }; "
            "$p=Start-Process -FilePath $env:BOVARY_VC_INSTALLER -ArgumentList '/install /passive /norestart' -Wait -PassThru; "
            "if ($p.ExitCode -notin 0,3010) { throw ('Microsoft installer exit '+$p.ExitCode) }",
            env=dict(os.environ, BOVARY_VC_INSTALLER=str(installer)))


def install(*, check_native=True):
    payload = ROOT / "setup"
    manifest = (payload / "SHA256SUMS.txt").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_HASH:
        raise ValueError("The checksum file has changed. Extract a fresh copy of the ZIP.")
    checks = {name: digest for digest, name in (row.split() for row in manifest.decode().splitlines())}
    for name in (ARCHIVE, WHEEL):
        if hashlib.sha256((payload / name).read_bytes()).hexdigest() != checks[name]:
            raise ValueError(f"{name} failed verification. Extract a fresh copy of the ZIP.")
    installed = STATE / BOOTSTRAP_MARKER
    if not (installed.exists() and PYTHON.is_file() and WORKER.is_file()):
        print("Installing the verified Bovary CPU package in this folder...", flush=True)
        with zipfile.ZipFile(payload / ARCHIVE) as archive:
            archive.extractall(RUNTIME)
        run(sys.executable, "-I", "-m", "venv", STATE / "env")
        run(PYTHON, "-I", "-m", "pip", "install", "--no-index", "--no-deps", payload / WHEEL)
        installed.touch()
    if check_native:
        verify_native()


def verify_native():
    install_native_dependencies()
    try:
        result = run(WORKER, "--require-cpu-only", "--protocol-version", capture_output=True, text=True)
        if result.stdout.strip() != "bovary-worker 2":
            raise ValueError("Unexpected CPU worker version.")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("The CPU worker could not start. Check that this PC supports AVX2/FMA and install the Microsoft x64 Visual C++ runtime from https://aka.ms/vs/17/release/vc_redist.x64.exe then reopen Bovary.") from error


def write_setting(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def model_storage():
    """Return the configured directory, or the per-user default for new models."""
    preference = STATE / "model-storage.json"
    if preference.exists():
        try:
            saved = json.loads(preference.read_text(encoding="utf8"))
        except ValueError:
            saved = None
        if not isinstance(saved, str) or not Path(saved).is_absolute():
            raise ValueError("Could not read the saved model folder. Choose S to set it again.")
        return Path(saved), True
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local") / "Bovary/Models", False


def choose_storage():
    path = input("Enter a full path to a model folder, or press Enter to keep the current setting: ").strip().strip('"')
    if not path:
        return
    directory = Path(path).expanduser()
    if not directory.is_absolute():
        raise ValueError("Enter a full path, for example D:\\Bovary Models.")
    current_drive(directory)
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    write_setting(STATE / "model-storage.json", str(directory))
    print(f"Model folder saved as {directory}. Existing files stay where they are.")


def current_drive(directory):
    if not Path(directory.anchor).is_dir():
        raise ValueError(f"The model drive for {directory} is unavailable. Reconnect it or choose S for another folder.")


def verify_storage(directory, selected):
    current_drive(directory)
    if selected and not directory.is_dir():
        raise ValueError(f"The selected model folder is unavailable: {directory}. Restore it or choose S.")


def inkling_folder(storage, filename):
    directory, selected = storage
    legacy = STATE / "inkling"
    if not selected and any((legacy / name).is_file() for name in (filename, filename + ".partial", "inputs.json")):
        return legacy
    folder = directory / "inkling"
    if (directory / filename).is_file() and not any(
        (folder / name).is_file() for name in (filename, filename + ".partial", "inputs.json")
    ):
        return directory
    return folder


def verified(model, proof):
    if not (model.is_file() and proof.is_file()):
        return False
    result = subprocess.run([PYTHON, RUNTIME / "tools/verify_model.py", "--model", model,
                             "--manifest", proof, "--check"], capture_output=True)
    return result.returncode == 0


def model_setup(use_existing=False, *, storage=None):
    settings = STATE / "model-path.json"
    directory, selected = storage or model_storage()
    if selected:
        key = hashlib.sha256(os.path.normcase(str(directory)).encode("utf-8")).hexdigest()[:16]
        settings = STATE / f"model-path-{key}.json"
    verifier = RUNTIME / "tools/verify_model.py"
    if use_existing:
        model = Path(input(f"Drop {MODEL} into this window or type its path, then press Enter: ").strip().strip('"')).resolve()
        if not model.is_file():
            raise ValueError("Model file not found.")
        proof = model.parent / "model-provenance.json"
        if not verified(model, proof):
            proof = STATE / f"verified-{uuid4().hex}.json"
            print("Checking the model checksum. This may take a minute...", flush=True)
            run(PYTHON, verifier, "--model", model, "--manifest", proof)
        write_setting(settings, [str(model), str(proof)])
        return model, proof

    verify_storage(directory, selected)
    saved_paths = None
    if settings.is_file():
        try:
            saved = json.loads(settings.read_text(encoding="utf8"))
            if not isinstance(saved, list) or len(saved) != 2:
                raise ValueError("Expected model and receipt paths.")
            if not all(isinstance(path, str) and path.strip() for path in saved):
                raise ValueError("Expected nonempty model and receipt paths.")
            saved_paths = tuple(map(Path, saved))
        except ValueError:
            raise ValueError("Could not read the saved Gemma paths. Choose E to register an existing Gemma file.") from None
    legacy = STATE / "models"
    if saved_paths:
        model, proof = saved_paths
        if verified(model, proof):
            return model, proof
        if model.is_file():
            proof = STATE / f"verified-{uuid4().hex}.json"
            print("Rechecking the registered Gemma model...", flush=True)
            run(PYTHON, verifier, "--model", model, "--manifest", proof)
            write_setting(settings, [str(model), str(proof)])
            return model, proof
        moved_legacy = (not selected and model == proof.parent / MODEL
                        and model.parent.name == "models" and model.parent.parent.name == STATE.name
                        and (legacy / MODEL).is_file())
        if not moved_legacy:
            raise ValueError(f"The registered Gemma file is unavailable: {model}. Restore it, choose E, or choose S.")
    if not selected and any((legacy / name).is_file() for name in (MODEL, MODEL + ".partial")):
        directory = legacy

    model, proof = directory / MODEL, directory / "model-provenance.json"
    if verified(model, proof):
        if saved_paths:
            write_setting(settings, [str(model), str(proof)])
        return model, proof
    if model.is_file():
        proof = STATE / f"verified-{uuid4().hex}.json"
        print(f"Verifying the existing Gemma model in {directory}...", flush=True)
        run(PYTHON, verifier, "--model", model, "--manifest", proof)
        write_setting(settings, [str(model), str(proof)])
    else:
        print(f"\nGemma 4 26B is a separate 14.6 GB download to {directory}.")
        print("The downloader checks remaining space on that drive before it starts.")
        print("Model terms: https://ai.google.dev/gemma/docs/gemma_4_license")
        print("Downloading and verifying. If interrupted, reopen Start Bovary and choose Gemma again.", flush=True)
        download_env = os.environ.copy()
        # The Python installer supplies this CA bundle with pip. A fresh
        # Windows profile may not yet have cached the model CDN's root.
        bundle = STATE / "env/Lib/site-packages/pip/_vendor/certifi/cacert.pem"
        if bundle.is_file():
            download_env.setdefault("SSL_CERT_FILE", str(bundle))
        override = ()
        if proof.exists():
            proof = STATE / f"verified-{uuid4().hex}.json"
            override = ("--manifest", proof)
        run(PYTHON, RUNTIME / "tools/download_model.py", "--directory", directory, *override, env=download_env)
        if override:
            write_setting(settings, [str(model), str(proof)])
    return model, proof


@contextmanager
def exclusive_lock(path, busy_message):
    import msvcrt
    if path.is_symlink():
        raise ValueError(f"Lock file cannot be a symbolic link: {path}")
    with path.open("a+b") as lock:
        if os.fstat(lock.fileno()).st_nlink > 1:
            raise ValueError(f"Lock file cannot be a shared hard link: {path}")
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise ValueError(busy_message) from None
        yield


@contextmanager
def gemma_environment(environment):
    keys = ("BOVARY_HOME", "BOVARY_MODEL", "BOVARY_MODEL_MANIFEST", "BOVARY_CUDA_EXE", "PATH")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: environment[key] for key in keys if key in environment})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def menu():
    sys.path.extend([str(ROOT / "setup"), str(RUNTIME / "previews/inkling")])
    from ask import ask as ask_inkling
    from bovary.checkpoint import PinnedGemma
    from bovary.cli import configure_input, configure_output
    from bovary.launcher_chat import SWITCH_MODEL, chat
    from inkling_setup import MODEL as INKLING_MODEL, gemma_cuda_environment, install as install_inkling
    configure_input()
    configure_output()
    verify_native()
    while True:
        print("\nChoose a model to download or use. Already installed models are reused.")
        print("G  Gemma 26B (default): prefer NVIDIA CUDA when available, otherwise CPU.")
        print("   14.6 GB model, plus 426 MB for CUDA on first use. Allow 20 GB free.")
        print("I  Inkling 2 Mini: stronger on our coding pilot (4/6 vs 3/6), usually slower there.")
        print("   NVIDIA CUDA 13.5-compatible driver, local NTFS SSD. 18.3 GB download including CUDA, allow 39 GB free.")
        print("B  Both: allow 54 GB free for a fresh setup. Switch with /models.")
        print("C  Gemma on CPU only     E  Use an existing Gemma file")
        print("S  Model storage folder  T  Test my laptop                Q  Quit")
        try:
            directory, selected = model_storage()
            print(f"Model storage: {directory}" + ("" if selected else " (new downloads; legacy files are reused)"))
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
        print("Model terms: https://ai.google.dev/gemma/docs/gemma_4_license")
        print("Inkling model card: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF")
        choice = input("Choose G, I, B, C, E, S, T, or Q [G]: ").strip().lower() or "g"
        if choice == "q":
            return
        if choice not in {"g", "i", "b", "c", "e", "s", "t"}:
            continue
        try:
            if choice == "s":
                choose_storage()
                continue
            storage = model_storage()
            directory, selected = storage
            if choice != "e":
                verify_storage(directory, selected)
            if choice in {"i", "b"}:
                folder = inkling_folder(storage, INKLING_MODEL["filename"])
                folder.mkdir(parents=True, exist_ok=True)
                with exclusive_lock(folder / ".bovary-inkling.lock", "Another launcher is preparing Inkling in this folder. Retry when it finishes."):
                    install_inkling(STATE, ROOT / "setup", model_folder=folder)
            if choice in {"g", "e", "b", "c", "t"}:
                model, proof = model_setup(use_existing=choice == "e", storage=storage)
                env = dict(os.environ, BOVARY_HOME=str(RUNTIME), BOVARY_MODEL=str(model),
                           BOVARY_MODEL_MANIFEST=str(proof))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"Setup could not finish: {error}", file=sys.stderr)
            continue
        if choice == "b":
            print("Both models are installed.")
            choice = input("Use [G]emma or [I]nkling now? ").strip().lower()
            if choice not in {"g", "i"}:
                continue
        preset = "cpu-only"
        if choice not in {"i", "c"}:
            try:
                cuda_env = gemma_cuda_environment(STATE, ROOT / "setup", env, GEMMA_CUDA_SHA256)
                if cuda_env is not None:
                    env, preset = cuda_env, "gpu-prefill-2g"
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
                print(f"Gemma CUDA setup unavailable: {error}. Using CPU.", file=sys.stderr)
        if choice == "t":
            print("The local test uses built-in prompts and sends no data. Model setup may require downloads.")
            result = subprocess.run([PYTHON, "-I", "-m", "bovary.laptop_benchmark", "--preset", preset],
                                    env=env, stdin=subprocess.DEVNULL)
            if result.returncode not in {0, 2}:
                print("The laptop test did not finish. See the message above.", file=sys.stderr)
            continue
        if choice != "i":
            with gemma_environment(env), PinnedGemma() as worker:
                outcome = chat(worker, preset)
            if outcome == SWITCH_MODEL:
                continue
            return
        print("\nInkling 2 Mini (CUDA). Each question starts fresh and releases the model afterward.")
        print("Press Enter for a sample, type a question, /models to switch, or /quit.")
        while True:
            prompt = input("\nAsk Bovary: ").strip()
            if prompt.lower() == "/quit":
                return
            if prompt.lower() == "/models":
                break
            if prompt.lower() in {"/keep", "/new", "/unload"}:
                print("Inkling releases the model after each question. Choose Gemma for a conversation.")
                continue
            prompt = prompt or "What is 17 plus 28? Answer with just the number."
            print("Thinking with Inkling 2 Mini...", flush=True)
            try:
                print(ask_inkling(prompt, timeout=600), flush=True)
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"Inkling request failed: {error}", file=sys.stderr)


def main():
    print("Bovary Mini for Windows", flush=True)
    STATE.mkdir(exist_ok=True)
    with exclusive_lock(STATE / "launcher.lock", "Bovary is already open from this folder."):
        install(check_native=False)
        return subprocess.run([PYTHON, "-I", __file__, "--installed-menu"]).returncode


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--installed-menu"] and Path(sys.prefix).resolve() == PYTHON.parent.parent:
            menu()
        else:
            sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nClosed.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup or startup failed: {error}", file=sys.stderr)
        sys.exit(1)
