"""Set up and run the two-model Bovary Windows kit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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


def install():
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
    install_native_dependencies()
    try:
        result = run(WORKER, "--require-cpu-only", "--protocol-version", capture_output=True, text=True)
        if result.stdout.strip() != "bovary-worker 2":
            raise ValueError("Unexpected CPU worker version.")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("The CPU worker could not start. Check that this PC supports AVX2/FMA and install the Microsoft x64 Visual C++ runtime from https://aka.ms/vs/17/release/vc_redist.x64.exe then reopen Bovary.") from error


def model_setup(use_existing=False):
    settings = STATE / "model-path.json"
    if not use_existing and settings.is_file():
        try:
            saved = json.loads(settings.read_text(encoding="utf8"))
            if not isinstance(saved, list) or len(saved) != 2:
                raise ValueError("Expected model and receipt paths.")
            if not all(isinstance(path, str) and path.strip() for path in saved):
                raise ValueError("Expected nonempty model and receipt paths.")
            model, proof = map(Path, saved)
        except ValueError:
            raise ValueError("Could not read the saved Gemma paths. Choose E to register an existing Gemma file.") from None
        if model.is_file() and proof.is_file():
            return model, proof
    print("\nGemma 4 26B is a separate 14.6 GB download. Allow about 18 GB free on this drive.")
    print("Model terms: https://ai.google.dev/gemma/docs/gemma_4_license")
    if not use_existing:
        directory = STATE / "models"
        print("Downloading and verifying. If interrupted, reopen Start Bovary and choose Gemma again.", flush=True)
        download_env = os.environ.copy()
        # The Python installer supplies this CA bundle with pip. A fresh
        # Windows profile may not yet have cached the model CDN's root.
        bundle = STATE / "env/Lib/site-packages/pip/_vendor/certifi/cacert.pem"
        if bundle.is_file():
            download_env.setdefault("SSL_CERT_FILE", str(bundle))
        run(PYTHON, RUNTIME / "tools/download_model.py", "--directory", directory, env=download_env)
        model, proof = directory / MODEL, directory / "model-provenance.json"
    else:
        model = Path(input(f"Drop {MODEL} into this window or type its path, then press Enter: ").strip().strip('"')).resolve()
        if not model.is_file():
            raise ValueError("Model file not found.")
        from uuid import uuid4
        proof = STATE / f"verified-{uuid4().hex}.json"
        print("Checking the model checksum. This may take a minute...", flush=True)
        run(PYTHON, RUNTIME / "tools/verify_model.py", "--model", model, "--manifest", proof)
    temporary = settings.with_suffix(".tmp")
    temporary.write_text(json.dumps([str(model), str(proof)]), encoding="utf8")
    temporary.replace(settings)
    return model, proof


def main():
    print("Bovary Mini for Windows", flush=True)
    STATE.mkdir(exist_ok=True)
    import msvcrt
    with (STATE / "launcher.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise ValueError("Bovary is already open from this folder.") from None
        install()
        sys.path.extend([str(ROOT / "setup"), str(RUNTIME / "previews/inkling")])
        from ask import ask as ask_inkling
        from inkling_setup import install as install_inkling
        while True:
            print("\nChoose a model to download or use. Already installed models are reused.")
            print("G  Gemma 26B: faster on our coding pilot; works on CPU. 14.6 GB download, allow 18 GB free.")
            print("I  Inkling 2 Mini: stronger on our coding pilot (4/6 vs 3/6), usually slower there.")
            print("   NVIDIA CUDA 13.5-compatible driver, local NTFS SSD. 18.3 GB download including CUDA, allow 39 GB free.")
            print("B  Both: allow 54 GB free for a fresh setup. Switch with /models.")
            print("E  Use an existing Gemma file     Q  Quit")
            print("Model terms: https://ai.google.dev/gemma/docs/gemma_4_license")
            print("Inkling model card: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF")
            choice = input("Choose G, I, B, E, or Q: ").strip().lower()
            if choice == "q":
                return
            if choice not in {"g", "i", "b", "e"}:
                continue
            try:
                if choice in {"i", "b"}:
                    install_inkling(STATE, ROOT / "setup")
                if choice in {"g", "e", "b"}:
                    model, proof = model_setup(use_existing=choice == "e")
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
            inkling = choice == "i"
            name = "Inkling 2 Mini (CUDA)" if inkling else "Gemma 26B (CPU)"
            print(f"\n{name} ready. Press Enter for a sample, type a question, /models to switch, or /quit.")
            while True:
                prompt = input("\nAsk Bovary: ").strip()
                if prompt.lower() == "/quit":
                    return
                if prompt.lower() == "/models":
                    break
                prompt = prompt or "What is 17 plus 28? Answer with just the number."
                print(f"Thinking with {name}...", flush=True)
                if inkling:
                    try:
                        print(ask_inkling(prompt, timeout=600), flush=True)
                    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
                        print(f"Inkling request failed: {error}", file=sys.stderr)
                else:
                    result = subprocess.run([PYTHON, "-I", "-m", "bovary", "mini", "--preset", "cpu-only",
                                             "--prompt", prompt], env=env)
                    if result.returncode:
                        print("This request did not finish. See the message above.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nClosed.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup or startup failed: {error}", file=sys.stderr)
        sys.exit(1)
