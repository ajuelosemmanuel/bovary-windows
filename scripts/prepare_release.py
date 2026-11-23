"""Create review-only source/CPU archives from an explicit public allowlist."""
import argparse
import base64
import configparser
import csv
from email.parser import BytesParser
from fnmatch import fnmatchcase
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import stat
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"work", "outputs", "dist", "build", ".git", ".venv", ".aws", ".codex", ".agents", "__pycache__"}
ALLOWED_SUFFIXES = {".py", ".cpp", ".h", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cmd", ".ps1", ".patch", ".gitignore", ".svg", ".png"}
SECRET = re.compile(rb"(?<![A-Za-z0-9_-])(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
UTF16_SECRET = re.compile(
    rb"s\0k\0-\0(?:p\0r\0o\0j\0-\0)?(?:[A-Za-z0-9_-]\0){24,}"
    rb"|\0s\0k\0-(?:\0p\0r\0o\0j\0-)?(?:\0[A-Za-z0-9_-]){24,}"
    rb"|h\0f\0_\0(?:[A-Za-z0-9]\0){24,}|\0h\0f\0_(?:\0[A-Za-z0-9]){24,}"
    + b"|" + b"|".join(re.escape(('-----BEGIN '+kind+'PRIVATE KEY-----').encode(encoding))
                       for kind in ('RSA ', 'EC ', 'OPENSSH ', '') for encoding in ('utf-16le','utf-16be')))
PRIVATE_PATH = re.compile(rb"[A-Za-z]:[/\\]+Users[/\\]+", re.I)
MAX_WHEEL_BYTES = 8 * 1024 * 1024
MAX_WHEEL_ENTRY_BYTES = 2 * 1024 * 1024
MAX_WHEEL_ENTRIES = 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def project_metadata(root=ROOT):
    project = tomllib.loads((Path(root) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", project["name"]):
        raise ValueError("Unsafe distribution name in pyproject.toml")
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", project["version"]):
        raise ValueError("Unsafe distribution version in pyproject.toml")
    return project


def distribution_component(name):
    return re.sub(r"[-_.]+", "_", name).lower()


def read_wheel_bytes(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_WHEEL_BYTES + 1)
    if len(data) > MAX_WHEEL_BYTES:
        raise ValueError("Wheel archive exceeds the 8 MiB review bound")
    return data


def wheel_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name or "\\" in name
            or ":" in name or any(ord(character) < 32 or ord(character) == 127 for character in name)
            or any(part in (".", "..") or part.endswith((" ", ".")) for part in path.parts)):
        raise ValueError("Unsafe wheel member path")
    for part in path.parts:
        if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", part.split(".")[0], re.I):
            raise ValueError("Wheel member uses a reserved Windows filename")
    return name


def wheel_headers(data, label):
    message = BytesParser().parsebytes(data)
    if message.defects:
        raise ValueError(f"Malformed wheel {label} headers")
    return message


def header_equals(message, name, value):
    if message.get_all(name, []) != [value]:
        raise ValueError(f"Wheel {name} differs from the reviewed metadata")


def check_wheel(path, files, project):
    """Check the project's pure-Python wheel, including every byte and metadata file."""
    path = Path(path)
    component = distribution_component(project["name"])
    stem = component + "-" + project["version"]
    if path.name != stem + "-py3-none-any.whl":
        raise ValueError("Expected the canonical project-version-py3-none-any.whl filename")
    raw = read_wheel_bytes(path)
    # Comments, extra fields, ZIP headers and tolerated trailing bytes also ship
    # to consumers. Member-content checks below cover compressed payloads.
    if has_private_payload(raw):
        raise ValueError("Possible credential or private user path in wheel archive metadata")
    package = {name.removeprefix("python/"): data for name, data in files.items() if name.startswith("python/")}
    info = stem + ".dist-info/"
    licenses = {name for name in files if any(fnmatchcase(name, pattern) for pattern in project.get("license-files", []))}
    required = {info + name for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")}
    allowed = set(package) | required | {info + "licenses/" + name for name in licenses}
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        footer = raw.rfind(b"PK\x05\x06", max(0, len(raw) - 22 - 65535))
        if footer < 0 or footer + 22 > len(raw):
            raise ValueError("Malformed wheel archive footer")
        comment_size = int.from_bytes(raw[footer+20:footer+22], "little")
        if footer + 22 + comment_size != len(raw):
            raise ValueError("Wheel archive has unsupported trailing bytes")
        members = wheel.infolist()
        if not members or len(members) > MAX_WHEEL_ENTRIES:
            raise ValueError("Wheel member count exceeds the review bound")
        if has_private_payload(wheel.comment):
            raise ValueError("Possible credential or private user path in wheel archive metadata")
        for member in members:
            local = member.header_offset
            if local < 0 or local + 30 > len(raw) or raw[local:local+4] != b"PK\x03\x04":
                raise ValueError("Malformed wheel local header")
            name_size = int.from_bytes(raw[local+26:local+28], "little")
            extra_size = int.from_bytes(raw[local+28:local+30], "little")
            extra_start = local + 30 + name_size
            extra_end = extra_start + extra_size
            if extra_end > len(raw) or raw[extra_start:extra_end] != member.extra:
                raise ValueError("Wheel local and central extra fields differ")
            if has_private_payload(member.comment) or has_private_payload(member.extra):
                raise ValueError("Possible credential or private user path in wheel archive metadata")
            # ZIP fields abut filenames or binary length bytes. Scan each extra
            # value separately so those bytes cannot hide a credential boundary.
            offset = 0
            while offset < len(member.extra):
                if offset + 4 > len(member.extra):
                    raise ValueError("Malformed wheel archive extra field")
                size = int.from_bytes(member.extra[offset+2:offset+4], "little")
                end = offset + 4 + size
                if end > len(member.extra):
                    raise ValueError("Malformed wheel archive extra field")
                if has_private_payload(member.extra[offset+4:end]):
                    raise ValueError("Possible credential or private user path in wheel archive metadata")
                offset = end
        names = [wheel_path(member.filename) for member in members]
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("Wheel contains duplicate or case-colliding entries")
        if set(names) != allowed:
            raise ValueError("Wheel file inventory differs from the reviewed package and metadata")
        if sum(member.file_size for member in members) > MAX_WHEEL_BYTES:
            raise ValueError("Expanded wheel exceeds the 8 MiB review bound")
        content = {}
        for member in members:
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (member.is_dir() or member.orig_filename != member.filename or mode not in (0, stat.S_IFREG)
                    or member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("Wheel contains a linked, encrypted or unsupported member")
            if not 0 <= member.file_size <= MAX_WHEEL_ENTRY_BYTES:
                raise ValueError("Wheel member exceeds the 2 MiB review bound")
            with wheel.open(member) as source:
                data = source.read(MAX_WHEEL_ENTRY_BYTES + 1)
            if len(data) != member.file_size or len(data) > MAX_WHEEL_ENTRY_BYTES:
                raise ValueError("Wheel member size disagrees with its archive metadata")
            if has_private_payload(data):
                raise ValueError("Possible credential or private user path in wheel payload")
            content[member.filename] = data
    if any(content[name] != data for name, data in package.items()):
        raise ValueError("Wheel package files differ from the public source; rebuild the wheel")
    if any(content[info + "licenses/" + name] != files[name] for name in licenses):
        raise ValueError("Wheel license files differ from the public source")

    check_package_metadata(content[info + "METADATA"], files, project)
    wheel_metadata = wheel_headers(content[info + "WHEEL"], "WHEEL")
    for name, value in {"Wheel-Version": "1.0", "Root-Is-Purelib": "true", "Tag": "py3-none-any"}.items():
        header_equals(wheel_metadata, name, value)
    if (set(key.lower() for key in wheel_metadata.keys()) != {"wheel-version", "root-is-purelib", "tag", "generator"}
            or len(wheel_metadata.get_all("Generator", [])) != 1 or not wheel_metadata["Generator"].strip()
            or wheel_metadata.get_payload().strip()):
        raise ValueError("Wheel contains unsupported format, build or tag metadata")
    check_entry_points(content[info + "entry_points.txt"], project)
    check_top_level(content[info + "top_level.txt"], files)

    record_name = info + "RECORD"
    try:
        rows = list(csv.reader(io.StringIO(content[record_name].decode("utf-8")), strict=True))
    except csv.Error as error:
        raise ValueError("Malformed wheel RECORD CSV") from error
    recorded = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError("Wheel RECORD rows require path, SHA256 and size")
        name, encoded, size = row
        wheel_path(name)
        if name in recorded or name not in content:
            raise ValueError("Wheel RECORD contains duplicate or unlisted paths")
        recorded.add(name)
        if name == record_name:
            if encoded or size:
                raise ValueError("Wheel RECORD must leave its own hash and size empty")
        else:
            expected_hash = base64.urlsafe_b64encode(hashlib.sha256(content[name]).digest()).rstrip(b"=").decode("ascii")
            if encoded != "sha256=" + expected_hash or size != str(len(content[name])):
                raise ValueError("Wheel RECORD hash or size differs from its payload")
    if recorded != set(content):
        raise ValueError("Wheel RECORD does not inventory every member")
    return {"filename": path.name, "sha256": digest(raw), "entries": len(content),
            "expanded_bytes": sum(map(len, content.values())), "tag": "py3-none-any", "record_verified": True}


def check_package_metadata(metadata_bytes, files, project):
    """Validate wheel METADATA or sdist PKG-INFO against reviewed project inputs."""
    metadata = wheel_headers(metadata_bytes, "package metadata")
    licenses = {name for name in files if any(fnmatchcase(name, pattern) for pattern in project.get("license-files", []))}
    expected_headers = {"Metadata-Version": "2.4", "Name": project["name"], "Version": project["version"],
                        "Summary": project["description"], "Requires-Python": project["requires-python"],
                        "License-Expression": project["license"], "Description-Content-Type": "text/markdown"}
    for name, value in expected_headers.items():
        header_equals(metadata, name, value)
    if set(key.lower() for key in metadata.keys()) - {key.lower() for key in expected_headers} - {"license-file", "dynamic"}:
        raise ValueError("Wheel contains unreviewed package metadata headers")
    if metadata.get_all("Dynamic", []) not in ([], ["license-file"]):
        raise ValueError("Wheel declares unreviewed dynamic metadata")
    notices = metadata.get_all("License-File", [])
    if len(notices) != len(set(notices)) or set(notices) != licenses:
        raise ValueError("Wheel license-file inventory differs from pyproject.toml")
    description = metadata_bytes.decode("utf-8").replace("\r\n", "\n").partition("\n\n")[2].rstrip()
    readme = files[project["readme"]].decode("utf-8").replace("\r\n", "\n").rstrip()
    if description != readme:
        raise ValueError("Wheel README metadata differs from the public source; rebuild the wheel")


def check_entry_points(data, project):
    entries = configparser.ConfigParser(interpolation=None)
    entries.optionxform = str
    entries.read_string(data.decode("utf-8"))
    if entries.defaults() or entries.sections() != ["console_scripts"] or dict(entries["console_scripts"]) != project["scripts"]:
        raise ValueError("Wheel console entry points differ from pyproject.toml")


def check_top_level(data, files):
    roots = {name.removeprefix("python/").split("/")[0] for name in files if name.startswith("python/")}
    if data.decode("utf-8").strip().splitlines() != sorted(roots):
        raise ValueError("Wheel top-level package inventory differs from the source")

def digest(data):
    return hashlib.sha256(data).hexdigest()


def has_private_build_paths(data):
    return bool(PRIVATE_PATH.search(data) or PRIVATE_PATH.search(data.replace(b"\0", b"")))


def has_private_payload(data):
    """Scan bounded payloads for ASCII/UTF-16 credential and user-path shapes.

    Removing NUL bytes also covers either UTF-16 byte order without decoding
    arbitrary binary payloads or creating an unbounded text expansion.
    """
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("Payload exceeds the 64 MiB privacy scan bound")
    if SECRET.search(data) or PRIVATE_PATH.search(data) or UTF16_SECRET.search(data):
        return True
    without_nuls = data.replace(b"\0", b"")
    return bool(SECRET.search(without_nuls) or PRIVATE_PATH.search(without_nuls))


def check_cpu_capture(manifest, root=ROOT):
    root = Path(root).resolve()
    if manifest.get("smoke", {}).get("passed") is not True:
        raise ValueError("CPU build manifest lacks a successful native smoke check")
    home = manifest.get("build_options", {}).get("CMAKE_HOME_DIRECTORY")
    if not isinstance(home, str) or not home.strip() or not Path(home).is_absolute() or Path(home).resolve() != root:
        raise ValueError("CPU build manifest belongs to a missing or different source tree")
    freshness = manifest.get("build_freshness", {})
    targets = freshness.get("targets")
    if (freshness.get("tool") != "ninja dry-run" or freshness.get("up_to_date") is not True
            or not isinstance(targets, list) or "bovary-infer.exe" not in targets):
        raise ValueError("CPU build manifest lacks the inference target dependency freshness proof")
    snapshots = {row["source"]: row for row in manifest.get("source_snapshots", [])}
    for name in ("src/infer.cpp", "src/quantize.cpp", "src/native_paths.h", "CMakeLists.txt",
                 "scripts/build.cmd", "scripts/find_build_tools.cmd"):
        if name not in snapshots or snapshots[name].get("sha256") != digest((root / name).read_bytes()):
            raise ValueError("CPU build manifest has missing or stale build-source snapshots")


def check_cpu_payload(data, options):
    if options.get("BOVARY_SANITIZE_PATHS") != "ON" or has_private_payload(data):
        raise ValueError("CPU binary contains credentials/private build paths or lacks the sanitized-build flag; keep local binaries private")


def check_tracked(root, public_names):
    """Require a source checkout to track exactly its reviewed public boundary."""
    root = Path(root).resolve()
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise ValueError("--check-tracked requires the project root to be a Git checkout")
    result = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "-z"], capture_output=True)
    if result.returncode:
        raise ValueError("Could not read the source checkout's tracked-file list")
    tracked = set(result.stdout.decode("utf-8").rstrip("\0").split("\0")) if result.stdout else set()
    extra, missing = sorted(tracked - set(public_names)), sorted(set(public_names) - tracked)
    if extra or missing:
        raise ValueError(f"Tracked files differ from public-files.txt; unlisted: {extra}; not tracked: {missing}")


def collect(root, allowlist):
    files = {}
    names = [line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError("Duplicate or case-colliding public allowlist entries")
    root = root.resolve()
    for name in names:
        try:
            wheel_path(name)
        except ValueError as error:
            raise ValueError(f"Unsafe public path: {name}") from error
        path = PurePosixPath(name)
        if any(part.casefold() in FORBIDDEN for part in path.parts):
            raise ValueError(f"Unsafe public path: {name}")
        source = root / name
        if not source.is_file() or source.is_symlink() or source.resolve() != source:
            raise ValueError(f"Missing or linked public source: {name}")
        if (
            source.suffix.lower() not in ALLOWED_SUFFIXES
            and source.name not in {"LICENSE", "MANIFEST.in", ".gitignore", ".gitattributes"}
            and name not in {"docs/images/bovary-mini-repair-demo.gif", "launcher/windows-setup.iss"}
        ):
            raise ValueError(f"Unapproved public source type: {name}")
        with source.open("rb") as stream:
            data = stream.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError(f"Oversized public source: {name}")
        if has_private_payload(data):
            raise ValueError(f"Possible credential/private absolute user path in {name}; inspect before publication")
        files[name] = data
    if not files:
        raise ValueError("Public allowlist is empty")
    return files


def archive(path, files):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)


def first_commit_diff(files):
    """Produce a real binary-capable Git patch without creating a repository."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "empty").mkdir()
        for name, data in files.items():
            target = root / "public" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        result = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--no-ext-diff", "--no-textconv", "--binary", "--", "empty", "public"],
                                cwd=root, capture_output=True)
        if result.returncode not in (0, 1):
            raise ValueError("Git could not create the public review diff")
        lines = result.stdout.decode("utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.startswith(("diff --git ", "--- ", "+++ ")):
                lines[index] = line.replace("a/public/", "a/").replace("b/public/", "b/")
        return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-wheel", type=Path, help="Check a built wheel without creating review artifacts")
    parser.add_argument("--check-tracked", action="store_true", help="Also require Git-tracked files to match the public allowlist exactly")
    parser.add_argument("--cpu-build-manifest", type=Path, help="Optional current build manifest; adds CPU review artifact without bundling DLLs")
    parser.add_argument("--wheel", type=Path, help="Optional freshly built Python wheel; its package files must match the public source")
    args = parser.parse_args()
    try:
        files = collect(ROOT, ROOT / "public-files.txt")
        project = project_metadata()
        if args.check_tracked:
            check_tracked(ROOT, files)
        entries = [{"path": name, "bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())]
        if args.check_wheel:
            if args.output_dir or args.wheel or args.cpu_build_manifest:
                parser.error("--check-wheel cannot be combined with artifact output options")
            checked = check_wheel(args.check_wheel, files, project)
            print(json.dumps({"public_files": len(entries), "wheel": checked}))
            return
        if args.check:
            print(json.dumps({"public_files": len(entries), "bytes": sum(x["bytes"] for x in entries), "sanity_scan_passed": True}))
            return
        if args.output_dir is None:
            parser.error("--output-dir is required unless --check")
        wheel_data, checked_wheel = None, None
        if args.wheel:
            checked_wheel = check_wheel(args.wheel, files, project)
            wheel_data = read_wheel_bytes(args.wheel)
            if digest(wheel_data) != checked_wheel["sha256"]:
                raise ValueError("Wheel changed during validation; retry with a completed build")
        prefix = re.sub(r"[-_.]+", "-", project["name"]).lower() + "-" + project["version"]
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
        archive(out / (prefix + "-source-review.zip"), files)
        (out / "proposed-first-commit.diff").write_text(first_commit_diff(files), encoding="utf-8", newline="")
        provenance = {"schema_version": 1, "purpose": "private review; not published", "project": {"name": project["name"], "version": project["version"], "license_expression": project.get("license")}, "source_files": entries,
                      "source_tree_sha256": digest(json.dumps(entries, sort_keys=True).encode()),
                      "license_status": "project code licensed under " + project.get("license", "unspecified") + "; third-party terms remain separate", "models_bundled": False,
                      "engine": json.loads((ROOT / "engine.lock.json").read_text())}
        if args.wheel:
            (out / args.wheel.name).write_bytes(wheel_data)
            provenance["wheel"] = {**checked_wheel, "native_executable_bundled": False}
        if args.cpu_build_manifest:
            manifest = json.loads(args.cpu_build_manifest.read_text())
            check_cpu_capture(manifest)
            binary = Path(manifest["binary"]["path"]).resolve()
            if not binary.is_relative_to((ROOT / "work/build").resolve()):
                raise ValueError("CPU artifact must come from an identified project build directory")
            with binary.open("rb") as stream:
                data = stream.read(MAX_PAYLOAD_BYTES + 1)
            options = manifest.get("build_options", {})
            if options.get("BOVARY_CPU_ONLY") != "ON" or options.get("GGML_CUDA") != "OFF":
                raise ValueError("Manifest must prove CPU-only build with CUDA disabled")
            check_cpu_payload(data, options)
            if manifest["binary"]["sha256"] != digest(data) or manifest["application_source_sha256"] != digest((ROOT / "src/infer.cpp").read_bytes()):
                raise ValueError("CPU build manifest is stale; capture a new manifest after building")
            if manifest["root_cmake_sha256"] != digest((ROOT / "CMakeLists.txt").read_bytes()):
                raise ValueError("CMake build manifest is stale")
            if (manifest.get("engine_commit") != provenance["engine"]["revision"]
                    or manifest.get("engine_diff_sha256") != provenance["engine"]["patch_sha256"]):
                raise ValueError("CPU build manifest differs from the pinned patched engine")
            portable = {**files, "work/build/bovary-cpu/bovary-infer.exe": data}
            archive(out / (prefix + "-cpu-review.zip"), portable)
            provenance["cpu"] = {"sha256": digest(data), "bytes": len(data), "source_sha256": manifest["application_source_sha256"],
                                 "private_build_path_scan_passed": True, "path_mapping": "MSVC /experimental:deterministic and /pathmap",
                                 "runtime_prerequisites": ["Windows x64 AVX2", "Python 3.11+ with Tk", "Microsoft VC++ x64 runtime including OpenMP"],
                                 "bundled_dlls": [], "clean_machine_verified": False}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        checksums = [f"{digest(path.read_bytes())}  {path.name}" for path in sorted(out.iterdir()) if path.is_file()]
        (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "files": len(entries), "published": False}))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, configparser.Error) as error:
        parser.exit(2, f"Release preparation: {error}\n")


if __name__ == "__main__":
    main()
