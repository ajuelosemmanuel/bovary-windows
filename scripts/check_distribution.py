"""Check the explicit source manifest and a built sdist against public-files.txt."""
import argparse
import configparser
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_release import (collect, project_metadata, distribution_component,
                                     check_package_metadata, check_entry_points, check_top_level,
                                     wheel_path, has_private_payload)

MAX_SDIST_BYTES = 8 * 1024 * 1024
MAX_SDIST_MEMBERS = 1024

def generated_metadata(root):
    name = distribution_component(project_metadata(root)["name"])
    return {"PKG-INFO", "setup.cfg"} | {
        "python/" + name + ".egg-info/" + entry for entry in
        ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "top_level.txt")
    }


def manifest_text(names):
    if any(any(character.isspace() or character in "*?[]" for character in name) for name in names):
        raise ValueError("Public source paths must be literal manifest paths without whitespace or glob characters")
    header = "# Generated from public-files.txt by python scripts/check_distribution.py --write-manifest.\n"
    return header + "".join("include " + name + "\n" for name in sorted(names))


def check_manifest(root=ROOT):
    root = Path(root)
    files = collect(root, root / "public-files.txt")
    if (root / "MANIFEST.in").read_text(encoding="utf-8") != manifest_text(files):
        raise ValueError("MANIFEST.in differs from public-files.txt; run python scripts/check_distribution.py --write-manifest")
    return files


def checked_sdist_files(archive, root=ROOT):
    """Read and validate the pinned backend's sdist without extracting its paths."""
    expected = check_manifest(root)
    project = project_metadata(root)
    actual, prefixes = {}, set()
    with Path(archive).open('rb') as stream:
        raw = stream.read(MAX_SDIST_BYTES + 1)
    if len(raw) > MAX_SDIST_BYTES:
        raise ValueError('Source archive exceeds the 8 MiB bound')
    if has_private_payload(raw):
        raise ValueError('Source archive wrapper has private payload')
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as compressed:
        tar_bytes = compressed.read(MAX_SDIST_BYTES + 1)
    if len(tar_bytes) > MAX_SDIST_BYTES:
        raise ValueError('Expanded source archive exceeds the 8 MiB bound')
    # Inspect TAR/PAX/global headers and padding as well as file contents. The
    # expanded buffer is bounded before either scanning or parsing its metadata.
    if has_private_payload(tar_bytes):
        raise ValueError('Expanded source archive has private payload')
    seen, expanded = set(), 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as source:
        for count, member in enumerate(source, 1):
            if count > MAX_SDIST_MEMBERS:
                raise ValueError('Source archive member count exceeds the review bound')
            metadata = [member.name, member.linkname, member.uname, member.gname]
            metadata.extend(member.pax_headers)
            metadata.extend(member.pax_headers.values())
            if any(has_private_payload(value.encode('utf-8', errors='surrogateescape')) for value in metadata):
                raise ValueError('Source archive metadata has private payload')
            # Directory entries may carry the conventional final slash.
            name = member.name.rstrip('/') if member.isdir() else member.name
            wheel_path(name)
            if name.casefold() in seen:
                raise ValueError('Duplicate or case-colliding source archive path')
            seen.add(name.casefold())
            prefix, _, relative = name.partition('/')
            prefixes.add(prefix)
            if member.isdir():
                continue
            if (not member.isfile() or member.issparse() or not relative
                    or not 0 <= member.size <= 2_000_000):
                raise ValueError("Unsupported source archive member")
            expanded += member.size
            if expanded > MAX_SDIST_BYTES:
                raise ValueError('Expanded source archive exceeds the 8 MiB bound')
            data = source.extractfile(member).read(2_000_001)
            if len(data) != member.size or has_private_payload(data):
                raise ValueError('Source archive has invalid size or private payload')
            actual[relative] = data
    canonical_prefix = distribution_component(project['name']) + '-' + project['version']
    if prefixes != {canonical_prefix}:
        raise ValueError("Source archive must have the canonical project-version directory")
    generated = generated_metadata(root)
    missing, extra = sorted((set(expected) | generated) - set(actual)), sorted(set(actual) - set(expected) - generated)
    if missing or extra:
        raise ValueError(f"Source archive differs from public boundary; missing: {missing}; unlisted: {extra}")
    mismatches = [name for name, data in expected.items() if actual[name] != data]
    if mismatches:
        raise ValueError(f"Source archive changes reviewed bytes: {mismatches}")
    egg = 'python/' + distribution_component(project['name']) + '.egg-info/'
    for name in ('PKG-INFO', egg + 'PKG-INFO'):
        check_package_metadata(actual[name], expected, project)
    if actual['PKG-INFO'].replace(b'\r\n', b'\n') != actual[egg + 'PKG-INFO'].replace(b'\r\n', b'\n'):
        raise ValueError('Source archive package metadata copies disagree')
    check_entry_points(actual[egg + 'entry_points.txt'], project)
    check_top_level(actual[egg + 'top_level.txt'], expected)
    if actual[egg + 'dependency_links.txt'].strip():
        raise ValueError('Source archive contains unexpected dependency links')
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(actual['setup.cfg'].decode('utf-8'))
    if (config.defaults() or config.sections() != ['egg_info']
            or dict(config['egg_info']) != {'tag_build':'', 'tag_date':'0'}):
        raise ValueError('Source archive setup.cfg contains unreviewed build settings')
    sources = actual[egg + 'SOURCES.txt'].decode('utf-8').splitlines()
    if (len(sources) != len(set(sources))
            or set(sources) != set(actual) - {'PKG-INFO', 'setup.cfg'}):
        raise ValueError('Source archive SOURCES.txt differs from its complete source inventory')
    return expected, actual


def check_sdist(archive, root=ROOT):
    expected, actual = checked_sdist_files(archive, root)
    return {"public_files": len(expected), "public_bytes": sum(map(len, expected.values())),
            "generated_metadata_files": len(set(actual) - set(expected)), "source_bytes_match": True,
            "generated_metadata_verified": True}


def extract_sdist(archive, destination, root=ROOT):
    """Write only validated regular-file bytes into a new source directory."""
    _, actual = checked_sdist_files(archive, root)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in actual.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--extract-to", type=Path, help="Validate --sdist and extract reviewed bytes to a new source directory")
    args = parser.parse_args()
    try:
        if args.write_manifest:
            names = [line.strip() for line in (ROOT / "public-files.txt").read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")]
            (ROOT / "MANIFEST.in").write_text(manifest_text(names), encoding="utf-8", newline="")
        if args.extract_to and not args.sdist:
            parser.error('--extract-to requires --sdist')
        files = check_manifest()
        result = check_sdist(args.sdist) if args.sdist else {"public_files": len(files), "manifest_matches": True}
        if args.extract_to:
            result['extracted_source'] = str(extract_sdist(args.sdist, args.extract_to))
        print(json.dumps(result))
    except (OSError, ValueError, KeyError, configparser.Error, tarfile.TarError) as error:
        parser.exit(2, f"Distribution check: {error}\n")


if __name__ == "__main__":
    main()
