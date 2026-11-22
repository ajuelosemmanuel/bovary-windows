"""Bind experimental configurations to captured build and verified model identities."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.modelproof import check_binding


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load_build(path, backend, root, executable_name='bovary-infer.exe'):
    root = Path(root).resolve()
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    options = manifest.get('build_options', {})
    if manifest.get('smoke', {}).get('passed') is not True:
        raise ValueError('Build manifest does not contain a successful native smoke check')
    home = options.get('CMAKE_HOME_DIRECTORY')
    if not isinstance(home, str) or not home.strip() or Path(home).resolve() != root:
        raise ValueError('Build manifest belongs to another source tree')
    expected_cpu = 'ON' if backend == 'cpu' else 'OFF'
    if (manifest.get('backend') != backend or options.get('BOVARY_CPU_ONLY') != expected_cpu
            or options.get('GGML_CUDA') != ('OFF' if backend == 'cpu' else 'ON')):
        raise ValueError('Build manifest does not prove the requested backend')
    lock = json.loads((root/'engine.lock.json').read_text(encoding='utf-8'))
    if manifest.get('engine_commit') != lock['revision'] or manifest.get('engine_diff_sha256') != lock['patch_sha256']:
        raise ValueError('Build manifest does not match the current engine lock')
    checks = {'application_source_sha256':'src/infer.cpp', 'root_cmake_sha256':'CMakeLists.txt',
              'build_script_sha256':'scripts/build.cmd'}
    for key, name in checks.items():
        if manifest.get(key) != digest(root/name):
            raise ValueError(f'Stale build manifest: {name}')
    snapshots = {row['source']: row for row in manifest.get('source_snapshots', [])}
    for name in ('src/infer.cpp', 'src/quantize.cpp', 'src/native_paths.h', 'CMakeLists.txt',
                 'scripts/build.cmd', 'scripts/find_build_tools.cmd'):
        if name not in snapshots or snapshots[name].get('sha256') != digest(root/name):
            raise ValueError(f'Missing or stale build source identity: {name}')
    entries = manifest.get('executables', [manifest.get('binary', {})])
    matches = [row for row in entries if Path(row.get('path', '')).name == executable_name]
    if len(matches) != 1:
        raise ValueError('Manifest must identify exactly one requested executable')
    row = matches[0]
    freshness = manifest.get('build_freshness', {})
    if freshness.get('up_to_date') is not True or executable_name not in freshness.get('targets', []):
        raise ValueError('Manifest lacks a successful no-build dependency freshness check')
    binary = Path(row['path']).resolve()
    if not binary.is_file() or row.get('bytes') != binary.stat().st_size or row.get('sha256') != digest(binary):
        raise ValueError('Build executable no longer matches its manifest')
    target_source = 'src/quantize.cpp' if executable_name == 'bovary-quantize.exe' else 'src/infer.cpp'
    metadata = {'provenance_status':'verified_build_manifest', 'build_manifest_sha256':digest(manifest_path),
                'native_exe_sha256':row['sha256'], 'source_commit':manifest['engine_commit'],
                'stream_patch_sha256':manifest['engine_diff_sha256'], 'native_source_sha256':snapshots[target_source]['sha256'],
                'native_source_file':target_source,
                'execution_backend':backend, 'build_options':options, 'driver':None,
                'driver_evidence':'Not queried by configuration generation; retain measured runtime inventory separately.'}
    return binary, metadata


def verified_model(root):
    root = Path(root)
    proof_path = Path(os.environ.get('BOVARY_MODEL_MANIFEST') or 'work/models/model-provenance.json').expanduser()
    manifest = json.loads((root / proof_path).read_text(encoding='utf-8-sig'))
    if (manifest.get('verified') is not True or type(manifest.get('bytes')) is not int or manifest['bytes'] < 1
            or not isinstance(manifest.get('sha256'), str) or not re.fullmatch(r'[a-fA-F0-9]{64}', manifest['sha256'])
            or not isinstance(manifest.get('filename'), str) or not manifest['filename']):
        raise ValueError('A completed verified model manifest is required')
    override = os.environ.get('BOVARY_MODEL')
    model = root / (Path(override).expanduser() if override else Path('work/models') / Path(manifest['filename']).name)
    if not model.is_file() or model.stat().st_size != manifest['bytes']:
        raise ValueError('Model size differs from its verification manifest')
    binding = check_binding(model, manifest)
    return model, {'model_sha256':manifest['sha256'], 'model_bytes':manifest['bytes'],
                   'model_revision':manifest.get('revision'),
                   'model_verification': ('Prior full verification; file identity checked during configuration generation.'
                                          if binding else 'Prior full verification; size checked during configuration generation.')}
