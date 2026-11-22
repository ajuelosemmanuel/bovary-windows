"""Record the built executable, exact Ninja commands, options, source diff, and no-model smoke.

Run after scripts/build.cmd cuda|cpu N. This tool never loads a model or runs inference.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def verified_pin_patch(engine, revision):
    """Use the lock's serialization only after checking the actual engine tree."""
    lock = json.loads((ROOT/'engine.lock.json').read_text(encoding='utf-8'))
    if revision != lock['revision']:
        return None
    result = subprocess.run(
        [sys.executable, str(ROOT/'scripts/bootstrap_engine.py'), '--check', '--directory', str(engine)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        return None
    proof = json.loads(result.stdout)
    patch = (ROOT/lock['patch']).read_bytes()
    if (proof != {'revision': revision, 'patch_sha256': lock['patch_sha256'], 'verified': True}
            or hashlib.sha256(patch).hexdigest() != lock['patch_sha256']):
        raise ValueError('Engine lock or patch changed while verifying the tree')
    return patch


def create_capture(output_dir, backend):
    if output_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output_dir = ROOT/'outputs/build-inventory'/f'{backend}-{stamp}-{uuid.uuid4().hex[:8]}'
    out = Path(output_dir).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError('Capture directory already exists; use a new --output-dir to preserve previous build evidence') from error
    return out


def read_options(build):
    options = {}
    for line in (build/'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith(('#','//')) or '=' not in line:
            continue
        key,value=line.split('=',1)
        key=key.split(':',1)[0]
        if key.startswith(('CMAKE_','GGML_','LLAMA_','BOVARY_','BUILD_SHARED_LIBS','CUDAToolkit_')):
            options[key]=value
    return options


def check_build_freshness(build, options, targets):
    """Refuse to associate current sources with stale binaries; never build here."""
    home = options.get('CMAKE_HOME_DIRECTORY')
    if not isinstance(home, str) or not home.strip() or Path(home).resolve() != ROOT.resolve():
        raise ValueError('Build was configured from another source tree; run its own inventory tool')
    result = subprocess.run([options['CMAKE_MAKE_PROGRAM'], '-C', str(build), '-n', *targets],
                            capture_output=True, timeout=60)
    lines = result.stdout.decode('utf-8', errors='replace').strip().splitlines()
    if result.returncode or not lines or lines[-1].strip() != 'ninja: no work to do.':
        raise ValueError('Build is not up to date; rebuild the requested targets before capturing source/binary provenance')
    return {'tool':'ninja dry-run', 'targets':targets, 'up_to_date':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cuda','cpu'), default='cpu')
    parser.add_argument('--build-dir',type=Path,help='Explicit isolated build directory; default work/build/bovary-<backend>')
    parser.add_argument('--build-command',help='Exact high-level build invocation, if known; otherwise left unrecorded')
    parser.add_argument('--output-dir',type=Path,
                        help='New capture directory; default creates a unique outputs/build-inventory capture.')
    args=parser.parse_args()
    build=args.build_dir.resolve() if args.build_dir else ROOT/'work/build'/('bovary-'+args.backend); binary=build/'bovary-infer.exe'
    if not binary.exists(): raise SystemExit(args.backend.upper()+' binary is absent')
    options=read_options(build)
    binaries=[p for p in (binary,build/'bovary-quantize.exe') if p.is_file()]
    targets=[p.name for p in binaries]
    freshness=check_build_freshness(build, options, targets)
    out=create_capture(args.output_dir, args.backend)
    def reference(path):
        try: return path.relative_to(ROOT).as_posix()
        except ValueError: return str(path)
    snapshots=[]
    for relative in ('src/infer.cpp','src/native_paths.h','CMakeLists.txt','src/quantize.cpp','scripts/build.cmd','scripts/find_build_tools.cmd'):
        source=ROOT/relative
        if not source.is_file(): continue
        snapshot=out/'source-snapshots'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,snapshot)
        snapshots.append({'source':relative,'snapshot':reference(snapshot),'sha256':digest(snapshot),'bytes':snapshot.stat().st_size})
    env=os.environ.copy()
    if args.backend == 'cuda':
        env['PATH']=str(ROOT/'work/deps/cuda-13.3/bin/x64')+os.pathsep+env.get('PATH','')
    else:
        env.pop('GGML_BACKEND_PATH',None)
        env['BOVARY_STREAM']='0'
    smoke=subprocess.run([str(binary)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    smoke_record={'command':[str(binary)],'exit_code':smoke.returncode,'stdout':smoke.stdout,'stderr':smoke.stderr,
                  'expected_exit_code':1,'passed':smoke.returncode==1 and 'bovary-infer: invalid arguments' in smoke.stderr,
                  'scope':'Executable/runtime DLL loading and argument validation only. No model loaded; no GPU inference.'}
    (out/'build-no-model-smoke.json').write_text(json.dumps(smoke_record,indent=2),encoding='utf-8')
    engine=Path(options.get('BOVARY_ENGINE_DIR',ROOT/'work/engine/bovary-llama.cpp'))
    git=lambda *args:subprocess.check_output(['git','-C',str(engine),*args])
    revision=git('rev-parse','HEAD').decode().strip()
    observed_diff=git('diff','--binary','HEAD')
    untracked=[]
    for relative in git('ls-files','--others','--exclude-standard').decode().splitlines():
        source=engine/relative
        if not source.is_file(): continue
        untracked.append({'path':relative,'sha256':digest(source),'bytes':source.stat().st_size})
        addition=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',relative],cwd=engine,capture_output=True)
        if addition.returncode not in (0,1): raise RuntimeError(addition.stderr.decode(errors='replace'))
        observed_diff+=addition.stdout
    pinned_diff=verified_pin_patch(engine, revision)
    diff=observed_diff if pinned_diff is None else pinned_diff
    if observed_diff != diff:
        (out/'build-engine-observed.patch').write_bytes(observed_diff)
    (out/'build-engine.patch').write_bytes(diff)
    commands=subprocess.run([options['CMAKE_MAKE_PROGRAM'],'-C',str(build),'-t','commands',*[p.name for p in binaries]],capture_output=True,check=True)
    (out/'build-commands.txt').write_bytes(commands.stdout)
    shutil.copyfile(build/'CMakeCache.txt',out/'build-CMakeCache.txt')
    log_snapshots=[]
    for source in sorted((ROOT/'work/environment').glob('build*.log')):
        if args.build_dir: continue
        if args.backend == 'cpu' and source.name != 'build-cpu.log': continue
        snapshot=out/'build-logs'/source.name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,snapshot)
        log_snapshots.append({'source':reference(source),'snapshot':reference(snapshot),'sha256':digest(snapshot)})
    manifest={'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'build_command':args.build_command,
        'backend':args.backend, 'build_freshness':freshness,
        'previous_incremental_stage':None,
        'binary':{'path':str(binary),'bytes':binary.stat().st_size,'sha256':digest(binary),'mtime_utc':dt.datetime.fromtimestamp(binary.stat().st_mtime,dt.timezone.utc).isoformat()},
        'executables':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in binaries],
        'engine_commit':revision,'engine_diff_sha256':hashlib.sha256(diff).hexdigest(),
        'engine_untracked_source_files':untracked,
        'engine_patch_verified_against_lock':pinned_diff is not None,
        'observed_engine_diff_sha256':hashlib.sha256(observed_diff).hexdigest(),
        'engine_diff_file':reference(out/'build-engine.patch'),'build_options':options,'ninja_commands_file':reference(out/'build-commands.txt'),
        'ninja_commands_sha256':digest(out/'build-commands.txt'),'cmake_cache_file':reference(out/'build-CMakeCache.txt'),
        'source_snapshots':snapshots,
        'root_cmake_sha256':digest(ROOT/'CMakeLists.txt'),'application_source_sha256':digest(ROOT/'src/infer.cpp'),
        'build_script_sha256':digest(ROOT/'scripts/build.cmd'),'portable_cuda_manifest':str(ROOT/'work/deps/cuda-13.3/portable-install.json') if args.backend == 'cuda' else None,
        'logs':[row['snapshot'] for row in log_snapshots],'build_history_log_snapshots':log_snapshots, 'smoke':smoke_record,
        'notes':['Backend label is supplied by the caller; inspect captured CMake options and native commands for the actual build configuration.',
                 'Compiler versions, high-level build commands and previous build history are not inferred from older runs.',
                 'No model inference, GPU telemetry or performance benchmark is performed by this manifest tool.']}
    check_build_freshness(build, options, targets)
    if any(digest(ROOT/row['source']) != row['sha256'] for row in snapshots):
        raise ValueError('Sources changed while capturing; preserve this incomplete capture and use a fresh directory')
    (out/'build-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'binary':manifest['binary'],'engine_commit':manifest['engine_commit'],'smoke_passed':smoke_record['passed']},indent=2))
    if not smoke_record['passed']: raise SystemExit('No-model smoke failed; see outputs/build-no-model-smoke.json')

if __name__=='__main__': main()
