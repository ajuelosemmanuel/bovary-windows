"""Check a trusted built Bovary wheel in a fresh venv; never run model inference.

This script is standalone so CI can copy it with the wheel/native executable to
another hosted image without checking out the product source.
"""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

MAX_WHEEL_BYTES = 8 * 1024 * 1024


def clean_environment(home):
    environment = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if (upper.startswith(('PYTHON', 'PIP_', 'BOVARY_', 'OPENAI_', 'AZURE_OPENAI_', 'ANTHROPIC_'))
                or upper in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GOOGLE_API_KEY', 'GEMINI_API_KEY'}):
            continue
        environment[name] = value
    environment.update(BOVARY_HOME=str(home), PYTHONIOENCODING='utf-8', PIP_CONFIG_FILE=os.devnull,
                       PIP_DISABLE_PIP_VERSION_CHECK='1')
    return environment


def file_sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def wheel_identity(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_WHEEL_BYTES:
        raise ValueError('Expected a completed wheel within the 8 MiB bound')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) > 1024 or len({item.filename.casefold() for item in members}) != len(members)
                or sum(item.file_size for item in members) > MAX_WHEEL_BYTES):
            raise ValueError('Wheel inventory exceeds the installed-check bounds')
        names = [item.filename for item in members if re.fullmatch(r'bovary-[^/]+\.dist-info/METADATA', item.filename)]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 2 * 1024 * 1024:
            raise ValueError('Wheel must contain one Bovary metadata identity')
        metadata = BytesParser().parsebytes(archive.read(names[0]))
        version = metadata.get('Version', '')
        if (metadata.get_all('Name') != ['bovary'] or metadata.get_all('Version') != [version]
                or not re.fullmatch(r'[0-9][A-Za-z0-9.!+_-]*', version)
                or path.name != 'bovary-' + version + '-py3-none-any.whl'
                or names[0] != 'bovary-' + version + '.dist-info/METADATA'):
            raise ValueError('Wheel filename and Bovary metadata identity disagree')
    return version


PROBE = r'''
import contextlib, importlib.metadata, importlib.util, io, json, pathlib, socket, sys, time
from unittest.mock import patch
expected, venv, home = sys.argv[1:]
venv, home = pathlib.Path(venv).resolve(), pathlib.Path(home).resolve()
def forbidden(*args, **kwargs): raise AssertionError('Installed smoke must not start a backend or network connection')
with patch('subprocess.Popen', side_effect=forbidden), patch('socket.create_connection', side_effect=forbidden), patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden):
    import tkinter as tk
    assert tk._default_root is None
    import bovary
    from bovary import cli, Request, Result, ExecutionError, create_workbench
    from bovary.contract import Request as DirectRequest
    from bovary.runtime import create_workbench as direct_factory
    from bovary.gui import build_request, DEFAULTS, SettingsWindow
    assert tk._default_root is None
    assert pathlib.Path(bovary.__file__).resolve().is_relative_to(venv)
    assert pathlib.Path(sys.prefix).resolve() == venv
    assert bovary.__version__ == expected
    assert Request is DirectRequest and create_workbench is direct_factory
    assert Request.__module__ == 'bovary.contract'
    assert importlib.util.find_spec('lowcostgemma') is None and importlib.util.find_spec('shardrail') is None
    distribution=importlib.metadata.distribution('bovary')
    assert distribution.version == expected
    assert {(item.name,item.value) for item in distribution.entry_points} == {('bovary','bovary.cli:main'),('bovary-settings','bovary.gui:main')}
    direct=json.loads(distribution.read_text('direct_url.json') or '{}')
    assert not direct.get('dir_info',{}).get('editable',False)
    assert Request('default').output_tokens == 128
    Request('until end',output_tokens=None,deadline=time.monotonic()+30).check()
    for change in ({'deadline':None},{'mode':'commercial'},{'mode':'suggest'},{'allow_escalation':True}):
        try: Request(**{'prompt':'blocked','output_tokens':None,'deadline':time.monotonic()+30,**change}).check()
        except ExecutionError: pass
        else: raise AssertionError('Invalid unlimited request was accepted')
    assert build_request({**DEFAULTS,'tokens':None},'GUI probe',deadline_s=30).output_tokens is None
    assert build_request(DEFAULTS,'GUI default',deadline_s=30).output_tokens == 128
    class Recorder:
        def __init__(self): self.requests=[]
        def execute(self,request,**kwargs):
            request.check(); self.requests.append(request)
            return Result(request.request_id,'local',status='complete')
        def unload(self): pass
    recorder=Recorder()
    for arguments,limit in (([],128),(['--tokens','7'],7),(['--tokens','unlimited'],None),(['--until-finished'],None)):
        with patch('bovary.cli.create_workbench',return_value=recorder), contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(['--prompt','installed input-control probe','--json',*arguments]) == 0
        assert recorder.requests[-1].output_tokens == limit
    root=tk.Tk();root.withdraw()
    try:
        window=SettingsWindow(root,home/'bovary-settings.json')
        root.update_idletasks()
        assert window.process is None and window.workbench is None
        assert window.settings()['tokens']==128
        window.until_finished.set(True)
        assert window.settings()['tokens'] is None
    finally: root.destroy()
print(json.dumps({'installed_import_inside_venv':True,'noneditable':True,'canonical_api_and_commands':True,
    'default_128':True,'until_finished_contract_and_cli_flags':True,'gui_request_and_withdrawn_constructor':True,
    'model_or_network_started':False}))
'''


def run(command, *, cwd, environment, timeout, accepted_exit_codes=(0,)):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode not in accepted_exit_codes:
        raise ValueError('Installed check command failed: ' + result.stderr[-4000:] + result.stdout[-1000:])
    return result.stdout


def check_installed(wheel, work, *, native=None, expected_version=None, wheel_sha256=None, native_sha256=None, timeout=120):
    wheel = Path(wheel).resolve()
    version = wheel_identity(wheel)
    wheel_digest = file_sha(wheel)
    if expected_version is not None and version != expected_version:
        raise ValueError('Wheel version differs from the expected candidate')
    if wheel_sha256 is not None and wheel_digest != wheel_sha256.lower():
        raise ValueError('Wheel SHA256 differs from the expected artifact')
    native = Path(native).resolve() if native else None
    if native and (not native.is_file() or native.stat().st_size > 64 * 1024 * 1024):
        raise ValueError('Expected a bounded native executable file')
    native_digest = file_sha(native) if native else None
    if native_sha256 is not None and native_digest != native_sha256.lower():
        raise ValueError('Native SHA256 differs from the expected artifact')
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    home, empty, venv = work/'portable home with spaces', work/'empty working directory', work/'isolated python'
    home.mkdir();empty.mkdir()
    environment = clean_environment(home)
    run([sys.executable,'-I','-m','venv',str(venv)],cwd=empty,environment=environment,timeout=timeout)
    scripts = venv/('Scripts' if os.name=='nt' else 'bin')
    python = scripts/('python.exe' if os.name=='nt' else 'python')
    run([str(python),'-I','-m','pip','install','--disable-pip-version-check','--no-index','--no-deps',
         '--no-compile','--no-cache-dir',str(wheel)],cwd=empty,environment=environment,timeout=timeout)
    probe = json.loads(run([str(python),'-I','-c',PROBE,version,str(venv),str(home)],
                          cwd=empty,environment=environment,timeout=timeout))
    for command, arguments in (('bovary',['--version']),('bovary-settings',['--help'])):
        executable=scripts/(command+'.exe' if os.name=='nt' else command)
        output=run([str(executable),*arguments],cwd=empty,environment=environment,timeout=timeout)
        if command=='bovary' and output.strip()!='Bovary '+version:
            raise ValueError('Installed console command reports a different version')
    run([str(python),'-I','-m','bovary','--help'],cwd=empty,environment=environment,timeout=timeout)
    suggest=json.loads(run([str(python),'-I','-m','bovary','--mode','suggest','--task','extract',
                           '--prompt','name: Ada','--json'],cwd=empty,environment=environment,timeout=timeout))
    if suggest.get('status')!='suggested' or suggest.get('attempts')!=[]:
        raise ValueError('Installed Suggest executed a backend or failed')
    doctor=json.loads(run([str(python),'-I','-m','bovary','doctor','--json','--timeout-s','10'],
                          cwd=empty,environment=environment,timeout=timeout,accepted_exit_codes=(1,)))
    diagnostic_checks={item['id']:item for item in doctor.get('checks',[])}
    if (doctor.get('schema_version')!=1 or doctor.get('exit_code')!=1
            or doctor.get('paths_redacted') is not True or doctor.get('full_hash_requested') is not False
            or diagnostic_checks.get('native_executable',{}).get('status')!='fail'
            or diagnostic_checks.get('model_manifest',{}).get('status')!='fail'
            or diagnostic_checks.get('tk',{}).get('status')!='pass'):
        raise ValueError('Installed doctor did not diagnose the intentionally unconfigured runtime')
    if str(home) in json.dumps(doctor):
        raise ValueError('Installed doctor exposed private local paths by default')
    for old in ('lowcostgemma','lowcostgemma-settings','shardrail','shardrail-settings'):
        if (scripts/(old+'.exe' if os.name=='nt' else old)).exists():
            raise ValueError('Unexpected retired console entrypoint in the fresh environment')
    native_probe=False
    if native:
        relocated=home/'work/build/bovary-cpu/bovary-infer.exe';relocated.parent.mkdir(parents=True)
        shutil.copyfile(native,relocated)
        if file_sha(relocated)!=native_digest:
            raise ValueError('Relocated native executable bytes changed')
        output=run([str(relocated),'--require-cpu-only','--protocol-version'],cwd=empty,environment=environment,timeout=timeout)
        if output.strip()!='bovary-worker 2':
            raise ValueError('Native worker protocol marker differs from the candidate contract')
        native_probe=True
    return {'schema_version':1,'version':version,'wheel_sha256':wheel_digest,'native_sha256':native_digest,
            **probe,'installed_module_and_console_smoke':True,'suggest_without_attempts':True,
            'doctor_without_model_or_native':True,'doctor_private_paths_redacted':True,
            'relocated_native_protocol_2':native_probe,'clean_hosted_image_is_not_clean_desktop_install':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',required=True,type=Path)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--expected-version')
    parser.add_argument('--wheel-sha256')
    parser.add_argument('--native-sha256')
    parser.add_argument('--work-dir',type=Path,help='New check directory; default is disposable OS temporary storage')
    parser.add_argument('--output',type=Path,help='Optional new JSON report; never overwritten')
    args=parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise ValueError('Installed-check report already exists; choose a new report')
        def execute(work):
            return check_installed(args.wheel,work,native=args.native,expected_version=args.expected_version,
                                   wheel_sha256=args.wheel_sha256,native_sha256=args.native_sha256)
        if args.work_dir:
            result=execute(args.work_dir)
        else:
            with tempfile.TemporaryDirectory(prefix='bovary-installed-') as temporary:
                result=execute(Path(temporary)/'consumer')
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as target:
                target.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,subprocess.SubprocessError) as error:
        parser.exit(2,'Installed check: '+str(error)+'\n')


if __name__=='__main__':
    main()
