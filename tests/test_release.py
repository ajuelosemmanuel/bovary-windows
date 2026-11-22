import importlib.util
import base64
import csv
import io
import stat
import warnings
from unittest import mock
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load("prepare_release")
bootstrap = load("bootstrap_engine")
installed = load("check_installed")


class ReleaseTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows package quick start")
    def test_packaged_checkpoint_commands_select_the_files_returned_by_the_helpers(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        quick_start = readme.split("## Install the Windows alpha from release files", 1)[1]
        quick_start = quick_start.split("\n## ", 1)[0]
        commands = re.findall(r"(?m)^```powershell\n(.*?)\n```$", quick_start, re.DOTALL)
        with tempfile.TemporaryDirectory(prefix="Bovary downloads ") as temporary:
            directory = Path(temporary)
            runtime = directory / "runtime"
            helper_dir = runtime / "tools"
            helper_dir.mkdir(parents=True)
            interpreter = r".\.venv\Scripts\python.exe"
            model_name = "gemma-4-26B-A4B-it-Q4_0.gguf"
            fixture_model = directory / model_name
            downloader = helper_dir / "download_model.py"
            downloader.write_text(
                "import argparse, pathlib\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--directory', required=True, type=pathlib.Path)\n"
                "directory = parser.parse_args().directory\n"
                "directory.mkdir(parents=True)\n"
                f"(directory / {model_name!r}).touch()\n"
                "(directory / 'model-provenance.json').touch()\n",
                encoding="utf-8",
            )
            verifier = helper_dir / "verify_model.py"
            verifier.write_text(
                "import argparse, pathlib\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--model', required=True, type=pathlib.Path)\n"
                "parser.add_argument('--manifest', required=True, type=pathlib.Path)\n"
                "args = parser.parse_args()\n"
                "args.model.touch()\n"
                "args.manifest.touch()\n",
                encoding="utf-8",
            )
            model_dir = runtime / "work" / "models"
            for helper, model, proof in (
                (verifier, fixture_model, runtime / "model-provenance.json"),
                (downloader, model_dir / model_name, model_dir / "model-provenance.json"),
            ):
                with self.subTest(helper=helper.name):
                    command, = (block for block in commands if helper.name in block)
                    command = command.replace(interpreter, "& $env:BOVARY_TEST_PYTHON", 1)
                    command = re.sub(r"'D:\\Models\\[^']+'", "$env:BOVARY_TEST_MODEL", command)
                    completed = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                         "$ErrorActionPreference = 'Stop'\n" + command + "\n"
                         "$env:BOVARY_MODEL\n$env:BOVARY_MODEL_MANIFEST\n"],
                        cwd=directory, capture_output=True, text=True, timeout=30,
                        env={**os.environ, "BOVARY_HOME": str(runtime),
                             "BOVARY_TEST_MODEL": str(fixture_model),
                             "BOVARY_TEST_PYTHON": sys.executable},
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(completed.stdout.splitlines(), [str(model), str(proof)])
                    self.assertTrue(model.is_file())
                    self.assertTrue(proof.is_file())

    @staticmethod
    def wheel_fixture():
        project = {'name':'fixture-route', 'version':'1.0a1', 'description':'Public fixture',
                   'requires-python':'>=3.11', 'license':'MIT', 'readme':'README.md',
                   'scripts':{'fixture-route':'bovary.cli:main', 'fixture-route-settings':'bovary.gui:main'},
                   'license-files':['LICENSE']}
        files = {'python/bovary/__init__.py':b'__version__ = "1.0a1"\n',
                 'README.md':'Public README: caf\u00e9 \u2192 local\n'.encode(), 'LICENSE':b'MIT fixture\n'}
        info = 'fixture_route-1.0a1.dist-info/'
        metadata = ('Metadata-Version: 2.4\nName: fixture-route\nVersion: 1.0a1\nSummary: Public fixture\n'
                    'Requires-Python: >=3.11\nLicense-Expression: MIT\nDescription-Content-Type: text/markdown\n'
                    'License-File: LICENSE\nDynamic: license-file\n\n').encode()+files['README.md']
        contents = {'bovary/__init__.py':files['python/bovary/__init__.py'], info+'METADATA':metadata,
                    info+'WHEEL':b'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n',
                    info+'entry_points.txt':b'[console_scripts]\nfixture-route = bovary.cli:main\nfixture-route-settings = bovary.gui:main\n',
                    info+'top_level.txt':b'bovary\n', info+'licenses/LICENSE':files['LICENSE']}
        return project, files, contents, info

    @staticmethod
    def record_text(contents, info):
        output=io.StringIO(newline='')
        writer=csv.writer(output,lineterminator='\n')
        for name,data in sorted(contents.items()):
            if name!=info+'RECORD':
                encoded=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                writer.writerow((name,'sha256='+encoded,str(len(data))))
        writer.writerow((info+'RECORD','',''))
        return output.getvalue().encode()

    def write_wheel(self, path, contents, info, *, record=True, attributes=None, duplicate=False):
        contents=contents.copy()
        if record and info+'RECORD' not in contents:
            contents[info+'RECORD']=self.record_text(contents,info)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as output:
            for name,data in contents.items():
                if attributes and name in attributes:
                    entry=zipfile.ZipInfo(name);entry.external_attr=attributes[name]<<16
                    output.writestr(entry,data)
                else:
                    output.writestr(name,data)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    output.writestr('bovary/__init__.py',contents['bovary/__init__.py'])
        return path

    def test_complete_wheel_fixture_checks_every_record_and_unicode_readme(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            wheel=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',contents,info)
            checked=release.check_wheel(wheel,files,project)
            self.assertTrue(checked['record_verified'])
            self.assertEqual(checked['entries'],len(contents)+1)
            self.assertEqual(checked['sha256'],hashlib.sha256(wheel.read_bytes()).hexdigest())

    def test_wheel_rejects_source_identity_metadata_commands_and_license_changes(self):
        project,files,contents,info=self.wheel_fixture();metadata=contents[info+'METADATA']
        changes={'package':{'bovary/__init__.py':b'unreviewed'},
                 'name':{info+'METADATA':metadata.replace(b'Name: fixture-route',b'Name: other')},
                 'version':{info+'METADATA':metadata.replace(b'Version: 1.0a1',b'Version: 0.9')},
                 'python':{info+'METADATA':metadata.replace(b'>=3.11',b'>=3.14')},
                 'license':{info+'METADATA':metadata.replace(b'License-Expression: MIT',b'License-Expression: Apache-2.0')},
                 'readme':{info+'METADATA':metadata.replace(b'Public README',b'Stale README')},
                 'missing-notice':{info+'METADATA':metadata.replace(b'License-File: LICENSE\n',b'')},
                 'notice-bytes':{info+'licenses/LICENSE':b'Stale license'},
                 'dependency':{info+'METADATA':metadata.replace(b'\n\n',b'\nRequires-Dist: unexpected\n\n',1)},
                 'duplicate-header':{info+'METADATA':metadata.replace(b'Name: fixture-route\n',b'Name: fixture-route\nName: fixture-route\n')},
                 'extra-command':{info+'entry_points.txt':contents[info+'entry_points.txt']+b'old-product = bovary.cli:main\n'},
                 'missing-command':{info+'entry_points.txt':b'[console_scripts]\nfixture-route = bovary.cli:main\n'},
                 'package-list':{info+'top_level.txt':b'other\n'}}
        with tempfile.TemporaryDirectory() as temporary:
            for kind,changed in changes.items():
                with self.subTest(kind=kind):
                    wheel=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',{**contents,**changed},info)
                    with self.assertRaises(ValueError):release.check_wheel(wheel,files,project)

    def test_wheel_requires_structure_and_reviewed_file_inventory(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl'
            for missing in ('WHEEL','RECORD','METADATA','entry_points.txt','top_level.txt'):
                with self.subTest(missing=missing):
                    chosen={n:d for n,d in contents.items() if n!=info+missing}
                    wheel=self.write_wheel(path,chosen,info,record=missing!='RECORD')
                    with self.assertRaises(ValueError):release.check_wheel(wheel,files,project)
            for unsafe in ('../escape.py','/escape.py','bovary\\escape.py','bovary/../escape.py','bovary//escape.py','bovary/CON.txt','bovary/trailing.','BOVARY/__INIT__.PY',info+'unreviewed.txt'):
                with self.subTest(path=unsafe):
                    wheel=self.write_wheel(path,{**contents,unsafe:b'not reviewed'},info)
                    with self.assertRaises(ValueError):release.check_wheel(wheel,files,project)
            self.write_wheel(path,contents,info,duplicate=True)
            with self.assertRaises(ValueError):release.check_wheel(path,files,project)
            self.write_wheel(path,contents,info,attributes={'bovary/__init__.py':stat.S_IFLNK|0o777})
            with self.assertRaises(ValueError):release.check_wheel(path,files,project)

    def test_wheel_rejects_wrong_filename_and_format_tags(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for name in ('other-1.0a1-py3-none-any.whl','fixture_route-2.0-py3-none-any.whl',
                         'fixture_route-1.0a1-cp313-cp313-win_amd64.whl','fixture_route-1.0a1-1-py3-none-any.whl'):
                self.write_wheel(Path(temporary)/name,contents,info)
                with self.subTest(filename=name),self.assertRaises(ValueError):release.check_wheel(Path(temporary)/name,files,project)
            for old,new in ((b'Wheel-Version: 1.0',b'Wheel-Version: 2.0'),(b'Root-Is-Purelib: true',b'Root-Is-Purelib: false'),
                            (b'Tag: py3-none-any',b'Tag: cp313-none-any'),(b'Tag: py3-none-any',b'Tag: py3-none-any\nTag: py3-none-any')):
                wheel=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',{**contents,info+'WHEEL':contents[info+'WHEEL'].replace(old,new)},info)
                with self.subTest(format=new),self.assertRaises(ValueError):release.check_wheel(wheel,files,project)

    def test_wheel_record_requires_exact_paths_hashes_sizes_and_self_entry(self):
        project,files,contents,info=self.wheel_fixture()
        rows=list(csv.reader(io.StringIO(self.record_text(contents,info).decode())))
        variations={}
        variations['missing']=rows[1:]
        variations['duplicate']=rows+[rows[0]]
        variations['extra']=rows+[['unlisted','sha256=bad','1']]
        for kind,column,value in (('hash',1,'sha256=bad'),('size',2,'999'),('unhashed',1,''),('algorithm',1,rows[0][1].replace('sha256=','sha1=')),('path',0,'../escape.py')):
            changed=[row.copy() for row in rows];changed[0][column]=value;variations[kind]=changed
        changed=[row.copy() for row in rows];changed[-1][1]='sha256=notempty';variations['self-hash']=changed
        variations['columns']=rows+[[]]
        with tempfile.TemporaryDirectory() as temporary:
            for kind,chosen in variations.items():
                output=io.StringIO(newline='');csv.writer(output,lineterminator='\n').writerows(chosen)
                path=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',{**contents,info+'RECORD':output.getvalue().encode()},info)
                with self.subTest(kind=kind),self.assertRaises(ValueError):release.check_wheel(path,files,project)

    def test_wheel_bounds_and_metadata_privacy_are_checked_before_acceptance(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',contents,info)
            with mock.patch.object(release,'MAX_WHEEL_BYTES',path.stat().st_size-1),self.assertRaisesRegex(ValueError,'archive exceeds'):
                release.check_wheel(path,files,project)
            with mock.patch.object(release,'MAX_WHEEL_ENTRIES',1),self.assertRaisesRegex(ValueError,'member count'):
                release.check_wheel(path,files,project)
            with mock.patch.object(release,'MAX_WHEEL_ENTRY_BYTES',1),self.assertRaisesRegex(ValueError,'member exceeds'):
                release.check_wheel(path,files,project)
            large={**contents,'bovary/__init__.py':b'x'*20000};largefiles={**files,'python/bovary/__init__.py':large['bovary/__init__.py']}
            self.write_wheel(path,large,info)
            with mock.patch.object(release,'MAX_WHEEL_BYTES',path.stat().st_size+10),self.assertRaisesRegex(ValueError,'Expanded wheel'):
                release.check_wheel(path,largefiles,project)
            private='C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'build'
            for content in (value.encode(encoding) for value in (private,'sk-'+'x'*30,'hf_'+'x'*30)
                            for encoding in ('ascii','utf-16le','utf-16be')):
                self.write_wheel(path,{**contents,info+'WHEEL':contents[info+'WHEEL'].replace(b'fixture',content)},info)
                with self.subTest(content_type=content[:2]),self.assertRaisesRegex(ValueError,'credential or private'):
                    release.check_wheel(path,files,project)

    def test_check_wheel_cli_is_read_only_and_needs_no_output_directory(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path=self.write_wheel(Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl',contents,info)
            capture=io.StringIO()
            with mock.patch('sys.argv',['prepare_release.py','--check-wheel',str(path)]),mock.patch.object(release,'collect',return_value=files),mock.patch.object(release,'project_metadata',return_value=project),mock.patch('sys.stdout',capture):
                release.main()
            self.assertTrue(json.loads(capture.getvalue())['wheel']['record_verified'])
            self.assertEqual([item.name for item in Path(temporary).iterdir()],[path.name])

    def test_wheel_container_privacy_covers_comments_extra_fields_and_trailing_bytes(self):
        project,files,contents,info=self.wheel_fixture()
        def decorate(path, location, payload):
            with zipfile.ZipFile(path) as archive:
                entries=[(item,archive.read(item)) for item in archive.infolist()]
            with zipfile.ZipFile(path,'w') as archive:
                if location=='archive-comment':archive.comment=payload
                for index,(item,data) in enumerate(entries):
                    if index==0:
                        if location=='member-comment':item.comment=payload
                        if location=='extra-field':item.extra=b'\xfe\xca'+len(payload).to_bytes(2,'little')+payload
                    archive.writestr(item,data)
            if location=='trailing-bytes':
                with path.open('ab') as output:output.write(payload)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl'
            private='C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'build'
            for location in ('archive-comment','member-comment','extra-field','trailing-bytes'):
                self.write_wheel(path,contents,info)
                decorate(path,location,'Reviewed archive caf\u00e9'.encode('utf-16le'))
                if location=='trailing-bytes':
                    with self.assertRaisesRegex(ValueError,'trailing bytes'):release.check_wheel(path,files,project)
                else:
                    self.assertTrue(release.check_wheel(path,files,project)['record_verified'])
                for value in (private,'sk-'+'x'*30,'hf_'+'x'*30):
                    for encoding in ('ascii','utf-16le','utf-16be'):
                        self.write_wheel(path,contents,info)
                        decorate(path,location,value.encode(encoding))
                        with self.subTest(location=location,encoding=encoding,prefix=value[:3]),self.assertRaisesRegex(ValueError,'archive metadata'):
                            release.check_wheel(path,files,project)
            # A large field's binary length can end with an ASCII letter, which
            # masks the following token from a whole-buffer boundary matcher.
            secret=b'sk-'+b'x'*30
            payload=secret+b'\0'*(0x4100-len(secret))
            self.write_wheel(path,contents,info);decorate(path,'extra-field',payload)
            with self.assertRaisesRegex(ValueError,'archive metadata'):release.check_wheel(path,files,project)

    def test_wheel_rejects_joined_comment_trailer_and_local_only_extra_values(self):
        project,files,contents,info=self.wheel_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fixture_route-1.0a1-py3-none-any.whl'
            self.write_wheel(path,contents,info)
            with zipfile.ZipFile(path,'a') as archive:archive.comment=b'safe'
            with path.open('ab') as output:output.write(b'sk-'+b'x'*30)
            with self.assertRaisesRegex(ValueError,'trailing bytes'):release.check_wheel(path,files,project)

            # Keep offsets and lengths valid: only the local extra value carries
            # the canary; the central-directory copy remains safe. Its size high
            # byte is ASCII A, masking a joined raw-buffer credential boundary.
            payload_size=0x4100
            extra=b'\xfe\xca'+payload_size.to_bytes(2,'little')+b'\0'*payload_size
            self.write_wheel(path,contents,info)
            with zipfile.ZipFile(path) as archive:
                entries=[(item,archive.read(item)) for item in archive.infolist()]
            with zipfile.ZipFile(path,'w') as archive:
                for index,(item,data) in enumerate(entries):
                    if index==0:item.extra=extra
                    archive.writestr(item,data)
            self.assertTrue(release.check_wheel(path,files,project)['record_verified'])
            raw=bytearray(path.read_bytes())
            with zipfile.ZipFile(path) as archive:local=archive.infolist()[0].header_offset
            name_size=int.from_bytes(raw[local+26:local+28],'little')
            start=local+30+name_size+4
            secret=b'sk-'+b'x'*30
            raw[start:start+len(secret)]=secret
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'local and central extra fields differ'):
                release.check_wheel(path,files,project)

    def test_installed_smoke_environment_removes_checkout_credentials_and_pip_overrides(self):
        with mock.patch.object(os, 'environ', {'PATH':'tools','PYTHONPATH':'source','PYTHONHOME':'other-python',
                'PYTHONUSERBASE':'user-packages','BOVARY_HOME':'old-home','BOVARY_MODEL':'private-model',
                'OPENAI_API_KEY':'synthetic-key','HF_TOKEN':'synthetic-token','PIP_TARGET':'old-target'}):
            environment=installed.clean_environment(Path('isolated-home'))
        self.assertEqual(environment['PATH'],'tools')
        self.assertEqual(environment['BOVARY_HOME'],'isolated-home')
        for name in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','BOVARY_MODEL','OPENAI_API_KEY','HF_TOKEN','PIP_TARGET'):
            self.assertNotIn(name,environment)
        self.assertEqual(environment['PIP_CONFIG_FILE'],os.devnull)

    def test_installed_smoke_rejects_wrong_artifacts_before_creating_environment(self):
        _,_,contents,info=self.wheel_fixture()
        new_info=info.replace('fixture_route','bovary')
        chosen={name.replace(info,new_info):data for name,data in contents.items()}
        chosen[new_info+'METADATA']=chosen[new_info+'METADATA'].replace(b'Name: fixture-route',b'Name: bovary')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            wheel=self.write_wheel(root/'bovary-1.0a1-py3-none-any.whl',chosen,new_info)
            self.assertEqual(installed.wheel_identity(wheel),'1.0a1')
            with mock.patch.object(installed,'run',side_effect=AssertionError('Must fail before creating a venv')) as runner:
                for options in ({'expected_version':'2.0'},{'wheel_sha256':'0'*64},{'native':root/'missing.exe'},
                                {'native_sha256':'0'*64}):
                    with self.subTest(options=options),self.assertRaises(ValueError):
                        installed.check_installed(wheel,root/'not-created',**options)
                    self.assertFalse((root/'not-created').exists())
                existing=root/'existing';existing.mkdir();(existing/'preserve.txt').write_text('old evidence')
                with self.assertRaises(FileExistsError):installed.check_installed(wheel,existing)
                self.assertEqual((existing/'preserve.txt').read_text(),'old evidence')
                runner.assert_not_called()

    def test_installed_smoke_cli_rejects_existing_report_before_any_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            report=Path(temporary)/'older.json';report.write_text('{"older":true}')
            with mock.patch('sys.argv',['check_installed.py','--wheel','not-opened.whl','--output',str(report)]), \
                    mock.patch.object(installed,'check_installed') as execute, mock.patch('sys.stderr',io.StringIO()), \
                    self.assertRaises(SystemExit) as result:
                installed.main()
            self.assertEqual(result.exception.code,2)
            self.assertEqual(report.read_text(),'{"older":true}')
            execute.assert_not_called()

    def test_cpu_release_requires_real_freshness_source_home_smoke_and_header_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            names=('src/infer.cpp','src/quantize.cpp','src/native_paths.h','CMakeLists.txt',
                   'scripts/build.cmd','scripts/find_build_tools.cmd')
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'captured source')
            manifest={'smoke':{'passed':True},'build_options':{'CMAKE_HOME_DIRECTORY':str(root)},
                      'build_freshness':{'tool':'ninja dry-run','up_to_date':True,'targets':['bovary-infer.exe']},
                      'source_snapshots':[{'source':name,'sha256':release.digest((root/name).read_bytes())} for name in names]}
            release.check_cpu_capture(manifest,root)
            for field,value in (('smoke',{}),('build_options',{}),('build_options',{'CMAKE_HOME_DIRECTORY':''}),
                                ('build_options',{'CMAKE_HOME_DIRECTORY':str(root/'foreign')}),('build_freshness',{}),
                                ('build_freshness',{'tool':'ninja dry-run','up_to_date':True,'targets':['bovary-quantize.exe']})):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    release.check_cpu_capture({**manifest,field:value},root)
            (root/'src/native_paths.h').write_bytes(b'changed without rebuilding')
            with self.assertRaisesRegex(ValueError,'snapshots'):
                release.check_cpu_capture(manifest,root)

    @staticmethod
    def bootstrap_fixture(root):
        upstream = root / 'producer'
        upstream.mkdir()
        environment = {**os.environ, 'GIT_AUTHOR_NAME':'Bootstrap fixture', 'GIT_AUTHOR_EMAIL':'fixture@example.invalid',
                       'GIT_COMMITTER_NAME':'Bootstrap fixture', 'GIT_COMMITTER_EMAIL':'fixture@example.invalid'}
        def git(directory, *args):
            return subprocess.check_output(['git', '-C', str(directory), *args], env=environment, stderr=subprocess.DEVNULL).decode().strip()
        git(upstream, 'init')
        (upstream / 'value.txt').write_bytes(b'old\n')
        git(upstream, 'add', 'value.txt')
        git(upstream, 'commit', '-m', 'Ephemeral bootstrap fixture')
        revision = git(upstream, 'rev-parse', 'HEAD')
        patch = root / 'fixture.patch'
        patch.write_bytes(b'diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-old\n+new\n')
        lock = root / 'lock.json'
        lock.write_text(json.dumps({'repository':str(upstream), 'revision':revision, 'patch':str(patch),
                                   'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest()}))
        return upstream, lock, git

    @unittest.skipUnless(shutil.which('git'), 'Git required for dependency bootstrap')
    def test_bootstrap_recovers_only_an_empty_interrupted_first_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer, lock, git = self.bootstrap_fixture(root)
            unavailable = root / 'temporarily unavailable upstream'
            value = json.loads(lock.read_text())
            value['repository'] = str(unavailable)
            lock.write_text(json.dumps(value))
            target = root / 'interrupted'
            with self.assertRaises(RuntimeError):
                bootstrap.bootstrap(target, lock_path=lock)
            self.assertTrue((target / '.git').is_dir())
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                bootstrap.bootstrap(target, check=True, lock_path=lock)
            git(producer, 'clone', '--bare', str(producer), str(unavailable))
            self.assertTrue(bootstrap.bootstrap(target, lock_path=lock)['verified'])
            self.assertEqual((target / 'value.txt').read_text(), 'new\n')

    @unittest.skipUnless(shutil.which('git'), 'Git required for dependency bootstrap')
    def test_unborn_bootstrap_preserves_wrong_origin_user_files_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer, lock, git = self.bootstrap_fixture(root)
            for kind in ('origin', 'files', 'index'):
                target = root / kind
                target.mkdir()
                git(target, 'init')
                git(target, 'remote', 'add', 'origin', str(root / 'wrong') if kind == 'origin' else str(producer))
                if kind == 'files':
                    (target / 'user.txt').write_text('preserve this user file')
                if kind == 'index':
                    (target / 'staged.txt').write_text('preserve this index entry')
                    git(target, 'add', 'staged.txt')
                    (target / 'staged.txt').unlink()
                with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'no changes were made'):
                    bootstrap.bootstrap(target, lock_path=lock)
                if kind == 'files':
                    self.assertEqual((target / 'user.txt').read_text(), 'preserve this user file')
                if kind == 'index':
                    self.assertEqual(git(target, 'ls-files'), 'staged.txt')
                self.assertFalse((target / 'value.txt').exists())

    def test_binary_private_path_gate_handles_ascii_and_utf16(self):
        private = "C:" + chr(92) + "Users" + chr(92) + "private" + chr(92) + "source.cpp"
        self.assertTrue(release.has_private_build_paths(private.encode("ascii")))
        self.assertTrue(release.has_private_build_paths(private.encode("utf-16le")))
        self.assertFalse(release.has_private_build_paths(b".\\engine\\source.cpp"))

    def test_shared_privacy_scanner_and_cpu_gate_cover_utf16_credentials_and_paths(self):
        private = 'C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'build'
        secret_values = (private, 'sk-'+'x'*30, 'sk-proj-'+'x'*30, 'hf_'+'x'*30,
                         '-----BEGIN '+'RSA PRIVATE KEY-----')
        release.check_cpu_payload(b'bounded safe native bytes',{'BOVARY_SANITIZE_PATHS':'ON'})
        with self.assertRaises(ValueError):release.check_cpu_payload(b'safe',{})
        for value in secret_values:
            for encoding in ('ascii','utf-16le','utf-16be'):
                payload=value.encode(encoding)
                with self.subTest(prefix=value[:4],encoding=encoding):
                    self.assertTrue(release.has_private_payload(payload))
                    with self.assertRaises(ValueError):release.check_cpu_payload(payload,{'BOVARY_SANITIZE_PATHS':'ON'})
        self.assertFalse(release.has_private_payload('Public caf\u00e9 model/path'.encode('utf-16le')))
        self.assertFalse(release.has_private_payload(b'\0disk-size-versus-quality-evidence\0'))
        with mock.patch.object(release,'MAX_PAYLOAD_BYTES',4),self.assertRaisesRegex(ValueError,'scan bound'):
            release.has_private_payload(b'12345')

    def test_source_scan_rejects_utf16_credentials_and_private_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);allowlist=root/'public-files.txt';allowlist.write_text('asset.png\n')
            asset=root/'asset.png'
            private='C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'build'
            for value in (private,'sk-'+'x'*30,'hf_'+'x'*30):
                for encoding in ('ascii','utf-16le','utf-16be'):
                    asset.write_bytes(b'asset prefix\0'+value.encode(encoding))
                    with self.subTest(prefix=value[:3],encoding=encoding),self.assertRaisesRegex(ValueError,'credential/private'):
                        release.collect(root,allowlist)

    @unittest.skipUnless(shutil.which("git"), "Git required for dependency bootstrap")
    def test_fresh_bootstrap_idempotence_and_unexpected_source_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            upstream.mkdir()
            env = {**os.environ, "GIT_AUTHOR_NAME": "Bootstrap fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                   "GIT_COMMITTER_NAME": "Bootstrap fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
            def run(*args):
                return subprocess.check_output(["git", "-C", str(upstream), *args], env=env, stderr=subprocess.DEVNULL).decode().strip()
            run("init")
            (upstream / "value.txt").write_text("old\n", encoding="utf-8")
            run("add", "value.txt")
            run("commit", "-m", "Ephemeral bootstrap test fixture")
            revision = run("rev-parse", "HEAD")
            patch = root / "fixture.patch"
            patch.write_bytes(b"diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-old\n+new\n")
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps({"repository": str(upstream), "revision": revision,
                                           "patch": str(patch), "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}))
            target = root / "fetched"
            self.assertTrue(bootstrap.bootstrap(target, lock_path=lock_path)["verified"])
            self.assertEqual((target / "value.txt").read_text(), "new\n")
            self.assertTrue(bootstrap.bootstrap(target, check=True, lock_path=lock_path)["verified"])
            (target / "value.txt").write_text("user edit\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                bootstrap.bootstrap(target, lock_path=lock_path)
            self.assertEqual((target / "value.txt").read_text(), "user edit\n")
            (target / "value.txt").write_text("new\n")
            (target / "unexpected.cpp").write_text("user source")
            with self.assertRaisesRegex(RuntimeError, "unexpected.cpp"):
                bootstrap.bootstrap(target, lock_path=lock_path)

    def test_source_boundary_rejects_private_paths_secrets_and_large_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowlist = root / "public-files.txt"
            safe = root / "safe.py"
            safe.write_text("print('safe')\n")
            allowlist.write_text("safe.py\n")
            self.assertEqual(list(release.collect(root, allowlist)), ["safe.py"])
            for name in ("../escape.py", "work/run.py", "model.gguf", "C:/secret.py"):
                allowlist.write_text(name + "\n")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    release.collect(root, allowlist)
            allowlist.write_text("safe.py\n")
            for content in ("sk-proj-" + "x" * 30, "C:" + chr(92) + "Users" + chr(92) + "private", "x" * 2_000_001):
                safe.write_text(content)
                with self.assertRaises(ValueError):
                    release.collect(root, allowlist)

    def test_source_boundary_rejects_case_collisions_and_noncanonical_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowlist = root / "public-files.txt"
            (root / "safe.py").write_text("pass\n")
            for names, message in (
                ("safe.py\nSAFE.py\n", "case-colliding"),
                ("./safe.py\n", "Unsafe public path"),
                ("WORK/run.py\n", "Unsafe public path"),
            ):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, message):
                    allowlist.write_text(names)
                    release.collect(root, allowlist)

    def test_source_zip_is_deterministic_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a.zip", Path(temporary) / "b.zip"
            release.archive(a, {"example.py": b"safe"})
            release.archive(b, {"example.py": b"safe"})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with self.assertRaises(FileExistsError):
                release.archive(a, {})

    @unittest.skipUnless(shutil.which("git"), "Git required for public diff")
    def test_public_diff_includes_binary_and_applies_to_empty_directory(self):
        files = {"example.py": b"print('safe')\n", "asset.png": b"\x89PNG\x00\xff\x00binary"}
        patch = release.first_commit_diff(files)
        self.assertIn("GIT binary patch", patch)
        self.assertNotIn("b/public/", patch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_path = root / "review.patch"
            patch_path.write_text(patch, encoding="utf-8", newline="")
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch_path)], cwd=root, check=True, capture_output=True)
            for name, data in files.items():
                self.assertEqual((root / name).read_bytes(), data)

    def test_bootstrap_wrong_patch_hash_fails_before_touching_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"patch": "patches/llama-streaming.patch", "patch_sha256": "wrong"}))
            target = root / "engine"
            with self.assertRaisesRegex(RuntimeError, "Patch hash"):
                bootstrap.bootstrap(target, lock_path=lock)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
