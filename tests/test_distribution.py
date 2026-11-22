import importlib.metadata
import gzip
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_distribution as distribution
from scripts import prepare_release as release


class DistributionTests(unittest.TestCase):
    def test_public_manifest_matches_complete_allowlist(self):
        self.assertEqual(set(distribution.check_manifest()), set(release.collect(ROOT, ROOT / 'public-files.txt')))

    @staticmethod
    def fixture(root):
        names = ('README.md', 'MANIFEST.in', 'public-files.txt', 'pyproject.toml', 'LICENSE', 'python/bovary/__init__.py')
        (root / 'README.md').write_bytes(b'Public source fixture\n')
        (root / 'LICENSE').write_bytes(b'MIT fixture\n')
        (root / 'python/bovary').mkdir(parents=True)
        (root / 'python/bovary/__init__.py').write_bytes(b'__version__="1.0a1"\n')
        (root / 'pyproject.toml').write_text('[project]\nname = "fixture-route"\nversion = "1.0a1"\n'
            'description = "Public fixture"\nrequires-python = ">=3.11"\nlicense = "MIT"\n'
            'license-files = ["LICENSE"]\nreadme = "README.md"\n'
            '[project.scripts]\nfixture-route = "bovary.cli:main"\n', encoding='utf-8')
        (root / 'public-files.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        (root / 'MANIFEST.in').write_text(distribution.manifest_text(names), encoding='utf-8')
        return release.collect(root, root / 'public-files.txt')

    @staticmethod
    def content(expected):
        egg = 'python/fixture_route.egg-info/'
        metadata = (b'Metadata-Version: 2.4\nName: fixture-route\nVersion: 1.0a1\nSummary: Public fixture\n'
                    b'Requires-Python: >=3.11\nLicense-Expression: MIT\nDescription-Content-Type: text/markdown\n'
                    b'License-File: LICENSE\nDynamic: license-file\n\n') + expected['README.md']
        result = {**expected, 'PKG-INFO':metadata, egg+'PKG-INFO':metadata,
                  'setup.cfg':b'[egg_info]\ntag_build = \ntag_date = 0\n\n',
                  egg+'dependency_links.txt':b'\n', egg+'top_level.txt':b'bovary\n',
                  egg+'entry_points.txt':b'[console_scripts]\nfixture-route = bovary.cli:main\n',
                  egg+'SOURCES.txt':b''}
        result[egg+'SOURCES.txt'] = ('\n'.join(sorted(set(result)-{'PKG-INFO','setup.cfg'}))+'\n').encode()
        return result

    @staticmethod
    def write_archive(path, content, *, prefix='fixture_route-1.0a1', extra=None):
        with tarfile.open(path, 'w:gz') as output:
            for name, data in content.items():
                info = tarfile.TarInfo(prefix + '/' + name)
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            if extra:
                info = tarfile.TarInfo(extra)
                info.size = 0
                output.addfile(info, io.BytesIO())
        return path

    def test_sdist_rejects_missing_unlisted_or_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            for kind in ('good', 'missing', 'unlisted', 'changed', 'wrong-metadata'):
                content = self.content(expected)
                if kind == 'missing':
                    content.pop('README.md')
                elif kind == 'unlisted':
                    content['work/private-run.json'] = b'{"synthetic_canary":true}'
                elif kind == 'changed':
                    content['README.md'] = b'Unreviewed content'
                elif kind == 'wrong-metadata':
                    content['python/other_distribution.egg-info/SOURCES.txt'] = content.pop('python/fixture_route.egg-info/SOURCES.txt')
                archive = root / (kind + '.tar.gz')
                self.write_archive(archive, content)
                with self.subTest(kind=kind):
                    if kind == 'good':
                        self.assertTrue(distribution.check_sdist(archive, root)['source_bytes_match'])
                    else:
                        with self.assertRaises(ValueError):
                            distribution.check_sdist(archive, root)

    def test_generated_metadata_rejects_wrong_identity_readme_license_and_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = self.content(self.fixture(root))
            for name in ('PKG-INFO', 'python/fixture_route.egg-info/PKG-INFO'):
                for old, new in ((b'Name: fixture-route',b'Name: wrong'), (b'Version: 1.0a1',b'Version: 2.0'),
                        (b'License-Expression: MIT',b'License-Expression: Apache-2.0'),
                        (b'License-File: LICENSE\n',b''), (b'Public source fixture',b'Outdated README'),
                        (b'\n\n',b'\nRequires-Dist: unexpected\n\n'),
                        (b'Name: fixture-route\n',b'Name: fixture-route\nName: fixture-route\n')):
                    archive = self.write_archive(root/'malformed.tar.gz', {**content,name:content[name].replace(old,new)})
                    with self.subTest(name=name, changed=new), self.assertRaises(ValueError):
                        distribution.check_sdist(archive, root)

    def test_generated_build_configuration_commands_and_sources_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = self.content(self.fixture(root));egg='python/fixture_route.egg-info/'
            changes = {'command':{egg+'entry_points.txt':b'[console_scripts]\nfixture-route = other:main\n'},
                       'extra-command':{egg+'entry_points.txt':content[egg+'entry_points.txt']+b'old = bovary.cli:main\n'},
                       'top-level':{egg+'top_level.txt':b'other\n'},
                       'dependency-link':{egg+'dependency_links.txt':b'https://example.invalid\n'},
                       'build-settings':{'setup.cfg':content['setup.cfg']+b'[options]\nscripts = other.py\n'},
                       'tag':{'setup.cfg':content['setup.cfg'].replace(b'tag_date = 0',b'tag_date = 1')},
                       'sources-missing':{egg+'SOURCES.txt':content[egg+'SOURCES.txt'].replace(b'README.md\n',b'')},
                       'sources-extra':{egg+'SOURCES.txt':content[egg+'SOURCES.txt']+b'private.txt\n'},
                       'sources-duplicate':{egg+'SOURCES.txt':content[egg+'SOURCES.txt']+b'README.md\n'},
                       'unreviewed-metadata':{egg+'requires.txt':b'unexpected-dependency\n'}}
            for kind, changed in changes.items():
                archive = self.write_archive(root/'malformed.tar.gz',{**content,**changed})
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    distribution.check_sdist(archive, root)
            for name in distribution.generated_metadata(root):
                archive = self.write_archive(root/'missing.tar.gz',{n:d for n,d in content.items() if n!=name})
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    distribution.check_sdist(archive, root)

    def test_checked_extraction_refuses_unsafe_archives_and_existing_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);content=self.content(self.fixture(root))
            archive=self.write_archive(root/'good.tar.gz',content)
            destination=root/'fresh source'
            distribution.extract_sdist(archive,destination,root)
            self.assertEqual({p.relative_to(destination).as_posix():p.read_bytes() for p in destination.rglob('*') if p.is_file()},content)
            with self.assertRaises(FileExistsError):distribution.extract_sdist(archive,destination,root)
            for extra in ('fixture_route-1.0a1/../escape.py','fixture_route-1.0a1/README.md',
                          'fixture_route-1.0a1/readme.MD','fixture_route-1.0a1/CON.txt'):
                archive=self.write_archive(root/'bad.tar.gz',content,extra=extra)
                with self.subTest(extra=extra),self.assertRaises(ValueError):
                    distribution.extract_sdist(archive,root/'not-created',root)
                self.assertFalse((root/'not-created').exists())
            archive=self.write_archive(root/'prefix.tar.gz',content,prefix='other-1.0')
            with self.assertRaisesRegex(ValueError,'canonical'):distribution.check_sdist(archive,root)

    def test_source_archive_bounds_and_generated_metadata_privacy(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);content=self.content(self.fixture(root))
            archive=self.write_archive(root/'good.tar.gz',content)
            for field,value in (('MAX_SDIST_BYTES',archive.stat().st_size-1),('MAX_SDIST_BYTES',8000),('MAX_SDIST_MEMBERS',1)):
                with self.subTest(field=field),mock.patch.object(distribution,field,value),self.assertRaises(ValueError):
                    distribution.check_sdist(archive,root)
            for encoding in ('ascii','utf-16le','utf-16be'):
                private='C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'source'
                for value in (private,'sk-'+'x'*30):
                    archive=self.write_archive(root/'private.tar.gz',{**content,'setup.cfg':value.encode(encoding)})
                    with self.subTest(encoding=encoding,value=value[:3]),self.assertRaisesRegex(ValueError,'private payload'):
                        distribution.check_sdist(archive,root)

    def test_real_sdist_preserves_every_reviewed_source_byte(self):
        try:
            versions = {name: importlib.metadata.version(name) for name in ('setuptools', 'wheel')}
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('Install requirements-build.txt for the actual source-distribution regression')
        if versions != {'setuptools': '80.9.0', 'wheel': '0.45.1'}:
            self.skipTest('Install requirements-build.txt for the actual source-distribution regression')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source'
            root.mkdir()
            for name, data in distribution.check_manifest().items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            out = Path(temporary) / 'dist'
            environment = os.environ.copy()
            if environment.get('PYTHONPATH'):
                environment['PYTHONPATH'] = os.pathsep.join(str(Path(entry).resolve()) for entry in environment['PYTHONPATH'].split(os.pathsep) if entry)
            result = subprocess.run([sys.executable, '-c', 'from setuptools.build_meta import build_sdist; import sys; build_sdist(sys.argv[1])', str(out)],
                                    cwd=root, env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archives = list(out.glob('*.tar.gz'))
            self.assertEqual(len(archives), 1)
            self.assertTrue(distribution.check_sdist(archives[0], root)['source_bytes_match'])
            extracted=distribution.extract_sdist(archives[0],Path(temporary)/'extracted source',root)
            wheel_out=Path(temporary)/'rebuilt wheels'
            result=subprocess.run([sys.executable,'-c','from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])',str(wheel_out)],
                                  cwd=extracted,env=environment,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            wheels=list(wheel_out.glob('*.whl'));self.assertEqual(len(wheels),1)
            self.assertTrue(release.check_wheel(wheels[0],distribution.check_manifest(root),release.project_metadata(root))['record_verified'])

    def test_tar_container_privacy_covers_pax_global_headers_and_encoded_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);content=self.content(self.fixture(root));path=root/'fixture_route-1.0a1.tar.gz'
            private='C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'build'
            def write_container(location, payload):
                # PAX values are UTF-8 strings, but may themselves carry embedded
                # UTF-16 byte sequences. Exercise the actual emitted header bytes.
                value=payload.decode('latin-1')
                with tarfile.open(path,'w:gz',format=tarfile.PAX_FORMAT,
                                  pax_headers={'comment':value} if location=='global-pax' else None) as archive:
                    for index,(name,data) in enumerate(content.items()):
                        item=tarfile.TarInfo('fixture_route-1.0a1/'+name);item.size=len(data)
                        if index==0:
                            if location=='member-pax':item.pax_headers={'comment':value}
                            if location=='header-field':item.linkname=value
                        archive.addfile(item,io.BytesIO(data))
                if location=='tar-padding':
                    path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes())+payload))
                elif location.startswith('gzip-'):
                    raw=gzip.compress(gzip.decompress(path.read_bytes()));assert raw[:4]==b'\x1f\x8b\x08\x00'
                    if location=='gzip-extra':
                        field=b'BC'+len(payload).to_bytes(2,'little')+payload
                        flags,header=4,len(field).to_bytes(2,'little')+field
                    else:
                        # Gzip filenames/comments are NUL-terminated, so only
                        # FEXTRA can carry a complete embedded UTF-16 value.
                        flags,header=(8 if location=='gzip-filename' else 16),payload+b'\0'
                    path.write_bytes(raw[:3]+bytes([flags])+raw[4:10]+header+raw[10:])
            for location in ('member-pax','global-pax','header-field','tar-padding','gzip-filename','gzip-comment','gzip-extra'):
                write_container(location,b'Reviewed safe archive metadata')
                self.assertTrue(distribution.check_sdist(path,root)['generated_metadata_verified'])
                for value in (private,'sk-'+'x'*30,'hf_'+'x'*30):
                    for encoding in (('ascii',) if location in ('gzip-filename','gzip-comment') else ('ascii','utf-16le','utf-16be')):
                        write_container(location,value.encode(encoding))
                        with self.subTest(location=location,encoding=encoding,prefix=value[:3]),self.assertRaisesRegex(ValueError,'private payload'):
                            distribution.check_sdist(path,root)

    @unittest.skipUnless(shutil.which('git'), 'Git required for tracked-file boundary checks')
    def test_tracked_boundary_rejects_extra_or_untracked_public_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
            git('init')
            git('add', '--', *expected)
            release.check_tracked(root, expected)
            (root / 'private-run-canary.json').write_text('{"synthetic_canary":true}')
            git('add', '--', 'private-run-canary.json')
            with self.assertRaisesRegex(ValueError, 'private-run-canary.json'):
                release.check_tracked(root, expected)
            git('rm', '--cached', '--', 'private-run-canary.json')
            with self.assertRaisesRegex(ValueError, 'not tracked'):
                release.check_tracked(root, set(expected) | {'new-public.py'})


if __name__ == '__main__':
    unittest.main()
