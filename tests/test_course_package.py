"""Adversarial, self-contained ZIP tests; no Git history or AFQMC inputs."""
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('build_course_package'), 'package builder missing')
        self.p = importlib.import_module('build_course_package')
        self.tmp = tempfile.TemporaryDirectory(prefix='解压 安全 ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'hello.py').write_bytes(b'print("hello")\n')
        self.zip = self.root / 'course.zip'

    def build(self, names=None):
        return self.p.write_package(self.root, names or ['hello.py'], self.zip, 'a' * 40)

    def rewrite(self, change):
        with zipfile.ZipFile(self.zip) as source:
            members = [(i, source.read(i)) for i in source.infolist()]
        with zipfile.ZipFile(self.zip, 'w') as target:
            for info, data in members:
                name, content = change(info.filename, data)
                target.writestr(name, content)

    def test_reproducible_bytes_and_real_extraction(self):
        a = self.build()
        second = self.root / 'second.zip'
        b = self.p.write_package(self.root, ['hello.py'], second, 'a' * 40)
        self.assertEqual(self.zip.read_bytes(), second.read_bytes())
        dest = self.root / '中文 新目录'
        report = self.p.verify_package(self.zip, dest)
        self.assertEqual((dest / 'afqmc-course-source/hello.py').read_bytes(), b'print("hello")\n')
        self.assertEqual(report['files'], 1)
        self.assertEqual(a['sha256'], hashlib.sha256(self.zip.read_bytes()).hexdigest())

    def test_modified_member_rejected_before_any_extraction(self):
        self.build()
        self.rewrite(lambda name, data: (name, b'changed' if name.endswith('hello.py') else data))
        dest = self.root / 'must-not-exist'
        with self.assertRaises(self.p.DeliveryError):
            self.p.verify_package(self.zip, dest)
        self.assertFalse(dest.exists())

    def test_mac_launcher_keeps_executable_bit(self):
        (self.root / 'Run_Mac.command').write_bytes(b'#!/bin/sh\nexit 0\n')
        self.build(['Run_Mac.command'])
        with zipfile.ZipFile(self.zip) as archive:
            self.assertEqual((archive.getinfo(self.p.PREFIX + 'Run_Mac.command').external_attr >> 16) & 0o777, 0o755)
        import os
        if os.name != 'nt':
            out = self.root / 'executable'
            self.p.verify_package(self.zip, out)
            self.assertTrue(os.access(out / self.p.PREFIX / 'Run_Mac.command', os.X_OK))

    def test_path_traversal_rejected(self):
        self.build()
        self.rewrite(lambda name, data: ('../escape.py' if name.endswith('hello.py') else name, data))
        with self.assertRaises(self.p.DeliveryError):
            self.p.verify_package(self.zip)
        self.assertFalse((self.root.parent / 'escape.py').exists())

    def test_unlisted_member_rejected(self):
        self.build()
        with zipfile.ZipFile(self.zip, 'a') as z:
            z.writestr('afqmc-course-source/extra.py', b'rogue')
        with self.assertRaises(self.p.DeliveryError):
            self.p.verify_package(self.zip)

    def test_case_collision_rejected(self):
        (self.root / 'Hello.PY').write_bytes(b'case')
        with self.assertRaises(self.p.DeliveryError):
            self.build(['hello.py', 'Hello.PY'])
        self.assertFalse(self.zip.exists())

    def test_windows_paths_reserved_and_private_paths_rejected(self):
        for name in ('../escape', '/absolute', 'C:/data', 'a\\b', 'NUL.txt', 'a:stream', 'tail.', '.git/config', 'local_only/report.md', 'stage01_data_audit/raw/data.zip', 'private.docx', 'venv/bin/python'):
            with self.subTest(name=name), self.assertRaises(self.p.DeliveryError):
                self.p.validate_package_names([name])

    def test_source_symlink_rejected(self):
        link = self.root / 'link.py'
        try:
            link.symlink_to(self.root / 'hello.py')
        except OSError:
            self.skipTest('OS account cannot create symlinks')
        with self.assertRaises(self.p.DeliveryError):
            self.build(['link.py'])
        self.assertFalse(self.zip.exists())

    def test_existing_zip_and_extraction_are_preserved(self):
        self.build(); old = self.zip.read_bytes()
        with self.assertRaises(self.p.DeliveryError):
            self.build()
        self.assertEqual(self.zip.read_bytes(), old)
        dest = self.root / 'existing'; dest.mkdir()
        with self.assertRaises(self.p.DeliveryError):
            self.p.verify_package(self.zip, dest)
        self.assertEqual(list(dest.iterdir()), [])

    def test_scanner_accepts_its_own_source_but_rejects_private_path(self):
        self.p.scan_text('scripts/build_course_package.py', (ROOT / 'scripts/build_course_package.py').read_bytes())
        with self.assertRaises(self.p.DeliveryError):
            self.p.scan_text('leak.txt', ('/' + 'Users/person/private.txt').encode())

    def test_file_directory_conflict_rejected(self):
        with self.assertRaises(self.p.DeliveryError):
            self.p.validate_package_names(['a', 'a/b.py'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
