"""Private packaging boundaries; tiny artificial fixtures contain no AFQMC rows."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('build_course_submission'), 'private submission builder missing')
        self.s = importlib.import_module('build_course_submission')
        self.p = importlib.import_module('build_course_package')
        self.tmp = tempfile.TemporaryDirectory(prefix='私有 提交 ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'Run_Mac.command').write_bytes(b'#!/bin/sh\nexit 0\n')
        self.base = self.root / 'source.zip'
        self.p.write_package(self.root, ['Run_Mac.command'], self.base, 'a' * 40)
        self.data = {'stage01_data_audit/raw/afqmc_public.zip': b'artificial archive fixture',
                     'stage01_data_audit/split_manifest.csv': b'artificial manifest fixture'}
        self.hashes = {n: hashlib.sha256(d).hexdigest() for n, d in self.data.items()}
        self.patch = patch.object(self.s, 'DATA_FILES', self.hashes)
        self.patch.start(); self.addCleanup(self.patch.stop)
        for name, data in self.data.items():
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.dest = self.root / 'private.zip'

    def build(self):
        return self.s.build_submission(self.base, self.root, self.dest)

    def test_roundtrip_and_reproducibility_with_exact_private_allowlist(self):
        report = self.build()
        other = self.root / 'other.zip'
        self.s.build_submission(self.base, self.root, other)
        self.assertEqual(self.dest.read_bytes(), other.read_bytes())
        out = self.root / '解压 新路径'
        self.s.verify_submission(self.dest, out)
        for name, data in self.data.items():
            self.assertEqual((out / self.p.PREFIX / name).read_bytes(), data)
        self.assertEqual(report['data_files'], 2)
        self.assertEqual(report['source_sha256'], hashlib.sha256(self.base.read_bytes()).hexdigest())

    def test_wrong_raw_fingerprint_prevents_creation(self):
        (self.root / next(iter(self.data))).write_bytes(b'wrong version')
        with self.assertRaises(self.s.DeliveryError):
            self.build()
        self.assertFalse(self.dest.exists())

    def test_unlisted_private_file_is_not_collected(self):
        (self.root / 'secret.env').write_bytes(b'private')
        self.build()
        with zipfile.ZipFile(self.dest) as archive:
            self.assertFalse(any('secret' in n for n in archive.namelist()))

    def test_public_verifier_still_rejects_private_package(self):
        self.build()
        with self.assertRaises(self.p.DeliveryError):
            self.p.verify_package(self.dest)

    def test_unlisted_member_rejected_before_extraction(self):
        self.build()
        with zipfile.ZipFile(self.dest, 'a') as archive:
            archive.writestr(self.p.PREFIX + 'unlisted.txt', b'rogue')
        out = self.root / 'not-created'
        with self.assertRaises(self.s.DeliveryError):
            self.s.verify_submission(self.dest, out)
        self.assertFalse(out.exists())

    def test_tampered_member_rejected_before_extraction(self):
        self.build()
        with zipfile.ZipFile(self.dest) as archive:
            members = [(i, archive.read(i)) for i in archive.infolist()]
        with zipfile.ZipFile(self.dest, 'w') as archive:
            for info, data in members:
                archive.writestr(info, b'changed' if info.filename.endswith('split_manifest.csv') else data)
        out = self.root / 'not-created'
        with self.assertRaises(self.s.DeliveryError):
            self.s.verify_submission(self.dest, out)
        self.assertFalse(out.exists())

    def test_traversal_rejected_before_extraction(self):
        self.build()
        with zipfile.ZipFile(self.dest, 'a') as archive:
            archive.writestr('../escape.txt', b'rogue')
        out = self.root / 'not-created'
        with self.assertRaises(self.s.DeliveryError):
            self.s.verify_submission(self.dest, out)
        self.assertFalse(out.exists())

    def test_existing_files_preserved(self):
        self.build(); before = self.dest.read_bytes()
        with self.assertRaises(self.s.DeliveryError):
            self.build()
        self.assertEqual(before, self.dest.read_bytes())
        out = self.root / 'already-there'; out.mkdir()
        with self.assertRaises(self.s.DeliveryError):
            self.s.verify_submission(self.dest, out)
        self.assertEqual(list(out.iterdir()), [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
