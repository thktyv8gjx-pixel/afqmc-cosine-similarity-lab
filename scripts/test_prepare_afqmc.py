"""Synthetic-only tests of the real preparation functions; no AFQMC opens."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

import prepare_afqmc as prep

ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def geturl(self):
        return prep.URL


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'stage01_data_audit/raw'
        self.raw.mkdir(parents=True)
        self.archive = self.raw / 'afqmc_public.zip'
        self.destination = self.raw.parent / 'split_manifest.csv'
        self.records = {
            'train.json': [{'sentence1': f'author toy left {i}', 'sentence2': f'author toy right {i}', 'label': str(i % 2)} for i in range(20)],
            'dev.json': [{'sentence1': f'author dev left {i}', 'sentence2': f'author dev right {i}', 'label': str(i % 2)} for i in range(4)],
            'test.json': [{'id': 0, 'sentence1': 'author test left', 'sentence2': 'author test right'}],
        }
        self.data = {name: ('\n'.join(json.dumps(r) for r in rows) + '\n').encode() for name, rows in self.records.items()}
        self.members = {name: (len(self.records[name]), len(data), hashlib.sha256(data).hexdigest()) for name, data in self.data.items()}
        self.archive_bytes = self.make_zip(self.data)
        self.digest = hashlib.sha256(self.archive_bytes).hexdigest()
        for name, value in [('MEMBERS', self.members), ('ARCHIVE_SHA256', self.digest)]:
            patch = mock.patch.object(prep, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def make_zip(self, members):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as z:
            for name, data in members.items():
                z.writestr(name, data)
        return buf.getvalue()

    def save(self, data=None):
        self.archive.write_bytes(self.archive_bytes if data is None else data)

    def test_valid_schema_counts_crc_and_hashes(self):
        self.save()
        result = prep.validate_archive(self.archive)
        self.assertEqual(result['train.json']['rows'], 20)
        self.assertEqual(result['test.json']['label_counts'], {})

    def test_wrong_archive_preserved(self):
        self.save(b'wrong version')
        with self.assertRaises(prep.PreparationError):
            prep.obtain_archive(self.archive, offline=False, timeout=1)
        self.assertEqual(self.archive.read_bytes(), b'wrong version')

    def test_download_atomic_and_no_parts(self):
        with mock.patch.object(prep.urllib.request, 'urlopen', return_value=Response(self.archive_bytes)) as request:
            _, mode = prep.obtain_archive(self.archive, offline=False, timeout=7)
        request.assert_called_once_with(prep.URL, timeout=7)
        self.assertEqual(mode, 'downloaded_and_verified')
        self.assertEqual(self.archive.read_bytes(), self.archive_bytes)
        self.assertFalse(list(self.raw.glob('*.part')))

    def test_existing_valid_is_offline_idempotent(self):
        self.save()
        with mock.patch.object(prep.urllib.request, 'urlopen', side_effect=AssertionError('network forbidden')):
            self.assertEqual(prep.obtain_archive(self.archive, offline=False, timeout=1)[1], 'existing_verified')

    def test_network_error_cleans_temporary_and_explains_manual_download(self):
        with mock.patch.object(prep.urllib.request, 'urlopen', side_effect=urllib.error.URLError('fake failure')), contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = prep.main(['--root', str(self.root), '--archive-only'])
        self.assertEqual(code, 2)
        self.assertIn(prep.URL, stderr.getvalue())
        self.assertIn('Manual download', stderr.getvalue())
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_incomplete_download_never_published(self):
        with mock.patch.object(prep.urllib.request, 'urlopen', return_value=Response(self.archive_bytes[:20])):
            with self.assertRaises(prep.PreparationError):
                prep.obtain_archive(self.archive, offline=False, timeout=1)
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_offline_missing_never_network(self):
        with mock.patch.object(prep.urllib.request, 'urlopen', side_effect=AssertionError('network forbidden')):
            with self.assertRaises(prep.PreparationError):
                prep.obtain_archive(self.archive, offline=True, timeout=1)

    def test_redirect_rejected(self):
        response = Response(self.archive_bytes)
        response.geturl = lambda: 'https://example.invalid/unregistered.zip'
        with mock.patch.object(prep.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(prep.PreparationError):
                prep.obtain_archive(self.archive, offline=False, timeout=1)
        self.assertFalse(self.archive.exists())

    def test_download_limit(self):
        with mock.patch.object(prep, 'MAX_DOWNLOAD_BYTES', 1), mock.patch.object(prep.urllib.request, 'urlopen', return_value=Response(self.archive_bytes)):
            with self.assertRaises(prep.PreparationError):
                prep.obtain_archive(self.archive, offline=False, timeout=1)
        self.assertFalse(self.archive.exists())

    def test_member_path_traversal_rejected_without_extraction(self):
        data = self.make_zip({'../secret.json': b'{}'})
        self.save(data)
        with mock.patch.object(prep, 'ARCHIVE_SHA256', hashlib.sha256(data).hexdigest()):
            with self.assertRaises(prep.PreparationError):
                prep.validate_archive(self.archive)
        self.assertFalse((self.root / 'secret.json').exists())

    def test_crc_corruption_rejected(self):
        data = bytearray(self.archive_bytes)
        offset = data.index(self.data['train.json'])
        data[offset + 3] ^= 1
        self.save(data)
        with mock.patch.object(prep, 'ARCHIVE_SHA256', hashlib.sha256(data).hexdigest()):
            with self.assertRaisesRegex(prep.PreparationError, 'CRC'):
                prep.validate_archive(self.archive)

    def test_member_fingerprint_rejected(self):
        self.save()
        bad = dict(self.members)
        bad['dev.json'] = (4, len(self.data['dev.json']), '0' * 64)
        with mock.patch.object(prep, 'MEMBERS', bad), self.assertRaises(prep.PreparationError):
            prep.validate_archive(self.archive)

    def test_row_count_rejected(self):
        with self.assertRaises(prep.PreparationError):
            prep.validate_records('dev.json', self.data['dev.json'], 5)

    def test_missing_fields_rejected_without_text_leak(self):
        secret = 'synthetic_private_marker'
        with self.assertRaises(prep.PreparationError) as raised:
            prep.validate_records('dev.json', json.dumps({'sentence1': secret}).encode(), 1)
        self.assertNotIn(secret, str(raised.exception))

    def test_bool_label_rejected(self):
        data = json.dumps({'sentence1': 'toy a', 'sentence2': 'toy b', 'label': True}).encode()
        with self.assertRaises(prep.PreparationError):
            prep.validate_records('train.json', data, 1)

    def test_test_labels_not_inferred(self):
        data = json.dumps({'id': 0, 'sentence1': 'toy a', 'sentence2': 'toy b', 'label': '0'}).encode()
        with self.assertRaises(prep.PreparationError):
            prep.validate_records('test.json', data, 1)

    def test_test_id_required_and_integer(self):
        data = json.dumps({'id': True, 'sentence1': 'toy a', 'sentence2': 'toy b'}).encode()
        with self.assertRaises(prep.PreparationError):
            prep.validate_records('test.json', data, 1)

    def test_test_ids_unique(self):
        row = json.dumps({'id': 0, 'sentence1': 'toy a', 'sentence2': 'toy b'})
        with self.assertRaises(prep.PreparationError):
            prep.validate_records('test.json', (row+'\n'+row).encode(), 2)

    def test_invalid_json_sanitized(self):
        with self.assertRaises(prep.PreparationError) as raised:
            prep.validate_records('dev.json', b'not json secret marker', 1)
        self.assertNotIn('secret marker', str(raised.exception))

    def test_stage1_source_fingerprint_gate(self):
        (self.raw.parent / 'audit_afqmc.py').write_text('raise RuntimeError("must not import")')
        with self.assertRaises(prep.PreparationError):
            prep.load_stage1(self.root)

    def test_dependency_gate(self):
        with mock.patch.object(prep.importlib.metadata, 'version', return_value='0.0'):
            with self.assertRaises(prep.PreparationError):
                prep.check_environment()

    def test_original_writer_and_manifest_pipeline_deterministic(self):
        self.save()
        stage1 = prep.load_stage1(ROOT)
        records, _, issues = stage1.load_archive(self.archive)
        self.assertFalse(issues)
        rows, info = prep.manifest_rows(stage1, records)
        expected = self.raw / 'expected.csv'
        stage1.write_csv(expected, rows, prep.FIELDNAMES)
        digest = prep.sha256_file(expected)
        with mock.patch.object(stage1, 'EXPECTED', {'train': 20, 'dev': 4, 'test': 1}), mock.patch.object(prep, 'load_stage1', return_value=stage1), mock.patch.object(prep, 'MANIFEST_SHA256', digest):
            first = prep.rebuild_manifest(ROOT, self.archive, self.destination)
            second = prep.rebuild_manifest(ROOT, self.archive, self.destination)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(first['rows'], 25)
        self.assertTrue(second['existing_manifest_preserved'])
        self.assertEqual(self.destination.read_bytes(), expected.read_bytes())
        self.assertTrue(expected.read_bytes().startswith(b'\xef\xbb\xbf'))
        self.assertIn(b'\r\n', expected.read_bytes())
        self.assertEqual(info['rows']['fit'] + info['rows']['tune'], 20)

    def test_manifest_hash_mismatch_does_not_publish(self):
        self.save()
        stage1 = prep.load_stage1(ROOT)
        with mock.patch.object(stage1, 'EXPECTED', {'train': 20, 'dev': 4, 'test': 1}), mock.patch.object(prep, 'load_stage1', return_value=stage1):
            with self.assertRaisesRegex(prep.PreparationError, 'Manifest SHA-256 mismatch'):
                prep.rebuild_manifest(ROOT, self.archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_wrong_manifest_rejected_before_record_read(self):
        self.destination.write_bytes(b'wrong historical file')
        with mock.patch.object(prep, 'load_stage1', side_effect=AssertionError('must not read')):
            with self.assertRaises(prep.PreparationError):
                prep.rebuild_manifest(ROOT, self.archive, self.destination)
        self.assertEqual(self.destination.read_bytes(), b'wrong historical file')

    def test_atomic_publication_never_overwrites(self):
        source = self.raw / 'source'
        source.write_bytes(b'candidate')
        self.destination.write_bytes(b'keep')
        with self.assertRaises(prep.PreparationError):
            prep.publish_without_overwrite(source, self.destination)
        self.assertEqual(self.destination.read_bytes(), b'keep')

    def test_archive_only_never_loads_model_or_splitter(self):
        self.save()
        with mock.patch.object(prep, 'rebuild_manifest', side_effect=AssertionError('not allowed')):
            result = prep.prepare(self.root, offline=True, archive_only=True)
        self.assertEqual(result['status'], 'ARCHIVE_VALIDATED')
        self.assertFalse(result['dev_performance_evaluation'])
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
