"""Safety boundaries for the additive course entry; no AFQMC is read."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

class CourseDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('course_delivery'), 'delivery wrapper is not implemented')
        self.c = importlib.import_module('course_delivery')
        self.tmp = tempfile.TemporaryDirectory(prefix='课程 验证 ')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def copied_freeze(self):
        root = self.work / 'copy'
        baseline = json.loads((ROOT / 'delivery/frozen_baseline.json').read_text(encoding='utf-8'))
        for name in [*baseline['files'], 'delivery/frozen_baseline.json']:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        return root

    def test_actual_frozen_sources_and_saved_counts(self):
        r = self.c.verify_frozen(ROOT)
        self.assertEqual(r['frozen_source_count'], 34)
        self.assertEqual(r['tfidf_confusion'], {'tp': 1186, 'fp': 2425, 'fn': 152, 'tn': 553})
        self.assertFalse(r['dev_replayed'])

    def test_single_byte_source_change_is_rejected(self):
        root = self.copied_freeze()
        p = root / 'stage03a_internal_experiment/src/features.py'
        p.write_bytes(p.read_bytes() + b'\n')
        with self.assertRaisesRegex(self.c.DeliveryError, 'fingerprint'):
            self.c.verify_frozen(root)

    def test_crlf_conversion_is_not_normalized_away(self):
        root = self.copied_freeze()
        p = root / 'stage03a_internal_experiment/config/experiment.json'
        p.write_bytes(p.read_bytes().replace(b'\n', b'\r\n'))
        with self.assertRaisesRegex(self.c.DeliveryError, 'fingerprint'):
            self.c.verify_frozen(root)

    def test_saved_metric_change_is_rejected(self):
        root = self.copied_freeze()
        p = root / 'stage03b_official_validation/results/dev_metrics.json'
        p.write_bytes(p.read_bytes().replace(b'0.47928874520105075', b'0.99'))
        with self.assertRaisesRegex(self.c.DeliveryError, 'fingerprint'):
            self.c.verify_frozen(root)

    def test_new_output_refuses_even_an_empty_existing_directory(self):
        with self.assertRaises(self.c.DeliveryError):
            self.c.new_output(self.work)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_new_output_rejects_symlink_parent(self):
        dest = self.work / 'real'; dest.mkdir()
        link = self.work / 'link'
        try:
            link.symlink_to(dest, target_is_directory=True)
        except OSError:
            self.skipTest('OS account cannot create symlinks')
        with self.assertRaises(self.c.DeliveryError):
            self.c.new_output(link / 'output')
        self.assertEqual(list(dest.iterdir()), [])

    def test_new_output_unicode_spaces_works(self):
        dest = self.c.new_output(self.work / '中文 空格')
        self.assertTrue(dest.is_dir())

    def test_missing_train_inputs_names_paths_and_hashes(self):
        with self.assertRaises(self.c.DeliveryError) as error:
            self.c.check_train_inputs(self.work)
        message = str(error.exception)
        self.assertIn('afqmc_public.zip', message)
        self.assertIn('split_manifest.csv', message)
        self.assertIn('5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc', message)
        self.assertFalse((self.work / 'stage01_data_audit').exists())

    def test_existing_wrong_train_input_is_preserved(self):
        p = self.work / 'stage01_data_audit/raw/afqmc_public.zip'
        p.parent.mkdir(parents=True); p.write_bytes(b'keep this invalid input')
        with self.assertRaises(self.c.DeliveryError):
            self.c.check_train_inputs(self.work)
        self.assertEqual(p.read_bytes(), b'keep this invalid input')

    def test_wrong_architecture_never_claims_native_support(self):
        with mock.patch.object(self.c.platform, 'system', return_value='Darwin'), mock.patch.object(self.c.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(self.c.DeliveryError, 'architecture'):
                self.c.environment()

    def test_wrong_python_version_rejected(self):
        with mock.patch.object(self.c.platform, 'python_version', return_value='3.12.11'):
            with self.assertRaisesRegex(self.c.DeliveryError, '3.12.10'):
                self.c.environment()

    def test_help_explains_modes_without_loading_data(self):
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_course_delivery.py'), '--help'], cwd=self.work, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(p.returncode, 0, p.stderr)
        for mode in ('synthetic', 'prepare', 'train', 'frozen'):
            self.assertIn(mode, p.stdout)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_dev_is_not_an_available_mode(self):
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_course_delivery.py'), '--mode', 'dev'], cwd=self.work, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(p.returncode, 2)
        self.assertIn('invalid choice', p.stderr)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_train_dotdot_output_cannot_escape_private_directory(self):
        root = self.copied_freeze().resolve()
        candidate = root / 'local_delivery/../escaped'
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_course_delivery.py'),
                            '--root', str(root), '--mode', 'train', '--output', str(candidate)],
                           capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(p.returncode, 2)
        self.assertFalse((root / 'escaped').exists(), 'private Train output escaped to a publishable path')

    def test_font_resolver_rejects_missing_explicit_font(self):
        with self.assertRaises(self.c.DeliveryError):
            self.c.resolve_font(self.work / 'missing.ttf')

if __name__ == '__main__':
    unittest.main(verbosity=2)
