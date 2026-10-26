"""Regression for a real macOS tempfile precedence failure; only synthetic data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TemporaryIsolationTests(unittest.TestCase):
    def test_parent_tmpdir_cannot_escape_synthetic_guard(self):
        # Removing the child TMPDIR override must fail on every OS, not just Mac.
        with tempfile.TemporaryDirectory(prefix='outside synthetic 中文 ') as outer:
            proc = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/check_afqmc_synthetic.py')],
                cwd=outer, env={**os.environ, 'TMPDIR': outer}, capture_output=True,
                text=True, encoding='utf-8')
            self.assertEqual(proc.returncode, 0, proc.stderr[-6000:])
            record = json.loads(proc.stdout)
            self.assertEqual(record['runs'][0]['result']['tests_run'], 49)
            self.assertEqual(record['runs'][0]['result']['failures'], 0)
            self.assertTrue(record['runs'][1]['result']['passed'])
            self.assertFalse(record['raw_afqmc_read'])
            self.assertEqual(list(Path(outer).iterdir()), [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
