"""Launcher regressions: real files/processes; never downloads or runs official Dev."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('start_course'), 'one-click launcher is not implemented')
        self.m = importlib.import_module('start_course')
        self.tmp = tempfile.TemporaryDirectory(prefix='启动 中文 空格 ')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name).resolve()

    def copy_static(self):
        root = self.work / '课程 源码'; root.mkdir()
        baseline = json.loads((ROOT / 'delivery/frozen_baseline.json').read_text(encoding='utf-8'))
        names = [*baseline['files'], 'delivery/frozen_baseline.json',
                 'delivery/requirements-cp312-macos-arm64.lock.txt',
                 'stage05_p1_closure_review/requirements-cp312-win_amd64.lock.txt']
        for name in names:
            dest = root / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        return root

    def test_static_preflight_accepts_real_frozen_project_without_creating_runtime(self):
        root = self.copy_static()
        result = self.m.static_preflight(root)
        self.assertEqual(result['frozen']['frozen_source_count'], 34)
        self.assertFalse((root / '.course_runtime').exists())

    def test_changed_hash_lock_blocks_before_runtime_or_dependency_install(self):
        root = self.copy_static()
        from course_delivery import lock_path
        lock = root / lock_path(); lock.write_bytes(lock.read_bytes() + b'\n')
        with self.assertRaisesRegex(self.m.LaunchError, 'lock'):
            self.m.static_preflight(root)
        self.assertFalse((root / '.course_runtime').exists())

    def test_changed_frozen_science_blocks_before_runtime(self):
        root = self.copy_static()
        source = root / 'stage03a_internal_experiment/src/features.py'
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaisesRegex(self.m.LaunchError, 'fingerprint'):
            self.m.static_preflight(root)
        self.assertFalse((root / '.course_runtime').exists())

    def test_runtime_symlink_cannot_write_outside_project(self):
        root = self.work / 'source'; root.mkdir()
        outside = self.work / 'outside'; outside.mkdir()
        try:
            (root / '.course_runtime').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Account cannot create symlinks')
        with self.assertRaises(self.m.LaunchError):
            self.m.runtime_directory(root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_offline_empty_runtime_never_installs_or_creates_environment(self):
        root = self.copy_static()
        with self.assertRaises(self.m.LaunchError) as caught:
            self.m.ensure_environment(root, self.work / 'nonexistent-uv', offline=True)
        self.assertEqual(caught.exception.code, 20)
        self.assertFalse((root / '.course_runtime/envs').exists())

    def test_summary_pointer_refuses_symlink_and_preserves_target(self):
        root = self.work / 'source'; root.mkdir()
        runtime = self.m.runtime_directory(root)
        outside = self.work / 'keep'; outside.write_text('preserve')
        try:
            (runtime / 'last_run.json').symlink_to(outside)
        except OSError:
            self.skipTest('Account cannot create symlinks')
        with self.assertRaises(self.m.LaunchError):
            self.m.write_latest(root, {'passed': True})
        self.assertEqual(outside.read_text(), 'preserve')

    def test_latest_pointer_has_no_absolute_paths_and_leaves_old_result(self):
        root = self.work / 'source'; root.mkdir()
        old = root / 'course_validation/previous'; old.mkdir(parents=True)
        marker = old / 'summary.json'; marker.write_text('old result')
        summary = {'passed': True, 'output': 'course_validation/new', 'dev_evaluation': False}
        self.m.write_latest(root, summary)
        self.assertEqual(json.loads((root / '.course_runtime/last_run.json').read_text()), summary)
        self.assertEqual(marker.read_text(), 'old result')
        self.assertNotIn(str(root), (root / '.course_runtime/last_run.json').read_text())

    def test_runtime_environment_ignores_external_uv_and_python_overrides(self):
        env = self.m.runtime_environment(self.work, offline=True, inherited={
            'HOME': '/do-not-change', 'PATH': '/some/path', 'UV_INDEX_URL': 'http://wrong.invalid',
            'UV_PYTHON_INSTALL_DIR': '/outside', 'PYTHONPATH': '/outside', 'VIRTUAL_ENV': '/outside',
            'PIP_TARGET': '/outside', 'HTTPS_PROXY': 'http://proxy.invalid'})
        self.assertEqual(env['HOME'], '/do-not-change')
        self.assertEqual(env['HTTPS_PROXY'], 'http://proxy.invalid')
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PIP_TARGET', env)
        self.assertNotIn('UV_INDEX_URL', env)
        self.assertEqual(env['UV_OFFLINE'], '1')
        self.assertEqual(env['UV_PYTHON_INSTALL_REGISTRY'], '0')
        self.assertEqual(env['UV_PYTHON_INSTALL_DIR'], str(self.work / 'python'))

    def test_summary_distinguishes_actual_toy_scores_from_saved_dev_metrics(self):
        synthetic = {'checks': [{'name': 'synthetic', 'result': {'runs': [
            {'name': 'tests', 'result': {'tests_run': 49, 'passed': True}},
            {'name': 'demo', 'result': {'passed': True, 'rows': [
                {'example_id': 'pair', 'method': 'count_char2', 'score': 1 / 3}],
                'formal_afqmc_model_used': False}}]}}]}
        result = self.m.demonstration_summary(synthetic)
        self.assertEqual(result['toy_cosine']['rows'][0]['score'], 1 / 3)
        self.assertFalse(result['saved_official_dev']['recomputed'])
        self.assertEqual(result['saved_official_dev']['tfidf_confusion'], {'tp': 1186, 'fp': 2425, 'fn': 152, 'tn': 553})


class NativeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='启动 入口 ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        names = ['Run_Mac.command', 'Run_Windows.cmd', 'scripts/start_course.ps1']
        for name in names:
            self.assertTrue((ROOT / name).is_file(), 'native launcher is not implemented: ' + name)
            dest = self.root / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        if platform.system() == 'Darwin':
            self.command = ['/bin/bash', str(self.root / 'Run_Mac.command')]
            self.archive = self.root / '.course_runtime/uv-0.13.5-macos-arm64.tar.gz'
        elif platform.system() == 'Windows':
            self.command = ['cmd.exe', '/d', '/c', str(self.root / 'Run_Windows.cmd')]
            self.archive = self.root / '.course_runtime/uv-0.13.5-windows-x64.zip'
        else:
            self.skipTest('Native entry is only supported on macOS/Windows')

    def run_entry(self, *args):
        return subprocess.run([*self.command, '--no-pause', *args], cwd=self.root.parent,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)

    def test_help_is_available_without_network_python_or_runtime(self):
        result = self.run_entry('--help')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--offline', result.stdout)
        self.assertFalse((self.root / '.course_runtime').exists())

    def test_unknown_argument_fails_before_runtime(self):
        result = self.run_entry('--not-an-option')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse((self.root / '.course_runtime').exists())

    def test_cold_offline_fails_clearly_without_network(self):
        result = self.run_entry('--offline')
        self.assertEqual(result.returncode, 10, result.stdout + result.stderr)
        self.assertIn('OFFLINE', result.stdout + result.stderr)
        self.assertFalse(self.archive.exists())

    def test_corrupt_cached_uv_is_never_executed_or_overwritten(self):
        self.archive.parent.mkdir(); self.archive.write_bytes(b'keep corrupt original')
        result = self.run_entry('--offline')
        self.assertEqual(result.returncode, 11, result.stdout + result.stderr)
        self.assertIn('SHA', result.stdout + result.stderr)
        self.assertEqual(self.archive.read_bytes(), b'keep corrupt original')

    def test_runtime_symlink_fails_before_download(self):
        outside = self.root / 'outside'; outside.mkdir()
        try:
            (self.root / '.course_runtime').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Account cannot create symlinks')
        result = self.run_entry('--offline')
        self.assertEqual(result.returncode, 12, result.stdout + result.stderr)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
