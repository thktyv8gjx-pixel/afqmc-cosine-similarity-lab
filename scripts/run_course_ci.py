"""Native CI driver with exact exit codes and explicitly selected safe artifacts.

Runs installation, worktree validation and a new ZIP extraction/new virtualenv.
Only --prepare performs official data preparation, after the data-free checks.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv
import tempfile

from course_delivery import ROOT, DeliveryError, lock_path, new_output, sha, write_json_new


def safe_diagnostic(text: str, root: Path = ROOT) -> str:
    """Bounded data-free test diagnostics; reject credentials after path redaction."""
    from build_course_package import scan_text
    for path in (str(root), str(Path.home()), tempfile.gettempdir()):
        for form in (path, path.replace('\\', '/'), path.replace('\\', '\\\\')):
            text = text.replace(form, '<local-path>')
    try:
        scan_text('ci-diagnostic.txt', text.encode('utf-8'))
    except DeliveryError:
        return 'Diagnostic withheld by private-path/credential scan'
    return text[-12000:]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-system', required=True, choices=['Darwin', 'Windows'])
    p.add_argument('--expected-machine', required=True, choices=['arm64', 'amd64'])
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args(argv)
    evidence = new_output(ROOT / 'course_validation/ci')
    record = {'passed': False, 'system': platform.system(), 'machine': platform.machine(),
              'python': platform.python_version(), 'prepare_requested': a.prepare, 'steps': []}

    def execute(name, args, cwd=ROOT, env=None):
        proc = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True,
                              text=True, encoding='utf-8', errors='replace', env=env)
        log = evidence / (name + '.log')
        log.write_text(proc.stdout + proc.stderr, encoding='utf-8', newline='\n')
        record['steps'].append({'name': name, 'exit_code': proc.returncode, 'log_sha256': sha(log)})
        print(f'{name}: exit {proc.returncode}', flush=True)
        if proc.returncode:
            if name in ('worktree-validation', 'zip-validation'):
                validation_output = Path(args[args.index('--output') + 1])
                test_log = validation_output / 'delivery-tests.log'
                if test_log.is_file():
                    diagnostic = safe_diagnostic(test_log.read_text(encoding='utf-8'), ROOT)
                    record['steps'][-1]['delivery_test_diagnostic'] = diagnostic
                    print(diagnostic, flush=True)
            elif name.startswith('launcher-'):
                diagnostic = safe_diagnostic(proc.stdout + proc.stderr, ROOT)
                record['steps'][-1]['launcher_diagnostic'] = diagnostic
                print(diagnostic, flush=True)
            # The CLI summary is sanitized; pip/build stderr may contain host paths.
            if name.endswith('validation') or name == 'official-preparation':
                try:
                    print(json.loads(proc.stdout).get('error', 'Validation failed; inspect summary artifact'))
                except ValueError:
                    pass
            raise DeliveryError(f'{name} failed')
        return proc

    def make_env(name, base):
        if base.exists():
            raise DeliveryError('CI requires a new virtual environment')
        execute(name, [sys.executable, '-m', 'venv', base])
        return base / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')

    def install(name, python, source):
        execute(name, [python, '-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:',
                       '--require-hashes', '-r', source / lock_path()])
        execute(name + '-pip-check', [python, '-m', 'pip', 'check'])

    try:
        if (platform.system(), platform.machine().lower(), platform.python_version()) != (a.expected_system, a.expected_machine, '3.12.10'):
            raise DeliveryError('Runner architecture/Python does not match the declared native target')
        python = make_env('worktree-venv', ROOT / '.venv')
        install('worktree-install', python, ROOT)
        validation = ROOT / 'course_validation/worktree'
        try:
            execute('worktree-validation', [python, 'scripts/verify_course_delivery.py', '--output', validation])
        finally:
            if (validation / 'summary.json').is_file():
                shutil.copyfile(validation / 'summary.json', evidence / 'worktree-summary.json')
        package = ROOT / 'dist/afqmc-course-source.zip'
        result = execute('package-build', [python, 'scripts/build_course_package.py', '--output', package])
        write_json_new(evidence / 'package-summary.json', json.loads(result.stdout))
        extracted = ROOT / 'course_validation/解压 verification'
        execute('package-extract', [python, 'scripts/build_course_package.py', '--verify', package, '--extract', extracted])
        source = extracted / 'afqmc-course-source'
        second = make_env('zip-venv', ROOT / 'course_validation/zip-venv')
        install('zip-install', second, source)
        validation = source / 'course_validation/extracted'
        try:
            # Different working directory proves the teacher CLI uses its script root.
            execute('zip-validation', [second, source / 'scripts/verify_course_delivery.py', '--output', validation], cwd=evidence)
        finally:
            if (validation / 'summary.json').is_file():
                shutil.copyfile(validation / 'summary.json', evidence / 'extracted-summary.json')
        if a.prepare:
            validation = source / 'course_validation/official'
            try:
                execute('official-preparation', [second, source / 'scripts/verify_course_delivery.py', '--mode', 'prepare', '--output', validation], cwd=evidence)
            finally:
                if (validation / 'summary.json').is_file():
                    shutil.copyfile(validation / 'summary.json', evidence / 'preparation-summary.json')
            # Only the explicitly dispatched preparation job handles real data.
            # The private ZIP never becomes a GitHub artifact.
            private_zip = ROOT / 'local_delivery/private-course-submission.zip'
            result = execute('private-package-build', [second, source / 'scripts/build_course_submission.py',
                             '--source', package, '--data-root', source, '--output', private_zip])
            write_json_new(evidence / 'submission-summary.json', json.loads(result.stdout))
            private_extract = ROOT / 'course_validation/教师 私有解压'
            execute('private-package-extract', [second, source / 'scripts/build_course_submission.py',
                    '--verify', private_zip, '--extract', private_extract])
            private_source = private_extract / 'afqmc-course-source'
            if (private_source / '.course_runtime').exists():
                raise DeliveryError('Cold launcher test found a pre-existing runtime')
            launcher = ([private_source / 'Run_Mac.command'] if platform.system() == 'Darwin'
                        else ['cmd.exe', '/d', '/c', private_source / 'Run_Windows.cmd'])
            execute('launcher-online', [*launcher, '--no-pause'], cwd=evidence)
            latest = private_source / '.course_runtime/last_run.json'
            first = json.loads(latest.read_text(encoding='utf-8'))
            if not first.get('passed') or first.get('exit_code') != 0 or first.get('dev_evaluation') is not False:
                raise DeliveryError('First launcher result failed')
            write_json_new(evidence / 'launcher-online-summary.json', first)
            offline_environment = os.environ.copy()
            offline_environment.update({'UV_OFFLINE': '1', 'PIP_NO_INDEX': '1',
                'HTTP_PROXY': 'http://127.0.0.1:9', 'HTTPS_PROXY': 'http://127.0.0.1:9',
                'ALL_PROXY': 'http://127.0.0.1:9', 'NO_PROXY': ''})
            execute('launcher-offline', [*launcher, '--offline', '--no-pause'], cwd=evidence, env=offline_environment)
            second_run = json.loads(latest.read_text(encoding='utf-8'))
            if not second_run.get('passed') or second_run.get('exit_code') != 0 or second_run.get('dev_evaluation') is not False:
                raise DeliveryError('Offline launcher result failed')
            if second_run.get('output') == first.get('output'):
                raise DeliveryError('Launcher reused an earlier result directory')
            write_json_new(evidence / 'launcher-offline-summary.json', second_run)
            # A normal second double-click has no --offline flag. It must also
            # work with cached runtime/data when network proxies are unavailable.
            execute('launcher-default-offline', [*launcher, '--no-pause'], cwd=evidence, env=offline_environment)
            default_offline = json.loads(latest.read_text(encoding='utf-8'))
            if not default_offline.get('passed') or not default_offline.get('environment_reused') or default_offline.get('output') in (first.get('output'), second_run.get('output')):
                raise DeliveryError('Normal launcher did not reuse its environment offline with a new output')
            write_json_new(evidence / 'launcher-default-offline-summary.json', default_offline)
            train_output = private_source / 'local_delivery/native-ci-train'
            try:
                execute('private-train-validation', [second, private_source / 'scripts/verify_course_delivery.py',
                        '--mode', 'train', '--output', train_output], cwd=evidence)
            finally:
                if (train_output / 'summary.json').is_file():
                    train_record = json.loads((train_output / 'summary.json').read_text(encoding='utf-8'))
                    # Explicit aggregate-only projection, never copy Train output trees.
                    train = train_record.get('train', {})
                    safe_train = {key: train.get(key) for key in ('passed', 'scope', 'inputs', 'metadata',
                        'methods', 'threshold_comparison', 'zip_members_opened', 'font_filename', 'font_sha256')}
                    write_json_new(evidence / 'train-summary.json', {
                        'passed': train_record.get('passed', False), 'exit_code': train_record.get('exit_code'),
                        'environment': train_record.get('environment'), 'dev_evaluation': False,
                        'train': safe_train, 'frozen_after': train_record.get('frozen_after')})
        record['passed'] = True
    except Exception as error:
        record['error'] = str(error) if isinstance(error, DeliveryError) else type(error).__name__
    record['exit_code'] = 0 if record['passed'] else 2
    write_json_new(evidence / 'summary.json', record)
    print(json.dumps(record, indent=2))
    return record['exit_code']

if __name__ == '__main__':
    raise SystemExit(main())
