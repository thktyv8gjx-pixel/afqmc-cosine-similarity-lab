"""Native CI driver with exact exit codes and explicitly selected safe artifacts.

Runs installation, worktree validation and a new ZIP extraction/new virtualenv.
Only --prepare performs official data preparation, after the data-free checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

from course_delivery import ROOT, DeliveryError, lock_path, new_output, sha, write_json_new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-system', required=True, choices=['Darwin', 'Windows'])
    p.add_argument('--expected-machine', required=True, choices=['arm64', 'amd64'])
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args(argv)
    evidence = new_output(ROOT / 'course_validation/ci')
    record = {'passed': False, 'system': platform.system(), 'machine': platform.machine(),
              'python': platform.python_version(), 'prepare_requested': a.prepare, 'steps': []}

    def execute(name, args, cwd=ROOT):
        proc = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True,
                              text=True, encoding='utf-8', errors='replace')
        log = evidence / (name + '.log')
        log.write_text(proc.stdout + proc.stderr, encoding='utf-8', newline='\n')
        record['steps'].append({'name': name, 'exit_code': proc.returncode, 'log_sha256': sha(log)})
        print(f'{name}: exit {proc.returncode}', flush=True)
        if proc.returncode:
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
        record['passed'] = True
    except Exception as error:
        record['error'] = str(error) if isinstance(error, DeliveryError) else type(error).__name__
    record['exit_code'] = 0 if record['passed'] else 2
    write_json_new(evidence / 'summary.json', record)
    print(json.dumps(record, indent=2))
    return record['exit_code']

if __name__ == '__main__':
    raise SystemExit(main())
