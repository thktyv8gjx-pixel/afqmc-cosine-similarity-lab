"""Run the existing 49 tests and toy cosine demo in a data-free snapshot.

Only frozen source files, v1/v2 fixtures and three synthetic helpers are copied.
Raw AFQMC, .git, private reports, and unrelated coursework are never copied.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def checked_result(output: Path, returncode: int, name: str) -> dict:
    if returncode or not output.is_file():
        raise RuntimeError(f'{name} failed with exit {returncode} or missing result file')
    result = json.loads(output.read_text(encoding='utf-8'))
    if not isinstance(result, dict) or result.get('passed') is not True:
        raise RuntimeError(f'{name} did not report an explicit successful result')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    private = root / 'stage05_p1_closure_review/local_only'
    record = {'passed': False, 'raw_afqmc_read': False, 'dev_evaluation': False}
    try:
        private.mkdir(parents=True, exist_ok=True)
        freeze_path = 'stage03b_official_validation/pre_dev_freeze_manifest.v2.json'
        freeze = json.loads((root / freeze_path).read_text(encoding='utf-8'))
        claimed = freeze.pop('freeze_manifest_sha256')
        canonical = (json.dumps(freeze, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        if hashlib.sha256(canonical).hexdigest() != claimed:
            raise RuntimeError('v2 canonical fingerprint mismatch')
        v1_path = 'stage03b_official_validation/pre_dev_freeze_manifest.json'
        v1 = json.loads((root / v1_path).read_text(encoding='utf-8'))
        v1.pop('freeze_manifest_sha256')
        v1_canonical = (json.dumps(v1, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        if hashlib.sha256(v1_canonical).hexdigest() != freeze['recovery']['v1_manifest_canonical_sha256']:
            raise RuntimeError('Historical v1 fixture canonical fingerprint mismatch')
        inventory = json.loads((root / 'stage05_p1_closure_review/candidate_inventory.v2.json').read_text(encoding='utf-8'))
        helper_paths = ['stage05_p1_closure_review/run_synthetic_validation.py',
                        'stage05_p1_closure_review/demo_cosine.py',
                        'stage05_p1_closure_review/synthetic_example.json']
        hashes = dict(freeze['frozen_source_files_sha256'])
        hashes.update({item['path']: item['sha256'] for item in inventory['files'] if item['path'] in helper_paths})
        if not all(p in hashes for p in helper_paths):
            raise RuntimeError('Required synthetic helper missing from safe inventory')
        with tempfile.TemporaryDirectory(prefix='datafree-', dir=private) as work:
            work = Path(work)
            snapshot = work / 'snapshot'
            temp = work / 'synthetic_tmp'
            temp.mkdir()
            for name in [*hashes, freeze_path, v1_path]:
                relative = Path(name)
                source = root / relative
                if relative.is_absolute() or '..' in relative.parts or source.is_symlink():
                    raise RuntimeError('Invalid safe inventory path')
                if name in hashes and hashlib.sha256(source.read_bytes()).hexdigest() != hashes[name]:
                    raise RuntimeError(f'Safe candidate fingerprint mismatch: {relative.as_posix()}')
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            # tempfile prefers TMPDIR over TEMP/TMP on macOS. Keep all three
            # inside the guarded snapshot; do not weaken the frozen ZIP guard.
            env = {**os.environ, 'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp), 'PYTHONIOENCODING': 'utf-8'}
            runs = []
            for script, name in [('run_synthetic_validation.py', 'tests'), ('demo_cosine.py', 'demo')]:
                output = work / f'{name}.json'
                command = [sys.executable, '-s', '-E', '-X', 'utf8', str(snapshot / 'stage05_p1_closure_review' / script), '--root', str(snapshot), '--output', str(output)]
                process = subprocess.run(command, cwd=snapshot, env=env, capture_output=True, text=True, encoding='utf-8')
                if process.returncode:
                    # Only synthetic test messages; no external data is in snapshot.
                    print(process.stderr[-5000:], file=sys.stderr)
                result = checked_result(output, process.returncode, name)
                runs.append({'name': name, 'exit_code': process.returncode, 'result': result})
            record.update(passed=True, copied_files=len(hashes)+2, runs=runs,
                          json_fixture_check='registered v1/v2 canonical digests; original bytes copied unchanged',
                          python=sys.version.split()[0], isolated_environment=sys.prefix != sys.base_prefix,
                          checked_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        print(f'DATA_FREE_CHECK_FAILED: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
