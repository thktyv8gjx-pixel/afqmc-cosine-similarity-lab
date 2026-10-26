"""Shared, non-frozen course-delivery checks. No data/model work at import."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc'
MANIFEST_SHA = 'e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907'
V2_CANONICAL_SHA = 'dc8798a981ea15689a84005f256e292ddecb5397f56375394650b3699e21b6d2'
VERSIONS = {'numpy': '2.5.3', 'scipy': '1.17.0', 'scikit-learn': '1.9.1',
            'pillow': '12.3.0', 'joblib': '1.6.0', 'threadpoolctl': '3.7.0',
            'packaging': '26.3', 'cloudpickle': '3.1.2', 'narwhals': '2.27.1'}

class DeliveryError(Exception):
    """Sanitized error suitable for a course summary."""


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json_new(path: Path, value) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def safe_name(name: str) -> str:
    """Portable relative file name, including Windows device/ADS restrictions."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or path.as_posix() != name or '\\' in name:
        raise DeliveryError('Unsafe relative path')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    for part in path.parts:
        if part in ('.', '..') or part.endswith((' ', '.')) or any(ord(c) < 32 or c in '<>:"|?*' for c in part):
            raise DeliveryError('Unsafe relative path component')
        if part.split('.')[0].upper() in reserved:
            raise DeliveryError('Windows reserved path')
    return name


def check_names(names: list[str]) -> None:
    seen = set()
    for name in names:
        safe_name(name)
        key = unicodedata.normalize('NFC', name).casefold()
        if key in seen:
            raise DeliveryError('Case/Unicode path collision')
        seen.add(key)
    for key in seen:
        if any(parent.as_posix() in seen for parent in PurePosixPath(key).parents if parent.as_posix() != '.'):
            raise DeliveryError('File/directory path collision')


def no_symlink(path: Path) -> None:
    for p in (path, *path.parents):
        if not p.is_symlink():
            continue
        # Apple supplies these root aliases; tempfile legitimately uses /var.
        if platform.system() == 'Darwin' and p in (Path('/var'), Path('/tmp')) and p.resolve() == Path('/private') / p.name:
            continue
        raise DeliveryError('Symlink paths are not accepted')


def checked_source(root: Path, name: str) -> Path:
    safe_name(name)
    path = root / name
    # Resolve the trusted root once so /var -> /private/var is not mistaken for
    # a user-supplied file symlink. All descendants still must be regular files.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise DeliveryError('Symlink source is not accepted')
    if not path.is_file():
        raise DeliveryError(f'Required file missing: {name}')
    return path


def new_output(path: Path) -> Path:
    path = path.absolute()
    # Only inspect supplied descendants of the nearest existing real parent;
    # macOS system /var alias is legitimate, but a user-created symlink is not.
    no_symlink(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
    except OSError as error:
        raise DeliveryError('Output must be a new writable directory; existing output preserved') from error
    return path.resolve()


def lock_path(system=None, machine=None) -> str:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == 'Darwin' and machine == 'arm64':
        return 'delivery/requirements-cp312-macos-arm64.lock.txt'
    if system == 'Windows' and machine in ('amd64', 'x86_64'):
        return 'stage05_p1_closure_review/requirements-cp312-win_amd64.lock.txt'
    raise DeliveryError('Unsupported native architecture; require macOS arm64 or Windows x64')


def environment() -> dict:
    lock = lock_path()
    if platform.python_version() != '3.12.10':
        raise DeliveryError('Use native Python 3.12.10 in a new virtual environment')
    if sys.prefix == sys.base_prefix:
        raise DeliveryError('Use an isolated virtual environment (python -m venv .venv)')
    versions = {}
    for name, expected in VERSIONS.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise DeliveryError(f'Missing {name}; install --require-hashes -r {lock}') from None
        if versions[name] != expected:
            raise DeliveryError(f'{name} must be {expected}; install the exact platform lock: {lock}')
    # Loading compiled extensions is a separate gate from installed metadata.
    try:
        import numpy, scipy, sklearn
        from PIL import Image
        from scipy import sparse
        sparse.csr_matrix(numpy.eye(2)).multiply(numpy.eye(2))
        Image.new('RGB', (1, 1)).getpixel((0, 0))
    except Exception as error:
        raise DeliveryError(f'Native dependency import failed ({type(error).__name__})') from None
    return {'system': platform.system(), 'platform': platform.platform(), 'machine': platform.machine(),
            'python': platform.python_version(), 'isolated_venv': True, 'versions': versions, 'lock': lock,
            'native_imports': True, 'pip': importlib.metadata.version('pip')}


def verify_frozen(root: Path) -> dict:
    root = root.resolve()
    baseline = read_json(checked_source(root, 'delivery/frozen_baseline.json'))
    check_names(list(baseline['files']))
    for name, expected in baseline['files'].items():
        if sha(checked_source(root, name)) != expected:
            raise DeliveryError(f'Frozen fingerprint mismatch: {name}')
    f = read_json(root / 'stage03b_official_validation/pre_dev_freeze_manifest.v2.json')
    claimed = f.pop('freeze_manifest_sha256')
    canonical = (json.dumps(f, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if claimed != V2_CANONICAL_SHA or hashlib.sha256(canonical).hexdigest() != V2_CANONICAL_SHA:
        raise DeliveryError('Frozen canonical fingerprint mismatch')
    protected = f['frozen_source_files_sha256']
    if len(protected) != 34 or any(baseline['files'].get(n) != h for n, h in protected.items()):
        raise DeliveryError('Frozen inventory mismatch')
    inputs = f['inputs']; space = f['fit_space']
    expected_inputs = {'archive_sha256': ARCHIVE_SHA, 'manifest_sha256': MANIFEST_SHA,
                       'train_rows': 34334, 'dev_rows': 4316, 'test_rows': 3861,
                       'fit_tune_rows': {'fit': 27467, 'tune': 6867}}
    if any(inputs[k] != v for k, v in expected_inputs.items()) or space['unique_normalized_fit_question_count'] != 54330 or space['vocabulary_dimension'] != 35242:
        raise DeliveryError('Frozen data/feature-space mismatch')
    metrics = read_json(root / 'stage03b_official_validation/results/dev_metrics.json')
    all_metrics = {**metrics['metrics'], **metrics['constant_baselines']}
    for method, threshold in [('count_char2', 0.10660035817780521), ('tfidf_char2', 0.05574242073415668)]:
        if f['thresholds'][method]['value'] != threshold or all_metrics[method]['threshold'] != threshold:
            raise DeliveryError('Frozen threshold mismatch')
    for m in all_metrics.values():
        tp, fp, fn, tn = (m[k] for k in ('tp', 'fp', 'fn', 'tn'))
        if min(tp, fp, fn, tn) < 0 or sum((tp, fp, fn, tn)) != 4316:
            raise DeliveryError('Saved confusion counts mismatch')
        denom = 2 * tp + fp + fn
        if not math.isclose(m['f1'], 2 * tp / denom if denom else 0., rel_tol=0, abs_tol=2e-15):
            raise DeliveryError('Saved F1 does not match confusion counts')
        if not math.isclose(m['false_positive_rate'], fp / (fp + tn), rel_tol=0, abs_tol=2e-15):
            raise DeliveryError('Saved FPR does not match confusion counts')
    main = all_metrics['tfidf_char2']
    confusion = {k: main[k] for k in ('tp', 'fp', 'fn', 'tn')}
    if confusion != dict(tp=1186, fp=2425, fn=152, tn=553):
        raise DeliveryError('Official TF-IDF confusion mismatch')
    return {'passed': True, 'frozen_source_count': 34, 'protected_files': len(baseline['files']),
            'baseline_commit': baseline['baseline_commit'], 'v2_canonical_sha256': V2_CANONICAL_SHA,
            'tfidf_confusion': confusion, 'dev_replayed': False,
            'scope': 'Raw byte fingerprints and saved aggregate arithmetic; not row-level prediction replay'}


def check_train_inputs(root: Path) -> dict:
    required = {'stage01_data_audit/raw/afqmc_public.zip': ARCHIVE_SHA,
                'stage01_data_audit/split_manifest.csv': MANIFEST_SHA}
    issues = []
    for name, expected in required.items():
        try:
            if sha(checked_source(root.resolve(), name)) != expected:
                issues.append(f'{name}: SHA-256 mismatch; expected {expected}')
        except DeliveryError:
            issues.append(f'{name}: missing or unsafe; expected SHA-256 {expected}')
    if issues:
        raise DeliveryError('; '.join(issues) + '. Obtain from https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip then use --mode prepare; existing files are never replaced.')
    return {'archive_sha256': ARCHIVE_SHA, 'manifest_sha256': MANIFEST_SHA}


def resolve_font(explicit: Path | None = None) -> Path:
    from PIL import ImageFont
    candidates = [explicit] if explicit else [
        Path('/System/Library/Fonts/PingFang.ttc'),
        Path('/System/Library/Fonts/STHeiti Medium.ttc'),
        Path('/System/Library/Fonts/Supplemental/Songti.ttc'),
        Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf')]
    for path in candidates:
        if path and path.is_file():
            try:
                font = ImageFont.truetype(str(path), 24, index=0)
                missing = bytes(font.getmask('\u0378'))
                if all(bytes(font.getmask(c)) not in (b'', missing) for c in '中文相似问句'):
                    return path.resolve()
            except (OSError, ValueError):
                pass
    raise DeliveryError('Chinese font unavailable; supply --font pointing to a legally installed CJK TTF/TTC. No font is bundled.')


def run_logged(name: str, arguments: list[str], root: Path, output: Path) -> dict:
    command = [sys.executable, '-s', '-E', '-X', 'utf8', *arguments]
    proc = subprocess.run(command, cwd=root, capture_output=True, encoding='utf-8', errors='replace')
    log = output / f'{name}.log'
    with log.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(proc.stdout + proc.stderr)
    record = {'name': name, 'exit_code': proc.returncode, 'log': log.name, 'log_sha256': sha(log)}
    match = re.search(r'Ran (\d+) tests? in ', proc.stderr)
    if match:
        record['tests_run'] = int(match[1])
    if name == 'synthetic' and proc.returncode == 0:
        record['result'] = json.loads(proc.stdout)
    return record
