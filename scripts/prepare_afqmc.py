"""Download and audit the frozen AFQMC inputs, without model evaluation.

Raw records are read locally to validate schema and rebuild the Stage 1 manifest.
No records are extracted, logged, published, or sent to a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

URL = 'https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip'
ARCHIVE_SHA256 = '5a4cb1556b833010c329fa2ad2207d9e98fc94071b7e474015e9dd7c385db4dc'
MANIFEST_SHA256 = 'e856f4b9fd42be0ecfe530c002048933e31d971a3ba8d070a0c13daed241d907'
STAGE1_SHA256 = '53efd7b02db0ab9466317238b5b917eb528ae0e0affd6536a5053b94466cb318'
MEMBERS = {
    'train.json': (34334, 4392205, '41c0ed462fe3da60c575c4f81f4799ba8fbe51155f962adcfc19c26b2de3f012'),
    'dev.json': (4316, 551449, '20b35a40e1aec2141e50247f798b2eef0e974accb9ca7196105ecb547c65c7b8'),
    'test.json': (3861, 481865, '9d2a78d43215a0d4d34230f4aaca1d2cdbad03e1f268bef9da0468e0dc9eb4e5'),
}
FIELDNAMES = ['source_file', 'source_row_id', 'official_split', 'internal_split', 'label', 'pair_group_id']
DEPENDENCIES = {'numpy': '2.5.3', 'scipy': '1.17.0', 'scikit-learn': '1.9.1'}
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024


class PreparationError(Exception):
    """A sanitized, actionable data-preparation error."""


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_records(name: str, data: bytes, expected_rows: int) -> dict:
    try:
        lines = data.decode('utf-8-sig').splitlines()
    except UnicodeDecodeError:
        raise PreparationError(f'{name}: invalid UTF-8; no record contents logged') from None
    if len(lines) != expected_rows:
        raise PreparationError(f'{name}: row count {len(lines)} != {expected_rows}')
    labels = Counter()
    fields = {'sentence1', 'sentence2'} | ({'id'} if name == 'test.json' else {'label'})
    test_ids = set()
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            raise PreparationError(f'{name}: invalid JSON at physical row {number}') from None
        if not isinstance(record, dict) or set(record) != fields:
            raise PreparationError(f'{name}: unexpected fields at physical row {number}')
        if any(not isinstance(record[k], str) for k in ('sentence1', 'sentence2')):
            raise PreparationError(f'{name}: non-string sentence at physical row {number}')
        if name == 'test.json':
            if type(record['id']) is not int or record['id'] in test_ids:
                raise PreparationError(f'{name}: invalid or duplicate test id at physical row {number}')
            test_ids.add(record['id'])
        else:
            value = record['label']
            if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) not in ('0', '1'):
                raise PreparationError(f'{name}: invalid label at physical row {number}')
            labels[str(value)] += 1
    return {'rows': len(lines), 'fields': sorted(fields), 'label_counts': dict(sorted(labels.items()))}


def validate_archive(path: Path) -> dict:
    if path.is_symlink():
        raise PreparationError('Refusing a symlink archive path')
    if sha256_file(path) != ARCHIVE_SHA256:
        raise PreparationError('Archive SHA-256 mismatch; existing file preserved. Obtain the exact official version.')
    result = {}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != len(MEMBERS) or {i.filename for i in infos} != set(MEMBERS):
                raise PreparationError('ZIP members differ from the exact three registered members')
            for info in infos:
                rows, size, digest = MEMBERS[info.filename]
                if info.flag_bits & 1 or info.is_dir() or info.file_size != size:
                    raise PreparationError('ZIP encryption, member type, or member size differs')
                # Reading to EOF checks CRC; nothing is extracted to the filesystem.
                data = archive.read(info)
                if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                    raise PreparationError(f'{info.filename}: member SHA-256/size mismatch')
                result[info.filename] = {**validate_records(info.filename, data, rows),
                                         'bytes': size, 'sha256': digest, 'crc': 'PASS'}
    except (zipfile.BadZipFile, RuntimeError, EOFError):
        raise PreparationError('ZIP format/CRC validation failed; no record contents logged') from None
    return result


def publish_without_overwrite(temporary: Path, destination: Path) -> None:
    """Atomically link verified bytes on the same filesystem; never replace a file."""
    if destination.is_symlink():
        raise PreparationError('Refusing a symlink destination')
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if sha256_file(destination) != sha256_file(temporary):
            raise PreparationError('Destination appeared with different bytes; preserved without overwrite') from None
    except OSError:
        raise PreparationError('Cannot atomically publish verified file on this filesystem; no fallback overwrite attempted') from None


def obtain_archive(path: Path, *, offline: bool, timeout: float) -> tuple[dict, str]:
    if path.exists() or path.is_symlink():
        return validate_archive(path), 'existing_verified'
    if offline:
        raise PreparationError('OFFLINE: official archive is missing')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.afqmc-', suffix='.part', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        try:
            with os.fdopen(fd, 'wb') as stream, urllib.request.urlopen(URL, timeout=timeout) as response:
                if response.geturl() != URL:
                    raise PreparationError('Unexpected download redirect; no alternative data version accepted')
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise PreparationError('Download exceeds the registered safety size limit')
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        except (urllib.error.URLError, OSError, TimeoutError):
            raise PreparationError('Official download failed (network, timeout, or local write error); temporary data discarded') from None
        result = validate_archive(temporary)
        publish_without_overwrite(temporary, path)
        return result, 'downloaded_and_verified'
    finally:
        temporary.unlink(missing_ok=True)


def check_environment() -> dict:
    versions = {'python': platform.python_version()}
    for name, version in DEPENDENCIES.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise PreparationError(f'Manifest dependency missing: {name}. Install the AFQMC requirements; use --archive-only for standard-library validation.') from None
        if versions[name] != version:
            raise PreparationError(f'Manifest dependency version mismatch: {name}; expected {version}. Use the pinned AFQMC environment.')
    if versions['python'] != '3.12.10':
        raise PreparationError('Manifest rebuild requires the recorded Python 3.12.10 environment; --archive-only remains available')
    return versions


def load_stage1(root: Path):
    path = root / 'stage01_data_audit/audit_afqmc.py'
    if not path.is_file() or sha256_file(path) != STAGE1_SHA256:
        raise PreparationError('Historical Stage 1 source missing or fingerprint mismatch; rebuild blocked')
    spec = importlib.util.spec_from_file_location('afqmc_original_stage1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_rows(stage1, split_rows: dict) -> tuple[list, dict]:
    assignments, info = stage1.build_internal_split(split_rows['train'])
    rows = []
    for split in stage1.SPLITS:
        for row in split_rows[split]:
            internal = assignments.get(row['source_row_id'], 'excluded_invalid') if split == 'train' else 'official_' + split
            rows.append({'source_file': row['source_file'], 'source_row_id': row['source_row_id'],
                         'official_split': split, 'internal_split': internal,
                         'label': row['label'] or '', 'pair_group_id': row['pair_group_id']})
    if len({r['source_row_id'] for r in rows}) != len(rows):
        raise PreparationError('Duplicate source row identifiers; manifest blocked')
    return rows, info


def rebuild_manifest(root: Path, archive: Path, destination: Path) -> dict:
    # Reject an existing wrong manifest before reading records or writing anything.
    if destination.is_symlink() or (destination.exists() and sha256_file(destination) != MANIFEST_SHA256):
        raise PreparationError('Existing Manifest differs from frozen SHA-256; preserved without overwrite')
    versions = check_environment()
    stage1 = load_stage1(root)
    split_rows, _, issues = stage1.load_archive(archive)
    if issues:
        raise PreparationError('Stage 1 parser found invalid rows; no record contents logged')
    rows, split_info = manifest_rows(stage1, split_rows)
    if {s: len(split_rows[s]) for s in stage1.SPLITS} != stage1.EXPECTED:
        raise PreparationError('Stage 1 row coverage mismatch')
    # Temp output stays in the ignored raw directory; the original writer fixes
    # UTF-8 BOM, column order and CSV CRLF independently of host line endings.
    with tempfile.TemporaryDirectory(prefix='.manifest-', dir=archive.parent) as temporary:
        candidate = Path(temporary) / 'split_manifest.csv'
        stage1.write_csv(candidate, rows, FIELDNAMES)
        digest = sha256_file(candidate)
        if digest != MANIFEST_SHA256:
            raise PreparationError(f'Rebuilt Manifest SHA-256 mismatch ({digest}); historical inputs unchanged')
        existed = destination.exists()
        publish_without_overwrite(candidate, destination)
    return {'sha256': digest, 'rows': len(rows), 'internal_split_rows': split_info['rows'],
            'seed': stage1.SEED, 'repeated_split_generation_identical': split_info['repeated_generation_identical'],
            'existing_manifest_preserved': existed, 'versions': versions,
            'serialization': 'Stage 1 write_csv; UTF-8 BOM; fixed columns; CRLF',
            'status': 'BYTE_IDENTICAL_TO_FROZEN_MANIFEST'}


def prepare(root: Path, *, offline=False, archive_only=False, timeout=30.0) -> dict:
    root = root.resolve()
    base = root / 'stage01_data_audit'
    archive = base / 'raw/afqmc_public.zip'
    destination = base / 'split_manifest.csv'
    # Prevent symlink parents from sending private outputs outside the target.
    if any(p.is_symlink() for p in (base, archive.parent)):
        raise PreparationError('Refusing symlink data directories')
    members, acquisition = obtain_archive(archive, offline=offline, timeout=timeout)
    result = {'status': 'ARCHIVE_VALIDATED' if archive_only else 'DATA_PREPARATION_PASS',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_url': URL,
              'archive_path': 'stage01_data_audit/raw/afqmc_public.zip', 'archive_sha256': ARCHIVE_SHA256,
              'acquisition': acquisition, 'members': members,
              'manifest': None, 'model_fit': False, 'model_predictions': False,
              'threshold_search': False, 'dev_performance_evaluation': False,
              'raw_data_redistributed': False}
    if not archive_only:
        result['manifest'] = rebuild_manifest(root, archive, destination)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Clone root; defaults to script repository, not working directory')
    parser.add_argument('--offline', action='store_true', help='Only check existing official archive; never access network')
    parser.add_argument('--archive-only', action='store_true', help='Validate download/ZIP/schema without scientific dependencies or Manifest rebuild')
    parser.add_argument('--timeout', type=float, default=30.0, help='Positive per-network-operation timeout in seconds')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 300:
        parser.error('--timeout must be between 0 and 300 seconds')
    try:
        result = prepare(args.root, offline=args.offline, archive_only=args.archive_only, timeout=args.timeout)
    except Exception as error:
        message = str(error) if isinstance(error, PreparationError) else f'Preparation failed ({type(error).__name__}); no record contents logged. Check environment and filesystem permissions'
        print(f'DATA_PREPARATION_BLOCKED: {message}', file=sys.stderr)
        print(f'Manual download: {URL}\nSave unchanged as: {args.root.resolve() / "stage01_data_audit/raw/afqmc_public.zip"}\nExpected ZIP SHA-256: {ARCHIVE_SHA256}\nDo not change hashes, labels, groups, or thresholds. No Dev evaluation is performed.', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
