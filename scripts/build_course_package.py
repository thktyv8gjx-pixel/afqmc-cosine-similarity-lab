"""Build a reviewed current-HEAD source ZIP, or verify/extract it without Git.

No globbing of the working directory and no historical 64-file inventory reuse.
The explicit delivery/package_allowlist.json must be reviewed and committed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile

from course_delivery import (ROOT, DeliveryError, check_names, checked_source, new_output,
                             no_symlink, read_json, sha, verify_frozen)

PREFIX = 'afqmc-course-source/'
MANIFEST = 'PACKAGE_MANIFEST.json'
MAX_BYTES = 128 * 1024 * 1024
DENIED_PARTS = {'.git', '.venv', 'venv', '__pycache__', 'local_only', 'local_delivery',
                'raw', 'outputs', 'course_validation', '.aws', '.codex', '.agents'}
DENIED_SUFFIXES = {'.zip', '.docx', '.pdf', '.whl', '.pyc', '.log', '.pem', '.key'}


def validate_package_names(names: list[str]) -> None:
    check_names(names)
    for name in names:
        path = Path(name)
        if any(p in DENIED_PARTS or p.startswith('.env') for p in path.parts) or path.suffix.lower() in DENIED_SUFFIXES or path.name in ('split_manifest.csv', MANIFEST):
            raise DeliveryError('Private/generated file is forbidden in source package')


def scan_text(name: str, data: bytes) -> None:
    if Path(name).suffix.lower() == '.png':
        return
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise DeliveryError(f'Unexpected binary file: {name}') from None
    # Stop obvious credentials/private host paths; human review still required.
    patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                '/' + r'Users/[^/\s]+/', r'[A-Z]:' + r'\\Users\\[^\\\s]+\\']
    if any(re.search(pattern, text) for pattern in patterns):
        raise DeliveryError(f'Potential credential/private host path in {name}; inspect privately')


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def write_package(root: Path, names: list[str], destination: Path, head: str) -> dict:
    root = root.resolve()
    validate_package_names(names)
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise DeliveryError('Expected a full source commit SHA')
    files = []
    payloads = []
    for name in sorted(names):
        data = checked_source(root, name).read_bytes()
        scan_text(name, data)
        files.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        payloads.append((name, data))
    if sum(len(data) for _, data in payloads) > MAX_BYTES:
        raise DeliveryError('Source package exceeds size limit')
    manifest = {'schema': 1, 'source_commit': head, 'files': files,
                'scope': 'source, tests, synthetic examples and aggregate evidence; external AFQMC/report prerequisites excluded'}
    payloads.append((MANIFEST, json_bytes(manifest)))
    no_symlink(destination.absolute())
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = destination.open('xb')
    except OSError as error:
        raise DeliveryError('ZIP destination must be new and writable; existing file preserved') from error
    # ZIP_STORED avoids cross-platform zlib differences. Fixed headers/order produce
    # byte-identical ZIPs on Mac and Windows from the same clean source commit.
    with stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in payloads:
            info = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 10, 26, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    result = verify_package(destination)
    return result


def verify_package(path: Path, extract: Path | None = None) -> dict:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            all_names = [i.filename for i in infos]
            check_names(all_names)
            if sum(i.file_size for i in infos) > MAX_BYTES or any(i.file_size > MAX_BYTES for i in infos):
                raise DeliveryError('Uncompressed package exceeds size limit')
            if any(not i.filename.startswith(PREFIX) or i.is_dir() or i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16) for i in infos):
                raise DeliveryError('Unsafe ZIP member')
            manifest = json.loads(archive.read(PREFIX + MANIFEST))
            if manifest.get('schema') != 1 or not re.fullmatch('[0-9a-f]{40}', manifest.get('source_commit', '')):
                raise DeliveryError('Invalid package manifest')
            entries = manifest['files']
            names = [e['path'] for e in entries]
            validate_package_names(names)
            if set(all_names) != {PREFIX + n for n in [*names, MANIFEST]}:
                raise DeliveryError('ZIP entries differ from manifest allowlist')
            content = {}
            for entry in entries:
                data = archive.read(PREFIX + entry['path'])  # CRC checked by zipfile
                if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise DeliveryError('ZIP member fingerprint mismatch')
                scan_text(entry['path'], data)
                content[entry['path']] = data
            content[MANIFEST] = archive.read(PREFIX + MANIFEST)
    except (KeyError, ValueError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        raise DeliveryError('Invalid or corrupted source package') from error
    if extract is not None:
        target = new_output(extract)
        for name, data in content.items():
            output = target / PREFIX / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                stream.write(data)
            if sha(output) != hashlib.sha256(data).hexdigest():
                raise DeliveryError('Extracted file fingerprint mismatch')
    return {'passed': True, 'source_commit': manifest['source_commit'], 'files': len(entries),
            'bytes': path.stat().st_size, 'sha256': sha(path),
            'all_member_hashes_and_crc_verified': True, 'extracted': extract is not None}


def build(root: Path, destination: Path) -> dict:
    root = root.resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        raise DeliveryError('Commit all reviewed changes before packaging current HEAD; keep outputs in ignored directories')
    head = git('rev-parse', 'HEAD').decode().strip()
    names = read_json(root / 'delivery/package_allowlist.json')['files']
    validate_package_names(names)
    tracked = set(git('ls-files', '-z').decode('utf-8').rstrip('\0').split('\0'))
    if not set(names).issubset(tracked):
        raise DeliveryError('Package allowlist contains untracked files')
    verify_frozen(root)
    for name in names:
        if checked_source(root, name).read_bytes() != git('show', f'{head}:{name}'):
            raise DeliveryError(f'Working bytes differ from HEAD (including line endings): {name}')
    return write_package(root, names, destination, head)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, help='New ZIP path; default dist/afqmc-course-source.zip')
    p.add_argument('--verify', type=Path, help='Verify an existing ZIP without building')
    p.add_argument('--extract', type=Path, help='New destination; only with --verify')
    a = p.parse_args(argv)
    if a.extract and not a.verify:
        p.error('--extract requires --verify')
    try:
        result = verify_package(a.verify, a.extract) if a.verify else build(a.root, a.output or a.root / 'dist/afqmc-course-source.zip')
    except (DeliveryError, OSError, subprocess.CalledProcessError) as error:
        message = str(error) if isinstance(error, DeliveryError) else type(error).__name__
        print(json.dumps({'passed': False, 'error': message, 'exit_code': 2})); return 2
    print(json.dumps({**result, 'exit_code': 0}, indent=2)); return 0

if __name__ == '__main__':
    raise SystemExit(main())
