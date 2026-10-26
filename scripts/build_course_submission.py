"""Private course submission ZIP: reviewed source plus two fixed data files.

Never upload the output to a public repository or CI artifacts. Public source
packaging remains data-free. No directory globbing and no model execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile

from build_course_package import PREFIX, MANIFEST, MAX_BYTES, json_bytes, verify_package
from course_delivery import (ROOT, ARCHIVE_SHA, MANIFEST_SHA, DeliveryError,
                             check_names, checked_source, new_output, no_symlink, sha)

PRIVATE_MANIFEST = 'SUBMISSION_MANIFEST.json'
DATA_FILES = {'stage01_data_audit/raw/afqmc_public.zip': ARCHIVE_SHA,
              'stage01_data_audit/split_manifest.csv': MANIFEST_SHA}


def _new_info(name):
    info = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 10, 26, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def build_submission(source_zip: Path, data_root: Path, destination: Path) -> dict:
    source = verify_package(source_zip)
    data = {name: checked_source(data_root.resolve(), name).read_bytes() for name in DATA_FILES}
    if any(hashlib.sha256(data[name]).hexdigest() != expected for name, expected in DATA_FILES.items()):
        raise DeliveryError('Private data SHA-256 differs from the frozen official version')
    manifest = {'schema': 1, 'private_course_submission': True,
                'source_commit': source['source_commit'], 'source_sha256': source['sha256'],
                'data_files': [{'path': name, 'bytes': len(data[name]), 'sha256': DATA_FILES[name]} for name in sorted(data)],
                'scope': 'Private course submission only; source plus original AFQMC archive and exact full Manifest; no personal reports or per-row predictions'}
    no_symlink(destination.absolute())
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = destination.open('xb')
    except OSError as error:
        raise DeliveryError('Submission destination must be new and writable; existing file preserved') from error
    with stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as output, zipfile.ZipFile(source_zip) as source_archive:
        for info in source_archive.infolist():
            output.writestr(info, source_archive.read(info))
        for name in sorted(data):
            output.writestr(_new_info(name), data[name])
        output.writestr(_new_info(PRIVATE_MANIFEST), json_bytes(manifest))
    return verify_submission(destination)


def verify_submission(path: Path, extract: Path | None = None) -> dict:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            check_names(names)
            if sum(i.file_size for i in infos) > MAX_BYTES or any(i.file_size > MAX_BYTES for i in infos):
                raise DeliveryError('Submission exceeds size limit')
            if any(not i.filename.startswith(PREFIX) or i.is_dir() or i.flag_bits & 1 or
                   stat.S_IFMT(i.external_attr >> 16) not in (0, stat.S_IFREG) for i in infos):
                raise DeliveryError('Unsafe submission member')
            manifest = json.loads(archive.read(PREFIX + PRIVATE_MANIFEST))
            if manifest.get('schema') != 1 or manifest.get('private_course_submission') is not True:
                raise DeliveryError('Invalid private submission manifest')
            data_entries = manifest['data_files']
            if len(data_entries) != len(DATA_FILES) or {x['path']: x['sha256'] for x in data_entries} != DATA_FILES:
                raise DeliveryError('Private data inventory is not the exact frozen pair')
            source_manifest = json.loads(archive.read(PREFIX + MANIFEST))
            source_names = [entry['path'] for entry in source_manifest['files']] + [MANIFEST]
            if set(names) != {PREFIX + n for n in [*source_names, *DATA_FILES, PRIVATE_MANIFEST]}:
                raise DeliveryError('Submission members differ from explicit manifests')
            content = {i.filename[len(PREFIX):]: archive.read(i) for i in infos}  # CRC on every member
            for entry in data_entries:
                payload = content[entry['path']]
                if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != DATA_FILES[entry['path']]:
                    raise DeliveryError('Private data fingerprint mismatch')
            # Reconstitute the original public package, so its strict privacy,
            # source hash and safety checks are reused without any exemption.
            with tempfile.TemporaryDirectory(prefix='afqmc-source-verification-') as temporary:
                source_path = Path(temporary) / 'source.zip'
                with zipfile.ZipFile(source_path, 'w', compression=zipfile.ZIP_STORED) as source_archive:
                    for info in infos:
                        name = info.filename[len(PREFIX):]
                        if name in source_names:
                            source_archive.writestr(info, content[name])
                if sha(source_path) != manifest['source_sha256']:
                    raise DeliveryError('Submission source package fingerprint mismatch')
                source = verify_package(source_path)
                if source['source_commit'] != manifest['source_commit']:
                    raise DeliveryError('Submission source commit mismatch')
    except (KeyError, ValueError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        raise DeliveryError('Invalid or corrupted private submission') from error
    if extract is not None:
        target = new_output(extract)
        for name, payload in content.items():
            output = target / PREFIX / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                stream.write(payload)
            if name.endswith('.command') and os.name != 'nt':
                output.chmod(0o755)
            if sha(output) != hashlib.sha256(payload).hexdigest():
                raise DeliveryError('Extracted submission fingerprint mismatch')
    return {'passed': True, 'private_course_submission': True, 'source_commit': source['source_commit'],
            'source_files': source['files'], 'source_sha256': source['sha256'],
            'data_files': len(DATA_FILES), 'data_sha256': DATA_FILES,
            'bytes': path.stat().st_size, 'sha256': sha(path),
            'all_member_hashes_and_crc_verified': True, 'extracted': extract is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--extract', type=Path)
    args = parser.parse_args(argv)
    if bool(args.verify) == bool(args.source) or (args.source and not args.output) or (args.extract and not args.verify):
        parser.error('Use --source SOURCE --output NEW_PRIVATE_ZIP, or --verify ZIP [--extract NEW_DIRECTORY]')
    try:
        result = verify_submission(args.verify, args.extract) if args.verify else build_submission(args.source, args.data_root, args.output)
    except (DeliveryError, OSError) as error:
        print(json.dumps({'passed': False, 'error': str(error) if isinstance(error, DeliveryError) else type(error).__name__}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
