"""Private, additive course launcher. Never trains or replays official Dev.

Native entries verify uv before invoking this file with isolated CPython 3.12.10.
The private runtime is independent of an existing project .venv and user config.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

# -I deliberately ignores PYTHONPATH and the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_delivery import (ROOT, DeliveryError, checked_source, environment, lock_path,
                             new_output, no_symlink, read_json, run_logged, safe_name,
                             sha, verify_frozen, write_json_new)

LOCK_HASHES = {
    'delivery/requirements-cp312-macos-arm64.lock.txt':
        '11546f8847c698002cf3c0cdc9134bd722367207a04cfe73ba53b13d9bb4fab0',
    'stage05_p1_closure_review/requirements-cp312-win_amd64.lock.txt':
        '0eb524a8dbc1992d4c3151906abb9f6d73103361ec07767c630b64d9873208e2',
}


class LaunchError(DeliveryError):
    def __init__(self, message: str, code=32):
        super().__init__(message)
        self.code = code


def safe_path(path: Path) -> None:
    """Reject symlink and Windows junction ancestors before any write."""
    try:
        no_symlink(path.absolute())
        for part in (path, *path.parents):
            if getattr(part, 'is_junction', lambda: False)():
                raise DeliveryError('Junction paths are not accepted')
    except DeliveryError as error:
        raise LaunchError(str(error), 12) from None


def static_preflight(root: Path) -> dict:
    """Validate immutable science and the exact lock before probing/installing."""
    try:
        safe_path(root)
        native_lock = lock_path()
        if platform.system() == 'Darwin' and int(platform.mac_ver()[0].split('.')[0]) < 14:
            raise LaunchError('Require macOS 14 or newer', 14)
        for name, digest in LOCK_HASHES.items():
            if sha(checked_source(root, name)) != digest:
                raise LaunchError('Dependency hash-lock fingerprint mismatch: ' + name)
        frozen = verify_frozen(root)
        return {'lock': native_lock, 'lock_sha256': LOCK_HASHES[native_lock], 'frozen': frozen}
    except DeliveryError as error:
        if isinstance(error, LaunchError):
            raise
        raise LaunchError(str(error)) from None


def runtime_directory(root: Path) -> Path:
    runtime = root / '.course_runtime'
    safe_path(runtime)
    runtime.mkdir(exist_ok=True)
    for name in ('bootstrap', 'cache', 'python', 'envs', 'bin', 'tools'):
        safe_path(runtime / name)
    return runtime


def runtime_environment(runtime: Path, *, offline=False, inherited=None) -> dict:
    source = os.environ if inherited is None else inherited
    env = {k: v for k, v in source.items()
           if not k.upper().startswith(('UV_', 'PIP_', 'PYTHON'))
           and k.upper() not in ('VIRTUAL_ENV', 'CONDA_PREFIX')}
    env.update(UV_NO_CONFIG='1', UV_NO_PROGRESS='1', UV_PYTHON_INSTALL_BIN='0',
               UV_PYTHON_INSTALL_REGISTRY='0', UV_PYTHON_NO_REGISTRY='1',
               UV_PYTHON_PREFERENCE='only-managed', UV_DEFAULT_INDEX='https://pypi.org/simple',
               UV_CACHE_DIR=str(runtime / 'cache'), UV_PYTHON_INSTALL_DIR=str(runtime / 'python'),
               UV_PYTHON_BIN_DIR=str(runtime / 'bin'), UV_TOOL_BIN_DIR=str(runtime / 'bin'),
               UV_TOOL_DIR=str(runtime / 'tools'), UV_HTTP_TIMEOUT='60', UV_HTTP_RETRIES='2',
               PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    if offline:
        env.update(UV_OFFLINE='1', PIP_NO_INDEX='1')
    return env


def replace_pointer(path: Path, record: dict) -> None:
    safe_path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    write_json_new(temporary, record)
    safe_path(path)
    os.replace(temporary, path)


def write_latest(root: Path, summary: dict) -> None:
    replace_pointer(runtime_directory(root) / 'last_run.json', summary)


def environment_probe(python: Path, root: Path, env: dict) -> dict | None:
    if not python.is_file():
        return None
    # uv creates interpreter symlinks on macOS; permit only this private runtime.
    safe_path(python.parent)
    if not python.resolve().is_relative_to((root / '.course_runtime').resolve()):
        return None
    try:
        result = subprocess.run([str(python), '-I', '-X', 'utf8', str(root / 'scripts/start_course.py'), '--probe'],
                                cwd=root, env=env, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=90)
        value = json.loads(result.stdout) if result.returncode == 0 else None
        return value if isinstance(value, dict) and value.get('native_imports') is True else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def ensure_environment(root: Path, uv: Path, *, offline=False) -> tuple[Path, dict, bool]:
    static = static_preflight(root)
    runtime = runtime_directory(root)
    env = runtime_environment(runtime, offline=offline)
    state_path = runtime / 'environment.json'
    safe_path(state_path)
    root_digest = hashlib.sha256(str(root.resolve()).encode('utf-8')).hexdigest()
    state = None
    if state_path.exists():
        try:
            state = read_json(state_path)
            relative = safe_name(state['environment'])
            if not relative.startswith('envs/') or len(Path(relative).parts) != 2:
                raise ValueError('Unsafe environment pointer')
            folder = runtime / relative
            safe_path(folder)
            python = folder / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')
            if state.get('root_sha256') == root_digest and state.get('lock_sha256') == static['lock_sha256']:
                info = environment_probe(python, root, env)
                if info is not None:
                    print('Private environment verified; no downloads or installation needed.', flush=True)
                    return python, info, True
        except (ValueError, KeyError, TypeError, OSError, DeliveryError):
            raise LaunchError('Runtime environment record is invalid; original preserved. Use a new extraction.', 22) from None
    if offline:
        raise LaunchError('OFFLINE: complete verified private environment is missing, moved, or damaged. Run once online in this directory; existing runtime preserved.', 20)
    # Always create a fresh folder. A partial/old environment is never overwritten.
    safe_path(uv)
    if not uv.is_file():
        raise LaunchError('Verified uv executable unavailable; use the native launcher.', 21)
    folder = runtime / 'envs' / ('py312-' + uuid.uuid4().hex[:12])
    safe_path(folder)
    folder.parent.mkdir(exist_ok=True)
    python = folder / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')
    log_path = runtime / ('setup-' + uuid.uuid4().hex[:12] + '.log')
    commands = [
        ('Creating isolated CPython 3.12.10 with pip',
         [uv, 'venv', '--no-config', '--python', sys.executable, '--seed', folder]),
        ('Installing nine exact SHA-256 locked dependencies',
         [uv, 'pip', 'install', '--no-config', '--python', python, '--require-hashes',
          '--only-binary', ':all:', '--no-deps', '--index-url', 'https://pypi.org/simple',
          '-r', root / static['lock']]),
    ]
    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        for label, command in commands:
            print(label + '...', flush=True)
            try:
                process = subprocess.run([str(x) for x in command], cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=900)
            except (OSError, subprocess.TimeoutExpired):
                raise LaunchError('Environment setup failed or timed out; retry online. See ' + log_path.relative_to(root).as_posix(), 21) from None
            log.flush()
            if process.returncode:
                raise LaunchError('Environment setup failed (download/hash/cache/native wheel); existing files preserved. See ' + log_path.relative_to(root).as_posix(), 21)
    info = environment_probe(python, root, env)
    if info is None:
        raise LaunchError('Installed environment failed version/native-import/pip validation; see private setup log. Original environments preserved.', 22)
    replace_pointer(state_path, {'environment': folder.relative_to(runtime).as_posix(),
                                'lock_sha256': static['lock_sha256'], 'root_sha256': root_digest})
    return python, info, False


def demonstration_summary(synthetic: dict) -> dict:
    result = next(check['result'] for check in synthetic['checks'] if check['name'] == 'synthetic')
    demo = next(run['result'] for run in result['runs'] if run['name'] == 'demo')
    metrics = read_json(ROOT / 'stage03b_official_validation/results/dev_metrics.json')['metrics']['tfidf_char2']
    return {'toy_cosine': demo,
            'saved_official_dev': {'recomputed': False, 'source': 'stage03b_official_validation/results/dev_metrics.json',
                                  'method': 'tfidf_char2', 'f1': metrics['f1'],
                                  'precision': metrics['precision'], 'recall': metrics['recall'],
                                  'false_positive_rate': metrics['false_positive_rate'],
                                  'tfidf_confusion': {k: metrics[k] for k in ('tp', 'fp', 'fn', 'tn')}}}


def run_validation(root: Path, *, offline=False, environment_reused=False) -> int:
    output = None
    summary = {'passed': False, 'exit_code': 32, 'output': None, 'offline_requested': offline,
               'environment_reused': environment_reused, 'dev_evaluation': False, 'train_run': False,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': []}
    try:
        static = static_preflight(root)
        summary['environment'] = environment()
        summary['frozen'] = static['frozen']
        summary['lock_sha256'] = static['lock_sha256']
        # Existing synthetic runner owns this scratch location; reject redirected parents.
        safe_path(root / 'stage05_p1_closure_review/local_only')
        output = new_output(root / 'course_validation' /
                            (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]))
        summary['output'] = output.relative_to(root).as_posix()
        print('Validating raw SHA-256 and rebuilding all 42,511 Manifest rows...', flush=True)
        from prepare_afqmc import prepare, PreparationError
        archive = root / 'stage01_data_audit/raw/afqmc_public.zip'
        # Already supplied data always takes the strictly offline preparation branch.
        data_offline = offline or archive.exists() or archive.is_symlink()
        try:
            data = prepare(root, offline=data_offline)
        except PreparationError as error:
            raise LaunchError(str(error), 30) from None
        summary['preparation'] = data
        summary['data_offline'] = data_offline
        summary['checks'].append({'name': 'data-preparation', 'exit_code': 0, 'manifest_rows': data['manifest']['rows']})
        print('Running A: 49 frozen synthetic tests, preparation/delivery regressions, toy cosine and frozen review...', flush=True)
        item = run_logged('synthetic-course', ['scripts/verify_course_delivery.py', '--mode', 'synthetic',
                                             '--root', str(root), '--output', str(output / 'synthetic')], root, output)
        summary['checks'].append(item)
        if item['exit_code']:
            raise LaunchError('Synthetic validation failed; inspect the new synthetic/summary.json and local log.', 31)
        synthetic = read_json(output / 'synthetic/summary.json')
        summary.update(demonstration_summary(synthetic))
        summary['checks'].extend(synthetic['checks'])
        # Keep compact safe check records, avoiding nested machine logs/predictions.
        summary['checks'] = [{k: v for k, v in check.items() if k != 'result'} for check in summary['checks']]
        summary['frozen_after'] = verify_frozen(root)
        summary['passed'] = True
        summary['exit_code'] = 0
    except LaunchError as error:
        summary.update(error=str(error), exit_code=error.code)
    except DeliveryError as error:
        summary.update(error=str(error), exit_code=32)
    except Exception as error:
        summary.update(error=f'{type(error).__name__}: validation failed; check local files and permissions. No raw records logged.', exit_code=32)
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    if output is not None:
        write_json_new(output / 'summary.json', summary)
        lines = ['AFQMC 课程验证 / Course validation',
                 '结果: ' + ('PASS' if summary['passed'] else 'FAIL'),
                 '结果目录: ' + summary['output'],
                 '本次未训练正式模型，未重跑官方 Dev。', '']
        if summary['passed']:
            lines += ['本次实际计算：自编合成例子的余弦相似度（不是正式 AFQMC 预测）']
            for row in summary['toy_cosine']['rows']:
                lines.append(f"  {row['example_id']} / {row['method']}: {row['score']:.12f}")
            saved = summary['saved_official_dev']
            lines += ['', f"历史冻结官方 Dev 指标（本次只校验文件与算术，未重新计算预测）: TF-IDF F1={saved['f1']:.6f}",
                      '原始数据 SHA 与全部 42,511 条 Manifest 已验证。',
                      '详细结果：summary.json；synthetic/summary.json；synthetic-course.log。']
        else:
            lines.append('错误: ' + summary.get('error', 'unknown'))
        with (output / 'READ_ME.txt').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
    write_latest(root, summary)
    print(json.dumps(summary, ensure_ascii=False, separators=(',', ':'), allow_nan=False), flush=True)
    return summary['exit_code']


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-pause', action='store_true', help='Native wrapper keeps/ closes the window; Python never pauses')
    parser.add_argument('--uv', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--environment-reused', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.probe:
            print(json.dumps(environment(), ensure_ascii=False)); return 0
        if args.worker:
            return run_validation(ROOT, offline=args.offline, environment_reused=args.environment_reused)
        if args.uv is None:
            raise LaunchError('Start with Run_Mac.command or Run_Windows.cmd.', 2)
        python, _, reused = ensure_environment(ROOT, args.uv, offline=args.offline)
        command = [str(python), '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--worker', '--no-pause']
        if args.offline:
            command.append('--offline')
        if reused:
            command.append('--environment-reused')
        return subprocess.run(command, cwd=ROOT, env=runtime_environment(ROOT / '.course_runtime', offline=args.offline)).returncode
    except (LaunchError, DeliveryError) as error:
        code = getattr(error, 'code', 32)
        record = {'passed': False, 'exit_code': code, 'output': None, 'dev_evaluation': False,
                  'train_run': False, 'error': str(error)}
        print('COURSE ERROR: ' + str(error), file=sys.stderr)
        # A failing preflight does not write inside an untrusted runtime.
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return code
    except Exception as error:
        print(f'COURSE ERROR 32: {type(error).__name__}; inspect local setup logs. Existing files preserved.', file=sys.stderr)
        return 32


if __name__ == '__main__':
    raise SystemExit(main())
