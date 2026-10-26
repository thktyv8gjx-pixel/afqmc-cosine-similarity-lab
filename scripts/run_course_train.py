"""Explicit private Train adapter: immutable sources, new outputs, host CJK font."""
from pathlib import Path
import shutil
import sys
import zipfile

from course_delivery import (DeliveryError, check_train_inputs, read_json, resolve_font,
                             sha, verify_frozen)


def run_train(root: Path, output: Path, font: Path | None = None) -> dict:
    verify_frozen(root)
    inputs = check_train_inputs(root)
    font = resolve_font(font)  # fail before reading Train or creating model output
    experiment = output / 'snapshot/stage03a_internal_experiment'
    freeze = read_json(root / 'stage03b_official_validation/pre_dev_freeze_manifest.v2.json')
    for name in freeze['frozen_source_files_sha256']:
        if name.startswith('stage03a_internal_experiment/'):
            target = output / 'snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    sys.path.insert(0, str(experiment / 'src'))
    import pipeline
    import visualize
    original_font = visualize._font_path
    original_open = zipfile.ZipFile.open
    opened = []
    archive_path = (root / 'stage01_data_audit/raw/afqmc_public.zip').resolve()

    def train_only(z, name, mode='r', *args, **kwargs):
        member = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if Path(z.filename).resolve() != archive_path or member != 'train.json' or mode != 'r':
            raise DeliveryError('Train adapter blocked a non-Train ZIP access')
        opened.append(member)
        return original_open(z, name, mode, *args, **kwargs)

    try:
        visualize._font_path = lambda: font
        zipfile.ZipFile.open = train_only
        result = pipeline.run_experiment(root, experiment)
    finally:
        zipfile.ZipFile.open = original_open
        visualize._font_path = original_font
        sys.path.pop(0)
    metadata = result['run_metadata']
    expected = {'internal_fit_rows': 27467, 'internal_tune_rows': 6867,
                'unique_idf_questions': 54330, 'vocabulary_size': 35242,
                'fit_corpus_content_sha256': freeze['fit_space']['idf_corpus_content_sha256'],
                'vocabulary_sha256': freeze['fit_space']['vocabulary_index_sha256']}
    if any(metadata[k] != v for k, v in expected.items()) or opened != ['train.json']:
        raise DeliveryError('Train reproduction counts/fingerprints or access boundary mismatch; outputs retained privately')
    methods = result['summary']['methods']
    # Keep observed floating-point thresholds separate from the unchanged official freeze.
    comparison = {m: {'observed_internal_threshold': methods[m]['threshold'],
                      'frozen_official_threshold': freeze['thresholds'][m]['value'],
                      'exact_equal': methods[m]['threshold'] == freeze['thresholds'][m]['value']}
                  for m in ('count_char2', 'tfidf_char2')}
    return {'passed': True, 'scope': 'new internal Train experiment, not official Dev replay',
            'inputs': inputs, 'metadata': expected, 'methods': methods,
            'threshold_comparison': comparison, 'zip_members_opened': opened,
            'font_filename': font.name, 'font_sha256': sha(font),
            'figures': result['plots']['figures'] if 'figures' in result['plots'] else result['plots'],
            'raw_outputs_private': True}
