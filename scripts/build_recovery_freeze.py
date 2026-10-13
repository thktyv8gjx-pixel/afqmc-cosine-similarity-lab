"""Additive incident evidence and v2 freeze; never reads AFQMC Dev."""
from __future__ import annotations
import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'stage03b_official_validation'
sys.path.insert(0,str(STAGE))
from recovery_runtime import canonical_bytes,digest,method_consistency,validate_manifest_digest,write_exclusive_json,utc_now

ORIGINAL='6deb2ace9f433173b483338172db07318a17a735'


def record_incident():
    v1_raw=(STAGE/'pre_dev_freeze_manifest.json').read_bytes()
    v1=json.loads(v1_raw); validate_manifest_digest(v1)
    original_manifest=subprocess.check_output(['git','show',ORIGINAL+':stage03b_official_validation/pre_dev_freeze_manifest.json'],cwd=ROOT)
    if json.loads(original_manifest)!=v1 or original_manifest.replace(b'\r\n',b'\n')!=v1_raw.replace(b'\r\n',b'\n'):
        raise AssertionError('v1 manifest content differs from original commit')
    source=subprocess.check_output(['git','show',ORIGINAL+':stage03b_official_validation/run_official_dev.py'],cwd=ROOT)
    if digest(source)!=v1['frozen_source_files_sha256']['stage03b_official_validation/run_official_dev.py']:
        raise AssertionError('original runner does not match v1 source fingerprint')
    log_path=STAGE/'qa_review/official_dev_execution.log'
    raw=log_path.read_bytes(); log=raw.decode('utf-8',errors='replace')
    if 'TypeError' not in log or 'float(thresholds[method])' not in log or 'compute_run' not in log:
        raise AssertionError('failure evidence does not support the recorded incident')
    if (STAGE/'outputs').exists(): raise AssertionError('unexpected result directory before recovery')
    stat=log_path.stat()
    evidence={}
    for p in (log_path,STAGE/'qa_review/official_dev_attempt_blocker.md',STAGE/'local_only/pre_dev_freeze_commit.json'):
        evidence[p.relative_to(ROOT).as_posix()]=digest(p.read_bytes())
    incident={
        'attempt_id':'attempt_001','status':'failed_after_partial_dev_access','recorded_retrospectively_utc':utc_now(),
        'original_freeze_commit_sha':ORIGINAL,'v1_manifest_raw_bytes_sha256':digest(v1_raw),
        'v1_manifest_git_blob_bytes_sha256':digest(original_manifest),
        'v1_worktree_vs_git_blob_difference':'CRLF versus LF only; parsed contents and canonical digest agree',
        'v1_manifest_canonical_sha256':v1['freeze_manifest_sha256'],'original_runner_sha256':digest(source),
        'execution_command':'python stage03b_official_validation/run_official_dev.py --execute-official-dev',
        'command_evidence':'prior session tool invocation; command is not embedded in the original log',
        'file_observed_creation_utc':datetime.fromtimestamp(stat.st_ctime,timezone.utc).isoformat(),
        'file_observed_last_write_utc':datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
        'exact_process_start_end_utc':'unknown; filesystem times are approximate evidence, not process timestamps',
        'exception_type':'TypeError','exception_message':"float() argument must be a string or a real number, not 'dict'",
        'failure_location':'v1 run_official_dev.py:289',
        'completed_stages_inferred_from_frozen_control_flow_and_trace':['Dev member read','archive/member checks','4316-row parsing check','ordered Manifest label/ID alignment','Count transform and cosine scores'],
        'in_memory_inferred':['raw Dev member bytes','parsed sentence pairs and labels','normalized fit vectorizers','raw-input Count matrices and cosine scores'],
        'not_completed_inferred':['Count threshold prediction','formal metrics','TF-IDF evaluation','final result writes'],
        'observed_files':evidence,'observed_outputs_exists':False,
        'information_exposure_limits':'available tool/log evidence contains PASS and traceback, no sample text or score arrays; other memory inspection and unrelated logs were not audited',
        'evaluation_attempt_number_scope':'Stage 3B evaluation attempts, not all historical data-audit reads',
    }
    path=STAGE/'local_only/attempts/attempt_001'
    path.mkdir(parents=True,exist_ok=False)
    (path/'original_execution.log').write_bytes(raw)
    (path/'original_runner.py').write_bytes(source)
    write_exclusive_json(path/'record.json',incident)
    old=json.loads((STAGE/'local_only/pre_dev_freeze_commit.json').read_text())
    correction={'record_type':'additive_historical_digest_correction','old_record_unchanged':True,
        'old_value':old['freeze_manifest_sha256'],'old_length':len(old['freeze_manifest_sha256']),
        'authoritative_manifest_raw_bytes_sha256':digest(v1_raw),'authoritative_manifest_canonical_sha256':v1['freeze_manifest_sha256'],
        'canonical_algorithm':'UTF-8 JSON ensure_ascii=False, sort_keys=True, separators=(comma,colon), allow_nan=False, exclude freeze_manifest_sha256, append LF',
        'derived_from_actual_bytes':True,'recorded_utc':utc_now()}
    write_exclusive_json(STAGE/'local_only/historical_digest_correction.json',correction)
    auth={'source':'explicit human recovery authorization in current session, dated 2026-10-13 Asia/Shanghai',
        'authorized_attempt_id':'attempt_002','recovery_of':'attempt_001','original_freeze_commit_sha':ORIGINAL,
        'attempt_001_record_sha256':digest((path/'record.json').read_bytes()),
        'scope':'repair schema and registered normalization, freeze v2, push, one controlled recovery evaluation; no tuning and no third attempt',
        'recorded_utc':utc_now()}
    write_exclusive_json(STAGE/'local_only/recovery_authorization.json',auth)
    public=STAGE/'recovery'; public.mkdir(exist_ok=True)
    public_incident={k:v for k,v in incident.items() if k not in ('in_memory_inferred','file_observed_creation_utc')}
    write_exclusive_json(public/'attempt_001_incident_summary.json',public_incident)
    write_exclusive_json(public/'historical_digest_correction.json',correction)
    print('Recorded attempt_001 and additive hash correction; original evidence retained.')


def build_freeze():
    raw=(STAGE/'pre_dev_freeze_manifest.json').read_bytes(); v1=json.loads(raw); validate_manifest_digest(v1)
    checks=json.loads((STAGE/'recovery/synthetic_test_results.json').read_text())
    if not checks['passed']: raise AssertionError('synthetic checks did not pass')
    v2=copy.deepcopy(v1)
    v2.update(freeze_version=2,status='engineering_recovery_freeze_after_partial_dev_access',created_utc=utc_now(),
        stage_scope='Registered methods unchanged; attempt_002 is a controlled recovery following partial Dev access in attempt_001')
    v2['recovery']={
        'original_freeze_commit_sha':ORIGINAL,'v1_manifest_raw_bytes_sha256':digest(raw),'v1_manifest_canonical_sha256':v1['freeze_manifest_sha256'],
        'prior_attempt_id':'attempt_001','prior_status':'failed_after_partial_dev_access','prior_partial_dev_access':True,
        'authorized_attempt_id':'attempt_002','authorized_by':'explicit user instruction in current session, 2026-10-13 Asia/Shanghai',
        'authorization_record_sha256':digest((STAGE/'local_only/recovery_authorization.json').read_bytes()),
        'attempt_001_record_sha256':digest((STAGE/'local_only/attempts/attempt_001/record.json').read_bytes()),
        'repair_reasons':['nested frozen threshold objects require strict value parsing before Dev read','reuse registered normalized_pair_texts before both transforms'],
        'execution_record_policy':'create attempt_002 before Dev read; immutable stage events and final record; no third attempt',
        'canonical_algorithm':'UTF-8 sorted compact JSON excluding freeze_manifest_sha256, ensure_ascii=False, allow_nan=False, append LF',
        'tests':checks,
    }
    paths=list(v1['frozen_source_files_sha256'])+[
        'stage03b_official_validation/recovery_runtime.py','stage03b_official_validation/test_recovery_runner.py',
        'stage03b_official_validation/recheck_saved_results.py','scripts/build_recovery_freeze.py','scripts/run_recovery_checks.py',
        'scripts/review_recovery_staging.py','scripts/synthetic_guard/sitecustomize.py']
    v2['frozen_source_files_sha256']={p:digest((ROOT/p).read_bytes()) for p in paths}
    result=method_consistency(v1,v2)
    unchanged_sources={p:v1['frozen_source_files_sha256'][p]==v2['frozen_source_files_sha256'][p]
                       for p in v1['frozen_source_files_sha256'] if p.startswith(('stage01_data_audit/','stage02_math_protocol/','stage03a_internal_experiment/'))}
    result.update(protected_implementation_sources_unchanged=unchanged_sources)
    result['passed']=result['passed'] and all(unchanged_sources.values())
    if not result['passed']: raise AssertionError('experimental method changed')
    write_exclusive_json(STAGE/'recovery/method_consistency_v1_v2.json',result)
    v2.pop('freeze_manifest_sha256'); v2['freeze_manifest_sha256']=digest(canonical_bytes(v2))
    validate_manifest_digest(v2)
    write_exclusive_json(STAGE/'pre_dev_freeze_manifest.v2.json',v2)
    write_exclusive_json(STAGE/'recovery/freeze_fingerprints.json',{
        'v1_raw_bytes_sha256':digest(raw),'v1_canonical_sha256':v1['freeze_manifest_sha256'],
        'v2_raw_bytes_sha256':digest((STAGE/'pre_dev_freeze_manifest.v2.json').read_bytes()),'v2_canonical_sha256':v2['freeze_manifest_sha256'],
        'final_commit_record':'stored only after commit in local-only publication gate; not a self-hash input'})
    print(json.dumps({'v2_canonical_sha256':v2['freeze_manifest_sha256'],'method_consistency_passed':result['passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--record-incident',action='store_true'); p.add_argument('--freeze',action='store_true'); a=p.parse_args()
    if a.record_incident: record_incident()
    elif a.freeze: build_freeze()
    else: p.error('select one mode')
