"""Read-only source/dependency admission audit. Never imports experiment modules.

No dataset ZIP, Manifest contents, predictions, model fitting or tests are read/run.
Only writes audit artifacts in this directory. Does not create a delivery ZIP.
"""
from __future__ import annotations
import ast,csv,hashlib,importlib.metadata as md,json,re,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).parent
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    freeze=read(ROOT/'stage03b_official_validation/pre_dev_freeze_manifest.v2.json')
    tracked={x.decode() for x in git('ls-files','-z').split(b'\0') if x}
    frozen=freeze['frozen_source_files_sha256'];inventory=[]
    candidates=sorted({p for p in tracked if p.endswith('.py') and p.startswith(('stage01_data_audit/','stage02_math_protocol/','stage03a_internal_experiment/','stage03b_official_validation/','stage04_report_draft/'))}|{p for p in frozen if p.endswith('.py')})
    # Two local QA scripts are checked, but not silently added to the public package.
    qa=[p.relative_to(ROOT).as_posix() for p in (ROOT/'stage03a_internal_experiment/qa_review').glob('*.py')]
    candidates+=sorted(set(qa)-set(candidates))
    local_modules={Path(p).stem for p in candidates}|{'tests'}
    third_party={};unresolved=[];test_cases=[];resources=[]
    for p in candidates:
        raw=(ROOT/p).read_bytes();source=raw.decode('utf-8-sig');tree=ast.parse(source,filename=p);compile(tree,p,'exec')
        imports=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.update(x.name.split('.')[0] for x in n.names)
            elif isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_'):test_cases.append({'source':p,'name':n.name,'line':n.lineno})
            if isinstance(n,ast.Constant) and isinstance(n.value,str):
                s=n.value
                if any(s.endswith(ext) for ext in ('.json','.csv','.jsonl','.zip','.ttf','.ttc','.docx')) and len(s)<180:resources.append({'source':p,'line':getattr(n,'lineno',None),'resource_literal':s})
        for mod in imports:
            if mod not in sys.stdlib_module_names and mod not in local_modules and mod!='__future__':
                third_party.setdefault(mod,[]).append(p)
        inventory.append({'path':p,'role':'optional_report_source' if p.startswith('stage04_') else 'local_historical_qa' if p in qa else 'frozen_or_runtime_source','tracked':p in tracked,'bytes':len(raw),'sha256':sha_bytes(raw),'syntax':'PASS','imports':sorted(imports)})
    package_map={'numpy':'numpy','scipy':'scipy','sklearn':'scikit-learn','PIL':'Pillow','matplotlib':'matplotlib'}
    for mod,paths in third_party.items():
        if mod not in package_map:unresolved.append({'module':mod,'sources':paths})
    frozen_checks=[]
    for p,want in frozen.items():
        raw=(ROOT/p).read_bytes();blob=git('show','HEAD:'+p) if p in tracked else None
        frozen_checks.append({'path':p,'expected_sha256':want,'worktree_sha256':sha_bytes(raw),'match':sha_bytes(raw)==want,'git_blob_sha256':sha_bytes(blob) if blob is not None else None,'git_blob_matches_frozen':blob is not None and sha_bytes(blob)==want})
    assert all(x['match'] for x in frozen_checks)
    versions={p:md.version(p) for p in ['numpy','scipy','scikit-learn','Pillow','joblib','threadpoolctl','packaging']}
    env={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':versions['numpy'],'scipy':versions['scipy'],'scikit_learn':versions['scikit-learn'],'pillow':versions['Pillow']}
    assert env==freeze['environment'],env
    resolved=[]
    for parent in versions:
        for raw_req in md.requires(parent) or []:
            req=Requirement(raw_req)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            actual=md.version(req.name);assert actual in req.specifier,(parent,req.name,actual)
            resolved.append({'parent':parent,'dependency':req.name,'constraint':str(req.specifier),'installed':actual,'matches':True})
    # Full pip check is environmental evidence only, not installation or experiment execution.
    pipcheck=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,encoding='utf-8')
    (OUT/'local_only').mkdir(exist_ok=True)
    dump(OUT/'local_only/pip_check.json',{'returncode':pipcheck.returncode,'stdout':pipcheck.stdout,'stderr':pipcheck.stderr})
    req=(ROOT/'stage03a_internal_experiment/requirements.txt').read_text(encoding='utf-8')
    for p,v in [(k,versions[k]) for k in ['numpy','scipy','scikit-learn','Pillow']]:assert p.lower()+'=='+v in req.lower()
    lock='# AFQMC audit snapshot of the existing frozen environment; no installation performed.\n# Python 3.12.10; independent from root five-experiment requirements.\n'+'\n'.join(k+'=='+v for k,v in versions.items())+'\n'
    (OUT/'requirements-afqmc.lock.txt').write_bytes(lock.encode())
    baseline={}
    for path in ['stage04_final_report/revision_highest_audit/document_gate.json','stage03b_official_validation/results/dev_metrics.json','stage03b_official_validation/pre_dev_freeze_manifest.json','stage03b_official_validation/pre_dev_freeze_manifest.v2.json']:
        baseline[path]=sha_bytes((ROOT/path).read_bytes())
    findings=[
        {'id':'A01','severity':'P1','status':'mitigated_by_dedicated_instructions','fact':'Root README/requirements refer to other mathematics experiments. Use AFQMC-specific instructions and a separate environment.'},
        {'id':'A02','severity':'P1','status':'documented_prerequisite','source':'stage03a_internal_experiment/tests/test_stage3a.py:97','fact':'test_zip_loader_opens_only_train_member requires actual local Train archive; suite is not fully data-free.'},
        {'id':'A03','severity':'P1','status':'documented_prerequisite','source':'stage03b_official_validation/test_recovery_runner.py:226','fact':'--prove-v1 requires the original Git commit; normal source ZIP has no Git history.'},
        {'id':'A04','severity':'P1','status':'not_executed','fact':'No clean environment installation, offline wheel installation or another-machine reproduction has been done by this admission audit.'},
        {'id':'A05','severity':'P1','status':'hold_for_data_terms_and_course_acceptance','fact':'AFQMC redistribution permission not established. Plan excludes original data; teacher acceptance of acquisition instructions remains unconfirmed.'},
        {'id':'A06','severity':'P1','status':'documented_no_replay','source':'stage03b_official_validation/run_official_dev.py:207','fact':'Official runner uses private incident/authorization/gate records and exact historical Git state; it is historical source, not a portable default entry.'},
        {'id':'A07','severity':'P1','status':'documented_missing_evidence','fact':'Row-level predictions and Manifest are intentionally local. Public aggregate evidence cannot regenerate per-row scores or validate individual classifications.'},
        {'id':'A08','severity':'P2','status':'documented_optional_toolchain','fact':'Report builders need separate plot dependencies, fonts and nonpublic inputs; final report is not an automatically reproducible output of the public-only package.'},
        {'id':'A09','severity':'P1','status':'historical_source_only_not_default','source':'stage03a_internal_experiment/qa_review/run_qa_diagnostics.py:333','fact':'Historical QA runner also executes registered char(2,3) exploration and reads private manual notes; do not offer it as the formal default or execute it during this audit.'},
    ]
    result={'audit_utc':datetime.now(timezone.utc).isoformat(),'source_commit':git('rev-parse','HEAD').decode().strip(),'syntax_checked_source_files':len(inventory),'unresolved_external_imports':unresolved,'third_party_imports':third_party,'environment_matches_frozen':True,'installed_versions':versions,'transitive_runtime_requirements':resolved,'pip_check_returncode':pipcheck.returncode,'pip_check_clean':pipcheck.returncode==0,'frozen_source_files_verified':len(frozen_checks),'git_blob_fingerprint_differences':[x['path'] for x in frozen_checks if not x['git_blob_matches_frozen']],'frozen_checks':frozen_checks,'source_inventory':inventory,'static_test_method_count':len(test_cases),'tests_executed':0,'models_imported_or_executed':0,'raw_archives_opened':0,'raw_dev_read_or_evaluated':False,'created_zip':False,'protected_report_and_metric_hashes':baseline,'findings':findings,'verdict':'CONDITIONAL_ADMISSION_FOR_PACKAGING_PREPARATION_ONLY'}
    dump(OUT/'admission_checks.json',result)
    with (OUT/'source_inventory.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','role','tracked','bytes','sha256','syntax']);w.writeheader();w.writerows({k:x[k] for k in w.fieldnames} for x in inventory)
    dump(OUT/'static_test_inventory.json',{'test_methods':test_cases,'executed':False})
    # Resource literals may contain sensitive local data identifiers; retained privately.
    dump(OUT/'local_only/resource_literals.json',resources)
    assert not unresolved,unresolved
    print(json.dumps({k:result[k] for k in ['syntax_checked_source_files','frozen_source_files_verified','git_blob_fingerprint_differences','environment_matches_frozen','pip_check_clean','static_test_method_count','tests_executed','raw_dev_read_or_evaluated','verdict']},ensure_ascii=False))
if __name__=='__main__':main()
