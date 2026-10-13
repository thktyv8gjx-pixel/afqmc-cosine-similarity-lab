"""Run existing and added suites with an OS-process real-Dev ZIP guard."""
import json
import os
import re
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'stage03b_official_validation'
def main():
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    private=STAGE/'local_only/test_runs'/now; private.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy(); env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONPATH=str(ROOT/'scripts/synthetic_guard'),RECOVERY_FORBIDDEN_ARCHIVE=str(ROOT/'stage01_data_audit/raw/afqmc_public.zip'),
        RECOVERY_GUARD_AUDIT=str(private/'dev_guard_audit.jsonl'))
    commands=[('stage2_math',ROOT/'stage02_math_protocol',['-m','unittest','discover','-s','tests','-t','.','-v']),
              ('stage3a_implementation',ROOT/'stage03a_internal_experiment',['-m','unittest','discover','-s','tests','-t','.','-v']),
              ('stage3b_original_helpers',ROOT,['stage03b_official_validation/test_official_dev_runner.py']),
              ('v1_regression_proof',ROOT,['stage03b_official_validation/test_recovery_runner.py','--prove-v1']),
              ('stage3b_recovery_integration',ROOT,['stage03b_official_validation/test_recovery_runner.py'])]
    results={}; combined=[]
    for name,cwd,args in commands:
        r=subprocess.run([sys.executable,*args],cwd=cwd,env=env,capture_output=True,text=True,encoding='utf-8')
        log=r.stdout+r.stderr
        (private/(name+'.txt')).write_text(log,encoding='utf-8')
        safe=log.replace(str(ROOT),'<PROJECT_ROOT>').replace(str(ROOT).replace('\\','/'),'<PROJECT_ROOT>')
        combined.append('['+name+']\n'+safe)
        count=re.search(r'Ran (\d+) tests?',log)
        results[name]={'exit_code':r.returncode,'tests_run':int(count[1]) if count else None,
            'passed':r.returncode==0,'expected_v1_failures':name=='v1_regression_proof'}
        print(json.dumps({name:results[name]}))
    audit=private/'dev_guard_audit.jsonl'
    events=[json.loads(x) for x in audit.read_text().splitlines()] if audit.exists() else []
    passed=all(r['passed'] for r in results.values()) and not any(x['dev_open']=='forbidden_real' for x in events)
    payload={'scope':'synthetic tests plus existing config/source only; no real Dev read','suites':results,
        'passed':passed,'registered_tests_passed':sum(r['tests_run'] or 0 for k,r in results.items() if k!='v1_regression_proof'),
        'real_dev_open_guard_enforced':True,'forbidden_real_dev_open_attempts':sum(x['dev_open']=='forbidden_real' for x in events),
        'allowed_synthetic_dev_member_opens':sum(x['dev_open']=='synthetic' for x in events),
        'preliminary_sandbox_run':{'tests_run':10,'errors':20,'cause':'temporary directory PermissionError before fixture creation; rerun with requested sandbox escalation'},
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    (private/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    if passed:
        public=STAGE/'recovery'; public.mkdir(exist_ok=True)
        (public/'synthetic_test_results.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
        (public/'synthetic_test_log.txt').write_text('\n'.join(combined),encoding='utf-8')
    return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
